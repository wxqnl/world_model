"""WorldVLA/RynnVLA-compatible LIBERO world-model benchmark utilities.

This module is deliberately narrow: it prepares and exports the public
WorldVLA/RynnVLA LIBERO world-model validation trajectories, using their JSON
trajectory lists and MP4 filename convention. The official metric script can
then be run directly on the generated folder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from wm3d_v3.benchmarks.online_tokenizer import FrameWindow, OnlineObservationTokenizer
from wm3d_v3.data.action_condition import make_action_condition_np
from wm3d_v3.eval.run_eval import build_model


SUITE_TO_LIBERO = {
    "10": "libero_10",
    "long": "libero_10",
    "goal": "libero_goal",
    "object": "libero_object",
    "spatial": "libero_spatial",
}


@dataclass(frozen=True)
class TrajectorySpec:
    suite: str
    episode_index: int
    source_path: str
    dataset_dir: str
    task_name: str
    trj_name: str
    trj_index: int


def _suite_key(value: str) -> str:
    key = str(value).strip().lower()
    if key not in SUITE_TO_LIBERO:
        raise ValueError(f"unsupported suite {value!r}; expected one of {sorted(SUITE_TO_LIBERO)}")
    return "10" if key == "long" else key


def _official_json_path(official_root: Path, suite: str) -> Path:
    return official_root / "exps_libero_world_model" / f"{_suite_key(suite)}_val_ind_trajectory_paths.json"


def _extract_traj_spec(suite: str, episode_index: int, source_path: str) -> TrajectorySpec:
    parts = Path(source_path).parts
    dataset_pos = None
    for idx, part in enumerate(parts):
        if part.startswith("libero_") and "_image_state_action_t_" in part:
            dataset_pos = idx
            break
    if dataset_pos is None or dataset_pos + 2 >= len(parts):
        raise ValueError(f"cannot parse WorldVLA trajectory path: {source_path}")
    dataset_dir = parts[dataset_pos]
    task_name = parts[dataset_pos + 1]
    trj_name = parts[dataset_pos + 2]
    match = re.fullmatch(r"trj_(\d+)", trj_name)
    if match is None:
        raise ValueError(f"cannot parse trajectory id from {source_path}")
    return TrajectorySpec(
        suite=_suite_key(suite),
        episode_index=int(episode_index),
        source_path=source_path,
        dataset_dir=dataset_dir,
        task_name=task_name,
        trj_name=trj_name,
        trj_index=int(match.group(1)),
    )


def load_trajectory_specs(official_root: Path, suite: str) -> list[TrajectorySpec]:
    json_path = _official_json_path(official_root, suite)
    data = json.loads(json_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"expected list in {json_path}")
    return [_extract_traj_spec(suite, idx, str(path)) for idx, path in enumerate(data)]


def _ensure_worldvla_on_path(official_root: Path) -> None:
    for sub in (official_root, official_root / "libero_util"):
        text = str(sub)
        if text not in sys.path:
            sys.path.insert(0, text)


def _load_libero_task_map(suite: str) -> dict[str, Any]:
    from libero.libero import benchmark

    suite_name = SUITE_TO_LIBERO[_suite_key(suite)]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    tasks = {}
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        tasks[task.name] = task
    return tasks


def _is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


def _mkdir_clean(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_png(path: Path, frame: np.ndarray) -> None:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def prepare_val_trajectories(args: argparse.Namespace) -> dict[str, Any]:
    official_root = args.official_root.resolve()
    _ensure_worldvla_on_path(official_root)
    from libero_util.libero_utils import get_libero_dummy_action, get_libero_env
    import h5py
    import robosuite.utils.transform_utils as T

    processed_root = args.processed_root.resolve()
    raw_root = args.raw_data_root.resolve()
    processed_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "protocol": "worldvla_rynnvla_libero_world_model_val_ind_replay",
        "resolution": int(args.resolution),
        "processed_root": str(processed_root),
        "raw_data_root": str(raw_root),
        "suites": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for suite in [_suite_key(s) for s in args.suites]:
        specs_all = load_trajectory_specs(official_root, suite)
        specs = specs_all
        if args.rank is not None and args.world_size is not None:
            specs = [spec for i, spec in enumerate(specs_all) if i % int(args.world_size) == int(args.rank)]
        if args.limit:
            specs = specs[: int(args.limit)]
        task_map = _load_libero_task_map(suite)
        suite_name = SUITE_TO_LIBERO[suite]
        env_cache: dict[str, Any] = {}
        suite_rows = []
        try:
            for spec in specs:
                if spec.task_name not in task_map:
                    raise KeyError(f"{spec.task_name} not found in LIBERO task map for {suite_name}")
                out_trj = processed_root / spec.dataset_dir / spec.task_name / spec.trj_name
                done_marker = out_trj / "metadata.json"
                if done_marker.exists() and args.skip_existing:
                    suite_rows.append({"trajectory": spec.source_path, "status": "skipped_existing", "out": str(out_trj)})
                    continue
                _mkdir_clean(out_trj, overwrite=True)
                dirs = {
                    name: out_trj / name
                    for name in (
                        "action",
                        "ee_state",
                        "gripper_state",
                        "eef_gripper_state",
                        "robot_state",
                        "imgs_third_view",
                        "imgs_wrist",
                    )
                }
                for d in dirs.values():
                    d.mkdir(parents=True, exist_ok=True)

                raw_hdf5 = raw_root / suite_name / f"{spec.task_name}_demo.hdf5"
                if not raw_hdf5.exists():
                    raise FileNotFoundError(f"missing raw LIBERO demo: {raw_hdf5}")
                task = task_map[spec.task_name]
                if spec.task_name not in env_cache:
                    env_cache[spec.task_name] = get_libero_env(task, resolution=int(args.resolution))[0]
                env = env_cache[spec.task_name]

                with h5py.File(raw_hdf5, "r") as h5:
                    demo_key = f"demo_{spec.trj_index}"
                    if demo_key not in h5["data"]:
                        raise KeyError(f"{demo_key} missing in {raw_hdf5}")
                    demo = h5["data"][demo_key]
                    orig_actions = demo["actions"][()]
                    orig_states = demo["states"][()]
                    orig_robot_states = demo["robot_states"][()]

                    env.reset()
                    env.set_init_state(orig_states[0])
                    obs = None
                    reward = done = info = None
                    for _ in range(10):
                        obs, reward, done, info = env.step(get_libero_dummy_action())

                    states = []
                    actions = []
                    ee_states = []
                    gripper_states = []
                    joint_states = []
                    robot_states = []
                    front_images = []
                    wrist_images = []
                    noops = 0

                    for action in orig_actions:
                        prev_action = actions[-1] if actions else None
                        if _is_noop(action, prev_action):
                            noops += 1
                            continue
                        if states == []:
                            states.append(orig_states[0])
                            robot_states.append(orig_robot_states[0])
                        else:
                            states.append(env.sim.get_state().flatten())
                            robot_states.append(
                                np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                            )
                        actions.append(action)
                        if "robot0_gripper_qpos" in obs:
                            gripper_states.append(obs["robot0_gripper_qpos"])
                        joint_states.append(obs["robot0_joint_pos"])
                        ee_states.append(
                            np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
                        )
                        front_images.append(obs["agentview_image"])
                        wrist_images.append(obs["robot0_eye_in_hand_image"])
                        obs, reward, done, info = env.step(action.tolist())

                ready_status = "ready"
                if not bool(done) and bool(args.state_render_fallback):
                    print(
                        f"[prepare-val] replay did not finish for {spec.task_name}/{spec.trj_name}; "
                        "using state-render fallback",
                        flush=True,
                    )
                    states = []
                    actions = []
                    ee_states = []
                    gripper_states = []
                    robot_states = []
                    front_images = []
                    wrist_images = []
                    noops = 0
                    env.reset()
                    for state_idx, action in enumerate(orig_actions):
                        prev_action = actions[-1] if actions else None
                        if _is_noop(action, prev_action):
                            noops += 1
                            continue
                        env.sim.set_state_from_flattened(orig_states[state_idx])
                        env.sim.forward()
                        obs_getter = getattr(env, "_get_observations", None) or getattr(env.env, "_get_observations")
                        obs = obs_getter()
                        states.append(orig_states[state_idx])
                        robot_states.append(orig_robot_states[state_idx])
                        actions.append(action)
                        gripper_states.append(obs["robot0_gripper_qpos"])
                        ee_states.append(
                            np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
                        )
                        front_images.append(obs["agentview_image"])
                        wrist_images.append(obs["robot0_eye_in_hand_image"])
                    done = bool(actions)
                    ready_status = "ready_state_render_fallback"

                if not bool(done):
                    suite_rows.append(
                        {
                            "trajectory": spec.source_path,
                            "status": "failed_replay",
                            "out": str(out_trj),
                            "num_actions": len(actions),
                            "num_noops": noops,
                        }
                    )
                    continue
                for j, action in enumerate(actions):
                    np.save(dirs["action"] / f"action_{j}.npy", np.asarray(action))
                    ee = np.asarray(ee_states[j])
                    grip = np.asarray(gripper_states[j])
                    np.save(dirs["ee_state"] / f"ee_state_{j}.npy", ee)
                    np.save(dirs["gripper_state"] / f"gripper_state_{j}.npy", grip)
                    np.save(dirs["eef_gripper_state"] / f"eef_gripper_state_{j}.npy", np.concatenate([ee, grip]))
                    np.save(dirs["robot_state"] / f"robot_state_{j}.npy", np.asarray(robot_states[j]))
                    _save_png(dirs["imgs_third_view"] / f"image_{j}.png", np.asarray(front_images[j])[::-1, ::-1])
                    _save_png(dirs["imgs_wrist"] / f"image_{j}.png", np.asarray(wrist_images[j])[::-1, ::-1])

                meta = {
                    "trajectory": spec.source_path,
                    "suite": suite,
                    "task_name": spec.task_name,
                    "trj_name": spec.trj_name,
                    "episode_index": spec.episode_index,
                    "num_actions": len(actions),
                    "num_noops": int(noops),
                    "resolution": int(args.resolution),
                    "status": ready_status,
                }
                done_marker.write_text(json.dumps(meta, indent=2, sort_keys=True))
                suite_rows.append({"trajectory": spec.source_path, "status": ready_status, "out": str(out_trj), "num_actions": len(actions)})
        finally:
            for env in env_cache.values():
                try:
                    env.close()
                except Exception:
                    pass
        summary["suites"][suite] = {
            "requested": len(specs),
            "ready": sum(
                1
                for row in suite_rows
                if row["status"] in {"ready", "ready_state_render_fallback", "skipped_existing"}
            ),
            "failed": sum(1 for row in suite_rows if row["status"] == "failed_replay"),
            "rows": suite_rows,
        }

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "suites"}, indent=2, sort_keys=True))
    for suite, info in summary["suites"].items():
        print(f"{suite}: requested={info['requested']} ready={info['ready']} failed={info['failed']}")
    return summary


def _read_image(path: Path, *, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _read_action(path: Path) -> np.ndarray:
    return np.asarray(np.load(path), dtype=np.float32).reshape(-1)[:7]


def _numeric_file_order(path: Path, prefix: str, suffix: str) -> list[Path]:
    def key(p: Path) -> int:
        m = re.fullmatch(re.escape(prefix) + r"(\d+)" + re.escape(suffix), p.name)
        return int(m.group(1)) if m else -1

    files = [p for p in path.glob(f"{prefix}*{suffix}") if key(p) >= 0]
    return sorted(files, key=key)


def _video_name(run_id: str, episode: str | int, success: str, task: str) -> str:
    task_clean = task.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    return f"{run_id}--episode={episode}--success={success}--task={task_clean}.mp4"


def _save_video(path: Path, frames: Iterable[np.ndarray], *, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps), macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _tensor_to_uint8(frame: torch.Tensor, *, size: int) -> np.ndarray:
    x = frame.detach().float().clamp(0.0, 1.0)
    if x.ndim != 3:
        raise ValueError(f"expected [3,H,W] frame tensor, got {tuple(x.shape)}")
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    arr = (x.permute(1, 2, 0).cpu().numpy() * 255.0).round()
    return np.clip(arr, 0, 255).astype(np.uint8)


def _load_action_stats(path: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if path is None:
        return None, None
    if not path.exists():
        raise FileNotFoundError(f"action_stats not found: {path}")
    data = np.load(path)
    return data["mean"].astype(np.float32), data["std"].astype(np.float32)


def _load_model_and_cfg(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if args.cfg is not None:
        cfg = yaml.safe_load(args.cfg.read_text())
        cfg_source = str(args.cfg)
    else:
        if "cfg" not in sd:
            raise KeyError("checkpoint has no cfg; pass --cfg explicitly")
        cfg = sd["cfg"]
        cfg_source = f"{args.ckpt}:cfg"
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(sd["model"], strict=True)
    stats_path = cfg.get("data", {}).get("action_stats")
    if stats_path:
        mean, std = _load_action_stats(Path(stats_path))
        model.load_action_stats(torch.as_tensor(mean[:6], device=device), torch.as_tensor(std[:6], device=device))
    report = {
        "checkpoint": str(args.ckpt),
        "cfg_source": cfg_source,
        "checkpoint_step": sd.get("step"),
        "checkpoint_epoch": sd.get("epoch"),
        "best_val": sd.get("best_val"),
        "model_hidden": cfg.get("model", {}).get("state", {}).get("hidden"),
        "model_layers": cfg.get("model", {}).get("state", {}).get("n_layers"),
    }
    return model, cfg, report


def _pad_action_chunk(actions: list[np.ndarray], start: int, k: int) -> np.ndarray:
    if not actions:
        return np.zeros((k, 7), dtype=np.float32)
    chunk = [actions[min(i, len(actions) - 1)] for i in range(start, start + k)]
    return np.stack(chunk, axis=0).astype(np.float32)


def _task_text_map(suite: str) -> dict[str, str]:
    return {name: str(task.language) for name, task in _load_libero_task_map(suite).items()}


def export_worldvla_videos(args: argparse.Namespace) -> dict[str, Any]:
    official_root = args.official_root.resolve()
    _ensure_worldvla_on_path(official_root)
    processed_root = args.processed_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, cfg, ckpt_report = _load_model_and_cfg(args, device)
    data_cfg = cfg["data"]
    token_grid = 16 if data_cfg.get("tokens_subdir") == "vggt_p256" else 8
    tokenizer = OnlineObservationTokenizer(
        T=int(data_cfg["T"]),
        token_grid=token_grid,
        task_cache_dir=args.task_cache_dir,
        device=str(device),
        image_size=int(args.vggt_image_size),
        context_rgb_size=int(args.context_rgb_size),
        qwen_device=args.qwen_device,
        allow_zero_task_fallback=bool(args.allow_zero_task_fallback),
    )
    mean, std = _load_action_stats(Path(data_cfg["action_stats"]) if data_cfg.get("action_stats") else None)

    run_id = args.run_id or time.strftime("wm3d_%Y_%m_%d-%H_%M_%S")
    report: dict[str, Any] = {
        "protocol": "worldvla_rynnvla_libero_world_model_mp4_export",
        "rollout": "autoregressive_first_prediction_per_action",
        "run_id": run_id,
        "checkpoint": ckpt_report,
        "processed_root": str(processed_root),
        "out_dir": str(out_dir),
        "video_size": int(args.video_size),
        "suites": {},
    }

    for suite in [_suite_key(s) for s in args.suites]:
        specs_all = load_trajectory_specs(official_root, suite)
        specs = specs_all
        if args.rank is not None and args.world_size is not None:
            specs = [spec for i, spec in enumerate(specs_all) if i % int(args.world_size) == int(args.rank)]
        if args.limit:
            specs = specs[: int(args.limit)]
        text_map = _task_text_map(suite)
        suite_rows = []
        for spec in specs:
            trj = processed_root / spec.dataset_dir / spec.task_name / spec.trj_name
            if not trj.exists():
                raise FileNotFoundError(f"processed trajectory missing: {trj}")
            task_text = text_map.get(spec.task_name, spec.task_name.replace("_", " "))
            action_files = _numeric_file_order(trj / "action", "action_", ".npy")
            actions = [_read_action(p) for p in action_files]
            if len(actions) <= 21:
                suite_rows.append({"trajectory": spec.source_path, "status": "skipped_short", "num_actions": len(actions)})
                continue
            for camera_dir, camera_name in (("imgs_third_view", "front"), ("imgs_wrist", "wrist")):
                image_files = _numeric_file_order(trj / camera_dir, "image_", ".png")
                if len(image_files) <= 21:
                    suite_rows.append(
                        {
                            "trajectory": spec.source_path,
                            "camera": camera_name,
                            "status": "skipped_short_images",
                            "num_images": len(image_files),
                        }
                    )
                    continue
                n_steps = min(len(actions), len(image_files)) - 1
                if args.max_steps:
                    n_steps = min(n_steps, int(args.max_steps))
                gt_frames = [_read_image(p, size=int(args.video_size)) for p in image_files[: n_steps + 1]]
                rollout_frames = [gt_frames[0]]
                frame_window = FrameWindow(int(data_cfg["T"]))
                frame_window.reset(gt_frames[0])

                for step in range(n_steps):
                    obs = tokenizer.tokenize(frame_window.list(), task_text)
                    action_chunk = _pad_action_chunk(actions, step, int(data_cfg["k"]))
                    action_cond_np = make_action_condition_np(action_chunk, mean=mean, std=std)
                    s = obs.context_tokens.to(device, non_blocking=True)
                    c = obs.task_emb.to(device, non_blocking=True)
                    context_rgb = obs.context_rgb.to(device, non_blocking=True) if obs.context_rgb is not None else None
                    action_cond = torch.from_numpy(action_cond_np).unsqueeze(0).to(device, non_blocking=True)
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
                    if "rgb" not in out:
                        raise RuntimeError("model output has no rgb")
                    pred_frame = _tensor_to_uint8(out["rgb"][0, 0], size=int(args.video_size))
                    rollout_frames.append(pred_frame)
                    frame_window.append(pred_frame)
                    if args.log_every and (step + 1) % int(args.log_every) == 0:
                        print(f"{suite} episode={spec.episode_index} camera={camera_name} step={step + 1}/{n_steps}", flush=True)

                base_episode = f"{suite}_{spec.episode_index:03d}"
                gt_path = out_dir / _video_name(run_id, base_episode, "gt", camera_name)
                recon_path = out_dir / _video_name(run_id, base_episode, "gt_recons", camera_name)
                inf_path = out_dir / _video_name(run_id, base_episode, "inf", camera_name)
                _save_video(gt_path, gt_frames, fps=int(args.fps))
                _save_video(recon_path, gt_frames, fps=int(args.fps))
                _save_video(inf_path, rollout_frames, fps=int(args.fps))
                suite_rows.append(
                    {
                        "trajectory": spec.source_path,
                        "camera": camera_name,
                        "status": "ready",
                        "num_frames": len(gt_frames),
                        "gt": str(gt_path),
                        "gt_recons": str(recon_path),
                        "inf": str(inf_path),
                    }
                )
        report["suites"][suite] = {
            "requested_trajectories": len(specs),
            "ready_videos": sum(1 for row in suite_rows if row["status"] == "ready"),
            "rows": suite_rows,
        }

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in report.items() if k != "suites"}, indent=2, sort_keys=True))
    for suite, info in report["suites"].items():
        print(f"{suite}: requested={info['requested_trajectories']} ready_videos={info['ready_videos']}")
    return report


def _parse_suites(value: str) -> list[str]:
    return [_suite_key(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare-val", help="Replay official val trajectories into WorldVLA processed_data format.")
    prep.add_argument("--official_root", type=Path, default=Path("/data/Minko/external/world_model_eval_sources/WorldVLA/rynnvla-002"))
    prep.add_argument("--raw_data_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO/datasets"))
    prep.add_argument("--processed_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO/processed_data_worldvla_val"))
    prep.add_argument("--suites", type=_parse_suites, default=["10", "goal", "object", "spatial"])
    prep.add_argument("--resolution", type=int, default=512)
    prep.add_argument("--skip_existing", action="store_true")
    prep.add_argument("--no_state_render_fallback", action="store_false", dest="state_render_fallback", default=True)
    prep.add_argument("--rank", type=int, default=None)
    prep.add_argument("--world_size", type=int, default=None)
    prep.add_argument("--limit", type=int, default=0)
    prep.add_argument("--out_summary", type=Path, required=True)
    prep.set_defaults(func=prepare_val_trajectories)

    exp = sub.add_parser("export-videos", help="Export WM3D rollouts as official WorldVLA-format MP4 files.")
    exp.add_argument("--official_root", type=Path, default=Path("/data/Minko/external/world_model_eval_sources/WorldVLA/rynnvla-002"))
    exp.add_argument("--processed_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO/processed_data_worldvla_val"))
    exp.add_argument("--suites", type=_parse_suites, default=["10", "goal", "object", "spatial"])
    exp.add_argument("--ckpt", type=Path, required=True)
    exp.add_argument("--cfg", type=Path, default=None)
    exp.add_argument("--out_dir", type=Path, required=True)
    exp.add_argument("--out_summary", type=Path, required=True)
    exp.add_argument("--device", default="cuda:0")
    exp.add_argument("--qwen_device", default=None)
    exp.add_argument("--task_cache_dir", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3/online_taskemb"))
    exp.add_argument("--allow_zero_task_fallback", action="store_true")
    exp.add_argument("--vggt_image_size", type=int, default=224)
    exp.add_argument("--context_rgb_size", type=int, default=256)
    exp.add_argument("--video_size", type=int, default=256)
    exp.add_argument("--fps", type=int, default=30)
    exp.add_argument("--rank", type=int, default=None)
    exp.add_argument("--world_size", type=int, default=None)
    exp.add_argument("--limit", type=int, default=0)
    exp.add_argument("--max_steps", type=int, default=0)
    exp.add_argument("--run_id", default=None)
    exp.add_argument("--log_every", type=int, default=25)
    exp.set_defaults(func=export_worldvla_videos)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
