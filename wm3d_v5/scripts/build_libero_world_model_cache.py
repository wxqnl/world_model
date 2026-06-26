#!/usr/bin/env python3
"""Build WM3D cache records for LIBERO world-model SFT.

The split follows the public WorldVLA/RynnVLA conversation-generation script:
sort tasks lexicographically, keep the first ceil(90%) tasks, then keep the
first ceil(90%) trajectories within each task for training. Official val-ind
trajectory IDs are also excluded as a guard against benchmark leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from wm3d_v3.data.manifest import OXEClipRecord, read_manifest, write_manifest
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder

from scripts.cache_geom_utils import (
    atomic_savez_compressed,
    frame_count_npy,
    validate_actions_npy,
    validate_geom_npz,
    validate_pooled_npy,
    validate_qwen_npy,
    validate_rgb_npy,
)


SUITE_TO_LIBERO = {
    "10": "libero_10",
    "long": "libero_10",
    "goal": "libero_goal",
    "object": "libero_object",
    "spatial": "libero_spatial",
}

CAMERA_TO_OBS = {
    "front": "agentview_rgb",
    "wrist": "eye_in_hand_rgb",
}

CAMERA_TO_PROCESSED_DIR = {
    "front": "imgs_third_view",
    "wrist": "imgs_wrist",
}


@dataclass(frozen=True)
class TrainTrajectory:
    suite_name: str
    task_name: str
    demo_key: str
    task_text: str

    @property
    def trj_name(self) -> str:
        return f"trj_{int(self.demo_key.split('_')[-1])}"


def suite_key(value: str) -> str:
    key = str(value).strip().lower()
    if key not in SUITE_TO_LIBERO:
        raise ValueError(f"unsupported suite {value!r}; expected one of {sorted(SUITE_TO_LIBERO)}")
    return "10" if key == "long" else key


def suite_label_from_libero(suite_name: str) -> str:
    if suite_name == "libero_10":
        return "10"
    if suite_name.startswith("libero_"):
        return suite_name[len("libero_") :]
    raise ValueError(f"unsupported LIBERO suite name: {suite_name}")


def safe_id(clip_id: str) -> str:
    return str(clip_id).replace("/", "__")


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npy")
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def resize_image_batch(imgs: np.ndarray, size: int) -> torch.Tensor:
    """[T,H,W,3] uint8 -> [T,3,size,size] float32 in [0,1]."""
    imgs = np.asarray(imgs, dtype=np.uint8)
    t = torch.from_numpy(np.array(imgs, copy=True)).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-2:] != (int(size), int(size)):
        t = F.interpolate(t, size=(int(size), int(size)), mode="bilinear", align_corners=False, antialias=True)
    return t


def _is_noop(action: np.ndarray, prev_action: np.ndarray | None = None, threshold: float = 1e-4) -> bool:
    action = np.asarray(action)
    if prev_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1])


def task_name_from_hdf5(path: Path) -> str:
    name = path.name
    suffix = "_demo.hdf5"
    if not name.endswith(suffix):
        raise ValueError(f"unexpected LIBERO demo filename: {path}")
    return name[: -len(suffix)]


def parse_clip_id(clip_id: str) -> tuple[str, str, str, str]:
    parts = str(clip_id).split("/")
    if len(parts) != 4:
        raise ValueError(f"expected clip_id suite/task/demo/camera, got {clip_id}")
    suite_name, task_name, demo_key, camera = parts
    if camera not in CAMERA_TO_OBS:
        raise ValueError(f"unknown camera {camera!r} in {clip_id}")
    return suite_name, task_name, demo_key, camera


def load_task_texts(suites: list[str]) -> dict[tuple[str, str], str]:
    from libero.libero import benchmark

    out: dict[tuple[str, str], str] = {}
    for suite in suites:
        suite_name = SUITE_TO_LIBERO[suite_key(suite)]
        bench = benchmark.get_benchmark_dict()[suite_name]()
        for task_id in range(bench.n_tasks):
            task = bench.get_task(task_id)
            out[(suite_name, task.name)] = str(task.language)
    return out


def official_val_ids(official_root: Path) -> set[tuple[str, str, str]]:
    root = official_root / "exps_libero_world_model"
    if not root.exists():
        raise FileNotFoundError(f"official WorldVLA val-ind root not found: {root}")
    val: set[tuple[str, str, str]] = set()
    for key in ("10", "goal", "object", "spatial"):
        path = root / f"{key}_val_ind_trajectory_paths.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for item in data:
            parts = Path(str(item)).parts
            dataset_idx = None
            for idx, part in enumerate(parts):
                if part.startswith("libero_") and "_image_state_action_t_" in part:
                    dataset_idx = idx
                    break
            if dataset_idx is None or dataset_idx + 2 >= len(parts):
                continue
            task_name = parts[dataset_idx + 1]
            trj_name = parts[dataset_idx + 2]
            match = re.fullmatch(r"trj_(\d+)", trj_name)
            if match:
                val.add((SUITE_TO_LIBERO[key], task_name, f"demo_{int(match.group(1))}"))
    return val


def ensure_worldvla_on_path(official_root: Path) -> None:
    for sub in (official_root, official_root / "libero_util"):
        text = str(sub)
        if text not in sys.path:
            sys.path.insert(0, text)


def filtered_demo_arrays(h5_path: Path, demo_key: str, camera: str) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(h5_path, "r") as h5:
        demo = h5["data"][demo_key]
        actions_raw = np.asarray(demo["actions"][()], dtype=np.float32)
        frames_raw = np.asarray(demo["obs"][CAMERA_TO_OBS[camera]][()], dtype=np.uint8)
    keep: list[int] = []
    actions: list[np.ndarray] = []
    noops = 0
    for idx, action in enumerate(actions_raw):
        prev_action = actions[-1] if actions else None
        if _is_noop(action, prev_action):
            noops += 1
            continue
        keep.append(idx)
        actions.append(np.asarray(action[:7], dtype=np.float32))
    if not keep:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0, 1, 1, 3), dtype=np.uint8), int(noops)
    acts = np.stack(actions, axis=0).astype(np.float32)
    # Match the WorldVLA LIBERO preprocessing convention.
    imgs = np.asarray(frames_raw[keep][:, ::-1, ::-1, :], dtype=np.uint8)
    return acts, imgs, int(noops)


def numeric_file_order(path: Path, prefix: str, suffix: str) -> list[Path]:
    def key(p: Path) -> int:
        match = re.fullmatch(re.escape(prefix) + r"(\d+)" + re.escape(suffix), p.name)
        return int(match.group(1)) if match else -1

    return sorted((p for p in path.glob(f"{prefix}*{suffix}") if key(p) >= 0), key=key)


def processed_dataset_dir(suite_name: str, resolution: int) -> str:
    return f"{suite_name}_image_state_action_t_{int(resolution)}"


def processed_traj_dir(processed_root: Path, suite_name: str, task_name: str, demo_key: str, resolution: int) -> Path:
    trj_name = f"trj_{int(str(demo_key).split('_')[-1])}"
    return processed_root / processed_dataset_dir(suite_name, resolution) / task_name / trj_name


def processed_demo_arrays(
    processed_root: Path,
    suite_name: str,
    task_name: str,
    demo_key: str,
    camera: str,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    trj_dir = processed_traj_dir(processed_root, suite_name, task_name, demo_key, resolution)
    action_files = numeric_file_order(trj_dir / "action", "action_", ".npy")
    image_files = numeric_file_order(trj_dir / CAMERA_TO_PROCESSED_DIR[camera], "image_", ".png")
    if not action_files or not image_files:
        raise FileNotFoundError(f"processed trajectory missing actions/images: {trj_dir}")
    n = min(len(action_files), len(image_files))
    actions = np.stack([np.asarray(np.load(path), dtype=np.float32).reshape(-1)[:7] for path in action_files[:n]], axis=0)
    imgs = np.stack([np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in image_files[:n]], axis=0)
    return actions.astype(np.float32), imgs.astype(np.uint8), 0


def filtered_action_count(h5_path: Path, demo_key: str) -> tuple[int, int]:
    with h5py.File(h5_path, "r") as h5:
        actions_raw = np.asarray(h5["data"][demo_key]["actions"][()], dtype=np.float32)
    actions: list[np.ndarray] = []
    noops = 0
    for action in actions_raw:
        prev_action = actions[-1] if actions else None
        if _is_noop(action, prev_action):
            noops += 1
            continue
        actions.append(np.asarray(action[:7], dtype=np.float32))
    return len(actions), int(noops)


def unique_trajectories_from_manifest(path: Path) -> list[TrainTrajectory]:
    records = read_manifest(path)
    out: dict[tuple[str, str, str], TrainTrajectory] = {}
    for record in records:
        suite_name, task_name, demo_key, _camera = parse_clip_id(record.clip_id)
        key = (suite_name, task_name, demo_key)
        out.setdefault(
            key,
            TrainTrajectory(
                suite_name=suite_name,
                task_name=task_name,
                demo_key=demo_key,
                task_text=str(record.task_text or task_name.replace("_", " ")),
            ),
        )
    return [out[key] for key in sorted(out)]


def save_png(path: Path, frame: np.ndarray) -> None:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def replay_one_trajectory(
    traj: TrainTrajectory,
    args: argparse.Namespace,
    env_cache: dict[str, object],
    task_maps: dict[str, dict[str, object]],
) -> dict:
    ensure_worldvla_on_path(args.official_root)
    from libero_util.libero_utils import get_libero_dummy_action, get_libero_env
    import robosuite.utils.transform_utils as T

    out_trj = processed_traj_dir(
        args.processed_root,
        traj.suite_name,
        traj.task_name,
        traj.demo_key,
        args.resolution,
    )
    meta_path = out_trj / "metadata.json"
    if meta_path.exists() and args.skip_existing:
        return {"trajectory": f"{traj.suite_name}/{traj.task_name}/{traj.demo_key}", "status": "skipped_existing", "out": str(out_trj)}

    task_map = task_maps[traj.suite_name]
    if traj.task_name not in task_map:
        raise KeyError(f"{traj.task_name} not found in task map for {traj.suite_name}")
    task = task_map[traj.task_name]
    env_key = f"{traj.suite_name}/{traj.task_name}"
    if env_key not in env_cache:
        env_cache[env_key] = get_libero_env(task, resolution=int(args.resolution))[0]
    env = env_cache[env_key]

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
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    raw_hdf5 = args.raw_data_root / traj.suite_name / f"{traj.task_name}_demo.hdf5"
    with h5py.File(raw_hdf5, "r") as h5:
        demo = h5["data"][traj.demo_key]
        orig_actions = demo["actions"][()]
        orig_states = demo["states"][()]
        orig_robot_states = demo["robot_states"][()]

    env.reset()
    env.set_init_state(orig_states[0])
    obs = None
    done = False
    for _ in range(10):
        obs, _reward, done, _info = env.step(get_libero_dummy_action())

    states = []
    actions = []
    ee_states = []
    gripper_states = []
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
        actions.append(np.asarray(action[:7], dtype=np.float32))
        gripper_states.append(obs["robot0_gripper_qpos"])
        ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))
        front_images.append(obs["agentview_image"])
        wrist_images.append(obs["robot0_eye_in_hand_image"])
        obs, _reward, done, _info = env.step(action.tolist())

    ready_status = "ready"
    if (not bool(done)) and bool(args.state_render_fallback):
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
            actions.append(np.asarray(action[:7], dtype=np.float32))
            gripper_states.append(obs["robot0_gripper_qpos"])
            ee_states.append(np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]))))
            front_images.append(obs["agentview_image"])
            wrist_images.append(obs["robot0_eye_in_hand_image"])
        done = bool(actions)
        ready_status = "ready_state_render_fallback"

    if not bool(done):
        return {
            "trajectory": f"{traj.suite_name}/{traj.task_name}/{traj.demo_key}",
            "status": "failed_replay",
            "out": str(out_trj),
            "num_actions": len(actions),
            "num_noops": int(noops),
        }
    if len(actions) < int(args.min_frames):
        return {
            "trajectory": f"{traj.suite_name}/{traj.task_name}/{traj.demo_key}",
            "status": "skipped_short",
            "out": str(out_trj),
            "num_actions": len(actions),
            "num_noops": int(noops),
        }

    for j, action in enumerate(actions):
        np.save(dirs["action"] / f"action_{j}.npy", np.asarray(action, dtype=np.float32))
        ee = np.asarray(ee_states[j])
        grip = np.asarray(gripper_states[j])
        np.save(dirs["ee_state"] / f"ee_state_{j}.npy", ee)
        np.save(dirs["gripper_state"] / f"gripper_state_{j}.npy", grip)
        np.save(dirs["eef_gripper_state"] / f"eef_gripper_state_{j}.npy", np.concatenate([ee, grip]))
        np.save(dirs["robot_state"] / f"robot_state_{j}.npy", np.asarray(robot_states[j]))
        save_png(dirs["imgs_third_view"] / f"image_{j}.png", np.asarray(front_images[j])[::-1, ::-1])
        save_png(dirs["imgs_wrist"] / f"image_{j}.png", np.asarray(wrist_images[j])[::-1, ::-1])

    meta = {
        "trajectory": f"{traj.suite_name}/{traj.task_name}/{traj.demo_key}",
        "suite": traj.suite_name,
        "task_name": traj.task_name,
        "trj_name": traj.trj_name,
        "num_actions": len(actions),
        "num_noops": int(noops),
        "resolution": int(args.resolution),
        "status": ready_status,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return {"trajectory": meta["trajectory"], "status": ready_status, "out": str(out_trj), "num_actions": len(actions)}


def run_prepare_replay(args: argparse.Namespace) -> dict:
    ensure_worldvla_on_path(args.official_root)
    from libero.libero import benchmark

    trajectories = unique_trajectories_from_manifest(args.manifest)
    if args.limit_records:
        trajectories = trajectories[: int(args.limit_records)]
    shard_trajectories = [traj for idx, traj in enumerate(trajectories) if idx % int(args.world) == int(args.shard)]

    suite_names = sorted({traj.suite_name for traj in trajectories})
    task_maps: dict[str, dict[str, object]] = {}
    for suite_name in suite_names:
        bench = benchmark.get_benchmark_dict()[suite_name]()
        task_maps[suite_name] = {bench.get_task(task_id).name: bench.get_task(task_id) for task_id in range(bench.n_tasks)}

    env_cache: dict[str, object] = {}
    counts: dict[str, int] = defaultdict(int)
    rows = []
    try:
        pbar = tqdm(shard_trajectories, desc=f"replay shard {args.shard}/{args.world}", dynamic_ncols=True)
        for traj in pbar:
            try:
                row = replay_one_trajectory(traj, args, env_cache, task_maps)
            except Exception as exc:
                row = {
                    "trajectory": f"{traj.suite_name}/{traj.task_name}/{traj.demo_key}",
                    "status": "error",
                    "error": str(exc),
                }
                print(f"[replay-error] {row['trajectory']}: {exc}", flush=True)
                if args.verbose:
                    traceback.print_exc()
            rows.append(row)
            counts[str(row["status"])] += 1
            pbar.set_postfix(dict(counts))
    finally:
        for env in env_cache.values():
            try:
                env.close()
            except Exception:
                pass
    summary = {
        "mode": "prepare_replay",
        "manifest": str(args.manifest),
        "processed_root": str(args.processed_root),
        "resolution": int(args.resolution),
        "shard": int(args.shard),
        "world": int(args.world),
        "trajectories_seen": len(shard_trajectories),
        "counts": dict(counts),
        "rows": rows,
    }
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True))
    if counts.get("error", 0) or counts.get("failed_replay", 0) or counts.get("skipped_short", 0):
        raise SystemExit(f"replay preparation failed or skipped trajectories: {dict(counts)}")
    return summary


def build_records(args: argparse.Namespace) -> tuple[list[OXEClipRecord], dict]:
    suites = [suite_key(s) for s in args.suites.split(",") if s.strip()]
    task_texts = load_task_texts(suites)
    official_val = official_val_ids(args.official_root)
    records: list[OXEClipRecord] = []
    summary = {
        "protocol": "worldvla_rynnvla_train_split",
        "raw_data_root": str(args.raw_data_root),
        "official_root": str(args.official_root),
        "min_frames": int(args.min_frames),
        "records": 0,
        "trajectories": 0,
        "skipped": defaultdict(int),
        "suites": {},
    }

    for suite in suites:
        suite_name = SUITE_TO_LIBERO[suite]
        suite_dir = args.raw_data_root / suite_name
        task_files = sorted(suite_dir.glob("*_demo.hdf5"))
        split_task_idx = math.ceil(len(task_files) * 0.9)
        suite_info = {
            "tasks_total": len(task_files),
            "tasks_train": split_task_idx,
            "trajectories_train": 0,
            "records_train": 0,
            "skipped": defaultdict(int),
        }
        for task_idx, h5_path in enumerate(task_files):
            task_name = task_name_from_hdf5(h5_path)
            if task_idx >= split_task_idx:
                suite_info["skipped"]["ood_task"] += 1
                continue
            with h5py.File(h5_path, "r") as h5:
                demo_keys = sorted(h5["data"].keys())
            split_demo_idx = math.ceil(len(demo_keys) * 0.9)
            for demo_pos, demo_key in enumerate(demo_keys):
                if demo_pos >= split_demo_idx:
                    suite_info["skipped"]["val_ind_split"] += 1
                    continue
                if (suite_name, task_name, demo_key) in official_val:
                    suite_info["skipped"]["official_val_guard"] += 1
                    continue
                try:
                    n_frames, _noops = filtered_action_count(h5_path, demo_key)
                except Exception:
                    suite_info["skipped"]["read_error"] += 1
                    if args.verbose:
                        traceback.print_exc()
                    continue
                if n_frames < int(args.min_frames):
                    suite_info["skipped"]["short_after_noop_filter"] += 1
                    continue
                task_text = task_texts.get((suite_name, task_name), task_name.replace("_", " "))
                for camera in CAMERA_TO_OBS:
                    clip_id = f"{suite_name}/{task_name}/{demo_key}/{camera}"
                    records.append(
                        OXEClipRecord(
                            clip_id=clip_id,
                            dataset=f"{suite_name}_{camera}",
                            tar_path="",
                            pickle_member="",
                            n_frames=int(n_frames),
                            fps=30,
                            robot="libero_panda",
                            task_text=task_text,
                            action_dim=7,
                            action_kind="libero_delta_xyz+rpy+gripper",
                            image_keys=[camera],
                            repeat_weight=1.0,
                        )
                    )
                suite_info["trajectories_train"] += 1
                suite_info["records_train"] += len(CAMERA_TO_OBS)
        summary["suites"][suite] = {
            **suite_info,
            "skipped": dict(suite_info["skipped"]),
        }
        summary["trajectories"] += suite_info["trajectories_train"]
        summary["records"] += suite_info["records_train"]
    summary["skipped"] = dict(summary["skipped"])
    return records, summary


def cache_complete(cache_root: Path, cid: str, *, need_qwen: bool) -> bool:
    pooled = cache_root / "vggt_pooled" / f"{cid}.npy"
    actions = cache_root / "actions" / f"{cid}.npy"
    n = frame_count_npy(actions)
    if n is None:
        return False
    if not validate_actions_npy(actions, expected_frames=n):
        return False
    if not validate_rgb_npy(cache_root / "rgb_256" / f"{cid}.npy", expected_frames=n):
        return False
    if not validate_pooled_npy(pooled, expected_frames=n):
        return False
    if not validate_geom_npz(cache_root / "vggt_geom" / f"{cid}.npz", expected_frames=n, require_geom_extra=False):
        return False
    if need_qwen and not validate_qwen_npy(cache_root / "qwen_taskemb" / f"{cid}.npy"):
        return False
    return True


def cache_one_record(
    record: OXEClipRecord,
    args: argparse.Namespace,
    enc: VGGTEncoder | None,
    qwen: QwenVLEmbed | None,
) -> str:
    suite_name, task_name, demo_key, camera = parse_clip_id(record.clip_id)
    cid = safe_id(record.clip_id)
    cache_root = args.cache_root
    need_qwen = qwen is not None
    if not args.force and cache_complete(cache_root, cid, need_qwen=need_qwen):
        return "skipped_existing"

    if args.source == "processed":
        actions, imgs, _noops = processed_demo_arrays(
            args.processed_root,
            suite_name,
            task_name,
            demo_key,
            camera,
            args.resolution,
        )
    else:
        h5_path = args.raw_data_root / suite_name / f"{task_name}_demo.hdf5"
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        actions, imgs, _noops = filtered_demo_arrays(h5_path, demo_key, camera)
    if actions.shape[0] < int(args.min_frames):
        return "skipped_short"

    pool_path = cache_root / "vggt_pooled" / f"{cid}.npy"
    geom_path = cache_root / "vggt_geom" / f"{cid}.npz"
    rgb_path = cache_root / "rgb_256" / f"{cid}.npy"
    act_path = cache_root / "actions" / f"{cid}.npy"
    qwen_path = cache_root / "qwen_taskemb" / f"{cid}.npy"

    n = int(actions.shape[0])
    if args.force or not validate_actions_npy(act_path, expected_frames=n):
        atomic_save_npy(act_path, actions.astype(np.float32))

    if args.force or not validate_rgb_npy(rgb_path, expected_frames=n):
        rgb256 = resize_image_batch(imgs, 256)
        rgb256_u8 = (rgb256.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        atomic_save_npy(rgb_path, rgb256_u8)

    need_pooled = args.force or not validate_pooled_npy(pool_path, expected_frames=n)
    need_geom = args.force or not validate_geom_npz(geom_path, expected_frames=n, require_geom_extra=False)
    if need_pooled or need_geom:
        if enc is None:
            raise RuntimeError("VGGT encoder is required for pooled/geom cache")
        frames_224 = resize_image_batch(imgs, 224)
        pooled_chunks: list[np.ndarray] = []
        depth_chunks: list[np.ndarray] = []
        for start in range(0, n, int(args.batch_frames)):
            chunk = frames_224[start : start + int(args.batch_frames)].unsqueeze(0).to("cuda")
            with torch.inference_mode():
                out = enc(chunk)
            if need_pooled:
                pooled_chunks.append(out["pooled"][0].detach().cpu().numpy().astype(np.float16))
            if need_geom:
                if "depth" not in out:
                    raise RuntimeError(f"VGGT did not return depth for {record.clip_id}")
                depth_chunks.append(out["depth"][0].detach().cpu().numpy().astype(np.float16))
        if need_pooled:
            atomic_save_npy(pool_path, np.concatenate(pooled_chunks, axis=0).astype(np.float16))
        if need_geom:
            atomic_savez_compressed(geom_path, depth=np.concatenate(depth_chunks, axis=0).astype(np.float16))

    if qwen is not None and (args.force or not validate_qwen_npy(qwen_path)):
        first_img = Image.fromarray(imgs[0])
        emb = qwen.embed(record.task_text or "robot manipulation", first_img)
        atomic_save_npy(qwen_path, emb.detach().cpu().numpy().astype(np.float16))

    return "cached"


def run_cache(args: argparse.Namespace) -> dict:
    records = read_manifest(args.manifest)
    if args.limit_records:
        records = records[: int(args.limit_records)]
    shard_records = [r for idx, r in enumerate(records) if idx % int(args.world) == int(args.shard)]
    for subdir in ("vggt_pooled", "vggt_geom", "rgb_256", "actions", "qwen_taskemb"):
        (args.cache_root / subdir).mkdir(parents=True, exist_ok=True)
    enc = None if args.only_qwen else VGGTEncoder(device="cuda", return_depth=True, return_geom_extra=False)
    qwen = None if args.skip_qwen else QwenVLEmbed()
    counts: dict[str, int] = defaultdict(int)
    pbar = tqdm(shard_records, desc=f"cache shard {args.shard}/{args.world}", dynamic_ncols=True)
    for record in pbar:
        try:
            status = cache_one_record(record, args, enc, qwen)
        except Exception as exc:
            status = "error"
            print(f"[cache-error] {record.clip_id}: {exc}", flush=True)
            if args.verbose:
                traceback.print_exc()
        counts[status] += 1
        pbar.set_postfix(dict(counts))
    summary = {
        "mode": "cache",
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "shard": int(args.shard),
        "world": int(args.world),
        "records_seen": len(shard_records),
        "counts": dict(counts),
    }
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if counts.get("missing", 0):
        raise SystemExit(f"cache validation failed: {counts['missing']} records are missing or incomplete")
    return summary


def run_validate(args: argparse.Namespace) -> dict:
    records = read_manifest(args.manifest)
    if args.limit_records:
        records = records[: int(args.limit_records)]
    counts: dict[str, int] = defaultdict(int)
    for record in tqdm(records, desc="validate cache", dynamic_ncols=True):
        cid = safe_id(record.clip_id)
        counts["complete" if cache_complete(args.cache_root, cid, need_qwen=not args.skip_qwen) else "missing"] += 1
    summary = {
        "mode": "validate",
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "records_seen": len(records),
        "counts": dict(counts),
    }
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if counts.get("error", 0) or counts.get("skipped_short", 0):
        raise SystemExit(f"cache shard failed or skipped records: {dict(counts)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("manifest", "prepare_replay", "cache", "all", "validate"), default="all")
    parser.add_argument("--raw_data_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO/datasets"))
    parser.add_argument("--official_root", type=Path, default=Path("/data/Minko/external/world_model_eval_sources/WorldVLA/rynnvla-002"))
    parser.add_argument("--manifest", type=Path, default=Path("/data/Minko/world_model/wm3d_v5/manifests/libero_world_model_sft_train_v1.jsonl"))
    parser.add_argument("--cache_root", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v5_libero_world_sft_v1"))
    parser.add_argument("--processed_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO/processed_data_worldvla_train"))
    parser.add_argument("--source", choices=("raw_obs", "processed"), default="processed")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out_summary", type=Path, default=None)
    parser.add_argument("--suites", type=str, default="10,goal,object,spatial")
    parser.add_argument("--min_frames", type=int, default=24)
    parser.add_argument("--batch_frames", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--limit_records", type=int, default=0)
    parser.add_argument("--skip_qwen", action="store_true")
    parser.add_argument("--only_qwen", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--state_render_fallback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.only_qwen and args.skip_qwen:
        raise SystemExit("--only_qwen and --skip_qwen are mutually exclusive")

    if args.mode in {"manifest", "all"}:
        records, summary = build_records(args)
        write_manifest(args.manifest, records)
        summary["manifest"] = str(args.manifest)
        summary["records_written"] = len(records)
        summary_path = args.out_summary or args.manifest.with_suffix(".summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, indent=2, sort_keys=True))

    if args.mode == "prepare_replay":
        run_prepare_replay(args)
    elif args.mode in {"cache", "all"}:
        run_cache(args)
    elif args.mode == "validate":
        run_validate(args)


if __name__ == "__main__":
    main()
