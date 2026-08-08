#!/usr/bin/env python3
"""Paired real-simulator closed-loop evaluation of V7 Stage0 vs Stage1-P."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cache_robocasa365_v7_compact import _encode_clip, _load_adapter, _load_codec  # noqa: E402
from scripts.generate_robocasa_same_root_cf import (  # noqa: E402
    _episode,
    _make_env,
    _render_views,
    _reset_episode,
    array_bytes_equal,
)
from scripts.harvest_wm3d_v7_stage1_planner_candidates import propose_h32  # noqa: E402
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder  # noqa: E402
from wm3d_v3.stage1.action_window_geometry import VGGT_MODEL_REVISION  # noqa: E402
from wm3d_v3.stage1_planner.action_bridge import canonical_model_actions_to_simulator  # noqa: E402
from wm3d_v3.stage1_planner.dataset import SCHEMA as DATA_SCHEMA  # noqa: E402
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig  # noqa: E402
from wm3d_v3.stage1_planner.train import _load_stage1_state, sha256_file  # noqa: E402
from wm3d_v3.training.train import build_model, config_sha256, load_train_config  # noqa: E402


SCHEMA = "wm3d_v7_stage1_planner_closed_loop_eval_v1"


def _dequantize(encoded: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(encoded["codes"], dtype=np.int8).astype(np.float32) * np.asarray(
        encoded["scale"], dtype=np.float32
    )


def _encode_context(
    history: list[np.ndarray],
    *,
    encoder: VGGTEncoder,
    codec,
) -> tuple[np.ndarray, np.ndarray]:
    if not history:
        raise ValueError("visual history is empty")
    padded = [history[0]] * max(0, 16 - len(history)) + history[-16:]
    if len(padded) != 16:
        raise RuntimeError("T16 context padding failed")
    anchor = _encode_clip(
        [frame[0] for frame in padded],
        encoder=encoder,
        codec=codec,
        batch_frames=16,
        keep_geometry=False,
    )
    wrist = _encode_clip(
        [frame[2] for frame in padded],
        encoder=encoder,
        codec=codec,
        batch_frames=16,
        keep_geometry=False,
    )
    return _dequantize(anchor), _dequantize(wrist)


def _reset_to_root(env, episode: dict, t0: int, *, height: int, width: int) -> list[np.ndarray]:
    _reset_episode(env, episode)
    visual = [_render_views(env, height, width)]
    for offset in range(t0):
        env.step(episode["actions"][offset])
        if (offset + 1) % 4 == 0:
            visual.append(_render_views(env, height, width))
    actual = np.asarray(env.sim.get_state().flatten())
    if not array_bytes_equal(actual, episode["states"][t0]):
        raise RuntimeError("paired evaluation could not reconstruct the exact root")
    return visual[-16:]


def _condition_actions(actions: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    conditioned = actions.copy()
    conditioned[..., :6] = (conditioned[..., :6] - mean[None, None]) / std[None, None]
    conditioned[..., 6] = np.clip((conditioned[..., 6] + 1.0) * 0.5, 0.0, 1.0)
    return torch.from_numpy(conditioned)


@torch.inference_mode()
def _run_path(
    *,
    mode: str,
    env,
    episode: dict,
    t0: int,
    task: torch.Tensor,
    action_history_physical: np.ndarray,
    baseline_model,
    planning_system: NativePlanningSystem,
    encoder: VGGTEncoder,
    codec,
    adapter,
    mean_np: np.ndarray,
    std_np: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    max_model_steps: int,
    seed: int,
    height: int,
    width: int,
) -> dict:
    if mode not in {"stage0_direct", "stage1_planner"}:
        raise ValueError(mode)
    visual = _reset_to_root(env, episode, t0, height=height, width=width)
    history_physical = np.asarray(action_history_physical, dtype=np.float32).copy()
    history_policy = history_physical.copy()
    history_policy[:, 6] = np.clip((history_policy[:, 6] + 1.0) * 0.5, 0.0, 1.0)
    low, high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
    # Non-arm fields are held at the last observed command.  No future
    # demonstration action is read by either policy.
    observed_template = np.repeat(episode["actions"][max(0, t0 - 1)][None], 4, axis=0)
    success = bool(env._check_success())
    done = False
    selected_roles: list[int] = []
    executed: list[list[float]] = []
    for model_step in range(max_model_steps):
        if success or done:
            break
        anchor_np, wrist_np = _encode_context(visual, encoder=encoder, codec=codec)
        context = torch.from_numpy(anchor_np).to(device)[None]
        wrist = torch.from_numpy(wrist_np).to(device)[None]
        history_tensor = torch.from_numpy(history_policy).to(device)[None]
        proposer = baseline_model if mode == "stage0_direct" else planning_system.world
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            candidates = propose_h32(
                proposer,
                context,
                wrist,
                task,
                history_tensor,
                mean,
                std,
                seed=seed + model_step,
            )
            if mode == "stage0_direct":
                selected = 0
            else:
                conditioned = _condition_actions(candidates, mean_np, std_np).to(device)[None]
                result = planning_system(
                    context,
                    task,
                    conditioned,
                    wrist=wrist,
                    view_mask=torch.ones((1, 16, 2), dtype=torch.bool, device=device),
                    score_planner=True,
                )
                selected = int(result["planner"]["score"][0].argmax().item())
        action = candidates[selected, 0]
        bridge = canonical_model_actions_to_simulator(
            action[None],
            adapter,
            source_hz=20.0,
            target_hz=5.0,
            template=observed_template,
            action_low=low,
            action_high=high,
        )
        for dense in bridge.simulator_actions:
            _obs, _reward, step_done, _info = env.step(dense)
            success = success or bool(env._check_success())
            done = done or bool(step_done)
            if success or done:
                break
        visual.append(_render_views(env, height, width))
        visual = visual[-16:]
        history_physical = np.concatenate((history_physical, action[None]), axis=0)[-4:]
        history_policy = history_physical.copy()
        history_policy[:, 6] = np.clip((history_policy[:, 6] + 1.0) * 0.5, 0.0, 1.0)
        selected_roles.append(selected)
        executed.append(action.astype(float).tolist())
    return {
        "success": success,
        "done": done,
        "model_steps": len(executed),
        "selected_candidate_indices": selected_roles,
        "executed_actions": executed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--overlay-sha256", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--max-roots", type=int, default=100)
    parser.add_argument("--max-model-steps", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_roots <= 0 or args.max_model_steps <= 0:
        raise SystemExit("closed-loop root/step limits must be positive")
    if sha256_file(args.overlay) != args.overlay_sha256:
        raise SystemExit("overlay SHA256 mismatch")
    cfg = load_train_config(args.cfg)
    phase_cfg = dict(cfg["planner_stage"])
    data_cfg = dict(cfg["planner_data"])
    source_path = Path(phase_cfg["source_checkpoint"])
    source_sha = sha256_file(source_path)
    if source_sha != phase_cfg["source_checkpoint_sha256"]:
        raise SystemExit("Stage0 checkpoint SHA256 mismatch")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    baseline = build_model(cfg)
    planning_world = build_model(cfg)
    for model in (baseline, planning_world):
        loaded = model.load_state_dict(source["model"], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError("Stage0 strict load was not clean")
    system = NativePlanningSystem(
        planning_world,
        Stage1SystemConfig(candidate_microbatch=1, activation_checkpointing=False),
    )
    overlay = torch.load(args.overlay, map_location="cpu", weights_only=False)
    if overlay.get("source_checkpoint_sha256") != source_sha:
        raise RuntimeError("overlay/source mismatch")
    _load_stage1_state(system, overlay["stage1_state"])
    device = torch.device(args.device)
    baseline.to(device).eval()
    system.to(device).eval()
    codec = _load_codec(Path(cfg["model"]["token_codec_checkpoint"]), device)
    encoder = VGGTEncoder(
        device=str(device),
        return_depth=False,
        return_depth_conf=False,
        return_geom_extra=False,
        model_revision=VGGT_MODEL_REVISION,
        local_files_only=True,
    )
    adapter = _load_adapter(
        Path("/data/Minko/world_model/wm3d_v7/manifests/audits/robocasa365_atomic_factual_action_v2.json"),
        allow_legacy_proof_audit=False,
    )
    with np.load(data_cfg["action_stats"], allow_pickle=False) as stats:
        mean_np = np.asarray(stats["mean"], dtype=np.float32)
        std_np = np.asarray(stats["std"], dtype=np.float32)
    mean = torch.from_numpy(mean_np).to(device).view(1, 1, 6)
    std = torch.from_numpy(std_np).to(device).view(1, 1, 6)
    rows = [
        json.loads(line)
        for line in Path(data_cfg["index"]).read_text().splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == args.split][: args.max_roots]
    if not rows or any(row.get("schema") != DATA_SCHEMA for row in rows):
        raise RuntimeError("closed-loop split is empty or has the wrong schema")

    results = []
    current_dataset = None
    env = None
    try:
        for ordinal, row in enumerate(rows):
            dataset = Path(row["source_dataset"])
            if current_dataset != dataset:
                if env is not None:
                    env.close()
                env, _metadata = _make_env(dataset, render_rgb=True)
                current_dataset = dataset
            episode = _episode(dataset, int(row["episode_id"]))
            with np.load(row["path"], allow_pickle=False) as archive:
                task = torch.from_numpy(np.asarray(archive["task_emb"], dtype=np.float32)).to(device)[None]
                history = np.asarray(archive["action_history_physical"], dtype=np.float32)
            common = {
                "env": env,
                "episode": episode,
                "t0": int(row["t0"]),
                "task": task,
                "action_history_physical": history,
                "baseline_model": baseline,
                "planning_system": system,
                "encoder": encoder,
                "codec": codec,
                "adapter": adapter,
                "mean_np": mean_np,
                "std_np": std_np,
                "mean": mean,
                "std": std,
                "device": device,
                "max_model_steps": args.max_model_steps,
                "seed": args.seed + ordinal * 1009,
                "height": args.height,
                "width": args.width,
            }
            direct = _run_path(mode="stage0_direct", **common)
            planner = _run_path(mode="stage1_planner", **common)
            results.append(
                {
                    "root_id": row["root_id"],
                    "task": row["task"],
                    "split_group": row["split_group"],
                    "stage0_direct": direct,
                    "stage1_planner": planner,
                }
            )
            print(json.dumps({"root_id": row["root_id"], "stage0": direct["success"], "stage1": planner["success"]}), flush=True)
    finally:
        if env is not None:
            env.close()
    stage0_rate = float(np.mean([row["stage0_direct"]["success"] for row in results]))
    stage1_rate = float(np.mean([row["stage1_planner"]["success"] for row in results]))
    report = {
        "schema": SCHEMA,
        "config": str(args.cfg.resolve()),
        "config_sha256": config_sha256(cfg),
        "source_checkpoint_sha256": source_sha,
        "overlay": str(args.overlay.resolve()),
        "overlay_sha256": args.overlay_sha256,
        "split": args.split,
        "roots": len(results),
        "stage0_direct_success": stage0_rate,
        "stage1_planner_success": stage1_rate,
        "success_uplift_vs_stage0": stage1_rate - stage0_rate,
        "paired_exact_roots": True,
        "planner_action_inputs": False,
        "future_demonstration_actions_used": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
