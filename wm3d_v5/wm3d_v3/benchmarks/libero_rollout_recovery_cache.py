"""Build WM3D action-policy recovery cache from failed LIBERO rollouts.

The cache mirrors `libero_expert_cache.py`, but the visual/lowdim context comes
from online rollout states while the target action chunk comes from the nearest
expert phase in the reference hdf5 trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image

from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _resolve_path(path: str | Path, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def _read_frame(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _expert_lowdim(h5: h5py.File, demo_id: str) -> np.ndarray:
    obs = h5["data"][demo_id]["obs"]
    return np.concatenate(
        [
            np.asarray(obs["ee_pos"], dtype=np.float32),
            np.asarray(obs["gripper_states"], dtype=np.float32),
            np.asarray(obs["joint_states"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _align_expert_index(
    rollout_lowdim: np.ndarray,
    expert_lowdim: np.ndarray,
    *,
    rollout_object_state: np.ndarray | None = None,
    expert_object_state: np.ndarray | None = None,
    object_state_weight: float = 0.0,
    rollout_step: int,
    rollout_steps: int,
    phase_prior_weight: float,
    min_idx: int = 0,
    max_idx: int | None = None,
) -> tuple[int, float]:
    scale = expert_lowdim.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    diff = (expert_lowdim - rollout_lowdim[None].astype(np.float32)) / scale[None]
    dist = np.sqrt(np.mean(diff * diff, axis=1))
    if (
        object_state_weight > 0
        and rollout_object_state is not None
        and expert_object_state is not None
        and len(expert_object_state) == len(expert_lowdim)
    ):
        obj_scale = expert_object_state.std(axis=0).astype(np.float32)
        obj_scale = np.maximum(obj_scale, 1e-4)
        obj_diff = (expert_object_state - rollout_object_state[None].astype(np.float32)) / obj_scale[None]
        obj_dist = np.sqrt(np.mean(obj_diff * obj_diff, axis=1))
        dist = dist + float(object_state_weight) * obj_dist
    if phase_prior_weight > 0:
        phase = float(rollout_step) / max(1.0, float(rollout_steps - 1))
        expert_phase = np.arange(len(expert_lowdim), dtype=np.float32) / max(1.0, float(len(expert_lowdim) - 1))
        dist = dist + float(phase_prior_weight) * np.abs(expert_phase - phase)
    lo = max(0, int(min_idx))
    hi = len(dist) - 1 if max_idx is None else min(len(dist) - 1, int(max_idx))
    if lo > 0:
        dist[:lo] = np.inf
    if hi + 1 < len(dist):
        dist[hi + 1 :] = np.inf
    if not np.isfinite(dist).any():
        return lo, float("inf")
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def _lowdim_distance(rollout_lowdim: np.ndarray, expert_lowdim: np.ndarray, idx: int) -> float:
    scale = expert_lowdim.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    diff = (expert_lowdim[int(idx)] - rollout_lowdim.astype(np.float32)) / scale
    return float(np.sqrt(np.mean(diff * diff)))


def _chunk(actions: np.ndarray, start: int, k: int) -> np.ndarray:
    start = min(max(0, int(start)), len(actions) - 1)
    out = actions[start : start + k].astype(np.float32)
    if len(out) < k:
        out = np.concatenate([out, np.repeat(out[-1:], k - len(out), axis=0)], axis=0)
    return out


def _context_frames(
    traces: list[dict[str, Any]],
    idx: int,
    *,
    T: int,
    root: Path,
) -> list[np.ndarray] | None:
    paths: list[Path] = []
    first_path = traces[0].get("frame_path")
    if not first_path:
        return None
    first_resolved = _resolve_path(str(first_path), root)
    for j in range(idx - T + 1, idx + 1):
        if j < 0:
            paths.append(first_resolved)
            continue
        frame_path = traces[j].get("frame_path")
        if not frame_path:
            return None
        resolved = _resolve_path(str(frame_path), root)
        if not resolved.exists():
            return None
        paths.append(resolved)
    return [_read_frame(path) for path in paths]


def _is_task1_put_cream_butter(text: str) -> bool:
    lowered = text.lower()
    return ("cream_cheese" in lowered or "cream cheese" in lowered) and "butter" in lowered and "basket" in lowered


def _stage_from_progress(progress: float) -> int:
    p = min(1.0, max(0.0, float(progress)))
    if p < 0.23:
        return 0
    if p < 0.52:
        return 1
    if p < 0.77:
        return 2
    return 3


def _entity_pos(named_poses: dict[str, Any] | None, entity: str) -> np.ndarray | None:
    if not isinstance(named_poses, dict):
        return None
    fields = named_poses.get(entity) or {}
    value = fields.get("pos")
    if value is None and entity == "robot0":
        value = fields.get("eef_pos")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr[:3] if arr.size >= 3 else None


def _plan_state_from_stage(
    stage: int,
    named_poses: dict[str, Any] | None,
    *,
    dim: int,
) -> np.ndarray:
    if dim < 8:
        raise ValueError(f"plan_state_dim must be >= 8, got {dim}")
    stage = min(3, max(0, int(stage)))
    target = 0 if stage < 2 else 1
    subgoal = 0 if stage in (0, 2) else 1
    out = np.zeros(int(dim), dtype=np.float32)
    out[stage] = 1.0
    out[4 + target] = 1.0
    out[6 + subgoal] = 1.0
    if dim >= 17:
        target_entity = "cream_cheese_1" if target == 0 else "butter_1"
        target_pos = _entity_pos(named_poses, target_entity)
        eef_pos = _entity_pos(named_poses, "robot0")
        basket_pos = _entity_pos(named_poses, "basket_1")
        if target_pos is not None and eef_pos is not None:
            out[8:11] = np.clip(target_pos[:3] - eef_pos[:3], -1.0, 1.0)
        if target_pos is not None and basket_pos is not None:
            out[11:14] = np.clip(basket_pos[:3] - target_pos[:3], -1.0, 1.0)
        if eef_pos is not None and basket_pos is not None:
            out[14:17] = np.clip(basket_pos[:3] - eef_pos[:3], -1.0, 1.0)
    return out


def _select_indices(n: int, stride: int, max_windows: int) -> list[int]:
    indices = list(range(0, n, max(1, int(stride))))
    if max_windows <= 0 or len(indices) <= max_windows:
        return indices
    if max_windows == 1:
        return [indices[0]]
    return [indices[round(i * (len(indices) - 1) / (max_windows - 1))] for i in range(max_windows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_json", type=Path, required=True)
    ap.add_argument("--expert_hdf5", type=Path, required=True)
    ap.add_argument("--demo_id", default="demo_0")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--action_stats", type=Path, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_windows", type=int, default=96)
    ap.add_argument("--min_rollout_step", type=int, default=0)
    ap.add_argument("--max_rollout_step", type=int, default=0)
    ap.add_argument("--min_expert_idx", type=int, default=0)
    ap.add_argument("--max_expert_idx", type=int, default=0)
    ap.add_argument("--plan_state_dim", type=int, default=8)
    ap.add_argument("--token_grid", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_device", default=None)
    ap.add_argument("--task_cache_dir", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3/libero_taskemb"))
    ap.add_argument("--allow_zero_task_fallback", action="store_true")
    ap.add_argument("--phase_prior_weight", type=float, default=0.05)
    ap.add_argument("--align_mode", choices=("nearest_lowdim", "time"), default="nearest_lowdim")
    ap.add_argument("--expert_object_state_npz", type=Path, default=None)
    ap.add_argument("--object_state_weight", type=float, default=0.0)
    ap.add_argument("--monotonic", action="store_true")
    ap.add_argument("--monotonic_slack", type=int, default=0)
    ap.add_argument("--max_align_distance", type=float, default=0.0)
    ap.add_argument("--sample_weight", type=float, default=4.0)
    ap.add_argument("--log_every", type=int, default=8)
    args = ap.parse_args()

    project_root = Path.cwd()
    rollout = _load_json(args.rollout_json)
    if not rollout.get("results"):
        raise RuntimeError(f"rollout has no results: {args.rollout_json}")
    episode = rollout["results"][0]
    traces = episode.get("step_trace") or []
    if not traces:
        raise RuntimeError(f"rollout has no step_trace: {args.rollout_json}")
    instruction = str(episode.get("instruction") or "robot manipulation")
    task_name = str(episode.get("task_name") or "")

    stats = np.load(args.action_stats)
    mean = stats["mean"][:6].astype(np.float32)
    std = np.maximum(stats["std"][:6].astype(np.float32), 1e-4)
    pos_rate = float(stats.get("pos_rate", np.asarray([0.5]))[0])

    with h5py.File(args.expert_hdf5, "r") as h5:
        expert_actions = np.asarray(h5["data"][args.demo_id]["actions"], dtype=np.float32)
        expert_lowdim = _expert_lowdim(h5, args.demo_id)
    expert_object_state = None
    if args.expert_object_state_npz is not None:
        ref = np.load(args.expert_object_state_npz)
        expert_object_state = np.asarray(ref["object_state"], dtype=np.float32)
        expert_lowdim_ref = np.asarray(ref["lowdim"], dtype=np.float32) if "lowdim" in ref.files else expert_lowdim
        n_ref = min(len(expert_actions), len(expert_object_state), len(expert_lowdim_ref))
        expert_actions = expert_actions[:n_ref]
        expert_lowdim = expert_lowdim_ref[:n_ref]
        expert_object_state = expert_object_state[:n_ref]

    tokenizer = OnlineObservationTokenizer(
        T=args.T,
        token_grid=args.token_grid,
        task_cache_dir=args.task_cache_dir,
        device=args.device,
        qwen_device=args.qwen_device or args.device,
        allow_zero_task_fallback=args.allow_zero_task_fallback,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "action_stats.npz", mean=mean, std=std, pos_rate=np.asarray([pos_rate], dtype=np.float32))
    manifest_path = args.out_dir / "manifest.jsonl"
    selected = _select_indices(len(traces), args.stride, args.max_windows)
    if args.min_rollout_step > 0 or args.max_rollout_step > 0:
        filtered = []
        for trace_idx in selected:
            step = int(traces[trace_idx].get("step", trace_idx + 1))
            if args.min_rollout_step > 0 and step < int(args.min_rollout_step):
                continue
            if args.max_rollout_step > 0 and step > int(args.max_rollout_step):
                continue
            filtered.append(trace_idx)
        selected = filtered
    written = 0
    skipped = 0
    align_distances: list[float] = []
    last_align_idx = 0

    with manifest_path.open("w") as mf:
        for out_idx, trace_idx in enumerate(selected):
            trace = traces[trace_idx]
            frames = _context_frames(traces, trace_idx, T=args.T, root=project_root)
            lowdim = trace.get("lowdim_state")
            object_state = trace.get("object_state")
            action_history = trace.get("action_history")
            if frames is None or lowdim is None or action_history is None:
                skipped += 1
                continue
            lowdim_arr = np.asarray(lowdim, dtype=np.float32).reshape(-1)
            if lowdim_arr.shape != (12,):
                skipped += 1
                continue
            object_state_arr = None
            if object_state is not None:
                object_state_arr = np.asarray(object_state, dtype=np.float32).reshape(-1)
            rollout_step = int(trace.get("step", trace_idx + 1)) - 1
            if args.align_mode == "time":
                phase = float(rollout_step) / max(1.0, float(len(traces) - 1))
                align_idx = int(round(phase * max(0, len(expert_actions) - 1)))
                if args.min_expert_idx > 0:
                    align_idx = max(align_idx, int(args.min_expert_idx))
                if args.max_expert_idx > 0:
                    align_idx = min(align_idx, int(args.max_expert_idx))
                align_dist = _lowdim_distance(lowdim_arr, expert_lowdim, align_idx)
            else:
                min_idx = max(0, last_align_idx - int(args.monotonic_slack)) if args.monotonic else 0
                if args.min_expert_idx > 0:
                    min_idx = max(min_idx, int(args.min_expert_idx))
                max_idx = int(args.max_expert_idx) if args.max_expert_idx > 0 else None
                align_idx, align_dist = _align_expert_index(
                    lowdim_arr,
                    expert_lowdim,
                    rollout_object_state=object_state_arr,
                    expert_object_state=expert_object_state,
                    object_state_weight=float(args.object_state_weight),
                    rollout_step=rollout_step,
                    rollout_steps=len(traces),
                    phase_prior_weight=float(args.phase_prior_weight),
                    min_idx=min_idx,
                    max_idx=max_idx,
                )
            if args.max_align_distance > 0 and align_dist > float(args.max_align_distance):
                skipped += 1
                continue
            if args.monotonic:
                last_align_idx = max(last_align_idx, int(align_idx))
            action_tgt = _chunk(expert_actions, align_idx, args.k)
            action_tgt_norm = ((action_tgt[:, :6] - mean[None]) / std[None]).astype(np.float32)
            obs = tokenizer.tokenize(frames, instruction)
            progress_tgt = float(align_idx) / max(1.0, float(len(expert_actions) - 1))
            if _is_task1_put_cream_butter(f"{task_name} {instruction}"):
                plan_state = _plan_state_from_stage(
                    _stage_from_progress(progress_tgt),
                    trace.get("named_poses"),
                    dim=int(args.plan_state_dim),
                )
            else:
                plan_state = np.zeros(int(args.plan_state_dim), dtype=np.float32)
            cache_path = args.out_dir / f"window_{written:06d}.npz"
            np.savez(
                cache_path,
                s_in=obs.context_tokens.squeeze(0).numpy().astype(np.float16),
                c=obs.task_emb.squeeze(0).numpy().astype(np.float16),
                context_rgb=obs.context_rgb.squeeze(0).numpy().astype(np.float16),
                action_tgt=action_tgt.astype(np.float32),
                action_tgt_norm=action_tgt_norm,
                terminal_success_tgt=np.asarray(1.0, dtype=np.float32),
                plausibility_tgt=np.asarray(1.0, dtype=np.float32),
                lowdim_state=lowdim_arr.astype(np.float32),
                action_history=np.asarray(action_history, dtype=np.float32),
                object_state=object_state_arr.astype(np.float32) if object_state_arr is not None else np.zeros(0, dtype=np.float32),
                plan_state=plan_state.astype(np.float32),
                proposer_weight=np.asarray(float(args.sample_weight), dtype=np.float32),
                progress_tgt=np.asarray(progress_tgt, dtype=np.float32),
            )
            rec = {
                "cache_path": str(cache_path),
                "source_format": "libero_failed_rollout_recovery",
                "source_rollout": str(args.rollout_json),
                "hdf5_path": str(args.expert_hdf5),
                "demo_id": args.demo_id,
                "task_name": task_name,
                "instruction": instruction,
                "rollout_step": int(trace.get("step", trace_idx + 1)),
                "rollout_trace_idx": int(trace_idx),
                "target_start": int(align_idx),
                "episode_len": int(len(expert_actions)),
                "expert_align_distance": align_dist,
                "expert_align_mode": args.align_mode,
                "object_state": bool(object_state_arr is not None),
                "object_state_weight": float(args.object_state_weight),
                "plan_state_dim": int(args.plan_state_dim),
                "proposer_weight": float(args.sample_weight),
                "T": int(args.T),
                "k": int(args.k),
                "lowdim_state": True,
                "action_history_len": int(np.asarray(action_history).shape[0]),
                "terminal_success_tgt": 1.0,
                "benchmark_success": True,
            }
            mf.write(json.dumps(rec, sort_keys=True) + "\n")
            written += 1
            align_distances.append(align_dist)
            if args.log_every and written % args.log_every == 0:
                print(json.dumps({"cached": written, "selected": len(selected), "align_idx": align_idx, "align_dist": align_dist}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "rollout_json": str(args.rollout_json),
        "expert_hdf5": str(args.expert_hdf5),
        "demo_id": args.demo_id,
        "manifest": str(manifest_path),
        "cached_windows": written,
        "skipped_windows": skipped,
        "selected_windows": len(selected),
        "min_rollout_step": int(args.min_rollout_step),
        "max_rollout_step": int(args.max_rollout_step),
        "min_expert_idx": int(args.min_expert_idx),
        "max_expert_idx": int(args.max_expert_idx),
        "plan_state_dim": int(args.plan_state_dim),
        "T": args.T,
        "k": args.k,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "phase_prior_weight": args.phase_prior_weight,
        "align_mode": args.align_mode,
        "expert_object_state_npz": str(args.expert_object_state_npz) if args.expert_object_state_npz else None,
        "object_state_weight": float(args.object_state_weight),
        "monotonic": bool(args.monotonic),
        "monotonic_slack": int(args.monotonic_slack),
        "max_align_distance": float(args.max_align_distance),
        "sample_weight": args.sample_weight,
        "align_distance_mean": float(np.mean(align_distances)) if align_distances else None,
        "align_distance_max": float(np.max(align_distances)) if align_distances else None,
        "action_stats": str(args.action_stats),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
