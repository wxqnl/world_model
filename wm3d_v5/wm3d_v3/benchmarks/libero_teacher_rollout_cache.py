"""Tokenize current-environment LIBERO BC teacher rollouts for WM3D SFT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as fh:
            report = json.load(fh)
        rows = report.get("results")
        if not rows and report.get("step_trace"):
            rows = [report]
        for row in rows or []:
            row = dict(row)
            row["_rollout_json"] = str(path)
            episodes.append(row)
    return episodes


def _read_frame(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _context_frames(traces: list[dict[str, Any]], idx: int, T: int) -> list[np.ndarray] | None:
    first = traces[0].get("frame_path")
    if not first:
        return None
    paths: list[str] = []
    for trace_idx in range(idx - T + 1, idx + 1):
        path = first if trace_idx < 0 else traces[trace_idx].get("frame_path")
        if not path or not Path(path).exists():
            return None
        paths.append(str(path))
    return [_read_frame(path) for path in paths]


def _action_chunk(traces: list[dict[str, Any]], idx: int, k: int) -> np.ndarray:
    actions = []
    for trace_idx in range(idx, min(len(traces), idx + k)):
        trace = traces[trace_idx]
        action = trace.get("teacher_action", trace.get("action"))
        if action is None:
            raise ValueError(f"trace {trace_idx} has no teacher/action target")
        actions.append(np.asarray(action, dtype=np.float32))
    if not actions:
        raise ValueError(f"empty action chunk at trace {idx}")
    while len(actions) < k:
        actions.append(actions[-1].copy())
    out = np.stack(actions, axis=0)
    if out.shape != (k, 7):
        raise ValueError(f"expected action chunk {(k, 7)}, got {out.shape}")
    return out


def _select_indices(n: int, stride: int, max_windows: int) -> list[int]:
    indices = list(range(0, n, max(1, stride)))
    if max_windows <= 0 or len(indices) <= max_windows:
        return indices
    if max_windows == 1:
        return [indices[0]]
    return [
        indices[round(i * (len(indices) - 1) / (max_windows - 1))]
        for i in range(max_windows)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_json", type=Path, nargs="+", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--action_stats", type=Path, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max_windows_per_episode", type=int, default=0)
    ap.add_argument("--token_grid", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_device", default=None)
    ap.add_argument(
        "--task_cache_dir",
        type=Path,
        default=Path("/data/Minko/datasets/cache/wm3d_v3/libero_taskemb"),
    )
    ap.add_argument("--allow_zero_task_fallback", action="store_true")
    ap.add_argument("--sample_weight", type=float, default=3.0)
    ap.add_argument("--grip_transition_boost", type=float, default=3.0)
    ap.add_argument("--successful_only", action="store_true")
    ap.add_argument("--log_every", type=int, default=16)
    args = ap.parse_args()

    episodes = _load_reports(args.rollout_json)
    if args.successful_only:
        episodes = [episode for episode in episodes if episode.get("success")]
    if not episodes:
        raise RuntimeError("no rollout episodes selected")

    stats = np.load(args.action_stats)
    mean = stats["mean"][:6].astype(np.float32)
    std = np.maximum(stats["std"][:6].astype(np.float32), 1e-4)
    pos_rate = float(stats.get("pos_rate", np.asarray([0.5]))[0])

    tokenizer = OnlineObservationTokenizer(
        T=args.T,
        token_grid=args.token_grid,
        task_cache_dir=args.task_cache_dir,
        device=args.device,
        qwen_device=args.qwen_device or args.device,
        allow_zero_task_fallback=args.allow_zero_task_fallback,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "action_stats.npz",
        mean=mean,
        std=std,
        pos_rate=np.asarray([pos_rate], dtype=np.float32),
    )

    manifest_path = args.out_dir / "manifest.jsonl"
    written = 0
    skipped = 0
    per_task: dict[str, int] = {}
    with manifest_path.open("w") as manifest:
        for episode_idx, episode in enumerate(episodes):
            traces = episode.get("step_trace") or []
            instruction = str(episode.get("instruction") or "robot manipulation")
            task_name = str(episode.get("task_name") or "")
            indices = _select_indices(
                len(traces),
                args.stride,
                args.max_windows_per_episode,
            )
            for trace_idx in indices:
                trace = traces[trace_idx]
                frames = _context_frames(traces, trace_idx, args.T)
                lowdim = trace.get("lowdim_state")
                if frames is None or lowdim is None:
                    skipped += 1
                    continue
                lowdim_arr = np.asarray(lowdim, dtype=np.float32).reshape(-1)
                if lowdim_arr.shape != (12,):
                    skipped += 1
                    continue
                action_tgt = _action_chunk(traces, trace_idx, args.k)
                grip = action_tgt[:, 6] > 0.5
                has_grip_transition = bool(np.any(grip[1:] != grip[:-1]))
                if trace_idx > 0:
                    prev = traces[trace_idx - 1].get(
                        "teacher_action",
                        traces[trace_idx - 1].get("action"),
                    )
                    if prev is not None:
                        has_grip_transition = has_grip_transition or bool(
                            (np.asarray(prev, dtype=np.float32)[6] > 0.5) != grip[0]
                        )
                sample_weight = args.sample_weight * (
                    args.grip_transition_boost if has_grip_transition else 1.0
                )
                action_tgt_norm = (
                    (action_tgt[:, :6] - mean[None]) / std[None]
                ).astype(np.float32)
                obs = tokenizer.tokenize(frames, instruction)
                cache_path = args.out_dir / f"window_{written:07d}.npz"
                payload: dict[str, np.ndarray] = {
                    "s_in": obs.context_tokens.squeeze(0).numpy().astype(np.float16),
                    "c": obs.task_emb.squeeze(0).numpy().astype(np.float16),
                    "context_rgb": obs.context_rgb.squeeze(0).numpy().astype(np.float16),
                    "action_tgt": action_tgt,
                    "action_tgt_norm": action_tgt_norm,
                    "terminal_success_tgt": np.asarray(1.0, dtype=np.float32),
                    "plausibility_tgt": np.asarray(1.0, dtype=np.float32),
                    "lowdim_state": lowdim_arr,
                    "proposer_weight": np.asarray(sample_weight, dtype=np.float32),
                }
                object_state = trace.get("object_state")
                if object_state is not None:
                    payload["object_state"] = np.asarray(object_state, dtype=np.float32)
                np.savez(cache_path, **payload)
                record = {
                    "cache_path": str(cache_path),
                    "source_format": "libero_bc_teacher_current_env",
                    "source_rollout": episode.get("_rollout_json"),
                    "task_id": episode.get("task_id"),
                    "task_name": task_name,
                    "instruction": instruction,
                    "init_state_id": episode.get("init_state_id"),
                    "rollout_trace_idx": trace_idx,
                    "target_start": trace_idx,
                    "episode_len": len(traces),
                    "T": args.T,
                    "k": args.k,
                    "lowdim_state": True,
                    "action_history_len": 0,
                    "proposer_weight": sample_weight,
                    "grip_transition_window": has_grip_transition,
                    "terminal_success_tgt": 1.0,
                    "benchmark_success": bool(episode.get("success")),
                }
                manifest.write(json.dumps(record, sort_keys=True) + "\n")
                written += 1
                per_task[task_name] = per_task.get(task_name, 0) + 1
                if args.log_every and written % args.log_every == 0:
                    print(json.dumps({"cached": written, "task": task_name}), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = {
        "rollout_json": [str(path) for path in args.rollout_json],
        "episodes": len(episodes),
        "cached_windows": written,
        "skipped_windows": skipped,
        "per_task": per_task,
        "manifest": str(manifest_path),
        "action_stats": str(args.action_stats),
        "T": args.T,
        "k": args.k,
        "stride": args.stride,
        "sample_weight": args.sample_weight,
        "grip_transition_boost": args.grip_transition_boost,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
