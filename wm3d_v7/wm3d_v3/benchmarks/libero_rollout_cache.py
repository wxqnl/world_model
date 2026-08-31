"""Tokenize LIBERO rollout traces into WM3D negative/feedback cache files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer


def _load_frame(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def _read_rollout(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _action_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(path)
    return stats["mean"][:6].astype(np.float32), np.maximum(stats["std"][:6].astype(np.float32), 1e-6)


def _pad_context(frames: list[np.ndarray], T: int) -> list[np.ndarray]:
    if not frames:
        raise ValueError("empty frame context")
    if len(frames) < T:
        return [frames[0]] * (T - len(frames)) + frames
    return frames[-T:]


def _trace_action(item: dict[str, Any], *, key: str | None = None) -> np.ndarray:
    action = item.get(key) if key is not None else None
    if action is None:
        action = item.get("policy_action") or item.get("action")
    if action is None:
        raise KeyError("trace item has no policy_action or action")
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape != (7,):
        raise ValueError(f"trace action must be 7D, got {arr.shape}")
    return arr


def _future_action_chunk(trace: list[dict[str, Any]], idx: int, k: int, *, key: str | None = None) -> np.ndarray:
    actions: list[np.ndarray] = []
    for j in range(idx, min(len(trace), idx + k)):
        actions.append(_trace_action(trace[j], key=key))
    if not actions:
        raise ValueError("cannot build future action chunk from empty trace")
    while len(actions) < k:
        actions.append(actions[-1].copy())
    return np.stack(actions, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_json", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--action_stats", type=Path, required=True)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth trace step when building windows.")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--token_grid", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_device", default=None)
    ap.add_argument("--task_cache_dir", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3/libero_taskemb"))
    ap.add_argument("--allow_zero_task_fallback", action="store_true")
    ap.add_argument(
        "--supervision_mode",
        default="terminal",
        choices=("terminal", "episode_success", "all_positive"),
        help="How to weight rollout rows for action BC/proposer training.",
    )
    ap.add_argument(
        "--target_chunk_source",
        default="trace_or_repeat_current",
        choices=("trace_or_repeat_current", "future_trace", "teacher_future_trace"),
        help="Use stored action_chunk_raw, policy future actions, or teacher future labels.",
    )
    ap.add_argument("--include_lowdim", action="store_true")
    ap.add_argument("--include_object_state", action="store_true")
    ap.add_argument("--include_plan_state", action="store_true")
    ap.add_argument("--include_action_history", action="store_true")
    ap.add_argument("--include_progress", action="store_true")
    ap.add_argument("--min_step", type=int, default=0)
    ap.add_argument("--max_step", type=int, default=0)
    ap.add_argument("--sample_weight", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=8)
    args = ap.parse_args()

    payload = _read_rollout(args.rollout_json)
    mean, std = _action_stats(args.action_stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = OnlineObservationTokenizer(
        T=args.T,
        token_grid=args.token_grid,
        task_cache_dir=args.task_cache_dir,
        device=args.device,
        qwen_device=args.qwen_device or args.device,
        allow_zero_task_fallback=args.allow_zero_task_fallback,
    )

    manifest_path = args.out_dir / "manifest.jsonl"
    written = 0
    episodes_seen = 0
    with manifest_path.open("w") as mf:
        for ep in payload.get("results") or []:
            episodes_seen += 1
            episode_success = bool(ep.get("success", False))
            frame_history: list[np.ndarray] = []
            trace = ep.get("step_trace") or []
            for trace_idx, item in enumerate(trace):
                step_id = int(item.get("step", trace_idx + 1))
                frame_path = item.get("frame_path")
                if not frame_path:
                    continue
                frame_history.append(_load_frame(frame_path))
                if args.stride > 1 and (trace_idx % int(args.stride)) != 0:
                    continue
                if args.min_step > 0 and step_id < int(args.min_step):
                    continue
                if args.max_step > 0 and step_id > int(args.max_step):
                    continue
                if args.target_chunk_source == "future_trace":
                    action_tgt = _future_action_chunk(trace, trace_idx, args.k)
                elif args.target_chunk_source == "teacher_future_trace":
                    action_tgt = _future_action_chunk(trace, trace_idx, args.k, key="teacher_action")
                else:
                    action_chunk = item.get("action_chunk_raw")
                    if action_chunk is None:
                        action = _trace_action(item)
                        action_chunk = np.repeat(action[None], args.k, axis=0).tolist()
                    action_tgt = np.asarray(action_chunk, dtype=np.float32)
                if action_tgt.ndim != 2 or action_tgt.shape[1] != 7:
                    raise ValueError(f"action chunk must be [k,7], got {action_tgt.shape}")
                if action_tgt.shape[0] < args.k:
                    pad = np.repeat(action_tgt[-1:], args.k - action_tgt.shape[0], axis=0)
                    action_tgt = np.concatenate([action_tgt, pad], axis=0)
                action_tgt = action_tgt[: args.k]
                action_tgt_norm = ((action_tgt[:, :6] - mean[None]) / std[None]).astype(np.float32)
                obs = tokenizer.tokenize(_pad_context(frame_history, args.T), str(ep.get("instruction") or "robot manipulation"))

                cache_path = args.out_dir / f"window_{written:06d}.npz"
                step_success = bool(item.get("success", False))
                if args.supervision_mode == "all_positive":
                    terminal = 1.0
                    proposer_weight = float(args.sample_weight)
                    benchmark_success = True
                elif args.supervision_mode == "episode_success":
                    terminal = float(episode_success)
                    proposer_weight = float(args.sample_weight) if episode_success else 0.0
                    benchmark_success = episode_success
                else:
                    terminal = float(step_success)
                    proposer_weight = float(args.sample_weight) if step_success else 0.0
                    benchmark_success = step_success
                payload_npz: dict[str, np.ndarray] = {
                    "s_in": obs.context_tokens.squeeze(0).numpy().astype(np.float16),
                    "c": obs.task_emb.squeeze(0).numpy().astype(np.float16),
                    "context_rgb": obs.context_rgb.squeeze(0).numpy().astype(np.float16),
                    "action_tgt": action_tgt.astype(np.float32),
                    "action_tgt_norm": action_tgt_norm,
                    "terminal_success_tgt": np.asarray(terminal, dtype=np.float32),
                    "plausibility_tgt": np.asarray(terminal, dtype=np.float32),
                    "proposer_weight": np.asarray(proposer_weight, dtype=np.float32),
                }
                if args.include_lowdim and item.get("lowdim_state") is not None:
                    payload_npz["lowdim_state"] = np.asarray(item["lowdim_state"], dtype=np.float32).reshape(-1)
                if args.include_object_state and item.get("object_state") is not None:
                    payload_npz["object_state"] = np.asarray(item["object_state"], dtype=np.float32).reshape(-1)
                if args.include_plan_state and item.get("plan_state") is not None:
                    payload_npz["plan_state"] = np.asarray(item["plan_state"], dtype=np.float32).reshape(-1)
                if args.include_action_history and item.get("action_history") is not None:
                    payload_npz["action_history"] = np.asarray(item["action_history"], dtype=np.float32).reshape(-1, 7)
                if args.include_progress and item.get("progress_state") is not None:
                    payload_npz["progress_tgt"] = np.asarray(float(item["progress_state"]), dtype=np.float32)
                np.savez(cache_path, **payload_npz)
                rec = {
                    "cache_path": str(cache_path),
                    "source_json": str(args.rollout_json),
                    "source_format": "libero_rollout_trace",
                    "task_name": ep.get("task_name"),
                    "instruction": ep.get("instruction"),
                    "task_id": ep.get("task_id"),
                    "init_state_id": ep.get("init_state_id"),
                    "step": item.get("step"),
                    "T": args.T,
                    "k": args.k,
                    "terminal_success_tgt": terminal,
                    "benchmark_success": bool(benchmark_success),
                    "proposer_weight": float(proposer_weight),
                    "supervision_mode": args.supervision_mode,
                    "target_chunk_source": args.target_chunk_source,
                    "lowdim_state": bool("lowdim_state" in payload_npz),
                    "object_state": bool("object_state" in payload_npz),
                    "plan_state": bool("plan_state" in payload_npz),
                    "action_history_len": int(payload_npz["action_history"].shape[0]) if "action_history" in payload_npz else 0,
                }
                if "progress_tgt" in payload_npz:
                    rec["progress_tgt"] = float(payload_npz["progress_tgt"])
                mf.write(json.dumps(rec, sort_keys=True) + "\n")
                written += 1
                if args.log_every and written % args.log_every == 0:
                    print(json.dumps({"cached": written, "cache_path": str(cache_path)}), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if args.max_windows > 0 and written >= args.max_windows:
                    break
            if args.max_windows > 0 and written >= args.max_windows:
                break

    summary = {
        "rollout_json": str(args.rollout_json),
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "episodes_seen": episodes_seen,
        "cached_windows": written,
        "T": args.T,
        "k": args.k,
        "stride": int(args.stride),
        "token_grid": args.token_grid,
        "action_stats": str(args.action_stats),
        "training_signal": {
            "negative_terminal_supervision": args.supervision_mode == "terminal" and written > 0,
            "proposer_supervision": args.supervision_mode in ("episode_success", "all_positive"),
            "supervision_mode": args.supervision_mode,
            "notes": "Use all_positive only for successful teacher/expert traces where every row is valid action supervision.",
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"manifest": str(manifest_path), "cached_windows": written}, sort_keys=True))


if __name__ == "__main__":
    main()
