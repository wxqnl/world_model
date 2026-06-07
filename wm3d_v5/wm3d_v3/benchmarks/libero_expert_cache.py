"""Tokenize LIBERO expert windows into compact WM3D training cache files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_rows(rows: list[dict[str, Any]], max_windows: int) -> list[dict[str, Any]]:
    if max_windows <= 0 or len(rows) <= max_windows:
        return rows
    if max_windows == 1:
        return [rows[0]]
    return [rows[round(i * (len(rows) - 1) / (max_windows - 1))] for i in range(max_windows)]


def _action_stats(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, float]:
    chunks = [np.asarray(row["action_chunk"], dtype=np.float32) for row in rows]
    actions = np.concatenate(chunks, axis=0)
    mean = actions[:, :6].mean(axis=0).astype(np.float32)
    std = actions[:, :6].std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-4).astype(np.float32)
    pos_rate = float((actions[:, 6] > 0.5).mean())
    return mean, std, pos_rate


def _frames_from_hdf5(row: dict[str, Any]) -> list[np.ndarray]:
    hdf5_path = Path(row["hdf5_path"])
    demo_id = str(row["demo_id"])
    camera_key = str(row.get("camera_key") or "agentview_rgb")
    start = int(row["context_start"])
    T = int(row["T"])
    with h5py.File(hdf5_path, "r") as h5:
        obs = h5["data"][demo_id]["obs"]
        all_frames = np.asarray(obs[camera_key])
        if start < 0:
            pad = np.repeat(all_frames[:1], -start, axis=0)
            arr = np.concatenate([pad, all_frames[: start + T]], axis=0)
        else:
            arr = all_frames[start: start + T]
        if arr.shape[0] < T:
            pad = np.repeat(all_frames[-1:], T - arr.shape[0], axis=0)
            arr = np.concatenate([arr, pad], axis=0)
    if arr.ndim != 4:
        raise ValueError(f"expected [{T},H,W,3] or [{T},3,H,W] frames, got {arr.shape}")
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        raise ValueError(f"expected RGB frames with 3 channels, got {arr.shape}")
    return [frame for frame in arr]


def _lowdim_from_hdf5(row: dict[str, Any]) -> np.ndarray:
    hdf5_path = Path(row["hdf5_path"])
    demo_id = str(row["demo_id"])
    target_start = int(row["target_start"])
    with h5py.File(hdf5_path, "r") as h5:
        obs = h5["data"][demo_id]["obs"]
        idx = min(max(target_start, 0), int(obs["ee_pos"].shape[0]) - 1)
        return np.concatenate([
            np.asarray(obs["ee_pos"][idx], dtype=np.float32).reshape(-1),
            np.asarray(obs["gripper_states"][idx], dtype=np.float32).reshape(-1),
            np.asarray(obs["joint_states"][idx], dtype=np.float32).reshape(-1),
        ]).astype(np.float32)


def _action_history_from_hdf5(row: dict[str, Any], history_len: int) -> np.ndarray:
    hdf5_path = Path(row["hdf5_path"])
    demo_id = str(row["demo_id"])
    target_start = int(row["target_start"])
    with h5py.File(hdf5_path, "r") as h5:
        actions = np.asarray(h5["data"][demo_id]["actions"], dtype=np.float32)
    hist = np.zeros((history_len, 7), dtype=np.float32)
    if target_start <= 0:
        return hist
    src_start = max(0, target_start - history_len)
    src = actions[src_start: target_start]
    hist[-len(src):] = src
    return hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--token_grid", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_device", default=None)
    ap.add_argument("--task_cache_dir", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3/libero_taskemb"))
    ap.add_argument("--allow_zero_task_fallback", action="store_true")
    ap.add_argument("--action_stats", type=Path, default=None)
    ap.add_argument("--include_lowdim", action="store_true")
    ap.add_argument("--action_history_len", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=8)
    args = ap.parse_args()

    all_rows = _read_jsonl(args.input_jsonl)
    if not all_rows:
        raise RuntimeError(f"empty input jsonl: {args.input_jsonl}")
    mean, std, pos_rate = (
        _action_stats(all_rows)
        if args.action_stats is None
        else (
            np.load(args.action_stats)["mean"][:6].astype(np.float32),
            np.load(args.action_stats)["std"][:6].astype(np.float32),
            float(np.load(args.action_stats).get("pos_rate", np.asarray([0.5]))[0]),
        )
    )
    rows = _select_rows(all_rows, args.max_windows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "action_stats.npz", mean=mean, std=std, pos_rate=np.asarray([pos_rate], dtype=np.float32))

    tokenizer = OnlineObservationTokenizer(
        T=args.T,
        token_grid=args.token_grid,
        task_cache_dir=args.task_cache_dir,
        device=args.device,
        qwen_device=args.qwen_device or args.device,
        allow_zero_task_fallback=args.allow_zero_task_fallback,
    )

    manifest_path = args.out_dir / "manifest.jsonl"
    manifest_rows: list[dict[str, Any]] = []
    with manifest_path.open("w") as mf:
        for idx, row in enumerate(rows):
            if int(row["T"]) != args.T:
                raise ValueError(f"row T={row['T']} does not match --T={args.T}")
            frames = _frames_from_hdf5(row)
            obs = tokenizer.tokenize(frames, str(row["instruction"]))
            action_tgt = np.asarray(row["action_chunk"], dtype=np.float32)
            if action_tgt.ndim != 2 or action_tgt.shape[1] != 7:
                raise ValueError(f"action_chunk must be [k,7], got {action_tgt.shape}")
            action_tgt_norm = ((action_tgt[:, :6] - mean[None]) / std[None]).astype(np.float32)
            cache_name = f"window_{idx:06d}.npz"
            cache_path = args.out_dir / cache_name
            payload = {
                "s_in": obs.context_tokens.squeeze(0).numpy().astype(np.float16),
                "c": obs.task_emb.squeeze(0).numpy().astype(np.float16),
                "context_rgb": obs.context_rgb.squeeze(0).numpy().astype(np.float16),
                "action_tgt": action_tgt.astype(np.float32),
                "action_tgt_norm": action_tgt_norm,
                "terminal_success_tgt": np.asarray(float(row.get("terminal_success_tgt", 1.0)), dtype=np.float32),
                "plausibility_tgt": np.asarray(float(row.get("plausibility_tgt", 1.0)), dtype=np.float32),
            }
            if args.include_lowdim:
                payload["lowdim_state"] = _lowdim_from_hdf5(row)
            if args.action_history_len > 0:
                payload["action_history"] = _action_history_from_hdf5(row, args.action_history_len)
            np.savez(cache_path, **payload)
            rec = {
                "cache_path": str(cache_path),
                "source_jsonl": str(args.input_jsonl),
                "source_format": row.get("source_format", "libero_hdf5"),
                "hdf5_path": row.get("hdf5_path"),
                "task_name": row.get("task_name"),
                "instruction": row.get("instruction"),
                "demo_id": row.get("demo_id"),
                "context_start": row.get("context_start"),
                "target_start": row.get("target_start"),
                "T": int(row["T"]),
                "k": int(row["k"]),
                "lowdim_state": bool(args.include_lowdim),
                "action_history_len": int(args.action_history_len),
                "terminal_success_tgt": float(row.get("terminal_success_tgt", 1.0)),
                "benchmark_success": bool(row.get("benchmark_success", True)),
            }
            mf.write(json.dumps(rec, sort_keys=True) + "\n")
            manifest_rows.append(rec)
            if args.log_every and (idx + 1) % args.log_every == 0:
                print(json.dumps({"cached": idx + 1, "total": len(rows), "cache_path": str(cache_path)}), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "out_dir": str(args.out_dir),
        "manifest": str(manifest_path),
        "total_input_rows": len(all_rows),
        "cached_windows": len(manifest_rows),
        "T": args.T,
        "token_grid": args.token_grid,
        "action_stats": {
            "path": str(args.out_dir / "action_stats.npz"),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "pos_rate": pos_rate,
        },
        "task_cache_dir": str(args.task_cache_dir),
        "lowdim_state": bool(args.include_lowdim),
        "action_history_len": int(args.action_history_len),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({
        "manifest": str(manifest_path),
        "cached_windows": len(manifest_rows),
        "action_stats": str(args.out_dir / "action_stats.npz"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
