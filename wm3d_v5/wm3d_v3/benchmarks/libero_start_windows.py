"""Build start-padded LIBERO expert windows for closed-loop BC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _unique_demos(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    demos: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["hdf5_path"]), str(row["demo_id"]))
        if key in seen:
            continue
        seen.add(key)
        demos.append(row)
    return demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task_name", default=None)
    ap.add_argument("--demo_id", default=None)
    ap.add_argument("--max_demos", type=int, default=0)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--camera_key", default="agentview_rgb")
    args = ap.parse_args()

    rows = _read_jsonl(args.source_jsonl)
    demos = _unique_demos(rows)
    if args.task_name:
        demos = [row for row in demos if str(row.get("task_name")) == args.task_name]
    if args.demo_id:
        demos = [row for row in demos if str(row.get("demo_id")) == args.demo_id]
    if args.max_demos > 0:
        demos = demos[: args.max_demos]
    if not demos:
        raise RuntimeError("no demos matched")

    written = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for demo_row in demos:
            hdf5_path = Path(demo_row["hdf5_path"])
            demo_id = str(demo_row["demo_id"])
            with h5py.File(hdf5_path, "r") as h5:
                actions = np.asarray(h5["data"][demo_id]["actions"], dtype=np.float32)
                obs_keys = sorted(h5["data"][demo_id]["obs"].keys())
            episode_len = int(actions.shape[0])
            for target_start in range(0, max(1, episode_len - args.k + 1), args.stride):
                action_chunk = actions[target_start: target_start + args.k]
                if action_chunk.shape[0] < args.k:
                    pad = np.repeat(action_chunk[-1:], args.k - action_chunk.shape[0], axis=0)
                    action_chunk = np.concatenate([action_chunk, pad], axis=0)
                rec = {
                    "row_type": "libero_start_padded_window",
                    "source_format": "libero_hdf5",
                    "hdf5_path": str(hdf5_path),
                    "task_name": demo_row.get("task_name"),
                    "instruction": demo_row.get("instruction"),
                    "demo_id": demo_id,
                    "context_start": int(target_start - args.T),
                    "target_start": int(target_start),
                    "episode_len": episode_len,
                    "T": int(args.T),
                    "k": int(args.k),
                    "stride": int(args.stride),
                    "camera_key": args.camera_key,
                    "camera_available": args.camera_key in obs_keys,
                    "obs_keys": obs_keys,
                    "action_chunk": action_chunk.astype(float).tolist(),
                    "terminal_success_tgt": 1.0,
                    "plausibility_tgt": 1.0,
                    "benchmark_success": True,
                }
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                written += 1

    print(json.dumps({
        "source_jsonl": str(args.source_jsonl),
        "out": str(args.out),
        "demos": len(demos),
        "windows": written,
        "T": args.T,
        "k": args.k,
        "stride": args.stride,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
