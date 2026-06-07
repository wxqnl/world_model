#!/usr/bin/env python3
"""Build the Stage-1 mixed OXE+DROID manifest and action statistics."""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.cache_geom_utils import (
    validate_actions_npy,
    validate_geom_npz,
    validate_pooled_npy,
    validate_qwen_npy,
    validate_rgb_npy,
)


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cache_ready(record: dict, cache_root: Path, require_task_emb: bool, min_frames: int) -> bool:
    cid = safe_id(str(record["clip_id"]))
    n_frames = int(record.get("n_frames", 0) or 0)
    if n_frames < min_frames:
        return False
    if not validate_pooled_npy(cache_root / "vggt_pooled" / f"{cid}.npy", expected_frames=n_frames):
        return False
    if not validate_actions_npy(cache_root / "actions" / f"{cid}.npy", expected_frames=n_frames):
        return False
    if not validate_rgb_npy(cache_root / "rgb_256" / f"{cid}.npy", expected_frames=n_frames):
        return False
    if not validate_geom_npz(cache_root / "vggt_geom" / f"{cid}.npz", expected_frames=n_frames, require_geom_extra=False):
        return False
    if require_task_emb and not validate_qwen_npy(cache_root / "qwen_taskemb" / f"{cid}.npy"):
        return False
    return True


def estimate_windows(records: list[dict], t: int, k: int, stride: int) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    win = t + k
    for r in records:
        n = int(r.get("n_frames", 0) or 0)
        out[str(r.get("dataset", "unknown"))] += max(0, (n - win) // stride + 1)
    return dict(out)


def compute_action_stats(records: list[dict], cache_root: Path, out_path: Path) -> None:
    total = np.zeros(7, dtype=np.float64)
    total_sq = np.zeros(7, dtype=np.float64)
    count = 0
    for r in records:
        p = cache_root / "actions" / f"{safe_id(str(r['clip_id']))}.npy"
        if not p.exists():
            continue
        a = np.load(p)
        if a.ndim == 2 and a.shape[1] >= 7:
            x = np.asarray(a[:, :7], dtype=np.float64)
            total += x.sum(axis=0)
            total_sq += np.square(x).sum(axis=0)
            count += int(x.shape[0])
    if count <= 0:
        raise RuntimeError("no action arrays found for action stats")
    mean64 = total / float(count)
    var64 = np.maximum(total_sq / float(count) - np.square(mean64), 1e-12)
    mean = mean64.astype(np.float32)
    std = np.maximum(np.sqrt(var64).astype(np.float32), 1e-6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean, std=std, count=np.array([count], dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oxe_manifest", type=Path, required=True)
    ap.add_argument("--droid_manifest_glob", required=True)
    ap.add_argument("--cache_root", type=Path, required=True)
    ap.add_argument("--out_manifest", type=Path, required=True)
    ap.add_argument("--out_action_stats", type=Path, required=True)
    ap.add_argument("--require_task_emb", action="store_true")
    ap.add_argument("--min_droid_records", type=int, default=0)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    min_frames = int(args.T + args.k)
    oxe = read_jsonl(args.oxe_manifest)
    droid_paths = [Path(p) for p in sorted(glob.glob(args.droid_manifest_glob))]
    droid: list[dict] = []
    for p in droid_paths:
        droid.extend(read_jsonl(p))

    oxe_ready = [r for r in oxe if cache_ready(r, args.cache_root, args.require_task_emb, min_frames)]
    droid_ready = [r for r in droid if cache_ready(r, args.cache_root, args.require_task_emb, min_frames)]
    if len(droid_ready) < int(args.min_droid_records):
        raise RuntimeError(
            f"only {len(droid_ready)} DROID records ready, expected at least {args.min_droid_records}"
        )

    combined = oxe_ready + droid_ready
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w") as f:
        for r in combined:
            f.write(json.dumps(r) + "\n")
    compute_action_stats(combined, args.cache_root, args.out_action_stats)

    counts = Counter(str(r.get("dataset", "unknown")) for r in combined)
    windows = estimate_windows(combined, args.T, args.k, args.stride)
    print(json.dumps({
        "out_manifest": str(args.out_manifest),
        "out_action_stats": str(args.out_action_stats),
        "records": dict(counts),
        "total_records": len(combined),
        "windows_est": windows,
        "total_windows_est": sum(windows.values()),
        "droid_manifest_files": [str(p) for p in droid_paths],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
