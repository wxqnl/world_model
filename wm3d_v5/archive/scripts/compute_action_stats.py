"""One-shot: compute pooled action mean/std over all cached actions.

Output keys:
  mean[7]       : per-axis mean
  std[7]        : per-axis std (clipped to 1e-3 to avoid div-by-zero)
  pos_rate[1]   : fraction of frames with gripper closed (gt > 0.5)
  n_frames      : total frames scanned
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=Path,
                    default=Path("/home/user01/Minko/datasets/cache/wm3d_v3"))
    args = ap.parse_args()

    act_dir = args.cache_root / "actions"
    files = sorted(act_dir.glob("*.npy"))
    if not files:
        raise SystemExit(f"no action files under {act_dir}")
    print(f"scanning {len(files)} action files")

    sum_x = np.zeros(7, dtype=np.float64)
    sum_sq = np.zeros(7, dtype=np.float64)
    n = 0
    n_pos = 0
    for i, f in enumerate(files):
        a = np.load(f).astype(np.float64).reshape(-1, 7)
        sum_x += a.sum(axis=0)
        sum_sq += (a ** 2).sum(axis=0)
        n += a.shape[0]
        n_pos += int((a[:, 6] > 0.5).sum())
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)} ({n} frames)")
    mean = sum_x / max(1, n)
    var = sum_sq / max(1, n) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-6))
    std = np.maximum(std, 1e-3)
    pos_rate = n_pos / max(1, n)

    out = args.cache_root / "action_stats.npz"
    np.savez(out,
             mean=mean.astype(np.float32),
             std=std.astype(np.float32),
             pos_rate=np.array([pos_rate], dtype=np.float32),
             n_frames=np.array([n], dtype=np.int64))
    print(f"wrote {out}")
    print(f"  mean = {mean.tolist()}")
    print(f"  std  = {std.tolist()}")
    print(f"  pos_rate = {pos_rate:.4f}")


if __name__ == "__main__":
    main()
