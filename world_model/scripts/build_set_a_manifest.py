"""Extract DROID-100 episode frames and build Set A manifest.

DROID-100 packs all episodes of `exterior_image_1_left` into one MP4 with
per-episode [from_timestamp, to_timestamp) slices recorded in the episodes
parquet. We decode each episode's slice, sub-sample to ≤150 frames per clip,
save JPGs, and write a manifest entry.

Takes ~a few minutes per 30 episodes using imageio-ffmpeg (single pass).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DROID = ROOT / "data/raw/droid_100"
EP_PARQUET = DROID / "meta/episodes/chunk-000/file-000.parquet"
MP4 = DROID / "videos/observation.images.exterior_image_1_left/chunk-000/file-000.mp4"
OUT_FRAMES = ROOT / "data/raw/droid_frames"
MANIFEST = ROOT / "data/manifests/set_a.json"

FPS = 15
MAX_FRAMES = 150
MIN_FRAMES = 60


def main(n_episodes: int):
    ep = pd.read_parquet(EP_PARQUET).sort_values("episode_index").reset_index(drop=True)
    # Pick episodes with length in [MIN_FRAMES, ... up to truncation]
    ep = ep[ep["length"] >= MIN_FRAMES].reset_index(drop=True)
    chosen = ep.head(n_episodes).copy()
    print(f"Selected {len(chosen)} episodes (length ≥{MIN_FRAMES}).")

    OUT_FRAMES.mkdir(parents=True, exist_ok=True)

    # Stream-decode the MP4 once, indexing frames by global index.
    # DROID exterior video is concatenation of per-episode segments at 15 fps.
    props = iio.improps(str(MP4), plugin="pyav")
    print(f"Video: {props.shape}, dtype={props.dtype}")

    # Build per-frame assignment: for each chosen episode, compute [start_frame, end_frame).
    intervals = []
    for _, row in chosen.iterrows():
        t0 = row["videos/observation.images.exterior_image_1_left/from_timestamp"]
        t1 = row["videos/observation.images.exterior_image_1_left/to_timestamp"]
        f0 = int(round(t0 * FPS))
        f1 = int(round(t1 * FPS))
        n = f1 - f0
        if n > MAX_FRAMES:
            f1 = f0 + MAX_FRAMES
            n = MAX_FRAMES
        intervals.append((int(row["episode_index"]), row["tasks"][0] if len(row["tasks"]) else "", f0, f1))

    max_frame = max(i[3] for i in intervals)
    print(f"Decoding up to frame {max_frame}")

    # Decode required frames with seeking: imageio can read by index.
    manifest = []
    reader = iio.imopen(str(MP4), "r", plugin="pyav")
    for epi_idx, task, f0, f1 in intervals:
        ep_dir = OUT_FRAMES / f"episode_{epi_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        frames_written = []
        for i, fi in enumerate(range(f0, f1)):
            out_path = ep_dir / f"frame_{i:04d}.jpg"
            if out_path.exists():
                frames_written.append(str(out_path))
                continue
            arr = reader.read(index=fi)
            iio.imwrite(str(out_path), arr, quality=90)
            frames_written.append(str(out_path))
        print(f"  episode {epi_idx}: wrote {len(frames_written)} frames  (task: {task[:40]})")
        manifest.append({
            "clip_id": f"droid_{epi_idx:04d}",
            "set": "A",
            "frames": frames_written,
            "fps": FPS,
            "meta": {"dataset": "droid_100", "task": task, "episode_index": epi_idx},
            "gt_depth": None,
        })
    reader.close()

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nManifest → {MANIFEST}  ({len(manifest)} clips)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    main(args.n)
