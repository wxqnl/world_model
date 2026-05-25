"""Set C — Autonomous driving clips from KITTI raw.

Downloads a handful of 2011_09_26 drive sequences (each ~100-300 frames, ~30-100MB).
KITTI raw is freely downloadable from the AWS mirror. We decimate to ~60 frames
per clip to match the brief.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/raw/kitti_raw"
MANIFEST = ROOT / "data/manifests/set_c.json"

# Small drives (diverse, from 2011_09_26 synced+rectified release).
DRIVES = [
    "2011_09_26_drive_0001",  # residential, ~108 frames
    "2011_09_26_drive_0002",  # residential, ~77 frames
    "2011_09_26_drive_0005",  # residential, ~153 frames
    "2011_09_26_drive_0009",  # residential, ~447 frames (sub-sample)
    "2011_09_26_drive_0013",  # residential, ~144 frames
    "2011_09_26_drive_0014",  # residential, ~314 frames
    "2011_09_26_drive_0017",  # city, ~114 frames
    "2011_09_26_drive_0018",  # city, ~270 frames
    "2011_09_26_drive_0019",  # city, ~481 frames (sub-sample)
    "2011_09_26_drive_0027",  # city, ~188 frames
    "2011_09_26_drive_0028",  # city, ~430 frames (sub-sample)
    "2011_09_26_drive_0029",  # city, ~430 frames (sub-sample)
]
TARGET_FRAMES = 60
URL_FMT = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{d}/{d}_sync.zip"
CAM = "image_02"  # left color


def download_drive(drive: str) -> Path:
    url = URL_FMT.format(d=drive)
    zip_path = OUT_DIR / f"{drive}_sync.zip"
    if zip_path.exists():
        return zip_path
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    return zip_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for drive in DRIVES:
        try:
            zip_path = download_drive(drive)
            ex_dir = OUT_DIR / drive
            if not ex_dir.exists():
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(OUT_DIR)
            frames_dir = next(OUT_DIR.rglob(f"{drive}_sync/{CAM}/data"), None)
            if frames_dir is None:
                print(f"  SKIP {drive}: no {CAM}/data"); continue
            all_frames = sorted(frames_dir.glob("*.png"))
            if not all_frames:
                print(f"  SKIP {drive}: empty"); continue
            step = max(1, len(all_frames) // TARGET_FRAMES)
            chosen = [str(p) for p in all_frames[::step][:TARGET_FRAMES]]
            manifest.append({
                "clip_id": drive,
                "set": "C",
                "frames": chosen,
                "fps": 10,  # KITTI raw native
                "meta": {"dataset": "kitti_raw", "drive": drive, "cam": CAM},
                "gt_depth": None,
            })
            print(f"  {drive}: {len(chosen)} frames")
        except Exception as e:
            print(f"  ERR {drive}: {e}")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nManifest → {MANIFEST}  ({len(manifest)} drives)")


if __name__ == "__main__":
    main()
