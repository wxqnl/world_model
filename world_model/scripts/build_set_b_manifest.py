"""Download 20 ScanNet scenes from fjd/scannet-processed-test and build Set B manifest."""
from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/raw/scannet_test"
MANIFEST = ROOT / "data/manifests/set_b.json"
REPO = "fjd/scannet-processed-test"
N_SCENES = 20
FRAMES_PER_SCENE = 20


def main():
    api = HfApi()
    info = api.dataset_info(REPO)
    all_files = [s.rfilename for s in info.siblings]
    scenes = sorted({f.split("/")[1] for f in all_files if f.startswith("test/scene")})[:N_SCENES]
    print(f"Selecting {len(scenes)} scenes from {REPO}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parallel bulk download: only first FRAMES_PER_SCENE items per scene (via name patterns).
    patterns = []
    for scene in scenes:
        # ScanNet frames are sampled at large strides (0, 20, 40, ...).
        # Fetch everything for these scenes; counts are small (<=60 files per scene).
        patterns.append(f"test/{scene}/color/*")
        patterns.append(f"test/{scene}/depth/*")

    print(f"Downloading {len(patterns)} patterns via snapshot_download ...")
    snapshot_download(REPO, repo_type="dataset", local_dir=str(OUT_DIR),
                      allow_patterns=patterns, max_workers=16)

    manifest = []
    for scene in scenes:
        color = sorted((OUT_DIR / "test" / scene / "color").glob("*.jpg"))
        depth = sorted((OUT_DIR / "test" / scene / "depth").glob("*.npy"))
        stems_c = {p.stem for p in color}; stems_d = {p.stem for p in depth}
        common = sorted(stems_c & stems_d)[:FRAMES_PER_SCENE]
        if len(common) < 8:
            print(f"  SKIP {scene}: only {len(common)} common stems"); continue
        frames = [str(OUT_DIR / f"test/{scene}/color/{s}.jpg") for s in common]
        gts    = [str(OUT_DIR / f"test/{scene}/depth/{s}.npy") for s in common]
        manifest.append({
            "clip_id": scene, "set": "B", "frames": frames, "fps": 30,
            "meta": {"dataset": "scannet_test", "scene": scene}, "gt_depth": gts,
        })
        print(f"  {scene}: {len(common)} frames")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2)
    print(f"\nManifest → {MANIFEST}  ({len(manifest)} scenes)")


if __name__ == "__main__":
    main()
