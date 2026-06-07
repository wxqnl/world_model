"""Build a bounded v5 native3D experiment manifest from cache-ready clips."""
from __future__ import annotations

import argparse
import json
import sys
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wm3d_v3.data.manifest import read_manifest  # noqa: E402
from scripts.cache_geom_utils import frame_count_npy  # noqa: E402


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def cache_ready(cache_root: Path, rec, *, min_frames: int, require_task_emb: bool) -> bool:
    cid = safe_id(str(rec.clip_id))
    n = frame_count_npy(cache_root / "vggt_pooled" / f"{cid}.npy")
    if n is None or n < min_frames:
        return False
    if frame_count_npy(cache_root / "actions" / f"{cid}.npy") != n:
        return False
    if frame_count_npy(cache_root / "rgb_256" / f"{cid}.npy") != n:
        return False
    if not (cache_root / "vggt_geom" / f"{cid}.npz").exists():
        return False
    qwen_path = cache_root / "qwen_taskemb" / f"{cid}.npy"
    if require_task_emb and (not qwen_path.exists() or qwen_path.stat().st_size <= 0):
        return False
    return True


def record_to_json(rec) -> str:
    if hasattr(rec, "to_json"):
        return rec.to_json()
    payload = rec.__dict__ if hasattr(rec, "__dict__") else dict(rec)
    return json.dumps(payload, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_manifest", type=Path, required=True)
    ap.add_argument("--cache_root", type=Path, required=True)
    ap.add_argument("--out_manifest", type=Path, required=True)
    ap.add_argument("--limit_clips", type=int, default=8192)
    ap.add_argument("--min_frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1405)
    ap.add_argument("--require_task_emb", action="store_true")
    args = ap.parse_args()

    records = read_manifest(args.source_manifest)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    selected = []
    scanned = 0
    for rec in records:
        scanned += 1
        if cache_ready(args.cache_root, rec, min_frames=args.min_frames, require_task_emb=args.require_task_emb):
            selected.append(rec)
            if len(selected) >= args.limit_clips:
                break
    if not selected:
        raise SystemExit("no cache-ready records found")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(record_to_json(rec) + "\n")

    counts = Counter(str(getattr(r, "dataset", "unknown")) for r in selected)
    print(
        json.dumps(
            {
                "source": str(args.source_manifest),
                "out": str(args.out_manifest),
                "scanned_records": scanned,
                "selected_records": len(selected),
                "dataset_counts": dict(sorted(counts.items())),
                "min_frames": args.min_frames,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
