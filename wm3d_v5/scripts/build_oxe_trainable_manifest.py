"""Build OXE manifests for cached OXE training paths."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.cache_geom_utils import (
    validate_actions_npy,
    validate_geom_npz,
    validate_pooled_npy,
    validate_qwen_npy,
    validate_rgb_npy,
)


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def has_required_cache(
    cache_root: Path,
    clip_id: str,
    *,
    require_policy: bool,
    require_qwen: bool,
    require_rgb: bool,
    require_geom: bool,
    expected_frames: int | None = None,
    require_geom_extra: bool = False,
) -> bool:
    cid = safe_id(clip_id)
    if require_policy:
        if not validate_pooled_npy(cache_root / "vggt_pooled" / f"{cid}.npy", expected_frames=expected_frames):
            return False
        if not validate_actions_npy(cache_root / "actions" / f"{cid}.npy", expected_frames=expected_frames):
            return False
    if require_qwen and not validate_qwen_npy(cache_root / "qwen_taskemb" / f"{cid}.npy"):
        return False
    if require_rgb and not validate_rgb_npy(cache_root / "rgb_256" / f"{cid}.npy", expected_frames=expected_frames):
        return False
    if require_geom and not validate_geom_npz(
        cache_root / "vggt_geom" / f"{cid}.npz",
        expected_frames=expected_frames,
        require_geom_extra=require_geom_extra,
    ):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cache_root", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3"))
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--require_policy_cache", action="store_true")
    ap.add_argument("--require_rgb", action="store_true")
    ap.add_argument("--require_geom", action="store_true")
    ap.add_argument("--allow_missing_qwen", action="store_true")
    args = ap.parse_args()

    win = args.T + args.k
    counts: Counter[str] = Counter()
    frames: Counter[str] = Counter()
    windows: Counter[str] = Counter()
    skipped_short: Counter[str] = Counter()
    skipped_cache: Counter[str] = Counter()
    total_frames = 0
    total_windows = 0
    kept = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as src, args.output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            dataset = str(rec.get("dataset", "unknown"))
            n_frames = int(rec.get("n_frames", 0))
            if n_frames < win:
                skipped_short[dataset] += 1
                continue
            require_cache = args.require_policy_cache or args.require_rgb or args.require_geom
            if require_cache and not has_required_cache(
                args.cache_root,
                str(rec.get("clip_id", "")),
                require_policy=args.require_policy_cache,
                require_qwen=not args.allow_missing_qwen,
                require_rgb=args.require_rgb,
                require_geom=args.require_geom,
                expected_frames=n_frames,
            ):
                skipped_cache[dataset] += 1
                continue

            num_windows = (n_frames - win) // args.stride + 1
            kept += 1
            counts[dataset] += 1
            frames[dataset] += n_frames
            windows[dataset] += num_windows
            total_frames += n_frames
            total_windows += num_windows
            dst.write(line)

    print(f"wrote={args.output}")
    print(f"clips={kept} frames={total_frames} windows={total_windows}")
    print(f"by_dataset={dict(counts)}")
    print(f"frames_by_dataset={dict(frames)}")
    print(f"windows_by_dataset={dict(windows)}")
    print(f"skipped_short={dict(skipped_short)}")
    print(f"skipped_cache={dict(skipped_cache)}")


if __name__ == "__main__":
    main()
