"""Subsample the OXE manifest to fit a target frame budget."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from wm3d_v3.data.manifest import read_manifest, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", type=Path,
                    default=Path("/home/user01/Minko/newwm/wm3d_v3/manifests/oxe.jsonl"))
    ap.add_argument("--out_manifest", type=Path,
                    default=Path("/home/user01/Minko/newwm/wm3d_v3/manifests/oxe_train.jsonl"))
    ap.add_argument("--target_frames", type=int, default=800_000)
    ap.add_argument("--per_dataset_caps", type=str,
                    default="bridge:600000,fractal20220817_data:300000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    caps = dict(x.split(":") for x in args.per_dataset_caps.split(","))
    caps = {k: int(v) for k, v in caps.items()}
    rng = random.Random(args.seed)

    all_recs = read_manifest(args.in_manifest)
    by_ds: dict[str, list] = {}
    for r in all_recs:
        by_ds.setdefault(r.dataset, []).append(r)
    kept = []
    for ds, recs in by_ds.items():
        rng.shuffle(recs)
        target = caps.get(ds, args.target_frames)
        frames = 0
        ds_kept = []
        for r in recs:
            if frames + r.n_frames > target:
                continue
            ds_kept.append(r)
            frames += r.n_frames
        print(f"{ds}: kept {len(ds_kept)} eps ({frames:,} frames)")
        kept.extend(ds_kept)
    write_manifest(args.out_manifest, kept)
    print(f"\nTotal: {len(kept)} eps, {sum(r.n_frames for r in kept):,} frames -> {args.out_manifest}")


if __name__ == "__main__":
    main()
