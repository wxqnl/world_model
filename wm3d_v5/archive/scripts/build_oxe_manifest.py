"""Scan OXE tars and build a unified manifest jsonl."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from tqdm import tqdm

from wm3d_v3.data.oxe_loader import quick_manifest_from_tar
from wm3d_v3.data.manifest import write_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/data/user01/world_model_data/oxe_hf"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/user01/Minko/newwm/wm3d_v3/manifests/oxe.jsonl"),
    )
    ap.add_argument("--datasets", nargs="+",
                    default=["bridge", "fractal20220817_data", "taco_play"])
    ap.add_argument("--fractal_subsample", type=float, default=0.20,
                    help="keep this fraction of fractal episodes (default 20%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    all_recs = []
    for ds in args.datasets:
        ds_root = args.root / ds
        if not ds_root.is_dir():
            print(f"[skip] {ds} (not found)")
            continue
        tars = sorted(ds_root.glob(f"{ds}_*.tar"))
        print(f"[ingest] {ds}: {len(tars)} tars")
        ds_recs = []
        for i, t in enumerate(tqdm(tars, desc=ds)):
            try:
                recs = quick_manifest_from_tar(t, ds, i)
            except Exception as e:
                print(f"  [err] {t.name}: {e}")
                continue
            ds_recs.extend(recs)
        if ds == "fractal20220817_data" and args.fractal_subsample < 1.0:
            n_keep = int(len(ds_recs) * args.fractal_subsample)
            rng.shuffle(ds_recs)
            ds_recs = ds_recs[:n_keep]
        print(f"  -> {len(ds_recs)} episodes kept")
        all_recs.extend(ds_recs)
    write_manifest(args.out, all_recs)
    print(f"\nTotal: {len(all_recs)} episodes -> {args.out}")
    print(f"Frame total: {sum(r.n_frames for r in all_recs):,}")


if __name__ == "__main__":
    main()
