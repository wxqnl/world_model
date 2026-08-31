"""Build safe anchor-plus-residual candidate shards from aligned extractions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _build(old_path: Path, warm_path: Path, out_path: Path, alphas: tuple[float, ...]) -> dict:
    old = np.load(old_path)
    warm = np.load(warm_path)
    if not np.array_equal(old["rows_json"], warm["rows_json"]):
        raise ValueError(f"row mismatch: {old_path} vs {warm_path}")
    if not np.allclose(old["expert_action"], warm["expert_action"]):
        raise ValueError(f"expert action mismatch: {old_path} vs {warm_path}")
    old_cond = np.asarray(old["candidate_cond"], dtype=np.float32)
    warm_cond = np.asarray(warm["candidate_cond"], dtype=np.float32)
    if old_cond.ndim != 4 or warm_cond.ndim != 4:
        raise ValueError("candidate tensors must be [N,K,T,7]")
    if old_cond.shape[0] != warm_cond.shape[0] or old_cond.shape[2:] != warm_cond.shape[2:]:
        raise ValueError(f"candidate shape mismatch: {old_cond.shape} vs {warm_cond.shape}")
    anchor = old_cond[:, 0]
    residuals = [
        anchor[:, None] + alpha * (warm_cond - anchor[:, None])
        for alpha in alphas
    ]
    candidate_cond = np.concatenate([anchor[:, None], *residuals], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        candidate_cond=candidate_cond.astype(np.float32),
        expert_action=np.asarray(old["expert_action"], dtype=np.float32),
        rows_json=old["rows_json"],
        alphas=np.asarray(alphas, dtype=np.float32),
        source_old=np.asarray(str(old_path)),
        source_warm=np.asarray(str(warm_path)),
    )
    return {
        "out": str(out_path),
        "rows": int(candidate_cond.shape[0]),
        "candidates": int(candidate_cond.shape[1]),
        "alphas": alphas,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_dir", type=Path, required=True)
    ap.add_argument("--warm_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.25])
    args = ap.parse_args()
    if not args.alpha or any(not 0.0 < alpha <= 1.0 for alpha in args.alpha):
        raise ValueError("all alpha values must be in (0,1]")
    alphas = tuple(args.alpha)
    old_paths = sorted(args.old_dir.glob("shard_*.npz"))
    if not old_paths:
        raise RuntimeError(f"no shards in {args.old_dir}")
    reports = []
    for old_path in old_paths:
        warm_path = args.warm_dir / old_path.name
        if not warm_path.exists():
            raise FileNotFoundError(warm_path)
        reports.append(_build(old_path, warm_path, args.out_dir / old_path.name, alphas))
    print(json.dumps(
        {"shards": reports, "total_rows": sum(r["rows"] for r in reports)},
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
