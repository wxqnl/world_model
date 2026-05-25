"""Exp 1 — Temporal coherence on sliding windows (Set A).

Runs VGGT on W=8/S=4 overlapping windows; measures depth/pose/Chamfer consistency
on the 4 overlapping frames between consecutive windows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_frames, load_manifest, sliding_windows
from src.metrics import chamfer_distance, depth_rel_err, so3_geodesic_deg, translation_err
from src.vggt_wrapper import VGGTConfig, VGGTWrapper
from src.viz import save_depth_pair, save_histogram


def realign_extrinsics(extri: torch.Tensor, shared_local_idx: int, shared_reference: torch.Tensor) -> torch.Tensor:
    """Transform window poses so the shared overlapping frame matches a given reference."""
    # extri: [N,3,4]; make 4x4
    T = torch.eye(4, device=extri.device).repeat(extri.shape[0], 1, 1)
    T[:, :3, :4] = extri
    T_ref_world = torch.eye(4, device=extri.device); T_ref_world[:3, :4] = shared_reference
    T_shared = T[shared_local_idx]
    # world_new = T_ref_world * inv(T_shared) applied on the left
    M = T_ref_world @ torch.linalg.inv(T_shared)
    T_new = M.unsqueeze(0) @ T
    return T_new[:, :3, :4]


def run(cfg_path: str = "configs/default.yaml", out_dir: str = "results", limit: int | None = None):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    model = VGGTWrapper(VGGTConfig(name=cfg["model"]["name"]))
    clips = load_manifest(cfg["datasets"]["set_a"]["manifest"])
    if limit: clips = clips[:limit]

    W, S = cfg["exp1"]["window_size"], cfg["exp1"]["stride"]
    rows = []

    for clip in tqdm(clips, desc="exp1"):
        frames = load_frames(clip, target_size=cfg["inference"]["image_size"])
        windows = sliding_windows(frames.shape[0], W, S)
        if len(windows) < 2: continue

        results = []
        for (a, b) in windows:
            out = model.full_inference(frames[a:b])
            results.append({
                "range": (a, b),
                "depth": out["depth"][0].detach().float().cpu(),        # [N,H,W,1] or [N,H,W]
                "point": out["point_map"][0].detach().float().cpu(),
                "extri": out["camera_extrinsic"][0].detach().float().cpu(),
            })

        depth_errs, rot_errs, trans_errs, chamfers = [], [], [], []
        for i in range(len(results) - 1):
            r1, r2 = results[i], results[i + 1]
            overlap_start = r2["range"][0]; overlap_end = r1["range"][1]
            local1 = slice(overlap_start - r1["range"][0], overlap_end - r1["range"][0])
            local2 = slice(0, overlap_end - overlap_start)

            d1 = r1["depth"][local1].squeeze(-1)
            d2 = r2["depth"][local2].squeeze(-1)
            for k in range(d1.shape[0]):
                depth_errs.append(depth_rel_err(d1[k], d2[k]))

            extri2_al = realign_extrinsics(r2["extri"], 0, r1["extri"][local1.start])
            R1 = r1["extri"][local1, :3, :3]; R2 = extri2_al[local2, :3, :3]
            t1 = r1["extri"][local1, :3, 3]; t2 = extri2_al[local2, :3, 3]
            rot_errs.append(so3_geodesic_deg(R1, R2))
            trans_errs.append(translation_err(t1, t2))

            p1 = r1["point"][local1].reshape(-1, 3)
            p2 = r2["point"][local2].reshape(-1, 3)
            chamfers.append(chamfer_distance(p1, p2, max_pts=cfg["exp1"]["chamfer_samples"]))

            if i == 0 and len(rows) < 3:
                save_depth_pair(d1[0], d2[0],
                                Path(out_dir) / "qualitative" / f"exp1_{clip.clip_id}.png",
                                title=f"{clip.clip_id} overlap frame 0")

        rows.append({
            "clip_id": clip.clip_id,
            "n_windows": len(results),
            "depth_rel_err_mean": float(np.mean(depth_errs)),
            "depth_rel_err_std":  float(np.std(depth_errs)),
            "rot_deg_mean":       float(np.mean(rot_errs)),
            "trans_err_mean":     float(np.mean(trans_errs)),
            "chamfer_mean":       float(np.mean(chamfers)),
        })

    out_dir = Path(out_dir)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics/exp1_consistency.csv", index=False)
    for col in ["depth_rel_err_mean", "rot_deg_mean", "trans_err_mean", "chamfer_mean"]:
        save_histogram(df[col].values, out_dir / f"plots/exp1_{col}_hist.png", xlabel=col)
    json.dump(cfg, open(out_dir / "metrics/exp1_config.json", "w"), indent=2)
    print(df.describe())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.config, a.out, a.limit)
