"""Exp 2 — Static vs dynamic reconstruction quality.

- Set B (static, with GT depth): AbsRel, delta<1.25
- Set A (dynamic, no GT): photometric self-consistency
- Motion bins via RAFT optical flow magnitude
- Confidence map distributions
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_frames, load_manifest
from src.metrics import abs_rel, delta_1_25, photometric_reprojection_l1
from src.vggt_wrapper import VGGTConfig, VGGTWrapper
from src.viz import save_histogram, save_scatter


def raft_flow_magnitudes(frames: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """frames [N,3,H,W] in [0,1] -> [N-1] mean optical flow magnitudes."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    def _to_div8(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        h8 = ((h + 7) // 8) * 8; w8 = ((w + 7) // 8) * 8
        if (h, w) == (h8, w8): return x
        return torch.nn.functional.interpolate(x, size=(h8, w8), mode="bilinear", align_corners=False)

    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).eval().to(device)
    prep = weights.transforms()
    mags = []
    with torch.no_grad():
        for i in range(frames.shape[0] - 1):
            a = _to_div8(frames[i:i+1].to(device)); b = _to_div8(frames[i+1:i+2].to(device))
            a, b = prep(a, b); a = _to_div8(a); b = _to_div8(b)
            flow = model(a, b)[-1][0]
            mags.append(flow.norm(dim=0).mean().item())
    del model
    torch.cuda.empty_cache()
    return np.array(mags)


def load_gt_depth(path: str, target_hw: tuple[int, int]) -> torch.Tensor:
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = np.asarray(Image.open(path)).astype(np.float32)
    # ScanNet depth is uint16 in mm
    if d.max() > 1000: d = d / 1000.0
    d = torch.from_numpy(d)
    # resize to match prediction
    d = torch.nn.functional.interpolate(d[None, None], size=target_hw, mode="nearest")[0, 0]
    return d


def run_static(cfg, model, out_dir: Path) -> pd.DataFrame:
    clips = load_manifest(cfg["datasets"]["set_b"]["manifest"])
    rows = []
    for clip in tqdm(clips, desc="exp2/static"):
        frames = load_frames(clip, target_size=cfg["inference"]["image_size"])
        out = model.full_inference(frames[: cfg["exp2"]["window_size"]])
        pred_depth = out["depth"][0].squeeze(-1).float().cpu()    # [N,H,W]
        conf = out["depth_conf"][0].float().cpu()
        if clip.gt_depth:
            gt = torch.stack([load_gt_depth(p, pred_depth.shape[-2:]) for p in clip.gt_depth[:pred_depth.shape[0]]])
            ar = abs_rel(pred_depth, gt)
            d1 = delta_1_25(pred_depth, gt)
        else:
            ar, d1 = float("nan"), float("nan")
        rows.append({"clip_id": clip.clip_id, "domain": "static",
                     "abs_rel": ar, "delta_1_25": d1,
                     "conf_mean": conf.mean().item(), "conf_std": conf.std().item()})
    return pd.DataFrame(rows)


def run_dynamic(cfg, model, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clips = load_manifest(cfg["datasets"]["set_a"]["manifest"])
    bins = [float(b) for b in cfg["exp2"]["motion_bins"]]
    rows, bin_rows = [], []
    for clip in tqdm(clips, desc="exp2/dynamic"):
        frames = load_frames(clip, target_size=cfg["inference"]["image_size"])
        W = cfg["exp2"]["window_size"]
        frames = frames[:W]
        out = model.full_inference(frames)
        pred_depth = out["depth"][0].squeeze(-1).float().cpu()
        extri = out["camera_extrinsic"][0].float().cpu()
        intri = out["camera_intrinsic"][0].float().cpu()
        conf = out["depth_conf"][0].float().cpu()
        frames_cpu = frames.float().cpu()

        flows = raft_flow_magnitudes(frames)
        photo = []
        for i in range(frames.shape[0] - 1):
            e = photometric_reprojection_l1(frames_cpu[i], frames_cpu[i+1],
                                            pred_depth[i], extri[i], extri[i+1], intri[i])
            photo.append(e)
        photo = np.array(photo)
        mean_flow = float(np.nanmean(flows))
        mean_photo = float(np.nanmean(photo))

        rows.append({"clip_id": clip.clip_id, "domain": "dynamic",
                     "photometric_l1_mean": mean_photo, "flow_mag_mean": mean_flow,
                     "conf_mean": conf.mean().item(), "conf_std": conf.std().item()})

        b_idx = int(np.digitize(mean_flow, bins) - 1)
        bin_rows.append({"clip_id": clip.clip_id, "bin": b_idx, "flow_mag": mean_flow,
                         "photometric_l1": mean_photo})

    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def run(cfg_path="configs/default.yaml", out_dir="results", limit=None):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    out_dir = Path(out_dir)

    model = VGGTWrapper(VGGTConfig(name=cfg["model"]["name"]))
    df_s = run_static(cfg, model, out_dir)
    df_d, df_bins = run_dynamic(cfg, model, out_dir)
    df = pd.concat([df_s, df_d], ignore_index=True)
    df.to_csv(out_dir / "metrics/exp2_static_vs_dynamic.csv", index=False)
    df_bins.to_csv(out_dir / "metrics/exp2_motion_bins.csv", index=False)

    save_scatter(df_bins["flow_mag"], df_bins["photometric_l1"],
                 out_dir / "plots/exp2_quality_vs_motion.png",
                 "mean optical flow magnitude (px)", "photometric L1")
    save_histogram(df_s["conf_mean"].values, out_dir / "plots/exp2_conf_static.png", "conf mean (static)")
    save_histogram(df_d["conf_mean"].values, out_dir / "plots/exp2_conf_dynamic.png", "conf mean (dynamic)")
    json.dump(cfg, open(out_dir / "metrics/exp2_config.json", "w"), indent=2)
    print(df.groupby("domain").describe(include="all"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.config, a.out, a.limit)
