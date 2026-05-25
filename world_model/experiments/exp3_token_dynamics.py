"""Exp 3 — Do VGGT aggregated tokens carry dynamics signal?

This is the most important experiment for Phase 1 viability.
For each clip: correlate per-frame token-difference magnitude with RAFT optical flow magnitude.
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

from src.data_loader import load_frames, load_manifest
from src.metrics import pearson_r
from src.vggt_wrapper import VGGTConfig, VGGTWrapper
from src.viz import save_heatmap_pair, save_histogram


def per_frame_token_delta(aggregated_tokens: list[torch.Tensor]) -> tuple[np.ndarray, torch.Tensor]:
    """Return (per-frame mean delta [N-1], per-frame-per-patch delta [N-1, P])."""
    t = aggregated_tokens[-1]  # last layer, shape [B, N, P, D] or [B, N*P, D] per VGGT impl
    if t.ndim == 4:
        B, N, P, D = t.shape
        t = t[0]  # [N, P, D]
    elif t.ndim == 3:
        t = t[0]
        raise ValueError("Expected 4D aggregated token tensor [B,N,P,D]")
    deltas = (t[1:] - t[:-1]).norm(dim=-1)            # [N-1, P]
    return deltas.mean(dim=-1).float().cpu().numpy(), deltas


def _reshape_patch_map(pmap_1d: np.ndarray, img_hw: tuple[int, int], patch: int = 14,
                        n_extra: int = 5) -> np.ndarray | None:
    """Reshape a per-patch vector [P] into (h,w) grid assuming VGGT's 14-patch tokenizer.
    Returns None if dims don't match (then viz is skipped).
    """
    H, W = img_hw
    h, w = H // patch, W // patch
    total = h * w
    # VGGT adds 4 register + 1 camera token (tail). Strip both ends as needed.
    for skip_prefix in (0, 1, n_extra):
        for skip_suffix in (0, n_extra, 1):
            core = pmap_1d[skip_prefix: len(pmap_1d) - skip_suffix] if skip_suffix else pmap_1d[skip_prefix:]
            if core.size == total:
                return core.reshape(h, w)
    return None


def raft_flow_per_frame(frames: torch.Tensor, device="cuda", target=520) -> tuple[np.ndarray, list[torch.Tensor]]:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    # RAFT needs H,W divisible by 8. VGGT uses 518 which isn't — resize up to nearest /8.
    def _to_div8(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        h8 = ((h + 7) // 8) * 8
        w8 = ((w + 7) // 8) * 8
        if (h, w) == (h8, w8): return x
        return torch.nn.functional.interpolate(x, size=(h8, w8), mode="bilinear", align_corners=False)

    w = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=w).eval().to(device)
    prep = w.transforms()
    mags, fields = [], []
    with torch.no_grad():
        for i in range(frames.shape[0] - 1):
            a = _to_div8(frames[i:i+1].to(device))
            b = _to_div8(frames[i+1:i+2].to(device))
            a, b = prep(a, b)
            a = _to_div8(a); b = _to_div8(b)
            f = model(a, b)[-1][0]
            mags.append(f.norm(dim=0).mean().item())
            fields.append(f.norm(dim=0).cpu())
    del model
    torch.cuda.empty_cache()
    return np.array(mags), fields


def run(cfg_path="configs/default.yaml", out_dir="results", limit=None, manifest_key="set_a"):
    cfg = yaml.safe_load(open(cfg_path))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    out_dir = Path(out_dir)

    model = VGGTWrapper(VGGTConfig(name=cfg["model"]["name"]))
    clips = load_manifest(cfg["datasets"][manifest_key]["manifest"])
    if limit: clips = clips[:limit]

    rows, all_deltas, all_flows = [], [], []
    for idx, clip in enumerate(tqdm(clips, desc=f"exp3/{manifest_key}")):
        frames = load_frames(clip, target_size=cfg["inference"]["image_size"])
        frames = frames[: cfg["inference"]["max_frames_per_clip"]]

        enc = model.encode(frames)
        token_delta, delta_map = per_frame_token_delta(enc["aggregated_tokens"])
        flow_mag, flow_fields = raft_flow_per_frame(frames)

        n = min(len(token_delta), len(flow_mag))
        r, p, n_pts = pearson_r(token_delta[:n], flow_mag[:n])

        rows.append({"clip_id": clip.clip_id, "manifest": manifest_key,
                     "pearson_r": r, "p_value": p, "n_pairs": n_pts,
                     "mean_token_delta": float(np.mean(token_delta)),
                     "mean_flow_mag": float(np.mean(flow_mag))})
        all_deltas.extend(token_delta.tolist()); all_flows.extend(flow_mag.tolist())

        if idx < 3 and delta_map is not None:
            try:
                patch_map = delta_map[0].float().cpu().numpy()            # [P]
                img_hw = tuple(frames.shape[-2:])
                grid = _reshape_patch_map(patch_map, img_hw)
                flow_img = flow_fields[0].numpy()
                if grid is not None:
                    save_heatmap_pair(grid, flow_img,
                                      out_dir / f"qualitative/exp3_{manifest_key}_{clip.clip_id}.png",
                                      labels=("token Δ (patch)", "optical flow mag"))
            except Exception as e:
                print(f"  [viz skipped for {clip.clip_id}: {e}]")

    df = pd.DataFrame(rows)
    tag = "" if manifest_key == "set_a" else f"_{manifest_key}"
    df.to_csv(out_dir / f"metrics/exp3_token_flow_correlation{tag}.csv", index=False)
    save_histogram(df["pearson_r"].values, out_dir / f"plots/exp3_correlation_distribution{tag}.png",
                   xlabel="per-clip Pearson r (token Δ vs flow)")
    json.dump(cfg, open(out_dir / f"metrics/exp3_config{tag}.json", "w"), indent=2)
    print("median r:", df["pearson_r"].median(), "n clips:", len(df))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest", default="set_a")
    a = ap.parse_args()
    run(a.config, a.out, a.limit, a.manifest)
