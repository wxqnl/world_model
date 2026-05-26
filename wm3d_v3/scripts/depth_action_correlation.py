"""Step 0 sanity check: does VGGT depth carry motion signal correlated with action?

For N random val windows, compute:
  - per-window scalar: ||depth_tgt[k] - depth_tgt[0]||_F  (total depth change across future window)
  - per-window scalar: sum_t ||action_tgt[t, :3]||_2     (total translation magnitude in same window)
  - per-frame: ||depth_tgt[t+1] - depth_tgt[t]||_F  vs  ||action_tgt[t, :3]||_2

Report Pearson correlation per dataset and pooled. Output a couple of scatter PNGs
so the strength is visually obvious.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path,
                    default=Path("/home/user01/Minko/newwm/wm3d_v3/configs/v3_vla.yaml"))
    ap.add_argument("--n_windows", type=int, default=500)
    ap.add_argument("--out", type=Path,
                    default=Path("/home/user01/Minko/newwm/results/wm3d_v3/eval/vla_analysis"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    records = read_manifest(cfg["data"]["manifest"])
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]),
                        action_stats=Path(cfg["data"]["action_stats"]))
    ds = OXEWindowDataset(records, wcfg)
    print(f"total windows: {len(ds)}")

    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(len(ds), generator=g).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))
    val_idx = perm[:n_val]
    # sample N from val
    rng = np.random.default_rng(0)
    sel = rng.choice(len(val_idx), size=min(args.n_windows, len(val_idx)), replace=False)
    sel = [val_idx[i] for i in sel]
    sub = Subset(ds, sel)
    loader = DataLoader(sub, batch_size=8, shuffle=False, num_workers=4)

    # Per-window scalars
    pw_depth_total = []     # ||depth_tgt[k-1] - depth_tgt[0]||_F
    pw_action_total = []    # sum_t ||action[t,:3]||_2
    pw_action_rot = []      # sum_t ||action[t,3:6]||_2
    pw_dataset = []

    # Per-frame paired arrays (within target window)
    pf_depth = []
    pf_act_trans = []
    pf_act_rot = []
    pf_dataset = []

    for batch in loader:
        depth_tgt = batch["depth_tgt"]    # [B, k, 224, 224]
        action_tgt = batch["action_tgt"]  # [B, k, 7]
        datasets = batch["dataset"]
        B, k, H, W = depth_tgt.shape
        # per-window
        ddiff_total = (depth_tgt[:, -1] - depth_tgt[:, 0]).reshape(B, -1).norm(dim=-1)  # [B]
        adiff_trans = action_tgt[:, :, :3].norm(dim=-1).sum(dim=-1)                     # [B]
        adiff_rot = action_tgt[:, :, 3:6].norm(dim=-1).sum(dim=-1)                      # [B]
        # per-frame (k-1 pairs)
        d_per_frame = (depth_tgt[:, 1:] - depth_tgt[:, :-1]).reshape(B, k - 1, -1).norm(dim=-1)  # [B, k-1]
        a_trans_pf = action_tgt[:, :-1, :3].norm(dim=-1)                                          # [B, k-1]
        a_rot_pf = action_tgt[:, :-1, 3:6].norm(dim=-1)
        for i in range(B):
            pw_depth_total.append(float(ddiff_total[i]))
            pw_action_total.append(float(adiff_trans[i]))
            pw_action_rot.append(float(adiff_rot[i]))
            pw_dataset.append(datasets[i])
            for j in range(k - 1):
                pf_depth.append(float(d_per_frame[i, j]))
                pf_act_trans.append(float(a_trans_pf[i, j]))
                pf_act_rot.append(float(a_rot_pf[i, j]))
                pf_dataset.append(datasets[i])

    pw_depth_total = np.array(pw_depth_total)
    pw_action_total = np.array(pw_action_total)
    pw_action_rot = np.array(pw_action_rot)
    pw_dataset = np.array(pw_dataset)
    pf_depth = np.array(pf_depth)
    pf_act_trans = np.array(pf_act_trans)
    pf_act_rot = np.array(pf_act_rot)
    pf_dataset = np.array(pf_dataset)

    def corr(x, y):
        if len(x) < 5:
            return float("nan")
        m = np.isfinite(x) & np.isfinite(y) & (x.std() > 1e-9) & (y.std() > 1e-9)
        if m.sum() < 5:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    print(f"\n{'='*72}")
    print("Per-window correlations (||depth[k-1]-depth[0]|| vs sum action norm)")
    print(f"{'='*72}")
    for d in sorted(set(pw_dataset.tolist())):
        m = pw_dataset == d
        n = m.sum()
        c_t = corr(pw_depth_total[m], pw_action_total[m])
        c_r = corr(pw_depth_total[m], pw_action_rot[m])
        print(f"  {d:30s} N={n:4d}  depth↔Δtrans: {c_t:+.3f}  depth↔Δrot: {c_r:+.3f}")
    c_t_all = corr(pw_depth_total, pw_action_total)
    c_r_all = corr(pw_depth_total, pw_action_rot)
    print(f"  {'ALL':30s} N={len(pw_dataset):4d}  depth↔Δtrans: {c_t_all:+.3f}  depth↔Δrot: {c_r_all:+.3f}")

    print(f"\n{'='*72}")
    print("Per-frame correlations (||depth[t+1]-depth[t]|| vs ||action[t,:3]||)")
    print(f"{'='*72}")
    for d in sorted(set(pf_dataset.tolist())):
        m = pf_dataset == d
        n = m.sum()
        c_t = corr(pf_depth[m], pf_act_trans[m])
        c_r = corr(pf_depth[m], pf_act_rot[m])
        print(f"  {d:30s} N={n:5d}  depth↔trans: {c_t:+.3f}  depth↔rot: {c_r:+.3f}")
    c_t_all = corr(pf_depth, pf_act_trans)
    c_r_all = corr(pf_depth, pf_act_rot)
    print(f"  {'ALL':30s} N={len(pf_dataset):5d}  depth↔trans: {c_t_all:+.3f}  depth↔rot: {c_r_all:+.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"bridge": "#2c7fb8", "fractal20220817_data": "#d73027"}
    for d in sorted(set(pw_dataset.tolist())):
        m = pw_dataset == d
        axes[0].scatter(pw_depth_total[m], pw_action_total[m], s=8, alpha=0.5,
                         label=d, c=colors.get(d, "#888"))
        axes[1].scatter(pw_depth_total[m], pw_action_rot[m], s=8, alpha=0.5,
                         label=d, c=colors.get(d, "#888"))
    axes[0].set_xlabel("||depth[k-1] - depth[0]||_F (per-window total depth change)")
    axes[0].set_ylabel("Σ ||action[:, :3]||_2 (per-window total translation)")
    axes[0].set_title(f"depth-motion vs translation  r={c_t_all:+.3f}")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("||depth[k-1] - depth[0]||_F")
    axes[1].set_ylabel("Σ ||action[:, 3:6]||_2 (rotation)")
    axes[1].set_title(f"depth-motion vs rotation  r={c_r_all:+.3f}")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out / "depth_action_corr.png", dpi=120)
    plt.close()
    print(f"\nwrote {args.out/'depth_action_corr.png'}")


if __name__ == "__main__":
    main()
