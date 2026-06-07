"""VLA offline analysis (Option A).

Outputs:
  - report.json : per-dataset per-axis RMSE, per-step MSE, baseline comparisons,
                  gripper switch precision/recall/F1
  - plots/per_axis_rmse_<ds>.png    : 6-DoF axis breakdown vs baselines
  - plots/per_step_mse_<ds>.png     : MSE vs future step (1..k) vs baselines
  - plots/trajectories/<clip>.png   : cumulative xyz + gripper from demo npz files
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


def build_model(cfg: dict, variant: str = "a"):
    if variant == "c":
        from wm3d_v3.models.idm_stream import IDMStreamConfig
        from wm3d_v3.models.depth_encoder import DepthEncoderConfig
        from wm3d_v3.models.joint_model_c import JointCConfig, JointWorldModelC
        sc = StateConfig(**cfg["model"]["state"])
        ic = IDMStreamConfig(**cfg["model"]["idm"])
        dec = DepthEncoderConfig(**cfg["model"]["depth_enc"])
        jc = JointCConfig(
            state=sc, idm=ic, depth_enc=dec,
            action_proj_hidden=cfg["model"]["action_proj_hidden"],
            action_proj_layers=cfg["model"]["action_proj_layers"],
            geom_hidden=cfg["model"]["geom_hidden"],
            pixel_hidden=cfg["model"]["pixel_hidden"],
            pixel_n_res=cfg["model"]["pixel_n_res"],
            enable_pixel=cfg["model"].get("enable_pixel", False),
        )
        return JointWorldModelC(jc)
    if variant == "b":
        from wm3d_v3.models.idm_stream import IDMStreamConfig
        from wm3d_v3.models.joint_model_b import JointBConfig, JointWorldModelB
        sc = StateConfig(**cfg["model"]["state"])
        ic = IDMStreamConfig(**cfg["model"]["idm"])
        jc = JointBConfig(
            state=sc, idm=ic,
            action_proj_hidden=cfg["model"]["action_proj_hidden"],
            action_proj_layers=cfg["model"]["action_proj_layers"],
            geom_hidden=cfg["model"]["geom_hidden"],
            pixel_hidden=cfg["model"]["pixel_hidden"],
            pixel_n_res=cfg["model"]["pixel_n_res"],
            enable_pixel=cfg["model"].get("enable_pixel", False),
        )
        return JointWorldModelB(jc)
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    jc = JointConfig(
        dual=dc,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_bridging=cfg["model"].get("enable_bridging", True),
        enable_aux_idm=cfg["model"].get("enable_aux_idm", False),
        aux_idm_hidden=cfg["model"].get("aux_idm_hidden", 1024),
        aux_idm_layers=cfg["model"].get("aux_idm_layers", 3),
    )
    return JointWorldModel(jc)


@torch.no_grad()
def collect(args):
    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    records = read_manifest(cfg["data"]["manifest"])
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]))
    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * cfg["data"]["val_frac"]))
    val = Subset(ds, perm[:n_val])
    print(f"val windows: {len(val)}")

    model = build_model(cfg, variant=args.variant).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={sd.get('val_total'):.4f} variant={args.variant}")

    loader = DataLoader(val, batch_size=cfg["train"]["batch_size_per_gpu"],
                        shuffle=False, num_workers=cfg["train"]["num_workers"],
                        pin_memory=True)

    per_ds = defaultdict(lambda: {"pose_pred": [], "pose_tgt": [],
                                   "grip_logit": [], "grip_tgt": []})
    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if args.variant == "c":
                depth_in = batch["depth_in"].to(device, non_blocking=True)
                out = model(s, c, depth_in=depth_in, pixel=False)
            elif args.variant == "b":
                out = model(s, c, pixel=False)
            else:
                out = model(s, c, pixel=False, bridging=False)
        pose = out["pose"].float().cpu()
        grip_logit = out["gripper_logit"].float().cpu()
        for i in range(s.shape[0]):
            d = batch["dataset"][i]
            per_ds[d]["pose_pred"].append(pose[i].numpy())
            per_ds[d]["pose_tgt"].append(action_tgt[i, :, :6].numpy())
            per_ds[d]["grip_logit"].append(grip_logit[i].numpy())
            per_ds[d]["grip_tgt"].append((action_tgt[i, :, 6] > 0.5).numpy())
        if (bi + 1) % 25 == 0:
            n_so_far = sum(len(v["pose_pred"]) for v in per_ds.values())
            print(f"[{bi+1}/{len(loader)}] collected {n_so_far} windows")
    return per_ds


def compute_report(per_ds: dict) -> dict:
    report = {}
    for d, dat in per_ds.items():
        pp = np.stack(dat["pose_pred"])                          # [N, k, 6]
        pt = np.stack(dat["pose_tgt"])
        gl = np.stack(dat["grip_logit"])                         # [N, k]
        gt = np.stack(dat["grip_tgt"]).astype(bool)
        gp = (1.0 / (1.0 + np.exp(-gl))) > 0.5
        N, k, _ = pp.shape

        per_axis_rmse = np.sqrt(((pp - pt) ** 2).mean(axis=(0, 1)))
        per_step_mse = ((pp - pt) ** 2).mean(axis=(0, 2))
        zero_rmse_axis = np.sqrt((pt ** 2).mean(axis=(0, 1)))
        zero_mse_step = (pt ** 2).mean(axis=(0, 2))
        mean_pred = pt.mean(axis=(0, 1), keepdims=True)
        mean_rmse_axis = np.sqrt(((pt - mean_pred) ** 2).mean(axis=(0, 1)))
        mean_mse_step = ((pt - mean_pred) ** 2).mean(axis=(0, 2))

        gt_switch = np.zeros_like(gt)
        gt_switch[:, 1:] = gt[:, 1:] != gt[:, :-1]
        gp_switch = np.zeros_like(gp)
        gp_switch[:, 1:] = gp[:, 1:] != gp[:, :-1]
        tp = int((gt_switch & gp_switch).sum())
        fp = int((~gt_switch & gp_switch).sum())
        fn = int((gt_switch & ~gp_switch).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-9, prec + rec)

        report[d] = {
            "n_windows": int(N),
            "k": int(k),
            "per_axis_rmse": per_axis_rmse.tolist(),
            "per_axis_rmse_zero_baseline": zero_rmse_axis.tolist(),
            "per_axis_rmse_mean_baseline": mean_rmse_axis.tolist(),
            "per_step_mse": per_step_mse.tolist(),
            "per_step_mse_zero_baseline": zero_mse_step.tolist(),
            "per_step_mse_mean_baseline": mean_mse_step.tolist(),
            "pose_mse_overall": float(((pp - pt) ** 2).mean()),
            "pose_mse_zero_baseline": float((pt ** 2).mean()),
            "pose_mse_mean_baseline": float(((pt - mean_pred) ** 2).mean()),
            "grip_acc": float((gp == gt).mean()),
            "grip_switch_count": int(gt_switch.sum()),
            "grip_switch_precision": float(prec),
            "grip_switch_recall": float(rec),
            "grip_switch_f1": float(f1),
        }
    return report


AXIS_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz"]


def plot_quant(report: dict, plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(AXIS_NAMES))
    for d, r in report.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        w = 0.27
        ax.bar(x - w, r["per_axis_rmse_zero_baseline"], w, label="zero-pred", color="#c0c0c0")
        ax.bar(x,      r["per_axis_rmse_mean_baseline"], w, label="dataset-mean", color="#9db4d4")
        ax.bar(x + w,  r["per_axis_rmse"], w, label="v3 model", color="#2c7fb8")
        ax.set_xticks(x); ax.set_xticklabels(AXIS_NAMES)
        ax.set_ylabel("RMSE")
        ax.set_title(f"Per-axis pose RMSE — {d}  (N={r['n_windows']})")
        ax.legend(); ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(plot_dir / f"per_axis_rmse_{d}.png", dpi=120)
        plt.close()

        steps = np.arange(1, r["k"] + 1)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(steps, r["per_step_mse"], "-o", label="v3 model", color="#2c7fb8")
        ax.plot(steps, r["per_step_mse_mean_baseline"], "--", label="dataset-mean", color="#9db4d4")
        ax.plot(steps, r["per_step_mse_zero_baseline"], "--", label="zero-pred", color="#c0c0c0")
        ax.set_xlabel("future step k")
        ax.set_ylabel("pose MSE")
        ax.set_title(f"Pose MSE vs future step — {d}  (N={r['n_windows']})")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"per_step_mse_{d}.png", dpi=120)
        plt.close()


def plot_trajectories(demo_dirs: list[Path], plot_dir: Path) -> None:
    out = plot_dir / "trajectories"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for dd in demo_dirs:
        if dd.exists():
            files.extend(sorted(dd.glob("*_action.npz")))
    for f in files:
        d = np.load(f)
        if "action_gt" not in d.files:
            continue  # long-rollout npz has no GT
        pose_pred = d["pose_pred"]           # [k, 6]
        grip_pred = d["grip_pred"]           # [k]  (sigmoid)
        action_gt = d["action_gt"]           # [k, 7]
        k = pose_pred.shape[0]
        xyz_pred = np.cumsum(pose_pred[:, :3], axis=0)
        xyz_gt = np.cumsum(action_gt[:, :3], axis=0)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ai, name in enumerate(["x", "y", "z"]):
            axes[ai].plot(xyz_gt[:, ai], "-o", label="GT", color="#1a9850")
            axes[ai].plot(xyz_pred[:, ai], "--x", label="pred", color="#d73027")
            axes[ai].set_title(f"cumulative {name}")
            axes[ai].grid(alpha=0.3)
            axes[ai].legend()
        axes[3].step(range(k), action_gt[:, 6], where="post", label="GT", color="#1a9850")
        axes[3].step(range(k), (grip_pred > 0.5).astype(int), where="post",
                     label="pred (sigmoid>0.5)", color="#d73027", linestyle="--")
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].set_title("gripper closed")
        axes[3].grid(alpha=0.3); axes[3].legend()
        fig.suptitle(f.stem, fontsize=10)
        plt.tight_layout()
        plt.savefig(out / (f.stem + ".png"), dpi=110)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="0 = full val set")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--variant", choices=["a", "b", "c"], default="a",
                    help="a = JointWorldModel (Stage A), b = JointWorldModelB (Stage B), c = JointWorldModelC (Stage C)")
    ap.add_argument("--demo_dirs", nargs="*", default=[
        "/home/user01/Minko/newwm/results/wm3d_v3/eval/demo",
        "/home/user01/Minko/newwm/results/wm3d_v3/eval/demo_full",
        "/home/user01/Minko/newwm/results/wm3d_v3/eval/demo_long",
    ])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out / "plots"

    per_ds = collect(args)
    report = compute_report(per_ds)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out/'report.json'}")
    print(json.dumps(report, indent=2))

    plot_quant(report, plot_dir)
    plot_trajectories([Path(p) for p in args.demo_dirs], plot_dir)
    print(f"plots in {plot_dir}")


if __name__ == "__main__":
    main()
