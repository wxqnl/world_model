"""v3 quantitative eval: per-dataset breakdown on val set + best.pt."""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import lpips
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.splits import random_window_indices, split_mode_from_config, split_records
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import _normalize_depth
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


def build_model(cfg: dict) -> JointWorldModel:
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
        enable_geom_extra=cfg["model"].get("enable_geom_extra", True),
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_context_pixel=cfg["model"].get("enable_context_pixel", False),
        context_pixel_hidden=cfg["model"].get("context_pixel_hidden", 384),
        context_pixel_action_dim=cfg["model"].get("context_pixel_action_dim", 7),
        context_pixel_task_dim=cfg["model"].get("context_pixel_task_dim"),
        context_pixel_residual_scale=cfg["model"].get("context_pixel_residual_scale", 0.75),
        context_pixel_use_action=cfg["model"].get("context_pixel_use_action", True),
        context_pixel_use_task=cfg["model"].get("context_pixel_use_task", True),
        context_pixel_predict_motion=cfg["model"].get("context_pixel_predict_motion", False),
        context_pixel_motion_blend_gain=cfg["model"].get("context_pixel_motion_blend_gain", 0.0),
        enable_control_head=cfg["model"].get("enable_control_head", False),
        control_hidden=cfg["model"].get("control_hidden", 128),
        control_output_size=cfg["model"].get("control_output_size", 256),
        control_fuse_size=cfg["model"].get("control_fuse_size", 64),
        control_refine_channels=cfg["model"].get("control_refine_channels", 16),
        control_use_refine=cfg["model"].get("control_use_refine", True),
        control_action_dim=cfg["model"].get("control_action_dim", 7),
        control_task_dim=cfg["model"].get("control_task_dim"),
        control_use_context=cfg["model"].get("control_use_context", True),
        control_use_action=cfg["model"].get("control_use_action", True),
        control_use_task=cfg["model"].get("control_use_task", True),
        enable_progress_head=cfg["model"].get("enable_progress_head", False),
        progress_hidden=cfg["model"].get("progress_hidden", 256),
        progress_layers=cfg["model"].get("progress_layers", 2),
        progress_heads=cfg["model"].get("progress_heads", 4),
        progress_action_dim=cfg["model"].get("progress_action_dim", 7),
        progress_task_dim=cfg["model"].get("progress_task_dim"),
        progress_max_horizon=cfg["model"].get("progress_max_horizon", 32),
        progress_use_action=cfg["model"].get("progress_use_action", True),
        progress_use_task=cfg["model"].get("progress_use_task", True),
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def window_config_from_cfg(cfg: dict) -> WindowConfig:
    data = cfg["data"]
    action_stats = data.get("action_stats")
    return WindowConfig(
        T=data["T"],
        k=data["k"],
        stride=data["stride"],
        cache_root=Path(data["cache_root"]),
        tokens_subdir=data.get("tokens_subdir", "vggt_pooled"),
        action_stats=Path(action_stats) if action_stats else None,
    )


def build_dataset_for_split(records, cfg: dict, split: str = "val"):
    """Build a window dataset/subset using config-defined split semantics."""
    if split not in {"train", "val", "all"}:
        raise ValueError(f"split must be train/val/all, got {split}")
    wcfg = window_config_from_cfg(cfg)
    if split == "all":
        return OXEWindowDataset(records, wcfg)
    data_cfg = cfg["data"]
    mode = split_mode_from_config(data_cfg)
    if mode == "episode":
        train_records, val_records = split_records(records, data_cfg)
        selected = train_records if split == "train" else val_records
        ds = OXEWindowDataset(selected, wcfg)
        if len(ds) == 0:
            raise RuntimeError(f"episode {split} split empty — caches missing?")
        return ds
    if mode != "random_window":
        raise ValueError(f"unsupported data.split.mode: {mode}")
    ds = OXEWindowDataset(records, wcfg)
    if len(ds) == 0:
        raise RuntimeError("OXEWindowDataset empty — caches missing?")
    train_idx, val_idx = random_window_indices(
        len(ds),
        val_frac=float(data_cfg.get("val_frac", 0.05)),
        seed=int(data_cfg.get("seed", 0)),
    )
    return Subset(ds, train_idx if split == "train" else val_idx)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="cap batches per dataset for quicker eval (0=all)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    val = build_dataset_for_split(records, cfg, split="val")
    print(f"val windows: {len(val)}")

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={sd.get('val_total'):.4f}")

    lp = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lp.parameters():
        p.requires_grad = False

    loader = DataLoader(val, batch_size=cfg["train"]["batch_size_per_gpu"],
                        shuffle=False, num_workers=cfg["train"]["num_workers"],
                        pin_memory=True)

    by_ds: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    cnt: dict[str, int] = defaultdict(int)

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        s_tgt = batch["s_tgt"].to(device, non_blocking=True)
        depth_tgt = batch["depth_tgt"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            kwargs = dict(action_cond=action_cond, pixel=True, bridging=False)
            if cfg["model"].get("enable_context_pixel", False):
                kwargs["context_rgb"] = context_rgb
            out = model(s, c, **kwargs)
        pred_s = out["pred_tokens"]
        L_state_mse = F.mse_loss(pred_s.float(), s_tgt.float(), reduction="none").mean(dim=(1, 2, 3))
        cos = F.cosine_similarity(pred_s.float().flatten(-2),
                                  s_tgt.float().flatten(-2), dim=-1).mean(dim=-1)
        depth_n = _normalize_depth(out["depth"].float())
        depth_tn = _normalize_depth(depth_tgt.float())
        L_depth = (depth_n - depth_tn).abs().mean(dim=(1, 2, 3))
        L_pose = (out["pose"].float() - action_tgt[..., :6]).pow(2).mean(dim=(1, 2))
        grip_tgt = (action_tgt[..., 6] > 0.5).float()
        grip_acc = ((out["gripper_logit"].float().sigmoid() > 0.5).float() == grip_tgt).float().mean(dim=-1)
        rgb_pred = out["rgb"].float()
        L_rgb_l1 = (rgb_pred - rgb_tgt).abs().mean(dim=(1, 2, 3, 4))
        rgb_ref = context_rgb.unsqueeze(1)
        motion = (rgb_tgt - rgb_ref).abs().mean(dim=2, keepdim=True)
        motion_mask = (motion > 0.03).float()
        motion_denom = (motion_mask.sum(dim=(1, 2, 3, 4)) * rgb_tgt.shape[2]).clamp_min(1.0)
        L_rgb_motion_l1 = (
            (rgb_pred - rgb_tgt).abs() * motion_mask
        ).sum(dim=(1, 2, 3, 4)) / motion_denom
        motion_frac = motion_mask.mean(dim=(1, 2, 3, 4))
        with torch.autocast(device_type="cuda", enabled=False):
            rp = (rgb_pred.flatten(0, 1) * 2 - 1)
            rt = (rgb_tgt.flatten(0, 1) * 2 - 1)
            lpv = lp(rp, rt).view(rgb_pred.shape[0], rgb_pred.shape[1]).mean(dim=-1)
        for i in range(s.shape[0]):
            d = batch["dataset"][i]
            by_ds[d]["L_state_mse"] += float(L_state_mse[i])
            by_ds[d]["cos_sim"] += float(cos[i])
            by_ds[d]["L_depth_rel_L1"] += float(L_depth[i])
            by_ds[d]["L_pose_mse"] += float(L_pose[i])
            by_ds[d]["grip_acc"] += float(grip_acc[i])
            by_ds[d]["L_rgb_L1"] += float(L_rgb_l1[i])
            by_ds[d]["L_rgb_lpips"] += float(lpv[i])
            by_ds[d]["L_rgb_motion_L1"] += float(L_rgb_motion_l1[i])
            by_ds[d]["motion_frac"] += float(motion_frac[i])
            cnt[d] += 1
        by_ds["ALL"]["L_state_mse"] += float(L_state_mse.sum())
        by_ds["ALL"]["cos_sim"] += float(cos.sum())
        by_ds["ALL"]["L_depth_rel_L1"] += float(L_depth.sum())
        by_ds["ALL"]["L_pose_mse"] += float(L_pose.sum())
        by_ds["ALL"]["grip_acc"] += float(grip_acc.sum())
        by_ds["ALL"]["L_rgb_L1"] += float(L_rgb_l1.sum())
        by_ds["ALL"]["L_rgb_lpips"] += float(lpv.sum())
        by_ds["ALL"]["L_rgb_motion_L1"] += float(L_rgb_motion_l1.sum())
        by_ds["ALL"]["motion_frac"] += float(motion_frac.sum())
        cnt["ALL"] += s.shape[0]
        if (bi + 1) % 25 == 0:
            print(f"[{bi+1}/{len(loader)}] L_state {by_ds['ALL']['L_state_mse']/cnt['ALL']:.4f} "
                  f"L_depth {by_ds['ALL']['L_depth_rel_L1']/cnt['ALL']:.4f} "
                  f"L_pose {by_ds['ALL']['L_pose_mse']/cnt['ALL']:.4f} "
                  f"grip {by_ds['ALL']['grip_acc']/cnt['ALL']:.4f} "
                  f"rgb_L1 {by_ds['ALL']['L_rgb_L1']/cnt['ALL']:.4f} "
                  f"lpips {by_ds['ALL']['L_rgb_lpips']/cnt['ALL']:.4f} "
                  f"motion_L1 {by_ds['ALL']['L_rgb_motion_L1']/cnt['ALL']:.4f}")

    report = {"counts": dict(cnt), "metrics": {}}
    for d, mvals in by_ds.items():
        n = max(1, cnt[d])
        report["metrics"][d] = {k: v / n for k, v in mvals.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
