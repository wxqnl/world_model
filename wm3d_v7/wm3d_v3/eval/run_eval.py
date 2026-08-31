"""v3 quantitative eval: per-dataset breakdown on val set + best.pt."""
from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

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
from wm3d_v3.models.model_factory import build_joint_world_model
from wm3d_v3.models.state_stream import StateConfig


def _legacy_build_model(cfg: dict) -> JointWorldModel:
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
        geom_upsample_mode=cfg["model"].get("geom_upsample_mode", "transpose"),
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
        enable_action_proposer=cfg["model"].get("enable_action_proposer", False),
        proposer_hidden=cfg["model"].get("proposer_hidden", 512),
        proposer_layers=cfg["model"].get("proposer_layers", 3),
        proposer_candidates=cfg["model"].get("proposer_candidates", 4),
        proposer_horizon=cfg["model"].get("proposer_horizon"),
        proposer_task_dim=cfg["model"].get("proposer_task_dim"),
        proposer_dropout=cfg["model"].get("proposer_dropout", 0.0),
        proposer_use_task=cfg["model"].get("proposer_use_task", True),
        enable_action_policy=cfg["model"].get("enable_action_policy", False),
        policy_hidden=cfg["model"].get("policy_hidden", 768),
        policy_layers=cfg["model"].get("policy_layers", 6),
        policy_heads=cfg["model"].get("policy_heads", 8),
        policy_chunk_layers=cfg["model"].get("policy_chunk_layers", 2),
        policy_horizon=cfg["model"].get("policy_horizon"),
        policy_task_dim=cfg["model"].get("policy_task_dim"),
        policy_max_context=cfg["model"].get("policy_max_context"),
        policy_dropout=cfg["model"].get("policy_dropout", 0.1),
        policy_use_task=cfg["model"].get("policy_use_task", True),
        policy_patch_pool=cfg["model"].get("policy_patch_pool", "mean"),
        policy_max_spatial_tokens=cfg["model"].get("policy_max_spatial_tokens", 64),
        policy_use_context_rgb=cfg["model"].get("policy_use_context_rgb", False),
        policy_context_source=cfg["model"].get("policy_context_source", "input"),
        policy_core_action_cond=cfg["model"].get("policy_core_action_cond", "same"),
        policy_rgb_spatial_tokens=cfg["model"].get("policy_rgb_spatial_tokens", 64),
        policy_lowdim_dim=cfg["model"].get("policy_lowdim_dim", 0),
        policy_object_state_dim=cfg["model"].get("policy_object_state_dim", 0),
        policy_plan_state_dim=cfg["model"].get("policy_plan_state_dim", 0),
        policy_action_history_len=cfg["model"].get("policy_action_history_len", 0),
        policy_action_history_dim=cfg["model"].get("policy_action_history_dim", 7),
        policy_action_history_as_token=cfg["model"].get("policy_action_history_as_token", True),
        policy_grip_history_adapter=cfg["model"].get("policy_grip_history_adapter", False),
        policy_grip_history_hidden=cfg["model"].get("policy_grip_history_hidden", 128),
        policy_grip_history_zero_init=cfg["model"].get("policy_grip_history_zero_init", True),
        policy_enable_grip_delta_head=cfg["model"].get("policy_enable_grip_delta_head", False),
        policy_grip_delta_hidden=cfg["model"].get("policy_grip_delta_hidden", 256),
        policy_grip_delta_zero_init=cfg["model"].get("policy_grip_delta_zero_init", True),
        policy_grip_delta_use_composed_action_cond=cfg["model"].get("policy_grip_delta_use_composed_action_cond", False),
        policy_grip_delta_soft_compose_action_cond=cfg["model"].get("policy_grip_delta_soft_compose_action_cond", False),
        policy_grip_delta_straight_through_action_cond=cfg["model"].get(
            "policy_grip_delta_straight_through_action_cond", False
        ),
        policy_use_progress=cfg["model"].get("policy_use_progress", False),
        policy_progress_dim=cfg["model"].get("policy_progress_dim", 1),
        policy_progress_mode=cfg["model"].get("policy_progress_mode", "token"),
        policy_enable_local_residual=cfg["model"].get("policy_enable_local_residual", False),
        policy_local_hidden=cfg["model"].get("policy_local_hidden", 256),
        policy_local_layers=cfg["model"].get("policy_local_layers", 2),
        policy_local_residual_scale=cfg["model"].get("policy_local_residual_scale", 1.0),
        policy_local_use_lowdim=cfg["model"].get("policy_local_use_lowdim", True),
        policy_local_use_plan_state=cfg["model"].get("policy_local_use_plan_state", True),
        policy_local_use_progress=cfg["model"].get("policy_local_use_progress", True),
        policy_local_use_action_history=cfg["model"].get("policy_local_use_action_history", True),
        policy_enable_waypoint_head=cfg["model"].get("policy_enable_waypoint_head", False),
        policy_waypoint_hidden=cfg["model"].get("policy_waypoint_hidden", 256),
        policy_waypoint_layers=cfg["model"].get("policy_waypoint_layers", 2),
        policy_waypoint_num_stages=cfg["model"].get("policy_waypoint_num_stages", 4),
        policy_waypoint_stage_dim=cfg["model"].get("policy_waypoint_stage_dim", 4),
        policy_waypoint_active_stages=tuple(cfg["model"].get("policy_waypoint_active_stages", ())),
        policy_waypoint_residual_scale=cfg["model"].get("policy_waypoint_residual_scale", 1.0),
        policy_waypoint_mode=cfg["model"].get("policy_waypoint_mode", "residual"),
        policy_waypoint_use_summary=cfg["model"].get("policy_waypoint_use_summary", True),
        policy_waypoint_use_lowdim=cfg["model"].get("policy_waypoint_use_lowdim", True),
        policy_waypoint_use_plan_state=cfg["model"].get("policy_waypoint_use_plan_state", True),
        policy_waypoint_use_progress=cfg["model"].get("policy_waypoint_use_progress", True),
        policy_waypoint_use_action_history=cfg["model"].get("policy_waypoint_use_action_history", True),
        policy_enable_prior=cfg["model"].get("policy_enable_prior", False),
        policy_prior_chunk_layers=cfg["model"].get("policy_prior_chunk_layers", 1),
        policy_action_add_trunk=cfg["model"].get("policy_action_add_trunk", True),
        policy_enable_flow_head=cfg["model"].get("policy_enable_flow_head", False),
        policy_flow_use_as_policy=cfg["model"].get("policy_flow_use_as_policy", False),
        policy_flow_layers=cfg["model"].get("policy_flow_layers", 2),
        policy_flow_hidden=cfg["model"].get("policy_flow_hidden", cfg["model"].get("policy_hidden", 768)),
        policy_flow_action_dim=cfg["model"].get("policy_flow_action_dim", 7),
        policy_flow_default_steps=cfg["model"].get("policy_flow_default_steps", 8),
        policy_flow_noise_scale=cfg["model"].get("policy_flow_noise_scale", 1.0),
        policy_flow_zero_init_output=cfg["model"].get("policy_flow_zero_init_output", False),
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def build_model(cfg: dict) -> JointWorldModel:
    return build_joint_world_model(cfg["model"])


def window_config_from_cfg(cfg: dict) -> WindowConfig:
    data = cfg["data"]
    model_cfg = cfg.get("model", {})
    policy_state_default = any(
        int(model_cfg.get(key, 0) or 0) > 0
        for key in (
            "policy_lowdim_dim",
            "policy_object_state_dim",
            "policy_plan_state_dim",
            "policy_action_history_len",
        )
    ) or bool(model_cfg.get("policy_use_progress", False))
    action_stats = data.get("action_stats")
    return WindowConfig(
        T=data["T"],
        k=data["k"],
        stride=data["stride"],
        cache_root=Path(data["cache_root"]),
        tokens_subdir=data.get("tokens_subdir", "vggt_pooled"),
        action_stats=Path(action_stats) if action_stats else None,
        require_task_emb=bool(data.get("require_task_emb", False)),
        load_task_text=bool(data.get("load_task_text", False)),
        load_rgb=bool(data.get("load_rgb", True)),
        load_geom=bool(data.get("load_geom", True)),
        load_state_tgt=bool(data.get("load_state_tgt", True)),
        load_geom_extra=bool(data.get("load_geom_extra", False)),
        require_geom_extra=bool(data.get("require_geom_extra", False)),
        window_geom_subdir=data.get("window_geom_subdir", "vggt_window_geom_p64"),
        window_geom_shard_index=Path(data["window_geom_shard_index"])
        if data.get("window_geom_shard_index") else None,
        window_geom_shard_root=Path(data["window_geom_shard_root"])
        if data.get("window_geom_shard_root") else None,
        use_window_tokens=bool(data.get("use_window_tokens", False)),
        max_windows_per_episode=int(data.get("max_windows_per_episode", 0) or 0),
        trust_window_geom_cache=bool(data.get("trust_window_geom_cache", False)),
        allow_pseudo_progress_targets=bool(data.get("allow_pseudo_progress_targets", False)),
        require_progress=bool(data.get("require_progress", bool(model_cfg.get("policy_use_progress", False)))),
        load_policy_state=bool(data.get("load_policy_state", policy_state_default)),
        require_policy_state=bool(data.get("require_policy_state", False)),
        policy_lowdim_dim=int(data.get("policy_lowdim_dim", model_cfg.get("policy_lowdim_dim", 0)) or 0),
        policy_object_state_dim=int(data.get("policy_object_state_dim", model_cfg.get("policy_object_state_dim", 0)) or 0),
        policy_plan_state_dim=int(data.get("policy_plan_state_dim", model_cfg.get("policy_plan_state_dim", 0)) or 0),
        policy_action_history_len=int(data.get("policy_action_history_len", model_cfg.get("policy_action_history_len", 0)) or 0),
        policy_action_history_dim=int(data.get("policy_action_history_dim", model_cfg.get("policy_action_history_dim", 7)) or 7),
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
    ap.add_argument("--batch_size", type=int, default=0,
                    help="override eval batch size (0=use train.batch_size_per_gpu)")
    ap.add_argument(
        "--skip_rgb_metrics",
        action="store_true",
        help="evaluate latent/action/progress prediction only; do not activate RGB/video renderer or LPIPS",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    model_cfg = cfg.get("model", {})
    state_cfg = model_cfg.get("state") or {}
    state_k = int(state_cfg.get("k", cfg.get("data", {}).get("k", 0)) or 0)
    policy_horizon = int(model_cfg.get("policy_horizon", 0) or 0)
    if bool(model_cfg.get("enable_action_policy", False)) and policy_horizon > 0 and state_k > 0 and policy_horizon != state_k:
        raise RuntimeError(
            "run_eval.py evaluates the world-core/K8 prediction heads and is not a valid gate for "
            f"K{policy_horizon} action-policy checkpoints. Use scripts/action_policy_direct_gate.py "
            f"--require-horizon {policy_horizon} or a benchmark rollout policy path instead."
        )
    if bool(model_cfg.get("policy_enable_flow_head", False)) and bool(model_cfg.get("policy_flow_use_as_policy", False)):
        raise RuntimeError(
            "run_eval.py evaluates the world-core/K8 prediction heads and is not a valid gate for "
            "flow action-policy checkpoints. Use scripts/action_policy_direct_gate.py "
            "or a benchmark rollout policy path instead."
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    rgb_metrics = bool(cfg["model"].get("enable_pixel", True)) and not args.skip_rgb_metrics

    records = read_manifest(cfg["data"]["manifest"])
    val = build_dataset_for_split(records, cfg, split="val")
    print(f"val windows: {len(val)}")

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    val_total = sd.get("val_total")
    val_total_text = f"{val_total:.4f}" if isinstance(val_total, (float, int)) else str(val_total)
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={val_total_text}")

    lp = None
    if rgb_metrics:
        import lpips

        lp = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lp.parameters():
            p.requires_grad = False

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["train"]["batch_size_per_gpu"])
    loader = DataLoader(val, batch_size=batch_size,
                        shuffle=False, num_workers=cfg["train"]["num_workers"],
                        pin_memory=True)

    by_ds: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    cnt: dict[str, int] = defaultdict(int)
    grip_transition_correct: dict[str, float] = defaultdict(float)
    grip_transition_count: dict[str, float] = defaultdict(float)

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        s_tgt = batch["s_tgt"].to(device, non_blocking=True)
        depth_tgt = batch["depth_tgt"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        rgb_tgt = None
        if rgb_metrics:
            rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            kwargs = dict(action_cond=action_cond, pixel=rgb_metrics, bridging=False)
            if cfg["model"].get("enable_context_pixel", False):
                kwargs["context_rgb"] = context_rgb
            out = model(s, c, **kwargs)
        pred_s = out["pred_tokens"]
        L_state_mse = F.mse_loss(pred_s.float(), s_tgt.float(), reduction="none").mean(dim=(1, 2, 3))
        cos = F.cosine_similarity(pred_s.float().flatten(-2),
                                  s_tgt.float().flatten(-2), dim=-1).mean(dim=-1)
        depth_n = _normalize_depth(out["depth"].float())
        depth_tn = _normalize_depth(depth_tgt.float())
        if depth_n.shape[-2:] != depth_tn.shape[-2:]:
            bsz_d, horizon_d = depth_n.shape[:2]
            depth_n = F.interpolate(
                depth_n.reshape(bsz_d * horizon_d, 1, *depth_n.shape[-2:]),
                size=depth_tn.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(bsz_d, horizon_d, *depth_tn.shape[-2:])
        L_depth = (depth_n - depth_tn).abs().mean(dim=(1, 2, 3))
        pose_err = (out["pose"].float() - action_tgt[..., :6]).pow(2)
        L_pose = pose_err.mean(dim=(1, 2))
        L_pose_first = pose_err[:, 0].mean(dim=1)
        grip_tgt = (action_tgt[..., 6] > 0.5).float()
        grip_pred = (out["gripper_logit"].float().sigmoid() > 0.5).float()
        grip_step_acc = (grip_pred == grip_tgt).float()
        grip_acc = grip_step_acc.mean(dim=-1)
        grip_transition = (grip_tgt[:, 1:] != grip_tgt[:, :-1])
        grip_transition_n = grip_transition.float().sum(dim=-1)
        grip_transition_ok = (grip_step_acc[:, 1:] * grip_transition.float()).sum(dim=-1)
        grip_transition_rate = grip_transition_n / max(1, grip_tgt.shape[1] - 1)
        if "progress" in out and "progress_tgt" in batch:
            progress_tgt = batch["progress_tgt"].to(device, non_blocking=True)
            L_progress = (out["progress"].float().sigmoid() - progress_tgt.float()).abs().mean(dim=1)
        else:
            L_progress = None
        if "proposer_pose_norm" in out:
            prop_pose = out["proposer_pose_norm"].float()
            prop_pose_tgt = action_tgt_norm.float()
            prop_pose_err = (prop_pose - prop_pose_tgt[:, None]).abs().mean(dim=(2, 3))
            prop_best = prop_pose_err.min(dim=1).values
            prop_anchor = prop_pose_err[:, 0]
            prop_grip_logits = out["proposer_gripper_logit"].float()
            prop_grip_tgt = grip_tgt[:, None].expand_as(prop_grip_logits)
            prop_grip_err = F.binary_cross_entropy_with_logits(
                prop_grip_logits,
                prop_grip_tgt,
                reduction="none",
            ).mean(dim=2).min(dim=1).values
        else:
            prop_best = prop_anchor = prop_grip_err = None
        if rgb_metrics:
            if "rgb" not in out:
                raise RuntimeError("RGB metrics requested, but model output has no 'rgb'. Use --skip_rgb_metrics for action/prediction-only eval.")
            if lp is None or rgb_tgt is None:
                raise RuntimeError("internal RGB metric setup failed")
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
            by_ds[d]["L_pose_first_mse"] += float(L_pose_first[i])
            by_ds[d]["grip_acc"] += float(grip_acc[i])
            by_ds[d]["grip_transition_rate"] += float(grip_transition_rate[i])
            grip_transition_correct[d] += float(grip_transition_ok[i])
            grip_transition_count[d] += float(grip_transition_n[i])
            if L_progress is not None:
                by_ds[d]["progress_abs_err"] += float(L_progress[i])
            if prop_best is not None:
                by_ds[d]["proposer_best_pose_L1"] += float(prop_best[i])
                by_ds[d]["proposer_anchor_pose_L1"] += float(prop_anchor[i])
                by_ds[d]["proposer_best_grip_bce"] += float(prop_grip_err[i])
            if rgb_metrics:
                by_ds[d]["L_rgb_L1"] += float(L_rgb_l1[i])
                by_ds[d]["L_rgb_lpips"] += float(lpv[i])
                by_ds[d]["L_rgb_motion_L1"] += float(L_rgb_motion_l1[i])
                by_ds[d]["motion_frac"] += float(motion_frac[i])
            cnt[d] += 1
        by_ds["ALL"]["L_state_mse"] += float(L_state_mse.sum())
        by_ds["ALL"]["cos_sim"] += float(cos.sum())
        by_ds["ALL"]["L_depth_rel_L1"] += float(L_depth.sum())
        by_ds["ALL"]["L_pose_mse"] += float(L_pose.sum())
        by_ds["ALL"]["L_pose_first_mse"] += float(L_pose_first.sum())
        by_ds["ALL"]["grip_acc"] += float(grip_acc.sum())
        by_ds["ALL"]["grip_transition_rate"] += float(grip_transition_rate.sum())
        grip_transition_correct["ALL"] += float(grip_transition_ok.sum())
        grip_transition_count["ALL"] += float(grip_transition_n.sum())
        if L_progress is not None:
            by_ds["ALL"]["progress_abs_err"] += float(L_progress.sum())
        if prop_best is not None:
            by_ds["ALL"]["proposer_best_pose_L1"] += float(prop_best.sum())
            by_ds["ALL"]["proposer_anchor_pose_L1"] += float(prop_anchor.sum())
            by_ds["ALL"]["proposer_best_grip_bce"] += float(prop_grip_err.sum())
        if rgb_metrics:
            by_ds["ALL"]["L_rgb_L1"] += float(L_rgb_l1.sum())
            by_ds["ALL"]["L_rgb_lpips"] += float(lpv.sum())
            by_ds["ALL"]["L_rgb_motion_L1"] += float(L_rgb_motion_l1.sum())
            by_ds["ALL"]["motion_frac"] += float(motion_frac.sum())
        cnt["ALL"] += s.shape[0]
        if (bi + 1) % 25 == 0:
            msg = (f"[{bi+1}/{len(loader)}] L_state {by_ds['ALL']['L_state_mse']/cnt['ALL']:.4f} "
                   f"L_depth {by_ds['ALL']['L_depth_rel_L1']/cnt['ALL']:.4f} "
                   f"L_pose {by_ds['ALL']['L_pose_mse']/cnt['ALL']:.4f} "
                   f"grip {by_ds['ALL']['grip_acc']/cnt['ALL']:.4f}")
            if rgb_metrics:
                msg += (f" rgb_L1 {by_ds['ALL']['L_rgb_L1']/cnt['ALL']:.4f} "
                        f"lpips {by_ds['ALL']['L_rgb_lpips']/cnt['ALL']:.4f} "
                        f"motion_L1 {by_ds['ALL']['L_rgb_motion_L1']/cnt['ALL']:.4f}")
            print(msg)

    report = {
        "mode": {
            "rgb_metrics": rgb_metrics,
            "video_generation_active": rgb_metrics,
        },
        "counts": dict(cnt),
        "metrics": {},
    }
    report["grip_transition_counts"] = dict(grip_transition_count)
    for d, mvals in by_ds.items():
        n = max(1, cnt[d])
        report["metrics"][d] = {k: v / n for k, v in mvals.items()}
        report["metrics"][d]["grip_transition_acc"] = (
            grip_transition_correct[d] / max(1.0, grip_transition_count[d])
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
