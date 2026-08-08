#!/usr/bin/env python3
"""Balanced WM3D-v7 Stage-0 checkpoint sweep and Base-vs-V7 demos.

This evaluator deliberately reuses the formal training loader/model path.  The
legacy eval entry points only understand a single OXE manifest and therefore
cannot evaluate the V7 OXE + compact RoboCasa mixture faithfully.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset, default_collate

from wm3d_v3.eval.action_sensitivity import (
    compute_counterfactual_metrics,
    make_action_counterfactuals,
    motion_target_from_rgb,
)
from wm3d_v3.losses import LossWeights, _normalize_depth, compute_losses
from wm3d_v3.training.train import (
    _forward_joint_model,
    action_policy_kwargs_from_targets,
    batch_to_device,
    build_gripper_event_partitions,
    build_datasets,
    build_model,
    compute_direct_policy_loss,
    compute_native_no_teacher_action_loss,
    decode_codec_targets,
    load_action_stats_if_available,
    load_compatible_state_dict,
    load_train_config,
    multiview_kwargs_from_targets,
    normalize_action_grip_contract,
    resolve_action_training_weights,
    targets_with_close01_grip,
    validate_stage0_native_warm_start_load,
)


def _configured_grip_contract(cfg: Mapping[str, Any]) -> str:
    return normalize_action_grip_contract(
        cfg.get("train", {}).get("action_grip_contract", "close01")
    )


def _batch_to_device_for_eval(
    batch: dict[str, Any],
    device: torch.device,
    horizon: int,
    cfg: Mapping[str, Any],
) -> tuple:
    """Use the exact factual gripper contract used by formal training."""

    return batch_to_device(
        batch,
        device,
        horizon,
        action_grip_contract=_configured_grip_contract(cfg),
    )


def _action_counterfactuals_from_real_condition(
    real: torch.Tensor,
    variants: tuple[str, ...],
    *,
    generator: torch.Generator,
    grip_contract: str,
) -> dict[str, torch.Tensor]:
    """Perturb the actual action tokens seen by the trained dynamics model."""

    out = {"real": real}
    for variant in variants:
        cur = real.clone()
        if variant == "zero":
            cur.zero_()
        elif variant == "shuffled":
            if cur.shape[0] > 1:
                perm = torch.randperm(cur.shape[0], generator=generator, device=cur.device)
                if torch.equal(perm, torch.arange(cur.shape[0], device=cur.device)):
                    perm = torch.roll(perm, shifts=1)
                cur = cur[perm]
            elif cur.shape[1] > 1:
                perm = torch.randperm(cur.shape[1], generator=generator, device=cur.device)
                if torch.equal(perm, torch.arange(cur.shape[1], device=cur.device)):
                    perm = torch.roll(perm, shifts=1)
                cur = cur[:, perm]
        elif variant == "sign_flip":
            cur[..., :6].neg_()
        elif variant == "scaled":
            cur[..., :6].mul_(2.0)
        elif variant == "grip_toggle":
            if grip_contract == "signed_close":
                cur[..., 6:7].neg_()
            else:
                cur[..., 6:7] = 1.0 - cur[..., 6:7]
        else:
            raise ValueError(f"unknown action variant: {variant}")
        out[variant] = cur
    return out


def psnr_video(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    mse = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3, 4))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(float(eps)))


def ssim_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    bsz, horizon, channels = pred.shape[:3]
    x = pred.float().flatten(0, 1)
    y = target.float().flatten(0, 1)
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, window_size, window_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    sigma_x = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return score.mean(dim=(1, 2, 3)).reshape(bsz, horizon).mean(dim=1).clamp(-1.0, 1.0)


def motion_region_rgb_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    threshold: float = 0.03,
) -> torch.Tensor:
    motion = (target.float() - context_rgb.float().unsqueeze(1)).abs().mean(dim=2, keepdim=True)
    mask = (motion > float(threshold)).float()
    denom = (mask.sum(dim=(1, 2, 3, 4)) * target.shape[2]).clamp_min(1.0)
    return ((pred.float() - target.float()).abs() * mask).sum(dim=(1, 2, 3, 4)) / denom


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    raise TypeError(type(value).__name__)


class MetricMoments:
    def __init__(self) -> None:
        self.sum: dict[str, float] = defaultdict(float)
        self.sumsq: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)

    def update(self, values: Mapping[str, torch.Tensor | float], *, repeat: int | None = None) -> None:
        for key, raw in values.items():
            if isinstance(raw, torch.Tensor):
                array = raw.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            else:
                n = max(1, int(repeat or 1))
                array = np.full(n, float(raw), dtype=np.float64)
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                continue
            self.sum[key] += float(finite.sum())
            self.sumsq[key] += float(np.square(finite).sum())
            self.count[key] += int(finite.size)

    def report(self) -> dict[str, dict[str, float | int]]:
        report: dict[str, dict[str, float | int]] = {}
        for key in sorted(self.sum):
            n = max(1, self.count[key])
            mean = self.sum[key] / n
            var = max(0.0, self.sumsq[key] / n - mean * mean)
            report[key] = {
                "mean": mean,
                "sem": math.sqrt(var / n),
                "count": self.count[key],
            }
        return report


class HorizonMoments:
    def __init__(self, horizon: int) -> None:
        self.steps = [MetricMoments() for _ in range(int(horizon))]

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        for step, accumulator in enumerate(self.steps):
            accumulator.update({key: value[:, step] for key, value in values.items()})

    def report(self) -> list[dict[str, Any]]:
        return [
            {"horizon_step": index + 1, "metrics": accumulator.report()}
            for index, accumulator in enumerate(self.steps)
        ]


def _sample_indices(length: int, count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(int(length), generator=generator)[: min(int(count), int(length))].tolist()


def _grip_transition_indices(dataset: Any, count: int, seed: int) -> list[int]:
    selected: list[int] = []
    for index in _sample_indices(len(dataset), len(dataset), seed):
        sample = dataset[index]
        grip = sample["action_tgt"][:, 6].float() > 0.5
        previous = sample.get("action_prev_grip")
        if previous is not None:
            grip = torch.cat([previous.float().reshape(-1) > 0.5, grip])
        if grip.numel() > 1 and bool((grip[1:] != grip[:-1]).any()):
            selected.append(index)
            if len(selected) >= int(count):
                break
    return selected


def _source_datasets(cfg: dict) -> tuple[dict[str, Any], dict[str, int]]:
    _train, val = build_datasets(cfg)
    if not hasattr(val, "source_names") or not hasattr(val, "datasets"):
        raise RuntimeError("expected formal V7 mixed validation dataset")
    sources = {str(name): dataset for name, dataset in zip(val.source_names, val.datasets)}
    lengths = {name: len(dataset) for name, dataset in sources.items()}
    return sources, lengths


def _load_model(
    cfg: dict,
    ckpt: Path,
    device: torch.device,
    *,
    base: bool,
    allow_future_value_extra: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    # The fuser output projection is zero-initialized, so the V6 warm-start Base
    # is an exact anchor-only identity even though its internal attention weights
    # are newly constructed here.  The token codec basis is loaded from the
    # immutable config-bound checkpoint during model construction.
    torch.manual_seed(0)
    model = build_model(cfg).to(device).eval()
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    if base:
        result = load_compatible_state_dict(model, payload["model"], strict=False)
        validate_stage0_native_warm_start_load(result)
        load_report = {
            "mode": "v6_native_base_reconstruction",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "skipped_keys": list(result.skipped_keys),
            "expanded_keys": list(result.expanded_keys),
            "fuser_output_projection_abs_max": float(
                model.multiview_fuser.output_projection.weight.detach().abs().max().cpu()
            ),
        }
        if load_report["fuser_output_projection_abs_max"] != 0.0:
            raise RuntimeError("Base fuser is not an exact identity")
    elif allow_future_value_extra:
        result = load_compatible_state_dict(model, payload["model"], strict=False)
        bad_skipped = [
            key
            for key in result.skipped_keys
            if not key.startswith("future_value_head.")
        ]
        if (
            result.missing_keys
            or result.unexpected_keys
            or result.expanded_keys
            or bad_skipped
        ):
            raise RuntimeError(
                "S1 checkpoint is not S0-world compatible: "
                f"missing={list(result.missing_keys)[:8]} "
                f"unexpected={list(result.unexpected_keys)[:8]} "
                f"expanded={list(result.expanded_keys)[:8]} "
                f"bad_skipped={bad_skipped[:8]}"
            )
        load_report = {
            "mode": "strict_v7_world_with_future_value_ignored",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
            "skipped_keys": list(result.skipped_keys),
            "expanded_keys": list(result.expanded_keys),
        }
    else:
        result = model.load_state_dict(payload["model"], strict=True)
        load_report = {
            "mode": "strict_v7_checkpoint",
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": list(result.unexpected_keys),
        }
    load_action_stats_if_available(model, cfg, 1, device)
    load_report.update(
        {
            "path": str(ckpt),
            "step": payload.get("step"),
            "epoch": payload.get("epoch"),
            "stored_val_total": payload.get("val_total"),
        }
    )
    return model, load_report


def _resize_depth(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    bsz, horizon = pred.shape[:2]
    return F.interpolate(
        pred.float().reshape(bsz * horizon, 1, *pred.shape[-2:]),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).reshape(bsz, horizon, *target.shape[-2:])


def _motion_metrics(out: Mapping[str, torch.Tensor], rgb_tgt: torch.Tensor, context: torch.Tensor) -> dict[str, torch.Tensor]:
    target = motion_target_from_rgb(rgb_tgt, context)
    if "motion_logit" in out:
        logits = out["motion_logit"].float()
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target.flatten(0, 1), size=logits.shape[-2:], mode="nearest").reshape_as(logits)
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3, 4))
    elif "motion_hint" in out:
        prob = out["motion_hint"].float().clamp(1e-6, 1.0 - 1e-6)
        if prob.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target.flatten(0, 1), size=prob.shape[-2:], mode="nearest").reshape_as(prob)
        bce = F.binary_cross_entropy(prob, target, reduction="none").mean(dim=(1, 2, 3, 4))
    else:
        return {}
    reduce_dims = (1, 2, 3, 4)
    intersection = (prob * target).sum(dim=reduce_dims)
    dice = (2.0 * intersection + 1e-6) / (prob.sum(dim=reduce_dims) + target.sum(dim=reduce_dims) + 1e-6)
    binary = prob > 0.5
    truth = target > 0.5
    inter_hard = (binary & truth).float().sum(dim=reduce_dims)
    union_hard = (binary | truth).float().sum(dim=reduce_dims)
    return {
        "motion_bce": bce,
        "motion_soft_dice": dice,
        "motion_iou_at_0p5": (inter_hard + 1e-6) / (union_hard + 1e-6),
        "motion_pred_fraction_at_0p5": binary.float().mean(dim=reduce_dims),
        "motion_gt_fraction": truth.float().mean(dim=reduce_dims),
    }


def _common_metrics(out: Mapping[str, torch.Tensor], tgt: Mapping[str, torch.Tensor], context: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_tokens = out["pred_tokens"].float()
    target_tokens = tgt["s_tgt"].float()
    token_mse = (pred_tokens - target_tokens).pow(2).mean(dim=(1, 2, 3))
    token_cos = F.cosine_similarity(
        pred_tokens.flatten(2), target_tokens.flatten(2), dim=-1
    ).mean(dim=1)

    depth_target = tgt["depth_tgt"].float()
    depth_pred = _resize_depth(out["depth"].float(), depth_target)
    depth_pred_n = _normalize_depth(depth_pred)
    depth_target_n = _normalize_depth(depth_target)
    depth_l1 = (depth_pred_n - depth_target_n).abs().mean(dim=(1, 2, 3))
    if depth_pred.shape[1] > 1:
        depth_temporal = (
            (depth_pred_n[:, 1:] - depth_pred_n[:, :-1])
            - (depth_target_n[:, 1:] - depth_target_n[:, :-1])
        ).abs().mean(dim=(1, 2, 3))
    else:
        depth_temporal = torch.zeros_like(depth_l1)

    rgb_target = tgt["rgb_tgt_p"].float()
    rgb_pred = out["rgb"].float().clamp(0.0, 1.0)
    last_frame = context.float().unsqueeze(1).expand_as(rgb_target)
    rgb_l1 = (rgb_pred - rgb_target).abs().mean(dim=(1, 2, 3, 4))
    last_l1 = (last_frame - rgb_target).abs().mean(dim=(1, 2, 3, 4))
    endpoint_l1 = (rgb_pred[:, -1] - rgb_target[:, -1]).abs().mean(dim=(1, 2, 3))
    if rgb_pred.shape[1] > 1:
        rgb_temporal = (
            (rgb_pred[:, 1:] - rgb_pred[:, :-1])
            - (rgb_target[:, 1:] - rgb_target[:, :-1])
        ).abs().mean(dim=(1, 2, 3, 4))
    else:
        rgb_temporal = torch.zeros_like(rgb_l1)
    metrics = {
        "token_mse": token_mse,
        "token_cos": token_cos,
        "depth_relative_l1": depth_l1,
        "depth_temporal_delta_l1": depth_temporal,
        "rgb_l1": rgb_l1,
        "rgb_psnr": psnr_video(rgb_pred, rgb_target),
        "rgb_ssim": ssim_video(rgb_pred, rgb_target),
        "rgb_motion_region_l1": motion_region_rgb_l1(rgb_pred, rgb_target, context),
        "rgb_endpoint_l1": endpoint_l1,
        "rgb_temporal_delta_l1": rgb_temporal,
        "last_frame_rgb_l1": last_l1,
        "last_frame_rgb_psnr": psnr_video(last_frame, rgb_target),
        "last_frame_rgb_ssim": ssim_video(last_frame, rgb_target),
        "rgb_l1_gain_vs_last_frame": last_l1 - rgb_l1,
        "rgb_psnr_gain_vs_last_frame": psnr_video(rgb_pred, rgb_target) - psnr_video(last_frame, rgb_target),
    }
    metrics.update(_motion_metrics(out, rgb_target, context))
    if "pose" in out:
        action = tgt["action_tgt"]
        horizon = min(out["pose"].shape[1], action.shape[1])
        metrics["aux_pose_mse"] = (
            out["pose"][:, :horizon].float() - action[:, :horizon, :6].float()
        ).pow(2).mean(dim=(1, 2))
        grip_pred = torch.sigmoid(out["gripper_logit"][:, :horizon].float()) > 0.5
        grip_tgt = action[:, :horizon, 6] > 0.5
        metrics["aux_grip_acc"] = (grip_pred == grip_tgt).float().mean(dim=1)
    return metrics


def _direct_policy_output_keys(train_cfg: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve the serving direct head without falling back to flow outputs."""

    head = str(train_cfg.get("direct_policy_head", "policy")).strip().lower()
    if head in {"policy", "direct", "full"}:
        return "policy_pose_norm", "policy_gripper_logit"
    if head in {"base", "base_policy"}:
        return "base_policy_pose_norm", "base_policy_gripper_logit"
    if head in {"prior", "prior_policy", "oxe_prior"}:
        return "prior_policy_pose_norm", "prior_policy_gripper_logit"
    raise ValueError(f"unsupported direct_policy_head for evaluation: {head!r}")


def _serving_grip_probability(
    out: Mapping[str, torch.Tensor],
    train_cfg: Mapping[str, Any],
) -> torch.Tensor:
    """Return the probability/state owned by the configured serving contract."""

    owner = str(train_cfg.get("direct_policy_grip_owner", "absolute")).strip().lower()
    if owner == "delta_composed":
        if "policy_gripper_composed" not in out:
            raise RuntimeError(
                "delta_composed serving requires policy_gripper_composed output"
            )
        return out["policy_gripper_composed"].float().clamp(0.0, 1.0)
    if owner not in {"auto", "absolute"}:
        raise ValueError(f"unsupported direct_policy_grip_owner={owner!r}")
    _, grip_key = _direct_policy_output_keys(train_cfg)
    return torch.sigmoid(out[grip_key].float())


def _serving_action_metrics(
    out: Mapping[str, torch.Tensor],
    tgt: Mapping[str, torch.Tensor],
    train_cfg: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Human-readable direct-policy metrics at the actual 0.5 serving threshold."""

    pose_key, _ = _direct_policy_output_keys(train_cfg)
    pose_pred = out[pose_key].float()
    grip_prob = _serving_grip_probability(out, train_cfg)
    pose_tgt_norm = tgt["action_tgt_norm"].float()
    action_tgt = tgt["action_tgt"].float()
    horizon = min(
        int(pose_pred.shape[1]),
        int(grip_prob.shape[1]),
        int(pose_tgt_norm.shape[1]),
        int(action_tgt.shape[1]),
    )
    pose_pred = pose_pred[:, :horizon]
    pose_tgt_norm = pose_tgt_norm[:, :horizon]
    grip_prob = grip_prob[:, :horizon]
    grip_tgt = action_tgt[:, :horizon, 6] > 0.5
    grip_pred = grip_prob >= 0.5

    pose_mean = tgt["action_pose_mean"].float()
    pose_std = tgt["action_pose_std"].float()
    while pose_mean.ndim < pose_pred.ndim:
        pose_mean = pose_mean.unsqueeze(1)
        pose_std = pose_std.unsqueeze(1)
    pose_pred_phys = pose_pred * pose_std + pose_mean
    pose_tgt_phys = action_tgt[:, :horizon, :6]
    pose_abs_phys = (pose_pred_phys - pose_tgt_phys).abs()

    eps = 1.0e-6
    positive = grip_tgt.float()
    negative = (~grip_tgt).float()
    pos_recall = ((grip_pred & grip_tgt).float().sum(dim=1) / positive.sum(dim=1).clamp_min(1.0))
    neg_recall = (((~grip_pred) & (~grip_tgt)).float().sum(dim=1) / negative.sum(dim=1).clamp_min(1.0))
    pos_valid = positive.sum(dim=1) > 0
    neg_valid = negative.sum(dim=1) > 0
    balanced_parts = []
    for recall, valid in ((pos_recall, pos_valid), (neg_recall, neg_valid)):
        balanced_parts.append(torch.where(valid, recall, torch.nan))
    stacked = torch.stack(balanced_parts, dim=1)
    balanced = torch.nanmean(stacked, dim=1)
    partitions = build_gripper_event_partitions(
        grip_tgt.float(), tgt.get("action_prev_grip")
    )
    transition = partitions["boundary_up"] | partitions["boundary_down"] | partitions["inclip_up"] | partitions["inclip_down"]
    transition_correct = (grip_pred == grip_tgt) & transition
    transition_recall = transition_correct.float().sum(dim=1) / transition.float().sum(dim=1).clamp_min(1.0)
    transition_recall = torch.where(
        transition.any(dim=1), transition_recall, torch.nan
    )

    return {
        "serving_pose_norm_l1": (pose_pred - pose_tgt_norm).abs().mean(dim=(1, 2)),
        "serving_first_pose_norm_l1": (pose_pred[:, 0] - pose_tgt_norm[:, 0]).abs().mean(dim=1),
        "serving_endpoint_pose_norm_l1": (
            pose_pred.sum(dim=1) - pose_tgt_norm.sum(dim=1)
        ).abs().mean(dim=1),
        "serving_translation_mae_mm": pose_abs_phys[..., :3].mean(dim=(1, 2)) * 1000.0,
        "serving_rotation_mae_deg": pose_abs_phys[..., 3:6].mean(dim=(1, 2)) * (180.0 / math.pi),
        "serving_grip_bce": F.binary_cross_entropy(
            grip_prob.clamp(1.0e-6, 1.0 - 1.0e-6),
            grip_tgt.float(),
            reduction="none",
        ).mean(dim=1),
        "serving_grip_accuracy": (grip_pred == grip_tgt).float().mean(dim=1),
        "serving_grip_balanced_accuracy": balanced,
        "serving_grip_positive_recall": torch.where(pos_valid, pos_recall, torch.nan),
        "serving_grip_negative_recall": torch.where(neg_valid, neg_recall, torch.nan),
        "serving_grip_transition_recall": transition_recall,
        "serving_grip_true_positive_rate": positive.mean(dim=1),
        "serving_grip_predicted_positive_rate": grip_pred.float().mean(dim=1),
        "serving_grip_probability_mean": grip_prob.mean(dim=1),
        "serving_grip_probability_prior_gap": (
            grip_prob.mean(dim=1) - positive.mean(dim=1)
        ).abs().clamp_min(eps),
    }


@torch.no_grad()
def run_quality(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.cfg)
    grip_contract = _configured_grip_contract(cfg)
    train_cfg = cfg.get("train") or {}
    action_policy_enabled = bool((cfg.get("model") or {}).get("enable_action_policy", False))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sources, source_lengths = _source_datasets(cfg)
    model, load_report = _load_model(
        cfg,
        args.ckpt,
        device,
        base=args.base,
        allow_future_value_extra=args.allow_future_value_extra,
    )
    import lpips

    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad = False
    weights = LossWeights(**cfg["loss"])
    _, _, native_action_weights = resolve_action_training_weights(
        weights, train_cfg, strict=False
    )
    accumulators = {name: MetricMoments() for name in [*sources, "ALL"]}
    selected: dict[str, list[int]] = {}
    causal_runtime_audit: dict[str, dict[str, float]] = {}
    for source_id, (source_name, dataset) in enumerate(sources.items()):
        indices = _sample_indices(len(dataset), args.samples_per_source, args.seed + source_id * 1009)
        selected[source_name] = indices
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for batch in loader:
            s, c, action_cond, context_rgb, tgt = _batch_to_device_for_eval(
                batch, device, cfg["data"]["k"], cfg
            )
            decode_codec_targets(model, tgt)
            loss_tgt = targets_with_close01_grip(tgt, grip_contract)
            policy_kwargs = action_policy_kwargs_from_targets(loss_tgt)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _forward_joint_model(
                    model,
                    s,
                    c,
                    action_cond=action_cond,
                    context_rgb=context_rgb,
                    pixel=True,
                    bridging=False,
                    policy_kwargs=policy_kwargs,
                    multiview_kwargs=multiview_kwargs_from_targets(tgt),
                    native_action_no_teacher=action_policy_enabled,
                )
                losses = compute_losses(out, loss_tgt, weights, lpips_model)
                direct_losses = (
                    compute_direct_policy_loss(
                        out,
                        loss_tgt["action_tgt"],
                        loss_tgt["action_tgt_norm"],
                        train_cfg,
                        action_prev_grip=loss_tgt.get("action_prev_grip"),
                        step=int(load_report.get("step") or 0),
                    )
                    if action_policy_enabled
                    else {}
                )
                native_losses = (
                    compute_native_no_teacher_action_loss(
                        out,
                        loss_tgt,
                        native_action_weights,
                        train_cfg=train_cfg,
                    )
                    if action_policy_enabled
                    else {}
                )
            batch_n = int(s.shape[0])
            scalar_losses = {f"objective/{key}": float(value.detach().float()) for key, value in losses.items()}
            common = _common_metrics(out, tgt, context_rgb)
            if action_policy_enabled:
                common.update(_serving_action_metrics(out, loss_tgt, train_cfg))
                scalar_losses.update(
                    {
                        f"direct/{key}": float(value.detach().float())
                        for key, value in direct_losses.items()
                    }
                )
                scalar_losses.update(
                    {
                        f"native_no_teacher/{key}": float(value.detach().float())
                        for key, value in native_losses.items()
                    }
                )
            for name in (source_name, "ALL"):
                accumulators[name].update(scalar_losses, repeat=batch_n)
                accumulators[name].update(common)
            if action_policy_enabled and source_name not in causal_runtime_audit:
                # The serving policy must read the action-free native core. A
                # changed teacher action may alter the factual world rollout,
                # but it must not alter policy context or policy output.
                counterfactual_action = action_cond.clone()
                counterfactual_action[..., :6].neg_()
                counterfactual_action[..., 6:7] = 1.0 - counterfactual_action[..., 6:7]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    counterfactual_out = _forward_joint_model(
                        model,
                        s,
                        c,
                        action_cond=counterfactual_action,
                        context_rgb=context_rgb,
                        pixel=False,
                        bridging=False,
                        policy_kwargs=policy_kwargs,
                        multiview_kwargs=multiview_kwargs_from_targets(tgt),
                        native_action_no_teacher=True,
                    )
                pose_key, _ = _direct_policy_output_keys(train_cfg)
                grip_value = _serving_grip_probability(out, train_cfg)
                counterfactual_grip_value = _serving_grip_probability(
                    counterfactual_out, train_cfg
                )
                causal_runtime_audit[source_name] = {
                    "policy_pose_teacher_action_max_abs": float(
                        (out[pose_key] - counterfactual_out[pose_key]).abs().max().float().cpu()
                    ),
                    "policy_grip_teacher_action_max_abs": float(
                        (grip_value - counterfactual_grip_value)
                        .abs().max().float().cpu()
                    ),
                    "policy_context_teacher_action_max_abs": float(
                        (
                            out["policy_context_tokens"]
                            - counterfactual_out["policy_context_tokens"]
                        ).abs().max().float().cpu()
                    ),
                    "factual_world_teacher_action_mean_abs": float(
                        (out["pred_tokens"] - counterfactual_out["pred_tokens"])
                        .abs().mean().float().cpu()
                    ),
                }
                del counterfactual_out
            del s, c, action_cond, context_rgb, tgt, loss_tgt, out, losses, direct_losses, native_losses
    report = {
        "mode": "balanced_v7_stage0_quality",
        "config": str(args.cfg),
        "checkpoint": load_report,
        "sampling": {
            "seed": args.seed,
            "samples_per_source": args.samples_per_source,
            "source_lengths": source_lengths,
            "selected_local_indices": selected,
            "equal_source_weighting": True,
        },
        "causal_runtime_audit": causal_runtime_audit,
        "metrics": {name: acc.report() for name, acc in accumulators.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=_jsonable))
    print(json.dumps({"out": str(args.out), "ALL": {k: v["mean"] for k, v in report["metrics"]["ALL"].items()}}, indent=2))
    return report


@torch.no_grad()
def run_action(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.cfg)
    grip_contract = _configured_grip_contract(cfg)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sources, source_lengths = _source_datasets(cfg)
    model, load_report = _load_model(
        cfg,
        args.ckpt,
        device,
        base=args.base,
        allow_future_value_extra=args.allow_future_value_extra,
    )
    variants = tuple(args.variants)
    accumulators: dict[str, dict[str, MetricMoments]] = {
        name: {variant: MetricMoments() for variant in variants}
        for name in [*sources, "ALL"]
    }
    selected: dict[str, list[int]] = {}
    generator = torch.Generator(device=device).manual_seed(args.seed + 99991)
    for source_id, (source_name, dataset) in enumerate(sources.items()):
        if args.grip_transition_only:
            indices = _grip_transition_indices(
                dataset, args.samples_per_source, args.seed + source_id * 1009
            )
        else:
            indices = _sample_indices(
                len(dataset), args.samples_per_source, args.seed + source_id * 1009
            )
        if not indices:
            raise RuntimeError(f"no eligible action samples for source={source_name}")
        selected[source_name] = indices
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for batch in loader:
            s, c, action_cond, context_rgb, tgt = _batch_to_device_for_eval(
                batch, device, cfg["data"]["k"], cfg
            )
            decode_codec_targets(model, tgt)
            conds = _action_counterfactuals_from_real_condition(
                action_cond,
                variants,
                generator=generator,
                grip_contract=grip_contract,
            )
            kwargs = {
                "context_rgb": context_rgb,
                "pixel": True,
                "bridging": False,
                "multiview_kwargs": multiview_kwargs_from_targets(tgt),
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                real_out = _forward_joint_model(model, s, c, action_cond=conds["real"], **kwargs)
            targets = {
                "s_tgt": tgt["s_tgt"],
                "depth_tgt": tgt["depth_tgt"],
                "rgb_tgt": tgt["rgb_tgt_p"],
                "motion_tgt": motion_target_from_rgb(tgt["rgb_tgt_p"], context_rgb),
            }
            for variant in variants:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    variant_out = _forward_joint_model(model, s, c, action_cond=conds[variant], **kwargs)
                metrics = compute_counterfactual_metrics(real_out, variant_out, targets)
                for name in (source_name, "ALL"):
                    accumulators[name][variant].update(metrics)
                del variant_out, metrics
            del s, c, context_rgb, tgt, conds, real_out
    report = {
        "mode": "balanced_v7_stage0_action_counterfactual",
        "config": str(args.cfg),
        "checkpoint": load_report,
        "sampling": {
            "seed": args.seed,
            "samples_per_source": args.samples_per_source,
            "source_lengths": source_lengths,
            "selected_local_indices": selected,
            "equal_source_weighting": True,
            "grip_transition_only": bool(args.grip_transition_only),
        },
        "variants": list(variants),
        "metrics": {
            source: {variant: acc.report() for variant, acc in variant_map.items()}
            for source, variant_map in accumulators.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=_jsonable))
    compact = {
        variant: {
            key: values["mean"]
            for key, values in report["metrics"]["ALL"][variant].items()
            if key in {
                "pred_tokens_mse_gap",
                "pred_tokens_gt_mse_gap",
                "pred_tokens_gt_mse_acc",
                "rgb_l1_gap",
                "rgb_gt_l1_gap",
                "rgb_gt_l1_acc",
                "motion_region_rgb_gt_l1_gap",
                "motion_region_rgb_gt_l1_acc",
                "depth_l1_gap",
                "depth_gt_l1_gap",
                "depth_gt_l1_acc",
            }
        }
        for variant in variants
    }
    print(json.dumps({"out": str(args.out), "ALL": compact}, indent=2))
    return report


def _gt_motion_fraction(sample: Mapping[str, Any], threshold: float = 0.03) -> float:
    target = sample["rgb_tgt"].float()
    context = sample["rgb_in"][-1].float()
    return float(((target - context[None]).abs().mean(dim=-1) > threshold).float().mean())


def _high_motion_indices(dataset: Any, candidate_count: int, selected_count: int, seed: int) -> tuple[list[int], list[float]]:
    rows = []
    for index in _sample_indices(len(dataset), candidate_count, seed):
        rows.append((index, _gt_motion_fraction(dataset[index])))
    rows.sort(key=lambda row: (-row[1], row[0]))
    chosen = rows[: min(int(selected_count), len(rows))]
    return [row[0] for row in chosen], [row[1] for row in chosen]


def _masked_video_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[2] == 1 and values.shape[2] != 1:
        mask = mask.expand(-1, -1, values.shape[2], -1, -1)
    denominator = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)
    return (values * mask).sum(dim=(2, 3, 4)) / denominator


def _horizon_ghost_metrics(
    out: Mapping[str, torch.Tensor],
    tgt: Mapping[str, torch.Tensor],
    context: torch.Tensor,
    threshold: float = 0.03,
) -> dict[str, torch.Tensor]:
    pred = out["rgb"].float().clamp(0.0, 1.0)
    target = tgt["rgb_tgt_p"].float()
    context_seq = context[:, None].expand_as(target)
    previous = torch.cat([context[:, None], target[:, :-1]], dim=1)
    motion = (target - context_seq).abs().mean(dim=2, keepdim=True)
    motion_mask = (motion > threshold).float()
    step_motion = (target - previous).abs().mean(dim=2, keepdim=True)
    step_mask = (step_motion > threshold).float()

    current_error = (pred - target).abs()
    context_error = (pred - context_seq).abs()
    previous_error = (pred - previous).abs()
    context_ghost_rank = _masked_video_mean(
        F.relu(current_error - context_error), motion_mask
    )
    previous_ghost_rank = _masked_video_mean(
        F.relu(current_error - previous_error), step_mask
    )
    current_step_error = _masked_video_mean(current_error, step_mask)
    previous_step_error = _masked_video_mean(previous_error, step_mask)
    lag_preference = (previous_step_error < current_step_error).float()

    target_from_context = _masked_video_mean((target - context_seq).abs(), motion_mask)
    pred_from_context = _masked_video_mean((pred - context_seq).abs(), motion_mask)
    displacement_ratio = pred_from_context / target_from_context.clamp_min(1e-6)
    target_step = _masked_video_mean((target - previous).abs(), step_mask)
    pred_step = _masked_video_mean((pred - previous).abs(), step_mask)
    step_motion_ratio = pred_step / target_step.clamp_min(1e-6)

    rgb_l1 = current_error.mean(dim=(2, 3, 4))
    motion_l1 = _masked_video_mean(current_error, motion_mask)
    depth_target = tgt["depth_tgt"].float()
    depth_pred = _resize_depth(out["depth"].float(), depth_target)
    depth_l1 = (
        _normalize_depth(depth_pred) - _normalize_depth(depth_target)
    ).abs().mean(dim=(2, 3))

    metrics = {
        "rgb_l1": rgb_l1,
        "motion_rgb_l1": motion_l1,
        "depth_relative_l1": depth_l1,
        "context_ghost_rank": context_ghost_rank,
        "previous_ghost_rank": previous_ghost_rank,
        "lag_preference_rate": lag_preference,
        "motion_displacement_ratio": displacement_ratio,
        "step_motion_ratio": step_motion_ratio,
        "gt_motion_fraction": motion_mask.mean(dim=(2, 3, 4)),
        "gt_step_motion_fraction": step_mask.mean(dim=(2, 3, 4)),
    }
    if "rgb_blend" in out:
        blend = out["rgb_blend"].float()
        metrics["blend_on_motion"] = _masked_video_mean(blend, motion_mask)
        metrics["blend_on_step_motion"] = _masked_video_mean(blend, step_mask)
        metrics["blend_on_static"] = _masked_video_mean(blend, 1.0 - motion_mask)
    if "motion_hint" in out:
        hint = out["motion_hint"].float()
        predicted = hint > 0.5
        truth = motion_mask > 0.5
        metrics["motion_recall_at_0p5"] = (
            (predicted & truth).float().sum(dim=(2, 3, 4))
            / truth.float().sum(dim=(2, 3, 4)).clamp_min(1.0)
        )
        metrics["motion_precision_at_0p5"] = (
            (predicted & truth).float().sum(dim=(2, 3, 4))
            / predicted.float().sum(dim=(2, 3, 4)).clamp_min(1.0)
        )
    return metrics


@torch.no_grad()
def run_horizon(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.cfg)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sources, source_lengths = _source_datasets(cfg)
    model, load_report = _load_model(
        cfg,
        args.ckpt,
        device,
        base=args.base,
        allow_future_value_extra=args.allow_future_value_extra,
    )
    horizon = int(cfg["data"]["k"])
    accumulators = {
        name: HorizonMoments(horizon) for name in [*sources, "ALL"]
    }
    selection: dict[str, Any] = {}
    for source_id, (source_name, dataset) in enumerate(sources.items()):
        indices, motion_fractions = _high_motion_indices(
            dataset,
            args.candidates_per_source,
            args.samples_per_source,
            args.seed + source_id * 1009,
        )
        selection[source_name] = {
            "local_indices": indices,
            "gt_motion_fractions": motion_fractions,
        }
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for batch in loader:
            s, c, action_cond, context_rgb, tgt = _batch_to_device_for_eval(
                batch, device, horizon, cfg
            )
            decode_codec_targets(model, tgt)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _forward_joint_model(
                    model,
                    s,
                    c,
                    action_cond=action_cond,
                    context_rgb=context_rgb,
                    pixel=True,
                    bridging=False,
                    multiview_kwargs=multiview_kwargs_from_targets(tgt),
                )
            metrics = _horizon_ghost_metrics(out, tgt, context_rgb)
            accumulators[source_name].update(metrics)
            accumulators["ALL"].update(metrics)
            del s, c, action_cond, context_rgb, tgt, out, metrics
    report = {
        "mode": "v7_stage0_high_motion_horizon_ghost_diagnostic",
        "config": str(args.cfg),
        "checkpoint": load_report,
        "sampling": {
            "seed": args.seed,
            "candidate_count_per_source": args.candidates_per_source,
            "selected_count_per_source": args.samples_per_source,
            "source_lengths": source_lengths,
            "selection": selection,
            "selection_uses_model_outputs": False,
        },
        "per_horizon": {
            source: accumulator.report()
            for source, accumulator in accumulators.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=_jsonable))
    compact = []
    for row in report["per_horizon"]["ALL"]:
        compact.append(
            {
                "h": row["horizon_step"],
                **{
                    key: row["metrics"][key]["mean"]
                    for key in (
                        "rgb_l1",
                        "motion_rgb_l1",
                        "context_ghost_rank",
                        "previous_ghost_rank",
                        "lag_preference_rate",
                        "motion_displacement_ratio",
                        "blend_on_motion",
                        "motion_recall_at_0p5",
                    )
                    if key in row["metrics"]
                },
            }
        )
    print(json.dumps({"out": str(args.out), "ALL": compact}, indent=2))
    return report


def _candidate_metadata(source: str, local_index: int, sample: Mapping[str, Any]) -> dict[str, Any]:
    rgb_target = sample["rgb_tgt"].float()
    context = sample["rgb_in"][-1].float()
    motion = (rgb_target - context.unsqueeze(0)).abs().mean(dim=-1) > 0.03
    action_norm = sample["action_tgt_norm"][:, :6].float().norm(dim=-1)
    grip = sample["action_tgt"][:, 6].float() > 0.5
    previous = sample.get("action_prev_grip")
    if previous is not None:
        grip_full = torch.cat([previous.float().reshape(-1) > 0.5, grip])
    else:
        grip_full = grip
    return {
        "source": source,
        "local_index": int(local_index),
        "clip_id": str(sample["clip_id"]),
        "start": int(sample["start"]),
        "dataset": str(sample["dataset"]),
        "motion_fraction": float(motion.float().mean()),
        "action_energy": float(action_norm.mean()),
        "grip_transition": bool((grip_full[1:] != grip_full[:-1]).any()) if grip_full.numel() > 1 else False,
    }


def _select_demo_samples(sources: Mapping[str, Any], count: int, seed: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for source_id, (source, dataset) in enumerate(sources.items()):
        rows = []
        for index in _sample_indices(len(dataset), count, seed + source_id * 1009):
            sample = dataset[index]
            rows.append((_candidate_metadata(source, index, sample), sample))
        candidates[source] = rows

    picked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    quantiles = {"oxe_bridge": 0.75, "robocasa_atomic": 0.50, "robocasa_composite": 0.75}
    for source, rows in candidates.items():
        key = "action_energy" if source == "robocasa_composite" else "motion_fraction"
        values = np.asarray([meta[key] for meta, _sample in rows], dtype=np.float64)
        target = float(np.quantile(values, quantiles.get(source, 0.5)))
        chosen = min(rows, key=lambda pair: (abs(float(pair[0][key]) - target), pair[0]["local_index"]))
        chosen[0]["selection_rule"] = f"GT-only {key} q{quantiles.get(source, 0.5):.2f}"
        picked.append(chosen)

    used = {(meta["source"], meta["local_index"]) for meta, _sample in picked}
    extras = [
        pair
        for rows in candidates.values()
        for pair in rows
        if (pair[0]["source"], pair[0]["local_index"]) not in used and pair[0]["grip_transition"]
    ]
    if not extras:
        extras = [
            pair
            for rows in candidates.values()
            for pair in rows
            if (pair[0]["source"], pair[0]["local_index"]) not in used
        ]
    if extras:
        chosen = max(extras, key=lambda pair: (pair[0]["motion_fraction"], -pair[0]["local_index"]))
        chosen[0]["selection_rule"] = "GT-only grip transition, then highest motion"
        picked.append(chosen)
    return picked[:4]


@torch.no_grad()
def _predict_samples(
    cfg: dict,
    ckpt: Path,
    device: torch.device,
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    base: bool,
    allow_future_value_extra: bool = False,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    model, load_report = _load_model(
        cfg,
        ckpt,
        device,
        base=base,
        allow_future_value_extra=allow_future_value_extra,
    )
    predictions: list[dict[str, np.ndarray]] = []
    for _meta, sample in selected:
        batch = default_collate([sample])
        s, c, action_cond, context_rgb, tgt = _batch_to_device_for_eval(
            batch, device, cfg["data"]["k"], cfg
        )
        decode_codec_targets(model, tgt)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = _forward_joint_model(
                model,
                s,
                c,
                action_cond=action_cond,
                context_rgb=context_rgb,
                pixel=True,
                bridging=False,
                multiview_kwargs=multiview_kwargs_from_targets(tgt),
            )
        predictions.append(
            {
                "rgb": out["rgb"][0].float().clamp(0, 1).cpu().numpy(),
                "depth": out["depth"][0].float().cpu().numpy(),
                "tokens": out["pred_tokens"][0].float().cpu().numpy(),
            }
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions, load_report


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _rgb_uint8(frame_chw: np.ndarray) -> np.ndarray:
    return (np.clip(frame_chw.transpose(1, 2, 0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _depth_uint8(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import matplotlib

    norm = np.clip((depth - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    return (matplotlib.colormaps["viridis"](norm)[..., :3] * 255.0 + 0.5).astype(np.uint8)


def _labeled_cell(array: np.ndarray, label: str, *, color: tuple[int, int, int]) -> np.ndarray:
    image = Image.fromarray(array).resize((256, 256), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (256, 288), (15, 17, 22))
    canvas.paste(image, (0, 32))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 255, 31), fill=color)
    draw.text((8, 6), label, font=_font(17, bold=True), fill=(255, 255, 255))
    return np.asarray(canvas)


def _info_cell(meta: Mapping[str, Any], frame: int, action: np.ndarray) -> np.ndarray:
    canvas = Image.new("RGB", (256, 288), (20, 23, 30))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 255, 31), fill=(54, 63, 78))
    draw.text((8, 6), "Action / sample", font=_font(17, bold=True), fill="white")
    lines = [
        f"source: {meta['source']}",
        f"future: {frame + 1}/8",
        f"motion: {meta['motion_fraction']:.3f}",
        f"action E: {meta['action_energy']:.3f}",
        f"grip transition: {meta['grip_transition']}",
        "a_xyz: " + " ".join(f"{x:+.2f}" for x in action[:3]),
        "a_rot: " + " ".join(f"{x:+.2f}" for x in action[3:6]),
        f"grip: {action[6]:.0f}",
    ]
    y = 47
    for line in lines:
        draw.text((10, y), line, font=_font(13), fill=(226, 231, 239))
        y += 27
    return np.asarray(canvas)


def _demo_metrics(sample: Mapping[str, Any], base: Mapping[str, np.ndarray], final: Mapping[str, np.ndarray]) -> dict[str, Any]:
    target_rgb = sample["rgb_tgt"].float().permute(0, 3, 1, 2).numpy()
    context = sample["rgb_in"][-1].float().permute(2, 0, 1).numpy()
    target_depth = sample["depth_tgt"].float().numpy()

    def metrics(pred: Mapping[str, np.ndarray]) -> dict[str, float]:
        rgb = pred["rgb"]
        depth_tensor = torch.from_numpy(pred["depth"])[None]
        target_depth_tensor = torch.from_numpy(target_depth)[None]
        depth_tensor = _resize_depth(depth_tensor, target_depth_tensor)
        depth_error = (
            _normalize_depth(depth_tensor) - _normalize_depth(target_depth_tensor)
        ).abs().mean().item()
        motion = np.abs(target_rgb - context[None]).mean(axis=1, keepdims=True) > 0.03
        motion_denom = max(1.0, float(motion.sum() * 3))
        temporal = np.abs(
            (rgb[1:] - rgb[:-1]) - (target_rgb[1:] - target_rgb[:-1])
        ).mean()
        return {
            "rgb_l1": float(np.abs(rgb - target_rgb).mean()),
            "rgb_psnr": float(10.0 * np.log10(1.0 / max(1e-8, np.square(rgb - target_rgb).mean()))),
            "rgb_motion_region_l1": float((np.abs(rgb - target_rgb) * motion).sum() / motion_denom),
            "rgb_temporal_delta_l1": float(temporal),
            "depth_relative_l1": float(depth_error),
        }

    base_m = metrics(base)
    final_m = metrics(final)
    return {
        "base": base_m,
        "v7_30k": final_m,
        "relative_change_percent": {
            key: 100.0 * (final_m[key] - base_m[key]) / max(1e-12, abs(base_m[key]))
            for key in base_m
        },
    }


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_train_config(args.cfg)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sources, source_lengths = _source_datasets(cfg)
    selected = _select_demo_samples(sources, args.candidates_per_source, args.seed)
    base_pred, base_load = _predict_samples(cfg, args.base_ckpt, device, selected, base=True)
    final_pred, final_load = _predict_samples(
        cfg,
        args.ckpt,
        device,
        selected,
        base=False,
        allow_future_value_extra=args.allow_future_value_extra,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    demos = []
    contact_frames = []
    for demo_id, ((meta, sample), base, final) in enumerate(zip(selected, base_pred, final_pred)):
        target_rgb = sample["rgb_tgt"].float().permute(0, 3, 1, 2).numpy()
        context_rgb = sample["rgb_in"][-1].float().permute(2, 0, 1).numpy()
        target_depth = sample["depth_tgt"].float().numpy()
        base_depth = _resize_depth(torch.from_numpy(base["depth"])[None], torch.from_numpy(target_depth)[None])[0].numpy()
        final_depth = _resize_depth(torch.from_numpy(final["depth"])[None], torch.from_numpy(target_depth)[None])[0].numpy()
        depth_all = np.concatenate([target_depth.reshape(-1), base_depth.reshape(-1), final_depth.reshape(-1)])
        lo, hi = np.nanpercentile(depth_all, [2, 98])
        actions = sample["action_tgt"].float().numpy()
        frames = []
        for frame_id in range(target_rgb.shape[0]):
            top = np.concatenate(
                [
                    _labeled_cell(_rgb_uint8(context_rgb), "Context (last)", color=(66, 82, 112)),
                    _labeled_cell(_rgb_uint8(target_rgb[frame_id]), "GT future", color=(39, 117, 86)),
                    _labeled_cell(_rgb_uint8(base["rgb"][frame_id]), "Base (V6)", color=(137, 82, 50)),
                    _labeled_cell(_rgb_uint8(final["rgb"][frame_id]), "V7 S0 @30K", color=(72, 78, 164)),
                ],
                axis=1,
            )
            bottom = np.concatenate(
                [
                    _info_cell(meta, frame_id, actions[frame_id]),
                    _labeled_cell(_depth_uint8(target_depth[frame_id], lo, hi), "GT depth", color=(39, 117, 86)),
                    _labeled_cell(_depth_uint8(base_depth[frame_id], lo, hi), "Base depth", color=(137, 82, 50)),
                    _labeled_cell(_depth_uint8(final_depth[frame_id], lo, hi), "V7 depth", color=(72, 78, 164)),
                ],
                axis=1,
            )
            frames.append(np.concatenate([top, bottom], axis=0))
        safe_clip = str(meta["clip_id"]).replace("/", "__")[:48]
        gif_path = args.out_dir / f"demo_{demo_id:02d}_{meta['source']}_{safe_clip}.gif"
        # imageio's Pillow GIF writer expects milliseconds (unlike its video
        # writers, which use fps).  Use 320 ms so all eight future frames remain
        # inspectable instead of being rounded to inconsistent 10 ms delays.
        imageio.mimsave(gif_path, frames, duration=320, loop=0)
        frame_paths = []
        for frame_id in (0, min(3, len(frames) - 1), len(frames) - 1):
            png_path = args.out_dir / f"demo_{demo_id:02d}_frame_{frame_id + 1:02d}.png"
            imageio.imwrite(png_path, frames[frame_id])
            frame_paths.append(str(png_path))
            contact_frames.append(frames[frame_id])
        metrics = _demo_metrics(sample, base, final)
        demos.append({"selection": meta, "gif": str(gif_path), "keyframes": frame_paths, "metrics": metrics})
        print(f"wrote {gif_path}")
    if contact_frames:
        width = max(frame.shape[1] for frame in contact_frames)
        resized = [frame if frame.shape[1] == width else np.asarray(Image.fromarray(frame).resize((width, frame.shape[0]))) for frame in contact_frames]
        contact = np.concatenate(resized, axis=0)
        contact_path = args.out_dir / "base_vs_v7_all_demo_keyframes.png"
        imageio.imwrite(contact_path, contact)
    else:
        contact_path = None
    report = {
        "mode": "base_vs_v7_fixed_gt_only_demo_selection",
        "config": str(args.cfg),
        "sampling": {
            "seed": args.seed,
            "candidates_per_source": args.candidates_per_source,
            "source_lengths": source_lengths,
            "selection_uses_model_outputs": False,
        },
        "base_checkpoint": base_load,
        "v7_checkpoint": final_load,
        "contact_sheet": str(contact_path) if contact_path else None,
        "demos": demos,
    }
    report_path = args.out_dir / "demo_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_jsonable))
    print(json.dumps({"report": str(report_path), "demos": [row["gif"] for row in demos]}, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cfg", type=Path, required=True)
    common.add_argument("--ckpt", type=Path, required=True)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--seed", type=int, default=20260716)
    common.add_argument(
        "--allow_future_value_extra",
        action="store_true",
        help="evaluate an S1 checkpoint with only future_value_head keys ignored",
    )

    quality = sub.add_parser("quality", parents=[common])
    quality.add_argument("--out", type=Path, required=True)
    quality.add_argument("--base", action="store_true")
    quality.add_argument("--samples_per_source", type=int, default=32)
    quality.add_argument("--batch_size", type=int, default=2)
    quality.add_argument("--num_workers", type=int, default=2)

    action = sub.add_parser("action", parents=[common])
    action.add_argument("--out", type=Path, required=True)
    action.add_argument("--base", action="store_true")
    action.add_argument("--samples_per_source", type=int, default=32)
    action.add_argument("--batch_size", type=int, default=2)
    action.add_argument("--num_workers", type=int, default=2)
    action.add_argument("--variants", nargs="+", default=["zero", "sign_flip", "grip_toggle"])
    action.add_argument("--grip_transition_only", action="store_true")

    horizon = sub.add_parser("horizon", parents=[common])
    horizon.add_argument("--out", type=Path, required=True)
    horizon.add_argument("--base", action="store_true")
    horizon.add_argument("--candidates_per_source", type=int, default=128)
    horizon.add_argument("--samples_per_source", type=int, default=16)
    horizon.add_argument("--batch_size", type=int, default=2)
    horizon.add_argument("--num_workers", type=int, default=2)

    demo = sub.add_parser("demo", parents=[common])
    demo.add_argument("--base_ckpt", type=Path, required=True)
    demo.add_argument("--out_dir", type=Path, required=True)
    demo.add_argument("--candidates_per_source", type=int, default=48)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quality":
        run_quality(args)
    elif args.command == "action":
        run_action(args)
    elif args.command == "demo":
        run_demo(args)
    elif args.command == "horizon":
        run_horizon(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
