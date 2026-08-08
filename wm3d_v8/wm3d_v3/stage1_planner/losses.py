"""Grounded dynamics and planning objectives for V7 Stage1-P."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .rollout import NativeRollout


@dataclass(frozen=True)
class DynamicsLossConfig:
    token_mse_weight: float = 1.0
    token_cosine_weight: float = 0.1
    effect_weight: float = 1.0
    depth_weight: float = 0.25
    point_weight: float = 0.25
    pose_weight: float = 0.1
    effect_floor: float = 1.0e-6


@dataclass(frozen=True)
class PlannerLossConfig:
    progress_weight: float = 0.5
    success_weight: float = 1.0
    risk_weight: float = 0.5
    uncertainty_weight: float = 0.25
    ranking_weight: float = 1.0
    ranking_margin: float = 0.05


def _valid_expand(valid: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    mask = valid.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return mask.expand_as(value)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    if not selected.numel():
        raise ValueError("loss mask contains no valid elements")
    return selected.mean()


def _pool_depth(value: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    b, c, h, y, x = value.shape
    return F.adaptive_avg_pool2d(value.reshape(b * c * h, 1, y, x), target_hw).reshape(
        b, c, h, *target_hw
    )


def _pool_point(value: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    b, c, h, y, x, d = value.shape
    pooled = F.adaptive_avg_pool2d(
        value.permute(0, 1, 2, 5, 3, 4).reshape(b * c * h, d, y, x), target_hw
    )
    return pooled.reshape(b, c, h, d, *target_hw).permute(0, 1, 2, 4, 5, 3)


def native_dynamics_loss(
    rollout: NativeRollout,
    target_tokens: torch.Tensor,
    *,
    target_depth: torch.Tensor,
    target_depth_conf: torch.Tensor,
    target_point: torch.Tensor,
    target_point_conf: torch.Tensor,
    target_pose: torch.Tensor,
    branch_valid: torch.Tensor,
    factual_index: int = 0,
    cfg: DynamicsLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = cfg or DynamicsLossConfig()
    predicted = rollout.tokens.float()
    truth = target_tokens.float()
    if predicted.shape != truth.shape or predicted.ndim != 5:
        raise ValueError("predicted and true native tokens must share [B,C,H,P,D]")
    if branch_valid.shape != predicted.shape[:2]:
        raise ValueError("branch_valid must be [B,C]")
    if not 0 <= factual_index < predicted.shape[1]:
        raise ValueError("factual index is outside candidate dimension")
    valid = branch_valid.bool()

    token_error = (predicted - truth).square().mean(dim=(-1, -2))
    token_mse = _masked_mean(token_error, _valid_expand(valid, token_error))
    token_cosine_frame = 1.0 - F.cosine_similarity(
        predicted.flatten(start_dim=3), truth.flatten(start_dim=3), dim=-1, eps=1.0e-8
    )
    token_cosine = _masked_mean(
        token_cosine_frame, _valid_expand(valid, token_cosine_frame)
    )

    pred_effect = predicted - predicted[:, factual_index : factual_index + 1]
    true_effect = truth - truth[:, factual_index : factual_index + 1]
    effect_error = (pred_effect - true_effect).square().mean(dim=(-1, -2, -3))
    effect_energy = true_effect.square().mean(dim=(-1, -2, -3))
    effect_valid = valid.clone()
    effect_valid[:, factual_index] = False
    effect_normalized = effect_error / effect_energy.clamp_min(float(cfg.effect_floor))
    effect_loss = _masked_mean(effect_normalized, effect_valid)

    if rollout.depth is None or rollout.point is None or rollout.pose is None:
        raise ValueError("native explicit depth/point/pose predictions are required")
    depth_truth = target_depth.float()
    point_truth = target_point.float()
    pose_truth = target_pose.float()
    if depth_truth.ndim != 5 or point_truth.ndim != 6 or pose_truth.ndim != 4:
        raise ValueError("branch geometry targets have invalid ranks")
    depth_pred = _pool_depth(rollout.depth.float(), depth_truth.shape[-2:])
    point_pred = _pool_point(rollout.point.float(), point_truth.shape[-3:-1])
    if depth_pred.shape != depth_truth.shape or point_pred.shape != point_truth.shape or rollout.pose.shape != pose_truth.shape:
        raise ValueError("native geometry predictions do not match branch targets")
    depth_mask = _valid_expand(valid, depth_truth) & (target_depth_conf > 0)
    if target_point_conf.shape != point_truth.shape[:-1]:
        raise ValueError("point confidence must match [B,C,H,Y,X]")
    point_mask = _valid_expand(valid, point_truth) & (
        (target_point_conf > 0).unsqueeze(-1).expand_as(point_truth)
    )
    pose_mask = _valid_expand(valid, pose_truth)
    depth_loss = _masked_mean(
        (torch.log1p(depth_pred.clamp_min(0.0)) - torch.log1p(depth_truth.clamp_min(0.0))).abs(),
        depth_mask,
    )
    point_loss = _masked_mean(F.smooth_l1_loss(point_pred, point_truth, reduction="none"), point_mask)
    pose_loss = _masked_mean(F.smooth_l1_loss(rollout.pose.float(), pose_truth, reduction="none"), pose_mask)

    total = (
        float(cfg.token_mse_weight) * token_mse
        + float(cfg.token_cosine_weight) * token_cosine
        + float(cfg.effect_weight) * effect_loss
        + float(cfg.depth_weight) * depth_loss
        + float(cfg.point_weight) * point_loss
        + float(cfg.pose_weight) * pose_loss
    )
    result = {
        "loss": total,
        "token_mse": token_mse,
        "token_cosine": token_cosine,
        "effect_normalized_error": effect_loss,
        "effect_gain_vs_zero": 1.0 - effect_loss.detach(),
        "depth": depth_loss,
        "point": point_loss,
        "pose": pose_loss,
    }
    for prefix in (8, 16, 32):
        if predicted.shape[2] >= prefix:
            prefix_pred = pred_effect[:, :, :prefix]
            prefix_true = true_effect[:, :, :prefix]
            err = (prefix_pred - prefix_true).square().mean(dim=(-1, -2, -3))
            energy = prefix_true.square().mean(dim=(-1, -2, -3))
            normalized = err / energy.clamp_min(float(cfg.effect_floor))
            result[f"effect_gain_h{prefix}"] = 1.0 - _masked_mean(normalized, effect_valid).detach()
    return result


def imagined_uncertainty_target(
    predicted_tokens: torch.Tensor,
    true_tokens: torch.Tensor,
) -> torch.Tensor:
    """Detached normalized world error used as an uncertainty target."""

    error = (predicted_tokens.float() - true_tokens.float()).square().mean(dim=(-1, -2, -3))
    scale = true_tokens.float().square().mean(dim=(-1, -2, -3)).clamp_min(1.0e-6)
    return (error / scale).detach().clamp(0.0, 1.0)


def planner_loss(
    outputs: dict[str, torch.Tensor],
    *,
    branch_rewards: torch.Tensor,
    branch_dones: torch.Tensor,
    branch_success: torch.Tensor,
    branch_valid: torch.Tensor,
    uncertainty_target: torch.Tensor,
    cfg: PlannerLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = cfg or PlannerLossConfig()
    progress_logit = outputs["progress_logit"]
    success_logit = outputs["success_logit"]
    risk_logit = outputs["risk_logit"]
    uncertainty_logit = outputs["uncertainty_logit"]
    if branch_success.shape != progress_logit.shape:
        raise ValueError("simulator outcome trajectory must match planner horizon")
    if branch_valid.shape != success_logit.shape or uncertainty_target.shape != success_logit.shape:
        raise ValueError("candidate validity/uncertainty must match terminal logits")
    valid = branch_valid.bool()
    success = branch_success.float()
    rewards = branch_rewards.float().clamp(0.0, 1.0)
    cumulative_success = torch.cummax(success, dim=-1).values
    cumulative_reward = torch.cummax(rewards, dim=-1).values
    progress_target = torch.maximum(cumulative_success, cumulative_reward)
    risk_target = branch_dones.bool() & ~cumulative_success.bool()
    terminal_target = cumulative_success.amax(dim=-1)
    trajectory_mask = _valid_expand(valid, progress_logit)
    progress_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(progress_logit, progress_target, reduction="none"),
        trajectory_mask,
    )
    risk_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(risk_logit, risk_target.float(), reduction="none"),
        trajectory_mask,
    )
    success_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(success_logit, terminal_target, reduction="none"), valid
    )
    uncertainty_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(
            uncertainty_logit, uncertainty_target.float(), reduction="none"
        ),
        valid,
    )

    utility = (
        2.0 * terminal_target
        + 0.5 * progress_target.mean(dim=-1)
        - 0.5 * risk_target.float().mean(dim=-1)
    )
    predicted_utility = (
        success_logit.sigmoid()
        + 0.5 * progress_logit.sigmoid().mean(dim=-1)
        - 0.5 * risk_logit.sigmoid().mean(dim=-1)
    )
    pair_mask = valid[:, :, None] & valid[:, None, :]
    target_margin = utility[:, :, None] - utility[:, None, :]
    pair_mask &= target_margin > float(cfg.ranking_margin)
    predicted_margin = predicted_utility[:, :, None] - predicted_utility[:, None, :]
    ranking_loss = (
        F.softplus(-predicted_margin[pair_mask]).mean()
        if bool(pair_mask.any())
        else success_logit.sum() * 0.0
    )
    ranking_acc = (
        (predicted_margin[pair_mask] > 0).float().mean()
        if bool(pair_mask.any())
        else success_logit.new_zeros(())
    )
    total = (
        float(cfg.progress_weight) * progress_bce
        + float(cfg.success_weight) * success_bce
        + float(cfg.risk_weight) * risk_bce
        + float(cfg.uncertainty_weight) * uncertainty_bce
        + float(cfg.ranking_weight) * ranking_loss
    )
    return {
        "loss": total,
        "progress_bce": progress_bce,
        "success_bce": success_bce,
        "risk_bce": risk_bce,
        "uncertainty_bce": uncertainty_bce,
        "ranking_loss": ranking_loss,
        "ranking_acc": ranking_acc,
        "ranking_pairs": pair_mask.sum().to(dtype=success_logit.dtype),
    }
