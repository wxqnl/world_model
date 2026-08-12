"""Grounded planning objective for WM3D-V8 Stage1."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
        raise ValueError("planner loss mask contains no valid elements")
    return selected.mean()


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
    if (
        branch_valid.shape != success_logit.shape
        or uncertainty_target.shape != success_logit.shape
    ):
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
        F.binary_cross_entropy_with_logits(
            progress_logit, progress_target, reduction="none"
        ),
        trajectory_mask,
    )
    risk_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(
            risk_logit, risk_target.float(), reduction="none"
        ),
        trajectory_mask,
    )
    success_bce = _masked_mean(
        F.binary_cross_entropy_with_logits(
            success_logit, terminal_target, reduction="none"
        ),
        valid,
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
    predicted_utility = outputs.get("score")
    if predicted_utility is None:
        raise ValueError("planner loss requires the exact serving score")
    if predicted_utility.shape != success_logit.shape:
        raise ValueError("planner serving score must match terminal logits")
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
