"""Ground candidate futures with true same-root simulator outcomes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FutureValueLossConfig:
    trajectory_weight: float = 1.0
    terminal_weight: float = 1.0
    ranking_weight: float = 0.0
    positive_weight: float = 1.0


def true_branch_future_value_loss(
    candidate_progress_logit: torch.Tensor,
    candidate_success_logit: torch.Tensor,
    branch_success: torch.Tensor,
    *,
    branch_valid: torch.Tensor | None = None,
    cfg: FutureValueLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise value without exposing candidate actions to the value head.

    ``branch_success`` is the simulator success state at every future step,
    never a pseudo-label.  Terminal success means that the task succeeded at
    least once inside the rollout horizon.
    """

    cfg = cfg or FutureValueLossConfig()
    if candidate_progress_logit.ndim != 3:
        raise ValueError("candidate_progress_logit must be [B,K,H]")
    if candidate_success_logit.shape != candidate_progress_logit.shape[:2]:
        raise ValueError("candidate_success_logit must be [B,K]")
    if branch_success.shape != candidate_progress_logit.shape:
        raise ValueError("branch_success must match candidate progress [B,K,H]")

    success = branch_success.to(
        device=candidate_progress_logit.device,
        dtype=candidate_progress_logit.dtype,
    )
    if branch_valid is None:
        valid = torch.ones(
            candidate_progress_logit.shape[:2],
            device=candidate_progress_logit.device,
            dtype=torch.bool,
        )
    else:
        if branch_valid.shape != candidate_progress_logit.shape[:2]:
            raise ValueError("branch_valid must be [B,K]")
        valid = branch_valid.to(device=candidate_progress_logit.device, dtype=torch.bool)
    if not bool(valid.any().detach().cpu()):
        raise ValueError("future value batch has no valid true branches")

    positive_weight = candidate_progress_logit.new_tensor(float(cfg.positive_weight))
    trajectory_raw = F.binary_cross_entropy_with_logits(
        candidate_progress_logit,
        success,
        reduction="none",
        pos_weight=positive_weight,
    )
    trajectory_mask = valid[..., None].expand_as(trajectory_raw)
    trajectory_loss = trajectory_raw[trajectory_mask].mean()

    terminal_target = success.amax(dim=-1)
    terminal_raw = F.binary_cross_entropy_with_logits(
        candidate_success_logit,
        terminal_target,
        reduction="none",
        pos_weight=positive_weight,
    )
    terminal_loss = terminal_raw[valid].mean()
    ranking_terms = []
    ranking_correct = []
    for sample_logits, sample_target, sample_valid in zip(
        candidate_success_logit, terminal_target, valid
    ):
        positive = sample_logits[(sample_target > 0.5) & sample_valid]
        negative = sample_logits[(sample_target <= 0.5) & sample_valid]
        if positive.numel() and negative.numel():
            pairwise_margin = positive[:, None] - negative[None, :]
            ranking_terms.append(F.softplus(-pairwise_margin).mean())
            ranking_correct.append((pairwise_margin > 0).float().mean())
    if ranking_terms:
        ranking_loss = torch.stack(ranking_terms).mean()
        ranking_acc = torch.stack(ranking_correct).mean()
        ranking_pairs = candidate_success_logit.new_tensor(
            sum(
                int(((sample_target > 0.5) & sample_valid).sum())
                * int(((sample_target <= 0.5) & sample_valid).sum())
                for sample_target, sample_valid in zip(terminal_target, valid)
            )
        )
    else:
        ranking_loss = candidate_success_logit.sum() * 0.0
        ranking_acc = candidate_success_logit.new_zeros(())
        ranking_pairs = candidate_success_logit.new_zeros(())
    total = (
        float(cfg.trajectory_weight) * trajectory_loss
        + float(cfg.terminal_weight) * terminal_loss
        + float(cfg.ranking_weight) * ranking_loss
    )

    with torch.no_grad():
        trajectory_pred = candidate_progress_logit > 0
        terminal_pred = candidate_success_logit > 0
        trajectory_acc = (
            trajectory_pred[trajectory_mask] == (success[trajectory_mask] > 0.5)
        ).float().mean()
        terminal_acc = (
            terminal_pred[valid] == (terminal_target[valid] > 0.5)
        ).float().mean()
        terminal_positive = terminal_target[valid].mean()
        per_root_positive = ((terminal_target > 0.5) & valid).any(dim=1)
        per_root_negative = ((terminal_target <= 0.5) & valid).any(dim=1)
        mixed = (per_root_positive & per_root_negative).float().mean()

    return {
        "loss": total,
        "trajectory_bce": trajectory_loss,
        "terminal_bce": terminal_loss,
        "ranking_loss": ranking_loss,
        "ranking_acc": ranking_acc,
        "ranking_pairs": ranking_pairs,
        "trajectory_acc": trajectory_acc,
        "terminal_acc": terminal_acc,
        "terminal_positive_fraction": terminal_positive,
        "mixed_terminal_labels": mixed,
    }
