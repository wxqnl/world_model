"""Losses for true same-state simulator branches; pseudo CF is unsupported."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrueBranchLossConfig:
    temperature: float = 0.1
    reconstruction_weight: float = 1.0
    matching_weight: float = 1.0
    effect_temperature: float = 0.07
    effect_reconstruction_weight: float = 0.0
    effect_matching_weight: float = 0.0
    effect_min_rms: float = 1e-3


def pairwise_future_distance(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return all KxK candidate/target MSE distances as ``[B,K,K]``."""
    if predicted.shape != target.shape or predicted.ndim < 3:
        raise ValueError("predicted and target branches must have identical [B,K,...] shapes")
    difference = predicted[:, :, None].float() - target[:, None, :].float()
    reduce_dims = tuple(range(3, difference.ndim))
    return difference.square().mean(dim=reduce_dims)


def true_branch_reconstruction_matching_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    branch_valid: torch.Tensor | None = None,
    cfg: TrueBranchLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    settings = cfg or TrueBranchLossConfig()
    if settings.temperature <= 0:
        raise ValueError("temperature must be positive")
    if settings.effect_temperature <= 0:
        raise ValueError("effect_temperature must be positive")
    if settings.effect_min_rms <= 0:
        raise ValueError("effect_min_rms must be positive")
    bsz, candidates = predicted.shape[:2]
    if candidates < 2:
        raise ValueError("true counterfactual training requires at least two branches")
    if branch_valid is not None:
        if branch_valid.shape != (bsz, candidates) or not bool(torch.all(branch_valid)):
            raise ValueError("formal true-branch batches require every K branch target to be valid")
    distances = pairwise_future_distance(predicted, target)
    labels = torch.arange(candidates, device=predicted.device)[None].expand(bsz, -1)
    own_distance = distances.gather(2, labels[..., None]).squeeze(-1)
    reconstruction = own_distance.mean()
    logits = -distances / settings.temperature
    matching = F.cross_entropy(logits.reshape(bsz * candidates, candidates), labels.reshape(-1))
    top1 = (logits.argmax(dim=-1) == labels).float().mean()
    # Absolute future-token losses are dominated by the scene content shared
    # by every branch. Train the action effect explicitly relative to the
    # factual branch so a collapsed "average future" cannot satisfy S1.
    predicted_effect = predicted[:, 1:].float() - predicted[:, :1].float()
    target_effect = target[:, 1:].float() - target[:, :1].float()
    effect_reduce_dims = tuple(range(2, target_effect.ndim))
    target_effect_mse = target_effect.square().mean(dim=effect_reduce_dims)
    effect_error_mse = (predicted_effect - target_effect).square().mean(
        dim=effect_reduce_dims
    )
    effect_scale_sq = target_effect_mse.clamp_min(settings.effect_min_rms**2)
    effect_reconstruction = (effect_error_mse / effect_scale_sq).mean()

    predicted_effect_flat = predicted_effect.flatten(start_dim=2)
    target_effect_flat = target_effect.flatten(start_dim=2)
    predicted_effect_unit = F.normalize(
        predicted_effect_flat, dim=-1, eps=settings.effect_min_rms
    )
    target_effect_unit = F.normalize(
        target_effect_flat, dim=-1, eps=settings.effect_min_rms
    )
    effect_logits = torch.einsum(
        "bif,bjf->bij", predicted_effect_unit, target_effect_unit
    ) / settings.effect_temperature
    effect_candidates = candidates - 1
    effect_labels = torch.arange(
        effect_candidates, device=predicted.device
    )[None].expand(bsz, -1)
    effect_matching = F.cross_entropy(
        effect_logits.reshape(bsz * effect_candidates, effect_candidates),
        effect_labels.reshape(-1),
    )
    effect_top1 = (
        effect_logits.argmax(dim=-1) == effect_labels
    ).float().mean()
    effect_cosine = torch.diagonal(
        effect_logits * settings.effect_temperature, dim1=1, dim2=2
    ).mean()
    effect_norm_ratio = (
        predicted_effect_flat.norm(dim=-1)
        / target_effect_flat.norm(dim=-1).clamp_min(settings.effect_min_rms)
    ).mean()

    loss = (
        settings.reconstruction_weight * reconstruction
        + settings.matching_weight * matching
        + settings.effect_reconstruction_weight * effect_reconstruction
        + settings.effect_matching_weight * effect_matching
    )
    return {
        "loss": loss,
        "branch_reconstruction": reconstruction,
        "branch_matching": matching,
        "branch_matching_top1": top1,
        "effect_reconstruction": effect_reconstruction,
        "effect_matching": effect_matching,
        "effect_matching_top1": effect_top1,
        "effect_cosine": effect_cosine,
        "effect_norm_ratio": effect_norm_ratio,
        "pairwise_distance": distances,
    }


def centered_candidate_rank(candidate_futures: torch.Tensor, *, relative_tol: float = 1e-4) -> torch.Tensor:
    """Numerical rank of action-dependent candidate variation for each batch item."""
    if candidate_futures.ndim < 3:
        raise ValueError("candidate_futures must be [B,K,...]")
    flattened = candidate_futures.float().flatten(start_dim=2)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    threshold = singular_values[:, :1] * relative_tol
    return (singular_values > threshold).sum(dim=1)
