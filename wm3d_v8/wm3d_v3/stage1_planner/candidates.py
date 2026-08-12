"""Masked grouped action cost for the unified WM3D V8 planner."""
from __future__ import annotations

import torch


def deterministic_action_cost(
    actions: torch.Tensor,
    action_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    coarse_actions: torch.Tensor | None = None,
    coarse_action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a deterministic normalized-action cost with shape ``[B,C]``.

    ``actions`` is the same grouped, normalized candidate ABI consumed by the
    frozen Stage0 dynamics: ``[B,C,H,G,S,A]``.  No semantic, arm-count, action
    dimension, or source cadence is inferred here; masks from the sealed data
    profile decide which measured controller coordinates exist.
    """

    if actions.ndim != 6:
        raise ValueError("grouped candidate actions must be [B,C,H,G,S,A]")
    if action_mask.shape != actions.shape or action_mask.dtype != torch.bool:
        raise ValueError("candidate action mask must match actions")
    if sample_mask.shape != actions.shape[:-1] or sample_mask.dtype != torch.bool:
        raise ValueError("candidate sample mask must be [B,C,H,G,S]")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("candidate actions contain non-finite values")
    valid = action_mask & sample_mask[..., None]
    count = valid.sum(dim=(-1, -2, -3, -4))
    energy = (actions.float().square() * valid).sum(dim=(-1, -2, -3, -4))
    if (coarse_actions is None) != (coarse_action_mask is None):
        raise ValueError("coarse grouped actions and mask must be provided together")
    if coarse_actions is not None:
        if coarse_actions.ndim != 5 or coarse_actions.shape[:4] != actions.shape[:4]:
            raise ValueError("coarse grouped candidate actions must be [B,C,H,G,A]")
        if (
            coarse_action_mask is None
            or coarse_action_mask.shape != coarse_actions.shape
            or coarse_action_mask.dtype != torch.bool
        ):
            raise ValueError("coarse grouped candidate action mask must match actions")
        if not bool(torch.isfinite(coarse_actions).all()):
            raise ValueError("coarse grouped candidate actions contain non-finite values")
        count = count + coarse_action_mask.sum(dim=(-1, -2, -3))
        energy = energy + (
            coarse_actions.float().square() * coarse_action_mask
        ).sum(dim=(-1, -2, -3))
    return (energy / count.clamp_min(1)).sqrt()
