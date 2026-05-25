"""Cross-branch coupling between G_theta, D_psi, and g_phi (scaffold).

The coupling has two legs:

1. **Generate-then-predict.** Sample a future token sequence from G_theta;
   feed it into D_psi as if it were a real observation; require that D_psi's
   own internal consistency (next-token prediction error on the generated
   trajectory) is low. Intuition: real scenes should be easy for a frozen
   predictor to extrapolate; physically implausible generations should not.

2. **Physics distribution match.** ``g_phi(z_generated)`` should come from
   the same distribution as ``g_phi(z_real)``. Implemented via
   :func:`src.phase2.physics.physics_consistency_loss`.

The self-improvement loop (Phase 3) is bootstrapped from these two signals.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .physics import physics_consistency_loss


def predictor_selfconsistency_loss(
    predictor: nn.Module,
    z_gen: torch.Tensor,          # [B, T, P, D] generated token trajectory
    context_len: int,
    token_pool: str = "mean",
) -> torch.Tensor:
    """Run the (frozen) predictor on ``z_gen`` one step at a time and average
    the squared error. Low when ``z_gen`` looks like a plausible rollout.

    The predictor is assumed to have the Phase-1 interface:
        predictor(ctx_tokens [B, C, ...], ctx_actions [B, C, A], tgt_action [B, A])

    Because we have no real actions for generated trajectories, we use zero
    actions and rely on the ``vggt_noact`` ablation predictor (which Phase 1
    produces anyway). This is a design choice — revisit if the noact baseline
    turns out to be too weak.
    """
    B, T, P, D = z_gen.shape
    if token_pool == "mean":
        state = z_gen.mean(dim=2)          # [B, T, D]
    else:
        state = z_gen                      # [B, T, P, D]  (predictor must accept this)

    C = context_len
    if T <= C:
        raise ValueError("generated trajectory must be longer than context_len")

    losses = []
    for t in range(C - 1, T - 1):
        ctx = state[:, t - C + 1 : t + 1]
        A = 7  # Phase 1 DROID
        a_zero = torch.zeros(B, C, A, device=z_gen.device)
        a_tgt = torch.zeros(B, A, device=z_gen.device)
        pred = predictor(ctx, a_zero, a_tgt)
        real = state[:, t + 1]
        losses.append((pred - real).pow(2).mean())
    return torch.stack(losses).mean()


def coupling_total(
    predictor: nn.Module,
    physics: nn.Module,
    z_real: torch.Tensor,
    z_gen: torch.Tensor,
    context_len: int,
    w_selfconsistency: float = 1.0,
    w_physics: float = 0.1,
) -> dict[str, torch.Tensor]:
    # Physics sees the first ``physics.cfg.context_len`` frames on both sides.
    phys_C = physics.cfg.context_len
    z_real_ph = z_real[:, :phys_C]
    z_gen_ph = z_gen[:, :phys_C]
    sc = predictor_selfconsistency_loss(predictor, z_gen, context_len)
    ph = physics_consistency_loss(physics, z_real_ph, z_gen_ph)
    total = w_selfconsistency * sc + w_physics * ph
    return {"total": total, "self_consistency": sc.detach(), "physics": ph.detach()}
