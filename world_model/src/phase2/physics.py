"""Physics-inference module g_phi (scaffold).

Per §4 of the proposal, physical parameters are *latent variables* learned
end-to-end through downstream prediction loss, **not** supervised targets.
The module:

    g_phi : (z_{t0}, ..., z_{t}) ↦ phi ∈ R^K

where ``phi`` is a fixed-size vector meant to encode friction, mass
distribution, contact stiffness, etc., without a hand-designed semantics.

Two use sites:

1. **Phase 1+** (stretch): Feed ``phi`` as an extra conditioning to the
   predictor ``D_psi``. Improvement over Phase-1 D_psi is evidence of H3.
2. **Phase 2**: Enforce that ``g_phi(z_{gen})`` lies close to the
   distribution of ``g_phi(z_{real})``. This is the cross-branch
   consistency mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PhysicsConfig:
    token_dim: int = 2048
    token_grid: int = 8
    context_len: int = 4
    hidden_dim: int = 384
    n_layers: int = 3
    n_heads: int = 6
    phi_dim: int = 64           # dimensionality of latent physics vector


class PhysicsInference(nn.Module):
    def __init__(self, cfg: PhysicsConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.token_grid ** 2
        self.in_proj = nn.Linear(cfg.token_dim, cfg.hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, cfg.context_len * P + 1, cfg.hidden_dim))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.hidden_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.phi_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, C, P, D] -> phi: [B, phi_dim]."""
        B, C, P, D = z.shape
        x = self.in_proj(z.reshape(B, C * P, D))
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, : x.size(1)]
        x = self.backbone(x)
        return self.head(x[:, 0])


# ---------------------------------------------------------- distribution match
def _moments_match_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cheap distribution match: first + second moments. Swap for MMD when
    you have enough bandwidth to estimate it per batch."""
    loss = (a.mean(0) - b.mean(0)).pow(2).mean() \
         + (a.std(0) - b.std(0)).pow(2).mean()
    return loss


def physics_consistency_loss(
    phys: PhysicsInference,
    z_real: torch.Tensor,
    z_gen: torch.Tensor,
) -> torch.Tensor:
    """Enforces that generated token trajectories look physically like real ones
    under g_phi. Returns the loss you add to G_theta's training objective."""
    phi_real = phys(z_real)
    phi_gen = phys(z_gen)
    return _moments_match_loss(phi_real, phi_gen)
