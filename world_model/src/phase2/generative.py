"""Generative head G_theta — token-space flow matching (scaffold).

Design choices (staged so we can pick this up without re-reading the proposal):

* **Latent space** = VGGT pooled tokens (same shape as Phase 1 inputs,
  [T, G*G, D]). Keeps G_theta and D_psi in the same representation so the
  coupling loss is meaningful.
* **Conditioning** = (a) task text, encoded via a frozen text encoder; and
  (b) an initial frame's tokens, so the generator hallucinates the *future*
  rather than the whole sequence from scratch.
* **Algorithm** = rectified flow (Lipman 2022 / Liu 2023). Simple loss
  ``|v_theta(z_t, t, c) - (z_1 - z_0)|^2``, no SDE sampler needed at eval.

The implementation here is intentionally a small working prototype — enough
to unit-test shapes and gradient flow. Real training belongs in
``scripts/phase2/train_generative.py`` (not shipped).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GenerativeConfig:
    token_dim: int = 2048
    token_grid: int = 8
    seq_len: int = 8                # number of frames to synthesize
    cond_text_dim: int = 512        # frozen text encoder output dim (stub)
    hidden_dim: int = 768
    n_layers: int = 8
    n_heads: int = 12
    dropout: float = 0.0
    use_checkpoint: bool = False    # activation checkpointing on backbone


class TimeEmbed(nn.Module):
    """Sinusoidal embedding of the flow-matching time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowMatchingGenerator(nn.Module):
    """Predict velocity v_theta(z_t, t, cond)."""

    def __init__(self, cfg: GenerativeConfig):
        super().__init__()
        self.cfg = cfg
        P = cfg.token_grid ** 2
        self.P = P
        self.in_proj = nn.Linear(cfg.token_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.token_dim)
        self.time_embed = TimeEmbed(cfg.hidden_dim)
        self.cond_proj = nn.Linear(cfg.cond_text_dim, cfg.hidden_dim)
        self.init_proj = nn.Linear(cfg.token_dim, cfg.hidden_dim)

        # positional: (frame, patch) grid over T x P.
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len * P, cfg.hidden_dim))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim, nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        z_t: torch.Tensor,          # [B, T, P, D]
        t: torch.Tensor,            # [B]
        cond_text: torch.Tensor,    # [B, cond_text_dim]
        init_frame: torch.Tensor,   # [B, P, D]
    ) -> torch.Tensor:
        B, T, P, D = z_t.shape
        x = self.in_proj(z_t.reshape(B, T * P, D))
        # Additive conditioning on every token.
        x = x + self.time_embed(t).unsqueeze(1)
        x = x + self.cond_proj(cond_text).unsqueeze(1)
        init_tok = self.init_proj(init_frame)   # [B, P, H]
        # broadcast init tokens to frames (share across T so they act as a
        # "static reference" that the generator has to deform).
        x = x + init_tok.repeat(1, T, 1)
        x = x + self.pos[:, : x.size(1)]
        if self.cfg.use_checkpoint and self.training:
            from torch.utils.checkpoint import checkpoint
            h = x
            for layer in self.backbone.layers:
                h = checkpoint(layer, h, use_reentrant=False)
            if self.backbone.norm is not None:
                h = self.backbone.norm(h)
        else:
            h = self.backbone(x)
        h = self.norm(h)
        v = self.out_proj(h).reshape(B, T, P, D)
        return v

    @torch.no_grad()
    def sample(self, cond_text, init_frame, n_steps: int = 24) -> torch.Tensor:
        """Euler integrator for the flow ODE. Returns tokens at t=1."""
        B = cond_text.size(0)
        device = cond_text.device
        T, P, D = self.cfg.seq_len, self.P, self.cfg.token_dim
        z = torch.randn(B, T, P, D, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self(z, t, cond_text, init_frame)
            z = z + dt * v
        return z


def flow_matching_loss(
    model: FlowMatchingGenerator,
    z1: torch.Tensor,            # [B, T, P, D] real tokens
    cond_text: torch.Tensor,
    init_frame: torch.Tensor,
) -> torch.Tensor:
    B = z1.size(0)
    t = torch.rand(B, device=z1.device)
    z0 = torch.randn_like(z1)
    zt = (1 - t)[:, None, None, None] * z0 + t[:, None, None, None] * z1
    target = z1 - z0
    pred = model(zt, t, cond_text, init_frame)
    return (pred - target).pow(2).mean()
