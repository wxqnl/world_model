"""Predictive head D_psi for Phase 1.

A small transformer over the flattened context (C past per-frame tokens + C
action embeddings), producing the predicted next-frame token. We keep the
architecture intentionally vanilla so the comparison across backbones
(VGGT vs DINOv2) and ablations (zero-action) is controlled.

Input per sample (mean-pooled variant):
    ctx_tokens     [B, C, D_in]
    ctx_actions    [B, C, A]
    tgt_tokens     [B, D_in]     (only used in loss)
    tgt_action     [B, A]

Output:
    pred_tokens    [B, D_in]

The architecture:
    1. Linear project tokens D_in -> H (hidden_dim).
    2. Embed actions A -> A_emb and sum into the corresponding frame position.
       Also append a "query" slot that receives the target action (the action
       whose effect we're predicting).
    3. Add learned positional embedding over (C + 1) slots.
    4. N standard Transformer encoder layers.
    5. Read out the query slot, project back to D_in.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PredictiveHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        context_len: int = 4,
        action_embed_dim: int = 64,
        dropout: float = 0.1,
        use_actions: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.C = context_len
        self.use_actions = use_actions
        self.use_checkpoint = use_checkpoint
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, token_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.GELU(),
            nn.Linear(action_embed_dim, hidden_dim),
        )

        self.pos = nn.Parameter(torch.zeros(1, context_len + 1, hidden_dim))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def _run_backbone(self, h: torch.Tensor) -> torch.Tensor:
        if not (self.use_checkpoint and self.training):
            return self.backbone(h)
        from torch.utils.checkpoint import checkpoint
        for layer in self.backbone.layers:
            h = checkpoint(layer, h, use_reentrant=False)
        if self.backbone.norm is not None:
            h = self.backbone.norm(h)
        return h

    def forward(
        self,
        ctx_tokens: torch.Tensor,      # [B, C, D_in]
        ctx_actions: torch.Tensor,     # [B, C, A]
        tgt_action: torch.Tensor,      # [B, A]
    ) -> torch.Tensor:
        B, C, _ = ctx_tokens.shape
        x = self.token_proj(ctx_tokens)                         # [B, C, H]
        if self.use_actions:
            x = x + self.action_proj(ctx_actions)               # condition each frame
            q = self.query.expand(B, -1, -1) + self.action_proj(tgt_action).unsqueeze(1)
        else:
            q = self.query.expand(B, -1, -1)
        h = torch.cat([x, q], dim=1)                            # [B, C+1, H]
        h = h + self.pos[:, : h.size(1)]
        h = self._run_backbone(h)
        h = self.norm(h[:, -1])                                 # [B, H]
        return self.out_proj(h)                                 # [B, D_in]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
