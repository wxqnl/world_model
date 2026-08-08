"""Action proposal head for tau0-style test-time computation.

The proposer is intentionally small: it predicts a small set of candidate
future action chunks from the current context and task embedding. The world
model still owns the simulator; this module only supplies candidate actions
for later propose -> simulate -> rank loops.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ActionProposerConfig:
    token_dim: int = 2048
    task_dim: int = 2048
    hidden: int = 512
    n_layers: int = 3
    n_candidates: int = 4
    horizon: int = 8
    dropout: float = 0.0
    use_task: bool = True


class ActionProposer(nn.Module):
    """Predict K candidate action chunks from context tokens and task state."""

    def __init__(self, cfg: ActionProposerConfig | None = None):
        super().__init__()
        self.cfg = cfg or ActionProposerConfig()
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.candidate_embed = nn.Parameter(torch.zeros(1, self.cfg.n_candidates, 1, self.cfg.hidden))
        self.horizon_embed = nn.Parameter(torch.zeros(1, 1, self.cfg.horizon, self.cfg.hidden))
        nn.init.normal_(self.candidate_embed, std=0.02)
        nn.init.normal_(self.horizon_embed, std=0.02)

        layers: list[nn.Module] = []
        for _ in range(max(1, self.cfg.n_layers)):
            layers.extend([
                nn.LayerNorm(self.cfg.hidden),
                nn.Linear(self.cfg.hidden, self.cfg.hidden * 4),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden * 4, self.cfg.hidden),
            ])
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(self.cfg.hidden)
        self.pose_norm_head = nn.Linear(self.cfg.hidden, 6)
        self.gripper_head = nn.Linear(self.cfg.hidden, 1)

    def forward(self, context_tokens: torch.Tensor, task_emb: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return candidate chunks.

        Args:
            context_tokens: `[B, T, P, D]` context VGGT tokens.
            task_emb: optional `[B, task_dim]` task embedding.
        """
        if context_tokens.ndim != 4:
            raise ValueError(f"context_tokens must be [B,T,P,D], got {tuple(context_tokens.shape)}")
        bsz, _time, _patches, dim = context_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")

        pooled = context_tokens.mean(dim=(1, 2))
        base = self.context_proj(pooled)
        if self.cfg.use_task:
            if task_emb is None:
                task_emb = torch.zeros(bsz, self.cfg.task_dim, device=context_tokens.device, dtype=context_tokens.dtype)
            if task_emb.shape != (bsz, self.cfg.task_dim):
                raise ValueError(f"task_emb must be {(bsz, self.cfg.task_dim)}, got {tuple(task_emb.shape)}")
            base = base + self.task_proj(task_emb.to(device=context_tokens.device, dtype=base.dtype))

        x = base[:, None, None, :] + self.candidate_embed.to(dtype=base.dtype) + self.horizon_embed.to(dtype=base.dtype)
        h = self.mlp(x)
        h = self.norm(h)
        pose_norm = self.pose_norm_head(h)
        gripper_logit = self.gripper_head(h).squeeze(-1)
        proposer_action_cond = torch.cat([pose_norm, torch.sigmoid(gripper_logit)[..., None]], dim=-1)
        return {
            "proposer_pose_norm": pose_norm,
            "proposer_gripper_logit": gripper_logit,
            "proposer_action_cond": proposer_action_cond,
        }
