"""Independent progress prediction head for future token rollouts."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProgressHeadConfig:
    token_dim: int = 2048
    hidden: int = 256
    n_layers: int = 2
    n_heads: int = 4
    mlp_mult: int = 4
    dropout: float = 0.0
    action_dim: int = 7
    task_dim: int = 2048
    max_horizon: int = 32
    use_action: bool = True
    use_task: bool = True
    enable_plausibility: bool = True


class ProgressHead(nn.Module):
    """Predict rollout progress and sequence-level terminal scores."""

    def __init__(self, cfg: ProgressHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or ProgressHeadConfig()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(self.cfg.action_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        ) if self.cfg.use_action else None
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        ) if self.cfg.use_task else None
        self.pos = nn.Parameter(torch.zeros(1, self.cfg.max_horizon, self.cfg.hidden))
        nn.init.normal_(self.pos, std=0.02)

        if self.cfg.n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.hidden,
                nhead=self.cfg.n_heads,
                dim_feedforward=self.cfg.hidden * self.cfg.mlp_mult,
                dropout=self.cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.n_layers)
        else:
            self.encoder = nn.Identity()
        self.norm = nn.LayerNorm(self.cfg.hidden)
        self.progress_head = nn.Linear(self.cfg.hidden, 1)
        self.terminal_success_head = nn.Linear(self.cfg.hidden, 1)
        self.plausibility_head = (
            nn.Linear(self.cfg.hidden, 1) if self.cfg.enable_plausibility else None
        )

    def _pos(self, horizon: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if horizon <= self.cfg.max_horizon:
            return self.pos[:, :horizon].to(device=device, dtype=dtype)
        pos = self.pos.transpose(1, 2)
        pos = nn.functional.interpolate(pos, size=horizon, mode="linear", align_corners=False)
        return pos.transpose(1, 2).to(device=device, dtype=dtype)

    def forward(
        self,
        future_tokens: torch.Tensor,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-step progress and sequence-level terminal scores.

        Args:
            future_tokens: [B, k, P, D] future token rollout.
            action_cond: optional [B, k, 7] action condition.
            task_emb: optional [B, 2048] task embedding.
        """
        if future_tokens.ndim != 4:
            raise ValueError(f"future_tokens must be [B,k,P,D], got {tuple(future_tokens.shape)}")
        bsz, horizon, _patches, dim = future_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected D={self.cfg.token_dim}, got {dim}")

        x = self.token_proj(future_tokens.mean(dim=2))
        if self.cfg.use_action:
            assert self.action_proj is not None
            if action_cond is None:
                action_cond = torch.zeros(
                    bsz, horizon, self.cfg.action_dim, device=future_tokens.device, dtype=x.dtype
                )
            if action_cond.shape != (bsz, horizon, self.cfg.action_dim):
                raise ValueError(
                    f"action_cond must be {(bsz, horizon, self.cfg.action_dim)}, "
                    f"got {tuple(action_cond.shape)}"
                )
            x = x + self.action_proj(action_cond.to(device=future_tokens.device, dtype=x.dtype))
        if self.cfg.use_task:
            assert self.task_proj is not None
            if task_emb is None:
                task_emb = torch.zeros(bsz, self.cfg.task_dim, device=future_tokens.device, dtype=x.dtype)
            if task_emb.shape != (bsz, self.cfg.task_dim):
                raise ValueError(
                    f"task_emb must be {(bsz, self.cfg.task_dim)}, got {tuple(task_emb.shape)}"
                )
            task = self.task_proj(task_emb.to(device=future_tokens.device, dtype=x.dtype))
            x = x + task[:, None]

        x = x + self._pos(horizon, future_tokens.device, x.dtype)
        h = self.norm(self.encoder(x))
        pooled = h.mean(dim=1)
        result = {
            "progress": self.progress_head(h).squeeze(-1),
            "terminal_success_logit": self.terminal_success_head(pooled).squeeze(-1),
        }
        if self.plausibility_head is not None:
            result["plausibility_logit"] = self.plausibility_head(pooled).squeeze(-1)
        return result
