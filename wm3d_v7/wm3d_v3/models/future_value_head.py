"""Task-conditioned value head over candidate-specific frozen world futures."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .progress_head import ProgressHead, ProgressHeadConfig


@dataclass
class FutureValueConfig:
    token_dim: int = 2048
    task_dim: int = 2048
    hidden: int = 256
    n_layers: int = 2
    n_heads: int = 4
    max_horizon: int = 32


class FutureValueHead(nn.Module):
    """Score ``K`` imagined futures without seeing candidate actions directly."""

    def __init__(self, cfg: FutureValueConfig | None = None):
        super().__init__()
        self.cfg = cfg or FutureValueConfig()
        self.backbone = ProgressHead(
            ProgressHeadConfig(
                token_dim=self.cfg.token_dim,
                task_dim=self.cfg.task_dim,
                hidden=self.cfg.hidden,
                n_layers=self.cfg.n_layers,
                n_heads=self.cfg.n_heads,
                max_horizon=self.cfg.max_horizon,
                use_action=False,
                use_task=True,
                enable_plausibility=False,
            )
        )

    def forward(self, candidate_future_tokens: torch.Tensor, task_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        if candidate_future_tokens.ndim != 5:
            raise ValueError("candidate futures must be [B,K,H,P,D]")
        bsz, candidates, horizon, patches, dim = candidate_future_tokens.shape
        if task_emb.shape != (bsz, self.cfg.task_dim):
            raise ValueError(f"task_emb must be {(bsz, self.cfg.task_dim)}")
        # This stop-gradient is an architectural contract, not a training-loop convention.
        futures = candidate_future_tokens.detach().reshape(bsz * candidates, horizon, patches, dim)
        tasks = task_emb[:, None].expand(-1, candidates, -1).reshape(bsz * candidates, -1)
        result = self.backbone(futures, action_cond=None, task_emb=tasks)
        return {
            "candidate_progress_logit": result["progress"].reshape(bsz, candidates, horizon),
            "candidate_success_logit": result["terminal_success_logit"].reshape(bsz, candidates),
        }
