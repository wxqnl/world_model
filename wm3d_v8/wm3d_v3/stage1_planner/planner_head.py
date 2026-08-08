"""Task-conditioned planner over explicit native WM3D futures."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NativePlannerConfig:
    token_dim: int = 2048
    task_dim: int = 2048
    hidden: int = 384
    layers: int = 4
    heads: int = 8
    mlp_mult: int = 4
    dropout: float = 0.1
    max_horizon: int = 32
    patches: int = 64


def _pool_depth(depth: torch.Tensor, grid: int) -> torch.Tensor:
    n, horizon, height, width = depth.shape
    pooled = F.adaptive_avg_pool2d(depth.reshape(n * horizon, 1, height, width), (grid, grid))
    return pooled.reshape(n, horizon, grid * grid, 1)


def _pool_point(point: torch.Tensor, grid: int) -> torch.Tensor:
    if point.ndim != 5 or point.shape[-1] != 3:
        raise ValueError("point evidence must be [N,H,Y,X,3]")
    n, horizon, height, width, _ = point.shape
    channels = point.permute(0, 1, 4, 2, 3).reshape(n * horizon, 3, height, width)
    pooled = F.adaptive_avg_pool2d(channels, (grid, grid))
    return pooled.reshape(n, horizon, 3, grid * grid).permute(0, 1, 3, 2)


class NativePlannerHead(nn.Module):
    """Read token + depth + point + pose futures; never read candidate actions."""

    def __init__(self, cfg: NativePlannerConfig | None = None):
        super().__init__()
        self.cfg = cfg or NativePlannerConfig()
        grid = int(round(math.sqrt(self.cfg.patches)))
        if grid * grid != self.cfg.patches:
            raise ValueError("planner patches must form a square grid")
        if self.cfg.max_horizon <= 0:
            raise ValueError("planner max_horizon must be positive")
        self.grid = grid
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, self.cfg.hidden),
        )
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, self.cfg.hidden),
        )
        self.frame_pos = nn.Parameter(
            torch.zeros(1, self.cfg.max_horizon, 1, self.cfg.hidden)
        )
        self.patch_pos = nn.Parameter(
            torch.zeros(1, 1, self.cfg.patches, self.cfg.hidden)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden,
            nhead=self.cfg.heads,
            dim_feedforward=self.cfg.hidden * self.cfg.mlp_mult,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.cfg.layers)
        self.norm = nn.LayerNorm(self.cfg.hidden)
        self.progress_head = nn.Linear(self.cfg.hidden, 1)
        self.risk_head = nn.Linear(self.cfg.hidden, 1)
        self.success_head = nn.Linear(self.cfg.hidden, 1)
        self.uncertainty_head = nn.Linear(self.cfg.hidden, 1)
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)

    def forward(
        self,
        future_tokens: torch.Tensor,
        task_emb: torch.Tensor,
        *,
        depth: torch.Tensor,
        point: torch.Tensor,
        pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if future_tokens.ndim != 5:
            raise ValueError("future_tokens must be [B,C,H,P,D]")
        bsz, candidates, horizon, patches, token_dim = future_tokens.shape
        if patches != self.cfg.patches or token_dim != self.cfg.token_dim:
            raise ValueError("future token shape differs from planner contract")
        if horizon > self.cfg.max_horizon:
            raise ValueError("future horizon exceeds planner max_horizon")
        if task_emb.shape != (bsz, self.cfg.task_dim):
            raise ValueError("task embedding shape differs from planner contract")
        expected_prefix = (bsz, candidates, horizon)
        if depth.shape[:3] != expected_prefix or point.shape[:3] != expected_prefix or pose.shape != (*expected_prefix, 9):
            raise ValueError("native geometry evidence is not candidate/horizon aligned")

        n = bsz * candidates
        tokens = future_tokens.reshape(n, horizon, patches, token_dim).float()
        depth_patch = _pool_depth(depth.reshape(n, horizon, *depth.shape[-2:]).float(), self.grid)
        point_patch = _pool_point(point.reshape(n, horizon, *point.shape[-3:]).float(), self.grid)
        geometry = torch.cat((torch.log1p(depth_patch.clamp_min(0.0)), point_patch), dim=-1)
        pose_feature = self.pose_proj(pose.reshape(n, horizon, 9).float())[:, :, None]
        x = (
            self.token_proj(tokens)
            + self.geometry_proj(geometry)
            + pose_feature
            + self.frame_pos[:, :horizon]
            + self.patch_pos
        )
        task = self.task_proj(task_emb.float())[:, None].expand(-1, candidates, -1).reshape(
            n, 1, self.cfg.hidden
        )
        encoded = self.norm(
            self.transformer(torch.cat((task, x.reshape(n, horizon * patches, -1)), dim=1))
        )
        task_summary = encoded[:, 0]
        frame_summary = encoded[:, 1:].reshape(n, horizon, patches, -1).mean(dim=2)
        return {
            "progress_logit": self.progress_head(frame_summary).squeeze(-1).reshape(
                bsz, candidates, horizon
            ),
            "risk_logit": self.risk_head(frame_summary).squeeze(-1).reshape(
                bsz, candidates, horizon
            ),
            "success_logit": self.success_head(task_summary).squeeze(-1).reshape(
                bsz, candidates
            ),
            "uncertainty_logit": self.uncertainty_head(task_summary).squeeze(-1).reshape(
                bsz, candidates
            ),
        }


def planning_score(
    outputs: dict[str, torch.Tensor],
    action_cost: torch.Tensor,
    *,
    progress_weight: float = 0.5,
    success_weight: float = 1.0,
    risk_weight: float = 0.5,
    uncertainty_weight: float = 0.25,
    action_cost_weight: float = 0.05,
) -> torch.Tensor:
    """Return calibrated candidate scores; action cost stays outside the head."""

    success = outputs["success_logit"].sigmoid()
    progress = outputs["progress_logit"].sigmoid().mean(dim=-1)
    risk = outputs["risk_logit"].sigmoid().mean(dim=-1)
    uncertainty = outputs["uncertainty_logit"].sigmoid()
    if action_cost.shape != success.shape:
        raise ValueError("action_cost must match planner candidate logits")
    return (
        float(progress_weight) * progress
        + float(success_weight) * success
        - float(risk_weight) * risk
        - float(uncertainty_weight) * uncertainty
        - float(action_cost_weight) * action_cost.float()
    )
