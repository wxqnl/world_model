"""Action-blind planner over explicit WM3D native 3D futures."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NativePlannerConfig:
    token_dim: int = 0
    task_dim: int = 0
    hidden: int = 256
    spatial_layers: int = 2
    temporal_layers: int = 4
    heads: int = 8
    mlp_mult: int = 4
    dropout: float = 0.1
    max_horizon: int = 0
    patches: int = 0
    num_views: int = 0
    time_fourier_dim: int = 0
    time_min_period_s: float = 0.0
    time_max_period_s: float = 0.0


def _encoder(hidden: int, heads: int, mlp_mult: int, dropout: float, layers: int):
    layer = nn.TransformerEncoderLayer(
        d_model=hidden,
        nhead=heads,
        dim_feedforward=hidden * mlp_mult,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers)


class NativePlannerHead(nn.Module):
    """Read native tokens/depth/point/pose without reading candidate actions.

    Attention is deliberately factorized across the sealed native patch grid
    and then across the configured future horizon.  Candidate actions are not
    accepted by this module.  Real source timestamps and evidence masks are
    required, so ordinal frame position never masquerades as a fixed cadence.
    """

    def __init__(self, cfg: NativePlannerConfig | None = None):
        super().__init__()
        self.cfg = cfg or NativePlannerConfig()
        if self.cfg.patches <= 0 or self.cfg.token_dim <= 0 or self.cfg.task_dim <= 0:
            raise ValueError("planner token/task/patch dimensions must be positive")
        if self.cfg.max_horizon <= 0:
            raise ValueError("planner max_horizon must be positive")
        if min(self.cfg.spatial_layers, self.cfg.temporal_layers) <= 0:
            raise ValueError("planner spatial/temporal layers must be positive")
        if self.cfg.hidden <= 0 or self.cfg.heads <= 0 or self.cfg.hidden % self.cfg.heads:
            raise ValueError("planner hidden must be positive and divisible by heads")
        if (
            self.cfg.time_fourier_dim <= 0
            or self.cfg.time_fourier_dim % 2
            or not 0 < self.cfg.time_min_period_s < self.cfg.time_max_period_s
        ):
            raise ValueError("planner continuous-time contract is invalid")
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, self.cfg.hidden),
        )
        if self.cfg.num_views <= 0:
            raise ValueError("planner num_views must be positive")
        # Geometry values and their availability are both encoded.  A missing
        # depth/point measurement must not be indistinguishable from a real 0.
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, self.cfg.hidden),
        )
        half = self.cfg.time_fourier_dim // 2
        periods = torch.logspace(
            math.log10(self.cfg.time_min_period_s),
            math.log10(self.cfg.time_max_period_s),
            half,
        )
        self.register_buffer("time_frequency", (2.0 * math.pi / periods), persistent=True)
        self.time_proj = nn.Sequential(
            nn.Linear(self.cfg.time_fourier_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.patch_pos = nn.Parameter(
            torch.zeros(1, self.cfg.patches, self.cfg.hidden)
        )
        self.spatial = _encoder(
            self.cfg.hidden,
            self.cfg.heads,
            self.cfg.mlp_mult,
            self.cfg.dropout,
            self.cfg.spatial_layers,
        )
        self.temporal = _encoder(
            self.cfg.hidden,
            self.cfg.heads,
            self.cfg.mlp_mult,
            self.cfg.dropout,
            self.cfg.temporal_layers,
        )
        self.spatial_norm = nn.LayerNorm(self.cfg.hidden)
        self.temporal_norm = nn.LayerNorm(self.cfg.hidden)
        self.progress_head = nn.Linear(self.cfg.hidden, 1)
        self.risk_head = nn.Linear(self.cfg.hidden, 1)
        self.success_head = nn.Linear(self.cfg.hidden, 1)
        self.uncertainty_head = nn.Linear(self.cfg.hidden, 1)
        nn.init.normal_(self.patch_pos, std=0.02)

    def forward(
        self,
        future_tokens: torch.Tensor,
        task_emb: torch.Tensor,
        *,
        future_dt_s: torch.Tensor,
        token_mask: torch.Tensor,
        depth: torch.Tensor,
        depth_mask: torch.Tensor,
        point: torch.Tensor,
        point_mask: torch.Tensor,
        pose: torch.Tensor,
        pose_mask: torch.Tensor,
        geometry_confidence: torch.Tensor,
        view_mask: torch.Tensor,
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
        expected = (bsz, candidates, horizon)
        view_expected = (*expected, self.cfg.num_views)
        if (
            future_dt_s.shape != expected
            or token_mask.shape != (*expected, patches)
            or depth.shape != (*view_expected, patches)
            or depth_mask.shape != depth.shape
            or point.shape != (*view_expected, patches, 3)
            or point_mask.shape != point.shape[:-1]
            or pose.shape != (*view_expected, 9)
            or pose_mask.shape != view_expected
            or geometry_confidence.shape != depth.shape
            or view_mask.shape != view_expected
        ):
            raise ValueError("native geometry evidence is not candidate/horizon aligned")
        masks = (token_mask, depth_mask, point_mask, pose_mask, view_mask)
        if any(mask.dtype != torch.bool for mask in masks):
            raise ValueError("native evidence masks must be boolean")
        if not bool(torch.isfinite(future_dt_s).all()) or bool((future_dt_s <= 0).any()):
            raise ValueError("future_dt_s must contain finite positive offsets")
        if horizon > 1 and not bool(torch.diff(future_dt_s, dim=-1).gt(0).all()):
            raise ValueError("future_dt_s must be strictly increasing")
        if not bool(token_mask.any(dim=-1).all()):
            raise ValueError("every candidate future frame needs native token evidence")

        n = bsz * candidates
        tokens = future_tokens.reshape(n, horizon, patches, token_dim)
        token_valid = token_mask.reshape(n, horizon, patches)

        def masked_view_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            weight = (mask & view_mask[..., None]).float()
            denominator = weight.sum(dim=3).clamp_min(1.0)
            if value.ndim == weight.ndim + 1:
                return (value.float() * weight[..., None]).sum(dim=3) / denominator[..., None]
            return (value.float() * weight).sum(dim=3) / denominator

        visible_depth = depth_mask & view_mask[..., None]
        visible_point = point_mask & view_mask[..., None]
        depth_present = visible_depth.any(dim=3)
        point_present = visible_point.any(dim=3)
        view_coverage = view_mask.float().mean(dim=3)
        depth = masked_view_mean(depth, depth_mask)
        point = masked_view_mean(point, point_mask)
        confidence_mask = depth_mask | point_mask
        confidence = masked_view_mean(geometry_confidence, confidence_mask)
        depth_patch = depth.reshape(n, horizon, patches, 1)
        point_patch = point.reshape(n, horizon, patches, 3)
        confidence_patch = confidence.reshape(n, horizon, patches, 1)
        geometry = torch.cat(
            (
                torch.log1p(depth_patch.clamp_min(0.0)),
                point_patch,
                confidence_patch,
                depth_present.reshape(n, horizon, patches, 1).float(),
                point_present.reshape(n, horizon, patches, 1).float(),
                view_coverage.reshape(n, horizon, 1, 1).expand(-1, -1, patches, -1),
            ),
            dim=-1,
        )
        pose_weight = (pose_mask & view_mask).float()
        pose_mean = (pose.float() * pose_weight[..., None]).sum(dim=3)
        pose_mean = pose_mean / pose_weight.sum(dim=3).clamp_min(1.0)[..., None]
        pose_present = pose_weight.any(dim=3, keepdim=True).float()
        pose_input = torch.cat((pose_mean, pose_present), dim=-1)
        pose_feature = self.pose_proj(pose_input.reshape(n, horizon, 10))
        phase = future_dt_s.float()[..., None] * self.time_frequency
        time_feature = torch.cat((phase.sin(), phase.cos()), dim=-1)
        time_feature = self.time_proj(time_feature).reshape(n, horizon, self.cfg.hidden)
        spatial = (
            self.token_proj(tokens.float())
            + self.geometry_proj(geometry)
            + pose_feature[:, :, None]
            + time_feature[:, :, None]
            + self.patch_pos[:, None]
        )
        spatial = self.spatial(
            spatial.reshape(n * horizon, patches, self.cfg.hidden),
            src_key_padding_mask=~token_valid.reshape(n * horizon, patches),
        )
        spatial = self.spatial_norm(spatial).reshape(n, horizon, patches, self.cfg.hidden)
        token_weight = token_valid[..., None].to(spatial.dtype)
        frames = (spatial * token_weight).sum(dim=2) / token_weight.sum(dim=2).clamp_min(1.0)
        task = self.task_proj(task_emb.float())[:, None].expand(
            -1, candidates, -1
        ).reshape(n, 1, self.cfg.hidden)
        temporal = self.temporal(torch.cat((task, frames), dim=1))
        temporal = self.temporal_norm(temporal)
        task_summary = temporal[:, 0]
        frame_summary = temporal[:, 1:]
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
    """Calibrated score; physical action cost stays outside the planner head."""

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
