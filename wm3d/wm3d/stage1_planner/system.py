"""Frozen unified Stage0 plus a trainable action-blind native-future planner."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import torch
import torch.nn as nn

from .candidates import deterministic_action_cost
from .planner_head import NativePlannerConfig, NativePlannerHead, planning_score
from .rollout import NativeRollout, _world_config, single_horizon_native_rollout


@dataclass(frozen=True)
class Stage1SystemConfig:
    planner: NativePlannerConfig = field(default_factory=NativePlannerConfig)
    horizon: int = 0
    candidate_microbatch: int = 0
    progress_weight: float = 0.5
    success_weight: float = 1.0
    risk_weight: float = 0.5
    uncertainty_weight: float = 0.25
    action_cost_weight: float = 0.05


class NativePlanningSystem(nn.Module):
    """One Stage0 owner; candidate actions never enter the learned planner."""

    def __init__(self, world: nn.Module, cfg: Stage1SystemConfig | None = None):
        super().__init__()
        self.world = world
        requested = cfg or Stage1SystemConfig()
        world_cfg = _world_config(world)
        horizon = int(requested.horizon or world_cfg.K)
        if not 0 < horizon <= int(world_cfg.K):
            raise ValueError("Stage1 horizon must lie inside the trained Stage0 K")
        planner_cfg = requested.planner
        derived = {
            "token_dim": int(world_cfg.token_dim),
            "task_dim": int(world_cfg.task_dim),
            "patches": int(world_cfg.P),
            "max_horizon": horizon,
            "num_views": int(world_cfg.num_views),
            "time_fourier_dim": int(world_cfg.time_fourier_dim),
            "time_min_period_s": float(world_cfg.time_min_period_s),
            "time_max_period_s": float(world_cfg.time_max_period_s),
        }
        for name, value in derived.items():
            current = getattr(planner_cfg, name)
            if current not in {0, value}:
                raise ValueError(f"planner {name}={current} differs from sealed Stage0 {value}")
        planner_cfg = replace(planner_cfg, **derived)
        self.cfg = replace(requested, planner=planner_cfg, horizon=horizon)
        self.planner = NativePlannerHead(planner_cfg)

    def imagine(self, batch: Mapping[str, torch.Tensor]) -> NativeRollout:
        return single_horizon_native_rollout(
            self.world,
            batch,
            horizon=self.cfg.horizon,
            candidate_microbatch=self.cfg.candidate_microbatch,
        )

    def score_rollout(
        self,
        rollout: NativeRollout,
        task_embedding: torch.Tensor,
        action_cost: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = self.planner(
            rollout.tokens,
            task_embedding,
            future_dt_s=rollout.future_dt_s,
            token_mask=rollout.token_mask,
            depth=rollout.depth,
            depth_mask=rollout.depth_mask,
            point=rollout.point,
            point_mask=rollout.point_mask,
            pose=rollout.pose,
            pose_mask=rollout.pose_mask,
            geometry_confidence=rollout.confidence,
            view_mask=rollout.view_mask,
        )
        output["score"] = planning_score(
            output,
            action_cost,
            progress_weight=self.cfg.progress_weight,
            success_weight=self.cfg.success_weight,
            risk_weight=self.cfg.risk_weight,
            uncertainty_weight=self.cfg.uncertainty_weight,
            action_cost_weight=self.cfg.action_cost_weight,
        )
        return output

    def score_observed_batch(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        horizon = self.cfg.horizon
        cost = deterministic_action_cost(
            batch["candidate_fine_action_values"][:, :, :horizon],
            batch["candidate_fine_action_mask"][:, :, :horizon],
            batch["candidate_fine_sample_mask"][:, :, :horizon],
            batch["candidate_coarse_action_values"][:, :, :horizon],
            batch["candidate_coarse_action_mask"][:, :, :horizon],
        )
        rollout = NativeRollout(
            tokens=batch["branch_future_tokens"][:, :, : self.cfg.horizon],
            future_dt_s=batch["branch_future_dt_s"][:, :, : self.cfg.horizon],
            token_mask=batch["branch_token_mask"][:, :, : self.cfg.horizon],
            depth=batch["branch_depth"][:, :, : self.cfg.horizon],
            depth_mask=batch["branch_depth_mask"][:, :, : self.cfg.horizon],
            point=batch["branch_point"][:, :, : self.cfg.horizon],
            point_mask=batch["branch_point_mask"][:, :, : self.cfg.horizon],
            pose=batch["branch_camera_pose"][:, :, : self.cfg.horizon],
            pose_mask=batch["branch_camera_pose_mask"][:, :, : self.cfg.horizon],
            confidence=batch["branch_geometry_confidence"][:, :, : self.cfg.horizon],
            view_mask=batch["branch_view_mask"][:, :, : self.cfg.horizon],
        )
        return self.score_rollout(rollout, batch["task_embedding"], cost)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, object]:
        horizon = self.cfg.horizon
        cost = deterministic_action_cost(
            batch["candidate_fine_action_values"][:, :, :horizon],
            batch["candidate_fine_action_mask"][:, :, :horizon],
            batch["candidate_fine_sample_mask"][:, :, :horizon],
            batch["candidate_coarse_action_values"][:, :, :horizon],
            batch["candidate_coarse_action_mask"][:, :, :horizon],
        )
        with torch.no_grad():
            rollout = self.imagine(batch)
        detached = NativeRollout(
            **{name: value.detach() for name, value in rollout.__dict__.items()}
        )
        return {
            "rollout": rollout,
            "action_cost": cost,
            "planner": self.score_rollout(detached, batch["task_embedding"], cost),
        }
