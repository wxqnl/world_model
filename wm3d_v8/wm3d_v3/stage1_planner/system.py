"""Isolated V7 Stage1-P system built around the unchanged native world core."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .candidates import deterministic_action_cost
from .planner_head import NativePlannerConfig, NativePlannerHead, planning_score
from .rollout import NativeRollout, multichunk_native_rollout


@dataclass(frozen=True)
class Stage1SystemConfig:
    planner: NativePlannerConfig = field(default_factory=NativePlannerConfig)
    candidate_microbatch: int = 2
    detach_between_chunks: bool = False
    activation_checkpointing: bool = True
    progress_weight: float = 0.5
    success_weight: float = 1.0
    risk_weight: float = 0.5
    uncertainty_weight: float = 0.25
    action_cost_weight: float = 0.05


class NativePlanningSystem(nn.Module):
    """World rollout plus an action-blind reader of explicit imagined futures.

    ``world`` remains a normal :class:`JointWorldModel`.  It is registered as a
    submodule so DDP sees the exact parameters selected by the phase allowlist;
    Stage0 serving heads are frozen and hash-guarded by the trainer.
    """

    def __init__(self, world: nn.Module, cfg: Stage1SystemConfig | None = None):
        super().__init__()
        self.world = world
        self.cfg = cfg or Stage1SystemConfig()
        if self.cfg.planner.token_dim != int(world.cfg.dual.state.D):
            raise ValueError("planner/world token dimensions differ")
        if self.cfg.planner.task_dim != int(world.cfg.dual.state.cond_dim):
            raise ValueError("planner/world task dimensions differ")
        if self.cfg.planner.patches != int(world.cfg.dual.state.P):
            raise ValueError("planner/world patch counts differ")
        self.planner = NativePlannerHead(self.cfg.planner)

    def fuse_context(
        self,
        context: torch.Tensor,
        *,
        wrist: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.world.fuse_views(context, wrist, view_mask=view_mask)

    def imagine(
        self,
        context: torch.Tensor,
        task_emb: torch.Tensor,
        candidate_actions: torch.Tensor,
        *,
        wrist: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
    ) -> NativeRollout:
        fused = self.fuse_context(context, wrist=wrist, view_mask=view_mask)
        return multichunk_native_rollout(
            self.world,
            fused,
            task_emb,
            candidate_actions,
            include_geometry=True,
            candidate_microbatch=self.cfg.candidate_microbatch,
            detach_between_chunks=self.cfg.detach_between_chunks,
            activation_checkpointing=self.cfg.activation_checkpointing,
        )

    def score_rollout(
        self,
        rollout: NativeRollout,
        task_emb: torch.Tensor,
        action_cost: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rollout.depth is None or rollout.point is None or rollout.pose is None:
            raise ValueError("planner requires explicit native geometry")
        outputs = self.planner(
            rollout.tokens,
            task_emb,
            depth=rollout.depth,
            point=rollout.point,
            pose=rollout.pose,
        )
        outputs["score"] = planning_score(
            outputs,
            action_cost,
            progress_weight=self.cfg.progress_weight,
            success_weight=self.cfg.success_weight,
            risk_weight=self.cfg.risk_weight,
            uncertainty_weight=self.cfg.uncertainty_weight,
            action_cost_weight=self.cfg.action_cost_weight,
        )
        return outputs

    def score_true_futures(
        self,
        future_codec: torch.Tensor,
        task_emb: torch.Tensor,
        *,
        depth: torch.Tensor,
        point: torch.Tensor,
        pose: torch.Tensor,
        action_cost: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if future_codec.ndim != 5:
            raise ValueError("true branch futures must be [B,C,H,P,Dcodec]")
        bsz, candidates, horizon, patches, codec_dim = future_codec.shape
        decoded = self.world.decode_input_tokens(
            future_codec.reshape(bsz * candidates, horizon, patches, codec_dim)
        ).reshape(bsz, candidates, horizon, patches, -1)
        return self.score_rollout(
            NativeRollout(tokens=decoded, depth=depth, point=point, pose=pose),
            task_emb,
            action_cost,
        )

    def forward(
        self,
        context: torch.Tensor,
        task_emb: torch.Tensor,
        candidate_actions: torch.Tensor,
        *,
        wrist: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        score_planner: bool = True,
        true_future_codec: torch.Tensor | None = None,
        true_depth: torch.Tensor | None = None,
        true_point: torch.Tensor | None = None,
        true_pose: torch.Tensor | None = None,
    ) -> dict[str, object]:
        rollout = self.imagine(
            context,
            task_emb,
            candidate_actions,
            wrist=wrist,
            view_mask=view_mask,
        )
        cost = deterministic_action_cost(candidate_actions)
        result: dict[str, object] = {"rollout": rollout, "action_cost": cost}
        if score_planner:
            # Planner gradients are deliberately stopped at the imagined
            # evidence boundary.  Joint calibration updates dynamics through
            # grounded reconstruction, never by making futures easier to rank.
            detached = NativeRollout(
                tokens=rollout.tokens.detach(),
                depth=rollout.depth.detach() if rollout.depth is not None else None,
                point=rollout.point.detach() if rollout.point is not None else None,
                pose=rollout.pose.detach() if rollout.pose is not None else None,
            )
            result["planner"] = self.score_rollout(detached, task_emb, cost)
        if true_future_codec is not None:
            if any(value is None for value in (true_depth, true_point, true_pose)):
                raise ValueError("true future scoring requires depth, point and pose")
            result["true_planner"] = self.score_true_futures(
                true_future_codec,
                task_emb,
                depth=true_depth,
                point=true_point,
                pose=true_pose,
                action_cost=cost,
            )
        return result
