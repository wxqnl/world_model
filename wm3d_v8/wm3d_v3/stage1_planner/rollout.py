"""Autoregressive native WM3D rollout over K=8 core chunks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass
class NativeRollout:
    tokens: torch.Tensor
    depth: torch.Tensor | None
    point: torch.Tensor | None
    pose: torch.Tensor | None


def _reshape_candidate(value: torch.Tensor, bsz: int, candidates: int) -> torch.Tensor:
    return value.reshape(bsz, candidates, *value.shape[1:])


def multichunk_native_rollout(
    world,
    context_tokens: torch.Tensor,
    task_emb: torch.Tensor,
    candidate_actions: torch.Tensor,
    *,
    include_geometry: bool = True,
    candidate_microbatch: int = 0,
    detach_between_chunks: bool = False,
    activation_checkpointing: bool = False,
) -> NativeRollout:
    """Roll H=8/16/32 in native token/geometry space.

    The world model core still predicts explicit VGGT tokens and its native
    depth/point/pose heads.  No latent-3D replacement or future observation is
    accepted.  Candidate microbatching changes memory use only, not semantics.
    """

    if context_tokens.ndim != 4 or task_emb.ndim != 2 or candidate_actions.ndim != 4:
        raise ValueError("expected context[B,T,P,D], task[B,D], actions[B,C,H,7]")
    bsz, candidates, horizon, action_dim = candidate_actions.shape
    if bsz != context_tokens.shape[0] or bsz != task_emb.shape[0]:
        raise ValueError("rollout batch dimensions do not match")
    if action_dim != int(world.cfg.dual.state.action_cond_dim):
        raise ValueError("candidate action dimension differs from native core contract")
    core_horizon = int(world.cfg.dual.state.k)
    if horizon <= 0 or horizon % core_horizon:
        raise ValueError(f"planning horizon must be a positive multiple of K={core_horizon}")
    if candidates < 2:
        raise ValueError("planning requires at least two candidates")
    micro = candidates if candidate_microbatch <= 0 else int(candidate_microbatch)
    if micro <= 0:
        raise ValueError("candidate_microbatch must be non-negative")

    token_slices: list[torch.Tensor] = []
    depth_slices: list[torch.Tensor] = []
    point_slices: list[torch.Tensor] = []
    pose_slices: list[torch.Tensor] = []
    for candidate_start in range(0, candidates, micro):
        candidate_stop = min(candidates, candidate_start + micro)
        local_candidates = candidate_stop - candidate_start
        local_context = context_tokens[:, None].expand(
            -1, local_candidates, -1, -1, -1
        ).reshape(bsz * local_candidates, *context_tokens.shape[1:])
        local_task = task_emb[:, None].expand(-1, local_candidates, -1).reshape(
            bsz * local_candidates, -1
        )
        local_actions = candidate_actions[:, candidate_start:candidate_stop].reshape(
            bsz * local_candidates, horizon, action_dim
        )
        chunk_tokens: list[torch.Tensor] = []
        chunk_depth: list[torch.Tensor] = []
        chunk_point: list[torch.Tensor] = []
        chunk_pose: list[torch.Tensor] = []
        for start in range(0, horizon, core_horizon):
            action_chunk = local_actions[:, start : start + core_horizon]
            if activation_checkpointing and torch.is_grad_enabled():
                predicted = checkpoint(
                    lambda state, task, action: world.dual(
                        state, task, action_cond=action
                    )["pred_tokens"],
                    local_context,
                    local_task,
                    action_chunk,
                    use_reentrant=False,
                )
            else:
                predicted = world.dual(
                    local_context, local_task, action_cond=action_chunk
                )["pred_tokens"]
            if predicted.shape[1] != core_horizon:
                raise RuntimeError("native core returned an unexpected rollout horizon")
            chunk_tokens.append(predicted)
            if include_geometry:
                geometry = world.geom(predicted)
                chunk_depth.append(geometry["depth"])
                if "point" not in geometry or "pose" not in geometry:
                    raise RuntimeError("Stage1-P requires native depth, point and pose heads")
                chunk_point.append(geometry["point"])
                chunk_pose.append(geometry["pose"])
            appended = predicted.detach() if detach_between_chunks else predicted
            local_context = torch.cat((local_context, appended), dim=1)[
                :, -int(world.cfg.dual.state.T) :
            ]
        token_slices.append(
            _reshape_candidate(torch.cat(chunk_tokens, dim=1), bsz, local_candidates)
        )
        if include_geometry:
            depth_slices.append(
                _reshape_candidate(torch.cat(chunk_depth, dim=1), bsz, local_candidates)
            )
            point_slices.append(
                _reshape_candidate(torch.cat(chunk_point, dim=1), bsz, local_candidates)
            )
            pose_slices.append(
                _reshape_candidate(torch.cat(chunk_pose, dim=1), bsz, local_candidates)
            )
    return NativeRollout(
        tokens=torch.cat(token_slices, dim=1),
        depth=torch.cat(depth_slices, dim=1) if depth_slices else None,
        point=torch.cat(point_slices, dim=1) if point_slices else None,
        pose=torch.cat(pose_slices, dim=1) if pose_slices else None,
    )
