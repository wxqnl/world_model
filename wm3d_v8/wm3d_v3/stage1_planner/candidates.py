"""Candidate construction for V7 Stage1-P.

The direct head remains the serving owner.  Flow contributes pose proposals
only; the event head owns gripper state for every stochastic pose proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_ROLES = (
    "direct",
    "flow_0",
    "flow_1",
    "flow_2",
    "flow_3",
    "grip_open",
    "grip_close",
    "arm_hold",
    "pose_reverse",
    "pose_half",
)


@dataclass(frozen=True)
class CandidateSet:
    actions: torch.Tensor
    roles: tuple[str, ...]
    direct_index: int = 0


def _validate_direct(direct: torch.Tensor) -> None:
    if direct.ndim != 3 or direct.shape[-1] != 7:
        raise ValueError(f"direct actions must be [B,H,7], got {tuple(direct.shape)}")
    if not bool(torch.isfinite(direct).all()):
        raise ValueError("direct actions contain non-finite values")


def build_candidate_set(
    direct: torch.Tensor,
    flow_pose: torch.Tensor,
    *,
    roles: Sequence[str] = DEFAULT_ROLES,
    pose_clip: float = 4.0,
) -> CandidateSet:
    """Build deterministic roles from direct pose/gripper and pose-only flow.

    ``direct`` and ``flow_pose`` are in the world model's normalized action
    space.  Gripper values in the returned action condition are close
    probabilities in ``[0,1]``.  No candidate can replace the direct serving
    head outside the planner selection path.
    """

    _validate_direct(direct)
    if flow_pose.ndim != 4 or flow_pose.shape[0] != direct.shape[0] or flow_pose.shape[2] != direct.shape[1] or flow_pose.shape[3] != 6:
        raise ValueError(
            f"flow_pose must be [B,N,H,6] aligned with direct, got {tuple(flow_pose.shape)}"
        )
    if not bool(torch.isfinite(flow_pose).all()):
        raise ValueError("flow pose candidates contain non-finite values")
    if pose_clip <= 0:
        raise ValueError("pose_clip must be positive")

    role_tuple = tuple(str(role) for role in roles)
    if not role_tuple or role_tuple[0] != "direct" or len(set(role_tuple)) != len(role_tuple):
        raise ValueError("candidate roles must be unique and start with direct")
    result: list[torch.Tensor] = []
    base_pose = direct[..., :6].clamp(-pose_clip, pose_clip)
    base_grip = direct[..., 6:7].clamp(0.0, 1.0)
    for role in role_tuple:
        if role == "direct":
            candidate = torch.cat((base_pose, base_grip), dim=-1)
        elif role.startswith("flow_"):
            index = int(role.split("_", 1)[1])
            if index >= flow_pose.shape[1]:
                raise ValueError(f"role {role} requires flow sample {index}")
            candidate = torch.cat(
                (flow_pose[:, index].clamp(-pose_clip, pose_clip), base_grip), dim=-1
            )
        elif role == "grip_open":
            candidate = torch.cat((base_pose, torch.zeros_like(base_grip)), dim=-1)
        elif role == "grip_close":
            candidate = torch.cat((base_pose, torch.ones_like(base_grip)), dim=-1)
        elif role == "arm_hold":
            candidate = torch.cat((torch.zeros_like(base_pose), base_grip), dim=-1)
        elif role == "pose_reverse":
            candidate = torch.cat((-base_pose, base_grip), dim=-1)
        elif role == "pose_half":
            candidate = torch.cat((0.5 * base_pose, base_grip), dim=-1)
        else:
            raise ValueError(f"unsupported candidate role: {role}")
        result.append(candidate)
    return CandidateSet(actions=torch.stack(result, dim=1), roles=role_tuple)


def deterministic_action_cost(actions: torch.Tensor) -> torch.Tensor:
    """Candidate cost used outside the planner head; shape ``[B,C]``."""

    if actions.ndim != 4 or actions.shape[-1] != 7:
        raise ValueError("candidate actions must be [B,C,H,7]")
    pose = actions[..., :6].float()
    grip = actions[..., 6].float()
    magnitude = pose.square().mean(dim=(-1, -2)).sqrt()
    if pose.shape[2] > 1:
        jerk = (pose[:, :, 1:] - pose[:, :, :-1]).square().mean(dim=(-1, -2)).sqrt()
        grip_events = (grip[:, :, 1:] - grip[:, :, :-1]).abs().mean(dim=-1)
    else:
        jerk = torch.zeros_like(magnitude)
        grip_events = torch.zeros_like(magnitude)
    return magnitude + 0.25 * jerk + 0.05 * grip_events
