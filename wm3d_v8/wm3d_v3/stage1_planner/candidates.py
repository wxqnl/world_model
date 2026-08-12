"""Physical candidate cost kept outside the V8 planner head."""
from __future__ import annotations

import torch


def deterministic_action_cost(actions: torch.Tensor) -> torch.Tensor:
    """Return a deterministic physical-action cost with shape ``[B,C]``."""

    if actions.ndim != 4 or actions.shape[-1] != 7:
        raise ValueError("physical candidate actions must be [B,C,H,7]")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("physical candidate actions contain non-finite values")
    pose = actions[..., :6].float()
    grip = actions[..., 6].float()
    magnitude = pose.square().mean(dim=(-1, -2)).sqrt()
    if pose.shape[2] > 1:
        jerk = (pose[:, :, 1:] - pose[:, :, :-1]).square().mean(
            dim=(-1, -2)
        ).sqrt()
        grip_events = (grip[:, :, 1:] - grip[:, :, :-1]).abs().mean(dim=-1)
    else:
        jerk = torch.zeros_like(magnitude)
        grip_events = torch.zeros_like(magnitude)
    return magnitude + 0.25 * jerk + 0.05 * grip_events
