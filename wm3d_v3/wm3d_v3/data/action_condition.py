"""Helpers for building future-action conditioning tensors."""
from __future__ import annotations

import numpy as np
import torch


def make_action_condition(
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [dx,dy,dz,drx,dry,drz,grip_closed] conditioning tokens.

    Pose axes use the normalized target when available. The gripper channel is
    binarized from the canonical raw action so all callers use the same close
    convention as the action loss.
    """
    if action_tgt.shape[-1] < 7:
        raise ValueError(f"action_tgt must end with 7 dims, got {tuple(action_tgt.shape)}")
    pose = action_tgt[..., :6] if action_tgt_norm is None else action_tgt_norm
    if pose.shape != action_tgt[..., :6].shape:
        raise ValueError(
            f"action_tgt_norm shape {tuple(pose.shape)} must match action pose "
            f"{tuple(action_tgt[..., :6].shape)}"
        )
    pose = pose.to(device=action_tgt.device, dtype=action_tgt.dtype)
    grip = (action_tgt[..., 6:7] > 0.5).to(dtype=pose.dtype)
    return torch.cat([pose, grip], dim=-1)


def make_action_condition_np(
    actions: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Numpy equivalent for rollout scripts reading raw cached actions."""
    arr = np.asarray(actions, dtype=np.float32)
    if arr.shape[-1] < 7:
        raise ValueError(f"actions must end with 7 dims, got {arr.shape}")
    pose = arr[..., :6]
    if mean is not None and std is not None:
        pose = (pose - mean[:6].astype(np.float32)) / np.maximum(std[:6].astype(np.float32), 1e-6)
    grip = (arr[..., 6:7] > 0.5).astype(np.float32)
    return np.concatenate([pose.astype(np.float32), grip], axis=-1)
