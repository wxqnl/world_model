"""Protocol-locked RoboNet adapters used by the paper evaluation."""
from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def pad_two_frame_history(tokens: Tensor, native_t: int = 16) -> Tensor:
    """Repeat the first of exactly two real frames on the left to native_t."""
    if not isinstance(tokens, Tensor) or tokens.ndim < 2:
        raise ValueError("tokens must be a Tensor with a time dimension")
    if tokens.shape[1] != 2:
        raise ValueError("RoboNet history must contain exactly two real frames")
    if native_t < 2:
        raise ValueError("native_t must be at least 2")
    first = tokens[:, :1].expand(-1, native_t - 1, *([-1] * (tokens.ndim - 2)))
    return torch.cat((first, tokens[:, 1:2]), dim=1)


def robonet5_to_native7(
    actions: NDArray[np.generic],
    rotation_axis: str,
    grip_midpoint: float,
) -> NDArray[np.generic]:
    """Map [dx,dy,dz,drotation,gripper] to native canonical 7D actions."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("RoboNet actions must be a rank-2 array with width 5")
    if rotation_axis not in {"x", "y", "z"}:
        raise ValueError("rotation_axis must be one of: x, y, z")
    if not math.isfinite(float(grip_midpoint)):
        raise ValueError("grip_midpoint must be finite")
    if not np.isfinite(values).all():
        raise ValueError("RoboNet actions must contain only finite values")

    dtype = np.result_type(values.dtype, np.float32)
    out = np.zeros((values.shape[0], 7), dtype=dtype)
    out[:, :3] = values[:, :3]
    out[:, 3 + {"x": 0, "y": 1, "z": 2}[rotation_axis]] = values[:, 3]
    out[:, 6] = (values[:, 4] >= grip_midpoint).astype(dtype, copy=False)
    return out
