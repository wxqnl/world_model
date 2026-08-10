"""WM3D-V8 dual-rate action contract.

The native world model remains 5 Hz.  The policy lane predicts eight real
20 Hz controller commands.  Sources that only expose audited 5 Hz interval
actions never receive fabricated 20 Hz labels; they supervise the exact
composition of four predicted controller commands instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import torch

from .v7_action_contract import resample_canonical_actions


V8_DUAL_RATE_ACTION_SCHEMA = "wm3d_v8_dual_rate_action_v1"
V8_ACTION_SIDECAR_SCHEMA = "wm3d_v8_action20_sidecar_v1"
V8_ACTION_SIDECAR_INDEX_SCHEMA = "wm3d_v8_action20_sidecar_index_v1"
V8_ACTION_STATS_SCHEMA = "wm3d_v8_action20_stats_v1"
V8_POLICY_HISTORY_SCHEMA = "wm3d_v8_action_history_20hz_dt_valid_v1"
V8_ACTION_NORMALIZER_SCHEMA = "wm3d_v8_source_bound_pose_normalizer_v1"

WORLD_HZ = 5
POLICY_HZ = 20
WORLD_DT_SECONDS = 1.0 / WORLD_HZ
POLICY_DT_SECONDS = 1.0 / POLICY_HZ
SUBSTEPS_PER_WORLD = POLICY_HZ // WORLD_HZ
ACTION_DIM = 7
POSE_DIM = 6
POLICY_HORIZON = 8
POLICY_CHUNK_SECONDS = POLICY_HORIZON / POLICY_HZ
POLICY_HISTORY_LEN = 16
POLICY_HISTORY_DIM = 9
DYNAMICS_ACTION_DIM = (
    SUBSTEPS_PER_WORLD * ACTION_DIM
    + SUBSTEPS_PER_WORLD
    + SUBSTEPS_PER_WORLD
)
LOWER_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class V8ActionContractError(ValueError):
    """Raised when data cannot satisfy the V8 physical action contract."""


def require_v8_pinned_file(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
    chunk_bytes: int = 8 << 20,
) -> Path:
    """Resolve a regular file and bind the bytes to a sealed SHA256.

    Preflight is not a substitute for runtime identity: files can change
    between the preflight report and dataset construction.  V8 loaders call
    this function again before reading any action normalizer.
    """

    source = Path(path)
    if source.is_symlink():
        raise V8ActionContractError(f"{label} must not be a symlink: {source}")
    try:
        resolved = source.resolve(strict=True)
    except FileNotFoundError as exc:
        raise V8ActionContractError(f"{label} is missing: {source}") from exc
    if not resolved.is_file():
        raise V8ActionContractError(f"{label} is not a regular file: {resolved}")
    expected = str(expected_sha256 or "")
    if LOWER_HEX64.fullmatch(expected) is None:
        raise V8ActionContractError(
            f"{label} expected SHA256 is missing/invalid: {expected!r}"
        )
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise V8ActionContractError(
            f"{label} SHA256 mismatch: observed={observed} expected={expected}"
        )
    return resolved


@dataclass(frozen=True)
class PoseStats:
    mean: np.ndarray
    std: np.ndarray
    key: str

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        if mean.shape != (POSE_DIM,) or std.shape != (POSE_DIM,):
            raise V8ActionContractError(
                f"pose stats must be [6], got mean={mean.shape} std={std.shape}"
            )
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise V8ActionContractError("pose stats contain non-finite values")
        if np.any(std <= 0.0):
            raise V8ActionContractError("pose std must be strictly positive")
        if not str(self.key):
            raise V8ActionContractError("pose stats key must be non-empty")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def normalize(self, pose: np.ndarray) -> np.ndarray:
        values = np.asarray(pose, dtype=np.float32)
        if values.shape[-1] != POSE_DIM or not np.isfinite(values).all():
            raise V8ActionContractError(
                f"pose values must be finite [...,6], got {values.shape}"
            )
        return ((values - self.mean) / self.std).astype(np.float32)


def _require_actions(values: np.ndarray, *, label: str) -> np.ndarray:
    actions = np.asarray(values, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise V8ActionContractError(f"{label} must be [N,7], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise V8ActionContractError(f"{label} contains non-finite values")
    if actions.shape[0] <= 0:
        raise V8ActionContractError(f"{label} must not be empty")
    if np.any(actions[:, 6] < -1.0001) or np.any(actions[:, 6] > 1.0001):
        raise V8ActionContractError(f"{label} signed gripper is outside [-1,1]")
    return actions


def signed_grip_to_close01(values: np.ndarray) -> np.ndarray:
    grip = np.asarray(values, dtype=np.float32)
    if not np.isfinite(grip).all():
        raise V8ActionContractError("gripper contains non-finite values")
    if np.any(grip < -1.0001) or np.any(grip > 1.0001):
        raise V8ActionContractError("signed gripper is outside [-1,1]")
    return np.clip((grip + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def compose_base_delta_actions_np(actions: np.ndarray) -> np.ndarray:
    """Compose one or more fixed-size groups using the promoted V7 SO(3) rule.

    ``actions`` is ``[..., S, 7]``.  Translation is base-frame and therefore
    sums; rotation deltas are multiplied in temporal order; gripper is the
    final absolute state.  This is intentionally the same implementation used
    to produce the existing 5 Hz RoboCasa cache.
    """

    values = np.asarray(actions, dtype=np.float32)
    if values.ndim < 2 or values.shape[-1] != ACTION_DIM or values.shape[-2] <= 0:
        raise V8ActionContractError(
            f"actions to compose must be [...,S,7], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise V8ActionContractError("actions to compose contain non-finite values")
    group = int(values.shape[-2])
    flat = values.reshape(-1, group, ACTION_DIM)
    result = np.empty((flat.shape[0], ACTION_DIM), dtype=np.float32)
    for index, chunk in enumerate(flat):
        result[index] = resample_canonical_actions(
            chunk,
            source_hz=float(group),
            target_hz=1.0,
        )[0]
    return result.reshape(values.shape[:-2] + (ACTION_DIM,))


def _rotvec_to_quaternion_torch(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half) / angle.clamp_min(1.0e-8),
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half), rotvec * scale), dim=-1)


def _quaternion_multiply_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternion_to_rotvec_torch(quaternion: torch.Tensor) -> torch.Tensor:
    q = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    xyz = q[..., 1:]
    norm = torch.linalg.vector_norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, q[..., :1].clamp(-1.0, 1.0))
    scale = torch.where(
        norm > 1.0e-8,
        angle / norm.clamp_min(1.0e-8),
        2.0 + norm.square() / 3.0,
    )
    return xyz * scale


def compose_base_delta_actions_torch(
    actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Differentiably compose ``[..., S, 7]`` controller commands.

    Invalid slots are identities.  At least one valid slot is required for
    every group.  The final valid slot owns the absolute gripper state.
    """

    if actions.ndim < 2 or actions.shape[-1] != ACTION_DIM or actions.shape[-2] <= 0:
        raise V8ActionContractError(
            f"torch actions must be [...,S,7], got {tuple(actions.shape)}"
        )
    if not torch.isfinite(actions).all():
        raise V8ActionContractError("torch actions contain non-finite values")
    if valid_mask is None:
        valid = torch.ones(
            actions.shape[:-1], dtype=torch.bool, device=actions.device
        )
    else:
        valid = valid_mask.to(device=actions.device, dtype=torch.bool)
        if valid.shape != actions.shape[:-1]:
            raise V8ActionContractError(
                f"valid mask {tuple(valid.shape)} does not match {tuple(actions.shape[:-1])}"
            )
    if not bool(valid.any(dim=-1).all().detach().cpu()):
        raise V8ActionContractError("every action group needs at least one valid slot")

    translation = (
        actions[..., :3] * valid.to(dtype=actions.dtype).unsqueeze(-1)
    ).sum(dim=-2)
    identity = actions.new_zeros(actions.shape[:-2] + (4,))
    identity[..., 0] = 1.0
    quaternion = identity
    for substep in range(int(actions.shape[-2])):
        delta = _rotvec_to_quaternion_torch(actions[..., substep, 3:6])
        delta = torch.where(valid[..., substep, None], delta, identity)
        quaternion = _quaternion_multiply_torch(quaternion, delta)
    rotation = _quaternion_to_rotvec_torch(quaternion)

    indices = torch.arange(
        int(actions.shape[-2]), device=actions.device, dtype=torch.long
    )
    last = torch.where(valid, indices, indices.new_full((), -1)).amax(dim=-1)
    grip = torch.gather(
        actions[..., 6], dim=-1, index=last.unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat((translation, rotation, grip.unsqueeze(-1)), dim=-1)


def pack_dynamics_action_condition(
    normalized_actions: np.ndarray,
    valid_mask: np.ndarray,
    dt_seconds: np.ndarray,
) -> np.ndarray:
    """Pack ``[K,4,7] + [K,4] + [K,4]`` into the exact 36D lane."""

    actions = np.asarray(normalized_actions, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    dt = np.asarray(dt_seconds, dtype=np.float32)
    expected_actions = (actions.shape[0], SUBSTEPS_PER_WORLD, ACTION_DIM)
    if actions.shape != expected_actions:
        raise V8ActionContractError(
            f"normalized dynamics actions must be [K,4,7], got {actions.shape}"
        )
    if valid.shape != actions.shape[:2] or dt.shape != actions.shape[:2]:
        raise V8ActionContractError(
            f"valid/dt shapes must be {actions.shape[:2]}, got {valid.shape}/{dt.shape}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(dt).all():
        raise V8ActionContractError("dynamics condition contains non-finite values")
    if np.any(dt < 0.0) or np.any(dt[~valid] != 0.0):
        raise V8ActionContractError("invalid dynamics dt/valid combination")
    if not valid.any(axis=1).all():
        raise V8ActionContractError("every 5 Hz interval needs a real action token")
    masked_actions = np.where(valid[..., None], actions, 0.0)
    packed = np.concatenate(
        (
            masked_actions.reshape(actions.shape[0], -1),
            valid.astype(np.float32),
            dt,
        ),
        axis=1,
    ).astype(np.float32)
    if packed.shape[1] != DYNAMICS_ACTION_DIM:
        raise AssertionError(packed.shape)
    return packed


def _policy_target(
    actions_signed: np.ndarray,
    stats: PoseStats,
) -> tuple[np.ndarray, np.ndarray]:
    physical = np.asarray(actions_signed, dtype=np.float32).copy()
    physical[:, 6] = signed_grip_to_close01(physical[:, 6])
    return physical, stats.normalize(actions_signed[:, :POSE_DIM])


def _fine_history(
    fine_actions: np.ndarray,
    end_exclusive: int,
) -> np.ndarray:
    start = int(end_exclusive) - POLICY_HISTORY_LEN
    if start < 0:
        raise V8ActionContractError("real 20 Hz action history is too short")
    history_actions = np.asarray(
        fine_actions[start:end_exclusive], dtype=np.float32
    ).copy()
    if history_actions.shape != (POLICY_HISTORY_LEN, ACTION_DIM):
        raise V8ActionContractError(
            f"real 20 Hz history shape mismatch: {history_actions.shape}"
        )
    history_actions[:, 6] = signed_grip_to_close01(history_actions[:, 6])
    return np.concatenate(
        (
            history_actions,
            np.full((POLICY_HISTORY_LEN, 1), POLICY_DT_SECONDS, np.float32),
            np.ones((POLICY_HISTORY_LEN, 1), np.float32),
        ),
        axis=1,
    )


def _coarse_history(
    coarse_actions: np.ndarray,
    end_exclusive: int,
) -> np.ndarray:
    # Four factual 5 Hz intervals cover the same 0.8 seconds as 16 policy
    # ticks.  They occupy the interval-end slots 3/7/11/15; no 20 Hz command
    # is invented for the remaining slots.
    start = int(end_exclusive) - POLICY_HISTORY_LEN // SUBSTEPS_PER_WORLD
    if start < 0:
        raise V8ActionContractError("coarse 5 Hz action history is too short")
    source = np.asarray(coarse_actions[start:end_exclusive], dtype=np.float32)
    if source.shape != (POLICY_HISTORY_LEN // SUBSTEPS_PER_WORLD, ACTION_DIM):
        raise V8ActionContractError(
            f"coarse history shape mismatch: {source.shape}"
        )
    result = np.zeros((POLICY_HISTORY_LEN, POLICY_HISTORY_DIM), dtype=np.float32)
    slots = np.arange(SUBSTEPS_PER_WORLD - 1, POLICY_HISTORY_LEN, SUBSTEPS_PER_WORLD)
    result[slots, :ACTION_DIM] = source
    result[slots, 6] = signed_grip_to_close01(source[:, 6])
    result[slots, 7] = WORLD_DT_SECONDS
    result[slots, 8] = 1.0
    return result


def _common_output(
    *,
    dynamics_action_cond: np.ndarray,
    fine_tgt: np.ndarray,
    fine_tgt_norm: np.ndarray,
    fine_valid: np.ndarray,
    coarse_tgt: np.ndarray,
    coarse_tgt_norm: np.ndarray,
    coarse_valid: np.ndarray,
    fine_stats: PoseStats,
    coarse_stats: PoseStats,
    history: np.ndarray,
    previous_grip_close01: float,
    stats_key: str,
) -> dict[str, np.ndarray | str]:
    values: Mapping[str, np.ndarray] = {
        "v8_dynamics_action_cond": np.asarray(dynamics_action_cond, np.float32),
        "policy_action_tgt": np.asarray(fine_tgt, np.float32),
        "policy_action_tgt_norm": np.asarray(fine_tgt_norm, np.float32),
        "policy_action_valid_mask": np.asarray(fine_valid, np.bool_),
        "policy_action_coarse_tgt": np.asarray(coarse_tgt, np.float32),
        "policy_action_coarse_tgt_norm": np.asarray(coarse_tgt_norm, np.float32),
        "policy_action_coarse_valid_mask": np.asarray(coarse_valid, np.bool_),
        "policy_action_pose_mean": fine_stats.mean,
        "policy_action_pose_std": fine_stats.std,
        "policy_action_coarse_pose_mean": coarse_stats.mean,
        "policy_action_coarse_pose_std": coarse_stats.std,
        "policy_action_prev_grip": np.asarray([previous_grip_close01], np.float32),
        "action_history": np.asarray(history, np.float32),
    }
    expected = {
        "v8_dynamics_action_cond": (dynamics_action_cond.shape[0], DYNAMICS_ACTION_DIM),
        "policy_action_tgt": (POLICY_HORIZON, ACTION_DIM),
        "policy_action_tgt_norm": (POLICY_HORIZON, POSE_DIM),
        "policy_action_valid_mask": (POLICY_HORIZON,),
        "policy_action_coarse_tgt": (POLICY_HORIZON // SUBSTEPS_PER_WORLD, ACTION_DIM),
        "policy_action_coarse_tgt_norm": (POLICY_HORIZON // SUBSTEPS_PER_WORLD, POSE_DIM),
        "policy_action_coarse_valid_mask": (POLICY_HORIZON // SUBSTEPS_PER_WORLD,),
        "policy_action_pose_mean": (POSE_DIM,),
        "policy_action_pose_std": (POSE_DIM,),
        "policy_action_coarse_pose_mean": (POSE_DIM,),
        "policy_action_coarse_pose_std": (POSE_DIM,),
        "policy_action_prev_grip": (1,),
        "action_history": (POLICY_HISTORY_LEN, POLICY_HISTORY_DIM),
    }
    for key, expected_shape in expected.items():
        if values[key].shape != expected_shape:
            raise V8ActionContractError(
                f"{key} shape {values[key].shape} != {expected_shape}"
            )
        if not np.isfinite(values[key]).all():
            raise V8ActionContractError(f"{key} contains non-finite values")
    return {
        **values,
        "v8_action_contract_version": V8_DUAL_RATE_ACTION_SCHEMA,
        "v8_action_stats_key": str(stats_key),
        "v8_action_history_schema": V8_POLICY_HISTORY_SCHEMA,
    }


def build_real_20hz_window_contract(
    *,
    fine_actions: np.ndarray,
    world_actions: np.ndarray,
    world_action_start: int,
    world_horizon: int,
    fine_stats: PoseStats,
    coarse_stats: PoseStats,
    composition_atol: float = 2.0e-5,
) -> dict[str, np.ndarray | str]:
    """Build one RoboCasa sample from real controller-rate commands."""

    fine = _require_actions(fine_actions, label="fine_actions")
    world = _require_actions(world_actions, label="world_actions")
    start = int(world_action_start)
    k = int(world_horizon)
    if k <= 0 or start < 0 or start + k > len(world):
        raise V8ActionContractError(
            f"world action window [{start},{start + k}) is outside {len(world)}"
        )
    fine_start = start * SUBSTEPS_PER_WORLD
    fine_stop = (start + k) * SUBSTEPS_PER_WORLD
    if fine_stop > len(fine):
        raise V8ActionContractError(
            f"fine action window [{fine_start},{fine_stop}) is outside {len(fine)}"
        )
    if fine_start < POLICY_HISTORY_LEN:
        raise V8ActionContractError(
            "real 20 Hz action window does not have the required 16-step history"
        )
    fine_window = fine[fine_start:fine_stop].reshape(
        k, SUBSTEPS_PER_WORLD, ACTION_DIM
    )
    composed = compose_base_delta_actions_np(fine_window)
    world_window = world[start : start + k]
    if not np.allclose(composed, world_window, rtol=0.0, atol=composition_atol):
        error = np.max(np.abs(composed - world_window), axis=0)
        raise V8ActionContractError(
            "real 20 Hz actions do not reproduce the sealed 5 Hz interval "
            f"actions; max_abs_by_dim={error.tolist()}"
        )

    normalized_fine = fine_stats.normalize(fine_window[..., :POSE_DIM])
    dynamics_actions = np.concatenate(
        (normalized_fine, fine_window[..., 6:7]), axis=-1
    )
    valid = np.ones((k, SUBSTEPS_PER_WORLD), dtype=np.bool_)
    dt = np.full((k, SUBSTEPS_PER_WORLD), POLICY_DT_SECONDS, dtype=np.float32)
    dynamics = pack_dynamics_action_condition(dynamics_actions, valid, dt)

    policy_signed = fine[fine_start : fine_start + POLICY_HORIZON]
    if policy_signed.shape != (POLICY_HORIZON, ACTION_DIM):
        raise V8ActionContractError("real 20 Hz policy target is shorter than C=8")
    fine_tgt, fine_tgt_norm = _policy_target(policy_signed, fine_stats)
    coarse_signed = compose_base_delta_actions_np(
        policy_signed.reshape(-1, SUBSTEPS_PER_WORLD, ACTION_DIM)
    )
    coarse_tgt, coarse_tgt_norm = _policy_target(coarse_signed, coarse_stats)
    previous = float(signed_grip_to_close01(fine[fine_start - 1 : fine_start, 6])[0])
    return _common_output(
        dynamics_action_cond=dynamics,
        fine_tgt=fine_tgt,
        fine_tgt_norm=fine_tgt_norm,
        fine_valid=np.ones(POLICY_HORIZON, dtype=np.bool_),
        coarse_tgt=coarse_tgt,
        coarse_tgt_norm=coarse_tgt_norm,
        coarse_valid=np.ones(POLICY_HORIZON // SUBSTEPS_PER_WORLD, dtype=np.bool_),
        fine_stats=fine_stats,
        coarse_stats=coarse_stats,
        history=_fine_history(fine, fine_start),
        previous_grip_close01=previous,
        stats_key=f"{fine_stats.key}|{coarse_stats.key}",
    )


def build_coarse_5hz_window_contract(
    *,
    episode_actions: np.ndarray,
    action_indices: np.ndarray | list[int] | tuple[int, ...],
    coarse_stats: PoseStats,
) -> dict[str, np.ndarray | str]:
    """Build an OXE sample without inventing controller-rate labels."""

    episode = _require_actions(episode_actions, label="episode_actions")
    indices = np.asarray(action_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) <= 0:
        raise V8ActionContractError("action_indices must be a non-empty vector")
    if np.any(np.diff(indices) != 1):
        raise V8ActionContractError("coarse action indices must be contiguous")
    if indices[0] < 0 or indices[-1] >= len(episode):
        raise V8ActionContractError("coarse action indices are outside the episode")
    if int(indices[0]) < POLICY_HISTORY_LEN // SUBSTEPS_PER_WORLD:
        raise V8ActionContractError(
            "coarse 5 Hz action window does not have the required 0.8 s history"
        )
    k = int(len(indices))
    world_window = episode[indices]
    world_norm = coarse_stats.normalize(world_window[:, :POSE_DIM])
    dynamics_actions = np.zeros(
        (k, SUBSTEPS_PER_WORLD, ACTION_DIM), dtype=np.float32
    )
    # A factual 5 Hz interval is one token with dt=0.2, not four copied 20 Hz
    # commands.  Mask/dt make the two representations distinguishable.
    dynamics_actions[:, 0, :POSE_DIM] = world_norm
    dynamics_actions[:, 0, 6] = world_window[:, 6]
    valid = np.zeros((k, SUBSTEPS_PER_WORLD), dtype=np.bool_)
    valid[:, 0] = True
    dt = np.zeros((k, SUBSTEPS_PER_WORLD), dtype=np.float32)
    dt[:, 0] = WORLD_DT_SECONDS
    dynamics = pack_dynamics_action_condition(dynamics_actions, valid, dt)

    coarse_count = POLICY_HORIZON // SUBSTEPS_PER_WORLD
    if k < coarse_count:
        raise V8ActionContractError(
            f"coarse target needs {coarse_count} intervals, got {k}"
        )
    coarse_signed = world_window[:coarse_count]
    coarse_tgt, coarse_tgt_norm = _policy_target(coarse_signed, coarse_stats)
    # The output coordinates still need a numerical normalization for
    # differentiable composition.  mean/std divided by four is an interface
    # scale only; fine_valid remains false and no pseudo fine label is trained.
    fine_stats = PoseStats(
        mean=coarse_stats.mean / SUBSTEPS_PER_WORLD,
        std=coarse_stats.std / SUBSTEPS_PER_WORLD,
        key=f"{coarse_stats.key}:composition_interface_div4",
    )
    previous_signed = episode[int(indices[0]) - 1, 6]
    previous = float(signed_grip_to_close01(np.asarray([previous_signed]))[0])
    fine_tgt = np.zeros((POLICY_HORIZON, ACTION_DIM), dtype=np.float32)
    fine_tgt[:, 6] = previous
    fine_tgt_norm = np.zeros((POLICY_HORIZON, POSE_DIM), dtype=np.float32)
    return _common_output(
        dynamics_action_cond=dynamics,
        fine_tgt=fine_tgt,
        fine_tgt_norm=fine_tgt_norm,
        fine_valid=np.zeros(POLICY_HORIZON, dtype=np.bool_),
        coarse_tgt=coarse_tgt,
        coarse_tgt_norm=coarse_tgt_norm,
        coarse_valid=np.ones(coarse_count, dtype=np.bool_),
        fine_stats=fine_stats,
        coarse_stats=coarse_stats,
        history=_coarse_history(episode, int(indices[0])),
        previous_grip_close01=previous,
        stats_key=f"{fine_stats.key}|{coarse_stats.key}",
    )


def torchify_v8_action_fields(
    fields: Mapping[str, np.ndarray | str],
) -> dict[str, torch.Tensor | str]:
    """Convert one validated NumPy contract to dataset sample values."""

    result: dict[str, torch.Tensor | str] = {}
    for key, value in fields.items():
        if isinstance(value, str):
            result[key] = value
        else:
            array = np.asarray(value)
            result[key] = torch.from_numpy(array.copy())
    return result
