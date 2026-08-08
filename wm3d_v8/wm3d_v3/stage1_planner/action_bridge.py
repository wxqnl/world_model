"""Fail-closed inverse of the audited V7 canonical RoboCasa action adapter."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wm3d_v3.data.v7_action_contract import (
    ActionAdapter,
    canonicalize_dense_action,
    resample_canonical_actions,
)


@dataclass(frozen=True)
class ActionRoundTrip:
    simulator_actions: np.ndarray
    reconstructed_canonical: np.ndarray
    max_pose_abs_error: float
    max_gripper_abs_error: float


def _scale3(value: float | tuple[float, float, float], name: str) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float64)
    if scale.ndim == 0:
        scale = np.repeat(scale.reshape(1), 3)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale == 0):
        raise ValueError(f"{name} must be a finite non-zero scalar or length-3 vector")
    return scale


def canonical_model_actions_to_simulator(
    actions: np.ndarray,
    adapter: ActionAdapter,
    *,
    source_hz: float = 20.0,
    target_hz: float = 5.0,
    template: np.ndarray | None = None,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
    pose_tolerance: float = 2.0e-6,
    gripper_tolerance: float = 1.0e-6,
) -> ActionRoundTrip:
    """Convert physical V7 actions to exact dense simulator commands.

    A model-rate delta is split into identical source-rate increments.  Since
    all split rotation vectors share one axis, composing them with the audited
    downsampler recovers the original SO(3) delta.  Non-arm dimensions are
    copied from ``template``; clipping is forbidden because it would silently
    change the candidate being evaluated.
    """

    canonical = np.asarray(actions, dtype=np.float64)
    if canonical.ndim != 2 or canonical.shape[1] != 7 or not np.isfinite(canonical).all():
        raise ValueError(f"actions must be finite [H,7], got {canonical.shape}")
    if not adapter.is_delta:
        raise ValueError("Stage1-P requires a delta-action adapter")
    if adapter.rotation_repr != "axis_angle":
        raise ValueError(
            "Stage1-P inverse is intentionally restricted to audited axis-angle sources"
        )
    ratio = source_hz / target_hz
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1.0e-9:
        raise ValueError("source_hz must be an integer multiple of target_hz")

    dense_steps = canonical.shape[0] * stride
    if template is None:
        width = max(7, int(adapter.gripper_index) + 1)
        dense = np.zeros((dense_steps, width), dtype=np.float64)
    else:
        dense = np.asarray(template, dtype=np.float64).copy()
        if dense.ndim != 2 or dense.shape[0] != dense_steps:
            raise ValueError(
                f"template must be [{dense_steps},A], got {dense.shape}"
            )
        if dense.shape[1] <= adapter.gripper_index or not np.isfinite(dense).all():
            raise ValueError("template does not contain the audited gripper dimension")

    frame = np.asarray(adapter.base_from_source_rotation, dtype=np.float64)
    if frame.shape != (3, 3) or not np.allclose(frame @ frame.T, np.eye(3), atol=1.0e-6):
        raise ValueError("base_from_source_rotation must be an orthonormal 3x3 matrix")
    translation_scale = _scale3(adapter.translation_unit_scale, "translation_unit_scale")
    rotation_scale = _scale3(adapter.rotation_unit_scale, "rotation_unit_scale")

    source_translation = (canonical[:, :3] @ frame) / translation_scale
    source_rotation = (canonical[:, 3:6] @ frame) / rotation_scale
    source_translation /= float(stride)
    source_rotation /= float(stride)
    dense[:, :3] = np.repeat(source_translation, stride, axis=0)
    dense[:, 3:6] = np.repeat(source_rotation, stride, axis=0)

    close01 = np.clip((canonical[:, 6] + 1.0) * 0.5, 0.0, 1.0)
    raw_gripper = (
        float(adapter.gripper_open_value)
        + close01
        * (float(adapter.gripper_closed_value) - float(adapter.gripper_open_value))
    )
    dense[:, int(adapter.gripper_index)] = np.repeat(raw_gripper, stride)

    if (action_low is None) != (action_high is None):
        raise ValueError("action_low and action_high must be supplied together")
    if action_low is not None:
        low = np.asarray(action_low, dtype=np.float64)
        high = np.asarray(action_high, dtype=np.float64)
        if low.shape != (dense.shape[1],) or high.shape != low.shape:
            raise ValueError("simulator action bounds do not match the dense action width")
        below = dense < low[None] - 1.0e-9
        above = dense > high[None] + 1.0e-9
        if bool(below.any() or above.any()):
            bad = np.argwhere(below | above)[0].tolist()
            raise ValueError(f"candidate exceeds simulator action bounds at {bad}")

    reconstructed = resample_canonical_actions(
        canonicalize_dense_action(dense.astype(np.float32), adapter),
        source_hz=source_hz,
        target_hz=target_hz,
    ).astype(np.float64)
    pose_error = float(np.max(np.abs(reconstructed[:, :6] - canonical[:, :6])))
    grip_error = float(np.max(np.abs(reconstructed[:, 6] - canonical[:, 6])))
    if pose_error > float(pose_tolerance) or grip_error > float(gripper_tolerance):
        raise RuntimeError(
            "canonical/simulator round trip failed: "
            f"pose={pose_error:.3e} grip={grip_error:.3e}"
        )
    return ActionRoundTrip(
        simulator_actions=dense.astype(np.float32),
        reconstructed_canonical=reconstructed.astype(np.float32),
        max_pose_abs_error=pose_error,
        max_gripper_abs_error=grip_error,
    )
