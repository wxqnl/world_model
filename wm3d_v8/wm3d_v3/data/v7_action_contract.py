"""Canonical action conversion and per-source audit helpers for WM3D-v7."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .v7_contracts import CANONICAL_ACTION_VERSION, MODEL_HZ


@dataclass(frozen=True)
class ActionAdapter:
    source: str
    source_frame: str
    translation_unit_scale: float | tuple[float, float, float]
    rotation_unit_scale: float | tuple[float, float, float]
    rotation_repr: str
    gripper_index: int = 6
    gripper_open_value: float = -1.0
    gripper_closed_value: float = 1.0
    is_delta: bool = True
    nominal_hz: float = MODEL_HZ
    base_from_source_rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    adapter_version: str = CANONICAL_ACTION_VERSION


@dataclass(frozen=True)
class ActionAuditThresholds:
    max_translation_p99_m: float = 0.15
    max_rotation_p99_rad: float = 1.2
    hz_relative_tolerance: float = 0.20
    min_direction_cosine: float = 0.25
    max_scale_log_error: float = 1.5
    min_lag_peak_margin: float = 0.03


@dataclass(frozen=True)
class ActionAuditReport:
    source: str
    adapter_version: str
    sample_count: int
    finite_fraction: float
    translation_p99_m: float
    rotation_p99_rad: float
    observed_hz: float | None
    direction_cosine: float | None
    scale_log_error: float | None
    best_lag_steps: int | None
    lag_peak_margin: float | None
    passed: bool
    failures: tuple[str, ...]

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


def _rpy_to_axis_angle(rpy: np.ndarray) -> np.ndarray:
    """Convert batched XYZ Euler deltas to rotation vectors without scipy."""
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    xyz = np.stack((qx, qy, qz), axis=-1)
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(qw[..., None], -1.0, 1.0))
    axis = xyz / np.maximum(norm, 1e-8)
    return np.where(norm > 1e-8, axis * angle, 2.0 * xyz).astype(np.float32)


def canonicalize_dense_action(raw_action: np.ndarray, adapter: ActionAdapter) -> np.ndarray:
    """Map a source dense action to base-frame delta axis-angle + close gripper."""
    raw = np.asarray(raw_action, dtype=np.float32)
    if raw.shape[-1] < 7:
        raise ValueError(f"expected at least 7 action dimensions, got {raw.shape}")
    if not adapter.is_delta:
        raise ValueError("absolute actions require an explicit state-aware adapter")
    rotation_scale = np.asarray(adapter.rotation_unit_scale, dtype=np.float32)
    translation_scale = np.asarray(adapter.translation_unit_scale, dtype=np.float32)
    if rotation_scale.ndim > 1 or translation_scale.ndim > 1:
        raise ValueError("action scales must be scalars or length-3 vectors")
    rotation = raw[..., 3:6] * rotation_scale
    if adapter.rotation_repr == "rpy":
        rotation = _rpy_to_axis_angle(rotation)
    elif adapter.rotation_repr != "axis_angle":
        raise ValueError(f"unsupported rotation representation: {adapter.rotation_repr}")
    frame_rotation = np.asarray(adapter.base_from_source_rotation, dtype=np.float32)
    if frame_rotation.shape != (3, 3):
        raise ValueError("base_from_source_rotation must be 3x3")
    translation = (raw[..., :3] * translation_scale) @ frame_rotation.T
    rotation = rotation @ frame_rotation.T
    gripper_raw = raw[..., adapter.gripper_index]
    denominator = adapter.gripper_closed_value - adapter.gripper_open_value
    if abs(denominator) < 1e-8:
        raise ValueError("gripper open and closed values must differ")
    close01 = (gripper_raw - adapter.gripper_open_value) / denominator
    close_signed = np.clip(close01, 0.0, 1.0) * 2.0 - 1.0
    return np.concatenate((translation, rotation, close_signed[..., None]), axis=-1).astype(np.float32)


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.where(angle > 1e-8, np.sin(half) / np.maximum(angle, 1e-8), 0.5)
    xyz = rotvec * scale
    return np.concatenate((np.cos(half), xyz), axis=-1)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def _quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    q = quaternion / np.maximum(np.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    q = np.where(q[..., :1] < 0, -q, q)
    xyz = q[..., 1:]
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(q[..., :1], -1.0, 1.0))
    axis = np.zeros_like(xyz)
    np.divide(xyz, norm, out=axis, where=norm > 1e-8)
    return np.where(norm > 1e-8, axis * angle, 2.0 * xyz).astype(np.float32)


def resample_canonical_actions(
    actions: np.ndarray,
    *,
    source_hz: float,
    target_hz: float = MODEL_HZ,
) -> np.ndarray:
    """Compose consecutive base-frame deltas when downsampling to model rate."""
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("actions must be [N,7]")
    ratio = source_hz / target_hz
    group = int(round(ratio))
    if group < 1 or abs(ratio - group) > 1e-6:
        raise ValueError("source_hz must be an integer multiple of target_hz")
    count = len(values) // group
    output = np.empty((count, 7), dtype=np.float32)
    for index in range(count):
        chunk = values[index * group : (index + 1) * group]
        output[index, :3] = chunk[:, :3].sum(axis=0)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for delta in chunk[:, 3:6]:
            quaternion = _quaternion_multiply(quaternion, _rotvec_to_quaternion(delta))
        output[index, 3:6] = _quaternion_to_rotvec(quaternion)
        output[index, 6] = chunk[-1, 6]
    return output


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 4 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("-inf")
    return float(np.corrcoef(left, right)[0, 1])


def audit_canonical_actions(
    actions: np.ndarray,
    *,
    source: str,
    adapter: ActionAdapter,
    timestamps: np.ndarray | None = None,
    observed_ee_deltas: np.ndarray | None = None,
    thresholds: ActionAuditThresholds | None = None,
) -> ActionAuditReport:
    """Audit scale, sign, control rate and action/observation temporal alignment."""
    limits = thresholds or ActionAuditThresholds()
    values = np.asarray(actions, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"canonical actions must be [N,7], got {values.shape}")
    failures: list[str] = []
    finite_fraction = float(np.isfinite(values).mean()) if values.size else 0.0
    if finite_fraction != 1.0 or len(values) < 8:
        failures.append("finite_and_sample_count")
    translation_p99 = float(np.nanpercentile(np.linalg.norm(values[:, :3], axis=1), 99))
    rotation_p99 = float(np.nanpercentile(np.linalg.norm(values[:, 3:6], axis=1), 99))
    if translation_p99 > limits.max_translation_p99_m:
        failures.append("translation_scale")
    if rotation_p99 > limits.max_rotation_p99_rad:
        failures.append("rotation_scale")
    if np.nanmin(values[:, 6]) < -1.001 or np.nanmax(values[:, 6]) > 1.001:
        failures.append("gripper_range")

    observed_hz: float | None = None
    if timestamps is not None:
        time_values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if len(time_values) != len(values) or np.any(np.diff(time_values) <= 0):
            failures.append("timestamps")
        else:
            observed_hz = float(1.0 / np.median(np.diff(time_values)))
            relative_error = abs(observed_hz - adapter.nominal_hz) / adapter.nominal_hz
            if relative_error > limits.hz_relative_tolerance:
                failures.append("control_hz")

    direction_cosine: float | None = None
    scale_log_error: float | None = None
    best_lag: int | None = None
    lag_margin: float | None = None
    if observed_ee_deltas is not None:
        ee = np.asarray(observed_ee_deltas, dtype=np.float32)
        if ee.ndim != 2 or ee.shape != (len(values), 6):
            failures.append("ee_delta_shape")
        else:
            action_motion = values[:, :6]
            action_norm = np.linalg.norm(action_motion, axis=1)
            ee_norm = np.linalg.norm(ee, axis=1)
            correlations: dict[int, float] = {}
            for lag in range(-2, 3):
                if lag < 0:
                    aligned_action_norm, aligned_ee_norm = action_norm[-lag:], ee_norm[:lag]
                elif lag > 0:
                    aligned_action_norm, aligned_ee_norm = action_norm[:-lag], ee_norm[lag:]
                else:
                    aligned_action_norm, aligned_ee_norm = action_norm, ee_norm
                # Raw robot actions are strongly autocorrelated.  Differencing
                # makes the lag test identify command changes rather than the
                # width of a long constant-action plateau.
                correlations[lag] = _safe_corr(
                    np.diff(aligned_action_norm),
                    np.diff(aligned_ee_norm),
                )
            ordered = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
            best_lag, best_score = ordered[0]
            lag_margin = float(best_score - ordered[1][1])
            if best_lag not in (-1, 0, 1) or lag_margin < limits.min_lag_peak_margin:
                failures.append("action_observation_lag")
            if best_lag < 0:
                aligned_action, aligned_ee = action_motion[-best_lag:], ee[:best_lag]
            elif best_lag > 0:
                aligned_action, aligned_ee = action_motion[:-best_lag], ee[best_lag:]
            else:
                aligned_action, aligned_ee = action_motion, ee
            norms = np.linalg.norm(aligned_action, axis=1) * np.linalg.norm(aligned_ee, axis=1)
            valid = norms > 1e-8
            if np.any(valid):
                direction_cosine = float(
                    np.median(np.sum(aligned_action[valid] * aligned_ee[valid], axis=1) / norms[valid])
                )
                ratios = np.linalg.norm(aligned_ee[valid], axis=1) / np.maximum(
                    np.linalg.norm(aligned_action[valid], axis=1), 1e-8
                )
                scale_log_error = float(abs(np.median(np.log(np.maximum(ratios, 1e-8)))))
                if direction_cosine < limits.min_direction_cosine:
                    failures.append("action_direction")
                if scale_log_error > limits.max_scale_log_error:
                    failures.append("action_ee_scale")
            else:
                failures.append("no_motion_for_alignment")

    return ActionAuditReport(
        source=source,
        adapter_version=adapter.adapter_version,
        sample_count=len(values),
        finite_fraction=finite_fraction,
        translation_p99_m=translation_p99,
        rotation_p99_rad=rotation_p99,
        observed_hz=observed_hz,
        direction_cosine=direction_cosine,
        scale_log_error=scale_log_error,
        best_lag_steps=best_lag,
        lag_peak_margin=lag_margin,
        passed=not failures,
        failures=tuple(failures),
    )
