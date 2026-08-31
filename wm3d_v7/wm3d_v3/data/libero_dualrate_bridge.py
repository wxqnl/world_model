"""Audited LIBERO 20 Hz controller <-> WM3D-v7 5 Hz action bridge.

The native WM3D action is an observed base-frame end-effector effect, not a
renamed LIBERO controller command.  Four observed 20 Hz effects are composed
into one canonical 5 Hz action.  A train-split linear adapter is retained only
for initializing the inverse servo path and for contract auditing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


BRIDGE_SCHEMA = "wm3dv7_libero_dualrate_eef_native_multiview_postaction_v3"
RAW_HZ = 20
WORLD_HZ = 5
RAW_PER_WORLD = RAW_HZ // WORLD_HZ
CONTEXT_WORLD_FRAMES = 16
CONTEXT_RAW_SPAN = (CONTEXT_WORLD_FRAMES - 1) * RAW_PER_WORLD
WORLD_HORIZON = 8
SERVO_HORIZON = 8
FUTURE_RAW_ACTIONS = WORLD_HORIZON * RAW_PER_WORLD
# LIBERO's HDF5 exporter calls ``env.step(action[j])`` and then stores the
# returned observation at row ``j`` beside ``action[j]``.  Consequently an
# HDF5 observation row is post-action: obs[t] is the state before action[t+1].
# Keeping this offset explicit prevents a smooth-trajectory correlation from
# disguising an off-by-one policy target.
HDF5_NEXT_ACTION_OFFSET = 1


@dataclass(frozen=True)
class AdapterFit:
    translation_matrix: np.ndarray
    translation_previous_matrix: np.ndarray
    translation_bias: np.ndarray
    rotation_matrix: np.ndarray
    rotation_previous_matrix: np.ndarray
    rotation_bias: np.ndarray

    def raw_to_physical(
        self, raw_pose: np.ndarray, previous_raw_pose: np.ndarray
    ) -> np.ndarray:
        raw = np.asarray(raw_pose, dtype=np.float64)
        previous = np.asarray(previous_raw_pose, dtype=np.float64)
        if raw.shape != previous.shape:
            raise ValueError("raw and previous raw poses must have matching shapes")
        translation = (
            raw[..., :3] @ self.translation_matrix
            + previous[..., :3] @ self.translation_previous_matrix
            + self.translation_bias
        )
        rotation = (
            raw[..., 3:6] @ self.rotation_matrix
            + previous[..., 3:6] @ self.rotation_previous_matrix
            + self.rotation_bias
        )
        return np.concatenate((translation, rotation), axis=-1).astype(np.float32)

    def physical_to_raw(
        self, physical_pose: np.ndarray, previous_raw_pose: np.ndarray
    ) -> np.ndarray:
        physical = np.asarray(physical_pose, dtype=np.float64)
        previous = np.asarray(previous_raw_pose, dtype=np.float64)
        if physical.shape != previous.shape:
            raise ValueError("physical and previous raw poses must have matching shapes")
        translation = (
            physical[..., :3]
            - previous[..., :3] @ self.translation_previous_matrix
            - self.translation_bias
        ) @ np.linalg.pinv(self.translation_matrix)
        rotation = (
            physical[..., 3:6]
            - previous[..., 3:6] @ self.rotation_previous_matrix
            - self.rotation_bias
        ) @ np.linalg.pinv(self.rotation_matrix)
        return np.concatenate((translation, rotation), axis=-1).astype(np.float32)

    def save(self, path: str | Path, *, report_sha256: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            schema=np.asarray(BRIDGE_SCHEMA),
            raw_hz=np.asarray(RAW_HZ, dtype=np.int64),
            world_hz=np.asarray(WORLD_HZ, dtype=np.int64),
            raw_per_world=np.asarray(RAW_PER_WORLD, dtype=np.int64),
            adapter_kind=np.asarray("causal_two_tap_next_plus_previous_raw_v1"),
            translation_matrix=self.translation_matrix.astype(np.float64),
            translation_previous_matrix=self.translation_previous_matrix.astype(np.float64),
            translation_bias=self.translation_bias.astype(np.float64),
            rotation_matrix=self.rotation_matrix.astype(np.float64),
            rotation_previous_matrix=self.rotation_previous_matrix.astype(np.float64),
            rotation_bias=self.rotation_bias.astype(np.float64),
            translation_inverse=np.linalg.pinv(self.translation_matrix).astype(np.float64),
            rotation_inverse=np.linalg.pinv(self.rotation_matrix).astype(np.float64),
            audit_report_sha256=np.asarray(report_sha256),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AdapterFit":
        with np.load(path, allow_pickle=False) as payload:
            if str(payload["schema"].item()) != BRIDGE_SCHEMA:
                raise ValueError(f"unexpected bridge schema in {path}")
            if int(payload["raw_hz"]) != RAW_HZ or int(payload["world_hz"]) != WORLD_HZ:
                raise ValueError(f"unexpected bridge rate contract in {path}")
            if str(payload["adapter_kind"].item()) != "causal_two_tap_next_plus_previous_raw_v1":
                raise ValueError(f"unexpected bridge adapter kind in {path}")
            return cls(
                translation_matrix=np.asarray(payload["translation_matrix"], dtype=np.float64),
                translation_previous_matrix=np.asarray(
                    payload["translation_previous_matrix"], dtype=np.float64
                ),
                translation_bias=np.asarray(payload["translation_bias"], dtype=np.float64),
                rotation_matrix=np.asarray(payload["rotation_matrix"], dtype=np.float64),
                rotation_previous_matrix=np.asarray(
                    payload["rotation_previous_matrix"], dtype=np.float64
                ),
                rotation_bias=np.asarray(payload["rotation_bias"], dtype=np.float64),
            )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def base_frame_deltas(
    ee_pos: np.ndarray,
    ee_ori_rotvec: np.ndarray,
) -> np.ndarray:
    """Return observed ``obs[t] -> obs[t+1]`` effects in the robot base frame."""

    position = np.asarray(ee_pos, dtype=np.float64)
    orientation = np.asarray(ee_ori_rotvec, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"ee_pos must be [N,3], got {position.shape}")
    if orientation.shape != position.shape or len(position) < 2:
        raise ValueError(f"ee_ori must match ee_pos, got {orientation.shape}")
    rotations = Rotation.from_rotvec(orientation)
    # Base-frame delta is left-relative: R[t+1] @ inv(R[t]).  The local-frame
    # alternative is measurably worse on held-out LIBERO trajectories.
    delta_rotation = (rotations[1:] * rotations[:-1].inv()).as_rotvec()
    return np.concatenate((np.diff(position, axis=0), delta_rotation), axis=-1).astype(
        np.float32
    )


def context_indices(anchor: int) -> np.ndarray:
    """Exact ``[t-60,t-56,...,t]`` indices, left padded at episode start."""

    anchor = int(anchor)
    if anchor < 0:
        raise ValueError("anchor must be nonnegative")
    values = np.arange(anchor - CONTEXT_RAW_SPAN, anchor + 1, RAW_PER_WORLD)
    return np.maximum(values, 0).astype(np.int64)


def future_indices(anchor: int) -> np.ndarray:
    """Future observation indices corresponding to eight 5 Hz effects."""

    anchor = int(anchor)
    return (anchor + np.arange(1, WORLD_HORIZON + 1) * RAW_PER_WORLD).astype(
        np.int64
    )


def compose_native_targets(
    observed_effects: np.ndarray,
    raw_actions: np.ndarray,
    anchor: int,
) -> np.ndarray:
    """Compose 32 observed 20 Hz effects into canonical K8@5 Hz targets."""

    effects = np.asarray(observed_effects, dtype=np.float64)
    raw = np.asarray(raw_actions, dtype=np.float64)
    anchor = int(anchor)
    if effects.ndim != 2 or effects.shape[1] != 6:
        raise ValueError(f"observed_effects must be [N-1,6], got {effects.shape}")
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"raw_actions must be [N,7], got {raw.shape}")
    if anchor < 0 or anchor + FUTURE_RAW_ACTIONS > len(effects):
        raise IndexError("native target interval exceeds observed effects")
    if anchor + FUTURE_RAW_ACTIONS >= len(raw):
        raise IndexError("native gripper target interval exceeds raw actions")

    output = np.empty((WORLD_HORIZON, 7), dtype=np.float32)
    for world_step in range(WORLD_HORIZON):
        start = anchor + world_step * RAW_PER_WORLD
        stop = start + RAW_PER_WORLD
        chunk = effects[start:stop]
        output[world_step, :3] = chunk[:, :3].sum(axis=0)
        # Each effect is base-frame (left-relative), hence newest delta is
        # multiplied on the left.  This equals R[stop] @ inv(R[start]).
        composed = Rotation.identity()
        for delta in chunk[:, 3:6]:
            composed = Rotation.from_rotvec(delta) * composed
        output[world_step, 3:6] = composed.as_rotvec()
        # effects[start:stop] are caused by raw actions[start+1:stop+1]
        # because the stored observations are post-action.  The canonical
        # gripper state is therefore the final command at raw[stop].
        output[world_step, 6] = 1.0 if raw[stop, 6] > 0 else -1.0
    return output


def servo_target(raw_actions: np.ndarray, anchor: int) -> np.ndarray:
    raw = np.asarray(raw_actions, dtype=np.float32)
    start = int(anchor) + HDF5_NEXT_ACTION_OFFSET
    target = raw[start : start + SERVO_HORIZON]
    if target.shape != (SERVO_HORIZON, 7):
        raise IndexError("servo target interval exceeds raw actions")
    if not np.isclose(np.abs(target[:, 6]), 1.0, atol=1e-6, rtol=0).all():
        raise ValueError("LIBERO gripper commands must be -1/open or +1/close")
    return target.copy()


def fit_linear_adapter(
    raw_pose: np.ndarray,
    previous_raw_pose: np.ndarray,
    physical_pose: np.ndarray,
) -> AdapterFit:
    raw = np.asarray(raw_pose, dtype=np.float64)
    previous = np.asarray(previous_raw_pose, dtype=np.float64)
    physical = np.asarray(physical_pose, dtype=np.float64)
    if (
        raw.shape != previous.shape
        or raw.shape != physical.shape
        or raw.ndim != 2
        or raw.shape[1] != 6
    ):
        raise ValueError("raw, previous raw, and physical poses must all be [N,6]")

    def fit(
        source: np.ndarray, previous_source: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        design = np.concatenate(
            (source, previous_source, np.ones((len(source), 1))), axis=1
        )
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        return coefficients[:3], coefficients[3:6], coefficients[-1]

    translation_matrix, translation_previous_matrix, translation_bias = fit(
        raw[:, :3], previous[:, :3], physical[:, :3]
    )
    rotation_matrix, rotation_previous_matrix, rotation_bias = fit(
        raw[:, 3:6], previous[:, 3:6], physical[:, 3:6]
    )
    return AdapterFit(
        translation_matrix=translation_matrix,
        translation_previous_matrix=translation_previous_matrix,
        translation_bias=translation_bias,
        rotation_matrix=rotation_matrix,
        rotation_previous_matrix=rotation_previous_matrix,
        rotation_bias=rotation_bias,
    )


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = prediction - target
    denominator = np.mean((target - target.mean(axis=0)) ** 2)
    r2 = 1.0 - float(np.mean(residual**2) / max(denominator, 1e-16))
    norms = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    valid = norms > 1e-10
    cosine = float(
        np.median(np.sum(target[valid] * prediction[valid], axis=1) / norms[valid])
    )
    return {
        "r2": r2,
        "median_direction_cosine": cosine,
        "mae": float(np.mean(np.abs(residual))),
        "p99_error_norm": float(np.quantile(np.linalg.norm(residual, axis=1), 0.99)),
    }


def lag_metrics(
    raw_pose_episodes: Iterable[np.ndarray],
    physical_pose_episodes: Iterable[np.ndarray],
    *,
    lags: range = range(-3, 4),
) -> dict[str, object]:
    raw_blocks = [np.asarray(value, dtype=np.float64) for value in raw_pose_episodes]
    physical_blocks = [
        np.asarray(value, dtype=np.float64) for value in physical_pose_episodes
    ]
    if len(raw_blocks) != len(physical_blocks):
        raise ValueError("raw/physical episode counts differ")
    scores: dict[str, float] = {}
    for lag in lags:
        raw_norms: list[np.ndarray] = []
        physical_norms: list[np.ndarray] = []
        for raw, physical in zip(raw_blocks, physical_blocks):
            if lag >= 0:
                left, right = raw[: len(raw) - lag or None], physical[lag:]
            else:
                left, right = raw[-lag:], physical[:lag]
            if len(left) > 3:
                raw_norms.append(np.diff(np.linalg.norm(left, axis=1)))
                physical_norms.append(np.diff(np.linalg.norm(right, axis=1)))
        left = np.concatenate(raw_norms)
        right = np.concatenate(physical_norms)
        scores[str(lag)] = float(np.corrcoef(left, right)[0, 1])
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "scores": scores,
        "best_lag": int(ordered[0][0]),
        "peak_margin": float(ordered[0][1] - ordered[1][1]),
    }


def write_json(path: str | Path, payload: dict) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return sha256_file(destination)
