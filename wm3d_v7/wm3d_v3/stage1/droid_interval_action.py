from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


DROID_INTERVAL_ACTION_KIND = "cartesian_target_interval_delta+rpy+gripper"
DROID_INTERVAL_ACTION_CACHE_SUBDIR = "actions_stage1_rgb_world"
DROID_INTERVAL_STATE_CACHE_SUBDIR = "robot_state_stage1_rgb_world"
DROID_INTERVAL_ACTION_VALID_COUNT = "n_frames-1"
DROID_INTERVAL_STATE_COUNT = "n_frames"
DROID_INTERVAL_TERMINAL_POLICY = "omitted"
DROID_INTERVAL_TRANSLATION_ROTATION_FORMULA = (
    "cartesian interval-end command minus interval-start observed state"
)
DROID_INTERVAL_ROTATION_FORMULA = "wrapped_rpy_delta_in_[-pi,pi)"
DROID_INTERVAL_GRIPPER_FORMULA = "interval-end_absolute_command"
DROID_INTERVAL_END_FORMULA = "next_sample_index_minus_one"


class DroidIntervalActionError(ValueError):
    pass


@dataclass(frozen=True)
class DroidIntervalActionSeries:
    actions: np.ndarray
    state_pose: np.ndarray
    state_grip: np.ndarray
    sampled_row_indices: tuple[int, ...]
    source_kind: str = DROID_INTERVAL_ACTION_KIND


def wrap_angle_delta(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise DroidIntervalActionError("rotation delta contains non-finite values")
    return (array + np.pi) % (2.0 * np.pi) - np.pi


def _finite_vector(
    row: dict[str, Any],
    key: str,
    *,
    expected_size: int,
) -> np.ndarray:
    if key not in row:
        raise DroidIntervalActionError(f"missing DROID field {key}")
    value = np.asarray(row[key], dtype=np.float64).reshape(-1)
    if value.size != expected_size or not np.isfinite(value).all():
        raise DroidIntervalActionError(
            f"invalid DROID field {key}: shape={value.shape}, "
            f"expected=({expected_size},)"
        )
    return value


def _finite_scalar(row: dict[str, Any], key: str) -> float:
    value = _finite_vector(row, key, expected_size=1)
    return float(value[0])


def build_interval_action_series(
    rows: Sequence[dict[str, Any]],
    *,
    sampled_row_indices: Sequence[int],
) -> DroidIntervalActionSeries:
    sampled = tuple(int(value) for value in sampled_row_indices)
    if len(sampled) < 2:
        raise DroidIntervalActionError("at least two sampled rows are required")
    if any(right <= left for left, right in zip(sampled, sampled[1:])):
        raise DroidIntervalActionError(
            "sampled_row_indices must be strictly increasing"
        )
    if sampled[0] < 0 or sampled[-1] >= len(rows):
        raise DroidIntervalActionError(
            "sampled_row_indices leave the DROID row bounds"
        )

    episode_ids = {
        int(rows[index].get("episode_index", -1))
        for index in range(sampled[0], sampled[-1] + 1)
    }
    if len(episode_ids) != 1 or next(iter(episode_ids)) < 0:
        raise DroidIntervalActionError(
            "sampled interval crosses an episode boundary"
        )
    timestamps = np.asarray(
        [
            float(rows[index].get("timestamp", np.nan))
            for index in range(sampled[0], sampled[-1] + 1)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0.0):
        raise DroidIntervalActionError(
            "DROID timestamps must be finite and strictly increasing"
        )

    state_pose = np.stack(
        [
            _finite_vector(
                rows[index],
                "observation.state.cartesian_position",
                expected_size=6,
            )
            for index in sampled
        ]
    )
    state_grip = np.asarray(
        [
            _finite_scalar(
                rows[index],
                "observation.state.gripper_position",
            )
            for index in sampled
        ],
        dtype=np.float64,
    )
    if np.any((state_grip < 0.0) | (state_grip > 1.0)):
        raise DroidIntervalActionError(
            "observed DROID gripper state must lie in [0, 1]"
        )

    actions = np.empty((len(sampled) - 1, 7), dtype=np.float64)
    for output_index, (start, next_start) in enumerate(
        zip(sampled, sampled[1:])
    ):
        command_index = next_start - 1
        command_pose = _finite_vector(
            rows[command_index],
            "action.cartesian_position",
            expected_size=6,
        )
        delta = command_pose - state_pose[output_index]
        delta[3:] = wrap_angle_delta(delta[3:])
        grip = _finite_scalar(
            rows[command_index],
            "action.gripper_position",
        )
        if not (0.0 <= grip <= 1.0):
            raise DroidIntervalActionError(
                "DROID gripper command must lie in [0, 1]"
            )
        actions[output_index, :6] = delta
        actions[output_index, 6] = grip

    return DroidIntervalActionSeries(
        actions=actions.astype(np.float32),
        state_pose=state_pose.astype(np.float32),
        state_grip=state_grip.astype(np.float32),
        sampled_row_indices=sampled,
    )
