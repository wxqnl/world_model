"""Unified, embodiment-aware robot state and action contract for WM3D V8.

This module deliberately performs *alignment*, not resampling.  A source that
only provides a world-rate effect can supervise dynamics, but it can never be
promoted into a high-rate policy command by interpolation, repetition, or
padding.  Dataset adapters are responsible for constructing the raw series;
the trainer only consumes the padded, masked representation emitted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np


GROUPED_ROBOT_SCHEMA = "wm3d_v8_grouped_robot_v1"

FineOrCoarse = Literal["fine_command", "coarse_effect"]


ACTION_SEMANTIC_IDS: Mapping[str, int] = {
    "unused": 0,
    "delta_position_m": 1,
    "delta_rotation_6d": 2,
    "delta_rotation_axis_angle_rad": 3,
    "absolute_gripper_open01": 4,
    "absolute_gripper_close01": 5,
    "joint_position_rad": 6,
    "joint_delta_rad": 7,
    "base_velocity_mps": 8,
    "base_yaw_rate_rps": 9,
    "binary_contact": 10,
    "continuous_force": 11,
    # Opaque controller channels are permitted only after a source audit has
    # named their unit/frame in the data profile.  They preserve dimensions
    # that are not Cartesian pose commands instead of silently dropping them.
    "controller_command": 12,
    "controller_mode": 13,
    "joint_velocity_rps": 14,
    "joint_torque_nm": 15,
}

STATE_SEMANTIC_IDS: Mapping[str, int] = {
    "unused": 0,
    "eef_position_m": 1,
    "eef_rotation_6d": 2,
    "gripper_open01": 3,
    "gripper_close01": 4,
    "joint_position_rad": 5,
    "joint_velocity_rps": 6,
    "base_pose": 7,
    "base_velocity": 8,
    "force_torque": 9,
    "tactile": 10,
    "controller_state": 11,
}

COMPOSITION_OPERATOR_IDS: Mapping[str, int] = {
    "none": 0,
    "sum": 1,
    "so3_axis_angle_base_left": 2,
    "last": 3,
    "time_weighted_mean": 4,
    "logical_last": 5,
    "so3_axis_angle_body_right": 6,
}


class GroupedRobotContractError(ValueError):
    """Raised when a sample cannot satisfy the lossless V8 robot contract."""


@dataclass(frozen=True)
class ActionGroupSpec:
    """Static layout of one independently controlled robot group."""

    name: str
    group_id: int
    action_semantics: Tuple[str, ...]
    state_semantics: Tuple[str, ...] = ()
    action_frame: str = "unspecified"
    state_frame: str = "unspecified"
    composition_operators: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise GroupedRobotContractError("action group name must be non-empty")
        if self.group_id <= 0:
            raise GroupedRobotContractError("group_id=0 is reserved for padding")
        if not self.action_semantics:
            raise GroupedRobotContractError(f"group {self.name!r} has no action dimensions")
        unknown_action = sorted(set(self.action_semantics) - set(ACTION_SEMANTIC_IDS))
        unknown_state = sorted(set(self.state_semantics) - set(STATE_SEMANTIC_IDS))
        unknown_operators = sorted(
            set(self.composition_operators) - set(COMPOSITION_OPERATOR_IDS)
        )
        if unknown_action:
            raise GroupedRobotContractError(
                f"group {self.name!r} has unknown action semantics: {unknown_action}"
            )
        if unknown_state:
            raise GroupedRobotContractError(
                f"group {self.name!r} has unknown state semantics: {unknown_state}"
            )
        if not self.action_frame or self.action_frame == "unspecified":
            raise GroupedRobotContractError(
                f"group {self.name!r} must declare its action coordinate frame"
            )
        if self.state_semantics and (
            not self.state_frame or self.state_frame == "unspecified"
        ):
            raise GroupedRobotContractError(
                f"group {self.name!r} must declare its state coordinate frame"
            )
        if len(self.composition_operators) != len(self.action_semantics):
            raise GroupedRobotContractError(
                f"group {self.name!r} must declare one composition operator per action dimension"
            )
        if unknown_operators:
            raise GroupedRobotContractError(
                f"group {self.name!r} has unknown composition operators: {unknown_operators}"
            )

    @property
    def action_dim(self) -> int:
        return len(self.action_semantics)

    @property
    def state_dim(self) -> int:
        return len(self.state_semantics)


@dataclass(frozen=True)
class EmbodimentSpec:
    """A robot embodiment expressed as a collection of action groups."""

    name: str
    embodiment_id: int
    groups: Tuple[ActionGroupSpec, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise GroupedRobotContractError("embodiment name must be non-empty")
        if self.embodiment_id <= 0:
            raise GroupedRobotContractError("embodiment_id=0 is reserved for padding/unknown")
        if not self.groups:
            raise GroupedRobotContractError("embodiment must contain at least one action group")
        names = [group.name for group in self.groups]
        ids = [group.group_id for group in self.groups]
        if len(names) != len(set(names)):
            raise GroupedRobotContractError(f"duplicate action group names: {names}")
        if len(ids) != len(set(ids)):
            raise GroupedRobotContractError(f"duplicate action group ids: {ids}")


@dataclass(frozen=True)
class GroupedRobotLimits:
    """Padding limits shared by all model profiles.

    These are capacity limits, not a declaration that every source has all
    groups, dimensions, or substeps.  Presence is always carried by masks.
    """

    max_groups: int = 8
    max_substeps: int = 128
    max_action_dim: int = 16
    max_state_dim: int = 32
    timestamp_tolerance_s: float = 1.0e-6

    def __post_init__(self) -> None:
        for field_name in ("max_groups", "max_substeps", "max_action_dim", "max_state_dim"):
            if int(getattr(self, field_name)) <= 0:
                raise GroupedRobotContractError(f"{field_name} must be positive")
        if self.timestamp_tolerance_s < 0 or not np.isfinite(self.timestamp_tolerance_s):
            raise GroupedRobotContractError("timestamp_tolerance_s must be finite and non-negative")


@dataclass(frozen=True)
class RawActionSeries:
    """One source-native action sequence for one group.

    Fine commands carry their actual timestamps.  Coarse effects must carry an
    explicit world interval index for every value; inferring those intervals
    from a nominal source frequency would silently change their semantics.
    """

    group: str
    supervision: FineOrCoarse
    values: np.ndarray
    timestamps_s: Optional[np.ndarray] = None
    world_interval_indices: Optional[np.ndarray] = None
    value_mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RawStateSnapshot:
    """Current state at the first command timestamp of the policy chunk."""

    group: str
    timestamp_s: float
    values: np.ndarray
    value_mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RawStateSeries:
    """Source-native measured state sequence for one physical group."""

    group: str
    values: np.ndarray
    timestamps_s: np.ndarray
    value_mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class GroupedRobotWindow:
    """Padded representation consumed by the unified WM3D model."""

    schema: str
    embodiment_id: np.int64
    group_ids: np.ndarray
    group_mask: np.ndarray
    action_semantic_ids: np.ndarray
    state_semantic_ids: np.ndarray
    composition_operator_ids: np.ndarray
    fine_action_values: np.ndarray
    fine_action_mask: np.ndarray
    fine_action_dt: np.ndarray
    fine_sample_mask: np.ndarray
    coarse_action_values: np.ndarray
    coarse_action_mask: np.ndarray
    current_state_values: np.ndarray
    current_state_mask: np.ndarray
    world_boundaries_s: np.ndarray
    world_interval_dt: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema": self.schema,
            "embodiment_id": self.embodiment_id,
            "group_ids": self.group_ids,
            "group_mask": self.group_mask,
            "action_semantic_ids": self.action_semantic_ids,
            "state_semantic_ids": self.state_semantic_ids,
            "composition_operator_ids": self.composition_operator_ids,
            "fine_action_values": self.fine_action_values,
            "fine_action_mask": self.fine_action_mask,
            "fine_action_dt": self.fine_action_dt,
            "fine_sample_mask": self.fine_sample_mask,
            "coarse_action_values": self.coarse_action_values,
            "coarse_action_mask": self.coarse_action_mask,
            "current_state_values": self.current_state_values,
            "current_state_mask": self.current_state_mask,
            "world_boundaries_s": self.world_boundaries_s,
            "world_interval_dt": self.world_interval_dt,
        }


def _as_finite_float_array(value: object, *, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise GroupedRobotContractError(f"{name} must be rank {ndim}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise GroupedRobotContractError(f"{name} contains non-finite values")
    return array


def _as_mask(value: Optional[np.ndarray], shape: Tuple[int, ...], *, name: str) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=np.bool_)
    mask = np.asarray(value, dtype=np.bool_)
    if mask.shape != shape:
        raise GroupedRobotContractError(f"{name} must have shape {shape}, got {mask.shape}")
    return mask


def _validate_boundaries(boundaries: object) -> np.ndarray:
    result = np.asarray(boundaries, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise GroupedRobotContractError("world_boundaries_s must be rank-1 with at least two values")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0):
        raise GroupedRobotContractError("world_boundaries_s must be finite and strictly increasing")
    return result


def _group_lookup(
    embodiment: EmbodimentSpec, limits: GroupedRobotLimits
) -> Tuple[
    Dict[str, Tuple[int, ActionGroupSpec]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if len(embodiment.groups) > limits.max_groups:
        raise GroupedRobotContractError(
            f"embodiment {embodiment.name!r} has {len(embodiment.groups)} groups, "
            f"limit is {limits.max_groups}"
        )
    lookup: Dict[str, Tuple[int, ActionGroupSpec]] = {}
    group_ids = np.zeros((limits.max_groups,), dtype=np.int64)
    group_mask = np.zeros((limits.max_groups,), dtype=np.bool_)
    action_semantics = np.zeros(
        (limits.max_groups, limits.max_action_dim), dtype=np.int64
    )
    state_semantics = np.zeros((limits.max_groups, limits.max_state_dim), dtype=np.int64)
    composition = np.zeros((limits.max_groups, limits.max_action_dim), dtype=np.int64)
    for slot, group in enumerate(embodiment.groups):
        if group.action_dim > limits.max_action_dim:
            raise GroupedRobotContractError(
                f"group {group.name!r} action_dim={group.action_dim} exceeds "
                f"max_action_dim={limits.max_action_dim}"
            )
        if group.state_dim > limits.max_state_dim:
            raise GroupedRobotContractError(
                f"group {group.name!r} state_dim={group.state_dim} exceeds "
                f"max_state_dim={limits.max_state_dim}"
            )
        lookup[group.name] = (slot, group)
        group_ids[slot] = group.group_id
        group_mask[slot] = True
        action_semantics[slot, : group.action_dim] = [
            ACTION_SEMANTIC_IDS[name] for name in group.action_semantics
        ]
        state_semantics[slot, : group.state_dim] = [
            STATE_SEMANTIC_IDS[name] for name in group.state_semantics
        ]
        composition[slot, : group.action_dim] = [
            COMPOSITION_OPERATOR_IDS[name] for name in group.composition_operators
        ]
    return lookup, group_ids, group_mask, action_semantics, state_semantics, composition


def _locate_fine_interval(
    timestamp_s: float,
    boundaries: np.ndarray,
    tolerance_s: float,
) -> int:
    start = float(boundaries[0])
    end = float(boundaries[-1])
    if timestamp_s < start - tolerance_s or timestamp_s >= end:
        raise GroupedRobotContractError(
            f"fine command timestamp {timestamp_s:.9f} is outside half-open window [{start}, {end})"
        )
    # Only the outer lower edge may absorb serialization noise.  Internal
    # timestamps are never shifted by tolerance: a command immediately below
    # a boundary remains in the preceding interval, and an exact-boundary
    # command belongs to the following interval.
    exact = max(timestamp_s, start)
    snapped = int(np.searchsorted(boundaries, exact, side="right") - 1)
    return min(max(snapped, 0), len(boundaries) - 2)


def slice_fine_action_series(
    series: RawActionSeries,
    *,
    start_s: float,
    stop_s: float,
) -> RawActionSeries:
    """Select real fine commands inside one half-open physical-time window.

    This helper never snaps, repeats, or interpolates a command.  Boundary
    ownership is exact: ``start_s`` is included and ``stop_s`` is excluded.
    """

    if series.supervision != "fine_command" or series.timestamps_s is None:
        raise GroupedRobotContractError(
            "slice_fine_action_series requires a timestamped fine-command series"
        )
    if not np.isfinite(start_s) or not np.isfinite(stop_s) or stop_s <= start_s:
        raise GroupedRobotContractError("fine-command slice bounds are invalid")
    timestamps = np.asarray(series.timestamps_s, dtype=np.float64)
    values = _as_finite_float_array(series.values, name=f"{series.group}.values", ndim=2)
    if timestamps.shape != (values.shape[0],):
        raise GroupedRobotContractError("fine-command timestamp cardinality mismatch")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
        raise GroupedRobotContractError("fine-command timestamps must be strictly increasing")
    keep = (timestamps >= np.float64(start_s)) & (timestamps < np.float64(stop_s))
    mask = _as_mask(series.value_mask, values.shape, name=f"{series.group}.value_mask")
    return RawActionSeries(
        group=series.group,
        supervision="fine_command",
        values=values[keep].copy(),
        timestamps_s=timestamps[keep].copy(),
        value_mask=mask[keep].copy(),
    )


def _axis_angle_to_quaternion(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        # First-order form avoids dividing by a vanishing angle while keeping
        # the identity exact for a true zero delta.
        xyz = 0.5 * vector.astype(np.float64, copy=False)
        result = np.asarray((1.0, xyz[0], xyz[1], xyz[2]), dtype=np.float64)
    else:
        half = 0.5 * angle
        xyz = vector.astype(np.float64, copy=False) * (np.sin(half) / angle)
        result = np.asarray((np.cos(half), xyz[0], xyz[1], xyz[2]), dtype=np.float64)
    return result / np.linalg.norm(result)


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left.tolist()
    rw, rx, ry, rz = right.tolist()
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    value = quaternion.astype(np.float64, copy=False)
    value = value / np.linalg.norm(value)
    # q and -q encode the same rotation.  The non-negative scalar convention
    # gives the principal axis-angle representation deterministically.
    if value[0] < 0:
        value = -value
    xyz_norm = float(np.linalg.norm(value[1:]))
    if xyz_norm < 1.0e-12:
        return (2.0 * value[1:]).astype(np.float32)
    angle = 2.0 * np.arctan2(xyz_norm, float(value[0]))
    return (value[1:] * (angle / xyz_norm)).astype(np.float32)


def _compose_axis_angle_rows(values: np.ndarray, *, left_multiply: bool) -> np.ndarray:
    total = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    for row in values:
        delta = _axis_angle_to_quaternion(row)
        total = (
            _quaternion_multiply(delta, total)
            if left_multiply
            else _quaternion_multiply(total, delta)
        )
        total /= np.linalg.norm(total)
    return _quaternion_to_axis_angle(total)


def compose_coarse_effects_for_observed_window(
    series: RawActionSeries,
    *,
    group: ActionGroupSpec,
    source_world_times_s: Sequence[float],
    selected_boundary_indices: Sequence[int],
) -> RawActionSeries:
    """Compose source-interval effects onto selected observed-state intervals.

    Every raw interval between two selected boundaries must have exactly one
    recorded coarse-effect row.  Missing rows are not interpreted as zeros;
    duplicate rows are not averaged.  Rotation triples use the explicitly
    declared multiplication convention rather than vector addition.
    """

    if series.supervision != "coarse_effect" or series.world_interval_indices is None:
        raise GroupedRobotContractError(
            "coarse-effect composition requires explicit source interval indices"
        )
    clock = _validate_boundaries(source_world_times_s)
    boundaries = np.asarray(selected_boundary_indices, dtype=np.int64)
    if (
        boundaries.ndim != 1
        or boundaries.size < 2
        or np.any(np.diff(boundaries) <= 0)
        or boundaries[0] < 0
        or boundaries[-1] >= clock.size
    ):
        raise GroupedRobotContractError(
            "selected_boundary_indices must be valid strictly increasing source rows"
        )
    values = _as_finite_float_array(series.values, name=f"{series.group}.values", ndim=2)
    if values.shape[1] != group.action_dim:
        raise GroupedRobotContractError(
            f"{series.group}.values action_dim={values.shape[1]}, expected {group.action_dim}"
        )
    indices = np.asarray(series.world_interval_indices, dtype=np.int64)
    if indices.shape != (values.shape[0],) or np.any(indices < 0) or np.any(
        indices >= clock.size - 1
    ):
        raise GroupedRobotContractError("coarse source interval indices are invalid")
    if np.unique(indices).size != indices.size:
        raise GroupedRobotContractError("coarse source interval indices are duplicated")
    order = np.argsort(indices)
    indices = indices[order]
    values = values[order]
    raw_mask = _as_mask(
        series.value_mask, np.asarray(series.values).shape, name=f"{series.group}.value_mask"
    )[order]

    output_values = np.zeros((boundaries.size - 1, group.action_dim), dtype=np.float32)
    output_mask = np.zeros_like(output_values, dtype=np.bool_)
    rotation_slots: set[int] = set()
    for dimension, operator in enumerate(group.composition_operators):
        if operator in {
            "so3_axis_angle_base_left",
            "so3_axis_angle_body_right",
        }:
            rotation_slots.add(dimension)
    for start in sorted(rotation_slots):
        if start > 0 and start - 1 in rotation_slots:
            continue
        if {start, start + 1, start + 2} - rotation_slots:
            raise GroupedRobotContractError(
                f"group {group.name!r} SO(3) composition dimensions must be contiguous triples"
            )
        operator = group.composition_operators[start]
        if any(group.composition_operators[start + offset] != operator for offset in range(3)):
            raise GroupedRobotContractError(
                f"group {group.name!r} SO(3) triple must use one multiplication convention"
            )

    for output_interval, (left, right) in enumerate(
        zip(boundaries[:-1].tolist(), boundaries[1:].tolist())
    ):
        expected = np.arange(left, right, dtype=np.int64)
        selected_rows = np.flatnonzero((indices >= left) & (indices < right))
        if not np.array_equal(indices[selected_rows], expected):
            raise GroupedRobotContractError(
                f"group {series.group!r} does not provide exactly one coarse effect for "
                f"every source interval [{left}, {right})"
            )
        interval_values = values[selected_rows]
        interval_mask = raw_mask[selected_rows]
        duration = np.diff(clock)[expected]
        for dimension, operator in enumerate(group.composition_operators):
            if dimension in rotation_slots:
                continue
            valid = bool(interval_mask[:, dimension].all())
            output_mask[output_interval, dimension] = valid
            if not valid:
                continue
            column = interval_values[:, dimension]
            if operator == "sum":
                output_values[output_interval, dimension] = np.float32(column.sum(dtype=np.float64))
            elif operator in {"last", "logical_last"}:
                output_values[output_interval, dimension] = column[-1]
            elif operator == "time_weighted_mean":
                output_values[output_interval, dimension] = np.float32(
                    np.average(column.astype(np.float64), weights=duration)
                )
            elif operator == "none":
                if len(column) != 1:
                    raise GroupedRobotContractError(
                        f"group {group.name!r} dimension {dimension} cannot compose "
                        f"{len(column)} rows with operator 'none'"
                    )
                output_values[output_interval, dimension] = column[0]
            else:
                raise GroupedRobotContractError(
                    f"unsupported scalar composition operator {operator!r}"
                )
        for start in sorted(rotation_slots):
            if start > 0 and start - 1 in rotation_slots:
                continue
            valid = bool(interval_mask[:, start : start + 3].all())
            output_mask[output_interval, start : start + 3] = valid
            if valid:
                output_values[output_interval, start : start + 3] = _compose_axis_angle_rows(
                    interval_values[:, start : start + 3],
                    left_multiply=(
                        group.composition_operators[start]
                        == "so3_axis_angle_base_left"
                    ),
                )
    return RawActionSeries(
        group=series.group,
        supervision="coarse_effect",
        values=output_values,
        world_interval_indices=np.arange(boundaries.size - 1, dtype=np.int64),
        value_mask=output_mask,
    )


def pack_grouped_robot_window(
    *,
    embodiment: EmbodimentSpec,
    limits: GroupedRobotLimits,
    world_boundaries_s: Sequence[float],
    action_series: Iterable[RawActionSeries],
    current_state: Iterable[RawStateSnapshot],
    policy_chunk_start_s: float,
) -> GroupedRobotWindow:
    """Pack source-native robot signals without inventing supervision.

    Every fine command appears exactly once in the result.  Every coarse effect
    appears exactly once in its explicitly declared interval.  Empty slots stay
    masked and zero; they are never filled by interpolation or repetition.
    """

    boundaries = _validate_boundaries(world_boundaries_s)
    interval_count = boundaries.size - 1
    (
        group_lookup,
        group_ids,
        group_mask,
        action_semantics,
        state_semantics,
        composition_operators,
    ) = _group_lookup(embodiment, limits)

    fine_values = np.zeros(
        (
            interval_count,
            limits.max_groups,
            limits.max_substeps,
            limits.max_action_dim,
        ),
        dtype=np.float32,
    )
    fine_mask = np.zeros_like(fine_values, dtype=np.bool_)
    fine_dt = np.zeros(
        (interval_count, limits.max_groups, limits.max_substeps), dtype=np.float32
    )
    fine_sample_mask = np.zeros_like(fine_dt, dtype=np.bool_)
    coarse_values = np.zeros(
        (interval_count, limits.max_groups, limits.max_action_dim), dtype=np.float32
    )
    coarse_mask = np.zeros_like(coarse_values, dtype=np.bool_)
    current_values = np.zeros((limits.max_groups, limits.max_state_dim), dtype=np.float32)
    current_mask = np.zeros_like(current_values, dtype=np.bool_)

    seen_series: set[Tuple[str, FineOrCoarse]] = set()
    for series in action_series:
        if series.group not in group_lookup:
            raise GroupedRobotContractError(
                f"action series references unknown group {series.group!r}"
            )
        identity = (series.group, series.supervision)
        if identity in seen_series:
            raise GroupedRobotContractError(f"duplicate action series {identity}")
        seen_series.add(identity)
        group_slot, group = group_lookup[series.group]
        values = _as_finite_float_array(series.values, name=f"{series.group}.values", ndim=2)
        if values.shape[1] != group.action_dim:
            raise GroupedRobotContractError(
                f"{series.group}.values action_dim={values.shape[1]}, expected {group.action_dim}"
            )
        value_mask = _as_mask(series.value_mask, values.shape, name=f"{series.group}.value_mask")

        if series.supervision == "fine_command":
            if series.timestamps_s is None:
                raise GroupedRobotContractError(
                    f"fine action series {series.group!r} requires source timestamps"
                )
            if series.world_interval_indices is not None:
                raise GroupedRobotContractError(
                    "fine commands must be aligned from timestamps, not caller-assigned intervals"
                )
            timestamps = np.asarray(series.timestamps_s, dtype=np.float64)
            if timestamps.ndim != 1 or timestamps.shape[0] != values.shape[0]:
                raise GroupedRobotContractError(
                    f"{series.group}.timestamps_s must have shape ({values.shape[0]},)"
                )
            if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0):
                raise GroupedRobotContractError(
                    f"{series.group}.timestamps_s must be finite and strictly increasing"
                )
            next_slot = np.zeros((interval_count,), dtype=np.int64)
            for row, timestamp_s in enumerate(timestamps.tolist()):
                interval = _locate_fine_interval(
                    timestamp_s, boundaries, limits.timestamp_tolerance_s
                )
                substep = int(next_slot[interval])
                if substep >= limits.max_substeps:
                    raise GroupedRobotContractError(
                        f"group {series.group!r} interval {interval} contains more than "
                        f"{limits.max_substeps} real commands; increase max_substeps instead "
                        "of dropping or resampling them"
                    )
                next_slot[interval] += 1
                fine_values[interval, group_slot, substep, : group.action_dim] = values[row]
                fine_mask[interval, group_slot, substep, : group.action_dim] = value_mask[row]
                fine_dt[interval, group_slot, substep] = np.float32(
                    timestamp_s - float(boundaries[interval])
                )
                fine_sample_mask[interval, group_slot, substep] = True
        elif series.supervision == "coarse_effect":
            if series.timestamps_s is not None:
                raise GroupedRobotContractError(
                    "coarse effects use explicit world_interval_indices; timestamps must be omitted"
                )
            if series.world_interval_indices is None:
                raise GroupedRobotContractError(
                    f"coarse action series {series.group!r} requires world_interval_indices"
                )
            indices = np.asarray(series.world_interval_indices, dtype=np.int64)
            if indices.ndim != 1 or indices.shape[0] != values.shape[0]:
                raise GroupedRobotContractError(
                    f"{series.group}.world_interval_indices must have shape ({values.shape[0]},)"
                )
            if np.any(indices < 0) or np.any(indices >= interval_count):
                raise GroupedRobotContractError(
                    f"{series.group}.world_interval_indices contains out-of-window values"
                )
            if len(np.unique(indices)) != len(indices):
                raise GroupedRobotContractError(
                    f"{series.group} has multiple coarse effects for one world interval"
                )
            for row, interval in enumerate(indices.tolist()):
                coarse_values[interval, group_slot, : group.action_dim] = values[row]
                coarse_mask[interval, group_slot, : group.action_dim] = value_mask[row]
        else:
            raise GroupedRobotContractError(
                f"unsupported supervision {series.supervision!r}"
            )

    seen_state_groups: set[str] = set()
    for snapshot in current_state:
        if snapshot.group not in group_lookup:
            raise GroupedRobotContractError(
                f"state snapshot references unknown group {snapshot.group!r}"
            )
        if snapshot.group in seen_state_groups:
            raise GroupedRobotContractError(
                f"duplicate current-state snapshot for group {snapshot.group!r}"
            )
        seen_state_groups.add(snapshot.group)
        group_slot, group = group_lookup[snapshot.group]
        if group.state_dim == 0:
            raise GroupedRobotContractError(
                f"group {snapshot.group!r} does not declare current-state semantics"
            )
        if not np.isfinite(snapshot.timestamp_s) or not np.isclose(
            snapshot.timestamp_s,
            policy_chunk_start_s,
            rtol=0.0,
            atol=limits.timestamp_tolerance_s,
        ):
            raise GroupedRobotContractError(
                f"current state for {snapshot.group!r} is at {snapshot.timestamp_s:.9f}, "
                f"policy chunk starts at {policy_chunk_start_s:.9f}; interpolation/fallback is forbidden"
            )
        values = _as_finite_float_array(
            snapshot.values, name=f"{snapshot.group}.current_state", ndim=1
        )
        if values.shape[0] != group.state_dim:
            raise GroupedRobotContractError(
                f"{snapshot.group}.current_state dim={values.shape[0]}, expected {group.state_dim}"
            )
        value_mask = _as_mask(
            snapshot.value_mask, values.shape, name=f"{snapshot.group}.current_state_mask"
        )
        current_values[group_slot, : group.state_dim] = values
        current_mask[group_slot, : group.state_dim] = value_mask

    required_state_groups = {group.name for group in embodiment.groups if group.state_dim > 0}
    missing_state = sorted(required_state_groups - seen_state_groups)
    if missing_state:
        raise GroupedRobotContractError(
            f"missing exact current-state snapshots for groups: {missing_state}"
        )

    return GroupedRobotWindow(
        schema=GROUPED_ROBOT_SCHEMA,
        embodiment_id=np.int64(embodiment.embodiment_id),
        group_ids=group_ids,
        group_mask=group_mask,
        action_semantic_ids=action_semantics,
        state_semantic_ids=state_semantics,
        composition_operator_ids=composition_operators,
        fine_action_values=fine_values,
        fine_action_mask=fine_mask,
        fine_action_dt=fine_dt,
        fine_sample_mask=fine_sample_mask,
        coarse_action_values=coarse_values,
        coarse_action_mask=coarse_mask,
        current_state_values=current_values,
        current_state_mask=current_mask,
        world_boundaries_s=boundaries,
        world_interval_dt=np.diff(boundaries).astype(np.float32, copy=False),
    )


def panda_single_arm_spec(*, embodiment_id: int = 1) -> EmbodimentSpec:
    """Compatibility profile for V8's normalized Panda policy ABI."""

    return EmbodimentSpec(
        name="panda_single_arm",
        embodiment_id=embodiment_id,
        groups=(
            ActionGroupSpec(
                name="arm",
                group_id=1,
                action_semantics=(
                    "delta_position_m",
                    "delta_position_m",
                    "delta_position_m",
                    "delta_rotation_axis_angle_rad",
                    "delta_rotation_axis_angle_rad",
                    "delta_rotation_axis_angle_rad",
                    "absolute_gripper_open01",
                ),
                state_semantics=(
                    "eef_position_m",
                    "eef_position_m",
                    "eef_position_m",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "gripper_close01",
                ),
                action_frame="robot_base",
                state_frame="robot_base",
                composition_operators=(
                    "sum",
                    "sum",
                    "sum",
                    "so3_axis_angle_base_left",
                    "so3_axis_angle_base_left",
                    "so3_axis_angle_base_left",
                    "logical_last",
                ),
            ),
        ),
    )


def bimanual_arm_spec(*, embodiment_id: int = 2) -> EmbodimentSpec:
    """Reference dual-arm layout used by tests and adapter implementations."""

    action_semantics = (
        "delta_position_m",
        "delta_position_m",
        "delta_position_m",
        "delta_rotation_axis_angle_rad",
        "delta_rotation_axis_angle_rad",
        "delta_rotation_axis_angle_rad",
        "absolute_gripper_open01",
    )
    state_semantics = (
        "eef_position_m",
        "eef_position_m",
        "eef_position_m",
        "eef_rotation_6d",
        "eef_rotation_6d",
        "eef_rotation_6d",
        "eef_rotation_6d",
        "eef_rotation_6d",
        "eef_rotation_6d",
        "gripper_close01",
    )
    return EmbodimentSpec(
        name="bimanual_arm",
        embodiment_id=embodiment_id,
        groups=(
            ActionGroupSpec(
                "left_arm",
                1,
                action_semantics,
                state_semantics,
                "robot_base",
                "robot_base",
                ("sum", "sum", "sum", "so3_axis_angle_base_left", "so3_axis_angle_base_left", "so3_axis_angle_base_left", "logical_last"),
            ),
            ActionGroupSpec(
                "right_arm",
                2,
                action_semantics,
                state_semantics,
                "robot_base",
                "robot_base",
                ("sum", "sum", "sum", "so3_axis_angle_base_left", "so3_axis_angle_base_left", "so3_axis_angle_base_left", "logical_last"),
            ),
        ),
    )
