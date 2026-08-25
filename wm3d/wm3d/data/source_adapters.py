"""Config-driven raw episode adapters for the unified WM3D cache builder.

All source-specific field names live in audited YAML contracts.  Adapter code
uses declarative tensor mappings and therefore does not branch on dataset
names, robot brands, model size, or nominal frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
import yaml

from .grouped_robot import (
    EmbodimentSpec,
    RawActionSeries,
    RawStateSeries,
    RawStateSnapshot,
)
from .manifest_contract import sha256_file


ADAPTER_SCHEMA = "wm3d_v8_source_adapter_v3"
ADAPTER_COLOR_SCHEMA = "wm3d_source_adapter_v4"
ADAPTER_CANONICAL_SCHEMA = "wm3d_source_adapter_v5"

CANONICAL_ACTION_TRANSFORMS = frozenset(
    {
        "axis_angle_residual",
        "base_rpy_close",
        "base_rpy_open",
        "droid_target",
        "drop_last",
        "euler_xyz_no_grip",
        "euler_xyz_open",
        "euler_zxy_open",
        "furniture_bench",
        "nyu_franka",
        "open_to_close",
        "robocasa",
        "signed_to_close",
        "state_euler_residual",
        "taco_world",
        "xy_only",
        "xyz_only",
    }
)
CANONICAL_STATE_TRANSFORMS = frozenset(
    {
        "austin_buds",
        "cmu_stretch",
        "droid_state",
        "language_table",
        "nyu_franka",
        "pos_euler_7d",
        "pos_euler_8d",
        "pos_quat",
        "pos_quat_no_grip",
        "robocasa",
        "zero",
    }
)


class AdapterContractError(ValueError):
    pass


class EpisodeAccessor(Protocol):
    def array(self, key: str) -> np.ndarray: ...


@dataclass(frozen=True)
class MappingTerm:
    key: str
    columns: tuple[int, ...]
    scale: tuple[float, ...]
    offset: tuple[float, ...]


@dataclass(frozen=True)
class ViewMapping:
    """One source RGB stream mapped into a canonical view slot.

    View names are intentionally not restricted to head/left/right.  The data
    profile declares the ordered view vocabulary and each source may provide
    any non-empty subset of it.  Missing views are represented by a false
    mask at cache time; they are never copied from another camera.
    """

    name: str
    key: str
    color_order: str = "rgb"


@dataclass(frozen=True)
class GroupMapping:
    group: str
    supervision: str
    action: tuple[MappingTerm, ...]
    state: tuple[MappingTerm, ...]
    action_time_key: Optional[str]
    state_time_key: Optional[str]
    world_interval_index_key: Optional[str]
    action_transform: str = "identity"
    state_transform: str = "identity"
    action_value_mask: Optional[tuple[bool, ...]] = None
    state_value_mask: Optional[tuple[bool, ...]] = None


@dataclass(frozen=True)
class AdapterContract:
    path: Path
    sha256: str
    name: str
    raw_format: str
    observation_time_key: str
    views: tuple[ViewMapping, ...]
    groups: tuple[GroupMapping, ...]

    @property
    def required_array_keys(self) -> tuple[str, ...]:
        keys: set[str] = {self.observation_time_key}
        for group in self.groups:
            keys.update(term.key for term in group.action)
            keys.update(term.key for term in group.state)
            for key in (
                group.action_time_key,
                group.state_time_key,
                group.world_interval_index_key,
            ):
                if key is not None:
                    keys.add(key)
        return tuple(sorted(keys))


def _term(value: Mapping[str, Any]) -> MappingTerm:
    required = {"key", "columns", "scale", "offset"}
    if set(value) != required:
        raise AdapterContractError(
            f"mapping term keys mismatch: missing={sorted(required - set(value))} "
            f"unknown={sorted(set(value) - required)}"
        )
    columns = tuple(int(item) for item in value["columns"])
    if not columns or any(item < 0 for item in columns):
        raise AdapterContractError("mapping columns must be non-empty/non-negative")
    scale = tuple(float(item) for item in value["scale"])
    offset = tuple(float(item) for item in value["offset"])
    if len(scale) != len(columns) or len(offset) != len(columns):
        raise AdapterContractError("mapping scale/offset must match columns")
    if not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise AdapterContractError("mapping scale/offset contains NaN/Inf")
    return MappingTerm(str(value["key"]), columns, scale, offset)


def load_adapter_contract(path: Path, *, expected_sha256: str) -> AdapterContract:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise AdapterContractError(f"adapter contract is not a regular file: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise AdapterContractError(
            f"adapter contract SHA mismatch: {observed} != {expected_sha256}"
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "name",
        "raw_format",
        "observation_time_key",
        "views",
        "groups",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AdapterContractError("adapter contract root fields mismatch")
    schema = str(value["schema"])
    if schema not in {
        ADAPTER_SCHEMA,
        ADAPTER_COLOR_SCHEMA,
        ADAPTER_CANONICAL_SCHEMA,
    }:
        raise AdapterContractError(f"unsupported adapter schema {schema!r}")
    raw_format = str(value["raw_format"])
    if raw_format not in {"lerobot_parquet_video", "agibot_parquet_video", "npz"}:
        raise AdapterContractError(f"unsupported raw_format {raw_format!r}")
    observation_time_key = str(value["observation_time_key"])
    if not observation_time_key:
        raise AdapterContractError("observation_time_key cannot be empty")
    raw_views = value["views"]
    if not isinstance(raw_views, list) or not raw_views:
        raise AdapterContractError("adapter must declare at least one RGB view")
    views: list[ViewMapping] = []
    view_names: set[str] = set()
    view_keys: set[str] = set()
    for raw_view in raw_views:
        view_fields = (
            {"name", "key"}
            if schema == ADAPTER_SCHEMA
            else {"name", "key", "color_order"}
        )
        if not isinstance(raw_view, dict) or set(raw_view) != view_fields:
            raise AdapterContractError(
                f"adapter view fields must be exactly {sorted(view_fields)}"
            )
        name = str(raw_view["name"])
        key = str(raw_view["key"])
        if not name or not key:
            raise AdapterContractError("adapter view name/key cannot be empty")
        if name in view_names or key in view_keys:
            raise AdapterContractError("adapter view names/keys must be unique")
        view_names.add(name)
        view_keys.add(key)
        color_order = str(raw_view.get("color_order", "rgb"))
        if color_order not in {"rgb", "bgr"}:
            raise AdapterContractError(
                f"adapter view color_order must be rgb/bgr, got {color_order!r}"
            )
        views.append(ViewMapping(name=name, key=key, color_order=color_order))
    groups: list[GroupMapping] = []
    seen: set[str] = set()
    for raw in value["groups"]:
        fields = {
            "group",
            "supervision",
            "action",
            "state",
            "action_time_key",
            "state_time_key",
            "world_interval_index_key",
        }
        if schema == ADAPTER_CANONICAL_SCHEMA:
            fields.update(
                {
                    "action_transform",
                    "state_transform",
                    "action_value_mask",
                    "state_value_mask",
                }
            )
        if not isinstance(raw, dict) or set(raw) != fields:
            raise AdapterContractError("adapter group fields mismatch")
        name = str(raw["group"])
        if not name or name in seen:
            raise AdapterContractError(f"duplicate/empty adapter group {name!r}")
        seen.add(name)
        supervision = str(raw["supervision"])
        if supervision not in {"fine_command", "coarse_effect"}:
            raise AdapterContractError(f"invalid supervision {supervision!r}")
        action_time = raw["action_time_key"]
        interval_key = raw["world_interval_index_key"]
        if supervision == "fine_command" and not action_time:
            raise AdapterContractError("fine command group requires action_time_key")
        if supervision == "coarse_effect" and not interval_key:
            raise AdapterContractError(
                "coarse effect group requires world_interval_index_key"
            )
        action_terms = tuple(_term(item) for item in raw["action"])
        state_terms = tuple(_term(item) for item in raw["state"])
        if not action_terms:
            raise AdapterContractError(f"group {name!r} has no action mapping")
        action_transform = str(raw.get("action_transform", "identity"))
        state_transform = str(raw.get("state_transform", "identity"))
        if (
            action_transform != "identity"
            and action_transform not in CANONICAL_ACTION_TRANSFORMS
        ):
            raise AdapterContractError(
                f"unsupported canonical action transform {action_transform!r}"
            )
        if (
            state_transform != "identity"
            and state_transform not in CANONICAL_STATE_TRANSFORMS
        ):
            raise AdapterContractError(
                f"unsupported canonical state transform {state_transform!r}"
            )
        if action_transform != "identity" and state_transform == "identity":
            raise AdapterContractError(
                "canonical action transforms require an explicit canonical state transform"
            )
        if state_transform == "zero" and (
            state_terms or raw["state_time_key"] is not None
        ):
            raise AdapterContractError(
                "zero state transform must not map or timestamp fabricated state"
            )
        action_value_mask = (
            None
            if raw.get("action_value_mask") is None
            else tuple(bool(item) for item in raw["action_value_mask"])
        )
        state_value_mask = (
            None
            if raw.get("state_value_mask") is None
            else tuple(bool(item) for item in raw["state_value_mask"])
        )
        if action_transform != "identity" and (
            action_value_mask is None or len(action_value_mask) != 7
        ):
            raise AdapterContractError(
                "canonical action_value_mask must declare exactly seven dimensions"
            )
        if state_transform == "zero" and state_value_mask not in {None, ()}:
            raise AdapterContractError(
                "zero state transform cannot declare a value mask"
            )
        if state_transform not in {"identity", "zero"} and (
            state_value_mask is None or len(state_value_mask) != 10
        ):
            raise AdapterContractError(
                "canonical state_value_mask must declare exactly ten dimensions"
            )
        groups.append(
            GroupMapping(
                group=name,
                supervision=supervision,
                action=action_terms,
                state=state_terms,
                action_time_key=None if action_time is None else str(action_time),
                state_time_key=(
                    None
                    if raw["state_time_key"] is None
                    else str(raw["state_time_key"])
                ),
                world_interval_index_key=(
                    None if interval_key is None else str(interval_key)
                ),
                action_transform=action_transform,
                state_transform=state_transform,
                action_value_mask=action_value_mask,
                state_value_mask=state_value_mask,
            )
        )
    return AdapterContract(
        path,
        observed,
        str(value["name"]),
        raw_format,
        observation_time_key,
        tuple(views),
        tuple(groups),
    )


def _mapped(accessor: EpisodeAccessor, terms: Sequence[MappingTerm]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    row_count: Optional[int] = None
    for term in terms:
        source = np.asarray(accessor.array(term.key))
        if source.ndim == 1:
            source = source[:, None]
        if source.ndim != 2:
            raise AdapterContractError(f"raw field {term.key!r} must be [N,D]")
        if max(term.columns) >= source.shape[1]:
            raise AdapterContractError(
                f"raw field {term.key!r} has {source.shape[1]} columns, "
                f"mapping requests {max(term.columns)}"
            )
        selected = source[:, term.columns].astype(np.float32, copy=False)
        selected = selected * np.asarray(term.scale, np.float32)
        selected = selected + np.asarray(term.offset, np.float32)
        if row_count is None:
            row_count = selected.shape[0]
        elif selected.shape[0] != row_count:
            raise AdapterContractError("mapped raw fields have different row counts")
        pieces.append(selected)
    if not pieces:
        raise AdapterContractError("mapping contains no terms")
    result = np.concatenate(pieces, axis=1)
    if not np.isfinite(result).all():
        raise AdapterContractError("mapped values contain NaN/Inf")
    return result


def _pad7(value: np.ndarray) -> np.ndarray:
    if value.shape[1] >= 7:
        return value[:, :7]
    return np.concatenate(
        (value, np.zeros((len(value), 7 - value.shape[1]), dtype=np.float32)),
        axis=1,
    )


def _rpy_xyz_to_matrix(rpy: np.ndarray) -> np.ndarray:
    values = np.asarray(rpy, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(values, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def _euler_zxy_to_matrix(euler: np.ndarray) -> np.ndarray:
    values = np.asarray(euler, dtype=np.float64)
    z, x, y = np.moveaxis(values, -1, 0)
    cz, sz = np.cos(z), np.sin(z)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    result = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cy * cz + sy * sx * sz
    result[..., 0, 1] = -cy * sz + sy * sx * cz
    result[..., 0, 2] = sy * cx
    result[..., 1, 0] = cx * sz
    result[..., 1, 1] = cx * cz
    result[..., 1, 2] = -sx
    result[..., 2, 0] = -sy * cz + cy * sx * sz
    result[..., 2, 1] = sy * sz + cy * sx * cz
    result[..., 2, 2] = cy * cx
    return result


def _quat_xyzw_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-12):
        raise AdapterContractError("state quaternion has zero norm")
    x, y, z, w = np.moveaxis(value / norm, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
    output = np.empty((len(values), 3, 3), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    for index, vector in enumerate(values):
        angle = float(np.linalg.norm(vector))
        x, y, z = vector
        skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
        if angle < 1.0e-8:
            a = 1.0 - angle * angle / 6.0
            b = 0.5 - angle * angle / 24.0
        else:
            a = np.sin(angle) / angle
            b = (1.0 - np.cos(angle)) / (angle * angle)
        output[index] = identity + a * skew + b * (skew @ skew)
    return output.reshape(np.asarray(rotvec).shape[:-1] + (3, 3))


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64).reshape(-1, 3, 3)
    output = np.empty((len(values), 3), dtype=np.float64)
    for index, rotation in enumerate(values):
        trace = float(np.trace(rotation))
        angle = float(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        vector = np.asarray(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ),
            dtype=np.float64,
        )
        if angle < 1.0e-7:
            output[index] = 0.5 * vector
        elif np.pi - angle < 1.0e-5:
            diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
            axis = np.sqrt(diagonal)
            dominant = int(np.argmax(axis))
            if dominant == 0:
                axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
                axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
            elif dominant == 1:
                axis[0] = np.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
                axis[2] = np.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
            else:
                axis[0] = np.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
                axis[1] = np.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
            output[index] = axis / max(float(np.linalg.norm(axis)), 1.0e-12) * angle
        else:
            output[index] = vector * (angle / (2.0 * np.sin(angle)))
    return output.reshape(np.asarray(matrix).shape[:-2] + (3,)).astype(np.float32)


def _canonical_state7(raw: np.ndarray, transform: str) -> np.ndarray:
    state = np.asarray(raw, dtype=np.float32)
    zero = np.zeros((len(state), 1), dtype=np.float32)
    if transform == "pos_euler_8d":
        result = np.concatenate((state[:, :6], state[:, 7:8]), axis=1)
    elif transform == "pos_euler_7d":
        result = _pad7(state)
    elif transform == "pos_quat":
        result = np.concatenate(
            (state[:, :3], _quat_xyzw_to_rpy(state[:, 3:7]), state[:, 7:8]),
            axis=1,
        )
    elif transform == "pos_quat_no_grip":
        result = np.concatenate(
            (state[:, :3], _quat_xyzw_to_rpy(state[:, 3:7]), zero), axis=1
        )
    elif transform == "droid_state":
        result = np.concatenate((state[:, :6], state[:, 6:7]), axis=1)
    elif transform == "austin_buds":
        result = np.concatenate((state[:, :6], state[:, 7:8]), axis=1)
    elif transform == "nyu_franka":
        result = np.concatenate((state[:, -6:], zero), axis=1)
    elif transform == "cmu_stretch":
        result = np.concatenate(
            (state[:, :3], np.zeros((len(state), 2), np.float32), state[:, 3:4], zero),
            axis=1,
        )
    elif transform == "language_table":
        result = np.concatenate(
            (state[:, :2], np.zeros((len(state), 5), np.float32)), axis=1
        )
    elif transform == "robocasa":
        aperture = np.abs(state[:, 14:15] - state[:, 15:16])
        if np.any(aperture > 0.120001):
            raise AdapterContractError(
                "RoboCasa Panda gripper aperture exceeds the audited 0.12 m envelope"
            )
        close01 = 1.0 - np.clip(aperture / 0.08, 0.0, 1.0)
        result = np.concatenate(
            (
                state[:, 7:10],
                _quat_xyzw_to_rpy(state[:, 10:14]),
                close01,
            ),
            axis=1,
        )
    else:
        raise AdapterContractError(
            f"unsupported canonical state transform {transform!r}"
        )
    if result.shape != (len(state), 7) or not np.isfinite(result).all():
        raise AdapterContractError("canonical state transform produced invalid values")
    return result.astype(np.float32, copy=False)


def _state7_to_v8(state: np.ndarray) -> np.ndarray:
    rotation = _rpy_xyz_to_matrix(state[:, 3:6])
    rotation6d = np.swapaxes(rotation[:, :, :2], 1, 2).reshape(len(state), 6)
    return np.concatenate((state[:, :3], rotation6d, state[:, 6:7]), axis=1).astype(
        np.float32, copy=False
    )


def _canonical_action(
    raw: np.ndarray,
    transform: str,
    *,
    state7: Optional[np.ndarray],
) -> np.ndarray:
    action = np.asarray(raw, dtype=np.float32)
    if state7 is not None and len(state7) != len(action):
        raise AdapterContractError(
            "action/state row counts differ during canonicalization"
        )
    if transform == "droid_target":
        if state7 is None:
            raise AdapterContractError("DROID target conversion requires current state")
        dpos = action[:, :3] - state7[:, :3]
        drot = _matrix_to_rotvec(
            _rpy_xyz_to_matrix(action[:, 3:6])
            @ np.swapaxes(_rpy_xyz_to_matrix(state7[:, 3:6]), -1, -2)
        )
        # DROID stores gripper_position as an open fraction.  The V8 grouped
        # action ABI uses absolute_gripper_close01, so invert it at the source
        # boundary instead of asking the shared head to learn source polarity.
        result = np.concatenate((dpos, drot, 1.0 - action[:, 6:7]), axis=1)
    elif transform == "state_euler_residual":
        if state7 is None:
            raise AdapterContractError("Bridge conversion requires current state")
        target = state7[:, 3:6] + action[:, 3:6]
        drot = _matrix_to_rotvec(
            _rpy_xyz_to_matrix(target)
            @ np.swapaxes(_rpy_xyz_to_matrix(state7[:, 3:6]), -1, -2)
        )
        result = np.concatenate((action[:, :3], drot, 1.0 - action[:, 6:7]), axis=1)
    elif transform == "axis_angle_residual":
        if state7 is None:
            raise AdapterContractError("BC-Z conversion requires current state")
        current = _rotvec_to_matrix(state7[:, 3:6])
        target = _rotvec_to_matrix(state7[:, 3:6] + action[:, 3:6])
        result = np.concatenate(
            (
                action[:, :3],
                _matrix_to_rotvec(target @ np.swapaxes(current, -1, -2)),
                1.0 - action[:, 6:7],
            ),
            axis=1,
        )
    elif transform == "furniture_bench":
        if state7 is None:
            raise AdapterContractError(
                "FurnitureBench conversion requires current state"
            )
        current = _rpy_xyz_to_matrix(state7[:, 3:6])
        local = _rpy_xyz_to_matrix(action[:, 3:6])
        result = np.concatenate(
            (
                action[:, :3],
                _matrix_to_rotvec(current @ local @ np.swapaxes(current, -1, -2)),
                1.0 - action[:, 6:7],
            ),
            axis=1,
        )
    elif transform in {"base_rpy_open", "base_rpy_close"}:
        grip = action[:, 6:7] if transform.endswith("close") else 1.0 - action[:, 6:7]
        result = np.concatenate(
            (
                action[:, :3],
                _matrix_to_rotvec(_rpy_xyz_to_matrix(action[:, 3:6])),
                grip,
            ),
            axis=1,
        )
    elif transform == "taco_world":
        result = np.concatenate(
            (
                action[:, :3] / 50.0,
                _matrix_to_rotvec(_rpy_xyz_to_matrix(action[:, 3:6] / 20.0)),
                1.0 - action[:, 6:7],
            ),
            axis=1,
        )
    elif transform in {"euler_xyz_open", "euler_xyz_no_grip", "euler_zxy_open"}:
        matrix = (
            _euler_zxy_to_matrix(action[:, 3:6])
            if transform == "euler_zxy_open"
            else _rpy_xyz_to_matrix(action[:, 3:6])
        )
        grip = (
            1.0 - action[:, 6:7]
            if transform.endswith("open")
            else np.zeros((len(action), 1), np.float32)
        )
        result = np.concatenate(
            (action[:, :3], _matrix_to_rotvec(matrix), grip), axis=1
        )
    elif transform == "open_to_close":
        result = _pad7(action).copy()
        result[:, 6] = 1.0 - result[:, 6]
    elif transform == "signed_to_close":
        result = _pad7(action).copy()
        result[:, 6] = (result[:, 6] + 1.0) * 0.5
    elif transform == "xyz_only":
        result = np.concatenate(
            (action[:, :3], np.zeros((len(action), 4), np.float32)), axis=1
        )
    elif transform == "nyu_franka":
        eef = action[:, -8:-2]
        result = np.concatenate(
            (
                eef[:, :3],
                _matrix_to_rotvec(_rpy_xyz_to_matrix(eef[:, 3:6])),
                # NYU uses a signed/open command in this slot.  Upstream OXE
                # canonicalization clips it to {0, 1} before converting open
                # to close; without the clip raw -1 becomes an invalid 2.
                1.0 - np.clip(action[:, -2:-1], 0.0, 1.0),
            ),
            axis=1,
        )
    elif transform == "drop_last":
        result = _pad7(action[:, :-1])
    elif transform == "xy_only":
        result = np.concatenate(
            (action[:, :2], np.zeros((len(action), 5), np.float32)), axis=1
        )
    elif transform == "robocasa":
        result = _pad7(action[:, 5:12] if action.shape[1] >= 12 else action).copy()
        result[:, 6] = np.clip((result[:, 6] + 1.0) * 0.5, 0.0, 1.0)
    else:
        raise AdapterContractError(
            f"unsupported canonical action transform {transform!r}"
        )
    result = _pad7(np.asarray(result, dtype=np.float32))
    if not np.isfinite(result).all():
        raise AdapterContractError("canonical action transform produced NaN/Inf")
    return result


def _constant_mask(
    mask: Optional[tuple[bool, ...]], rows: int, width: int
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if len(mask) != width:
        raise AdapterContractError(
            f"value mask width {len(mask)} != transformed width {width}"
        )
    return np.broadcast_to(np.asarray(mask, dtype=np.bool_), (rows, width)).copy()


def adapt_action_series(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
) -> list[RawActionSeries]:
    """Decode complete source-native action series without windowing them."""

    specs = {group.name: group for group in embodiment.groups}
    if set(specs) != {group.group for group in contract.groups}:
        raise AdapterContractError(
            "adapter groups do not exactly match embodiment groups"
        )
    actions: list[RawActionSeries] = []
    for mapping in contract.groups:
        spec = specs[mapping.group]
        action = _mapped(accessor, mapping.action)
        if mapping.action_transform != "identity":
            raw_state = _mapped(accessor, mapping.state) if mapping.state else None
            state7 = (
                None
                if raw_state is None
                else _canonical_state7(raw_state, mapping.state_transform)
            )
            action = _canonical_action(
                action,
                mapping.action_transform,
                state7=state7,
            )
        if action.shape[1] != spec.action_dim:
            raise AdapterContractError(
                f"group {mapping.group!r} action dim {action.shape[1]} != {spec.action_dim}"
            )
        timestamps: Optional[np.ndarray] = None
        interval_indices: Optional[np.ndarray] = None
        if mapping.supervision == "fine_command":
            timestamps = np.asarray(
                accessor.array(mapping.action_time_key), dtype=np.float64
            ).reshape(-1)
            if timestamps.shape != (len(action),):
                raise AdapterContractError("action timestamp cardinality mismatch")
        else:
            interval_indices = np.asarray(
                accessor.array(mapping.world_interval_index_key), dtype=np.int64
            ).reshape(-1)
            if interval_indices.shape != (len(action),):
                raise AdapterContractError("coarse interval cardinality mismatch")
        actions.append(
            RawActionSeries(
                mapping.group,
                mapping.supervision,  # type: ignore[arg-type]
                action,
                timestamps_s=timestamps,
                world_interval_indices=interval_indices,
                value_mask=_constant_mask(
                    mapping.action_value_mask, len(action), action.shape[1]
                ),
            )
        )
    return actions


def adapt_current_state(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
    policy_chunk_start_s: float,
) -> list[RawStateSnapshot]:
    """Read measured state at the exact first policy-command timestamp."""

    specs = {group.name: group for group in embodiment.groups}
    if set(specs) != {group.group for group in contract.groups}:
        raise AdapterContractError(
            "adapter groups do not exactly match embodiment groups"
        )
    states: list[RawStateSnapshot] = []
    series_by_group = {
        item.group: item
        for item in adapt_state_series(
            accessor=accessor, contract=contract, embodiment=embodiment
        )
    }
    for mapping in contract.groups:
        spec = specs[mapping.group]

        # A mode/trigger group may genuinely have no measured state.  Keep it
        # as a real action group, but never fabricate a zero state token.
        if spec.state_dim == 0:
            if mapping.state:
                raise AdapterContractError(
                    f"group {mapping.group!r} declares no state semantics but "
                    "its adapter maps state fields"
                )
            if mapping.state_time_key is not None:
                raise AdapterContractError(
                    f"group {mapping.group!r} declares no state semantics but "
                    "its adapter maps a state time key"
                )
            continue
        raw = series_by_group[mapping.group]
        state = raw.values
        state_times = raw.timestamps_s
        exact = np.flatnonzero(state_times == np.float64(policy_chunk_start_s))
        if len(exact) != 1:
            raise AdapterContractError(
                f"group {mapping.group!r} has {len(exact)} exact current-state matches; "
                "interpolation/nearest fallback is forbidden"
            )
        row = int(exact[0])
        if state.shape[1] != spec.state_dim:
            raise AdapterContractError(
                f"group {mapping.group!r} state dim {state.shape[1]} != {spec.state_dim}"
            )
        states.append(
            RawStateSnapshot(
                mapping.group,
                float(state_times[row]),
                state[row],
                value_mask=(None if raw.value_mask is None else raw.value_mask[row]),
            )
        )
    return states


def adapt_state_series(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
) -> list[RawStateSeries]:
    """Decode all measured state rows; stateless action groups remain absent."""

    specs = {group.name: group for group in embodiment.groups}
    if set(specs) != {group.group for group in contract.groups}:
        raise AdapterContractError(
            "adapter groups do not exactly match embodiment groups"
        )
    output: list[RawStateSeries] = []
    for mapping in contract.groups:
        spec = specs[mapping.group]
        if spec.state_dim == 0:
            if mapping.state or mapping.state_time_key is not None:
                raise AdapterContractError(
                    f"group {mapping.group!r} declares no state semantics but maps state"
                )
            continue
        if not mapping.state or mapping.state_time_key is None:
            raise AdapterContractError(
                f"group {mapping.group!r} requires measured state and state_time_key"
            )
        values = _mapped(accessor, mapping.state)
        if mapping.state_transform != "identity":
            values = _state7_to_v8(_canonical_state7(values, mapping.state_transform))
        if values.shape[1] != spec.state_dim:
            raise AdapterContractError(
                f"group {mapping.group!r} state dim {values.shape[1]} != {spec.state_dim}"
            )
        timestamps = np.asarray(
            accessor.array(mapping.state_time_key), dtype=np.float64
        ).reshape(-1)
        if timestamps.shape != (len(values),):
            raise AdapterContractError("state timestamp cardinality mismatch")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise AdapterContractError(
                "state timestamps must be finite/strictly increasing"
            )
        output.append(
            RawStateSeries(
                mapping.group,
                values,
                timestamps,
                value_mask=_constant_mask(
                    mapping.state_value_mask, len(values), values.shape[1]
                ),
            )
        )
    return output


def adapt_robot_signals(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
    policy_chunk_start_s: float,
) -> tuple[list[RawActionSeries], list[RawStateSnapshot]]:
    """Compatibility wrapper for callers that need both complete actions and state."""

    return (
        adapt_action_series(
            accessor=accessor,
            contract=contract,
            embodiment=embodiment,
        ),
        adapt_current_state(
            accessor=accessor,
            contract=contract,
            embodiment=embodiment,
            policy_chunk_start_s=policy_chunk_start_s,
        ),
    )
