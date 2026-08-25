#!/usr/bin/env python3
"""Materialize the already-downloaded GAM/OXE + RoboCasa WM3D canary mix.

This is deliberately a local-reuse materializer: it does not download data or
build a full visual cache.  It restores the audited GAM/OXE canonical action
transforms and the audited grouped RoboCasa adapter before emitting a WM3D
data-profile.  Unsupported source semantics fail closed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

import av
from av.error import FFmpegError
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from wm3d.data.episode_io import EpisodeIOError, select_episode_cache_rows
from wm3d.data.manifest_contract import sha256_file
from wm3d.data.source_inventory import (
    _existing_relative,
    _format,
    _normalize_episode_ranges,
    _path_values,
    deterministic_split,
)
from wm3d.data.window_selection import (
    WindowSelectionError,
    select_observed_world_window,
)


@dataclass(frozen=True)
class SourcePlan:
    name: str
    root_kind: str
    relative_root: str
    weight: int
    views: tuple[str, ...]
    action_transform: str
    state_transform: str
    action_keys: tuple[str, ...] = ("action",)
    state_keys: tuple[str, ...] = ("observation.state",)
    action_mask: tuple[bool, ...] = (True,) * 7
    bgr_views: tuple[str, ...] = ()


def _truthy(value: object) -> bool:
    return value is not None and str(value).strip().lower() not in {
        "",
        "none",
        "null",
        "nan",
        "not provided",
    }


def _droid_keep_ranges(joint_velocity: np.ndarray) -> list[list[int]]:
    """Match the released GAM DROID non-idle contract exactly."""

    velocity = np.asarray(joint_velocity, dtype=np.float32)
    if velocity.ndim != 2 or not len(velocity):
        return []
    idle = np.concatenate(
        (
            np.array([False]),
            np.all(np.abs(np.diff(velocity, axis=0)) < 1.0e-3, axis=1),
        )
    )
    padded = np.concatenate((np.array([False]), idle, np.array([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    keep = np.ones(len(velocity), dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        if end - start >= 7:
            keep[start:end] = False
    padded = np.concatenate((np.array([False]), keep, np.array([False])))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        [int(start), int(end - 10)]
        for start, end in zip(starts, ends, strict=True)
        if end - start >= 16 and end - 10 > start
    ]


def _fixed_list_array(table: object, key: str) -> np.ndarray:
    array = table[key].combine_chunks()
    lengths = np.asarray(
        pc.list_value_length(array).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    if not len(lengths):
        return np.empty((0, 0), dtype=np.float32)
    if not np.all(lengths == lengths[0]):
        raise RuntimeError(f"{key} has variable-length rows")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(len(lengths), int(lengths[0]))


def _compute_droid_nonidle_ranges(root: Path) -> dict[int, tuple[tuple[int, int], ...]]:
    columns = (
        "action.joint_velocity",
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
        "is_episode_successful",
        "reward",
    )
    ranges: dict[int, tuple[tuple[int, int], ...]] = {}
    info = json.loads(_regular(root / "meta/info.json", "DROID info").read_text())
    data_template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    metadata_rows = list(_episode_candidates(root))
    if not metadata_rows:
        raise RuntimeError(f"DROID has no episode metadata under {root}")

    file_origins: dict[tuple[int, int, str], int] = {}
    seen_indices: set[int] = set()
    for row in metadata_rows:
        index = int(row["episode_index"])
        if index in seen_indices:
            raise RuntimeError(f"DROID has duplicate episode metadata for {index}")
        seen_indices.add(index)
        if row.get("data/from_index") is not None:
            continue
        values = _path_values(row, index)
        key = (
            values["chunk_index"],
            values["file_index"],
            str(row.get("data_path", "")),
        )
        dataset_start = int(row.get("dataset_from_index", 0))
        file_origins[key] = min(file_origins.get(key, dataset_start), dataset_start)

    payload_episodes: dict[Path, list[tuple[int, int, int]]] = {}
    for row in metadata_rows:
        index = int(row["episode_index"])
        length = int(row.get("length", row.get("episode_length", 0)))
        if length < 2:
            continue
        values = _path_values(row, index)
        candidates = []
        if row.get("data_path"):
            candidates.append(str(row["data_path"]))
        candidates.extend(
            [
                _format(data_template, values),
                f"data/chunk-{values['chunk_index']:03d}/episode_{index:06d}.parquet",
                f"data/chunk-{values['chunk_index']:03d}/file-{values['file_index']:03d}.parquet",
            ]
        )
        payload = root / _existing_relative(root, candidates)
        explicit_start = row.get("data/from_index")
        explicit_stop = row.get("data/to_index")
        if (explicit_start is None) != (explicit_stop is None):
            raise RuntimeError(f"DROID episode {index} has partial row bounds")
        if explicit_start is not None:
            row_start, row_stop = int(explicit_start), int(explicit_stop)
        else:
            dataset_start = int(row.get("dataset_from_index", 0))
            dataset_stop = int(row.get("dataset_to_index", dataset_start + length))
            key = (
                values["chunk_index"],
                values["file_index"],
                str(row.get("data_path", "")),
            )
            row_start = dataset_start - file_origins[key]
            row_stop = dataset_stop - file_origins[key]
        if row_start < 0 or row_stop - row_start != length:
            raise RuntimeError(f"DROID episode {index} has invalid payload row slice")
        payload_episodes.setdefault(payload, []).append((index, row_start, row_stop))

    for path, episodes in sorted(payload_episodes.items()):
        table = pq.read_table(_regular(path, "DROID payload"), columns=list(columns))
        if max(stop for _, _, stop in episodes) > table.num_rows:
            raise RuntimeError(f"DROID episode slice exceeds payload {path}")
        velocity = _fixed_list_array(table, "action.joint_velocity")
        language = [table[key].combine_chunks().to_pylist() for key in columns[1:4]]
        successful = np.asarray(
            table["is_episode_successful"].to_numpy(zero_copy_only=False), dtype=bool
        )
        reward = np.asarray(
            table["reward"].to_numpy(zero_copy_only=False), dtype=np.float32
        )
        for index, start, stop in episodes:
            has_language = any(
                _truthy(values[position])
                for values in language
                for position in range(start, stop)
            )
            has_success = bool(
                successful[start:stop].any() or np.abs(reward[start:stop]).any()
            )
            if not has_language or not has_success:
                continue
            keep = _droid_keep_ranges(velocity[start:stop])
            if keep:
                ranges[index] = tuple((item[0], item[1]) for item in keep)
    if not ranges:
        raise RuntimeError("DROID non-idle scan produced no usable episodes")
    return ranges


OXE = (
    SourcePlan(
        "oxe_bridge",
        "oxe",
        "BrunoM42/bridge_orig_lerobot",
        224,
        ("observation.images.image_0", "observation.images.image_1"),
        "state_euler_residual",
        "pos_euler_8d",
    ),
    SourcePlan(
        "oxe_droid",
        "droid",
        ".",
        303,
        (
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
            "observation.images.exterior_1_left",
        ),
        "droid_target",
        "droid_state",
        action_keys=("action.cartesian_position", "action.gripper_position"),
        state_keys=(
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
        ),
    ),
    SourcePlan(
        "oxe_taco_play",
        "oxe",
        "lerobot/taco_play",
        60,
        ("observation.images.rgb_static", "observation.images.rgb_gripper"),
        "taco_world",
        "pos_euler_7d",
    ),
    SourcePlan(
        "oxe_utaustin_mutex",
        "oxe",
        "lerobot/utaustin_mutex",
        39,
        ("observation.images.image", "observation.images.wrist_image"),
        "open_to_close",
        "pos_euler_8d",
        bgr_views=("observation.images.image", "observation.images.wrist_image"),
    ),
    SourcePlan(
        "oxe_stanford_hydra",
        "oxe",
        "lerobot/stanford_hydra_dataset",
        24,
        ("observation.images.image", "observation.images.wrist_image"),
        "euler_xyz_open",
        "pos_euler_8d",
        bgr_views=("observation.images.image", "observation.images.wrist_image"),
    ),
    SourcePlan(
        "oxe_berkeley_autolab_ur5",
        "oxe",
        "lerobot/berkeley_autolab_ur5",
        32,
        ("observation.images.image", "observation.images.hand_image"),
        "base_rpy_open",
        "pos_quat",
        bgr_views=("observation.images.hand_image",),
    ),
    SourcePlan(
        "oxe_austin_sailor",
        "oxe",
        "lerobot/austin_sailor_dataset",
        15,
        ("observation.images.image", "observation.images.wrist_image"),
        "open_to_close",
        "pos_quat",
        action_mask=(True, True, True, False, False, True, True),
    ),
    SourcePlan(
        "oxe_austin_sirius",
        "oxe",
        "lerobot/austin_sirius_dataset",
        24,
        ("observation.images.image", "observation.images.wrist_image"),
        "open_to_close",
        "pos_euler_8d",
        action_mask=(True, True, True, False, False, True, True),
    ),
    SourcePlan(
        "oxe_berkeley_fanuc",
        "oxe",
        "lerobot/berkeley_fanuc_manipulation",
        20,
        ("observation.images.image", "observation.images.wrist_image"),
        "euler_xyz_no_grip",
        "pos_euler_8d",
        action_mask=(True, True, True, True, True, True, False),
        bgr_views=("observation.images.image", "observation.images.wrist_image"),
    ),
    SourcePlan(
        "oxe_jaco_play",
        "oxe",
        "lerobot/jaco_play",
        33,
        ("observation.images.image", "observation.images.image_wrist"),
        "open_to_close",
        "pos_quat",
        action_mask=(True, True, True, False, False, False, True),
    ),
    SourcePlan(
        "oxe_fmb",
        "oxe",
        "lerobot/fmb",
        42,
        ("observation.images.image_side_1", "observation.images.image_wrist_1"),
        "base_rpy_close",
        "pos_quat",
        bgr_views=(
            "observation.images.image_side_1",
            "observation.images.image_side_2",
            "observation.images.image_wrist_1",
            "observation.images.image_wrist_2",
        ),
    ),
    SourcePlan(
        "oxe_kuka",
        "oxe",
        "lerobot/stanford_kuka_multimodal_dataset",
        50,
        ("observation.images.image",),
        "xyz_only",
        "pos_quat_no_grip",
        action_mask=(True, True, True, False, False, False, False),
    ),
    SourcePlan(
        "oxe_fractal",
        "oxe",
        "BrunoM42/fractal20220817_data_lerobot",
        271,
        ("observation.images.image",),
        "base_rpy_open",
        "pos_quat",
    ),
    SourcePlan(
        "oxe_berkeley_cable",
        "oxe",
        "lerobot/berkeley_cable_routing",
        8,
        ("observation.images.image", "observation.images.wrist45_image"),
        "velocity_no_grip",
        "pos_quat",
        action_mask=(True, True, True, False, False, True, False),
    ),
    SourcePlan(
        "oxe_roboturk",
        "oxe",
        "lerobot/roboturk",
        20,
        ("observation.images.front_rgb",),
        "open_to_close",
        "zero",
        state_keys=(),
    ),
    SourcePlan(
        "oxe_dlr_edan",
        "oxe",
        "lerobot/dlr_edan_shared_control",
        1,
        ("observation.images.image",),
        "euler_zxy_open",
        "pos_euler_7d",
    ),
    SourcePlan(
        "oxe_austin_buds",
        "oxe",
        "lerobot/austin_buds_dataset",
        7,
        ("observation.images.image", "observation.images.wrist_image"),
        "signed_to_close",
        "austin_buds",
        action_mask=(True, True, True, False, False, False, True),
    ),
    SourcePlan(
        "oxe_nyu_franka",
        "oxe",
        "lerobot/nyu_franka_play_dataset",
        10,
        ("observation.images.image", "observation.images.image_additional_view"),
        "nyu_franka",
        "nyu_franka",
    ),
    SourcePlan(
        "oxe_nyu_door",
        "oxe",
        "lerobot/nyu_door_opening_surprising_effectiveness",
        10,
        ("observation.images.image",),
        "velocity_open",
        "zero",
        state_keys=(),
    ),
    SourcePlan(
        "oxe_cmu_stretch",
        "oxe",
        "lerobot/cmu_stretch",
        5,
        ("observation.images.image",),
        "drop_last",
        "cmu_stretch",
        action_mask=(True, False, True, False, False, False, False),
    ),
    SourcePlan(
        "oxe_furniture_bench",
        "oxe",
        "tailong-wu/furniture_bench_dataset_lerobot_v30",
        71,
        ("observation.images.image", "observation.images.wrist_image"),
        "furniture_bench",
        "pos_quat",
    ),
    SourcePlan(
        "oxe_bc_z",
        "oxe",
        "tailong-wu/bc_z_lerobot_v30",
        208,
        ("observation.images.image",),
        "axis_angle_residual",
        "pos_euler_8d",
    ),
    SourcePlan(
        "oxe_language_table",
        "oxe",
        "tailong-wu/language_table_lerobot_v30",
        100,
        ("observation.images.rgb",),
        "xy_only",
        "language_table",
        action_mask=(True, True, False, False, False, False, False),
    ),
)

ROBOCASA = (
    SourcePlan(
        "robocasa_atomic",
        "robocasa",
        "robocasa365-pretrain-atomic",
        7,
        (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
        ),
        "robocasa_grouped",
        "robocasa_grouped",
    ),
    SourcePlan(
        "robocasa_composite",
        "robocasa",
        "robocasa365-pretrain-composite",
        160,
        (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
        ),
        "robocasa_grouped",
        "robocasa_grouped",
    ),
    SourcePlan(
        "robocasa_mg",
        "robocasa",
        "robocasa365-pretrain-mg",
        52,
        (
            "observation.images.robot0_agentview_left",
            "observation.images.robot0_eye_in_hand",
        ),
        "robocasa_grouped",
        "robocasa_grouped",
    ),
)

# Exact 20 Hz source-controller scales from the three passed RoboCasa factual
# action audits.  The partitions use different controller gains; sharing the
# atomic scale silently changes the physical action represented by composite
# and MG rows.
ROBOCASA_ACTION_SCALES = {
    "robocasa_atomic": (
        (0.010797412134706974, 0.011273571290075779, 0.010780979879200459),
        (0.0976465567946434, 0.09016355127096176, 0.10008708387613297),
    ),
    "robocasa_composite": (
        (0.012059568427503109, 0.01184189785271883, 0.011475762352347374),
        (0.09586925804615021, 0.09779707342386246, 0.10560151189565659),
    ),
    "robocasa_mg": (
        (0.008660104125738144, 0.00796457938849926, 0.005903083365410566),
        (0.07884584367275238, 0.03510546311736107, 0.04012652486562729),
    ),
}

PLANS = OXE + ROBOCASA
VIEW_SLOTS = ("head", "left_wrist", "right_wrist")


def _real_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be an absolute real directory: {path}")
    return path.resolve(strict=True)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _root(plan: SourcePlan, *, oxe: Path, droid: Path, robocasa: Path) -> Path:
    base = {"oxe": oxe, "droid": droid, "robocasa": robocasa}[plan.root_kind]
    root = base if plan.relative_root == "." else base / plan.relative_root
    return _real_dir(root, plan.name)


def _width(features: dict, key: str, maximum: int) -> int:
    value = features.get(key)
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 1:
        raise RuntimeError(f"{key} must declare a one-dimensional shape")
    width = int(shape[0])
    if not 0 < width <= maximum:
        raise RuntimeError(f"{key} width {width} exceeds WM3D capacity {maximum}")
    return width


def _terms(features: dict, keys: tuple[str, ...], *, maximum: int = 64) -> list[dict]:
    output = []
    for key in keys:
        width = _width(features, key, maximum)
        output.append(
            {
                "key": key,
                "columns": list(range(width)),
                "scale": [1.0] * width,
                "offset": [0.0] * width,
            }
        )
    return output


def _oxe_adapter(plan: SourcePlan, info: dict) -> tuple[dict, dict]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise RuntimeError(f"{plan.name}: missing features")
    if plan.action_transform in {"velocity_no_grip", "velocity_open"}:
        raise RuntimeError(
            f"{plan.name}: velocity commands require an audited interval integration; "
            "exclude this source instead of treating nominal frequency as exact time"
        )
    missing = set(plan.views) - set(features)
    if missing:
        raise RuntimeError(
            f"{plan.name}: selected RGB views are missing: {sorted(missing)}"
        )
    action_terms = _terms(features, plan.action_keys)
    state_terms = (
        _terms(features, plan.state_keys) if plan.state_transform != "zero" else []
    )
    state_mask = (
        []
        if plan.state_transform == "zero"
        else [True] * 9
        + [
            plan.state_transform
            not in {"nyu_franka", "cmu_stretch", "pos_quat_no_grip"}
        ]
    )
    adapter = {
        "schema": "wm3d_source_adapter_v5",
        "name": f"{plan.name}_gam_canonical_arm",
        "raw_format": "lerobot_parquet_video",
        "observation_time_key": "timestamp",
        "views": [
            {
                "name": slot,
                "key": key,
                "color_order": "bgr" if key in plan.bgr_views else "rgb",
            }
            for slot, key in zip(VIEW_SLOTS, plan.views, strict=False)
        ],
        "groups": [
            {
                "group": "arm",
                "supervision": "fine_command",
                "action": action_terms,
                "state": state_terms,
                "action_time_key": "timestamp",
                "state_time_key": (
                    None if plan.state_transform == "zero" else "timestamp"
                ),
                "world_interval_index_key": None,
                "action_transform": plan.action_transform,
                "state_transform": plan.state_transform,
                "action_value_mask": list(plan.action_mask),
                "state_value_mask": state_mask,
            }
        ],
    }
    embodiment = {
        "name": f"{plan.name}_canonical_arm",
        "embodiment_id": 200 + PLANS.index(plan),
        "groups": [
            {
                "name": "arm",
                "group_id": 1,
                "action_semantics": (
                    ["delta_position_m"] * 3
                    + ["delta_rotation_axis_angle_rad"] * 3
                    + ["absolute_gripper_close01"]
                ),
                "state_semantics": (
                    []
                    if plan.state_transform == "zero"
                    else ["eef_position_m"] * 3
                    + ["eef_rotation_6d"] * 6
                    # OXE state grippers are heterogeneous physical positions
                    # (metres, radians, device units, and occasionally fractions).
                    # Keep the semantic honest so grouped continuous
                    # normalization is applied per source instead of treating the
                    # state as a universal close01 binary channel.
                    + ["controller_state"]
                ),
                "action_frame": "robot_base",
                "state_frame": (
                    "not_applicable" if plan.state_transform == "zero" else "robot_base"
                ),
                "composition_operators": (
                    ["sum"] * 3 + ["so3_axis_angle_base_left"] * 3 + ["logical_last"]
                ),
            }
        ],
    }
    return adapter, embodiment


def _robocasa_adapter(plan: SourcePlan, info: dict) -> tuple[dict, dict]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise RuntimeError(f"{plan.name}: missing features")
    if _width(features, "action", 16) != 12:
        raise RuntimeError(f"{plan.name}: audited RoboCasa action must be 12D")
    if _width(features, "observation.state", 32) != 16:
        raise RuntimeError(f"{plan.name}: audited RoboCasa state must be 16D")
    missing = set(plan.views) - set(features)
    if missing:
        raise RuntimeError(
            f"{plan.name}: selected RGB views are missing: {sorted(missing)}"
        )
    try:
        translation_scale, rotation_scale = ROBOCASA_ACTION_SCALES[plan.name]
    except KeyError as exc:
        raise RuntimeError(
            f"{plan.name}: no passed RoboCasa factual-action scale contract"
        ) from exc

    def term(
        key: str, columns: list[int], scale: list[float], offset: list[float]
    ) -> dict:
        return {"key": key, "columns": columns, "scale": scale, "offset": offset}

    adapter = {
        "schema": "wm3d_source_adapter_v5",
        "name": f"{plan.name}_panda_omron_audited",
        "raw_format": "lerobot_parquet_video",
        "observation_time_key": "timestamp",
        "views": [
            {"name": slot, "key": key, "color_order": "rgb"}
            for slot, key in zip(VIEW_SLOTS, plan.views, strict=False)
        ],
        "groups": [
            {
                "group": "base",
                "supervision": "fine_command",
                "action": [term("action", [0, 1, 2, 3], [1.0] * 4, [0.0] * 4)],
                "state": [
                    term("observation.state", list(range(7)), [1.0] * 7, [0.0] * 7)
                ],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
                "action_transform": "identity",
                "state_transform": "identity",
                "action_value_mask": None,
                "state_value_mask": None,
            },
            {
                "group": "mode",
                "supervision": "fine_command",
                "action": [term("action", [4], [0.5], [0.5])],
                "state": [],
                "action_time_key": "timestamp",
                "state_time_key": None,
                "world_interval_index_key": None,
                "action_transform": "identity",
                "state_transform": "identity",
                "action_value_mask": None,
                "state_value_mask": None,
            },
            {
                "group": "arm",
                "supervision": "fine_command",
                "action": [
                    term(
                        "action",
                        [5, 6, 7],
                        list(translation_scale),
                        [0.0] * 3,
                    ),
                    term(
                        "action",
                        [8, 9, 10],
                        list(rotation_scale),
                        [0.0] * 3,
                    ),
                    # Preserve the audited signed close-positive command here;
                    # the canonical transform below maps it once to close01.
                    term("action", [11], [1.0], [0.0]),
                ],
                "state": [
                    term(
                        "observation.state",
                        list(range(16)),
                        [1.0] * 16,
                        [0.0] * 16,
                    ),
                ],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
                "action_transform": "robocasa",
                "state_transform": "robocasa",
                "action_value_mask": [True] * 7,
                "state_value_mask": [True] * 10,
            },
        ],
    }
    embodiment = {
        # All RoboCasa partitions and downstream LIBERO share one physical
        # Panda ABI. Dataset identity remains in source_id and normalization;
        # it must not create three incompatible robot identities.
        "name": "panda_robocasa_libero",
        "embodiment_id": 2,
        "groups": [
            {
                "name": "base",
                "group_id": 10,
                "action_semantics": ["controller_command"] * 4,
                "state_semantics": ["base_pose"] * 7,
                "action_frame": "omron_controller_native",
                "state_frame": "robot_base_pose_position_quaternion_native",
                "composition_operators": ["last"] * 4,
            },
            {
                "name": "mode",
                "group_id": 11,
                "action_semantics": ["controller_mode"],
                "state_semantics": [],
                "action_frame": "arm_base_controller_mode_binary01",
                "state_frame": "not_applicable",
                "composition_operators": ["logical_last"],
            },
            {
                "name": "arm",
                "group_id": 12,
                "action_semantics": (
                    ["delta_position_m"] * 3
                    + ["delta_rotation_axis_angle_rad"] * 3
                    + ["absolute_gripper_close01"]
                ),
                "state_semantics": (
                    ["eef_position_m"] * 3
                    + ["eef_rotation_6d"] * 6
                    + ["gripper_close01"]
                ),
                "action_frame": "robot_base",
                "state_frame": "robot_base_relative_eef_position_quaternion_and_gripper_qpos",
                "composition_operators": (
                    ["sum"] * 3 + ["so3_axis_angle_base_left"] * 3 + ["logical_last"]
                ),
            },
        ],
    }
    return adapter, embodiment


def _adapter(plan: SourcePlan, info: dict) -> tuple[dict, dict]:
    return (
        _robocasa_adapter(plan, info)
        if plan.root_kind == "robocasa"
        else _oxe_adapter(plan, info)
    )


_EPISODE_METADATA_COLUMNS = (
    "episode_index",
    "length",
    "episode_length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "data/from_index",
    "data/to_index",
    "data_path",
)


def _episode_candidates(root: Path):
    jsonl = root / "meta/episodes.jsonl"
    if jsonl.is_file() and not jsonl.is_symlink():
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"episode metadata row is not an object: {jsonl}"
                    )
                yield row
        return
    paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise RuntimeError(f"no episode metadata under {root}")
    for path in paths:
        safe = _regular(path, "episode metadata")
        parquet = pq.ParquetFile(safe)
        columns = [
            name
            for name in _EPISODE_METADATA_COLUMNS
            if name in parquet.schema_arrow.names
        ]
        columns.extend(
            name
            for name in parquet.schema_arrow.names
            if name.startswith("videos/") and name not in columns
        )
        if "episode_index" not in columns or not {
            "length",
            "episode_length",
        }.intersection(columns):
            raise RuntimeError(f"episode metadata misses index/length: {safe}")
        for batch in parquet.iter_batches(columns=columns, batch_size=4096):
            yield from batch.to_pylist()


def _window_evidence(
    clock: np.ndarray, model_profile: dict, *, count_all: bool = True
) -> dict | None:
    sampling = model_profile["sampling"]
    model = model_profile["model"]
    try:
        selected_rows = select_episode_cache_rows(
            clock,
            minimum_separation_s=float(sampling["minimum_anchor_separation_seconds"]),
        )
    except EpisodeIOError:
        return None
    cached_clock = np.asarray(clock, dtype=np.float64)[selected_rows]
    first: dict | None = None
    valid_window_count = 0
    for anchor in range(len(cached_clock)):
        try:
            window = select_observed_world_window(
                cached_clock,
                anchor_index=anchor,
                context_samples=int(model["T"]),
                future_samples=int(model["K"]),
                context_horizon_s=float(sampling["context_horizon_seconds"]),
                future_horizon_s=float(sampling["future_horizon_seconds"]),
                minimum_horizon_coverage=float(sampling["minimum_horizon_coverage"]),
                future_offsets_s=sampling.get("future_offsets_seconds"),
            )
        except WindowSelectionError:
            continue
        valid_window_count += 1
        if first is None:
            first = {
                "raw_frame_count": int(len(clock)),
                "cache_frame_count": int(len(cached_clock)),
                "duration_s": float(cached_clock[-1] - cached_clock[0]),
                "first_valid_anchor_index": int(anchor),
                "context_coverage_s": float(
                    cached_clock[anchor] - cached_clock[window.context_indices[0]]
                ),
                "future_coverage_s": float(
                    cached_clock[window.future_indices[-1]] - cached_clock[anchor]
                ),
            }
            if not count_all:
                break
    if first is None:
        return None
    return {**first, "valid_window_count": valid_window_count}


def _segment_has_video_coverage(
    *,
    requested_start_s: float | None,
    requested_stop_s: float | None,
    available_start_s: float,
    available_stop_s: float,
    frame_count: int | None,
) -> bool:
    if frame_count is not None and frame_count < 2:
        return False
    available = np.asarray([available_start_s, available_stop_s], dtype=np.float64)
    if not bool(np.isfinite(available).all()) or available_stop_s <= available_start_s:
        return False
    if requested_start_s is None and requested_stop_s is None:
        return True
    if requested_start_s is None or requested_stop_s is None:
        return False
    requested = np.asarray([requested_start_s, requested_stop_s], dtype=np.float64)
    if not bool(np.isfinite(requested).all()) or requested_stop_s <= requested_start_s:
        return False
    tolerance_s = 1e-3
    return (
        requested_start_s >= available_start_s - tolerance_s
        and requested_stop_s <= available_stop_s + tolerance_s
    )


def _video_bounds(
    path: Path,
    cache: dict[Path, tuple[float, float, int | None] | None],
) -> tuple[float, float, int | None] | None:
    if path in cache:
        return cache[path]
    try:
        safe = _regular(path, "selected episode video")
        with av.open(str(safe), mode="r") as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            start_s = float((stream.start_time or 0) * time_base)
            if stream.duration is not None:
                duration_s = float(stream.duration * time_base)
            elif container.duration is not None:
                duration_s = float(container.duration) / float(av.time_base)
            else:
                cache[path] = None
                return None
            frames = int(stream.frames) if int(stream.frames or 0) > 0 else None
            result = (start_s, start_s + duration_s, frames)
    except (FFmpegError, OSError, ValueError, RuntimeError, IndexError):
        result = None
    cache[path] = result
    return result


def _episode_video_coverage(
    *,
    root: Path,
    row: dict,
    episode_index: int,
    view_keys: tuple[str, ...],
    video_template: str,
    cache: dict[Path, tuple[float, float, int | None] | None],
) -> bool:
    values = _path_values(row, episode_index)
    for view_key in view_keys:
        direct = row.get(f"videos/{view_key}/path")
        video_chunk_index = int(
            row.get(f"videos/{view_key}/chunk_index", values["chunk_index"])
        )
        video_file_index = int(
            row.get(f"videos/{view_key}/file_index", values["file_index"])
        )
        video_values = {
            **values,
            "video_key": view_key,
            "chunk_index": video_chunk_index,
            "file_index": video_file_index,
        }
        candidates = [str(direct)] if direct else []
        candidates.extend(
            [
                _format(video_template, video_values),
                (
                    "videos/chunk-"
                    f"{video_chunk_index:03d}/"
                    f"{view_key}/episode_{episode_index:06d}.mp4"
                ),
                (
                    f"videos/{view_key}/chunk-"
                    f"{video_chunk_index:03d}/file-{video_file_index:03d}.mp4"
                ),
            ]
        )
        try:
            relative = _existing_relative(root, candidates)
        except RuntimeError:
            return False
        bounds = _video_bounds(root / relative, cache)
        if bounds is None:
            return False
        if not _segment_has_video_coverage(
            requested_start_s=row.get(f"videos/{view_key}/from_timestamp"),
            requested_stop_s=row.get(f"videos/{view_key}/to_timestamp"),
            available_start_s=bounds[0],
            available_stop_s=bounds[1],
            frame_count=bounds[2],
        ):
            return False
    return True


def _selected_indices(
    source: str,
    root: Path,
    model_profile: dict,
    *,
    view_keys: tuple[str, ...],
    minimum_train_windows: int,
    minimum_eval_windows: int,
    select_all_usable: bool,
    episode_ranges: dict[int, tuple[tuple[int, int], ...]] | None = None,
) -> tuple[tuple[int, ...], dict[str, dict]]:
    info = json.loads(_regular(root / "meta/info.json", f"{source} info").read_text())
    data_template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    video_template = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        )
    )
    candidates: dict[str, list[tuple[int, int, dict]]] = {
        split: [] for split in ("train", "val", "test")
    }
    file_origins: dict[tuple[int, int, str], int] = {}
    for row in _episode_candidates(root):
        index = int(row["episode_index"])
        length = int(row.get("length", row.get("episode_length", 0)))
        if length < 2:
            continue
        ranges = None if episode_ranges is None else episode_ranges.get(index)
        if episode_ranges is not None and ranges is None:
            continue
        values = _path_values(row, index)
        key = (
            values["chunk_index"],
            values["file_index"],
            str(row.get("data_path", "")),
        )
        if row.get("data/from_index") is None:
            dataset_start = int(row.get("dataset_from_index", 0))
            file_origins[key] = min(file_origins.get(key, dataset_start), dataset_start)
        episode_id = f"{source}:{index:09d}"
        split = deterministic_split(
            source,
            episode_id,
            seed=3407,
            train_fraction=0.8,
            validation_fraction=0.1,
        )
        selection_length = (
            length if ranges is None else sum(stop - start for start, stop in ranges)
        )
        candidates[split].append((selection_length, index, row))

    selected: dict[str, list[int]] = {}
    evidence: dict[str, dict] = {}
    parquet_cache: dict[Path, pq.ParquetFile] = {}
    timestamp_cache: dict[Path, np.ndarray] = {}
    video_bounds_cache: dict[Path, tuple[float, float, int | None] | None] = {}
    for split in ("train", "val", "test"):
        target = minimum_train_windows if split == "train" else minimum_eval_windows
        selected[split] = []
        selected_evidence = []
        valid_window_count = 0
        for _selection_length, index, row in sorted(
            candidates[split], key=lambda item: (item[0], item[1]), reverse=True
        ):
            length = int(row.get("length", row.get("episode_length", 0)))
            values = _path_values(row, index)
            payload_candidates = []
            if row.get("data_path"):
                payload_candidates.append(str(row["data_path"]))
            payload_candidates.extend(
                [
                    _format(data_template, values),
                    (
                        "data/chunk-{chunk_index:03d}/"
                        "episode_{episode_index:06d}.parquet"
                    ).format(**values),
                    (
                        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                    ).format(**values),
                ]
            )
            try:
                relative = _existing_relative(root, payload_candidates)
            except RuntimeError:
                continue
            explicit_start = row.get("data/from_index")
            explicit_stop = row.get("data/to_index")
            if (explicit_start is None) != (explicit_stop is None):
                continue
            if explicit_start is not None:
                row_start, row_stop = int(explicit_start), int(explicit_stop)
            else:
                dataset_start = int(row.get("dataset_from_index", 0))
                dataset_stop = int(row.get("dataset_to_index", dataset_start + length))
                key = (
                    values["chunk_index"],
                    values["file_index"],
                    str(row.get("data_path", "")),
                )
                row_start = dataset_start - file_origins[key]
                row_stop = dataset_stop - file_origins[key]
            if row_start < 0 or row_stop - row_start != length:
                continue
            payload = root / relative
            parquet = parquet_cache.setdefault(payload, pq.ParquetFile(payload))
            try:
                all_timestamps = timestamp_cache.get(payload)
                if all_timestamps is None:
                    timestamp_column = parquet.read(columns=["timestamp"])[
                        "timestamp"
                    ].combine_chunks()
                    all_timestamps = timestamp_column.to_numpy(zero_copy_only=False)
                    timestamp_cache[payload] = all_timestamps
                if row_stop > len(all_timestamps):
                    continue
                clock = all_timestamps[row_start:row_stop]
            except (KeyError, ValueError):
                continue
            if len(clock) != length:
                continue
            if not _episode_video_coverage(
                root=root,
                row=row,
                episode_index=index,
                view_keys=view_keys,
                video_template=video_template,
                cache=video_bounds_cache,
            ):
                continue
            keep_ranges = (
                ((0, length),) if episode_ranges is None else episode_ranges[index]
            )
            segment_evidence = []
            episode_window_count = 0
            for range_start, range_stop in keep_ranges:
                if range_start < 0 or range_stop > length or range_stop <= range_start:
                    raise RuntimeError(
                        f"{source}: episode {index} has invalid non-idle range "
                        f"[{range_start}, {range_stop}) for length {length}"
                    )
                observed = _window_evidence(
                    clock[range_start:range_stop],
                    model_profile,
                    count_all=(
                        not select_all_usable
                        or valid_window_count + episode_window_count < target
                    ),
                )
                if observed is None:
                    continue
                episode_window_count += int(observed["valid_window_count"])
                if not select_all_usable:
                    segment_evidence.append(
                        {
                            "range_start": range_start,
                            "range_stop": range_stop,
                            **observed,
                        }
                    )
            if episode_window_count == 0:
                continue
            selected[split].append(index)
            if not select_all_usable:
                selected_evidence.append(
                    {
                        "episode_index": index,
                        "video_coverage": "container_bounds_verified",
                        "nonidle_segments": segment_evidence,
                        "valid_window_count": episode_window_count,
                    }
                )
            valid_window_count += episode_window_count
            if not select_all_usable and valid_window_count >= target:
                break
        if valid_window_count < target:
            raise RuntimeError(
                f"{source}: {split} provides only {valid_window_count} valid windows "
                f"but {target} are required by the sealed sampling contract; exclude "
                "it explicitly or use a compatible model profile"
            )
        evidence[split] = {
            "selection_mode": (
                "all_usable_episodes" if select_all_usable else "minimum_window_budget"
            ),
            "required_valid_window_count": target,
            "selected_episode_count": len(selected[split]),
        }
        evidence[split][
            (
                "selected_valid_window_count_lower_bound"
                if select_all_usable
                else "selected_valid_window_count"
            )
        ] = valid_window_count
        if not select_all_usable:
            evidence[split]["selected_episodes"] = selected_evidence
    return tuple(
        index for split in ("train", "val", "test") for index in selected[split]
    ), evidence


def _profiles(model_path: Path, encoder_path: Path) -> tuple[dict, dict, dict]:
    model = yaml.safe_load(_regular(model_path, "model profile").read_text())
    encoder = yaml.safe_load(_regular(encoder_path, "encoder contract").read_text())
    model_body = model["model"]
    sampling = model["sampling"]
    grid = int(encoder["token_grid"])
    observed = (
        grid * grid,
        int(encoder["token_dim"]),
        int(encoder["max_views"]),
        int(encoder["target_rgb_size"]),
    )
    expected = (
        int(model_body["P"]),
        int(model_body["token_dim"]),
        int(model_body["num_views"]),
        int(model_body["rgb_size"]),
    )
    if observed != expected:
        raise RuntimeError(
            f"model/encoder representation mismatch: {observed} != {expected}"
        )
    representation = {
        "schema": "wm3d_v8_episode_representation_v1",
        "token_grid": grid,
        "spatial_tokens": grid * grid,
        "token_dim": observed[1],
        "num_views": observed[2],
        "view_slots": list(VIEW_SLOTS),
        "rgb_size": observed[3],
        "time_binding": "episode_row_ordinal_with_pts_audit",
        "missing_view_policy": "mask_without_duplication",
        "state_frame_selection": {
            "mode": "observed_greedy_minimum_separation",
            "minimum_separation_seconds": float(
                sampling["minimum_anchor_separation_seconds"]
            ),
            "preserve_observed_timestamps": True,
            "interpolation": "forbidden",
        },
    }
    appearance_grid = int(encoder.get("appearance_token_grid", 0))
    if appearance_grid > 0:
        appearance_tokens = appearance_grid * appearance_grid
        if not bool(model_body.get("appearance_enabled", False)):
            raise RuntimeError(
                "encoder exposes appearance tokens but the model disables appearance"
            )
        if int(model_body.get("appearance_P", 0)) != appearance_tokens:
            raise RuntimeError(
                "model/encoder appearance representation mismatch: "
                f"{model_body.get('appearance_P')} != {appearance_tokens}"
            )
        representation["appearance_token_grid"] = appearance_grid
        representation["appearance_feature_layer"] = int(
            encoder["appearance_feature_layer"]
        )
    cache = {
        "schema": "wm3d_v8_unified_window_index_v3",
        "task_partition": "episode",
        "feature_workers_per_node": 8,
        "decode_workers_per_gpu": 4,
        "writer_threads_per_worker": 2,
        "task_claim": "atomic_no_clobber",
        "resume": "receipt_and_sha",
        "view_token_codec": "int8_per_vector",
        "depth_codec": "fp16",
        "point_codec": "fp16",
        "camera_pose_codec": "fp32",
        "rgb_codec": "jpeg_pack",
        "action_proprio_storage": "same_episode_artifact",
    }
    return representation, cache, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe-root", type=Path, required=True)
    parser.add_argument("--droid-root", type=Path, required=True)
    parser.add_argument("--robocasa-root", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--old-spec", type=Path, required=True)
    parser.add_argument("--old-config", type=Path, required=True)
    parser.add_argument("--old-log", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument("--minimum-train-windows", type=int, default=1)
    parser.add_argument("--minimum-eval-windows", type=int, default=1)
    parser.add_argument("--select-all-usable-episodes", action="store_true")
    args = parser.parse_args()
    if args.minimum_train_windows < 1:
        raise RuntimeError("--minimum-train-windows must be positive")
    if args.minimum_eval_windows < 1:
        raise RuntimeError("--minimum-eval-windows must be positive")
    roots = {
        "oxe": _real_dir(args.oxe_root, "OXE root"),
        "droid": _real_dir(args.droid_root, "DROID root"),
        "robocasa": _real_dir(args.robocasa_root, "RoboCasa root"),
    }
    output = args.output_root.absolute()
    output.mkdir(parents=True, exist_ok=True)
    representation, cache, model_profile = _profiles(
        args.model_profile, args.encoder_contract
    )
    excluded = frozenset(str(name) for name in args.exclude_source)
    known = {plan.name for plan in PLANS}
    unknown = sorted(excluded - known)
    if unknown:
        raise RuntimeError(f"unknown excluded sources: {unknown}")
    plans = tuple(plan for plan in PLANS if plan.name not in excluded)
    if not plans:
        raise RuntimeError("all existing robot sources were excluded")
    sources = []
    embodiments_by_name: dict[str, dict] = {}
    receipt_rows = []
    adapter_root = output / "adapters"
    episode_root = output / "episode_indices"
    episode_range_root = output / "episode_ranges"
    for plan in plans:
        root = _root(plan, **roots)
        info_path = _regular(root / "meta/info.json", f"{plan.name} info")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        adapter, embodiment = _adapter(plan, info)
        adapter_path = adapter_root / f"{plan.name}.yaml"
        _publish(
            adapter_path,
            yaml.safe_dump(adapter, sort_keys=False, allow_unicode=True).encode(
                "utf-8"
            ),
        )
        episode_ranges = None
        episode_range_path = None
        if plan.name == "oxe_droid":
            episode_range_path = episode_range_root / "oxe_droid.nonidle.json"
            if episode_range_path.is_file() and not episode_range_path.is_symlink():
                raw_ranges = json.loads(episode_range_path.read_text(encoding="utf-8"))
                episode_ranges = _normalize_episode_ranges(raw_ranges)
            else:
                episode_ranges = _compute_droid_nonidle_ranges(root)
                _publish(
                    episode_range_path,
                    (
                        json.dumps(
                            {
                                str(index): [list(bounds) for bounds in ranges]
                                for index, ranges in sorted(episode_ranges.items())
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8"),
                )
            if episode_ranges is None:
                raise RuntimeError("DROID non-idle range contract is empty")
        selected, selection_evidence = _selected_indices(
            plan.name,
            root,
            model_profile,
            view_keys=plan.views,
            minimum_train_windows=args.minimum_train_windows,
            minimum_eval_windows=args.minimum_eval_windows,
            select_all_usable=args.select_all_usable_episodes,
            episode_ranges=episode_ranges,
        )
        episode_path = episode_root / f"{plan.name}.txt"
        _publish(
            episode_path, ("\n".join(str(item) for item in selected) + "\n").encode()
        )
        sources.append(
            {
                "name": plan.name,
                "adapter": "lerobot",
                "raw_root": str(root),
                "adapter_config": str(adapter_path.absolute()),
                "adapter_contract_sha256": sha256_file(adapter_path.absolute()),
                "manifest": f"__MATERIALIZE_REQUIRED__/{plan.name}.jsonl",
                "manifest_sha256": "__MATERIALIZE_REQUIRED__",
                "embodiment": embodiment["name"],
                "weight": plan.weight,
                "nominal_hours": (
                    float(info["total_frames"]) / float(info["fps"]) / 3600.0
                ),
                "license_id": "operator_verified_upstream_license",
            }
        )
        previous_embodiment = embodiments_by_name.get(embodiment["name"])
        if previous_embodiment is None:
            embodiments_by_name[embodiment["name"]] = embodiment
        elif previous_embodiment != embodiment:
            raise RuntimeError(
                f"conflicting shared embodiment contract {embodiment['name']!r}"
            )
        receipt_rows.append(
            {
                "name": plan.name,
                "raw_root": str(root),
                "info_path": str(info_path),
                "info_sha256": sha256_file(info_path),
                "total_episodes": int(info["total_episodes"]),
                "total_frames": int(info["total_frames"]),
                "adapter_sha256": sha256_file(adapter_path.absolute()),
                "episode_index_sha256": sha256_file(episode_path.absolute()),
                "episode_range_path": (
                    None
                    if episode_range_path is None
                    else str(episode_range_path.absolute())
                ),
                "episode_range_sha256": (
                    None
                    if episode_range_path is None
                    else sha256_file(episode_range_path.absolute())
                ),
                "episode_range_count": (
                    0 if episode_ranges is None else len(episode_ranges)
                ),
                "nonidle_segment_count": (
                    0
                    if episode_ranges is None
                    else sum(len(ranges) for ranges in episode_ranges.values())
                ),
                "nonidle_row_count": (
                    0
                    if episode_ranges is None
                    else sum(
                        stop - start
                        for ranges in episode_ranges.values()
                        for start, stop in ranges
                    )
                ),
                "selected_episode_indices": list(selected),
                "action_transform": plan.action_transform,
                "state_transform": plan.state_transform,
                "action_value_mask": list(plan.action_mask),
                "selection_window_evidence": selection_evidence,
            }
        )
    template = {
        "schema": "wm3d_v8_data_profile_v4",
        "name": (
            "wm3d_1b_existing_robot_raw_formal"
            if args.select_all_usable_episodes
            else "wm3d_1b_existing_robot_raw_canary"
        ),
        "cache_representation": representation,
        "cache": cache,
        "sources": sources,
        "embodiments": list(embodiments_by_name.values()),
        "notes": {
            "purpose": (
                "no-PCA raw streaming formal training over all usable existing "
                "GAM/OXE and RoboCasa episodes"
                if args.select_all_usable_episodes
                else "no-PCA raw streaming canary over the existing GAM/OXE and RoboCasa data"
            ),
            "source_count": len(plans),
            "requested_source_count": len(PLANS),
            "excluded_sources": sorted(excluded),
            "split_policy": (
                "all physically usable episodes in each deterministic split"
                if args.select_all_usable_episodes
                else (
                    "longest deterministic train episodes until the minimum train-window "
                    "budget is met, with validation/test episodes selected until their "
                    "minimum eval-window budgets are met"
                )
            ),
            "selection_mode": (
                "all_usable_episodes"
                if args.select_all_usable_episodes
                else "minimum_window_budget"
            ),
            "minimum_train_windows_per_source": args.minimum_train_windows,
            "minimum_eval_windows_per_source": args.minimum_eval_windows,
            "action_state_policy": (
                "audited GAM/OXE canonical arm actions plus audited grouped "
                "RoboCasa base/mode/arm actions with recorded timestamps"
            ),
            "color_policy": "legacy GAM BGR declarations are explicitly restored before RGB supervision",
        },
    }
    template_path = output / "data_template.yaml"
    _publish(
        template_path,
        yaml.safe_dump(template, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )
    receipt = {
        "schema": "wm3d_existing_robot_mix_reuse_v1",
        "code_commit": args.code_commit,
        "model_profile_sha256": sha256_file(
            _regular(args.model_profile, "model profile")
        ),
        "encoder_contract_sha256": sha256_file(
            _regular(args.encoder_contract, "encoder contract")
        ),
        "legacy_provenance": {
            "pretraining_spec_sha256": sha256_file(_regular(args.old_spec, "old spec")),
            "training_config_sha256": sha256_file(
                _regular(args.old_config, "old config")
            ),
            "completed_training_log_sha256": sha256_file(
                _regular(args.old_log, "old log")
            ),
        },
        "data_template_path": str(template_path.absolute()),
        "data_template_sha256": sha256_file(template_path.absolute()),
        "requested_source_count": len(PLANS),
        "source_count": len(receipt_rows),
        "excluded_sources": sorted(excluded),
        "minimum_train_windows_per_source": args.minimum_train_windows,
        "minimum_eval_windows_per_source": args.minimum_eval_windows,
        "selection_mode": (
            "all_usable_episodes"
            if args.select_all_usable_episodes
            else "minimum_window_budget"
        ),
        "sources": receipt_rows,
    }
    receipt_path = output / "local_reuse_receipt.json"
    _publish(
        receipt_path,
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "source_count": len(plans),
                "excluded_sources": sorted(excluded),
                "data_template": str(template_path.absolute()),
                "data_template_sha256": sha256_file(template_path.absolute()),
                "local_reuse_receipt": str(receipt_path.absolute()),
                "local_reuse_receipt_sha256": sha256_file(receipt_path.absolute()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
