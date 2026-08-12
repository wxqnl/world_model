"""Materialize small 1B/5B window indices from one shared episode cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import numpy as np
import torch

from .episode_robot import (
    assemble_robot_window_from_prepared_episode,
    prepare_episode_robot_tensors,
)
from .grouped_robot import GroupedRobotLimits
from .manifest_contract import CACHE_INDEX_SCHEMA, CacheEpisodeEntry, DataProfile
from .window_selection import WindowSelectionError, select_observed_world_window
from wm3d_v3.models.native_world_model import native_config_from_mapping


class WindowIndexError(RuntimeError):
    pass


def _sample_id(
    *, episode: CacheEpisodeEntry, model_profile_sha256: str, anchor_index: int
) -> str:
    return hashlib.sha256(
        (
            f"{episode.episode_id}\x1f{episode.feature_sha256}\x1f"
            f"{episode.robot_sha256}\x1f{model_profile_sha256}\x1f{anchor_index}"
        ).encode()
    ).hexdigest()


def plan_episode_windows(
    *,
    episode: CacheEpisodeEntry,
    cache_root: Path,
    model_profile: Mapping[str, Any],
    model_profile_sha256: str,
    data_profile: DataProfile,
) -> tuple[dict[str, Any], ...]:
    """Select real observed rows; no feature/action payload is duplicated."""

    model = native_config_from_mapping(model_profile["model"]).__dict__
    sampling = model_profile["sampling"]
    path = Path(cache_root) / episode.feature_shard
    if path.is_symlink() or not path.is_file():
        raise WindowIndexError(f"feature shard is unavailable: {path}")
    with safe_open(path, framework="np") as handle:
        if not {"frame_time_s", "source_observation_row"}.issubset(handle.keys()):
            raise WindowIndexError(
                f"feature shard misses frame clock/source-row binding: {path}"
            )
        clock = np.asarray(handle.get_tensor("frame_time_s"), dtype=np.float64)
        source_rows = np.asarray(
            handle.get_tensor("source_observation_row"), dtype=np.int64
        )
    if clock.shape != (episode.frame_count,):
        raise WindowIndexError("episode index frame_count disagrees with feature clock")
    if (
        source_rows.shape != clock.shape
        or source_rows[0] < 0
        or np.any(np.diff(source_rows) <= 0)
    ):
        raise WindowIndexError("cached frame source-row binding is invalid")
    robot_path = Path(cache_root) / episode.robot_shard
    if robot_path.is_symlink() or not robot_path.is_file():
        raise WindowIndexError(f"robot shard is unavailable: {robot_path}")
    with safe_open(robot_path, framework="pt", device="cpu") as handle:
        robot = {name: handle.get_tensor(name) for name in handle.keys()}
    embodiment = data_profile.embodiments.get(episode.embodiment)
    if embodiment is None:
        raise WindowIndexError(
            f"episode references unknown embodiment {episode.embodiment!r}"
        )
    prepared_robot = prepare_episode_robot_tensors(robot, embodiment=embodiment)
    limits = GroupedRobotLimits(
        max_groups=int(model["max_action_groups"]),
        max_substeps=int(model["max_action_substeps"]),
        max_action_dim=int(model["max_action_dim"]),
        max_state_dim=int(model["max_state_dim"]),
    )
    separation = float(sampling["minimum_anchor_separation_seconds"])
    rows: list[dict[str, Any]] = []
    last_anchor_time = -np.inf
    for anchor in range(episode.frame_count):
        if float(clock[anchor]) - last_anchor_time + 1.0e-12 < separation:
            continue
        try:
            window = select_observed_world_window(
                clock,
                anchor_index=anchor,
                context_samples=int(model["T"]),
                future_samples=int(model["K"]),
                context_horizon_s=float(sampling["context_horizon_seconds"]),
                future_horizon_s=float(sampling["future_horizon_seconds"]),
                minimum_horizon_coverage=float(
                    sampling["minimum_horizon_coverage"]
                ),
                future_offsets_s=sampling.get("future_offsets_seconds"),
            )
        except WindowSelectionError:
            continue
        cached_boundary_rows = np.concatenate(
            (
                np.asarray([window.leading_boundary_index], dtype=np.int64),
                window.context_indices,
                window.future_indices,
            )
        )
        try:
            assemble_robot_window_from_prepared_episode(
                prepared=prepared_robot,
                embodiment=embodiment,
                selected_source_boundary_indices=source_rows[
                    cached_boundary_rows
                ].tolist(),
                limits=limits,
                context_samples=int(model["T"]),
                max_policy_queries=int(model["max_policy_queries"]),
                policy_target_horizon_s=float(
                    sampling["policy_target_horizon_seconds"]
                ),
            )
        except (ValueError, RuntimeError):
            # A cache episode may be visually long enough but lack exact
            # measured state/action coverage for this model horizon.  Such a
            # row is excluded at index materialization, never discovered by a
            # random dataloader worker during formal training.
            continue
        rows.append(
            {
                "schema": CACHE_INDEX_SCHEMA,
                "sample_id": _sample_id(
                    episode=episode,
                    model_profile_sha256=model_profile_sha256,
                    anchor_index=anchor,
                ),
                "source": episode.source,
                "split": episode.split,
                "embodiment": episode.embodiment,
                "feature_shard": episode.feature_shard,
                "feature_sha256": episode.feature_sha256,
                "robot_shard": episode.robot_shard,
                "robot_sha256": episode.robot_sha256,
                "rgb_pack": episode.rgb_pack,
                "rgb_pack_sha256": episode.rgb_pack_sha256,
                "leading_feature_row": int(window.leading_boundary_index),
                "context_feature_rows": window.context_indices.tolist(),
                "future_feature_rows": window.future_indices.tolist(),
            }
        )
        last_anchor_time = float(clock[anchor])
    return tuple(rows)


def plan_window_index(
    *,
    episodes: Sequence[CacheEpisodeEntry],
    cache_root: Path,
    model_profile: Mapping[str, Any],
    model_profile_sha256: str,
    data_profile: DataProfile,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=lambda item: (item.source, item.episode_id)):
        rows.extend(
            plan_episode_windows(
                episode=episode,
                cache_root=cache_root,
                model_profile=model_profile,
                model_profile_sha256=model_profile_sha256,
                data_profile=data_profile,
            )
        )
    if not rows:
        raise WindowIndexError("model sampling profile produced no valid cache windows")
    identities = [str(row["sample_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise WindowIndexError("window index produced duplicate sample identities")
    return tuple(rows)
