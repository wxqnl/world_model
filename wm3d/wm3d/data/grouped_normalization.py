"""Train-only normalization contract for grouped robot signals.

Episode caches remain in audited physical units.  This module materializes and
loads a separate, SHA-bound training view whose coordinates are keyed by the
data-profile source, embodiment, physical group, semantic and dimension.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open
import torch

from .episode_robot import (
    assemble_robot_window_from_prepared_episode,
    prepare_episode_robot_tensors,
    validate_episode_robot_tensors,
)
from .grouped_robot import (
    ACTION_SEMANTIC_IDS,
    STATE_SEMANTIC_IDS,
    GroupedRobotLimits,
)
from .manifest_contract import (
    DataProfile,
    SHA256_RE,
    canonical_sha256,
    load_cache_index,
    sha256_file,
)
from wm3d.models.native_world_model import native_config_from_mapping


GROUPED_NORMALIZATION_SCHEMA = "wm3d_v8_grouped_normalization_v2"
GROUPED_NORMALIZATION_ESTIMATOR = "masked_population_mean_std_float64_v1"
GROUPED_NORMALIZATION_SPLIT = "train_only"
ACTION_FINE_LANE = "fine_command"
ACTION_COARSE_LANE = "coarse_effect"
STATE_CURRENT_LANE = "current_state"
_EPISODE_LRU_SIZE = 8

_ACTION_IDENTITY_SEMANTICS = frozenset(
    {
        "absolute_gripper_open01",
        "absolute_gripper_close01",
        "binary_contact",
        "controller_mode",
    }
)
_STATE_IDENTITY_SEMANTICS = frozenset({"gripper_open01", "gripper_close01"})
_ROW_FIELDS = {
    "kind",
    "source",
    "source_id",
    "embodiment",
    "embodiment_id",
    "group",
    "group_id",
    "dimension",
    "semantic",
    "semantic_id",
    "lane",
    "transform",
    "count",
    "observed_mean",
    "observed_std",
    "observed_min",
    "observed_max",
    "offset",
    "scale",
}
_ARTIFACT_FIELDS = {
    "schema",
    "estimator",
    "split",
    "minimum_scale",
    "data_profile_path",
    "data_profile_sha256",
    "model_profile_sha256",
    "window_index_path",
    "window_index_sha256",
    "train_window_count_by_source",
    "rows",
    "rows_sha256",
}


class GroupedNormalizationError(ValueError):
    """Raised when normalization provenance or coordinates are incomplete."""


@dataclass(frozen=True)
class NormalizationTensors:
    fine_action_offset: torch.Tensor
    fine_action_scale: torch.Tensor
    fine_action_available: torch.Tensor
    coarse_action_offset: torch.Tensor
    coarse_action_scale: torch.Tensor
    coarse_action_available: torch.Tensor
    state_offset: torch.Tensor
    state_scale: torch.Tensor
    state_available: torch.Tensor


@dataclass
class _RunningMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        sample = np.asarray(values, dtype=np.float64).reshape(-1)
        if sample.size == 0:
            return
        if not np.isfinite(sample).all():
            raise GroupedNormalizationError("normalization input contains NaN/Inf")
        batch_count = int(sample.size)
        batch_mean = float(sample.mean(dtype=np.float64))
        centered = sample - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
            self.mean += delta * batch_count / total
            self.count = total
        self.minimum = min(self.minimum, float(sample.min()))
        self.maximum = max(self.maximum, float(sample.max()))

    @property
    def std(self) -> float:
        if self.count <= 0:
            raise GroupedNormalizationError("normalization dimension has no train values")
        return float(np.sqrt(max(0.0, self.m2 / self.count)))


def _safe_cache_file(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise GroupedNormalizationError(f"robot cache shard is not a regular file: {candidate}")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise GroupedNormalizationError(f"robot cache shard escapes cache root: {path}") from exc
    return path


def _read_robot_shard(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        values = {name: handle.get_tensor(name) for name in handle.keys()}
    validate_episode_robot_tensors(values)
    return values


def _source_by_name(profile: DataProfile) -> dict[str, Any]:
    return {source.name: source for source in profile.sources}


def _scale_floor(statistic: _RunningMoments, minimum_scale: float) -> float:
    reference = max(1.0, abs(statistic.minimum), abs(statistic.maximum))
    return max(statistic.std, minimum_scale * reference)


def build_grouped_normalization_artifact(
    *,
    data_profile: DataProfile,
    model_profile: Mapping[str, Any],
    model_profile_sha256: str,
    window_index_path: Path,
    window_index_sha256: str,
    cache_root: Path,
    minimum_scale: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute mask-aware moments from exactly the sampled train windows.

    Each source/group contributes only its declared supervision lane.  A
    fine-command group is never used to invent coarse-effect statistics and a
    coarse-effect group never invents fine-command statistics.  The artifact
    is model-profile/window bound because the selected train windows and their
    exact current-state/policy boundaries define the sampled distribution,
    while the episode cache itself remains model independent.
    """

    if not np.isfinite(minimum_scale) or minimum_scale <= 0:
        raise GroupedNormalizationError("minimum_scale must be finite and positive")
    if canonical_sha256(model_profile) != model_profile_sha256:
        raise GroupedNormalizationError("model profile digest does not match content")
    index_path = Path(window_index_path).resolve(strict=True)
    root = Path(cache_root).resolve(strict=True)
    entries = load_cache_index(index_path, expected_sha256=window_index_sha256)
    train = tuple(entry for entry in entries if entry.split == "train")
    if not train:
        raise GroupedNormalizationError("window index contains no train windows")
    native = native_config_from_mapping(model_profile["model"])
    sampling = model_profile["sampling"]
    limits = GroupedRobotLimits(
        max_groups=native.max_action_groups,
        max_substeps=native.max_action_substeps,
        max_action_dim=native.max_action_dim,
        max_state_dim=native.max_state_dim,
    )
    source_specs = _source_by_name(data_profile)
    source_ids = {name: index for index, name in enumerate(data_profile.source_order)}
    counts = {name: 0 for name in data_profile.source_order}
    moments: dict[tuple[str, str, str, int, int], _RunningMoments] = {}
    lane_by_group: dict[tuple[str, int], str] = {}

    verified: dict[str, str] = {}
    expected_feature_sha: dict[str, str] = {}
    # Window rows are materialized in source/episode order, so a bounded LRU
    # keeps the hot episode while preventing full-corpus robot/boundary tensors
    # from accumulating in RAM during 1B/5B artifact builds.
    prepared_by_robot: OrderedDict[tuple[str, str], Any] = OrderedDict()
    source_rows_by_feature: OrderedDict[
        tuple[str, str], torch.Tensor
    ] = OrderedDict()
    for entry in train:
        if entry.source not in source_specs:
            raise GroupedNormalizationError(
                f"train cache references source outside data profile: {entry.source!r}"
            )
        source = source_specs[entry.source]
        if entry.embodiment != source.embodiment:
            raise GroupedNormalizationError("episode source/embodiment binding drifted")
        embodiment = data_profile.embodiments[entry.embodiment]
        if entry.robot_shard in verified and verified[entry.robot_shard] != entry.robot_sha256:
            raise GroupedNormalizationError("window index assigns conflicting robot shard SHA")
        previous_feature_sha = expected_feature_sha.setdefault(
            entry.feature_shard, entry.feature_sha256
        )
        if previous_feature_sha != entry.feature_sha256:
            raise GroupedNormalizationError("window index assigns conflicting feature shard SHA")
        robot_key = (entry.robot_shard, entry.robot_sha256)
        prepared = prepared_by_robot.pop(robot_key, None)
        if prepared is None:
            path = _safe_cache_file(root, entry.robot_shard)
            observed_sha = verified.get(entry.robot_shard)
            if observed_sha is None:
                observed_sha = sha256_file(path)
                verified[entry.robot_shard] = observed_sha
            if observed_sha != entry.robot_sha256:
                raise GroupedNormalizationError(
                    f"robot shard SHA mismatch {entry.robot_shard}: "
                    f"{observed_sha} != {entry.robot_sha256}"
                )
            values = _read_robot_shard(path)
            if int(values["embodiment_id"]) != embodiment.embodiment_id:
                raise GroupedNormalizationError("robot shard embodiment id drifted")
            prepared = prepare_episode_robot_tensors(values, embodiment=embodiment)
        prepared_by_robot[robot_key] = prepared
        while len(prepared_by_robot) > _EPISODE_LRU_SIZE:
            prepared_by_robot.popitem(last=False)
        # The sealed window index already binds the complete feature payload by
        # SHA.  Normalization needs only its tiny source-observation boundary
        # column; re-hashing every 1B/5B feature shard here would rescan the
        # expensive shared payload once per model profile.  Actual training
        # still verifies each sampled feature shard on first open.
        feature_key = (entry.feature_shard, entry.feature_sha256)
        boundary_source_all = source_rows_by_feature.pop(feature_key, None)
        if boundary_source_all is None:
            feature_path = _safe_cache_file(root, entry.feature_shard)
            with safe_open(feature_path, framework="pt", device="cpu") as handle:
                if "source_observation_row" not in set(handle.keys()):
                    raise GroupedNormalizationError(
                        "feature shard omits source observation rows"
                    )
                boundary_source_all = handle.get_tensor(
                    "source_observation_row"
                ).to(torch.int64)
            if boundary_source_all.ndim != 1:
                raise GroupedNormalizationError(
                    "feature source-observation rows must be rank-1"
                )
        source_rows_by_feature[feature_key] = boundary_source_all
        while len(source_rows_by_feature) > _EPISODE_LRU_SIZE:
            source_rows_by_feature.popitem(last=False)
        boundary_rows = (
            entry.leading_feature_row,
            *entry.context_feature_rows,
            *entry.future_feature_rows,
        )
        if max(boundary_rows) >= boundary_source_all.shape[0]:
            raise GroupedNormalizationError("window boundary row exceeds feature shard")
        boundary_source = boundary_source_all[
            torch.tensor(boundary_rows, dtype=torch.int64)
        ]
        if not bool(torch.diff(boundary_source).gt(0).all()):
            raise GroupedNormalizationError("window source boundaries are not increasing")
        assembled = assemble_robot_window_from_prepared_episode(
            prepared=prepared,
            embodiment=embodiment,
            selected_source_boundary_indices=boundary_source.tolist(),
            limits=limits,
            context_samples=native.T,
            max_policy_queries=native.max_policy_queries,
            policy_target_horizon_s=float(sampling["policy_target_horizon_seconds"]),
        )
        counts[entry.source] += 1
        for slot, group in enumerate(embodiment.groups):
            if int(assembled["action_group_ids"][slot]) != group.group_id or not bool(
                assembled["action_group_mask"][slot]
            ):
                raise GroupedNormalizationError("assembled robot group layout drifted")
            action_ids = assembled["action_semantic_ids"][slot, : group.action_dim]
            expected_action_ids = torch.tensor(
                [ACTION_SEMANTIC_IDS[item] for item in group.action_semantics],
                dtype=torch.int64,
            )
            if not torch.equal(action_ids, expected_action_ids):
                raise GroupedNormalizationError("robot shard action semantics drifted")
            prepared_series = prepared.action_series[slot]
            lane = prepared_series.supervision
            group_key = (entry.source, group.group_id)
            previous = lane_by_group.setdefault(group_key, lane)
            if previous != lane:
                raise GroupedNormalizationError(
                    "one source/group changes fine/coarse supervision across train episodes"
                )
            fine_mask_tensors = (
                assembled["history_fine_action_mask"][:, slot, :, : group.action_dim],
                assembled["future_factual_fine_action_mask"][:, slot, :, : group.action_dim],
                assembled["target_fine_action_mask"][slot, :, : group.action_dim],
            )
            coarse_mask_tensors = (
                assembled["history_coarse_action_mask"][:, slot, : group.action_dim],
                assembled["future_factual_coarse_action_mask"][
                    :, slot, : group.action_dim
                ],
                assembled["target_coarse_action_mask"][:, slot, : group.action_dim],
            )
            if lane == ACTION_FINE_LANE:
                if any(bool(mask.any()) for mask in coarse_mask_tensors):
                    raise GroupedNormalizationError(
                        "fine-command group unexpectedly contains coarse-effect masks"
                    )
                action = torch.cat(
                    (
                        assembled["history_fine_action_values"][
                            :, slot, :, : group.action_dim
                        ].reshape(-1, group.action_dim),
                        assembled["future_factual_fine_action_values"][
                            :, slot, :, : group.action_dim
                        ].reshape(-1, group.action_dim),
                        assembled["target_fine_action"][
                            slot, :, : group.action_dim
                        ].reshape(-1, group.action_dim),
                    )
                ).numpy()
                action_mask = torch.cat(
                    tuple(
                        mask.reshape(-1, group.action_dim)
                        for mask in fine_mask_tensors
                    )
                ).numpy()
            elif lane == ACTION_COARSE_LANE:
                if any(bool(mask.any()) for mask in fine_mask_tensors):
                    raise GroupedNormalizationError(
                        "coarse-effect group unexpectedly contains fine-command masks"
                    )
                action = torch.cat(
                    (
                        assembled["history_coarse_action_values"][
                            :, slot, : group.action_dim
                        ],
                        assembled["target_coarse_action"][:, slot, : group.action_dim],
                    )
                ).numpy()
                # future_factual_coarse and target_coarse are the same K
                # physical effects exposed for different consumers.  Validate
                # both masks above, but count each effect exactly once.
                action_mask = torch.cat(
                    (coarse_mask_tensors[0], coarse_mask_tensors[2])
                ).numpy()
            else:
                raise GroupedNormalizationError(f"unsupported action lane {lane!r}")
            for dimension in range(group.action_dim):
                key = ("action", lane, entry.source, group.group_id, dimension)
                moments.setdefault(key, _RunningMoments()).update(
                    action[action_mask[:, dimension], dimension]
                )

            if group.state_dim:
                state_ids = assembled["state_semantic_ids"][slot, : group.state_dim]
                expected_state_ids = torch.tensor(
                    [STATE_SEMANTIC_IDS[item] for item in group.state_semantics],
                    dtype=torch.int64,
                )
                if not torch.equal(state_ids, expected_state_ids):
                    raise GroupedNormalizationError("robot shard state semantics drifted")
                state = assembled["current_state_values"][slot : slot + 1, : group.state_dim].numpy()
                state_mask = assembled["current_state_mask"][slot : slot + 1, : group.state_dim].numpy()
                for dimension in range(group.state_dim):
                    key = (
                        "state",
                        STATE_CURRENT_LANE,
                        entry.source,
                        group.group_id,
                        dimension,
                    )
                    moments.setdefault(key, _RunningMoments()).update(
                        state[state_mask[:, dimension], dimension]
                    )

    missing_sources = sorted(name for name, count in counts.items() if count <= 0)
    if missing_sources:
        raise GroupedNormalizationError(
            f"normalization has no train windows for sources: {missing_sources}"
        )

    rows: list[dict[str, Any]] = []
    for source_name in data_profile.source_order:
        source = source_specs[source_name]
        embodiment = data_profile.embodiments[source.embodiment]
        for group in embodiment.groups:
            for kind, lane, semantics in (
                (
                    "action",
                    lane_by_group[(source_name, group.group_id)],
                    group.action_semantics,
                ),
                ("state", STATE_CURRENT_LANE, group.state_semantics),
            ):
                semantic_ids = (
                    ACTION_SEMANTIC_IDS if kind == "action" else STATE_SEMANTIC_IDS
                )
                identities = (
                    _ACTION_IDENTITY_SEMANTICS
                    if kind == "action"
                    else _STATE_IDENTITY_SEMANTICS
                )
                for dimension, semantic in enumerate(semantics):
                    key = (kind, lane, source_name, group.group_id, dimension)
                    statistic = moments.get(key)
                    if statistic is None or statistic.count <= 0:
                        raise GroupedNormalizationError(
                            f"no masked train values for {key}; dimensions cannot be dropped"
                        )
                    observed_std = statistic.std
                    identity = semantic in identities
                    rows.append(
                        {
                            "kind": kind,
                            "source": source_name,
                            "source_id": source_ids[source_name],
                            "embodiment": embodiment.name,
                            "embodiment_id": embodiment.embodiment_id,
                            "group": group.name,
                            "group_id": group.group_id,
                            "dimension": dimension,
                            "semantic": semantic,
                            "semantic_id": semantic_ids[semantic],
                            "lane": lane,
                            "transform": "identity" if identity else "zscore",
                            "count": statistic.count,
                            "observed_mean": statistic.mean,
                            "observed_std": observed_std,
                            "observed_min": statistic.minimum,
                            "observed_max": statistic.maximum,
                            "offset": 0.0 if identity else statistic.mean,
                            "scale": 1.0
                            if identity
                            else _scale_floor(statistic, minimum_scale),
                        }
                    )
    return {
        "schema": GROUPED_NORMALIZATION_SCHEMA,
        "estimator": GROUPED_NORMALIZATION_ESTIMATOR,
        "split": GROUPED_NORMALIZATION_SPLIT,
        "minimum_scale": float(minimum_scale),
        "data_profile_path": str(data_profile.path),
        "data_profile_sha256": data_profile.profile_sha256,
        "model_profile_sha256": model_profile_sha256,
        "window_index_path": str(index_path),
        "window_index_sha256": window_index_sha256,
        "train_window_count_by_source": counts,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }


class GroupedRobotNormalizer:
    """Strict runtime lookup for one immutable normalization artifact."""

    def __init__(self, artifact: Mapping[str, Any], *, data_profile: DataProfile):
        if set(artifact) != _ARTIFACT_FIELDS:
            raise GroupedNormalizationError("normalization artifact fields mismatch")
        if artifact.get("schema") != GROUPED_NORMALIZATION_SCHEMA:
            raise GroupedNormalizationError("normalization artifact schema mismatch")
        if artifact.get("estimator") != GROUPED_NORMALIZATION_ESTIMATOR:
            raise GroupedNormalizationError("normalization estimator mismatch")
        if artifact.get("split") != GROUPED_NORMALIZATION_SPLIT:
            raise GroupedNormalizationError("normalization must be train-only")
        if artifact.get("data_profile_sha256") != data_profile.profile_sha256:
            raise GroupedNormalizationError("normalization/data profile SHA mismatch")
        minimum_scale = float(artifact.get("minimum_scale", 0.0))
        if not np.isfinite(minimum_scale) or minimum_scale <= 0:
            raise GroupedNormalizationError("normalization minimum scale is invalid")
        for field in (
            "data_profile_sha256",
            "model_profile_sha256",
            "window_index_sha256",
            "rows_sha256",
        ):
            if SHA256_RE.fullmatch(str(artifact.get(field, ""))) is None:
                raise GroupedNormalizationError(f"normalization {field} is invalid")
        rows = artifact.get("rows")
        if not isinstance(rows, list) or canonical_sha256(rows) != artifact.get(
            "rows_sha256"
        ):
            raise GroupedNormalizationError("normalization row index SHA mismatch")
        expected_counts = {name: 0 for name in data_profile.source_order}
        counts = artifact.get("train_window_count_by_source")
        if not isinstance(counts, dict) or set(counts) != set(expected_counts):
            raise GroupedNormalizationError("normalization source coverage mismatch")
        if any(int(value) <= 0 for value in counts.values()):
            raise GroupedNormalizationError("normalization source has no train episodes")

        self.profile = data_profile
        self.window_index_sha256 = str(artifact["window_index_sha256"])
        self.model_profile_sha256 = str(artifact["model_profile_sha256"])
        self._rows: dict[tuple[str, str, str, int, int], Mapping[str, Any]] = {}
        self._lane_by_group: dict[tuple[str, int], str] = {}
        source_ids = {name: index for index, name in enumerate(data_profile.source_order)}
        for raw in rows:
            if not isinstance(raw, dict) or set(raw) != _ROW_FIELDS:
                raise GroupedNormalizationError("normalization row fields mismatch")
            kind = str(raw["kind"])
            source_name = str(raw["source"])
            if kind not in {"action", "state"} or source_name not in source_ids:
                raise GroupedNormalizationError("normalization row identity is invalid")
            if int(raw["source_id"]) != source_ids[source_name]:
                raise GroupedNormalizationError("normalization source id drifted")
            source = next(item for item in data_profile.sources if item.name == source_name)
            embodiment = data_profile.embodiments[source.embodiment]
            if (
                str(raw["embodiment"]) != embodiment.name
                or int(raw["embodiment_id"]) != embodiment.embodiment_id
            ):
                raise GroupedNormalizationError("normalization embodiment drifted")
            group_id = int(raw["group_id"])
            group = next((item for item in embodiment.groups if item.group_id == group_id), None)
            if group is None or str(raw["group"]) != group.name:
                raise GroupedNormalizationError("normalization group drifted")
            dimension = int(raw["dimension"])
            semantics = group.action_semantics if kind == "action" else group.state_semantics
            semantic_ids = ACTION_SEMANTIC_IDS if kind == "action" else STATE_SEMANTIC_IDS
            if not 0 <= dimension < len(semantics):
                raise GroupedNormalizationError("normalization dimension is out of range")
            semantic = semantics[dimension]
            if (
                str(raw["semantic"]) != semantic
                or int(raw["semantic_id"]) != semantic_ids[semantic]
            ):
                raise GroupedNormalizationError("normalization semantic drifted")
            identity = semantic in (
                _ACTION_IDENTITY_SEMANTICS
                if kind == "action"
                else _STATE_IDENTITY_SEMANTICS
            )
            expected_transform = "identity" if identity else "zscore"
            if str(raw["transform"]) != expected_transform:
                raise GroupedNormalizationError("normalization transform drifted")
            count = int(raw["count"])
            offset = float(raw["offset"])
            scale = float(raw["scale"])
            numeric = [
                float(raw[name])
                for name in (
                    "observed_mean",
                    "observed_std",
                    "observed_min",
                    "observed_max",
                )
            ]
            if count <= 0 or not np.isfinite([offset, scale, *numeric]).all() or scale <= 0:
                raise GroupedNormalizationError("normalization row statistics are invalid")
            observed_mean, observed_std, observed_minimum, observed_maximum = numeric
            if (
                observed_std < 0
                or observed_minimum > observed_mean
                or observed_mean > observed_maximum
            ):
                raise GroupedNormalizationError("normalization observed moments are inconsistent")
            if identity and (offset != 0.0 or scale != 1.0):
                raise GroupedNormalizationError(
                    "gripper/binary/discrete normalization must be identity"
                )
            if identity and (observed_minimum < 0.0 or observed_maximum > 1.0):
                raise GroupedNormalizationError(
                    "gripper/binary/discrete values must remain in [0,1]"
                )
            if not identity:
                expected_scale = max(
                    observed_std,
                    minimum_scale
                    * max(1.0, abs(observed_minimum), abs(observed_maximum)),
                )
                if offset != observed_mean or not math.isclose(
                    scale,
                    expected_scale,
                    rel_tol=0.0,
                    abs_tol=0.0,
                ):
                    raise GroupedNormalizationError(
                        "zscore offset/scale do not match observed moments"
                    )
            lane = str(raw["lane"])
            if (kind == "action" and lane not in {ACTION_FINE_LANE, ACTION_COARSE_LANE}) or (
                kind == "state" and lane != STATE_CURRENT_LANE
            ):
                raise GroupedNormalizationError("normalization lane contract drifted")
            if kind == "action":
                group_key = (source_name, group_id)
                previous_lane = self._lane_by_group.setdefault(group_key, lane)
                if previous_lane != lane:
                    raise GroupedNormalizationError(
                        "one source/group has multiple action normalization lanes"
                    )
            key = (kind, lane, source_name, group_id, dimension)
            if key in self._rows:
                raise GroupedNormalizationError(f"duplicate normalization row {key}")
            self._rows[key] = raw

        expected_keys = {
            (kind, lane, source.name, group.group_id, dimension)
            for source in data_profile.sources
            for group in data_profile.embodiments[source.embodiment].groups
            for kind, lane, dimension_count in (
                (
                    "action",
                    self._lane_by_group.get((source.name, group.group_id), "missing"),
                    group.action_dim,
                ),
                ("state", STATE_CURRENT_LANE, group.state_dim),
            )
            for dimension in range(dimension_count)
        }
        if set(self._rows) != expected_keys:
            raise GroupedNormalizationError(
                "normalization row closure does not exactly cover profile dimensions"
            )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_sha256: str,
        expected_data_profile_sha256: str,
        expected_model_profile_sha256: str,
        expected_window_index_sha256: str,
        data_profile: DataProfile,
    ) -> "GroupedRobotNormalizer":
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise GroupedNormalizationError(
                f"normalization artifact is not a regular file: {path}"
            )
        path = path.resolve(strict=True)
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise GroupedNormalizationError(
                f"normalization artifact SHA mismatch: {observed} != {expected_sha256}"
            )
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise GroupedNormalizationError("normalization artifact root must be a mapping")
        if artifact.get("data_profile_sha256") != expected_data_profile_sha256:
            raise GroupedNormalizationError("normalization expected data profile SHA mismatch")
        if artifact.get("model_profile_sha256") != expected_model_profile_sha256:
            raise GroupedNormalizationError("normalization expected model profile SHA mismatch")
        if artifact.get("window_index_sha256") != expected_window_index_sha256:
            raise GroupedNormalizationError("normalization expected window index SHA mismatch")
        return cls(artifact, data_profile=data_profile)

    def tensors_for(
        self,
        *,
        source: str,
        embodiment_id: int,
        group_ids: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        state_semantic_ids: torch.Tensor,
    ) -> NormalizationTensors:
        source_spec = next((item for item in self.profile.sources if item.name == source), None)
        if source_spec is None:
            raise GroupedNormalizationError(f"unknown normalization source {source!r}")
        embodiment = self.profile.embodiments[source_spec.embodiment]
        if int(embodiment_id) != embodiment.embodiment_id:
            raise GroupedNormalizationError("normalization sample embodiment id drifted")
        if group_ids.ndim != 1 or action_semantic_ids.ndim != 2 or state_semantic_ids.ndim != 2:
            raise GroupedNormalizationError("normalization tensor metadata ranks are invalid")
        if action_semantic_ids.shape[0] != group_ids.numel() or state_semantic_ids.shape[0] != group_ids.numel():
            raise GroupedNormalizationError("normalization tensor metadata shapes disagree")
        fine_action_offset = torch.zeros_like(action_semantic_ids, dtype=torch.float32)
        fine_action_scale = torch.ones_like(action_semantic_ids, dtype=torch.float32)
        fine_action_available = torch.zeros_like(action_semantic_ids, dtype=torch.bool)
        coarse_action_offset = torch.zeros_like(action_semantic_ids, dtype=torch.float32)
        coarse_action_scale = torch.ones_like(action_semantic_ids, dtype=torch.float32)
        coarse_action_available = torch.zeros_like(action_semantic_ids, dtype=torch.bool)
        state_offset = torch.zeros_like(state_semantic_ids, dtype=torch.float32)
        state_scale = torch.ones_like(state_semantic_ids, dtype=torch.float32)
        state_available = torch.zeros_like(state_semantic_ids, dtype=torch.bool)
        for slot, group in enumerate(embodiment.groups):
            if int(group_ids[slot]) != group.group_id:
                raise GroupedNormalizationError("normalization sample group id drifted")
            action_lane = self._lane_by_group[(source, group.group_id)]
            action_offset = (
                fine_action_offset if action_lane == ACTION_FINE_LANE else coarse_action_offset
            )
            action_scale = (
                fine_action_scale if action_lane == ACTION_FINE_LANE else coarse_action_scale
            )
            action_available = (
                fine_action_available
                if action_lane == ACTION_FINE_LANE
                else coarse_action_available
            )
            for kind, lane, semantic_tensor, offset_tensor, scale_tensor, available_tensor, semantics in (
                (
                    "action",
                    action_lane,
                    action_semantic_ids,
                    action_offset,
                    action_scale,
                    action_available,
                    group.action_semantics,
                ),
                (
                    "state",
                    STATE_CURRENT_LANE,
                    state_semantic_ids,
                    state_offset,
                    state_scale,
                    state_available,
                    group.state_semantics,
                ),
            ):
                semantic_ids = ACTION_SEMANTIC_IDS if kind == "action" else STATE_SEMANTIC_IDS
                for dimension, semantic in enumerate(semantics):
                    if int(semantic_tensor[slot, dimension]) != semantic_ids[semantic]:
                        raise GroupedNormalizationError("normalization sample semantic id drifted")
                    row = self._rows[(kind, lane, source, group.group_id, dimension)]
                    offset_tensor[slot, dimension] = float(row["offset"])
                    scale_tensor[slot, dimension] = float(row["scale"])
                    available_tensor[slot, dimension] = True
                if bool(semantic_tensor[slot, len(semantics) :].ne(0).any()):
                    raise GroupedNormalizationError(
                        "normalization sample has nonzero padded semantic ids"
                    )
        if bool(group_ids[len(embodiment.groups) :].ne(0).any()):
            raise GroupedNormalizationError(
                "normalization sample has nonzero padded group ids"
            )
        if bool(action_semantic_ids[len(embodiment.groups) :].ne(0).any()) or bool(
            state_semantic_ids[len(embodiment.groups) :].ne(0).any()
        ):
            raise GroupedNormalizationError(
                "normalization sample has nonzero padded group semantics"
            )
        return NormalizationTensors(
            fine_action_offset=fine_action_offset,
            fine_action_scale=fine_action_scale,
            fine_action_available=fine_action_available,
            coarse_action_offset=coarse_action_offset,
            coarse_action_scale=coarse_action_scale,
            coarse_action_available=coarse_action_available,
            state_offset=state_offset,
            state_scale=state_scale,
            state_available=state_available,
        )


def normalize_grouped_masked(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    offset: torch.Tensor,
    scale: torch.Tensor,
    group_axis: int,
) -> torch.Tensor:
    """Return a zero-filled normalized view without mutating physical values."""

    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise GroupedNormalizationError("normalization value/mask shapes disagree")
    group_axis = int(group_axis) % values.ndim
    if values.ndim < 2 or group_axis == values.ndim - 1:
        raise GroupedNormalizationError("normalization group axis is invalid")
    if (
        offset.ndim != 2
        or offset.shape != scale.shape
        or values.shape[group_axis] != offset.shape[0]
        or values.shape[-1] != offset.shape[1]
    ):
        raise GroupedNormalizationError("normalization statistics do not match grouped values")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(offset).all()):
        raise GroupedNormalizationError("normalization inputs contain NaN/Inf")
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise GroupedNormalizationError("normalization scale must be finite/positive")
    broadcast = [1] * values.ndim
    broadcast[group_axis] = offset.shape[0]
    broadcast[-1] = offset.shape[1]
    offset_view = offset.reshape(broadcast)
    scale_view = scale.reshape(broadcast)
    return torch.where(
        mask,
        (values - offset_view) / scale_view,
        torch.zeros_like(values),
    )


def validate_grouped_lane_mask(
    mask: torch.Tensor,
    *,
    available: torch.Tensor,
    group_axis: int,
    lane: str,
) -> None:
    """Fail when a valid value is routed through an undeclared stats lane."""

    if mask.dtype != torch.bool or available.dtype != torch.bool or available.ndim != 2:
        raise GroupedNormalizationError("normalization lane mask metadata is invalid")
    group_axis = int(group_axis) % mask.ndim
    if mask.ndim < 2 or group_axis == mask.ndim - 1:
        raise GroupedNormalizationError("normalization lane group axis is invalid")
    if mask.shape[group_axis] != available.shape[0] or mask.shape[-1] != available.shape[1]:
        raise GroupedNormalizationError("normalization lane mask shapes disagree")
    broadcast = [1] * mask.ndim
    broadcast[group_axis] = available.shape[0]
    broadcast[-1] = available.shape[1]
    if bool((mask & ~available.reshape(broadcast)).any()):
        raise GroupedNormalizationError(
            f"valid values exist outside declared {lane!r} normalization lane"
        )
