"""Window-bounded raw RGB dataset for direct in-model VGGT training.

Unlike the streaming episode cache, this path never encodes or persists a full
episode.  It decodes only the sealed T+K observation rows for one training
window, keeps robot metadata in a small CPU LRU, and hands uint8 RGB to the
rank-local frozen VGGT adapter.
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .cache_tasks import CacheTask, cache_task_from_mapping
from .episode_io import (
    VerifiedAssetStore,
    open_episode_accessor,
    select_episode_cache_rows,
)
from .episode_robot import (
    PreparedEpisodeRobot,
    assemble_robot_window_from_prepared_episode,
    build_episode_robot_cache,
    prepare_episode_robot_tensors,
)
from .grouped_normalization import (
    GroupedNormalizationError,
    GroupedRobotNormalizer,
    normalize_grouped_masked,
    validate_grouped_lane_mask,
)
from .manifest_contract import (
    CacheIndexEntry,
    DataProfile,
    canonical_timestamp_sha256,
    load_cache_episode_index,
)
from .source_adapters import (
    adapt_action_series,
    adapt_state_series,
    load_adapter_contract,
)
from .task_embedding_store import TaskEmbeddingStore
from .unified_cache_dataset import CacheDataError, UnifiedCacheDataset
from .window_video import (
    VideoTimestampIndexStore,
    decode_episode_window_views,
)


DIRECT_RAW_DATA_CLOSURE_SCHEMA = "wm3d_direct_raw_data_closure_v1"


def _apply_ignored_action_dimensions(
    action: dict[str, torch.Tensor],
    ignored_by_slot: Mapping[int, Sequence[int]],
) -> None:
    """Remove source metadata fields that are not executable robot actions."""

    fine_fields = (
        ("history_fine_action_mask", 1),
        ("future_factual_fine_action_mask", 1),
        ("target_fine_action_mask", 0),
    )
    coarse_fields = (
        ("history_coarse_action_mask", 1),
        ("future_factual_coarse_action_mask", 1),
        ("target_coarse_action_mask", 1),
    )
    for raw_slot, raw_dimensions in ignored_by_slot.items():
        slot = int(raw_slot)
        dimensions = tuple(int(value) for value in raw_dimensions)
        for name, group_axis in (*fine_fields, *coarse_fields):
            mask = action[name]
            for dimension in dimensions:
                index = [slice(None)] * mask.ndim
                index[group_axis] = slot
                index[-1] = dimension
                mask[tuple(index)] = False
        for dimension in dimensions:
            action["action_semantic_ids"][slot, dimension] = 0
            action["composition_operator_ids"][slot, dimension] = 0


def _ignored_action_slots(
    raw: Any,
    *,
    data_profile: DataProfile,
) -> dict[str, dict[int, tuple[int, ...]]]:
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise CacheDataError("direct ignored action dimensions must be a list")
    result: dict[str, dict[int, tuple[int, ...]]] = {}
    source_by_name = {source.name: source for source in data_profile.sources}
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"source", "group", "dimensions"}:
            raise CacheDataError("direct ignored action dimension entry is invalid")
        source_name = str(item["source"])
        source = source_by_name.get(source_name)
        if source is None:
            raise CacheDataError(f"ignored action source is unknown: {source_name}")
        embodiment = data_profile.embodiments[source.embodiment]
        group_name = str(item["group"])
        matches = [
            (slot, group)
            for slot, group in enumerate(embodiment.groups)
            if group.name == group_name
        ]
        if len(matches) != 1:
            raise CacheDataError(
                f"ignored action group is unknown: {source_name}/{group_name}"
            )
        slot, group = matches[0]
        dimensions = item["dimensions"]
        if (
            not isinstance(dimensions, list)
            or not dimensions
            or any(isinstance(value, bool) for value in dimensions)
        ):
            raise CacheDataError(
                "ignored action dimensions must be a non-empty integer list"
            )
        normalized = tuple(sorted({int(value) for value in dimensions}))
        if len(normalized) != len(dimensions) or any(
            not 0 <= value < group.action_dim for value in normalized
        ):
            raise CacheDataError(
                "ignored action dimensions are duplicate or out of range"
            )
        source_slots = result.setdefault(source_name, {})
        if slot in source_slots:
            raise CacheDataError(
                f"ignored action group is duplicated: {source_name}/{group_name}"
            )
        source_slots[slot] = normalized
    return result


@dataclass(frozen=True)
class _DirectEpisode:
    selected_rows: np.ndarray
    prepared_robot: PreparedEpisodeRobot


@dataclass(frozen=True)
class _DirectTaskRecord:
    task_id: str
    byte_offset: int


@dataclass(frozen=True)
class _DirectWindowPlan:
    raw_rows: np.ndarray
    boundary_rows: np.ndarray
    frame_keys: np.ndarray
    key_values: tuple[int, ...]
    prepared_robot: PreparedEpisodeRobot


class _PreparedViewRowStore:
    """Thread-safe byte-bounded LRU of fully preprocessed camera rows."""

    def __init__(self, maximum_bytes: int = 0) -> None:
        if int(maximum_bytes) < 0:
            raise CacheDataError("prepared row cache bytes cannot be negative")
        self.maximum_bytes = int(maximum_bytes)
        self._values: OrderedDict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_many(
        self,
        keys: Sequence[int],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if self.maximum_bytes == 0:
            return {}
        result: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        with self._lock:
            for key in dict.fromkeys(int(value) for value in keys):
                cached = self._values.pop(key, None)
                if cached is None:
                    self.misses += 1
                    continue
                self._values[key] = cached
                self.hits += 1
                result[key] = cached
        return result

    def put_many(
        self,
        keys: Sequence[int],
        images_u8: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> None:
        if self.maximum_bytes == 0:
            return
        if (
            images_u8.device.type != "cpu"
            or view_mask.device.type != "cpu"
            or images_u8.dtype != torch.uint8
            or view_mask.dtype != torch.bool
            or images_u8.ndim != 5
            or view_mask.ndim != 2
            or images_u8.shape[:2] != view_mask.shape
            or len(keys) != images_u8.shape[0]
        ):
            raise CacheDataError("prepared row cache received an invalid batch")
        with self._lock:
            for key_value, image_value, mask_value in zip(
                keys, images_u8, view_mask
            ):
                key = int(key_value)
                image = image_value.contiguous().clone()
                mask = mask_value.contiguous().clone()
                size = (
                    image.numel() * image.element_size()
                    + mask.numel() * mask.element_size()
                )
                previous = self._values.pop(key, None)
                if previous is not None:
                    self._bytes -= (
                        previous[0].numel() * previous[0].element_size()
                        + previous[1].numel() * previous[1].element_size()
                    )
                if size > self.maximum_bytes:
                    continue
                self._values[key] = (image, mask)
                self._bytes += size
                while self._bytes > self.maximum_bytes:
                    _old_key, old = self._values.popitem(last=False)
                    self._bytes -= (
                        old[0].numel() * old[0].element_size()
                        + old[1].numel() * old[1].element_size()
                    )
                    self.evictions += 1

    @property
    def metrics(self) -> Mapping[str, int]:
        with self._lock:
            return {
                "prepared_row_cache_bytes": self._bytes,
                "prepared_row_cache_entries": len(self._values),
                "prepared_row_cache_hits": self.hits,
                "prepared_row_cache_misses": self.misses,
                "prepared_row_cache_evictions": self.evictions,
            }


_TASK_ID_RE = re.compile(rb'"task_id"\s*:\s*"([0-9a-f]{64})"')
_TASK_SOURCE_RE = re.compile(rb'"source"\s*:\s*"([^"]+)"')
_TASK_EPISODE_RE = re.compile(rb'"episode_id"\s*:\s*"([^"]+)"')


def _task_header(line: bytes, *, line_number: int) -> tuple[str, str, str]:
    values: list[str] = []
    for name, pattern in (
        ("task_id", _TASK_ID_RE),
        ("source", _TASK_SOURCE_RE),
        ("episode_id", _TASK_EPISODE_RE),
    ):
        match = pattern.search(line)
        if match is None:
            raise CacheDataError(
                f"direct raw task manifest line {line_number} misses {name}"
            )
        values.append(match.group(1).decode("utf-8"))
    return values[0], values[1], values[2]


class _DirectWindowSource:
    """Read sealed raw windows without materializing episode feature caches."""

    def __init__(
        self,
        *,
        closure: Mapping[str, Any],
        profile: DataProfile,
    ) -> None:
        self.profile = profile
        self.task_manifest_path = Path(
            str(closure["task_manifest_path"])
        ).resolve(strict=True)
        if (
            self.task_manifest_path.is_symlink()
            or not self.task_manifest_path.is_file()
        ):
            raise CacheDataError("direct raw task manifest is not a regular file")
        self._task_record_by_episode: dict[
            tuple[str, str], _DirectTaskRecord
        ] = {}
        self._task_ordinal_by_id: dict[str, int] = {}
        self._task_offset_by_id: dict[str, int] = {}
        first_task: CacheTask | None = None
        with self.task_manifest_path.open("rb") as handle:
            line_number = 0
            while True:
                byte_offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                task_id, source, episode_id = _task_header(
                    line, line_number=line_number
                )
                record = _DirectTaskRecord(
                    task_id=task_id,
                    byte_offset=byte_offset,
                )
                if task_id in self._task_offset_by_id:
                    raise CacheDataError(
                        "direct raw task manifest has duplicate task ids"
                    )
                episode_key = (source, episode_id)
                if episode_key in self._task_record_by_episode:
                    raise CacheDataError(
                        "direct raw task manifest has duplicate episodes"
                    )
                self._task_offset_by_id[task_id] = byte_offset
                self._task_ordinal_by_id[task_id] = line_number - 1
                self._task_record_by_episode[episode_key] = record
                if first_task is None:
                    first_task = cache_task_from_mapping(
                        json.loads(line)
                    )
        if first_task is None:
            raise CacheDataError("direct raw task manifest is empty")
        self._task_cache: OrderedDict[str, CacheTask] = OrderedDict()
        # Multiple future windows may decode concurrently.  Serialize only
        # metadata/LRU mutation; immutable video reads remain parallel.
        self._metadata_lock = threading.RLock()
        # Multiple prefetch batches may run concurrently. Windows from the
        # same episode share a stripe so overlapping frames are never decoded
        # twice, while unrelated videos may still progress in parallel.
        self._decode_locks = tuple(threading.Lock() for _ in range(64))
        self.task_loads = 0
        self.task_hits = 0
        self.sources = {source.name: source for source in profile.sources}
        self.adapters = {
            source.name: load_adapter_contract(
                source.adapter_config_path,
                expected_sha256=source.adapter_contract_sha256,
            )
            for source in profile.sources
        }
        self.task_store = TaskEmbeddingStore(
            root=Path(str(closure["task_bank_root"])),
            index_sha256=str(closure["task_bank_index_sha256"]),
            expected_data_profile_sha256=profile.profile_sha256,
            expected_source_manifest_sha256_by_name={
                source.name: source.manifest_sha256 for source in profile.sources
            },
            expected_encoder_contract_sha256=(
                first_task.task_encoder_contract_sha256
            ),
        )
        self.asset_verifier = VerifiedAssetStore()
        self.input_rgb_size = int(closure["direct_input_rgb_size"])
        self.decode_workers = int(
            os.environ.get(
                "WM3D_DIRECT_DECODE_WORKERS",
                closure.get("direct_decode_workers", 4),
            )
        )
        self.robot_cache_episodes = int(
            closure.get("direct_robot_cache_episodes", 8)
        )
        self.video_indices = VideoTimestampIndexStore(
            maximum_assets=int(
                closure.get("direct_video_index_cache_assets", 64)
            )
        )
        prepared_cache_bytes = int(
            os.environ.get(
                "WM3D_DIRECT_PREPARED_ROW_CACHE_BYTES_PER_RANK",
                closure.get("direct_prepared_row_cache_bytes_per_rank", 0),
            )
        )
        self.prepared_rows = _PreparedViewRowStore(
            maximum_bytes=prepared_cache_bytes
        )
        if (
            self.input_rgb_size <= 0
            or self.input_rgb_size % 14
            or self.decode_workers <= 0
            or self.robot_cache_episodes <= 0
        ):
            raise CacheDataError("direct raw decode/cache configuration is invalid")
        self._episodes: OrderedDict[str, _DirectEpisode] = OrderedDict()
        self.episode_loads = 0
        self.episode_hits = 0
        self.windows_decoded = 0
        self.decode_calls = 0
        self.coalesced_batches = 0
        self.coalesced_requested_rows = 0
        self.coalesced_unique_rows = 0
        self.decode_seconds = 0.0

    def task_id_for_episode(self, source: str, episode_id: str) -> str:
        try:
            return self._task_record_by_episode[
                (source, episode_id)
            ].task_id
        except KeyError as exc:
            raise CacheDataError(
                f"direct raw metadata episode is absent: {source}/{episode_id}"
            ) from exc

    def _load_task(self, task_id: str) -> CacheTask:
        cached = self._task_cache.pop(task_id, None)
        if cached is not None:
            self._task_cache[task_id] = cached
            self.task_hits += 1
            return cached
        try:
            byte_offset = self._task_offset_by_id[task_id]
        except KeyError as exc:
            raise CacheDataError(
                f"direct raw task id is absent: {task_id}"
            ) from exc
        with self.task_manifest_path.open("rb") as handle:
            handle.seek(byte_offset)
            line = handle.readline()
        task = cache_task_from_mapping(json.loads(line))
        if task.task_id != task_id:
            raise CacheDataError(
                "direct raw task offset resolved to another task"
            )
        self._task_cache[task_id] = task
        while len(self._task_cache) > self.robot_cache_episodes * 2:
            self._task_cache.popitem(last=False)
        self.task_loads += 1
        return task

    def _load_episode(self, task: CacheTask) -> _DirectEpisode:
        cached = self._episodes.pop(task.task_id, None)
        if cached is not None:
            self._episodes[task.task_id] = cached
            self.episode_hits += 1
            return cached
        source = self.sources.get(task.source)
        if source is None or source.embodiment != task.embodiment:
            raise CacheDataError("direct raw source/embodiment mismatch")
        adapter = self.adapters[task.source]
        accessor = open_episode_accessor(
            task=task,
            source_root=source.raw_root,
            adapter=adapter,
            asset_verifier=self.asset_verifier,
        )
        observation_clock = np.asarray(
            accessor.array(adapter.observation_time_key), dtype=np.float64
        ).reshape(-1)
        if (
            observation_clock.shape != (task.observation_samples,)
            or canonical_timestamp_sha256(observation_clock)
            != task.observation_clock["timestamp_sha256"]
        ):
            raise CacheDataError("direct raw observation clock differs from task")
        selection = self.profile.cache_representation["state_frame_selection"]
        selected_rows = select_episode_cache_rows(
            observation_clock,
            minimum_separation_s=float(selection["minimum_separation_seconds"]),
        )
        embodiment = self.profile.embodiments[task.embodiment]
        robot = build_episode_robot_cache(
            embodiment=embodiment,
            action_series=adapt_action_series(
                accessor=accessor,
                contract=adapter,
                embodiment=embodiment,
            ),
            state_series=adapt_state_series(
                accessor=accessor,
                contract=adapter,
                embodiment=embodiment,
            ),
            task_embedding=self.task_store.get(task.task_text),
            observation_times_s=observation_clock,
            max_groups=max(
                len(item.groups) for item in self.profile.embodiments.values()
            ),
            max_action_dim=max(
                group.action_dim
                for item in self.profile.embodiments.values()
                for group in item.groups
            ),
            max_state_dim=max(
                max((group.state_dim for group in item.groups), default=0)
                for item in self.profile.embodiments.values()
            ),
        )
        result = _DirectEpisode(
            selected_rows=np.asarray(selected_rows, dtype=np.int64),
            prepared_robot=prepare_episode_robot_tensors(
                robot.as_tensors(), embodiment=embodiment
            ),
        )
        self._episodes[task.task_id] = result
        while len(self._episodes) > self.robot_cache_episodes:
            self._episodes.popitem(last=False)
        self.episode_loads += 1
        return result

    def _plan_window(
        self,
        *,
        entry: CacheIndexEntry,
        task_id: str,
        episode: _DirectEpisode,
    ) -> _DirectWindowPlan:
        feature_rows = entry.context_feature_rows + entry.future_feature_rows
        boundary_feature_rows = (entry.leading_feature_row,) + feature_rows
        if (
            min(boundary_feature_rows) < 0
            or max(boundary_feature_rows) >= len(episode.selected_rows)
        ):
            raise CacheDataError("direct raw feature row lies outside episode")
        raw_rows = episode.selected_rows[np.asarray(feature_rows, dtype=np.int64)]
        boundary_rows = episode.selected_rows[
            np.asarray(boundary_feature_rows, dtype=np.int64)
        ]
        if np.any(np.diff(boundary_rows) <= 0):
            raise CacheDataError("direct raw window boundaries are not increasing")
        task_ordinal = self._task_ordinal_by_id[task_id]
        if (
            task_ordinal < 0
            or task_ordinal >= 2**31
            or raw_rows.min() < 0
            or raw_rows.max() >= 2**32
        ):
            raise CacheDataError("direct raw frame identity exceeds int64 packing")
        frame_keys = (
            (np.int64(task_ordinal) << np.int64(32))
            | raw_rows.astype(np.int64, copy=False)
        )
        return _DirectWindowPlan(
            raw_rows=raw_rows,
            boundary_rows=boundary_rows,
            frame_keys=frame_keys,
            key_values=tuple(int(value) for value in frame_keys),
            prepared_robot=episode.prepared_robot,
        )

    def decode_windows(
        self,
        requests: Sequence[tuple[int, CacheIndexEntry, str]],
    ) -> dict[
        int,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            PreparedEpisodeRobot,
        ],
    ]:
        """Decode one prefetch batch after coalescing episode-local frames."""

        if not requests:
            return {}
        started = time.perf_counter()
        grouped: OrderedDict[
            str, list[tuple[int, CacheIndexEntry]]
        ] = OrderedDict()
        seen: set[int] = set()
        for raw_index, entry, task_id in requests:
            index = int(raw_index)
            if index in seen:
                raise CacheDataError(
                    "direct raw coalesced request duplicated an index"
                )
            seen.add(index)
            grouped.setdefault(str(task_id), []).append((index, entry))

        result: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                PreparedEpisodeRobot,
            ],
        ] = {}
        decode_calls = 0
        requested_rows = 0
        unique_rows = 0
        for task_id, items in grouped.items():
            with self._metadata_lock:
                task = self._load_task(task_id)
                episode = self._load_episode(task)
            plans = {
                index: self._plan_window(
                    entry=entry,
                    task_id=task_id,
                    episode=episode,
                )
                for index, entry in items
            }
            key_to_raw_row: OrderedDict[int, int] = OrderedDict()
            for plan in plans.values():
                requested_rows += len(plan.key_values)
                for key, row in zip(plan.key_values, plan.raw_rows):
                    key_to_raw_row.setdefault(int(key), int(row))
            unique_rows += len(key_to_raw_row)

            stripe = self._decode_locks[
                self._task_ordinal_by_id[task_id] % len(self._decode_locks)
            ]
            with stripe:
                prepared = self.prepared_rows.get_many(tuple(key_to_raw_row))
                missing = sorted(
                    (
                        (raw_row, key)
                        for key, raw_row in key_to_raw_row.items()
                        if key not in prepared
                    ),
                    key=lambda item: item[0],
                )
                if missing:
                    source = self.sources[task.source]
                    adapter = self.adapters[task.source]
                    slots = tuple(
                        str(item)
                        for item in self.profile.cache_representation["view_slots"]
                    )
                    decoded, _evidence = decode_episode_window_views(
                        task=task,
                        source_root=source.raw_root,
                        canonical_view_slots=slots,
                        selected_observation_rows=np.asarray(
                            [raw_row for raw_row, _key in missing],
                            dtype=np.int64,
                        ),
                        asset_verifier=self.asset_verifier,
                        timestamp_indices=self.video_indices,
                        decode_workers=self.decode_workers,
                    )
                    from scripts.data.run_cache_worker import _view_batch

                    missing_images, missing_mask = _view_batch(
                        decoded=decoded,
                        slots=slots,
                        input_size=self.input_rgb_size,
                        color_order_by_view={
                            view.name: view.color_order for view in adapter.views
                        },
                    )
                    missing_images_u8 = (
                        missing_images.mul(255)
                        .round()
                        .clamp(0, 255)
                        .to(torch.uint8)
                    )
                    missing_keys = tuple(key for _raw_row, key in missing)
                    missing_mask_bool = missing_mask.bool()
                    self.prepared_rows.put_many(
                        missing_keys,
                        missing_images_u8,
                        missing_mask_bool,
                    )
                    prepared.update(
                        {
                            key: (image, mask)
                            for key, image, mask in zip(
                                missing_keys,
                                missing_images_u8,
                                missing_mask_bool,
                            )
                        }
                    )
                    decode_calls += 1

            for index, plan in plans.items():
                result[index] = (
                    torch.stack(
                        [prepared[key][0] for key in plan.key_values],
                        dim=0,
                    ),
                    torch.stack(
                        [prepared[key][1] for key in plan.key_values],
                        dim=0,
                    ),
                    torch.from_numpy(plan.frame_keys.copy()).to(torch.int64),
                    torch.from_numpy(plan.boundary_rows.copy()).to(torch.int64),
                    plan.prepared_robot,
                )

        with self._metadata_lock:
            self.windows_decoded += len(requests)
            self.decode_calls += decode_calls
            self.coalesced_batches += 1
            self.coalesced_requested_rows += requested_rows
            self.coalesced_unique_rows += unique_rows
            self.decode_seconds += time.perf_counter() - started
        return result

    def decode_window(
        self,
        *,
        entry: CacheIndexEntry,
        task_id: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        PreparedEpisodeRobot,
    ]:
        return self.decode_windows(((0, entry, task_id),))[0]

    def metrics(self) -> Mapping[str, float | int]:
        return {
            **self.video_indices.metrics,
            **self.prepared_rows.metrics,
            "task_loads": self.task_loads,
            "task_hits": self.task_hits,
            "task_resident": len(self._task_cache),
            "episode_loads": self.episode_loads,
            "episode_hits": self.episode_hits,
            "robot_resident_episodes": len(self._episodes),
            "windows_decoded": self.windows_decoded,
            "decode_calls": self.decode_calls,
            "coalesced_batches": self.coalesced_batches,
            "coalesced_requested_rows": self.coalesced_requested_rows,
            "coalesced_unique_rows": self.coalesced_unique_rows,
            "decode_seconds": self.decode_seconds,
        }


class DirectRawDataset(UnifiedCacheDataset):
    """Raw T+K window dataset consumed by a rank-local direct VGGT adapter."""

    requires_main_process = True

    def __init__(
        self,
        *,
        closure: Mapping[str, Any],
        data_profile: DataProfile,
        model_profile: Mapping[str, Any],
        split: str,
        grouped_normalizer: GroupedRobotNormalizer,
        rank: int,
    ) -> None:
        if closure.get("schema") != DIRECT_RAW_DATA_CLOSURE_SCHEMA:
            raise CacheDataError("direct raw closure schema mismatch")
        super().__init__(
            cache_root=Path(str(closure["metadata_root"])),
            index_path=Path(str(closure["cache_index_path"])),
            index_sha256=str(closure["cache_index_sha256"]),
            data_profile=data_profile,
            model_profile=model_profile,
            split=split,
            verify_shard_sha_on_open=False,
            jpeg_reader_cache_size=0,
            robot_reader_cache_size=0,
            appearance_cache_grid=int(
                closure.get(
                    "appearance_token_grid",
                    data_profile.cache_representation["token_grid"],
                )
            ),
            grouped_normalizer=grouped_normalizer,
        )
        episode_index = load_cache_episode_index(
            Path(str(closure["episode_index_path"])),
            expected_sha256=str(closure["episode_index_sha256"]),
        )
        by_thin_feature = {
            item.feature_shard: (item.source, item.episode_id)
            for item in episode_index
        }
        self._source = _DirectWindowSource(closure=closure, profile=data_profile)
        self._task_id_by_feature: dict[str, str] = {}
        for relative, identity in by_thin_feature.items():
            self._task_id_by_feature[relative] = (
                self._source.task_id_for_episode(*identity)
            )
        order = {
            name: index for index, name in enumerate(data_profile.source_order)
        }
        self.entries = tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    order[entry.source],
                    self._task_id_by_feature[entry.feature_shard],
                    entry.leading_feature_row,
                    entry.sample_id,
                ),
            )
        )
        spans: dict[str, tuple[int, int]] = {}
        episode_spans: dict[str, list[tuple[int, int]]] = {
            name: [] for name in self._source_names
        }
        cursor = 0
        for source in self._source_names:
            source_start = cursor
            while cursor < len(self.entries) and self.entries[cursor].source == source:
                task_id = self._task_id_by_feature[
                    self.entries[cursor].feature_shard
                ]
                episode_start = cursor
                while (
                    cursor < len(self.entries)
                    and self.entries[cursor].source == source
                    and self._task_id_by_feature[
                        self.entries[cursor].feature_shard
                    ]
                    == task_id
                ):
                    cursor += 1
                episode_spans[source].append((episode_start, cursor))
            spans[source] = (source_start, cursor)
        if cursor != len(self.entries):
            raise CacheDataError("direct raw entries do not form source spans")
        self._source_spans = spans
        self._source_episode_spans = {
            name: tuple(value) for name, value in episode_spans.items()
        }
        self.rank = int(rank)
        self._ignored_action_slots = _ignored_action_slots(
            closure.get("direct_ignored_action_dimensions"),
            data_profile=data_profile,
        )
        self.max_prefetch_windows = int(
            closure.get("direct_prefetch_windows", 32)
        )
        if self.max_prefetch_windows <= 0:
            raise CacheDataError("direct raw prefetch window count must be positive")
        self.prefetch_workers = int(
            os.environ.get("WM3D_DIRECT_PREFETCH_WORKERS", "1")
        )
        if not 0 < self.prefetch_workers <= self.max_prefetch_windows:
            raise CacheDataError(
                "WM3D_DIRECT_PREFETCH_WORKERS must be in "
                f"[1,{self.max_prefetch_windows}]"
            )
        self._executor = ThreadPoolExecutor(
            max_workers=self.prefetch_workers,
            thread_name_prefix=f"wm3d-direct-rank-{self.rank}",
        )
        self._futures: dict[int, Future[Any]] = {}
        self.prefetch_submitted = 0
        self.prefetch_consumed = 0
        self.prefetch_capacity_skips = 0
        self.prefetch_wait_seconds = 0.0

    @property
    def source_episode_spans(self) -> Mapping[str, tuple[tuple[int, int], ...]]:
        return dict(self._source_episode_spans)

    @property
    def direct_raw_metrics(self) -> Mapping[str, float | int]:
        return {
            **self._source.metrics(),
            "prefetch_submitted": self.prefetch_submitted,
            "prefetch_consumed": self.prefetch_consumed,
            "prefetch_capacity_skips": self.prefetch_capacity_skips,
            "prefetch_pending": len(self._futures),
            "prefetch_wait_seconds": self.prefetch_wait_seconds,
            "prefetch_workers": self.prefetch_workers,
        }

    def _decode_index(self, index: int) -> tuple[Any, ...]:
        entry = self.entries[int(index)]
        task_id = self._task_id_by_feature[entry.feature_shard]
        return self._source.decode_window(
            entry=entry,
            task_id=task_id,
        )

    def _decode_indices(
        self, indices: Sequence[int]
    ) -> dict[int, tuple[Any, ...]]:
        requests = tuple(
            (
                int(index),
                self.entries[int(index)],
                self._task_id_by_feature[
                    self.entries[int(index)].feature_shard
                ],
            )
            for index in indices
        )
        return self._source.decode_windows(requests)

    def prefetch_indices(self, indices: list[int]) -> None:
        selected: list[int] = []
        selected_set: set[int] = set()
        for raw_index in indices:
            index = int(raw_index)
            if index in self._futures or index in selected_set:
                continue
            if len(self._futures) + len(selected) >= self.max_prefetch_windows:
                self.prefetch_capacity_skips += 1
                break
            selected.append(index)
            selected_set.add(index)
        if not selected:
            return
        future = self._executor.submit(self._decode_indices, tuple(selected))
        for index in selected:
            self._futures[index] = future
        self.prefetch_submitted += len(selected)

    def _window(self, index: int) -> tuple[Any, ...]:
        future = self._futures.pop(int(index), None)
        if future is None:
            future = self._executor.submit(self._decode_indices, (int(index),))
            self.prefetch_submitted += 1
        started = time.perf_counter()
        result = future.result()[int(index)]
        self.prefetch_wait_seconds += time.perf_counter() - started
        self.prefetch_consumed += 1
        return result

    def _robot_fields(
        self,
        *,
        entry: CacheIndexEntry,
        boundary_source: torch.Tensor,
        prepared_robot: PreparedEpisodeRobot,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        embodiment = self.data_profile.embodiments[entry.embodiment]
        action = assemble_robot_window_from_prepared_episode(
            prepared=prepared_robot,
            embodiment=embodiment,
            selected_source_boundary_indices=boundary_source.tolist(),
            limits=self.limits,
            context_samples=self.T,
            max_policy_queries=int(self.model["max_policy_queries"]),
            policy_target_horizon_s=float(
                self.sampling["policy_target_horizon_seconds"]
            ),
        )
        normalization = self.grouped_normalizer.tensors_for(
            source=entry.source,
            embodiment_id=int(action["embodiment_ids"]),
            group_ids=action["action_group_ids"],
            action_semantic_ids=action["action_semantic_ids"],
            state_semantic_ids=action["state_semantic_ids"],
        )
        ignored = self._ignored_action_slots.get(entry.source)
        if ignored:
            _apply_ignored_action_dimensions(action, ignored)
        for slot, (group, series) in enumerate(
            zip(embodiment.groups, prepared_robot.action_series)
        ):
            fine_available = bool(
                normalization.fine_action_available[
                    slot, : group.action_dim
                ].any()
            )
            coarse_available = bool(
                normalization.coarse_action_available[
                    slot, : group.action_dim
                ].any()
            )
            if (
                (series.supervision == "fine_command") != fine_available
                or (series.supervision == "coarse_effect") != coarse_available
            ):
                raise CacheDataError(
                    f"sample {entry.sample_id} action normalization lane drifted"
                )
        for mask, group_axis, lane in (
            (action["history_fine_action_mask"], 1, "fine_command"),
            (action["future_factual_fine_action_mask"], 1, "fine_command"),
            (action["target_fine_action_mask"], 0, "fine_command"),
            (action["history_coarse_action_mask"], 1, "coarse_effect"),
            (action["future_factual_coarse_action_mask"], 1, "coarse_effect"),
            (action["target_coarse_action_mask"], 1, "coarse_effect"),
            (action["current_state_mask"], 0, "current_state"),
        ):
            available = (
                normalization.fine_action_available
                if lane == "fine_command"
                else (
                    normalization.coarse_action_available
                    if lane == "coarse_effect"
                    else normalization.state_available
                )
            )
            try:
                validate_grouped_lane_mask(
                    mask,
                    available=available,
                    group_axis=group_axis,
                    lane=lane,
                )
            except GroupedNormalizationError as exc:
                raise CacheDataError(
                    f"sample {entry.sample_id}: {exc}"
                ) from exc
        for name in (
            "history_fine_action_values",
            "future_factual_fine_action_values",
        ):
            action[name] = normalize_grouped_masked(
                action[name],
                action[name.replace("values", "mask")],
                offset=normalization.fine_action_offset,
                scale=normalization.fine_action_scale,
                group_axis=1,
            )
        for name in (
            "history_coarse_action_values",
            "future_factual_coarse_action_values",
        ):
            action[name] = normalize_grouped_masked(
                action[name],
                action[name.replace("values", "mask")],
                offset=normalization.coarse_action_offset,
                scale=normalization.coarse_action_scale,
                group_axis=1,
            )
        action["current_state_values"] = normalize_grouped_masked(
            action["current_state_values"],
            action["current_state_mask"],
            offset=normalization.state_offset,
            scale=normalization.state_scale,
            group_axis=0,
        )
        action["target_fine_action"] = normalize_grouped_masked(
            action["target_fine_action"],
            action["target_fine_action_mask"],
            offset=normalization.fine_action_offset,
            scale=normalization.fine_action_scale,
            group_axis=0,
        )
        action["target_coarse_action_normalized"] = normalize_grouped_masked(
            action["target_coarse_action"],
            action["target_coarse_action_mask"],
            offset=normalization.coarse_action_offset,
            scale=normalization.coarse_action_scale,
            group_axis=1,
        )
        action["action_normalization_offset"] = torch.where(
            normalization.fine_action_available,
            normalization.fine_action_offset,
            normalization.coarse_action_offset,
        )
        action["action_normalization_scale"] = torch.where(
            normalization.fine_action_available,
            normalization.fine_action_scale,
            normalization.coarse_action_scale,
        )
        action["state_normalization_offset"] = normalization.state_offset
        action["state_normalization_scale"] = normalization.state_scale
        return action, prepared_robot.task_embedding

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry = self.entries[int(index)]
        (
            images,
            view_mask,
            frame_keys,
            boundary_source,
            prepared_robot,
        ) = self._window(index)
        action, task_embedding = self._robot_fields(
            entry=entry,
            boundary_source=boundary_source,
            prepared_robot=prepared_robot,
        )
        if task_embedding.numel() != int(self.model["task_dim"]):
            raise CacheDataError("direct raw task embedding dimension mismatch")
        frame_rows = boundary_source[1:]
        world_times = torch.from_numpy(
            prepared_robot.observation_times_s[frame_rows.numpy()].copy()
        ).to(torch.float64)
        aux_values = torch.zeros(
            self.T,
            int(self.model["max_aux_tokens"]),
            int(self.model["aux_dim"]),
            dtype=torch.float32,
        )
        result = {
            **action,
            "direct_rgb_uint8": images,
            "direct_view_mask": view_mask.bool(),
            "direct_frame_keys": frame_keys,
            "world_times_s": world_times,
            "task_embedding": task_embedding.to(torch.float32),
            "aux_values": aux_values,
            "aux_mask": torch.zeros(aux_values.shape[:-1], dtype=torch.bool),
            "aux_type_ids": torch.zeros(aux_values.shape[:-1], dtype=torch.int64),
            "rgb_frame_indices": torch.tensor(
                self.rgb_indices, dtype=torch.int64
            ),
            "source_id": torch.tensor(
                self.source_to_id[entry.source], dtype=torch.long
            ),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
        self._validate_sample(result, entry)
        return result
