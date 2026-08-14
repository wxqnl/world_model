"""Bounded just-in-time episode cache for training directly from raw datasets.

The streaming path deliberately reuses the ordinary WM3D episode-cache writer
and reader.  Raw videos are decoded and encoded once when an episode first
enters a rank-local LRU; every window from that episode then follows the exact
same quantized feature/RGB/robot ABI as the normal precomputed-cache path.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import shutil
import socket
import time
from typing import Any, Mapping

import torch

from .cache_tasks import CacheTask, cache_task_from_mapping
from .episode_io import VerifiedAssetStore
from .manifest_contract import (
    CacheEpisodeEntry,
    CacheIndexEntry,
    DataProfile,
    iter_jsonl,
    load_cache_episode_index,
    sha256_file,
)
from .source_adapters import load_adapter_contract
from .task_embedding_store import TaskEmbeddingStore
from .unified_cache_dataset import (
    CacheDataError,
    UnifiedCacheDataset,
    _JpegStore,
    _RobotStore,
    _ShardStore,
)


STREAMING_METADATA_SEAL_SCHEMA = "wm3d_streaming_metadata_v1"
STREAMING_DATA_CLOSURE_SCHEMA = "wm3d_streaming_raw_data_closure_v1"


class StreamingRawError(RuntimeError):
    pass


def _episode_row(path: Path) -> CacheEpisodeEntry:
    if path.is_symlink() or not path.is_file():
        raise StreamingRawError(f"streaming cache episode fragment is missing: {path}")
    entries = load_cache_episode_index(path, expected_sha256=sha256_file(path))
    if len(entries) != 1:
        raise StreamingRawError("streaming cache fragment must contain one episode")
    return entries[0]


def load_streaming_metadata_seal(
    path: Path, *, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise StreamingRawError("streaming metadata seal must be a regular file")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise StreamingRawError("streaming metadata seal SHA mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "data_profile_path",
        "data_profile_sha256",
        "model_profile_path",
        "model_profile_sha256",
        "task_manifest_path",
        "task_manifest_sha256",
        "task_count",
        "metadata_root",
        "episode_index_path",
        "episode_index_sha256",
        "episode_count",
        "window_index_path",
        "window_index_sha256",
        "window_count",
        "grouped_normalization_path",
        "grouped_normalization_sha256",
        "encoder_contract_path",
        "encoder_contract_sha256",
        "task_bank_root",
        "task_bank_index_sha256",
        "task_encoder_contract_sha256",
        "representation_contract_sha256",
        "source_manifest_sha256_by_name",
        "adapter_contract_sha256_by_name",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise StreamingRawError("streaming metadata seal fields mismatch")
    if value.get("schema") != STREAMING_METADATA_SEAL_SCHEMA:
        raise StreamingRawError("streaming metadata seal schema mismatch")
    for count in ("task_count", "episode_count", "window_count"):
        if isinstance(value[count], bool) or int(value[count]) <= 0:
            raise StreamingRawError(f"streaming metadata {count} must be positive")
    if int(value["task_count"]) != int(value["episode_count"]):
        raise StreamingRawError("streaming metadata requires one episode per task")
    path_fields = {
        "data_profile_path": "data_profile_sha256",
        "model_profile_path": "model_profile_sha256",
        "task_manifest_path": "task_manifest_sha256",
        "episode_index_path": "episode_index_sha256",
        "window_index_path": "window_index_sha256",
        "grouped_normalization_path": "grouped_normalization_sha256",
        "encoder_contract_path": "encoder_contract_sha256",
    }
    for path_name, sha_name in path_fields.items():
        candidate = Path(str(value[path_name]))
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or not candidate.is_file()
            or sha256_file(candidate) != value[sha_name]
        ):
            raise StreamingRawError(f"streaming metadata {path_name} is invalid")
    for root_name in ("metadata_root", "task_bank_root"):
        root = Path(str(value[root_name]))
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise StreamingRawError(f"streaming metadata {root_name} is invalid")
    return value


class _StreamingEpisodeCache:
    """Generate full cache episodes lazily and keep a strict byte-bounded LRU."""

    def __init__(
        self,
        *,
        closure: Mapping[str, Any],
        profile: DataProfile,
        device: torch.device,
        rank: int,
    ) -> None:
        self.profile = profile
        self.device = torch.device(device)
        self.rank = int(rank)
        self.max_bytes = int(closure["lru_max_bytes_per_rank"])
        if self.max_bytes <= 0:
            raise StreamingRawError("streaming LRU budget must be positive")
        parent = Path(str(closure["lru_root"]))
        if not parent.is_absolute() or parent.is_symlink():
            raise StreamingRawError("streaming LRU root must be an absolute non-symlink")
        parent.mkdir(parents=True, exist_ok=True)
        self.root = parent / socket.gethostname() / f"rank_{self.rank:05d}"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise StreamingRawError("streaming rank LRU root cannot be a symlink")
        self.batch_frames = int(closure["encode_batch_frames"])
        self.decode_workers = int(closure["decode_workers"])
        if self.batch_frames <= 0 or self.decode_workers <= 0:
            raise StreamingRawError("streaming encode/decode concurrency is invalid")

        tasks = tuple(
            cache_task_from_mapping(dict(row))
            for _line, row in iter_jsonl(Path(str(closure["task_manifest_path"])))
        )
        if not tasks:
            raise StreamingRawError("streaming task manifest is empty")
        self.tasks = {task.task_id: task for task in tasks}
        if len(self.tasks) != len(tasks):
            raise StreamingRawError("streaming task manifest has duplicate identities")
        self.task_by_episode = {(task.source, task.episode_id): task for task in tasks}
        if len(self.task_by_episode) != len(tasks):
            raise StreamingRawError("streaming task manifest has duplicate source episodes")
        self.sources = {source.name: source for source in profile.sources}
        self.adapters = {
            source.name: load_adapter_contract(
                source.adapter_config_path,
                expected_sha256=source.adapter_contract_sha256,
            )
            for source in profile.sources
        }
        self.encoder_contract = Path(str(closure["encoder_contract_path"]))
        self.task_bank_root = Path(str(closure["task_bank_root"]))
        task_encoder_digests = {task.task_encoder_contract_sha256 for task in tasks}
        if len(task_encoder_digests) != 1:
            raise StreamingRawError("streaming tasks mix task-encoder contracts")
        self.task_store = TaskEmbeddingStore(
            root=self.task_bank_root,
            index_sha256=str(closure["task_bank_index_sha256"]),
            expected_data_profile_sha256=profile.profile_sha256,
            expected_source_manifest_sha256_by_name={
                source.name: source.manifest_sha256 for source in profile.sources
            },
            expected_encoder_contract_sha256=next(iter(task_encoder_digests)),
        )
        self.asset_verifier = VerifiedAssetStore()
        self._encoder: Any | None = None
        self._encoder_config: Any | None = None
        self._entries: OrderedDict[str, tuple[CacheEpisodeEntry, int]] = OrderedDict()
        self.generated_episodes = 0
        self.cache_hits = 0
        self.evicted_episodes = 0
        self.prepare_seconds = 0.0
        self.encode_seconds = 0.0

    def task_for_episode(self, source: str, episode_id: str) -> CacheTask:
        try:
            return self.task_by_episode[(source, episode_id)]
        except KeyError as exc:
            raise StreamingRawError(
                f"metadata episode {source}/{episode_id} is absent from task manifest"
            ) from exc

    @staticmethod
    def _entry_bytes(root: Path, entry: CacheEpisodeEntry) -> int:
        total = 0
        for relative in (entry.feature_shard, entry.robot_shard, entry.rgb_pack):
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise StreamingRawError(f"streaming LRU payload is missing: {path}")
            total += int(path.stat().st_size)
        return total

    def _load_existing(self, task: CacheTask) -> CacheEpisodeEntry | None:
        fragment = self.root / "episode_index_fragments" / f"{task.task_id}.jsonl"
        if not fragment.exists():
            return None
        entry = _episode_row(fragment)
        if (
            entry.episode_id != task.episode_id
            or entry.source != task.source
            or entry.split != task.split
            or entry.embodiment != task.embodiment
        ):
            raise StreamingRawError("streaming LRU episode/task identity mismatch")
        expected = (
            (entry.feature_shard, entry.feature_sha256),
            (entry.robot_shard, entry.robot_sha256),
            (entry.rgb_pack, entry.rgb_pack_sha256),
        )
        for relative, digest in expected:
            path = self.root / relative
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise StreamingRawError("streaming LRU episode payload SHA mismatch")
        return entry

    def _load_encoder(self) -> tuple[Any, Any]:
        if self._encoder is None:
            from scripts.data.run_cache_worker import _strict_encoder
            from wm3d.encoders.native_vggt import NativeVGGTEncoder

            self._encoder_config = _strict_encoder(self.encoder_contract)
            self._encoder = NativeVGGTEncoder(
                self._encoder_config, device=str(self.device)
            ).eval()
        return self._encoder, self._encoder_config

    def _materialize(self, task: CacheTask) -> CacheEpisodeEntry:
        from scripts.data.run_cache_worker import (
            _encode_task,
            _prepare_task,
            _write_task,
        )

        source = self.sources.get(task.source)
        if source is None or source.embodiment != task.embodiment:
            raise StreamingRawError("streaming task source/embodiment mismatch")
        encoder, config = self._load_encoder()
        prepared = _prepare_task(
            task=task,
            source=source,
            adapter=self.adapters[task.source],
            profile=self.profile,
            task_store=self.task_store,
            asset_verifier=self.asset_verifier,
            encoder_input_size=int(config.input_rgb_size),
            task_bank_index_sha256=task.task_bank_index_sha256,
            decode_workers=self.decode_workers,
        )
        encoded = _encode_task(
            prepared,
            encoder=encoder,
            device=self.device,
            batch_frames=self.batch_frames,
        )
        _write_task(encoded, cache_root=self.root)
        entry = self._load_existing(task)
        if entry is None:
            raise StreamingRawError("streaming cache writer did not publish an episode")
        self.generated_episodes += 1
        self.prepare_seconds += float(encoded.prepare_seconds)
        self.encode_seconds += float(encoded.encode_seconds)
        print(
            json.dumps(
                {
                    "streaming_raw_cache": "miss",
                    "task_id": task.task_id,
                    "source": task.source,
                    "episode_id": task.episode_id,
                    "frames": entry.frame_count,
                    "prepare_seconds": round(float(encoded.prepare_seconds), 3),
                    "encode_seconds": round(float(encoded.encode_seconds), 3),
                    "rank": self.rank,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return entry

    def _remove_task(self, task_id: str, entry: CacheEpisodeEntry) -> None:
        payload = self.root / "payload" / "tasks" / task_id[:2] / task_id
        if payload.is_symlink():
            raise StreamingRawError("streaming LRU payload directory became a symlink")
        shutil.rmtree(payload, ignore_errors=False)
        for path in (
            self.root / "episode_index_fragments" / f"{task_id}.jsonl",
            self.root / "receipts" / f"{task_id}.json",
            self.root / "claims" / f"{task_id}.claim",
        ):
            if path.exists() or path.is_symlink():
                path.unlink()
        self.evicted_episodes += 1

    def _trim(self, *, protected: str) -> None:
        total = sum(size for _entry, size in self._entries.values())
        while total > self.max_bytes and len(self._entries) > 1:
            task_id, (entry, size) = self._entries.popitem(last=False)
            if task_id == protected:
                self._entries[task_id] = (entry, size)
                continue
            self._remove_task(task_id, entry)
            total -= size
        if total > self.max_bytes:
            raise StreamingRawError(
                "one streaming episode exceeds the per-rank LRU byte budget"
            )

    def ensure(self, task: CacheTask) -> CacheEpisodeEntry:
        cached = self._entries.pop(task.task_id, None)
        if cached is not None:
            self.cache_hits += 1
            self._entries[task.task_id] = cached
            return cached[0]
        started = time.perf_counter()
        entry = self._load_existing(task)
        if entry is None:
            entry = self._materialize(task)
        else:
            self.cache_hits += 1
        size = self._entry_bytes(self.root, entry)
        self._entries[task.task_id] = (entry, size)
        self._trim(protected=task.task_id)
        _ = started
        return entry

    def metrics(self) -> Mapping[str, float | int]:
        return {
            "generated_episodes": self.generated_episodes,
            "cache_hits": self.cache_hits,
            "evicted_episodes": self.evicted_episodes,
            "prepare_seconds": self.prepare_seconds,
            "encode_seconds": self.encode_seconds,
            "resident_bytes": sum(size for _entry, size in self._entries.values()),
            "resident_episodes": len(self._entries),
        }


class StreamingRawDataset(UnifiedCacheDataset):
    """Window dataset backed by a bounded rank-local just-in-time cache."""

    requires_main_process = True

    def __init__(
        self,
        *,
        closure: Mapping[str, Any],
        data_profile: DataProfile,
        model_profile: Mapping[str, Any],
        split: str,
        grouped_normalizer: Any,
        device: torch.device,
        rank: int,
    ) -> None:
        super().__init__(
            cache_root=Path(str(closure["metadata_root"])),
            index_path=Path(str(closure["cache_index_path"])),
            index_sha256=str(closure["cache_index_sha256"]),
            data_profile=data_profile,
            model_profile=model_profile,
            split=split,
            verify_shard_sha_on_open=True,
            jpeg_reader_cache_size=0,
            robot_reader_cache_size=0,
            grouped_normalizer=grouped_normalizer,
        )
        episode_index = load_cache_episode_index(
            Path(str(closure["episode_index_path"])),
            expected_sha256=str(closure["episode_index_sha256"]),
        )
        by_thin_feature = {
            item.feature_shard: (item.source, item.episode_id) for item in episode_index
        }
        if len(by_thin_feature) != len(episode_index):
            raise CacheDataError("streaming metadata feature shards are not unique")
        self._manager = _StreamingEpisodeCache(
            closure=closure,
            profile=data_profile,
            device=device,
            rank=rank,
        )
        self._task_id_by_feature: dict[str, str] = {}
        for relative, (source, episode_id) in by_thin_feature.items():
            self._task_id_by_feature[relative] = self._manager.task_for_episode(
                source, episode_id
            ).task_id
        order = {name: index for index, name in enumerate(data_profile.source_order)}
        try:
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
        except KeyError as exc:
            raise CacheDataError("streaming window references unknown metadata episode") from exc
        spans: dict[str, tuple[int, int]] = {}
        episode_spans: dict[str, list[tuple[int, int]]] = {
            name: [] for name in self._source_names
        }
        cursor = 0
        for source in self._source_names:
            source_start = cursor
            while cursor < len(self.entries) and self.entries[cursor].source == source:
                task_id = self._task_id_by_feature[self.entries[cursor].feature_shard]
                episode_start = cursor
                while (
                    cursor < len(self.entries)
                    and self.entries[cursor].source == source
                    and self._task_id_by_feature[self.entries[cursor].feature_shard]
                    == task_id
                ):
                    cursor += 1
                episode_spans[source].append((episode_start, cursor))
            spans[source] = (source_start, cursor)
        if cursor != len(self.entries) or any(not value for value in episode_spans.values()):
            raise CacheDataError("streaming entries do not form source/episode spans")
        self._source_spans = spans
        self._source_episode_spans = {
            name: tuple(value) for name, value in episode_spans.items()
        }
        self.root = self._manager.root
        self.shards = _ShardStore(self.root, {}, verify_on_open=True)
        # Keep only the current episode pack open.  Linux keeps an already-open
        # inode valid during LRU eviction, and opening the next pack closes it.
        self.jpeg = _JpegStore(self.shards, 1)
        self.robot = _RobotStore(self.shards, 0)

    @property
    def source_episode_spans(self) -> Mapping[str, tuple[tuple[int, int], ...]]:
        return dict(self._source_episode_spans)

    @property
    def streaming_metrics(self) -> Mapping[str, float | int]:
        return self._manager.metrics()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        thin = self.entries[index]
        task_id = self._task_id_by_feature[thin.feature_shard]
        episode = self._manager.ensure(self._manager.tasks[task_id])
        for relative, digest in (
            (episode.feature_shard, episode.feature_sha256),
            (episode.robot_shard, episode.robot_sha256),
            (episode.rgb_pack, episode.rgb_pack_sha256),
        ):
            self.shards.register(relative, digest)
        entry = CacheIndexEntry(
            sample_id=thin.sample_id,
            source=thin.source,
            split=thin.split,
            embodiment=thin.embodiment,
            feature_shard=episode.feature_shard,
            feature_sha256=episode.feature_sha256,
            robot_shard=episode.robot_shard,
            robot_sha256=episode.robot_sha256,
            rgb_pack=episode.rgb_pack,
            rgb_pack_sha256=episode.rgb_pack_sha256,
            leading_feature_row=thin.leading_feature_row,
            context_feature_rows=thin.context_feature_rows,
            future_feature_rows=thin.future_feature_rows,
        )
        return self._sample_from_entry(entry, sample_index=index)
