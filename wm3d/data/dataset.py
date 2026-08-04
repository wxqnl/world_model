"""Immutable shard loader for WM3D windows.

Payload layout
--------------
Feature safetensors shards contain frame-major tensors:

``view_tokens_q`` [F,V,P,D] INT8
``view_tokens_scale`` [F,V,P,1] FP16
``view_mask`` [F,V] bool
``rgb_offsets``/``rgb_lengths`` [F,V] into an independent JPEG pack
``depth`` [F,V,P], ``point`` [F,V,P,3]
``geometry_confidence`` [F,V,P], ``camera_pose`` [F,V,9]
``aux_tokens`` [F,A,aux_dim], ``aux_mask`` [F,A] (optional)
``frame_summary_q``/``frame_summary_scale`` [F,D]/[F,1] (optional)

Action safetensors shards contain frame-aligned high-rate tensors:

``action_values`` [F,G,S,A], ``action_dim_mask`` [F,G,S,A]
``contact`` [F,G,S], ``contact_mask`` [F,G,S]

Parquet rows point to immutable shard offsets.  This avoids duplicating the
same encoded frame across overlapping T24/K16 windows.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from torch.utils.data import Dataset

from .contracts import (
    ContractError,
    DatasetContract,
    resolve_regular_file,
)
from .codec import JpegPackReader, dequantize_per_vector


WINDOW_INDEX_SCHEMA = "wm3d_v7_window_index_v1"
REQUIRED_WINDOW_COLUMNS = frozenset(
    {
        "window_id",
        "episode_id",
        "source",
        "feature_shard",
        "action_shard",
        "rgb_pack",
        "frame_offset",
        "action_offset",
        "frame_count",
        "episode_frame_start",
        "episode_frame_stop",
        "task_id",
        "embodiment_id",
        "action_group_ids",
        "action_group_mask",
    }
)


class DataIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowLoaderConfig:
    rgb_decode_indices: tuple[int, ...] = (3, 7, 11, 15)
    memory_slots: int = 12
    memory_stride_frames: int = 25
    row_group_cache_size: int = 4
    task_cache_size: int = 4096
    strict_shapes: bool = True

    def validate(self, contract: DatasetContract) -> None:
        if not self.rgb_decode_indices:
            raise ContractError("at least one RGB supervision frame is required")
        if any(index < 0 or index >= contract.K for index in self.rgb_decode_indices):
            raise ContractError("RGB decode index is outside K")
        if self.memory_slots < 0 or self.memory_stride_frames <= 0:
            raise ContractError("invalid low-frequency memory layout")
        if self.row_group_cache_size <= 0 or self.task_cache_size <= 0:
            raise ContractError("cache sizes must be positive")


class ParquetWindowIndex:
    """Random-access parquet index with bounded row-group caching."""

    def __init__(self, paths: Sequence[Path], cache_size: int = 4) -> None:
        self.paths = tuple(sorted(Path(path) for path in paths))
        self.cache_size = int(cache_size)
        if not self.paths:
            raise DataIntegrityError("window index contains no parquet parts")
        self._files = [pq.ParquetFile(path, memory_map=True) for path in self.paths]
        self._groups: list[tuple[int, int, int]] = []
        self._ends: list[int] = []
        total = 0
        required = set(REQUIRED_WINDOW_COLUMNS)
        for file_index, parquet_file in enumerate(self._files):
            columns = set(parquet_file.schema_arrow.names)
            missing = required.difference(columns)
            if missing:
                raise DataIntegrityError(
                    f"{self.paths[file_index]} misses columns {sorted(missing)}"
                )
            metadata = parquet_file.metadata
            for row_group in range(metadata.num_row_groups):
                rows = metadata.row_group(row_group).num_rows
                if rows <= 0:
                    raise DataIntegrityError(
                        f"empty row group in {self.paths[file_index]}"
                    )
                self._groups.append((file_index, row_group, total))
                total += rows
                self._ends.append(total)
        self.length = total
        self._cache: OrderedDict[tuple[int, int], pa.Table] = OrderedDict()

    def __len__(self) -> int:
        return self.length

    def _table(self, file_index: int, row_group: int) -> pa.Table:
        key = (file_index, row_group)
        table = self._cache.pop(key, None)
        if table is None:
            table = self._files[file_index].read_row_group(row_group)
        self._cache[key] = table
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return table

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        group_index = bisect_right(self._ends, index)
        file_index, row_group, start = self._groups[group_index]
        row = self._table(file_index, row_group).slice(index - start, 1)
        return {name: row.column(name)[0].as_py() for name in row.column_names}


class _SafeTensorShard:
    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve(strict=True)
        self._resolved: dict[str, Path] = {}

    def path(self, relative: str) -> Path:
        value = self._resolved.get(relative)
        if value is None:
            value = resolve_regular_file(self.dataset_root, relative)
            self._resolved[relative] = value
        return value

    def read_many(
        self,
        relative: str,
        slices: Mapping[str, tuple[int, int] | None],
    ) -> dict[str, torch.Tensor]:
        path = self.path(relative)
        output: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            missing = set(slices).difference(keys)
            if missing:
                raise DataIntegrityError(f"{relative} misses tensors {sorted(missing)}")
            for name, bounds in slices.items():
                if bounds is None:
                    output[name] = handle.get_tensor(name)
                else:
                    start, stop = bounds
                    output[name] = handle.get_slice(name)[start:stop]
        return output

    def optional(
        self,
        relative: str,
        names: Sequence[str],
        bounds: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        path = self.path(relative)
        output: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for name in names:
                if name in keys:
                    output[name] = handle.get_slice(name)[bounds[0] : bounds[1]]
        return output

    def read_quantized(
        self,
        relative: str,
        name: str,
        bounds: tuple[int, int],
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        tensors = self.read_many(
            relative,
            {
                f"{name}_q": bounds,
                f"{name}_scale": bounds,
            },
        )
        return dequantize_per_vector(
            tensors[f"{name}_q"],
            tensors[f"{name}_scale"],
            dtype=dtype,
        )

    def optional_quantized(
        self,
        relative: str,
        name: str,
        bounds: tuple[int, int],
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor | None:
        path = self.path(relative)
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            names = {f"{name}_q", f"{name}_scale"}
            if not names.issubset(keys):
                if names.intersection(keys):
                    raise DataIntegrityError(
                        f"{relative} has a partial quantized tensor {name}"
                    )
                return None
            quantized = handle.get_slice(f"{name}_q")[bounds[0] : bounds[1]]
            scale = handle.get_slice(f"{name}_scale")[bounds[0] : bounds[1]]
        return dequantize_per_vector(quantized, scale, dtype=dtype)


class _JpegPackStore:
    def __init__(self, shards: _SafeTensorShard, cache_size: int = 8) -> None:
        self.shards = shards
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[str, JpegPackReader] = OrderedDict()

    def reader(self, relative: str) -> JpegPackReader:
        reader = self.cache.pop(relative, None)
        if reader is None:
            reader = JpegPackReader(self.shards.path(relative))
        self.cache[relative] = reader
        while len(self.cache) > self.cache_size:
            _name, evicted = self.cache.popitem(last=False)
            evicted.close()
        return reader


class _TaskEmbeddingBank:
    def __init__(
        self,
        shards: _SafeTensorShard,
        relative: str,
        cache_size: int,
        expected_dim: int,
    ) -> None:
        self.shards = shards
        self.relative = relative
        self.cache_size = int(cache_size)
        self.expected_dim = int(expected_dim)
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        path = self.shards.path(relative)
        with safe_open(path, framework="pt", device="cpu") as handle:
            if "embeddings" not in handle.keys():
                raise DataIntegrityError(f"{relative} has no embeddings tensor")
            shape = handle.get_slice("embeddings").get_shape()
        if len(shape) != 2 or shape[1] != self.expected_dim:
            raise DataIntegrityError(
                f"task bank shape {shape} does not end in {self.expected_dim}"
            )
        self.length = int(shape[0])

    def __getitem__(self, task_id: int) -> torch.Tensor:
        task_id = int(task_id)
        if not 0 <= task_id < self.length:
            raise DataIntegrityError(f"task_id out of range: {task_id}")
        value = self.cache.pop(task_id, None)
        if value is None:
            value = self.shards.read_many(
                self.relative, {"embeddings": (task_id, task_id + 1)}
            )["embeddings"][0]
        self.cache[task_id] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value


def _pad_vector(
    values: Sequence[int | bool],
    length: int,
    *,
    dtype: torch.dtype,
    fill: int | bool,
) -> torch.Tensor:
    if len(values) > length:
        raise DataIntegrityError(
            f"metadata vector length {len(values)} exceeds {length}"
        )
    return torch.tensor(list(values) + [fill] * (length - len(values)), dtype=dtype)


class SourceDataset(Dataset[dict[str, torch.Tensor]]):
    """One immutable source/split of canonical wm3d windows."""

    def __init__(
        self,
        dataset_root: Path,
        contract: DatasetContract,
        *,
        source_name: str,
        split: str,
        config: WindowLoaderConfig | None = None,
    ) -> None:
        self.root = Path(dataset_root).resolve(strict=True)
        self.contract = contract
        self.source_name = str(source_name)
        self.split = str(split)
        self.config = config or WindowLoaderConfig()
        self.config.validate(contract)
        if self.source_name not in contract.source_order:
            raise DataIntegrityError(f"unknown source {self.source_name!r}")
        if self.split not in {"train", "val", "test"}:
            raise DataIntegrityError(f"unsupported split {self.split!r}")
        index_root = self.root / "indexes" / self.split / self.source_name
        if not index_root.is_dir() or index_root.is_symlink():
            raise DataIntegrityError(f"missing safe index directory {index_root}")
        paths = sorted(index_root.glob("part-*.parquet"))
        if any(path.is_symlink() for path in paths):
            raise DataIntegrityError("symlinked parquet indexes are forbidden")
        self.index = ParquetWindowIndex(
            paths, cache_size=self.config.row_group_cache_size
        )
        self.shards = _SafeTensorShard(self.root)
        self.jpeg_packs = _JpegPackStore(self.shards)
        self.tasks = _TaskEmbeddingBank(
            self.shards,
            "control/task_embeddings.safetensors",
            cache_size=self.config.task_cache_size,
            expected_dim=contract.task_dim,
        )

    def __len__(self) -> int:
        return len(self.index)

    def _memory(
        self,
        feature_relative: str,
        frame_offset: int,
        episode_start: int,
        episode_stop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = self.config.memory_slots
        dim = self.contract.token_dim
        if slots == 0:
            return torch.empty((0, dim), dtype=torch.bfloat16), torch.empty(
                0, dtype=torch.bool
            )
        positions = [
            frame_offset - self.config.memory_stride_frames * (slots - index)
            for index in range(slots)
        ]
        valid = [
            episode_start <= position < min(frame_offset, episode_stop)
            for position in positions
        ]
        output = torch.zeros((slots, dim), dtype=torch.bfloat16)
        if any(valid):
            valid_positions = [
                position for position, keep in zip(positions, valid) if keep
            ]
            start, stop = min(valid_positions), max(valid_positions) + 1
            summary = self.shards.optional_quantized(
                feature_relative,
                "frame_summary",
                (start, stop),
            )
            if summary is not None:
                for slot, (position, keep) in enumerate(zip(positions, valid)):
                    if keep:
                        output[slot] = summary[position - start]
            else:
                valid = [False] * slots
        return output, torch.tensor(valid, dtype=torch.bool)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.index[index]
        if str(row["source"]) != self.source_name:
            raise DataIntegrityError(
                f"index source drift: {row['source']!r} != {self.source_name!r}"
            )
        count = int(row["frame_count"])
        expected_count = self.contract.T + self.contract.K
        if count != expected_count:
            raise DataIntegrityError(
                f"window {row['window_id']} has {count} frames, expected {expected_count}"
            )
        frame_start = int(row["frame_offset"])
        frame_stop = frame_start + count
        action_start = int(row["action_offset"])
        action_stop = action_start + count
        feature_relative = str(row["feature_shard"])
        action_relative = str(row["action_shard"])
        feature = self.shards.read_many(
            feature_relative,
            {
                "view_mask": (frame_start, frame_stop),
                "rgb_offsets": (frame_start, frame_stop),
                "rgb_lengths": (frame_start, frame_stop),
                "depth": (frame_start, frame_stop),
                "point": (frame_start, frame_stop),
                "geometry_confidence": (frame_start, frame_stop),
                "camera_pose": (frame_start, frame_stop),
                "aux_tokens": (frame_start, frame_stop),
                "aux_mask": (frame_start, frame_stop),
            },
        )
        view_tokens = self.shards.read_quantized(
            feature_relative,
            "view_tokens",
            (frame_start, frame_stop),
        )
        action = self.shards.read_many(
            action_relative,
            {
                "action_values": (action_start, action_stop),
                "action_dim_mask": (action_start, action_stop),
                "contact": (action_start, action_stop),
                "contact_mask": (action_start, action_stop),
            },
        )
        T, K = self.contract.T, self.contract.K
        rgb_indices = torch.tensor(self.config.rgb_decode_indices, dtype=torch.long)
        rgb_absolute = rgb_indices + T
        selected_offsets = feature["rgb_offsets"].index_select(0, rgb_absolute)
        selected_lengths = feature["rgb_lengths"].index_select(0, rgb_absolute)
        rgb_reader = self.jpeg_packs.reader(str(row["rgb_pack"]))
        decoded_rgb = torch.stack(
            [
                rgb_reader.decode(offsets.tolist(), lengths.tolist())
                for offsets, lengths in zip(
                    selected_offsets,
                    selected_lengths,
                    strict=True,
                )
            ]
        )
        view_mask = feature["view_mask"].bool()
        if not bool(view_mask.any(dim=1).all()):
            raise DataIntegrityError(
                f"window {row['window_id']} contains a frame with no valid view"
            )
        confidence = (
            feature["geometry_confidence"].float().clamp_min(0.0) * view_mask[..., None]
        )
        confidence_sum = confidence.sum(dim=1)
        if not bool((confidence_sum[T : T + K] > 0.0).all()):
            raise DataIntegrityError(
                f"window {row['window_id']} target token has no geometry evidence"
            )
        weights = confidence[..., None]
        world_tokens = (view_tokens.float() * weights).sum(dim=1) / (
            confidence_sum.clamp_min(1.0e-6)[..., None]
        )
        group_ids = _pad_vector(
            row["action_group_ids"],
            self.contract.max_action_groups,
            dtype=torch.long,
            fill=0,
        )
        group_mask = _pad_vector(
            row["action_group_mask"],
            self.contract.max_action_groups,
            dtype=torch.bool,
            fill=False,
        )
        if not bool(group_mask.any()):
            raise DataIntegrityError(f"window {row['window_id']} has no action group")
        memory, memory_mask = self._memory(
            feature_relative,
            frame_start,
            int(row["episode_frame_start"]),
            int(row["episode_frame_stop"]),
        )
        result: dict[str, torch.Tensor] = {
            "world_tokens": view_tokens[:T],
            "view_mask": feature["view_mask"][:T].bool(),
            "target_view_mask": feature["view_mask"][T : T + K].bool(),
            "target_tokens": world_tokens[T : T + K].to(torch.bfloat16),
            "task_embedding": self.tasks[int(row["task_id"])],
            "context_action_values": action["action_values"][:T],
            "context_action_dim_mask": action["action_dim_mask"][:T].bool(),
            "future_factual_action_values": action["action_values"][T : T + K],
            "future_factual_action_dim_mask": action["action_dim_mask"][
                T : T + K
            ].bool(),
            "target_action_values": action["action_values"][T : T + K],
            "target_action_dim_mask": action["action_dim_mask"][T : T + K].bool(),
            "target_contact": action["contact"][T : T + K],
            "target_contact_mask": action["contact_mask"][T : T + K].bool(),
            "action_group_ids": group_ids,
            "action_group_mask": group_mask,
            "embodiment_ids": torch.tensor(int(row["embodiment_id"]), dtype=torch.long),
            "target_rgb": decoded_rgb.float().div_(255.0),
            "target_depth": feature["depth"][T : T + K],
            "target_point": feature["point"][T : T + K],
            "target_geometry_confidence": feature["geometry_confidence"][T : T + K],
            "target_camera_pose": feature["camera_pose"][T : T + K],
            "memory_tokens": memory,
            "memory_mask": memory_mask,
            "aux_tokens": feature["aux_tokens"][:T],
            "aux_mask": feature["aux_mask"][:T].bool(),
            "rgb_frame_indices": rgb_indices,
            "sample_index": torch.tensor(int(index), dtype=torch.long),
        }
        if self.config.strict_shapes:
            expected = {
                "world_tokens": (
                    T,
                    self.contract.num_views,
                    self.contract.P,
                    self.contract.token_dim,
                ),
                "target_tokens": (
                    K,
                    self.contract.P,
                    self.contract.token_dim,
                ),
                "target_rgb": (
                    len(self.config.rgb_decode_indices),
                    self.contract.num_views,
                    3,
                    decoded_rgb.shape[-2],
                    decoded_rgb.shape[-1],
                ),
                "target_depth": (
                    K,
                    self.contract.num_views,
                    self.contract.P,
                ),
                "target_point": (
                    K,
                    self.contract.num_views,
                    self.contract.P,
                    3,
                ),
                "aux_tokens": (
                    T,
                    self.contract.max_aux_tokens,
                    self.contract.aux_dim,
                ),
                "aux_mask": (
                    T,
                    self.contract.max_aux_tokens,
                ),
                "target_contact": (
                    K,
                    self.contract.max_action_groups,
                    self.contract.action_substeps,
                ),
                "target_contact_mask": (
                    K,
                    self.contract.max_action_groups,
                    self.contract.action_substeps,
                ),
            }
            for name, shape in expected.items():
                if tuple(result[name].shape) != shape:
                    raise DataIntegrityError(
                        f"window {row['window_id']} {name} shape "
                        f"{tuple(result[name].shape)} != {shape}"
                    )
        return result


class MixedDataset(Dataset[dict[str, torch.Tensor]]):
    """Concatenate sources in the exact sealed source order."""

    def __init__(self, sources: Sequence[tuple[str, SourceDataset]]) -> None:
        if not sources:
            raise DataIntegrityError("mixed wm3d dataset has no sources")
        self.source_names = tuple(str(name) for name, _dataset in sources)
        if len(set(self.source_names)) != len(self.source_names):
            raise DataIntegrityError("mixed wm3d source names are not unique")
        self.datasets = tuple(dataset for _name, dataset in sources)
        self.source_spans: dict[str, tuple[int, int]] = {}
        self._ends: list[int] = []
        offset = 0
        for name, dataset in sources:
            length = len(dataset)
            if length <= 0:
                raise DataIntegrityError(f"source {name} is empty")
            self.source_spans[str(name)] = (offset, offset + length)
            offset += length
            self._ends.append(offset)
        self.length = offset

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        index = int(index)
        if index < 0:
            index += self.length
        if index < 0 or index >= self.length:
            raise IndexError(index)
        source_id = bisect_right(self._ends, index)
        start = 0 if source_id == 0 else self._ends[source_id - 1]
        sample = self.datasets[source_id][index - start]
        sample["source_id"] = torch.tensor(source_id, dtype=torch.long)
        return sample
