"""Source discovery and normalized episode plans for native WM3D-V7 5B.

Raw vendors change layouts; the expensive encoder must not contain vendor
heuristics.  This module converts supported LeRobot or normalized manifests
into a small immutable episode plan.  Every downstream build consumes only
that plan plus the separately sealed dataset contract and source-layout file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .scale5b_contracts import (
    ContractError,
    atomic_write_json,
    canonical_sha256,
    resolve_real_directory,
    resolve_regular_file,
    safe_relative_path,
    sha256_file,
)


EPISODE_PLAN_SCHEMA = "wm3d_v7_native5b_episode_plan_v1"
SOURCE_LAYOUT_SCHEMA = "wm3d_v7_native5b_source_layout_v1"
SCAN_RECEIPT_SCHEMA = "wm3d_v7_native5b_source_scan_receipt_v1"


@dataclass(frozen=True)
class ActionColumnSpec:
    group_name: str
    column: str
    indices: tuple[int, ...]
    discrete: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActionColumnSpec":
        item = dict(value)
        item["indices"] = tuple(int(index) for index in item.get("indices", ()))
        result = cls(**item)
        if not result.group_name or not result.column or not result.indices:
            raise ContractError("action column requires group_name/column/indices")
        if len(result.indices) != len(set(result.indices)) or min(result.indices) < 0:
            raise ContractError(f"invalid action indices for {result.group_name}")
        return result


@dataclass(frozen=True)
class AuxiliaryColumnSpec:
    modality_name: str
    column: str
    indices: tuple[int, ...]
    discrete: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AuxiliaryColumnSpec":
        item = dict(value)
        item["indices"] = tuple(int(index) for index in item.get("indices", ()))
        result = cls(**item)
        if not result.modality_name or not result.column or not result.indices:
            raise ContractError(
                "auxiliary column requires modality_name/column/indices"
            )
        if len(result.indices) != len(set(result.indices)) or min(result.indices) < 0:
            raise ContractError(f"invalid auxiliary indices for {result.modality_name}")
        return result


@dataclass(frozen=True)
class SourceLayout:
    source: str
    adapter: str
    embodiment: str
    view_keys: Mapping[str, str | None]
    action_columns: tuple[ActionColumnSpec, ...]
    auxiliary_columns: tuple[AuxiliaryColumnSpec, ...] = ()
    timestamp_column: str = "timestamp"
    episode_column: str = "episode_index"
    task_column: str = "task_index"
    default_task: str = "robot manipulation"
    fps_override: float | None = None
    normalized_manifest_path: str | None = None
    collection_receipt_path: str | None = None
    collection_receipt_schema: str | None = None
    provenance_dataset: str | None = None
    forbidden_provenance_datasets: tuple[str, ...] = ()
    schema: str = SOURCE_LAYOUT_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceLayout":
        item = dict(value)
        item["view_keys"] = {
            str(name): None if key is None else str(key)
            for name, key in dict(item.get("view_keys", {})).items()
        }
        item["action_columns"] = tuple(
            ActionColumnSpec.from_mapping(spec)
            for spec in item.get("action_columns", ())
        )
        item["auxiliary_columns"] = tuple(
            AuxiliaryColumnSpec.from_mapping(spec)
            for spec in item.get("auxiliary_columns", ())
        )
        item["forbidden_provenance_datasets"] = tuple(
            str(name) for name in item.get("forbidden_provenance_datasets", ())
        )
        result = cls(**item)
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema != SOURCE_LAYOUT_SCHEMA:
            raise ContractError(f"unsupported source layout schema {self.schema}")
        if self.adapter not in {
            "lerobot",
            "lerobot_collection",
            "normalized_manifest",
        }:
            raise ContractError(f"unsupported source adapter {self.adapter}")
        if self.adapter == "normalized_manifest" and not self.normalized_manifest_path:
            raise ContractError(
                "normalized_manifest adapter requires its manifest path"
            )
        if self.normalized_manifest_path is not None:
            safe_relative_path(self.normalized_manifest_path)
        if (self.collection_receipt_path is None) != (
            self.collection_receipt_schema is None
        ):
            raise ContractError(
                "collection receipt path/schema must be provided together"
            )
        if self.collection_receipt_path is not None:
            if self.adapter != "lerobot_collection":
                raise ContractError(
                    "collection receipt only applies to lerobot_collection"
                )
            safe_relative_path(self.collection_receipt_path)
            if not self.collection_receipt_schema:
                raise ContractError("collection receipt schema cannot be empty")
        if self.provenance_dataset is not None and not self.provenance_dataset:
            raise ContractError("provenance_dataset cannot be empty")
        if len(self.forbidden_provenance_datasets) != len(
            set(self.forbidden_provenance_datasets)
        ) or any(not name for name in self.forbidden_provenance_datasets):
            raise ContractError(
                "forbidden provenance datasets must be unique/non-empty"
            )
        if self.adapter != "normalized_manifest" and self.forbidden_provenance_datasets:
            raise ContractError(
                "forbidden provenance datasets only apply to normalized manifests"
            )
        if tuple(self.view_keys) != ("head", "left_hand", "right_hand"):
            raise ContractError(
                "source view_keys must be ordered head,left_hand,right_hand"
            )
        available_views = [key for key in self.view_keys.values() if key is not None]
        if not available_views:
            raise ContractError("source layout must provide at least one RGB view")
        if len(set(available_views)) != len(available_views):
            raise ContractError("source view keys must be unique")
        if not self.action_columns:
            raise ContractError("source layout has no action columns")
        names = [item.group_name for item in self.action_columns]
        if len(names) != len(set(names)):
            raise ContractError("source layout has duplicate action group mappings")
        auxiliary_names = [item.modality_name for item in self.auxiliary_columns]
        if len(auxiliary_names) != len(set(auxiliary_names)):
            raise ContractError(
                "source layout has duplicate auxiliary modality mappings"
            )
        if self.fps_override is not None and float(self.fps_override) <= 0:
            raise ContractError("fps_override must be positive")


@dataclass(frozen=True)
class ViewSegment:
    canonical_name: str
    feature_key: str | None
    relative_path: str | None
    start_seconds: float
    stop_seconds: float


@dataclass(frozen=True)
class EpisodeDescriptor:
    source: str
    episode_id: str
    episode_index: int
    embodiment: str
    split: str
    task_text: str
    raw_root: str
    data_relative_path: str
    data_row_start: int
    data_row_stop: int
    timestamp_column: str
    episode_column: str
    source_fps: float
    duration_seconds: float
    views: tuple[ViewSegment, ...]
    action_columns: tuple[ActionColumnSpec, ...]
    auxiliary_columns: tuple[AuxiliaryColumnSpec, ...] = ()
    provenance_dataset: str | None = None
    schema: str = EPISODE_PLAN_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeDescriptor":
        item = dict(value)
        item["views"] = tuple(
            ViewSegment(**dict(view)) for view in item.get("views", ())
        )
        item["action_columns"] = tuple(
            ActionColumnSpec.from_mapping(spec)
            for spec in item.get("action_columns", ())
        )
        item["auxiliary_columns"] = tuple(
            AuxiliaryColumnSpec.from_mapping(spec)
            for spec in item.get("auxiliary_columns", ())
        )
        result = cls(**item)
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema != EPISODE_PLAN_SCHEMA:
            raise ContractError(f"unsupported episode schema {self.schema}")
        if self.split not in {"train", "val", "test"}:
            raise ContractError(f"unsupported episode split {self.split}")
        safe_relative_path(self.data_relative_path)
        if self.data_row_start < 0 or self.data_row_stop <= self.data_row_start:
            raise ContractError(f"invalid row interval for {self.episode_id}")
        if self.source_fps <= 0 or self.duration_seconds <= 0:
            raise ContractError(f"invalid timing for {self.episode_id}")
        if self.provenance_dataset is not None and not self.provenance_dataset:
            raise ContractError(f"empty provenance dataset for {self.episode_id}")
        if tuple(view.canonical_name for view in self.views) != (
            "head",
            "left_hand",
            "right_hand",
        ):
            raise ContractError(
                f"{self.episode_id} does not have canonical 3-view order"
            )
        if not any(view.relative_path is not None for view in self.views):
            raise ContractError(f"{self.episode_id} has no available RGB view")
        for view in self.views:
            if view.relative_path is None:
                if view.feature_key is not None:
                    raise ContractError(
                        f"missing view has a feature key for {self.episode_id}"
                    )
                continue
            safe_relative_path(view.relative_path)
            if view.start_seconds < 0 or view.stop_seconds <= view.start_seconds:
                raise ContractError(f"invalid video segment for {self.episode_id}")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _format_template(template: str, values: Mapping[str, Any]) -> str:
    fields = {
        name
        for _literal, name, _format, _conversion in Formatter().parse(template)
        if name
    }
    missing = fields.difference(values)
    if missing:
        raise ContractError(
            f"path template misses values {sorted(missing)}: {template}"
        )
    return template.format(**values)


def deterministic_split(
    source: str,
    episode_id: str,
    *,
    seed: int,
    train_fraction: float,
) -> str:
    if not 0.5 <= float(train_fraction) < 1.0:
        raise ContractError("train_fraction must lie in [0.5,1)")
    payload = f"{source}\x1f{episode_id}\x1f{int(seed)}".encode()
    unit = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    if unit < train_fraction:
        return "train"
    # Keep validation and test stable and equally sized.
    return "val" if unit < train_fraction + (1.0 - train_fraction) / 2.0 else "test"


def plan_shard(episode_id: str, num_shards: int) -> int:
    if int(num_shards) <= 0:
        raise ContractError("num_shards must be positive")
    digest = hashlib.sha256(str(episode_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % int(num_shards)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _metadata_rows(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.exists() or jsonl.is_symlink():
        return _read_jsonl(resolve_regular_file(root, "meta/episodes.jsonl"))
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise ContractError(f"cannot find LeRobot episode metadata under {root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        safe_path = resolve_regular_file(
            root,
            path.relative_to(root).as_posix(),
        )
        rows.extend(pq.read_table(safe_path).to_pylist())
    return rows


def _task_lookup(root: Path) -> dict[int, str]:
    rows: list[dict[str, Any]] = []
    jsonl = root / "meta" / "tasks.jsonl"
    parquet = root / "meta" / "tasks.parquet"
    if jsonl.exists() or jsonl.is_symlink():
        rows = _read_jsonl(resolve_regular_file(root, "meta/tasks.jsonl"))
    elif parquet.exists() or parquet.is_symlink():
        rows = pq.read_table(
            resolve_regular_file(root, "meta/tasks.parquet")
        ).to_pylist()
    result: dict[int, str] = {}
    for row in rows:
        index = int(row.get("task_index", row.get("index", len(result))))
        text = row.get("task", row.get("language_instruction", row.get("name")))
        if text:
            result[index] = str(text)
    return result


def _resolve_existing(root: Path, candidates: Sequence[str]) -> str:
    for relative in candidates:
        try:
            safe_relative_path(relative)
            resolve_regular_file(root, relative)
        except (ContractError, FileNotFoundError):
            continue
        return relative
    raise ContractError(f"none of the expected files exists under {root}: {candidates}")


def _lerobot_path_values(row: Mapping[str, Any], episode_index: int) -> dict[str, Any]:
    chunk = int(
        row.get(
            "chunk_index",
            row.get("data/chunk_index", episode_index // 1000),
        )
    )
    file_index = int(row.get("data/file_index", episode_index))
    return {
        "episode_index": int(episode_index),
        "episode_chunk": chunk,
        "chunk_index": chunk,
        "file_index": file_index,
    }


def _task_text(
    row: Mapping[str, Any],
    layout: SourceLayout,
    tasks: Mapping[int, str],
) -> str:
    direct = row.get("task", row.get("language_instruction"))
    if direct:
        return str(direct)
    values = row.get("tasks")
    if isinstance(values, Sequence) and not isinstance(values, str) and values:
        first = values[0]
        if isinstance(first, str):
            return first
        if int(first) in tasks:
            return tasks[int(first)]
    task_index = row.get(layout.task_column)
    if task_index is not None and int(task_index) in tasks:
        return tasks[int(task_index)]
    return layout.default_task


def scan_lerobot(
    root: Path,
    layout: SourceLayout,
    *,
    split_seed: int,
    train_fraction: float,
    episode_namespace: str | None = None,
) -> list[EpisodeDescriptor]:
    """Scan both file-per-episode and shared-file LeRobot layouts."""

    root = resolve_real_directory(root, f"{layout.source} raw root")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file() or info_path.is_symlink():
        raise ContractError(f"missing safe LeRobot metadata {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(layout.fps_override or info.get("fps", 0))
    if fps <= 0:
        raise ContractError("LeRobot source has no valid FPS")
    data_template = str(
        info.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        )
    )
    video_template = str(
        info.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4",
        )
    )
    tasks = _task_lookup(root)
    descriptors: list[EpisodeDescriptor] = []
    seen: set[int] = set()
    data_file_origins: dict[str, int] = {}
    for row in sorted(
        _metadata_rows(root), key=lambda item: int(item["episode_index"])
    ):
        episode_index = int(row["episode_index"])
        if episode_index in seen:
            raise ContractError(f"duplicate LeRobot episode {episode_index}")
        seen.add(episode_index)
        length = int(row.get("length", 0))
        if length <= 0:
            raise ContractError(f"episode {episode_index} has no positive length")
        values = _lerobot_path_values(row, episode_index)
        candidates: list[str] = []
        direct_data = row.get("data_path")
        if direct_data:
            candidates.append(str(direct_data))
        try:
            candidates.append(_format_template(data_template, values))
        except (KeyError, ValueError, ContractError):
            pass
        candidates.extend(
            [
                f"data/chunk-{values['chunk_index']:03d}/"
                f"episode_{episode_index:06d}.parquet",
                f"data/chunk-{values['chunk_index']:03d}/"
                f"file-{values['file_index']:03d}.parquet",
            ]
        )
        data_relative = _resolve_existing(root, candidates)
        explicit_start = row.get("data/from_index")
        explicit_stop = row.get("data/to_index")
        if (explicit_start is None) != (explicit_stop is None):
            raise ContractError(
                f"episode {episode_index} has an incomplete file-local row interval"
            )
        if explicit_start is not None:
            row_start = int(explicit_start)
            row_stop = int(explicit_stop)
        else:
            dataset_start = int(row.get("dataset_from_index", 0))
            dataset_stop = int(
                row.get("dataset_to_index", dataset_start + length)
            )
            origin = data_file_origins.setdefault(data_relative, dataset_start)
            if dataset_start < origin:
                raise ContractError(
                    f"episode {episode_index} precedes the first row of "
                    f"{data_relative}"
                )
            row_start = dataset_start - origin
            row_stop = dataset_stop - origin
        if row_start < 0 or row_stop - row_start != length:
            raise ContractError(
                f"episode {episode_index} row interval [{row_start}, {row_stop}) "
                f"does not match length {length}"
            )
        start_seconds = float(row.get("video/from_timestamp", 0.0))
        stop_seconds = float(row.get("video/to_timestamp", length / fps))
        views: list[ViewSegment] = []
        for canonical_name, feature_key in layout.view_keys.items():
            if feature_key is None:
                views.append(
                    ViewSegment(
                        canonical_name=canonical_name,
                        feature_key=None,
                        relative_path=None,
                        start_seconds=0.0,
                        stop_seconds=length / fps,
                    )
                )
                continue
            video_values = {**values, "video_key": feature_key}
            direct = row.get(f"videos/{feature_key}/path")
            video_candidates = [str(direct)] if direct else []
            try:
                video_candidates.append(_format_template(video_template, video_values))
            except (KeyError, ValueError, ContractError):
                pass
            video_chunk = int(
                row.get(f"videos/{feature_key}/chunk_index", values["chunk_index"])
            )
            video_file = int(
                row.get(f"videos/{feature_key}/file_index", values["file_index"])
            )
            video_candidates.extend(
                [
                    f"videos/chunk-{video_chunk:03d}/{feature_key}/"
                    f"episode_{episode_index:06d}.mp4",
                    f"videos/{feature_key}/chunk-{video_chunk:03d}/"
                    f"file-{video_file:03d}.mp4",
                ]
            )
            relative = _resolve_existing(root, video_candidates)
            view_start = float(
                row.get(f"videos/{feature_key}/from_timestamp", start_seconds)
            )
            view_stop = float(
                row.get(f"videos/{feature_key}/to_timestamp", stop_seconds)
            )
            views.append(
                ViewSegment(
                    canonical_name=canonical_name,
                    feature_key=feature_key,
                    relative_path=relative,
                    start_seconds=view_start,
                    stop_seconds=view_stop,
                )
            )
        namespace = "" if episode_namespace is None else f"{episode_namespace}:"
        episode_id = f"{layout.source}:{namespace}{episode_index:09d}"
        descriptors.append(
            EpisodeDescriptor(
                source=layout.source,
                episode_id=episode_id,
                episode_index=episode_index,
                embodiment=layout.embodiment,
                split=deterministic_split(
                    layout.source,
                    episode_id,
                    seed=split_seed,
                    train_fraction=train_fraction,
                ),
                task_text=_task_text(row, layout, tasks),
                raw_root=str(root),
                data_relative_path=data_relative,
                data_row_start=row_start,
                data_row_stop=row_stop,
                timestamp_column=layout.timestamp_column,
                episode_column=layout.episode_column,
                source_fps=fps,
                duration_seconds=min(
                    (row_stop - row_start) / fps,
                    min(
                        view.stop_seconds - view.start_seconds
                        for view in views
                        if view.relative_path is not None
                    ),
                ),
                views=tuple(views),
                action_columns=layout.action_columns,
                auxiliary_columns=layout.auxiliary_columns,
                provenance_dataset=layout.provenance_dataset or layout.source,
            )
        )
    if not descriptors:
        raise ContractError(f"LeRobot scan produced no episodes for {layout.source}")
    return descriptors


def scan_lerobot_collection(
    root: Path,
    layout: SourceLayout,
    *,
    split_seed: int,
    train_fraction: float,
) -> list[EpisodeDescriptor]:
    """Scan a directory containing independent LeRobot dataset roots.

    AgiBot releases one LeRobot tree per archive/task. Episode indices restart
    inside each tree, so a stable relative-root hash is included in episode IDs.
    The nested root stays in each descriptor, avoiding vendor path heuristics in
    the expensive encoder.
    """

    root = resolve_real_directory(root, f"{layout.source} collection root")
    validate_collection_receipt(root, layout)
    nested_roots: list[Path] = []
    for info_path in sorted(root.glob("**/meta/info.json")):
        if info_path.is_symlink() or not info_path.is_file():
            raise ContractError(f"unsafe LeRobot metadata path {info_path}")
        candidate = resolve_real_directory(
            info_path.parent.parent,
            f"{layout.source} nested LeRobot root",
        )
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ContractError(
                f"nested LeRobot root escapes collection: {candidate}"
            ) from exc
        nested_roots.append(candidate)
    if not nested_roots:
        raise ContractError(f"LeRobot collection has no nested meta/info.json: {root}")
    if len(nested_roots) != len(set(nested_roots)):
        raise ContractError(f"duplicate nested LeRobot roots under {root}")

    descriptors: list[EpisodeDescriptor] = []
    for nested_root in nested_roots:
        relative = nested_root.relative_to(root).as_posix()
        namespace = hashlib.sha256(relative.encode()).hexdigest()[:16]
        descriptors.extend(
            scan_lerobot(
                nested_root,
                layout,
                split_seed=split_seed,
                train_fraction=train_fraction,
                episode_namespace=namespace,
            )
        )
    episode_ids = [item.episode_id for item in descriptors]
    if len(episode_ids) != len(set(episode_ids)):
        raise ContractError(f"duplicate episode IDs in collection {root}")
    return descriptors


def validate_collection_receipt(
    root: Path,
    layout: SourceLayout,
) -> Path | None:
    if layout.collection_receipt_path is None:
        return None
    receipt = resolve_regular_file(root, layout.collection_receipt_path)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != layout.collection_receipt_schema
        or value.get("complete") is not True
    ):
        raise ContractError(
            f"{layout.source}: collection completion receipt 未完成或 schema 不匹配"
        )
    return receipt


def scan_normalized_manifest(
    path: Path,
    layout: SourceLayout,
    *,
    split_seed: int,
    train_fraction: float,
    raw_root: Path | None = None,
) -> list[EpisodeDescriptor]:
    expected_root = (
        None
        if raw_root is None
        else resolve_real_directory(raw_root, f"{layout.source} raw root")
    )
    descriptors: list[EpisodeDescriptor] = []
    for row in _read_jsonl(path):
        value = EpisodeDescriptor.from_mapping(row)
        if value.source != layout.source or value.embodiment != layout.embodiment:
            raise ContractError("normalized manifest source/embodiment mismatch")
        if value.provenance_dataset is None:
            raise ContractError(
                f"{value.episode_id}: normalized manifest 缺 provenance_dataset"
            )
        if value.provenance_dataset in layout.forbidden_provenance_datasets:
            raise ContractError(
                f"{value.episode_id}: forbidden provenance dataset "
                f"{value.provenance_dataset}"
            )
        if (
            expected_root is not None
            and Path(value.raw_root).resolve(strict=True) != expected_root
        ):
            raise ContractError(
                f"{value.episode_id}: normalized manifest raw_root differs "
                "from the dataset contract"
            )
        if value.timestamp_column != layout.timestamp_column:
            raise ContractError(
                f"{value.episode_id}: timestamp column differs from layout"
            )
        if value.episode_column != layout.episode_column:
            raise ContractError(
                f"{value.episode_id}: episode column differs from layout"
            )
        if value.action_columns != layout.action_columns:
            raise ContractError(
                f"{value.episode_id}: action mapping differs from layout"
            )
        if value.auxiliary_columns != layout.auxiliary_columns:
            raise ContractError(
                f"{value.episode_id}: auxiliary mapping differs from layout"
            )
        for view in value.views:
            expected_key = layout.view_keys[view.canonical_name]
            if view.relative_path is not None and view.feature_key != expected_key:
                raise ContractError(
                    f"{value.episode_id}: {view.canonical_name} feature key "
                    "differs from layout"
                )
            if expected_key is None and view.relative_path is not None:
                raise ContractError(
                    f"{value.episode_id}: layout forbids {view.canonical_name}"
                )
        # Split is recomputed so one global seed controls every adapter.
        mapping = value.as_dict()
        mapping["split"] = deterministic_split(
            value.source,
            value.episode_id,
            seed=split_seed,
            train_fraction=train_fraction,
        )
        descriptors.append(EpisodeDescriptor.from_mapping(mapping))
    if not descriptors:
        raise ContractError(f"normalized manifest {path} is empty")
    return descriptors


def _arrow_vector_width(value: pa.DataType) -> int | None:
    while isinstance(value, pa.ExtensionType):
        value = value.storage_type
    if pa.types.is_fixed_size_list(value):
        return int(value.list_size)
    if pa.types.is_list(value) or pa.types.is_large_list(value):
        return None
    if (
        pa.types.is_boolean(value)
        or pa.types.is_integer(value)
        or pa.types.is_floating(value)
        or pa.types.is_decimal(value)
    ):
        return 1
    raise ContractError(f"unsupported action/auxiliary Arrow type: {value}")


def _sample_cell_width(
    parquet: pq.ParquetFile,
    *,
    column: str,
    row_index: int,
) -> int:
    cursor = 0
    for row_group in range(parquet.num_row_groups):
        rows = int(parquet.metadata.row_group(row_group).num_rows)
        if row_index < cursor + rows:
            table = parquet.read_row_group(row_group, columns=[column])
            cell = table.column(0)[row_index - cursor].as_py()
            if cell is None:
                raise ContractError(f"{column} is null at parquet row {row_index}")
            if isinstance(cell, (list, tuple)):
                return len(cell)
            return 1
        cursor += rows
    raise ContractError(f"parquet row {row_index} is outside the file")


def validate_episode_inputs(
    episodes: Sequence[EpisodeDescriptor],
) -> dict[str, Any]:
    """Deeply validate the immutable plan before any expensive encoding.

    This is deliberately metadata-heavy and decode-light: all episode
    intervals, parquet schemas/vector widths and every referenced video are
    checked, while actual frame/timestamp decoding remains the encoder's
    transactional responsibility.
    """

    if not episodes:
        raise ContractError("cannot validate an empty episode collection")
    seen_ids: set[str] = set()
    seen_intervals: set[tuple[str, str, int, int]] = set()
    files: dict[Path, list[EpisodeDescriptor]] = {}
    videos: set[Path] = set()
    source_counts: dict[str, int] = {}
    source_seconds: dict[str, float] = {}
    for episode in episodes:
        episode.validate()
        if episode.episode_id in seen_ids:
            raise ContractError(f"duplicate episode ID {episode.episode_id}")
        seen_ids.add(episode.episode_id)
        root = resolve_real_directory(
            Path(episode.raw_root),
            f"{episode.episode_id} raw root",
        )
        data_path = resolve_regular_file(root, episode.data_relative_path)
        if data_path.stat().st_size <= 0:
            raise ContractError(f"empty parquet input: {data_path}")
        interval = (
            str(root),
            episode.data_relative_path,
            episode.data_row_start,
            episode.data_row_stop,
        )
        if interval in seen_intervals:
            raise ContractError(f"duplicate parquet interval for {episode.episode_id}")
        seen_intervals.add(interval)
        files.setdefault(data_path, []).append(episode)
        for view in episode.views:
            if view.relative_path is None:
                continue
            video = resolve_regular_file(root, view.relative_path)
            if video.stat().st_size <= 0:
                raise ContractError(f"empty video input: {video}")
            videos.add(video)
        source_counts[episode.source] = source_counts.get(episode.source, 0) + 1
        source_seconds[episode.source] = source_seconds.get(
            episode.source, 0.0
        ) + float(episode.duration_seconds)

    data_bytes = 0
    for path, bound_episodes in sorted(files.items(), key=lambda item: str(item[0])):
        parquet = pq.ParquetFile(path)
        total_rows = int(parquet.metadata.num_rows)
        data_bytes += path.stat().st_size
        schema = parquet.schema_arrow
        available = set(schema.names)
        required: set[str] = set()
        for episode in bound_episodes:
            required.update((episode.timestamp_column, episode.episode_column))
            required.update(item.column for item in episode.action_columns)
            required.update(item.column for item in episode.auxiliary_columns)
            if episode.data_row_stop > total_rows:
                raise ContractError(
                    f"{episode.episode_id}: row stop "
                    f"{episode.data_row_stop} exceeds {total_rows}"
                )
        missing = required.difference(available)
        if missing:
            raise ContractError(f"{path}: missing parquet columns {sorted(missing)}")
        representative = bound_episodes[0]
        for scalar_column in (
            representative.timestamp_column,
            representative.episode_column,
        ):
            width = _arrow_vector_width(schema.field(scalar_column).type)
            if width != 1:
                raise ContractError(
                    f"{path}: {scalar_column} must be scalar, width={width}"
                )
        specs = (
            *representative.action_columns,
            *representative.auxiliary_columns,
        )
        for spec in specs:
            width = _arrow_vector_width(schema.field(spec.column).type)
            if width is None:
                width = _sample_cell_width(
                    parquet,
                    column=spec.column,
                    row_index=representative.data_row_start,
                )
            if max(spec.indices) >= width:
                raise ContractError(
                    f"{path}: {spec.column} width {width} does not cover "
                    f"indices {spec.indices}"
                )

    video_bytes = sum(path.stat().st_size for path in videos)
    return {
        "episodes": len(episodes),
        "unique_data_files": len(files),
        "unique_video_files": len(videos),
        "data_bytes": data_bytes,
        "video_bytes": video_bytes,
        "sources": {
            source: {
                "episodes": source_counts[source],
                "hours": source_seconds[source] / 3600.0,
            }
            for source in sorted(source_counts)
        },
    }


def write_episode_plan(
    path: Path, episodes: Iterable[EpisodeDescriptor]
) -> dict[str, Any]:
    values = sorted(
        (episode.as_dict() for episode in episodes),
        key=lambda item: (
            item["source"],
            int(item["episode_index"]),
            item["episode_id"],
        ),
    )
    if not values:
        raise ContractError("cannot publish an empty episode plan")
    payload = b"".join(
        (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(
        path.name + f".tmp.{hashlib.sha256(payload).hexdigest()[:12]}"
    )
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return {
        "episodes": len(values),
        "sha256": sha256_file(path),
        "splits": {
            split: sum(item["split"] == split for item in values)
            for split in ("train", "val", "test")
        },
        "hours": sum(float(item["duration_seconds"]) for item in values) / 3600.0,
    }


def publish_scan_receipt(
    path: Path,
    *,
    layout_path: Path,
    plan_path: Path,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema": SCAN_RECEIPT_SCHEMA,
        "layout_sha256": sha256_file(layout_path),
        "plan_sha256": sha256_file(plan_path),
        "summary": dict(summary),
    }
    value["content_sha256"] = canonical_sha256(value)
    atomic_write_json(path, value, exclusive=True)
    return value
