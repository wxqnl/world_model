"""Audited raw-source inventory for the unified WM3D pipeline.

The scanner understands container structure, not robot semantics.  Robot and
camera field mappings come exclusively from a SHA-bound source adapter.  All
clock evidence is read from payload columns and hashed; no nominal frame rate
is converted into state or action timestamps.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from string import Formatter
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq

from .grouped_robot import EmbodimentSpec
from .manifest_contract import (
    SOURCE_MANIFEST_SCHEMA,
    ManifestContractError,
    canonical_sha256,
    safe_relative_path,
    sha256_file,
    timestamp_evidence,
    validate_source_manifest,
)
from .source_adapters import AdapterContract, AdapterContractError, MappingTerm


INVENTORY_RECEIPT_SCHEMA = "wm3d_v8_source_inventory_receipt_v1"


class SourceInventoryError(RuntimeError):
    pass


def _normalize_task_text(value: Any) -> str:
    """Return the real instruction text without converter debug wrappers."""

    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    text = str(value).strip()
    if not text:
        return ""
    # Some converted OXE metadata contains TensorFlow's repr rather than the
    # scalar string itself: ``tf.Tensor(b"instruction", shape=(),
    # dtype=string)``.  Parse only the literal first argument; never evaluate
    # the wrapper expression.
    try:
        expression = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        expression = None
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and isinstance(expression.func.value, ast.Name)
        and expression.func.value.id == "tf"
        and expression.func.attr == "Tensor"
        and expression.args
        and isinstance(expression.args[0], ast.Constant)
        and isinstance(expression.args[0].value, (str, bytes))
    ):
        scalar = expression.args[0].value
        return (
            scalar.decode("utf-8", errors="strict").strip()
            if isinstance(scalar, bytes)
            else scalar.strip()
        )
    return text


def deterministic_split(
    source: str,
    episode_id: str,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    if not 0.0 < validation_fraction < 1.0:
        raise SourceInventoryError("validation_fraction must lie in (0,1)")
    if not 0.0 < train_fraction < 1.0 - validation_fraction:
        raise SourceInventoryError(
            "train_fraction must leave non-empty validation and test ranges"
        )
    unit = (
        int.from_bytes(
            hashlib.sha256(f"{source}\x1f{episode_id}\x1f{seed}".encode()).digest()[:8],
            "big",
        )
        / 2**64
    )
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "val"
    return "test"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SourceInventoryError(f"{path}:{line_number}: blank row")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise SourceInventoryError(
                    f"{path}:{line_number}: row is not an object"
                )
            rows.append(value)
    return rows


def _episode_metadata(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    if path.is_file() and not path.is_symlink():
        return _read_jsonl(path)
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise SourceInventoryError(f"no LeRobot episode metadata under {root}")
    rows: list[dict[str, Any]] = []
    for candidate in paths:
        if candidate.is_symlink() or not candidate.is_file():
            raise SourceInventoryError(f"unsafe episode metadata {candidate}")
        rows.extend(pq.read_table(candidate).to_pylist())
    return rows


def _task_lookup(root: Path) -> dict[int, str]:
    rows: list[dict[str, Any]] = []
    jsonl = root / "meta" / "tasks.jsonl"
    parquet = root / "meta" / "tasks.parquet"
    if jsonl.is_file() and not jsonl.is_symlink():
        rows = _read_jsonl(jsonl)
    elif parquet.is_file() and not parquet.is_symlink():
        rows = pq.read_table(parquet).to_pylist()
    result: dict[int, str] = {}
    for row in rows:
        index = int(row.get("task_index", row.get("index", len(result))))
        text = _normalize_task_text(
            row.get("task", row.get("language_instruction", row.get("name")))
        )
        if text:
            result[index] = text
    return result


def _format(template: str, values: Mapping[str, Any]) -> str:
    fields = {
        name
        for _literal, name, _format_spec, _conversion in Formatter().parse(template)
        if name
    }
    missing = fields - set(values)
    if missing:
        raise SourceInventoryError(
            f"path template lacks values {sorted(missing)}: {template}"
        )
    return template.format(**values)


def _existing_relative(root: Path, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        try:
            candidate = safe_relative_path(candidate)
        except ManifestContractError:
            continue
        path = root / candidate
        if path.is_file() and not path.is_symlink():
            return candidate
    raise SourceInventoryError(f"none of the expected files exists under {root}")


def _path_values(row: Mapping[str, Any], episode_index: int) -> dict[str, int]:
    chunk = int(
        row.get("chunk_index", row.get("data/chunk_index", episode_index // 1000))
    )
    return {
        "episode_index": episode_index,
        "episode_chunk": chunk,
        "chunk_index": chunk,
        "file_index": int(row.get("data/file_index", episode_index)),
    }


def _task_text(
    row: Mapping[str, Any], tasks: Mapping[int, str], default_task: str
) -> str:
    direct = _normalize_task_text(row.get("task", row.get("language_instruction")))
    if direct:
        return direct
    values = row.get("tasks")
    if isinstance(values, Sequence) and not isinstance(values, str) and values:
        first = values[0]
        if isinstance(first, (str, bytes, np.str_, np.bytes_)):
            text = _normalize_task_text(first)
            if text:
                return text
        else:
            try:
                index = int(first)
            except (TypeError, ValueError):
                index = -1
            if index in tasks:
                return tasks[index]
    task_index = row.get("task_index")
    if task_index is not None and int(task_index) in tasks:
        return tasks[int(task_index)]
    default = _normalize_task_text(default_task)
    if not default:
        raise SourceInventoryError(
            "episode task text is unavailable and default is empty"
        )
    return default


def _slice_column(
    parquet: pq.ParquetFile, key: str, start: int, stop: int
) -> np.ndarray:
    """Read only row groups intersecting one episode slice.

    Shared-file LeRobot releases can contain thousands of episodes.  Reading
    the whole Parquet column once per episode turns inventory into an
    accidental quadratic I/O pass.  This implementation touches only the row
    groups that intersect ``[start, stop)`` and therefore keeps the inventory
    scan proportional to the source size.
    """

    if key not in parquet.schema_arrow.names:
        raise SourceInventoryError(f"payload misses adapter field {key!r}")
    if start < 0 or stop <= start or stop > parquet.metadata.num_rows:
        raise SourceInventoryError(f"invalid payload slice [{start}, {stop})")
    values: list[Any] = []
    row_group_start = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_count = parquet.metadata.row_group(row_group_index).num_rows
        row_group_stop = row_group_start + row_count
        overlap_start = max(start, row_group_start)
        overlap_stop = min(stop, row_group_stop)
        if overlap_start < overlap_stop:
            column = parquet.read_row_group(row_group_index, columns=[key]).column(0)
            local_start = overlap_start - row_group_start
            values.extend(
                column.slice(local_start, overlap_stop - overlap_start).to_pylist()
            )
        row_group_start = row_group_stop
        if row_group_start >= stop:
            break
    result = np.asarray(values)
    if len(result) != stop - start:
        raise SourceInventoryError(f"payload field {key!r} row slice is incomplete")
    return result


def _term_width(array: np.ndarray, term: MappingTerm, label: str) -> None:
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or max(term.columns) >= array.shape[1]:
        raise SourceInventoryError(
            f"{label}: mapping requests columns {term.columns} from shape {array.shape}"
        )


def _clock(
    arrays: Mapping[str, np.ndarray], key: str, expected_count: int, label: str
) -> dict[str, Any]:
    values = np.asarray(arrays[key], dtype=np.float64).reshape(-1)
    if values.shape != (expected_count,):
        raise SourceInventoryError(
            f"{label}: timestamp cardinality {values.shape} != {(expected_count,)}"
        )
    try:
        return timestamp_evidence(key=key, values=values)
    except ManifestContractError as exc:
        raise SourceInventoryError(f"{label}: {exc}") from exc


def _normalize_episode_ranges(
    episode_ranges: Optional[Mapping[int | str, Sequence[Sequence[int]]]],
) -> Optional[dict[int, tuple[tuple[int, int], ...]]]:
    if episode_ranges is None:
        return None
    if not isinstance(episode_ranges, Mapping) or not episode_ranges:
        raise SourceInventoryError("episode_ranges must be a non-empty mapping")
    normalized: dict[int, tuple[tuple[int, int], ...]] = {}
    for raw_index, raw_ranges in episode_ranges.items():
        try:
            episode_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise SourceInventoryError("episode_ranges keys must be integers") from exc
        if episode_index < 0 or episode_index in normalized:
            raise SourceInventoryError(
                "episode_ranges keys must be unique non-negative integers"
            )
        if (
            not isinstance(raw_ranges, Sequence)
            or isinstance(raw_ranges, (str, bytes))
            or not raw_ranges
        ):
            raise SourceInventoryError(
                f"episode {episode_index} must contain at least one keep range"
            )
        ranges: list[tuple[int, int]] = []
        previous_stop = -1
        for raw_range in raw_ranges:
            if (
                not isinstance(raw_range, Sequence)
                or isinstance(raw_range, (str, bytes))
                or len(raw_range) != 2
            ):
                raise SourceInventoryError(
                    f"episode {episode_index} keep ranges must be [start, stop] pairs"
                )
            try:
                start, stop = int(raw_range[0]), int(raw_range[1])
            except (TypeError, ValueError) as exc:
                raise SourceInventoryError(
                    f"episode {episode_index} keep range bounds must be integers"
                ) from exc
            if start < 0 or stop - start < 2 or start < previous_stop:
                raise SourceInventoryError(
                    f"episode {episode_index} keep ranges must be sorted, "
                    "non-overlapping, and contain at least two rows"
                )
            ranges.append((start, stop))
            previous_stop = stop
        normalized[episode_index] = tuple(ranges)
    return normalized


def _selected_metadata_rows(
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    adapter: AdapterContract,
    episode_indices: Optional[frozenset[int]],
    episode_ranges: Optional[Mapping[int, tuple[tuple[int, int], ...]]],
    file_origin: Mapping[tuple[int, int, str], int],
) -> list[dict[str, Any]]:
    """Select episodes and expand audited non-idle ranges into video-aligned segments."""

    available: set[int] = set()
    selected: list[dict[str, Any]] = []
    for raw_metadata in metadata_rows:
        metadata = dict(raw_metadata)
        episode_index = int(metadata["episode_index"])
        if episode_index in available:
            raise SourceInventoryError(f"duplicate episode_index {episode_index}")
        available.add(episode_index)
        if episode_indices is not None and episode_index not in episode_indices:
            continue
        if episode_ranges is None:
            selected.append(metadata)
            continue
        ranges = episode_ranges.get(episode_index)
        if ranges is None:
            continue
        length = int(metadata.get("length", metadata.get("episode_length", 0)))
        if length < 2:
            raise SourceInventoryError(
                f"episode {episode_index} has fewer than two rows"
            )
        values = _path_values(metadata, episode_index)
        explicit_start = metadata.get("data/from_index")
        explicit_stop = metadata.get("data/to_index")
        if (explicit_start is None) != (explicit_stop is None):
            raise SourceInventoryError(
                f"episode {episode_index} has partial row bounds"
            )
        if explicit_start is not None:
            base_row_start = int(explicit_start)
            base_row_stop = int(explicit_stop)
        else:
            dataset_start = int(metadata.get("dataset_from_index", 0))
            dataset_stop = int(metadata.get("dataset_to_index", dataset_start + length))
            key = (
                values["chunk_index"],
                values["file_index"],
                str(metadata.get("data_path", "")),
            )
            base_row_start = dataset_start - file_origin[key]
            base_row_stop = dataset_stop - file_origin[key]
        if base_row_start < 0 or base_row_stop - base_row_start != length:
            raise SourceInventoryError(f"episode {episode_index} has invalid row slice")
        for segment_index, (start, stop) in enumerate(ranges):
            if stop > length:
                raise SourceInventoryError(
                    f"episode {episode_index} keep range [{start}, {stop}) "
                    f"exceeds episode length {length}"
                )
            segment = dict(metadata)
            segment["length"] = stop - start
            segment["data/from_index"] = base_row_start + start
            segment["data/to_index"] = base_row_start + stop
            segment["_wm3d_segment_index"] = segment_index
            for view in adapter.views:
                start_key = f"videos/{view.key}/from_timestamp"
                stop_key = f"videos/{view.key}/to_timestamp"
                video_start = metadata.get(start_key)
                video_stop = metadata.get(stop_key)
                if video_start is None or video_stop is None:
                    raise SourceInventoryError(
                        f"episode {episode_index}/{view.name}: range filtering "
                        "requires a recorded video PTS range"
                    )
                video_start = float(video_start)
                video_stop = float(video_stop)
                if (
                    not np.isfinite([video_start, video_stop]).all()
                    or video_stop <= video_start
                ):
                    raise SourceInventoryError(
                        f"episode {episode_index}/{view.name}: invalid video PTS range"
                    )
                duration = video_stop - video_start
                segment[start_key] = video_start + duration * start / length
                segment[stop_key] = video_start + duration * stop / length
            selected.append(segment)
    requested = (
        set(episode_ranges or ()) if episode_indices is None else set(episode_indices)
    )
    missing = sorted(requested - available)
    if missing:
        raise SourceInventoryError(f"requested episode indices are absent: {missing}")
    if episode_ranges is not None and episode_indices is not None:
        missing_ranges = sorted(set(episode_indices) - set(episode_ranges))
        if missing_ranges:
            raise SourceInventoryError(
                f"requested episodes lack required keep ranges: {missing_ranges}"
            )
    return selected


def _episode_rows(
    *,
    root: Path,
    source: str,
    embodiment: EmbodimentSpec,
    adapter: AdapterContract,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
    default_task: str,
    episode_indices: Optional[frozenset[int]],
    episode_ranges: Optional[Mapping[int, tuple[tuple[int, int], ...]]],
) -> list[dict[str, Any]]:
    info_path = root / "meta" / "info.json"
    if info_path.is_symlink() or not info_path.is_file():
        raise SourceInventoryError(f"missing safe LeRobot info.json: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
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
    tasks = _task_lookup(root)
    rows: list[dict[str, Any]] = []
    metadata_rows = sorted(
        _episode_metadata(root), key=lambda value: int(value["episode_index"])
    )
    file_origin: dict[tuple[int, int, str], int] = {}
    for metadata in metadata_rows:
        episode_index = int(metadata["episode_index"])
        explicit_start = metadata.get("data/from_index")
        explicit_stop = metadata.get("data/to_index")
        if (explicit_start is None) != (explicit_stop is None):
            raise SourceInventoryError(
                f"episode {episode_index} has partial row bounds"
            )
        if explicit_start is None:
            values = _path_values(metadata, episode_index)
            key = (
                values["chunk_index"],
                values["file_index"],
                str(metadata.get("data_path", "")),
            )
            dataset_start = int(metadata.get("dataset_from_index", 0))
            previous = file_origin.get(key)
            file_origin[key] = (
                dataset_start if previous is None else min(previous, dataset_start)
            )
    digest_cache: dict[str, str] = {}

    def digest(relative: str) -> str:
        observed = digest_cache.get(relative)
        if observed is None:
            observed = sha256_file(root / relative)
            digest_cache[relative] = observed
        return observed

    payload_cache_path: str | None = None
    payload_cache_rows = 0
    payload_cache: dict[str, np.ndarray] = {}

    metadata_rows = _selected_metadata_rows(
        metadata_rows,
        adapter=adapter,
        episode_indices=episode_indices,
        episode_ranges=episode_ranges,
        file_origin=file_origin,
    )
    for metadata in metadata_rows:
        episode_index = int(metadata["episode_index"])
        length = int(metadata.get("length", metadata.get("episode_length", 0)))
        if length < 2:
            raise SourceInventoryError(
                f"episode {episode_index} has fewer than two rows"
            )
        values = _path_values(metadata, episode_index)
        candidates = []
        if metadata.get("data_path"):
            candidates.append(str(metadata["data_path"]))
        candidates.extend(
            [
                _format(data_template, values),
                f"data/chunk-{values['chunk_index']:03d}/episode_{episode_index:06d}.parquet",
                f"data/chunk-{values['chunk_index']:03d}/file-{values['file_index']:03d}.parquet",
            ]
        )
        payload = _existing_relative(root, candidates)
        explicit_start = metadata.get("data/from_index")
        explicit_stop = metadata.get("data/to_index")
        if (explicit_start is None) != (explicit_stop is None):
            raise SourceInventoryError(
                f"episode {episode_index} has partial row bounds"
            )
        if explicit_start is not None:
            row_start, row_stop = int(explicit_start), int(explicit_stop)
        else:
            dataset_start = int(metadata.get("dataset_from_index", 0))
            dataset_stop = int(metadata.get("dataset_to_index", dataset_start + length))
            key = (
                values["chunk_index"],
                values["file_index"],
                str(metadata.get("data_path", "")),
            )
            origin = file_origin[key]
            row_start, row_stop = dataset_start - origin, dataset_stop - origin
        if row_start < 0 or row_stop - row_start != length:
            raise SourceInventoryError(f"episode {episode_index} has invalid row slice")
        payload_path = root / payload
        if payload != payload_cache_path:
            required_keys = tuple(dict.fromkeys(adapter.required_array_keys))
            try:
                table = pq.read_table(payload_path, columns=list(required_keys))
            except (KeyError, OSError) as exc:
                raise SourceInventoryError(
                    f"payload misses an adapter field or cannot be read: {payload}"
                ) from exc
            payload_cache = {
                key: np.asarray(table[key].to_pylist()) for key in required_keys
            }
            payload_cache_rows = table.num_rows
            payload_cache_path = payload
        if row_stop > payload_cache_rows:
            raise SourceInventoryError(
                f"episode {episode_index} row slice exceeds payload"
            )
        arrays = {key: payload_cache[key][row_start:row_stop] for key in payload_cache}
        observation_clock = _clock(
            arrays,
            adapter.observation_time_key,
            length,
            f"episode {episode_index}/observation",
        )
        robot_groups: dict[str, Any] = {}
        for mapping in adapter.groups:
            for term in (*mapping.action, *mapping.state):
                _term_width(
                    arrays[term.key], term, f"episode {episode_index}/{mapping.group}"
                )
            action_count = len(arrays[mapping.action[0].key])
            state_count = len(arrays[mapping.state[0].key]) if mapping.state else 0
            action_clock = None
            interval_key = mapping.world_interval_index_key
            if mapping.supervision == "fine_command":
                action_clock = _clock(
                    arrays,
                    str(mapping.action_time_key),
                    action_count,
                    f"episode {episode_index}/{mapping.group}/action",
                )
            else:
                interval = np.asarray(
                    arrays[str(interval_key)], dtype=np.int64
                ).reshape(-1)
                if interval.shape != (action_count,) or bool((interval < 0).any()):
                    raise SourceInventoryError(
                        f"episode {episode_index}/{mapping.group}: invalid world intervals"
                    )
            state_clock = None
            if state_count:
                state_clock = _clock(
                    arrays,
                    str(mapping.state_time_key),
                    state_count,
                    f"episode {episode_index}/{mapping.group}/state",
                )
            robot_groups[mapping.group] = {
                "supervision": mapping.supervision,
                "action_samples": action_count,
                "state_samples": state_count,
                "action_clock": action_clock,
                "state_clock": state_clock,
                "world_interval_index_key": interval_key,
            }

        assets: list[dict[str, str]] = []
        payload_digest = digest(payload)
        assets.append(
            {"role": "primary_payload", "path": payload, "sha256": payload_digest}
        )
        views: list[dict[str, Any]] = []
        for view in adapter.views:
            direct = metadata.get(f"videos/{view.key}/path")
            video_chunk_index = int(
                metadata.get(f"videos/{view.key}/chunk_index", values["chunk_index"])
            )
            video_file_index = int(
                metadata.get(f"videos/{view.key}/file_index", values["file_index"])
            )
            video_values = {
                **values,
                "video_key": view.key,
                "chunk_index": video_chunk_index,
                "file_index": video_file_index,
            }
            video_candidates = [str(direct)] if direct else []
            video_candidates.extend(
                [
                    _format(video_template, video_values),
                    f"videos/chunk-{video_chunk_index:03d}/{view.key}/episode_{episode_index:06d}.mp4",
                    f"videos/{view.key}/chunk-{video_chunk_index:03d}/file-{video_file_index:03d}.mp4",
                ]
            )
            video = _existing_relative(root, video_candidates)
            role = f"rgb/{view.name}"
            video_digest = digest(video)
            assets.append({"role": role, "path": video, "sha256": video_digest})
            start = metadata.get(f"videos/{view.key}/from_timestamp")
            stop = metadata.get(f"videos/{view.key}/to_timestamp")
            if start is None and stop is None:
                views.append(
                    {
                        "name": view.name,
                        "asset_role": role,
                        "segment_kind": "entire_file",
                        "start_s": None,
                        "stop_s": None,
                    }
                )
            elif start is None or stop is None:
                raise SourceInventoryError(
                    f"episode {episode_index}/{view.name}: partial video PTS range"
                )
            else:
                views.append(
                    {
                        "name": view.name,
                        "asset_role": role,
                        "segment_kind": "recorded_pts_range",
                        "start_s": float(start),
                        "stop_s": float(stop),
                    }
                )
        clock_starts = [float(observation_clock["start_s"])]
        clock_ends = [float(observation_clock["end_s"])]
        clock_max_dt = [float(observation_clock["max_dt_s"])]
        for group in robot_groups.values():
            for clock_name in ("action_clock", "state_clock"):
                clock = group[clock_name]
                if clock is not None:
                    clock_starts.append(float(clock["start_s"]))
                    clock_ends.append(float(clock["end_s"]))
                    clock_max_dt.append(float(clock["max_dt_s"]))
        duration = max(clock_ends) - min(clock_starts) + max(clock_max_dt)
        base_episode_id = f"{source}:{episode_index:09d}"
        segment_index = metadata.get("_wm3d_segment_index")
        episode_id = (
            base_episode_id
            if segment_index is None
            else f"{base_episode_id}:segment{int(segment_index):03d}"
        )
        rows.append(
            {
                "schema": SOURCE_MANIFEST_SCHEMA,
                "episode_id": episode_id,
                "source": source,
                "payload": payload,
                "payload_sha256": payload_digest,
                "payload_row_start": row_start,
                "payload_row_stop": row_stop,
                "assets": assets,
                "views": views,
                "task_text": _task_text(metadata, tasks, default_task),
                "embodiment": embodiment.name,
                "split": deterministic_split(
                    source,
                    base_episode_id,
                    seed=split_seed,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                ),
                "duration_s": duration,
                "observation_samples": length,
                "observation_clock": observation_clock,
                "robot_groups": robot_groups,
            }
        )
    if not rows:
        raise SourceInventoryError(f"source {source!r} produced no episodes")
    return rows


def scan_lerobot_source(
    *,
    root: Path,
    source: str,
    embodiment: EmbodimentSpec,
    adapter: AdapterContract,
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
    default_task: str,
    episode_indices: Optional[Sequence[int]] = None,
    episode_ranges: Optional[Mapping[int | str, Sequence[Sequence[int]]]] = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Scan one file-per-episode/shared-file LeRobot source into WM3D ABI."""

    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise SourceInventoryError("raw root must be an absolute real directory")
    root = root.resolve(strict=True)
    if adapter.raw_format not in {"lerobot_parquet_video", "agibot_parquet_video"}:
        raise AdapterContractError(
            f"adapter raw format {adapter.raw_format!r} is not a LeRobot source"
        )
    expected_groups = {group.name for group in embodiment.groups}
    if expected_groups != {group.group for group in adapter.groups}:
        raise SourceInventoryError("adapter and embodiment groups differ")
    selected_indices: Optional[frozenset[int]] = None
    if episode_indices is not None:
        normalized = tuple(int(item) for item in episode_indices)
        if not normalized or any(item < 0 for item in normalized):
            raise SourceInventoryError(
                "episode_indices must contain non-negative episode indices"
            )
        if len(normalized) != len(set(normalized)):
            raise SourceInventoryError("episode_indices contains duplicates")
        selected_indices = frozenset(normalized)
    normalized_ranges = _normalize_episode_ranges(episode_ranges)
    rows = _episode_rows(
        root=root,
        source=source,
        embodiment=embodiment,
        adapter=adapter,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        default_task=default_task,
        episode_indices=selected_indices,
        episode_ranges=normalized_ranges,
    )
    manifest_digest = canonical_sha256(rows)
    receipt = {
        "schema": INVENTORY_RECEIPT_SCHEMA,
        "source": source,
        "raw_root": str(root),
        "adapter_contract_sha256": adapter.sha256,
        "episode_count": len(rows),
        "split_count": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "duration_s": sum(float(row["duration_s"]) for row in rows),
        "canonical_rows_sha256": manifest_digest,
        "selection": {
            "mode": (
                "all_episodes"
                if selected_indices is None and normalized_ranges is None
                else "explicit_episode_indices"
                if normalized_ranges is None
                else "nonidle_episode_ranges"
            ),
            "episode_indices": (
                None if selected_indices is None else sorted(selected_indices)
            ),
            "episode_indices_sha256": (
                None
                if selected_indices is None
                else canonical_sha256(sorted(selected_indices))
            ),
            "range_episode_count": (
                0 if normalized_ranges is None else len(normalized_ranges)
            ),
            "selected_segment_count": len(rows),
            "episode_ranges_sha256": (
                None
                if normalized_ranges is None
                else canonical_sha256(
                    {
                        str(index): [list(bounds) for bounds in ranges]
                        for index, ranges in sorted(normalized_ranges.items())
                    }
                )
            ),
        },
    }
    return tuple(rows), receipt


def validate_written_inventory(
    manifest_path: Path,
    *,
    source: str,
    embodiment: EmbodimentSpec,
) -> dict[str, Any]:
    return validate_source_manifest(
        manifest_path,
        expected_source=source,
        expected_embodiment=embodiment,
    )
