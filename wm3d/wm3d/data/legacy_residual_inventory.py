"""Strict metadata-only import of audited V7 residual episode plans into WM3D.

The legacy plan is used only to locate immutable raw slices and video segments.
Robot semantics, clock evidence, split assignment, duration, and payload digests
are rebuilt under the WM3D adapter/data-profile contracts.  No V7 training or
cache behavior is imported here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, Sequence

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
)
from .source_adapters import AdapterContract, MappingTerm
from .source_inventory import (
    INVENTORY_RECEIPT_SCHEMA,
    SourceInventoryError,
    deterministic_split,
)


LEGACY_PLAN_SCHEMA = "wm3d_v7_native5b_episode_plan_v1"
LEGACY_FORMAL_SOURCE = "legacy_v7_formal"
FORBIDDEN_PROVENANCE_DATASET = "robocasa365_mg"


class LegacyResidualImportError(SourceInventoryError):
    """Raised when legacy metadata cannot satisfy the WM3D source contract."""


_PLAN_FIELDS = {
    "schema",
    "source",
    "episode_id",
    "episode_index",
    "embodiment",
    "split",
    "task_text",
    "raw_root",
    "data_relative_path",
    "data_row_start",
    "data_row_stop",
    "timestamp_column",
    "episode_column",
    "source_fps",
    "duration_seconds",
    "views",
    "action_columns",
    "auxiliary_columns",
    "provenance_dataset",
}
_VIEW_FIELDS = {
    "canonical_name",
    "feature_key",
    "relative_path",
    "start_seconds",
    "stop_seconds",
}
_ACTION_FIELDS = {"group_name", "column", "indices", "discrete"}
_AUXILIARY_FIELDS = {"modality_name", "column", "indices", "discrete"}


def _regular_file(path: Path, *, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise LegacyResidualImportError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _real_root(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise LegacyResidualImportError("raw_root must be an absolute real directory")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise LegacyResidualImportError(
            "raw_root must be canonical and cannot contain symlink components"
        )
    return resolved


def _resolve_under_root(root: Path, relative: str, *, label: str) -> tuple[str, Path]:
    try:
        relative = safe_relative_path(str(relative))
    except ManifestContractError as exc:
        raise LegacyResidualImportError(f"{label}: {exc}") from exc
    current = root
    for component in Path(relative).parts:
        current = current / component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            raise LegacyResidualImportError(f"{label}: missing path {relative!r}") from exc
        if stat.S_ISLNK(mode):
            raise LegacyResidualImportError(
                f"{label}: symlink components are forbidden: {relative!r}"
            )
    if not stat.S_ISREG(os.stat(current, follow_symlinks=False).st_mode):
        raise LegacyResidualImportError(f"{label}: not a regular file: {relative!r}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LegacyResidualImportError(
            f"{label}: path escapes raw_root: {relative!r}"
        ) from exc
    return relative, resolved


def _fingerprint(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (int(value.st_dev), int(value.st_ino), int(value.st_size), int(value.st_mtime_ns))


class _StableDigestCache:
    def __init__(self) -> None:
        self._entries: dict[Path, tuple[str, tuple[int, int, int, int]]] = {}

    def digest(self, path: Path) -> str:
        observed = self._entries.get(path)
        if observed is not None:
            digest, expected = observed
            if _fingerprint(path) != expected:
                raise LegacyResidualImportError(f"raw asset changed during import: {path}")
            return digest
        before = _fingerprint(path)
        if before[2] <= 0:
            raise LegacyResidualImportError(f"raw asset is empty: {path}")
        digest = sha256_file(path)
        after = _fingerprint(path)
        if after != before:
            raise LegacyResidualImportError(f"raw asset changed while hashing: {path}")
        self._entries[path] = (digest, after)
        return digest

    def assert_stable(self) -> None:
        for path, (_digest, expected) in self._entries.items():
            if _fingerprint(path) != expected:
                raise LegacyResidualImportError(f"raw asset changed during import: {path}")


def _read_plan(path: Path) -> tuple[list[dict[str, Any]], Path, str]:
    safe = _regular_file(path, label="legacy episode plan")
    before = _fingerprint(safe)
    rows: list[dict[str, Any]] = []
    with safe.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise LegacyResidualImportError(f"{safe}:{line_number}: blank row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LegacyResidualImportError(
                    f"{safe}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, dict) or set(value) != _PLAN_FIELDS:
                raise LegacyResidualImportError(
                    f"{safe}:{line_number}: legacy plan fields mismatch"
                )
            rows.append(value)
    if not rows:
        raise LegacyResidualImportError("legacy episode plan is empty")
    digest = sha256_file(safe)
    if _fingerprint(safe) != before:
        raise LegacyResidualImportError("legacy episode plan changed during import")
    return rows, safe, digest


def _legacy_indices(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise LegacyResidualImportError(f"{label}: indices must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise LegacyResidualImportError(f"{label}: indices must be integers")
    indices = tuple(value)
    if any(item < 0 for item in indices) or len(indices) != len(set(indices)):
        raise LegacyResidualImportError(f"{label}: indices must be unique/non-negative")
    return indices


def _validate_mapping_term(
    array: np.ndarray,
    term: MappingTerm,
    *,
    expected_rows: int,
    label: str,
) -> np.ndarray:
    try:
        numeric = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LegacyResidualImportError(f"{label}: source field is not numeric") from exc
    if numeric.ndim != 2 or numeric.shape[0] != expected_rows:
        raise LegacyResidualImportError(
            f"{label}: source field must be [{expected_rows},D], got {numeric.shape}"
        )
    if max(term.columns) >= numeric.shape[1]:
        raise LegacyResidualImportError(
            f"{label}: mapping columns {term.columns} exceed width {numeric.shape[1]}"
        )
    selected = numeric[:, term.columns]
    selected = selected * np.asarray(term.scale, dtype=np.float64)
    selected = selected + np.asarray(term.offset, dtype=np.float64)
    if not bool(np.isfinite(selected).all()):
        raise LegacyResidualImportError(f"{label}: mapped values contain NaN/Inf")
    return selected


def _clock(array: np.ndarray, *, key: str, count: int, label: str) -> dict[str, Any]:
    try:
        values = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LegacyResidualImportError(f"{label}: clock is not numeric") from exc
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape != (count,):
        raise LegacyResidualImportError(
            f"{label}: clock must be a scalar series of shape {(count,)}, got {values.shape}"
        )
    try:
        return timestamp_evidence(key=key, values=values)
    except ManifestContractError as exc:
        raise LegacyResidualImportError(f"{label}: {exc}") from exc


def _slice_column(
    parquet: pq.ParquetFile, key: str, start: int, stop: int
) -> np.ndarray:
    if key not in parquet.schema_arrow.names:
        raise LegacyResidualImportError(f"payload misses required field {key!r}")
    if start < 0 or stop <= start or stop > parquet.metadata.num_rows:
        raise LegacyResidualImportError(f"invalid payload slice [{start}, {stop})")
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
        raise LegacyResidualImportError(f"payload field {key!r} slice is incomplete")
    return result


def _validate_episode_column(
    array: np.ndarray, *, expected_index: int, expected_rows: int, label: str
) -> None:
    try:
        values = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise LegacyResidualImportError(f"{label}: episode column is not numeric") from exc
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape != (expected_rows,):
        raise LegacyResidualImportError(f"{label}: invalid episode column shape")
    if not bool(np.isfinite(values).all()) or not bool(np.equal(values, np.floor(values)).all()):
        raise LegacyResidualImportError(f"{label}: episode column is not finite/integral")
    if not bool((values == expected_index).all()):
        raise LegacyResidualImportError(f"{label}: row slice crosses episode identity")


def _validate_old_action_contract(
    value: object,
    *,
    adapter: AdapterContract,
    identity: str,
) -> None:
    if not isinstance(value, list):
        raise LegacyResidualImportError(f"{identity}: action_columns must be a list")
    legacy_names: set[str] = set()
    legacy_coordinates: set[tuple[str, int]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != _ACTION_FIELDS:
            raise LegacyResidualImportError(
                f"{identity}.action_columns[{index}]: fields mismatch"
            )
        group = str(raw["group_name"])
        column = str(raw["column"])
        indices = _legacy_indices(
            raw["indices"], label=f"{identity}.action_columns[{index}]"
        )
        if (
            not group
            or not column
            or group in legacy_names
            or not isinstance(raw["discrete"], bool)
        ):
            raise LegacyResidualImportError(f"{identity}: invalid legacy action mapping")
        legacy_names.add(group)
        for coordinate in ((column, item) for item in indices):
            if coordinate in legacy_coordinates:
                raise LegacyResidualImportError(
                    f"{identity}: overlapping legacy action coordinate {coordinate}"
                )
            legacy_coordinates.add(coordinate)
    adapter_coordinates: set[tuple[str, int]] = set()
    for mapping in adapter.groups:
        for term in mapping.action:
            for coordinate in ((term.key, item) for item in term.columns):
                if coordinate in adapter_coordinates:
                    raise LegacyResidualImportError(
                        f"{identity}: WM3D adapter reuses action coordinate {coordinate}"
                    )
                adapter_coordinates.add(coordinate)
    # Legacy V7 commonly split arm6 and gripper1 while WM3D intentionally owns
    # one audited arm7 group.  Group names are therefore not an identity
    # boundary: the exact raw column coordinates are.  The audited adapter is
    # solely responsible for grouping/semantics in WM3D.
    if legacy_coordinates != adapter_coordinates:
        missing = sorted(adapter_coordinates - legacy_coordinates)
        extra = sorted(legacy_coordinates - adapter_coordinates)
        raise LegacyResidualImportError(
            f"{identity}: legacy/WM3D action coordinate coverage differs: "
            f"missing={missing} extra={extra}"
        )


def _validate_auxiliary_columns(
    value: object, *, identity: str
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not isinstance(value, list):
        raise LegacyResidualImportError(f"{identity}: auxiliary_columns must be a list")
    seen: set[str] = set()
    result: list[tuple[str, tuple[int, ...]]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != _AUXILIARY_FIELDS:
            raise LegacyResidualImportError(
                f"{identity}.auxiliary_columns[{index}]: fields mismatch"
            )
        name = str(raw["modality_name"])
        column = str(raw["column"])
        indices = _legacy_indices(
            raw["indices"], label=f"{identity}.auxiliary_columns[{index}]"
        )
        if not name or not column or name in seen or not isinstance(raw["discrete"], bool):
            raise LegacyResidualImportError(f"{identity}: invalid auxiliary mapping")
        seen.add(name)
        result.append((column, indices))
    return tuple(result)


def _strict_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LegacyResidualImportError(f"{label} must be an integer")
    return value


def _views(
    value: object,
    *,
    root: Path,
    adapter: AdapterContract,
    view_slots: Sequence[str],
    identity: str,
    digests: _StableDigestCache,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if not isinstance(value, list):
        raise LegacyResidualImportError(f"{identity}: views must be a list")
    legacy_names: list[str] = []
    present: dict[str, tuple[str, str, Path, float, float]] = {}
    probed_duration: dict[Path, float] = {}
    adapter_by_key = {view.key: view.name for view in adapter.views}
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != _VIEW_FIELDS:
            raise LegacyResidualImportError(f"{identity}.views[{index}]: fields mismatch")
        name = str(raw["canonical_name"])
        if not name or name in legacy_names:
            raise LegacyResidualImportError(f"{identity}: invalid/duplicate view {name!r}")
        legacy_names.append(name)
        feature = raw["feature_key"]
        relative = raw["relative_path"]
        if relative is None:
            if feature is not None:
                raise LegacyResidualImportError(
                    f"{identity}/{name}: missing view cannot carry a feature key"
                )
            continue
        if not isinstance(feature, str) or not feature:
            raise LegacyResidualImportError(f"{identity}/{name}: present view lacks feature key")
        canonical_name = adapter_by_key.get(feature)
        if canonical_name is None or canonical_name not in view_slots:
            raise LegacyResidualImportError(
                f"{identity}/{name}: feature key is not mapped into the WM3D view vocabulary"
            )
        if canonical_name in present:
            raise LegacyResidualImportError(
                f"{identity}: multiple legacy views map to WM3D view {canonical_name!r}"
            )
        relative, path = _resolve_under_root(
            root, str(relative), label=f"{identity}/{name}/video"
        )
        start = float(raw["start_seconds"])
        stop = float(raw["stop_seconds"])
        if not np.isfinite([start, stop]).all() or start < 0 or stop <= start:
            raise LegacyResidualImportError(f"{identity}/{name}: invalid video PTS range")
        duration = probed_duration.get(path)
        if duration is None:
            try:
                import av

                with av.open(str(path), mode="r") as container:
                    streams = [stream for stream in container.streams if stream.type == "video"]
                    if not streams:
                        raise LegacyResidualImportError(
                            f"{identity}/{name}: video container has no video stream"
                        )
                    stream = streams[0]
                    if stream.duration is not None and stream.time_base is not None:
                        duration = float(stream.duration * stream.time_base)
                    elif container.duration is not None:
                        duration = float(container.duration) / float(av.time_base)
                    else:
                        decoded_pts: list[float] = []
                        for frame in container.decode(stream):
                            if frame.pts is not None and stream.time_base is not None:
                                decoded_pts.append(float(frame.pts * stream.time_base))
                        if len(decoded_pts) < 2:
                            raise LegacyResidualImportError(
                                f"{identity}/{name}: video duration is unavailable"
                            )
                        duration = decoded_pts[-1] + (decoded_pts[-1] - decoded_pts[-2])
            except LegacyResidualImportError:
                raise
            except Exception as exc:
                raise LegacyResidualImportError(
                    f"{identity}/{name}: video container cannot be decoded/probed"
                ) from exc
            if duration is None or not np.isfinite(duration) or duration <= 0:
                raise LegacyResidualImportError(
                    f"{identity}/{name}: invalid video duration"
                )
            probed_duration[path] = duration
        tolerance = max(1.0e-6, duration * 1.0e-6)
        if stop > duration + tolerance:
            raise LegacyResidualImportError(
                f"{identity}/{name}: video PTS stop {stop} exceeds duration {duration}"
            )
        present[canonical_name] = (feature, relative, path, start, stop)
    expected_views = {view.name: view.key for view in adapter.views}
    observed_views = {name: item[0] for name, item in present.items()}
    if observed_views != expected_views:
        raise LegacyResidualImportError(
            f"{identity}: present legacy views do not exactly match WM3D adapter views"
        )
    if not present:
        raise LegacyResidualImportError(f"{identity}: no real RGB view")
    assets: list[dict[str, str]] = []
    views: list[dict[str, Any]] = []
    for name in view_slots:
        if name not in present:
            continue
        _feature, relative, path, start, stop = present[name]
        role = f"rgb/{name}"
        assets.append({"role": role, "path": relative, "sha256": digests.digest(path)})
        views.append(
            {
                "name": name,
                "asset_role": role,
                "segment_kind": "recorded_pts_range",
                "start_s": start,
                "stop_s": stop,
            }
        )
    return assets, views


def import_legacy_residual_plan(
    *,
    plan_path: Path,
    raw_root: Path,
    source: str,
    embodiment: EmbodimentSpec,
    adapter: AdapterContract,
    view_slots: Sequence[str],
    split_seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Rebuild a standard WM3D manifest from one audited V7 residual plan."""

    if source != LEGACY_FORMAL_SOURCE:
        raise LegacyResidualImportError(
            f"legacy residual import only accepts source={LEGACY_FORMAL_SOURCE!r}"
        )
    root = _real_root(raw_root)
    if adapter.raw_format not in {"lerobot_parquet_video", "agibot_parquet_video"}:
        raise LegacyResidualImportError("legacy residual import requires parquet/video adapter")
    if not view_slots or len(view_slots) != len(set(view_slots)):
        raise LegacyResidualImportError("WM3D view_slots must be unique/non-empty")
    if any(not isinstance(item, str) or not item for item in view_slots):
        raise LegacyResidualImportError("WM3D view_slots must be unique/non-empty")
    expected_groups = {group.name for group in embodiment.groups}
    if expected_groups != {group.group for group in adapter.groups}:
        raise LegacyResidualImportError("adapter and embodiment groups differ")

    plan_rows, safe_plan, plan_sha = _read_plan(plan_path)
    digests = _StableDigestCache()
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    physical_slices: set[tuple[str, int, int]] = set()
    provenance_roots: set[str] = set()

    for line_number, legacy in enumerate(plan_rows, 1):
        prefix = f"{safe_plan}:{line_number}"
        if legacy["schema"] != LEGACY_PLAN_SCHEMA:
            raise LegacyResidualImportError(f"{prefix}: unsupported legacy schema")
        if legacy["source"] != LEGACY_FORMAL_SOURCE:
            raise LegacyResidualImportError(
                f"{prefix}: source must be {LEGACY_FORMAL_SOURCE!r}"
            )
        identity = str(legacy["episode_id"])
        if not identity or identity in identities:
            raise LegacyResidualImportError(f"{prefix}: duplicate/empty episode_id {identity!r}")
        identities.add(identity)
        if legacy["embodiment"] != embodiment.name:
            raise LegacyResidualImportError(f"{identity}: embodiment does not match WM3D template")
        provenance = legacy["provenance_dataset"]
        if not isinstance(provenance, str) or not provenance.strip():
            raise LegacyResidualImportError(f"{identity}: provenance_dataset must be non-empty")
        if provenance.strip().casefold() == FORBIDDEN_PROVENANCE_DATASET:
            raise LegacyResidualImportError(
                f"{identity}: forbidden provenance {FORBIDDEN_PROVENANCE_DATASET!r}"
            )
        legacy_root = Path(str(legacy["raw_root"]))
        if not str(legacy["raw_root"]).strip() or not legacy_root.is_absolute():
            raise LegacyResidualImportError(
                f"{identity}: legacy raw_root provenance must be non-empty/absolute"
            )
        # This string is provenance only.  It may not exist on the receiving
        # cluster and is never joined with an asset path.  Every real file is
        # resolved exclusively below the explicit current ``raw_root``.
        provenance_roots.add(str(legacy_root))
        if str(legacy["timestamp_column"]) != adapter.observation_time_key:
            raise LegacyResidualImportError(
                f"{identity}: legacy timestamp column differs from WM3D adapter"
            )
        episode_column = str(legacy["episode_column"])
        if not episode_column:
            raise LegacyResidualImportError(f"{identity}: episode_column is empty")
        episode_index = _strict_int(
            legacy["episode_index"], label=f"{identity}.episode_index"
        )
        if episode_index < 0:
            raise LegacyResidualImportError(f"{identity}: episode_index is negative")
        task_text = legacy["task_text"]
        if not isinstance(task_text, str) or not task_text.strip():
            raise LegacyResidualImportError(f"{identity}: task_text is empty")
        _validate_old_action_contract(
            legacy["action_columns"], adapter=adapter, identity=identity
        )
        auxiliary_specs = _validate_auxiliary_columns(
            legacy["auxiliary_columns"], identity=identity
        )

        payload, payload_path = _resolve_under_root(
            root, str(legacy["data_relative_path"]), label=f"{identity}/payload"
        )
        if payload_path.suffix != ".parquet":
            raise LegacyResidualImportError(f"{identity}: primary payload must be Parquet")
        row_start = _strict_int(
            legacy["data_row_start"], label=f"{identity}.data_row_start"
        )
        row_stop = _strict_int(
            legacy["data_row_stop"], label=f"{identity}.data_row_stop"
        )
        physical_identity = (payload, row_start, row_stop)
        if physical_identity in physical_slices:
            raise LegacyResidualImportError(f"{identity}: duplicate physical episode slice")
        physical_slices.add(physical_identity)
        payload_sha = digests.digest(payload_path)
        parquet = pq.ParquetFile(payload_path)
        if row_start < 0 or row_stop <= row_start or row_stop > parquet.metadata.num_rows:
            raise LegacyResidualImportError(f"{identity}: invalid Parquet row slice")
        count = row_stop - row_start
        if count < 2:
            raise LegacyResidualImportError(f"{identity}: episode must contain at least two rows")
        required_keys = (
            set(adapter.required_array_keys)
            | {episode_column}
            | {column for column, _indices in auxiliary_specs}
        )
        arrays = {
            key: _slice_column(parquet, key, row_start, row_stop)
            for key in sorted(required_keys)
        }
        _validate_episode_column(
            arrays[episode_column],
            expected_index=episode_index,
            expected_rows=count,
            label=f"{identity}/{episode_column}",
        )
        for auxiliary_index, (column, indices) in enumerate(auxiliary_specs):
            try:
                auxiliary = np.asarray(arrays[column], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise LegacyResidualImportError(
                    f"{identity}/auxiliary[{auxiliary_index}]: field is not numeric"
                ) from exc
            if (
                auxiliary.ndim != 2
                or auxiliary.shape[0] != count
                or max(indices) >= auxiliary.shape[1]
            ):
                raise LegacyResidualImportError(
                    f"{identity}/auxiliary[{auxiliary_index}]: invalid field width"
                )
            if not bool(np.isfinite(auxiliary[:, indices]).all()):
                raise LegacyResidualImportError(
                    f"{identity}/auxiliary[{auxiliary_index}]: values contain NaN/Inf"
                )
        observation_clock = _clock(
            arrays[adapter.observation_time_key],
            key=adapter.observation_time_key,
            count=count,
            label=f"{identity}/observation",
        )

        specs = {group.name: group for group in embodiment.groups}
        robot_groups: dict[str, Any] = {}
        clocks: list[dict[str, Any]] = [observation_clock]
        for mapping in adapter.groups:
            action_parts = [
                _validate_mapping_term(
                    arrays[term.key],
                    term,
                    expected_rows=count,
                    label=f"{identity}/{mapping.group}/action/{term.key}",
                )
                for term in mapping.action
            ]
            if sum(part.shape[1] for part in action_parts) != specs[mapping.group].action_dim:
                raise LegacyResidualImportError(
                    f"{identity}/{mapping.group}: action width differs from embodiment"
                )
            state_parts = [
                _validate_mapping_term(
                    arrays[term.key],
                    term,
                    expected_rows=count,
                    label=f"{identity}/{mapping.group}/state/{term.key}",
                )
                for term in mapping.state
            ]
            state_count = count if state_parts else 0
            if sum(part.shape[1] for part in state_parts) != specs[mapping.group].state_dim:
                raise LegacyResidualImportError(
                    f"{identity}/{mapping.group}: state width differs from embodiment"
                )
            action_clock = None
            action_clock_values: np.ndarray | None = None
            interval_key = mapping.world_interval_index_key
            if mapping.supervision == "fine_command":
                if mapping.action_time_key is None:
                    raise LegacyResidualImportError(
                        f"{identity}/{mapping.group}: fine action clock is absent"
                    )
                action_clock = _clock(
                    arrays[mapping.action_time_key],
                    key=mapping.action_time_key,
                    count=count,
                    label=f"{identity}/{mapping.group}/action_clock",
                )
                action_clock_values = np.asarray(
                    arrays[mapping.action_time_key], dtype=np.float64
                ).reshape(-1)
                clocks.append(action_clock)
            else:
                if interval_key is None:
                    raise LegacyResidualImportError(
                        f"{identity}/{mapping.group}: coarse interval key is absent"
                    )
                try:
                    intervals = np.asarray(arrays[interval_key], dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise LegacyResidualImportError(
                        f"{identity}/{mapping.group}: interval index is not numeric"
                    ) from exc
                if intervals.ndim == 2 and intervals.shape[1] == 1:
                    intervals = intervals[:, 0]
                if (
                    intervals.shape != (count,)
                    or not bool(np.isfinite(intervals).all())
                    or not bool(np.equal(intervals, np.floor(intervals)).all())
                    or bool((intervals < 0).any())
                    or bool((intervals >= count - 1).any())
                ):
                    raise LegacyResidualImportError(
                        f"{identity}/{mapping.group}: invalid world interval indices"
                    )
            state_clock = None
            if state_count:
                if mapping.state_time_key is None:
                    raise LegacyResidualImportError(
                        f"{identity}/{mapping.group}: measured state clock is absent"
                    )
                state_clock = _clock(
                    arrays[mapping.state_time_key],
                    key=mapping.state_time_key,
                    count=state_count,
                    label=f"{identity}/{mapping.group}/state_clock",
                )
                state_clock_values = np.asarray(
                    arrays[mapping.state_time_key], dtype=np.float64
                ).reshape(-1)
                if (
                    action_clock_values is not None
                    and not np.array_equal(action_clock_values, state_clock_values)
                ):
                    raise LegacyResidualImportError(
                        f"{identity}/{mapping.group}: state clock is not exactly "
                        "anchored to the fine action clock"
                    )
                clocks.append(state_clock)
            elif mapping.state_time_key is not None:
                raise LegacyResidualImportError(
                    f"{identity}/{mapping.group}: stateless group carries a state clock"
                )
            robot_groups[mapping.group] = {
                "supervision": mapping.supervision,
                "action_samples": count,
                "state_samples": state_count,
                "action_clock": action_clock,
                "state_clock": state_clock,
                "world_interval_index_key": interval_key,
            }

        video_assets, views = _views(
            legacy["views"],
            root=root,
            adapter=adapter,
            view_slots=view_slots,
            identity=identity,
            digests=digests,
        )
        starts = [float(clock["start_s"]) for clock in clocks]
        ends = [float(clock["end_s"]) for clock in clocks]
        largest_dt = max(float(clock["max_dt_s"]) for clock in clocks)
        duration = max(ends) - min(starts) + largest_dt
        rows.append(
            {
                "schema": SOURCE_MANIFEST_SCHEMA,
                "episode_id": identity,
                "source": source,
                "payload": payload,
                "payload_sha256": payload_sha,
                "payload_row_start": row_start,
                "payload_row_stop": row_stop,
                "assets": [
                    {
                        "role": "primary_payload",
                        "path": payload,
                        "sha256": payload_sha,
                    },
                    *video_assets,
                ],
                "views": views,
                "task_text": task_text,
                "embodiment": embodiment.name,
                "split": deterministic_split(
                    source,
                    identity,
                    seed=split_seed,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                ),
                "duration_s": duration,
                "observation_samples": count,
                "observation_clock": observation_clock,
                "robot_groups": robot_groups,
            }
        )
    digests.assert_stable()
    rows.sort(key=lambda row: str(row["episode_id"]))
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
        "canonical_rows_sha256": canonical_sha256(rows),
        "selection": {
            "mode": "legacy_v7_formal_residual_plan",
            "legacy_plan_path": str(safe_plan),
            "legacy_plan_sha256": plan_sha,
            "legacy_provenance_raw_roots": sorted(provenance_roots),
            "legacy_provenance_raw_roots_sha256": canonical_sha256(
                sorted(provenance_roots)
            ),
        },
    }
    return tuple(rows), receipt
