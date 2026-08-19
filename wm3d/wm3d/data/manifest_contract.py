"""Dataset-agnostic manifest contracts for the unified WM3D pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

import yaml

from .grouped_robot import (
    ActionGroupSpec,
    EmbodimentSpec,
)


DATA_PROFILE_SCHEMA = "wm3d_v8_data_profile_v4"
SOURCE_MANIFEST_SCHEMA = "wm3d_v8_source_manifest_v4"
CACHE_EPISODE_INDEX_SCHEMA = "wm3d_v8_unified_episode_index_v1"
CACHE_INDEX_SCHEMA = "wm3d_v8_unified_window_index_v3"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestContractError(ValueError):
    pass


def sha256_file(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_timestamp_sha256(values: Sequence[float]) -> str:
    """Hash one recorded clock as canonical little-endian float64 bytes.

    The digest is intentionally defined here rather than by each dataset
    adapter.  It binds the exact recorded timestamps while remaining
    independent of the source container (Parquet/HDF5/NPZ).
    """

    import numpy as np

    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or array.size < 2:
        raise ManifestContractError(
            "recorded timestamp evidence requires at least two samples"
        )
    if not bool(np.isfinite(array).all()) or bool((np.diff(array) <= 0).any()):
        raise ManifestContractError(
            "recorded timestamps must be finite and strictly increasing"
        )
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def timestamp_evidence(*, key: str, values: Sequence[float]) -> dict[str, Any]:
    """Create auditable evidence for an observed source-native clock."""

    import numpy as np

    array = np.asarray(values, dtype="<f8")
    digest = canonical_timestamp_sha256(array)
    delta = np.diff(array)
    if not key:
        raise ManifestContractError("timestamp evidence key must be explicit")
    return {
        "key": str(key),
        "origin": "recorded_payload_timestamps",
        "unit": "seconds",
        "sample_count": int(array.size),
        "start_s": float(array[0]),
        "end_s": float(array[-1]),
        "min_dt_s": float(delta.min()),
        "max_dt_s": float(delta.max()),
        "timestamp_sha256": digest,
    }


def resolve_regular_file(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ManifestContractError(f"expected a regular file: {path}")
    return path.resolve(strict=True)


def safe_relative_path(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ManifestContractError(f"unsafe relative path {value!r}")
    normalized = path.as_posix()
    if normalized != value:
        raise ManifestContractError(f"non-canonical relative path {value!r}")
    return normalized


@dataclass(frozen=True)
class SourceSpec:
    name: str
    adapter: str
    raw_root: Path
    adapter_config_path: Path
    adapter_contract_sha256: str
    manifest_path: Path
    manifest_sha256: str
    embodiment: str
    weight: int
    nominal_hours: Optional[float]
    license_id: str


@dataclass(frozen=True)
class DataProfile:
    path: Path
    profile_sha256: str
    name: str
    sources: tuple[SourceSpec, ...]
    embodiments: Mapping[str, EmbodimentSpec]
    cache_representation: Mapping[str, Any]
    cache: Mapping[str, Any]

    @property
    def source_order(self) -> tuple[str, ...]:
        return tuple(source.name for source in self.sources)

    @property
    def source_weights(self) -> Mapping[str, int]:
        return {source.name: source.weight for source in self.sources}


@dataclass(frozen=True)
class CacheEpisodeEntry:
    episode_id: str
    source: str
    split: str
    embodiment: str
    feature_shard: str
    feature_sha256: str
    robot_shard: str
    robot_sha256: str
    rgb_pack: str
    rgb_pack_sha256: str
    frame_count: int


@dataclass(frozen=True)
class CacheIndexEntry:
    sample_id: str
    source: str
    split: str
    embodiment: str
    feature_shard: str
    feature_sha256: str
    robot_shard: str
    robot_sha256: str
    rgb_pack: str
    rgb_pack_sha256: str
    leading_feature_row: int
    context_feature_rows: tuple[int, ...]
    future_feature_rows: tuple[int, ...]


def load_cache_episode_index(
    path: Path, *, expected_sha256: str
) -> tuple[CacheEpisodeEntry, ...]:
    """Load the model-size-independent frame/robot episode cache index."""

    path = resolve_regular_file(path)
    expected_sha256 = _require_sha(expected_sha256, field="episode index SHA")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ManifestContractError(
            f"episode index SHA mismatch: {observed} != {expected_sha256}"
        )
    entries: list[CacheEpisodeEntry] = []
    identities: set[str] = set()
    required = {
        "schema",
        "episode_id",
        "source",
        "split",
        "embodiment",
        "feature_shard",
        "feature_sha256",
        "robot_shard",
        "robot_sha256",
        "rgb_pack",
        "rgb_pack_sha256",
        "frame_count",
    }
    for line_number, row in iter_jsonl(path):
        if row.get("schema") != CACHE_EPISODE_INDEX_SCHEMA or set(row) != required:
            raise ManifestContractError(
                f"{path}:{line_number}: invalid episode cache row fields/schema"
            )
        identity = str(row["episode_id"])
        if not identity or identity in identities:
            raise ManifestContractError(
                f"{path}:{line_number}: duplicate/empty episode_id {identity!r}"
            )
        identities.add(identity)
        split = str(row["split"])
        frame_count = int(row["frame_count"])
        if split not in {"train", "val", "test"} or frame_count < 2:
            raise ManifestContractError(
                f"{path}:{line_number}: invalid split/frame_count"
            )
        entries.append(
            CacheEpisodeEntry(
                episode_id=identity,
                source=str(row["source"]),
                split=split,
                embodiment=str(row["embodiment"]),
                feature_shard=safe_relative_path(str(row["feature_shard"])),
                feature_sha256=_require_sha(row["feature_sha256"], field="feature SHA"),
                robot_shard=safe_relative_path(str(row["robot_shard"])),
                robot_sha256=_require_sha(row["robot_sha256"], field="robot SHA"),
                rgb_pack=safe_relative_path(str(row["rgb_pack"])),
                rgb_pack_sha256=_require_sha(row["rgb_pack_sha256"], field="RGB pack SHA"),
                frame_count=frame_count,
            )
        )
    if not entries:
        raise ManifestContractError("episode cache index contains no entries")
    return tuple(entries)


def _load_structured(path: Path) -> Any:
    safe = resolve_regular_file(path)
    text = safe.read_text(encoding="utf-8")
    return json.loads(text) if safe.suffix == ".json" else yaml.safe_load(text)


def _require_sha(value: object, *, field: str) -> str:
    value = str(value)
    if SHA256_RE.fullmatch(value) is None:
        raise ManifestContractError(f"{field} must be a 64-character lowercase SHA256")
    return value


def _action_group_from_mapping(value: Mapping[str, Any]) -> ActionGroupSpec:
    required = {
        "name",
        "group_id",
        "action_semantics",
        "state_semantics",
        "action_frame",
        "state_frame",
        "composition_operators",
    }
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing or unknown:
        raise ManifestContractError(
            f"action group keys invalid: missing={missing} unknown={unknown}"
        )
    return ActionGroupSpec(
        name=str(value["name"]),
        group_id=int(value["group_id"]),
        action_semantics=tuple(str(item) for item in value["action_semantics"]),
        state_semantics=tuple(str(item) for item in value["state_semantics"]),
        action_frame=str(value["action_frame"]),
        state_frame=str(value["state_frame"]),
        composition_operators=tuple(
            str(item) for item in value["composition_operators"]
        ),
    )


def _embodiments(values: object) -> Mapping[str, EmbodimentSpec]:
    if not isinstance(values, list) or not values:
        raise ManifestContractError("data profile must declare embodiments")
    result: dict[str, EmbodimentSpec] = {}
    for raw in values:
        if not isinstance(raw, dict):
            raise ManifestContractError("embodiment entry must be a mapping")
        required = {"name", "embodiment_id", "groups"}
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required)
        if missing or unknown:
            raise ManifestContractError(
                f"embodiment keys invalid: missing={missing} unknown={unknown}"
            )
        groups = tuple(_action_group_from_mapping(item) for item in raw["groups"])
        spec = EmbodimentSpec(
            name=str(raw["name"]),
            embodiment_id=int(raw["embodiment_id"]),
            groups=groups,
        )
        if spec.name in result:
            raise ManifestContractError(f"duplicate embodiment {spec.name!r}")
        result[spec.name] = spec
    if len({item.embodiment_id for item in result.values()}) != len(result):
        raise ManifestContractError("embodiment ids must be unique")
    return result


def load_data_profile(path: Path, *, verify_source_manifests: bool = True) -> DataProfile:
    path = resolve_regular_file(path)
    value = _load_structured(path)
    if not isinstance(value, dict) or value.get("schema") != DATA_PROFILE_SCHEMA:
        raise ManifestContractError(f"data profile schema must be {DATA_PROFILE_SCHEMA}")
    allowed = {
        "schema",
        "name",
        "cache_representation",
        "cache",
        "sources",
        "embodiments",
        "notes",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManifestContractError(f"unknown data profile fields: {unknown}")
    embodiments = _embodiments(value.get("embodiments"))
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ManifestContractError("data profile must contain sources")
    sources: list[SourceSpec] = []
    seen: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ManifestContractError("source entry must be a mapping")
        required = {
            "name",
            "adapter",
            "raw_root",
            "adapter_config",
            "adapter_contract_sha256",
            "manifest",
            "manifest_sha256",
            "embodiment",
            "weight",
            "license_id",
        }
        allowed_source = required | {"nominal_hours"}
        missing = sorted(required - set(raw))
        unknown_source = sorted(set(raw) - allowed_source)
        if missing or unknown_source:
            raise ManifestContractError(
                f"source keys invalid: missing={missing} unknown={unknown_source}"
            )
        name = str(raw["name"])
        if name in seen:
            raise ManifestContractError(f"duplicate source {name!r}")
        seen.add(name)
        embodiment = str(raw["embodiment"])
        if embodiment not in embodiments:
            raise ManifestContractError(
                f"source {name!r} references unknown embodiment {embodiment!r}"
            )
        adapter_config_path = Path(str(raw["adapter_config"]))
        if not adapter_config_path.is_absolute():
            adapter_config_path = path.parent / adapter_config_path
        adapter_digest = _require_sha(
            raw["adapter_contract_sha256"],
            field=f"{name}.adapter_contract_sha256",
        )
        manifest_path = Path(str(raw["manifest"]))
        if not manifest_path.is_absolute():
            manifest_path = path.parent / manifest_path
        digest = _require_sha(raw["manifest_sha256"], field=f"{name}.manifest_sha256")
        if verify_source_manifests:
            raw_root = Path(str(raw["raw_root"]))
            if not raw_root.is_absolute() or raw_root.is_symlink() or not raw_root.is_dir():
                raise ManifestContractError(
                    f"source {name!r} raw_root must be an absolute real directory"
                )
            raw_root = raw_root.resolve(strict=True)
            adapter_config_path = resolve_regular_file(adapter_config_path)
            observed_adapter = sha256_file(adapter_config_path)
            if observed_adapter != adapter_digest:
                raise ManifestContractError(
                    f"source {name!r} adapter SHA mismatch: "
                    f"{observed_adapter} != {adapter_digest}"
                )
            manifest_path = resolve_regular_file(manifest_path)
            observed = sha256_file(manifest_path)
            if observed != digest:
                raise ManifestContractError(
                    f"source {name!r} manifest SHA mismatch: {observed} != {digest}"
                )
            validate_source_manifest(
                manifest_path,
                expected_source=name,
                expected_embodiment=embodiments[embodiment],
            )
        weight = int(raw["weight"])
        if weight <= 0:
            raise ManifestContractError(f"source {name!r} weight must be positive")
        nominal_hours = raw.get("nominal_hours")
        if nominal_hours is not None and float(nominal_hours) <= 0:
            raise ManifestContractError(f"source {name!r} nominal_hours must be positive")
        sources.append(
            SourceSpec(
                name=name,
                adapter=str(raw["adapter"]),
                raw_root=(
                    raw_root
                    if verify_source_manifests
                    else Path(str(raw["raw_root"]))
                ),
                adapter_config_path=adapter_config_path,
                adapter_contract_sha256=adapter_digest,
                manifest_path=manifest_path,
                manifest_sha256=digest,
                embodiment=embodiment,
                weight=weight,
                nominal_hours=None if nominal_hours is None else float(nominal_hours),
                license_id=str(raw["license_id"]),
            )
        )
    representation = value.get("cache_representation")
    cache = value.get("cache")
    if not isinstance(representation, dict) or not isinstance(cache, dict):
        raise ManifestContractError("cache_representation and cache must be mappings")
    required_representation = {
        "schema",
        "token_grid",
        "spatial_tokens",
        "token_dim",
        "num_views",
        "view_slots",
        "rgb_size",
        "time_binding",
        "missing_view_policy",
        "state_frame_selection",
    }
    optional_representation = {"appearance_token_grid"}
    missing_representation = sorted(required_representation - set(representation))
    unknown_representation = sorted(set(representation) - required_representation - optional_representation)
    if missing_representation or unknown_representation:
        raise ManifestContractError(
            "cache_representation fields invalid: "
            f"missing={missing_representation} unknown={unknown_representation}"
        )
    if representation["schema"] != "wm3d_v8_episode_representation_v1":
        raise ManifestContractError(
            "cache_representation.schema must be wm3d_v8_episode_representation_v1"
        )
    for field in (
        "token_grid",
        "spatial_tokens",
        "token_dim",
        "num_views",
        "rgb_size",
    ):
        if int(representation[field]) <= 0:
            raise ManifestContractError(
                f"cache_representation.{field} must be positive"
            )
    if "appearance_token_grid" in representation:
        appearance_grid = representation["appearance_token_grid"]
        if (
            isinstance(appearance_grid, bool)
            or int(appearance_grid) < int(representation["token_grid"])
        ):
            raise ManifestContractError(
                "cache_representation.appearance_token_grid must be at least token_grid"
            )
    if int(representation["token_grid"]) ** 2 != int(
        representation["spatial_tokens"]
    ):
        raise ManifestContractError(
            "cache representation spatial_tokens must equal token_grid squared"
        )
    view_slots = representation["view_slots"]
    if (
        not isinstance(view_slots, list)
        or len(view_slots) != int(representation["num_views"])
        or any(not isinstance(item, str) or not item for item in view_slots)
        or len(set(view_slots)) != len(view_slots)
    ):
        raise ManifestContractError(
            "cache_representation.view_slots must contain num_views unique "
            "non-empty canonical names"
        )
    if representation["time_binding"] != "episode_row_ordinal_with_pts_audit":
        raise ManifestContractError(
            "video frames must bind by episode row ordinal with recorded PTS audit"
        )
    if representation["missing_view_policy"] != "mask_without_duplication":
        raise ManifestContractError(
            "missing views must be masked without copying another camera"
        )
    state_selection = representation["state_frame_selection"]
    required_state_selection = {
        "mode",
        "minimum_separation_seconds",
        "preserve_observed_timestamps",
        "interpolation",
    }
    if not isinstance(state_selection, dict) or set(state_selection) != required_state_selection:
        raise ManifestContractError(
            "cache state_frame_selection fields must be exactly "
            f"{sorted(required_state_selection)}"
        )
    if state_selection["mode"] != "observed_greedy_minimum_separation":
        raise ManifestContractError("cache state frames must be selected from observations")
    if float(state_selection["minimum_separation_seconds"]) < 0:
        raise ManifestContractError(
            "cache state minimum separation must be non-negative; zero preserves all rows"
        )
    if state_selection["preserve_observed_timestamps"] is not True:
        raise ManifestContractError("cache state timestamps must be preserved")
    if state_selection["interpolation"] != "forbidden":
        raise ManifestContractError("cache state interpolation is forbidden")
    if cache.get("schema") != CACHE_INDEX_SCHEMA:
        raise ManifestContractError(f"cache.schema must be {CACHE_INDEX_SCHEMA}")
    if cache.get("task_partition") != "episode":
        raise ManifestContractError("cache.task_partition must be episode")
    if cache.get("task_claim") != "atomic_no_clobber":
        raise ManifestContractError("cache.task_claim must be atomic_no_clobber")
    if cache.get("resume") != "receipt_and_sha":
        raise ManifestContractError("cache.resume must be receipt_and_sha")
    return DataProfile(
        path=path,
        profile_sha256=sha256_file(path),
        name=str(value.get("name", "")),
        sources=tuple(sources),
        embodiments=embodiments,
        cache_representation=representation,
        cache=cache,
    )


def iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    safe = resolve_regular_file(path)
    with safe.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ManifestContractError(f"{safe}:{line_number}: blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ManifestContractError(f"{safe}:{line_number}: row must be object")
            yield line_number, value


_CLOCK_EVIDENCE_FIELDS = {
    "key",
    "origin",
    "unit",
    "sample_count",
    "start_s",
    "end_s",
    "min_dt_s",
    "max_dt_s",
    "timestamp_sha256",
}


def _validate_clock_evidence(
    value: object,
    *,
    identity: str,
    label: str,
    expected_samples: int,
) -> tuple[float, float, float]:
    if not isinstance(value, dict) or set(value) != _CLOCK_EVIDENCE_FIELDS:
        raise ManifestContractError(
            f"{identity}.{label}: clock evidence fields must be exactly "
            f"{sorted(_CLOCK_EVIDENCE_FIELDS)}"
        )
    if not str(value["key"]):
        raise ManifestContractError(f"{identity}.{label}: clock key is empty")
    if value["origin"] != "recorded_payload_timestamps":
        raise ManifestContractError(
            f"{identity}.{label}: only recorded payload timestamps are formal evidence"
        )
    if value["unit"] != "seconds":
        raise ManifestContractError(f"{identity}.{label}: clock unit must be seconds")
    sample_count = int(value["sample_count"])
    if sample_count != expected_samples or sample_count < 2:
        raise ManifestContractError(
            f"{identity}.{label}: clock sample_count={sample_count} "
            f"!= declared {expected_samples}"
        )
    start_s = float(value["start_s"])
    end_s = float(value["end_s"])
    min_dt_s = float(value["min_dt_s"])
    max_dt_s = float(value["max_dt_s"])
    if not all(
        item == item and abs(item) != float("inf")
        for item in (start_s, end_s, min_dt_s, max_dt_s)
    ):
        raise ManifestContractError(f"{identity}.{label}: clock contains NaN/Inf")
    if not start_s < end_s or not 0 < min_dt_s <= max_dt_s:
        raise ManifestContractError(
            f"{identity}.{label}: clock extent/delta is invalid"
        )
    _require_sha(
        value["timestamp_sha256"], field=f"{identity}.{label}.timestamp_sha256"
    )
    return start_s, end_s, max_dt_s


def validate_source_manifest(
    path: Path,
    *,
    expected_source: str,
    expected_embodiment: Optional[EmbodimentSpec] = None,
) -> dict[str, Any]:
    """Validate per-group source-native clock and supervision evidence.

    There is deliberately no ``hz`` field.  Uniform cadence, when present,
    is derived from the recorded clock evidence; irregular clocks remain
    valid and are passed to continuous-time model inputs unchanged.
    """

    identities: set[str] = set()
    episodes = 0
    duration_s = 0.0
    state_samples = 0
    action_samples = 0
    for line_number, row in iter_jsonl(path):
        if row.get("schema") != SOURCE_MANIFEST_SCHEMA:
            raise ManifestContractError(
                f"{path}:{line_number}: schema must be {SOURCE_MANIFEST_SCHEMA}"
            )
        required = {
            "schema",
            "episode_id",
            "source",
            "payload",
            "payload_sha256",
            "payload_row_start",
            "payload_row_stop",
            "assets",
            "views",
            "task_text",
            "embodiment",
            "split",
            "duration_s",
            "observation_samples",
            "observation_clock",
            "robot_groups",
        }
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - required)
        if missing or unknown:
            raise ManifestContractError(
                f"{path}:{line_number}: source row fields invalid: "
                f"missing={missing} unknown={unknown}"
            )
        identity = str(row["episode_id"])
        if not identity or identity in identities:
            raise ManifestContractError(
                f"{path}:{line_number}: duplicate/empty episode_id {identity!r}"
            )
        identities.add(identity)
        if row["source"] != expected_source:
            raise ManifestContractError(
                f"{path}:{line_number}: source {row['source']!r} != {expected_source!r}"
            )
        payload = safe_relative_path(str(row["payload"]))
        payload_sha = _require_sha(
            row["payload_sha256"], field=f"{identity}.payload_sha256"
        )
        row_start = int(row["payload_row_start"])
        row_stop = int(row["payload_row_stop"])
        if row_start < 0 or row_stop <= row_start:
            raise ManifestContractError(
                f"{identity}: payload row slice must satisfy 0 <= start < stop"
            )
        assets = row["assets"]
        if not isinstance(assets, list) or not assets:
            raise ManifestContractError(f"{identity}: assets must be a non-empty list")
        asset_roles: set[str] = set()
        asset_paths: set[str] = set()
        primary_matches = 0
        for asset_index, asset in enumerate(assets):
            fields = {"role", "path", "sha256"}
            if not isinstance(asset, dict) or set(asset) != fields:
                raise ManifestContractError(
                    f"{identity}.assets[{asset_index}]: fields must be exactly "
                    f"{sorted(fields)}"
                )
            role = str(asset["role"])
            asset_path = safe_relative_path(str(asset["path"]))
            asset_sha = _require_sha(
                asset["sha256"], field=f"{identity}.assets[{asset_index}].sha256"
            )
            if not role or role in asset_roles:
                raise ManifestContractError(
                    f"{identity}: duplicate/empty asset role {role!r}"
                )
            if asset_path in asset_paths:
                raise ManifestContractError(
                    f"{identity}: duplicate asset path {asset_path!r}"
                )
            asset_roles.add(role)
            asset_paths.add(asset_path)
            if role == "primary_payload":
                primary_matches += int(asset_path == payload and asset_sha == payload_sha)
        if primary_matches != 1:
            raise ManifestContractError(
                f"{identity}: assets must contain exactly one primary_payload "
                "matching payload/path SHA"
            )
        views = row["views"]
        if not isinstance(views, list) or not views:
            raise ManifestContractError(f"{identity}: views must be a non-empty list")
        view_names: set[str] = set()
        view_roles: set[str] = set()
        for view_index, view in enumerate(views):
            fields = {"name", "asset_role", "segment_kind", "start_s", "stop_s"}
            if not isinstance(view, dict) or set(view) != fields:
                raise ManifestContractError(
                    f"{identity}.views[{view_index}]: fields must be exactly "
                    f"{sorted(fields)}"
                )
            name = str(view["name"])
            role = str(view["asset_role"])
            segment_kind = str(view["segment_kind"])
            if not name or name in view_names or role in view_roles:
                raise ManifestContractError(
                    f"{identity}: view names and asset roles must be non-empty/unique"
                )
            if role not in asset_roles or role == "primary_payload":
                raise ManifestContractError(
                    f"{identity}: view {name!r} references unknown/non-video role {role!r}"
                )
            if segment_kind == "entire_file":
                if view["start_s"] is not None or view["stop_s"] is not None:
                    raise ManifestContractError(
                        f"{identity}: entire-file view {name!r} cannot carry a time range"
                    )
            elif segment_kind == "recorded_pts_range":
                start_s = float(view["start_s"])
                stop_s = float(view["stop_s"])
                if start_s < 0 or stop_s <= start_s:
                    raise ManifestContractError(
                        f"{identity}: view {name!r} has an invalid recorded PTS range"
                    )
            else:
                raise ManifestContractError(
                    f"{identity}: view {name!r} has invalid segment_kind {segment_kind!r}"
                )
            view_names.add(name)
            view_roles.add(role)
        task_text = row["task_text"]
        if not isinstance(task_text, str) or not task_text.strip():
            raise ManifestContractError(f"{identity}: task_text cannot be empty")
        if row["split"] not in {"train", "val", "test"}:
            raise ManifestContractError(f"{identity}: invalid split {row['split']!r}")
        episode_duration = float(row["duration_s"])
        if not episode_duration > 0:
            raise ManifestContractError(f"{identity}: duration_s must be positive")
        observation_samples = int(row["observation_samples"])
        observation_extent = _validate_clock_evidence(
            row["observation_clock"],
            identity=identity,
            label="observation_clock",
            expected_samples=observation_samples,
        )
        if observation_samples != row_stop - row_start:
            raise ManifestContractError(
                f"{identity}: observation_samples={observation_samples} must equal "
                f"the primary payload row span {row_stop-row_start}"
            )
        groups = row["robot_groups"]
        if not isinstance(groups, dict) or not groups:
            raise ManifestContractError(f"{identity}: robot_groups must be non-empty")
        if expected_embodiment is not None:
            expected_groups = {group.name for group in expected_embodiment.groups}
            if set(groups) != expected_groups:
                raise ManifestContractError(
                    f"{identity}: robot_groups {sorted(groups)} != embodiment "
                    f"groups {sorted(expected_groups)}"
                )
            if row["embodiment"] != expected_embodiment.name:
                raise ManifestContractError(
                    f"{identity}: embodiment {row['embodiment']!r} != "
                    f"{expected_embodiment.name!r}"
                )
        episode_state_count = 0
        episode_action_count = 0
        clock_extents: list[tuple[float, float, float]] = [observation_extent]
        for group_name, group in groups.items():
            group_fields = {
                "supervision",
                "action_samples",
                "state_samples",
                "action_clock",
                "state_clock",
                "world_interval_index_key",
            }
            if not isinstance(group, dict) or set(group) != group_fields:
                raise ManifestContractError(
                    f"{identity}.{group_name}: group fields must be exactly "
                    f"{sorted(group_fields)}"
                )
            supervision = group["supervision"]
            if supervision not in {"fine_command", "coarse_effect"}:
                raise ManifestContractError(
                    f"{identity}.{group_name}: invalid supervision {supervision!r}"
                )
            action_count = int(group["action_samples"])
            state_count = int(group["state_samples"])
            if action_count < 1 or state_count < 0:
                raise ManifestContractError(
                    f"{identity}.{group_name}: invalid action/state sample count"
                )
            if state_count == 1:
                raise ManifestContractError(
                    f"{identity}.{group_name}: one state sample cannot define a clock"
                )
            action_clock = group["action_clock"]
            interval_key = group["world_interval_index_key"]
            if supervision == "fine_command":
                if interval_key is not None:
                    raise ManifestContractError(
                        f"{identity}.{group_name}: fine commands cannot carry "
                        "world_interval_index_key"
                    )
                clock_extents.append(
                    _validate_clock_evidence(
                        action_clock,
                        identity=identity,
                        label=f"{group_name}.action_clock",
                        expected_samples=action_count,
                    )
                )
            else:
                if action_clock is not None or not str(interval_key or ""):
                    raise ManifestContractError(
                        f"{identity}.{group_name}: coarse effects require an explicit "
                        "world_interval_index_key and no fabricated action clock"
                    )
            state_clock = group["state_clock"]
            if state_count == 0:
                if state_clock is not None:
                    raise ManifestContractError(
                        f"{identity}.{group_name}: stateless group has a state clock"
                    )
            else:
                clock_extents.append(
                    _validate_clock_evidence(
                        state_clock,
                        identity=identity,
                        label=f"{group_name}.state_clock",
                        expected_samples=state_count,
                    )
                )
            episode_action_count += action_count
            episode_state_count += state_count
        if not clock_extents:
            raise ManifestContractError(f"{identity}: no recorded clock evidence")
        observed_span = max(end for _start, end, _dt in clock_extents) - min(
            start for start, _end, _dt in clock_extents
        )
        largest_dt = max(dt for _start, _end, dt in clock_extents)
        tolerance = max(1.0e-9, largest_dt * 1.0e-6)
        if episode_duration + tolerance < observed_span or episode_duration > (
            observed_span + largest_dt + tolerance
        ):
            raise ManifestContractError(
                f"{identity}: duration_s={episode_duration} is inconsistent with "
                f"recorded clocks span={observed_span}, max_dt={largest_dt}"
            )
        episodes += 1
        duration_s += episode_duration
        state_samples += episode_state_count
        action_samples += episode_action_count
    if episodes == 0:
        raise ManifestContractError(f"source manifest is empty: {path}")
    return {
        "episodes": episodes,
        "duration_s": duration_s,
        "state_samples": state_samples,
        "action_samples": action_samples,
    }


def load_cache_index(path: Path, *, expected_sha256: str) -> tuple[CacheIndexEntry, ...]:
    path = resolve_regular_file(path)
    expected_sha256 = _require_sha(expected_sha256, field="cache index SHA")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ManifestContractError(f"cache index SHA mismatch: {observed} != {expected_sha256}")
    entries: list[CacheIndexEntry] = []
    identities: set[str] = set()
    for line_number, row in iter_jsonl(path):
        if row.get("schema") != CACHE_INDEX_SCHEMA:
            raise ManifestContractError(
                f"{path}:{line_number}: cache schema must be {CACHE_INDEX_SCHEMA}"
            )
        required = {
            "schema",
            "sample_id",
            "source",
            "split",
            "embodiment",
            "feature_shard",
            "feature_sha256",
            "robot_shard",
            "robot_sha256",
            "rgb_pack",
            "rgb_pack_sha256",
            "leading_feature_row",
            "context_feature_rows",
            "future_feature_rows",
        }
        missing = sorted(required - set(row))
        unknown = sorted(set(row) - required)
        if missing or unknown:
            raise ManifestContractError(
                f"{path}:{line_number}: cache row fields invalid: "
                f"missing={missing} unknown={unknown}"
            )
        identity = str(row["sample_id"])
        if not identity or identity in identities:
            raise ManifestContractError(
                f"{path}:{line_number}: duplicate/empty sample_id {identity!r}"
            )
        identities.add(identity)
        context_rows = tuple(int(item) for item in row["context_feature_rows"])
        future_rows = tuple(int(item) for item in row["future_feature_rows"])
        leading_row = int(row["leading_feature_row"])
        if (
            leading_row < 0
            or not context_rows
            or not future_rows
            or any(item < 0 for item in context_rows + future_rows)
            or len(set((leading_row,) + context_rows + future_rows))
            != len((leading_row,) + context_rows + future_rows)
            or any(
                right <= left
                for left, right in zip(
                    (leading_row,) + context_rows + future_rows,
                    ((leading_row,) + context_rows + future_rows)[1:],
                )
            )
        ):
            raise ManifestContractError(
                f"{path}:{line_number}: leading/context/future feature rows must "
                "be non-empty, unique, non-negative and strictly increasing"
            )
        entries.append(
            CacheIndexEntry(
                sample_id=identity,
                source=str(row["source"]),
                split=str(row["split"]),
                embodiment=str(row["embodiment"]),
                feature_shard=safe_relative_path(str(row["feature_shard"])),
                feature_sha256=_require_sha(row["feature_sha256"], field="feature SHA"),
                robot_shard=safe_relative_path(str(row["robot_shard"])),
                robot_sha256=_require_sha(row["robot_sha256"], field="robot SHA"),
                rgb_pack=safe_relative_path(str(row["rgb_pack"])),
                rgb_pack_sha256=_require_sha(row["rgb_pack_sha256"], field="RGB pack SHA"),
                leading_feature_row=leading_row,
                context_feature_rows=context_rows,
                future_feature_rows=future_rows,
            )
        )
    if not entries:
        raise ManifestContractError("cache index contains no entries")
    return tuple(entries)
