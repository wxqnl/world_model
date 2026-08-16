"""Config-driven raw episode adapters for the unified WM3D cache builder.

All source-specific field names live in audited YAML contracts.  Adapter code
uses declarative tensor mappings and therefore does not branch on dataset
names, robot brands, model size, or nominal frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
import yaml

from .grouped_robot import (
    EmbodimentSpec,
    RawActionSeries,
    RawStateSeries,
    RawStateSnapshot,
)
from .manifest_contract import sha256_file


ADAPTER_SCHEMA = "wm3d_v8_source_adapter_v3"
ADAPTER_COLOR_SCHEMA = "wm3d_source_adapter_v4"


class AdapterContractError(ValueError):
    pass


class EpisodeAccessor(Protocol):
    def array(self, key: str) -> np.ndarray: ...


@dataclass(frozen=True)
class MappingTerm:
    key: str
    columns: tuple[int, ...]
    scale: tuple[float, ...]
    offset: tuple[float, ...]


@dataclass(frozen=True)
class ViewMapping:
    """One source RGB stream mapped into a canonical view slot.

    View names are intentionally not restricted to head/left/right.  The data
    profile declares the ordered view vocabulary and each source may provide
    any non-empty subset of it.  Missing views are represented by a false
    mask at cache time; they are never copied from another camera.
    """

    name: str
    key: str
    color_order: str = "rgb"


@dataclass(frozen=True)
class GroupMapping:
    group: str
    supervision: str
    action: tuple[MappingTerm, ...]
    state: tuple[MappingTerm, ...]
    action_time_key: Optional[str]
    state_time_key: Optional[str]
    world_interval_index_key: Optional[str]


@dataclass(frozen=True)
class AdapterContract:
    path: Path
    sha256: str
    name: str
    raw_format: str
    observation_time_key: str
    views: tuple[ViewMapping, ...]
    groups: tuple[GroupMapping, ...]

    @property
    def required_array_keys(self) -> tuple[str, ...]:
        keys: set[str] = {self.observation_time_key}
        for group in self.groups:
            keys.update(term.key for term in group.action)
            keys.update(term.key for term in group.state)
            for key in (
                group.action_time_key,
                group.state_time_key,
                group.world_interval_index_key,
            ):
                if key is not None:
                    keys.add(key)
        return tuple(sorted(keys))


def _term(value: Mapping[str, Any]) -> MappingTerm:
    required = {"key", "columns", "scale", "offset"}
    if set(value) != required:
        raise AdapterContractError(
            f"mapping term keys mismatch: missing={sorted(required-set(value))} "
            f"unknown={sorted(set(value)-required)}"
        )
    columns = tuple(int(item) for item in value["columns"])
    if not columns or any(item < 0 for item in columns):
        raise AdapterContractError("mapping columns must be non-empty/non-negative")
    scale = tuple(float(item) for item in value["scale"])
    offset = tuple(float(item) for item in value["offset"])
    if len(scale) != len(columns) or len(offset) != len(columns):
        raise AdapterContractError("mapping scale/offset must match columns")
    if not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise AdapterContractError("mapping scale/offset contains NaN/Inf")
    return MappingTerm(str(value["key"]), columns, scale, offset)


def load_adapter_contract(path: Path, *, expected_sha256: str) -> AdapterContract:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise AdapterContractError(f"adapter contract is not a regular file: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise AdapterContractError(
            f"adapter contract SHA mismatch: {observed} != {expected_sha256}"
        )
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "name",
        "raw_format",
        "observation_time_key",
        "views",
        "groups",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise AdapterContractError("adapter contract root fields mismatch")
    schema = str(value["schema"])
    if schema not in {ADAPTER_SCHEMA, ADAPTER_COLOR_SCHEMA}:
        raise AdapterContractError(f"unsupported adapter schema {schema!r}")
    raw_format = str(value["raw_format"])
    if raw_format not in {"lerobot_parquet_video", "agibot_parquet_video", "npz"}:
        raise AdapterContractError(f"unsupported raw_format {raw_format!r}")
    observation_time_key = str(value["observation_time_key"])
    if not observation_time_key:
        raise AdapterContractError("observation_time_key cannot be empty")
    raw_views = value["views"]
    if not isinstance(raw_views, list) or not raw_views:
        raise AdapterContractError("adapter must declare at least one RGB view")
    views: list[ViewMapping] = []
    view_names: set[str] = set()
    view_keys: set[str] = set()
    for raw_view in raw_views:
        view_fields = (
            {"name", "key"}
            if schema == ADAPTER_SCHEMA
            else {"name", "key", "color_order"}
        )
        if not isinstance(raw_view, dict) or set(raw_view) != view_fields:
            raise AdapterContractError(
                f"adapter view fields must be exactly {sorted(view_fields)}"
            )
        name = str(raw_view["name"])
        key = str(raw_view["key"])
        if not name or not key:
            raise AdapterContractError("adapter view name/key cannot be empty")
        if name in view_names or key in view_keys:
            raise AdapterContractError("adapter view names/keys must be unique")
        view_names.add(name)
        view_keys.add(key)
        color_order = str(raw_view.get("color_order", "rgb"))
        if color_order not in {"rgb", "bgr"}:
            raise AdapterContractError(
                f"adapter view color_order must be rgb/bgr, got {color_order!r}"
            )
        views.append(ViewMapping(name=name, key=key, color_order=color_order))
    groups: list[GroupMapping] = []
    seen: set[str] = set()
    for raw in value["groups"]:
        fields = {
            "group",
            "supervision",
            "action",
            "state",
            "action_time_key",
            "state_time_key",
            "world_interval_index_key",
        }
        if not isinstance(raw, dict) or set(raw) != fields:
            raise AdapterContractError("adapter group fields mismatch")
        name = str(raw["group"])
        if not name or name in seen:
            raise AdapterContractError(f"duplicate/empty adapter group {name!r}")
        seen.add(name)
        supervision = str(raw["supervision"])
        if supervision not in {"fine_command", "coarse_effect"}:
            raise AdapterContractError(f"invalid supervision {supervision!r}")
        action_time = raw["action_time_key"]
        interval_key = raw["world_interval_index_key"]
        if supervision == "fine_command" and not action_time:
            raise AdapterContractError("fine command group requires action_time_key")
        if supervision == "coarse_effect" and not interval_key:
            raise AdapterContractError(
                "coarse effect group requires world_interval_index_key"
            )
        action_terms = tuple(_term(item) for item in raw["action"])
        state_terms = tuple(_term(item) for item in raw["state"])
        if not action_terms:
            raise AdapterContractError(f"group {name!r} has no action mapping")
        groups.append(
            GroupMapping(
                group=name,
                supervision=supervision,
                action=action_terms,
                state=state_terms,
                action_time_key=None if action_time is None else str(action_time),
                state_time_key=(
                    None
                    if raw["state_time_key"] is None
                    else str(raw["state_time_key"])
                ),
                world_interval_index_key=(
                    None if interval_key is None else str(interval_key)
                ),
            )
        )
    return AdapterContract(
        path,
        observed,
        str(value["name"]),
        raw_format,
        observation_time_key,
        tuple(views),
        tuple(groups),
    )


def _mapped(accessor: EpisodeAccessor, terms: Sequence[MappingTerm]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    row_count: Optional[int] = None
    for term in terms:
        source = np.asarray(accessor.array(term.key))
        if source.ndim != 2:
            raise AdapterContractError(f"raw field {term.key!r} must be [N,D]")
        if max(term.columns) >= source.shape[1]:
            raise AdapterContractError(
                f"raw field {term.key!r} has {source.shape[1]} columns, "
                f"mapping requests {max(term.columns)}"
            )
        selected = source[:, term.columns].astype(np.float32, copy=False)
        selected = selected * np.asarray(term.scale, np.float32)
        selected = selected + np.asarray(term.offset, np.float32)
        if row_count is None:
            row_count = selected.shape[0]
        elif selected.shape[0] != row_count:
            raise AdapterContractError("mapped raw fields have different row counts")
        pieces.append(selected)
    if not pieces:
        raise AdapterContractError("mapping contains no terms")
    result = np.concatenate(pieces, axis=1)
    if not np.isfinite(result).all():
        raise AdapterContractError("mapped values contain NaN/Inf")
    return result


def adapt_action_series(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
) -> list[RawActionSeries]:
    """Decode complete source-native action series without windowing them."""

    specs = {group.name: group for group in embodiment.groups}
    if set(specs) != {group.group for group in contract.groups}:
        raise AdapterContractError(
            "adapter groups do not exactly match embodiment groups"
        )
    actions: list[RawActionSeries] = []
    for mapping in contract.groups:
        spec = specs[mapping.group]
        action = _mapped(accessor, mapping.action)
        if action.shape[1] != spec.action_dim:
            raise AdapterContractError(
                f"group {mapping.group!r} action dim {action.shape[1]} != {spec.action_dim}"
            )
        timestamps: Optional[np.ndarray] = None
        interval_indices: Optional[np.ndarray] = None
        if mapping.supervision == "fine_command":
            timestamps = np.asarray(
                accessor.array(mapping.action_time_key), dtype=np.float64
            ).reshape(-1)
            if timestamps.shape != (len(action),):
                raise AdapterContractError("action timestamp cardinality mismatch")
        else:
            interval_indices = np.asarray(
                accessor.array(mapping.world_interval_index_key), dtype=np.int64
            ).reshape(-1)
            if interval_indices.shape != (len(action),):
                raise AdapterContractError("coarse interval cardinality mismatch")
        actions.append(
            RawActionSeries(
                mapping.group,
                mapping.supervision,  # type: ignore[arg-type]
                action,
                timestamps_s=timestamps,
                world_interval_indices=interval_indices,
            )
        )
    return actions


def adapt_current_state(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
    policy_chunk_start_s: float,
) -> list[RawStateSnapshot]:
    """Read measured state at the exact first policy-command timestamp."""

    specs = {group.name: group for group in embodiment.groups}
    if set(specs) != {group.group for group in contract.groups}:
        raise AdapterContractError(
            "adapter groups do not exactly match embodiment groups"
        )
    states: list[RawStateSnapshot] = []
    series_by_group = {
        item.group: item
        for item in adapt_state_series(
            accessor=accessor, contract=contract, embodiment=embodiment
        )
    }
    for mapping in contract.groups:
        spec = specs[mapping.group]

        # A mode/trigger group may genuinely have no measured state.  Keep it
        # as a real action group, but never fabricate a zero state token.
        if spec.state_dim == 0:
            if mapping.state:
                raise AdapterContractError(
                    f"group {mapping.group!r} declares no state semantics but "
                    "its adapter maps state fields"
                )
            if mapping.state_time_key is not None:
                raise AdapterContractError(
                    f"group {mapping.group!r} declares no state semantics but "
                    "its adapter maps a state time key"
                )
            continue
        raw = series_by_group[mapping.group]
        state = raw.values
        state_times = raw.timestamps_s
        exact = np.flatnonzero(state_times == np.float64(policy_chunk_start_s))
        if len(exact) != 1:
            raise AdapterContractError(
                f"group {mapping.group!r} has {len(exact)} exact current-state matches; "
                "interpolation/nearest fallback is forbidden"
            )
        row = int(exact[0])
        if state.shape[1] != spec.state_dim:
            raise AdapterContractError(
                f"group {mapping.group!r} state dim {state.shape[1]} != {spec.state_dim}"
            )
        states.append(
            RawStateSnapshot(mapping.group, float(state_times[row]), state[row])
        )
    return states


def adapt_state_series(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
) -> list[RawStateSeries]:
    """Decode all measured state rows; stateless action groups remain absent."""

    specs = {group.name: group for group in embodiment.groups}
    if set(specs) != {group.group for group in contract.groups}:
        raise AdapterContractError("adapter groups do not exactly match embodiment groups")
    output: list[RawStateSeries] = []
    for mapping in contract.groups:
        spec = specs[mapping.group]
        if spec.state_dim == 0:
            if mapping.state or mapping.state_time_key is not None:
                raise AdapterContractError(
                    f"group {mapping.group!r} declares no state semantics but maps state"
                )
            continue
        if not mapping.state or mapping.state_time_key is None:
            raise AdapterContractError(
                f"group {mapping.group!r} requires measured state and state_time_key"
            )
        values = _mapped(accessor, mapping.state)
        if values.shape[1] != spec.state_dim:
            raise AdapterContractError(
                f"group {mapping.group!r} state dim {values.shape[1]} != {spec.state_dim}"
            )
        timestamps = np.asarray(
            accessor.array(mapping.state_time_key), dtype=np.float64
        ).reshape(-1)
        if timestamps.shape != (len(values),):
            raise AdapterContractError("state timestamp cardinality mismatch")
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
            raise AdapterContractError("state timestamps must be finite/strictly increasing")
        output.append(RawStateSeries(mapping.group, values, timestamps))
    return output


def adapt_robot_signals(
    *,
    accessor: EpisodeAccessor,
    contract: AdapterContract,
    embodiment: EmbodimentSpec,
    policy_chunk_start_s: float,
) -> tuple[list[RawActionSeries], list[RawStateSnapshot]]:
    """Compatibility wrapper for callers that need both complete actions and state."""

    return (
        adapt_action_series(
            accessor=accessor,
            contract=contract,
            embodiment=embodiment,
        ),
        adapt_current_state(
            accessor=accessor,
            contract=contract,
            embodiment=embodiment,
            policy_chunk_start_s=policy_chunk_start_s,
        ),
    )
