from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from wm3d_v3.stage1.action_evidence_sources import (
    EVIDENCE_CANDIDATE_OFFSETS,
    EVIDENCE_CONTEXT_LENGTH,
    EvidenceExtractionError,
    TEMPORAL_WINDOW_BINDING_SCHEMA,
    TEMPORAL_WINDOW_DERIVATION,
    bind_temporal_window as _bind_temporal_window,
    load_alignment_robot_state as _load_alignment_robot_state,
    temporal_window_binding_payload as _temporal_window_binding_payload,
)
from wm3d_v3.data.manifest import OXEClipRecord, read_manifest
from wm3d_v3.stage1.action_alignment import (
    AlignmentSignals,
    FrozenProjectionSet,
    ProjectionCalibrationClip,
    build_clip_offset_evidence,
    fit_frozen_offset_projections,
)
from wm3d_v3.stage1.action_contract import (
    action_contract_key,
    canonical_dataset_name,
)
from wm3d_v3.stage1.action_contract_split import (
    ActionContractSplitError,
    frozen_contract_split_from_mapping,
)
from wm3d_v3.stage1.action_signal_extractor import (
    ActionSignalConfig,
    FORMAL_ACTION_SIGNAL_CONFIG,
    extract_cache_alignment_signals,
)
from wm3d_v3.stage1.action_evidence_window_sources import (
    ValidatedActionEvidenceWindows,
    load_bound_npz,
    load_validated_action_evidence_windows,
)
from wm3d_v3.stage1.droid_interval_action import (
    DROID_INTERVAL_ACTION_KIND,
    DROID_INTERVAL_ACTION_VALID_COUNT,
    DROID_INTERVAL_STATE_COUNT,
    DROID_INTERVAL_TERMINAL_POLICY,
)
from wm3d_v3.stage1.immutable_artifact import (
    ImmutableArtifactConflict,
    PublishResult,
    publish_immutable_bytes,
    require_distinct_output_paths,
)



FORMULA_REGISTRY_SCHEMA = "wm3d_v6_action_formula_registry_v2"
SPLIT_ARTIFACT_SCHEMA = "wm3d_v6_action_contract_split_v1"
PROJECTION_SCHEMA = "wm3d_v6_action_frozen_projection_v2"
EVIDENCE_METADATA_SCHEMA = "wm3d_v6_action_evidence_metadata_v3"
DROID_INDEX_SCHEMA = "wm3d_v6_stage1_droid_interval_cache_v2"
EVIDENCE_PROVENANCE_SCHEMA = "wm3d_v6_action_evidence_source_provenance_v2"
SPLIT_TEMPORAL_BINDING_SCHEMA = "wm3d_v6_action_temporal_window_binding_spec_v1"
CANDIDATE_OFFSETS = EVIDENCE_CANDIDATE_OFFSETS
FORMAL_TARGET_LENGTH = 16
FORMAL_NULL_REPEATS = 256
NULL_BLOCK_SIZE = 2
VISUAL_MODALITY_FAMILIES = frozenset({"flow", "geometry"})
VISUAL_MODALITY_AUDIT_FIELDS = (
    "robot_mask_motion_coverage",
    "valid_depth_coverage",
    "support_count",
    "support_fraction",
    "minimum_support_count",
    "fallback_used",
)
PROVENANCE_SOURCE_FILES = (
    "wm3d_v3/stage1/action_evidence_pipeline.py",
    "wm3d_v3/stage1/action_evidence_sources.py",
    "wm3d_v3/stage1/action_evidence_window_sources.py",
    "wm3d_v3/stage1/action_window_geometry.py",
    "wm3d_v3/stage1/robot_mask_cache.py",
    "wm3d_v3/stage1/action_signal_extractor.py",
    "wm3d_v3/stage1/action_alignment.py",
    "wm3d_v3/stage1/action_contract_evidence.py",
    "wm3d_v3/stage1/droid_interval_action.py",
    "wm3d_v3/stage1/action_contract.py",
)


@dataclass(frozen=True)
class ClipBundle:
    record: OXEClipRecord
    actions: np.ndarray
    signals: AlignmentSignals
    target: tuple[int, ...]
    signal_target: tuple[int, ...]
    action_frame_indices_by_offset: Mapping[int, tuple[int, ...]]
    manifest_identity: Mapping[str, Any]
    manifest_identity_sha256: str
    cache_identity: Mapping[str, Any]
    cache_identity_sha256: str
    temporal_binding: Mapping[str, Any]
    temporal_binding_sha256: str
    binding_identity_sha256: str
    mask_sha256: str
    mask_informative: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    else:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _split_temporal_binding_spec() -> dict[str, Any]:
    return {
        "schema_version": SPLIT_TEMPORAL_BINDING_SCHEMA,
        "binding_schema_version": TEMPORAL_WINDOW_BINDING_SCHEMA,
        "candidate_offsets": list(CANDIDATE_OFFSETS),
        "context_length": EVIDENCE_CONTEXT_LENGTH,
        "start_derivation": TEMPORAL_WINDOW_DERIVATION,
        "resolver": {
            "path": "wm3d_v3/stage1/action_contract.py",
            "symbol": "resolve_action_window",
        },
    }


def _source_provenance_payload() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    files: list[dict[str, Any]] = []
    for relative in PROVENANCE_SOURCE_FILES:
        entry: dict[str, Any] = {
            "path": relative,
            "sha256": _sha256(repo_root / relative),
        }
        if relative.endswith("action_contract.py"):
            entry["symbols"] = ["resolve_action_window"]
        files.append(entry)
    return {
        "schema_version": EVIDENCE_PROVENANCE_SCHEMA,
        "tree_sha256": _sha256_payload(files),
        "files": files,
    }


def _bundle_ledger_entry(role: str, bundle: ClipBundle) -> dict[str, Any]:
    return {
        "role": role,
        "group_id": bundle.temporal_binding["group_id"],
        "manifest_identity": bundle.manifest_identity,
        "manifest_identity_sha256": bundle.manifest_identity_sha256,
        "cache_identity": bundle.cache_identity,
        "cache_identity_sha256": bundle.cache_identity_sha256,
        "temporal_window": bundle.temporal_binding,
        "action_frame_indices_by_offset": {
            str(offset): list(indices)
            for offset, indices in sorted(bundle.action_frame_indices_by_offset.items())
        },
        "temporal_window_sha256": bundle.temporal_binding_sha256,
        "binding_identity_sha256": bundle.binding_identity_sha256,
    }


def _write_immutable(path: Path, payload: bytes) -> PublishResult:
    try:
        return publish_immutable_bytes(path, payload)
    except ImmutableArtifactConflict as exc:
        raise EvidenceExtractionError(str(exc)) from exc


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceExtractionError(f"missing {label}: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise EvidenceExtractionError(f"{label} must be a JSON mapping")
    return payload


def _load_formula_registry(path: Path) -> dict[str, Any]:
    payload = _read_mapping(path, "formula registry")
    if payload.get("schema_version") != FORMULA_REGISTRY_SCHEMA:
        raise EvidenceExtractionError("unexpected formula registry schema")
    if payload.get("immutable") is not True:
        raise EvidenceExtractionError("formula registry must be immutable")
    modalities = payload.get("modalities")
    if not isinstance(modalities, dict) or not modalities:
        raise EvidenceExtractionError("formula registry has no modalities")
    for name, spec in modalities.items():
        if not isinstance(spec, dict):
            raise EvidenceExtractionError(f"invalid formula modality {name}")
        if float(spec.get("scale_epsilon", 0.0)) != 0.05:
            raise EvidenceExtractionError(
                f"formula modality {name} must use scale_epsilon=0.05"
            )
    if float(payload.get("ridge", 0.0)) <= 0.0:
        raise EvidenceExtractionError("formula registry ridge must be positive")
    return payload


def _split_ids(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    values = raw.get(key)
    if not isinstance(values, list):
        raise EvidenceExtractionError(f"split {key} must be a list")
    result = tuple(str(value) for value in values)
    if len(result) != 32 or len(set(result)) != 32:
        raise EvidenceExtractionError(f"split {key} must contain 32 unique clips")
    return result


def _load_splits(
    path: Path,
    *,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _read_mapping(path, "frozen split artifact")
    if payload.get("schema_version") != SPLIT_ARTIFACT_SCHEMA:
        raise EvidenceExtractionError("unexpected frozen split schema")
    if payload.get("immutable") is not True:
        raise EvidenceExtractionError("frozen split must be immutable")
    if payload.get("derivation") != "sha256(seed|contract_key|independence_group_id)":
        raise EvidenceExtractionError("unexpected frozen split derivation")
    if payload.get("temporal_window_binding") != _split_temporal_binding_spec():
        raise EvidenceExtractionError(
            "frozen split temporal_window_binding specification is invalid"
        )
    source = payload.get("source_manifest")
    if not isinstance(source, dict) or source.get("sha256") != _sha256(manifest_path):
        raise EvidenceExtractionError(
            "frozen split source manifest hash does not match evidence manifest"
        )
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise EvidenceExtractionError("frozen split has no groups")
    normalized: dict[str, dict[str, Any]] = {}
    for key, raw in sorted(groups.items()):
        if not isinstance(raw, dict) or raw.get("contract_key") != key:
            raise EvidenceExtractionError(f"invalid split group {key}")
        try:
            frozen = frozen_contract_split_from_mapping(raw)
        except ActionContractSplitError as exc:
            raise EvidenceExtractionError(
                f"invalid split group {key}: {exc}"
            ) from exc
        if frozen.contract_key != key:
            raise EvidenceExtractionError(f"invalid split contract key {key}")
        normalized[str(key)] = {
            **raw,
            "calibration_clip_ids": frozen.calibration_clip_ids,
            "qualification_clip_ids": frozen.qualification_clip_ids,
            "confirmation_clip_ids": frozen.confirmation_clip_ids,
        }
    return payload, normalized


def _record_map(
    manifest_path: Path,
) -> tuple[dict[str, OXEClipRecord], dict[str, set[str]]]:
    by_id: dict[str, OXEClipRecord] = {}
    by_contract: dict[str, set[str]] = {}
    for record in read_manifest(manifest_path):
        existing = by_id.get(record.clip_id)
        if existing is not None:
            raise EvidenceExtractionError(
                f"formal evidence manifest is not unique: {record.clip_id}"
            )
        by_id[record.clip_id] = record
        by_contract.setdefault(action_contract_key(record), set()).add(
            record.clip_id
        )
    return by_id, by_contract


def _load_droid_index(
    path: Path | None,
    *,
    manifest_path: Path,
    has_droid: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if not has_droid:
        return None, None
    if path is None:
        raise EvidenceExtractionError(
            "formal DROID evidence requires --droid-cache-index"
        )
    payload = _read_mapping(path, "DROID finalized cache index")
    if payload.get("schema_version") != DROID_INDEX_SCHEMA:
        raise EvidenceExtractionError("unexpected DROID finalized cache schema")
    plan_id = str(payload.get("plan_id", ""))
    if len(plan_id) != 64 or any(
        character not in "0123456789abcdef" for character in plan_id
    ):
        raise EvidenceExtractionError("DROID finalized cache plan_id is invalid")
    contract = payload.get("action_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("action_kind") != DROID_INTERVAL_ACTION_KIND
        or int(contract.get("action_dim", 0)) != 7
        or contract.get("valid_action_count")
        != DROID_INTERVAL_ACTION_VALID_COUNT
        or contract.get("state_count") != DROID_INTERVAL_STATE_COUNT
        or contract.get("terminal_policy") != DROID_INTERVAL_TERMINAL_POLICY
    ):
        raise EvidenceExtractionError(
            "DROID finalized cache action contract is invalid"
        )
    commit = payload.get("commit")
    output_manifest = payload.get("output_manifest")
    if (
        not isinstance(commit, dict)
        or commit.get("protocol") != "manifest_then_index_marker_v1"
        or not isinstance(output_manifest, dict)
    ):
        raise EvidenceExtractionError(
            "DROID finalized cache commit marker is invalid"
        )
    try:
        index_manifest_path = Path(
            str(output_manifest.get("path", ""))
        ).resolve(strict=True)
    except OSError as exc:
        raise EvidenceExtractionError(
            "DROID finalized cache output manifest is missing"
        ) from exc
    index_manifest_sha256 = _sha256(index_manifest_path)
    if (
        Path(str(commit.get("manifest_path", ""))).resolve()
        != index_manifest_path
        or output_manifest.get("sha256") != index_manifest_sha256
        or commit.get("manifest_sha256") != index_manifest_sha256
    ):
        raise EvidenceExtractionError(
            "DROID finalized cache commit marker manifest mismatch"
        )
    generation_payload = {
        key: value for key, value in payload.items() if key != "commit"
    }
    generation_id = hashlib.sha256(
        json.dumps(
            generation_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if commit.get("generation_id") != generation_id:
        raise EvidenceExtractionError(
            "DROID finalized cache commit marker generation mismatch"
        )
    coverage = payload.get("coverage")
    index_records = payload.get("records")
    manifest_droid_rows = {
        record.clip_id: record
        for record in read_manifest(manifest_path)
        if canonical_dataset_name(record.dataset) == "droid"
    }
    index_manifest_rows = read_manifest(index_manifest_path)
    index_manifest_droid_rows = {
        record.clip_id: record
        for record in index_manifest_rows
        if canonical_dataset_name(record.dataset) == "droid"
    }
    if len(index_manifest_droid_rows) != len(index_manifest_rows):
        raise EvidenceExtractionError(
            "DROID finalized cache output manifest contains non-DROID rows"
        )
    if any(
        index_manifest_droid_rows.get(clip_id) != record
        for clip_id, record in manifest_droid_rows.items()
    ):
        raise EvidenceExtractionError(
            "DROID finalized cache manifest does not cover the evidence "
            "manifest DROID subset"
        )
    manifest_droid_ids = set(manifest_droid_rows)
    if (
        not isinstance(coverage, dict)
        or coverage.get("exact") is not True
        or not isinstance(index_records, dict)
        or int(coverage.get("planned", -1)) != len(index_records)
        or int(coverage.get("built", -1)) != len(index_records)
        or not manifest_droid_ids.issubset(index_records)
    ):
        raise EvidenceExtractionError(
            "DROID finalized cache coverage is not exact"
        )
    return payload, _sha256(path)


def _clip_seed(seed: int, contract_key: str, clip_id: str) -> int:
    digest = hashlib.sha256(
        f"{int(seed)}|{contract_key}|{clip_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _load_window_state(
    record: OXEClipRecord,
    *,
    droid_index: Mapping[str, Any] | None,
    cache_root: Path,
    source_hash_cache: dict[str, str],
) -> tuple[np.ndarray | None, np.ndarray | None, Mapping[str, Any]]:
    if canonical_dataset_name(record.dataset) == "droid":
        if droid_index is None:
            raise EvidenceExtractionError("DROID finalized cache index is missing")
        records = droid_index.get("records")
        raw = records.get(record.clip_id) if isinstance(records, dict) else None
        state = raw.get("state") if isinstance(raw, dict) else None
        if not isinstance(state, dict):
            raise EvidenceExtractionError(f"DROID state binding is missing: {record.clip_id}")
        path = Path(str(state.get("path", ""))).resolve()
        expected_sha = str(state.get("sha256", ""))
        payload = load_bound_npz(path, expected_sha)
        if set(payload) != {"pose", "grip"}:
            raise EvidenceExtractionError(f"DROID state NPZ fields differ: {record.clip_id}")
        pose = np.asarray(payload["pose"], dtype=np.float64)
        grip = np.asarray(payload["grip"], dtype=np.float64).reshape(-1)
        identity = {"path": str(path), "sha256": expected_sha}
    else:
        pose, grip = _load_alignment_robot_state(record, cache_root=cache_root)
        source = Path(record.tar_path).resolve()
        if not source.is_file() or not record.pickle_member:
            raise EvidenceExtractionError(
                f"frozen formal clip lacks raw state source: {record.clip_id}"
            )
        source_key = str(source)
        source_sha = source_hash_cache.setdefault(source_key, _sha256(source))
        identity = {
            "path": source_key,
            "sha256": source_sha,
            "pickle_member": record.pickle_member,
        }
    if pose is not None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] < 3 or not np.isfinite(pose).all():
            raise EvidenceExtractionError(f"invalid state pose: {record.clip_id}")
    if grip is not None:
        grip = np.asarray(grip, dtype=np.float64).reshape(-1)
        if not np.isfinite(grip).all():
            raise EvidenceExtractionError(f"invalid state grip: {record.clip_id}")
    return pose, grip, identity


def _load_bundle(
    record: OXEClipRecord,
    *,
    role: str,
    cache_root: Path,
    group_id: str,
    droid_index: Mapping[str, Any] | None,
    windows: ValidatedActionEvidenceWindows,
    source_hash_cache: dict[str, str],
    signal_config: ActionSignalConfig,
) -> ClipBundle:
    contract_key = action_contract_key(record)
    matching = [
        identity for identity in windows
        if identity[0] == contract_key and identity[1] == role and identity[2] == record.clip_id
    ]
    if len(matching) != 1:
        raise EvidenceExtractionError(
            f"validated window identity is not unique: {contract_key} {role} {record.clip_id}"
        )
    identity = matching[0]
    window = windows[identity]
    if window.group_id != group_id:
        raise EvidenceExtractionError(f"validated window group mismatch: {identity}")
    arrays = windows.load_arrays(identity)
    start = int(identity[3])
    usable_frames = min(int(record.n_frames), int(window.source_rgb["shape"][0]))
    temporal_binding = _temporal_window_binding_payload(
        _bind_temporal_window(
            record,
            group_id=group_id,
            seed=1729,
            usable_frames=usable_frames,
            n_action_frames=int(arrays.actions.shape[0]),
            target_length=FORMAL_TARGET_LENGTH,
        )
    )
    if int(temporal_binding["start"]) != start:
        raise EvidenceExtractionError(f"validated window start mismatch: {identity}")
    target = tuple(int(value) for value in temporal_binding["target_frame_indices"])
    if target != tuple(window.frame_indices[1:]):
        raise EvidenceExtractionError(f"validated target-frame mismatch: {identity}")
    action_frame_indices_by_offset = {
        int(offset): tuple(int(value) for value in binding["action_frame_indices"])
        for offset, binding in temporal_binding["by_offset"].items()
    }
    pose, grip, state_identity = _load_window_state(
        record,
        droid_index=droid_index,
        cache_root=cache_root,
        source_hash_cache=source_hash_cache,
    )
    stop = start + 17
    if pose is not None:
        if pose.shape[0] < stop:
            raise EvidenceExtractionError(f"state pose window is short: {identity}")
        pose = np.array(pose[start:stop], copy=True)
    if grip is not None:
        if grip.shape[0] < stop:
            raise EvidenceExtractionError(f"state grip window is short: {identity}")
        grip = np.array(grip[start:stop], copy=True)
    signal_target = tuple(range(1, 17))
    signals = extract_cache_alignment_signals(
        rgb=arrays.rgb,
        depth=arrays.depth,
        target_frame_indices=signal_target,
        signal_config=signal_config,
        state_pose=pose,
        state_grip=grip,
        robot_masks=arrays.robot_masks,
    )
    if signals.state_pose_delta is None:
        raise EvidenceExtractionError(
            f"frozen formal clip lacks required state evidence: {record.clip_id}"
        )
    manifest_identity = asdict(record)
    cache_identity: dict[str, Any] = {
        "source_rgb": dict(window.source_rgb),
        "source_action": dict(window.source_action),
        "state_source": dict(state_identity),
        "geometry_index": {
            "path": windows.geometry_index.index_path,
            "sha256": windows.geometry_index.index_sha256,
        },
        "geometry_output": {"path": window.geometry_path, **dict(window.geometry_output)},
        "robot_mask_index": {
            "path": windows.robot_mask_index.index_path,
            "sha256": windows.robot_mask_index.index_sha256,
        },
        "robot_mask_output": {
            **({"path": window.mask_path} if window.mask_path else {}),
            **dict(window.mask_output),
        },
    }
    manifest_identity_sha256 = _sha256_payload(manifest_identity)
    cache_identity_sha256 = _sha256_payload(cache_identity)
    temporal_binding_sha256 = _sha256_payload(temporal_binding)
    binding_identity_sha256 = _sha256_payload(
        {
            "manifest_identity_sha256": manifest_identity_sha256,
            "cache_identity_sha256": cache_identity_sha256,
            "temporal_window_sha256": temporal_binding_sha256,
        }
    )
    return ClipBundle(
        record=record,
        actions=arrays.actions,
        signals=signals,
        target=target,
        signal_target=signal_target,
        action_frame_indices_by_offset=action_frame_indices_by_offset,
        manifest_identity=manifest_identity,
        manifest_identity_sha256=manifest_identity_sha256,
        cache_identity=cache_identity,
        cache_identity_sha256=cache_identity_sha256,
        temporal_binding=temporal_binding,
        temporal_binding_sha256=temporal_binding_sha256,
        binding_identity_sha256=binding_identity_sha256,
        mask_sha256=str(
            window.mask_output["mask_sha256"]
            if window.mask_informative
            else window.mask_output["binding_sha256"]
        ),
        mask_informative=bool(window.mask_informative),
    )


def _projection_payload(
    frozen_by_contract: Mapping[str, FrozenProjectionSet],
    *,
    calibration_binding_sha256_by_contract: Mapping[str, Mapping[str, str]],
    provenance: Mapping[str, Any],
    source_manifest: Path,
    split_path: Path,
    registry_path: Path,
    windows: ValidatedActionEvidenceWindows,
    signal_config: ActionSignalConfig,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECTION_SCHEMA,
        "immutable": True,
        "source_manifest_sha256": _sha256(source_manifest),
        "split_artifact_sha256": _sha256(split_path),
        "formula_registry_sha256": _sha256(registry_path),
        "geometry_index": {
            "path": windows.geometry_index.index_path,
            "sha256": windows.geometry_index.index_sha256,
        },
        "robot_mask_index": {
            "path": windows.robot_mask_index.index_path,
            "sha256": windows.robot_mask_index.index_sha256,
        },
        "action_signal_config": asdict(signal_config),
        "action_signal_config_sha256": _sha256_payload(asdict(signal_config)),
        "candidate_offsets": list(CANDIDATE_OFFSETS),
        "context_length": EVIDENCE_CONTEXT_LENGTH,
        "provenance": provenance,
        "provenance_sha256": _sha256_payload(provenance),
        "groups": {
            key: {
                "calibration_clip_ids": list(value.calibration_clip_ids),
                "calibration_binding_sha256_by_clip": dict(
                    sorted(calibration_binding_sha256_by_contract[key].items())
                ),
                "ridge": value.ridge,
                "by_offset": {
                    str(offset): asdict(projection)
                    for offset, projection in sorted(value.by_offset.items())
                },
            }
            for key, value in sorted(frozen_by_contract.items())
        },
    }


def _modality_payload(modality: Any) -> dict[str, Any]:
    payload = {
        "observed": float(modality.observed),
        "null_samples": [float(value) for value in modality.null_samples],
        "informative": bool(modality.informative),
    }
    if str(modality.family).strip() in VISUAL_MODALITY_FAMILIES:
        payload.update(
            {
                field: getattr(modality, field)
                for field in VISUAL_MODALITY_AUDIT_FIELDS
            }
        )
    return payload


def _evidence_row(
    *,
    contract_key: str,
    role: str,
    bundle: ClipBundle,
    row: Any,
    projection_sha256: str,
    registered_modalities: set[str],
) -> dict[str, Any]:
    names = set(row.modalities)
    if not names or not names.issubset(registered_modalities):
        raise EvidenceExtractionError(
            f"unregistered produced modalities for {bundle.record.clip_id}: "
            f"{sorted(names.difference(registered_modalities))}"
        )
    offset_binding = bundle.temporal_binding["by_offset"][str(int(row.offset))]
    return {
        "contract_key": contract_key,
        "clip_id": row.clip_id,
        "group_id": bundle.temporal_binding["group_id"],
        "split_role": role,
        "offset": int(row.offset),
        "start": int(bundle.temporal_binding["start"]),
        "target_frame_indices": list(bundle.target),
        "action_frame_indices": list(offset_binding["action_frame_indices"]),
        "previous_gripper_index": int(offset_binding["previous_gripper_index"]),
        "robot_mask_sha256": bundle.mask_sha256,
        "manifest_identity_sha256": bundle.manifest_identity_sha256,
        "cache_identity_sha256": bundle.cache_identity_sha256,
        "temporal_window_sha256": bundle.temporal_binding_sha256,
        "binding_identity_sha256": bundle.binding_identity_sha256,
        "projection_artifact_sha256": projection_sha256,
        "modalities": {
            name: _modality_payload(modality)
            for name, modality in sorted(row.modalities.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    cache_root = Path(args.cache_root).resolve()
    geometry_index_path = Path(args.geometry_index).resolve()
    robot_mask_index_path = Path(args.robot_mask_index).resolve()
    registry_path = Path(args.formula_registry).resolve()
    split_path = Path(args.split_artifact).resolve()
    projection_path = Path(args.projection_out).resolve()
    evidence_path = Path(args.evidence_out).resolve()
    metadata_path = Path(args.evidence_metadata_out).resolve()
    try:
        require_distinct_output_paths(
            {
                "projection": projection_path,
                "evidence": evidence_path,
                "metadata": metadata_path,
            }
        )
    except ImmutableArtifactConflict as exc:
        raise EvidenceExtractionError(str(exc)) from exc
    droid_index_path = (
        None
        if args.droid_cache_index is None
        else Path(args.droid_cache_index).resolve()
    )
    target_length = FORMAL_TARGET_LENGTH
    null_repeats = int(args.null_repeats)
    seed = int(args.seed)
    test_mode = bool(args.test_mode)
    signal_config = FORMAL_ACTION_SIGNAL_CONFIG
    if seed != 1729:
        raise EvidenceExtractionError("formal window/null seed must remain 1729")
    if test_mode:
        if null_repeats < 32:
            raise EvidenceExtractionError("test null_repeats cannot be below 32")
    elif null_repeats != FORMAL_NULL_REPEATS:
        raise EvidenceExtractionError("formal evidence requires null_repeats=256")

    registry = _load_formula_registry(registry_path)
    _, split_groups = _load_splits(split_path, manifest_path=manifest_path)
    records, contract_records = _record_map(manifest_path)
    if set(contract_records) != set(split_groups):
        raise EvidenceExtractionError(
            "manifest and frozen split contract groups differ"
        )
    has_droid = any(
        canonical_dataset_name(record.dataset) == "droid"
        for record in records.values()
    )
    droid_index, droid_index_sha = _load_droid_index(
        droid_index_path,
        manifest_path=manifest_path,
        has_droid=has_droid,
    )
    if droid_index_path is None:
        raise EvidenceExtractionError("formal evidence requires a DROID finalized cache index")
    try:
        windows = load_validated_action_evidence_windows(
            geometry_index_path=geometry_index_path,
            robot_mask_index_path=robot_mask_index_path,
            split_artifact=split_path,
            manifest_path=manifest_path,
            cache_root=cache_root,
            droid_cache_index=droid_index_path,
        )
    except Exception as exc:
        raise EvidenceExtractionError(f"validated action-evidence windows failed: {exc}") from exc
    provenance = _source_provenance_payload()
    source_hash_cache: dict[str, str] = {}
    cache_ledger: dict[str, dict[str, Any]] = {}
    calibration_binding_sha256_by_contract: dict[str, dict[str, str]] = {}
    frozen_by_contract: dict[str, FrozenProjectionSet] = {}

    for contract_key, split in sorted(split_groups.items()):
        all_ids = (
            split["calibration_clip_ids"]
            + split["qualification_clip_ids"]
            + split["confirmation_clip_ids"]
        )
        missing = set(all_ids).difference(records)
        if missing:
            raise EvidenceExtractionError(
                f"frozen split clips are absent from manifest: {sorted(missing)}"
            )
        calibration: list[ProjectionCalibrationClip] = []
        cache_ledger[contract_key] = {}
        calibration_binding_sha256_by_contract[contract_key] = {}
        for clip_id in split["calibration_clip_ids"]:
            record = records[clip_id]
            if action_contract_key(record) != contract_key:
                raise EvidenceExtractionError(
                    f"frozen calibration clip contract mismatch: {clip_id}"
                )
            bundle = _load_bundle(
                record,
                role="calibration",
                cache_root=cache_root,
                group_id=str(split["clip_to_group_id"][clip_id]),
                droid_index=droid_index,
                windows=windows,
                source_hash_cache=source_hash_cache,
                signal_config=signal_config,
            )
            cache_ledger[contract_key][clip_id] = _bundle_ledger_entry(
                "calibration",
                bundle,
            )
            calibration_binding_sha256_by_contract[contract_key][clip_id] = (
                bundle.binding_identity_sha256
            )
            calibration.append(
                ProjectionCalibrationClip(
                    clip_id=clip_id,
                    actions=bundle.actions,
                    target_frame_indices=bundle.signal_target,
                    action_frame_indices_by_offset=bundle.action_frame_indices_by_offset,
                    flow_vectors=bundle.signals.flow_vectors,
                    depth_delta=bundle.signals.depth_delta,
                    source_informative=bool(bundle.mask_informative),
                )
            )
        frozen_by_contract[contract_key] = fit_frozen_offset_projections(
            calibration,
            offsets=CANDIDATE_OFFSETS,
            ridge=float(registry["ridge"]),
            minimum_clips=24,
        )

    projection_payload = _projection_payload(
        frozen_by_contract,
        calibration_binding_sha256_by_contract=calibration_binding_sha256_by_contract,
        provenance=provenance,
        source_manifest=manifest_path,
        split_path=split_path,
        registry_path=registry_path,
        windows=windows,
        signal_config=signal_config,
    )
    projection_bytes = _canonical_bytes(projection_payload, pretty=True)
    projection_result = _write_immutable(
        projection_path,
        projection_bytes,
    )
    projection_sha = projection_result.sha256
    evidence_rows: list[dict[str, Any]] = []
    registered_modalities = set(registry["modalities"])
    group_metadata: dict[str, Any] = {}

    for contract_key, split in sorted(split_groups.items()):
        signature: tuple[str, ...] | None = None
        for role, clip_ids in (
            ("qualification", split["qualification_clip_ids"]),
            ("confirmation", split["confirmation_clip_ids"]),
        ):
            for clip_id in clip_ids:
                record = records[clip_id]
                if action_contract_key(record) != contract_key:
                    raise EvidenceExtractionError(
                        f"frozen evidence clip contract mismatch: {clip_id}"
                    )
                bundle = _load_bundle(
                    record,
                    role=role,
                    cache_root=cache_root,
                    group_id=str(split["clip_to_group_id"][clip_id]),
                    droid_index=droid_index,
                    windows=windows,
                    source_hash_cache=source_hash_cache,
                    signal_config=signal_config,
                )
                cache_ledger[contract_key][clip_id] = _bundle_ledger_entry(
                    role,
                    bundle,
                )
                rows = build_clip_offset_evidence(
                    clip_id=clip_id,
                    actions=bundle.actions,
                    signals=bundle.signals,
                    projection=None,
                    action_frame_indices_by_offset=bundle.action_frame_indices_by_offset,
                    offsets=CANDIDATE_OFFSETS,
                    null_repeats=null_repeats,
                    seed=_clip_seed(seed, contract_key, clip_id),
                    frozen_projections=frozen_by_contract[contract_key],
                )
                current_signature = tuple(sorted(rows[0].modalities))
                if any(
                    tuple(sorted(row.modalities)) != current_signature
                    for row in rows
                ):
                    raise EvidenceExtractionError(
                        f"offset modality signature mismatch: {clip_id}"
                    )
                if signature is None:
                    signature = current_signature
                elif signature != current_signature:
                    raise EvidenceExtractionError(
                        f"frozen clip modality signature mismatch: {clip_id}"
                    )
                evidence_rows.extend(
                    _evidence_row(
                        contract_key=contract_key,
                        role=role,
                        bundle=bundle,
                        row=row,
                        projection_sha256=projection_sha,
                        registered_modalities=registered_modalities,
                    )
                    for row in rows
                )
        group_metadata[contract_key] = {
            "calibration_clip_ids": list(split["calibration_clip_ids"]),
            "qualification_clip_ids": list(split["qualification_clip_ids"]),
            "confirmation_clip_ids": list(split["confirmation_clip_ids"]),
            "modality_signature": list(signature or ()),
            "cache_ledger_sha256": _sha256_payload(cache_ledger[contract_key]),
        }

    evidence_bytes = b"".join(
        _canonical_bytes(row)
        for row in evidence_rows
    )
    evidence_result = _write_immutable(evidence_path, evidence_bytes)
    metadata_payload = {
        "schema_version": EVIDENCE_METADATA_SCHEMA,
        "immutable": True,
        "source_manifest_sha256": _sha256(manifest_path),
        "formula_registry_sha256": _sha256(registry_path),
        "geometry_index": {
            "path": windows.geometry_index.index_path,
            "sha256": windows.geometry_index.index_sha256,
        },
        "robot_mask_index": {
            "path": windows.robot_mask_index.index_path,
            "sha256": windows.robot_mask_index.index_sha256,
        },
        "action_signal_config": asdict(signal_config),
        "action_signal_config_sha256": _sha256_payload(asdict(signal_config)),
        "split_artifact_sha256": _sha256(split_path),
        "projection_artifact_sha256": projection_sha,
        "evidence_sha256": evidence_result.sha256,
        "droid_cache_index_sha256": droid_index_sha,
        "extractor_sha256": _sha256(Path(__file__)),
        "provenance": provenance,
        "provenance_sha256": _sha256_payload(provenance),
        "candidate_offsets": list(CANDIDATE_OFFSETS),
        "context_length": EVIDENCE_CONTEXT_LENGTH,
        "seed": seed,
        "evaluation_mode": "test" if test_mode else "formal",
        "target_length": target_length,
        "null_repeats": null_repeats,
        "block_size": NULL_BLOCK_SIZE,
        "groups": group_metadata,
        "cache_ledger": cache_ledger,
    }
    metadata_bytes = _canonical_bytes(metadata_payload, pretty=True)
    metadata_result = _write_immutable(
        metadata_path,
        metadata_bytes,
    )
    return {
        "contract_count": len(split_groups),
        "evidence_rows": len(evidence_rows),
        "projection_sha256": projection_sha,
        "evidence_sha256": evidence_result.sha256,
        "evidence_metadata_sha256": metadata_result.sha256,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--geometry-index", required=True)
    parser.add_argument("--robot-mask-index", required=True)
    parser.add_argument("--formula-registry", required=True)
    parser.add_argument("--split-artifact", required=True)
    parser.add_argument("--projection-out", required=True)
    parser.add_argument("--evidence-out", required=True)
    parser.add_argument("--evidence-metadata-out", required=True)
    parser.add_argument("--droid-cache-index")
    parser.add_argument("--null-repeats", type=int, default=FORMAL_NULL_REPEATS)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args(argv)


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))
