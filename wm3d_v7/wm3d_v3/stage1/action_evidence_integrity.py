from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from wm3d_v3.stage1.action_evidence_window_sources import (
    ActionEvidenceWindowSourceError,
    ValidatedActionEvidenceWindows,
    load_validated_action_evidence_windows,
)


METADATA_SCHEMA = "wm3d_v6_action_evidence_metadata_v3"
PROJECTION_SCHEMA = "wm3d_v6_action_frozen_projection_v2"
PROVENANCE_SCHEMA = "wm3d_v6_action_evidence_source_provenance_v2"
CANDIDATE_OFFSETS = tuple(range(-2, 3))
CONTEXT_LENGTH = 1
ACTION_SIGNAL_CONFIG = {
    "motion_quantile": 0.7,
    "min_motion_pixels": 512,
}
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


class ActionEvidenceIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class ActionEvidenceIntegrity:
    metadata_sha256: str
    projection_sha256: str
    evidence_sha256: str
    source_manifest_sha256: str
    group_count: int
    row_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_sha256(path: Path, label: str) -> str:
    if not path.is_file():
        raise ActionEvidenceIntegrityError(f"missing {label}: {path}")
    return _sha256(path)


def _external_index_binding(path: Path, label: str) -> dict[str, str]:
    raw = Path(path)
    if ".." in raw.parts:
        raise ActionEvidenceIntegrityError(
            f"{label} contains path traversal: {raw}"
        )
    absolute = raw.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise ActionEvidenceIntegrityError(
                f"missing {label}: {absolute}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ActionEvidenceIntegrityError(
                f"{label} path contains symlink: {current}"
            )
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ActionEvidenceIntegrityError(
            f"{label} must be a non-symlink regular file: {absolute}"
        )
    return {"path": str(absolute), "sha256": _sha256(absolute)}


def _validate_index_binding(
    payload: Mapping[str, Any],
    *,
    payload_label: str,
    field: str,
    expected: Mapping[str, str],
) -> None:
    binding = payload.get(field)
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ActionEvidenceIntegrityError(
            f"{payload_label} {field} binding must contain path and sha256"
        )
    if binding.get("path") != expected["path"]:
        raise ActionEvidenceIntegrityError(
            f"{payload_label} {field} path mismatch"
        )
    if binding.get("sha256") != expected["sha256"]:
        raise ActionEvidenceIntegrityError(
            f"{payload_label} {field} hash mismatch"
        )


def _canonical_sha256(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ActionEvidenceIntegrityError(f"{label} must be a JSON mapping")
    return payload


def _digest(value: Any, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ActionEvidenceIntegrityError(f"{label} is not a SHA256 digest")
    return digest


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ActionEvidenceIntegrityError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ActionEvidenceIntegrityError(f"{label} must be a finite number")
    return result


def _integer_setting(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ActionEvidenceIntegrityError(f"metadata {label} must be an integer")
    return value


def _unit_interval(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ActionEvidenceIntegrityError(f"{label} must be in [0, 1]")
    return result


def _validate_visual_modality_payload(
    modality: Mapping[str, Any],
    *,
    contract_key: str,
    clip_id: str,
    offset: int,
    modality_name: str,
) -> None:
    missing = [
        field for field in VISUAL_MODALITY_AUDIT_FIELDS if field not in modality
    ]
    if missing:
        raise ActionEvidenceIntegrityError(
            "visual evidence support audit is incomplete: "
            f"contract_key={contract_key} clip_id={clip_id} "
            f"offset={offset} modality={modality_name} missing={missing}"
        )
    support_count = modality.get("support_count")
    minimum_support_count = modality.get("minimum_support_count")
    if isinstance(support_count, bool) or not isinstance(support_count, int):
        raise ActionEvidenceIntegrityError(
            f"visual evidence support_count must be an integer: {modality_name}"
        )
    if isinstance(minimum_support_count, bool) or not isinstance(
        minimum_support_count, int
    ):
        raise ActionEvidenceIntegrityError(
            "visual evidence minimum_support_count must be an integer: "
            f"{modality_name}"
        )
    if support_count < 0 or minimum_support_count < 0:
        raise ActionEvidenceIntegrityError(
            f"visual evidence support counts must be nonnegative: {modality_name}"
        )
    if not isinstance(modality.get("fallback_used"), bool):
        raise ActionEvidenceIntegrityError(
            f"visual evidence fallback_used must be boolean: {modality_name}"
        )
    _unit_interval(
        modality.get("robot_mask_motion_coverage"),
        f"{modality_name} robot_mask_motion_coverage",
    )
    _unit_interval(
        modality.get("valid_depth_coverage"),
        f"{modality_name} valid_depth_coverage",
    )
    _unit_interval(
        modality.get("support_fraction"),
        f"{modality_name} support_fraction",
    )


def _finite_vector(value: Any, length: int, label: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ActionEvidenceIntegrityError(f"{label} has invalid projection shape")
    for index, item in enumerate(value):
        _finite_number(item, f"{label}[{index}]")


def _finite_matrix(
    value: Any,
    rows: int,
    columns: int,
    label: str,
) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ActionEvidenceIntegrityError(f"{label} has invalid projection shape")
    for row_index, row in enumerate(value):
        _finite_vector(row, columns, f"{label}[{row_index}]")


def _validate_projection_offset(
    value: Any,
    *,
    offset: int,
    contract_key: str,
) -> None:
    label = f"projection offset {offset} for {contract_key}"
    if not isinstance(value, dict):
        raise ActionEvidenceIntegrityError(f"invalid {label}")
    if value.get("offset") != offset:
        raise ActionEvidenceIntegrityError(f"{label} has an invalid offset")
    _finite_matrix(value.get("flow_weights"), 3, 2, f"{label} flow_weights")
    _finite_vector(value.get("flow_bias"), 2, f"{label} flow_bias")
    _finite_vector(value.get("depth_weights"), 3, f"{label} depth_weights")
    _finite_number(value.get("depth_bias"), f"{label} depth_bias")


def _split_roles(split_group: Mapping[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    for role in ("calibration", "qualification", "confirmation"):
        values = split_group.get(f"{role}_clip_ids")
        if not isinstance(values, (list, tuple)) or len(values) != 32:
            raise ActionEvidenceIntegrityError(
                f"split {role} must contain exactly 32 clips"
            )
        for value in values:
            clip_id = str(value)
            if clip_id in roles:
                raise ActionEvidenceIntegrityError(
                    f"split clip appears in multiple roles: {clip_id}"
                )
            roles[clip_id] = role
    return roles


def _source_provenance_payload(extractor_path: Path) -> dict[str, Any]:
    repo_root = extractor_path.resolve().parents[2]
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
        "schema_version": PROVENANCE_SCHEMA,
        "tree_sha256": _canonical_sha256(files),
        "files": files,
    }


def validate_action_evidence_artifacts(
    *,
    source_manifest_path: Path,
    geometry_index_path: Path,
    robot_mask_index_path: Path,
    cache_root_path: Path | None = None,
    extractor_path: Path,
    droid_cache_index_path: Path | None,
    formula_registry_path: Path,
    split_artifact_path: Path,
    evidence_path: Path,
    metadata_path: Path,
    projection_path: Path,
    split_groups: Mapping[str, Mapping[str, Any]],
) -> ActionEvidenceIntegrity:
    metadata = _mapping(metadata_path, "evidence metadata")
    projection = _mapping(projection_path, "frozen projection")
    split_artifact = _mapping(split_artifact_path, "frozen split artifact")
    formula_registry = _mapping(formula_registry_path, "formula registry")
    if metadata.get("schema_version") != METADATA_SCHEMA:
        raise ActionEvidenceIntegrityError("unexpected evidence metadata schema")
    if metadata.get("immutable") is not True:
        raise ActionEvidenceIntegrityError("evidence metadata must be immutable")
    if projection.get("schema_version") != PROJECTION_SCHEMA:
        raise ActionEvidenceIntegrityError("unexpected frozen projection schema")
    if projection.get("immutable") is not True:
        raise ActionEvidenceIntegrityError("frozen projection must be immutable")
    for payload, label in ((metadata, "metadata"), (projection, "projection")):
        if "mask_registry" in payload or "mask_registry_sha256" in payload:
            raise ActionEvidenceIntegrityError(
                f"{label} contains obsolete mask registry binding"
            )

    source_sha = _external_sha256(source_manifest_path, "source manifest")
    geometry_index = _external_index_binding(
        geometry_index_path, "geometry index"
    )
    robot_mask_index = _external_index_binding(
        robot_mask_index_path, "robot-mask index"
    )
    extractor_sha = _external_sha256(extractor_path, "extractor")
    provenance = _source_provenance_payload(extractor_path)
    provenance_sha = _canonical_sha256(provenance)
    has_droid = any(
        str(contract_key).split("|", 1)[0] == "droid"
        for contract_key in split_groups
    )
    if has_droid and droid_cache_index_path is None:
        raise ActionEvidenceIntegrityError(
            "DROID evidence requires a DROID cache index path"
        )
    droid_index_sha = (
        None
        if droid_cache_index_path is None
        else _external_sha256(droid_cache_index_path, "DROID cache index")
    )
    formal_windows: ValidatedActionEvidenceWindows | None = None
    if metadata.get("evaluation_mode") == "formal":
        if cache_root_path is None or droid_cache_index_path is None:
            raise ActionEvidenceIntegrityError(
                "formal evidence requires cache root and DROID cache index"
            )
        canonical_extractor = (
            Path(__file__).resolve().parents[2]
            / "wm3d_v3/stage1/action_evidence_pipeline.py"
        )
        if Path(extractor_path).resolve() != canonical_extractor:
            raise ActionEvidenceIntegrityError(
                "formal evidence extractor must be the canonical pipeline module"
            )
        try:
            formal_windows = load_validated_action_evidence_windows(
                geometry_index_path=geometry_index_path,
                robot_mask_index_path=robot_mask_index_path,
                split_artifact=split_artifact_path,
                manifest_path=source_manifest_path,
                cache_root=cache_root_path,
                droid_cache_index=droid_cache_index_path,
            )
        except ActionEvidenceWindowSourceError as exc:
            raise ActionEvidenceIntegrityError(
                f"formal 576-window index validation failed: {exc}"
            ) from exc
        if len(formal_windows) != 576:
            raise ActionEvidenceIntegrityError(
                "formal evidence must bind exactly 576 validated windows"
            )
    formula_sha = _sha256(formula_registry_path)
    split_sha = _sha256(split_artifact_path)
    evidence_sha = _sha256(evidence_path)
    metadata_sha = _sha256(metadata_path)
    projection_sha = _sha256(projection_path)
    for payload, label in (
        (metadata, "metadata"),
        (projection, "projection"),
    ):
        _validate_index_binding(
            payload,
            payload_label=label,
            field="geometry_index",
            expected=geometry_index,
        )
        _validate_index_binding(
            payload,
            payload_label=label,
            field="robot_mask_index",
            expected=robot_mask_index,
        )
        if payload.get("action_signal_config") != ACTION_SIGNAL_CONFIG:
            raise ActionEvidenceIntegrityError(
                f"{label} action signal config mismatch"
            )
        if payload.get("action_signal_config_sha256") != _canonical_sha256(
            ACTION_SIGNAL_CONFIG
        ):
            raise ActionEvidenceIntegrityError(
                f"{label} action signal config hash mismatch"
            )
        if payload.get("formula_registry_sha256") != formula_sha:
            raise ActionEvidenceIntegrityError(
                f"{label} formula registry hash mismatch"
            )
        if payload.get("split_artifact_sha256") != split_sha:
            raise ActionEvidenceIntegrityError(
                f"{label} split artifact hash mismatch"
            )
        if payload.get("provenance") != provenance:
            raise ActionEvidenceIntegrityError(f"{label} provenance mismatch")
        if payload.get("provenance_sha256") != provenance_sha:
            raise ActionEvidenceIntegrityError(
                f"{label} provenance hash mismatch"
            )
    if metadata.get("projection_artifact_sha256") != projection_sha:
        raise ActionEvidenceIntegrityError("metadata projection hash mismatch")
    if metadata.get("evidence_sha256") != evidence_sha:
        raise ActionEvidenceIntegrityError("metadata evidence hash mismatch")
    metadata_source_sha = _digest(
        metadata.get("source_manifest_sha256"),
        "source manifest hash",
    )
    if metadata_source_sha != source_sha:
        raise ActionEvidenceIntegrityError("metadata source manifest hash mismatch")
    if projection.get("source_manifest_sha256") != source_sha:
        raise ActionEvidenceIntegrityError(
            "projection source manifest hash mismatch"
        )
    split_source = split_artifact.get("source_manifest")
    if (
        not isinstance(split_source, dict)
        or split_source.get("sha256") != source_sha
    ):
        raise ActionEvidenceIntegrityError("split source manifest hash mismatch")
    if metadata.get("extractor_sha256") != extractor_sha:
        raise ActionEvidenceIntegrityError("metadata extractor hash mismatch")
    if has_droid and metadata.get("droid_cache_index_sha256") != droid_index_sha:
        raise ActionEvidenceIntegrityError("metadata DROID cache index hash mismatch")
    if metadata.get("candidate_offsets") != list(CANDIDATE_OFFSETS):
        raise ActionEvidenceIntegrityError("metadata candidate offsets mismatch")
    if projection.get("candidate_offsets") != list(CANDIDATE_OFFSETS):
        raise ActionEvidenceIntegrityError("projection candidate offsets mismatch")
    if metadata.get("context_length") != CONTEXT_LENGTH:
        raise ActionEvidenceIntegrityError("metadata context_length mismatch")
    if projection.get("context_length") != CONTEXT_LENGTH:
        raise ActionEvidenceIntegrityError("projection context_length mismatch")

    metadata_groups = metadata.get("groups")
    cache_ledger = metadata.get("cache_ledger")
    projection_groups = projection.get("groups")
    expected_groups = set(split_groups)
    for value, label in (
        (metadata_groups, "metadata groups"),
        (cache_ledger, "cache ledger"),
        (projection_groups, "projection groups"),
    ):
        if not isinstance(value, dict) or set(value) != expected_groups:
            raise ActionEvidenceIntegrityError(
                f"{label} do not exactly match frozen split groups"
            )

    expected_role_by_key: dict[tuple[str, str], str] = {}
    ledger_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    signature_by_group: dict[str, set[str]] = {}
    evaluation_mode = metadata.get("evaluation_mode")
    if evaluation_mode not in {"formal", "test"}:
        raise ActionEvidenceIntegrityError(
            "metadata evaluation_mode must be formal or test"
        )
    target_length = _integer_setting(metadata.get("target_length"), "target_length")
    null_repeats = _integer_setting(metadata.get("null_repeats"), "null_repeats")
    block_size = _integer_setting(metadata.get("block_size"), "block_size")
    if block_size != 2:
        raise ActionEvidenceIntegrityError("metadata block_size must be 2")
    if evaluation_mode == "formal" and (target_length, null_repeats) != (16, 256):
        raise ActionEvidenceIntegrityError(
            "formal metadata requires target_length=16 and null_repeats=256"
        )
    if evaluation_mode == "test" and (target_length < 8 or null_repeats < 32):
        raise ActionEvidenceIntegrityError(
            "test metadata target/null settings are below minimum"
        )
    registered = formula_registry.get("modalities")
    if not isinstance(registered, dict):
        raise ActionEvidenceIntegrityError("formula registry has no modalities")
    formula_ridge = _finite_number(
        formula_registry.get("ridge"), "formula registry ridge"
    )
    if formula_ridge <= 0.0:
        raise ActionEvidenceIntegrityError("formula registry ridge must be positive")

    for contract_key, split_group in sorted(split_groups.items()):
        roles = _split_roles(split_group)
        ledger_group = cache_ledger[contract_key]
        metadata_group = metadata_groups[contract_key]
        projection_group = projection_groups[contract_key]
        if not isinstance(ledger_group, dict) or set(ledger_group) != set(roles):
            raise ActionEvidenceIntegrityError(
                f"cache ledger does not cover frozen clips for {contract_key}"
            )
        if not isinstance(metadata_group, dict):
            raise ActionEvidenceIntegrityError(
                f"invalid metadata group {contract_key}"
            )
        if metadata_group.get("cache_ledger_sha256") != _canonical_sha256(
            ledger_group
        ):
            raise ActionEvidenceIntegrityError(
                f"cache ledger hash mismatch for {contract_key}"
            )
        for role in ("calibration", "qualification", "confirmation"):
            expected = list(split_group[f"{role}_clip_ids"])
            if metadata_group.get(f"{role}_clip_ids") != expected:
                raise ActionEvidenceIntegrityError(
                    f"metadata {role} IDs differ for {contract_key}"
                )
        if not isinstance(projection_group, dict):
            raise ActionEvidenceIntegrityError(
                f"invalid projection group {contract_key}"
            )
        projection_ridge = _finite_number(
            projection_group.get("ridge"),
            f"projection ridge for {contract_key}",
        )
        if projection_ridge != formula_ridge:
            raise ActionEvidenceIntegrityError(
                f"projection ridge differs from formula registry for {contract_key}"
            )
        if set(projection_group.get("calibration_clip_ids", ())) != {
            str(value) for value in split_group["calibration_clip_ids"]
        }:
            raise ActionEvidenceIntegrityError(
                f"projection calibration IDs differ for {contract_key}"
            )
        calibration_binding_sha = projection_group.get(
            "calibration_binding_sha256_by_clip"
        )
        if not isinstance(calibration_binding_sha, dict):
            raise ActionEvidenceIntegrityError(
                f"projection calibration bindings missing for {contract_key}"
            )
        by_offset = projection_group.get("by_offset")
        if not isinstance(by_offset, dict) or set(by_offset) != {
            "-2",
            "-1",
            "0",
            "1",
            "2",
        }:
            raise ActionEvidenceIntegrityError(
                f"projection offsets are incomplete for {contract_key}"
            )
        for offset in range(-2, 3):
            _validate_projection_offset(
                by_offset[str(offset)],
                offset=offset,
                contract_key=contract_key,
            )
        signature = metadata_group.get("modality_signature")
        if (
            not isinstance(signature, list)
            or not signature
            or not set(signature).issubset(registered)
        ):
            raise ActionEvidenceIntegrityError(
                f"invalid modality signature for {contract_key}"
            )
        signature_by_group[contract_key] = set(signature)
        for clip_id, role in roles.items():
            entry = ledger_group[clip_id]
            if not isinstance(entry, dict) or entry.get("role") != role:
                raise ActionEvidenceIntegrityError(
                    f"cache ledger role mismatch for {clip_id}"
                )
            if entry.get("group_id") != split_group["clip_to_group_id"][clip_id]:
                raise ActionEvidenceIntegrityError(
                    f"cache ledger group identity mismatch for {clip_id}"
                )
            manifest_identity = entry.get("manifest_identity")
            cache_identity = entry.get("cache_identity")
            temporal_window = entry.get("temporal_window")
            if not isinstance(manifest_identity, dict):
                raise ActionEvidenceIntegrityError(
                    f"manifest identity missing for {clip_id}"
                )
            if not isinstance(cache_identity, dict):
                raise ActionEvidenceIntegrityError(
                    f"cache identity missing for {clip_id}"
                )
            if not isinstance(temporal_window, dict):
                raise ActionEvidenceIntegrityError(
                    f"temporal window missing for {clip_id}"
                )
            manifest_identity_sha = _canonical_sha256(manifest_identity)
            cache_identity_sha = _canonical_sha256(cache_identity)
            temporal_window_sha = _canonical_sha256(temporal_window)
            binding_identity_sha = _canonical_sha256(
                {
                    "manifest_identity_sha256": manifest_identity_sha,
                    "cache_identity_sha256": cache_identity_sha,
                    "temporal_window_sha256": temporal_window_sha,
                }
            )
            if entry.get("manifest_identity_sha256") != manifest_identity_sha:
                raise ActionEvidenceIntegrityError(
                    f"manifest identity hash mismatch for {clip_id}"
                )
            if entry.get("cache_identity_sha256") != cache_identity_sha:
                raise ActionEvidenceIntegrityError(
                    f"cache identity hash mismatch for {clip_id}"
                )
            if entry.get("temporal_window_sha256") != temporal_window_sha:
                raise ActionEvidenceIntegrityError(
                    f"temporal window hash mismatch for {clip_id}"
                )
            if entry.get("binding_identity_sha256") != binding_identity_sha:
                raise ActionEvidenceIntegrityError(
                    f"binding identity hash mismatch for {clip_id}"
                )
            if temporal_window.get("group_id") != split_group["clip_to_group_id"][clip_id]:
                raise ActionEvidenceIntegrityError(
                    f"temporal window group identity mismatch for {clip_id}"
                )
            if temporal_window.get("contract_key") != contract_key:
                raise ActionEvidenceIntegrityError(
                    f"temporal window contract mismatch for {clip_id}"
                )
            if temporal_window.get("clip_id") != clip_id:
                raise ActionEvidenceIntegrityError(
                    f"temporal window clip mismatch for {clip_id}"
                )
            if temporal_window.get("candidate_offsets") != list(CANDIDATE_OFFSETS):
                raise ActionEvidenceIntegrityError(
                    f"temporal window candidate offsets mismatch for {clip_id}"
                )
            if temporal_window.get("context_length") != CONTEXT_LENGTH:
                raise ActionEvidenceIntegrityError(
                    f"temporal window context mismatch for {clip_id}"
                )
            target = temporal_window.get("target_frame_indices")
            if formal_windows is not None:
                window_key = (
                    contract_key,
                    role,
                    clip_id,
                    int(temporal_window.get("start", -1)),
                )
                try:
                    window = formal_windows[window_key]
                except KeyError as exc:
                    raise ActionEvidenceIntegrityError(
                        f"ledger window is absent from validated indexes: {window_key}"
                    ) from exc
                expected_cache_fields = {
                    "source_rgb": dict(window.source_rgb),
                    "source_action": dict(window.source_action),
                    "geometry_index": {
                        "path": formal_windows.geometry_index.index_path,
                        "sha256": formal_windows.geometry_index.index_sha256,
                    },
                    "geometry_output": {
                        "path": window.geometry_path,
                        **dict(window.geometry_output),
                    },
                    "robot_mask_index": {
                        "path": formal_windows.robot_mask_index.index_path,
                        "sha256": formal_windows.robot_mask_index.index_sha256,
                    },
                    "robot_mask_output": {
                        "path": window.mask_path,
                        **dict(window.mask_output),
                    },
                }
                for field, expected_value in expected_cache_fields.items():
                    if cache_identity.get(field) != expected_value:
                        raise ActionEvidenceIntegrityError(
                            f"ledger {field} differs from validated indexes: {clip_id}"
                        )
            if (
                not isinstance(target, list)
                or len(target) != target_length
                or any(
                    int(target[index + 1]) != int(target[index]) + 1
                    for index in range(len(target) - 1)
                )
            ):
                raise ActionEvidenceIntegrityError(
                    f"temporal window target indices invalid for {clip_id}"
                )
            by_offset_temporal = temporal_window.get("by_offset")
            if not isinstance(by_offset_temporal, dict) or set(by_offset_temporal) != {
                str(offset) for offset in CANDIDATE_OFFSETS
            }:
                raise ActionEvidenceIntegrityError(
                    f"temporal window offsets are incomplete for {clip_id}"
                )
            for offset in CANDIDATE_OFFSETS:
                offset_payload = by_offset_temporal[str(offset)]
                if (
                    not isinstance(offset_payload, dict)
                    or offset_payload.get("offset") != offset
                ):
                    raise ActionEvidenceIntegrityError(
                        f"temporal window offset payload is invalid for {clip_id}"
                    )
                action_indices = offset_payload.get("action_frame_indices")
                if (
                    not isinstance(action_indices, list)
                    or len(action_indices) != target_length
                    or any(
                        int(action_indices[index]) != int(target[index]) + offset
                        for index in range(target_length)
                    )
                ):
                    raise ActionEvidenceIntegrityError(
                        f"temporal window action indices mismatch for {clip_id}"
                    )
                previous_index = int(offset_payload.get("previous_gripper_index", -1))
                if previous_index != int(action_indices[0]) - 1:
                    raise ActionEvidenceIntegrityError(
                        f"temporal window previous gripper mismatch for {clip_id}"
                    )
            if role == "calibration":
                if calibration_binding_sha.get(clip_id) != binding_identity_sha:
                    raise ActionEvidenceIntegrityError(
                        f"projection calibration binding mismatch for {clip_id}"
                    )
            expected_role_by_key[(contract_key, clip_id)] = role
            ledger_by_key[(contract_key, clip_id)] = entry

    seen: dict[tuple[str, str], set[int]] = {}
    row_count = 0
    with evidence_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ActionEvidenceIntegrityError(
                    f"evidence row {line_number} is not a mapping"
                )
            key = (str(row.get("contract_key")), str(row.get("clip_id")))
            expected_role = expected_role_by_key.get(key)
            if expected_role not in {"qualification", "confirmation"}:
                raise ActionEvidenceIntegrityError(
                    f"evidence contains non-heldout clip at row {line_number}"
                )
            if row.get("split_role") != expected_role:
                raise ActionEvidenceIntegrityError(
                    f"evidence split role mismatch at row {line_number}"
                )
            if row.get("projection_artifact_sha256") != projection_sha:
                raise ActionEvidenceIntegrityError(
                    f"evidence projection hash mismatch at row {line_number}"
                )
            ledger_entry = ledger_by_key[key]
            if row.get("group_id") != ledger_entry["group_id"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence group identity mismatch at row {line_number}"
                )
            if row.get("manifest_identity_sha256") != ledger_entry["manifest_identity_sha256"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence manifest identity mismatch at row {line_number}"
                )
            if row.get("cache_identity_sha256") != ledger_entry["cache_identity_sha256"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence cache identity mismatch at row {line_number}"
                )
            if row.get("temporal_window_sha256") != ledger_entry["temporal_window_sha256"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence temporal window mismatch at row {line_number}"
                )
            if row.get("binding_identity_sha256") != ledger_entry["binding_identity_sha256"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence binding identity mismatch at row {line_number}"
                )
            mask_sha = _digest(row.get("robot_mask_sha256"), "robot mask hash")
            expected_mask_sha = (
                ledger_entry.get("cache_identity", {})
                .get("robot_mask_output", {})
                .get("mask_sha256")
            )
            if mask_sha != expected_mask_sha:
                raise ActionEvidenceIntegrityError(
                    f"evidence robot mask hash mismatch at row {line_number}"
                )
            target = row.get("target_frame_indices")
            if (
                not isinstance(target, list)
                or len(target) != target_length
                or any(
                    int(target[index + 1]) != int(target[index]) + 1
                    for index in range(len(target) - 1)
                )
            ):
                raise ActionEvidenceIntegrityError(
                    f"evidence target indices invalid at row {line_number}"
                )
            if target != ledger_entry["temporal_window"]["target_frame_indices"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence target indices differ from ledger at row {line_number}"
                )
            if int(row.get("start", -1)) != int(ledger_entry["temporal_window"]["start"]):
                raise ActionEvidenceIntegrityError(
                    f"evidence start mismatch at row {line_number}"
                )
            offset = int(row.get("offset"))
            if offset not in CANDIDATE_OFFSETS:
                raise ActionEvidenceIntegrityError(
                    f"invalid evidence offset at row {line_number}"
                )
            temporal_offset = ledger_entry["temporal_window"]["by_offset"][str(offset)]
            if row.get("action_frame_indices") != temporal_offset["action_frame_indices"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence action indices mismatch at row {line_number}"
                )
            if row.get("previous_gripper_index") != temporal_offset["previous_gripper_index"]:
                raise ActionEvidenceIntegrityError(
                    f"evidence previous gripper mismatch at row {line_number}"
                )
            modalities = row.get("modalities")
            if (
                not isinstance(modalities, dict)
                or set(modalities) != signature_by_group[key[0]]
            ):
                raise ActionEvidenceIntegrityError(
                    f"evidence modality signature mismatch at row {line_number}"
                )
            for modality_name, modality in modalities.items():
                if (
                    not isinstance(modality, dict)
                    or not isinstance(modality.get("informative"), bool)
                    or len(modality.get("null_samples", ())) != null_repeats
                ):
                    raise ActionEvidenceIntegrityError(
                        f"evidence modality metadata invalid at row {line_number}"
                    )
                family = str(
                    registered.get(modality_name, {}).get("family", "")
                ).strip()
                if family in VISUAL_MODALITY_FAMILIES:
                    _validate_visual_modality_payload(
                        modality,
                        contract_key=key[0],
                        clip_id=key[1],
                        offset=offset,
                        modality_name=str(modality_name),
                    )
            offsets = seen.setdefault(key, set())
            if offset in offsets:
                raise ActionEvidenceIntegrityError(
                    f"duplicate evidence offset at row {line_number}"
                )
            offsets.add(offset)

    heldout_keys = {
        key
        for key, role in expected_role_by_key.items()
        if role in {"qualification", "confirmation"}
    }
    if set(seen) != heldout_keys or any(
        offsets != set(CANDIDATE_OFFSETS) for offsets in seen.values()
    ):
        raise ActionEvidenceIntegrityError(
            "evidence does not exactly cover five offsets for every heldout clip"
        )
    return ActionEvidenceIntegrity(
        metadata_sha256=metadata_sha,
        projection_sha256=projection_sha,
        evidence_sha256=evidence_sha,
        source_manifest_sha256=source_sha,
        group_count=len(expected_groups),
        row_count=row_count,
    )
