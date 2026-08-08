from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.action_contract_evidence import (
    FORMAL_DROID_CONTRACT_KEY,
    FORMAL_DROID_METHOD,
    FORMAL_OXE_COHORT_ID,
    FORMAL_OXE_COHORT_METHOD,
    FORMAL_OXE_METHOD,
    FORMAL_OXE_CONTRACT_KEYS,
)
from wm3d_v3.stage1.droid_interval_action import (
    DROID_INTERVAL_ACTION_KIND,
    DROID_INTERVAL_ACTION_VALID_COUNT,
    DROID_INTERVAL_END_FORMULA,
    DROID_INTERVAL_GRIPPER_FORMULA,
    DROID_INTERVAL_ROTATION_FORMULA,
    DROID_INTERVAL_STATE_COUNT,
    DROID_INTERVAL_TERMINAL_POLICY,
    DROID_INTERVAL_TRANSLATION_ROTATION_FORMULA,
)


ACTION_CONTRACT_SCHEMA = "wm3d_v6_action_frame_contract_v4"
LEGACY_ACTION_CONTRACT_SCHEMA = "wm3d_v6_action_frame_contract_v2"
FORMAL_REPORT_SCHEMA = "wm3d_v6_action_frame_contract_report_v4"
ACTION_GATE_SCHEMA = "wm3d_v6_stage1_action_gate_v2"
_ALLOWED_OFFSETS = frozenset(range(-2, 3))
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_FORMULA_REGISTRY = (
    _REPO_ROOT / "configs/stage1_rgb_world_action_formula_registry_v2.json"
)
_CANONICAL_GATE_CONFIG = (
    _REPO_ROOT / "configs/stage1_rgb_world_action_gate_v2.json"
)
_DATASET_ALIASES = {
    "bridge": "bridge",
    "droid": "droid",
    "fractal20220817": "fractal20220817_data",
    "fractal20220817_data": "fractal20220817_data",
    "taco_play": "taco_play",
    "jaco_play": "jaco_play",
    "kuka": "kuka",
}


class ActionContractError(ValueError):
    """Base error for an invalid action-frame contract."""


class UnknownDatasetAlias(ActionContractError):
    pass


class ActionContractBoundaryError(ActionContractError):
    def __init__(self, clip_id: str, start: int, offset: int):
        super().__init__(
            f"action contract leaves episode bounds: clip_id={clip_id} "
            f"start={start} offset={offset}"
        )


class ActionContractCoverageError(ActionContractError):
    pass


class ActionContractFileError(ActionContractError):
    pass


@dataclass(frozen=True)
class ActionWindowResolution:
    contract_key: str
    action_frame_offset: int
    target_frame_indices: tuple[int, ...]
    action_frame_indices: tuple[int, ...]
    previous_gripper_index: int


def canonical_dataset_name(name: str) -> str:
    normalized = str(name).strip().lower()
    try:
        return _DATASET_ALIASES[normalized]
    except KeyError as exc:
        raise UnknownDatasetAlias(f"unknown dataset alias: {name!r}") from exc


def _canonical_fps(value: int | float) -> str:
    fps = float(value)
    if not (fps > 0.0):
        raise ActionContractError(f"fps must be positive, got {value!r}")
    return format(fps, ".8g")


def action_contract_key(record: OXEClipRecord) -> str:
    action_kind = str(record.action_kind).strip()
    if not action_kind:
        raise ActionContractError(f"empty action_kind for {record.clip_id}")
    return "|".join(
        (
            canonical_dataset_name(record.dataset),
            _canonical_fps(record.fps),
            action_kind,
        )
    )


def resolve_action_window(
    record: OXEClipRecord,
    *,
    start: int,
    T: int,
    k: int,
    offset: int,
    n_action_frames: int,
) -> ActionWindowResolution:
    start = int(start)
    T = int(T)
    k = int(k)
    offset = int(offset)
    n_action_frames = int(n_action_frames)
    if start < 0 or T <= 0 or k <= 0 or n_action_frames <= 0:
        raise ActionContractError(
            f"invalid window geometry start={start} T={T} k={k} "
            f"n_action_frames={n_action_frames}"
        )
    if offset not in _ALLOWED_OFFSETS:
        raise ActionContractError(
            f"action offset must be one of {sorted(_ALLOWED_OFFSETS)}, got {offset}"
        )

    target_indices = tuple(range(start + T, start + T + k))
    action_indices = tuple(index + offset for index in target_indices)
    previous_gripper_index = action_indices[0] - 1
    if (
        previous_gripper_index < 0
        or min(target_indices) < 0
        or max(target_indices) >= int(record.n_frames)
        or min(action_indices) < 0
        or max(action_indices) >= n_action_frames
    ):
        raise ActionContractBoundaryError(record.clip_id, start, offset)

    return ActionWindowResolution(
        contract_key=action_contract_key(record),
        action_frame_offset=offset,
        target_frame_indices=target_indices,
        action_frame_indices=action_indices,
        previous_gripper_index=previous_gripper_index,
    )



_HASH_FIELDS = (
    "formula_registry_sha256",
    "gate_config_sha256",
    "split_artifact_sha256",
    "evidence_sha256",
    "evidence_metadata_sha256",
    "projection_artifact_sha256",
    "source_manifest_sha256",
    "geometry_index_sha256",
    "robot_mask_index_sha256",
    "extractor_sha256",
    "diagnostic_report_sha256",
)

_ARTIFACT_HASH_FIELDS = {
    "formula_registry": "formula_registry_sha256",
    "gate_config": "gate_config_sha256",
    "split_artifact": "split_artifact_sha256",
    "evidence": "evidence_sha256",
    "evidence_metadata": "evidence_metadata_sha256",
    "projection_artifact": "projection_artifact_sha256",
    "source_manifest": "source_manifest_sha256",
    "geometry_index": "geometry_index_sha256",
    "robot_mask_index": "robot_mask_index_sha256",
    "extractor": "extractor_sha256",
    "diagnostic_report": "diagnostic_report_sha256",
}

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_stable_regular_file(path: Path, label: str) -> bytes:
    raw = Path(path)
    if ".." in raw.parts:
        raise ActionContractFileError(f"{label} contains path traversal: {raw}")
    absolute = raw.absolute()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for part in absolute.parts[1:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=parent_descriptor,
            )
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        descriptor = os.open(
            absolute.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
    except OSError as exc:
        raise ActionContractFileError(
            f"{label} path is missing, replaced, or contains a symlink: {absolute}"
        ) from exc
    finally:
        os.close(parent_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ActionContractFileError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 4 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ActionContractFileError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validate_challenger_values(
    value: object,
    *,
    challengers: set[str],
    label: str,
    minimum: float | None = None,
    strict_maximum: float | None = None,
) -> None:
    if not isinstance(value, dict) or set(value) != challengers:
        raise ActionContractFileError(
            f"{label} must report all four challengers"
        )
    numbers = [float(value[challenger]) for challenger in challengers]
    if any(not math.isfinite(number) for number in numbers):
        raise ActionContractFileError(f"{label} contains non-finite values")
    if minimum is not None and any(number < minimum for number in numbers):
        raise ActionContractFileError(f"{label} is below {minimum}")
    if strict_maximum is not None and any(
        number < 0.0 or number >= strict_maximum for number in numbers
    ):
        raise ActionContractFileError(
            f"{label} must be in [0, {strict_maximum})"
        )


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ActionContractFileError(f"{label} is not a SHA256 digest")
    return digest


def _validate_bound_artifacts(
    contract_path: Path,
    payload: Mapping[str, object],
    *,
    has_droid: bool,
    expected_droid_cache_index: Path | None,
    expected_droid_cache_sha256: str | None = None,
) -> tuple[dict[str, Path], dict[str, bytes]]:
    required = dict(_ARTIFACT_HASH_FIELDS)
    if has_droid:
        required["droid_cache_index"] = "droid_cache_index_sha256"
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(required):
        raise ActionContractFileError(
            "formal action contract artifact bundle is incomplete"
        )

    resolved: dict[str, Path] = {}
    artifact_payloads: dict[str, bytes] = {}
    for name, hash_field in required.items():
        entry = artifacts[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ActionContractFileError(
                f"{name} artifact binding must contain path and sha256"
            )
        declared = _require_sha256(entry.get("sha256"), f"{name} artifact hash")
        if payload.get(hash_field) != declared:
            raise ActionContractFileError(
                f"{name} artifact hash differs from {hash_field}"
            )
        raw_path = str(entry.get("path") or "").strip()
        if not raw_path:
            raise ActionContractFileError(f"{name} artifact path is empty")
        artifact_path = Path(raw_path)
        if not artifact_path.is_absolute():
            artifact_path = contract_path.parent / artifact_path
        artifact_payload = _read_stable_regular_file(
            artifact_path,
            f"{name} artifact",
        )
        actual = hashlib.sha256(artifact_payload).hexdigest()
        if actual != declared:
            raise ActionContractFileError(f"{name} artifact hash mismatch")
        resolved[name] = artifact_path
        artifact_payloads[name] = artifact_payload

    if expected_droid_cache_index is not None:
        if not has_droid:
            raise ActionContractFileError(
                "runtime DROID cache index supplied for a non-DROID contract"
            )
        expected_path = Path(expected_droid_cache_index)
        expected_sha = hashlib.sha256(
            _read_stable_regular_file(expected_path, "runtime DROID cache index")
        ).hexdigest()
        if expected_sha != payload["droid_cache_index_sha256"]:
            raise ActionContractFileError(
                "runtime DROID cache index differs from promoted evidence index"
            )
    if expected_droid_cache_sha256 is not None:
        if not has_droid:
            raise ActionContractFileError(
                "runtime DROID cache hash supplied for a non-DROID contract"
            )
        expected_digest = _require_sha256(
            expected_droid_cache_sha256,
            "runtime DROID cache index hash",
        )
        if expected_digest != payload["droid_cache_index_sha256"]:
            raise ActionContractFileError(
                "runtime DROID cache index differs from promoted evidence index"
            )
    return resolved, artifact_payloads
def _validate_passed_group(key: str, group: object) -> int:
    if not isinstance(group, dict) or group.get("status") != "passed":
        raise ActionContractFileError(f"contract group is not passed: {key}")
    offset = int(group.get("offset"))
    if offset not in _ALLOWED_OFFSETS:
        raise ActionContractFileError(f"invalid offset for {key}: {offset}")
    if int(group.get("clip_count", 0)) != 64:
        raise ActionContractFileError(
            f"contract group must bind exactly 64 heldout clips: {key}"
        )
    required_families = tuple(group.get("required_families", ()))
    families = tuple(group.get("binding_families", required_families))
    if (
        required_families
        not in {("state", "flow"), ("state", "geometry")}
        or families
        not in {
            ("state", "flow"),
            ("state", "geometry"),
            ("state", "flow", "geometry"),
        }
    ):
        raise ActionContractFileError(
            f"contract group lacks binding state plus visual evidence: {key}"
        )
    if tuple(group.get("required_source_classes", ())) != (
        "proprioceptive",
        "exteroceptive",
    ):
        raise ActionContractFileError(
            f"contract group lacks independent source classes: {key}"
        )
    _require_sha256(
        group.get("split_partition_sha256"),
        f"split partition hash for {key}",
    )
    qualification = group.get("qualification")
    if not isinstance(qualification, dict):
        raise ActionContractFileError(f"qualification report missing for {key}")
    required_qualification = {
        "dz_by_challenger",
        "max_t_p_by_challenger",
        "bootstrap_win_frequency",
        "informative_clip_count_by_family",
        "family_best_by_family",
        "family_dz_by_challenger",
        "family_max_t_p_by_challenger",
        "family_bootstrap_win_frequency",
    }
    if set(qualification) != required_qualification:
        raise ActionContractFileError(
            f"qualification report fields are incomplete for {key}"
        )
    challengers = {
        str(candidate) for candidate in _ALLOWED_OFFSETS if candidate != offset
    }
    _validate_challenger_values(
        qualification["dz_by_challenger"],
        challengers=challengers,
        label=f"qualification dz for {key}",
        minimum=0.30,
    )
    _validate_challenger_values(
        qualification["max_t_p_by_challenger"],
        challengers=challengers,
        label=f"qualification maxT for {key}",
        strict_maximum=0.01,
    )
    bootstrap = float(qualification["bootstrap_win_frequency"])
    if not math.isfinite(bootstrap) or not 0.80 <= bootstrap <= 1.0:
        raise ActionContractFileError(
            f"qualification bootstrap is below 0.80 for {key}"
        )
    informative = qualification["informative_clip_count_by_family"]
    if not isinstance(informative, dict) or any(
        int(informative.get(family, 0)) < 24 for family in families
    ):
        raise ActionContractFileError(
            f"qualification evidence is below 24/32 for {key}"
        )

    family_best = qualification["family_best_by_family"]
    family_effects = qualification["family_dz_by_challenger"]
    family_p = qualification["family_max_t_p_by_challenger"]
    family_win = qualification["family_bootstrap_win_frequency"]
    for value, label in (
        (family_best, "best"),
        (family_effects, "effect"),
        (family_p, "maxT"),
        (family_win, "bootstrap"),
    ):
        if not isinstance(value, dict) or set(value) != set(families):
            raise ActionContractFileError(
                f"qualification family {label} reports are incomplete for {key}"
            )
    if any(int(family_best[family]) != offset for family in families):
        raise ActionContractFileError(
            f"qualification required families do not bind offset {offset}: {key}"
        )
    for family in families:
        _validate_challenger_values(
            family_effects[family],
            challengers=challengers,
            label=f"qualification dz for {key}/{family}",
            minimum=0.30,
        )
        _validate_challenger_values(
            family_p[family],
            challengers=challengers,
            label=f"qualification maxT for {key}/{family}",
            strict_maximum=0.01,
        )
        win_frequency = float(family_win[family])
        if (
            not math.isfinite(win_frequency)
            or not 0.80 <= win_frequency <= 1.0
        ):
            raise ActionContractFileError(
                f"qualification bootstrap is below 0.80 for {key}/{family}"
            )
    confirmation = group.get("confirmation")
    confirmation_families = (
        set(confirmation) if isinstance(confirmation, dict) else set()
    )
    if (
        not isinstance(confirmation, dict)
        or not set(families).issubset(confirmation_families)
        or not confirmation_families.issubset({"state", "flow", "geometry"})
    ):
        raise ActionContractFileError(
            f"confirmation families are incomplete for {key}"
        )
    for family in sorted(confirmation_families):
        report = confirmation[family]
        if (
            not isinstance(report, dict)
            or int(report.get("clip_count", 0)) < 24
            or "dz_by_challenger" not in report
            or "holm_p_by_challenger" not in report
        ):
            raise ActionContractFileError(
                f"confirmation evidence is incomplete for {key}/{family}"
            )
        _validate_challenger_values(
            report["dz_by_challenger"],
            challengers=challengers,
            label=f"confirmation dz for {key}/{family}",
            minimum=0.30,
        )
        _validate_challenger_values(
            report["holm_p_by_challenger"],
            challengers=challengers,
            label=f"confirmation Holm p for {key}/{family}",
            strict_maximum=0.01,
        )
    return offset


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _split_partition_sha256(
    contract_key: str,
    qualification: Sequence[str],
    confirmation: Sequence[str],
) -> str:
    identity = hashlib.sha256()
    identity.update(str(contract_key).encode("utf-8"))
    for label, values in (
        ("qualification", qualification),
        ("confirmation", confirmation),
    ):
        identity.update(b"\0")
        identity.update(label.encode("ascii"))
        for value in values:
            identity.update(b"\0")
            identity.update(str(value).encode("utf-8"))
    return identity.hexdigest()


def _resolve_declared_artifact_snapshot(
    binding: object,
    *,
    label: str,
) -> tuple[Path, str, bytes]:
    if (
        not isinstance(binding, dict)
        or not {"path", "sha256"}.issubset(binding)
    ):
        raise ActionContractFileError(
            f"DROID {label} binding must contain path and sha256"
        )
    digest = _require_sha256(binding.get("sha256"), f"DROID {label} hash")
    raw_path = str(binding.get("path") or "")
    path = Path(raw_path)
    try:
        payload = _read_stable_regular_file(path, f"DROID {label} artifact")
    except ActionContractFileError as exc:
        raise ActionContractFileError(
            f"missing DROID {label} provenance artifact: {raw_path}"
        ) from exc
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ActionContractFileError(
            f"DROID {label} provenance artifact hash mismatch: {path}"
        )
    return path.absolute(), digest, payload


def _resolve_declared_artifact(
    binding: object,
    *,
    label: str,
) -> tuple[Path, str]:
    path, digest, _ = _resolve_declared_artifact_snapshot(binding, label=label)
    return path, digest


def _read_json_artifact(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            _read_stable_regular_file(path, f"{label} provenance artifact")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionContractFileError(
            f"cannot read DROID {label} provenance artifact: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ActionContractFileError(
            f"DROID {label} provenance artifact must be a mapping"
        )
    return payload


def _read_json_payload(payload: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ActionContractFileError(
            f"cannot decode DROID {label} provenance artifact"
        ) from exc
    if not isinstance(parsed, dict):
        raise ActionContractFileError(
            f"DROID {label} provenance artifact must be a mapping"
        )
    return parsed


def validate_exact_droid_cache_index(
    index_path: str | Path,
    *,
    index_payload: bytes | None = None,
) -> dict[str, Any]:
    index_path = Path(index_path)
    if index_payload is None:
        index_payload = _read_stable_regular_file(
            index_path,
            "DROID cache index provenance artifact",
        )
    index = _read_json_payload(index_payload, "cache index")
    if index.get("schema_version") != "wm3d_v6_stage1_droid_interval_cache_v2":
        raise ActionContractFileError(
            "unexpected finalized DROID cache index schema"
        )
    plan_id = _require_sha256(index.get("plan_id"), "DROID plan_id")
    records = index.get("records")
    coverage = index.get("coverage")
    if (
        not isinstance(records, dict)
        or not records
        or not isinstance(coverage, dict)
        or coverage.get("exact") is not True
        or int(coverage.get("planned", -1)) != len(records)
        or int(coverage.get("built", -1)) != len(records)
    ):
        raise ActionContractFileError(
            "finalized DROID cache index coverage is not exact"
        )

    expected_formula = {
        "translation_rotation": DROID_INTERVAL_TRANSLATION_ROTATION_FORMULA,
        "rotation": DROID_INTERVAL_ROTATION_FORMULA,
        "gripper": DROID_INTERVAL_GRIPPER_FORMULA,
        "interval_end": DROID_INTERVAL_END_FORMULA,
        "terminal": DROID_INTERVAL_TERMINAL_POLICY,
    }
    action_contract = index.get("action_contract")
    if (
        not isinstance(action_contract, dict)
        or action_contract.get("action_kind") != DROID_INTERVAL_ACTION_KIND
        or int(action_contract.get("action_dim", 0)) != 7
        or action_contract.get("valid_action_count")
        != DROID_INTERVAL_ACTION_VALID_COUNT
        or action_contract.get("state_count") != DROID_INTERVAL_STATE_COUNT
        or action_contract.get("terminal_policy")
        != DROID_INTERVAL_TERMINAL_POLICY
        or action_contract.get("formula") != expected_formula
    ):
        raise ActionContractFileError(
            "finalized DROID exact action construction is invalid"
        )

    source_path, source_sha, _source_payload = (
        _resolve_declared_artifact_snapshot(
            index.get("source_manifest"),
            label="source manifest",
        )
    )
    output_path, output_sha, _output_payload = (
        _resolve_declared_artifact_snapshot(
            index.get("output_manifest"),
            label="output manifest",
        )
    )
    plan_path, plan_sha, plan_payload = _resolve_declared_artifact_snapshot(
        index.get("plan"),
        label="plan",
    )
    plan = _read_json_payload(plan_payload, "plan")
    plan_records = plan.get("records")
    if (
        plan.get("schema_version")
        != "wm3d_v6_stage1_droid_interval_cache_plan_v2"
        or plan.get("plan_id") != plan_id
        or not isinstance(plan_records, dict)
        or set(plan_records) != set(records)
        or plan.get("source_manifest") != index.get("source_manifest")
        or plan.get("action_contract") != action_contract
    ):
        raise ActionContractFileError(
            "DROID plan provenance does not match finalized index"
        )
    plan_identity_keys = (
        "schema_version",
        "source_manifest",
        "dataset_info",
        "builder",
        "lerobot_root",
        "cache_root",
        "action_contract",
        "records",
    )
    if any(key not in plan for key in plan_identity_keys) or _canonical_sha256(
        {key: plan[key] for key in plan_identity_keys}
    ) != plan_id:
        raise ActionContractFileError("DROID plan identity hash mismatch")

    build_bindings = index.get("build_indexes")
    if not isinstance(build_bindings, list) or not build_bindings:
        raise ActionContractFileError(
            "finalized DROID cache index has no build provenance"
        )
    merged_build_records: dict[str, Any] = {}
    build_hashes: list[str] = []
    for build_index, binding in enumerate(build_bindings):
        build_path, build_sha, build_payload = (
            _resolve_declared_artifact_snapshot(
                binding,
                label=f"build index {build_index}",
            )
        )
        build = _read_json_payload(
            build_payload,
            f"build index {build_index}",
        )
        build_records = build.get("records")
        if (
            build.get("schema_version")
            != "wm3d_v6_stage1_droid_interval_cache_build_v2"
            or build.get("plan_id") != plan_id
            or not isinstance(build_records, dict)
            or not set(build_records).issubset(plan_records)
        ):
            raise ActionContractFileError(
                f"DROID build index {build_index} provenance is invalid"
            )
        for clip_id, record in build_records.items():
            previous = merged_build_records.setdefault(clip_id, record)
            if previous != record:
                raise ActionContractFileError(
                    f"DROID build indexes conflict for {clip_id}"
                )
        build_hashes.append(build_sha)
    if merged_build_records != records:
        raise ActionContractFileError(
            "DROID build provenance does not exactly reconstruct final records"
        )

    for clip_id, record in records.items():
        if not isinstance(record, dict) or record.get("clip_id") != clip_id:
            raise ActionContractFileError(
                f"DROID finalized record identity is invalid: {clip_id}"
            )
        sampled = record.get("sampled_global_indices")
        actions = record.get("actions")
        state = record.get("state")
        if (
            not isinstance(sampled, list)
            or len(sampled) < 2
            or not isinstance(actions, dict)
            or not isinstance(state, dict)
        ):
            raise ActionContractFileError(
                f"DROID finalized record counts are incomplete: {clip_id}"
            )
        state_count = len(sampled)
        if (
            actions.get("shape") != [state_count - 1, 7]
            or int(actions.get("valid_count", -1)) != state_count - 1
            or actions.get("dtype") != "float32"
            or state.get("pose_shape") != [state_count, 6]
            or state.get("grip_shape") != [state_count]
            or state.get("pose_dtype") != "float32"
            or state.get("grip_dtype") != "float32"
            or record.get("formula") != expected_formula
        ):
            raise ActionContractFileError(
                f"DROID record is not exact N-1 action/N state: {clip_id}"
            )

    commit = index.get("commit")
    output_manifest = index.get("output_manifest")
    if (
        not isinstance(commit, dict)
        or commit.get("protocol") != "manifest_then_index_marker_v1"
        or Path(str(commit.get("manifest_path", ""))).absolute() != output_path
        or commit.get("manifest_sha256") != output_sha
    ):
        raise ActionContractFileError(
            "finalized DROID cache commit marker is invalid"
        )
    generation = {
        key: value for key, value in index.items() if key != "commit"
    }
    generation_id = _canonical_sha256(generation)
    if commit.get("generation_id") != generation_id:
        raise ActionContractFileError(
            "finalized DROID cache generation identity mismatch"
        )
    return {
        "index_schema": index["schema_version"],
        "plan_id": plan_id,
        "generation_id": generation_id,
        "commit_protocol": commit["protocol"],
        "record_count": len(records),
        "coverage_exact": True,
        "plan_sha256": plan_sha,
        "build_index_sha256": build_hashes,
        "source_manifest_sha256": source_sha,
        "output_manifest_sha256": output_sha,
        "action_kind": DROID_INTERVAL_ACTION_KIND,
        "action_dim": 7,
        "valid_action_count": DROID_INTERVAL_ACTION_VALID_COUNT,
        "state_count": DROID_INTERVAL_STATE_COUNT,
        "terminal_policy": DROID_INTERVAL_TERMINAL_POLICY,
        "formula": expected_formula,
        "target_frame_action_relation": (
            "target_frame_F_uses_interval_action_F-1"
        ),
        "derived_offset": -1,
    }


def _partition_from_report(
    report_group: Mapping[str, Any],
    contract_key: str,
) -> tuple[str, dict[str, int]]:
    frozen = report_group.get("frozen_split")
    if not isinstance(frozen, dict):
        raise ActionContractFileError(
            f"diagnostic report frozen split missing for {contract_key}"
        )
    clip_sets: dict[str, tuple[str, ...]] = {}
    group_sets: dict[str, tuple[str, ...]] = {}
    for role in ("calibration", "qualification", "confirmation"):
        clips = tuple(str(value) for value in frozen.get(f"{role}_clip_ids", ()))
        groups = tuple(
            str(value) for value in frozen.get(f"{role}_group_ids", ())
        )
        if (
            len(clips) != 32
            or len(set(clips)) != 32
            or len(groups) != 32
            or len(set(groups)) != 32
            or any(not value for value in clips + groups)
        ):
            raise ActionContractFileError(
                f"diagnostic report {role} split/group IDs are not 32 unique "
                f"values for {contract_key}"
            )
        clip_sets[role] = clips
        group_sets[role] = groups
    if any(
        set(group_sets[left]).intersection(group_sets[right])
        for left, right in (
            ("calibration", "qualification"),
            ("calibration", "confirmation"),
            ("qualification", "confirmation"),
        )
    ):
        raise ActionContractFileError(
            f"diagnostic report split group IDs overlap for {contract_key}"
        )
    return (
        _split_partition_sha256(
            contract_key,
            clip_sets["qualification"],
            clip_sets["confirmation"],
        ),
        {"qualification": 32, "confirmation": 32},
    )


def _require_exact_partition_clip_ids(
    partition: Mapping[str, Any],
    expected: Sequence[str],
    *,
    label: str,
) -> None:
    actual = tuple(str(value) for value in partition.get("clip_ids", ()))
    if actual != tuple(expected):
        raise ActionContractFileError(
            f"{label} clip IDs differ from the frozen split"
        )


def build_formal_contract_claims_from_report(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    cohorts, groups = _formal_claims_from_report(report)
    return {"cohorts": cohorts, "groups": groups}


def _validate_droid_exact_claim(
    claim: Mapping[str, Any],
    exact_construction: Mapping[str, Any],
    *,
    cache_index_sha256: str,
) -> None:
    expected_construction = dict(exact_construction)
    expected_construction["cache_index_sha256"] = cache_index_sha256
    if (
        claim.get("status") != "passed"
        or claim.get("method") != FORMAL_DROID_METHOD
        or int(claim.get("offset", 99)) != -1
        or claim.get("frozen_clip_counts")
        != {"qualification": 32, "confirmation": 32}
        or claim.get("exact_construction") != expected_construction
    ):
        raise ActionContractFileError(
            "DROID exact-derived construction claim is invalid"
        )
    separation = claim.get("separation")
    if separation != {
        "kind": "non_statistical_exact_construction",
        "basis": "exact_n_minus_one_interval_action_construction",
        "statistical_separation_claimed": False,
        "statistics_role": "expected_offset_falsification",
    }:
        raise ActionContractFileError(
            "DROID exact-derived non-statistical separation claim is invalid"
        )
    falsification = claim.get("falsification")
    if not isinstance(falsification, dict) or set(falsification) != {
        "qualification",
        "confirmation",
    }:
        raise ActionContractFileError("DROID falsification claim is missing")
    for partition in ("qualification", "confirmation"):
        _validate_expected_offset_falsification(
            f"{FORMAL_DROID_CONTRACT_KEY}/{partition}",
            falsification[partition],
            expected_offset=-1,
            families=("state", "flow"),
            partition_clip_count=32,
            min_informative=24,
        )


def _validate_v3_report_and_claims(
    payload: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    artifact_payloads: Mapping[str, bytes] | None = None,
) -> dict[str, int]:
    report_payload = (
        artifact_payloads["diagnostic_report"]
        if artifact_payloads is not None
        else _read_stable_regular_file(
            artifacts["diagnostic_report"],
            "diagnostic report artifact",
        )
    )
    report = _read_json_payload(report_payload, "diagnostic report")
    if (
        report.get("schema_version") != FORMAL_REPORT_SCHEMA
        or report.get("evaluation_mode") != "formal"
        or report.get("status") != "passed"
        or report.get("contract_written") is not True
        or report.get("resample_counts") != payload.get("resample_counts")
        or report.get("evidence_settings") != payload.get("evidence_settings")
    ):
        raise ActionContractFileError(
            "bound diagnostic report is not a formal v3 report"
        )
    report_hash_fields = tuple(
        field for field in _HASH_FIELDS if field != "diagnostic_report_sha256"
    ) + ("droid_cache_index_sha256",)
    for field in report_hash_fields:
        if report.get(field) != payload.get(field):
            raise ActionContractFileError(
                f"bound diagnostic report differs on {field}"
            )
    expected_source_artifacts = {
        name: value
        for name, value in payload["artifacts"].items()
        if name != "diagnostic_report"
    }
    if report.get("source_artifacts") != expected_source_artifacts:
        raise ActionContractFileError(
            "bound diagnostic report source artifact bindings differ"
        )

    expected = build_formal_contract_claims_from_report(report)
    if payload.get("cohorts") != expected["cohorts"]:
        raise ActionContractFileError(
            "contract diagnostic-only cohorts differ from diagnostic report"
        )
    if payload.get("groups") != expected["groups"]:
        raise ActionContractFileError(
            "contract group claims differ from diagnostic report"
        )

    exact = validate_exact_droid_cache_index(
        artifacts["droid_cache_index"],
        index_payload=(
            artifact_payloads["droid_cache_index"]
            if artifact_payloads is not None
            else None
        ),
    )
    cohort_claim = expected["cohorts"][FORMAL_OXE_COHORT_ID]
    _validate_pooled_cohort_claim(cohort_claim)
    droid_claim = expected["groups"][FORMAL_DROID_CONTRACT_KEY]
    _validate_droid_exact_claim(
        droid_claim,
        exact,
        cache_index_sha256=payload["droid_cache_index_sha256"],
    )
    offsets: dict[str, int] = {}
    for key in FORMAL_OXE_CONTRACT_KEYS:
        group = expected["groups"][key]
        if (
            group.get("status") != "passed"
            or group.get("method") != FORMAL_OXE_COHORT_METHOD
            or group.get("cohort_id") != FORMAL_OXE_COHORT_ID
            or group.get("cohort_sha256") != cohort_claim["cohort_sha256"]
            or int(group.get("offset", 99)) != -2
        ):
            raise ActionContractFileError(
                f"pooled OXE member claim is inconsistent: {key}"
            )
        offsets[key] = -2
    offsets[FORMAL_DROID_CONTRACT_KEY] = int(droid_claim["offset"])
    return offsets
def load_passed_contracts(
    path: str | Path,
    *,
    expected_droid_cache_index: Path | None = None,
    expected_droid_cache_sha256: str | None = None,
) -> dict[str, int]:
    contract_path = Path(path)
    try:
        payload = json.loads(
            _read_stable_regular_file(contract_path, "action contract")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ActionContractFileError(
            f"cannot read action contract {contract_path}: {exc}"
        ) from exc
    schema_version = payload.get("schema_version")
    if schema_version != ACTION_CONTRACT_SCHEMA:
        raise ActionContractFileError(
            f"unexpected action contract schema: {schema_version!r}"
        )
    if payload.get("statistical_gate_schema") != ACTION_GATE_SCHEMA:
        raise ActionContractFileError(
            "action contract does not bind the formal v2 statistical gate"
        )
    if payload.get("evaluation_mode") != "formal":
        raise ActionContractFileError(
            "action contract was not produced in formal evaluation mode"
        )
    resample_counts = payload.get("resample_counts")
    required_resamples = {
        "qualification_permutation",
        "qualification_bootstrap",
        "confirmation_sign_flip",
    }
    if (
        not isinstance(resample_counts, dict)
        or set(resample_counts) != required_resamples
        or any(int(resample_counts[name]) < 10000 for name in required_resamples)
    ):
        raise ActionContractFileError(
            "formal action contract resample counts must all be at least 10000"
        )
    if payload.get("evidence_settings") != {
        "target_length": 16,
        "null_repeats": 256,
        "block_size": 2,
    }:
        raise ActionContractFileError(
            "formal action contract evidence settings must be "
            "target_length=16, null_repeats=256, block_size=2"
        )
    groups = payload.get("groups")
    if not isinstance(groups, dict):
        raise ActionContractFileError("action contract groups must be a mapping")
    has_droid = any(
        str(key).split("|", 1)[0] == "droid" for key in groups
    )
    required_artifacts = set(_ARTIFACT_HASH_FIELDS)
    if has_droid:
        required_artifacts.add("droid_cache_index")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != required_artifacts:
        raise ActionContractFileError(
            "formal action contract artifact bundle is incomplete"
        )
    for field in _HASH_FIELDS:
        _require_sha256(payload.get(field), field)
    if payload.get("formula_registry_sha256") != _sha256(
        _CANONICAL_FORMULA_REGISTRY
    ):
        raise ActionContractFileError(
            "contract does not bind the canonical formula registry"
        )
    if payload.get("gate_config_sha256") != _sha256(_CANONICAL_GATE_CONFIG):
        raise ActionContractFileError(
            "contract does not bind the canonical action gate config"
        )
    if payload.get("source_digest") != payload.get("evidence_sha256"):
        raise ActionContractFileError(
            "action contract source_digest must equal evidence_sha256"
        )
    resolved_artifacts, artifact_payloads = _validate_bound_artifacts(
        contract_path,
        payload,
        has_droid=has_droid,
        expected_droid_cache_index=expected_droid_cache_index,
        expected_droid_cache_sha256=expected_droid_cache_sha256,
    )
    return _validate_v3_report_and_claims(
        payload,
        resolved_artifacts,
        artifact_payloads,
    )


def validate_manifest_contract_coverage(
    records: Sequence[OXEClipRecord],
    passed_offsets: Mapping[str, int],
) -> None:
    required = {action_contract_key(record) for record in records}
    missing = sorted(required.difference(passed_offsets))
    if missing:
        raise ActionContractCoverageError(
            "training manifest has action contract groups without a passed "
            f"offset: {missing}"
        )


def _formal_source_provenance(
    report: Mapping[str, Any],
    report_group: Mapping[str, Any],
    key: str,
) -> dict[str, str]:
    provenance = report_group.get("source_provenance")
    expected_fields = {
        "source_manifest_sha256",
        "evidence_metadata_sha256",
        "projection_artifact_sha256",
        "cache_ledger_sha256",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_fields:
        raise ActionContractFileError(
            f"formal exact source provenance is missing: {key}"
        )
    for field, value in provenance.items():
        _require_sha256(value, f"{key} source provenance {field}")
    for field in (
        "source_manifest_sha256",
        "evidence_metadata_sha256",
        "projection_artifact_sha256",
    ):
        if provenance[field] != report.get(field):
            raise ActionContractFileError(
                f"formal source provenance differs on {field}: {key}"
            )
    return dict(provenance)


def _qualification_claim_from_gate_result(
    qualification: Mapping[str, Any],
    families: Sequence[str],
) -> dict[str, Any]:
    return {
        "dz_by_challenger": qualification.get("dz_by_challenger"),
        "max_t_p_by_challenger": qualification.get(
            "max_t_p_by_challenger"
        ),
        "bootstrap_win_frequency": qualification.get(
            "bootstrap_win_frequency"
        ),
        "informative_clip_count_by_family": {
            family: int(
                qualification.get("informative_clip_count_by_family", {}).get(
                    family, 0
                )
            )
            for family in families
        },
        "family_best_by_family": {
            family: qualification.get("family_best_by_family", {}).get(family)
            for family in families
        },
        "family_dz_by_challenger": {
            family: qualification.get("family_dz_by_challenger", {}).get(family)
            for family in families
        },
        "family_max_t_p_by_challenger": {
            family: qualification.get("family_max_t_p_by_challenger", {}).get(
                family
            )
            for family in families
        },
        "family_bootstrap_win_frequency": {
            family: qualification.get(
                "family_bootstrap_win_frequency", {}
            ).get(family)
            for family in families
        },
    }


def _confirmation_claim_from_gate_result(
    confirmation: Mapping[str, Any],
    families: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    by_family = confirmation.get("by_family")
    if not isinstance(by_family, dict) or not set(families).issubset(by_family):
        raise ActionContractFileError(
            f"{label} confirmation families are incomplete"
        )
    return {
        family: {
            "clip_count": int(by_family[family].get("clip_count", 0)),
            "dz_by_challenger": by_family[family].get("dz_by_challenger"),
            "holm_p_by_challenger": by_family[family].get(
                "holm_p_by_challenger"
            ),
        }
        for family in families
    }


def _falsification_claim_from_gate_result(
    partition: Mapping[str, Any],
    families: Sequence[str],
    *,
    label: str,
) -> dict[str, Any]:
    by_family = partition.get("by_family")
    if not isinstance(by_family, dict) or set(by_family) != set(families):
        raise ActionContractFileError(
            f"{label} falsification families are incomplete"
        )
    clip_ids = tuple(str(value) for value in partition.get("clip_ids", ()))
    return {
        "validation_mode": "expected_offset_falsification_v1",
        "expected_offset": int(partition.get("expected_offset", 99)),
        "clip_count": len(set(clip_ids)),
        "by_family": {
            family: {
                "best_offset": int(by_family[family].get("best_offset", 99)),
                "clip_count": int(by_family[family].get("clip_count", 0)),
                "mean_score_by_offset": by_family[family].get(
                    "mean_score_by_offset"
                ),
                "challenger_over_expected_dz": by_family[family].get(
                    "challenger_over_expected_dz"
                ),
                "raw_p_by_challenger": by_family[family].get(
                    "raw_p_by_challenger"
                ),
                "holm_p_by_challenger": by_family[family].get(
                    "holm_p_by_challenger"
                ),
                "conflicting_challengers": by_family[family].get(
                    "conflicting_challengers"
                ),
            }
            for family in families
        },
    }


def _validate_expected_offset_falsification(
    key: str,
    partition: object,
    *,
    expected_offset: int,
    families: Sequence[str],
    partition_clip_count: int,
    min_informative: int,
    alpha: float = 0.01,
    min_dz: float = 0.30,
) -> None:
    if not isinstance(partition, dict) or (
        partition.get("validation_mode")
        != "expected_offset_falsification_v1"
        or int(partition.get("expected_offset", 99)) != expected_offset
        or int(partition.get("clip_count", 0)) != partition_clip_count
    ):
        raise ActionContractFileError(
            f"expected-offset falsification header is invalid for {key}"
        )
    by_family = partition.get("by_family")
    if not isinstance(by_family, dict) or set(by_family) != set(families):
        raise ActionContractFileError(
            f"expected-offset falsification families are incomplete for {key}"
        )
    challengers = {
        str(candidate)
        for candidate in _ALLOWED_OFFSETS
        if candidate != expected_offset
    }
    all_offsets = {str(candidate) for candidate in _ALLOWED_OFFSETS}
    for family in families:
        report = by_family[family]
        if (
            not isinstance(report, dict)
            or int(report.get("best_offset", 99)) not in _ALLOWED_OFFSETS
            or int(report.get("clip_count", 0)) < min_informative
            or int(report.get("clip_count", 0)) > partition_clip_count
            or set(report.get("mean_score_by_offset", {})) != all_offsets
        ):
            raise ActionContractFileError(
                f"expected-offset falsification evidence is incomplete for "
                f"{key}/{family}"
            )
        means = {
            int(offset): float(report["mean_score_by_offset"][offset])
            for offset in all_offsets
        }
        if any(not math.isfinite(value) for value in means.values()):
            raise ActionContractFileError(
                f"expected-offset means contain non-finite values for "
                f"{key}/{family}"
            )
        computed_best = sorted(
            means,
            key=lambda offset: (-means[offset], offset),
        )[0]
        if int(report["best_offset"]) != computed_best:
            raise ActionContractFileError(
                f"expected-offset best is inconsistent for {key}/{family}"
            )
        effects = report.get("challenger_over_expected_dz")
        raw_p = report.get("raw_p_by_challenger")
        holm_p = report.get("holm_p_by_challenger")
        for value, label in (
            (effects, "dz"),
            (raw_p, "raw p"),
            (holm_p, "Holm p"),
        ):
            if not isinstance(value, dict) or set(value) != challengers:
                raise ActionContractFileError(
                    f"expected-offset {label} must report all four "
                    f"challengers for {key}/{family}"
                )
        effect_values = {offset: float(effects[offset]) for offset in challengers}
        raw_values = {offset: float(raw_p[offset]) for offset in challengers}
        holm_values = {offset: float(holm_p[offset]) for offset in challengers}
        if any(not math.isfinite(value) for value in effect_values.values()):
            raise ActionContractFileError(
                f"expected-offset dz contains non-finite values for {key}/{family}"
            )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (*raw_values.values(), *holm_values.values())
        ) or any(
            holm_values[offset] + 1e-12 < raw_values[offset]
            for offset in challengers
        ):
            raise ActionContractFileError(
                f"expected-offset p-values are invalid for {key}/{family}"
            )
        ordered = sorted(
            raw_values,
            key=lambda offset: (raw_values[offset], int(offset)),
        )
        recomputed_holm: dict[str, float] = {}
        running = 0.0
        for rank, offset in enumerate(ordered):
            candidate = min(
                1.0,
                raw_values[offset] * (len(ordered) - rank),
            )
            running = max(running, candidate)
            recomputed_holm[offset] = running
        if any(
            not math.isclose(
                holm_values[offset],
                recomputed_holm[offset],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for offset in challengers
        ):
            raise ActionContractFileError(
                f"expected-offset Holm correction is inconsistent for "
                f"{key}/{family}"
            )
        derived_conflicts = sorted(
            int(offset)
            for offset in challengers
            if effect_values[offset] >= min_dz and holm_values[offset] < alpha
        )
        reported_conflicts = sorted(
            int(value) for value in report.get("conflicting_challengers", ())
        )
        if reported_conflicts != derived_conflicts:
            raise ActionContractFileError(
                f"expected-offset conflicts are inconsistent for {key}/{family}"
            )
        if derived_conflicts:
            raise ActionContractFileError(
                f"expected offset {expected_offset} is falsified for "
                f"{key}/{family}: {derived_conflicts}"
            )


def _independent_oxe_claim_from_report(
    report: Mapping[str, Any],
    report_group: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    if (
        report_group.get("status") != "passed"
        or report_group.get("method") != FORMAL_OXE_METHOD
    ):
        raise ActionContractFileError(
            f"formal OXE group is not independently passed: {key}"
        )
    partition, counts = _partition_from_report(report_group, key)
    if counts != {"qualification": 32, "confirmation": 32}:
        raise ActionContractFileError(f"formal OXE split counts are wrong: {key}")
    provenance = _formal_source_provenance(report, report_group, key)
    result = report_group.get("gate_result")
    if not isinstance(result, dict):
        raise ActionContractFileError(f"formal OXE gate result is missing: {key}")
    qualification = result.get("qualification")
    confirmation = result.get("confirmation")
    if not isinstance(qualification, dict) or not isinstance(confirmation, dict):
        raise ActionContractFileError(
            f"formal OXE qualification/confirmation is missing: {key}"
        )
    eligible = tuple(result.get("eligible_families", ()))
    binding = tuple(result.get("binding_families", ()))
    expected_binding = tuple(
        ["state"]
        + [family for family in ("flow", "geometry") if family in eligible]
    )
    diagnostic = set(result.get("diagnostic_families", ()))
    selected_offset = int(result.get("selected_offset", 99))
    if (
        selected_offset not in _ALLOWED_OFFSETS
        or int(result.get("clip_count", 0)) != 64
        or result.get("split_artifact_sha256")
        != report.get("split_artifact_sha256")
        or result.get("split_partition_sha256") != partition
        or int(qualification.get("selected_offset", 99)) != selected_offset
        or int(confirmation.get("tested_offset", 99)) != selected_offset
        or len(set(qualification.get("clip_ids", ()))) != 32
        or len(set(confirmation.get("clip_ids", ()))) != 32
        or binding != expected_binding
        or len(binding) < 2
        or diagnostic.intersection(binding)
        or (
            {"flow", "geometry"}.issubset(binding)
            and result.get("flow_geometry_agree") is not True
        )
    ):
        raise ActionContractFileError(
            f"formal OXE independent result is invalid: {key}"
        )
    required = tuple(result.get("required_families", ()))
    source_classes = tuple(result.get("required_source_classes", ()))
    qualification_claim = _qualification_claim_from_gate_result(
        qualification,
        binding,
    )
    confirmation_claim = _confirmation_claim_from_gate_result(
        confirmation,
        binding,
        label=f"formal OXE {key}",
    )
    claim = {
        "status": "passed",
        "method": FORMAL_OXE_METHOD,
        "offset": selected_offset,
        "clip_count": 64,
        "frozen_clip_counts": counts,
        "required_families": list(required),
        "required_source_classes": list(source_classes),
        "eligible_families": list(eligible),
        "binding_families": list(binding),
        "geometry_policy": "all_qualification_eligible_visual_families_are_binding",
        "split_partition_sha256": partition,
        "source_provenance": provenance,
        "qualification": qualification_claim,
        "confirmation": confirmation_claim,
    }
    _validate_passed_group(key, claim)
    return claim


def _independent_droid_claim_from_report(
    report: Mapping[str, Any],
    report_group: Mapping[str, Any],
) -> dict[str, Any]:
    key = FORMAL_DROID_CONTRACT_KEY
    if (
        report_group.get("status") != "passed"
        or report_group.get("method") != FORMAL_DROID_METHOD
    ):
        raise ActionContractFileError("formal DROID group is not passed")
    partition, counts = _partition_from_report(report_group, key)
    provenance = _formal_source_provenance(report, report_group, key)
    result = report_group.get("gate_result")
    exact = report_group.get("exact_construction")
    falsification = result.get("falsification") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or int(result.get("selected_offset", 99)) != -1
        or int(result.get("clip_count", 0)) != 64
        or result.get("split_artifact_sha256")
        != report.get("split_artifact_sha256")
        or result.get("split_partition_sha256") != partition
        or tuple(result.get("required_families", ())) != ("state", "flow")
        or tuple(result.get("required_source_classes", ()))
        != ("proprioceptive", "exteroceptive")
        or result.get("separation_kind")
        != "non_statistical_exact_construction"
        or result.get("separation_basis")
        != "exact_n_minus_one_interval_action_construction"
        or result.get("statistical_separation_claimed") is not False
        or result.get("statistics_role") != "expected_offset_falsification"
        or result.get("exact_construction") != exact
        or not isinstance(falsification, dict)
        or int(falsification.get("selected_offset", 99)) != -1
        or falsification.get("split_partition_sha256") != partition
        or not isinstance(exact, dict)
    ):
        raise ActionContractFileError(
            "DROID exact-derived diagnostic result is invalid"
        )
    qualification = falsification.get("qualification")
    confirmation = falsification.get("confirmation")
    if not isinstance(qualification, dict) or not isinstance(
        confirmation, dict
    ):
        raise ActionContractFileError("DROID falsification reports are missing")
    frozen = report_group["frozen_split"]
    qualification_clip_ids = tuple(frozen["qualification_clip_ids"])
    confirmation_clip_ids = tuple(frozen["confirmation_clip_ids"])
    _require_exact_partition_clip_ids(
        qualification,
        qualification_clip_ids,
        label="DROID qualification",
    )
    _require_exact_partition_clip_ids(
        confirmation,
        confirmation_clip_ids,
        label="DROID confirmation",
    )
    qualification_claim = _falsification_claim_from_gate_result(
        qualification,
        ("state", "flow"),
        label="formal DROID qualification",
    )
    confirmation_claim = _falsification_claim_from_gate_result(
        confirmation,
        ("state", "flow"),
        label="formal DROID confirmation",
    )
    _validate_expected_offset_falsification(
        f"{key}/qualification",
        qualification_claim,
        expected_offset=-1,
        families=("state", "flow"),
        partition_clip_count=32,
        min_informative=24,
    )
    _validate_expected_offset_falsification(
        f"{key}/confirmation",
        confirmation_claim,
        expected_offset=-1,
        families=("state", "flow"),
        partition_clip_count=32,
        min_informative=24,
    )
    claim: dict[str, Any] = {
        "status": "passed",
        "method": FORMAL_DROID_METHOD,
        "offset": -1,
        "frozen_clip_counts": counts,
        "required_families": ["state", "flow"],
        "required_source_classes": ["proprioceptive", "exteroceptive"],
        "split_partition_sha256": partition,
        "source_provenance": provenance,
        "exact_construction": exact,
        "falsification": {
            "qualification": qualification_claim,
            "confirmation": confirmation_claim,
        },
        "separation": {
            "kind": "non_statistical_exact_construction",
            "basis": "exact_n_minus_one_interval_action_construction",
            "statistical_separation_claimed": False,
            "statistics_role": "expected_offset_falsification",
        },
    }
    return claim


def _validate_pooled_cohort_claim(claim: Mapping[str, Any]) -> None:
    if (
        claim.get("status") != "passed"
        or claim.get("cohort_id") != FORMAL_OXE_COHORT_ID
        or claim.get("method") != FORMAL_OXE_COHORT_METHOD
        or int(claim.get("offset", 99)) != -2
        or tuple(claim.get("members", ())) != FORMAL_OXE_CONTRACT_KEYS
        or claim.get("frozen_clip_counts")
        != {"qualification": 160, "confirmation": 160}
        or tuple(claim.get("required_families", ())) != ("state", "flow")
        or tuple(claim.get("binding_families", ())) != ("state", "flow")
        or tuple(claim.get("required_source_classes", ()))
        != ("proprioceptive", "exteroceptive")
        or claim.get("geometry_policy")
        != "diagnostic_only_never_binding"
    ):
        raise ActionContractFileError("pooled OXE cohort claim is invalid")
    for partition in ("qualification", "confirmation"):
        _validate_expected_offset_falsification(
            f"{FORMAL_OXE_COHORT_ID}/{partition}",
            claim.get(partition),
            expected_offset=-2,
            families=("state", "flow"),
            partition_clip_count=160,
            min_informative=120,
        )
    member_falsification = claim.get("member_falsification")
    if not isinstance(member_falsification, dict) or set(member_falsification) != set(
        FORMAL_OXE_CONTRACT_KEYS
    ):
        raise ActionContractFileError("pooled OXE member falsification is incomplete")
    for key in FORMAL_OXE_CONTRACT_KEYS:
        member_claim = member_falsification[key]
        if (
            not isinstance(member_claim, dict)
            or tuple(member_claim.get("member_contract_keys", ())) != (key,)
            or int(member_claim.get("offset", 99)) != -2
            or member_claim.get("frozen_clip_counts")
            != {"qualification": 32, "confirmation": 32}
        ):
            raise ActionContractFileError(
                f"pooled OXE member falsification is invalid: {key}"
            )
        for partition in ("qualification", "confirmation"):
            _validate_expected_offset_falsification(
                f"{key}/member_falsification/{partition}",
                member_claim.get(partition),
                expected_offset=-2,
                families=("state", "flow"),
                partition_clip_count=32,
                min_informative=24,
            )
    expected_sha = _canonical_sha256(
        {key: value for key, value in claim.items() if key != "cohort_sha256"}
    )
    if claim.get("cohort_sha256") != expected_sha:
        raise ActionContractFileError("pooled OXE cohort SHA256 is invalid")


def _formal_claims_from_report(
    report: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    report_groups = report.get("groups")
    expected_keys = set(FORMAL_OXE_CONTRACT_KEYS) | {FORMAL_DROID_CONTRACT_KEY}
    if not isinstance(report_groups, dict) or set(report_groups) != expected_keys:
        raise ActionContractFileError(
            "formal diagnostic report does not cover exactly six runtime keys"
        )
    report_cohorts = report.get("cohorts")
    if not isinstance(report_cohorts, dict) or set(report_cohorts) != {
        FORMAL_OXE_COHORT_ID
    }:
        raise ActionContractFileError("formal report must contain one pooled OXE cohort")
    cohort_report = report_cohorts[FORMAL_OXE_COHORT_ID]
    result = cohort_report.get("gate_result") if isinstance(cohort_report, dict) else None
    if (
        not isinstance(result, dict)
        or cohort_report.get("method") != FORMAL_OXE_COHORT_METHOD
        or tuple(cohort_report.get("members", ())) != FORMAL_OXE_CONTRACT_KEYS
        or int(result.get("selected_offset", 99)) != -2
        or int(result.get("clip_count", 0)) != 320
        or int(result.get("frozen_qualification_clip_count", 0)) != 160
        or int(result.get("frozen_confirmation_clip_count", 0)) != 160
    ):
        raise ActionContractFileError("formal pooled OXE gate result is invalid")
    families = tuple(result.get("binding_families", ()))
    qualification = result.get("qualification")
    confirmation = result.get("confirmation")
    if not isinstance(qualification, dict) or not isinstance(confirmation, dict):
        raise ActionContractFileError("pooled OXE gate statistics are missing")

    member_partitions: dict[str, str] = {}
    member_counts: dict[str, dict[str, int]] = {}
    member_provenance: dict[str, dict[str, str]] = {}
    member_falsification_claims: dict[str, dict[str, Any]] = {}
    member_falsification = result.get("member_falsification")
    if not isinstance(member_falsification, dict) or set(member_falsification) != set(
        FORMAL_OXE_CONTRACT_KEYS
    ):
        raise ActionContractFileError("formal pooled OXE member guards are missing")
    for key in FORMAL_OXE_CONTRACT_KEYS:
        group = report_groups[key]
        if (
            not isinstance(group, dict)
            or group.get("method") != FORMAL_OXE_COHORT_METHOD
            or group.get("cohort_id") != FORMAL_OXE_COHORT_ID
        ):
            raise ActionContractFileError(f"pooled OXE member report is invalid: {key}")
        partition, counts = _partition_from_report(group, key)
        member_partitions[key] = partition
        member_counts[key] = counts
        member_provenance[key] = _formal_source_provenance(report, group, key)
        guard = member_falsification[key]
        if (
            not isinstance(guard, dict)
            or int(guard.get("selected_offset", 99)) != -2
            or tuple(guard.get("member_contract_keys", ())) != (key,)
            or int(guard.get("frozen_qualification_clip_count", 0)) != 32
            or int(guard.get("frozen_confirmation_clip_count", 0)) != 32
            or tuple(guard.get("binding_families", ())) != ("state", "flow")
        ):
            raise ActionContractFileError(
                f"formal pooled OXE member guard is invalid: {key}"
            )
        guard_qualification = guard.get("qualification")
        guard_confirmation = guard.get("confirmation")
        if not isinstance(guard_qualification, dict) or not isinstance(
            guard_confirmation, dict
        ):
            raise ActionContractFileError(
                f"formal pooled OXE member guard reports are missing: {key}"
            )
        frozen = group["frozen_split"]
        _require_exact_partition_clip_ids(
            guard_qualification,
            tuple(frozen["qualification_clip_ids"]),
            label=f"formal pooled OXE member qualification {key}",
        )
        _require_exact_partition_clip_ids(
            guard_confirmation,
            tuple(frozen["confirmation_clip_ids"]),
            label=f"formal pooled OXE member confirmation {key}",
        )
        member_claim = {
            "status": "passed",
            "offset": -2,
            "clip_count": 64,
            "member_contract_keys": [key],
            "frozen_clip_counts": {"qualification": 32, "confirmation": 32},
            "required_families": ["state", "flow"],
            "binding_families": ["state", "flow"],
            "required_source_classes": ["proprioceptive", "exteroceptive"],
            "split_partition_sha256": member_partitions[key],
            "qualification": _falsification_claim_from_gate_result(
                guard_qualification,
                ("state", "flow"),
                label=f"formal pooled OXE member qualification {key}",
            ),
            "confirmation": _falsification_claim_from_gate_result(
                guard_confirmation,
                ("state", "flow"),
                label=f"formal pooled OXE member confirmation {key}",
            ),
        }
        for partition in ("qualification", "confirmation"):
            _validate_expected_offset_falsification(
                f"{key}/member_falsification/{partition}",
                member_claim[partition],
                expected_offset=-2,
                families=("state", "flow"),
                partition_clip_count=32,
                min_informative=24,
            )
        member_falsification_claims[key] = member_claim
    if result.get("member_split_partition_sha256") != member_partitions:
        raise ActionContractFileError("pooled OXE member partitions differ from split")
    if cohort_report.get("member_source_provenance") != member_provenance:
        raise ActionContractFileError("pooled OXE member provenance is inconsistent")
    pooled_qualification_clip_ids = tuple(
        f"{key}\0{clip_id}"
        for key in FORMAL_OXE_CONTRACT_KEYS
        for clip_id in report_groups[key]["frozen_split"][
            "qualification_clip_ids"
        ]
    )
    pooled_confirmation_clip_ids = tuple(
        f"{key}\0{clip_id}"
        for key in FORMAL_OXE_CONTRACT_KEYS
        for clip_id in report_groups[key]["frozen_split"][
            "confirmation_clip_ids"
        ]
    )
    _require_exact_partition_clip_ids(
        qualification,
        pooled_qualification_clip_ids,
        label="formal pooled OXE qualification",
    )
    _require_exact_partition_clip_ids(
        confirmation,
        pooled_confirmation_clip_ids,
        label="formal pooled OXE confirmation",
    )

    cohort_claim: dict[str, Any] = {
        "status": "passed",
        "cohort_id": FORMAL_OXE_COHORT_ID,
        "method": FORMAL_OXE_COHORT_METHOD,
        "members": list(FORMAL_OXE_CONTRACT_KEYS),
        "offset": -2,
        "required_families": list(result.get("required_families", ())),
        "required_source_classes": list(result.get("required_source_classes", ())),
        "binding_families": list(families),
        "geometry_policy": result.get("geometry_policy"),
        "frozen_clip_counts": {"qualification": 160, "confirmation": 160},
        "member_frozen_clip_counts": member_counts,
        "member_split_partition_sha256": member_partitions,
        "member_source_provenance": member_provenance,
        "member_falsification": member_falsification_claims,
        "split_partition_sha256": result.get("split_partition_sha256"),
        "qualification": _falsification_claim_from_gate_result(
            qualification,
            families,
            label="formal pooled OXE qualification",
        ),
        "confirmation": _falsification_claim_from_gate_result(
            confirmation,
            families,
            label="formal pooled OXE confirmation",
        ),
    }
    cohort_claim["cohort_sha256"] = _canonical_sha256(cohort_claim)
    _validate_pooled_cohort_claim(cohort_claim)

    groups: dict[str, dict[str, Any]] = {}
    for key in FORMAL_OXE_CONTRACT_KEYS:
        groups[key] = {
            "status": "passed",
            "method": FORMAL_OXE_COHORT_METHOD,
            "cohort_id": FORMAL_OXE_COHORT_ID,
            "cohort_sha256": cohort_claim["cohort_sha256"],
            "offset": -2,
            "member_frozen_clip_counts": member_counts[key],
            "member_split_partition_sha256": member_partitions[key],
            "source_provenance": member_provenance[key],
        }
    groups[FORMAL_DROID_CONTRACT_KEY] = _independent_droid_claim_from_report(
        report,
        report_groups[FORMAL_DROID_CONTRACT_KEY],
    )
    return {FORMAL_OXE_COHORT_ID: cohort_claim}, groups
