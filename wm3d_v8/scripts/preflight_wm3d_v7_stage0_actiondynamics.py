#!/usr/bin/env python3
"""Fail-closed preflight for corrected WM3D-v7 S0 action dynamics.

``--mode static`` validates the immutable configuration contract and reports
unpinned runtime artifacts as blockers.  The default ``--mode full`` verifies
every file digest, canonical-action audit, source-specific train statistics,
train/val separation and fresh output directory.  Only a full PASS authorizes
the 200-step canary; this script never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Fail closed against the older sibling checkout at
# /data/Minko/world_model/wm3d_v3.  This script must validate the exact runtime
# implementation that the v7 launcher will execute, even under a polluted
# inherited PYTHONPATH.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from wm3d_v3.training import train as _runtime_train  # noqa: E402
from wm3d_v3.data import canonical_action as _runtime_canonical_action  # noqa: E402

RUNTIME_TRAIN_MODULE_PATH = Path(_runtime_train.__file__).resolve()
try:
    RUNTIME_TRAIN_MODULE_PATH.relative_to(PROJECT_ROOT)
except ValueError as exc:  # pragma: no cover - exercised by subprocess test
    raise ImportError(
        "preflight imported wm3d_v3.training.train outside wm3d_v7: "
        f"{RUNTIME_TRAIN_MODULE_PATH}"
    ) from exc
normalize_action_grip_contract = _runtime_train.normalize_action_grip_contract
RUNTIME_CANONICAL_ACTION_MODULE_PATH = Path(_runtime_canonical_action.__file__).resolve()
try:
    RUNTIME_CANONICAL_ACTION_MODULE_PATH.relative_to(PROJECT_ROOT)
except ValueError as exc:  # pragma: no cover - exercised by deployment audit
    raise ImportError(
        "preflight imported wm3d_v3.data.canonical_action outside wm3d_v7: "
        f"{RUNTIME_CANONICAL_ACTION_MODULE_PATH}"
    ) from exc


CANONICAL_VERSION = "wm3d_v7_base_delta_axisangle_gripclose_v1"
CANONICAL_LAYOUT = [
    "dx_m",
    "dy_m",
    "dz_m",
    "drot_x_rad",
    "drot_y_rad",
    "drot_z_rad",
    "gripper_close_signed",
]
EXPECTED_SOURCES = {
    "oxe_droid_action": 35,
    "oxe_bridge_action": 15,
    "robocasa_atomic": 10,
    "robocasa_composite": 20,
    "robocasa_mg": 20,
}
EXPECTED_OXE_DATASETS = {
    "oxe_droid_action": "droid",
    "oxe_bridge_action": "bridge",
}
EXPECTED_OXE_ACTION_KINDS = {
    "oxe_droid_action": "cartesian_target_interval_delta+rpy+gripper",
    "oxe_bridge_action": "delta_xyz+rpy+gripper",
}
EXPECTED_OXE_OFFSETS = {"oxe_droid_action": -1, "oxe_bridge_action": -2}
EXPECTED_ROTATION_CONVERSIONS = {
    "oxe_droid_action": "wrapped_rpy_interval_delta_to_so3_to_axisangle",
    "oxe_bridge_action": "fixed_extrinsic_xyz_rz_ry_rx_to_axisangle",
}
FORBIDDEN_DATASETS = {"fractal20220817_data", "taco_play", "jaco_play", "kuka"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PreflightError(RuntimeError):
    """Raised after all fail-closed violations have been collected."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    chain = set(seen or ())
    if path in chain:
        raise ValueError(f"cyclic config inheritance: {path}")
    chain.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _deep_merge(load_config(base_path, chain), payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        raise KeyError(key)
    value = archive[key]
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape={value.shape}")
    return value.reshape(()).item()


class _Checks:
    def __init__(self, mode: str):
        self.mode = mode
        self.errors: list[str] = []
        self.blockers: list[str] = []
        self.verified_artifacts: dict[str, str] = {}

    def expect(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def equal(self, observed: Any, expected: Any, label: str) -> None:
        if observed != expected:
            self.errors.append(f"{label}: observed={observed!r}, expected={expected!r}")

    def approx(self, observed: Any, expected: float, label: str) -> None:
        try:
            passed = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            passed = False
        if not passed:
            self.errors.append(f"{label}: observed={observed!r}, expected={expected!r}")

    def pinned_file(self, path_value: Any, digest_value: Any, label: str) -> Path | None:
        path = Path(str(path_value)) if path_value else None
        digest = str(digest_value or "")
        if digest.startswith("PENDING_"):
            message = f"{label}: unresolved digest {digest}"
            if self.mode == "full":
                self.errors.append(message)
            else:
                self.blockers.append(message)
            return None
        if not HEX64.fullmatch(digest):
            self.errors.append(f"{label}: expected a lowercase SHA256 pin, got {digest!r}")
            return None
        if path is None:
            self.errors.append(f"{label}: missing path")
            return None
        if self.mode != "full":
            return path
        if not path.is_file():
            self.errors.append(f"{label}: missing file {path}")
            return None
        observed = sha256_file(path)
        if observed != digest:
            self.errors.append(
                f"{label}: digest mismatch observed={observed} expected={digest} path={path}"
            )
            return None
        self.verified_artifacts[label] = observed
        return path


def _validate_canonical_gate(
    checks: _Checks,
    source: dict[str, Any],
    gate_path: Path | None,
) -> dict[str, Any] | None:
    if checks.mode != "full" or gate_path is None:
        return None
    label = source["source_name"]
    try:
        gate = json.loads(gate_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"{label}: unreadable action audit gate: {exc}")
        return None
    checks.equal(gate.get("schema"), "wm3d_v7_canonical_action_audit_gate_v1", f"{label}.gate.schema")
    checks.equal(gate.get("status"), "passed", f"{label}.gate.status")
    checks.equal(gate.get("source_name"), label, f"{label}.gate.source_name")
    checks.equal(gate.get("dataset"), EXPECTED_OXE_DATASETS[label], f"{label}.gate.dataset")
    canonical = gate.get("canonical") or {}
    checks.equal(canonical.get("version"), CANONICAL_VERSION, f"{label}.gate.canonical.version")
    checks.equal(canonical.get("layout"), CANONICAL_LAYOUT, f"{label}.gate.canonical.layout")
    checks.equal(canonical.get("source_frame"), "base", f"{label}.gate.canonical.source_frame")
    checks.equal(canonical.get("translation_unit"), "meter", f"{label}.gate.canonical.translation_unit")
    checks.equal(canonical.get("rotation_representation"), "axis_angle", f"{label}.gate.canonical.rotation")
    checks.equal(canonical.get("gripper_semantics"), "signed_close_positive_continuous", f"{label}.gate.canonical.gripper")
    required_boolean_checks = (
        "finite",
        "direction_passed",
        "scale_passed",
        "rotation_conversion_passed",
        "gripper_mapping_passed",
        "gripper_signed_close01_consistent",
        "source_frame_passed",
        "temporal_alignment_passed",
        "train_val_disjoint",
        "payload_digests_verified",
    )
    gate_checks = gate.get("checks") or {}
    for key in required_boolean_checks:
        checks.equal(gate_checks.get(key), True, f"{label}.gate.checks.{key}")
    checks.equal(
        gate.get("source_manifest_sha256"),
        source.get("manifest_sha256"),
        f"{label}.gate.source_manifest_sha256",
    )
    checks.equal(
        gate.get("canonical_action_cache_manifest_sha256"),
        source.get("canonical_action_cache_manifest_sha256"),
        f"{label}.gate.action_cache_manifest_sha256",
    )
    adapter = gate.get("adapter") or {}
    checks.equal(adapter.get("version"), CANONICAL_VERSION, f"{label}.gate.adapter.version")
    checks.expect(HEX64.fullmatch(str(adapter.get("implementation_sha256", ""))) is not None,
                  f"{label}.gate.adapter.implementation_sha256 is not pinned")
    return gate


def _validate_action_cache_split(
    checks: _Checks,
    source: dict[str, Any],
    manifest_path: Path | None,
) -> None:
    if checks.mode != "full" or manifest_path is None:
        return
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    label = source["source_name"]
    try:
        with manifest_path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("source_name") != label:
                    checks.errors.append(
                        f"{label}.action_cache row {line_number}: wrong source_name={row.get('source_name')!r}"
                    )
                    continue
                if row.get("dataset") != EXPECTED_OXE_DATASETS[label]:
                    checks.errors.append(
                        f"{label}.action_cache row {line_number}: wrong dataset={row.get('dataset')!r}"
                    )
                checks.equal(row.get("schema"), "wm3d_v7_canonical_action_cache_row_v1", f"{label}.action_cache row {line_number}.schema")
                checks.equal(row.get("action_kind"), EXPECTED_OXE_ACTION_KINDS[label], f"{label}.action_cache row {line_number}.action_kind")
                checks.equal(row.get("action_frame_offset"), EXPECTED_OXE_OFFSETS[label], f"{label}.action_cache row {line_number}.action_frame_offset")
                checks.equal(row.get("adapter_version"), CANONICAL_VERSION, f"{label}.action_cache row {line_number}.adapter_version")
                checks.equal(row.get("gripper_semantics"), "signed_close_positive_continuous", f"{label}.action_cache row {line_number}.gripper_semantics")
                checks.equal(row.get("action_contract_evidence_sha256"), source.get("action_contract_evidence_sha256"), f"{label}.action_cache row {line_number}.temporal_evidence")
                expected_actions = int(row.get("n_frames", 0) or 0) - (
                    1 if label == "oxe_droid_action" else 0
                )
                checks.expect(expected_actions > 0, f"{label}.action_cache row {line_number}: invalid n_frames")
                checks.equal(row.get("action_shape"), [expected_actions, 7], f"{label}.action_cache row {line_number}.action_shape")
                checks.equal(row.get("grip_close01_shape"), [expected_actions], f"{label}.action_cache row {line_number}.grip_close01_shape")
                checks.equal(row.get("action_dtype"), "float32", f"{label}.action_cache row {line_number}.action_dtype")
                checks.equal(row.get("grip_close01_dtype"), "float32", f"{label}.action_cache row {line_number}.grip_close01_dtype")
                for path_key, sha_key, payload_label in (
                    ("action_path", "action_sha256", "actions_signed"),
                    ("grip_close01_path", "grip_close01_sha256", "grip_close01"),
                ):
                    payload_path = Path(str(row.get(path_key, "")))
                    expected_digest = str(row.get(sha_key, ""))
                    if not payload_path.is_file():
                        checks.errors.append(
                            f"{label}.action_cache row {line_number}: missing "
                            f"{payload_label} payload {payload_path}"
                        )
                    elif not HEX64.fullmatch(expected_digest):
                        checks.errors.append(
                            f"{label}.action_cache row {line_number}: invalid "
                            f"{payload_label} digest"
                        )
                    elif sha256_file(payload_path) != expected_digest:
                        checks.errors.append(
                            f"{label}.action_cache row {line_number}: "
                            f"{payload_label} digest mismatch"
                        )
                split = row.get("split")
                clip_id = str(row.get("clip_id", ""))
                if not clip_id or split not in {"train", "val"}:
                    checks.errors.append(
                        f"{label}.action_cache row {line_number}: clip_id/split contract violation"
                    )
                elif split == "train":
                    train_ids.add(clip_id)
                else:
                    val_ids.add(clip_id)
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"{label}: unreadable canonical action manifest: {exc}")
        return
    checks.expect(bool(train_ids), f"{label}: canonical action manifest has no train clips")
    checks.expect(bool(val_ids), f"{label}: canonical action manifest has no val clips")
    overlap = train_ids & val_ids
    checks.expect(not overlap, f"{label}: train/val clip overlap ({len(overlap)} clips)")


def _validate_stats(
    checks: _Checks,
    source: dict[str, Any],
    stats_path: Path | None,
) -> None:
    if checks.mode != "full" or stats_path is None:
        return
    label = source["source_name"]
    try:
        with np.load(stats_path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float64)
            std = np.asarray(archive["std"], dtype=np.float64)
            checks.equal(mean.shape, (6,), f"{label}.stats.mean.shape")
            checks.equal(std.shape, (6,), f"{label}.stats.std.shape")
            checks.expect(bool(np.isfinite(mean).all()), f"{label}.stats.mean is non-finite")
            checks.expect(bool(np.isfinite(std).all()), f"{label}.stats.std is non-finite")
            checks.expect(bool((std > 1e-6).all()), f"{label}.stats.std has degenerate axes")
            checks.expect(int(_scalar(archive, "count")) > 0, f"{label}.stats.count is empty")
            checks.equal(_scalar(archive, "split"), "train", f"{label}.stats.split")
            checks.equal(_scalar(archive, "source_name"), label, f"{label}.stats.source_name")
            checks.equal(
                _scalar(archive, "dataset"),
                EXPECTED_OXE_DATASETS[label],
                f"{label}.stats.dataset",
            )
            checks.equal(
                _scalar(archive, "action_adapter_version"),
                CANONICAL_VERSION,
                f"{label}.stats.action_adapter_version",
            )
            checks.equal(
                _scalar(archive, "source_manifest_sha256"),
                source.get("manifest_sha256"),
                f"{label}.stats.source_manifest_sha256",
            )
            checks.equal(
                _scalar(archive, "action_cache_manifest_sha256"),
                source.get("canonical_action_cache_manifest_sha256"),
                f"{label}.stats.action_cache_manifest_sha256",
            )
            checks.equal(
                _scalar(archive, "action_audit_gate_sha256"),
                source.get("action_audit_gate_sha256"),
                f"{label}.stats.action_audit_gate_sha256",
            )
            checks.equal(
                _scalar(archive, "gripper_semantics"),
                "signed_close_positive_continuous",
                f"{label}.stats.gripper_semantics",
            )
            grip_rate = float(_scalar(archive, "grip_close_rate"))
            checks.expect(
                math.isfinite(grip_rate) and 0.0 < grip_rate < 1.0,
                f"{label}.stats.grip_close_rate must be strictly between zero and one",
            )
    except (OSError, KeyError, ValueError) as exc:
        checks.errors.append(f"{label}: invalid action stats: {exc}")


def _validate_robocasa_index(checks: _Checks, index_path: Path | None) -> None:
    if checks.mode != "full" or index_path is None:
        return
    allowed_partitions = {"atomic", "composite", "mg"}
    split_ids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    partition_split_counts = {
        partition: {"train": 0, "val": 0} for partition in allowed_partitions
    }
    try:
        with index_path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                clip_id = str(row.get("clip_hash", ""))
                split = str(row.get("split", ""))
                partition = str(row.get("v7_source", row.get("dataset", "")))
                checks.expect(bool(clip_id), f"robocasa index row {line_number}: missing clip_hash")
                checks.expect(split in split_ids, f"robocasa index row {line_number}: invalid split={split!r}")
                checks.expect(partition in allowed_partitions, f"robocasa index row {line_number}: invalid partition={partition!r}")
                checks.equal(row.get("action_adapter_version"), CANONICAL_VERSION, f"robocasa index row {line_number}.action_adapter_version")
                checks.equal(row.get("action_valid"), True, f"robocasa index row {line_number}.action_valid")
                checks.equal(row.get("factual_action_audit_passed"), True, f"robocasa index row {line_number}.factual_action_audit_passed")
                checks.equal(row.get("paired_views"), True, f"robocasa index row {line_number}.paired_views")
                if clip_id and split in split_ids:
                    split_ids[split].add(clip_id)
                if partition in allowed_partitions and split in {"train", "val"}:
                    partition_split_counts[partition][split] += 1
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"robocasa compact index is unreadable: {exc}")
        return
    checks.expect(bool(split_ids["train"]), "robocasa compact index has no train clips")
    checks.expect(bool(split_ids["val"]), "robocasa compact index has no val clips")
    checks.expect(not (split_ids["train"] & split_ids["val"]), "robocasa train/val clip overlap")
    for partition, counts in sorted(partition_split_counts.items()):
        checks.expect(counts["train"] > 0, f"robocasa {partition} has no train clips")
        checks.expect(counts["val"] > 0, f"robocasa {partition} has no val clips")


def validate_preflight(
    config: dict[str, Any],
    mode: str = "full",
    *,
    resume: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"static", "full"}:
        raise ValueError(f"unsupported mode: {mode}")
    checks = _Checks(mode)
    contract = config.get("contract") or {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    train = config.get("train") or {}
    loss = config.get("loss") or {}
    optimizer = config.get("optimizer") or {}
    schedule = config.get("lr_schedule") or {}
    sampler = train.get("mixed_batch_sampler") or {}
    profile = contract.get("profile")

    checks.equal(contract.get("schema"), "wm3d_v7_stage0_action_dynamics_contract_v1", "contract.schema")
    checks.expect(profile in {"main", "canary200"}, f"contract.profile is invalid: {profile!r}")
    checks.equal(contract.get("preflight_required"), True, "contract.preflight_required")
    checks.equal(contract.get("canary_required"), True, "contract.canary_required")
    checks.equal(contract.get("auto_promote"), False, "contract.auto_promote")
    canonical = contract.get("canonical_action") or {}
    checks.equal(canonical.get("version"), CANONICAL_VERSION, "contract.canonical.version")
    checks.equal(canonical.get("layout"), CANONICAL_LAYOUT, "contract.canonical.layout")
    checks.equal(canonical.get("source_frame"), "base", "contract.canonical.source_frame")
    checks.equal(canonical.get("translation_unit"), "meter", "contract.canonical.translation_unit")
    checks.equal(canonical.get("rotation_representation"), "axis_angle", "contract.canonical.rotation")
    checks.equal(canonical.get("gripper_semantics"), "signed_close_positive_continuous", "contract.canonical.gripper")

    checks.expect("oxe" not in data, "data.oxe pooled loader is forbidden; use data.oxe_sources")
    sources = data.get("oxe_sources")
    checks.expect(isinstance(sources, list), "data.oxe_sources must be a list")
    source_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict) or not source.get("source_name"):
                checks.errors.append("every data.oxe_sources entry needs source_name")
                continue
            name = str(source["source_name"])
            if name in source_by_name:
                checks.errors.append(f"duplicate OXE source_name: {name}")
            source_by_name[name] = source
    checks.equal(set(source_by_name), set(EXPECTED_OXE_DATASETS), "data.oxe_sources names")
    stats_paths: set[str] = set()
    for name, dataset in EXPECTED_OXE_DATASETS.items():
        source = source_by_name.get(name)
        if source is None:
            continue
        checks.expect(
            "action_cache_manifest" not in source,
            f"{name}.action_cache_manifest is a forbidden legacy alias; use canonical_action_cache_manifest only",
        )
        checks.expect(
            "action_stats" not in source and "action_stats_sha256" not in source,
            f"{name}: legacy source-local action_stats fields are forbidden; use canonical action stats maps only",
        )
        checks.equal(source.get("include_datasets"), [dataset], f"{name}.include_datasets")
        includes = {str(x) for x in source.get("include_datasets", ())}
        checks.expect(not (includes & FORBIDDEN_DATASETS), f"{name} contains a forbidden dataset")
        checks.equal(source.get("require_action_contract"), False, f"{name}.require_action_contract")
        checks.equal(source.get("require_canonical_action_contract"), True, f"{name}.require_canonical_action_contract")
        checks.equal(source.get("require_train_val_disjoint"), True, f"{name}.require_train_val_disjoint")
        checks.equal(source.get("allowed_action_kinds"), [EXPECTED_OXE_ACTION_KINDS[name]], f"{name}.allowed_action_kinds")
        checks.equal(source.get("default_action_frame_offset"), EXPECTED_OXE_OFFSETS[name], f"{name}.action_frame_offset")
        checks.equal(source.get("split"), {"mode": "episode", "val_frac": 0.03, "seed": 909}, f"{name}.split")
        adapter = source.get("action_adapter") or {}
        checks.equal(adapter.get("version"), CANONICAL_VERSION, f"{name}.action_adapter.version")
        checks.equal(adapter.get("rotation_conversion"), EXPECTED_ROTATION_CONVERSIONS[name], f"{name}.action_adapter.rotation_conversion")
        checks.equal(adapter.get("translation_unit"), "meter", f"{name}.action_adapter.translation_unit")
        checks.equal(adapter.get("source_frame"), "base", f"{name}.action_adapter.source_frame")
        checks.equal(adapter.get("gripper_semantics"), "signed_close_positive_continuous", f"{name}.action_adapter.gripper")
        checks.equal(source.get("canonical_action_enabled"), True, f"{name}.canonical_action_enabled")
        checks.equal(source.get("canonical_action_sources"), [dataset], f"{name}.canonical_action_sources")
        canonical_stats = source.get("canonical_action_stats_by_source") or {}
        canonical_stats_sha = source.get("canonical_action_stats_sha256_by_source") or {}
        checks.equal(set(canonical_stats), {dataset}, f"{name}.canonical_action_stats_by_source keys")
        checks.equal(set(canonical_stats_sha), {dataset}, f"{name}.canonical_action_stats_sha256_by_source keys")
        stats_paths.add(str(canonical_stats.get(dataset)))

        checks.pinned_file(source.get("manifest"), source.get("manifest_sha256"), f"{name}.manifest")
        checks.pinned_file(
            source.get("action_contract_evidence_path"),
            source.get("action_contract_evidence_sha256"),
            f"{name}.temporal_contract",
        )
        if name == "oxe_droid_action":
            checks.pinned_file(
                source.get("droid_provenance_index"),
                source.get("droid_provenance_index_sha256"),
                f"{name}.droid_provenance_index",
            )
        action_cache_path = checks.pinned_file(
            source.get("canonical_action_cache_manifest"),
            source.get("canonical_action_cache_manifest_sha256"),
            f"{name}.canonical_action_cache_manifest",
        )
        gate_path = checks.pinned_file(
            source.get("action_audit_gate"),
            source.get("action_audit_gate_sha256"),
            f"{name}.action_audit_gate",
        )
        stats_path = checks.pinned_file(
            canonical_stats.get(dataset),
            canonical_stats_sha.get(dataset),
            f"{name}.canonical_action_stats_by_source.{dataset}",
        )
        _validate_canonical_gate(checks, source, gate_path)
        _validate_action_cache_split(checks, source, action_cache_path)
        _validate_stats(checks, source, stats_path)
    checks.equal(len(stats_paths), 2, "OXE source-specific action_stats paths")

    checks.equal(data.get("dataset_type"), "v7_mixed", "data.dataset_type")
    checks.equal(data.get("robocasa_partitions"), ["atomic", "composite", "mg"], "data.robocasa_partitions")
    checks.approx(data.get("view_dropout"), 0.10, "data.view_dropout")
    checks.equal(data.get("require_action_stats"), True, "data.require_action_stats")
    compact_path = checks.pinned_file(data.get("compact_index"), data.get("compact_index_sha256"), "robocasa.compact_index")
    rc_stats_path = checks.pinned_file(data.get("action_stats"), data.get("action_stats_sha256"), "robocasa.action_stats")
    sidecar_pins = data.get("rgb_sidecar_sha256") or {}
    checks.equal(set(sidecar_pins), set(data.get("rgb_sidecar_indices", ())), "robocasa.rgb_sidecar pins")
    for path in data.get("rgb_sidecar_indices", ()):
        checks.pinned_file(path, sidecar_pins.get(path), f"robocasa.rgb_sidecar:{Path(path).parent.name}")
    _validate_robocasa_index(checks, compact_path)
    if checks.mode == "full" and compact_path is not None and rc_stats_path is not None:
        try:
            with np.load(rc_stats_path, allow_pickle=False) as archive:
                checks.equal(_scalar(archive, "split"), "train", "robocasa.stats.split")
                checks.equal(_scalar(archive, "index_sha256"), data.get("compact_index_sha256"), "robocasa.stats.index_sha256")
                checks.equal(np.asarray(archive["mean"]).shape, (6,), "robocasa.stats.mean.shape")
                checks.equal(np.asarray(archive["std"]).shape, (6,), "robocasa.stats.std.shape")
        except (OSError, KeyError, ValueError) as exc:
            checks.errors.append(f"robocasa.action_stats is invalid: {exc}")

    checks.equal(sampler.get("enabled"), True, "sampler.enabled")
    checks.equal(sampler.get("seed"), 1707, "sampler.seed")
    checks.equal(sampler.get("cycle_optimizer_steps"), 100, "sampler.cycle_optimizer_steps")
    checks.equal(sampler.get("shuffle_cycle"), True, "sampler.shuffle_cycle")
    checks.equal(sampler.get("synchronized_across_ranks"), True, "sampler.synchronized_across_ranks")
    checks.equal(sampler.get("accumulation_group_same_source"), True, "sampler.accumulation_group_same_source")
    rank_local = sampler.get("rank_local_cache_shards") or {}
    if profile == "canary200":
        checks.equal(rank_local.get("enabled"), True, "sampler.rank_local_cache_shards.enabled")
        checks.equal(rank_local.get("scope"), "canary_system_validation_only", "sampler.rank_local_cache_shards.scope")
        checks.equal(rank_local.get("allow_unequal_source_lengths"), True, "sampler.rank_local_cache_shards.allow_unequal_source_lengths")
        checks.equal(rank_local.get("require_each_source_at_least_global_batch"), True, "sampler.rank_local_cache_shards.require_each_source_at_least_global_batch")
        checks.equal(rank_local.get("record_startup_telemetry"), True, "sampler.rank_local_cache_shards.record_startup_telemetry")
        checks.equal(
            rank_local.get("node_roles"),
            {
                "0": {"host": "node43", "oxe_cache": "full_primary"},
                "1": {"host": "node44", "oxe_cache": "strict_subset_partial"},
            },
            "sampler.rank_local_cache_shards.node_roles",
        )
        checks.equal(
            rank_local.get("known_member_topology"),
            {
                "basis": "formal_manifest_last_wins",
                "unit": "canonical_windows",
                "droid": {
                    "relation": "node44_strict_subset_of_node43",
                    "train": {"node43": 130458, "node44": 68109},
                    "val": {"node43": 4044, "node44": 2019},
                    "train_plus_val": {
                        "node43": 134502,
                        "node44": 70128,
                        "intersection": 70128,
                        "node43_only": 64374,
                        "node44_only": 0,
                    },
                },
                "bridge": {
                    "relation": "node44_strict_subset_of_node43",
                    "train": {"node43": 26037, "node44": 4363},
                    "val": {"node43": 799, "node44": 134},
                    "train_plus_val": {
                        "node43": 26836,
                        "node44": 4497,
                        "intersection": 4497,
                        "node43_only": 22339,
                        "node44_only": 0,
                    },
                },
            },
            "sampler.rank_local_cache_shards.known_member_topology",
        )
    else:
        checks.equal(rank_local, {}, "sampler.rank_local_cache_shards formal prohibition")
    checks.equal(sampler.get("num_batches_per_epoch"), 50000, "sampler.num_batches_per_epoch")
    checks.equal(sampler.get("val_num_batches_per_epoch"), 100, "sampler.val_num_batches_per_epoch")
    counts = sampler.get("source_cycle_counts_exact") or {}
    checks.equal(counts, EXPECTED_SOURCES, "sampler.source_cycle_counts_exact")
    checks.equal(sum(int(v) for v in counts.values()) if isinstance(counts, dict) else -1, 100, "sampler.cycle count")
    checks.approx(sampler.get("expected_oxe_fraction"), 0.50, "sampler.expected_oxe_fraction")
    checks.approx(sampler.get("expected_robocasa_fraction"), 0.50, "sampler.expected_robocasa_fraction")
    checks.equal(set(sampler.get("forbidden_sources", ())), FORBIDDEN_DATASETS, "sampler.forbidden_sources")
    checks.expect(not (set(counts) & FORBIDDEN_DATASETS), "forbidden source has nonzero sampling probability")
    checks.expect(int(sampler.get("val_num_batches_per_epoch", 0) or 0) > 0, "validation sampler is empty")

    checks.equal(train.get("fresh_init_required"), True, "train.fresh_init_required")
    checks.equal(train.get("fresh_initialization_required"), True, "train.fresh_initialization_required")
    checks.equal(train.get("forbid_warm_start"), True, "train.forbid_warm_start")
    checks.equal(train.get("forbid_cross_run_resume"), True, "train.forbid_cross_run_resume")
    checks.equal(train.get("allow_exact_same_run_resume"), True, "train.allow_exact_same_run_resume")
    checks.equal(train.get("forbid_resume"), False, "train.forbid_resume")
    checks.equal(train.get("strict_v6_native_warm_start"), False, "train.strict_v6_native_warm_start")
    checks.equal(train.get("stage_transition"), None, "train.stage_transition")
    checks.equal(train.get("resume_checkpoint"), None, "train.resume_checkpoint")
    checks.expect(
        "reset_optimizer" not in train,
        "train.reset_optimizer must be absent; fresh start is selected by no --resume and exact resume restores optimizer/scheduler",
    )
    checks.equal(train.get("model_initialization"), "random_world_weights_with_frozen_pinned_codec", "train.model_initialization")
    checks.equal(train.get("pretrained_world_checkpoint"), None, "train.pretrained_world_checkpoint")
    checks.expect(bool(train.get("run_lineage")), "train.run_lineage is missing")
    resume_contract = train.get("resume_contract") or {}
    expected_resume_contract = {
        "mode": "exact_same_run_only",
        "require_same_run_lineage": True,
        "require_resume_compatible_config_digest": True,
        "require_same_output_root": True,
        "restore_model": True,
        "restore_optimizer": True,
        "restore_scheduler": True,
        "restore_global_step": True,
        "restore_sampler_cycle": True,
        "restore_rng": False,
        "rng_mode": "step_addressed_reconstruction",
        "verify_rng_contract": True,
        "require_step_named_checkpoint": True,
        "reject_latest_pt": True,
        "min_checkpoint_bytes": 15000000000,
    }
    checks.equal(resume_contract, expected_resume_contract, "train.resume_contract")
    checks.equal(train.get("num_nodes"), 2, "train.num_nodes")
    checks.equal(train.get("gpus_per_node"), 8, "train.gpus_per_node")
    checks.equal(train.get("batch_size_per_gpu"), 2, "train.batch_size_per_gpu")
    checks.equal(train.get("gradient_accumulation_steps"), 2, "train.gradient_accumulation_steps")
    checks.equal(data.get("node_sharded_window_cache"), profile == "canary200", "data.node_sharded_window_cache")
    checks.equal(bool(train.get("equalize_node_steps", False)), profile == "canary200", "train.equalize_node_steps")
    computed_gb = int(train.get("num_nodes", 0)) * int(train.get("gpus_per_node", 0)) * int(train.get("batch_size_per_gpu", 0)) * int(train.get("gradient_accumulation_steps", 0))
    checks.equal(computed_gb, 64, "computed global batch")
    checks.equal(train.get("effective_global_batch"), 64, "train.effective_global_batch")
    checks.equal(train.get("precision"), "bf16", "train.precision")
    checks.approx(train.get("grad_clip"), 1.0, "train.grad_clip")
    checks.equal(train.get("warmup_steps"), 20 if profile == "canary200" else 2000, "train.warmup_steps")
    if profile == "main":
        checks.equal(train.get("max_steps"), 100000, "train.max_steps")
        checks.equal(train.get("main_promotion_step"), 80000, "train.main_promotion_step")
        checks.equal(train.get("planned_review_stop_step"), 80000, "train.planned_review_stop_step")
        checks.equal(train.get("extension_cap_steps"), 100000, "train.extension_cap_steps")
        checks.equal(train.get("extension_requires_manual_decision"), True, "train.extension_requires_manual_decision")
    elif profile == "canary200":
        checks.equal(train.get("max_steps"), 200, "train.max_steps")
        checks.equal(train.get("planned_review_stop_step"), None, "train.planned_review_stop_step")
        checks.equal(train.get("extension_cap_steps"), 200, "train.extension_cap_steps")
        canary = config.get("canary") or {}
        checks.equal(canary.get("total_steps"), 200, "canary.total_steps")
        checks.equal(canary.get("phase1_stop_step"), 100, "canary.phase1_stop_step")
        checks.equal(canary.get("phase2_resume_step"), 100, "canary.phase2_resume_step")
        checks.equal(canary.get("phase2_stop_step"), 200, "canary.phase2_stop_step")
        checks.equal(canary.get("auto_promote"), False, "canary.auto_promote")

    checks.equal(train.get("oxe_representation_only"), False, "train.oxe_representation_only")
    factual = train.get("factual_action_conditioning") or {}
    checks.equal(factual.get("enabled"), True, "factual_action.enabled")
    checks.equal(factual.get("start_step"), 0, "factual_action.start_step")
    checks.equal(factual.get("detach_action_condition"), False, "factual_action.detach")
    checks.expect(
        "require_nonzero_action" not in factual,
        "factual_action.require_nonzero_action is forbidden: a finite canonical zero/hold action is valid",
    )
    checks.equal(factual.get("require_valid_action_contract"), True, "factual_action.require_valid_action_contract")
    checks.equal(factual.get("require_finite_action"), True, "factual_action.require_finite_action")
    checks.equal(factual.get("allow_zero_hold_action"), True, "factual_action.allow_zero_hold_action")
    checks.equal(set(factual.get("required_sources", ())), set(EXPECTED_SOURCES), "factual_action.required_sources")
    checks.equal(set(train.get("action_aux_sources", ())), set(EXPECTED_SOURCES), "train.action_aux_sources")
    checks.approx(loss.get("action"), 0.0, "loss.action")
    checks.equal(model.get("context_pixel_use_action"), True, "model.context_pixel_use_action")
    checks.equal((model.get("state") or {}).get("action_cond_dim"), 7, "model.state.action_cond_dim")
    checks.equal(model.get("token_codec_frozen"), True, "model.token_codec_frozen")
    checks.pinned_file(
        model.get("token_codec_checkpoint"),
        model.get("token_codec_checkpoint_sha256"),
        "model.token_codec_checkpoint",
    )
    checks.equal(set(train.get("factual_action_sources", ())), set(EXPECTED_SOURCES), "train.factual_action_sources")
    checks.equal(set(train.get("audited_action_sources", ())), set(EXPECTED_SOURCES), "train.audited_action_sources")
    checks.equal(train.get("representation_only_sources"), [], "train.representation_only_sources")
    configured_grip_contract = train.get("action_grip_contract")
    try:
        normalized_grip_contract = normalize_action_grip_contract(configured_grip_contract)
    except (TypeError, ValueError) as exc:
        checks.expect(False, f"train.action_grip_contract is not executable: {exc}")
        normalized_grip_contract = None
    checks.equal(configured_grip_contract, "signed_close", "train.action_grip_contract")
    checks.equal(normalized_grip_contract, "signed_close", "train.action_grip_contract.normalized")
    grip_semantics = train.get("action_grip_semantics") or {}
    checks.equal(
        grip_semantics.get("canonical_input"),
        "signed_close_positive_continuous",
        "train.action_grip_semantics.canonical_input",
    )
    checks.equal(
        grip_semantics.get("close01_supervision"),
        "exact_affine_g_plus_1_over_2",
        "train.action_grip_semantics.close01_supervision",
    )
    checks.approx(train.get("factual_main_action_loss_weight"), 0.0, "train.factual_main_action_loss_weight")

    checks.approx(train.get("native_action_no_teacher_weight"), 0.05, "no_teacher.weight")
    expected_aux_start = 0 if profile == "canary200" else 5000
    expected_aux_ramp = 20 if profile == "canary200" else 5000
    expected_no_teacher_every = 4
    checks.equal(train.get("native_action_no_teacher_start_step"), expected_aux_start, "no_teacher.start")
    checks.equal(train.get("native_action_no_teacher_ramp_steps"), expected_aux_ramp, "no_teacher.ramp")
    checks.equal(train.get("native_action_no_teacher_every"), expected_no_teacher_every, "no_teacher.every")
    checks.equal(train.get("native_core_action_cf_enabled"), True, "native_core_cf.enabled")
    expected_cf_start = 0 if profile == "canary200" else 10000
    expected_cf_ramp = 20 if profile == "canary200" else 5000
    expected_cf_every = 8
    checks.equal(train.get("native_core_action_cf_start_step"), expected_cf_start, "native_core_cf.start")
    checks.equal(train.get("native_core_action_cf_ramp_steps"), expected_cf_ramp, "native_core_cf.ramp")
    checks.equal(train.get("native_core_action_cf_every"), expected_cf_every, "native_core_cf.every")
    checks.expect(float(train.get("native_core_action_cf_rank_weight", 0.0) or 0.0) > 0.0, "native_core_cf rank weight is disabled")
    checks.expect(float(train.get("native_core_action_cf_separation_weight", 0.0) or 0.0) > 0.0, "native_core_cf separation weight is disabled")
    checks.approx(train.get("context_pixel_action_rank_weight"), 0.0, "legacy context_pixel rank weight")
    checks.approx(train.get("context_pixel_action_separation_weight"), 0.0, "legacy context_pixel separation weight")
    checks.equal(train.get("checkpoint_milestone_steps"), [5000, 10000, 15000], "train.checkpoint_milestone_steps")
    reviews = train.get("milestone_reviews") or {}
    checks.equal(reviews.get("fail_closed"), True, "milestone_reviews.fail_closed")
    checks.equal(reviews.get("pause_on_missing_or_failed_review"), True, "milestone_reviews.pause")
    checks.equal(
        [entry.get("step") for entry in reviews.get("required_review_steps", ())],
        [5000, 10000, 15000],
        "milestone_reviews.required_review_steps",
    )

    checks.approx(optimizer.get("peak_lr"), 1e-5, "optimizer.peak_lr")
    checks.approx(optimizer.get("grad_clip"), 1.0, "optimizer.grad_clip")
    checks.approx(schedule.get("peak_lr"), 1e-5, "lr_schedule.peak_lr")
    checks.equal(schedule.get("warmup_steps"), 20 if profile == "canary200" else 2000, "lr_schedule.warmup_steps")

    output_root = Path(str((config.get("out") or {}).get("root", "")))
    checks.expect(str(output_root) not in {"", "."}, "out.root is missing")
    checks.equal((config.get("out") or {}).get("require_empty_checkpoint_dir"), True, "out.require_empty_checkpoint_dir")
    if checks.mode == "full" and resume is None and output_root.exists():
        checkpoint_dir = output_root / str((config.get("out") or {}).get("ckpt_dir", "ckpt"))
        leftovers = []
        if checkpoint_dir.exists():
            leftovers = list(checkpoint_dir.glob("step_*.pt")) + list(checkpoint_dir.glob("latest.pt"))
        checks.expect(not leftovers, f"fresh-init output has checkpoint state: {leftovers[:3]}")
    if resume is not None:
        resume = resume.resolve()
        checks.expect(resume.name != "latest.pt", "resume must never use latest.pt")
        checks.expect(
            re.fullmatch(r"step_[0-9]{8}\.pt", resume.name) is not None,
            f"resume checkpoint is not step-addressed: {resume.name}",
        )
        expected_checkpoint_dir = (output_root / str((config.get("out") or {}).get("ckpt_dir", "ckpt"))).resolve()
        checks.equal(resume.parent, expected_checkpoint_dir, "resume checkpoint output lineage")
        if checks.mode == "full":
            checks.expect(resume.is_file(), f"resume checkpoint is missing: {resume}")
            if resume.is_file():
                checks.expect(
                    resume.stat().st_size >= int(resume_contract.get("min_checkpoint_bytes", 0)),
                    f"resume checkpoint is too small: {resume.stat().st_size}",
                )
                checks.expect(zipfile.is_zipfile(resume), "resume checkpoint is not a complete torch zip")

    report = {
        "schema": "wm3d_v7_stage0_action_dynamics_preflight_report_v1",
        "mode": mode,
        "profile": profile,
        "passed": not checks.errors,
        "launch_ready": mode == "full" and not checks.errors and not checks.blockers,
        "errors": checks.errors,
        "blockers": checks.blockers,
        "verified_artifacts": checks.verified_artifacts,
        "source_cycle_counts": EXPECTED_SOURCES,
        "global_batch": computed_gb,
        "initialization_mode": "exact_same_run_resume" if resume is not None else "fresh",
        "resolved_config_sha256": resolved_config_sha256(config),
        "runtime": {
            "train_path": str(RUNTIME_TRAIN_MODULE_PATH),
            "train_sha256": sha256_file(RUNTIME_TRAIN_MODULE_PATH),
            "canonical_action_path": str(RUNTIME_CANONICAL_ACTION_MODULE_PATH),
            "canonical_action_sha256": sha256_file(RUNTIME_CANONICAL_ACTION_MODULE_PATH),
        },
    }
    if checks.errors:
        raise PreflightError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "full"), default="full")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = validate_preflight(
            load_config(args.config), mode=args.mode, resume=args.resume
        )
    except (OSError, ValueError, yaml.YAMLError, PreflightError) as exc:
        if isinstance(exc, PreflightError):
            report = exc.report
        else:
            report = {
                "schema": "wm3d_v7_stage0_action_dynamics_preflight_report_v1",
                "mode": args.mode,
                "passed": False,
                "launch_ready": False,
                "errors": [str(exc)],
                "blockers": [],
            }
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
