#!/usr/bin/env python3
"""V8 Stage0 fail-closed preflight 的共享数据与资源校验。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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

MIN_DATA_FREE_BYTES = 200_000_000_000

def _validate_checkpoint_lineage(
    checks: _Checks,
    config: dict[str, Any],
    *,
    exact_resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    out = config.get("out") or {}
    root = Path(str(out.get("root") or ""))
    checks.expect(str(root) not in {"", "."}, "out.root is missing")
    checks.equal(
        out.get("require_empty_checkpoint_dir"),
        True,
        "out.require_empty_checkpoint_dir",
    )
    ckpt_dir = root / str(out.get("ckpt_dir", "ckpt"))
    existing = (
        sorted(
            (*ckpt_dir.glob("step_*.pt"), *ckpt_dir.glob("latest.pt")),
            key=lambda path: path.name,
        )
        if ckpt_dir.exists()
        else []
    )
    if exact_resume_checkpoint is None:
        checks.expect(
            not existing,
            f"formal output checkpoint lineage is not empty: {existing[:3]}",
        )
    else:
        expected = exact_resume_checkpoint.resolve()
        checks.expect(
            exact_resume_checkpoint.parent.resolve() == ckpt_dir.resolve(),
            "exact resume checkpoint must live in the configured checkpoint directory",
        )
        checks.expect(
            bool(re.fullmatch(r"step_[0-9]{8}\.pt", exact_resume_checkpoint.name)),
            f"exact resume checkpoint is not numbered: {exact_resume_checkpoint}",
        )
        checks.expect(
            exact_resume_checkpoint.is_file(),
            f"exact resume checkpoint is missing: {exact_resume_checkpoint}",
        )
        expected_names = {"latest.pt", exact_resume_checkpoint.name}
        observed_names = {path.name for path in existing}
        checks.expect(
            observed_names == expected_names,
            "exact resume checkpoint lineage mismatch: "
            f"expected={sorted(expected_names)} observed={sorted(observed_names)}",
        )
        latest = ckpt_dir / "latest.pt"
        checks.expect(
            latest.is_symlink() and latest.resolve() == expected,
            f"latest.pt does not resolve to exact resume checkpoint: {latest}",
        )
    return {
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_files": [str(path) for path in existing],
    }

def _validate_local_resources(
    checks: _Checks,
    config: dict[str, Any],
    *,
    exact_resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    health: dict[str, Any] = {}
    if checks.mode != "full":
        return health
    try:
        usage = shutil.disk_usage("/data")
        health["data_free_bytes"] = usage.free
        checks.expect(
            usage.free >= MIN_DATA_FREE_BYTES,
            f"/data free space is below {MIN_DATA_FREE_BYTES}: {usage.free}",
        )
    except OSError as exc:
        checks.errors.append(f"cannot inspect /data free space: {exc}")

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        health["gpu_ecc_rows"] = rows
        checks.equal(len(rows), 8, "local GPU count")
        for row in rows:
            fields = [part.strip() for part in row.split(",")]
            checks.expect(
                len(fields) == 3 and fields[1:] == ["0", "0"],
                f"uncorrected ECC is nonzero or unreadable: {row}",
            )
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        app_rows = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
        health["compute_apps"] = app_rows
        checks.expect(not app_rows, f"GPU compute applications are already active: {app_rows[:4]}")
    except (OSError, subprocess.CalledProcessError) as exc:
        checks.errors.append(f"cannot inspect GPU health: {exc}")

    health.update(
        _validate_checkpoint_lineage(
            checks, config, exact_resume_checkpoint=exact_resume_checkpoint
        )
    )
    return health
