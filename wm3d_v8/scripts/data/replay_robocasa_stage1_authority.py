#!/usr/bin/env python3
"""Freshly replay selected RoboCasa candidates and seal execution authority."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from wm3d_v3.data.manifest_contract import SHA256_RE, canonical_sha256
from wm3d_v3.stage1_planner.replay_authority import (
    REPLAY_AUTHORITY_FIELDS,
    REPLAY_AUTHORITY_ROW_FIELDS,
    REPLAY_AUTHORITY_ROW_SCHEMA,
    REPLAY_AUTHORITY_SCHEMA,
    REPLAY_BRANCH_FIELDS,
    REPLAY_ENVIRONMENT_FIELDS,
    REPLAY_ENVIRONMENT_SCHEMA,
    REPLAY_SOURCE_TREE_SCHEMA,
    PINNED_ACTION_AUDIT_SHA256,
    PINNED_ACTION_BRIDGE_SHA256,
    PINNED_ADAPTER_LOADER_SHA256,
    PINNED_REPLAY_HELPER_SHA256,
    PINNED_RUNTIME_GENERATOR_SHA256,
    PINNED_V7_ACTION_CONTRACT_SHA256,
    PINNED_V7_CONTRACTS_SHA256,
    REPLAY_SELECTION_FIELDS,
    REPLAY_SELECTION_ROW_FIELDS,
    REPLAY_SELECTION_SCHEMA,
    array_sha256,
    validate_replay_authority,
    validate_replay_environment,
)
from wm3d_v3.stage1_planner.execution_snapshot import (
    ExecutionSnapshotPlan,
    PinnedExecutionPath,
    ReadOnlyBindMount,
    enter_private_mount_namespace,
    scan_regular_tree,
)
from wm3d_v3.stage1_planner.rollout_audit import (
    TrustedOutputRoot,
    publish_no_clobber,
    read_regular_bytes,
)
from wm3d_v3.training.launch_qualification import verify_clean_runtime_checkout
from scripts.data.seal_robocasa_stage1_selection import rebuild_selection_receipt


LEGACY_CANDIDATE_SCHEMA = "wm3d_v7_stage1_planner_candidates_v2"
LEGACY_RUNTIME_SCHEMA = "wm3d_v7_stage1_planner_same_root_runtime_v3"
LEGACY_CANDIDATE_SEAL_SCHEMA = "wm3d_v7_stage1_planner_index_seal_v1"
_SPLITS = ("train", "val", "test")
_MODULES = ("numpy", "mujoco", "robosuite", "robocasa")
_COMPARE_ARRAYS = (
    "simulator_actions", "root_state", "root_rgb", "branch_rgb",
    "branch_rewards", "branch_dones", "branch_success",
)


def _sha(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA256 string")
    return value


def _json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _publish_json(path: Path, value: dict[str, Any], label: str) -> str:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    publish_no_clobber(path, payload, label)
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path, label: str) -> tuple[Path, bytes, str]:
    try:
        return read_regular_bytes(path, label)
    except (OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error


def _file_identity(path: Path, label: str) -> tuple[int, int, int, int, str]:
    resolved, payload, digest = _regular(path, label)
    metadata = resolved.stat(follow_symlinks=False)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        digest,
    )


def _source_tree_manifest(
    root: Path, *, name: str, package_name: str, commit: str
) -> tuple[dict[str, Any], bytes, str]:
    package = root / package_name
    if package.is_symlink() or not package.is_dir():
        raise RuntimeError("simulator source package root is invalid")
    rows: list[dict[str, Any]] = []
    for path in sorted(
        package.rglob("*"), key=lambda value: value.relative_to(package).as_posix()
    ):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise RuntimeError("simulator source tree contains a symlink")
        if not path.is_file():
            continue
        resolved, payload, digest = _regular(path, "simulator source file")
        rows.append({
            "path": resolved.relative_to(package).as_posix(),
            "size": len(payload),
            "sha256": digest,
        })
    if not rows:
        raise RuntimeError("simulator source tree is empty")
    tree_sha = canonical_sha256(rows)
    manifest = {
        "schema": REPLAY_SOURCE_TREE_SCHEMA,
        "root": str(root),
        "package": package_name,
        "commit": commit,
        "rows": rows,
        "rows_sha256": tree_sha,
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "passed": True,
    }
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    return manifest, payload, hashlib.sha256(payload).hexdigest()


def _rows(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} is not UTF-8 JSONL") from error
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} row {line_number} is not an object")
        result.append(value)
    if not result:
        raise RuntimeError(f"{label} is empty")
    return result


def _npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"{label} is not a valid no-pickle NPZ") from error


def _validate_candidate_payload(
    row: dict[str, Any], payload: dict[str, np.ndarray]
) -> None:
    required = {
        "schema", "root_id", "branch_roles", "branch_actions_physical",
        "action_history_physical", "root_context_sha256",
        "stage0_checkpoint_sha256", "simulator_action_low",
        "simulator_action_high", "simulator_executable_all_candidates",
        "counterfactual_pose_space",
    }
    if not required.issubset(payload):
        raise RuntimeError("candidate payload fields are incomplete")
    if (
        str(payload["schema"].item()) != LEGACY_CANDIDATE_SCHEMA
        or str(payload["root_id"].item()) != row["root_id"]
        or payload["branch_actions_physical"].shape != (10, 32, 7)
        or payload["action_history_physical"].shape != (4, 7)
        or payload["simulator_action_low"].shape != (12,)
        or payload["simulator_action_high"].shape != (12,)
        or payload["simulator_executable_all_candidates"].item() is not True
        or str(payload["counterfactual_pose_space"].item())
        != "physical_canonical_6d"
        or str(payload["root_context_sha256"].item())
        != row["root_context_sha256"]
        or str(payload["stage0_checkpoint_sha256"].item())
        != row["stage0_checkpoint_sha256"]
        or len(payload["branch_roles"]) != 10
    ):
        raise RuntimeError("candidate payload capacities/lineage are invalid")


def _source_episode_path(source: Path, episode_id: int) -> Path:
    return source / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"


def _legacy_runtime_rows(runtime_root: Path) -> dict[str, tuple[dict, Path, str]]:
    result: dict[str, tuple[dict, Path, str]] = {}
    for path in sorted(runtime_root.glob("index.shard-*.jsonl")):
        if path.name.endswith(".partial"):
            continue
        resolved, payload, digest = _regular(path, "legacy runtime index shard")
        for row in _rows(payload, str(resolved)):
            root_id = row.get("root_id")
            if type(root_id) is not str or root_id in result:
                raise RuntimeError("legacy runtime index has invalid/duplicate root")
            result[root_id] = (row, resolved, digest)
    if not result:
        raise RuntimeError("legacy runtime contains no sealed index rows")
    return result


def _selection_manifest(
    payload: bytes,
    *,
    code_commit: str,
    candidate_index_sha256: str,
    candidate_seal_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    value = _json(payload, "selection manifest")
    if set(value) != REPLAY_SELECTION_FIELDS:
        raise RuntimeError("selection manifest exact fields mismatch")
    if (
        value["schema"] != REPLAY_SELECTION_SCHEMA
        or value["passed"] is not True
        or value["code_commit"] != code_commit
        or value["candidate_index_sha256"] != candidate_index_sha256
        or value["candidate_index_seal_sha256"] != candidate_seal_sha256
    ):
        raise RuntimeError("selection manifest lineage mismatch")
    rebuild_selection_receipt(value, expected_code_commit=code_commit)
    counts = value["selection_count"]
    rows = value["rows"]
    if (
        not isinstance(counts, dict)
        or set(counts) != set(_SPLITS)
        or any(type(counts[name]) is not int or counts[name] <= 0 for name in _SPLITS)
        or not isinstance(rows, list)
        or sum(counts.values()) != len(rows)
        or canonical_sha256(rows) != _sha(value["rows_sha256"], "selection rows SHA")
    ):
        raise RuntimeError("selection manifest count/row closure mismatch")
    selected: dict[str, dict[str, Any]] = {}
    observed = {split: 0 for split in _SPLITS}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != REPLAY_SELECTION_ROW_FIELDS
            or row["split"] not in observed
        ):
            raise RuntimeError("selection manifest row fields mismatch")
        root_id = _sha(row["root_id"], "selection root id")
        if root_id in selected:
            raise RuntimeError("selection manifest contains duplicate root")
        selected[root_id] = row
        observed[row["split"]] += 1
    if observed != counts:
        raise RuntimeError("selection manifest split counts mismatch")
    return selected, value


def _module_probe_script() -> str:
    return """
import hashlib, json, os, platform
from pathlib import Path
modules = {}
for name in ("numpy", "mujoco", "robosuite", "robocasa"):
    module = __import__(name)
    path = Path(module.__file__).resolve(strict=True)
    modules[name] = {
        "version": str(getattr(module, "__version__", "unknown")),
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
from scripts import generate_robocasa_stage1_planner_branches as generator
from scripts import generate_robocasa_same_root_cf as helper
from scripts import robocasa_stage1_adapter_loader as adapter_loader
from wm3d_v3.data import v7_action_contract, v7_contracts
from wm3d_v3.stage1_planner import action_bridge
snapshot_modules = {}
for name, module in {
    "runtime_generator": generator,
    "replay_helper": helper,
    "adapter_loader": adapter_loader,
    "v7_action_contract": v7_action_contract,
    "v7_contracts": v7_contracts,
    "action_bridge": action_bridge,
}.items():
    path = Path(module.__file__).resolve(strict=True)
    snapshot_modules[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
print(json.dumps({
    "python_version": platform.python_version(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    "mujoco_gl": os.environ.get("MUJOCO_GL", "unset"),
    "modules": modules,
    "snapshot_modules": snapshot_modules,
}, sort_keys=True))
"""


def _environment_receipt(
    *,
    simulator_command: Path,
    simulator_execution_command: str,
    simulator_environment: dict[str, str],
    simulator_execution_environment: dict[str, str],
    simulator_site_packages: Path,
    simulator_site_packages_provenance: Path,
    simulator_stdlib_provenance: Path,
    simulator_stdlib_snapshot: Path,
    robocasa_source_root: Path,
    robocasa_source_provenance_root: Path,
    robosuite_source_root: Path,
    robosuite_source_provenance_root: Path,
    simulator_python_provenance_path: Path,
    simulator_python_provenance_sha256: str,
    egl_vendor_library_provenance_path: Path,
    egl_vendor_library_provenance_sha256: str,
    execution_snapshot_manifest_path: Path,
    execution_snapshot_manifest_sha256: str,
    source_trees: dict[str, dict[str, str]],
    code_commit: str,
    generator_sha256: str,
    generator_snapshot_sha256: str,
    helper_sha256: str,
    adapter_loader_sha256: str,
    v7_action_contract_sha256: str,
    v7_contracts_sha256: str,
    action_bridge_sha256: str,
    output: Path,
    output_scope: TrustedOutputRoot,
    execution_root: Path,
    execution_cwd: str,
    execution_path_aliases: dict[str, Path],
    pass_fds: tuple[int, ...],
) -> tuple[Path, str, dict[str, Any]]:
    python_path, _python_payload, python_sha = _regular(
        simulator_command, "simulator Python"
    )
    probe = subprocess.run(
        [simulator_execution_command, "-S", "-c", _module_probe_script()],
        check=True,
        capture_output=True,
        text=True,
        env=simulator_execution_environment,
        cwd=execution_cwd,
        pass_fds=pass_fds,
    )
    try:
        value = json.loads(probe.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("simulator environment probe did not emit JSON") from error
    for group in ("modules", "snapshot_modules"):
        for module in value[group].values():
            observed = module["path"]
            for alias, persistent in execution_path_aliases.items():
                if observed == alias or observed.startswith(alias + "/"):
                    suffix = observed[len(alias):].lstrip("/")
                    module["path"] = str(persistent / suffix)
                    break
    freeze = subprocess.run(
        [simulator_execution_command, "-S", "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        env=simulator_execution_environment,
        cwd=execution_cwd,
        pass_fds=pass_fds,
    ).stdout
    freeze_path = output.parent / "pip_freeze.txt"
    output_scope.publish(freeze_path, freeze, label="replay pip freeze")
    receipt = {
        "schema": REPLAY_ENVIRONMENT_SCHEMA,
        "code_commit": code_commit,
        "execution_snapshot_manifest_path": str(execution_snapshot_manifest_path),
        "execution_snapshot_manifest_sha256": execution_snapshot_manifest_sha256,
        "simulator_python_provenance_path": str(simulator_python_provenance_path),
        "simulator_python_provenance_sha256": simulator_python_provenance_sha256,
        "simulator_python_path": str(python_path),
        "simulator_python_sha256": python_sha,
        "simulator_python_device": python_path.stat(follow_symlinks=False).st_dev,
        "simulator_python_inode": python_path.stat(follow_symlinks=False).st_ino,
        "simulator_python_size": python_path.stat(follow_symlinks=False).st_size,
        "simulator_python_mtime_ns": python_path.stat(follow_symlinks=False).st_mtime_ns,
        "simulator_pythonpath": simulator_environment["PYTHONPATH"],
        "simulator_pythonhome": simulator_environment["PYTHONHOME"],
        "python_version": value["python_version"],
        "cuda_visible_devices": value["cuda_visible_devices"],
        "mujoco_gl": value["mujoco_gl"],
        "egl_vendor_library_path": simulator_environment[
            "__EGL_VENDOR_LIBRARY_FILENAMES"
        ],
        "egl_vendor_library_sha256": hashlib.sha256(
            Path(simulator_environment["__EGL_VENDOR_LIBRARY_FILENAMES"]).read_bytes()
        ).hexdigest(),
        "egl_vendor_library_provenance_path": str(
            egl_vendor_library_provenance_path
        ),
        "egl_vendor_library_provenance_sha256": (
            egl_vendor_library_provenance_sha256
        ),
        "pip_freeze_path": str(freeze_path),
        "pip_freeze_sha256": hashlib.sha256(freeze).hexdigest(),
        "simulator_site_packages_path": str(simulator_site_packages),
        "simulator_site_packages_provenance_path": str(
            simulator_site_packages_provenance
        ),
        "simulator_stdlib_provenance_root": str(simulator_stdlib_provenance),
        "simulator_stdlib_snapshot_root": str(simulator_stdlib_snapshot),
        "robocasa_source_root": str(robocasa_source_root),
        "robocasa_source_provenance_root": str(robocasa_source_provenance_root),
        "robocasa_source_commit": "8f3c96ec8d1bfcd8126cad2bca887da98d30e997",
        "robosuite_source_root": str(robosuite_source_root),
        "robosuite_source_provenance_root": str(robosuite_source_provenance_root),
        "robosuite_source_commit": "6c10ef24a4bb52f59199976125060ce793470e6e",
        "source_trees": source_trees,
        "modules": value["modules"],
        "snapshot_modules": value["snapshot_modules"],
        "runtime_generator_sha256": generator_sha256,
        "runtime_generator_snapshot_sha256": generator_snapshot_sha256,
        "replay_helper_sha256": helper_sha256,
        "adapter_loader_sha256": adapter_loader_sha256,
        "v7_action_contract_sha256": v7_action_contract_sha256,
        "v7_contracts_sha256": v7_contracts_sha256,
        "action_bridge_sha256": action_bridge_sha256,
        "passed": True,
    }
    if set(receipt) != REPLAY_ENVIRONMENT_FIELDS:
        raise AssertionError("internal environment receipt fields drifted")
    validate_replay_environment(
        receipt,
        expected_code_commit=code_commit,
        runtime_generator_sha256=generator_sha256,
        runtime_generator_snapshot_sha256=generator_snapshot_sha256,
        replay_helper_sha256=helper_sha256,
        adapter_loader_sha256=adapter_loader_sha256,
        v7_action_contract_sha256=v7_action_contract_sha256,
        v7_contracts_sha256=v7_contracts_sha256,
        action_bridge_sha256=action_bridge_sha256,
        verify_referents=True,
    )
    receipt_payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    output_scope.publish(output, receipt_payload, label="replay environment receipt")
    digest = hashlib.sha256(receipt_payload).hexdigest()
    return output.absolute(), digest, receipt


def _branch_rows(payload: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    roles = payload["branch_roles"]
    simulator = payload["simulator_actions"]
    rgb = payload["branch_rgb"]
    rewards = payload["branch_rewards"]
    dones = payload["branch_dones"]
    success = payload["branch_success"]
    render = payload["root_render_state_branch_sha256"]
    raw_rgb = payload["root_rgb_raw_sha256"]
    state_sha = array_sha256(payload["root_state"])
    result = []
    for index in range(len(roles)):
        row = {
            "index": index,
            "role": str(roles[index]),
            "root_state_sha256": state_sha,
            "root_render_state_sha256": str(render[index]),
            "root_rgb_raw_sha256": str(raw_rgb[index]),
            "executed_action_sha256": array_sha256(simulator[index]),
            "rgb_sha256": array_sha256(rgb[index]),
            "reward_sha256": array_sha256(rewards[index]),
            "done_sha256": array_sha256(dones[index]),
            "success_sha256": array_sha256(success[index]),
            "terminal_success": bool(success[index].any()),
            "max_reward": float(rewards[index].max()),
        }
        if set(row) != REPLAY_BRANCH_FIELDS:
            raise AssertionError("internal replay branch fields drifted")
        result.append(row)
    return result


def _candidate_seal(
    payload: bytes,
    *,
    candidate_index: Path,
    candidate_index_sha256: str,
) -> dict[str, Any]:
    seal = _json(payload, "candidate index seal")
    output = seal.get("output")
    if (
        seal.get("schema") != LEGACY_CANDIDATE_SEAL_SCHEMA
        or seal.get("passed") is not True
        or type(output) is not str
        or Path(output).is_symlink()
        or Path(output).resolve(strict=True) != candidate_index
        or seal.get("output_sha256") != candidate_index_sha256
        or type(seal.get("roots")) is not int
        or seal["roots"] <= 0
    ):
        raise RuntimeError("candidate index seal does not authorize the index")
    return seal


def main() -> None:
    enter_private_mount_namespace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-generator", type=Path, required=True)
    parser.add_argument("--replay-helper", type=Path, required=True)
    parser.add_argument("--adapter-loader", type=Path, required=True)
    parser.add_argument("--v7-action-contract", type=Path, required=True)
    parser.add_argument("--v7-contracts", type=Path, required=True)
    parser.add_argument("--action-bridge", type=Path, required=True)
    parser.add_argument("--simulator-python", type=Path, required=True)
    parser.add_argument("--simulator-stdlib", type=Path, required=True)
    parser.add_argument("--simulator-site-packages", type=Path, required=True)
    parser.add_argument("--robocasa-source-root", type=Path, required=True)
    parser.add_argument("--robosuite-source-root", type=Path, required=True)
    parser.add_argument("--egl-vendor-library", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--candidate-index-seal", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--code-repo",
        type=Path,
        help="Clean checkout to bind; defaults to the checkout containing this script.",
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    repo = (
        args.code_repo
        if args.code_repo is not None
        else Path(__file__).resolve().parents[2]
    )
    code_commit = verify_clean_runtime_checkout(repo, args.code_commit)
    if args.height <= 0 or args.width <= 0:
        raise RuntimeError("replay image dimensions must be positive")
    runtime_root = args.runtime_root
    if runtime_root.is_symlink():
        raise RuntimeError("legacy runtime root must not be a symlink")
    runtime_root = runtime_root.resolve(strict=True)
    if not runtime_root.is_dir():
        raise RuntimeError("legacy runtime root must be a real directory")
    output_root = args.output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("fresh replay output root must not already exist")
    if args.output.absolute().is_relative_to(output_root):
        raise RuntimeError("authority receipt must be outside fresh replay output root")

    generator, generator_payload, generator_sha = _regular(
        args.runtime_generator, "runtime generator"
    )
    helper, _helper_payload, helper_sha = _regular(
        args.replay_helper, "replay helper"
    )
    adapter_loader, adapter_loader_payload, adapter_loader_sha = _regular(
        args.adapter_loader, "adapter loader"
    )
    _v7_action_contract, v7_action_contract_payload, v7_action_contract_sha = _regular(
        args.v7_action_contract, "V7 action contract"
    )
    _v7_contracts, v7_contracts_payload, v7_contracts_sha = _regular(
        args.v7_contracts, "V7 contracts"
    )
    _action_bridge, action_bridge_payload, action_bridge_sha = _regular(
        args.action_bridge, "action bridge"
    )
    action_audit, _action_payload, action_sha = _regular(
        args.action_audit, "action audit"
    )
    if generator_sha != PINNED_RUNTIME_GENERATOR_SHA256:
        raise RuntimeError("runtime generator is not the pinned authority producer")
    if helper_sha != PINNED_REPLAY_HELPER_SHA256:
        raise RuntimeError("replay helper is not the pinned authority helper")
    if adapter_loader_sha != PINNED_ADAPTER_LOADER_SHA256:
        raise RuntimeError("adapter loader is not the pinned action helper")
    if v7_action_contract_sha != PINNED_V7_ACTION_CONTRACT_SHA256:
        raise RuntimeError("V7 action contract is not pinned")
    if v7_contracts_sha != PINNED_V7_CONTRACTS_SHA256:
        raise RuntimeError("V7 contracts module is not pinned")
    if action_bridge_sha != PINNED_ACTION_BRIDGE_SHA256:
        raise RuntimeError("action bridge is not pinned")
    if action_sha != PINNED_ACTION_AUDIT_SHA256:
        raise RuntimeError("action audit is not the pinned factual action authority")
    if b"from scripts.generate_robocasa_same_root_cf import" not in generator_payload:
        raise RuntimeError("pinned generator does not import the sealed replay helper")
    candidate_index, candidate_payload, candidate_sha = _regular(
        args.candidate_index, "candidate index"
    )
    candidate_seal, candidate_seal_payload, candidate_seal_sha = _regular(
        args.candidate_index_seal, "candidate index seal"
    )
    candidate_seal_value = _candidate_seal(
        candidate_seal_payload,
        candidate_index=candidate_index,
        candidate_index_sha256=candidate_sha,
    )
    selection_manifest, selection_manifest_payload, selection_manifest_sha = _regular(
        args.selection_manifest, "selection manifest"
    )
    selected, _selection_receipt = _selection_manifest(
        selection_manifest_payload,
        code_commit=code_commit,
        candidate_index_sha256=candidate_sha,
        candidate_seal_sha256=candidate_seal_sha,
    )
    candidates = {}
    for row in _rows(candidate_payload, "candidate index"):
        root_id = row.get("root_id")
        if (
            row.get("schema") != LEGACY_CANDIDATE_SCHEMA
            or type(root_id) is not str
            or root_id in candidates
        ):
            raise RuntimeError("candidate index row schema/identity is invalid")
        candidates[root_id] = row
    if set(selected) - set(candidates):
        raise RuntimeError("selected root is absent from candidate index")
    legacy_rows = _legacy_runtime_rows(runtime_root)
    if set(selected) - set(legacy_rows):
        raise RuntimeError("selected root is absent from legacy runtime")
    chosen = []
    for root_id, selection_row in selected.items():
        split = selection_row["split"]
        candidate = candidates[root_id]
        legacy, _shard, _digest = legacy_rows[root_id]
        if candidate.get("split") != split or legacy.get("split") != split:
            raise RuntimeError("selected split differs from candidate/runtime")
        if (
            candidate.get("future_frames") != 32
            or candidate.get("h32_factual_available") is not True
            or candidate.get("future_observation_leakage") is not False
            or candidate.get("simulator_executable_all_candidates") is not True
            or candidate.get("action_audit_sha256") != action_sha
            or candidate.get("stage0_checkpoint_sha256")
            != candidate_seal_value["stage0_checkpoint_sha256"]
        ):
            raise RuntimeError("selected candidate row contract is invalid")
        candidate_file, candidate_bytes, candidate_file_sha = _regular(
            Path(candidate["candidate_path"]), "selected candidate payload"
        )
        if candidate_file_sha != candidate["payload_sha256"]:
            raise RuntimeError("selected candidate payload SHA drift")
        _validate_candidate_payload(candidate, _npz(candidate_bytes, str(candidate_file)))
        chosen.append(candidate)
    chosen.sort(key=lambda row: (_SPLITS.index(row["split"]), row["root_id"]))
    output = TrustedOutputRoot(output_root.parent, label="replay authority parent")
    output_root_pin = output.pin_directory(
        output_root, label="fresh replay output root"
    )
    snapshot_root = output_root / "execution_snapshot"
    snapshot_input_root = snapshot_root / "inputs"
    selection_path = snapshot_input_root / "selected_candidates.jsonl"
    execution_root = snapshot_input_root / "execution"
    scripts_root = execution_root / "scripts"
    wm3d_root = execution_root / "wm3d_v3"
    data_root = wm3d_root / "data"
    planner_root = wm3d_root / "stage1_planner"
    output.mkdir(scripts_root, label="pinned replay script package")
    output.mkdir(data_root, label="pinned action data package")
    output.mkdir(planner_root, label="pinned action bridge package")
    output.publish(scripts_root / "__init__.py", b"", label="script package marker")
    output.publish(wm3d_root / "__init__.py", b"", label="wm3d package marker")
    output.publish(data_root / "__init__.py", b"", label="data package marker")
    output.publish(planner_root / "__init__.py", b"", label="planner package marker")
    generator_snapshot = scripts_root / "generate_robocasa_stage1_planner_branches.py"
    helper_snapshot = scripts_root / "generate_robocasa_same_root_cf.py"
    adapter_snapshot = scripts_root / "robocasa_stage1_adapter_loader.py"
    old_adapter_import = b"from scripts.cache_robocasa365_v7_compact import _load_adapter"
    new_adapter_import = b"from scripts.robocasa_stage1_adapter_loader import _load_adapter"
    if generator_payload.count(old_adapter_import) != 1:
        raise RuntimeError("pinned generator adapter import closure drifted")
    generator_snapshot_payload = generator_payload.replace(
        old_adapter_import, new_adapter_import
    )
    resolved_output_path = b'"path": str(destination.resolve()),'
    relative_output_path = b'"path": destination.as_posix(),'
    if generator_snapshot_payload.count(resolved_output_path) != 1:
        raise RuntimeError("pinned generator output-path closure drifted")
    generator_snapshot_payload = generator_snapshot_payload.replace(
        resolved_output_path, relative_output_path
    )
    generator_snapshot_sha = hashlib.sha256(generator_snapshot_payload).hexdigest()
    output.publish(
        generator_snapshot, generator_snapshot_payload, label="generator snapshot"
    )
    _helper_path, helper_payload, _helper_digest = _regular(
        helper, "replay helper snapshot source"
    )
    output.publish(helper_snapshot, helper_payload, label="replay helper snapshot")
    output.publish(adapter_snapshot, adapter_loader_payload, label="adapter loader snapshot")
    output.publish(
        data_root / "v7_action_contract.py", v7_action_contract_payload,
        label="V7 action contract snapshot",
    )
    output.publish(
        data_root / "v7_contracts.py", v7_contracts_payload,
        label="V7 contracts snapshot",
    )
    output.publish(
        planner_root / "action_bridge.py", action_bridge_payload,
        label="action bridge snapshot",
    )
    command_python = args.simulator_python.resolve(strict=True)
    if command_python.is_symlink() or not command_python.is_file():
        raise RuntimeError("resolved simulator Python must be a regular file")
    provenance_python_identity = _file_identity(
        command_python, "simulator Python provenance"
    )
    simulator_stdlib = args.simulator_stdlib
    simulator_site_packages = args.simulator_site_packages
    robocasa_source_root = args.robocasa_source_root
    robosuite_source_root = args.robosuite_source_root
    for path, label in (
        (simulator_stdlib, "simulator stdlib"),
        (simulator_site_packages, "simulator site-packages"),
        (robocasa_source_root, "RoboCasa source root"),
        (robosuite_source_root, "robosuite source root"),
    ):
        if path.is_symlink() or not path.resolve(strict=True).is_dir():
            raise RuntimeError(f"{label} must be a real directory")
    simulator_stdlib = simulator_stdlib.resolve(strict=True)
    simulator_site_packages = simulator_site_packages.resolve(strict=True)
    robocasa_source_root = robocasa_source_root.resolve(strict=True)
    robosuite_source_root = robosuite_source_root.resolve(strict=True)
    egl_vendor_path, _egl_vendor_payload, _egl_vendor_sha = _regular(
        args.egl_vendor_library, "EGL vendor library manifest"
    )
    for source_root, expected, label in (
        (robocasa_source_root, "8f3c96ec8d1bfcd8126cad2bca887da98d30e997", "RoboCasa"),
        (robosuite_source_root, "6c10ef24a4bb52f59199976125060ce793470e6e", "robosuite"),
    ):
        if source_root.name != f"{label.lower()}-{expected}":
            raise RuntimeError(f"{label} source root is not the pinned snapshot")
    source_tree_manifests: dict[str, dict[str, Any]] = {}
    source_tree_references: dict[str, dict[str, str]] = {}
    for name, source_root, package_name, commit in (
        ("robocasa", robocasa_source_root, "robocasa", "8f3c96ec8d1bfcd8126cad2bca887da98d30e997"),
        ("robosuite", robosuite_source_root, "robosuite", "6c10ef24a4bb52f59199976125060ce793470e6e"),
    ):
        manifest, manifest_payload, manifest_sha = _source_tree_manifest(
            source_root, name=name, package_name=package_name, commit=commit
        )
        source_tree_manifests[name] = manifest
    snapshot_plan = ExecutionSnapshotPlan(output, snapshot_input_root)
    action_audit_snapshot, _ = snapshot_plan.add_verified_file(
        action_audit, snapshot_input_root / "action_audit.json",
        kind="action audit",
    )
    egl_vendor_snapshot, _ = snapshot_plan.add_verified_file(
        egl_vendor_path, snapshot_input_root / "runtime/egl_vendor.json",
        kind="EGL vendor manifest",
    )
    simulator_python_snapshot = snapshot_plan.add_file(
        command_python, snapshot_input_root / "python/bin/python3.10",
        expected_sha256=provenance_python_identity[4],
        size=provenance_python_identity[2],
        kind="simulator Python", mode=0o750,
    )
    stdlib_rows = scan_regular_tree(
        simulator_stdlib, label="simulator stdlib",
        exclude_python_cache=True, materialize_file_symlinks=True,
    )
    site_rows = scan_regular_tree(
        simulator_site_packages, label="simulator site-packages",
        exclude_python_cache=True, materialize_file_symlinks=True,
    )
    simulator_stdlib_snapshot = snapshot_input_root / "python/lib/python3.10"
    simulator_site_packages_snapshot = snapshot_input_root / "python/site-packages"
    snapshot_plan.add_tree(
        simulator_stdlib, simulator_stdlib_snapshot, stdlib_rows,
        kind="simulator stdlib",
    )
    snapshot_plan.add_tree(
        simulator_site_packages, simulator_site_packages_snapshot, site_rows,
        kind="simulator site-packages",
    )
    source_snapshots: dict[str, Path] = {}
    for name, source_root, package_name, commit in (
        ("robocasa", robocasa_source_root, "robocasa", "8f3c96ec8d1bfcd8126cad2bca887da98d30e997"),
        ("robosuite", robosuite_source_root, "robosuite", "6c10ef24a4bb52f59199976125060ce793470e6e"),
    ):
        snapshot_source_root = snapshot_input_root / "sources" / f"{name}-{commit}"
        source_snapshots[name] = snapshot_source_root
        snapshot_plan.add_tree(
            source_root / package_name,
            snapshot_source_root / package_name,
            source_tree_manifests[name]["rows"],
            kind=f"{name} source tree",
        )
    execution_candidates: list[dict[str, Any]] = []
    snapshot_inputs: dict[str, dict[str, Path]] = {}
    selected_digest_fields = (
        ("source_episode", "source_episode_sha256"),
        ("states", "states_sha256"),
        ("model_xml_gz", "model_xml_gz_sha256"),
        ("ep_meta", "ep_meta_file_sha256"),
        ("dataset_meta", "dataset_meta_sha256"),
        ("modality", "modality_sha256"),
        ("candidate_payload", "candidate_payload_sha256"),
        ("root_context", "root_context_sha256"),
    )
    for candidate in chosen:
        root_id = candidate["root_id"]
        sealed = selected[root_id]
        dataset_root = snapshot_input_root / "selected" / root_id / "lerobot"
        paths: dict[str, Path] = {}
        for name, digest_field in selected_digest_fields:
            source_path = Path(sealed[f"{name}_path"])
            if name in {"source_episode", "states", "model_xml_gz", "ep_meta", "dataset_meta", "modality"}:
                relative = source_path.relative_to(Path(sealed["source_dataset_path"]))
                target = dataset_root / relative
            else:
                target = snapshot_input_root / "selected" / root_id / name / source_path.name
            _resolved, payload, observed = _regular(source_path, f"selected {name}")
            if observed != sealed[digest_field]:
                raise RuntimeError(f"selected {name} provenance SHA drift")
            paths[name] = snapshot_plan.add_file(
                source_path, target, expected_sha256=observed,
                size=len(payload), kind=f"selected {name}",
            )
        execution = dict(candidate)
        execution["source_dataset"] = "./" + dataset_root.relative_to(
            snapshot_root
        ).as_posix()
        execution["candidate_path"] = "./" + paths[
            "candidate_payload"
        ].relative_to(snapshot_root).as_posix()
        execution["root_context_path"] = "./" + paths[
            "root_context"
        ].relative_to(snapshot_root).as_posix()
        execution_candidates.append(execution)
        snapshot_inputs[root_id] = paths | {"source_dataset": dataset_root}
    selection_payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode()
        for row in execution_candidates
    )
    output.publish(selection_path, selection_payload, label="selected candidate index")
    snapshot_plan.add_verified_file(
        selection_path, selection_path, kind="selected candidate index"
    )
    for module_path in (
        generator_snapshot, helper_snapshot, adapter_snapshot,
        data_root / "v7_action_contract.py", data_root / "v7_contracts.py",
        planner_root / "action_bridge.py",
    ):
        snapshot_plan.add_verified_file(
            module_path, module_path, kind="execution module"
        )
    snapshot_manifest_path = output_root / "execution_snapshot_manifest.json"
    snapshot_manifest, snapshot_manifest_sha = snapshot_plan.seal(
        snapshot_manifest_path
    )
    robocasa_source_root = source_snapshots["robocasa"]
    robosuite_source_root = source_snapshots["robosuite"]
    for name, source_root, package_name, commit in (
        ("robocasa", robocasa_source_root, "robocasa", "8f3c96ec8d1bfcd8126cad2bca887da98d30e997"),
        ("robosuite", robosuite_source_root, "robosuite", "6c10ef24a4bb52f59199976125060ce793470e6e"),
    ):
        manifest = dict(source_tree_manifests[name], root=str(source_root))
        manifest_payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
        manifest_path = output_root / f"{name}_source_tree.json"
        output.publish(manifest_path, manifest_payload, label=f"{name} source-tree manifest")
        source_tree_references[name] = {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "tree_sha256": manifest["rows_sha256"],
        }
    command_python = simulator_python_snapshot
    simulator_site_packages = simulator_site_packages_snapshot
    simulator_environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "__EGL_VENDOR_LIBRARY_FILENAMES": str(egl_vendor_snapshot),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "4"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHOME": str(snapshot_input_root / "python"),
        "PYTHONPATH": os.pathsep.join((
            str(execution_root), str(robocasa_source_root),
            str(robosuite_source_root), str(simulator_site_packages),
        )),
    }
    snapshot_anchor = PinnedExecutionPath(
        snapshot_root, directory=True, label="replay execution snapshot"
    )
    immutable_inputs = ReadOnlyBindMount(
        snapshot_input_root, label="replay execution inputs"
    )
    try:
        immutable_inputs.__enter__()
        execution_environment = dict(simulator_environment)
        execution_environment.update({
            "__EGL_VENDOR_LIBRARY_FILENAMES": "./inputs/runtime/egl_vendor.json",
            "PYTHONHOME": "./inputs/python",
            "PYTHONPATH": os.pathsep.join((
                "./inputs/execution",
                f"./inputs/sources/{robocasa_source_root.name}",
                f"./inputs/sources/{robosuite_source_root.name}",
                "./inputs/python/site-packages",
            )),
        })
        execution_aliases = {
            snapshot_anchor.alias("inputs/execution"): execution_root,
            snapshot_anchor.alias(f"inputs/sources/{robocasa_source_root.name}"):
                robocasa_source_root,
            snapshot_anchor.alias(f"inputs/sources/{robosuite_source_root.name}"):
                robosuite_source_root,
            snapshot_anchor.alias("inputs/python/site-packages"):
                simulator_site_packages,
        }
        snapshot_python_identity = _file_identity(command_python, "simulator Python")
        environment_path, environment_sha, _environment = _environment_receipt(
            simulator_command=command_python,
            simulator_execution_command="./inputs/python/bin/python3.10",
            simulator_environment=simulator_environment,
            simulator_execution_environment=execution_environment,
            simulator_site_packages=simulator_site_packages,
            simulator_site_packages_provenance=args.simulator_site_packages.resolve(strict=True),
            simulator_stdlib_provenance=args.simulator_stdlib.resolve(strict=True),
            simulator_stdlib_snapshot=simulator_stdlib_snapshot,
            robocasa_source_root=robocasa_source_root,
            robocasa_source_provenance_root=args.robocasa_source_root.resolve(strict=True),
            robosuite_source_root=robosuite_source_root,
            robosuite_source_provenance_root=args.robosuite_source_root.resolve(strict=True),
            simulator_python_provenance_path=args.simulator_python.resolve(strict=True),
            simulator_python_provenance_sha256=provenance_python_identity[4],
            egl_vendor_library_provenance_path=egl_vendor_path,
            egl_vendor_library_provenance_sha256=_egl_vendor_sha,
            execution_snapshot_manifest_path=snapshot_manifest_path,
            execution_snapshot_manifest_sha256=snapshot_manifest_sha,
            source_trees=source_tree_references,
            code_commit=code_commit,
            generator_sha256=generator_sha,
            generator_snapshot_sha256=generator_snapshot_sha,
            helper_sha256=helper_sha,
            adapter_loader_sha256=adapter_loader_sha,
            v7_action_contract_sha256=v7_action_contract_sha,
            v7_contracts_sha256=v7_contracts_sha,
            action_bridge_sha256=action_bridge_sha,
            output=output_root / "simulator_environment_receipt.json",
            output_scope=output,
            execution_root=execution_root,
            execution_cwd=snapshot_anchor.alias(),
            execution_path_aliases=execution_aliases,
            pass_fds=(snapshot_anchor.fd,),
        )
        fresh_payload_root = snapshot_root / "generated/payloads"
        fresh_index_path = snapshot_root / "generated/index.jsonl"
        payload_root_pin = output.pin_directory(
            fresh_payload_root, label="fresh replay payload root"
        )
        split_pins = {
            split: output.pin_directory(
                fresh_payload_root / split, label=f"fresh replay {split} payloads"
            )
            for split in _SPLITS
        }
        command = [
            "./inputs/python/bin/python3.10", "-S",
            "./inputs/execution/scripts/generate_robocasa_stage1_planner_branches.py",
            "--candidate-index", "./inputs/selected_candidates.jsonl",
            "--action-audit", "./inputs/action_audit.json",
            "--output-root", "./generated/payloads",
            "--output-index", "./generated/index.jsonl",
            "--num-shards", "1", "--shard-index", "0", "--max-roots", "0",
            "--height", str(args.height), "--width", str(args.width),
        ]
        subprocess.run(
            command,
            cwd=snapshot_anchor.alias(),
            env=execution_environment,
            pass_fds=(snapshot_anchor.fd,),
            check=True,
        )
        if _file_identity(command_python, "simulator Python") != snapshot_python_identity:
            raise RuntimeError("simulator Python changed while replay was running")
        output.verify_pinned_directory(
            output_root_pin, label="fresh replay output root"
        )
        output.verify_pinned_directory(
            payload_root_pin, label="fresh replay payload root"
        )
        for split, pin in split_pins.items():
            output.verify_pinned_directory(
                pin, label=f"fresh replay {split} payloads"
            )
        fresh_index = fresh_index_path
        fresh_index_payload, fresh_index_sha = snapshot_anchor.read_regular(
            "generated/index.jsonl", label="fresh runtime index"
        )
        fresh_rows = {}
        for row in _rows(fresh_index_payload, "fresh runtime index"):
            root_id = row.get("root_id")
            if (
                row.get("schema") != LEGACY_RUNTIME_SCHEMA
                or type(root_id) is not str
                or root_id in fresh_rows
            ):
                raise RuntimeError("fresh runtime row schema/identity is invalid")
            fresh_rows[root_id] = row
        if set(fresh_rows) != set(selected):
            raise RuntimeError("fresh runtime root coverage differs from selection")

        authority_rows = []
        for candidate, execution_candidate in zip(chosen, execution_candidates, strict=True):
            root_id = candidate["root_id"]
            selection_row = selected[root_id]
            split = selection_row["split"]
            legacy, legacy_shard, legacy_shard_sha = legacy_rows[root_id]
            fresh = fresh_rows[root_id]
            legacy_path, legacy_payload, legacy_sha = _regular(
                Path(legacy["path"]), "legacy runtime payload"
            )
            expected_fresh_relative = (
                Path("generated") / "payloads" / split / f"{root_id}.npz"
            )
            fresh_path_value = fresh.get("path")
            if (
                type(fresh_path_value) is not str
                or Path(fresh_path_value) != expected_fresh_relative
                or Path(fresh_path_value).is_absolute()
                or ".." in Path(fresh_path_value).parts
            ):
                raise RuntimeError(
                    "fresh runtime payload path is outside the anchored output"
                )
            fresh_payload, fresh_sha = snapshot_anchor.read_regular(
                expected_fresh_relative, label="fresh runtime payload"
            )
            fresh_path = fresh_payload_root / split / f"{root_id}.npz"
            if legacy_sha != _sha(legacy["payload_sha256"], "legacy payload SHA"):
                raise RuntimeError("legacy runtime payload SHA drift")
            if fresh_sha != _sha(fresh["payload_sha256"], "fresh payload SHA"):
                raise RuntimeError("fresh runtime payload SHA drift")
            legacy_npz = _npz(legacy_payload, "legacy runtime payload")
            fresh_npz = _npz(fresh_payload, "fresh runtime payload")
            legacy_comparison = {}
            for name in _COMPARE_ARRAYS:
                if name not in legacy_npz or name not in fresh_npz:
                    raise RuntimeError(f"runtime payload lacks {name}")
                legacy_comparison[name] = bool(
                    np.array_equal(legacy_npz[name], fresh_npz[name])
                )
            candidate_path, _candidate_bytes, candidate_payload_sha = _regular(
                Path(candidate["candidate_path"]), "candidate payload"
            )
            if candidate_payload_sha != candidate["payload_sha256"]:
                raise RuntimeError("candidate payload SHA drift")
            root_context, _context_bytes, root_context_sha = _regular(
                Path(candidate["root_context_path"]), "root context"
            )
            if root_context_sha != candidate["root_context_sha256"]:
                raise RuntimeError("root context SHA drift")
            execution_source_episode = snapshot_inputs[root_id]["source_episode"]
            execution_source_episode_relative = execution_source_episode.relative_to(
                snapshot_root
            )
            _episode_bytes, execution_source_episode_sha = snapshot_anchor.read_regular(
                execution_source_episode_relative,
                label="execution source episode",
            )
            source_episode = Path(selection_row["source_episode_path"])
            source_episode_sha = selection_row["source_episode_sha256"]
            if execution_source_episode_sha != source_episode_sha:
                raise RuntimeError("execution/provenance source episode SHA mismatch")
            branches = _branch_rows(fresh_npz)
            row = {
                "schema": REPLAY_AUTHORITY_ROW_SCHEMA,
                "split": split,
                "source": selection_row["source"],
                "root_id": root_id,
                "episode_id": int(candidate["episode_id"]),
                "episode_root_index": int(candidate["episode_root_index"]),
                "t0": int(candidate["t0"]),
                "source_dataset_path": selection_row["source_dataset_path"],
                "states_path": selection_row["states_path"],
                "states_sha256": selection_row["states_sha256"],
                "model_xml_gz_path": selection_row["model_xml_gz_path"],
                "model_xml_gz_sha256": selection_row["model_xml_gz_sha256"],
                "ep_meta_path": selection_row["ep_meta_path"],
                "ep_meta_file_sha256": selection_row["ep_meta_file_sha256"],
                "dataset_meta_path": selection_row["dataset_meta_path"],
                "dataset_meta_sha256": selection_row["dataset_meta_sha256"],
                "modality_path": selection_row["modality_path"],
                "modality_sha256": selection_row["modality_sha256"],
                "candidate_seed": int(candidate["candidate_seed"]),
                "candidate_index_row_sha256": canonical_sha256(candidate),
                "execution_candidate_index_row_sha256": canonical_sha256(
                    execution_candidate
                ),
                "candidate_payload_path": str(candidate_path),
                "candidate_payload_sha256": candidate_payload_sha,
                "execution_candidate_payload_path": str(
                    snapshot_inputs[root_id]["candidate_payload"]
                ),
                "execution_candidate_payload_sha256": candidate_payload_sha,
                "root_context_path": str(root_context),
                "root_context_sha256": root_context_sha,
                "execution_root_context_path": str(
                    snapshot_inputs[root_id]["root_context"]
                ),
                "execution_root_context_sha256": root_context_sha,
                "source_episode_path": str(source_episode),
                "source_episode_sha256": source_episode_sha,
                "execution_source_dataset_path": str(
                    snapshot_inputs[root_id]["source_dataset"]
                ),
                "execution_source_episode_path": str(execution_source_episode),
                "execution_source_episode_sha256": execution_source_episode_sha,
                "execution_states_path": str(snapshot_inputs[root_id]["states"]),
                "execution_states_sha256": selection_row["states_sha256"],
                "execution_model_xml_gz_path": str(
                    snapshot_inputs[root_id]["model_xml_gz"]
                ),
                "execution_model_xml_gz_sha256": selection_row["model_xml_gz_sha256"],
                "execution_ep_meta_path": str(snapshot_inputs[root_id]["ep_meta"]),
                "execution_ep_meta_file_sha256": selection_row["ep_meta_file_sha256"],
                "execution_dataset_meta_path": str(
                    snapshot_inputs[root_id]["dataset_meta"]
                ),
                "execution_dataset_meta_sha256": selection_row["dataset_meta_sha256"],
                "execution_modality_path": str(snapshot_inputs[root_id]["modality"]),
                "execution_modality_sha256": selection_row["modality_sha256"],
                "source_manifest_path": selection_row["source_manifest_path"],
                "source_manifest_sha256": selection_row["source_manifest_sha256"],
                "source_manifest_row_sha256": selection_row[
                    "source_manifest_row_sha256"
                ],
                "legacy_runtime_index_shard_path": str(legacy_shard),
                "legacy_runtime_index_shard_sha256": legacy_shard_sha,
                "legacy_runtime_index_row_sha256": canonical_sha256(legacy),
                "legacy_runtime_payload_path": str(legacy_path),
                "legacy_runtime_payload_sha256": legacy_sha,
                "fresh_runtime_index_row_sha256": canonical_sha256(fresh),
                "fresh_runtime_payload_path": str(fresh_path),
                "fresh_runtime_payload_sha256": fresh_sha,
                "stage0_checkpoint_sha256": _sha(
                    candidate["stage0_checkpoint_sha256"], "Stage0 checkpoint SHA"
                ),
                "candidate_count": int(fresh["branches"]),
                "future_frames": int(fresh["future_frames"]),
                "root_state_sha256": array_sha256(fresh_npz["root_state"]),
                "root_render_state_sha256": str(fresh_npz["root_render_state_sha256"].item()),
                "root_rgb_sha256": array_sha256(fresh_npz["root_rgb"]),
                "executed_actions_sha256": array_sha256(fresh_npz["simulator_actions"]),
                "branch_rgb_sha256": array_sha256(fresh_npz["branch_rgb"]),
                "branch_rewards_sha256": array_sha256(fresh_npz["branch_rewards"]),
                "branch_dones_sha256": array_sha256(fresh_npz["branch_dones"]),
                "branch_success_sha256": array_sha256(fresh_npz["branch_success"]),
                "branch_roles": [str(value) for value in fresh_npz["branch_roles"].tolist()],
                "simulator_action_low_sha256": array_sha256(fresh_npz["simulator_action_low"]),
                "simulator_action_high_sha256": array_sha256(fresh_npz["simulator_action_high"]),
                "model_xml_sha256": selection_row["model_xml_sha256"],
                "ep_meta_sha256": selection_row["ep_meta_sha256"],
                "root_rgb_equivalence_contract": str(
                    fresh_npz["root_rgb_equivalence_contract"].item()
                ),
                "root_rgb_changed_fraction_sha256": array_sha256(
                    fresh_npz["root_rgb_changed_fraction"]
                ),
                "root_rgb_mean_abs_sha256": array_sha256(fresh_npz["root_rgb_mean_abs"]),
                "root_rgb_rmse_sha256": array_sha256(fresh_npz["root_rgb_rmse"]),
                "root_rgb_p99_abs_sha256": array_sha256(fresh_npz["root_rgb_p99_abs"]),
                "root_rgb_psnr_db_sha256": array_sha256(fresh_npz["root_rgb_psnr_db"]),
                "branches": branches,
                "branches_sha256": canonical_sha256(branches),
                "legacy_comparison_diagnostic": {
                    "available": True,
                    "all_core_equal": all(legacy_comparison.values()),
                    "fields": legacy_comparison,
                },
                "candidate_actions_executed_exact": bool(
                    fresh["candidate_actions_executed_exact"]
                ),
                "same_root_simulator_state_exact": bool(
                    fresh_npz["same_root_simulator_state_exact"].item()
                ),
                "same_root_render_state_exact": bool(
                    fresh_npz["same_root_render_state_exact"].item()
                ),
                "same_root_rgb_exact": bool(
                    fresh_npz["root_rgb_equivalence_all_passed"].item()
                    and fresh_npz["same_root_rgb_canonicalized"].item()
                ),
                "real_simulator_outcomes": fresh["pseudo_outcomes"] is False,
            }
            if set(row) != REPLAY_AUTHORITY_ROW_FIELDS:
                raise AssertionError("internal replay authority row fields drifted")
            authority_rows.append(row)
        authority_rows.sort(key=lambda row: (_SPLITS.index(row["split"]), row["root_id"]))
        counts = {
            split: sum(row["split"] == split for row in authority_rows)
            for split in _SPLITS
        }
        authority = {
            "schema": REPLAY_AUTHORITY_SCHEMA,
            "code_commit": code_commit,
            "simulator_environment_receipt_path": str(environment_path),
            "simulator_environment_receipt_sha256": environment_sha,
            "runtime_root": str(runtime_root),
            "runtime_generator_path": str(generator),
            "runtime_generator_sha256": generator_sha,
            "replay_helper_path": str(helper),
            "replay_helper_sha256": helper_sha,
            "adapter_loader_path": str(adapter_loader),
            "adapter_loader_sha256": adapter_loader_sha,
            "action_audit_path": str(action_audit),
            "action_audit_sha256": action_sha,
            "action_audit_snapshot_path": str(action_audit_snapshot),
            "action_audit_snapshot_sha256": action_sha,
            "execution_snapshot_manifest_path": str(snapshot_manifest_path),
            "execution_snapshot_manifest_sha256": snapshot_manifest_sha,
            "candidate_index_path": str(candidate_index),
            "candidate_index_sha256": candidate_sha,
            "candidate_index_seal_path": str(candidate_seal),
            "candidate_index_seal_sha256": candidate_seal_sha,
            "selected_candidate_index_path": str(selection_path),
            "selected_candidate_index_sha256": hashlib.sha256(
                selection_payload
            ).hexdigest(),
            "selection_manifest_path": str(selection_manifest),
            "selection_manifest_sha256": selection_manifest_sha,
            "fresh_runtime_root": str(fresh_payload_root.absolute()),
            "fresh_runtime_index_path": str(fresh_index),
            "fresh_runtime_index_sha256": fresh_index_sha,
            "selection_count": counts,
            "candidate_count": next(iter(fresh_rows.values()))["branches"],
            "future_frames": next(iter(fresh_rows.values()))["future_frames"],
            "rows": authority_rows,
            "rows_sha256": canonical_sha256(authority_rows),
            "passed": True,
        }
        if set(authority) != REPLAY_AUTHORITY_FIELDS:
            raise AssertionError("internal replay authority fields drifted")
        validate_replay_authority(
            authority,
            expected_code_commit=code_commit,
            verify_referents=True,
        )
        receipt_payload = (json.dumps(authority, sort_keys=True, indent=2) + "\n").encode()
        output.publish(args.output.absolute(), receipt_payload, label="replay authority")
        output._verify_namespace()
        receipt_sha = hashlib.sha256(receipt_payload).hexdigest()
        print(json.dumps({
            "schema": REPLAY_AUTHORITY_SCHEMA,
            "output": str(args.output.absolute()),
            "sha256": receipt_sha,
            "selection_count": counts,
            "candidate_count": authority["candidate_count"],
            "passed": True,
        }, sort_keys=True))
        immutable_inputs.verify(
            target=snapshot_anchor.alias("inputs"),
            pass_fds=(snapshot_anchor.fd,),
        )
    finally:
        try:
            immutable_inputs.close(
                target=snapshot_anchor.alias("inputs"),
                pass_fds=(snapshot_anchor.fd,),
            )
        finally:
            snapshot_anchor.close()
            output.close()


if __name__ == "__main__":
    main()
