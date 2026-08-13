"""Independent, fail-closed authority for real RoboCasa Stage1 replays.

The legacy runtime is useful supervision, but it is not its own execution
authority.  This contract binds a fresh invocation of the pinned simulator
producer and verifies its tensors against the sealed legacy payloads.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from wm3d_v3.data.manifest_contract import SHA256_RE, canonical_sha256
from wm3d_v3.stage1_planner.rollout_audit import (
    RolloutAuditError,
    read_regular_bytes,
)
from wm3d_v3.stage1_planner.execution_snapshot import (
    validate_execution_snapshot,
)


REPLAY_AUTHORITY_SCHEMA = "wm3d_v8_robocasa_stage1_replay_authority_v3"
REPLAY_AUTHORITY_ROW_SCHEMA = "wm3d_v8_robocasa_stage1_replay_authority_row_v3"
REPLAY_ENVIRONMENT_SCHEMA = "wm3d_v8_robocasa_stage1_replay_environment_v3"
REPLAY_SOURCE_TREE_SCHEMA = "wm3d_v8_robocasa_stage1_source_tree_v2"
PINNED_RUNTIME_GENERATOR_SHA256 = (
    "24bfe1cf34be7813d298713efc33ca49dfc9d81e77bdea5530710fcbf2ccfdba"
)
PINNED_RUNTIME_GENERATOR_SNAPSHOT_SHA256 = (
    "12ed0116fe73e0babc5282c4adb8dd866b64c9e159b9ad1b69d1b6788921895d"
)
PINNED_REPLAY_HELPER_SHA256 = (
    "cf93e5e5434fe65112a1f266b62afe0e47ccf732c61204fdd912e753ffe9d3e9"
)
PINNED_ADAPTER_LOADER_SHA256 = (
    "e39287c64dd4e9d0348522770c995a335bf3631093da54146e1df5cae72cd29a"
)
PINNED_ACTION_AUDIT_SHA256 = (
    "f0e3c99f5f5792d996473bf47035d989ebe17ff2665a3c5c07c73ea736a3c27f"
)
PINNED_V7_ACTION_CONTRACT_SHA256 = (
    "f6d1cdd4f5f52471c1b5dee349aee2379191437a10e962ee0e883fc1c620a702"
)
PINNED_V7_CONTRACTS_SHA256 = (
    "c8a9abcac096a6f2cbc8e939da9e870cba835ecf2f7af0c5f408628063db9684"
)
PINNED_ACTION_BRIDGE_SHA256 = (
    "14e91bdeb40926a7044c25ac65e682feed67141c3aa1386dd3146e61867b88d4"
)
PINNED_SOURCE_TREE_SHA256 = {
    "robocasa": "3101ac6ed711756a80bf624a4063d5443c2493b4c69efd91866726a414ef9078",
    "robosuite": "1a7eea7868f4dbcc26a2388213e7dfcf998bb427cd21f45dc9df33f452cb18f9",
}

REPLAY_AUTHORITY_FIELDS = {
    "schema", "code_commit", "simulator_environment_receipt_path",
    "simulator_environment_receipt_sha256", "runtime_root",
    "runtime_generator_path", "runtime_generator_sha256", "replay_helper_path",
    "replay_helper_sha256", "adapter_loader_path", "adapter_loader_sha256",
    "action_audit_path", "action_audit_sha256",
    "action_audit_snapshot_path", "action_audit_snapshot_sha256",
    "execution_snapshot_manifest_path", "execution_snapshot_manifest_sha256",
    "candidate_index_path", "candidate_index_sha256",
    "candidate_index_seal_path", "candidate_index_seal_sha256",
    "selected_candidate_index_path", "selected_candidate_index_sha256",
    "selection_manifest_path", "selection_manifest_sha256",
    "fresh_runtime_root", "fresh_runtime_index_path", "fresh_runtime_index_sha256",
    "selection_count", "candidate_count", "future_frames", "rows", "rows_sha256",
    "passed",
}
REPLAY_SELECTION_SCHEMA = "wm3d_v8_robocasa_stage1_replay_selection_v2"
REPLAY_SELECTION_FIELDS = {
    "schema", "code_commit", "code_repo_path", "selection_policy_path",
    "selection_policy_sha256", "data_profile_path", "data_profile_sha256",
    "candidate_index_path", "candidate_index_sha256",
    "candidate_index_seal_path", "candidate_index_seal_sha256",
    "selection_count", "rows", "rows_sha256", "passed",
}
REPLAY_SELECTION_ROW_FIELDS = {
    "split", "source", "root_id", "episode_id", "episode_root_index", "t0",
    "source_dataset_path",
    "source_manifest_path", "source_manifest_sha256",
    "source_manifest_row_sha256", "source_episode_path",
    "source_episode_sha256", "states_path", "states_sha256",
    "model_xml_gz_path", "model_xml_gz_sha256", "model_xml_sha256",
    "ep_meta_path", "ep_meta_file_sha256", "ep_meta_sha256",
    "dataset_meta_path", "dataset_meta_sha256",
    "modality_path", "modality_sha256", "candidate_index_row_sha256",
    "candidate_payload_path", "candidate_payload_sha256",
    "root_context_path", "root_context_sha256",
}
REPLAY_ENVIRONMENT_FIELDS = {
    "schema", "code_commit", "simulator_python_path", "simulator_python_sha256",
    "simulator_python_device", "simulator_python_inode", "simulator_python_size",
    "simulator_python_mtime_ns", "simulator_pythonpath",
    "simulator_pythonhome",
    "python_version", "cuda_visible_devices", "mujoco_gl",
    "execution_snapshot_manifest_path", "execution_snapshot_manifest_sha256",
    "simulator_python_provenance_path", "simulator_python_provenance_sha256",
    "simulator_stdlib_provenance_root", "simulator_stdlib_snapshot_root",
    "egl_vendor_library_path", "egl_vendor_library_sha256", "pip_freeze_path",
    "egl_vendor_library_provenance_path",
    "egl_vendor_library_provenance_sha256",
    "pip_freeze_sha256", "simulator_site_packages_path",
    "simulator_site_packages_provenance_path",
    "robocasa_source_root", "robocasa_source_commit",
    "robocasa_source_provenance_root",
    "robosuite_source_root", "robosuite_source_commit", "source_trees",
    "robosuite_source_provenance_root",
    "modules", "snapshot_modules", "runtime_generator_sha256",
    "runtime_generator_snapshot_sha256", "replay_helper_sha256",
    "adapter_loader_sha256", "v7_action_contract_sha256",
    "v7_contracts_sha256", "action_bridge_sha256", "passed",
}
REPLAY_AUTHORITY_ROW_FIELDS = {
    "schema", "split", "source", "root_id", "episode_id", "t0", "candidate_seed",
    "candidate_index_row_sha256", "candidate_payload_path", "candidate_payload_sha256",
    "execution_candidate_index_row_sha256",
    "execution_candidate_payload_path", "execution_candidate_payload_sha256",
    "root_context_path", "root_context_sha256", "source_episode_path",
    "execution_root_context_path", "execution_root_context_sha256",
    "execution_source_dataset_path", "execution_source_episode_path",
    "execution_source_episode_sha256", "execution_states_path",
    "execution_states_sha256", "execution_model_xml_gz_path",
    "execution_model_xml_gz_sha256", "execution_ep_meta_path",
    "execution_ep_meta_file_sha256", "execution_dataset_meta_path",
    "execution_dataset_meta_sha256", "execution_modality_path",
    "execution_modality_sha256",
    "source_episode_sha256", "source_manifest_path", "source_manifest_sha256",
    "source_manifest_row_sha256", "legacy_runtime_index_shard_path",
    "legacy_runtime_index_shard_sha256", "legacy_runtime_index_row_sha256",
    "legacy_runtime_payload_path", "legacy_runtime_payload_sha256",
    "fresh_runtime_index_row_sha256", "fresh_runtime_payload_path",
    "fresh_runtime_payload_sha256", "stage0_checkpoint_sha256", "candidate_count",
    "future_frames", "root_state_sha256", "root_render_state_sha256",
    "root_rgb_sha256", "executed_actions_sha256", "branch_rgb_sha256",
    "branch_rewards_sha256", "branch_dones_sha256", "branch_success_sha256",
    "legacy_comparison_diagnostic",
    "branch_roles", "simulator_action_low_sha256", "simulator_action_high_sha256",
    "source_dataset_path", "episode_root_index", "states_path", "states_sha256",
    "model_xml_gz_path", "model_xml_gz_sha256", "model_xml_sha256",
    "ep_meta_path", "ep_meta_file_sha256", "ep_meta_sha256",
    "dataset_meta_path", "dataset_meta_sha256", "modality_path", "modality_sha256",
    "root_rgb_equivalence_contract",
    "root_rgb_changed_fraction_sha256", "root_rgb_mean_abs_sha256",
    "root_rgb_rmse_sha256", "root_rgb_p99_abs_sha256", "root_rgb_psnr_db_sha256",
    "branches", "branches_sha256",
    "candidate_actions_executed_exact", "same_root_simulator_state_exact",
    "same_root_render_state_exact", "same_root_rgb_exact", "real_simulator_outcomes",
}
REPLAY_BRANCH_FIELDS = {
    "index", "role", "root_state_sha256", "root_render_state_sha256",
    "root_rgb_raw_sha256", "executed_action_sha256", "rgb_sha256",
    "reward_sha256", "done_sha256", "success_sha256", "terminal_success",
    "max_reward",
}
_SPLITS = {"train", "val", "test"}
_MODULES = {"numpy", "mujoco", "robosuite", "robocasa"}
_SOURCE_TREE_FIELDS = {
    "schema", "root", "package", "commit", "rows", "rows_sha256", "file_count",
    "total_bytes", "passed",
}
_SOURCE_TREE_ROW_FIELDS = {"path", "size", "sha256"}
_NPZ_ARRAYS = {
    "schema", "root_id", "branch_roles", "simulator_actions",
    "branch_actions_physical", "branch_actions_requested_physical",
    "action_history_physical", "root_state", "root_rgb", "branch_rgb",
    "root_render_state_sha256", "root_render_state_branch_sha256",
    "root_rgb_raw_sha256", "root_rgb_equivalence_contract",
    "root_rgb_canonical_source", "root_rgb_changed_fraction",
    "root_rgb_mean_abs", "root_rgb_rmse", "root_rgb_p99_abs",
    "root_rgb_psnr_db", "same_root_simulator_state_exact",
    "same_root_render_state_exact", "root_rgb_equivalence_all_passed",
    "same_root_rgb_canonicalized", "model_xml_sha256", "ep_meta_sha256",
    "branch_rewards", "branch_dones", "branch_success",
    "stage0_checkpoint_sha256", "root_context_sha256",
    "simulator_action_low", "simulator_action_high",
    "h32_factual_available", "counterfactual_pose_space",
}
_LEGACY_COMPARISON_FIELDS = {
    "simulator_actions", "root_state", "root_rgb", "branch_rgb",
    "branch_rewards", "branch_dones", "branch_success",
}


class ReplayAuthorityError(RolloutAuditError):
    pass


def _sha(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ReplayAuthorityError(f"{label} must be a lowercase SHA256 string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayAuthorityError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayAuthorityError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayAuthorityError(f"{label} must be a finite JSON number")
    return result


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{array.shape}".encode("ascii")
    import hashlib

    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _load_npz(payload: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, KeyError) as error:
        raise ReplayAuthorityError(f"{label} is not a valid no-pickle NPZ") from error


def _verify_path(path: object, expected: object, label: str) -> tuple[Path, bytes]:
    if type(path) is not str or not path:
        raise ReplayAuthorityError(f"{label} path must be a non-empty string")
    resolved, payload, observed = read_regular_bytes(Path(path), label)
    if observed != _sha(expected, f"{label} SHA"):
        raise ReplayAuthorityError(f"{label} SHA mismatch")
    return resolved, payload


def _verify_selection_row_referents(row: dict[str, Any]) -> None:
    """Verify selection inputs against their raw-file digests.

    ``ep_meta_sha256`` is the canonical JSON semantic digest consumed by the
    simulator contract, while ``ep_meta_file_sha256`` seals the bytes at
    ``ep_meta_path``.  Path verification must use the latter.
    """

    digest_fields = {
        "source_manifest": "source_manifest_sha256",
        "source_episode": "source_episode_sha256",
        "states": "states_sha256",
        "model_xml_gz": "model_xml_gz_sha256",
        "ep_meta": "ep_meta_file_sha256",
        "dataset_meta": "dataset_meta_sha256",
        "modality": "modality_sha256",
        "candidate_payload": "candidate_payload_sha256",
        "root_context": "root_context_sha256",
    }
    for name, digest_field in digest_fields.items():
        _verify_path(
            row[f"{name}_path"], row[digest_field], f"selection row {name}"
        )


def _validate_source_tree(
    reference: object,
    *,
    name: str,
    expected_root: Path,
    expected_package: str,
    expected_commit: str,
    verify_referents: bool,
) -> None:
    if not isinstance(reference, dict) or set(reference) != {
        "manifest_path", "manifest_sha256", "tree_sha256"
    }:
        raise ReplayAuthorityError(f"replay {name} source-tree reference mismatch")
    tree_sha = _sha(reference["tree_sha256"], f"replay {name} tree SHA")
    if tree_sha != PINNED_SOURCE_TREE_SHA256[name]:
        raise ReplayAuthorityError(f"replay {name} source-tree SHA is not pinned")
    if not verify_referents:
        return
    _path, payload = _verify_path(
        reference["manifest_path"], reference["manifest_sha256"],
        f"replay {name} source-tree manifest",
    )
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayAuthorityError(f"replay {name} source-tree manifest invalid") from error
    if not isinstance(manifest, dict) or set(manifest) != _SOURCE_TREE_FIELDS:
        raise ReplayAuthorityError(f"replay {name} source-tree fields mismatch")
    rows = manifest["rows"]
    if (
        manifest["schema"] != REPLAY_SOURCE_TREE_SCHEMA
        or manifest["passed"] is not True
        or manifest["root"] != str(expected_root)
        or manifest["package"] != expected_package
        or manifest["commit"] != expected_commit
        or not isinstance(rows, list)
        or not rows
        or type(manifest["file_count"]) is not int
        or manifest["file_count"] != len(rows)
        or type(manifest["total_bytes"]) is not int
        or manifest["total_bytes"] < 0
        or manifest["rows_sha256"] != tree_sha
        or canonical_sha256(rows) != tree_sha
    ):
        raise ReplayAuthorityError(f"replay {name} source-tree closure mismatch")
    root = expected_root / expected_package
    if (
        expected_root.is_symlink()
        or root.is_symlink()
        or not expected_root.is_dir()
        or not root.is_dir()
    ):
        raise ReplayAuthorityError(f"replay {name} source-tree root invalid")
    observed_paths: set[str] = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != _SOURCE_TREE_ROW_FIELDS:
            raise ReplayAuthorityError(f"replay {name} source-tree row mismatch")
        relative = row["path"]
        if (
            type(relative) is not str
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in observed_paths
            or type(row["size"]) is not int
            or row["size"] < 0
        ):
            raise ReplayAuthorityError(f"replay {name} source-tree row invalid")
        observed_paths.add(relative)
        current = root
        for part in Path(relative).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ReplayAuthorityError(
                    f"replay {name} source-tree contains a directory symlink"
                )
        _file, file_payload = _verify_path(
            str(root / relative), row["sha256"],
            f"replay {name} source-tree file",
        )
        if len(file_payload) != row["size"]:
            raise ReplayAuthorityError(f"replay {name} source-tree size mismatch")
        total += len(file_payload)
    current_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if current_paths != observed_paths or total != manifest["total_bytes"]:
        raise ReplayAuthorityError(f"replay {name} source-tree membership mismatch")


def _validate_environment(
    value: object,
    *,
    code_commit: str,
    generator_sha256: str,
    generator_snapshot_sha256: str,
    helper_sha256: str,
    adapter_loader_sha256: str,
    v7_action_contract_sha256: str,
    v7_contracts_sha256: str,
    action_bridge_sha256: str,
    verify_referents: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != REPLAY_ENVIRONMENT_FIELDS:
        raise ReplayAuthorityError("replay environment fields mismatch")
    if value["schema"] != REPLAY_ENVIRONMENT_SCHEMA or value["passed"] is not True:
        raise ReplayAuthorityError("replay environment schema/pass mismatch")
    if value["code_commit"] != code_commit:
        raise ReplayAuthorityError("replay environment code commit mismatch")
    if value["runtime_generator_sha256"] != generator_sha256:
        raise ReplayAuthorityError("replay environment generator SHA mismatch")
    if value["runtime_generator_snapshot_sha256"] != _sha(
        generator_snapshot_sha256, "runtime generator snapshot SHA"
    ):
        raise ReplayAuthorityError("replay environment generator snapshot SHA mismatch")
    if generator_snapshot_sha256 != PINNED_RUNTIME_GENERATOR_SNAPSHOT_SHA256:
        raise ReplayAuthorityError("replay environment generator snapshot is unpinned")
    if value["replay_helper_sha256"] != helper_sha256:
        raise ReplayAuthorityError("replay environment helper SHA mismatch")
    if value["adapter_loader_sha256"] != adapter_loader_sha256:
        raise ReplayAuthorityError("replay environment adapter-loader SHA mismatch")
    for field, expected in (
        ("v7_action_contract_sha256", v7_action_contract_sha256),
        ("v7_contracts_sha256", v7_contracts_sha256),
        ("action_bridge_sha256", action_bridge_sha256),
    ):
        if value[field] != _sha(expected, field):
            raise ReplayAuthorityError(f"replay environment {field} mismatch")
    for field in ("python_version", "cuda_visible_devices", "mujoco_gl"):
        if type(value[field]) is not str or not value[field]:
            raise ReplayAuthorityError(f"replay environment {field} is invalid")
    if verify_referents:
        _manifest_path, manifest_payload = _verify_path(
            value["execution_snapshot_manifest_path"],
            value["execution_snapshot_manifest_sha256"],
            "execution snapshot manifest",
        )
        try:
            snapshot_manifest = json.loads(manifest_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayAuthorityError("execution snapshot manifest invalid") from error
        validate_execution_snapshot(
            snapshot_manifest, verify_provenance=True, verify_snapshots=True
        )
        snapshot_rows_by_path = {
            row["snapshot_path"]: row for row in snapshot_manifest["rows"]
        }
        _verify_path(
            value["simulator_python_provenance_path"],
            value["simulator_python_provenance_sha256"],
            "simulator Python provenance",
        )
        _verify_path(
            value["egl_vendor_library_provenance_path"],
            value["egl_vendor_library_provenance_sha256"],
            "EGL vendor library provenance",
        )
        _verify_path(
            value["egl_vendor_library_path"],
            value["egl_vendor_library_sha256"],
            "EGL vendor library manifest",
        )
    else:
        snapshot_manifest = None
        snapshot_rows_by_path = {}
        _sha(value["egl_vendor_library_sha256"], "EGL vendor manifest SHA")
    resolved_directories: dict[str, Path] = {}
    for field in (
        "simulator_site_packages_path", "simulator_site_packages_provenance_path",
        "simulator_stdlib_provenance_root", "simulator_stdlib_snapshot_root",
        "robocasa_source_root", "robocasa_source_provenance_root",
        "robosuite_source_root", "robosuite_source_provenance_root",
    ):
        if type(value[field]) is not str or not value[field]:
            raise ReplayAuthorityError(f"replay environment {field} is invalid")
        path = Path(value[field])
        if verify_referents and (path.is_symlink() or not path.is_dir()):
            raise ReplayAuthorityError(f"replay environment {field} is not a real directory")
        if verify_referents:
            resolved_directories[field] = path.resolve(strict=True)
    if type(value["simulator_pythonhome"]) is not str or not value[
        "simulator_pythonhome"
    ]:
        raise ReplayAuthorityError("replay environment simulator PYTHONHOME is invalid")
    if verify_referents:
        pythonhome = Path(value["simulator_pythonhome"])
        if pythonhome.is_symlink() or not pythonhome.is_dir():
            raise ReplayAuthorityError(
                "replay environment simulator PYTHONHOME is not a real directory"
            )
        pythonhome = pythonhome.resolve(strict=True)
        assert snapshot_manifest is not None
        snapshot_root = Path(snapshot_manifest["root"]).resolve(strict=True)
        if pythonhome != snapshot_root / "python":
            raise ReplayAuthorityError(
                "replay environment simulator PYTHONHOME is not the snapshot"
            )
    expected_sources = {
        "robocasa": (
            "robocasa_source_root", "robocasa_source_commit", "robocasa",
            "8f3c96ec8d1bfcd8126cad2bca887da98d30e997",
        ),
        "robosuite": (
            "robosuite_source_root", "robosuite_source_commit", "robosuite",
            "6c10ef24a4bb52f59199976125060ce793470e6e",
        ),
    }
    for name, (root_field, commit_field, _package, expected_commit) in (
        expected_sources.items()
    ):
        if value[commit_field] != expected_commit:
            raise ReplayAuthorityError(
                f"replay environment {name} source commit is not pinned"
            )
        if verify_referents and resolved_directories[root_field].name != (
            f"{name}-{expected_commit}"
        ):
            raise ReplayAuthorityError(
                f"replay environment {name} source root is not pinned"
            )
        if verify_referents and resolved_directories[root_field] != (
            snapshot_root / "sources" / f"{name}-{expected_commit}"
        ):
            raise ReplayAuthorityError(
                f"replay environment {name} source root is outside snapshot"
            )
    source_trees = value["source_trees"]
    if not isinstance(source_trees, dict) or set(source_trees) != {
        "robocasa", "robosuite"
    }:
        raise ReplayAuthorityError("replay environment source-tree closure mismatch")
    for name, reference in source_trees.items():
        root_field, _commit_field, package, expected_commit = expected_sources[name]
        _validate_source_tree(
            reference,
            name=name,
            expected_root=resolved_directories.get(
                root_field, Path(value[root_field])
            ),
            expected_package=package,
            expected_commit=expected_commit,
            verify_referents=verify_referents,
        )
    _sha(value["pip_freeze_sha256"], "pip freeze SHA")
    if verify_referents:
        _verify_path(
            value["pip_freeze_path"], value["pip_freeze_sha256"], "pip freeze"
        )
    _sha(value["simulator_python_sha256"], "simulator Python SHA")
    for field in (
        "simulator_python_device", "simulator_python_inode",
        "simulator_python_size", "simulator_python_mtime_ns",
    ):
        _integer(value[field], f"replay environment {field}")
    if type(value["simulator_pythonpath"]) is not str:
        raise ReplayAuthorityError("replay environment simulator PYTHONPATH is invalid")
    if verify_referents:
        python_path, python_payload = _verify_path(
            value["simulator_python_path"],
            value["simulator_python_sha256"],
            "simulator Python",
        )
        python_stat = python_path.stat(follow_symlinks=False)
        expected_python_identity = (
            value["simulator_python_device"], value["simulator_python_inode"],
            value["simulator_python_size"], value["simulator_python_mtime_ns"],
        )
        if (
            python_path.is_symlink()
            or len(python_payload) != value["simulator_python_size"]
            or (
                python_stat.st_dev, python_stat.st_ino, python_stat.st_size,
                python_stat.st_mtime_ns,
            ) != expected_python_identity
        ):
            raise ReplayAuthorityError("replay simulator Python identity mismatch")
        if python_path != pythonhome / "bin/python3.10":
            raise ReplayAuthorityError(
                "replay simulator Python is outside snapshot PYTHONHOME"
            )
        for path, digest, kind in (
            (
                value["simulator_python_path"],
                value["simulator_python_sha256"],
                "simulator Python",
            ),
            (
                value["egl_vendor_library_path"],
                value["egl_vendor_library_sha256"],
                "EGL vendor manifest",
            ),
        ):
            manifest_row = snapshot_rows_by_path.get(path)
            if (
                manifest_row is None
                or manifest_row["snapshot_sha256"] != digest
                or manifest_row["kind"] != kind
            ):
                raise ReplayAuthorityError(
                    f"replay {kind} is not bound by the execution snapshot"
                )
    modules = value["modules"]
    if not isinstance(modules, dict) or set(modules) != _MODULES:
        raise ReplayAuthorityError("replay environment module closure mismatch")
    for name, module in modules.items():
        if not isinstance(module, dict) or set(module) != {"version", "path", "sha256"}:
            raise ReplayAuthorityError(f"replay environment {name} fields mismatch")
        if type(module["version"]) is not str or not module["version"]:
            raise ReplayAuthorityError(f"replay environment {name} version invalid")
        _sha(module["sha256"], f"replay environment {name} SHA")
        if verify_referents:
            _verify_path(module["path"], module["sha256"], f"replay module {name}")
    snapshot_modules = value["snapshot_modules"]
    expected_snapshots = {
        "runtime_generator", "replay_helper", "adapter_loader",
        "v7_action_contract", "v7_contracts", "action_bridge",
    }
    if (
        not isinstance(snapshot_modules, dict)
        or set(snapshot_modules) != expected_snapshots
    ):
        raise ReplayAuthorityError("replay snapshot module closure mismatch")
    for name, module in snapshot_modules.items():
        if not isinstance(module, dict) or set(module) != {"path", "sha256"}:
            raise ReplayAuthorityError(f"replay snapshot {name} fields mismatch")
        _sha(module["sha256"], f"replay snapshot {name} SHA")
        if verify_referents:
            _verify_path(module["path"], module["sha256"], f"replay snapshot {name}")
    if verify_referents:
        execution_root = Path(
            snapshot_modules["runtime_generator"]["path"]
        ).parents[1]
        expected_pythonpath = ":".join((
            str(execution_root),
            str(resolved_directories["robocasa_source_root"]),
            str(resolved_directories["robosuite_source_root"]),
            str(resolved_directories["simulator_site_packages_path"]),
        ))
        if value["simulator_pythonpath"] != expected_pythonpath:
            raise ReplayAuthorityError("replay simulator PYTHONPATH is not exact")
        if (
            resolved_directories["simulator_site_packages_path"]
            != pythonhome / "site-packages"
            or resolved_directories["simulator_stdlib_snapshot_root"]
            != pythonhome / "lib/python3.10"
        ):
            raise ReplayAuthorityError(
                "replay Python library roots are not the execution snapshot"
            )
        expected_snapshot_paths = {
            "runtime_generator": execution_root / "scripts/generate_robocasa_stage1_planner_branches.py",
            "replay_helper": execution_root / "scripts/generate_robocasa_same_root_cf.py",
            "adapter_loader": execution_root / "scripts/robocasa_stage1_adapter_loader.py",
            "v7_action_contract": execution_root / "wm3d_v3/data/v7_action_contract.py",
            "v7_contracts": execution_root / "wm3d_v3/data/v7_contracts.py",
            "action_bridge": execution_root / "wm3d_v3/stage1_planner/action_bridge.py",
        }
        expected_snapshot_shas = {
            "runtime_generator": generator_snapshot_sha256,
            "replay_helper": helper_sha256,
            "adapter_loader": adapter_loader_sha256,
            "v7_action_contract": v7_action_contract_sha256,
            "v7_contracts": v7_contracts_sha256,
            "action_bridge": action_bridge_sha256,
        }
        for name in expected_snapshots:
            if (
                Path(snapshot_modules[name]["path"]) != expected_snapshot_paths[name]
                or snapshot_modules[name]["sha256"] != expected_snapshot_shas[name]
            ):
                raise ReplayAuthorityError(
                    f"replay snapshot {name} path/SHA is not the pinned execution input"
                )
    return value


def validate_replay_environment(
    value: object,
    *,
    expected_code_commit: str,
    runtime_generator_sha256: str,
    runtime_generator_snapshot_sha256: str,
    replay_helper_sha256: str,
    adapter_loader_sha256: str,
    v7_action_contract_sha256: str,
    v7_contracts_sha256: str,
    action_bridge_sha256: str,
    verify_referents: bool = True,
) -> dict[str, Any]:
    """Public strict validator used by the replay orchestration command."""
    return _validate_environment(
        value,
        code_commit=expected_code_commit,
        generator_sha256=_sha(
            runtime_generator_sha256, "runtime generator SHA"
        ),
        generator_snapshot_sha256=_sha(
            runtime_generator_snapshot_sha256,
            "runtime generator snapshot SHA",
        ),
        helper_sha256=_sha(replay_helper_sha256, "replay helper SHA"),
        adapter_loader_sha256=_sha(
            adapter_loader_sha256, "adapter loader SHA"
        ),
        v7_action_contract_sha256=_sha(
            v7_action_contract_sha256, "V7 action contract SHA"
        ),
        v7_contracts_sha256=_sha(v7_contracts_sha256, "V7 contracts SHA"),
        action_bridge_sha256=_sha(action_bridge_sha256, "action bridge SHA"),
        verify_referents=verify_referents,
    )


def _runtime_index_row(
    payload: bytes,
    *,
    root_id: str,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ReplayAuthorityError(f"{label} is not UTF-8 JSONL") from error
    for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and value.get("root_id") == root_id:
            found.append(value)
    if len(found) != 1 or canonical_sha256(found[0]) != expected_sha256:
        raise ReplayAuthorityError(f"{label} row closure mismatch for {root_id}")
    return found[0]


def _validate_fresh_index_row(
    fresh: dict[str, Any],
    *,
    selection: dict[str, Any],
    candidate: dict[str, Any],
    authority_row: dict[str, Any],
    action_audit_sha256: str,
    selected_candidate_index_sha256: str,
    fresh_payload: dict[str, np.ndarray],
) -> None:
    """Close every fresh producer identity back to independently sealed inputs."""
    source_dataset = Path(authority_row["execution_source_dataset_path"])
    expected_suffix = Path("inputs/selected") / authority_row["root_id"] / "lerobot"
    suffix_parts = expected_suffix.parts
    if source_dataset.parts[-len(suffix_parts):] != suffix_parts:
        raise ReplayAuthorityError(
            "authority execution source dataset is outside selected snapshot"
        )
    snapshot_root = source_dataset.parents[3]

    def child_path(value: str) -> str:
        try:
            relative = Path(value).relative_to(snapshot_root)
        except ValueError as error:
            raise ReplayAuthorityError(
                "authority execution path escapes snapshot"
            ) from error
        return "./" + relative.as_posix()

    expected = {
        "schema": "wm3d_v7_stage1_planner_same_root_runtime_v3",
        "root_id": authority_row["root_id"],
        "split": selection["split"],
        "source_dataset": authority_row["execution_source_dataset_path"],
        "root_context_path": authority_row["execution_root_context_path"],
        "root_context_sha256": selection["root_context_sha256"],
        "episode_id": selection["episode_id"],
        "episode_root_index": selection["episode_root_index"],
        "t0": selection["t0"],
        "candidate_index_sha256": selected_candidate_index_sha256,
        "candidate_payload_sha256": selection["candidate_payload_sha256"],
        "action_audit_sha256": action_audit_sha256,
        "stage0_checkpoint_sha256": authority_row["stage0_checkpoint_sha256"],
        "model_xml_sha256": selection["model_xml_sha256"],
        "ep_meta_sha256": selection["ep_meta_sha256"],
        "branches": authority_row["candidate_count"],
        "future_frames": authority_row["future_frames"],
        "branch_roles": authority_row["branch_roles"],
        "candidate_actions_executed_exact": True,
        "pseudo_outcomes": False,
        "future_observation_leakage": False,
    }
    for field, value in expected.items():
        if fresh.get(field) != value:
            raise ReplayAuthorityError(
                f"fresh runtime index {field} differs from sealed inputs"
            )
    for field in (
        "split_group", "task", "task_text",
        "episode_id", "episode_root_index", "t0", "stage0_checkpoint_sha256",
    ):
        if field in candidate and fresh.get(field) != candidate[field]:
            raise ReplayAuthorityError(
                f"fresh runtime index {field} differs from candidate index"
            )
    selection_candidate_expected = {
        "root_id": selection["root_id"],
        "split": selection["split"],
        "source_dataset": child_path(
            authority_row["execution_source_dataset_path"]
        ),
        "episode_id": selection["episode_id"],
        "episode_root_index": selection["episode_root_index"],
        "t0": selection["t0"],
        "candidate_path": child_path(
            authority_row["execution_candidate_payload_path"]
        ),
        "payload_sha256": selection["candidate_payload_sha256"],
        "root_context_path": child_path(
            authority_row["execution_root_context_path"]
        ),
        "root_context_sha256": selection["root_context_sha256"],
        "stage0_checkpoint_sha256": authority_row["stage0_checkpoint_sha256"],
        "action_audit_sha256": action_audit_sha256,
    }
    for field, value in selection_candidate_expected.items():
        if candidate.get(field) != value:
            raise ReplayAuthorityError(
                f"candidate index {field} differs from sealed selection"
            )
    authority_selection_expected = {
        key: authority_row[key] for key in REPLAY_SELECTION_ROW_FIELDS
    }
    if authority_selection_expected != selection:
        raise ReplayAuthorityError(
            "authority row source closure differs from sealed selection"
        )
    payload_expected = {
        "root_id": authority_row["root_id"],
        "root_context_sha256": authority_row["root_context_sha256"],
        "stage0_checkpoint_sha256": authority_row["stage0_checkpoint_sha256"],
        "model_xml_sha256": selection["model_xml_sha256"],
        "ep_meta_sha256": selection["ep_meta_sha256"],
    }
    for field, value in payload_expected.items():
        if str(fresh_payload[field].item()) != value:
            raise ReplayAuthorityError(
                f"fresh replay payload {field} differs from sealed inputs"
            )
    array_expected = {
        "root_state_sha256": "root_state",
        "root_rgb_sha256": "root_rgb",
    }
    for field, array_name in array_expected.items():
        if fresh.get(field) != array_sha256(fresh_payload[array_name]):
            raise ReplayAuthorityError(
                f"fresh runtime index {field} differs from replay payload"
            )


def _validate_payload_pair(
    row: dict[str, Any],
    *,
    selection: dict[str, Any],
    candidate_index_row: dict[str, Any],
    fresh_index_row: dict[str, Any],
    action_audit_sha256: str,
    selected_candidate_index_sha256: str,
    candidate_payload: bytes,
    legacy_payload: bytes,
    fresh_payload: bytes,
) -> None:
    candidate = _load_npz(candidate_payload, "candidate payload")
    legacy = _load_npz(legacy_payload, "legacy replay payload")
    fresh = _load_npz(fresh_payload, "fresh replay payload")
    missing = _NPZ_ARRAYS - set(fresh)
    if missing:
        raise ReplayAuthorityError(f"replay payload lacks arrays: {sorted(missing)}")
    _validate_fresh_index_row(
        fresh_index_row,
        selection=selection,
        candidate=candidate_index_row,
        authority_row=row,
        action_audit_sha256=action_audit_sha256,
        selected_candidate_index_sha256=selected_candidate_index_sha256,
        fresh_payload=fresh,
    )
    legacy_missing = _LEGACY_COMPARISON_FIELDS - set(legacy)
    if legacy_missing:
        raise ReplayAuthorityError(
            f"legacy replay lacks diagnostic arrays: {sorted(legacy_missing)}"
        )
    diagnostic = row["legacy_comparison_diagnostic"]
    if (
        not isinstance(diagnostic, dict)
        or set(diagnostic) != {"available", "all_core_equal", "fields"}
        or diagnostic["available"] is not True
        or type(diagnostic["all_core_equal"]) is not bool
    ):
        raise ReplayAuthorityError("authority legacy diagnostic fields mismatch")
    comparison = diagnostic["fields"]
    if not isinstance(comparison, dict) or set(comparison) != _LEGACY_COMPARISON_FIELDS:
        raise ReplayAuthorityError("authority legacy comparison fields mismatch")
    for name in _LEGACY_COMPARISON_FIELDS:
        if comparison[name] is not bool(np.array_equal(legacy[name], fresh[name])):
            raise ReplayAuthorityError(f"authority legacy comparison mismatch: {name}")
    if diagnostic["all_core_equal"] is not all(comparison.values()):
        raise ReplayAuthorityError("authority legacy diagnostic aggregate mismatch")
    candidate_count = _integer(row["candidate_count"], "authority candidate count", minimum=2)
    future_frames = _integer(row["future_frames"], "authority future frames", minimum=1)
    expected_shapes = {
        "simulator_actions": (candidate_count, future_frames * 4, 12),
        "branch_rgb": (candidate_count, future_frames + 1),
        "branch_rewards": (candidate_count, future_frames),
        "branch_dones": (candidate_count, future_frames),
        "branch_success": (candidate_count, future_frames),
    }
    for name, prefix in expected_shapes.items():
        if fresh[name].shape[: len(prefix)] != prefix:
            raise ReplayAuthorityError(f"fresh replay {name} shape mismatch")
    if (
        str(fresh["schema"].item())
        != "wm3d_v7_stage1_planner_same_root_runtime_v3"
        or str(fresh["root_id"].item()) != row["root_id"]
        or fresh["branch_actions_physical"].shape
        != (candidate_count, future_frames, 7)
        or fresh["branch_actions_requested_physical"].shape
        != (candidate_count, future_frames, 7)
        or fresh["action_history_physical"].shape != (4, 7)
        or fresh["simulator_action_low"].shape != (12,)
        or fresh["simulator_action_high"].shape != (12,)
        or str(fresh["stage0_checkpoint_sha256"].item())
        != row["stage0_checkpoint_sha256"]
        or str(fresh["root_context_sha256"].item())
        != row["root_context_sha256"]
        or fresh["same_root_simulator_state_exact"].item() is not True
        or fresh["same_root_render_state_exact"].item() is not True
        or fresh["root_rgb_equivalence_all_passed"].item() is not True
        or fresh["same_root_rgb_canonicalized"].item() is not True
        or fresh["h32_factual_available"].item() is not True
        or str(fresh["counterfactual_pose_space"].item())
        != "physical_canonical_6d"
        or str(fresh["root_rgb_canonical_source"].item())
        != "sealed_root_context"
    ):
        raise ReplayAuthorityError("fresh replay self-contained contract mismatch")
    candidate_actions = candidate.get("branch_actions_physical")
    candidate_roles = candidate.get("branch_roles")
    if (
        candidate_actions is None
        or candidate_roles is None
        or candidate_actions.shape != (candidate_count - 1, future_frames, 7)
        or [str(value) for value in candidate_roles.tolist()]
        != [str(value) for value in fresh["branch_roles"][1:].tolist()]
        or not np.array_equal(
            candidate_actions,
            fresh["branch_actions_requested_physical"][1:],
        )
        or not np.array_equal(
            fresh["branch_actions_requested_physical"][0],
            fresh["branch_actions_physical"][0],
        )
    ):
        raise ReplayAuthorityError(
            "fresh replay actions do not exactly execute sealed candidates"
        )
    digest_fields = {
        "root_state_sha256": fresh["root_state"],
        "root_rgb_sha256": fresh["root_rgb"],
        "executed_actions_sha256": fresh["simulator_actions"],
        "branch_rgb_sha256": fresh["branch_rgb"],
        "branch_rewards_sha256": fresh["branch_rewards"],
        "branch_dones_sha256": fresh["branch_dones"],
        "branch_success_sha256": fresh["branch_success"],
        "simulator_action_low_sha256": fresh["simulator_action_low"],
        "simulator_action_high_sha256": fresh["simulator_action_high"],
        "root_rgb_changed_fraction_sha256": fresh["root_rgb_changed_fraction"],
        "root_rgb_mean_abs_sha256": fresh["root_rgb_mean_abs"],
        "root_rgb_rmse_sha256": fresh["root_rgb_rmse"],
        "root_rgb_p99_abs_sha256": fresh["root_rgb_p99_abs"],
        "root_rgb_psnr_db_sha256": fresh["root_rgb_psnr_db"],
    }
    for field, array in digest_fields.items():
        if array_sha256(array) != row[field]:
            raise ReplayAuthorityError(f"authority {field} differs from fresh replay")
    roles = fresh.get("branch_roles")
    render = fresh.get("root_render_state_branch_sha256")
    raw_rgb = fresh.get("root_rgb_raw_sha256")
    if roles is None or render is None or raw_rgb is None:
        raise ReplayAuthorityError("fresh replay lacks per-branch provenance")
    if row["branch_roles"] != [str(value) for value in roles.tolist()]:
        raise ReplayAuthorityError("authority branch roles mismatch")
    for field in ("model_xml_sha256", "ep_meta_sha256"):
        if row[field] != str(fresh[field].item()):
            raise ReplayAuthorityError(f"authority {field} mismatch")
    if row["root_rgb_equivalence_contract"] != str(
        fresh["root_rgb_equivalence_contract"].item()
    ):
        raise ReplayAuthorityError("authority root RGB equivalence contract mismatch")
    branches = row["branches"]
    if not isinstance(branches, list) or len(branches) != candidate_count:
        raise ReplayAuthorityError("authority branch coverage mismatch")
    for index, branch in enumerate(branches):
        if not isinstance(branch, dict) or set(branch) != REPLAY_BRANCH_FIELDS:
            raise ReplayAuthorityError("authority branch fields mismatch")
        if _integer(branch["index"], "authority branch index") != index:
            raise ReplayAuthorityError("authority branch ordering mismatch")
        role = str(roles[index])
        if type(branch["role"]) is not str or branch["role"] != role:
            raise ReplayAuthorityError("authority branch role mismatch")
        expected = {
            "root_state_sha256": row["root_state_sha256"],
            "root_render_state_sha256": str(render[index]),
            "root_rgb_raw_sha256": str(raw_rgb[index]),
            "executed_action_sha256": array_sha256(fresh["simulator_actions"][index]),
            "rgb_sha256": array_sha256(fresh["branch_rgb"][index]),
            "reward_sha256": array_sha256(fresh["branch_rewards"][index]),
            "done_sha256": array_sha256(fresh["branch_dones"][index]),
            "success_sha256": array_sha256(fresh["branch_success"][index]),
        }
        for field, digest in expected.items():
            if _sha(branch[field], f"authority branch {field}") != digest:
                raise ReplayAuthorityError(f"authority branch {field} mismatch")
        terminal = bool(fresh["branch_success"][index].any())
        if branch["terminal_success"] is not terminal:
            raise ReplayAuthorityError("authority branch terminal success mismatch")
        maximum = float(fresh["branch_rewards"][index].max())
        if _number(branch["max_reward"], "authority branch max reward") != maximum:
            raise ReplayAuthorityError("authority branch max reward mismatch")
    if canonical_sha256(branches) != row["branches_sha256"]:
        raise ReplayAuthorityError("authority branches SHA mismatch")


def validate_replay_authority(
    authority: object,
    *,
    expected_code_commit: str,
    verify_referents: bool = True,
) -> dict[str, Any]:
    if not isinstance(authority, dict) or set(authority) != REPLAY_AUTHORITY_FIELDS:
        raise ReplayAuthorityError("replay authority top-level fields mismatch")
    if authority["schema"] != REPLAY_AUTHORITY_SCHEMA or authority["passed"] is not True:
        raise ReplayAuthorityError("replay authority schema/pass mismatch")
    if authority["code_commit"] != expected_code_commit:
        raise ReplayAuthorityError("replay authority code commit mismatch")
    for name in (
        "runtime_generator", "replay_helper", "adapter_loader", "action_audit",
        "action_audit_snapshot", "execution_snapshot_manifest",
        "candidate_index", "candidate_index_seal", "selected_candidate_index",
        "fresh_runtime_index",
    ):
        _sha(authority[f"{name}_sha256"], f"authority {name} SHA")
        if verify_referents:
            _verify_path(
                authority[f"{name}_path"], authority[f"{name}_sha256"], name
            )
    if authority["action_audit_snapshot_sha256"] != authority["action_audit_sha256"]:
        raise ReplayAuthorityError("action audit snapshot differs from provenance")
    snapshot_rows_by_path: dict[str, dict[str, Any]] = {}
    if verify_referents:
        _snapshot_path, snapshot_payload = _verify_path(
            authority["execution_snapshot_manifest_path"],
            authority["execution_snapshot_manifest_sha256"],
            "execution snapshot manifest",
        )
        try:
            snapshot_manifest = json.loads(snapshot_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayAuthorityError("execution snapshot manifest invalid") from error
        validate_execution_snapshot(
            snapshot_manifest, verify_provenance=True, verify_snapshots=True
        )
        snapshot_rows_by_path = {
            row["snapshot_path"]: row for row in snapshot_manifest["rows"]
        }
        action_snapshot = snapshot_rows_by_path.get(
            authority["action_audit_snapshot_path"]
        )
        if (
            action_snapshot is None
            or action_snapshot["snapshot_sha256"]
            != authority["action_audit_snapshot_sha256"]
            or action_snapshot["kind"] != "action audit"
        ):
            raise ReplayAuthorityError(
                "action audit execution path is not bound by snapshot"
            )
    selection_sha = _sha(
        authority["selection_manifest_sha256"], "selection manifest SHA"
    )
    selection_rows: list[dict[str, str]] = []
    selection_counts: dict[str, int] = {}
    if verify_referents:
        _selection_path, selection_payload = _verify_path(
            authority["selection_manifest_path"],
            selection_sha,
            "selection manifest",
        )
        try:
            selection = json.loads(selection_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayAuthorityError("selection manifest is invalid JSON") from error
        if not isinstance(selection, dict) or set(selection) != REPLAY_SELECTION_FIELDS:
            raise ReplayAuthorityError("selection manifest fields mismatch")
        try:
            from scripts.data.seal_robocasa_stage1_selection import (
                rebuild_selection_receipt,
            )

            rebuild_selection_receipt(
                selection, expected_code_commit=expected_code_commit
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise ReplayAuthorityError(
                "selection manifest cannot be re-derived from committed inputs"
            ) from error
        if (
            selection["schema"] != REPLAY_SELECTION_SCHEMA
            or selection["passed"] is not True
            or selection["code_commit"] != expected_code_commit
            or selection["candidate_index_sha256"]
            != authority["candidate_index_sha256"]
            or selection["candidate_index_seal_sha256"]
            != authority["candidate_index_seal_sha256"]
        ):
            raise ReplayAuthorityError("selection manifest lineage mismatch")
        for name in (
            "selection_policy", "data_profile", "candidate_index",
            "candidate_index_seal",
        ):
            _verify_path(
                selection[f"{name}_path"],
                selection[f"{name}_sha256"],
                f"selection {name}",
            )
        selection_rows = selection["rows"]
        selection_counts = selection["selection_count"]
        if (
            not isinstance(selection_rows, list)
            or not selection_rows
            or canonical_sha256(selection_rows)
            != _sha(selection["rows_sha256"], "selection rows SHA")
            or not isinstance(selection_counts, dict)
            or set(selection_counts) != _SPLITS
            or any(
                type(selection_counts[name]) is not int
                or selection_counts[name] <= 0
                for name in _SPLITS
            )
            or sum(selection_counts.values()) != len(selection_rows)
        ):
            raise ReplayAuthorityError("selection manifest row closure mismatch")
        for row in selection_rows:
            if (
                not isinstance(row, dict)
                or set(row) != REPLAY_SELECTION_ROW_FIELDS
                or row["split"] not in _SPLITS
            ):
                raise ReplayAuthorityError("selection manifest row fields mismatch")
            _sha(row["root_id"], "selection root id")
            if (
                type(row["source"]) is not str
                or not row["source"]
                or type(row["episode_id"]) is not int
                or row["episode_id"] < 0
                or type(row["episode_root_index"]) is not int
                or row["episode_root_index"] < 0
                or type(row["t0"]) is not int
                or row["t0"] < 0
            ):
                raise ReplayAuthorityError("selection manifest row identity invalid")
            _verify_selection_row_referents(row)
            for field in (
                "source_manifest_row_sha256", "model_xml_sha256",
                "ep_meta_sha256", "candidate_index_row_sha256",
            ):
                _sha(row[field], f"selection row {field}")
            if (
                type(row["source_dataset_path"]) is not str
                or not row["source_dataset_path"]
            ):
                raise ReplayAuthorityError(
                    "selection manifest source dataset path invalid"
                )
    environment_sha = _sha(
        authority["simulator_environment_receipt_sha256"],
        "simulator environment receipt SHA",
    )
    if verify_referents:
        _environment_path, environment_payload = _verify_path(
            authority["simulator_environment_receipt_path"],
            environment_sha,
            "simulator environment receipt",
        )
        try:
            environment = json.loads(environment_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayAuthorityError("simulator environment receipt is invalid JSON") from error
        environment = _validate_environment(
            environment,
            code_commit=expected_code_commit,
            generator_sha256=authority["runtime_generator_sha256"],
            generator_snapshot_sha256=environment[
                "runtime_generator_snapshot_sha256"
            ],
            helper_sha256=authority["replay_helper_sha256"],
            adapter_loader_sha256=authority["adapter_loader_sha256"],
            v7_action_contract_sha256=PINNED_V7_ACTION_CONTRACT_SHA256,
            v7_contracts_sha256=PINNED_V7_CONTRACTS_SHA256,
            action_bridge_sha256=PINNED_ACTION_BRIDGE_SHA256,
            verify_referents=True,
        )
        if (
            environment["execution_snapshot_manifest_path"]
            != authority["execution_snapshot_manifest_path"]
            or environment["execution_snapshot_manifest_sha256"]
            != authority["execution_snapshot_manifest_sha256"]
        ):
            raise ReplayAuthorityError(
                "environment and authority execution snapshots differ"
            )
    for field in ("runtime_root", "fresh_runtime_root"):
        value = authority[field]
        if type(value) is not str or not value:
            raise ReplayAuthorityError(f"authority {field} must be a path string")
        path = Path(value)
        if verify_referents and (path.is_symlink() or not path.is_dir()):
            raise ReplayAuthorityError(f"authority {field} is not a real directory")
    candidate_count = _integer(
        authority["candidate_count"], "authority candidate count", minimum=2
    )
    future_frames = _integer(
        authority["future_frames"], "authority future frames", minimum=1
    )
    counts = authority["selection_count"]
    rows = authority["rows"]
    if (
        not isinstance(counts, dict)
        or set(counts) != _SPLITS
        or any(type(counts[name]) is not int or counts[name] < 0 for name in _SPLITS)
        or not isinstance(rows, list)
        or not rows
        or sum(counts.values()) != len(rows)
    ):
        raise ReplayAuthorityError("replay authority selection closure mismatch")
    if canonical_sha256(rows) != _sha(authority["rows_sha256"], "authority rows SHA"):
        raise ReplayAuthorityError("replay authority rows SHA mismatch")
    observed = {name: 0 for name in _SPLITS}
    roots: set[str] = set()
    fresh_index_payload = b""
    if verify_referents:
        _path, fresh_index_payload = _verify_path(
            authority["fresh_runtime_index_path"],
            authority["fresh_runtime_index_sha256"],
            "fresh runtime index",
        )
        _path, selected_candidate_index_payload = _verify_path(
            authority["selected_candidate_index_path"],
            authority["selected_candidate_index_sha256"],
            "selected candidate index",
        )
    else:
        selected_candidate_index_payload = b""
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or set(row) != REPLAY_AUTHORITY_ROW_FIELDS:
            raise ReplayAuthorityError(f"replay authority row {number} fields mismatch")
        if row["schema"] != REPLAY_AUTHORITY_ROW_SCHEMA:
            raise ReplayAuthorityError(f"replay authority row {number} schema mismatch")
        split = row["split"]
        if split not in _SPLITS or type(row["source"]) is not str or not row["source"]:
            raise ReplayAuthorityError(f"replay authority row {number} split/source invalid")
        observed[split] += 1
        root_id = _sha(row["root_id"], f"authority row {number} root id")
        if root_id in roots:
            raise ReplayAuthorityError("replay authority contains duplicate roots")
        roots.add(root_id)
        for field in (
            "episode_id", "episode_root_index", "t0", "candidate_seed"
        ):
            _integer(row[field], f"authority row {number} {field}")
        if row["candidate_count"] != candidate_count or row["future_frames"] != future_frames:
            raise ReplayAuthorityError("replay authority row capacities mismatch")
        for field in (
            "candidate_index_row_sha256", "candidate_payload_sha256",
            "execution_candidate_index_row_sha256",
            "execution_candidate_payload_sha256",
            "execution_root_context_sha256",
            "execution_source_episode_sha256", "execution_states_sha256",
            "execution_model_xml_gz_sha256", "execution_ep_meta_file_sha256",
            "execution_dataset_meta_sha256", "execution_modality_sha256",
            "root_context_sha256", "source_episode_sha256",
            "source_manifest_sha256", "source_manifest_row_sha256",
            "legacy_runtime_index_shard_sha256", "legacy_runtime_index_row_sha256",
            "legacy_runtime_payload_sha256", "fresh_runtime_index_row_sha256",
            "fresh_runtime_payload_sha256", "stage0_checkpoint_sha256",
            "root_state_sha256", "root_render_state_sha256", "root_rgb_sha256",
            "executed_actions_sha256", "branch_rgb_sha256", "branch_rewards_sha256",
            "branch_dones_sha256", "branch_success_sha256", "branches_sha256",
            "simulator_action_low_sha256", "simulator_action_high_sha256",
            "states_sha256", "model_xml_gz_sha256", "model_xml_sha256",
            "ep_meta_file_sha256", "ep_meta_sha256", "dataset_meta_sha256",
            "modality_sha256",
            "root_rgb_changed_fraction_sha256", "root_rgb_mean_abs_sha256",
            "root_rgb_rmse_sha256", "root_rgb_p99_abs_sha256",
            "root_rgb_psnr_db_sha256",
        ):
            _sha(row[field], f"authority row {number} {field}")
        if (
            not isinstance(row["branch_roles"], list)
            or len(row["branch_roles"]) != candidate_count
            or any(type(value) is not str or not value for value in row["branch_roles"])
            or type(row["root_rgb_equivalence_contract"]) is not str
            or not row["root_rgb_equivalence_contract"]
        ):
            raise ReplayAuthorityError("authority row branch/RGB contract invalid")
        for gate in (
            "candidate_actions_executed_exact",
            "same_root_simulator_state_exact", "same_root_render_state_exact",
            "same_root_rgb_exact", "real_simulator_outcomes",
        ):
            if row[gate] is not True:
                raise ReplayAuthorityError(f"replay authority row {number} gate failed: {gate}")
        if verify_referents:
            matching_selection = [
                selected for selected in selection_rows
                if selected["root_id"] == root_id
            ]
            if len(matching_selection) != 1:
                raise ReplayAuthorityError(
                    "replay authority root is not unique in selection manifest"
                )
            selection_row = matching_selection[0]
            selected_candidate_index_row = _runtime_index_row(
                selected_candidate_index_payload,
                root_id=root_id,
                expected_sha256=row["execution_candidate_index_row_sha256"],
                label="selected candidate index",
            )
            _candidate_index_path, candidate_index_payload = _verify_path(
                authority["candidate_index_path"],
                authority["candidate_index_sha256"],
                "candidate index",
            )
            candidate_index_row = _runtime_index_row(
                candidate_index_payload,
                root_id=root_id,
                expected_sha256=row["candidate_index_row_sha256"],
                label="candidate index",
            )
            execution_expected = dict(candidate_index_row)
            snapshot_root = Path(
                row["execution_source_dataset_path"]
            ).parents[3]
            for field, authority_field in (
                ("source_dataset", "execution_source_dataset_path"),
                ("candidate_path", "execution_candidate_payload_path"),
                ("root_context_path", "execution_root_context_path"),
            ):
                try:
                    relative = Path(row[authority_field]).relative_to(
                        snapshot_root
                    )
                except ValueError as error:
                    raise ReplayAuthorityError(
                        f"authority {authority_field} escapes execution snapshot"
                    ) from error
                execution_expected[field] = "./" + relative.as_posix()
            if execution_expected != selected_candidate_index_row:
                raise ReplayAuthorityError(
                    "selected candidate row is not the exact snapshot path rewrite"
                )
            _candidate_path, candidate_payload = _verify_path(
                row["candidate_payload_path"], row["candidate_payload_sha256"],
                f"authority row {number} candidate payload",
            )
            _execution_candidate_path, execution_candidate_payload = _verify_path(
                row["execution_candidate_payload_path"],
                row["execution_candidate_payload_sha256"],
                f"authority row {number} execution candidate payload",
            )
            if execution_candidate_payload != candidate_payload:
                raise ReplayAuthorityError("execution candidate bytes differ from provenance")
            candidate = _load_npz(candidate_payload, "candidate payload")
            if (
                str(candidate.get("root_id", np.asarray("")).item()) != root_id
                or candidate.get("simulator_executable_all_candidates") is None
                or candidate["simulator_executable_all_candidates"].item() is not True
            ):
                raise ReplayAuthorityError("authority candidate payload is not executable")
            _verify_path(
                row["root_context_path"], row["root_context_sha256"],
                f"authority row {number} root context",
            )
            for name, provenance_digest, execution_digest in (
                ("root_context", "root_context_sha256", "execution_root_context_sha256"),
                ("source_episode", "source_episode_sha256", "execution_source_episode_sha256"),
                ("states", "states_sha256", "execution_states_sha256"),
                ("model_xml_gz", "model_xml_gz_sha256", "execution_model_xml_gz_sha256"),
                ("ep_meta", "ep_meta_file_sha256", "execution_ep_meta_file_sha256"),
                ("dataset_meta", "dataset_meta_sha256", "execution_dataset_meta_sha256"),
                ("modality", "modality_sha256", "execution_modality_sha256"),
            ):
                _verify_path(
                    row[f"execution_{name}_path"], row[execution_digest],
                    f"authority row {number} execution {name}",
                )
                if row[execution_digest] != row[provenance_digest]:
                    raise ReplayAuthorityError(
                        f"execution {name} SHA differs from provenance"
                    )
                snapshot_row = snapshot_rows_by_path.get(
                    row[f"execution_{name}_path"]
                )
                if (
                    snapshot_row is None
                    or snapshot_row["snapshot_sha256"] != row[execution_digest]
                    or snapshot_row["kind"] != f"selected {name}"
                ):
                    raise ReplayAuthorityError(
                        f"execution {name} is not bound by snapshot"
                    )
            _verify_path(
                row["source_episode_path"], row["source_episode_sha256"],
                f"authority row {number} source episode",
            )
            _verify_path(
                row["source_manifest_path"], row["source_manifest_sha256"],
                f"authority row {number} source manifest",
            )
            _legacy_index_path, legacy_index_payload = _verify_path(
                row["legacy_runtime_index_shard_path"],
                row["legacy_runtime_index_shard_sha256"],
                f"authority row {number} legacy runtime index shard",
            )
            _runtime_index_row(
                legacy_index_payload,
                root_id=root_id,
                expected_sha256=row["legacy_runtime_index_row_sha256"],
                label="legacy runtime index",
            )
            fresh_index_row = _runtime_index_row(
                fresh_index_payload,
                root_id=root_id,
                expected_sha256=row["fresh_runtime_index_row_sha256"],
                label="fresh runtime index",
            )
            _legacy_path, legacy_payload = _verify_path(
                row["legacy_runtime_payload_path"],
                row["legacy_runtime_payload_sha256"],
                f"authority row {number} legacy runtime payload",
            )
            _fresh_path, fresh_payload = _verify_path(
                row["fresh_runtime_payload_path"],
                row["fresh_runtime_payload_sha256"],
                f"authority row {number} fresh runtime payload",
            )
            _validate_payload_pair(
                row,
                selection=selection_row,
                candidate_index_row=selected_candidate_index_row,
                fresh_index_row=fresh_index_row,
                action_audit_sha256=authority["action_audit_sha256"],
                selected_candidate_index_sha256=authority[
                    "selected_candidate_index_sha256"
                ],
                candidate_payload=candidate_payload,
                legacy_payload=legacy_payload,
                fresh_payload=fresh_payload,
            )
    if observed != counts:
        raise ReplayAuthorityError("replay authority split counts mismatch")
    if verify_referents:
        authority_selection = [{
            key: row[key]
            for key in REPLAY_SELECTION_ROW_FIELDS
        } for row in rows]
        if selection_counts != counts or selection_rows != authority_selection:
            raise ReplayAuthorityError("replay authority differs from selection manifest")
    return authority


def load_replay_authority(
    path: Path,
    *,
    expected_code_commit: str,
) -> tuple[dict[str, Any], str]:
    _resolved, payload, digest = read_regular_bytes(path, "replay authority")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayAuthorityError("replay authority is not valid JSON") from error
    return validate_replay_authority(
        value,
        expected_code_commit=expected_code_commit,
        verify_referents=True,
    ), digest
