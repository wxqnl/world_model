from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import wm3d_v3.stage1_planner.rollout_audit as rollout_audit_contract
import wm3d_v3.stage1_planner.replay_authority as replay_authority_contract
import wm3d_v3.stage1_planner.train as stage1_train

from scripts.eval_wm3d_v8_stage1 import (
    _ACTION_FIELDS,
    _LEARNED_FIELDS,
    _action_shuffle_invariant,
    _auc,
    _require_authority_coverage,
)
from wm3d_v3.models.native_world_model import NativeWorldModel, NativeWorldModelConfig
from wm3d_v3.data.unified_cache_dataset import CacheDataError, _active_source_names
from wm3d_v3.stage1_planner.candidates import deterministic_action_cost
from wm3d_v3.stage1_planner.dataset import (
    BRANCH_INDEX_SCHEMA,
    BRANCH_SCHEMA,
    BRANCH_SEAL_SCHEMA,
    GENERATOR_RECEIPT_FIELDS,
    GENERATOR_RECEIPT_SCHEMA,
    Stage1BranchError,
    validate_rollout_audit_binding,
    _validate_candidate_action_shapes,
)
from wm3d_v3.stage1_planner.losses import planner_loss
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig, NativePlannerHead, planning_score
from wm3d_v3.stage1_planner.rollout import single_horizon_native_rollout
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig
from wm3d_v3.stage1_planner.train import (
    _expectations,
    _load_stage1,
    _prepare_stage1_launch,
    _planner_contract_sha,
    _stage1_run_contract,
    _topology_sha,
    _verify_runtime_checkout,
)
from scripts.materialize_wm3d_v8_stage1_branches import (
    _load_json as load_materializer_json,
    _output_directory as materializer_output_directory,
    _publish as publish_materialized_branch,
    _validate_candidate_payload,
    _validate_stage0_window_clock,
)
from scripts.produce_wm3d_v8_robocasa_stage1_candidates import (
    _output_directory as producer_output_directory,
    _publish as publish_candidate_output,
    _verified_payload as verified_producer_payload,
)
from scripts.data.audit_robocasa_real_rollouts import (
    _rows as load_audit_runtime_rows,
)
from scripts.data.robocasa_stage1_adapter_loader import _load_adapter
from scripts.data.seal_robocasa_stage1_selection import _policy
from wm3d_v3.data.manifest_contract import canonical_sha256
from wm3d_v3.stage1_planner.rollout_audit import (
    ROLLOUT_AUDIT_FIELDS,
    ROLLOUT_AUDIT_ROW_FIELDS,
    ROLLOUT_AUDIT_SCHEMA,
    RolloutAuditError,
    load_rollout_audit,
    validate_rollout_audit,
)
from wm3d_v3.stage1_planner.replay_authority import (
    REPLAY_AUTHORITY_FIELDS,
    REPLAY_AUTHORITY_ROW_FIELDS,
    REPLAY_AUTHORITY_ROW_SCHEMA,
    REPLAY_AUTHORITY_SCHEMA,
    REPLAY_BRANCH_FIELDS,
    ReplayAuthorityError,
    REPLAY_ENVIRONMENT_FIELDS,
    REPLAY_ENVIRONMENT_SCHEMA,
    REPLAY_SELECTION_ROW_FIELDS,
    REPLAY_SOURCE_TREE_SCHEMA,
    PINNED_RUNTIME_GENERATOR_SNAPSHOT_SHA256,
    _validate_environment,
    _validate_fresh_index_row,
    _validate_source_tree,
    array_sha256,
    validate_replay_authority,
)
from wm3d_v3.training.launch_qualification import LaunchQualificationError
from wm3d_v3.training.distributed_checkpoint import (
    CheckpointIntegrityError,
    _validate_metadata,
)


def test_multisource_split_requires_all_train_sources_but_not_eval_sources() -> None:
    entries = (
        SimpleNamespace(source="blender", split="train"),
        SimpleNamespace(source="coffee", split="train"),
        SimpleNamespace(source="blender", split="val"),
        SimpleNamespace(source="blender", split="test"),
    )
    kwargs = {
        "source_order": ("blender", "coffee"),
        "selected_sources": ("blender", "coffee"),
        "entries": entries,
    }
    assert _active_source_names(**kwargs, split="train") == ("blender", "coffee")
    assert _active_source_names(**kwargs, split="val") == ("blender",)
    assert _active_source_names(**kwargs, split="test") == ("blender",)

    missing_train = tuple(entry for entry in entries if entry.source != "coffee")
    with pytest.raises(CacheDataError, match="training sources have no cache windows"):
        _active_source_names(
            source_order=("blender", "coffee"),
            selected_sources=("blender", "coffee"),
            entries=missing_train,
            split="train",
        )

    with pytest.raises(CacheDataError, match="cache selection produced no samples"):
        _active_source_names(**kwargs, split="missing")


def test_stage1_new_closure_requires_rollout_audit_binding() -> None:
    assert BRANCH_SCHEMA.endswith("_v3")
    assert BRANCH_INDEX_SCHEMA.endswith("_v3")
    assert BRANCH_SEAL_SCHEMA.endswith("_v3")
    assert GENERATOR_RECEIPT_SCHEMA.endswith("_v2")
    digest = "a" * 64
    row = {"schema": BRANCH_SCHEMA, "rollout_audit_sha256": digest}
    receipt = {
        name: False if name == "future_observation_leakage" else True
        for name in GENERATOR_RECEIPT_FIELDS
    }
    receipt.update(
        schema=GENERATOR_RECEIPT_SCHEMA,
        rollout_audit_sha256=digest,
    )
    assert validate_rollout_audit_binding(row, receipt) == digest
    missing = dict(row)
    missing.pop("rollout_audit_sha256")
    with pytest.raises(Stage1BranchError, match="binding is missing"):
        validate_rollout_audit_binding(missing, receipt)
    tampered = dict(receipt, rollout_audit_sha256="b" * 64)
    with pytest.raises(Stage1BranchError, match="binding mismatch"):
        validate_rollout_audit_binding(row, tampered)

def _audit_fixture(tmp_path: Path) -> dict:
    digest = "a" * 64
    row = {
        field: digest for field in ROLLOUT_AUDIT_ROW_FIELDS
    }
    row.update({
        "split": "train", "source": "source", "episode_id": 1, "t0": 2,
        "task_text": "task", "candidate_seed": 3, "candidate_count": 11,
        "factual_simulator_action_source_byte_exact": True,
        "candidate_actions_executed_exact": True,
        "real_simulator_outcomes": True,
        "future_observation_leakage": False,
        "outcome_indices": [2], "future_offsets_seconds": [0.6],
        "branch_rgb_indices": [3], "source_future_row_offsets": [12],
        "replay_authority_row_sha256": digest,
    })
    rows = []
    for split, suffix in (("train", "a"), ("val", "b"), ("test", "c")):
        rows.append(dict(row, split=split, root_id=suffix * 64))
    audit = {field: digest for field in ROLLOUT_AUDIT_FIELDS}
    audit.update({
        "schema": ROLLOUT_AUDIT_SCHEMA, "code_commit": "c" * 40,
        "runtime_root": str(tmp_path), "source_roots": {"source": str(tmp_path)},
        "source_metadata_sha256": {"source": {
            name: digest for name in (
                "info.json", "modality.json", "embodiment.json", "episodes.jsonl"
            )
        }},
        "simulator_revision": {
            "source_repo": "repo", "source_revision": "revision",
            "robocasa_commit": "robocasa", "robocasa_dataset_version": "dataset",
            "robosuite_version": "version", "robosuite_commit": "robosuite",
            "mujoco_version": "mujoco",
        },
        "camera_order": ["left", "right", "wrist"],
        "simulator_action_order": [
            "eef_position3", "eef_rotation3", "gripper_close1", "base_motion4",
            "control_mode1",
        ],
        "source_action_order": [
            "base_motion4", "control_mode1", "eef_position3", "eef_rotation3",
            "gripper_close1",
        ],
        "simulator_action_period_seconds": 0.05,
        "selection_count": {"train": 1, "val": 1, "test": 1},
        "replay_authority_path": str(tmp_path / "replay-authority.json"),
        "replay_authority_sha256": digest,
        "rows": rows, "passed": True,
    })
    audit["simulator_revision_sha256"] = canonical_sha256(audit["simulator_revision"])
    audit["rows_sha256"] = canonical_sha256(rows)
    return audit


def _authority_fixture(tmp_path: Path) -> dict:
    digest = "a" * 64
    rows = []
    for split, suffix in (("train", "a"), ("val", "b"), ("test", "c")):
        branches = []
        for index in range(2):
            branch = {field: digest for field in REPLAY_BRANCH_FIELDS}
            branch.update({
                "index": index,
                "role": f"role_{index}",
                "terminal_success": bool(index),
                "max_reward": float(index),
            })
            branches.append(branch)
        row = {field: digest for field in REPLAY_AUTHORITY_ROW_FIELDS}
        row.update({
            "schema": REPLAY_AUTHORITY_ROW_SCHEMA,
            "split": split,
            "source": "source",
            "root_id": suffix * 64,
            "episode_id": 1,
            "episode_root_index": 2,
            "t0": 2,
            "candidate_seed": 3,
            "candidate_count": 2,
            "future_frames": 1,
            "legacy_comparison_diagnostic": {
                "available": True,
                "all_core_equal": True,
                "fields": {
                    name: True for name in (
                        "simulator_actions",
                        "root_state",
                        "root_rgb",
                        "branch_rgb",
                        "branch_rewards",
                        "branch_dones",
                        "branch_success",
                    )
                },
            },
            "branch_roles": ["role_0", "role_1"],
            "root_rgb_equivalence_contract": "exact",
            "branches": branches,
            "branches_sha256": canonical_sha256(branches),
            "candidate_actions_executed_exact": True,
            "same_root_simulator_state_exact": True,
            "same_root_render_state_exact": True,
            "same_root_rgb_exact": True,
            "real_simulator_outcomes": True,
        })
        rows.append(row)
    authority = {field: digest for field in REPLAY_AUTHORITY_FIELDS}
    authority.update({
        "schema": REPLAY_AUTHORITY_SCHEMA,
        "code_commit": "c" * 40,
        "runtime_root": str(tmp_path),
        "fresh_runtime_root": str(tmp_path),
        "selection_count": {"train": 1, "val": 1, "test": 1},
        "candidate_count": 2,
        "future_frames": 1,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
        "passed": True,
    })
    return authority


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda value: value.update(candidate_count=True), "candidate count"),
        (lambda value: value.update(selection_count={"train": True, "val": 1, "test": 1}), "selection closure"),
        (lambda value: value["rows"][0].update(candidate_seed=True), "candidate_seed"),
        (
            lambda value: value["rows"][0].update(episode_root_index=True),
            "episode_root_index",
        ),
        (
            lambda value: value.update(selected_candidate_index_sha256=1),
            "selected_candidate_index SHA",
        ),
        (lambda value: value["rows"][0].update(root_id=1), "root id"),
        (lambda value: value["rows"][0].update(same_root_rgb_exact=False), "gate failed"),
    ),
)
def test_replay_authority_rejects_exact_type_and_gate_tamper(
    tmp_path: Path, mutate, match: str
) -> None:
    authority = _authority_fixture(tmp_path)
    mutate(authority)
    authority["rows_sha256"] = canonical_sha256(authority["rows"])
    with pytest.raises(ReplayAuthorityError, match=match):
        validate_replay_authority(
            authority,
            expected_code_commit="c" * 40,
            verify_referents=False,
        )


def test_replay_authority_array_digest_is_dtype_and_shape_bound() -> None:
    value = np.arange(8, dtype=np.float32)
    assert array_sha256(value) != array_sha256(value.astype(np.float64))
    assert array_sha256(value) != array_sha256(value.reshape(2, 4))


def _fresh_binding_fixture() -> tuple[dict, dict, dict, dict, dict[str, np.ndarray]]:
    digest = "a" * 64
    selection = {field: digest for field in REPLAY_SELECTION_ROW_FIELDS}
    selection.update({
        "split": "train", "source": "source", "root_id": digest,
        "episode_id": 1, "episode_root_index": 2, "t0": 2,
        "source_dataset_path": "/source", "candidate_payload_path": "/candidate.npz",
        "root_context_path": "/root.npz",
    })
    authority = dict(selection)
    authority.update({
        "candidate_count": 2, "future_frames": 1,
        "stage0_checkpoint_sha256": "b" * 64,
        "branch_roles": ["factual_teacher", "direct"],
    })
    candidate = {
        "root_id": digest, "split": "train", "source_dataset": "/source",
        "episode_id": 1, "episode_root_index": 2, "t0": 2,
        "candidate_path": "/candidate.npz", "payload_sha256": digest,
        "root_context_path": "/root.npz", "root_context_sha256": digest,
        "stage0_checkpoint_sha256": "b" * 64, "action_audit_sha256": "c" * 64,
    }
    payload = {
        "root_id": np.asarray(digest), "root_context_sha256": np.asarray(digest),
        "stage0_checkpoint_sha256": np.asarray("b" * 64),
        "model_xml_sha256": np.asarray(digest), "ep_meta_sha256": np.asarray(digest),
        "root_state": np.zeros((1,), dtype=np.float32),
        "root_rgb": np.zeros((1,), dtype=np.uint8),
    }
    fresh = {
        "schema": "wm3d_v7_stage1_planner_same_root_runtime_v3",
        "root_id": digest, "split": "train", "source_dataset": "/source",
        "root_context_path": "/root.npz", "root_context_sha256": digest,
        "episode_id": 1, "episode_root_index": 2, "t0": 2,
        "candidate_index_sha256": "d" * 64, "candidate_payload_sha256": digest,
        "action_audit_sha256": "c" * 64, "stage0_checkpoint_sha256": "b" * 64,
        "model_xml_sha256": digest, "ep_meta_sha256": digest,
        "branches": 2, "future_frames": 1,
        "branch_roles": ["factual_teacher", "direct"],
        "candidate_actions_executed_exact": True, "pseudo_outcomes": False,
        "future_observation_leakage": False,
        "root_state_sha256": array_sha256(payload["root_state"]),
        "root_rgb_sha256": array_sha256(payload["root_rgb"]),
    }
    return selection, authority, candidate, fresh, payload


@pytest.mark.parametrize(
    "field",
    (
        "source_dataset_path", "source_episode_path", "source_episode_sha256",
        "states_path", "states_sha256", "model_xml_gz_path",
        "model_xml_gz_sha256", "model_xml_sha256", "ep_meta_path",
        "ep_meta_file_sha256", "ep_meta_sha256", "dataset_meta_path",
        "dataset_meta_sha256", "modality_path", "modality_sha256",
    ),
)
def test_replay_selection_extra_tamper_is_rejected(field: str) -> None:
    selection, authority, candidate, fresh, payload = _fresh_binding_fixture()
    selection[field] = "e" * 64
    with pytest.raises(ReplayAuthorityError, match="source closure|sealed inputs"):
        _validate_fresh_index_row(
            fresh, selection=selection, candidate=candidate,
            authority_row=authority, action_audit_sha256="c" * 64,
            selected_candidate_index_sha256="d" * 64, fresh_payload=payload,
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_dataset", "split", "episode_id", "episode_root_index", "t0",
        "candidate_index_sha256", "root_context_path", "root_context_sha256",
        "action_audit_sha256", "stage0_checkpoint_sha256", "model_xml_sha256",
        "ep_meta_sha256", "candidate_payload_sha256",
    ),
)
def test_fresh_runtime_index_identity_tamper_is_rejected(field: str) -> None:
    selection, authority, candidate, fresh, payload = _fresh_binding_fixture()
    fresh[field] = "tampered"
    with pytest.raises(ReplayAuthorityError, match="fresh runtime index"):
        _validate_fresh_index_row(
            fresh, selection=selection, candidate=candidate,
            authority_row=authority, action_audit_sha256="c" * 64,
            selected_candidate_index_sha256="d" * 64, fresh_payload=payload,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("root", "/self/reported/root"),
        ("package", "self_reported_package"),
        ("commit", "0" * 40),
        ("tree_sha256", "f" * 64),
    ),
)
def test_replay_source_tree_self_report_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    root = tmp_path / "robocasa-pinned"
    package = root / "robocasa"
    package.mkdir(parents=True)
    source = package / "source.py"
    source.write_bytes(b"source")
    rows = [{
        "path": "source.py", "size": len(b"source"),
        "sha256": hashlib.sha256(b"source").hexdigest(),
    }]
    tree_sha = canonical_sha256(rows)
    monkeypatch.setitem(
        replay_authority_contract.PINNED_SOURCE_TREE_SHA256,
        "robocasa", tree_sha,
    )
    manifest = {
        "schema": REPLAY_SOURCE_TREE_SCHEMA, "root": str(root),
        "package": "robocasa", "commit": "1" * 40, "rows": rows,
        "rows_sha256": tree_sha, "file_count": 1,
        "total_bytes": len(b"source"), "passed": True,
    }
    reference = {
        "manifest_path": str(tmp_path / "tree.json"),
        "manifest_sha256": "a" * 64, "tree_sha256": tree_sha,
    }
    if field == "tree_sha256":
        reference[field] = value
    else:
        manifest[field] = value
    payload = json.dumps(manifest).encode()
    monkeypatch.setattr(
        replay_authority_contract, "_verify_path",
        lambda *_args, **_kwargs: (tmp_path / "tree.json", payload),
    )
    with pytest.raises(ReplayAuthorityError, match="source-tree"):
        _validate_source_tree(
            reference, name="robocasa", expected_root=root,
            expected_package="robocasa", expected_commit="1" * 40,
            verify_referents=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("robocasa_source_commit", "0" * 40, "commit"),
        ("simulator_python_inode", 99, "identity"),
        ("simulator_pythonpath", "/ambient", "PYTHONPATH"),
        (
            "snapshot_runtime_generator_path",
            "/self/reported.py",
            "PYTHONPATH|path/SHA",
        ),
        ("snapshot_replay_helper_sha256", "f" * 64, "path/SHA"),
    ),
)
def test_replay_environment_identity_tamper_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, value: object, match: str,
) -> None:
    execution = tmp_path / "execution"
    paths = {
        "runtime_generator": execution / "scripts/generate_robocasa_stage1_planner_branches.py",
        "replay_helper": execution / "scripts/generate_robocasa_same_root_cf.py",
        "adapter_loader": execution / "scripts/robocasa_stage1_adapter_loader.py",
        "v7_action_contract": execution / "wm3d_v3/data/v7_action_contract.py",
        "v7_contracts": execution / "wm3d_v3/data/v7_contracts.py",
        "action_bridge": execution / "wm3d_v3/stage1_planner/action_bridge.py",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    python = tmp_path / "python"
    python.write_bytes(b"x")
    site = tmp_path / "site"
    site.mkdir()
    casa = tmp_path / ("robocasa-" + "8f3c96ec8d1bfcd8126cad2bca887da98d30e997")
    suite = tmp_path / ("robosuite-" + "6c10ef24a4bb52f59199976125060ce793470e6e")
    casa.mkdir()
    suite.mkdir()
    stat = python.stat()
    helper_sha, adapter_sha, action_sha = "b" * 64, "c" * 64, "d" * 64
    contracts_sha, bridge_sha = "e" * 64, "9" * 64
    snapshot_shas = {
        "runtime_generator": PINNED_RUNTIME_GENERATOR_SNAPSHOT_SHA256,
        "replay_helper": helper_sha, "adapter_loader": adapter_sha,
        "v7_action_contract": action_sha, "v7_contracts": contracts_sha,
        "action_bridge": bridge_sha,
    }
    environment = {field: "a" * 64 for field in REPLAY_ENVIRONMENT_FIELDS}
    environment.update({
        "schema": REPLAY_ENVIRONMENT_SCHEMA, "code_commit": "1" * 40,
        "simulator_python_path": str(python),
        "simulator_python_sha256": hashlib.sha256(b"x").hexdigest(),
        "simulator_python_device": stat.st_dev, "simulator_python_inode": stat.st_ino,
        "simulator_python_size": stat.st_size,
        "simulator_python_mtime_ns": stat.st_mtime_ns,
        "simulator_pythonpath": ":".join(map(str, (execution, casa, suite, site))),
        "python_version": "3.10", "cuda_visible_devices": "4", "mujoco_gl": "egl",
        "simulator_site_packages_path": str(site), "robocasa_source_root": str(casa),
        "robocasa_source_commit": "8f3c96ec8d1bfcd8126cad2bca887da98d30e997",
        "robosuite_source_root": str(suite),
        "robosuite_source_commit": "6c10ef24a4bb52f59199976125060ce793470e6e",
        "source_trees": {name: {
            "manifest_path": str(tmp_path / name), "manifest_sha256": "a" * 64,
            "tree_sha256": replay_authority_contract.PINNED_SOURCE_TREE_SHA256[name],
        } for name in ("robocasa", "robosuite")},
        "modules": {name: {"version": "1", "path": str(python),
                            "sha256": hashlib.sha256(b"x").hexdigest()}
                    for name in ("numpy", "mujoco", "robosuite", "robocasa")},
        "snapshot_modules": {name: {"path": str(paths[name]), "sha256": snapshot_shas[name]}
                             for name in paths},
        "runtime_generator_sha256": "a" * 64,
        "runtime_generator_snapshot_sha256": PINNED_RUNTIME_GENERATOR_SNAPSHOT_SHA256,
        "replay_helper_sha256": helper_sha, "adapter_loader_sha256": adapter_sha,
        "v7_action_contract_sha256": action_sha,
        "v7_contracts_sha256": contracts_sha, "action_bridge_sha256": bridge_sha,
        "passed": True,
    })
    monkeypatch.setattr(replay_authority_contract, "_validate_source_tree", lambda *a, **k: None)
    monkeypatch.setattr(
        replay_authority_contract,
        "_verify_path",
        lambda path, *_args, **_kwargs: (Path(path), b"x"),
    )
    if field.startswith("snapshot_"):
        if field == "snapshot_runtime_generator_path":
            environment["snapshot_modules"]["runtime_generator"]["path"] = value
        else:
            environment["snapshot_modules"]["replay_helper"]["sha256"] = value
    else:
        environment[field] = value
    with pytest.raises(ReplayAuthorityError, match=match):
        _validate_environment(
            environment, code_commit="1" * 40, generator_sha256="a" * 64,
            generator_snapshot_sha256=PINNED_RUNTIME_GENERATOR_SNAPSHOT_SHA256,
            helper_sha256=helper_sha, adapter_loader_sha256=adapter_sha,
            v7_action_contract_sha256=action_sha,
            v7_contracts_sha256=contracts_sha, action_bridge_sha256=bridge_sha,
            verify_referents=True,
        )


def test_stage1_selection_policy_rejects_three_root_self_selection() -> None:
    value = {
        "notes": {
            "stage1_selection_policy": {
                "schema": "wm3d_v8_robocasa_stage1_selection_policy_v1",
                "expected_split_counts": {"train": 1, "val": 1, "test": 1},
                "expected_data_profile_sha256": "d" * 64,
                "expected_candidate_index_sha256": "e" * 64,
                "expected_candidate_seal_sha256": "f" * 64,
                "expected_stage0_checkpoint_sha256": "0" * 64,
                "rows": [
                    {
                        "split": split,
                        "source": "source",
                        "root_id": character * 64,
                        "episode_id": index,
                        "t0": index,
                    }
                    for index, (split, character) in enumerate(
                        (("train", "a"), ("val", "b"), ("test", "c"))
                    )
                ],
            }
        }
    }
    with pytest.raises(RuntimeError, match="train2/val1/test1"):
        _policy(value)


def test_stage1_formal_selection_policy_pins_external_authorities() -> None:
    import yaml

    profile = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/data/stage1_robocasa_real_4roots.template.yaml"
        ).read_text(encoding="utf-8")
    )
    policy = profile["notes"]["stage1_selection_policy"]
    assert policy["expected_split_counts"] == {
        "train": 2,
        "val": 1,
        "test": 1,
    }
    for field in (
        "expected_data_profile_sha256",
        "expected_candidate_index_sha256",
        "expected_candidate_seal_sha256",
        "expected_stage0_checkpoint_sha256",
    ):
        assert isinstance(policy[field], str)
        assert len(policy[field]) == 64
        assert policy[field] != "0" * 64


def test_minimal_replay_adapter_loader_is_strict(tmp_path: Path) -> None:
    adapter = {
        "source": "robocasa",
        "source_frame": "robot_base",
        "translation_unit_scale": 1.0,
        "rotation_unit_scale": 1.0,
        "rotation_repr": "axis_angle",
    }
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps({
            "factual_action_audit": {"passed": True},
            "adapter": adapter,
        }),
        encoding="utf-8",
    )
    assert _load_adapter(
        audit, allow_legacy_proof_audit=False
    ).source == "robocasa"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"adapter": adapter}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="factual_action_audit"):
        _load_adapter(bad, allow_legacy_proof_audit=False)
    symlink = tmp_path / "audit-link.json"
    symlink.symlink_to(audit)
    with pytest.raises(RuntimeError, match="symlink"):
        _load_adapter(symlink, allow_legacy_proof_audit=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("replay_authority_sha256", 1, "replay authority SHA"),
        ("selection_count", {"train": True, "val": 1, "test": 1}, "selection counts"),
        ("simulator_action_period_seconds", True, "cadence"),
    ),
)
def test_rollout_audit_rejects_bool_as_number_and_nonstring_sha(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    audit = _audit_fixture(tmp_path)
    audit[field] = value
    with pytest.raises(RolloutAuditError, match=match):
        validate_rollout_audit(
            audit, expected_code_commit="c" * 40, verify_referents=False
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        (lambda value: value.update(extra=True), "top-level fields"),
        (lambda value: value.pop("rows_sha256"), "top-level fields"),
        (lambda value: value.update(rows=[]), "selection counts"),
        (lambda value: value.update(rows_sha256="b" * 64), "rows SHA mismatch"),
        (lambda value: value["rows"][0].update(split="invalid"), "split/source"),
        (lambda value: value["rows"][0].update(real_simulator_outcomes=False), "gate failed"),
        (lambda value: value["rows"][0].update(extra=True), "row 1 fields"),
    ),
)
def test_rollout_audit_contract_rejects_malformed_authority(
    tmp_path, mutation, match
) -> None:
    audit = _audit_fixture(tmp_path)
    mutation(audit)
    if "rows_sha256" in audit and audit.get("rows"):
        if match not in {"rows SHA mismatch", "gate failed", "split/source", "row 1 fields"}:
            audit["rows_sha256"] = canonical_sha256(audit["rows"])
        elif match in {"gate failed", "split/source", "row 1 fields"}:
            audit["rows_sha256"] = canonical_sha256(audit["rows"])
    with pytest.raises(RolloutAuditError, match=match):
        validate_rollout_audit(
            audit, expected_code_commit="c" * 40, verify_referents=False
        )


def test_rollout_audit_contract_accepts_exact_authority(tmp_path) -> None:
    audit = _audit_fixture(tmp_path)
    assert validate_rollout_audit(
        audit, expected_code_commit="c" * 40, verify_referents=False
    ) is audit
    with pytest.raises(RolloutAuditError, match="code commit differs"):
        validate_rollout_audit(
            audit, expected_code_commit="d" * 40, verify_referents=False
        )


def test_rollout_audit_loader_rejects_symlink_and_referent_tamper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = _audit_fixture(tmp_path)
    referent_fields = (
        "launch_receipt", "runtime_generator", "replay_helper", "action_audit",
        "candidate_index", "candidate_index_seal",
    )
    for name in referent_fields:
        path = tmp_path / f"{name}.json"
        path.write_bytes(name.encode())
        audit[f"{name}_path"] = str(path)
        audit[f"{name}_sha256"] = hashlib.sha256(name.encode()).hexdigest()
    (tmp_path / "meta").mkdir()
    for filename in ("info.json", "modality.json", "embodiment.json", "episodes.jsonl"):
        path = tmp_path / "meta" / filename
        path.write_bytes(filename.encode())
        audit["source_metadata_sha256"]["source"][filename] = hashlib.sha256(
            filename.encode()
        ).hexdigest()
    for index, row in enumerate(audit["rows"]):
        for name in ("runtime_payload", "runtime_index_shard", "root_context"):
            path = tmp_path / f"{index}_{name}.bin"
            path.write_bytes(f"{index}_{name}".encode())
            row[f"{name}_path"] = str(path)
            row[f"{name}_sha256"] = hashlib.sha256(
                f"{index}_{name}".encode()
            ).hexdigest()
    authority_rows = []
    for row in audit["rows"]:
        authority_row = {
            "split": row["split"],
            "root_id": row["root_id"],
            "episode_id": row["episode_id"],
            "t0": row["t0"],
            "candidate_seed": row["candidate_seed"],
            "candidate_payload_sha256": row["candidate_payload_sha256"],
            "root_context_sha256": row["root_context_sha256"],
            "legacy_runtime_payload_sha256": row["runtime_payload_sha256"],
            "legacy_runtime_index_shard_sha256": row[
                "runtime_index_shard_sha256"
            ],
            "legacy_runtime_index_row_sha256": row["runtime_index_row_sha256"],
            "candidate_count": row["candidate_count"],
        }
        row["replay_authority_row_sha256"] = canonical_sha256(authority_row)
        authority_rows.append(authority_row)
    monkeypatch.setattr(
        "wm3d_v3.stage1_planner.replay_authority.load_replay_authority",
        lambda *_args, **_kwargs: (
            {
                "selection_manifest_path": audit["selection_manifest_path"],
                "selection_manifest_sha256": audit["selection_manifest_sha256"],
                "rows": authority_rows,
            },
            audit["replay_authority_sha256"],
        ),
    )
    audit["rows_sha256"] = canonical_sha256(audit["rows"])
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    loaded, _digest = load_rollout_audit(
        audit_path, expected_code_commit="c" * 40
    )
    assert loaded["passed"] is True
    symlink = tmp_path / "audit-link.json"
    symlink.symlink_to(audit_path)
    with pytest.raises(RolloutAuditError, match="must not be a symlink"):
        load_rollout_audit(symlink, expected_code_commit="c" * 40)
    (tmp_path / "0_runtime_payload.bin").write_bytes(b"tampered")
    with pytest.raises(RolloutAuditError, match="runtime_payload SHA mismatch"):
        load_rollout_audit(audit_path, expected_code_commit="c" * 40)


def test_rollout_audit_loader_rejects_path_replacement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path = tmp_path / "audit.json"
    replacement = tmp_path / "replacement.json"
    audit_path.write_text("{}", encoding="utf-8")
    replacement.write_text("{}", encoding="utf-8")
    original_read = rollout_audit_contract.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, size)
        if payload and not replaced:
            replacement.replace(audit_path)
            replaced = True
        return payload

    monkeypatch.setattr(rollout_audit_contract.os, "read", replacing_read)
    with pytest.raises(RolloutAuditError, match="replaced while it was read"):
        load_rollout_audit(audit_path, expected_code_commit="c" * 40)


def test_rollout_audit_runtime_index_rejects_path_replacement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shard = tmp_path / "index.shard-00000.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    shard.write_text(json.dumps({"root_id": "a" * 64}) + "\n", encoding="utf-8")
    replacement.write_text(
        json.dumps({"root_id": "b" * 64}) + "\n", encoding="utf-8"
    )
    original_read = rollout_audit_contract.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        payload = original_read(descriptor, size)
        if payload and not replaced:
            replacement.replace(shard)
            replaced = True
        return payload

    monkeypatch.setattr(rollout_audit_contract.os, "read", replacing_read)
    with pytest.raises(RolloutAuditError, match="replaced while it was read"):
        load_audit_runtime_rows(tmp_path)


def test_stage1_producer_uses_one_verified_payload_snapshot(tmp_path) -> None:
    path = tmp_path / "runtime.npz"
    payload = b"sealed runtime snapshot"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    resolved, observed, digest = verified_producer_payload(
        path, expected, "runtime payload"
    )
    assert resolved == path.resolve()
    assert observed == payload
    assert digest == expected
    path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        verified_producer_payload(path, expected, "runtime payload")


@pytest.mark.parametrize(
    "publisher",
    (publish_candidate_output, publish_materialized_branch),
)
def test_stage1_publish_rejects_existing_symlink_even_for_same_bytes(
    tmp_path, publisher
) -> None:
    referent = tmp_path / "referent"
    referent.write_bytes(b"same")
    target = tmp_path / "published"
    target.symlink_to(referent)
    with pytest.raises(FileExistsError, match="symlink"):
        publisher(target, b"same")
    assert target.is_symlink()
    assert referent.read_bytes() == b"same"


@pytest.mark.parametrize(
    ("guard", "error"),
    (
        (producer_output_directory, RuntimeError),
        (materializer_output_directory, ValueError),
    ),
)
def test_stage1_output_root_rejects_directory_symlink(
    tmp_path, guard, error
) -> None:
    real = tmp_path / "real-output"
    real.mkdir()
    link = tmp_path / "output-root"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(error, match="must not be a symlink"):
        guard(link, "output root")
    assert not list(real.iterdir())


def test_stage1_publish_fsyncs_parent_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[int] = []
    original_fsync = rollout_audit_contract.os.fsync

    def recording_fsync(descriptor: int) -> None:
        observed.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(rollout_audit_contract.os, "fsync", recording_fsync)
    target = tmp_path / "published"
    publish_candidate_output(target, b"payload")
    assert target.read_bytes() == b"payload"
    assert len(observed) >= 2


def test_stage1_stable_read_and_publish_reject_ancestor_symlink(
    tmp_path,
) -> None:
    real = tmp_path / "real" / "nested"
    real.mkdir(parents=True)
    source = real / "source.bin"
    source.write_bytes(b"sealed")
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(RolloutAuditError, match="symlink"):
        rollout_audit_contract.read_regular_bytes(
            linked / "nested" / "source.bin", "ancestor-symlink input"
        )
    with pytest.raises(RolloutAuditError, match="symlink"):
        publish_candidate_output(
            linked / "nested" / "published.bin", b"payload"
        )
    assert not (real / "published.bin").exists()


def test_stage1_publish_parent_replacement_is_pinned_and_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "trusted"
    parent.mkdir()
    moved = tmp_path / "trusted-pinned"
    original_link = rollout_audit_contract.os.link
    replaced = False

    def replacing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal replaced
        if not replaced:
            parent.rename(moved)
            parent.mkdir()
            replaced = True
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(rollout_audit_contract.os, "link", replacing_link)
    with pytest.raises(RolloutAuditError, match="ancestor was replaced"):
        publish_candidate_output(parent / "published.bin", b"payload")
    assert not (parent / "published.bin").exists()
    assert (moved / "published.bin").read_bytes() == b"payload"


def test_trusted_output_root_rejects_nested_parent_replacement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope = tmp_path / "scope"
    nested = scope / "payloads" / "train"
    nested.mkdir(parents=True)
    moved = scope / "payloads-pinned"
    output = rollout_audit_contract.TrustedOutputRoot(scope)
    original_link = rollout_audit_contract.os.link
    replaced = False

    def replacing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        nonlocal replaced
        if not replaced:
            (scope / "payloads").rename(moved)
            (scope / "payloads" / "train").mkdir(parents=True)
            replaced = True
        original_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(rollout_audit_contract.os, "link", replacing_link)
    try:
        with pytest.raises(RolloutAuditError, match="ancestor was replaced"):
            output.publish(
                nested / "branch.pt", b"payload", label="nested branch"
            )
    finally:
        output.close()
    assert not (scope / "payloads" / "train" / "branch.pt").exists()
    assert (moved / "train" / "branch.pt").read_bytes() == b"payload"


def test_trusted_output_root_directory_pin_rejects_replacement(tmp_path) -> None:
    scope_path = tmp_path / "scope"
    child = scope_path / "payloads"
    with rollout_audit_contract.TrustedOutputRoot(scope_path) as scope:
        pin = scope.pin_directory(child, label="payload directory")
        moved = scope_path / "payloads-original"
        child.rename(moved)
        child.mkdir()
        with pytest.raises(RolloutAuditError, match="ancestor was replaced"):
            scope.verify_pinned_directory(pin, label="payload directory")


def test_materializer_json_rejects_symlink(tmp_path) -> None:
    referent = tmp_path / "receipt.json"
    referent.write_text("{}", encoding="utf-8")
    link = tmp_path / "receipt-link.json"
    link.symlink_to(referent)
    with pytest.raises(ValueError, match="symlink"):
        load_materializer_json(link)


def test_stage1_release_manual_tracks_required_audit_cli_and_schemas() -> None:
    manual = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "WM3D_V8_STAGE1_UNIFIED.md"
    ).read_text(encoding="utf-8")
    audit_block = manual.split(
        "./run_v8.sh stage1-audit-rollouts \\", 1
    )[1].split("```", 1)[0]
    replay_block = manual.split(
        "./run_v8.sh stage1-replay-authority \\", 1
    )[1].split("```", 1)[0]
    assert '--code-commit "$CODE_COMMIT"' in audit_block
    assert '--code-commit "$CODE_COMMIT"' in replay_block
    assert "--selection-manifest" in replay_block
    assert "--simulator-site-packages" in replay_block
    assert "robocasa_stage1_adapter_loader.py" in replay_block
    assert '--replay-authority "$STAGE1_ROOT/replay_authority.json"' in audit_block
    assert BRANCH_SCHEMA in manual
    assert GENERATOR_RECEIPT_SCHEMA in manual
    assert "rollout_audit_sha256" in manual
    assert "wm3d_v8_unified_stage1_branch_v2`" not in manual
    assert "wm3d_v8_unified_stage1_candidate_generator_receipt_v1`" not in manual
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "stage1-audit-rollouts" in readme
    assert "stage1-replay-authority" in readme
    assert "stage1-seal-selection" in readme
    assert "stage1-produce" in readme
    validation = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "WM3D_V8_RELEASE_VALIDATION.md"
    ).read_text(encoding="utf-8")
    assert "旧 SHA 因此仅是开发记录" in validation
    assert "59c6af619650e2114ca280cb87b0cd1198741d3be19faae06e0871cb99aa80c3" not in validation


def test_stage1_checkout_uses_shared_clean_runtime_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    observed = {}

    def fake(repo, commit):
        observed.update(repo=repo, commit=commit)
        return commit

    monkeypatch.setattr(
        "wm3d_v3.stage1_planner.train.verify_clean_runtime_checkout", fake
    )
    runtime = {"run": {"code_commit": "c" * 40}}
    assert _verify_runtime_checkout(runtime, tmp_path) == "c" * 40
    assert observed == {"repo": tmp_path, "commit": "c" * 40}


def test_stage1_checkout_rejects_dirty_repository(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    (tmp_path / "code.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True
    )
    head = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert _verify_runtime_checkout(
        {"run": {"code_commit": head}}, tmp_path
    ) == head
    (tmp_path / "untracked.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(LaunchQualificationError, match="dirty"):
        _verify_runtime_checkout({"run": {"code_commit": head}}, tmp_path)


def _planner() -> NativePlannerHead:
    return NativePlannerHead(NativePlannerConfig(
        token_dim=16, task_dim=12, hidden=32, spatial_layers=1,
        temporal_layers=1, heads=4, mlp_mult=2, dropout=0.0,
        max_horizon=4, patches=4, num_views=2, time_fourier_dim=8,
        time_min_period_s=0.01, time_max_period_s=10.0))


def _evidence() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.randn(2, 3, 4, 4, 16),
        "future_dt_s": torch.tensor([.1, .3, .8, 1.4]).view(1,1,4).expand(2,3,4),
        "token_mask": torch.ones(2,3,4,4,dtype=torch.bool),
        "task": torch.randn(2, 12),
        "depth": torch.rand(2, 3, 4, 2, 4),
        "depth_mask": torch.ones(2,3,4,2,4,dtype=torch.bool),
        "point": torch.randn(2, 3, 4, 2, 4, 3),
        "point_mask": torch.ones(2,3,4,2,4,dtype=torch.bool),
        "pose": torch.randn(2, 3, 4, 2, 9),
        "pose_mask": torch.ones(2,3,4,2,dtype=torch.bool),
        "confidence": torch.rand(2, 3, 4, 2, 4),
        "view_mask": torch.ones(2,3,4,2,dtype=torch.bool),
    }


def _forward(planner: NativePlannerHead, evidence: dict[str, torch.Tensor]):
    return planner(evidence["tokens"], evidence["task"],
        future_dt_s=evidence["future_dt_s"], token_mask=evidence["token_mask"],
        depth=evidence["depth"], depth_mask=evidence["depth_mask"],
        point=evidence["point"], point_mask=evidence["point_mask"],
        pose=evidence["pose"], pose_mask=evidence["pose_mask"],
        geometry_confidence=evidence["confidence"], view_mask=evidence["view_mask"])


def test_planner_action_shuffle_invariance() -> None:
    torch.manual_seed(7)
    planner = _planner().eval()
    evidence = _evidence()
    actions = torch.randn(2, 3, 4, 2, 5, 6)
    first = _forward(planner, evidence)
    shuffled_actions = actions[:, torch.tensor([2, 0, 1])]
    second = _forward(planner, evidence)
    assert not torch.equal(actions, shuffled_actions)
    for name in first:
        assert torch.equal(first[name], second[name])
    assert not any("action" in name for name, _ in planner.named_parameters())


def test_eval_action_shuffle_gate_uses_one_execution_mode() -> None:
    class ModeSensitiveActionBlindSystem:
        def score_observed_batch(self, batch):
            bias = float(torch.is_grad_enabled())
            return {
                name: batch["fixed_evidence"] + bias for name in _LEARNED_FIELDS
            }

    batch = {"fixed_evidence": torch.tensor([[3.0, 5.0, 7.0]])}
    for index, name in enumerate(_ACTION_FIELDS):
        batch[name] = torch.tensor([[index, index + 10, index + 20]])
    system = ModeSensitiveActionBlindSystem()
    grad_output = system.score_observed_batch(batch)
    with torch.no_grad():
        no_grad_output = system.score_observed_batch(batch)
    assert not torch.equal(
        grad_output["progress_logit"], no_grad_output["progress_logit"]
    )
    assert _action_shuffle_invariant(system, batch, torch.tensor([2, 0, 1]))


def test_eval_auc_uses_average_ranks_for_score_ties() -> None:
    assert _auc(
        labels=torch.tensor([True, False]).numpy(),
        scores=torch.tensor([4.0, 4.0]).numpy(),
    ) == 0.5
    labels = torch.tensor([False, True, True, False]).numpy()
    scores = torch.tensor([0.0, 1.0, 1.0, 1.0]).numpy()
    assert _auc(labels=labels, scores=scores) == 0.75
    permutation = [2, 3, 0, 1]
    assert _auc(labels=labels[permutation], scores=scores[permutation]) == 0.75


def test_masked_native_evidence_is_not_treated_as_a_real_zero() -> None:
    torch.manual_seed(19)
    planner = _planner().eval()
    evidence = _evidence()
    missing = copy.deepcopy(evidence)
    missing["depth"].zero_()
    missing["depth_mask"].zero_()
    missing["point"].zero_()
    missing["point_mask"].zero_()
    missing["pose"].zero_()
    missing["pose_mask"].zero_()
    invalid_values = copy.deepcopy(missing)
    invalid_values["depth"].fill_(1e6)
    invalid_values["point"].fill_(-1e6)
    invalid_values["pose"].fill_(1e6)
    missing_out = _forward(planner, missing)
    invalid_out = _forward(planner, invalid_values)
    assert all(torch.equal(missing_out[name], invalid_out[name]) for name in missing_out)
    measured_zero = copy.deepcopy(missing)
    measured_zero["depth_mask"].fill_(True)
    measured_zero["point_mask"].fill_(True)
    measured_zero["pose_mask"].fill_(True)
    measured_out = _forward(planner, measured_zero)
    assert any(not torch.equal(missing_out[name], measured_out[name]) for name in missing_out)


def test_label_shuffle_sensitivity_and_finite_planner_gradients() -> None:
    torch.manual_seed(11)
    planner = _planner().train()
    evidence = _evidence()
    evidence = {name: value[:1] for name, value in evidence.items()}
    labels = torch.tensor([[[0,0,0,0],[0,1,1,1],[0,0,0,0]]], dtype=torch.bool)
    def loss_for(current: torch.Tensor) -> torch.Tensor:
        output = _forward(planner, evidence)
        output["score"] = planning_score(output, torch.zeros(1,3))
        return planner_loss(output, branch_rewards=current.float(), branch_dones=torch.zeros_like(current),
            branch_success=current, branch_valid=torch.ones(1,3,dtype=torch.bool),
            uncertainty_target=torch.zeros(1,3))["loss"]
    original = loss_for(labels)
    shuffled = loss_for(labels[:, torch.tensor([1,0,2])])
    assert not torch.allclose(original, shuffled)
    original.backward()
    grads=[p.grad for p in planner.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(value).all() for value in grads)
    assert sum(float(value.abs().sum()) for value in grads) > 0


def test_grouped_action_cost_is_masked_and_not_fixed_7d() -> None:
    actions=torch.zeros(2,3,4,2,5,9)
    mask=torch.zeros_like(actions,dtype=torch.bool)
    samples=torch.zeros(actions.shape[:-1],dtype=torch.bool)
    mask[...,0,:3]=True
    samples[...,0]=True
    actions[...,0,0]=2
    cost=deterministic_action_cost(actions,mask,samples)
    assert cost.shape==(2,3) and bool((cost>0).all())
    coarse=torch.full((2,3,4,2,9),3.0)
    coarse_mask=torch.ones_like(coarse,dtype=torch.bool)
    coarse_only=deterministic_action_cost(
        torch.zeros_like(actions), torch.zeros_like(mask), torch.zeros_like(samples),
        coarse, coarse_mask)
    assert bool((coarse_only == 3).all())
    with pytest.raises(ValueError, match="grouped"):
        deterministic_action_cost(torch.zeros(2,3,4,7), torch.zeros(2,3,4,7,dtype=torch.bool), torch.zeros(2,3,4,dtype=torch.bool))


def test_branch_clocks_must_match_stage0_window_and_interval_ownership() -> None:
    world_times = torch.tensor(
        [0.0, 0.3, 0.9000001430511475, 1.700000047683716],
        dtype=torch.float64,
    )
    future_dt = world_times[2:] - world_times[1]
    candidate = {
        "branch_future_dt_s": future_dt[None].expand(2, -1).clone(),
        "candidate_fine_action_dt": torch.tensor(
            [[[[.0, .2]], [[.0, .3]]], [[[.1, .2]], [[.1, .3]]]]
        ),
        "candidate_fine_sample_mask": torch.ones(2, 2, 1, 2, dtype=torch.bool),
    }
    sample = {
        "world_times_s": world_times,
        "future_world_boundaries_dt": torch.tensor([0., .4, .9]),
    }
    _validate_stage0_window_clock(candidate, sample=sample, context=2, K=2, horizon=2)
    drifted = copy.deepcopy(candidate)
    drifted["branch_future_dt_s"][0, 0] += .01
    with pytest.raises(ValueError, match="differ from Stage0"):
        _validate_stage0_window_clock(drifted, sample=sample, context=2, K=2, horizon=2)
    escaped = copy.deepcopy(candidate)
    escaped["candidate_fine_action_dt"][0, 0, 0, 1] = .4
    with pytest.raises(ValueError, match="outside its world interval"):
        _validate_stage0_window_clock(escaped, sample=sample, context=2, K=2, horizon=2)
    rounded = copy.deepcopy(candidate)
    rounded["branch_future_dt_s"] = rounded["branch_future_dt_s"].to(torch.float32)
    with pytest.raises(ValueError, match="differ from Stage0"):
        _validate_stage0_window_clock(
            rounded, sample=sample, context=2, K=2, horizon=2
        )


def test_candidate_payload_accepts_exact_ckgsa_grouped_action_abi() -> None:
    C, K, H, G, S, A, V, P, D = 3, 2, 2, 2, 4, 5, 2, 4, 8
    value = {
        "candidate_fine_action_values": torch.ones(C, K, G, S, A),
        "candidate_fine_action_mask": torch.ones(C, K, G, S, A, dtype=torch.bool),
        "candidate_fine_action_dt": torch.zeros(C, K, G, S),
        "candidate_fine_sample_mask": torch.ones(C, K, G, S, dtype=torch.bool),
        "candidate_coarse_action_values": torch.zeros(C, K, G, A),
        "candidate_coarse_action_mask": torch.zeros(C, K, G, A, dtype=torch.bool),
        "branch_future_tokens": torch.randn(C, H, P, D),
        "branch_future_dt_s": torch.tensor([[0.2, 0.5]]).expand(C, -1).clone(),
        "branch_token_mask": torch.ones(C, H, P, dtype=torch.bool),
        "branch_depth": torch.ones(C, H, V, P),
        "branch_depth_mask": torch.ones(C, H, V, P, dtype=torch.bool),
        "branch_point": torch.ones(C, H, V, P, 3),
        "branch_point_mask": torch.ones(C, H, V, P, dtype=torch.bool),
        "branch_camera_pose": torch.ones(C, H, V, 9),
        "branch_camera_pose_mask": torch.ones(C, H, V, dtype=torch.bool),
        "branch_geometry_confidence": torch.ones(C, H, V, P),
        "branch_view_mask": torch.ones(C, H, V, dtype=torch.bool),
        "branch_rewards": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "branch_dones": torch.zeros(C, H, dtype=torch.bool),
        "branch_success": torch.tensor(
            [[False, False], [False, True], [False, False]]
        ),
        "branch_valid": torch.ones(C, dtype=torch.bool),
    }
    _validate_candidate_payload(
        value,
        model={
            "K": K,
            "P": P,
            "token_dim": D,
            "num_views": V,
            "max_action_groups": G,
            "max_action_substeps": S,
            "max_action_dim": A,
        },
        horizon=H,
    )
    malformed = copy.deepcopy(value)
    malformed["candidate_fine_action_values"] = malformed[
        "candidate_fine_action_values"
    ].unsqueeze(0)
    with pytest.raises(ValueError, match="same sealed K"):
        _validate_candidate_payload(
            malformed,
            model={
                "K": K,
                "P": P,
                "token_dim": D,
                "num_views": V,
                "max_action_groups": G,
                "max_action_substeps": S,
                "max_action_dim": A,
            },
            horizon=H,
        )


def test_stage1_dataset_accepts_ckgsa_and_rejects_extra_or_missing_axis() -> None:
    C, K, G, S, A = 11, 8, 8, 128, 16
    payload = {
        "candidate_fine_action_values": torch.zeros(C, K, G, S, A),
        "candidate_fine_action_mask": torch.zeros(
            C, K, G, S, A, dtype=torch.bool
        ),
        "candidate_fine_action_dt": torch.zeros(C, K, G, S),
        "candidate_fine_sample_mask": torch.zeros(C, K, G, S, dtype=torch.bool),
        "candidate_coarse_action_values": torch.zeros(C, K, G, A),
        "candidate_coarse_action_mask": torch.zeros(C, K, G, A, dtype=torch.bool),
    }
    model = {
        "max_action_groups": G,
        "max_action_substeps": S,
        "max_action_dim": A,
    }
    _validate_candidate_action_shapes(payload, candidates=C, K=K, model=model)

    for malformed in (
        payload["candidate_fine_action_values"].unsqueeze(0),
        payload["candidate_fine_action_values"][..., 0],
    ):
        broken = dict(payload)
        broken["candidate_fine_action_values"] = malformed
        with pytest.raises(Stage1BranchError, match="do not cover sealed K"):
            _validate_candidate_action_shapes(broken, candidates=C, K=K, model=model)


def test_stage1_dcp_exact_resume_contract_is_planner_ddp_and_branch_bound() -> None:
    planner=NativePlannerConfig(token_dim=0,task_dim=0,patches=0,num_views=0,max_horizon=0,
        time_fourier_dim=0,time_min_period_s=0,time_max_period_s=0)
    rollout_audit_sha256 = "d" * 64
    stage1={"run":{"lineage":"s1","global_batch_size":8},"branch":{
        "index_sha256":"a"*64, "rollout_audit_sha256": rollout_audit_sha256,
        "stage0_checkpoint_commit_sha256": "e" * 64},
        "planner":{"horizon":4,"model":planner.__dict__,"score":{
            "progress_weight":.5,"success_weight":1.,"risk_weight":.5,
            "uncertainty_weight":.25,"action_cost_weight":.05}}}
    runtime={"bindings":{"model_contract_sha256":"b"*64},"model_profile":{"model":{
        "token_dim":16,"task_dim":12,"P":4,"num_views":2,"time_fourier_dim":8,
        "time_min_period_s":.01,"time_max_period_s":10.0}}}
    expected=_expectations(step=10,stage1=stage1,stage1_sha="c"*64,runtime=runtime,world_size=4)
    assert expected.distributed_strategy == "ddp"
    assert expected.shard_degree == 1 and expected.world_size == 4
    assert expected.allow_topology_reshard is False
    assert expected.topology_contract_sha256 == _topology_sha(stage1,runtime)
    assert expected.model_contract_sha256 == _planner_contract_sha(stage1,runtime)
    assert expected.extra_immutable_metadata == {
        "rollout_audit_sha256": rollout_audit_sha256,
        "stage0_checkpoint_commit_sha256": "e" * 64,
        "branch_index_sha256": "a" * 64,
        "stage0_frozen": True,
        "planner_action_inputs": False,
        "imagined_rollout": "single_trained_K_only",
    }

    metadata = {
        "step": expected.step,
        "run_lineage": expected.run_lineage,
        "runtime_config_sha256": expected.runtime_config_sha256,
        "data_closure_sha256": expected.data_closure_sha256,
        "model_contract_sha256": expected.model_contract_sha256,
        "world_size": expected.world_size,
        "shard_degree": expected.shard_degree,
        "distributed_strategy": expected.distributed_strategy,
        "global_batch_size": expected.global_batch_size,
        "topology_contract_sha256": expected.topology_contract_sha256,
        "rollout_audit_sha256": rollout_audit_sha256,
        "stage0_checkpoint_commit_sha256": "e" * 64,
        "branch_index_sha256": "a" * 64,
        "stage0_frozen": True,
        "planner_action_inputs": False,
        "imagined_rollout": "single_trained_K_only",
    }
    assert _validate_metadata(metadata, expected) == "exact"

    missing = dict(metadata)
    del missing["rollout_audit_sha256"]
    with pytest.raises(CheckpointIntegrityError, match="rollout_audit_sha256"):
        _validate_metadata(missing, expected)

    wrong = dict(metadata, rollout_audit_sha256="e" * 64)
    with pytest.raises(CheckpointIntegrityError, match="rollout_audit_sha256"):
        _validate_metadata(wrong, expected)

    for name, expected_value in expected.extra_immutable_metadata.items():
        missing = dict(metadata)
        del missing[name]
        with pytest.raises(CheckpointIntegrityError, match=name):
            _validate_metadata(missing, expected)
        wrong_value = not expected_value if isinstance(expected_value, bool) else "f" * 64
        wrong = dict(metadata, **{name: wrong_value})
        with pytest.raises(CheckpointIntegrityError, match=name):
            _validate_metadata(wrong, expected)


def test_stage1_authority_coverage_is_full_split_only() -> None:
    assert _require_authority_coverage(
        evaluated_count=7, sealed_split_count=7, gate_count=7
    ) == {
        "evaluated_branch_count": 7,
        "sealed_split_count": 7,
        "gated_branch_count": 7,
    }
    for evaluated, gated in ((6, 7), (7, 6), (0, 0)):
        with pytest.raises(ValueError, match="does not cover"):
            _require_authority_coverage(
                evaluated_count=evaluated,
                sealed_split_count=7,
                gate_count=gated,
            )


def test_stage1_runtime_loader_uses_one_regular_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    planner = NativePlannerConfig(
        token_dim=0,
        task_dim=0,
        patches=0,
        num_views=0,
        max_horizon=0,
        time_fourier_dim=0,
        time_min_period_s=0,
        time_max_period_s=0,
    )
    runtime = {
        "schema": "wm3d_v8_unified_stage1_runtime_v2",
        "stage0_runtime": "/sealed/stage0.yaml",
        "stage0_checkpoint": "/sealed/step_00000001",
        "branch": {
            "index": "/sealed/index.jsonl",
            "index_sha256": "a" * 64,
            "seal": "/sealed/seal.json",
            "seal_sha256": "b" * 64,
            "stage0_runtime_sha256": "c" * 64,
            "stage0_checkpoint_commit_sha256": "d" * 64,
            "rollout_audit_sha256": "e" * 64,
        },
        "planner": {
            "horizon": 4,
            "candidate_microbatch": 1,
            "model": planner.__dict__,
            "loss": {
                "progress_weight": 0.5,
                "success_weight": 1.0,
                "risk_weight": 0.5,
                "uncertainty_weight": 0.25,
                "ranking_weight": 1.0,
                "ranking_margin": 0.05,
            },
            "score": {
                "progress_weight": 0.5,
                "success_weight": 1.0,
                "risk_weight": 0.5,
                "uncertainty_weight": 0.25,
                "action_cost_weight": 0.05,
            },
        },
        "run": {
            "lineage": "s1",
            "output_root": "/sealed/output",
            "seed": 1,
            "total_steps": 2,
            "checkpoint_interval": 1,
            "micro_batch_size": 1,
            "gradient_accumulation": 1,
            "global_batch_size": 1,
            "num_workers": 0,
            "lr": 0.001,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
        },
    }
    path = tmp_path / "stage1.yaml"
    payload = yaml.safe_dump(runtime, sort_keys=True).encode()
    path.write_bytes(payload)
    loaded, digest = _load_stage1(path)
    assert loaded == runtime
    assert digest == hashlib.sha256(payload).hexdigest()
    symlink = tmp_path / "stage1-link.yaml"
    symlink.symlink_to(path)
    with pytest.raises(ValueError, match="stable Stage1 runtime"):
        _load_stage1(symlink)

    replacement = tmp_path / "replacement.yaml"
    replacement.write_bytes(payload)
    original_read = rollout_audit_contract.os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            moved = tmp_path / "replacement-old.yaml"
            replacement.rename(moved)
            replacement.write_bytes(payload)
        return chunk

    monkeypatch.setattr(rollout_audit_contract.os, "read", replacing_read)
    with pytest.raises(ValueError, match="stable Stage1 runtime"):
        _load_stage1(replacement)


def test_stage1_run_contract_binds_sealed_stage0_resources() -> None:
    planner = NativePlannerConfig(
        token_dim=0, task_dim=0, patches=0, num_views=0, max_horizon=0,
        time_fourier_dim=0, time_min_period_s=0, time_max_period_s=0,
    )
    stage1 = {
        "run": {"lineage": "s1", "global_batch_size": 8},
        "branch": {
            "index_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "rollout_audit_sha256": "c" * 64,
            "stage0_checkpoint_commit_sha256": "d" * 64,
        },
        "planner": {"horizon": 4, "model": planner.__dict__, "score": {
            "progress_weight": .5, "success_weight": 1., "risk_weight": .5,
            "uncertainty_weight": .25, "action_cost_weight": .05,
        }},
    }
    stage0 = {
        "run": {"code_commit": "1" * 40, "environment_lock_sha256": "e" * 64},
        "bindings": {"model_contract_sha256": "f" * 64},
        "runtime_profile": {"expected_world_size": 8, "resources": {"gpu": "H200"}},
        "model_profile": {"model": {"token_dim": 16, "task_dim": 12, "P": 4,
            "num_views": 2, "time_fourier_dim": 8, "time_min_period_s": .01,
            "time_max_period_s": 10.0}},
    }
    contract = _stage1_run_contract(
        stage1=stage1, stage1_sha="0" * 64, stage0=stage0,
        stage0_sha="9" * 64, planner_parameter_count=123,
    )
    assert contract["expected_world_size"] == 8
    assert contract["resource_contract_sha256"] == canonical_sha256({"gpu": "H200"})
    assert contract["stage0_frozen"] is True
    assert contract["planner_action_inputs"] is False


@pytest.mark.parametrize("launch_kind", ("fresh", "exact_resume", "eval"))
def test_stage1_launch_qualification_separates_stage1_and_resource_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    launch_kind: str,
) -> None:
    planner = NativePlannerConfig(
        token_dim=0,
        task_dim=0,
        patches=0,
        num_views=0,
        max_horizon=0,
        time_fourier_dim=0,
        time_min_period_s=0,
        time_max_period_s=0,
    )
    stage1 = {
        "run": {
            "lineage": "s1",
            "output_root": str(tmp_path / "output"),
            "global_batch_size": 8,
        },
        "branch": {
            "index_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "rollout_audit_sha256": "c" * 64,
            "stage0_checkpoint_commit_sha256": "d" * 64,
        },
        "planner": {
            "horizon": 4,
            "model": planner.__dict__,
            "score": {
                "progress_weight": 0.5,
                "success_weight": 1.0,
                "risk_weight": 0.5,
                "uncertainty_weight": 0.25,
                "action_cost_weight": 0.05,
            },
        },
    }
    stage0 = {
        "run": {
            "code_commit": "1" * 40,
            "environment_lock_sha256": "e" * 64,
        },
        "bindings": {"model_contract_sha256": "f" * 64},
        "runtime_profile": {
            "expected_world_size": 2,
            "resources": {"gpu": "H200"},
        },
        "model_profile": {
            "model": {
                "token_dim": 16,
                "task_dim": 12,
                "P": 4,
                "num_views": 2,
                "time_fourier_dim": 8,
                "time_min_period_s": 0.01,
                "time_max_period_s": 10.0,
            }
        },
        "data_closure": {"cache_root": str(tmp_path / "cache")},
    }
    context = SimpleNamespace(is_rank0=True, world_size=2)
    receipt = {
        "path": str(tmp_path / "resource.json"),
        "sha256": "9" * 64,
        "created_unix_ns": 1,
    }
    source = None if launch_kind == "fresh" else {"resume_mode": "exact"}
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        stage1_train,
        "_require_recent_resource_preflight",
        lambda config, config_sha, observed_context: (
            calls.update(
                receipt_runtime_sha=config_sha,
                receipt_context=observed_context,
            )
            or receipt
        ),
    )
    monkeypatch.setattr(
        stage1_train,
        "_atomic_json_no_clobber",
        lambda path, value: calls.update(contract_mode="fresh", contract=value),
    )
    monkeypatch.setattr(
        stage1_train,
        "_require_stable_run_contract",
        lambda path, value: calls.update(contract_mode="existing", contract=value),
    )
    monkeypatch.setattr(
        stage1_train.torch.distributed,
        "broadcast_object_list",
        lambda values, src: calls.update(broadcast_src=src),
    )

    def publish(**kwargs: object) -> tuple[str, str]:
        calls.update(kwargs)
        return str(tmp_path / "qualification.json"), "8" * 64

    monkeypatch.setattr(stage1_train, "_publish_and_validate_launch", publish)
    path, digest, contract = _prepare_stage1_launch(
        stage1=stage1,
        stage1_sha="2" * 64,
        stage0=stage0,
        stage0_sha="3" * 64,
        context=context,
        planner_parameter_count=123,
        source_checkpoint=source,
        launch_kind=launch_kind,
    )
    assert path.endswith("qualification.json") and digest == "8" * 64
    assert calls["receipt_runtime_sha"] == "3" * 64
    assert calls["config_sha"] == "2" * 64
    assert calls["resource_runtime_config_sha256"] == "3" * 64
    assert calls["source_checkpoint"] == source
    assert calls["launch_kind"] == launch_kind
    assert calls["contract_mode"] == (
        "fresh" if launch_kind == "fresh" else "existing"
    )
    assert contract["runtime_config_sha256"] == "2" * 64
    assert contract["stage0_runtime_sha256"] == "3" * 64


def test_stage1_run_contract_failure_is_collective_before_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = {
        "run": {
            "lineage": "s1",
            "output_root": str(tmp_path / "output"),
            "global_batch_size": 1,
        },
        "branch": {
            "index_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "rollout_audit_sha256": "c" * 64,
            "stage0_checkpoint_commit_sha256": "d" * 64,
        },
        "planner": {
            "horizon": 1,
            "model": NativePlannerConfig(
                token_dim=0,
                task_dim=0,
                patches=0,
                num_views=0,
                max_horizon=0,
                time_fourier_dim=0,
                time_min_period_s=0,
                time_max_period_s=0,
            ).__dict__,
            "score": {
                "progress_weight": 0.5,
                "success_weight": 1.0,
                "risk_weight": 0.5,
                "uncertainty_weight": 0.25,
                "action_cost_weight": 0.05,
            },
        },
    }
    stage0 = {
        "run": {
            "code_commit": "1" * 40,
            "environment_lock_sha256": "e" * 64,
        },
        "bindings": {"model_contract_sha256": "f" * 64},
        "runtime_profile": {"expected_world_size": 1, "resources": None},
        "model_profile": {
            "model": {
                "token_dim": 16,
                "task_dim": 12,
                "P": 4,
                "num_views": 2,
                "time_fourier_dim": 8,
                "time_min_period_s": 0.01,
                "time_max_period_s": 10.0,
            }
        },
        "data_closure": {"cache_root": str(tmp_path / "cache")},
    }
    context = SimpleNamespace(is_rank0=True, world_size=1)
    monkeypatch.setattr(
        stage1_train,
        "_require_recent_resource_preflight",
        lambda config, config_sha, observed_context: None,
    )
    monkeypatch.setattr(
        stage1_train,
        "_atomic_json_no_clobber",
        lambda path, value: (_ for _ in ()).throw(ValueError("sealed mismatch")),
    )
    broadcasted: list[dict[str, object]] = []
    monkeypatch.setattr(
        stage1_train.torch.distributed,
        "broadcast_object_list",
        lambda values, src: broadcasted.append(dict(values[0])),
    )
    monkeypatch.setattr(
        stage1_train,
        "_publish_and_validate_launch",
        lambda **kwargs: pytest.fail("qualification must not be published"),
    )
    with pytest.raises(ValueError, match="run-contract publication failed"):
        _prepare_stage1_launch(
            stage1=stage1,
            stage1_sha="2" * 64,
            stage0=stage0,
            stage0_sha="3" * 64,
            context=context,
            planner_parameter_count=1,
            source_checkpoint=None,
            launch_kind="fresh",
        )
    assert broadcasted and broadcasted[0]["ok"] is False


def test_stage1_checkpoint_payload_loads_follow_launch_qualification() -> None:
    train_source = Path(stage1_train.__file__).read_text(encoding="utf-8")
    train_qualification = train_source.index(
        "launch_path, launch_sha, _run_contract = _prepare_stage1_launch("
    )
    train_stage0_load = train_source.index(
        "source_manager.load_model_for_evaluation("
    )
    train_resume_load = train_source.index("_metadata, progress = manager.load(")
    assert train_qualification < train_stage0_load < train_resume_load

    eval_source = Path(__file__).parents[1] / "scripts" / "eval_wm3d_v8_stage1.py"
    eval_text = eval_source.read_text(encoding="utf-8")
    eval_qualification = eval_text.index(
        "launch_path, launch_sha, _run_contract = _prepare_stage1_launch("
    )
    eval_stage0_load = eval_text.index(
        "stage0_manager.load_model_for_evaluation("
    )
    eval_stage1_load = eval_text.index(
        "stage1_manager.load_model_for_evaluation("
    )
    assert eval_qualification < eval_stage0_load < eval_stage1_load


def _tiny_world() -> NativeWorldModel:
    cfg=NativeWorldModelConfig(T=2,P=4,K=2,token_dim=8,task_dim=6,num_views=2,
        state_hidden=16,state_layers=1,state_heads=4,state_ff_mult=2,
        action_hidden=16,action_layers=1,action_heads=4,action_ff_mult=2,
        bridge_layers_state=(0,),bridge_heads=4,dynamics_layers=1,view_hidden=16,view_heads=4,view_ff_mult=2,
        max_action_groups=2,max_action_dim=5,max_state_dim=6,max_action_substeps=3,max_policy_queries=4,
        max_group_id=8,max_embodiments=8,max_action_semantic_id=16,max_state_semantic_id=16,
        time_fourier_dim=8,max_aux_tokens=2,aux_dim=4,max_aux_type_id=4,rgb_hidden=8,rgb_size=8,
        rgb_decode_indices=(1,),geom_hidden=8,dropout=0,activation_checkpointing=False)
    return NativeWorldModel(cfg)


def _rollout_batch() -> dict[str, torch.Tensor]:
    B,C,T,K,V,P,D,G,S,A,Q=1,3,2,2,2,4,8,2,3,5,2
    batch={
        "world_tokens":torch.randn(B,T,V,P,D),"view_mask":torch.ones(B,T,V,dtype=torch.bool),
        "world_times_s":torch.tensor([[0.,.3,.7,1.1]]),"task_embedding":torch.randn(B,6),
        "history_fine_action_values":torch.randn(B,T,G,S,A),"history_fine_action_mask":torch.ones(B,T,G,S,A,dtype=torch.bool),
        "history_fine_action_dt":torch.tensor([[[[.05,.15,.25]]*G]*T]),"history_fine_sample_mask":torch.ones(B,T,G,S,dtype=torch.bool),
        "history_coarse_action_values":torch.zeros(B,T,G,A),"history_coarse_action_mask":torch.zeros(B,T,G,A,dtype=torch.bool),
        "action_group_ids":torch.tensor([[1,2]]),"action_group_mask":torch.ones(B,G,dtype=torch.bool),
        "action_semantic_ids":torch.ones(B,G,A,dtype=torch.long),"current_state_values":torch.randn(B,G,6),
        "current_state_mask":torch.ones(B,G,6,dtype=torch.bool),"state_semantic_ids":torch.ones(B,G,6,dtype=torch.long),
        "embodiment_ids":torch.ones(B,dtype=torch.long),"policy_query_dt":torch.tensor([[[0.,.2],[0.,.2]]]),
        "policy_query_mask":torch.ones(B,G,Q,dtype=torch.bool),"action_normalization_offset":torch.zeros(B,G,A),
        "action_normalization_scale":torch.ones(B,G,A),"aux_values":torch.zeros(B,T,2,4),
        "aux_mask":torch.zeros(B,T,2,dtype=torch.bool),"aux_type_ids":torch.zeros(B,T,2,dtype=torch.long),
        "candidate_fine_action_values":torch.randn(B,C,K,G,S,A),"candidate_fine_action_mask":torch.ones(B,C,K,G,S,A,dtype=torch.bool),
        "candidate_fine_action_dt":torch.tensor([[[[[.05,.15,.25]]*G]*K]*C]),"candidate_fine_sample_mask":torch.ones(B,C,K,G,S,dtype=torch.bool),
        "candidate_coarse_action_values":torch.zeros(B,C,K,G,A),"candidate_coarse_action_mask":torch.zeros(B,C,K,G,A,dtype=torch.bool),
    }
    return batch


def test_rollout_is_single_trained_horizon_and_gradient_owner_is_planner() -> None:
    torch.manual_seed(3)
    world=_tiny_world().eval()
    batch=_rollout_batch()
    with pytest.raises(ValueError, match="H <= K"):
        single_horizon_native_rollout(world,batch,horizon=3)
    rollout=single_horizon_native_rollout(world,batch,horizon=1,candidate_microbatch=2)
    assert rollout.tokens.shape==(1,3,1,4,8)
    assert rollout.future_dt_s.shape == (1,3,1)
    assert bool((rollout.future_dt_s > 0).all())
    system=NativePlanningSystem(world,Stage1SystemConfig(planner=NativePlannerConfig(
        token_dim=0,task_dim=0,hidden=16,spatial_layers=1,temporal_layers=1,heads=4,mlp_mult=2,
        dropout=0,max_horizon=0,patches=0,num_views=0,time_fourier_dim=0,
        time_min_period_s=0,time_max_period_s=0),horizon=1))
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    cost=deterministic_action_cost(batch["candidate_fine_action_values"],batch["candidate_fine_action_mask"],batch["candidate_fine_sample_mask"])
    out=system.score_rollout(rollout,batch["task_embedding"],cost)
    out["score"].sum().backward()
    assert all(parameter.grad is None for parameter in world.parameters())
    assert any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in system.planner.parameters())
