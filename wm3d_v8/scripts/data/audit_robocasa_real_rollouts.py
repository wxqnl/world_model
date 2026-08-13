#!/usr/bin/env python3
"""Audit pinned real RoboCasa same-root rollouts before V8 re-encoding.

This command does not trust the old D=384 branch features. It verifies the
real simulator payload, its pinned generator/replay code, its root receipts,
and byte-exact equality between factual simulator commands and source rows.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from wm3d_v3.data.manifest_contract import canonical_sha256
from wm3d_v3.stage1_planner.rollout_audit import (
    ROLLOUT_AUDIT_SCHEMA,
    publish_no_clobber,
    read_regular_bytes,
)
from wm3d_v3.stage1_planner.replay_authority import load_replay_authority
from wm3d_v3.training.launch_qualification import verify_clean_runtime_checkout


SCHEMA = ROLLOUT_AUDIT_SCHEMA
RUNTIME_SCHEMA = "wm3d_v7_stage1_planner_same_root_runtime_v3"
LAUNCH_SCHEMA = "wm3d_v7_stage1p_success_pool_runtime_launch_world16_v2"
CANDIDATE_SCHEMA = "wm3d_v7_stage1_planner_candidates_v2"
CANDIDATE_SEAL_SCHEMA = "wm3d_v7_stage1_planner_index_seal_v1"
_LAUNCH_FIELDS = {
    "schema", "host_rank", "num_shards", "first_shard", "last_shard",
    "gpu_ids", "pids", "runtime_generator_sha256", "replay_helper_sha256",
    "candidate_index_sha256", "candidate_seal_sha256",
    "stage0_checkpoint_sha256", "roots", "branches_per_root", "future_frames",
    "same_root_simulator_state_exact", "same_root_render_state_exact",
    "same_root_rgb_canonicalized", "root_rgb_equivalence_contract",
    "root_rgb_canonical_source", "candidate_actions_executed_exact",
    "counterfactual_pose_space", "simulator_bounds_exact", "pseudo_outcomes",
    "future_observation_leakage", "success_aligned_screening",
    "terminal_unrecorded_next_state_diagnostic_skipped", "pass",
}
_CANDIDATE_SEAL_FIELDS = {
    "schema", "kind", "output", "output_sha256", "roots", "splits",
    "payload_schema", "stage0_checkpoint_sha256", "input_indices", "passed",
}
_CANDIDATE_FIELDS = {
    "schema", "root_id", "candidate_path", "payload_sha256", "split",
    "split_group", "task", "task_text", "source_dataset", "episode_id",
    "episode_root_index", "t0", "branch_roles", "candidate_seed",
    "flow_sample_stream_indices", "rejected_flow_samples", "max_flow_attempts",
    "root_context_path", "root_context_sha256", "root_context_index_sha256",
    "stage0_checkpoint", "stage0_checkpoint_sha256", "model_config_sha256",
    "action_audit_sha256", "action_stats_sha256", "future_frames",
    "h32_factual_available", "counterfactual_pose_space",
    "simulator_executable_all_candidates", "future_observation_leakage",
    "max_candidate_roundtrip_pose_abs_error",
}
_RUNTIME_FIELDS = {
    "schema", "root_id", "path", "split", "split_group", "task", "task_text",
    "source_dataset", "root_context_path", "root_context_sha256", "episode_id",
    "episode_root_index", "t0", "branch_roles", "branches", "context_frames",
    "future_frames", "h32_factual_available", "same_root_current_runtime_exact",
    "same_root_simulator_state_exact", "same_root_render_state_exact",
    "root_rgb_equivalence_contract", "root_rgb_equivalence_all_passed",
    "same_root_rgb_canonicalized", "root_rgb_canonical_source",
    "simulator_bounds_exact", "candidate_actions_executed_exact",
    "counterfactual_pose_space", "pseudo_outcomes", "future_observation_leakage",
    "outcome_source", "terminal_positive_branches", "terminal_negative_branches",
    "mixed_terminal_outcomes", "root_state_sha256", "model_xml_sha256",
    "ep_meta_sha256", "root_render_state_sha256", "root_rgb_sha256",
    "root_rgb_raw_sha256", "max_root_rgb_changed_fraction",
    "max_root_rgb_mean_abs", "max_root_rgb_rmse", "max_root_rgb_p99_abs",
    "min_root_rgb_psnr_db", "candidate_index_sha256", "candidate_payload_sha256",
    "payload_sha256", "action_audit_sha256", "stage0_checkpoint_sha256",
    "max_roundtrip_pose_abs_error",
}


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(f"{label} must be a lowercase SHA256 string")
    return value


def _validate_launch(
    launch: object,
    *,
    generator_sha256: str,
    helper_sha256: str,
    candidate_index_sha256: str,
    candidate_seal_sha256: str,
    candidate_seal: dict,
) -> dict:
    if not isinstance(launch, dict) or set(launch) != _LAUNCH_FIELDS:
        raise RuntimeError("legacy launch receipt exact fields mismatch")
    if launch["schema"] != LAUNCH_SCHEMA or launch["pass"] is not True:
        raise RuntimeError("legacy launch receipt schema/pass mismatch")
    for name in (
        "runtime_generator_sha256", "replay_helper_sha256",
        "candidate_index_sha256", "candidate_seal_sha256",
        "stage0_checkpoint_sha256",
    ):
        _sha(launch[name], f"legacy launch {name}")
    expected = {
        "runtime_generator_sha256": generator_sha256,
        "replay_helper_sha256": helper_sha256,
        "candidate_index_sha256": candidate_index_sha256,
        "candidate_seal_sha256": candidate_seal_sha256,
        "stage0_checkpoint_sha256": candidate_seal["stage0_checkpoint_sha256"],
        "roots": candidate_seal["roots"],
        "branches_per_root": 11,
        "future_frames": 32,
    }
    if any(launch[name] != value for name, value in expected.items()):
        raise RuntimeError("legacy launch differs from candidate closure")
    for name in (
        "host_rank", "num_shards", "first_shard", "last_shard", "roots",
        "branches_per_root", "future_frames",
    ):
        _exact_int(launch[name], f"legacy launch {name}")
    if (
        not isinstance(launch["gpu_ids"], list)
        or not launch["gpu_ids"]
        or any(type(value) is not int or value < 0 for value in launch["gpu_ids"])
        or not isinstance(launch["pids"], list)
        or len(launch["pids"]) != len(launch["gpu_ids"])
        or any(type(value) is not int or value <= 0 for value in launch["pids"])
        or launch["num_shards"] <= launch["last_shard"]
        or launch["first_shard"] > launch["last_shard"]
    ):
        raise RuntimeError("legacy launch shard/process closure is invalid")
    true_gates = {
        "same_root_simulator_state_exact", "same_root_render_state_exact",
        "same_root_rgb_canonicalized", "candidate_actions_executed_exact",
        "simulator_bounds_exact", "success_aligned_screening",
        "terminal_unrecorded_next_state_diagnostic_skipped",
    }
    if any(launch[name] is not True for name in true_gates):
        raise RuntimeError("legacy launch gate failed")
    if launch["pseudo_outcomes"] is not False or launch["future_observation_leakage"] is not False:
        raise RuntimeError("legacy launch is not causal real supervision")
    if (
        launch["counterfactual_pose_space"] != "physical_canonical_6d"
        or launch["root_rgb_canonical_source"] != "sealed_root_context"
        or type(launch["root_rgb_equivalence_contract"]) is not str
    ):
        raise RuntimeError("legacy launch action/RGB contract mismatch")
    return launch


def _validate_candidate_seal(
    seal: object,
    *,
    index_path: Path,
    index_sha256: str,
) -> dict:
    if not isinstance(seal, dict) or set(seal) != _CANDIDATE_SEAL_FIELDS:
        raise RuntimeError("candidate index seal exact fields mismatch")
    if (
        seal["schema"] != CANDIDATE_SEAL_SCHEMA
        or seal["passed"] is not True
        or seal["kind"] != "candidates"
        or seal["payload_schema"] != CANDIDATE_SCHEMA
    ):
        raise RuntimeError("candidate index seal schema/pass mismatch")
    _sha(seal["output_sha256"], "candidate seal output SHA")
    _sha(seal["stage0_checkpoint_sha256"], "candidate seal Stage0 SHA")
    if type(seal["output"]) is not str:
        raise RuntimeError("candidate seal output path must be a string")
    output = Path(seal["output"])
    if output.is_symlink() or output.resolve(strict=True) != index_path:
        raise RuntimeError("candidate index seal output path mismatch")
    if seal["output_sha256"] != index_sha256:
        raise RuntimeError("candidate index seal output SHA mismatch")
    roots = _exact_int(seal["roots"], "candidate seal roots", minimum=1)
    splits = seal["splits"]
    if (
        not isinstance(splits, dict)
        or set(splits) != {"train", "val", "test"}
        or any(type(value) is not int or value < 0 for value in splits.values())
        or sum(splits.values()) != roots
    ):
        raise RuntimeError("candidate seal split counts mismatch")
    inputs = seal["input_indices"]
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError("candidate seal input closure is empty")
    for value in inputs:
        if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
            raise RuntimeError("candidate seal input fields mismatch")
        if type(value["path"]) is not str:
            raise RuntimeError("candidate seal input path must be a string")
        _sha(value["sha256"], "candidate seal input SHA")
        _path, _payload, observed = read_regular_bytes(
            Path(value["path"]), "candidate seal input"
        )
        if observed != value["sha256"]:
            raise RuntimeError("candidate seal input SHA mismatch")
    return seal


def _publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    publish_no_clobber(path, payload, "rollout audit")


def _constants(payload: bytes, path: Path) -> dict[str, object]:
    tree = ast.parse(payload.decode("utf-8"), filename=str(path))
    names = {
        "ROBOSUITE_VERSION", "ROBOSUITE_COMMIT", "MUJOCO_VERSION",
        "ROBOCASA_COMMIT", "ROBOCASA_DATASET_VERSION", "SOURCE_REVISION",
        "CAMERAS", "ACTION_KEY_ORDERING_HDF5",
    }
    result: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != names:
        raise RuntimeError(f"replay helper lacks pinned constants: {sorted(names-set(result))}")
    return result


def _rows(runtime_root: Path) -> dict[str, tuple[dict, Path, str]]:
    result: dict[str, tuple[dict, Path, str]] = {}
    for shard in sorted(runtime_root.glob("index.shard-*.jsonl")):
        if shard.name.endswith(".partial"):
            continue
        shard_path, payload, digest = read_regular_bytes(shard, "runtime index shard")
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            raw = json.loads(line)
            root_id = str(raw.get("root_id", ""))
            if not root_id or root_id in result:
                raise RuntimeError(f"invalid/duplicate root in {shard}:{line_number}")
            result[root_id] = (raw, shard_path, digest)
    return result


def _candidate_rows(payload: bytes, path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        root_id = str(row.get("root_id", ""))
        if not root_id or root_id in result:
            raise RuntimeError(f"invalid/duplicate candidate root at {path}:{line_number}")
        if not isinstance(row, dict) or set(row) != _CANDIDATE_FIELDS:
            raise RuntimeError(f"candidate index fields mismatch at {path}:{line_number}")
        if row.get("schema") != CANDIDATE_SCHEMA:
            raise RuntimeError(f"candidate index schema mismatch at {path}:{line_number}")
        if type(row.get("candidate_seed")) is not int:
            raise RuntimeError(f"candidate seed is absent at {path}:{line_number}")
        for field in (
            "root_id", "payload_sha256", "root_context_sha256",
            "root_context_index_sha256", "stage0_checkpoint_sha256",
            "model_config_sha256", "action_audit_sha256", "action_stats_sha256",
        ):
            _sha(row[field], f"candidate row {field}")
        for field in (
            "episode_id", "episode_root_index", "t0", "candidate_seed",
            "rejected_flow_samples", "max_flow_attempts", "future_frames",
        ):
            _exact_int(row[field], f"candidate row {field}")
        if (
            row["simulator_executable_all_candidates"] is not True
            or row["h32_factual_available"] is not True
            or row["future_observation_leakage"] is not False
            or row["counterfactual_pose_space"] != "physical_canonical_6d"
            or not isinstance(row["branch_roles"], list)
            or len(row["branch_roles"]) != 10
        ):
            raise RuntimeError("candidate row executable/causal contract mismatch")
        result[root_id] = row
    if not result:
        raise RuntimeError("candidate index is empty")
    return result


def _episode_action(
    root: Path,
    episode_id: int,
    *,
    episodes_payload: bytes,
) -> np.ndarray:
    metadata = None
    for line in episodes_payload.decode("utf-8").splitlines():
        row = json.loads(line)
        if int(row["episode_index"]) == episode_id:
            metadata = row
            break
    if metadata is None:
        raise RuntimeError(f"source episode {episode_id} is absent")
    relative = f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
    _path, payload, _digest = read_regular_bytes(
        root / relative, "source episode payload"
    )
    action = np.asarray(
        pq.read_table(pa.BufferReader(payload), columns=["action"]).column(0).to_pylist(),
        dtype=np.float32,
    )
    if action.shape != (int(metadata["length"]), 12) or not np.isfinite(action).all():
        raise RuntimeError("source action payload shape/values are invalid")
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--runtime-generator", type=Path, required=True)
    parser.add_argument("--replay-helper", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--candidate-index-seal", type=Path, required=True)
    parser.add_argument("--replay-authority", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="SOURCE_NAME=/absolute/LeRobot/root; repeat for every selected source",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    code_commit = verify_clean_runtime_checkout(repo, args.code_commit)
    replay_authority, replay_authority_digest = load_replay_authority(
        args.replay_authority, expected_code_commit=code_commit
    )
    replay_authority_rows = {
        row["root_id"]: row for row in replay_authority["rows"]
    }
    selection_path, _selection_payload, selection_digest = read_regular_bytes(
        args.selection_manifest, "selection manifest"
    )
    if (
        str(selection_path) != replay_authority["selection_manifest_path"]
        or selection_digest != replay_authority["selection_manifest_sha256"]
    ):
        raise RuntimeError("audit selection manifest differs from replay authority")

    if args.runtime_root.is_symlink():
        raise RuntimeError("runtime root must not be a symlink")
    runtime_root = args.runtime_root.resolve(strict=True)
    if not runtime_root.is_dir():
        raise RuntimeError("runtime root must be a real directory")
    source_roots: dict[str, Path] = {}
    for raw in args.source_root:
        source_name, separator, raw_path = raw.partition("=")
        if not separator or not source_name or source_name in source_roots:
            raise RuntimeError(f"invalid/duplicate source root {raw!r}")
        raw_root = Path(raw_path)
        if raw_root.is_symlink():
            raise RuntimeError("source root must not be a symlink")
        source_root = raw_root.resolve(strict=True)
        if not source_root.is_dir():
            raise RuntimeError("source root must be a real directory")
        source_roots[source_name] = source_root
    launch_path, launch_payload, launch_digest = read_regular_bytes(
        args.launch_receipt, "launch receipt"
    )
    generator, generator_payload, generator_digest = read_regular_bytes(
        args.runtime_generator, "runtime generator"
    )
    replay, replay_payload, replay_digest = read_regular_bytes(
        args.replay_helper, "replay helper"
    )
    action_audit_path, action_audit_payload, action_audit_digest = read_regular_bytes(
        args.action_audit, "action audit"
    )
    candidate_index_path, candidate_index_payload, candidate_index_digest = (
        read_regular_bytes(args.candidate_index, "candidate index")
    )
    candidate_seal_path, candidate_seal_payload, candidate_seal_digest = (
        read_regular_bytes(args.candidate_index_seal, "candidate index seal")
    )
    launch = json.loads(launch_payload)
    action_audit = json.loads(action_audit_payload)
    if (
        action_audit_digest != "f0e3c99f5f5792d996473bf47035d989ebe17ff2665a3c5c07c73ea736a3c27f"
        or action_audit.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError("factual action semantics audit is invalid")
    candidate_seal = _validate_candidate_seal(
        json.loads(candidate_seal_payload),
        index_path=candidate_index_path,
        index_sha256=candidate_index_digest,
    )
    _validate_launch(
        launch,
        generator_sha256=generator_digest,
        helper_sha256=replay_digest,
        candidate_index_sha256=candidate_index_digest,
        candidate_seal_sha256=candidate_seal_digest,
        candidate_seal=candidate_seal,
    )
    candidates = _candidate_rows(candidate_index_payload, candidate_index_path)
    pinned = _constants(replay_payload, replay)
    camera_order = list(pinned["CAMERAS"])
    if camera_order != [
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"
    ]:
        raise RuntimeError("simulator camera ordering differs from V8 adapter")

    selection_rows = replay_authority["rows"]
    selections: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in selection_rows:
        selections[row["split"]].append(row["root_id"])
    selected_roots = {
        root_id for roots in selections.values() for root_id in roots
    }
    if set(selected_roots) != set(replay_authority_rows):
        raise RuntimeError("selection differs from fresh replay authority coverage")
    available = _rows(runtime_root)
    source_meta: dict[str, dict[str, str]] = {}
    source_episode_metadata: dict[str, bytes] = {}
    for source_name, source_root in source_roots.items():
        digests: dict[str, str] = {}
        for name in ("info.json", "modality.json", "embodiment.json", "episodes.jsonl"):
            _path, payload, digest = read_regular_bytes(
                source_root / "meta" / name, f"{source_name} {name}"
            )
            digests[name] = digest
            if name == "episodes.jsonl":
                source_episode_metadata[source_name] = payload
        source_meta[source_name] = digests
    observed_hz = float(action_audit["audit"]["observed_hz"])
    if not np.isfinite(observed_hz) or observed_hz <= 0:
        raise RuntimeError("action audit does not seal a valid observed cadence")
    simulator_action_period_s = 1.0 / observed_hz
    audited = []
    for split in ("train", "val", "test"):
      for root_id in selections[split]:
        if root_id not in available:
            raise RuntimeError(f"selected root is absent: {root_id}")
        row, shard, shard_digest = available[root_id]
        row_source_path = Path(row.get("source_dataset", ""))
        if row_source_path.is_symlink():
            raise RuntimeError("runtime row source root must not be a symlink")
        row_source_root = row_source_path.resolve(strict=True)
        matching_sources = [
            name for name, root in source_roots.items() if root == row_source_root
        ]
        if len(matching_sources) != 1:
            raise RuntimeError(
                f"selected root {root_id} belongs to an undeclared/ambiguous source root"
            )
        source_name = matching_sources[0]
        source_root = source_roots[source_name]
        candidate = candidates.get(root_id)
        if candidate is None:
            raise RuntimeError(f"selected root has no sealed candidate row: {root_id}")
        replay_row = replay_authority_rows[root_id]
        required_true = (
            "candidate_actions_executed_exact", "same_root_current_runtime_exact",
            "same_root_simulator_state_exact", "same_root_render_state_exact",
            "root_rgb_equivalence_all_passed", "simulator_bounds_exact",
        )
        if not isinstance(row, dict) or set(row) != _RUNTIME_FIELDS:
            raise RuntimeError(f"selected root {root_id} runtime fields mismatch")
        if row.get("schema") != RUNTIME_SCHEMA or any(row.get(name) is not True for name in required_true):
            raise RuntimeError(f"selected root {root_id} did not pass real-simulator gates")
        if row.get("pseudo_outcomes") is not False or row.get("future_observation_leakage") is not False:
            raise RuntimeError(f"selected root {root_id} is not causal real supervision")
        if row.get("split") != split:
            raise RuntimeError("selected root split/source differs from requested closure")
        if (
            int(candidate.get("episode_id", -1)) != int(row.get("episode_id", -2))
            or int(candidate.get("episode_root_index", -1)) != int(row.get("t0", -2))
            or candidate.get("split") != split
            or candidate.get("payload_sha256") != row.get("candidate_payload_sha256")
            or candidate.get("root_context_sha256") != row.get("root_context_sha256")
        ):
            raise RuntimeError("candidate seed row differs from real-runtime root identity")
        if (
            row["candidate_index_sha256"] != candidate_index_digest
            or row["action_audit_sha256"] != action_audit_digest
            or row["stage0_checkpoint_sha256"] != candidate_seal[
                "stage0_checkpoint_sha256"
            ]
            or row["candidate_payload_sha256"] != candidate[
                "payload_sha256"
            ]
            or int(row["branches"]) != 11
            or int(row["future_frames"]) != 32
        ):
            raise RuntimeError("runtime row differs from sealed candidate/launch closure")
        payload, payload_bytes, payload_digest = read_regular_bytes(
            Path(row["path"]), "runtime payload"
        )
        if payload_digest != row["payload_sha256"]:
            raise RuntimeError("runtime payload SHA drift")
        root_context, _root_context_bytes, root_context_digest = read_regular_bytes(
            Path(row["root_context_path"]), "root context"
        )
        if root_context_digest != row["root_context_sha256"]:
            raise RuntimeError("root context SHA drift")
        episode = int(row["episode_id"])
        t0 = int(row["t0"])
        with np.load(io.BytesIO(payload_bytes), allow_pickle=False) as npz:
            simulator = np.asarray(npz["simulator_actions"], dtype=np.float32)
            branch_rgb = np.asarray(npz["branch_rgb"])
            reward = np.asarray(npz["branch_rewards"])
            done = np.asarray(npz["branch_dones"])
            success = np.asarray(npz["branch_success"])
            if (
                simulator.shape != (int(row["branches"]), 128, 12)
                or branch_rgb.shape[:3] != (int(row["branches"]), 33, 3)
                or reward.shape != done.shape
                or done.shape != success.shape
                or reward.shape != (int(row["branches"]), 32)
            ):
                raise RuntimeError("runtime tensor closure differs from row contract")
            factual_source_order = np.concatenate(
                (simulator[0, :, 7:11], simulator[0, :, 11:12], simulator[0, :, 0:7]),
                axis=-1,
            )
            source = _episode_action(
                source_root,
                episode,
                episodes_payload=source_episode_metadata[source_name],
            )
            source_slice = source[t0 : t0 + 128]
            if source_slice.shape != factual_source_order.shape or not np.array_equal(source_slice, factual_source_order):
                raise RuntimeError("factual simulator actions are not byte-exact source rows")
            outcome_indices = np.asarray((2, 6, 9, 13, 16, 20, 23, 27), dtype=np.int64)
            utility = success[:, outcome_indices].any(axis=1).astype(np.float32)
            utility = utility + reward[:, outcome_indices].max(axis=1)
            if np.unique(utility).size < 2:
                raise RuntimeError("selected K8 outcome times contain no planning signal")
        audited.append({
            "split": split, "source": source_name,
            "root_id": root_id, "episode_id": episode, "t0": t0,
            "task_text": row["task_text"], "runtime_payload_path": str(payload),
            "runtime_payload_sha256": row["payload_sha256"],
            "runtime_index_shard_path": str(shard),
            "runtime_index_shard_sha256": shard_digest,
            "runtime_index_row_sha256": canonical_sha256(row),
            "root_context_path": str(root_context), "root_context_sha256": row["root_context_sha256"],
            "root_state_sha256": row["root_state_sha256"],
            "model_xml_sha256": row["model_xml_sha256"], "ep_meta_sha256": row["ep_meta_sha256"],
            "candidate_seed": int(candidate["candidate_seed"]),
            "candidate_payload_sha256": str(candidate["payload_sha256"]),
            "candidate_index_row_sha256": canonical_sha256(candidate),
            "source_action_slice_sha256": canonical_sha256(source_slice.tolist()),
            "factual_simulator_action_source_byte_exact": True,
            "candidate_actions_executed_exact": True,
            "real_simulator_outcomes": True,
            "future_observation_leakage": False,
            "outcome_indices": outcome_indices.tolist(),
            "future_offsets_seconds": [0.6, 1.4, 2.0, 2.8, 3.4, 4.2, 4.8, 5.6],
            "branch_rgb_indices": [3, 7, 10, 14, 17, 21, 24, 28],
            "source_future_row_offsets": [12, 28, 40, 56, 68, 84, 96, 112],
            "candidate_count": int(row["branches"]),
            "replay_authority_row_sha256": canonical_sha256(replay_row),
        })
    audited.sort(key=lambda row: (str(row["split"]), int(row["episode_id"]), int(row["t0"]), str(row["root_id"])))
    simulator_revision = {
        "source_repo": "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos",
        "source_revision": pinned["SOURCE_REVISION"],
        "robocasa_commit": pinned["ROBOCASA_COMMIT"],
        "robocasa_dataset_version": pinned["ROBOCASA_DATASET_VERSION"],
        "robosuite_version": pinned["ROBOSUITE_VERSION"],
        "robosuite_commit": pinned["ROBOSUITE_COMMIT"],
        "mujoco_version": pinned["MUJOCO_VERSION"],
    }
    receipt = {
        "schema": SCHEMA,
        "code_commit": code_commit,
        "runtime_root": str(runtime_root),
        "launch_receipt_path": str(launch_path), "launch_receipt_sha256": launch_digest,
        "runtime_generator_path": str(generator),
        "runtime_generator_sha256": generator_digest,
        "replay_helper_path": str(replay), "replay_helper_sha256": replay_digest,
        "action_audit_path": str(action_audit_path),
        "action_audit_sha256": action_audit_digest,
        "candidate_index_path": str(candidate_index_path),
        "candidate_index_sha256": candidate_index_digest,
        "candidate_index_seal_path": str(candidate_seal_path),
        "candidate_index_seal_sha256": candidate_seal_digest,
        "replay_authority_path": str(args.replay_authority.absolute()),
        "replay_authority_sha256": replay_authority_digest,
        "selection_manifest_path": str(selection_path),
        "selection_manifest_sha256": selection_digest,
        "source_roots": {name: str(root) for name, root in source_roots.items()},
        "source_metadata_sha256": source_meta,
        "simulator_revision": simulator_revision,
        "simulator_revision_sha256": canonical_sha256(simulator_revision),
        "camera_order": camera_order,
        "simulator_action_order": ["eef_position3", "eef_rotation3", "gripper_close1", "base_motion4", "control_mode1"],
        "source_action_order": ["base_motion4", "control_mode1", "eef_position3", "eef_rotation3", "gripper_close1"],
        "simulator_action_period_seconds": simulator_action_period_s,
        "selection_count": {split: len(roots) for split, roots in selections.items()},
        "rows": audited,
        "rows_sha256": canonical_sha256(audited),
        "passed": True,
    }
    _publish(args.output, receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
