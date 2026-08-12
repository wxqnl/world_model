#!/usr/bin/env python3
"""Audit pinned real RoboCasa same-root rollouts before V8 re-encoding.

This command does not trust the old D=384 branch features. It verifies the
real simulator payload, its pinned generator/replay code, its root receipts,
and byte-exact equality between factual simulator commands and source rows.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pyarrow.parquet as pq

from wm3d_v3.data.manifest_contract import canonical_sha256, sha256_file


SCHEMA = "wm3d_v8_robocasa_real_rollout_audit_v1"
RUNTIME_SCHEMA = "wm3d_v7_stage1_planner_same_root_runtime_v3"


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical audit: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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


def _rows(runtime_root: Path) -> dict[str, tuple[dict, Path]]:
    result: dict[str, tuple[dict, Path]] = {}
    for shard in sorted(runtime_root.glob("index.shard-*.jsonl")):
        if shard.name.endswith(".partial") or shard.is_symlink() or not shard.is_file():
            continue
        for line_number, line in enumerate(shard.read_text(encoding="utf-8").splitlines(), 1):
            raw = json.loads(line)
            root_id = str(raw.get("root_id", ""))
            if not root_id or root_id in result:
                raise RuntimeError(f"invalid/duplicate root in {shard}:{line_number}")
            result[root_id] = (raw, shard.resolve(strict=True))
    return result


def _candidate_rows(path: Path) -> dict[str, dict]:
    path = _regular(path, "candidate index")
    result: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        root_id = str(row.get("root_id", ""))
        if not root_id or root_id in result:
            raise RuntimeError(f"invalid/duplicate candidate root at {path}:{line_number}")
        if row.get("schema") != "wm3d_v7_stage1_planner_candidates_v2":
            raise RuntimeError(f"candidate index schema mismatch at {path}:{line_number}")
        if not isinstance(row.get("candidate_seed"), int):
            raise RuntimeError(f"candidate seed is absent at {path}:{line_number}")
        result[root_id] = row
    if not result:
        raise RuntimeError("candidate index is empty")
    return result


def _episode_action(root: Path, episode_id: int) -> np.ndarray:
    metadata = None
    for line in _regular(root / "meta" / "episodes.jsonl", "episode metadata").read_text().splitlines():
        row = json.loads(line)
        if int(row["episode_index"]) == episode_id:
            metadata = row
            break
    if metadata is None:
        raise RuntimeError(f"source episode {episode_id} is absent")
    relative = f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
    payload = _regular(root / relative, "source episode payload")
    action = np.asarray(pq.read_table(payload, columns=["action"]).column(0).to_pylist(), dtype=np.float32)
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
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="SOURCE_NAME=/absolute/LeRobot/root; repeat for every selected source",
    )
    parser.add_argument(
        "--selection", action="append", required=True,
        help=(
            "SPLIT=ROOT_SHA256; repeat for every distinct real root. "
            "Each split must be represented and a split may contain multiple roots."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve(strict=True)
    if runtime_root.is_symlink() or not runtime_root.is_dir():
        raise RuntimeError("runtime root must be a real directory")
    source_roots: dict[str, Path] = {}
    for raw in args.source_root:
        source_name, separator, raw_path = raw.partition("=")
        if not separator or not source_name or source_name in source_roots:
            raise RuntimeError(f"invalid/duplicate source root {raw!r}")
        source_root = Path(raw_path).resolve(strict=True)
        if source_root.is_symlink() or not source_root.is_dir():
            raise RuntimeError("source root must be a real directory")
        source_roots[source_name] = source_root
    launch_path = _regular(args.launch_receipt, "launch receipt")
    generator = _regular(args.runtime_generator, "runtime generator")
    replay = _regular(args.replay_helper, "replay helper")
    action_audit_path = _regular(args.action_audit, "action audit")
    candidate_index_path = _regular(args.candidate_index, "candidate index")
    candidate_seal_path = _regular(args.candidate_index_seal, "candidate index seal")
    launch = json.loads(launch_path.read_text())
    if launch.get("pass") is not True:
        raise RuntimeError("legacy real-runtime launch did not pass")
    for path, key in ((generator, "runtime_generator_sha256"), (replay, "replay_helper_sha256")):
        if sha256_file(path) != launch.get(key):
            raise RuntimeError(f"{key} differs from launch receipt")
    action_audit = json.loads(action_audit_path.read_text())
    if (
        sha256_file(action_audit_path) != "f0e3c99f5f5792d996473bf47035d989ebe17ff2665a3c5c07c73ea736a3c27f"
        or action_audit.get("audit", {}).get("passed") is not True
    ):
        raise RuntimeError("factual action semantics audit is invalid")
    candidate_seal = json.loads(candidate_seal_path.read_text())
    if (
        candidate_seal.get("passed") is not True
        or Path(candidate_seal.get("output", "")).resolve(strict=True) != candidate_index_path
        or candidate_seal.get("output_sha256") != sha256_file(candidate_index_path)
    ):
        raise RuntimeError("candidate index seal/path/SHA is invalid")
    candidates = _candidate_rows(candidate_index_path)
    pinned = _constants(replay)
    camera_order = list(pinned["CAMERAS"])
    if camera_order != [
        "robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"
    ]:
        raise RuntimeError("simulator camera ordering differs from V8 adapter")

    selections: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    selected_roots: set[str] = set()
    for raw in args.selection:
        split, separator, root_id = raw.partition("=")
        if (
            not separator
            or split not in selections
            or len(root_id) != 64
            or root_id in selected_roots
        ):
            raise RuntimeError(f"invalid/duplicate selection {raw!r}")
        selections[split].append(root_id)
        selected_roots.add(root_id)
    if any(not roots for roots in selections.values()):
        raise RuntimeError("selection must contain at least one distinct train/val/test root")
    available = _rows(runtime_root)
    source_meta = {
        source_name: {
            name: sha256_file(
                _regular(source_root / "meta" / name, f"{source_name} {name}")
            )
            for name in ("info.json", "modality.json", "embodiment.json", "episodes.jsonl")
        }
        for source_name, source_root in source_roots.items()
    }
    observed_hz = float(action_audit["audit"]["observed_hz"])
    if not np.isfinite(observed_hz) or observed_hz <= 0:
        raise RuntimeError("action audit does not seal a valid observed cadence")
    simulator_action_period_s = 1.0 / observed_hz
    audited = []
    for split in ("train", "val", "test"):
      for root_id in selections[split]:
        if root_id not in available:
            raise RuntimeError(f"selected root is absent: {root_id}")
        row, shard = available[root_id]
        row_source_root = Path(row.get("source_dataset", "")).resolve(strict=True)
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
        required_true = (
            "candidate_actions_executed_exact", "same_root_current_runtime_exact",
            "same_root_simulator_state_exact", "same_root_render_state_exact",
            "root_rgb_equivalence_all_passed", "simulator_bounds_exact",
        )
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
        payload = _regular(Path(row["path"]), "runtime payload")
        if sha256_file(payload) != row["payload_sha256"]:
            raise RuntimeError("runtime payload SHA drift")
        root_context = _regular(Path(row["root_context_path"]), "root context")
        if sha256_file(root_context) != row["root_context_sha256"]:
            raise RuntimeError("root context SHA drift")
        episode = int(row["episode_id"])
        t0 = int(row["t0"])
        with np.load(payload, allow_pickle=False) as npz:
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
            source = _episode_action(source_root, episode)
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
            "runtime_index_shard_path": str(shard), "runtime_index_shard_sha256": sha256_file(shard),
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
        "runtime_root": str(runtime_root),
        "launch_receipt_path": str(launch_path), "launch_receipt_sha256": sha256_file(launch_path),
        "runtime_generator_path": str(generator), "runtime_generator_sha256": sha256_file(generator),
        "replay_helper_path": str(replay), "replay_helper_sha256": sha256_file(replay),
        "action_audit_path": str(action_audit_path), "action_audit_sha256": sha256_file(action_audit_path),
        "candidate_index_path": str(candidate_index_path),
        "candidate_index_sha256": sha256_file(candidate_index_path),
        "candidate_index_seal_path": str(candidate_seal_path),
        "candidate_index_seal_sha256": sha256_file(candidate_seal_path),
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
