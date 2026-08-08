#!/usr/bin/env python3
"""Execute harvested V7 Stage1-P candidates in the pinned RoboCasa runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cache_robocasa365_v7_compact import _load_adapter
from scripts.generate_robocasa_same_root_cf import (
    _episode,
    _make_env,
    _reset_episode,
    _roll_branch,
    array_bytes_equal,
    sha256_array,
)
from wm3d_v3.data.v7_action_contract import (
    canonicalize_dense_action,
    resample_canonical_actions,
)
from wm3d_v3.stage1_planner.action_bridge import canonical_model_actions_to_simulator


SCHEMA = "wm3d_v7_stage1_planner_same_root_runtime_v1"
ROOT_CONTEXT_SCHEMA = "wm3d_v7_stage1_planner_root_context_v1"


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def reduce_outcomes(values: np.ndarray, *, kind: str) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 128:
        raise ValueError("native outcomes must be [C,128]")
    shaped = values.reshape(values.shape[0], 32, 4)
    if kind == "reward":
        return shaped.max(axis=-1)
    if kind in {"done", "success"}:
        return shaped.any(axis=-1)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard contract")
    rows = [
        json.loads(line)
        for line in args.candidate_index.read_text().splitlines()
        if line.strip()
    ][args.shard_index :: args.num_shards]
    if args.max_roots > 0:
        rows = rows[: args.max_roots]
    if not rows:
        raise RuntimeError("runtime generation shard is empty")
    adapter = _load_adapter(args.action_audit, allow_legacy_proof_audit=False)
    candidate_index_sha = sha256_file(args.candidate_index)
    action_audit_sha = sha256_file(args.action_audit)
    output_rows: list[dict] = []
    current_dataset: Path | None = None
    env = None
    try:
        for row in sorted(rows, key=lambda value: (value["source_dataset"], value["episode_id"], value["t0"])):
            if row.get("future_observation_leakage") is not False:
                raise RuntimeError("candidate row does not prohibit future observations")
            dataset = Path(row["source_dataset"])
            if current_dataset != dataset:
                if env is not None:
                    env.close()
                env, _metadata = _make_env(dataset, render_rgb=True)
                current_dataset = dataset
            episode = _episode(dataset, int(row["episode_id"]))
            t0 = int(row["t0"])
            root_context_path = Path(row["root_context_path"])
            root_context_sha = sha256_file(root_context_path)
            if root_context_sha != row["root_context_sha256"]:
                raise RuntimeError(f"root-context SHA mismatch: {row['root_id']}")
            with np.load(root_context_path, allow_pickle=False) as root_context:
                if str(root_context["schema"].item()) != ROOT_CONTEXT_SCHEMA:
                    raise RuntimeError("root-context payload schema mismatch")
                if str(root_context["root_id"].item()) != row["root_id"]:
                    raise RuntimeError("root-context identity mismatch")
                context_root_rgb = np.asarray(root_context["root_rgb"], dtype=np.uint8)
                context_root_state = np.asarray(root_context["root_state"])
            factual_dense = episode["actions"][t0 : t0 + 128]
            if factual_dense.shape != (128, 12):
                raise RuntimeError(f"short H32 factual horizon: {row['root_id']}")
            with np.load(row["candidate_path"], allow_pickle=False) as candidate:
                if str(candidate["schema"].item()) != "wm3d_v7_stage1_planner_candidates_v1":
                    raise RuntimeError("candidate payload schema mismatch")
                if str(candidate["root_id"].item()) != row["root_id"]:
                    raise RuntimeError("candidate/root identity mismatch")
                roles = tuple(str(value) for value in candidate["branch_roles"].tolist())
                proposals = np.asarray(candidate["branch_actions_physical"], dtype=np.float32)
                action_history = np.asarray(candidate["action_history_physical"], dtype=np.float32)
                candidate_context_sha = str(candidate["root_context_sha256"].item())
                candidate_stage0_sha = str(candidate["stage0_checkpoint_sha256"].item())
            if proposals.shape != (10, 32, 7) or action_history.shape != (4, 7):
                raise RuntimeError(f"candidate/action-history shape mismatch: {row['root_id']}")
            if candidate_context_sha != root_context_sha:
                raise RuntimeError(f"candidate/root-context mismatch: {row['root_id']}")
            if candidate_stage0_sha != row["stage0_checkpoint_sha256"]:
                raise RuntimeError(f"candidate/Stage0 provenance mismatch: {row['root_id']}")
            _reset_episode(env, episode)
            low, high = (np.asarray(value, dtype=np.float64) for value in env.action_spec)
            simulator_proposals = []
            roundtrip_pose_error = []
            for proposal in proposals:
                bridge = canonical_model_actions_to_simulator(
                    proposal,
                    adapter,
                    source_hz=20.0,
                    target_hz=5.0,
                    template=factual_dense,
                    action_low=low,
                    action_high=high,
                )
                simulator_proposals.append(bridge.simulator_actions)
                roundtrip_pose_error.append(bridge.max_pose_abs_error)
            all_dense = np.concatenate(
                (factual_dense[None].astype(np.float32), np.stack(simulator_proposals)),
                axis=0,
            )
            all_roles = ("factual_teacher", *roles)
            factual_physical = resample_canonical_actions(
                canonicalize_dense_action(factual_dense, adapter),
                source_hz=20.0,
                target_hz=5.0,
            )
            all_physical = np.concatenate((factual_physical[None], proposals), axis=0)
            branches = [
                _roll_branch(
                    env,
                    episode,
                    t0,
                    dense,
                    factual=index == 0,
                    render_rgb=True,
                    rgb_stride=4,
                    height=args.height,
                    width=args.width,
                )
                for index, dense in enumerate(all_dense)
            ]
            roots = [branch["root_state"] for branch in branches]
            root_rgbs = [branch["root_rgb"] for branch in branches]
            if not all(array_bytes_equal(roots[0], value) for value in roots[1:]):
                raise RuntimeError(f"same-root state mismatch: {row['root_id']}")
            if not all(array_bytes_equal(root_rgbs[0], value) for value in root_rgbs[1:]):
                raise RuntimeError(f"same-root RGB mismatch: {row['root_id']}")
            if not array_bytes_equal(roots[0], context_root_state):
                raise RuntimeError(f"candidate/context root-state mismatch: {row['root_id']}")
            if not array_bytes_equal(root_rgbs[0], context_root_rgb):
                raise RuntimeError(f"candidate/context root-RGB mismatch: {row['root_id']}")
            branch_rgb = np.stack([branch["rgb"] for branch in branches])
            if branch_rgb.shape[:2] != (11, 33):
                raise RuntimeError(f"model-rate RGB horizon mismatch: {row['root_id']}")
            rewards_native = np.stack([branch["rewards"] for branch in branches])
            dones_native = np.stack([branch["dones"] for branch in branches])
            success_native = np.stack([branch["success"] for branch in branches])
            rewards = reduce_outcomes(rewards_native, kind="reward").astype(np.float32)
            dones = reduce_outcomes(dones_native, kind="done").astype(np.bool_)
            success = reduce_outcomes(success_native, kind="success").astype(np.bool_)
            destination = args.output_root / row["split"] / f"{row['root_id']}.npz"
            atomic_savez(
                destination,
                schema=np.asarray(SCHEMA),
                root_id=np.asarray(row["root_id"]),
                branch_roles=np.asarray(all_roles),
                simulator_actions=all_dense,
                branch_actions_physical=all_physical,
                action_history_physical=action_history,
                root_state=roots[0],
                root_rgb=root_rgbs[0],
                branch_rgb=branch_rgb,
                branch_rewards=rewards,
                branch_dones=dones,
                branch_success=success,
                stage0_checkpoint_sha256=np.asarray(row["stage0_checkpoint_sha256"]),
                root_context_sha256=np.asarray(root_context_sha),
            )
            terminal = success.any(axis=-1)
            output_rows.append({
                "schema": SCHEMA,
                "root_id": row["root_id"],
                "path": str(destination.resolve()),
                "split": row["split"],
                "split_group": row["split_group"],
                "task": row["task"],
                "task_text": row["task_text"],
                "source_dataset": str(dataset.resolve()),
                "root_context_path": str(root_context_path.resolve()),
                "root_context_sha256": root_context_sha,
                "episode_id": int(row["episode_id"]),
                "episode_root_index": int(row["episode_root_index"]),
                "t0": t0,
                "branch_roles": list(all_roles),
                "branches": len(all_roles),
                "context_frames": 16,
                "future_frames": 32,
                "same_root_current_runtime_exact": True,
                "pseudo_outcomes": False,
                "future_observation_leakage": False,
                "outcome_source": "current_pinned_robocasa_simulator",
                "terminal_positive_branches": int(terminal.sum()),
                "terminal_negative_branches": int((~terminal).sum()),
                "mixed_terminal_outcomes": bool(terminal.any() and not terminal.all()),
                "root_state_sha256": sha256_array(roots[0]),
                "root_rgb_sha256": sha256_array(root_rgbs[0]),
                "candidate_index_sha256": candidate_index_sha,
                "candidate_payload_sha256": sha256_file(Path(row["candidate_path"])),
                "action_audit_sha256": action_audit_sha,
                "stage0_checkpoint_sha256": row["stage0_checkpoint_sha256"],
                "max_roundtrip_pose_abs_error": float(max(roundtrip_pose_error)),
            })
            atomic_jsonl(args.output_index.with_suffix(args.output_index.suffix + ".partial"), output_rows)
            print(json.dumps({"root_id": row["root_id"], "status": "executed"}), flush=True)
    finally:
        if env is not None:
            env.close()
    os.replace(args.output_index.with_suffix(args.output_index.suffix + ".partial"), args.output_index)


if __name__ == "__main__":
    main()
