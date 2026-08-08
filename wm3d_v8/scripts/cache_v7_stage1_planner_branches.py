#!/usr/bin/env python3
"""Encode exact H32 same-root RoboCasa branches for V7 Stage1-P.

Every branch is encoded independently in one VGGT gauge containing the root
frame and all 32 future frames.  Unlike the legacy Stage1 cache, geometry is
stored for every candidate; counterfactual geometry is never copied from the
factual branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cache_robocasa365_v7_compact import _encode_clip, _load_codec
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.stage1.action_window_geometry import VGGT_MODEL_REVISION
from wm3d_v3.stage1_planner.dataset import SCHEMA


RUNTIME_SCHEMA = "wm3d_v7_stage1_planner_same_root_runtime_v1"
ROOT_CONTEXT_SCHEMA = "wm3d_v7_stage1_planner_root_context_v1"
EXPECTED_ROLES = (
    "factual_teacher",
    "direct",
    "flow_0",
    "flow_1",
    "flow_2",
    "flow_3",
    "grip_open",
    "grip_close",
    "arm_hold",
    "pose_reverse",
    "pose_half",
)
CAMERA_INDEX = {"anchor": 0, "alternate": 1, "wrist": 2}


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def encode_all_branches(
    root_rgb: np.ndarray,
    branch_rgb: np.ndarray,
    *,
    encoder: VGGTEncoder,
    codec,
    batch_frames: int,
) -> dict[str, np.ndarray]:
    if branch_rgb.shape[:2] != (len(EXPECTED_ROLES), 33):
        raise ValueError(f"expected branch RGB [11,33,...], got {branch_rgb.shape}")
    if not np.array_equal(
        branch_rgb[:, 0], np.broadcast_to(root_rgb, branch_rgb[:, 0].shape)
    ):
        raise ValueError("branch roots do not byte-match root_rgb")
    codes, scales = [], []
    depth, depth_conf, point, point_conf, pose = [], [], [], [], []
    for branch in branch_rgb:
        # Root and all future frames must stay in one VGGT call/gauge.
        frames = [
            np.asarray(frame, dtype=np.uint8)
            for frame in branch[:, CAMERA_INDEX["anchor"]]
        ]
        encoded = _encode_clip(
            frames,
            encoder=encoder,
            codec=codec,
            batch_frames=batch_frames,
            keep_geometry=True,
        )
        segment_id = np.asarray(encoded["geometry_segment_id"])
        if segment_id.shape != (33,) or np.any(segment_id != segment_id[0]):
            raise RuntimeError("a branch crossed VGGT gauge segments")
        codes.append(encoded["codes"][1:])
        scales.append(encoded["scale"][1:])
        depth.append(encoded["depth_patch"][1:])
        depth_conf.append(encoded["depth_conf_patch"][1:])
        point.append(encoded["point_patch"][1:])
        point_conf.append(encoded["point_conf_patch"][1:])
        pose.append(encoded["pose_enc"][1:])
    return {
        "branch_codes": np.stack(codes),
        "branch_scales": np.stack(scales),
        "branch_depth_tgt": np.stack(depth),
        "branch_depth_conf_tgt": np.stack(depth_conf),
        "branch_point_tgt": np.stack(point),
        "branch_point_conf_tgt": np.stack(point_conf),
        "branch_pose_geom_tgt": np.stack(pose),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-index", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-downstream-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--future-frames", type=int, default=32)
    parser.add_argument("--batch-frames", type=int, default=33)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if (args.context_frames, args.future_frames) != (16, 32):
        raise SystemExit("V7 Stage1-P cache requires T16/H32")
    if args.batch_frames < 33:
        raise SystemExit("batch-frames must be >=33 to preserve each VGGT gauge")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard contract")
    downstream = json.loads(args.codec_downstream_report.read_text())
    if not bool(downstream.get("formal_cache_allowed")) or not bool(
        downstream.get("strict_train_split")
    ):
        raise SystemExit("strict token-codec downstream gate failed")
    all_rows = [
        json.loads(line)
        for line in args.runtime_index.read_text().splitlines()
        if line.strip()
    ]
    rows = all_rows[args.shard_index :: args.num_shards]
    if args.max_roots > 0:
        rows = rows[: args.max_roots]
    if not rows:
        raise SystemExit("runtime index shard is empty")
    seen: set[str] = set()
    for row in rows:
        root_id = str(row.get("root_id", ""))
        if not root_id or root_id in seen:
            raise RuntimeError(f"blank/duplicate root_id: {root_id!r}")
        seen.add(root_id)
        if row.get("schema") != RUNTIME_SCHEMA:
            raise RuntimeError(f"runtime schema mismatch: {root_id}")
        if tuple(row.get("branch_roles") or ()) != EXPECTED_ROLES:
            raise RuntimeError(f"branch role mismatch: {root_id}")
        if not row.get("same_root_current_runtime_exact"):
            raise RuntimeError(f"non-exact root: {root_id}")
        if row.get("pseudo_outcomes") is not False:
            raise RuntimeError(f"pseudo outcomes forbidden: {root_id}")
        if row.get("future_observation_leakage") is not False:
            raise RuntimeError(f"future leakage contract missing: {root_id}")
        if not row.get("split_group"):
            raise RuntimeError(f"split-group provenance missing: {root_id}")

    device = torch.device(args.device)
    codec = _load_codec(args.codec, device)
    encoder = VGGTEncoder(
        device=str(device),
        return_depth=True,
        return_depth_conf=True,
        return_geom_extra=True,
        model_revision=VGGT_MODEL_REVISION,
        local_files_only=True,
    )
    runtime_index_sha = sha256_file(args.runtime_index)
    codec_sha = sha256_file(args.codec)
    downstream_sha = sha256_file(args.codec_downstream_report)
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        source_path = Path(row["path"])
        root_context_path = Path(row["root_context_path"])
        root_context_sha = sha256_file(root_context_path)
        if root_context_sha != row["root_context_sha256"]:
            raise RuntimeError(f"root-context SHA mismatch: {row['root_id']}")
        with np.load(root_context_path, allow_pickle=False) as context_archive:
            if str(context_archive["schema"].item()) != ROOT_CONTEXT_SCHEMA:
                raise RuntimeError(f"root-context schema mismatch: {row['root_id']}")
            if str(context_archive["root_id"].item()) != row["root_id"]:
                raise RuntimeError(f"root-context identity mismatch: {row['root_id']}")
            anchor_codes = np.asarray(context_archive["anchor_codes"], dtype=np.int8)
            anchor_scale = np.asarray(context_archive["anchor_scale"], dtype=np.float16)
            wrist_codes = np.asarray(context_archive["wrist_codes"], dtype=np.int8)
            wrist_scale = np.asarray(context_archive["wrist_scale"], dtype=np.float16)
            task_emb = np.asarray(context_archive["task_emb"], dtype=np.float16)
            context_root_rgb = np.asarray(context_archive["root_rgb"], dtype=np.uint8)
        with np.load(source_path, allow_pickle=False) as archive:
            if str(archive["schema"].item()) != RUNTIME_SCHEMA:
                raise RuntimeError(f"payload schema mismatch: {row['root_id']}")
            if str(archive["root_id"].item()) != row["root_id"]:
                raise RuntimeError(f"payload identity mismatch: {row['root_id']}")
            roles = tuple(str(value) for value in archive["branch_roles"].tolist())
            root_rgb = np.asarray(archive["root_rgb"], dtype=np.uint8)
            branch_rgb = np.asarray(archive["branch_rgb"], dtype=np.uint8)
            actions = np.asarray(archive["branch_actions_physical"], dtype=np.float32)
            history = np.asarray(archive["action_history_physical"], dtype=np.float32)
            rewards = np.asarray(archive["branch_rewards"], dtype=np.float32)
            dones = np.asarray(archive["branch_dones"], dtype=np.bool_)
            success = np.asarray(archive["branch_success"], dtype=np.bool_)
            stage0_sha = str(archive["stage0_checkpoint_sha256"].item())
            runtime_context_sha = str(archive["root_context_sha256"].item())
        if roles != EXPECTED_ROLES:
            raise RuntimeError(f"payload role mismatch: {row['root_id']}")
        if root_rgb.ndim != 4 or root_rgb.shape[0] != 3:
            raise RuntimeError(f"root RGB shape mismatch: {row['root_id']}")
        if actions.shape != (11, 32, 7) or history.shape != (4, 7):
            raise RuntimeError(f"action/history shape mismatch: {row['root_id']}")
        if rewards.shape != (11, 32) or dones.shape != rewards.shape or success.shape != rewards.shape:
            raise RuntimeError(f"outcome shape mismatch: {row['root_id']}")
        if stage0_sha != row["stage0_checkpoint_sha256"]:
            raise RuntimeError(f"Stage0 provenance mismatch: {row['root_id']}")
        if runtime_context_sha != root_context_sha:
            raise RuntimeError(f"runtime/root-context mismatch: {row['root_id']}")
        if not np.array_equal(root_rgb, context_root_rgb):
            raise RuntimeError(f"runtime/context root RGB mismatch: {row['root_id']}")
        if anchor_codes.shape != (16, 64, 384) or anchor_scale.shape != (16, 1, 1):
            raise RuntimeError(f"anchor T16 context mismatch: {row['root_id']}")
        if wrist_codes.shape != anchor_codes.shape or wrist_scale.shape != anchor_scale.shape:
            raise RuntimeError(f"wrist T16 context mismatch: {row['root_id']}")
        if task_emb.shape != (2048,) or not np.isfinite(task_emb).all():
            raise RuntimeError(f"task embedding mismatch: {row['root_id']}")
        if not np.isfinite(actions).all() or not np.isfinite(history).all() or not np.isfinite(rewards).all():
            raise RuntimeError(f"non-finite runtime payload: {row['root_id']}")

        branch_payload = encode_all_branches(
            root_rgb,
            branch_rgb,
            encoder=encoder,
            codec=codec,
            batch_frames=args.batch_frames,
        )
        destination = args.output_root / row["split"] / f"{row['root_id']}.npz"
        atomic_savez(
            destination,
            schema=np.asarray(SCHEMA),
            root_id=np.asarray(row["root_id"]),
            episode_id=np.asarray(int(row["episode_id"]), dtype=np.int64),
            episode_root_index=np.asarray(int(row["episode_root_index"]), dtype=np.int64),
            split=np.asarray(row["split"]),
            split_group=np.asarray(row["split_group"]),
            task=np.asarray(row["task"]),
            task_text=np.asarray(row["task_text"]),
            task_emb=task_emb,
            branch_roles=np.asarray(EXPECTED_ROLES),
            anchor_codes=anchor_codes,
            anchor_scale=anchor_scale,
            wrist_codes=wrist_codes,
            wrist_scale=wrist_scale,
            branch_actions_physical=actions,
            action_history_physical=history,
            branch_valid=np.ones(11, dtype=np.bool_),
            branch_rewards=rewards,
            branch_dones=dones,
            branch_success=success,
            factual_index=np.asarray(0, dtype=np.int64),
            direct_index=np.asarray(1, dtype=np.int64),
            stage0_checkpoint_sha256=np.asarray(stage0_sha),
            runtime_payload_sha256=np.asarray(sha256_file(source_path)),
            root_context_sha256=np.asarray(root_context_sha),
            codec_sha256=np.asarray(codec_sha),
            **branch_payload,
        )
        terminal = success.any(axis=-1)
        output_rows.append(
            {
                "schema": SCHEMA,
                "root_id": row["root_id"],
                "path": str(destination.resolve()),
                "split": row["split"],
                "split_group": row["split_group"],
                "task": row["task"],
                "task_text": row["task_text"],
                "source_dataset": row["source_dataset"],
                "t0": int(row["t0"]),
                "episode_id": int(row["episode_id"]),
                "episode_root_index": int(row["episode_root_index"]),
                "branch_roles": list(EXPECTED_ROLES),
                "branches": 11,
                "context_frames": 16,
                "future_frames": 32,
                "action_history_len": 4,
                "factual_index": 0,
                "direct_index": 1,
                "same_root_current_runtime_exact": True,
                "pseudo_outcomes": False,
                "future_observation_leakage": False,
                "all_branch_native_geometry": True,
                "single_vggt_gauge_per_branch": True,
                "outcome_source": "current_pinned_robocasa_simulator",
                "terminal_positive_branches": int(terminal.sum()),
                "terminal_negative_branches": int((~terminal).sum()),
                "mixed_terminal_outcomes": bool(terminal.any() and not terminal.all()),
                "runtime_index_sha256": runtime_index_sha,
                "runtime_payload_sha256": sha256_file(source_path),
                "root_context_path": str(root_context_path.resolve()),
                "root_context_sha256": root_context_sha,
                "context_source": "current_pinned_robocasa_runtime_causal_replay",
                "stage0_checkpoint_sha256": stage0_sha,
                "codec_sha256": codec_sha,
                "geometry_teacher": {
                    "name": "VGGT",
                    "revision": VGGT_MODEL_REVISION,
                    "all_branches": True,
                    "confidence_stored": True,
                },
            }
        )
        atomic_jsonl(
            args.output_index.with_suffix(args.output_index.suffix + ".partial"),
            output_rows,
        )
        print(json.dumps({"root_id": row["root_id"], "status": "cached"}), flush=True)

    del encoder
    torch.cuda.empty_cache()
    partial = args.output_index.with_suffix(args.output_index.suffix + ".partial")
    os.replace(partial, args.output_index)
    report = {
        "schema": SCHEMA,
        "roots": len(output_rows),
        "branches": 11 * len(output_rows),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "splits": {
            split: sum(row["split"] == split for row in output_rows)
            for split in ("train", "val", "test")
        },
        "mixed_terminal_roots": sum(row["mixed_terminal_outcomes"] for row in output_rows),
        "runtime_index_sha256": runtime_index_sha,
        "codec_sha256": codec_sha,
        "codec_downstream_report_sha256": downstream_sha,
        "all_branch_native_geometry": True,
        "single_vggt_gauge_per_branch": True,
        "task_embedding_real": True,
        "output_index": str(args.output_index.resolve()),
        "output_index_sha256": sha256_file(args.output_index),
    }
    args.output_index.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
