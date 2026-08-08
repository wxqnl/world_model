#!/usr/bin/env python3
"""Cache exact causal T16 contexts for V7 Stage1-P roots.

The current pinned RoboCasa runtime is replayed only up to each root.  Sixteen
real 5 Hz observations ending at the root are encoded; no future simulator
step or future observation is read.  These are the exact contexts consumed by
the frozen Stage0 policy when harvesting candidate plans.
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

from scripts.cache_robocasa365_v7_compact import (
    _encode_clip,
    _load_adapter,
    _load_codec,
)
from scripts.generate_robocasa_same_root_cf import (
    _collection_warmup,
    _episode,
    _make_env,
    _render_views,
    _reset_episode,
    sha256_array,
)
from wm3d_v3.data.v7_action_contract import (
    canonicalize_dense_action,
    resample_canonical_actions,
)
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed
from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.stage1.action_window_geometry import VGGT_MODEL_REVISION


SCHEMA = "wm3d_v7_stage1_planner_root_context_v1"
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


def context_state_steps(t0: int, *, frames: int = 16, stride: int = 4) -> tuple[int, ...]:
    """Return the causal native-rate state indices ending exactly at ``t0``."""

    if frames <= 0 or stride <= 0:
        raise ValueError("frames and stride must be positive")
    first = int(t0) - (int(frames) - 1) * int(stride)
    if first < 0:
        raise ValueError(f"root t0={t0} has fewer than {frames} causal frames")
    steps = tuple(range(first, int(t0) + 1, int(stride)))
    if len(steps) != frames or steps[-1] != int(t0):
        raise RuntimeError("causal context schedule construction failed")
    return steps


def render_causal_context(
    env,
    episode: dict[str, Any],
    t0: int,
    *,
    frames: int,
    stride: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    wanted = set(context_state_steps(t0, frames=frames, stride=stride))
    _reset_episode(env, episode)
    _collection_warmup(env, episode)
    rendered: list[np.ndarray] = []
    prefix_l2_max = 0.0
    if 0 in wanted:
        rendered.append(_render_views(env, height, width))
    for action_index in range(t0):
        env.step(episode["actions"][action_index])
        state_step = action_index + 1
        actual = np.asarray(env.sim.get_state().flatten()).copy()
        expected = np.asarray(episode["states"][state_step])
        prefix_l2_max = max(prefix_l2_max, float(np.linalg.norm(actual - expected)))
        if state_step in wanted:
            rendered.append(_render_views(env, height, width))
    if len(rendered) != frames:
        raise RuntimeError(f"rendered {len(rendered)} causal frames, expected {frames}")
    root_state = np.asarray(env.sim.get_state().flatten()).copy()
    return np.stack(rendered), root_state, prefix_l2_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-branch-index", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, action="append", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-downstream-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=16)
    parser.add_argument("--native-stride", type=int, default=4)
    parser.add_argument("--batch-frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard contract")
    if (args.context_frames, args.native_stride) != (16, 4):
        raise SystemExit("V7 Stage1-P requires real T16 at 5 Hz from the 20 Hz runtime")
    if args.batch_frames < args.context_frames:
        raise SystemExit("batch-frames must keep the full T16 context in one VGGT call")
    downstream = json.loads(args.codec_downstream_report.read_text())
    if not bool(downstream.get("formal_cache_allowed")) or not bool(
        downstream.get("strict_train_split")
    ):
        raise SystemExit("strict token-codec downstream gate failed")

    source_payload = json.loads(args.source_audit.read_text())
    source_by_task = {
        str(row["task"]): Path(row["dataset_path"])
        for row in source_payload["rows"]
        if row.get("dataset_path")
    }
    raw_rows = []
    for path in args.raw_manifest:
        raw_rows.extend(
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    raw_by_key = {
        (str(row["task"]), int(row["episode_id"]), int(row["episode_root_index"])): row
        for row in raw_rows
    }
    legacy_all = [
        json.loads(line)
        for line in args.legacy_branch_index.read_text().splitlines()
        if line.strip()
    ]
    legacy_rows = []
    root_signatures: dict[str, tuple[Any, ...]] = {}
    duplicate_root_rows = 0
    for legacy in legacy_all:
        key = (
            str(legacy["task"]),
            int(legacy["episode_id"]),
            int(legacy["episode_root_index"]),
        )
        raw = raw_by_key.get(key)
        if raw is None:
            raise RuntimeError(f"no raw same-root source for {key}")
        root_id = str(raw["root_id"])
        signature = (
            *key,
            int(raw["t0"]),
            str(legacy["split"]),
            str(legacy["split_group"]),
        )
        previous = root_signatures.get(root_id)
        if previous is not None:
            if previous != signature:
                raise RuntimeError(f"duplicate root metadata conflict: {root_id}")
            duplicate_root_rows += 1
            continue
        root_signatures[root_id] = signature
        legacy_rows.append(legacy)
    legacy_rows = legacy_rows[args.shard_index :: args.num_shards]
    if args.max_roots > 0:
        legacy_rows = legacy_rows[: args.max_roots]
    if not legacy_rows:
        raise RuntimeError("root-context shard is empty")

    device = torch.device(args.device)
    adapter = _load_adapter(args.action_audit, allow_legacy_proof_audit=False)
    codec = _load_codec(args.codec, device)
    eligible_task_texts: set[str] = set()
    for legacy in legacy_rows:
        key = (
            str(legacy["task"]),
            int(legacy["episode_id"]),
            int(legacy["episode_root_index"]),
        )
        raw = raw_by_key.get(key)
        if raw is None:
            raise RuntimeError(f"no raw same-root source for {key}")
        try:
            context_state_steps(
                int(raw["t0"]),
                frames=args.context_frames,
                stride=args.native_stride,
            )
        except ValueError:
            continue
        eligible_task_texts.add(str(raw["task_text"]))
    if not eligible_task_texts:
        raise RuntimeError("all selected roots lacked a full causal T16 context")
    task_encoder = QwenVLEmbed(device=str(device))
    task_cache: dict[str, np.ndarray] = {}
    for task_text in sorted(eligible_task_texts):
        task_emb = task_encoder.embed(task_text).numpy().astype(np.float16)
        if task_emb.shape != (2048,) or not np.isfinite(task_emb).all():
            raise RuntimeError(f"invalid task embedding for {task_text!r}")
        task_cache[task_text] = task_emb
    del task_encoder
    torch.cuda.empty_cache()
    encoder = VGGTEncoder(
        device=str(device),
        return_depth=False,
        return_depth_conf=False,
        return_geom_extra=False,
        model_revision=VGGT_MODEL_REVISION,
        local_files_only=True,
    )
    codec_sha = sha256_file(args.codec)
    audit_sha = sha256_file(args.action_audit)
    episode_cache: dict[tuple[str, int], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    skipped_short = 0
    current_dataset: Path | None = None
    env = None
    try:
        for legacy in legacy_rows:
            key = (
                str(legacy["task"]),
                int(legacy["episode_id"]),
                int(legacy["episode_root_index"]),
            )
            raw = raw_by_key.get(key)
            if raw is None:
                raise RuntimeError(f"no raw same-root source for {key}")
            t0 = int(raw["t0"])
            try:
                context_state_steps(
                    t0, frames=args.context_frames, stride=args.native_stride
                )
            except ValueError:
                skipped_short += 1
                continue
            dataset = source_by_task.get(key[0])
            if dataset is None:
                raise RuntimeError(f"task has no pinned dataset source: {key[0]}")
            if current_dataset != dataset:
                if env is not None:
                    env.close()
                env, _metadata = _make_env(dataset, render_rgb=True)
                current_dataset = dataset
            episode_key = (str(dataset), key[1])
            if episode_key not in episode_cache:
                episode_cache[episode_key] = _episode(dataset, key[1])
            episode = episode_cache[episode_key]
            context_rgb, root_state, prefix_l2_max = render_causal_context(
                env,
                episode,
                t0,
                frames=args.context_frames,
                stride=args.native_stride,
                height=args.height,
                width=args.width,
            )
            anchor = _encode_clip(
                [frame[CAMERA_INDEX["anchor"]] for frame in context_rgb],
                encoder=encoder,
                codec=codec,
                batch_frames=args.batch_frames,
                keep_geometry=False,
            )
            wrist = _encode_clip(
                [frame[CAMERA_INDEX["wrist"]] for frame in context_rgb],
                encoder=encoder,
                codec=codec,
                batch_frames=args.batch_frames,
                keep_geometry=False,
            )
            task_text = str(raw["task_text"])
            history_dense = episode["actions"][t0 - 16 : t0]
            history = resample_canonical_actions(
                canonicalize_dense_action(history_dense, adapter),
                source_hz=20.0,
                target_hz=5.0,
            ).astype(np.float32)
            if history.shape != (4, 7):
                raise RuntimeError("causal action-history reconstruction failed")

            destination = args.output_root / legacy["split"] / f"{raw['root_id']}.npz"
            atomic_savez(
                destination,
                schema=np.asarray(SCHEMA),
                root_id=np.asarray(raw["root_id"]),
                anchor_codes=anchor["codes"],
                anchor_scale=anchor["scale"],
                wrist_codes=wrist["codes"],
                wrist_scale=wrist["scale"],
                task_emb=task_cache[task_text],
                action_history_physical=history,
                root_rgb=context_rgb[-1],
                root_state=root_state,
                t0=np.asarray(t0, dtype=np.int64),
                codec_sha256=np.asarray(codec_sha),
                action_audit_sha256=np.asarray(audit_sha),
            )
            rows.append(
                {
                    "schema": SCHEMA,
                    "root_id": raw["root_id"],
                    "path": str(destination.resolve()),
                    "split": legacy["split"],
                    "split_group": legacy["split_group"],
                    "task": raw["task"],
                    "task_text": task_text,
                    "source_dataset": str(dataset.resolve()),
                    "episode_id": key[1],
                    "episode_root_index": key[2],
                    "t0": t0,
                    "context_frames": 16,
                    "native_stride": 4,
                    "context_source": "current_pinned_robocasa_runtime_causal_replay",
                    "future_observation_leakage": False,
                    "max_observation_state_step": t0,
                    "root_state_sha256": sha256_array(root_state),
                    "root_rgb_sha256": sha256_array(context_rgb[-1]),
                    "prefix_historical_l2_max": prefix_l2_max,
                    "codec_sha256": codec_sha,
                    "action_audit_sha256": audit_sha,
                }
            )
            atomic_jsonl(
                args.output_index.with_suffix(args.output_index.suffix + ".partial"),
                rows,
            )
            print(json.dumps({"root_id": raw["root_id"], "status": "cached"}), flush=True)
    finally:
        if env is not None:
            env.close()
    if not rows:
        raise RuntimeError("all selected roots lacked a full causal T16 context")
    partial = args.output_index.with_suffix(args.output_index.suffix + ".partial")
    os.replace(partial, args.output_index)
    report = {
        "schema": SCHEMA,
        "roots": len(rows),
        "skipped_short_context": skipped_short,
        "duplicate_root_rows_removed_before_sharding": duplicate_root_rows,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "causal_t16": True,
        "future_observation_leakage": False,
        "codec_sha256": codec_sha,
        "action_audit_sha256": audit_sha,
        "output_index": str(args.output_index.resolve()),
        "output_index_sha256": sha256_file(args.output_index),
    }
    args.output_index.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
