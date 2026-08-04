#!/usr/bin/env python3
"""Distributed robust action-statistics builder for the WM3D corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from wm3d.data.action import robust_action_normalization
from wm3d.data.contracts import (
    atomic_write_json,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
)
from wm3d.data.sources import EpisodeDescriptor, plan_shard


PARTIAL_SCHEMA = "wm3d_v7_action_stats_partial_v2"
STATS_SCHEMA = "wm3d_v7_action_stats_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    partial = subparsers.add_parser("partial")
    partial.add_argument("--episode-plan", type=Path, required=True)
    partial.add_argument("--output", type=Path, required=True)
    partial.add_argument("--shard-id", type=int, required=True)
    partial.add_argument("--num-shards", type=int, required=True)
    partial.add_argument(
        "--global-sample-budget",
        type=int,
        default=8_000_000,
        help=(
            "Maximum samples across the complete shard set. Each shard "
            "deterministically receives ceil(global/num_shards)."
        ),
    )
    merge = subparsers.add_parser("merge")
    merge.add_argument("--partials", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--clip", type=float, default=5.0)
    return parser.parse_args()


def _episodes(path: Path) -> list[EpisodeDescriptor]:
    result: list[EpisodeDescriptor] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = EpisodeDescriptor.from_mapping(json.loads(line))
                if value.split == "train":
                    result.append(value)
    return result


def _sample_positions(length: int, count: int, seed_text: str) -> np.ndarray:
    count = min(int(count), int(length))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count == length:
        return np.arange(length, dtype=np.int64)
    digest = hashlib.sha256(seed_text.encode()).digest()
    offset = int.from_bytes(digest[:8], "big") / 2**64
    positions = ((np.arange(count, dtype=np.float64) + offset) * length / count).astype(
        np.int64
    )
    return np.minimum(positions, length - 1)


def _episode_sample_positions(
    episodes: list[EpisodeDescriptor],
    *,
    sample_budget: int,
    seed_text: str,
) -> dict[str, np.ndarray]:
    """Allocate one bounded deterministic sample over concatenated episodes."""

    if sample_budget <= 0:
        raise ValueError("sample budget must be positive")
    lengths = np.asarray(
        [episode.data_row_stop - episode.data_row_start for episode in episodes],
        dtype=np.int64,
    )
    if lengths.size == 0 or bool((lengths <= 0).any()):
        raise ValueError("action-statistics episodes have invalid row counts")
    stops = np.cumsum(lengths, dtype=np.int64)
    global_positions = _sample_positions(
        int(stops[-1]),
        min(int(sample_budget), int(stops[-1])),
        seed_text,
    )
    episode_indices = np.searchsorted(stops, global_positions, side="right")
    starts = np.concatenate((np.zeros(1, dtype=np.int64), stops[:-1]))
    result: dict[str, np.ndarray] = {}
    for episode_index, episode in enumerate(episodes):
        selected = global_positions[episode_indices == episode_index]
        if selected.size:
            result[episode.episode_id] = selected - starts[episode_index]
    if sum(len(value) for value in result.values()) != len(global_positions):
        raise RuntimeError("deterministic action-statistics allocation drift")
    return result


def _column_matrix(
    values: list[Any],
    indices: tuple[int, ...],
    *,
    allow_missing: bool,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[1] <= max(indices):
        raise ValueError(
            f"action column has shape {matrix.shape}, cannot select {indices}"
        )
    result = matrix[:, indices]
    valid = np.isfinite(result)
    if not allow_missing and not bool(valid.all()):
        raise ValueError("factual action column contains NaN or Inf")
    return np.where(valid, result, 0.0), valid


def command_partial(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_id < args.num_shards or args.global_sample_budget <= 0:
        raise ValueError("invalid shard or sample budget")
    plan = resolve_regular_file(
        args.episode_plan.parent,
        args.episode_plan.name,
    )
    selected = [
        episode
        for episode in _episodes(plan)
        if plan_shard(episode.episode_id, args.num_shards) == args.shard_id
    ]
    if not selected:
        raise ValueError("action-statistics shard has no training episodes")
    selected = sorted(
        selected,
        key=lambda episode: (episode.source, episode.episode_id),
    )
    plan_sha = sha256_file(plan)
    shard_sample_budget = math.ceil(args.global_sample_budget / args.num_shards)
    positions_by_episode = _episode_sample_positions(
        selected,
        sample_budget=shard_sample_budget,
        seed_text=(
            f"{plan_sha}:{args.shard_id}:{args.num_shards}:{args.global_sample_budget}"
        ),
    )
    buffers: dict[str, list[np.ndarray]] = {}
    valid_buffers: dict[str, list[np.ndarray]] = {}
    dimensions: dict[str, int] = {}
    sampled_rows = 0
    for episode in selected:
        positions = positions_by_episode.get(episode.episode_id)
        if positions is None:
            continue
        root = resolve_real_directory(
            Path(episode.raw_root),
            f"{episode.episode_id} raw root",
        )
        path = resolve_regular_file(root, episode.data_relative_path)
        columns = sorted(
            {episode.episode_column}
            | {spec.column for spec in episode.action_columns}
            | {spec.column for spec in episode.auxiliary_columns}
        )
        table = pq.read_table(path, columns=columns).slice(
            episode.data_row_start,
            episode.data_row_stop - episode.data_row_start,
        )
        episode_values = np.asarray(table[episode.episode_column].to_numpy())
        if episode_values.size and not np.all(episode_values == episode.episode_index):
            raise ValueError(f"parquet episode interval drift for {episode.episode_id}")
        if int(positions[-1]) >= table.num_rows:
            raise ValueError(
                f"episode row-count drift for {episode.episode_id}: "
                f"sample={int(positions[-1])} rows={table.num_rows}"
            )
        sampled_rows += int(positions.size)
        payload = table.to_pydict()
        for spec in episode.action_columns:
            key = f"{episode.embodiment}::{spec.group_name}"
            matrix, valid = _column_matrix(
                payload[spec.column],
                spec.indices,
                allow_missing=False,
            )
            sample = matrix[positions]
            sample_valid = valid[positions]
            dimensions.setdefault(key, sample.shape[1])
            if dimensions[key] != sample.shape[1]:
                raise ValueError(f"action dimension drift for {key}")
            buffers.setdefault(key, []).append(sample)
            valid_buffers.setdefault(key, []).append(sample_valid)
        for spec in episode.auxiliary_columns:
            key = f"{episode.embodiment}::aux::{spec.modality_name}"
            matrix, valid = _column_matrix(
                payload[spec.column],
                spec.indices,
                allow_missing=True,
            )
            sample = matrix[positions]
            sample_valid = valid[positions]
            dimensions.setdefault(key, sample.shape[1])
            if dimensions[key] != sample.shape[1]:
                raise ValueError(f"auxiliary dimension drift for {key}")
            buffers.setdefault(key, []).append(sample)
            valid_buffers.setdefault(key, []).append(sample_valid)
    values_by_key = {
        key: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for key, parts in buffers.items()
    }
    validity_by_key = {
        key: np.concatenate(parts, axis=0).astype(bool, copy=False)
        for key, parts in valid_buffers.items()
    }
    arrays = {
        **values_by_key,
        **{f"{key}__valid": validity_by_key[key] for key in sorted(values_by_key)},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    metadata = {
        "schema": PARTIAL_SCHEMA,
        "episode_plan_sha256": plan_sha,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "episodes": len(selected),
        "sampled_episodes": len(positions_by_episode),
        "global_sample_budget": args.global_sample_budget,
        "shard_sample_budget": shard_sample_budget,
        "sampled_rows": sampled_rows,
        "keys": sorted(values_by_key),
    }
    np.savez_compressed(
        temporary,
        **arrays,
        __metadata__=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    actual_temporary = (
        temporary
        if temporary.suffix == ".npz"
        else temporary.with_name(temporary.name + ".npz")
    )
    descriptor = os.open(actual_temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(actual_temporary, output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(
        json.dumps({**metadata, "output_sha256": sha256_file(output)}, sort_keys=True)
    )


def command_merge(args: argparse.Namespace) -> None:
    partials = sorted(
        resolve_regular_file(path.parent, path.name) for path in args.partials
    )
    metadata: list[dict[str, Any]] = []
    buffers: dict[str, list[np.ndarray]] = {}
    valid_buffers: dict[str, list[np.ndarray]] = {}
    for path in partials:
        with np.load(path, allow_pickle=False) as payload:
            meta = json.loads(str(payload["__metadata__"]))
            if meta.get("schema") != PARTIAL_SCHEMA:
                raise ValueError(f"partial schema mismatch: {path}")
            metadata.append(meta)
            expected_arrays = set(meta["keys"]) | {
                f"{key}__valid" for key in meta["keys"]
            }
            actual_arrays = set(payload.files).difference({"__metadata__"})
            if actual_arrays != expected_arrays:
                raise ValueError(
                    f"partial array set mismatch for {path}: "
                    f"missing={sorted(expected_arrays - actual_arrays)} "
                    f"extra={sorted(actual_arrays - expected_arrays)}"
                )
            for key in meta["keys"]:
                values = np.asarray(payload[key], dtype=np.float32)
                valid = np.asarray(payload[f"{key}__valid"], dtype=bool)
                if values.shape != valid.shape or values.ndim != 2:
                    raise ValueError(f"partial value/mask shape mismatch for {key}")
                buffers.setdefault(key, []).append(values)
                valid_buffers.setdefault(key, []).append(valid)
    plan_hashes = {item["episode_plan_sha256"] for item in metadata}
    num_shards = {int(item["num_shards"]) for item in metadata}
    global_budgets = {int(item["global_sample_budget"]) for item in metadata}
    shard_ids = {int(item["shard_id"]) for item in metadata}
    if len(plan_hashes) != 1 or len(num_shards) != 1 or len(global_budgets) != 1:
        raise ValueError("action-statistics partial lineage mismatch")
    expected_shards = next(iter(num_shards))
    if (
        len(metadata) != expected_shards
        or len(shard_ids) != len(metadata)
        or shard_ids != set(range(expected_shards))
    ):
        raise ValueError(
            f"action-statistics partials are incomplete: {sorted(shard_ids)}"
        )
    global_budget = next(iter(global_budgets))
    maximum_total = math.ceil(global_budget / expected_shards) * expected_shards
    sampled_total = sum(int(item["sampled_rows"]) for item in metadata)
    if sampled_total > maximum_total:
        raise ValueError(
            "action-statistics partials exceed the global sample budget: "
            f"{sampled_total} > {maximum_total}"
        )
    groups: dict[str, Any] = {}
    for key, parts in sorted(buffers.items()):
        values = np.concatenate(parts, axis=0)
        valid = np.concatenate(valid_buffers[key], axis=0)
        normalization = robust_action_normalization(values, valid)
        q01 = []
        q99 = []
        valid_count = []
        for dimension in range(values.shape[1]):
            selected = values[valid[:, dimension], dimension]
            if selected.size < 32:
                raise ValueError(
                    f"{key} dimension {dimension} has only "
                    f"{selected.size} valid samples"
                )
            low, high = np.quantile(selected, [0.01, 0.99])
            q01.append(float(low))
            q99.append(float(high))
            valid_count.append(int(selected.size))
        groups[key] = {
            "count": int(values.shape[0]),
            "valid_count": valid_count,
            "dimension": int(values.shape[1]),
            "center": normalization.center.tolist(),
            "scale": normalization.scale.tolist(),
            "clip": float(args.clip),
            "q01": q01,
            "q99": q99,
        }
    value = {
        "schema": STATS_SCHEMA,
        "episode_plan_sha256": next(iter(plan_hashes)),
        "global_sample_budget": global_budget,
        "sampled_rows": sampled_total,
        "partial_sha256": {path.name: sha256_file(path) for path in partials},
        "groups": groups,
    }
    atomic_write_json(args.output.resolve(), value, exclusive=True)
    print(
        json.dumps(
            {
                "pass": True,
                "groups": sorted(groups),
                "output_sha256": sha256_file(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.command == "partial":
        command_partial(args)
    else:
        command_merge(args)


if __name__ == "__main__":
    main()
