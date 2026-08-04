#!/usr/bin/env python3
"""Merge committed WM3D encoder parts and publish the dataset seal."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import uuid
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wm3d.data.contracts import (
    DATASET_SCHEMA,
    DatasetSeal,
    atomic_write_json,
    canonical_sha256,
    evidence_for,
    load_contract,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    utc_now,
)


PART_SCHEMA = "wm3d_v7_encoded_part_v2"
PART_COMMIT_SCHEMA = "wm3d_v7_encoded_part_commit_v2"
WORKER_RECEIPT_SCHEMA = "wm3d_v7_encode_worker_receipt_v1"
PART_PAYLOAD_FILES = {
    "features.safetensors",
    "actions.safetensors",
    "rgb.jpgpack",
    "windows.parquet",
}
PART_NAME_RE = re.compile(r"^part-([0-9]{5})-([0-9]{6})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--num-encoder-shards", type=int, required=True)
    parser.add_argument("--index-rows-per-file", type=int, default=1_000_000)
    return parser.parse_args()


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_part(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    commit_path = path / "COMMITTED.json"
    if (
        path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not commit_path.is_file()
        or commit_path.is_symlink()
    ):
        raise ValueError(f"part is not safely committed: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != PART_SCHEMA
        or commit.get("schema") != PART_COMMIT_SCHEMA
    ):
        raise ValueError(f"part schema mismatch: {path}")
    if manifest.get("part_name") != path.name or commit.get("part_name") != path.name:
        raise ValueError(f"part identity mismatch: {path}")
    match = PART_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"part name is not canonical: {path}")
    expected_shard_id = int(match.group(1))
    expected_part_index = int(match.group(2))
    if (
        int(manifest.get("worker_shard_id", -1)) != expected_shard_id
        or int(manifest.get("part_index", -1)) != expected_part_index
        or int(manifest.get("worker_num_shards", -1)) <= expected_shard_id
    ):
        raise ValueError(f"part worker/index identity mismatch: {path}")
    if sha256_file(manifest_path) != commit.get("manifest_sha256"):
        raise ValueError(f"part manifest digest mismatch: {path}")
    if canonical_sha256(manifest) != commit.get("manifest_content_sha256"):
        raise ValueError(f"part canonical manifest digest mismatch: {path}")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path}: {key} lineage mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != PART_PAYLOAD_FILES:
        raise ValueError(f"{path}: payload file contract mismatch")
    expected_files = PART_PAYLOAD_FILES | {
        "manifest.json",
        "COMMITTED.json",
    }
    actual_entries = {item.name for item in path.iterdir()}
    if actual_entries != expected_files:
        raise ValueError(
            f"{path}: file set mismatch "
            f"missing={sorted(expected_files - actual_entries)} "
            f"extra={sorted(actual_entries - expected_files)}"
        )
    for name, evidence in files.items():
        file_path = resolve_regular_file(path, name)
        if (
            file_path.stat().st_size != int(evidence["size"])
            or sha256_file(file_path) != evidence["sha256"]
        ):
            raise ValueError(f"{path}: payload digest mismatch for {name}")
    return manifest


class _RotatingParquetWriters:
    def __init__(self, root: Path, maximum_rows: int) -> None:
        self.root = root
        self.maximum_rows = int(maximum_rows)
        self.state: dict[tuple[str, str], dict[str, Any]] = {}
        self.files: list[str] = []

    def write(self, split: str, source: str, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        key = (split, source)
        state = self.state.get(key)
        if state is None or (
            state["rows"] > 0 and state["rows"] + table.num_rows > self.maximum_rows
        ):
            if state is not None:
                state["writer"].close()
                _fsync(state["path"])
            file_index = 0 if state is None else state["file_index"] + 1
            directory = self.root / split / source
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{file_index:06d}.parquet"
            writer = pq.ParquetWriter(
                path,
                table.schema,
                compression="zstd",
                write_statistics=True,
            )
            state = {
                "writer": writer,
                "path": path,
                "rows": 0,
                "file_index": file_index,
            }
            self.state[key] = state
            self.files.append(path.relative_to(self.root.parent).as_posix())
        state["writer"].write_table(table, row_group_size=8192)
        state["rows"] += table.num_rows

    def close(self) -> None:
        for state in self.state.values():
            state["writer"].close()
            _fsync(state["path"])


def _load_worker_receipts(
    root: Path,
    count: int,
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    receipts = []
    part_summaries: dict[str, dict[str, int]] = {}
    for shard_id in range(count):
        path = root / "receipts" / "encode_workers" / f"worker_{shard_id:05d}.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing encoder worker receipt {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema") != WORKER_RECEIPT_SCHEMA
            or int(value.get("shard_id", -1)) != shard_id
            or int(value.get("num_shards", -1)) != count
        ):
            raise ValueError(f"encoder worker receipt identity mismatch: {path}")
        parts = value.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"encoder worker receipt has no parts: {path}")
        shard_part_indices: list[int] = []
        for part in parts:
            if not isinstance(part, dict):
                raise ValueError(f"encoder worker part summary is invalid: {path}")
            part_name = str(part.get("part", ""))
            match = PART_NAME_RE.fullmatch(part_name)
            if (
                match is None
                or int(match.group(1)) != shard_id
                or int(part.get("frames", 0)) <= 0
                or int(part.get("windows", 0)) <= 0
            ):
                raise ValueError(f"encoder worker part summary is invalid: {path}")
            part_index = int(match.group(2))
            shard_part_indices.append(part_index)
            if part_name in part_summaries:
                raise ValueError(
                    f"duplicate encoded part identity across receipts: {part_name}"
                )
            part_summaries[part_name] = {
                "frames": int(part["frames"]),
                "windows": int(part["windows"]),
            }
        if sorted(shard_part_indices) != list(range(len(parts))):
            raise ValueError(
                f"encoder worker part indexes are not contiguous from zero: {path}"
            )
        receipts.append(value)
    lineage_keys = (
        "episode_plan_sha256",
        "dataset_contract_sha256",
        "action_stats_sha256",
        "task_index_sha256",
        "encoder_asset_receipt_sha256",
        "vggt_model",
        "vggt_revision",
    )
    lineage = {}
    for key in lineage_keys:
        values = {receipt[key] for receipt in receipts}
        if len(values) != 1:
            raise ValueError(f"encoder worker lineage drift for {key}")
        lineage[key] = next(iter(values))
    lineage["worker_num_shards"] = count
    return lineage, part_summaries


def _committed_part_names(root: Path) -> set[str]:
    parts_root = resolve_real_directory(
        root / "payload" / "parts",
        "encoded payload parts root",
    )
    result: set[str] = set()
    for path in parts_root.iterdir():
        if path.name.startswith("."):
            continue
        info = os.lstat(path)
        if (
            PART_NAME_RE.fullmatch(path.name) is None
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
        ):
            raise ValueError(f"unexpected non-committed payload entry: {path}")
        result.add(path.name)
    return result


def main() -> None:
    args = parse_args()
    if args.num_encoder_shards <= 0 or args.index_rows_per_file <= 0:
        raise ValueError("invalid merge arguments")
    root = resolve_real_directory(args.dataset_root, "dataset root")
    contract_path = resolve_regular_file(root, "control/dataset_contract.json")
    contract = load_contract(contract_path)
    lineage, expected_part_summaries = _load_worker_receipts(
        root,
        args.num_encoder_shards,
    )
    expected_parts = set(expected_part_summaries)
    if contract.sha256 != lineage["dataset_contract_sha256"]:
        raise ValueError("encoder/dataset contract digest mismatch")
    direct_evidence = {
        "episode_plan_sha256": root / "control" / "episode_plan.jsonl",
        "action_stats_sha256": root / "control" / "action_stats.json",
        "task_index_sha256": root / "control" / "task_index.json",
        "encoder_asset_receipt_sha256": root / "control" / "encoder_asset_receipt.json",
    }
    for key, path in direct_evidence.items():
        path = resolve_regular_file(root, path.relative_to(root).as_posix())
        if sha256_file(path) != lineage[key]:
            raise ValueError(f"sealed control input drift for {key}")
    actual_parts = _committed_part_names(root)
    if actual_parts != expected_parts:
        raise ValueError(
            "committed part set mismatch: "
            f"missing={sorted(expected_parts - actual_parts)} "
            f"extra={sorted(actual_parts - expected_parts)}"
        )

    indexes = root / "indexes"
    if indexes.exists():
        raise FileExistsError(f"indexes already exist: {indexes}")
    temporary_indexes = root / f".indexes.incomplete.{uuid.uuid4().hex}"
    temporary_indexes.mkdir(mode=0o750)
    writers = _RotatingParquetWriters(
        temporary_indexes,
        args.index_rows_per_file,
    )
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    database = sqlite3.connect(temporary_indexes / "window_ids.sqlite")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("CREATE TABLE windows (window_id TEXT PRIMARY KEY)")
    payload_manifest_relatives: list[str] = []
    try:
        for part_name in sorted(expected_parts):
            part = root / "payload" / "parts" / part_name
            manifest = _verify_part(part, lineage)
            summary = expected_part_summaries[part_name]
            if (
                int(manifest.get("frames", -1)) != summary["frames"]
                or int(manifest.get("windows", -1)) != summary["windows"]
            ):
                raise ValueError(f"{part_name}: worker receipt/manifest count mismatch")
            payload_manifest_relatives.extend(
                [
                    f"payload/parts/{part_name}/manifest.json",
                    f"payload/parts/{part_name}/COMMITTED.json",
                ]
            )
            table = pq.read_table(part / "windows.parquet")
            if table.num_rows != int(manifest["windows"]):
                raise ValueError(f"{part_name}: window count mismatch")
            required_columns = {
                "window_id",
                "episode_id",
                "source",
                "split",
                "feature_shard",
                "action_shard",
                "rgb_pack",
                "frame_offset",
                "action_offset",
                "frame_count",
                "episode_frame_start",
                "episode_frame_stop",
                "task_id",
                "embodiment_id",
                "action_group_ids",
                "action_group_mask",
            }
            if set(table.column_names) != required_columns:
                raise ValueError(
                    f"{part_name}: window schema mismatch "
                    f"missing={sorted(required_columns - set(table.column_names))} "
                    f"extra={sorted(set(table.column_names) - required_columns)}"
                )
            rows = table.select(
                [
                    "window_id",
                    "source",
                    "split",
                    "feature_shard",
                    "action_shard",
                    "rgb_pack",
                    "frame_offset",
                    "action_offset",
                    "frame_count",
                    "episode_frame_start",
                    "episode_frame_stop",
                ]
            ).to_pylist()
            for row in rows:
                try:
                    database.execute(
                        "INSERT INTO windows(window_id) VALUES (?)",
                        (str(row["window_id"]),),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"duplicate global window id {row['window_id']}"
                    ) from exc
                expected_paths = {
                    "feature_shard": (
                        f"payload/parts/{part_name}/features.safetensors"
                    ),
                    "action_shard": (f"payload/parts/{part_name}/actions.safetensors"),
                    "rgb_pack": f"payload/parts/{part_name}/rgb.jpgpack",
                }
                if any(
                    str(row[name]) != expected
                    for name, expected in expected_paths.items()
                ):
                    raise ValueError(f"{part_name}: window payload path mismatch")
                if (
                    str(row["source"]) not in contract.source_order
                    or str(row["split"]) not in {"train", "val", "test"}
                    or int(row["frame_count"]) != contract.T + contract.K
                    or int(row["frame_offset"]) < 0
                    or int(row["action_offset"]) != int(row["frame_offset"])
                    or int(row["frame_offset"]) + int(row["frame_count"])
                    > int(manifest["frames"])
                    or int(row["episode_frame_start"]) < 0
                    or int(row["episode_frame_stop"]) > int(manifest["frames"])
                    or int(row["episode_frame_start"]) > int(row["frame_offset"])
                    or int(row["episode_frame_stop"])
                    < int(row["frame_offset"]) + int(row["frame_count"])
                ):
                    raise ValueError(f"{part_name}: window bounds are invalid")
            database.commit()
            for split in ("train", "val", "test"):
                split_table = table.filter(pc.equal(table["split"], split))
                for source in contract.source_order:
                    selected = split_table.filter(
                        pc.equal(split_table["source"], source)
                    )
                    if selected.num_rows:
                        selected = selected.drop(["split"])
                        writers.write(split, source, selected)
                        counts[source][split] += selected.num_rows
        writers.close()
    finally:
        database.close()
    for source in contract.source_order:
        if counts[source]["train"] <= 0 or counts[source]["val"] <= 0:
            raise ValueError(f"{source}: sealed index lacks train or val windows")
    (temporary_indexes / "window_ids.sqlite").unlink()
    for suffix in ("-wal", "-shm"):
        sidecar = temporary_indexes / f"window_ids.sqlite{suffix}"
        if sidecar.exists():
            sidecar.unlink()
    _fsync(temporary_indexes)
    os.replace(temporary_indexes, indexes)
    _fsync(root)

    hours: dict[str, float] = defaultdict(float)
    with (root / "control" / "episode_plan.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                hours[str(row["source"])] += float(row["duration_seconds"]) / 3600.0

    control_relatives = [
        "control/dataset_contract.json",
        "control/source_layouts.json",
        "control/episode_plan.jsonl",
        "control/action_stats.json",
        "control/task_index.json",
        "control/task_embeddings.safetensors",
        "control/encoder_asset_receipt.json",
        "receipts/source_scan.json",
    ]
    control_relatives.extend(
        path.relative_to(root).as_posix()
        for path in sorted(indexes.glob("*/*/part-*.parquet"))
    )
    control_relatives.extend(
        f"receipts/encode_workers/worker_{shard_id:05d}.json"
        for shard_id in range(args.num_encoder_shards)
    )
    receipt = DatasetSeal(
        dataset_schema=DATASET_SCHEMA,
        dataset_contract_sha256=contract.sha256,
        control_files=evidence_for(root, control_relatives),
        payload_manifest_files=evidence_for(root, payload_manifest_relatives),
        source_window_counts={
            source: {
                split: int(value)
                for split, value in counts[source].items()
                if int(value) > 0
            }
            for source in contract.source_order
        },
        source_hours={source: float(hours[source]) for source in contract.source_order},
        created_at_utc=utc_now(),
    )
    receipt.validate()
    receipt_path = root / "receipts" / "dataset_seal.json"
    atomic_write_json(receipt_path, receipt.as_dict(), exclusive=True)
    print(
        json.dumps(
            {
                "pass": True,
                "dataset_contract_sha256": contract.sha256,
                "dataset_seal_sha256": receipt.sha256,
                "source_window_counts": receipt.source_window_counts,
                "source_hours": receipt.source_hours,
                "parts": len(expected_parts),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
