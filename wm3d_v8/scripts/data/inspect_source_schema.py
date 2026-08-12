#!/usr/bin/env python3
"""Read-only schema audit; deliberately does not guess an action adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wm3d_v3.data.manifest_contract import canonical_sha256, sha256_file


SCHEMA = "wm3d_v8_raw_schema_audit_v1"


def _regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return path.resolve(strict=True)


def _real_root(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"raw root must be a real directory: {path}")
    return path.resolve(strict=True)


def _roots(root: Path, collection: bool, maximum: int) -> tuple[list[Path], int]:
    if (root / "meta/info.json").is_file() and not collection:
        return [root], 1
    candidates = sorted(
        {
            _real_root(path.parent.parent)
            for path in root.glob("**/meta/info.json")
            if path.is_file() and not path.is_symlink()
        }
    )
    if not candidates:
        legacy = root / "meta_data/info.json"
        if legacy.is_file() and not legacy.is_symlink():
            raise RuntimeError(
                "legacy LeRobot v1/v2 layout detected (meta_data/); convert it with a "
                "version-pinned upstream converter before V8 inventory"
            )
        raise ValueError(f"no meta/info.json found under {root}")
    for candidate in candidates:
        candidate.relative_to(root)
    if len(candidates) > maximum:
        raise RuntimeError(
            f"collection contains {len(candidates)} LeRobot roots, above --max-roots "
            f"{maximum}; raise the limit so every root is audited"
        )
    return candidates, len(candidates)


def _type(value: pa.DataType) -> dict[str, Any]:
    result: dict[str, Any] = {"arrow_type": str(value)}
    if pa.types.is_fixed_size_list(value) or pa.types.is_list(value):
        result["value_type"] = str(value.value_type)
        if pa.types.is_fixed_size_list(value):
            result["list_size"] = int(value.list_size)
    return result


def _observed_list_widths(
    parquet: pq.ParquetFile,
    *,
    field_name: str,
    sample_rows: int = 32,
) -> dict[str, Any]:
    """Record payload evidence for variable-size Arrow list columns.

    Recent LeRobot releases often store a logically fixed ``[D]`` tensor as
    Arrow ``list<float>`` rather than ``fixed_size_list<float, D>``.  Treating
    that schema as dimensionless makes a valid audited adapter impossible;
    trusting only ``info.json`` would not inspect the payload.  Read a bounded
    prefix from the first row group and record every observed width.  Full
    episode inventory later checks every selected payload row again.
    """

    if parquet.metadata.num_row_groups <= 0:
        raise RuntimeError(f"Parquet payload has no row groups: {parquet}")
    column = parquet.read_row_group(0, columns=[field_name]).column(0)
    count = min(int(sample_rows), len(column))
    values = column.slice(0, count).to_pylist()
    widths: set[int] = set()
    null_rows = 0
    for value in values:
        if value is None:
            null_rows += 1
        elif isinstance(value, list):
            widths.add(len(value))
        else:
            raise RuntimeError(
                f"expected Arrow list payload for {field_name!r}, got {type(value)}"
            )
    return {
        "observed_list_widths": sorted(widths),
        "observed_list_rows": count,
        "observed_list_null_rows": null_rows,
    }


def _one(root: Path, collection_root: Path, max_data_files: int, max_videos: int) -> dict[str, Any]:
    info_path = _regular(root / "meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    candidates = sorted((root / "data").glob("**/*.parquet"))
    if not candidates:
        raise ValueError(f"no Parquet payload under {root}")
    samples = []
    for path in candidates[:max_data_files]:
        safe = _regular(path)
        parquet = pq.ParquetFile(safe)
        schema = parquet.schema_arrow
        columns: dict[str, dict[str, Any]] = {}
        for field in schema:
            evidence = _type(field.type)
            if pa.types.is_list(field.type):
                evidence.update(
                    _observed_list_widths(parquet, field_name=field.name)
                )
            columns[field.name] = evidence
        samples.append(
            {
                "relative_path": safe.relative_to(root).as_posix(),
                "size_bytes": safe.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "columns": columns,
            }
        )
    videos = []
    for path in (root / "videos").glob("**/*.mp4"):
        if len(videos) >= max_videos:
            break
        safe = _regular(path)
        videos.append(
            {
                "relative_path": safe.relative_to(root).as_posix(),
                "size_bytes": safe.stat().st_size,
            }
        )
    signature = {
        "info_schema_version": info.get("codebase_version"),
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
        "features": info.get("features"),
        "sample_columns": [row["columns"] for row in samples],
    }
    return {
        "relative_root": root.relative_to(collection_root).as_posix() or ".",
        "info_path_sha256": sha256_file(info_path),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "declared_nominal_fps_for_audit_only": info.get("fps"),
        "declared_features": info.get("features"),
        "data_path_template": info.get("data_path"),
        "video_path_template": info.get("video_path"),
        "sample_data": samples,
        "sample_videos": videos,
        "schema_signature_sha256": canonical_sha256(signature),
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical schema audit: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--collection", action="store_true")
    parser.add_argument("--max-roots", type=int, default=32)
    parser.add_argument("--max-data-files", type=int, default=2)
    parser.add_argument("--max-video-files", type=int, default=8)
    parser.add_argument("--require-homogeneous", action="store_true")
    parser.add_argument(
        "--upstream-receipt",
        type=Path,
        required=True,
        help="下载或 collection receipt；审计报告会绑定其字节 SHA。",
    )
    parser.add_argument(
        "--candidate-output",
        type=Path,
        required=True,
        help="只生成字段候选，不生成或猜测正式 adapter。",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_roots, args.max_data_files, args.max_video_files) <= 0:
        raise ValueError("all sample limits must be positive")
    root = _real_root(args.root)
    roots, total = _roots(root, args.collection, args.max_roots)
    rows = [_one(item, root, args.max_data_files, args.max_video_files) for item in roots]
    signatures = sorted({row["schema_signature_sha256"] for row in rows})
    homogeneous = len(signatures) == 1 and len(rows) == total
    upstream_receipt = _regular(args.upstream_receipt)
    field_candidates: dict[str, list[dict[str, Any]]] = {}
    view_candidates: set[str] = set()
    for row in rows:
        declared = row.get("declared_features")
        if isinstance(declared, dict):
            for key, value in declared.items():
                if isinstance(value, dict) and str(value.get("dtype", "")).lower() in {
                    "video",
                    "image",
                }:
                    view_candidates.add(str(key))
        for sample in row["sample_data"]:
            for key, value in sample["columns"].items():
                bucket = field_candidates.setdefault(str(key), [])
                if value not in bucket:
                    bucket.append(value)
    candidate = {
        "schema": "wm3d_v8_source_adapter_candidate_v1",
        "raw_root": str(root),
        "upstream_receipt_path": str(upstream_receipt),
        "upstream_receipt_sha256": sha256_file(upstream_receipt),
        "fields": dict(sorted(field_candidates.items())),
        "possible_rgb_features": sorted(view_candidates),
        "operator_must_supply": [
            "canonical view-slot mapping",
            "action/state fields and exact columns",
            "units and affine conversion",
            "coordinate frames and composition operators",
            "gripper polarity/absolute-vs-delta semantics",
            "source-native observation/action/state clock fields",
            "fine_command or coarse_effect supervision for every group",
        ],
        "formal_adapter": False,
    }
    _publish(args.candidate_output, candidate)
    report = {
        "schema": SCHEMA,
        "raw_root": str(root),
        "roots_total": total,
        "roots_inspected": len(rows),
        "all_roots_inspected": len(rows) == total,
        "homogeneous": homogeneous,
        "schema_signatures": signatures,
        "upstream_receipt_path": str(upstream_receipt),
        "upstream_receipt_sha256": sha256_file(upstream_receipt),
        "adapter_candidate_generated": True,
        "adapter_candidate_path": str(args.candidate_output.absolute()),
        "adapter_candidate_sha256": sha256_file(args.candidate_output.absolute()),
        "formal_inventory_ready": False,
        "next_required_action": (
            "audit view/action/state/time columns, units, frames, gripper polarity, "
            "group layout and supervision; then write a strict source adapter YAML"
        ),
        "roots": rows,
    }
    _publish(args.output, report)
    print(json.dumps({key: report[key] for key in report if key != "roots"}, sort_keys=True))
    if args.require_homogeneous and not homogeneous:
        raise SystemExit("collection schemas differ or the collection was only partially inspected")


if __name__ == "__main__":
    main()
