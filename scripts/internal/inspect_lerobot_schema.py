#!/usr/bin/env python3
"""只读审计一个 LeRobot root 或由多个独立 root 组成的 collection。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wm3d.data.contracts import (
    canonical_sha256,
    resolve_real_directory,
    resolve_regular_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--collection", action="store_true")
    parser.add_argument("--max-roots", type=int, default=32)
    parser.add_argument("--max-data-files", type=int, default=2)
    parser.add_argument("--max-video-samples", type=int, default=16)
    parser.add_argument("--require-homogeneous", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _type_description(value: pa.DataType) -> dict[str, Any]:
    result: dict[str, Any] = {"arrow_type": str(value)}
    if pa.types.is_fixed_size_list(value) or pa.types.is_list(value):
        result["value_type"] = str(value.value_type)
        if pa.types.is_fixed_size_list(value):
            result["list_size"] = int(value.list_size)
    return result


def _roots(root: Path, collection: bool, maximum: int) -> tuple[list[Path], int]:
    direct = root / "meta" / "info.json"
    if direct.is_file() and not direct.is_symlink() and not collection:
        return [root], 1
    candidates = []
    for info_path in sorted(root.glob("**/meta/info.json")):
        if info_path.is_symlink() or not info_path.is_file():
            raise ValueError(f"LeRobot metadata 不是普通文件: {info_path}")
        nested = resolve_real_directory(info_path.parent.parent, "nested LeRobot root")
        try:
            nested.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"LeRobot root 逃逸 collection: {nested}") from exc
        candidates.append(nested)
    if not candidates:
        raise ValueError(f"找不到 meta/info.json: {root}")
    if len(candidates) != len(set(candidates)):
        raise ValueError("collection 中出现重复 LeRobot root")
    return candidates[:maximum], len(candidates)


def _inspect_one(
    root: Path,
    collection_root: Path,
    *,
    max_data_files: int,
    max_video_samples: int,
) -> dict[str, Any]:
    info_path = resolve_regular_file(root, "meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data_candidates = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_candidates:
        data_candidates = sorted((root / "data").glob("**/*.parquet"))
    files = [
        resolve_regular_file(root, path.relative_to(root).as_posix())
        for path in data_candidates[:max_data_files]
    ]
    if not files:
        raise ValueError(f"没有 data parquet: {root}")
    schemas = []
    for path in files:
        schema = pq.read_schema(path)
        schemas.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "columns": {
                    field.name: _type_description(field.type) for field in schema
                },
            }
        )
    video_samples = []
    for path in (root / "videos").glob("**/*.mp4"):
        if len(video_samples) >= max_video_samples:
            break
        safe = resolve_regular_file(root, path.relative_to(root).as_posix())
        video_samples.append(safe.relative_to(root).as_posix())
    signature = {
        "fps": info.get("fps"),
        "data_path": info.get("data_path"),
        "video_path": info.get("video_path"),
        "features": info.get("features", {}),
        "sample_columns": [item["columns"] for item in schemas],
    }
    return {
        "relative_root": root.relative_to(collection_root).as_posix(),
        "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "data_path_template": info.get("data_path"),
        "video_path_template": info.get("video_path"),
        "declared_features": info.get("features", {}),
        "sample_data_schemas": schemas,
        "video_prefix_samples": video_samples,
        "schema_sha256": canonical_sha256(signature),
    }


def main() -> None:
    args = parse_args()
    for name in ("max_roots", "max_data_files", "max_video_samples"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须为正数")
    root = resolve_real_directory(args.root, "LeRobot source/collection root")
    selected, total = _roots(root, args.collection, args.max_roots)
    reports = [
        _inspect_one(
            nested,
            root,
            max_data_files=args.max_data_files,
            max_video_samples=args.max_video_samples,
        )
        for nested in selected
    ]
    schema_hashes = sorted({item["schema_sha256"] for item in reports})
    homogeneous = len(schema_hashes) == 1 and len(reports) == total
    report = {
        "pass": not args.require_homogeneous or homogeneous,
        "root": str(root),
        "collection": total > 1 or args.collection,
        "roots_total": total,
        "roots_inspected": len(reports),
        "all_roots_inspected": len(reports) == total,
        "homogeneous": homogeneous,
        "schema_sha256": schema_hashes,
        "roots": reports,
    }
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    if not report["pass"]:
        raise SystemExit(
            "collection schema 非同构；必须拆分 embodiment/layout 后再 cache"
        )


if __name__ == "__main__":
    main()
