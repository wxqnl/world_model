#!/usr/bin/env python3
"""Read-only schema inventory used before mapping a new LeRobot source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wm3d_v3.data.scale5b_contracts import (
    resolve_real_directory,
    resolve_regular_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-data-files", type=int, default=4)
    return parser.parse_args()


def _type_description(value: pa.DataType) -> dict[str, Any]:
    result: dict[str, Any] = {"arrow_type": str(value)}
    if pa.types.is_fixed_size_list(value) or pa.types.is_list(value):
        result["value_type"] = str(value.value_type)
        if pa.types.is_fixed_size_list(value):
            result["list_size"] = int(value.list_size)
    return result


def main() -> None:
    args = parse_args()
    if args.max_data_files <= 0:
        raise ValueError("max-data-files must be positive")
    root = resolve_real_directory(args.root, "LeRobot source root")
    info_path = resolve_regular_file(root, "meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    files = [
        resolve_regular_file(root, path.relative_to(root).as_posix())
        for path in sorted((root / "data").glob("chunk-*/*.parquet"))[
            : args.max_data_files
        ]
    ]
    if not files:
        raise ValueError(f"no data parquet under {root}")
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
    video_files = [
        resolve_regular_file(root, path.relative_to(root).as_posix())
        for path in sorted((root / "videos").glob("**/*.mp4"))
    ]
    report = {
        "pass": True,
        "root": str(root),
        "fps": info.get("fps"),
        "data_path_template": info.get("data_path"),
        "video_path_template": info.get("video_path"),
        "declared_features": info.get("features", {}),
        "sample_data_schemas": schemas,
        "video_prefix_samples": [
            path.relative_to(root).as_posix() for path in video_files[:32]
        ],
        "video_file_count": len(video_files),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
