#!/usr/bin/env python3
"""从 V7 formal 全量 manifest 中精确剔除被完整 MG 替换的旧子集。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from wm3d_v3.data.scale5b_contracts import (
    atomic_write_json,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    utc_now,
)
from wm3d_v3.data.scale5b_sources import EpisodeDescriptor


RECEIPT_SCHEMA = "wm3d_v7_native5b_legacy_residual_receipt_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-provenance",
        action="append",
        required=True,
        help="精确 provenance_dataset 值；可重复提供。",
    )
    return parser.parse_args()


def _read(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON object")
            descriptor = EpisodeDescriptor.from_mapping(value)
            if descriptor.provenance_dataset is None:
                raise ValueError(
                    f"{path}:{line_number} 缺 provenance_dataset，禁止猜来源"
                )
            values.append(descriptor.as_dict())
    if not values:
        raise ValueError("输入 legacy manifest 为空")
    return values


def _publish(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists():
        raise FileExistsError(f"输出或临时文件已存在: {path}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def main() -> None:
    args = parse_args()
    input_path = resolve_regular_file(args.input.parent, args.input.name)
    output_parent = resolve_real_directory(args.output.parent, "legacy output parent")
    output_path = output_parent / args.output.name
    exclusions = tuple(str(value) for value in args.exclude_provenance)
    if any(not value for value in exclusions) or len(exclusions) != len(
        set(exclusions)
    ):
        raise ValueError("--exclude-provenance 必须非空且唯一")

    rows = _read(input_path)
    seen_ids: set[str] = set()
    kept = []
    excluded: dict[str, dict[str, float | int]] = {
        name: {"episodes": 0, "hours": 0.0} for name in exclusions
    }
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id in seen_ids:
            raise ValueError(f"legacy manifest episode_id 重复: {episode_id}")
        seen_ids.add(episode_id)
        provenance = str(row["provenance_dataset"])
        if provenance in excluded:
            excluded[provenance]["episodes"] = int(excluded[provenance]["episodes"]) + 1
            excluded[provenance]["hours"] = (
                float(excluded[provenance]["hours"])
                + float(row["duration_seconds"]) / 3600.0
            )
        else:
            kept.append(row)
    missing = [name for name, value in excluded.items() if int(value["episodes"]) == 0]
    if missing:
        raise ValueError(f"要求剔除的 provenance 在输入中不存在: {missing}")
    if not kept:
        raise ValueError("剔除后 legacy residual 为空")

    _publish(output_path, kept)
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "complete": True,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "input_episodes": len(rows),
        "kept_episodes": len(kept),
        "kept_hours": sum(float(row["duration_seconds"]) for row in kept) / 3600.0,
        "excluded": excluded,
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(receipt_path, receipt, exclusive=True)
    print(json.dumps({"pass": True, **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
