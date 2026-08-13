#!/usr/bin/env python3
"""Create the deterministic task-id input for the pinned Beta converter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import uuid

from wm3d.data.manifest_contract import sha256_file


DOWNLOAD_SCHEMA = "wm3d_v8_raw_snapshot_receipt_v1"
TASK = re.compile(r"task_([0-9]+)\.json")


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_root.is_symlink() or not args.raw_root.is_dir():
        raise RuntimeError("raw root must be a real directory")
    root = args.raw_root.resolve(strict=True)
    receipt_path = _regular(args.download_receipt, "download receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != DOWNLOAD_SCHEMA
        or receipt.get("source") != "agibot_beta"
        or Path(str(receipt.get("snapshot_path", ""))).resolve(strict=True) != root
    ):
        raise RuntimeError("download receipt does not bind the Beta raw root")
    task_root = root / "task_info"
    if task_root.is_symlink() or not task_root.is_dir():
        raise RuntimeError("Beta raw root has no safe task_info directory")
    tasks: list[int] = []
    episodes: set[int] = set()
    task_files: dict[str, str] = {}
    for candidate in sorted(task_root.glob("task_*.json")):
        path = _regular(candidate, "Beta task-info file")
        match = TASK.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"invalid Beta task-info filename: {path.name}")
        task_id = int(match.group(1))
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"Beta task-info must be a non-empty list: {path}")
        for row in rows:
            if not isinstance(row, dict) or int(row.get("task_id", -1)) != task_id:
                raise RuntimeError(f"Beta task identity mismatch: {path}")
            episode = int(row.get("episode_id", -1))
            if episode < 0 or episode in episodes:
                raise RuntimeError(f"Beta episode is invalid/duplicated: {path}")
            episodes.add(episode)
        tasks.append(task_id)
        task_files[path.relative_to(root).as_posix()] = sha256_file(path)
    if not tasks or len(tasks) != len(set(tasks)):
        raise RuntimeError("Beta task set is empty or duplicated")
    payload = "".join(f"{value}\n" for value in tasks).encode()
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    except FileExistsError:
        if output.read_bytes() != payload:
            raise
    finally:
        temporary.unlink(missing_ok=True)
    value = {
        "schema": "wm3d_v8_agibot_beta_task_list_receipt_v1",
        "download_receipt_path": str(receipt_path),
        "download_receipt_sha256": sha256_file(receipt_path),
        "task_count": len(tasks),
        "episode_count": len(episodes),
        "task_info_sha256_by_path": task_files,
        "task_list_path": str(output),
        "task_list_sha256": sha256_file(output),
    }
    report = args.receipt.absolute()
    report_payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if report.exists():
        if report.read_bytes() != report_payload:
            raise FileExistsError(f"non-identical task-list receipt exists: {report}")
    else:
        report.write_bytes(report_payload)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
