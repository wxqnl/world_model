#!/usr/bin/env python3
"""从冻结的 AgiBot Beta 快照生成确定性 task-id 列表。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from wm3d.data.contracts import (
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
)


TASK_RE = re.compile(r"task_([0-9]+)\.json")
DOWNLOAD_RECEIPT_SCHEMA = "wm3d_v7_raw_download_receipt_v1"
EXPECTED_SOURCE = "agibot_beta_snapshot"
EXPECTED_REPO_ID = "agibot-world/AgiBotWorld-Beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_real_directory(args.raw_root, "AgiBot Beta raw root")
    download_receipt = resolve_regular_file(
        root,
        ".wm3d_v7_download_receipt.json",
    )
    download = json.loads(download_receipt.read_text(encoding="utf-8"))
    revision = str(download.get("revision", ""))
    if (
        not isinstance(download, dict)
        or download.get("schema") != DOWNLOAD_RECEIPT_SCHEMA
        or download.get("complete") is not True
        or download.get("source") != EXPECTED_SOURCE
        or download.get("repo_id") != EXPECTED_REPO_ID
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or download.get("resolved_revision") != revision
        or download.get("target") != str(root)
    ):
        raise ValueError("AgiBot Beta download receipt 未完成或身份不匹配")
    task_info = resolve_real_directory(root / "task_info", "AgiBot Beta task_info")
    for directory_name in ("observations", "parameters", "proprio_stats"):
        resolve_real_directory(
            root / directory_name,
            f"AgiBot Beta {directory_name}",
        )
    ids = []
    episode_ids: set[int] = set()
    for candidate in sorted(task_info.glob("task_*.json")):
        path = resolve_regular_file(root, candidate.relative_to(root).as_posix())
        match = TASK_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"无法解析 task id: {path}")
        task_id = int(match.group(1))
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not value:
            raise ValueError(f"task_info 必须是非空 episode 列表: {path}")
        for row in value:
            if not isinstance(row, dict) or int(row.get("task_id", -1)) != task_id:
                raise ValueError(f"task_info task_id 与文件名不一致: {path}")
            episode_id = int(row.get("episode_id", -1))
            if episode_id < 0 or episode_id in episode_ids:
                raise ValueError(f"task_info episode_id 非法或跨 task 重复: {path}")
            episode_ids.add(episode_id)
        ids.append(task_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("AgiBot Beta task 列表为空或重复")
    output_parent = resolve_real_directory(
        args.output.parent, "task-list output parent"
    )
    output = output_parent / args.output.name
    with output.open("x", encoding="utf-8") as handle:
        handle.write("".join(f"{task_id}\n" for task_id in ids))
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "pass": True,
                "tasks": len(ids),
                "episodes": len(episode_ids),
                "output": str(output),
                "download_receipt_sha256": sha256_file(download_receipt),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
