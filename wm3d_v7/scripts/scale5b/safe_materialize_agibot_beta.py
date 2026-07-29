#!/usr/bin/env python3
"""安全、可分片地把 AgiBot Beta tar 快照物化为官方 converter 所需目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from wm3d_v3.data.scale5b_contracts import (
    atomic_write_json,
    canonical_sha256,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    utc_now,
)


PLAN_SCHEMA = "wm3d_v7_native5b_agibot_beta_materialization_plan_v1"
PREPARE_SCHEMA = "wm3d_v7_native5b_agibot_beta_prepare_receipt_v1"
ARCHIVE_SCHEMA = "wm3d_v7_native5b_agibot_beta_archive_receipt_v1"
FINAL_SCHEMA = "wm3d_v7_native5b_agibot_beta_materialization_receipt_v1"
CATEGORIES = ("observations", "parameters", "proprio_stats")
TASK_RE = re.compile(r"task_([0-9]+)\.json")
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz")
DOWNLOAD_RECEIPT_SCHEMA = "wm3d_v7_native5b_raw_download_receipt_v1"
EXPECTED_SOURCE = "agibot_beta_snapshot"
EXPECTED_REPO_ID = "agibot-world/AgiBotWorld-Beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "extract", "finalize"))
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def _is_archive(path: Path) -> bool:
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def _task_catalog(
    snapshot_root: Path,
) -> tuple[dict[int, tuple[int, ...]], dict[int, int]]:
    task_info = resolve_real_directory(
        snapshot_root / "task_info",
        "AgiBot Beta task_info",
    )
    tasks: dict[int, tuple[int, ...]] = {}
    episode_to_task: dict[int, int] = {}
    for candidate in sorted(task_info.glob("task_*.json")):
        path = resolve_regular_file(
            snapshot_root,
            candidate.relative_to(snapshot_root).as_posix(),
        )
        match = TASK_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"无法解析 task id: {path}")
        task_id = int(match.group(1))
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not value:
            raise ValueError(f"task_info 必须是非空 episode 列表: {path}")
        episode_ids = []
        for row in value:
            if not isinstance(row, dict):
                raise ValueError(f"task_info episode 不是对象: {path}")
            if int(row.get("task_id", -1)) != task_id:
                raise ValueError(f"task_info task_id 与文件名不一致: {path}")
            episode_id = int(row.get("episode_id", -1))
            if episode_id < 0:
                raise ValueError(f"task_info episode_id 非法: {path}")
            previous = episode_to_task.setdefault(episode_id, task_id)
            if previous != task_id:
                raise ValueError(
                    f"episode {episode_id} 同时属于 task {previous}/{task_id}"
                )
            episode_ids.append(episode_id)
        if len(episode_ids) != len(set(episode_ids)):
            raise ValueError(f"task {task_id} episode_id 重复")
        tasks[task_id] = tuple(sorted(episode_ids))
    if not tasks:
        raise ValueError("AgiBot Beta task_info 为空")
    return tasks, episode_to_task


def _archives(snapshot_root: Path) -> list[dict[str, Any]]:
    values = []
    for category in CATEGORIES:
        root = resolve_real_directory(
            snapshot_root / category,
            f"AgiBot Beta {category}",
        )
        for candidate in sorted(root.rglob("*")):
            if not _is_archive(candidate):
                continue
            path = resolve_regular_file(
                snapshot_root,
                candidate.relative_to(snapshot_root).as_posix(),
            )
            values.append(
                {
                    "relative_path": path.relative_to(snapshot_root).as_posix(),
                    "size": path.stat().st_size,
                }
            )
    if not values:
        raise ValueError("AgiBot Beta 快照没有 observations/parameters/proprio tar")
    return values


def _member_parts(name: str) -> tuple[str, ...]:
    pure = PurePosixPath(name)
    parts = tuple(part for part in pure.parts if part != ".")
    if pure.is_absolute() or not parts or any(part in {"", ".."} for part in parts):
        raise ValueError(f"归档成员路径不安全: {name!r}")
    return parts


def _normalized_member(
    archive_relative: str,
    member_name: str,
    *,
    task_ids: set[int],
    episode_to_task: Mapping[int, int],
) -> Path:
    archive_parts = Path(archive_relative).parts
    category = archive_parts[0]
    if category not in CATEGORIES:
        raise ValueError(f"未知 AgiBot Beta archive category: {archive_relative}")
    parts = _member_parts(member_name)
    if category in parts:
        parts = parts[parts.index(category) :]
    else:
        numeric_index = None
        numeric_value = None
        for index, part in enumerate(parts):
            if not part.isdigit():
                continue
            value = int(part)
            if value in task_ids or value in episode_to_task:
                numeric_index = index
                numeric_value = value
                break
        if numeric_index is None or numeric_value is None:
            raise ValueError(
                f"无法把 archive member 映射到 task/episode: "
                f"{archive_relative}:{member_name}"
            )
        tail = parts[numeric_index:]
        if category == "observations":
            if len(archive_parts) < 3 or not archive_parts[1].isdigit():
                raise ValueError(
                    f"observations archive 缺 task 目录: {archive_relative}"
                )
            archive_task = int(archive_parts[1])
            if numeric_value == archive_task:
                parts = (category, *tail)
            elif episode_to_task.get(numeric_value) == archive_task:
                parts = (category, str(archive_task), *tail)
            else:
                raise ValueError(
                    f"observations member 不属于 archive task: "
                    f"{archive_relative}:{member_name}"
                )
        elif numeric_value in task_ids:
            parts = (category, *tail)
        else:
            task_id = episode_to_task[numeric_value]
            parts = (category, str(task_id), *tail)

    if len(parts) < 3 or parts[0] != category:
        raise ValueError(
            f"归档成员没有 category/task/episode: {archive_relative}:{member_name}"
        )
    if not parts[1].isdigit() or int(parts[1]) not in task_ids:
        raise ValueError(f"归档成员 task 非法: {archive_relative}:{member_name}")
    if not parts[2].isdigit():
        raise ValueError(f"归档成员 episode 非法: {archive_relative}:{member_name}")
    task_id = int(parts[1])
    episode_id = int(parts[2])
    if episode_to_task.get(episode_id) != task_id:
        raise ValueError(
            f"归档成员 task/episode 不匹配: {archive_relative}:{member_name}"
        )
    return Path(*parts)


def _publish_stream(stream: BinaryIO, target: Path) -> int:
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.tmp.{os.getpid()}.{hashlib.sha256(str(target).encode()).hexdigest()[:8]}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"发布临时文件已存在: {temporary}")
    size = 0
    try:
        with temporary.open("xb") as handle:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
                size += len(block)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o640)
        os.link(temporary, target)
        temporary.unlink()
    except BaseException:
        # 保留未完成临时文件；finalize 会拒绝它。
        raise
    return size


def _prepare(snapshot_root: Path, output_root: Path) -> dict[str, Any]:
    snapshot_root = resolve_real_directory(snapshot_root, "AgiBot Beta snapshot")
    parent = resolve_real_directory(output_root.parent, "materialization parent")
    output_root = parent / output_root.name
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"materialization output 已存在: {output_root}")
    tasks, _ = _task_catalog(snapshot_root)
    archives = _archives(snapshot_root)
    download_receipt = resolve_regular_file(
        snapshot_root,
        ".wm3d_v7_download_receipt.json",
    )
    download_value = json.loads(download_receipt.read_text(encoding="utf-8"))
    revision = str(download_value.get("revision", ""))
    if (
        download_value.get("schema") != DOWNLOAD_RECEIPT_SCHEMA
        or download_value.get("complete") is not True
        or download_value.get("source") != EXPECTED_SOURCE
        or download_value.get("repo_id") != EXPECTED_REPO_ID
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or download_value.get("resolved_revision") != revision
        or download_value.get("target") != str(snapshot_root)
        or int(download_value.get("payload_files", 0)) <= 0
        or int(download_value.get("payload_bytes", 0)) <= 0
    ):
        raise ValueError("AgiBot Beta download receipt 未完成或身份不匹配")

    output_root.mkdir(mode=0o750)
    for category in (*CATEGORIES, "task_info", "receipts"):
        (output_root / category).mkdir(mode=0o750)
    for task_id in sorted(tasks):
        source = resolve_regular_file(
            snapshot_root,
            f"task_info/task_{task_id}.json",
        )
        target = output_root / "task_info" / source.name
        with source.open("rb") as reader:
            _publish_stream(reader, target)
    plan = {
        "schema": PLAN_SCHEMA,
        "snapshot_root": str(snapshot_root),
        "download_receipt_sha256": sha256_file(download_receipt),
        "tasks": {str(key): list(value) for key, value in sorted(tasks.items())},
        "archives": archives,
    }
    plan_path = output_root / "materialization_plan.json"
    atomic_write_json(plan_path, plan, exclusive=True)
    receipt = {
        "schema": PREPARE_SCHEMA,
        "complete": True,
        "plan_sha256": sha256_file(plan_path),
        "tasks": len(tasks),
        "episodes": sum(len(value) for value in tasks.values()),
        "archives": len(archives),
        "created_at_utc": utc_now(),
    }
    atomic_write_json(
        output_root / "receipts" / "prepare.json",
        receipt,
        exclusive=True,
    )
    return receipt


def _load_plan(
    snapshot_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], dict[int, tuple[int, ...]], dict[int, int]]:
    snapshot_root = resolve_real_directory(snapshot_root, "AgiBot Beta snapshot")
    output_root = resolve_real_directory(output_root, "AgiBot Beta materialized root")
    plan_path = resolve_regular_file(output_root, "materialization_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or plan.get("snapshot_root") != str(
        snapshot_root
    ):
        raise ValueError("AgiBot Beta materialization plan 与 snapshot 不匹配")
    download_receipt = resolve_regular_file(
        snapshot_root,
        ".wm3d_v7_download_receipt.json",
    )
    if plan.get("download_receipt_sha256") != sha256_file(download_receipt):
        raise ValueError("AgiBot Beta download receipt 在 prepare 后发生漂移")
    prepare = json.loads(
        resolve_regular_file(output_root, "receipts/prepare.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        prepare.get("schema") != PREPARE_SCHEMA
        or prepare.get("complete") is not True
        or prepare.get("plan_sha256") != sha256_file(plan_path)
    ):
        raise ValueError("AgiBot Beta prepare receipt 不匹配")
    tasks = {
        int(key): tuple(int(item) for item in value)
        for key, value in plan["tasks"].items()
    }
    episode_to_task = {
        episode_id: task_id
        for task_id, episodes in tasks.items()
        for episode_id in episodes
    }
    return plan, tasks, episode_to_task


def _archive_receipt_path(output_root: Path, relative: str) -> Path:
    digest = hashlib.sha256(relative.encode()).hexdigest()
    return output_root / "receipts" / f"archive_{digest}.json"


def _extract(
    snapshot_root: Path,
    output_root: Path,
    *,
    shard_id: int,
    num_shards: int,
) -> dict[str, Any]:
    if num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ValueError("要求 0 <= shard-id < num-shards")
    snapshot_root = resolve_real_directory(snapshot_root, "AgiBot Beta snapshot")
    output_root = resolve_real_directory(output_root, "AgiBot Beta materialized root")
    plan, tasks, episode_to_task = _load_plan(snapshot_root, output_root)
    selected = []
    for archive in plan["archives"]:
        relative = str(archive["relative_path"])
        partition = int.from_bytes(
            hashlib.sha256(relative.encode()).digest()[:8],
            "big",
        )
        if partition % num_shards == shard_id:
            selected.append(archive)
    results = []
    for archive in selected:
        relative = str(archive["relative_path"])
        expected_size = int(archive["size"])
        path = resolve_regular_file(snapshot_root, relative)
        if path.stat().st_size != expected_size:
            raise ValueError(f"archive size 漂移: {relative}")
        archive_sha = sha256_file(path)
        receipt_path = _archive_receipt_path(output_root, relative)
        if receipt_path.exists() or receipt_path.is_symlink():
            receipt = json.loads(
                resolve_regular_file(
                    output_root,
                    receipt_path.relative_to(output_root).as_posix(),
                ).read_text(encoding="utf-8")
            )
            if (
                receipt.get("schema") == ARCHIVE_SCHEMA
                and receipt.get("complete") is True
                and receipt.get("archive_relative_path") == relative
                and receipt.get("archive_size") == expected_size
                and receipt.get("archive_sha256") == archive_sha
            ):
                results.append({"archive": relative, "status": "already_complete"})
                continue
            raise FileExistsError(f"archive receipt 已存在但不匹配: {receipt_path}")

        files = 0
        extracted_bytes = 0
        with tarfile.open(path, mode="r:*") as handle:
            for member in handle:
                if member.isdir():
                    # 文件发布会创建父目录；目录成员只做路径安全检查，避免
                    # 合法的 task/episode 前缀目录被误判为缺少第三级。
                    _member_parts(member.name)
                    continue
                relative_target = _normalized_member(
                    relative,
                    member.name,
                    task_ids=set(tasks),
                    episode_to_task=episode_to_task,
                )
                target = output_root / relative_target
                if not member.isfile():
                    raise ValueError(
                        f"archive 含链接、设备或特殊成员: {relative}:{member.name}"
                    )
                stream = handle.extractfile(member)
                if stream is None:
                    raise ValueError(
                        f"无法读取 archive member: {relative}:{member.name}"
                    )
                with stream:
                    extracted_bytes += _publish_stream(stream, target)
                files += 1
        receipt = {
            "schema": ARCHIVE_SCHEMA,
            "complete": True,
            "archive_relative_path": relative,
            "archive_size": expected_size,
            "archive_sha256": archive_sha,
            "files": files,
            "extracted_bytes": extracted_bytes,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(receipt_path, receipt, exclusive=True)
        results.append({"archive": relative, "status": "extracted", "files": files})
    return {
        "pass": True,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "archives": len(selected),
        "results": results,
    }


def _numeric_children(path: Path) -> set[int]:
    values = set()
    for child in path.iterdir():
        if child.is_symlink() or not child.is_dir() or not child.name.isdigit():
            raise ValueError(f"任务目录含非数字或非普通目录: {child}")
        values.add(int(child.name))
    return values


def _finalize(snapshot_root: Path, output_root: Path) -> dict[str, Any]:
    snapshot_root = resolve_real_directory(snapshot_root, "AgiBot Beta snapshot")
    output_root = resolve_real_directory(output_root, "AgiBot Beta materialized root")
    plan, tasks, _ = _load_plan(snapshot_root, output_root)
    if (output_root / ".wm3d_v7_beta_materialization_receipt.json").exists():
        raise FileExistsError("AgiBot Beta final materialization receipt 已存在")
    archive_receipts = []
    for archive in plan["archives"]:
        relative = str(archive["relative_path"])
        receipt_path = _archive_receipt_path(output_root, relative)
        receipt = json.loads(
            resolve_regular_file(
                output_root,
                receipt_path.relative_to(output_root).as_posix(),
            ).read_text(encoding="utf-8")
        )
        if (
            receipt.get("schema") != ARCHIVE_SCHEMA
            or receipt.get("complete") is not True
            or receipt.get("archive_relative_path") != relative
            or int(receipt.get("archive_size", -1)) != int(archive["size"])
        ):
            raise ValueError(f"archive receipt 不完整或错绑: {relative}")
        archive_receipts.append(receipt)
    leftovers = sorted(output_root.glob("**/.*.tmp.*"))
    if leftovers:
        raise ValueError(f"存在未完成发布临时文件: {leftovers[:8]}")

    for task_id, expected_episodes in sorted(tasks.items()):
        copied_task = resolve_regular_file(
            output_root,
            f"task_info/task_{task_id}.json",
        )
        if not copied_task.is_file():
            raise ValueError(f"缺 task_info: {task_id}")
        expected = set(expected_episodes)
        for category in CATEGORIES:
            task_root = resolve_real_directory(
                output_root / category / str(task_id),
                f"{category} task {task_id}",
            )
            actual = _numeric_children(task_root)
            if actual != expected:
                missing = sorted(expected.difference(actual))
                extra = sorted(actual.difference(expected))
                raise ValueError(
                    f"{category} task {task_id} episode 集合不精确: "
                    f"missing={missing[:8]} extra={extra[:8]}"
                )
    value = {
        "schema": FINAL_SCHEMA,
        "complete": True,
        "snapshot_root": str(snapshot_root),
        "download_receipt_sha256": plan["download_receipt_sha256"],
        "materialization_plan_sha256": sha256_file(
            output_root / "materialization_plan.json"
        ),
        "archive_receipts_content_sha256": canonical_sha256(archive_receipts),
        "archives": len(archive_receipts),
        "tasks": len(tasks),
        "episodes": sum(len(item) for item in tasks.values()),
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(
        output_root / ".wm3d_v7_beta_materialization_receipt.json",
        value,
        exclusive=True,
    )
    return value


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        result = _prepare(args.snapshot_root, args.output_root)
    elif args.mode == "extract":
        result = _extract(
            args.snapshot_root,
            args.output_root,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )
    else:
        result = _finalize(args.snapshot_root, args.output_root)
    print(json.dumps({"pass": True, "mode": args.mode, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
