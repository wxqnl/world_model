#!/usr/bin/env python3
"""事务化调用 AgiBot Beta 官方 converter，把一个 task 变为 LeRobot root。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.scale5b.verify_agibot_converter_environment import (
    RECEIPT_SCHEMA as CONVERTER_ENVIRONMENT_RECEIPT_SCHEMA,
)
from scripts.scale5b.verify_agibot_converter_environment import validate_receipt
from wm3d_v3.data.scale5b_contracts import (
    atomic_write_json,
    canonical_sha256,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    utc_now,
)


RECEIPT_SCHEMA = "wm3d_v7_native5b_agibot_beta_conversion_receipt_v1"
DOWNLOAD_RECEIPT_SCHEMA = "wm3d_v7_native5b_raw_download_receipt_v1"
DOWNLOAD_RECEIPT_NAME = ".wm3d_v7_download_receipt.json"
CONVERTER_SOURCE = "agibot_alpha_converter_snapshot"
CONVERTER_REPO_ID = "agibot-world/AgiBotWorld-Alpha"
CONVERTER_RELATIVE_PATH = Path("scripts/convert_to_lerobot.py")
MATERIALIZATION_RECEIPT_SCHEMA = (
    "wm3d_v7_native5b_agibot_beta_materialization_receipt_v1"
)
MATERIALIZATION_RECEIPT_NAME = ".wm3d_v7_beta_materialization_receipt.json"
COLLECTION_RECEIPT_SCHEMA = (
    "wm3d_v7_native5b_agibot_beta_conversion_collection_receipt_v1"
)
COLLECTION_RECEIPT_NAME = ".wm3d_v7_beta_conversion_collection_receipt.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vendor-converter", type=Path, required=True)
    parser.add_argument(
        "--converter-download-receipt",
        type=Path,
        required=True,
        help="冻结的 AgiBotWorld-Alpha 官方转换器快照下载 receipt。",
    )
    parser.add_argument(
        "--converter-environment-receipt",
        type=Path,
        required=True,
        help="独立 LeRobot dataset-v2 转换镜像内的 environment receipt。",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", type=int)
    group.add_argument("--task-list", type=Path)
    parser.add_argument(
        "--array-index",
        type=int,
        help="使用 --task-list 时选择的 0-based 行号，通常传 SLURM_ARRAY_TASK_ID。",
    )
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def _task_ids(path: Path) -> tuple[int, ...]:
    safe = resolve_regular_file(path.parent, path.name)
    values = tuple(
        int(line.strip())
        for line in safe.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values or any(value < 0 for value in values):
        raise ValueError("task list 必须含非负 task id")
    if len(values) != len(set(values)):
        raise ValueError("task list 含重复 task id")
    return values


def _task_id(args: argparse.Namespace) -> int:
    if args.task_id is not None:
        if args.array_index is not None:
            raise ValueError("--task-id 不能与 --array-index 同用")
        value = int(args.task_id)
    else:
        if args.array_index is None or args.array_index < 0:
            raise ValueError("--task-list 必须同时提供非负 --array-index")
        values = _task_ids(args.task_list)
        if args.array_index >= len(values):
            raise IndexError(
                f"array-index {args.array_index} 超出 task list 长度 {len(values)}"
            )
        value = int(values[args.array_index])
    if value < 0:
        raise ValueError("task-id 必须非负")
    return value


def _materialization_receipt(raw_root: Path) -> tuple[Path, dict[str, object]]:
    path = resolve_regular_file(raw_root, MATERIALIZATION_RECEIPT_NAME)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("AgiBot Beta materialization receipt 不是 JSON object")
    if (
        value.get("schema") != MATERIALIZATION_RECEIPT_SCHEMA
        or value.get("complete") is not True
        or int(value.get("archives", 0)) <= 0
        or int(value.get("tasks", 0)) <= 0
        or int(value.get("episodes", 0)) <= 0
    ):
        raise ValueError("AgiBot Beta materialization receipt 未完成或身份不匹配")
    plan = resolve_regular_file(raw_root, "materialization_plan.json")
    if value.get("materialization_plan_sha256") != sha256_file(plan):
        raise ValueError("AgiBot Beta materialization plan 与 final receipt 不匹配")
    return path, value


def _converter_receipt(
    converter: Path,
    receipt_path: Path,
) -> tuple[Path, dict[str, object]]:
    receipt = resolve_regular_file(receipt_path.parent, receipt_path.name)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("converter download receipt 不是 JSON object")
    revision = str(value.get("revision", ""))
    resolved_revision = str(value.get("resolved_revision", ""))
    payload_sha256 = value.get("payload_sha256")
    if (
        receipt.name != DOWNLOAD_RECEIPT_NAME
        or value.get("schema") != DOWNLOAD_RECEIPT_SCHEMA
        or value.get("complete") is not True
        or value.get("source") != CONVERTER_SOURCE
        or value.get("repo_id") != CONVERTER_REPO_ID
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or resolved_revision != revision
        or int(value.get("payload_files", 0)) < 1
        or int(value.get("payload_bytes", 0)) <= 0
        or not isinstance(payload_sha256, dict)
    ):
        raise ValueError("converter download receipt 未完成或身份不匹配")
    snapshot_root = resolve_real_directory(receipt.parent, "converter snapshot root")
    if value.get("target") != str(snapshot_root):
        raise ValueError("converter download receipt target 与所在目录不匹配")
    expected_converter = snapshot_root / CONVERTER_RELATIVE_PATH
    if converter != expected_converter:
        raise ValueError(
            f"vendor converter 必须是冻结快照内的 {CONVERTER_RELATIVE_PATH}"
        )
    if payload_sha256.get(CONVERTER_RELATIVE_PATH.as_posix()) != sha256_file(converter):
        raise ValueError("converter 文件 SHA 与冻结 download receipt 不匹配")
    return receipt, value


def _converter_environment_receipt(path: Path) -> tuple[Path, dict[str, object]]:
    receipt = resolve_regular_file(path.parent, path.name)
    value = validate_receipt(receipt, check_current=True)
    if (
        value.get("schema") != CONVERTER_ENVIRONMENT_RECEIPT_SCHEMA
        or value.get("pass") is not True
        or value.get("python_executable") != os.path.abspath(sys.executable)
    ):
        raise ValueError("converter environment receipt 与当前 Python 不匹配")
    return receipt, value


def _finalize_collection(
    *,
    raw_root: Path,
    output_root: Path,
    converter: Path,
    converter_download_receipt: Path,
    converter_environment_receipt: Path,
    task_list: Path,
    materialization_receipt: Path,
) -> dict[str, object]:
    task_ids = _task_ids(task_list)
    expected_names = {f"task_{task_id:06d}" for task_id in task_ids}
    converter_sha256 = sha256_file(converter)
    converter_download_receipt_sha256 = sha256_file(converter_download_receipt)
    converter_environment_receipt_sha256 = sha256_file(converter_environment_receipt)
    converter_environment_value = validate_receipt(
        converter_environment_receipt,
        check_current=True,
    )
    lerobot_revision = str(converter_environment_value["lerobot_revision"])
    materialization_receipt_sha256 = sha256_file(materialization_receipt)
    receipts = []
    lerobot_roots = 0
    for task_id in task_ids:
        destination = resolve_real_directory(
            output_root / f"task_{task_id:06d}",
            f"converted task {task_id}",
        )
        receipt_path = resolve_regular_file(
            destination,
            ".wm3d_v7_conversion_receipt.json",
        )
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != RECEIPT_SCHEMA
            or value.get("complete") is not True
            or value.get("task_id") != task_id
            or value.get("converter_sha256") != converter_sha256
            or value.get("converter_download_receipt_sha256")
            != converter_download_receipt_sha256
            or value.get("converter_environment_receipt_sha256")
            != converter_environment_receipt_sha256
            or value.get("lerobot_revision") != lerobot_revision
            or value.get("materialization_receipt_sha256")
            != materialization_receipt_sha256
            or value.get("raw_root") != str(raw_root)
        ):
            raise ValueError(f"Beta task {task_id} conversion receipt 未完成或错绑")
        info_files = sorted(destination.glob("**/meta/info.json"))
        actual_roots = {
            path.parent.parent.relative_to(destination).as_posix()
            for path in info_files
            if path.is_file() and not path.is_symlink()
        }
        if not actual_roots or actual_roots != set(value.get("lerobot_roots", [])):
            raise ValueError(f"Beta task {task_id} LeRobot root 集合与 receipt 不一致")
        receipts.append(value)
        lerobot_roots += len(actual_roots)

    actual_names = set()
    for child in output_root.iterdir():
        if child.name == COLLECTION_RECEIPT_NAME:
            continue
        if child.is_symlink() or not child.is_dir():
            raise ValueError(f"Beta conversion output 含未知或未完成条目: {child}")
        actual_names.add(child.name)
    if actual_names != expected_names:
        missing = sorted(expected_names.difference(actual_names))
        extra = sorted(actual_names.difference(expected_names))
        raise ValueError(
            f"Beta task 集合不精确: missing={missing[:8]} extra={extra[:8]}"
        )

    stable = {
        "schema": COLLECTION_RECEIPT_SCHEMA,
        "complete": True,
        "raw_root": str(raw_root),
        "task_list_sha256": sha256_file(task_list),
        "converter_sha256": converter_sha256,
        "converter_download_receipt_sha256": converter_download_receipt_sha256,
        "converter_environment_receipt_sha256": (converter_environment_receipt_sha256),
        "lerobot_revision": lerobot_revision,
        "materialization_receipt_sha256": materialization_receipt_sha256,
        "tasks": len(task_ids),
        "lerobot_roots": lerobot_roots,
        "task_receipts_content_sha256": canonical_sha256(receipts),
    }
    final_path = output_root / COLLECTION_RECEIPT_NAME
    if final_path.exists() or final_path.is_symlink():
        current = json.loads(
            resolve_regular_file(output_root, COLLECTION_RECEIPT_NAME).read_text(
                encoding="utf-8"
            )
        )
        if all(current.get(key) == value for key, value in stable.items()):
            return {**stable, "status": "already_complete"}
        raise FileExistsError("Beta conversion final receipt 已存在但不匹配")
    atomic_write_json(
        final_path,
        {**stable, "completed_at_utc": utc_now()},
        exclusive=True,
    )
    return {**stable, "status": "finalized"}


def main() -> None:
    args = parse_args()
    raw_root = resolve_real_directory(args.raw_root, "AgiBot Beta raw root")
    materialization_receipt, _materialization_value = _materialization_receipt(raw_root)
    materialization_receipt_sha256 = sha256_file(materialization_receipt)
    output_root = resolve_real_directory(
        args.output_root, "AgiBot Beta collection root"
    )
    converter = resolve_regular_file(
        args.vendor_converter.parent,
        args.vendor_converter.name,
    )
    converter_download_receipt, converter_download_value = _converter_receipt(
        converter,
        args.converter_download_receipt,
    )
    converter_download_receipt_sha256 = sha256_file(converter_download_receipt)
    converter_environment_receipt, converter_environment_value = (
        _converter_environment_receipt(args.converter_environment_receipt)
    )
    converter_environment_receipt_sha256 = sha256_file(converter_environment_receipt)
    lerobot_revision = str(converter_environment_value["lerobot_revision"])
    if args.finalize:
        if (
            args.task_list is None
            or args.task_id is not None
            or args.array_index is not None
        ):
            raise ValueError(
                "--finalize 只接受 --task-list，不能提供 --task-id/--array-index"
            )
        result = _finalize_collection(
            raw_root=raw_root,
            output_root=output_root,
            converter=converter,
            converter_download_receipt=converter_download_receipt,
            converter_environment_receipt=converter_environment_receipt,
            task_list=args.task_list,
            materialization_receipt=materialization_receipt,
        )
        print(json.dumps({"pass": True, **result}, sort_keys=True))
        return
    task_id = _task_id(args)
    destination = output_root / f"task_{task_id:06d}"
    receipt_name = ".wm3d_v7_conversion_receipt.json"
    if destination.exists() or destination.is_symlink():
        destination = resolve_real_directory(destination, "converted task root")
        receipt = destination / receipt_name
        if receipt.is_file() and not receipt.is_symlink():
            value = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                value.get("schema") == RECEIPT_SCHEMA
                and value.get("task_id") == task_id
                and value.get("converter_sha256") == sha256_file(converter)
                and value.get("converter_download_receipt_sha256")
                == converter_download_receipt_sha256
                and value.get("converter_environment_receipt_sha256")
                == converter_environment_receipt_sha256
                and value.get("lerobot_revision") == lerobot_revision
                and value.get("materialization_receipt_sha256")
                == materialization_receipt_sha256
                and value.get("complete") is True
            ):
                print(
                    json.dumps(
                        {"pass": True, "status": "already_complete", "task_id": task_id}
                    )
                )
                return
        raise FileExistsError(f"转换目标已存在但无匹配完成 receipt: {destination}")

    temporary = output_root / f".convert-task-{task_id:06d}-{os.getpid()}"
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"转换临时目录已存在: {temporary}")
    command = [
        sys.executable,
        str(converter),
        "--src_path",
        str(raw_root),
        "--task_id",
        str(task_id),
        "--tgt_path",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    info_files = sorted(temporary.glob("**/meta/info.json"))
    if not info_files:
        raise ValueError(
            f"官方 converter 未产生 LeRobot meta/info.json: task {task_id}"
        )
    for path in info_files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"转换 metadata 不是普通文件: {path}")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "complete": True,
        "task_id": task_id,
        "converter": str(converter),
        "converter_sha256": sha256_file(converter),
        "converter_repo_id": converter_download_value["repo_id"],
        "converter_revision": converter_download_value["revision"],
        "converter_download_receipt": str(converter_download_receipt),
        "converter_download_receipt_sha256": converter_download_receipt_sha256,
        "converter_environment_receipt": str(converter_environment_receipt),
        "converter_environment_receipt_sha256": (converter_environment_receipt_sha256),
        "lerobot_revision": lerobot_revision,
        "raw_root": str(raw_root),
        "materialization_receipt": str(materialization_receipt),
        "materialization_receipt_sha256": materialization_receipt_sha256,
        "lerobot_roots": [
            path.parent.parent.relative_to(temporary).as_posix() for path in info_files
        ],
        "completed_at_utc": utc_now(),
    }
    atomic_write_json(temporary / receipt_name, receipt, exclusive=True)
    os.replace(temporary, destination)
    directory_fd = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(
        json.dumps(
            {
                "pass": True,
                "status": "converted",
                "task_id": task_id,
                "target": str(destination),
                "lerobot_roots": len(info_files),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
