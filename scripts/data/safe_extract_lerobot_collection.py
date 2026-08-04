#!/usr/bin/env python3
"""安全、可分片地把供应商归档展开成独立 LeRobot root collection。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import zipfile

from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    utc_now,
)


RECEIPT_SCHEMA = "wm3d_v7_archive_extract_receipt_v1"
COLLECTION_RECEIPT_SCHEMA = "wm3d_v7_collection_materialization_receipt_v1"
DOWNLOAD_RECEIPT_SCHEMA = "wm3d_v7_raw_download_receipt_v1"
COLLECTION_RECEIPT_NAME = ".wm3d_v7_collection_materialization_receipt.json"
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--download-receipt", type=Path)
    return parser.parse_args()


def _is_archive(path: Path) -> bool:
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def _member_relative(name: str) -> Path:
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"归档成员路径不安全: {name!r}")
    return Path(*pure.parts)


def _write_stream(stream: object, target: Path) -> None:
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with target.open("xb") as handle:
        shutil.copyfileobj(stream, handle, length=8 * 1024 * 1024)  # type: ignore[arg-type]
        handle.flush()
        os.fsync(handle.fileno())
    target.chmod(0o640)


def _extract_tar(archive: Path, target: Path) -> int:
    files = 0
    with tarfile.open(archive, mode="r:*") as handle:
        for member in handle:
            relative = _member_relative(member.name)
            output = target / relative
            if member.isdir():
                output.mkdir(mode=0o750, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"归档含链接、设备或其他特殊成员，拒绝: {archive}:{member.name}"
                )
            stream = handle.extractfile(member)
            if stream is None:
                raise ValueError(f"无法读取归档成员: {archive}:{member.name}")
            with stream:
                _write_stream(stream, output)
            files += 1
    return files


def _extract_zip(archive: Path, target: Path) -> int:
    files = 0
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            relative = _member_relative(member.filename)
            output = target / relative
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"zip 含符号链接，拒绝: {archive}:{member.filename}")
            if member.is_dir():
                output.mkdir(mode=0o750, parents=True, exist_ok=True)
                continue
            with handle.open(member, "r") as stream:
                _write_stream(stream, output)
            files += 1
    return files


def _destination_name(relative: str) -> str:
    digest = hashlib.sha256(relative.encode()).hexdigest()[:16]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", relative)[-96:]
    return f"{digest}--{stem}"


def _already_complete(
    target: Path,
    relative: str,
    size: int,
    archive_sha256: str,
) -> bool:
    receipt = target / ".wm3d_v7_extract_receipt.json"
    if not receipt.is_file() or receipt.is_symlink():
        return False
    value = json.loads(receipt.read_text(encoding="utf-8"))
    return (
        value.get("schema") == RECEIPT_SCHEMA
        and value.get("archive_relative_path") == relative
        and value.get("archive_size") == size
        and value.get("archive_sha256") == archive_sha256
        and value.get("complete") is True
    )


def _all_archives(archive_root: Path) -> list[tuple[str, Path]]:
    archives = []
    for candidate in sorted(archive_root.rglob("*")):
        if not _is_archive(candidate):
            continue
        relative = candidate.relative_to(archive_root).as_posix()
        archives.append((relative, resolve_regular_file(archive_root, relative)))
    if not archives:
        raise ValueError("archive root 没有 tar/zip 归档")
    return archives


def _finalize(
    archive_root: Path,
    output_root: Path,
    archives: list[tuple[str, Path]],
    download_receipt: Path | None,
) -> dict[str, object]:
    if download_receipt is None:
        raise ValueError("--finalize 必须提供 --download-receipt")
    receipt_path = resolve_regular_file(
        download_receipt.parent,
        download_receipt.name,
    )
    download = json.loads(receipt_path.read_text(encoding="utf-8"))
    revision = str(download.get("revision", ""))
    if (
        not isinstance(download, dict)
        or download.get("schema") != DOWNLOAD_RECEIPT_SCHEMA
        or download.get("complete") is not True
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or download.get("resolved_revision") != revision
        or int(download.get("payload_files", 0)) <= 0
        or int(download.get("payload_bytes", 0)) <= 0
    ):
        raise ValueError("download receipt 未完成或 schema 不匹配")
    snapshot_root = resolve_real_directory(
        Path(str(download.get("target", ""))),
        "download receipt target",
    )
    try:
        archive_root.relative_to(snapshot_root)
    except ValueError as exc:
        raise ValueError("archive root 不属于 download receipt target") from exc

    expected_names = set()
    receipts = []
    lerobot_roots = 0
    extracted_files = 0
    for relative, archive in archives:
        destination_name = _destination_name(relative)
        expected_names.add(destination_name)
        destination = resolve_real_directory(
            output_root / destination_name,
            f"materialized archive {relative}",
        )
        extract_receipt = resolve_regular_file(
            destination,
            ".wm3d_v7_extract_receipt.json",
        )
        value = json.loads(extract_receipt.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema") != RECEIPT_SCHEMA
            or value.get("complete") is not True
            or value.get("archive_relative_path") != relative
            or int(value.get("archive_size", -1)) != archive.stat().st_size
            or value.get("archive_sha256") != sha256_file(archive)
        ):
            raise ValueError(f"archive extract receipt 未完成或错绑: {relative}")
        info_files = sorted(destination.glob("**/meta/info.json"))
        if len(info_files) != len(value.get("lerobot_roots", [])) or not info_files:
            raise ValueError(f"LeRobot root 数与 receipt 不一致: {relative}")
        for info in info_files:
            if info.is_symlink() or not info.is_file():
                raise ValueError(f"LeRobot metadata 不是普通文件: {info}")
        receipts.append(value)
        lerobot_roots += len(info_files)
        extracted_files += int(value.get("extracted_files", 0))

    final_path = output_root / COLLECTION_RECEIPT_NAME
    actual_names = set()
    for child in output_root.iterdir():
        if child.name == COLLECTION_RECEIPT_NAME:
            continue
        if child.is_symlink() or not child.is_dir():
            raise ValueError(f"collection output 含未知或未完成条目: {child}")
        actual_names.add(child.name)
    if actual_names != expected_names:
        missing = sorted(expected_names.difference(actual_names))
        extra = sorted(actual_names.difference(expected_names))
        raise ValueError(
            f"collection archive 集合不精确: missing={missing[:8]} extra={extra[:8]}"
        )

    stable = {
        "schema": COLLECTION_RECEIPT_SCHEMA,
        "complete": True,
        "archive_root": str(archive_root),
        "download_receipt_sha256": sha256_file(receipt_path),
        "archives": len(archives),
        "lerobot_roots": lerobot_roots,
        "extracted_files": extracted_files,
        "archive_receipts_content_sha256": canonical_sha256(receipts),
    }
    if final_path.exists() or final_path.is_symlink():
        current = json.loads(
            resolve_regular_file(output_root, COLLECTION_RECEIPT_NAME).read_text(
                encoding="utf-8"
            )
        )
        if all(current.get(key) == value for key, value in stable.items()):
            return {**stable, "status": "already_complete"}
        raise FileExistsError("collection final receipt 已存在但不匹配")
    atomic_write_json(
        final_path,
        {**stable, "completed_at_utc": utc_now()},
        exclusive=True,
    )
    return {**stable, "status": "finalized"}


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("要求 0 <= shard-id < num-shards")
    archive_root = resolve_real_directory(args.archive_root, "archive root")
    output_root = resolve_real_directory(args.output_root, "collection output root")
    all_archives = _all_archives(archive_root)
    if args.finalize:
        result = _finalize(
            archive_root,
            output_root,
            all_archives,
            args.download_receipt,
        )
        print(json.dumps({"pass": True, **result}, sort_keys=True))
        return
    archives = []
    for relative, candidate in all_archives:
        digest = int.from_bytes(hashlib.sha256(relative.encode()).digest()[:8], "big")
        if digest % args.num_shards != args.shard_id:
            continue
        archives.append(candidate)
    if not archives:
        print(
            json.dumps(
                {"pass": True, "status": "empty_shard", "results": []},
                sort_keys=True,
            )
        )
        return

    results = []
    for archive in archives:
        relative = archive.relative_to(archive_root).as_posix()
        size = archive.stat().st_size
        archive_sha256 = sha256_file(archive)
        destination = output_root / _destination_name(relative)
        if destination.exists() or destination.is_symlink():
            destination = resolve_real_directory(destination, "extracted dataset root")
            if _already_complete(
                destination,
                relative,
                size,
                archive_sha256,
            ):
                results.append({"archive": relative, "status": "already_complete"})
                continue
            raise FileExistsError(
                f"目标已存在但无匹配完成 receipt，禁止覆盖: {destination}"
            )
        temporary = output_root / f".extract-{destination.name}-{os.getpid()}"
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError(f"临时目录已存在: {temporary}")
        temporary.mkdir(mode=0o750)
        try:
            if archive.name.lower().endswith(".zip"):
                file_count = _extract_zip(archive, temporary)
            else:
                file_count = _extract_tar(archive, temporary)
            info_files = sorted(temporary.glob("**/meta/info.json"))
            if not info_files:
                raise ValueError(
                    f"归档不是 LeRobot 数据：缺少 meta/info.json: {archive}"
                )
            for path in info_files:
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"LeRobot metadata 不是普通文件: {path}")
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "complete": True,
                "archive_relative_path": relative,
                "archive_size": size,
                "archive_sha256": archive_sha256,
                "extracted_files": file_count,
                "lerobot_roots": [
                    path.parent.parent.relative_to(temporary).as_posix()
                    for path in info_files
                ],
                "completed_at_utc": utc_now(),
            }
            atomic_write_json(
                temporary / ".wm3d_v7_extract_receipt.json",
                receipt,
                exclusive=True,
            )
            os.replace(temporary, destination)
            directory_fd = os.open(output_root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            results.append(
                {
                    "archive": relative,
                    "status": "extracted",
                    "target": str(destination),
                    "files": file_count,
                    "lerobot_roots": len(info_files),
                }
            )
        except BaseException:
            # 保留临时目录作为中断证据；绝不自动删除或覆盖。
            raise
    print(json.dumps({"pass": True, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
