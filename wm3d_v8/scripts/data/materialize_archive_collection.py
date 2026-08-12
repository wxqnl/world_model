#!/usr/bin/env python3
"""Safely expand a frozen archive subset into a sealed LeRobot collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import uuid
import zipfile

from wm3d_v3.data.manifest_contract import canonical_sha256, sha256_file


DOWNLOAD_SCHEMA = "wm3d_v8_raw_snapshot_receipt_v1"
EXTRACT_SCHEMA = "wm3d_v8_archive_extract_receipt_v1"
COLLECTION_SCHEMA = "wm3d_v8_lerobot_collection_receipt_v1"
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".zip")


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _member(name: str) -> Path:
    pure = PurePosixPath(name)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return Path(*pure.parts)


def _stream(reader: object, target: Path) -> int:
    target.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    with target.open("xb") as writer:
        while True:
            block = reader.read(8 * 1024 * 1024)  # type: ignore[attr-defined]
            if not block:
                break
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    target.chmod(0o640)
    return int(target.stat().st_size)


def _extract_tar(archive: Path, target: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    with tarfile.open(archive, mode="r:*") as handle:
        for item in handle:
            relative = _member(item.name)
            destination = target / relative
            if item.isdir():
                destination.mkdir(mode=0o750, parents=True, exist_ok=True)
                continue
            if not item.isfile():
                raise RuntimeError(f"archive contains a link/device/special member: {item.name}")
            reader = handle.extractfile(item)
            if reader is None:
                raise RuntimeError(f"cannot read archive member: {item.name}")
            with reader:
                total_bytes += _stream(reader, destination)
            files += 1
    return files, total_bytes


def _extract_zip(archive: Path, target: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    with zipfile.ZipFile(archive) as handle:
        for item in handle.infolist():
            relative = _member(item.filename)
            destination = target / relative
            mode = (item.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RuntimeError(f"zip contains a symbolic link: {item.filename}")
            if item.is_dir():
                destination.mkdir(mode=0o750, parents=True, exist_ok=True)
                continue
            with handle.open(item, "r") as reader:
                total_bytes += _stream(reader, destination)
            files += 1
    return files, total_bytes


def _archives(root: Path, prefix: str) -> list[tuple[str, Path]]:
    prefix_path = Path(prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts:
        raise RuntimeError("--source-prefix must be a safe relative path")
    subset = _directory(root / prefix_path, "archive subset")
    rows = []
    for candidate in sorted(subset.rglob("*")):
        if not candidate.name.lower().endswith(ARCHIVE_SUFFIXES):
            continue
        path = _regular(candidate, "archive")
        rows.append((path.relative_to(root).as_posix(), path))
    if not rows:
        raise RuntimeError(f"no tar/zip archives under frozen subset {prefix!r}")
    return rows


def _target_name(relative: str) -> str:
    digest = hashlib.sha256(relative.encode()).hexdigest()[:20]
    suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", relative)[-100:]
    return f"{digest}--{suffix}"


def _load_download(receipt: Path, snapshot: Path, source: str) -> tuple[Path, dict]:
    safe = _regular(receipt, "download receipt")
    value = json.loads(safe.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != DOWNLOAD_SCHEMA
        or value.get("source") != source
        or Path(str(value.get("snapshot_path", ""))).resolve(strict=True) != snapshot
        or int(value.get("snapshot_file_count", 0)) <= 0
        or int(value.get("snapshot_total_bytes", 0)) <= 0
    ):
        raise RuntimeError("download receipt does not bind this frozen snapshot/source")
    return safe, value


def _extract_one(
    relative: str,
    archive: Path,
    output: Path,
    *,
    download_receipt_sha256: str,
) -> dict:
    destination = output / _target_name(relative)
    stable_without_archive_digest = {
        "schema": EXTRACT_SCHEMA,
        "archive_relative_path": relative,
        "archive_size": int(archive.stat().st_size),
        "download_receipt_sha256": download_receipt_sha256,
    }
    receipt_name = ".wm3d_v8_extract_receipt.json"
    if destination.exists() or destination.is_symlink():
        destination = _directory(destination, "materialized archive")
        receipt = json.loads(_regular(destination / receipt_name, "extract receipt").read_text())
        if (
            all(
                receipt.get(key) == value
                for key, value in stable_without_archive_digest.items()
            )
            and re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", "")))
        ):
            return {"archive": relative, "status": "verified-skip"}
        raise FileExistsError(f"extract target exists with different identity: {destination}")
    temporary = output / f".extract-{destination.name}-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o750)
    # The first materialization is the only pass that hashes the multi-TB raw
    # archive.  Resume/finalize bind this immutable receipt and exact size;
    # they never perform another full payload scan.
    archive_sha = sha256_file(archive)
    if archive.name.lower().endswith(".zip"):
        files, total_bytes = _extract_zip(archive, temporary)
    else:
        files, total_bytes = _extract_tar(archive, temporary)
    info_files = sorted(temporary.glob("**/meta/info.json"))
    if not info_files:
        raise RuntimeError(f"archive has no LeRobot meta/info.json: {relative}")
    roots = []
    for info in info_files:
        _regular(info, "LeRobot info.json")
        roots.append(info.parent.parent.relative_to(temporary).as_posix())
    receipt = {
        **stable_without_archive_digest,
        "archive_sha256": archive_sha,
        "extracted_files": files,
        "extracted_bytes": total_bytes,
        "lerobot_roots": roots,
    }
    _publish(
        temporary / receipt_name,
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(),
    )
    os.replace(temporary, destination)
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"archive": relative, "status": "extracted", "lerobot_roots": len(roots)}


def _finalize(
    archives: list[tuple[str, Path]],
    output: Path,
    download_receipt: Path,
    source: str,
    prefix: str,
) -> dict:
    download_receipt_sha256 = sha256_file(download_receipt)
    receipt_rows = []
    expected = set()
    roots = []
    for relative, archive in archives:
        name = _target_name(relative)
        expected.add(name)
        destination = _directory(output / name, "materialized archive")
        receipt_path = _regular(destination / ".wm3d_v8_extract_receipt.json", "extract receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != EXTRACT_SCHEMA
            or receipt.get("archive_relative_path") != relative
            or receipt.get("archive_size") != int(archive.stat().st_size)
            or receipt.get("download_receipt_sha256") != download_receipt_sha256
            or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("archive_sha256", "")))
            is None
            or not receipt.get("lerobot_roots")
        ):
            raise RuntimeError(f"extract receipt mismatch: {relative}")
        receipt_rows.append(receipt)
        for root in receipt["lerobot_roots"]:
            root_path = destination / str(root)
            _regular(root_path / "meta" / "info.json", "LeRobot info.json")
            roots.append(str(root_path.resolve(strict=True)))
    extras = {
        item.name
        for item in output.iterdir()
        if item.name != "collection_receipt.json"
    }
    if extras != expected:
        raise RuntimeError(
            f"collection output closure mismatch: missing={sorted(expected-extras)[:8]} "
            f"extra={sorted(extras-expected)[:8]}"
        )
    value = {
        "schema": COLLECTION_SCHEMA,
        "source": source,
        "source_prefix": prefix,
        "download_receipt_path": str(download_receipt),
        "download_receipt_sha256": download_receipt_sha256,
        "archive_count": len(archives),
        "lerobot_root_count": len(roots),
        "lerobot_roots": sorted(roots),
        "archive_receipts_content_sha256": canonical_sha256(receipt_rows),
    }
    _publish(
        output / "collection_receipt.json",
        (json.dumps(value, sort_keys=True, indent=2) + "\n").encode(),
    )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--download-source", required=True)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise RuntimeError("require 0 <= worker-index < worker-count")
    snapshot = _directory(args.snapshot_root, "download snapshot")
    download_receipt, _ = _load_download(
        args.download_receipt, snapshot, args.download_source
    )
    output = args.output_root.absolute()
    if output.is_symlink():
        raise RuntimeError("output root cannot be a symlink")
    output.mkdir(mode=0o750, parents=True, exist_ok=True)
    archives = _archives(snapshot, args.source_prefix)
    if args.finalize:
        print(
            json.dumps(
                _finalize(
                    archives,
                    output,
                    download_receipt,
                    args.download_source,
                    args.source_prefix,
                ),
                sort_keys=True,
            )
        )
        return
    results = []
    for relative, archive in archives:
        owner = int.from_bytes(hashlib.sha256(relative.encode()).digest()[:8], "big")
        if owner % args.worker_count == args.worker_index:
            results.append(
                _extract_one(
                    relative,
                    archive,
                    output,
                    download_receipt_sha256=sha256_file(download_receipt),
                )
            )
    print(
        json.dumps(
            {
                "passed": True,
                "worker_index": args.worker_index,
                "worker_count": args.worker_count,
                "archives": len(results),
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
