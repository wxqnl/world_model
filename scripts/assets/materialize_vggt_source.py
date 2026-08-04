#!/usr/bin/env python3
"""Materialize a pinned VGGT source tree from GitHub's official codeload."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import time
import uuid

import httpx


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wm3d.data.assets import (  # noqa: E402
    VGGT_SOURCE_RECEIPT_NAME,
    VGGT_SOURCE_RECEIPT_SCHEMA,
    verify_vggt_source,
    vggt_source_evidence,
    vggt_source_tree_sha256,
)
from wm3d.data.contracts import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _summary(report: dict[str, object], *, status: str) -> dict[str, object]:
    files = report.get("files")
    if not isinstance(files, dict):
        raise ValueError("VGGT source verification returned invalid file evidence")
    return {
        "pass": bool(report.get("pass")),
        "status": status,
        "root": report.get("root"),
        "commit": report.get("commit"),
        "archive_sha256": report.get("archive_sha256"),
        "tree_sha256": report.get("tree_sha256"),
        "receipt_sha256": report.get("receipt_sha256"),
        "files": len(files),
        "bytes": sum(int(value["size"]) for value in files.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--tree-sha256", required=True)
    parser.add_argument("--network-attempts", type=int, default=3)
    return parser.parse_args()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _download(
    archive: Path,
    *,
    url: str,
    expected_sha256: str,
    attempts: int,
) -> None:
    if archive.exists() or archive.is_symlink():
        info = os.lstat(archive)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ValueError(f"VGGT archive is not regular: {archive}")
        if sha256_file(archive) != expected_sha256:
            raise ValueError(f"VGGT archive SHA mismatch: {archive}")
        return
    if attempts < 1:
        raise ValueError("network attempts must be >= 1")
    archive.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        temporary = archive.parent / f".{archive.name}.incomplete.{uuid.uuid4().hex}"
        try:
            with httpx.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, connect=30.0),
            ) as response:
                response.raise_for_status()
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(temporary, flags, 0o640)
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as handle:
                        for chunk in response.iter_bytes(8 * 1024 * 1024):
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    os.close(descriptor)
            actual_sha256 = sha256_file(temporary)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"VGGT archive SHA {actual_sha256} != {expected_sha256}"
                )
            os.replace(temporary, archive)
            _fsync_directory(archive.parent)
            return
        except Exception:
            if attempt == attempts:
                raise
            print(
                f"VGGT codeload attempt {attempt}/{attempts} failed; "
                f"evidence kept at {temporary}",
                flush=True,
            )
            time.sleep(float(attempt * 2))
    raise AssertionError("unreachable")


def _relative_member(name: str, prefix: str) -> Path | None:
    if name == prefix.rstrip("/"):
        return None
    if not name.startswith(prefix):
        raise ValueError(f"VGGT archive member escaped prefix: {name!r}")
    relative = name[len(prefix) :]
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe VGGT archive member: {name!r}")
    return Path(*pure.parts)


def _extract(archive: Path, temporary: Path, *, commit: str) -> int:
    prefix = f"vggt-{commit}/"
    files = 0
    with tarfile.open(archive, mode="r:gz") as source:
        for member in source:
            relative = _relative_member(member.name, prefix)
            if relative is None:
                continue
            output = temporary / relative
            if member.isdir():
                output.mkdir(mode=0o750, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"VGGT archive contains a link or special file: {member.name}"
                )
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read VGGT archive member: {member.name}")
            output.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(output, flags, 0o640)
            try:
                with stream, os.fdopen(descriptor, "wb", closefd=False) as target:
                    shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                os.close(descriptor)
            files += 1
    if files < 1:
        raise ValueError("VGGT archive contained no source files")
    return files


def main() -> None:
    args = parse_args()
    if COMMIT_RE.fullmatch(args.commit) is None:
        raise ValueError("VGGT commit must be 40 lowercase hex digits")
    for name, value in (
        ("archive", args.archive_sha256),
        ("tree", args.tree_sha256),
    ):
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"VGGT {name} SHA must be 64 lowercase hex digits")
    output = args.output_root.absolute()
    if output.exists() or output.is_symlink():
        report = verify_vggt_source(
            output,
            expected_commit=args.commit,
            expected_archive_sha256=args.archive_sha256,
            expected_tree_sha256=args.tree_sha256,
        )
        print(
            json.dumps(
                _summary(report, status="already_complete"),
                sort_keys=True,
            )
        )
        return
    archive_root = args.archive_root.absolute()
    if archive_root.exists() or archive_root.is_symlink():
        info = os.lstat(archive_root)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"archive root is not a real directory: {archive_root}")
    else:
        archive_root.mkdir(mode=0o750, parents=True)
    archive = archive_root / f"vggt-{args.commit}.tar.gz"
    url = f"https://codeload.github.com/facebookresearch/vggt/tar.gz/{args.commit}"
    _download(
        archive,
        url=url,
        expected_sha256=args.archive_sha256,
        attempts=args.network_attempts,
    )
    output.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.incomplete.{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o750)
    try:
        _extract(archive, temporary, commit=args.commit)
        files = vggt_source_evidence(temporary)
        tree_sha256 = vggt_source_tree_sha256(files)
        if tree_sha256 != args.tree_sha256:
            raise ValueError(
                f"VGGT tree SHA {tree_sha256} != {args.tree_sha256}"
            )
        receipt = {
            "schema": VGGT_SOURCE_RECEIPT_SCHEMA,
            "commit": args.commit,
            "archive_url": url,
            "archive_sha256": args.archive_sha256,
            "tree_sha256": tree_sha256,
            "files": files,
        }
        receipt["content_sha256"] = canonical_sha256(receipt)
        atomic_write_json(
            temporary / VGGT_SOURCE_RECEIPT_NAME,
            receipt,
            exclusive=True,
        )
        verify_vggt_source(
            temporary,
            expected_commit=args.commit,
            expected_archive_sha256=args.archive_sha256,
            expected_tree_sha256=args.tree_sha256,
        )
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except Exception:
        # Keep the uniquely named incomplete tree as forensic evidence.
        raise
    report = verify_vggt_source(
        output,
        expected_commit=args.commit,
        expected_archive_sha256=args.archive_sha256,
        expected_tree_sha256=args.tree_sha256,
    )
    print(json.dumps(_summary(report, status="materialized"), sort_keys=True))


if __name__ == "__main__":
    main()
