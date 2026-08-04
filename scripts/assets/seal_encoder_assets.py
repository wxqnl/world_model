#!/usr/bin/env python3
"""Build a portable no-symlink VGGT/T5 asset bundle for offline encoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import uuid
from typing import Iterable

from wm3d.data.assets import (
    ASSET_RECEIPT_SCHEMA,
    verify_vggt_source,
    vggt_source_tree_sha256,
)
from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)


REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vggt-source-root", type=Path, required=True)
    parser.add_argument("--vggt-source-commit", required=True)
    parser.add_argument("--vggt-source-archive-sha256", required=True)
    parser.add_argument("--vggt-source-tree-sha256", required=True)
    parser.add_argument("--vggt-model", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-snapshot", type=Path, required=True)
    parser.add_argument("--vggt-revision", required=True)
    parser.add_argument("--task-model", default="google/flan-t5-xl")
    parser.add_argument("--task-snapshot", type=Path, required=True)
    parser.add_argument("--task-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _command(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _copy_file(source: Path, destination: Path) -> None:
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"asset source is not regular: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("rb", buffering=0) as reader:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, 0o640)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as writer:
                shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
        finally:
            os.close(descriptor)


def _snapshot_files(root: Path) -> Iterable[tuple[Path, Path]]:
    root = root.resolve(strict=True)
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for dirname in dirnames:
            path = directory_path / dirname
            if stat.S_ISLNK(os.lstat(path).st_mode):
                raise ValueError(f"directory symlink in model snapshot: {path}")
        for filename in filenames:
            path = directory_path / filename
            resolved = path.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"non-regular model snapshot entry: {path}")
            yield path, path.relative_to(root)


def _copy_snapshot(source: Path, destination: Path) -> int:
    files = list(_snapshot_files(source))
    if not files:
        raise ValueError(f"empty model snapshot: {source}")
    for path, relative in files:
        _copy_file(path, destination / relative)
    return len(files)


def _copy_vggt_source(
    source: Path,
    destination: Path,
    expected_commit: str,
    expected_archive_sha256: str,
    expected_tree_sha256: str,
) -> tuple[int, str]:
    input_source = Path(source)
    mode = os.lstat(input_source).st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError(f"VGGT source root is not a real directory: {input_source}")
    source = input_source.resolve(strict=True)
    if (source / ".git").is_dir():
        actual_commit = _command("git", "rev-parse", "HEAD", cwd=source)
        if actual_commit != expected_commit:
            raise ValueError(
                f"VGGT source commit {actual_commit} != {expected_commit}"
            )
        status = _command(
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            cwd=source,
        )
        if status:
            raise ValueError("VGGT tracked source is dirty")
        files = [
            item
            for item in _command("git", "ls-files", "-z", cwd=source).split("\0")
            if item
        ]
        if not files:
            raise ValueError("VGGT repository contains no tracked files")
        evidence = {}
        for relative in files:
            path = source / relative
            mode = os.lstat(path).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(f"unsupported tracked VGGT entry: {relative}")
            evidence[relative] = {
                "size": int(os.lstat(path).st_size),
                "sha256": sha256_file(path),
            }
        if vggt_source_tree_sha256(evidence) != expected_tree_sha256:
            raise ValueError("VGGT Git checkout tree SHA mismatch")
        provenance = "git_checkout"
    else:
        report = verify_vggt_source(
            source,
            expected_commit=expected_commit,
            expected_archive_sha256=expected_archive_sha256,
            expected_tree_sha256=expected_tree_sha256,
        )
        files = sorted(report["files"])
        provenance = "github_codeload_archive"
    for relative in files:
        path = source / relative
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ValueError(f"unsupported tracked VGGT entry: {relative}")
        _copy_file(path, destination / relative)
    return len(files), provenance


def _evidence(root: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink escaped into encoder asset bundle: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = {
                "size": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    return result


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    directories.append(root)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main() -> None:
    args = parse_args()
    revisions = (
        args.vggt_source_commit,
        args.vggt_revision,
        args.task_revision,
    )
    if any(not REVISION_RE.fullmatch(value) for value in revisions):
        raise ValueError("asset revisions must be immutable 40-64 digit hex commits")
    for name, value in (
        ("VGGT source archive", args.vggt_source_archive_sha256),
        ("VGGT source tree", args.vggt_source_tree_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{name} SHA must be 64 lowercase hex digits")
    vggt_snapshot = args.vggt_snapshot.resolve(strict=True)
    task_snapshot = args.task_snapshot.resolve(strict=True)
    if vggt_snapshot.name != args.vggt_revision:
        raise ValueError("VGGT snapshot basename does not match its revision")
    if task_snapshot.name != args.task_revision:
        raise ValueError("task snapshot basename does not match its revision")

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.incomplete.{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o750)
    try:
        source_relative = Path("vggt/source")
        vggt_relative = Path("vggt/model") / args.vggt_revision
        task_relative = Path("task/model") / args.task_revision
        source_files, source_provenance = _copy_vggt_source(
            args.vggt_source_root,
            temporary / source_relative,
            args.vggt_source_commit,
            args.vggt_source_archive_sha256,
            args.vggt_source_tree_sha256,
        )
        vggt_files = _copy_snapshot(
            vggt_snapshot,
            temporary / vggt_relative,
        )
        task_files = _copy_snapshot(
            task_snapshot,
            temporary / task_relative,
        )
        files = _evidence(temporary)
        receipt = {
            "schema": ASSET_RECEIPT_SCHEMA,
            "assets": {
                "vggt_source": {
                    "path": source_relative.as_posix(),
                    "commit": args.vggt_source_commit,
                    "archive_sha256": args.vggt_source_archive_sha256,
                    "tree_sha256": args.vggt_source_tree_sha256,
                    "provenance": source_provenance,
                    "files": source_files,
                },
                "vggt_model": {
                    "path": vggt_relative.as_posix(),
                    "repo_id": args.vggt_model,
                    "revision": args.vggt_revision,
                    "files": vggt_files,
                },
                "task_model": {
                    "path": task_relative.as_posix(),
                    "repo_id": args.task_model,
                    "revision": args.task_revision,
                    "files": task_files,
                },
            },
            "files": files,
        }
        receipt["content_sha256"] = canonical_sha256(receipt)
        atomic_write_json(
            temporary / "receipt.json",
            receipt,
            exclusive=True,
        )
        _fsync_directories(temporary)
        os.replace(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        # Keep the uniquely named incomplete tree as forensic evidence.
        raise
    print(
        json.dumps(
            {
                "pass": True,
                "output_root": str(output),
                "receipt_sha256": canonical_sha256(receipt),
                "files": len(files),
                "bytes": sum(int(item["size"]) for item in files.values()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
