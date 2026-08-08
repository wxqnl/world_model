from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping


class ImmutableArtifactConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishResult:
    path: Path
    sha256: str
    created: bool


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file_no_follow(path: Path) -> bytes | None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode):
        raise ImmutableArtifactConflict(
            f"immutable artifact destination is a symlink: {path}"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise ImmutableArtifactConflict(
            f"immutable artifact destination is not a regular file: {path}"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ImmutableArtifactConflict(
            f"cannot open immutable artifact without following links: {path}"
        ) from exc
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ImmutableArtifactConflict(
                f"immutable artifact changed type while opening: {path}"
            )
        return stream.read()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_result(path: Path, payload: bytes) -> PublishResult:
    existing = _read_regular_file_no_follow(path)
    if existing is None:
        raise FileNotFoundError(path)
    if existing != payload:
        raise ImmutableArtifactConflict(
            f"immutable artifact already exists with different content: {path}"
        )
    return PublishResult(path=path, sha256=_sha256(payload), created=False)


def publish_immutable_bytes(
    path: str | Path,
    payload: bytes,
) -> PublishResult:
    destination = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("immutable artifact payload must be bytes")
    destination.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_regular_file_no_follow(destination)
    if existing is not None:
        return _existing_result(destination, payload)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    created = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
            created = True
        except FileExistsError:
            return _existing_result(destination, payload)
        _fsync_directory(destination.parent)
        return PublishResult(
            path=destination,
            sha256=_sha256(payload),
            created=True,
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if created:
            _fsync_directory(destination.parent)


def require_distinct_output_paths(
    outputs: Mapping[str, str | Path],
) -> None:
    canonical: dict[Path, str] = {}
    inodes: dict[tuple[int, int], str] = {}
    for label, raw_path in outputs.items():
        path = Path(raw_path)
        resolved = path.resolve(strict=False)
        previous = canonical.get(resolved)
        if previous is not None:
            raise ImmutableArtifactConflict(
                f"output path alias: {label} aliases {previous}: {resolved}"
            )
        canonical[resolved] = str(label)
        try:
            metadata = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            continue
        inode = (int(metadata.st_dev), int(metadata.st_ino))
        previous = inodes.get(inode)
        if previous is not None:
            raise ImmutableArtifactConflict(
                f"output path alias: {label} aliases {previous}: {path}"
            )
        inodes[inode] = str(label)
