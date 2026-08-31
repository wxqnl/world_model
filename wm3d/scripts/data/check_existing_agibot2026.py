#!/usr/bin/env python3
"""Bounded compatibility probe for an already-downloaded AgiBotWorld2026 tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile


SCHEMA = "wm3d_v8_agibot2026_existing_snapshot_probe_v1"
DEFAULT_PREFIXES = (
    "ImitationLearning",
    "RichInteraction",
    "ReinforcementLearning",
)
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".zip")


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return path


def _is_archive(path: Path) -> bool:
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def _inspect_info(payload: bytes, archive: Path) -> dict[str, object]:
    if len(payload) > 8 * 1024 * 1024:
        raise RuntimeError(f"meta/info.json is unexpectedly large in {archive}")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"meta/info.json is not a JSON object in {archive}")
    if not isinstance(value.get("features"), dict):
        raise RuntimeError(f"meta/info.json has no features mapping in {archive}")
    return {
        "codebase_version": value.get("codebase_version"),
        "fps": value.get("fps"),
        "feature_count": len(value["features"]),
    }


def _inspect_tar(path: Path) -> dict[str, object]:
    info: dict[str, object] | None = None
    has_data = False
    has_visual = False
    members = 0
    with tarfile.open(path, mode="r:*") as handle:
        for member in handle:
            relative = _safe_member(member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"archive contains link/device member: {member.name}")
            if member.isfile():
                members += 1
                normalized = relative.as_posix()
                has_data = has_data or (
                    "/data/" in f"/{normalized}" and normalized.endswith(".parquet")
                )
                has_visual = has_visual or any(
                    marker in f"/{normalized}" for marker in ("/videos/", "/images/")
                )
                if normalized.endswith("/meta/info.json") or normalized == "meta/info.json":
                    reader = handle.extractfile(member)
                    if reader is None:
                        raise RuntimeError(f"cannot read meta/info.json in {path}")
                    with reader:
                        info = _inspect_info(reader.read(8 * 1024 * 1024 + 1), path)
    return _finish_archive(path, members, has_data, has_visual, info)


def _inspect_zip(path: Path) -> dict[str, object]:
    info: dict[str, object] | None = None
    has_data = False
    has_visual = False
    members = 0
    with zipfile.ZipFile(path) as handle:
        for member in handle.infolist():
            relative = _safe_member(member.filename)
            mode = (member.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise RuntimeError(f"archive contains symlink: {member.filename}")
            if member.is_dir():
                continue
            members += 1
            normalized = relative.as_posix()
            has_data = has_data or (
                "/data/" in f"/{normalized}" and normalized.endswith(".parquet")
            )
            has_visual = has_visual or any(
                marker in f"/{normalized}" for marker in ("/videos/", "/images/")
            )
            if normalized.endswith("/meta/info.json") or normalized == "meta/info.json":
                with handle.open(member, "r") as reader:
                    info = _inspect_info(reader.read(8 * 1024 * 1024 + 1), path)
    return _finish_archive(path, members, has_data, has_visual, info)


def _finish_archive(
    path: Path,
    members: int,
    has_data: bool,
    has_visual: bool,
    info: dict[str, object] | None,
) -> dict[str, object]:
    if members <= 0 or info is None or not has_data or not has_visual:
        raise RuntimeError(
            f"{path} is not a usable LeRobot archive: members={members}, "
            f"info={info is not None}, data={has_data}, visual={has_visual}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "members": members,
        "has_data": has_data,
        "has_visual": has_visual,
        "info": info,
    }


def _inspect_archive(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"archive must be a regular non-symlink file: {path}")
    if path.name.lower().endswith(".zip"):
        return _inspect_zip(path)
    return _inspect_tar(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--sample-archives-per-prefix", type=int, default=1)
    args = parser.parse_args()
    if args.sample_archives_per_prefix <= 0:
        raise ValueError("--sample-archives-per-prefix must be positive")
    root = _directory(args.snapshot_root, "AgiBotWorld2026 snapshot")
    prefixes = tuple(args.prefix) if args.prefix else DEFAULT_PREFIXES
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("--prefix values must be unique")
    evidence: dict[str, object] = {}
    for raw_prefix in prefixes:
        prefix = PurePosixPath(raw_prefix)
        if prefix.is_absolute() or not prefix.parts or ".." in prefix.parts:
            raise ValueError(f"unsafe prefix: {raw_prefix!r}")
        subset = _directory(root.joinpath(*prefix.parts), f"AgiBot prefix {raw_prefix}")
        archives = sorted(path for path in subset.rglob("*") if _is_archive(path))
        if not archives:
            raise RuntimeError(f"no archives found under {subset}")
        regular = []
        total_bytes = 0
        for archive in archives:
            if archive.is_symlink() or not archive.is_file():
                raise RuntimeError(f"archive must be regular and non-symlink: {archive}")
            regular.append(archive)
            total_bytes += archive.stat().st_size
        samples = [
            _inspect_archive(path)
            for path in regular[: args.sample_archives_per_prefix]
        ]
        evidence[raw_prefix] = {
            "archive_count": len(regular),
            "total_bytes": total_bytes,
            "sampled_archives": samples,
        }
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "passed": True,
                "snapshot_root": str(root),
                "prefixes": evidence,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
