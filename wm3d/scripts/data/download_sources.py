#!/usr/bin/env python3
"""Resume immutable raw snapshots from a sealed, transport-neutral source lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import uuid
from fnmatch import fnmatch
from pathlib import PurePosixPath

from huggingface_hub import snapshot_download
import yaml

from wm3d.data.manifest_contract import sha256_file


LOCK_SCHEMA = "wm3d_v8_raw_source_lock_v1"
RECEIPT_SCHEMA = "wm3d_v8_raw_snapshot_receipt_v1"
LOCK_RECEIPT_SCHEMA = "wm3d_v8_raw_source_lock_receipt_v1"
FILE_LIST_SCHEMA = "wm3d_v8_raw_source_file_list_v1"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _token(path: Path | None) -> str | None:
    if path is None:
        return None
    path = path.resolve(strict=True)
    info = path.stat()
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError("token file must not be group/world readable")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("token file is empty")
    return value


def _publish(path: Path, value: object) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"non-identical receipt exists: {path}")
        return
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def _file_list_evidence(lock: Path, name: str, revision: str) -> tuple[Path, str, int]:
    receipt_path = lock.with_suffix(lock.suffix + ".receipt.json")
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError(f"source lock receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != LOCK_RECEIPT_SCHEMA
        or receipt.get("lock_sha256") != sha256_file(lock)
    ):
        raise RuntimeError("source lock receipt schema/SHA mismatch")
    path = Path(str(receipt["file_list_path_by_source"][name])).resolve(strict=True)
    digest = str(receipt["file_list_sha256_by_source"][name])
    if sha256_file(path) != digest:
        raise RuntimeError(f"{name}: upstream file-list SHA drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != FILE_LIST_SCHEMA
        or value.get("source") != name
        or value.get("revision") != revision
        or int(value.get("file_count", 0)) != len(value.get("files", []))
        or int(value.get("file_count", 0)) <= 0
    ):
        raise RuntimeError(f"{name}: upstream file-list identity mismatch")
    return path, digest, int(value["file_count"])


def _matches_any(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        normalized = str(pattern)
        if fnmatch(path, normalized):
            return True
        # Python's fnmatch treats ``prefix/**`` as requiring a slash after the
        # prefix.  Accept the directory marker itself for closure accounting.
        if normalized.endswith("/**") and path == normalized[:-3].rstrip("/"):
            return True
    return False


def _verify_snapshot_closure(snapshot: Path, selected_files: list[str]) -> dict[str, int]:
    """Verify the frozen path closure without re-hashing multi-TB raw payloads."""
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError(f"snapshot is not a real directory: {snapshot}")
    root = snapshot.resolve(strict=True)
    total_bytes = 0
    for raw_relative in selected_files:
        relative = PurePosixPath(raw_relative)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"unsafe upstream path in frozen file list: {raw_relative!r}")
        target = root.joinpath(*relative.parts)
        try:
            info = target.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"snapshot closure is missing {raw_relative!r}") from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(
                f"snapshot closure member must be a regular non-symlink file: {target}"
            )
        total_bytes += int(info.st_size)
    return {"file_count": len(selected_files), "total_bytes": total_bytes}


def main() -> None:
    args = parse_args()
    lock = args.lock.resolve(strict=True)
    value = yaml.safe_load(lock.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "sources"}:
        raise RuntimeError("raw source lock fields mismatch")
    if value["schema"] != LOCK_SCHEMA:
        raise RuntimeError(f"raw source lock schema must be {LOCK_SCHEMA}")
    selected = set(args.source)
    names = {str(item["name"]) for item in value["sources"]}
    unknown = selected - names
    if unknown:
        raise RuntimeError(f"unknown selected sources: {sorted(unknown)}")
    root = args.raw_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    token = _token(args.token_file)
    for source in value["sources"]:
        required = {
            "name",
            "transport",
            "repo_id",
            "revision",
            "include",
            "destination",
            "gated",
        }
        if set(source) != required:
            raise RuntimeError(f"source lock fields mismatch for {source.get('name')}")
        name = str(source["name"])
        if selected and name not in selected:
            continue
        if source["transport"] != "huggingface_dataset":
            raise RuntimeError(f"unsupported transport {source['transport']!r}")
        revision = str(source["revision"])
        if COMMIT_RE.fullmatch(revision) is None:
            raise RuntimeError(f"{name}: revision must be a 40-char commit SHA")
        if bool(source["gated"]) and token is None:
            raise RuntimeError(f"{name}: gated source requires --token-file")
        file_list_path, file_list_sha, file_count = _file_list_evidence(
            lock, name, revision
        )
        file_list_value = json.loads(file_list_path.read_text(encoding="utf-8"))
        include = [str(item) for item in source["include"]]
        selected_files = [
            str(path)
            for path in file_list_value["files"]
            if _matches_any(str(path), include)
        ]
        if not selected_files:
            raise RuntimeError(
                f"{name}: include patterns select no files from the frozen revision"
            )
        destination = root / str(source["destination"])
        receipt_path = root / "receipts" / f"{name}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            snapshot_path = Path(str(receipt.get("snapshot_path", "")))
            if (
                receipt.get("schema") == RECEIPT_SCHEMA
                and receipt.get("lock_sha256") == sha256_file(lock)
                and receipt.get("revision") == revision
                and receipt.get("upstream_file_list_sha256") == file_list_sha
                and receipt.get("selected_upstream_file_count") == len(selected_files)
            ):
                closure = _verify_snapshot_closure(snapshot_path, selected_files)
                if receipt.get("snapshot_file_count") != closure["file_count"]:
                    raise RuntimeError(f"{name}: snapshot receipt file-count drift")
                if receipt.get("snapshot_total_bytes") != closure["total_bytes"]:
                    raise RuntimeError(f"{name}: snapshot receipt byte-count drift")
                print(
                    json.dumps(
                        {"source": name, "status": "verified-skip", **closure},
                        sort_keys=True,
                    )
                )
                continue
            raise RuntimeError(f"{name}: existing receipt does not match source lock")
        snapshot = snapshot_download(
            repo_id=str(source["repo_id"]),
            repo_type="dataset",
            revision=revision,
            local_dir=destination,
            allow_patterns=include,
            token=token,
            max_workers=int(args.max_workers),
        )
        snapshot_path = Path(snapshot).resolve(strict=True)
        closure = _verify_snapshot_closure(snapshot_path, selected_files)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "source": name,
            "repo_id": str(source["repo_id"]),
            "revision": revision,
            "snapshot_path": str(snapshot_path),
            "lock_sha256": sha256_file(lock),
            "upstream_file_list_path": str(file_list_path),
            "upstream_file_list_sha256": file_list_sha,
            "upstream_file_count": file_count,
            "selected_upstream_file_count": len(selected_files),
            "snapshot_file_count": closure["file_count"],
            "snapshot_total_bytes": closure["total_bytes"],
        }
        _publish(receipt_path, receipt)
        print(json.dumps({"source": name, "status": "downloaded", "path": snapshot}))


if __name__ == "__main__":
    main()
