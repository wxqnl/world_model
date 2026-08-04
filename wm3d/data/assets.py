"""Portable, immutable offline-encoder assets for WM3D."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any

from .contracts import (
    ContractError,
    canonical_sha256,
    resolve_regular_file,
    safe_relative_path,
    sha256_file,
)


ASSET_RECEIPT_SCHEMA = "wm3d_v7_encoder_assets_v1"
VGGT_SOURCE_RECEIPT_SCHEMA = "wm3d_v7_vggt_source_receipt_v1"
VGGT_SOURCE_RECEIPT_NAME = ".wm3d_v7_vggt_source_receipt.json"


def vggt_source_evidence(root: Path) -> dict[str, dict[str, int | str]]:
    """Hash every regular source file while rejecting links and special files."""
    input_root = Path(root)
    info = os.lstat(input_root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"VGGT source root is not a real directory: {input_root}")
    source_root = input_root.resolve(strict=True)
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(source_root.rglob("*")):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"VGGT source symlink is forbidden: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ContractError(f"VGGT source special file is forbidden: {path}")
        relative = path.relative_to(source_root).as_posix()
        if relative == VGGT_SOURCE_RECEIPT_NAME:
            continue
        safe_relative_path(relative)
        files[relative] = {
            "size": int(info.st_size),
            "sha256": sha256_file(path),
        }
    if not files:
        raise ContractError("VGGT source contains no files")
    return files


def vggt_source_tree_sha256(
    files: dict[str, dict[str, int | str]],
) -> str:
    return canonical_sha256({"files": files})


def verify_vggt_source(
    root: Path,
    *,
    expected_commit: str,
    expected_archive_sha256: str,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    input_root = Path(root)
    info = os.lstat(input_root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"VGGT source root is not a real directory: {input_root}")
    source_root = input_root.resolve(strict=True)
    receipt_path = resolve_regular_file(source_root, VGGT_SOURCE_RECEIPT_NAME)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != VGGT_SOURCE_RECEIPT_SCHEMA:
        raise ContractError("VGGT source receipt schema mismatch")
    content = dict(receipt)
    content_digest = content.pop("content_sha256", None)
    if canonical_sha256(content) != content_digest:
        raise ContractError("VGGT source receipt digest mismatch")
    expected = {
        "commit": expected_commit,
        "archive_sha256": expected_archive_sha256,
        "tree_sha256": expected_tree_sha256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ContractError(f"VGGT source {key} mismatch")
    files = vggt_source_evidence(source_root)
    if files != receipt.get("files"):
        raise ContractError("VGGT source file evidence mismatch")
    actual_tree = vggt_source_tree_sha256(files)
    if actual_tree != expected_tree_sha256:
        raise ContractError("VGGT source tree digest mismatch")
    return {
        "pass": True,
        "root": str(source_root),
        "commit": expected_commit,
        "archive_sha256": expected_archive_sha256,
        "tree_sha256": actual_tree,
        "files": files,
        "receipt_sha256": canonical_sha256(receipt),
    }


def load_asset_receipt(path: Path) -> dict[str, Any]:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"asset receipt is not a regular file: {input_path}")
    receipt_path = input_path.resolve(strict=True)
    value = json.loads(receipt_path.read_text(encoding="utf-8"))
    if value.get("schema") != ASSET_RECEIPT_SCHEMA:
        raise ContractError("encoder asset receipt schema mismatch")
    assets = value.get("assets")
    files = value.get("files")
    if not isinstance(assets, dict) or set(assets) != {
        "vggt_source",
        "vggt_model",
        "task_model",
    }:
        raise ContractError("encoder asset identity set mismatch")
    if not isinstance(files, dict) or not files:
        raise ContractError("encoder asset receipt contains no files")
    for name, asset in assets.items():
        if not isinstance(asset, dict):
            raise ContractError(f"invalid encoder asset record {name}")
        safe_relative_path(str(asset["path"]))
    for relative, evidence in files.items():
        safe_relative_path(str(relative))
        if (
            not isinstance(evidence, dict)
            or int(evidence.get("size", -1)) < 0
            or len(str(evidence.get("sha256", ""))) != 64
        ):
            raise ContractError(f"invalid encoder asset evidence {relative}")
    expected_content = value.get("content_sha256")
    content = dict(value)
    content.pop("content_sha256", None)
    if canonical_sha256(content) != expected_content:
        raise ContractError("encoder asset receipt content digest mismatch")
    return value


def verify_asset_bundle(
    root: Path,
    *,
    deep: bool,
) -> dict[str, Any]:
    input_root = Path(root)
    info = os.lstat(input_root)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"asset root is not a real directory: {input_root}")
    root = input_root.resolve(strict=True)
    receipt = load_asset_receipt(root / "receipt.json")
    errors: list[str] = []
    total_bytes = 0
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode):
            errors.append(f"{path.relative_to(root)}: symlink is forbidden")
        elif stat.S_ISREG(info.st_mode):
            relative = path.relative_to(root).as_posix()
            if relative != "receipt.json":
                actual_files.add(relative)
        elif not stat.S_ISDIR(info.st_mode):
            errors.append(f"{path.relative_to(root)}: special file is forbidden")
    expected_files = set(receipt["files"])
    if actual_files != expected_files:
        errors.append(
            "asset file set mismatch: "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )
    for relative, evidence in sorted(receipt["files"].items()):
        try:
            path = resolve_regular_file(root, relative)
        except Exception as exc:
            errors.append(f"{relative}: {exc}")
            continue
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            errors.append(f"{relative}: not a regular file")
            continue
        total_bytes += int(info.st_size)
        if int(info.st_size) != int(evidence["size"]):
            errors.append(f"{relative}: size mismatch")
        elif deep and sha256_file(path) != str(evidence["sha256"]):
            errors.append(f"{relative}: sha256 mismatch")
    for name, asset in receipt["assets"].items():
        asset_path = root / str(asset["path"])
        if asset_path.is_symlink() or not asset_path.is_dir():
            errors.append(f"{name}: missing real asset directory {asset_path}")
    if errors:
        raise ContractError("encoder asset verification failed:\n" + "\n".join(errors))
    return {
        "pass": True,
        "receipt": receipt,
        "receipt_sha256": canonical_sha256(receipt),
        "files": len(receipt["files"]),
        "bytes": total_bytes,
        "deep": bool(deep),
    }
