#!/usr/bin/env python3
"""Resolve mutable Hugging Face refs once and publish an immutable WM3D lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import uuid
from typing import Any

from huggingface_hub import HfApi
import yaml

from wm3d.data.manifest_contract import sha256_file


LOCK_SCHEMA = "wm3d_v8_raw_source_lock_v1"
RECEIPT_SCHEMA = "wm3d_v8_raw_source_lock_receipt_v1"
FILE_LIST_SCHEMA = "wm3d_v8_raw_source_file_list_v1"
REVISION_RE = re.compile(r"[0-9a-f]{40}")
LICENSE_CONFIRMATION = "YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES"


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _token(path: Path | None, *, required: bool) -> str | None:
    if path is None:
        if required:
            raise PermissionError("a gated source requires --token-file")
        return None
    safe = _regular(path, "token file")
    if stat.S_IMODE(safe.stat().st_mode) & 0o077:
        raise PermissionError("token file permissions must be 0600 or stricter")
    value = safe.read_text(encoding="utf-8").strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError("token file must contain exactly one non-empty token")
    return value


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(_regular(path, "source lock template").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema", "sources"}:
        raise ValueError("source lock root fields mismatch")
    if value["schema"] != LOCK_SCHEMA:
        raise ValueError(f"source lock schema must be {LOCK_SCHEMA}")
    if not isinstance(value["sources"], list) or not value["sources"]:
        raise ValueError("source lock must contain a non-empty source list")
    names: set[str] = set()
    required = {
        "name", "transport", "repo_id", "revision", "include", "destination", "gated"
    }
    for source in value["sources"]:
        if not isinstance(source, dict) or set(source) != required:
            raise ValueError("source lock entry fields mismatch")
        name = str(source["name"])
        if not name or name in names:
            raise ValueError(f"duplicate/empty source name {name!r}")
        names.add(name)
        if source["transport"] != "huggingface_dataset":
            raise ValueError(f"unsupported source transport {source['transport']!r}")
        if not isinstance(source["include"], list) or not source["include"]:
            raise ValueError(f"{name}: include patterns must be non-empty")
    return value


def _overrides(values: list[str], names: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        name, separator, revision = raw.partition("=")
        revision = revision.lower()
        if not separator or name not in names or name in result:
            raise ValueError(f"invalid/duplicate --revision {raw!r}")
        if revision != "auto" and REVISION_RE.fullmatch(revision) is None:
            raise ValueError("revision override must be AUTO or a 40-char lowercase commit")
        result[name] = revision
    return result


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical lock artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--confirm-licenses", required=True)
    parser.add_argument("--revision", action="append", default=[], metavar="SOURCE=AUTO_OR_SHA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.confirm_licenses != LICENSE_CONFIRMATION:
        raise PermissionError(
            "accept every selected upstream license first, then pass the exact confirmation literal"
        )
    template = _regular(args.template, "source lock template")
    value = _load(template)
    names = {str(source["name"]) for source in value["sources"]}
    overrides = _overrides(args.revision, names)
    token = _token(
        args.token_file,
        required=any(bool(source["gated"]) for source in value["sources"]),
    )
    api = HfApi(token=token)
    resolved: dict[str, str] = {}
    file_lists: dict[str, dict[str, Any]] = {}
    for source in value["sources"]:
        name = str(source["name"])
        requested = overrides.get(name, "auto")
        info = api.repo_info(
            repo_id=str(source["repo_id"]),
            repo_type="dataset",
            revision="main" if requested == "auto" else requested,
        )
        revision = str(info.sha)
        if REVISION_RE.fullmatch(revision) is None:
            raise RuntimeError(f"{name}: upstream did not resolve to a 40-char commit")
        if requested != "auto" and revision != requested:
            raise RuntimeError(f"{name}: requested revision resolved to a different commit")
        source["revision"] = revision
        resolved[name] = revision
        files = sorted(
            api.list_repo_files(
                repo_id=str(source["repo_id"]),
                repo_type="dataset",
                revision=revision,
            )
        )
        if not files:
            raise RuntimeError(f"{name}: upstream revision contains no files")
        file_lists[name] = {
            "schema": FILE_LIST_SCHEMA,
            "source": name,
            "repo_id": str(source["repo_id"]),
            "revision": revision,
            "file_count": len(files),
            "files": files,
        }
    payload = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    _publish(args.output, payload)
    output = args.output.absolute().resolve(strict=True)
    file_list_sha_by_source: dict[str, str] = {}
    file_list_path_by_source: dict[str, str] = {}
    file_list_root = output.parent / f"{output.name}.file_lists"
    for name, file_list in file_lists.items():
        file_list_path = file_list_root / f"{name}.json"
        _publish(
            file_list_path,
            (json.dumps(file_list, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        safe = file_list_path.resolve(strict=True)
        file_list_path_by_source[name] = str(safe)
        file_list_sha_by_source[name] = sha256_file(safe)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "template_path": str(template),
        "template_sha256": sha256_file(template),
        "lock_path": str(output),
        "lock_sha256": sha256_file(output),
        "resolved_revision_by_source": resolved,
        "file_list_path_by_source": file_list_path_by_source,
        "file_list_sha256_by_source": file_list_sha_by_source,
        "operator_confirmed_upstream_licenses": True,
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    _publish(
        receipt_path,
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
