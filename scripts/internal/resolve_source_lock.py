#!/usr/bin/env python3
"""Resolve mutable Hugging Face names once and publish an immutable raw lock."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any

from huggingface_hub import HfApi
import yaml


REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CONFIRM = "YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--confirm-licenses", required=True)
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="SOURCE=AUTO_OR_40_HEX",
    )
    return parser.parse_args()


def _regular(path: Path, label: str) -> Path:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} 必须是普通文件：{path}")
    return path.resolve(strict=True)


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(_regular(path, "source lock").read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != "wm3d_v7_raw_sources_lock_v1"
    ):
        raise ValueError("raw-source lock schema 不匹配")
    sources = value.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("raw-source lock 没有 sources")
    return value


def _validate_frozen(value: dict[str, Any]) -> None:
    for name, source in value["sources"].items():
        revision = str(source.get("revision", ""))
        if not REVISION_RE.fullmatch(revision):
            raise ValueError(f"{name} revision 不是 40 位 commit：{revision}")


def _overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        name, separator, revision = item.partition("=")
        if not separator or not name or not revision or name in result:
            raise ValueError(f"非法或重复 --revision：{item}")
        if revision != "AUTO" and not REVISION_RE.fullmatch(revision):
            raise ValueError(f"revision 必须是 AUTO 或 40 位小写 commit：{item}")
        result[name] = revision
    return result


def _token(path: Path) -> str:
    safe = _regular(path, "HF token file")
    if stat.S_IMODE(safe.stat().st_mode) & 0o077:
        raise PermissionError("HF token file 权限必须不高于 0600")
    value = safe.read_text(encoding="utf-8").strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError("HF token file 必须只含一行非空 token")
    return value


def _atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"拒绝覆盖 raw lock：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    if args.confirm_licenses != CONFIRM:
        raise PermissionError("先接受 AgiBot Alpha/Beta 上游许可，再提供明确确认串")
    if args.output.exists():
        value = _load(args.output)
        _validate_frozen(value)
        print(
            yaml.safe_dump(
                {
                    "pass": True,
                    "status": "already_complete",
                    "output": str(args.output),
                },
                sort_keys=True,
            ).strip()
        )
        return

    value = _load(args.template)
    overrides = _overrides(args.revision)
    unknown = set(overrides).difference(value["sources"])
    if unknown:
        raise ValueError(f"revision override 包含未知 source：{sorted(unknown)}")
    token = _token(args.token_file)
    api = HfApi(token=token)
    for name, source in value["sources"].items():
        requested = overrides.get(name, "AUTO")
        if requested == "AUTO":
            info = api.repo_info(
                repo_id=str(source["repo_id"]),
                repo_type=str(source["repo_type"]),
                revision="main",
            )
            revision = str(info.sha)
        else:
            revision = requested
            info = api.repo_info(
                repo_id=str(source["repo_id"]),
                repo_type=str(source["repo_type"]),
                revision=revision,
            )
            if str(info.sha) != revision:
                raise ValueError(f"{name} 请求 revision 与上游解析结果不一致")
        if not REVISION_RE.fullmatch(revision):
            raise ValueError(f"{name} 上游未返回 40 位 commit：{revision}")
        source["revision"] = revision
    value["resolved_at_utc"] = datetime.now(timezone.utc).isoformat()
    value["license_confirmation"] = {
        "operator_confirmed_upstream_access": True,
        "confirmation_literal": CONFIRM,
    }
    _validate_frozen(value)
    _atomic_yaml(args.output, value)
    print(
        yaml.safe_dump(
            {"pass": True, "status": "created", "output": str(args.output)},
            sort_keys=True,
        ).strip()
    )


if __name__ == "__main__":
    main()
