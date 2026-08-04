#!/usr/bin/env python3
"""按不可变 Hugging Face revision 下载 WM3D 原始数据快照。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from huggingface_hub import HfApi, snapshot_download
import yaml

from wm3d.data.contracts import (
    atomic_write_json,
    resolve_real_directory,
    resolve_regular_file,
    safe_relative_path,
    sha256_file,
    utc_now,
)


LOCK_SCHEMA = "wm3d_v7_raw_sources_lock_v1"
RECEIPT_SCHEMA = "wm3d_v7_raw_download_receipt_v1"
REVISION_RE = re.compile(r"[0-9a-f]{40}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="lock 中的 source 名；可重复。不填时处理全部 source。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="允许 huggingface_hub 续传无最终 receipt 的已有目录。",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_lock(path: Path) -> dict[str, Any]:
    safe = resolve_regular_file(path.parent, path.name)
    value = yaml.safe_load(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != LOCK_SCHEMA:
        raise ValueError(f"raw source lock schema 必须是 {LOCK_SCHEMA}")
    sources = value.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("raw source lock 没有 sources")
    for name, raw in sources.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{name}: source 配置不是对象")
        revision = str(raw.get("revision", ""))
        if REVISION_RE.fullmatch(revision) is None:
            raise ValueError(f"{name}: revision 必须是 40 位小写提交 SHA")
        repo_id = str(raw.get("repo_id", ""))
        if repo_id.count("/") != 1:
            raise ValueError(f"{name}: repo_id 非法: {repo_id!r}")
        target = str(raw.get("target_subdir", ""))
        safe_relative_path(target)
        if Path(target).parent != Path("."):
            raise ValueError(f"{name}: target_subdir 必须是 raw-root 的直接子目录")
        if raw.get("repo_type") != "dataset":
            raise ValueError(f"{name}: 本工具只允许 dataset repo_type")
        for key in ("allow_patterns", "ignore_patterns"):
            patterns = raw.get(key, [])
            if not isinstance(patterns, list) or not all(
                isinstance(item, str) and item for item in patterns
            ):
                raise ValueError(f"{name}: {key} 必须是非空字符串列表")
    return value


def _payload_inventory(
    root: Path,
    *,
    hash_content: bool = False,
) -> tuple[int, int, dict[str, str]]:
    files = 0
    total_bytes = 0
    payload_sha256: dict[str, str] = {}
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            if path.is_symlink():
                raise ValueError(f"下载快照含符号链接，拒绝发布: {path}")
        for name in filenames:
            path = directory_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"下载快照含非普通文件，拒绝发布: {path}")
            if name == ".wm3d_v7_download_receipt.json":
                continue
            files += 1
            total_bytes += path.stat().st_size
            if hash_content:
                payload_sha256[path.relative_to(root).as_posix()] = sha256_file(path)
    return files, total_bytes, payload_sha256


def _receipt_matches(
    path: Path,
    name: str,
    source: Mapping[str, Any],
    target: Path,
) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    identity_matches = (
        value.get("schema") == RECEIPT_SCHEMA
        and value.get("source") == name
        and value.get("repo_id") == source["repo_id"]
        and value.get("revision") == source["revision"]
        and value.get("resolved_revision") == source["revision"]
        and value.get("target") == str(target)
        and value.get("allow_patterns") == source.get("allow_patterns", [])
        and value.get("ignore_patterns") == source.get("ignore_patterns", [])
        and value.get("complete") is True
    )
    if not identity_matches:
        return False
    files, total_bytes, payload_sha256 = _payload_inventory(
        target,
        hash_content=source.get("materialization") == "vendor_tool_bundle",
    )
    return (
        value.get("payload_files") == files
        and value.get("payload_bytes") == total_bytes
        and value.get("payload_sha256") == payload_sha256
    )


def main() -> None:
    args = parse_args()
    lock = _load_lock(args.lock)
    raw_root = resolve_real_directory(args.raw_root, "raw snapshot root")
    sources: dict[str, Mapping[str, Any]] = lock["sources"]
    selected = tuple(args.source) if args.source else tuple(sources)
    unknown = sorted(set(selected).difference(sources))
    if unknown:
        raise ValueError(f"lock 中不存在 source: {unknown}")
    if len(selected) != len(set(selected)):
        raise ValueError("--source 重复")

    token = os.environ.get("HF_TOKEN")
    results = []
    for name in selected:
        source = sources[name]
        if bool(source.get("gated")) and not token and not args.dry_run:
            raise ValueError(f"{name}: gated 数据必须由 secret manager 注入 HF_TOKEN")
        target = raw_root / str(source["target_subdir"])
        receipt_path = target / ".wm3d_v7_download_receipt.json"
        if target.exists() or target.is_symlink():
            target = resolve_real_directory(target, f"{name} download target")
            if _receipt_matches(receipt_path, name, source, target):
                results.append({"source": name, "status": "already_complete"})
                continue
            if not args.resume:
                raise FileExistsError(
                    f"{name}: 目标已存在但无匹配完成 receipt；显式加 --resume 才能续传"
                )
        elif not args.dry_run:
            target.mkdir(mode=0o750)
        if args.dry_run:
            results.append(
                {
                    "source": name,
                    "status": "dry_run",
                    "repo_id": source["repo_id"],
                    "revision": source["revision"],
                    "target": str(target),
                }
            )
            continue

        api = HfApi(token=token)
        resolved = api.repo_info(
            repo_id=str(source["repo_id"]),
            repo_type="dataset",
            revision=str(source["revision"]),
        ).sha
        if resolved != source["revision"]:
            raise ValueError(
                f"{name}: 远端解析 SHA {resolved} 与 lock {source['revision']} 不一致"
            )
        snapshot_download(
            repo_id=str(source["repo_id"]),
            repo_type="dataset",
            revision=str(source["revision"]),
            local_dir=target,
            token=token,
            allow_patterns=source.get("allow_patterns") or None,
            ignore_patterns=source.get("ignore_patterns") or None,
        )
        files, total_bytes, payload_sha256 = _payload_inventory(
            target,
            hash_content=source.get("materialization") == "vendor_tool_bundle",
        )
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "complete": True,
            "source": name,
            "repo_id": source["repo_id"],
            "repo_type": "dataset",
            "revision": source["revision"],
            "resolved_revision": resolved,
            "target": str(target),
            "allow_patterns": source.get("allow_patterns", []),
            "ignore_patterns": source.get("ignore_patterns", []),
            "payload_files": files,
            "payload_bytes": total_bytes,
            "payload_sha256": payload_sha256,
            "completed_at_utc": utc_now(),
        }
        atomic_write_json(receipt_path, receipt, exclusive=True)
        results.append(
            {
                "source": name,
                "status": "downloaded",
                "files": files,
                "bytes": total_bytes,
                "receipt": str(receipt_path),
            }
        )
    print(json.dumps({"pass": True, "results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
