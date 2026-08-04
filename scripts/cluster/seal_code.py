#!/usr/bin/env python3
"""Seal the complete WM3D training runtime surface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

from wm3d.data.contracts import (
    FileEvidence,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
    utc_now,
)
from wm3d.training.config import CODE_RECEIPT_SCHEMA
from wm3d.training.runtime import assert_dependency_boundary


DEFAULT_PATTERNS = (
    "wm3d/__init__.py",
    "wm3d/data/*.py",
    "wm3d/encoders/*.py",
    "wm3d/models/*.py",
    "wm3d/training/*.py",
    "scripts/pipeline.py",
    "scripts/README.md",
    "scripts/assets/*",
    "scripts/cluster/*",
    "scripts/data/*",
    "scripts/slurm/*",
    "scripts/smoke/*",
    "scripts/tools/*",
    "configs/cluster/*",
    "configs/data/*",
    "configs/smoke/*",
    "configs/train/*",
    "environments/*",
    "tests/test_*.py",
    "wm3d.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include", action="append", default=[])
    return parser.parse_args()


def _files(root: Path, patterns: Iterable[str]) -> list[Path]:
    values: set[Path] = set()
    for pattern in patterns:
        matches = tuple(root.glob(pattern))
        if not matches:
            raise ValueError(f"code seal pattern matched nothing: {pattern}")
        for path in matches:
            if "__pycache__" in path.parts or path.name.endswith((".pyc", ".pyo")):
                continue
            if path.is_symlink():
                raise ValueError(f"code seal forbids symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise ValueError(f"code seal matched a special file: {path}")
            values.add(path.resolve(strict=True))
    if not values:
        raise ValueError("code seal matched no files")
    for path in values:
        if root not in path.parents:
            raise ValueError(f"code seal path escaped repo: {path}")
    return sorted(values)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve(strict=True)
    patterns = tuple(args.include) or DEFAULT_PATTERNS
    files = _files(root, patterns)
    assert_dependency_boundary(
        path for path in files if path.suffix in {".py", ".yaml", ".yml", ".json"}
    )
    relatives = [path.relative_to(root).as_posix() for path in files]
    scoped_status = _git(root, "status", "--porcelain", "--", *relatives)
    if scoped_status:
        raise ValueError(
            "refusing to seal a dirty WM3D scope:\n" + scoped_status
        )
    evidence = {
        relative: FileEvidence(
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for relative, path in zip(relatives, files, strict=True)
    }
    value = {
        "schema": CODE_RECEIPT_SCHEMA,
        "root_layout": "wm3d",
        "created_at_utc": utc_now(),
        "include_patterns": list(patterns),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "scoped_git_status": scoped_status,
        "files": {
            relative: {"size": item.size, "sha256": item.sha256}
            for relative, item in evidence.items()
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, value, exclusive=True)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(
        json.dumps(
            {
                "pass": True,
                "files": len(files),
                "receipt": str(output),
                "receipt_sha256": canonical_sha256(value),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
