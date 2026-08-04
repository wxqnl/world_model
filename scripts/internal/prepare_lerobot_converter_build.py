#!/usr/bin/env python3
"""为冻结的 LeRobot 0.1.0 准备可复现的 Linux 转换环境。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


UPSTREAM_PYAV_BLOCK_SHA256 = (
    "75fc1fa85a291240a975505bea6724f998409f94a2e011b11d047a1991459814"
)
OFFICIAL_AV_CP310_LINUX_X86_64_SHA256 = (
    "1d568c4d7a36df52c0774d52e6d730148775ead16daed81c10dafc2569b5a38d"
)
ORIGINAL_PYPROJECT_SHA256 = (
    "34a923b9d6739c52d63af14d20282d5cbebbc78a46a81d76600ad33ae4057d66"
)
PATCHED_PYPROJECT_SHA256 = (
    "d867205f8ec7c6f2e2049de8ecc34381809cda1c31f0df1619b97a74326e8214"
)
UPSTREAM_REQUIREMENT_PREFIX = "pyav==13.1.0 ; "
NEWLINE = bytes((10,))
CONTINUATION = bytes((92, 10))
REPLACEMENT_REQUIREMENT = (
    b"av==13.1.0 "
    + CONTINUATION
    + f"    --hash=sha256:{OFFICIAL_AV_CP310_LINUX_X86_64_SHA256}".encode()
    + NEWLINE
)
ORIGINAL_DEPENDENCY = NEWLINE + b'pyav = ">=12.0.5"' + NEWLINE
PATCHED_DEPENDENCY = NEWLINE + b'av = ">=12.0.5"' + NEWLINE


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_requirements(
    raw: bytes,
    *,
    expected_block_sha256: str = UPSTREAM_PYAV_BLOCK_SHA256,
) -> bytes:
    """把已下架的 pyav 发行物替换为官方 av wheel，其他行保持不变。"""
    lines = raw.splitlines(keepends=True)
    starts = [
        index
        for index, line in enumerate(lines)
        if line.decode("utf-8").startswith(UPSTREAM_REQUIREMENT_PREFIX)
    ]
    if len(starts) != 1:
        raise ValueError(f"pyav requirement 数量异常: {len(starts)}")
    start = starts[0]
    end = start + 1
    while end < len(lines) and lines[end].startswith(b"    --hash="):
        end += 1
    block = b"".join(lines[start:end])
    actual = sha256_bytes(block)
    if actual != expected_block_sha256:
        raise ValueError(
            "上游 pyav requirement 已变化，拒绝静默修补: "
            f"expected={expected_block_sha256} actual={actual}"
        )
    normalized = b"".join([*lines[:start], REPLACEMENT_REQUIREMENT, *lines[end:]])
    if b"pyav==" in normalized:
        raise ValueError("normalized requirements 仍包含 pyav")
    return normalized


def patch_pyproject(
    raw: bytes,
    *,
    original_sha256: str = ORIGINAL_PYPROJECT_SHA256,
    patched_sha256: str = PATCHED_PYPROJECT_SHA256,
) -> bytes:
    """只在构建副本中修正 wheel 的 Requires-Dist 名称。"""
    actual = sha256_bytes(raw)
    if actual == patched_sha256:
        return raw
    if actual != original_sha256:
        raise ValueError(
            f"LeRobot pyproject 身份不匹配: expected={original_sha256} actual={actual}"
        )
    if raw.count(ORIGINAL_DEPENDENCY) != 1:
        raise ValueError("LeRobot pyproject 的 pyav 声明不唯一")
    result = raw.replace(ORIGINAL_DEPENDENCY, PATCHED_DEPENDENCY)
    actual_patched = sha256_bytes(result)
    if actual_patched != patched_sha256:
        raise ValueError(
            "LeRobot pyproject 修补结果不匹配: "
            f"expected={patched_sha256} actual={actual_patched}"
        )
    return result


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} 必须是普通文件: {path}")
    return path


def _atomic_replace(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"拒绝覆盖不一致文件: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requirements-input", type=Path, required=True)
    parser.add_argument("--requirements-output", type=Path, required=True)
    parser.add_argument("--build-pyproject", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements_input = _regular_file(args.requirements_input, "raw requirements")
    build_pyproject = _regular_file(args.build_pyproject, "build pyproject")
    normalized = normalize_requirements(requirements_input.read_bytes())
    patched = patch_pyproject(build_pyproject.read_bytes())
    _write_once(args.requirements_output, normalized)
    if patched != build_pyproject.read_bytes():
        _atomic_replace(build_pyproject, patched)
    print(
        json.dumps(
            {
                "pass": True,
                "normalized_requirements_sha256": sha256_bytes(normalized),
                "patched_pyproject_sha256": sha256_bytes(patched),
                "av_wheel_sha256": OFFICIAL_AV_CP310_LINUX_X86_64_SHA256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
