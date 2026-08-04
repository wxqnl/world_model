#!/usr/bin/env python3
"""生成或复核 AgiBot dataset-v2 转换 venv 的运行时 receipt。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping


CONTRACT_SCHEMA = "wm3d_v7_agibot_converter_environment_contract_v1"
RECEIPT_SCHEMA = "wm3d_v7_agibot_converter_environment_receipt_v1"
CONTRACT_BUNDLE_NAME = "environment_contract.json"
REVISION_BUNDLE_NAME = "LEROBOT_REVISION"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--revision-file", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--receipt", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} 必须是普通文件: {path}")
    return path.resolve(strict=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    safe = _regular_file(path, label)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON object")
    return value


def _contract(path: Path) -> tuple[Path, dict[str, Any]]:
    safe = _regular_file(path, "converter environment contract")
    value = _load_json(safe, "converter environment contract")
    lerobot = value.get("lerobot")
    packages = value.get("packages")
    imports = value.get("required_imports")
    commands = value.get("required_commands")
    if (
        value.get("schema") != CONTRACT_SCHEMA
        or value.get("python_major_minor") != "3.10"
        or not isinstance(lerobot, dict)
        or len(str(lerobot.get("revision", ""))) != 40
        or not isinstance(packages, dict)
        or not all(
            isinstance(name, str) and name and isinstance(version, str) and version
            for name, version in packages.items()
        )
        or not isinstance(imports, list)
        or not all(isinstance(name, str) and name for name in imports)
        or not isinstance(commands, list)
        or not all(isinstance(name, str) and name for name in commands)
    ):
        raise ValueError("converter environment contract 字段不完整")
    return safe, value


def current_environment(
    *,
    contract_path: Path,
    revision_file: Path,
) -> dict[str, Any]:
    contract_path, contract = _contract(contract_path)
    revision_file = _regular_file(revision_file, "LeRobot revision file")
    if contract_path.name != CONTRACT_BUNDLE_NAME:
        raise ValueError(f"converter contract 文件名必须是 {CONTRACT_BUNDLE_NAME}")
    if revision_file.name != REVISION_BUNDLE_NAME:
        raise ValueError(f"LeRobot revision 文件名必须是 {REVISION_BUNDLE_NAME}")
    python_version = platform.python_version()
    python_major_minor = ".".join(python_version.split(".")[:2])
    if python_major_minor != contract["python_major_minor"]:
        raise ValueError(
            "Python 版本不匹配: "
            f"expected={contract['python_major_minor']}.x actual={python_version}"
        )
    expected_revision = str(contract["lerobot"]["revision"])
    actual_revision = revision_file.read_text(encoding="utf-8").strip()
    if actual_revision != expected_revision:
        raise ValueError(
            f"LeRobot revision 不匹配: expected={expected_revision} actual={actual_revision}"
        )

    actual_packages: dict[str, str] = {}
    for name, expected in sorted(contract["packages"].items()):
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(
                f"package 版本不匹配: {name} expected={expected} actual={actual}"
            )
        actual_packages[name] = actual

    imported = []
    for name in contract["required_imports"]:
        importlib.import_module(name)
        imported.append(name)

    command_versions: dict[str, str] = {}
    for name in contract["required_commands"]:
        result = subprocess.run(
            [name, "-version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        first_line = result.stdout.splitlines()[0].strip()
        if not first_line:
            raise ValueError(f"{name} -version 没有输出")
        command_versions[name] = first_line

    return {
        "schema": RECEIPT_SCHEMA,
        "pass": True,
        "environment_contract": CONTRACT_BUNDLE_NAME,
        "environment_contract_sha256": sha256_file(contract_path),
        "python_executable": os.path.abspath(sys.executable),
        "python_version": python_version,
        "lerobot_revision": actual_revision,
        "lerobot_revision_file": REVISION_BUNDLE_NAME,
        "lerobot_revision_file_sha256": sha256_file(revision_file),
        "packages": actual_packages,
        "imports": imported,
        "commands": command_versions,
    }


def _stable_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "schema",
            "pass",
            "environment_contract",
            "environment_contract_sha256",
            "python_executable",
            "python_version",
            "lerobot_revision",
            "lerobot_revision_file",
            "lerobot_revision_file_sha256",
            "packages",
            "imports",
            "commands",
        )
    }


def validate_receipt(
    receipt_path: Path,
    *,
    check_current: bool = True,
) -> dict[str, Any]:
    receipt_path = _regular_file(receipt_path, "converter environment receipt")
    value = _load_json(receipt_path, "converter environment receipt")
    contract_relative = Path(str(value.get("environment_contract", "")))
    revision_relative = Path(str(value.get("lerobot_revision_file", "")))
    if (
        value.get("schema") != RECEIPT_SCHEMA
        or value.get("pass") is not True
        or contract_relative != Path(CONTRACT_BUNDLE_NAME)
        or revision_relative != Path(REVISION_BUNDLE_NAME)
    ):
        raise ValueError("converter environment receipt 身份或路径布局不匹配")
    contract = receipt_path.parent / contract_relative
    revision_file = receipt_path.parent / revision_relative
    contract, contract_value = _contract(contract)
    revision_file = _regular_file(revision_file, "LeRobot revision file")
    revision = revision_file.read_text(encoding="utf-8").strip()
    packages = value.get("packages")
    imports = value.get("imports")
    commands = value.get("commands")
    if (
        value.get("environment_contract_sha256") != sha256_file(contract)
        or value.get("lerobot_revision_file_sha256") != sha256_file(revision_file)
        or ".".join(str(value.get("python_version", "")).split(".")[:2])
        != contract_value["python_major_minor"]
        or value.get("lerobot_revision") != revision
        or revision != contract_value["lerobot"]["revision"]
        or packages != contract_value["packages"]
        or imports != contract_value["required_imports"]
        or not isinstance(commands, dict)
        or set(commands) != set(contract_value["required_commands"])
        or not all(
            isinstance(command, str) and command for command in commands.values()
        )
        or not isinstance(value.get("python_executable"), str)
        or not value["python_executable"]
    ):
        raise ValueError("converter environment receipt 内容与绑定文件不匹配")
    if check_current:
        current = current_environment(
            contract_path=contract,
            revision_file=revision_file,
        )
        if _stable_receipt(value) != _stable_receipt(current):
            raise ValueError("converter environment receipt 与当前运行时不匹配")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
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
    if args.receipt is not None:
        value = validate_receipt(args.receipt, check_current=True)
        print(
            json.dumps(
                {
                    "pass": True,
                    "receipt": str(args.receipt),
                    "receipt_sha256": sha256_file(args.receipt),
                    "lerobot_revision": value["lerobot_revision"],
                },
                sort_keys=True,
            )
        )
        return
    value = current_environment(
        contract_path=args.contract,
        revision_file=args.revision_file,
    )
    value["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write(args.output, value)
    print(
        json.dumps(
            {
                "pass": True,
                "receipt": str(args.output),
                "receipt_sha256": sha256_file(args.output),
                "lerobot_revision": value["lerobot_revision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
