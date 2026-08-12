#!/usr/bin/env python3
"""Run a pinned upstream converter and seal its LeRobot collection output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

import yaml

from wm3d_v3.data.manifest_contract import canonical_sha256, sha256_file


CONTRACT_SCHEMA = "wm3d_v8_external_converter_contract_v1"
DOWNLOAD_SCHEMA = "wm3d_v8_raw_snapshot_receipt_v1"
RECEIPT_SCHEMA = "wm3d_v8_external_conversion_receipt_v1"
COLLECTION_SCHEMA = "wm3d_v8_lerobot_collection_receipt_v1"
SHA64 = re.compile(r"[0-9a-f]{64}")


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file: {path}")
    return path.resolve(strict=True)


def _root(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def _executable(path: Path, label: str) -> Path:
    safe = path.resolve(strict=True)
    if not safe.is_file() or not os.access(safe, os.X_OK):
        raise RuntimeError(
            f"{label} must resolve to an executable regular file: {path}"
        )
    return safe


def _publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical receipt: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _download(path: Path, source: str) -> tuple[Path, dict]:
    safe = _regular(path, "download receipt")
    value = json.loads(safe.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema") != DOWNLOAD_SCHEMA
        or value.get("source") != source
        or int(value.get("snapshot_file_count", 0)) <= 0
        or int(value.get("snapshot_total_bytes", 0)) <= 0
    ):
        raise RuntimeError(f"download receipt does not bind source {source!r}")
    return safe, value


def _contract(path: Path) -> tuple[Path, dict]:
    safe = _regular(path, "converter contract")
    value = yaml.safe_load(safe.read_text(encoding="utf-8"))
    required = {
        "schema", "name", "input_source", "converter_source", "converter_relative_path",
        "environment_receipt_sha256", "required_bindings", "argv", "output_kind",
        "lerobot_root_glob",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("external converter contract fields mismatch")
    if value["schema"] != CONTRACT_SCHEMA:
        raise RuntimeError(f"converter contract schema must be {CONTRACT_SCHEMA}")
    if (
        not isinstance(value["required_bindings"], list)
        or any(
            re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(item)) is None
            for item in value["required_bindings"]
        )
        or len(value["required_bindings"]) != len(set(value["required_bindings"]))
        or not isinstance(value["argv"], list)
        or not value["argv"]
        or any(not isinstance(item, str) or not item for item in value["argv"])
        or value["output_kind"] != "lerobot_collection"
        or not str(value["lerobot_root_glob"]).endswith("meta/info.json")
        or SHA64.fullmatch(str(value["environment_receipt_sha256"])) is None
    ):
        raise RuntimeError("external converter contract is incomplete")
    return safe, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--input-download-receipt", type=Path, required=True)
    parser.add_argument("--converter-root", type=Path, required=True)
    parser.add_argument("--converter-download-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--binding", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument(
        "--binding-file",
        action="append",
        default=[],
        metavar="NAME=FILE",
        help="逐行执行一个绑定；当前只允许一个数组绑定。",
    )
    args = parser.parse_args()

    contract_path, contract = _contract(args.contract)
    input_root = _root(args.input_root, "converter input root")
    converter_root = _root(args.converter_root, "converter snapshot root")
    input_receipt, input_value = _download(
        args.input_download_receipt, str(contract["input_source"])
    )
    converter_receipt, converter_value = _download(
        args.converter_download_receipt, str(contract["converter_source"])
    )
    if Path(str(input_value["snapshot_path"])).resolve(strict=True) != input_root:
        raise RuntimeError("input download receipt snapshot path mismatch")
    if Path(str(converter_value["snapshot_path"])).resolve(strict=True) != converter_root:
        raise RuntimeError("converter download receipt snapshot path mismatch")
    relative_converter = Path(str(contract["converter_relative_path"]))
    if relative_converter.is_absolute() or ".." in relative_converter.parts:
        raise RuntimeError("converter_relative_path must be safe and relative")
    converter = _regular(converter_root / relative_converter, "upstream converter")
    environment = _regular(args.environment_receipt, "converter environment receipt")
    if sha256_file(environment) != contract["environment_receipt_sha256"]:
        raise RuntimeError("converter environment receipt SHA mismatch")
    python = _executable(args.python_bin, "converter Python")

    output = args.output_root.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"conversion output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.convert.{os.getpid()}.{uuid.uuid4().hex}"
    bindings: dict[str, str] = {}
    for raw in args.binding:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value or name in bindings:
            raise RuntimeError(f"invalid/duplicate --binding {raw!r}")
        bindings[name] = value
    binding_files: dict[str, Path] = {}
    for raw in args.binding_file:
        name, separator, value = raw.partition("=")
        if not separator or not name or not value or name in binding_files:
            raise RuntimeError(f"invalid/duplicate --binding-file {raw!r}")
        binding_files[name] = _regular(Path(value), f"binding file {name}")
    if len(binding_files) > 1 or set(bindings) & set(binding_files):
        raise RuntimeError("only one disjoint --binding-file is supported")
    if set(bindings) | set(binding_files) != set(map(str, contract["required_bindings"])):
        raise RuntimeError(
            "converter binding set mismatch: "
            f"required={contract['required_bindings']} "
            f"actual={sorted(set(bindings) | set(binding_files))}"
        )
    series: list[dict[str, str]] = [dict(bindings)]
    binding_file_evidence: dict[str, dict[str, object]] = {}
    if binding_files:
        name, path = next(iter(binding_files.items()))
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not values or len(values) != len(set(values)):
            raise RuntimeError("binding file is empty or contains duplicate values")
        series = [{**bindings, name: value} for value in values]
        binding_file_evidence[name] = {
            "path": str(path), "sha256": sha256_file(path), "count": len(values)
        }
    temporary.mkdir(mode=0o750)
    commands: list[list[str]] = []
    for ordinal, one_binding in enumerate(series):
        job_output = temporary / "jobs" / f"{ordinal:08d}"
        placeholders = {
            "python": str(python),
            "converter": str(converter),
            "input_root": str(input_root),
            "output_root": str(job_output),
            **one_binding,
        }
        allowed = {"{" + key + "}" for key in placeholders}
        argv = []
        for item in contract["argv"]:
            observed = {part for part in allowed if part in item}
            remainder = item
            for token in observed:
                remainder = remainder.replace(token, "")
            if "{" in remainder or "}" in remainder:
                raise RuntimeError(f"unknown converter argv placeholder: {item!r}")
            argv.append(item.format(**placeholders))
        if not argv or Path(argv[0]).resolve(strict=True) != python:
            raise RuntimeError("converter argv must execute the bound --python-bin")
        subprocess.run(argv, check=True)
        commands.append(
            [item.replace(str(temporary), str(output), 1) for item in argv]
        )
    roots = sorted(
        {
            _root(path.parent.parent, "converted LeRobot root")
            for path in temporary.glob(str(contract["lerobot_root_glob"]))
            if path.is_file() and not path.is_symlink()
        }
    )
    if not roots:
        raise RuntimeError("upstream converter produced no LeRobot meta/info.json")
    relative_roots = [root.relative_to(temporary).as_posix() for root in roots]
    stable = {
        "schema": RECEIPT_SCHEMA,
        "converter_contract_path": str(contract_path),
        "converter_contract_sha256": sha256_file(contract_path),
        "input_download_receipt_path": str(input_receipt),
        "input_download_receipt_sha256": sha256_file(input_receipt),
        "converter_download_receipt_path": str(converter_receipt),
        "converter_download_receipt_sha256": sha256_file(converter_receipt),
        "converter_path": str(converter),
        "converter_sha256": sha256_file(converter),
        "environment_receipt_path": str(environment),
        "environment_receipt_sha256": sha256_file(environment),
        "python_path": str(python),
        "commands": commands,
        "bindings": dict(sorted(bindings.items())),
        "binding_files": binding_file_evidence,
        "lerobot_roots": relative_roots,
    }
    _publish(temporary / "conversion_receipt.json", stable)
    os.replace(temporary, output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    roots_final = [str((output / relative).resolve(strict=True)) for relative in relative_roots]
    collection = {
        "schema": COLLECTION_SCHEMA,
        "source": str(contract["input_source"]),
        "source_prefix": "external_converter",
        "download_receipt_path": str(input_receipt),
        "download_receipt_sha256": sha256_file(input_receipt),
        "converter_contract_sha256": sha256_file(contract_path),
        "conversion_receipt_sha256": sha256_file(output / "conversion_receipt.json"),
        "archive_count": 0,
        "lerobot_root_count": len(roots_final),
        "lerobot_roots": roots_final,
        "archive_receipts_content_sha256": canonical_sha256(stable),
    }
    _publish(output / "collection_receipt.json", collection)
    print(
        json.dumps(
            {**collection, "commands": [shlex.join(command) for command in commands]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
