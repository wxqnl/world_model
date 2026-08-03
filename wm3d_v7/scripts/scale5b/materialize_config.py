#!/usr/bin/env python3
"""Bind a V7 native-5B template to one sealed dataset/code/topology."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import yaml

from wm3d_v3.data.scale5b_contracts import (
    canonical_sha256,
    load_contract,
    load_seal,
    resolve_regular_file,
    verify_dataset_seal,
)
from wm3d_v3.training.scale5b_config import (
    TRAIN_CONFIG_SCHEMA,
    training_contract_sha256,
    verify_code_receipt,
)
from wm3d_v3.training.scale5b_environment import (
    load_environment_contract,
    verify_environment_receipt,
)


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
PLACEHOLDER = "__MATERIALIZE_REQUIRED__"


def _load_regular_json(path: Path) -> tuple[Path, dict[str, Any]]:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular JSON file: {input_path}")
    resolved = input_path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {resolved}")
    return resolved, value


def _load_regular_yaml(path: Path) -> tuple[Path, dict[str, Any]]:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular YAML file: {input_path}")
    resolved = input_path.resolve(strict=True)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be an object: {resolved}")
    return resolved, value


def _real_directory(path: Path, name: str) -> Path:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} is not a real directory: {input_path}")
    return input_path.resolve(strict=True)


def _placeholder_paths(value: Any, prefix: str = "$") -> list[str]:
    if isinstance(value, dict):
        return [
            path
            for key, child in value.items()
            for path in _placeholder_paths(child, f"{prefix}.{key}")
        ]
    if isinstance(value, list):
        return [
            path
            for index, child in enumerate(value)
            for path in _placeholder_paths(child, f"{prefix}[{index}]")
        ]
    if isinstance(value, str) and PLACEHOLDER in value:
        return [prefix]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--environment-contract", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-lineage", required=True)
    parser.add_argument("--world-size", type=int, choices=(2, 64, 128), default=128)
    parser.add_argument(
        "--smoke-confirmation",
        default="",
        help="Required only for the isolated 2-GPU public smoke template.",
    )
    parser.add_argument("--shard-degree", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not SHA_RE.fullmatch(args.run_lineage):
        raise ValueError("run-lineage must be an explicit lowercase SHA-256")
    if not RUN_NAME_RE.fullmatch(args.run_name):
        raise ValueError("run-name must be a canonical lowercase run identity")
    if args.world_size == 2 and args.smoke_confirmation != (
        "EXECUTE_V7_NATIVE5B_PUBLIC_SMOKE"
    ):
        raise ValueError("2-GPU topology requires the exact smoke confirmation")
    if args.world_size != 2 and args.smoke_confirmation:
        raise ValueError("smoke confirmation is forbidden for formal topology")
    if args.world_size % args.shard_degree:
        raise ValueError("world-size must be divisible by shard-degree")
    denominator = args.world_size * args.micro_batch_size
    if args.global_batch_size % denominator:
        raise ValueError("global batch is not divisible by world*micro batch")
    accumulation = args.global_batch_size // denominator
    _template_path, config = _load_regular_yaml(args.template)
    if config.get("schema") != TRAIN_CONFIG_SCHEMA:
        raise ValueError("training template schema mismatch")
    if args.world_size == 2:
        if "smoke" not in _template_path.name:
            raise ValueError("2-GPU topology accepts only an explicit smoke template")
        if int(config.get("train", {}).get("total_steps", 0)) > 2:
            raise ValueError("small-topology smoke may run at most two optimizer steps")
        if not bool(config.get("model", {}).get("activation_checkpointing")):
            raise ValueError("small-topology smoke requires activation checkpointing")
    dataset_root = _real_directory(args.dataset_root, "dataset root")
    contract_relative = "control/dataset_contract.json"
    receipt_relative = "receipts/dataset_seal.json"
    contract = load_contract(resolve_regular_file(dataset_root, contract_relative))
    receipt = load_seal(resolve_regular_file(dataset_root, receipt_relative))
    if contract.sha256 != receipt.dataset_contract_sha256:
        raise ValueError("dataset contract/seal mismatch")
    seal_report = verify_dataset_seal(dataset_root, receipt_relative)
    if not seal_report["pass"]:
        raise ValueError(
            "dataset seal verification failed:\n" + "\n".join(seal_report["errors"])
        )
    code_receipt_path, code_receipt_value = _load_regular_json(args.code_receipt)
    code_receipt_sha = canonical_sha256(code_receipt_value)
    verify_code_receipt(
        code_receipt_path,
        expected_sha256=code_receipt_sha,
        repo_root=_real_directory(args.code_root, "code root"),
    )
    environment_contract = load_environment_contract(args.environment_contract)
    environment_contract_path = args.environment_contract.resolve(strict=True)
    environment_contract_sha = canonical_sha256(environment_contract)
    environment_receipt_path, environment_receipt_value = _load_regular_json(
        args.environment_receipt
    )
    environment_receipt_sha = canonical_sha256(environment_receipt_value)
    environment_receipt = verify_environment_receipt(
        environment_receipt_path,
        expected_sha256=environment_receipt_sha,
        contract_path=environment_contract_path,
        check_current=True,
    )

    config["data"].update(
        {
            "root": str(dataset_root),
            "contract": contract_relative,
            "contract_sha256": contract.sha256,
            "seal_receipt": receipt_relative,
            "seal_receipt_sha256": receipt.sha256,
            "source_order": list(contract.source_order),
            "source_weights": contract.source_weights,
        }
    )
    config["distributed"].update(
        {
            "expected_world_size": args.world_size,
            "shard_degree": args.shard_degree,
        }
    )
    config["train"].update(
        {
            "global_batch_size": args.global_batch_size,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation": accumulation,
        }
    )
    config["run"].update(
        {
            "name": args.run_name,
            "run_lineage": args.run_lineage,
            "output_root": str(args.output_root.resolve()),
            "code_receipt_path": str(code_receipt_path),
            "code_receipt_sha256": code_receipt_sha,
            "environment_contract_path": str(environment_contract_path),
            "environment_contract_sha256": environment_contract_sha,
            "environment_receipt_path": str(environment_receipt_path),
            "environment_receipt_sha256": environment_receipt_sha,
        }
    )
    config["run"]["training_contract_sha256"] = training_contract_sha256(config)
    remaining_placeholders = _placeholder_paths(config)
    if remaining_placeholders:
        raise ValueError(
            "materialized config still contains placeholders at "
            + ", ".join(remaining_placeholders)
        )
    output = args.output_config.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(
        json.dumps(
            {
                "pass": True,
                "output_config": str(output),
                "training_contract_sha256": config["run"]["training_contract_sha256"],
                "dataset_seal_sha256": receipt.sha256,
                "code_receipt_sha256": code_receipt_sha,
                "environment_contract_sha256": environment_contract_sha,
                "environment_receipt_sha256": environment_receipt_sha,
                "environment_fingerprint_sha256": environment_receipt["environment"][
                    "fingerprint_sha256"
                ],
                "world_size": args.world_size,
                "global_batch_size": args.global_batch_size,
                "gradient_accumulation": accumulation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
