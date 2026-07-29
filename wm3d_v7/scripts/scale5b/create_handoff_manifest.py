#!/usr/bin/env python3
"""Bind every immutable artifact in one WM3D-V7 native-5B handoff."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat

import yaml

from wm3d_v3.data.scale5b_assets import verify_asset_bundle
from wm3d_v3.data.scale5b_contracts import (
    atomic_write_json,
    canonical_sha256,
    load_seal,
    resolve_regular_file,
    sha256_file,
    utc_now,
    verify_dataset_seal,
)
from wm3d_v3.training.scale5b_config import (
    TRAIN_CONFIG_SCHEMA,
    training_contract_sha256,
    verify_code_receipt,
)
from wm3d_v3.training.scale5b_environment import (
    verify_environment_receipt,
)


SCHEMA = "wm3d_v7_native5b_handoff_manifest_v1"


def _regular_file(path: Path) -> Path:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"not a regular file: {input_path}")
    return input_path.resolve(strict=True)


def _file_evidence(path: Path) -> dict[str, int | str]:
    value = _regular_file(path)
    return {
        "path": str(value),
        "size": int(value.stat().st_size),
        "sha256": sha256_file(value),
    }


def _real_directory(path: Path, name: str) -> Path:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{name} is not a real directory: {input_path}")
    return input_path.resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--container-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = _regular_file(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != TRAIN_CONFIG_SCHEMA:
        raise ValueError("training config schema mismatch")
    contract_sha = training_contract_sha256(config)
    if contract_sha != str(config["run"]["training_contract_sha256"]):
        raise ValueError("training config contract SHA mismatch")

    code_receipt = verify_code_receipt(
        Path(config["run"]["code_receipt_path"]),
        expected_sha256=str(config["run"]["code_receipt_sha256"]),
        repo_root=args.repo_root,
    )
    environment_receipt = verify_environment_receipt(
        Path(config["run"]["environment_receipt_path"]),
        expected_sha256=str(config["run"]["environment_receipt_sha256"]),
        contract_path=Path(config["run"]["environment_contract_path"]),
        check_current=True,
    )

    dataset_root = _real_directory(args.dataset_root, "dataset root")
    if dataset_root != Path(config["data"]["root"]).resolve(strict=True):
        raise ValueError("supplied dataset root differs from training config")
    dataset_report = verify_dataset_seal(
        dataset_root,
        str(config["data"]["seal_receipt"]),
    )
    if not dataset_report["pass"]:
        raise ValueError(
            "dataset seal verification failed:\n"
            + "\n".join(dataset_report["errors"])
        )
    dataset_seal = load_seal(
        resolve_regular_file(
            dataset_root,
            str(config["data"]["seal_receipt"]),
        )
    )
    if dataset_seal.sha256 != str(config["data"]["seal_receipt_sha256"]):
        raise ValueError("dataset seal SHA differs from training config")

    asset_report = verify_asset_bundle(args.asset_root, deep=True)
    dataset_asset_receipt = json.loads(
        resolve_regular_file(
            dataset_root,
            "control/encoder_asset_receipt.json",
        ).read_text(encoding="utf-8")
    )
    if canonical_sha256(dataset_asset_receipt) != asset_report["receipt_sha256"]:
        raise ValueError("dataset and supplied encoder asset receipts differ")

    manifest = {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "run_name": config["run"]["name"],
        "run_lineage": config["run"]["run_lineage"],
        "training_contract_sha256": contract_sha,
        "config": _file_evidence(config_path),
        "code_receipt": {
            **_file_evidence(Path(config["run"]["code_receipt_path"])),
            "canonical_sha256": canonical_sha256(code_receipt),
        },
        "environment_receipt": {
            **_file_evidence(Path(config["run"]["environment_receipt_path"])),
            "canonical_sha256": canonical_sha256(environment_receipt),
            "fingerprint_sha256": environment_receipt["environment"][
                "fingerprint_sha256"
            ],
        },
        "dataset": {
            "root": str(dataset_root),
            "seal_canonical_sha256": dataset_seal.sha256,
            "seal_file": _file_evidence(
                dataset_root / str(config["data"]["seal_receipt"])
            ),
        },
        "encoder_assets": {
            "root": str(Path(args.asset_root).resolve(strict=True)),
            "receipt_canonical_sha256": asset_report["receipt_sha256"],
            "files": asset_report["files"],
            "bytes": asset_report["bytes"],
        },
        "container_artifact": _file_evidence(args.container_artifact),
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    atomic_write_json(args.output.resolve(), manifest, exclusive=True)
    print(
        json.dumps(
            {
                "pass": True,
                "output": str(args.output.resolve()),
                "content_sha256": manifest["content_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
