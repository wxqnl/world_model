"""Shared configuration and code-lineage contracts for V7 native 5B."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from wm3d_v3.data.scale5b_contracts import (
    canonical_sha256,
    resolve_regular_file,
    sha256_file,
)


TRAIN_CONFIG_SCHEMA = "wm3d_v7_native5b_pretrain_config_v1"
CODE_RECEIPT_SCHEMA = "wm3d_v7_native5b_code_receipt_v1"


def semantic_training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return exactly the fields whose drift invalidates exact resume."""

    return {
        "schema": TRAIN_CONFIG_SCHEMA,
        "model": config["model"],
        "data": config["data"],
        "distributed": config["distributed"],
        "optimizer": config["optimizer"],
        "schedule": config["schedule"],
        "loss": config["loss"],
        "code_receipt_sha256": config["run"]["code_receipt_sha256"],
        "environment_contract_sha256": config["run"]["environment_contract_sha256"],
        "environment_receipt_sha256": config["run"]["environment_receipt_sha256"],
        "train_semantics": {
            key: config["train"][key]
            for key in (
                "total_steps",
                "micro_batch_size",
                "gradient_accumulation",
                "seed",
                "gradient_clip",
            )
        },
    }


def training_contract_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(semantic_training_contract(config))


def verify_code_receipt(
    receipt_path: Path,
    *,
    expected_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    input_path = Path(receipt_path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"code receipt is not a regular file: {input_path}")
    path = input_path.resolve(strict=True)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema") != CODE_RECEIPT_SCHEMA:
        raise ValueError(f"code receipt schema mismatch: {path}")
    actual_receipt_sha = canonical_sha256(receipt)
    if actual_receipt_sha != str(expected_sha256):
        raise ValueError(
            f"code receipt digest {actual_receipt_sha} != {expected_sha256}"
        )
    root = Path(repo_root).resolve(strict=True)
    if receipt.get("root_layout") != "wm3d_v7":
        raise ValueError("code receipt is not rooted at wm3d_v7")
    if receipt.get("scoped_git_status") != "":
        raise ValueError("formal code receipt was created from a dirty scope")
    if not receipt.get("files"):
        raise ValueError("code receipt binds no files")
    errors = []
    for relative, evidence in sorted(receipt.get("files", {}).items()):
        try:
            file_path = resolve_regular_file(root, relative)
        except Exception as exc:
            errors.append(f"{relative}: {exc}")
            continue
        if file_path.stat().st_size != int(evidence["size"]):
            errors.append(f"{relative}: size mismatch")
        elif sha256_file(file_path) != evidence["sha256"]:
            errors.append(f"{relative}: sha256 mismatch")
    if errors:
        raise ValueError("code receipt verification failed:\n" + "\n".join(errors))
    return receipt
