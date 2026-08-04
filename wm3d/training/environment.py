"""Pinned software-environment contract for WM3D.

The code receipt binds source files.  This module separately binds the
interpreter, CUDA-enabled PyTorch build, NCCL ABI, and Python distribution
versions that were qualified with the FSDP2/DCP smoke test.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
from typing import Any, Mapping

import torch

from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
)


ENVIRONMENT_CONTRACT_SCHEMA = "wm3d_v7_environment_contract_v1"
ENVIRONMENT_RECEIPT_SCHEMA = "wm3d_v7_environment_receipt_v1"


class EnvironmentContractError(RuntimeError):
    """Raised when a node differs from the qualified Python runtime."""


def _normalized_distribution_name(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def load_environment_contract(path: Path) -> dict[str, Any]:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EnvironmentContractError(
            f"environment contract is not a regular file: {input_path}"
        )
    contract_path = input_path.resolve(strict=True)
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    if value.get("schema") != ENVIRONMENT_CONTRACT_SCHEMA:
        raise EnvironmentContractError(
            f"environment contract schema mismatch: {contract_path}"
        )
    packages = value.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise EnvironmentContractError("environment contract has no package pins")
    normalized = {
        _normalized_distribution_name(name): str(version)
        for name, version in packages.items()
    }
    if len(normalized) != len(packages):
        raise EnvironmentContractError(
            "environment package names collide after normalization"
        )
    value["packages"] = dict(sorted(normalized.items()))
    python_minor = str(value.get("python_major_minor", ""))
    if python_minor.count(".") != 1:
        raise EnvironmentContractError("python_major_minor must look like 3.10")
    expected_nccl = tuple(int(item) for item in value.get("minimum_nccl", ()))
    if len(expected_nccl) != 3:
        raise EnvironmentContractError("minimum_nccl must contain three integers")
    return value


def current_environment_report(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Collect and validate the software-only runtime identity."""

    expected_packages = {
        _normalized_distribution_name(name): str(version)
        for name, version in dict(contract["packages"]).items()
    }
    installed: dict[str, str] = {}
    errors: list[str] = []
    for name, expected in sorted(expected_packages.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        installed[name] = actual
        if actual != expected:
            errors.append(f"package {name}: {actual} != {expected}")

    python_major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_major_minor != str(contract["python_major_minor"]):
        errors.append(
            f"python {python_major_minor} != {contract['python_major_minor']}"
        )
    if torch.__version__ != str(contract["torch"]):
        errors.append(f"torch {torch.__version__} != {contract['torch']}")
    torch_cuda = str(torch.version.cuda)
    if torch_cuda != str(contract["torch_cuda"]):
        errors.append(f"torch CUDA {torch_cuda} != {contract['torch_cuda']}")
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        errors.append("PyTorch NCCL distributed backend is unavailable")
        nccl = (0, 0, 0)
    else:
        try:
            nccl = tuple(int(item) for item in torch.cuda.nccl.version())
        except (AttributeError, RuntimeError, TypeError) as exc:
            nccl = (0, 0, 0)
            errors.append(f"unable to query NCCL version: {exc}")
        else:
            minimum_nccl = tuple(int(item) for item in contract["minimum_nccl"])
            if nccl < minimum_nccl:
                errors.append(f"NCCL {nccl} is below {minimum_nccl}")

    capabilities: dict[str, bool] = {}
    try:
        from torch.distributed.checkpoint.state_dict import (  # noqa: F401
            StateDictOptions,
            get_state_dict,
            set_state_dict,
        )
        from torch.distributed.fsdp import (  # noqa: F401
            FSDPModule,
            MixedPrecisionPolicy,
            fully_shard,
        )

        capabilities["fsdp2"] = True
        capabilities["distributed_checkpoint_state_dict"] = True
    except ImportError as exc:
        capabilities["fsdp2"] = False
        capabilities["distributed_checkpoint_state_dict"] = False
        errors.append(f"required FSDP2/DCP API is unavailable: {exc}")

    report = {
        "python_major_minor": python_major_minor,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda": torch_cuda,
        "nccl": list(nccl),
        "packages": installed,
        "capabilities": capabilities,
    }
    report["fingerprint_sha256"] = canonical_sha256(report)
    report["errors"] = errors
    report["pass"] = not errors
    return report


def create_environment_receipt(
    *,
    contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    contract = load_environment_contract(contract_path)
    report = current_environment_report(contract)
    if not report["pass"]:
        raise EnvironmentContractError(
            "environment qualification failed:\n" + "\n".join(report["errors"])
        )
    receipt = {
        "schema": ENVIRONMENT_RECEIPT_SCHEMA,
        "contract_sha256": canonical_sha256(contract),
        "environment": report,
    }
    atomic_write_json(Path(output_path), receipt, exclusive=True)
    return receipt


def verify_environment_receipt(
    receipt_path: Path,
    *,
    expected_sha256: str,
    contract_path: Path,
    check_current: bool,
) -> dict[str, Any]:
    input_path = Path(receipt_path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise EnvironmentContractError(
            f"environment receipt is not a regular file: {input_path}"
        )
    receipt_file = input_path.resolve(strict=True)
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("schema") != ENVIRONMENT_RECEIPT_SCHEMA:
        raise EnvironmentContractError("environment receipt schema mismatch")
    actual_sha = canonical_sha256(receipt)
    if actual_sha != str(expected_sha256):
        raise EnvironmentContractError(
            f"environment receipt SHA {actual_sha} != {expected_sha256}"
        )
    contract = load_environment_contract(contract_path)
    contract_sha = canonical_sha256(contract)
    if receipt.get("contract_sha256") != contract_sha:
        raise EnvironmentContractError(
            "environment receipt does not bind the current environment contract"
        )
    frozen_report = receipt.get("environment")
    if not isinstance(frozen_report, dict) or not frozen_report.get("pass"):
        raise EnvironmentContractError("environment receipt did not qualify")
    if check_current:
        current = current_environment_report(contract)
        if not current["pass"]:
            raise EnvironmentContractError(
                "current environment failed:\n" + "\n".join(current["errors"])
            )
        if current["fingerprint_sha256"] != frozen_report.get("fingerprint_sha256"):
            raise EnvironmentContractError(
                "current environment fingerprint differs from image receipt"
            )
    return receipt
