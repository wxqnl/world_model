#!/usr/bin/env python3
"""Read-only audit of a V8 Stage0 checkpoint before LIBERO fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wm3d_v3.training.train import (  # noqa: E402
    _resume_compat_config,
    build_model,
    build_v8_action_policy_contract,
    config_sha256,
    load_train_config,
)
from wm3d_v3.training.v8_action_policy_transition import (  # noqa: E402
    load_v8_stage0_for_libero_strict,
    sha256_file,
    validate_v8_stage0_checkpoint_payload,
)


NUMBERED_CHECKPOINT = re.compile(r"^step_([0-9]{8})\.pt$")


def _expected_runtime_config_sha256(config: dict) -> str:
    train = config.get("train") or {}
    data = config.get("data") or {}
    base_seed = int(train.get("seed", data.get("seed", 0)) or 0)
    train["resolved_seed"] = base_seed
    resume_sha = config_sha256(_resume_compat_config(config))
    lineage_payload = (
        str(train.get("run_lineage") or "")
        or f"{(config.get('out') or {}).get('root','')}|"
        f"{(config.get('contract') or {}).get('schema','')}|{resume_sha}"
    )
    train["resolved_resume_compat_sha256"] = resume_sha
    train["resolved_run_lineage"] = hashlib.sha256(
        lineage_payload.encode("utf-8")
    ).hexdigest()
    return config_sha256(config)


def _materialize_and_strict_load_target(
    config: dict,
    payload: dict,
    expected_contract: dict,
) -> dict:
    """Exercise the production strict loader against the configured target.

    Payload-only validation cannot prove that every native-core and policy
    tensor required by the downstream model exists with the exact shape.  The
    audit therefore constructs the CPU target model from the sealed config and
    performs the same strict load that the downstream trainer must call.
    """

    target = build_model(config)
    try:
        return load_v8_stage0_for_libero_strict(
            target,
            payload,
            expected_contract=expected_contract,
        )
    finally:
        del target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-config", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(f"numbered Stage0 checkpoint is missing/not regular: {checkpoint}")
    step_match = NUMBERED_CHECKPOINT.fullmatch(checkpoint.name)
    if step_match is None:
        raise RuntimeError(
            "transition audit accepts only step_XXXXXXXX.pt numbered checkpoints"
        )
    config = load_train_config(args.expected_config.resolve())
    expected_contract = build_v8_action_policy_contract(config)
    if expected_contract is None:
        raise RuntimeError("expected config does not resolve the V8 unified action ABI")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    transition = validate_v8_stage0_checkpoint_payload(
        payload, expected_contract=expected_contract
    )
    expected_config_sha = _expected_runtime_config_sha256(config)
    if transition["resolved_config_sha256"] != expected_config_sha:
        raise RuntimeError(
            "checkpoint does not belong to expected config: "
            f"checkpoint={transition['resolved_config_sha256']} "
            f"expected={expected_config_sha}"
        )
    filename_step = int(step_match.group(1))
    if transition["step"] != filename_step:
        raise RuntimeError(
            "checkpoint filename/payload step mismatch: "
            f"filename={filename_step} payload={transition['step']}"
        )
    strict_load = _materialize_and_strict_load_target(
        config,
        payload,
        expected_contract,
    )
    report = {
        "schema": "wm3d_v8_stage0_libero_transition_audit_v1",
        "passed": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        **transition,
        "strict_target_model_load": {
            "strict": bool(strict_load["strict"]),
            "loaded_tensor_count": int(strict_load["loaded_tensor_count"]),
            "target_config": str(args.expected_config.resolve()),
        },
    }
    encoded = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.report is not None:
        path = args.report.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to overwrite non-identical report: {path}")
        if not path.exists():
            path.write_bytes(encoded)
    print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
