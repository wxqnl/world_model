#!/usr/bin/env python3
"""Publish one immutable evidence report for the public WM3D smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
    load_seal,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
    verify_dataset_seal,
)
from wm3d.training.checkpoint import CheckpointManager


SCHEMA = "wm3d_v7_public_smoke_report_v1"
UPSTREAM_REVISION = "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--code-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> tuple[Path, dict]:
    safe = resolve_regular_file(path.parent, path.name)
    value = json.loads(safe.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {safe}")
    return safe, value


def main() -> None:
    args = parse_args()
    work = resolve_real_directory(args.work_root, "smoke work root")
    dataset = resolve_real_directory(args.dataset_root, "smoke dataset root")
    train_root = resolve_real_directory(args.train_root, "smoke train root")
    eval_root = resolve_real_directory(args.eval_root, "smoke eval root")
    if args.output.parent.resolve(strict=True) != work:
        raise ValueError("smoke report must be published at the work-root top level")

    raw_path, raw = _json(args.raw_receipt)
    if (
        raw.get("schema") != "wm3d_v7_raw_download_receipt_v1"
        or raw.get("complete") is not True
        or raw.get("repo_id") != "lerobot/aloha_sim_insertion_human"
        or raw.get("revision") != UPSTREAM_REVISION
        or raw.get("resolved_revision") != UPSTREAM_REVISION
    ):
        raise ValueError("raw ALOHA receipt identity or completion mismatch")
    dataset_report = verify_dataset_seal(dataset)
    if not dataset_report["pass"]:
        raise ValueError("dataset seal verification failed")
    seal = load_seal(resolve_regular_file(dataset, "receipts/dataset_seal.json"))

    config_path = resolve_regular_file(args.train_config.parent, args.train_config.name)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if int(config["distributed"]["expected_world_size"]) != 2:
        raise ValueError("public smoke report expects world size 2")
    run_contract_path, run_contract = _json(train_root / "run_contract.json")
    parameter_count = int(run_contract["parameter_counts"]["total"])
    if parameter_count != 4_956_589_929:
        raise ValueError(f"unexpected WM3D parameter count {parameter_count}")

    checkpoint = train_root / "checkpoints" / "step_00000001"
    checkpoint_metadata = CheckpointManager(checkpoint.parent).verify(checkpoint)
    if int(checkpoint_metadata.get("step", -1)) != 1:
        raise ValueError("verified checkpoint metadata is not step 1")
    manifest_path, checkpoint_manifest = _json(checkpoint / "MANIFEST.json")
    checkpoint_files = checkpoint_manifest.get("files")
    if not isinstance(checkpoint_files, dict) or not checkpoint_files:
        raise ValueError("verified checkpoint manifest has no payload files")
    checkpoint_bytes = sum(
        int(item["size"]) for item in checkpoint_files.values()
    )
    eval_report_path, evaluation = _json(eval_root / "report.json")
    if evaluation.get("pass") is not True or int(evaluation.get("checkpoint_step", -1)) != 1:
        raise ValueError("smoke eval did not pass the explicit step-1 checkpoint")
    if int(evaluation["bindings"]["parameter_count"]) != parameter_count:
        raise ValueError("eval/model parameter binding mismatch")
    code_path, code = _json(args.code_receipt)
    environment_path, environment = _json(args.environment_receipt)

    report = {
        "schema": SCHEMA,
        "pass": True,
        "meaning": "infrastructure_correctness_smoke_not_quality_claim",
        "upstream": {
            "repo_id": raw["repo_id"],
            "revision": raw["revision"],
            "payload_files": raw["payload_files"],
            "payload_bytes": raw["payload_bytes"],
            "receipt_sha256": sha256_file(raw_path),
        },
        "dataset": {
            "root": str(dataset),
            "seal_sha256": seal.sha256,
            "source_window_counts": seal.source_window_counts,
            "source_hours": seal.source_hours,
        },
        "training": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "run_contract_sha256": sha256_file(run_contract_path),
            "world_size": 2,
            "parameter_count": parameter_count,
            "checkpoint": str(checkpoint),
            "checkpoint_commit_sha256": sha256_file(checkpoint / "COMMITTED.json"),
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "checkpoint_metadata_sha256": sha256_file(checkpoint / "metadata.json"),
            "checkpoint_files": len(checkpoint_files),
            "checkpoint_bytes": checkpoint_bytes,
        },
        "evaluation": {
            "report": str(eval_report_path),
            "report_sha256": sha256_file(eval_report_path),
            "metrics": evaluation["metrics"],
            "checks": evaluation["checks"],
        },
        "bindings": {
            "code_receipt_sha256": canonical_sha256(code),
            "environment_receipt_sha256": canonical_sha256(environment),
        },
    }
    atomic_write_json(args.output, report, exclusive=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
