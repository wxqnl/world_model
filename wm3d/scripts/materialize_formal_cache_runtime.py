#!/usr/bin/env python3
"""Seal the existing formal cache into the current WM3D runtime contract."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

from wm3d.data.formal_cache_adapter import (
    FORMAL_CACHE_CLOSURE_SCHEMA,
    FORMAL_CACHE_RECEIPT_SCHEMA,
    validate_formal_cache_closure,
)
from wm3d.data.manifest_contract import sha256_file
from wm3d.training.runtime_contract import (
    RUNTIME_CONFIG_SCHEMA,
    canonical_sha256,
    load_yaml,
    validate_materialized_runtime,
    validate_runtime_profile,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical file: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--legacy-reader-root", type=Path, required=True)
    parser.add_argument("--legacy-runtime-config", type=Path, required=True)
    parser.add_argument("--closure-report", type=Path, required=True)
    parser.add_argument("--token-codec", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-lineage", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("formal runtime materialization requires a clean current checkout")
    commit = _git(repo, "rev-parse", "HEAD")
    reader = args.legacy_reader_root.resolve(strict=True)
    if _git(reader, "status", "--porcelain"):
        raise RuntimeError("formal cache reader checkout must be clean")
    reader_commit = _git(reader, "rev-parse", "HEAD")
    reader_tree = _git(reader, "rev-parse", "HEAD^{tree}")

    model = load_yaml(args.model)
    runtime = load_yaml(args.runtime)
    validate_runtime_profile(runtime)
    objective = load_yaml(args.objective)
    cache_root = args.cache_root.resolve(strict=True)
    legacy_config = args.legacy_runtime_config.resolve(strict=True)
    closure_report = args.closure_report.resolve(strict=True)
    token_codec = args.token_codec.resolve(strict=True)
    environment = args.environment_lock.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.is_symlink():
        raise RuntimeError("output root cannot be a symlink")

    reader_text = str(reader)
    if reader_text not in sys.path:
        sys.path.insert(0, reader_text)
    from wm3d_v3.training.train import build_datasets, load_train_config  # type: ignore[import-not-found]

    legacy = load_train_config(legacy_config)
    legacy["data"]["view_dropout"] = 0.0
    train, val = build_datasets(legacy)
    source_order = tuple(str(name) for name in train.source_names)
    expected_source_order = (
        "oxe_droid_action",
        "oxe_bridge_action",
        "robocasa_atomic",
        "robocasa_composite",
        "robocasa_mg",
    )
    if source_order != expected_source_order or tuple(val.source_names) != expected_source_order:
        raise RuntimeError("formal cache source order drifted")
    weights = legacy["train"]["mixed_batch_sampler"]["source_cycle_counts_exact"]
    expected_weights = {
        "oxe_droid_action": 35,
        "oxe_bridge_action": 15,
        "robocasa_atomic": 10,
        "robocasa_composite": 20,
        "robocasa_mg": 20,
    }
    if weights != expected_weights:
        raise RuntimeError("formal cache source schedule drifted")
    source_lengths = {
        "train": {name: int(stop - start) for name, (start, stop) in train.source_spans.items()},
        "val": {name: int(stop - start) for name, (start, stop) in val.source_spans.items()},
    }
    receipt_value = {
        "schema": FORMAL_CACHE_RECEIPT_SCHEMA,
        "passed": True,
        "cache_root": str(cache_root),
        "legacy_reader_root": str(reader),
        "legacy_reader_commit": reader_commit,
        "legacy_reader_tree": reader_tree,
        "legacy_runtime_config_path": str(legacy_config),
        "legacy_runtime_config_sha256": sha256_file(legacy_config),
        "closure_report_path": str(closure_report),
        "closure_report_sha256": sha256_file(closure_report),
        "token_codec_path": str(token_codec),
        "token_codec_sha256": sha256_file(token_codec),
        "source_order": list(source_order),
        "source_weights": expected_weights,
        "source_lengths_by_split": source_lengths,
        "cache_representation": {
            "spatial_tokens": 64,
            "token_grid": 8,
            "stored_token_dim": 384,
            "token_dim": 2048,
            "num_views": 3,
            "rgb_size": 256,
        },
    }
    receipt_payload = json.dumps(
        receipt_value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"
    _publish(args.receipt.absolute(), receipt_payload)
    closure = {
        "schema": FORMAL_CACHE_CLOSURE_SCHEMA,
        "name": "formal_world16_existing_cache",
        "cache_root": str(cache_root),
        "receipt_path": str(args.receipt.absolute()),
        "receipt_sha256": sha256_file(args.receipt.absolute()),
    }
    validate_formal_cache_closure(closure)
    value = {
        "schema": RUNTIME_CONFIG_SCHEMA,
        "run": {
            "name": args.run_name,
            "lineage": args.run_lineage,
            "output_root": str(output_root),
            "code_commit": commit,
            "environment_lock_path": str(environment),
            "environment_lock_sha256": sha256_file(environment),
        },
        "model_profile": model,
        "data_closure": closure,
        "runtime_profile": runtime,
        "objective_profile": objective,
        "bindings": {
            "model_profile_sha256": canonical_sha256(model),
            "data_closure_sha256": canonical_sha256(closure),
            "runtime_profile_sha256": canonical_sha256(runtime),
            "objective_profile_sha256": canonical_sha256(objective),
            "model_contract_sha256": canonical_sha256(
                {"architecture": model["architecture"], "model": model["model"]}
            ),
        },
    }
    validate_materialized_runtime(value)
    runtime_payload = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    _publish(args.output.absolute(), runtime_payload)
    print(
        json.dumps(
            {
                "runtime": str(args.output.absolute()),
                "runtime_sha256": sha256_file(args.output.absolute()),
                "receipt": str(args.receipt.absolute()),
                "receipt_sha256": sha256_file(args.receipt.absolute()),
                "train_windows": sum(source_lengths["train"].values()),
                "val_windows": sum(source_lengths["val"].values()),
                "world_size": runtime["expected_world_size"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
