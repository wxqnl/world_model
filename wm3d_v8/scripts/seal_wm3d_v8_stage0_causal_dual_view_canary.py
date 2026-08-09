#!/usr/bin/env python3
"""Seal real five-source indices for the V8 causal dual-view canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    CAUSAL_DUAL_VIEW_REPRESENTATION,
    CAUSAL_DUAL_VIEW_SCHEMA,
    CausalDualViewPreflightError,
    load_config,
    resolved_config_sha256,
    validate_preflight,
)


REPORT_SCHEMA = "wm3d_v8_stage0_causal_dual_view_canary_seal_v1"
ROBOCASA_PARTITIONS = ("atomic", "composite", "mg")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_text_no_clobber(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"existing output is non-identical: {path}")
        return _sha256_file(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise FileExistsError(
                    f"existing output is non-identical: {path}"
                )
        return _sha256_file(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be a mapping")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty RoboCasa index: {path}")
    return rows


def _merge_robocasa_indices(
    inputs: dict[str, Path], output: Path
) -> tuple[str, list[dict[str, Any]]]:
    if set(inputs) != set(ROBOCASA_PARTITIONS):
        raise ValueError(
            "RoboCasa input partitions must be exactly atomic/composite/mg"
        )
    merged: list[dict[str, Any]] = []
    seen_clip_hashes: set[str] = set()
    for partition in ROBOCASA_PARTITIONS:
        path = Path(inputs[partition]).resolve()
        splits: set[str] = set()
        for row in _read_jsonl(path):
            if row.get("schema") != CAUSAL_DUAL_VIEW_SCHEMA:
                raise ValueError(f"{path}: invalid causal schema")
            if row.get("representation") != CAUSAL_DUAL_VIEW_REPRESENTATION:
                raise ValueError(f"{path}: invalid causal representation")
            if row.get("context_future_leakage") is not False:
                raise ValueError(f"{path}: future leakage contract is not false")
            source = str(row.get("source") or "")
            if source not in {"robocasa365", partition}:
                raise ValueError(
                    f"{path}: unexpected source {source!r} for {partition}"
                )
            existing_partition = row.get("v7_source")
            if existing_partition not in (None, partition):
                raise ValueError(
                    f"{path}: v7_source={existing_partition!r} conflicts with "
                    f"partition={partition!r}"
                )
            split = str(row.get("split") or "")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{path}: invalid split {split!r}")
            clip_hash = str(row.get("clip_hash") or "")
            if not clip_hash:
                raise ValueError(f"{path}: missing clip_hash")
            if clip_hash in seen_clip_hashes:
                raise ValueError(f"duplicate clip_hash across partitions: {clip_hash}")
            seen_clip_hashes.add(clip_hash)
            sealed = dict(row)
            sealed["v7_source"] = partition
            merged.append(sealed)
            splits.add(split)
        if not {"train", "val"}.issubset(splits):
            raise ValueError(
                f"{path}: {partition} must contain both train and val rows"
            )
    merged.sort(
        key=lambda row: (
            str(row["v7_source"]), str(row["split"]), str(row["clip_hash"])
        )
    )
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in merged
    )
    return _publish_text_no_clobber(output.resolve(), text), merged


def _runtime_overlay(
    *,
    base_config: Path,
    oxe_paths: dict[str, list[Path]],
    combined_robocasa_index: Path,
) -> dict[str, Any]:
    combined = combined_robocasa_index.resolve()
    combined_sha = _sha256_file(combined)
    indices: dict[str, Any] = {}
    for source in ("oxe_droid_action", "oxe_bridge_action"):
        paths = [path.resolve() for path in oxe_paths[source]]
        indices[source] = {
            "kind": "oxe",
            "paths": [str(path) for path in paths],
            "sha256": [_sha256_file(path) for path in paths],
            "paired_views": False,
        }
    for partition in ROBOCASA_PARTITIONS:
        indices[f"robocasa_{partition}"] = {
            "kind": "compact",
            "paths": [str(combined)],
            "sha256": [combined_sha],
            "partition": partition,
            "paired_views": True,
        }
    return {
        "_base_": str(base_config.resolve()),
        "data": {
            "compact_index": str(combined),
            "compact_index_sha256": combined_sha,
            "causal_dual_view_indices": indices,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--oxe-droid-train-index", type=Path, required=True)
    parser.add_argument("--oxe-droid-val-index", type=Path, required=True)
    parser.add_argument("--oxe-bridge-train-index", type=Path, required=True)
    parser.add_argument("--oxe-bridge-val-index", type=Path, required=True)
    parser.add_argument("--robocasa-atomic-index", type=Path, required=True)
    parser.add_argument("--robocasa-composite-index", type=Path, required=True)
    parser.add_argument("--robocasa-mg-index", type=Path, required=True)
    parser.add_argument("--combined-robocasa-index", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-training-assets", action="store_true")
    parser.add_argument("--skip-local-resources", action="store_true")
    args = parser.parse_args()

    robo_inputs = {
        "atomic": args.robocasa_atomic_index,
        "composite": args.robocasa_composite_index,
        "mg": args.robocasa_mg_index,
    }
    combined_sha, merged = _merge_robocasa_indices(
        robo_inputs, args.combined_robocasa_index
    )
    oxe_paths = {
        "oxe_droid_action": [
            args.oxe_droid_train_index, args.oxe_droid_val_index
        ],
        "oxe_bridge_action": [
            args.oxe_bridge_train_index, args.oxe_bridge_val_index
        ],
    }
    overlay = _runtime_overlay(
        base_config=args.base_config,
        oxe_paths=oxe_paths,
        combined_robocasa_index=args.combined_robocasa_index,
    )
    runtime_text = yaml.safe_dump(overlay, sort_keys=False)
    runtime_sha = _publish_text_no_clobber(args.runtime_config, runtime_text)
    resolved = load_config(args.runtime_config)
    try:
        preflight = validate_preflight(
            resolved,
            mode="full",
            verify_training_assets=not args.skip_training_assets,
            verify_local_resources=not args.skip_local_resources,
        )
        exit_code = 0
    except CausalDualViewPreflightError as exc:
        preflight = exc.report
        exit_code = 1
    report = {
        "schema": REPORT_SCHEMA,
        "passed": bool(preflight.get("passed")),
        "launch_ready": bool(preflight.get("launch_ready")),
        "base_config": str(args.base_config.resolve()),
        "runtime_config": str(args.runtime_config.resolve()),
        "runtime_config_sha256": runtime_sha,
        "resolved_config_sha256": resolved_config_sha256(resolved),
        "oxe_index_sha256": {
            source: [_sha256_file(path.resolve()) for path in paths]
            for source, paths in oxe_paths.items()
        },
        "robocasa_input_sha256": {
            partition: _sha256_file(path.resolve())
            for partition, path in robo_inputs.items()
        },
        "combined_robocasa_index_sha256": combined_sha,
        "combined_robocasa_rows": len(merged),
        "preflight": preflight,
    }
    _publish_text_no_clobber(
        args.report,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
