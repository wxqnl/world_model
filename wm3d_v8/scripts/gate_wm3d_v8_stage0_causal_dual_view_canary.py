#!/usr/bin/env python3
"""Publish the immutable V8 Stage0 canary-to-formal gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any


REVIEW_SCHEMA = "wm3d_v8_stage0_causal_dual_view_canary_review_v1"
GATE_SCHEMA = "wm3d_v8_stage0_causal_dual_view_canary_gate_v1"
REVIEW_DECISION = "PASS_STAGE0_CAUSAL_DUAL_VIEW_CANARY"
GATE_DECISION = "PASS_STAGE0_CAUSAL_DUAL_VIEW_CANARY_TO_FORMAL"
CONFIRM = "EXECUTE_WM3D_V8_STAGE0_CANARY_GATE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_no_clobber(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"existing gate is non-identical: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"concurrent gate is non-identical: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def evaluate(
    review_path: Path,
    checkpoint_path: Path,
    *,
    min_checkpoint_bytes: int = 15_000_000_000,
) -> dict[str, Any]:
    review_path = review_path.resolve(strict=True)
    checkpoint_path = checkpoint_path.resolve(strict=True)
    review = json.loads(review_path.read_text())
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError("unexpected canary review schema")
    if review.get("passed") is not True or review.get("decision") != REVIEW_DECISION:
        raise ValueError("canary review did not pass")
    if review.get("errors") != []:
        raise ValueError("canary review contains errors")
    if (
        review.get("scope")
        != "bounded_objective_wiring_and_exact_resume_not_downstream_score"
    ):
        raise ValueError("canary review scope mismatch")

    step100 = (review.get("checkpoints") or {}).get("step100") or {}
    if int(step100.get("step", -1)) != 100:
        raise ValueError("review is not bound to step 100")
    if Path(str(step100.get("path") or "")).resolve() != checkpoint_path:
        raise ValueError("review checkpoint path mismatch")
    if step100.get("zip_complete") is not True:
        raise ValueError("review checkpoint is not ZIP-complete")
    size = checkpoint_path.stat().st_size
    if size < min_checkpoint_bytes or int(step100.get("size_bytes", -1)) != size:
        raise ValueError("checkpoint size mismatch")
    if not zipfile.is_zipfile(checkpoint_path):
        raise ValueError("checkpoint is not a torch ZIP")
    with zipfile.ZipFile(checkpoint_path) as archive:
        bad_member = archive.testzip()
        members = len(archive.namelist())
    if bad_member is not None:
        raise ValueError(f"checkpoint ZIP CRC failed at {bad_member}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != step100.get("sha256"):
        raise ValueError("checkpoint SHA mismatch")

    training = review.get("training") or {}
    required_finite = training.get("required_metrics_finite") or {}
    if not required_finite or not all(
        value is True for value in required_finite.values()
    ):
        raise ValueError("canary metrics are not all finite")
    gradients = training.get("positive_gradient_steps") or {}
    if not gradients or not all(int(value) == 100 for value in gradients.values()):
        raise ValueError("canary gradient coverage is incomplete")
    if training.get("fatal_hits") != []:
        raise ValueError("canary training contains fatal hits")
    if training.get("steps_exact_0_to_99") is not True:
        raise ValueError("canary optimizer steps are not exactly 0..99")
    if (review.get("resume") or {}).get("verified") is not True:
        raise ValueError("canary exact resume was not verified")

    return {
        "schema": GATE_SCHEMA,
        "passed": True,
        "decision": GATE_DECISION,
        "scope": "formal_fresh_initialization_gate_not_checkpoint_promotion",
        "review": {
            "path": str(review_path),
            "sha256": sha256_file(review_path),
            "schema": REVIEW_SCHEMA,
            "decision": REVIEW_DECISION,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "step": 100,
            "size_bytes": size,
            "sha256": checkpoint_sha,
            "zip_members": members,
            "zip_complete": True,
        },
        "formal_initialization": "fresh_random_world_with_frozen_pinned_codec",
        "canary_checkpoint_must_not_be_loaded": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("dry-run", "execute"), required=True)
    args = parser.parse_args()

    report = evaluate(args.review, args.checkpoint)
    report["mode"] = args.mode
    report["mutated"] = False
    if args.mode == "execute":
        if os.environ.get("WM3D_V8_STAGE0_CANARY_GATE") != CONFIRM:
            raise SystemExit(f"set WM3D_V8_STAGE0_CANARY_GATE={CONFIRM}")
        report["mutated"] = True
        payload = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
        _publish_no_clobber(args.output.resolve(), payload)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
