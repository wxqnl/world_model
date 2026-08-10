#!/usr/bin/env python3
"""Review the bounded V8 Stage0 causal dual-view exact-resume canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import yaml


SCHEMA = "wm3d_v8_stage0_causal_dual_view_canary_review_v1"
SEAL_SCHEMA = "wm3d_v8_stage0_causal_dual_view_canary_seal_v1"
RESUME_SCHEMA = "wm3d_v7_action_dynamics_resume_telemetry_v1"
RNG_SCHEMA = "wm3d_v7_step_addressed_rng_v1"
SAMPLER_SCHEMA = "wm3d_v7_exact_source_cycle_v1"
EXPECTED_SOURCE_COUNTS = {
    "oxe_droid_action": 35,
    "oxe_bridge_action": 15,
    "robocasa_atomic": 10,
    "robocasa_composite": 20,
    "robocasa_mg": 20,
}
REQUIRED_FINITE_METRICS = (
    "L_total",
    "rgb_L1",
    "lpips",
    "depth",
    "native_action",
    "native_future",
    "direct",
    "direct_pose",
    "policy_flow",
    "policy_flow_pose",
)
REQUIRED_POSITIVE_GRADIENTS = (
    "factual_grad_action_proj",
    "factual_grad_state_dynamics",
    "factual_grad_no_teacher_head",
    "native_future_grad",
)
ERROR_RE = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|"
    r"NCCL.*(?:error|failed)|non[-_ ]?finite|No space left|"
    r"Input/output error|DataLoader worker.*exited",
    re.IGNORECASE,
)
STEP_RE = re.compile(
    r"^\[rank0\] step (\d+) \(ep \d+\) src=([^ ]+) (.*)$"
)
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _parse_kv(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw in KV_RE.findall(text):
        try:
            result[key] = float(raw)
        except ValueError:
            continue
    return result


def _parse_logs(paths: tuple[Path, Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fatal_hits: list[str] = []
    log_evidence: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(errors="replace")
        log_evidence.append(
            {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        for line_number, line in enumerate(text.splitlines(), 1):
            if ERROR_RE.search(line):
                fatal_hits.append(f"{path.name}:{line_number}:{line[:300]}")
            match = STEP_RE.match(line)
            if match:
                rows.append(
                    {
                        "step": int(match.group(1)),
                        "source": match.group(2),
                        "metrics": _parse_kv(match.group(3)),
                    }
                )
    return {
        "rows": rows,
        "fatal_hits": fatal_hits,
        "logs": log_evidence,
    }


def _checkpoint_evidence(
    path: Path, expected_step: int, min_checkpoint_bytes: int
) -> dict[str, Any]:
    expected_name = f"step_{expected_step:08d}.pt"
    if path.name != expected_name:
        raise ValueError(
            f"checkpoint basename {path.name!r} is not {expected_name!r}"
        )
    if not path.is_file():
        raise ValueError(f"checkpoint is missing: {path}")
    size = path.stat().st_size
    if size < min_checkpoint_bytes:
        raise ValueError(
            f"checkpoint {path} is only {size} bytes; "
            f"minimum is {min_checkpoint_bytes}"
        )
    if not zipfile.is_zipfile(path):
        raise ValueError(f"checkpoint is not a torch ZIP: {path}")
    with zipfile.ZipFile(path) as archive:
        members = len(archive.namelist())
        bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"checkpoint ZIP CRC failed at {bad_member}: {path}")

    payload = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False
    )
    required = (
        "model",
        "opt",
        "sched",
        "step",
        "run_lineage",
        "resolved_config_sha256",
        "resume_compat_sha256",
        "sampler_state",
        "rng_contract_rank0",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"checkpoint {path.name} is missing: {', '.join(missing)}"
        )
    if int(payload["step"]) != expected_step:
        raise ValueError(
            f"checkpoint {path.name} payload step={payload['step']}, "
            f"expected {expected_step}"
        )

    sampler = payload["sampler_state"]
    expected_micro_batches = expected_step * 4
    expected_cycle_position = expected_step % 100
    if not isinstance(sampler, dict) or sampler.get("schema") != SAMPLER_SCHEMA:
        raise ValueError(f"{path.name}: invalid sampler schema")
    sampler_expectations = {
        "micro_batches_consumed_in_epoch": expected_micro_batches,
        "gradient_accumulation_steps": 4,
        "sampler_num_replicas": 8,
        "source_cycle_optimizer_steps": 100,
        "source_cycle_position": expected_cycle_position,
    }
    for key, expected in sampler_expectations.items():
        if int(sampler.get(key, -1)) != expected:
            raise ValueError(
                f"{path.name}: sampler {key}={sampler.get(key)!r}, "
                f"expected {expected}"
            )

    rng = payload["rng_contract_rank0"]
    if not isinstance(rng, dict) or rng.get("schema") != RNG_SCHEMA:
        raise ValueError(f"{path.name}: invalid RNG contract")
    cpu_rng = rng.get("torch_cpu_state")
    cuda_rng = rng.get("torch_cuda_state")
    if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.numel() <= 0:
        raise ValueError(f"{path.name}: CPU RNG evidence is empty")
    if isinstance(cuda_rng, torch.Tensor):
        cuda_state_count = int(cuda_rng.shape[0]) if cuda_rng.ndim else int(cuda_rng.numel())
    elif isinstance(cuda_rng, (list, tuple)):
        cuda_state_count = len(cuda_rng)
    else:
        cuda_state_count = 0
    if cuda_state_count <= 0:
        raise ValueError(f"{path.name}: CUDA RNG evidence is empty")

    optimizer = payload["opt"]
    scheduler = payload["sched"]
    if not isinstance(optimizer, dict) or not isinstance(
        optimizer.get("state"), dict
    ):
        raise ValueError(f"{path.name}: optimizer state is not restorable")
    if not isinstance(scheduler, dict) or not scheduler:
        raise ValueError(f"{path.name}: scheduler state is not restorable")

    evidence = {
        "path": str(path.resolve()),
        "step": expected_step,
        "size_bytes": size,
        "sha256": sha256_file(path),
        "zip_members": members,
        "zip_complete": True,
        "run_lineage": str(payload["run_lineage"]),
        "resolved_config_sha256": str(payload["resolved_config_sha256"]),
        "resume_compat_sha256": str(payload["resume_compat_sha256"]),
        "model_state_entries": len(payload["model"]),
        "optimizer_state_entries": len(optimizer["state"]),
        "scheduler_state_entries": len(scheduler),
        "sampler_state": dict(sampler),
        "rng": {
            "schema": rng["schema"],
            "base_seed": int(rng["base_seed"]),
            "rank": int(rng["rank"]),
            "cpu_state_elements": int(cpu_rng.numel()),
            "cuda_state_count": cuda_state_count,
        },
    }
    del payload
    return evidence


def _read_telemetry(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid telemetry JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: telemetry row is not a mapping")
        rows.append(row)
    return rows, sha256_file(path)


def evaluate(
    *,
    runtime_config: Path,
    seal_report: Path,
    fresh_log: Path,
    resume_log: Path,
    telemetry: Path,
    step20_checkpoint: Path,
    step100_checkpoint: Path,
    min_checkpoint_bytes: int = 15_000_000_000,
) -> dict[str, Any]:
    errors: list[str] = []

    runtime = yaml.safe_load(runtime_config.read_text()) or {}
    train = runtime.get("train") or {}
    if int(train.get("max_steps", -1)) != 100:
        errors.append("runtime config max_steps is not 100")
    if int(train.get("canary_initial_stop_step", -1)) != 20:
        errors.append("runtime config initial stop is not 20")

    seal = json.loads(seal_report.read_text())
    if seal.get("schema") != SEAL_SCHEMA:
        errors.append("seal report schema mismatch")
    if seal.get("passed") is not True or seal.get("launch_ready") is not True:
        errors.append("seal report is not passed and launch-ready")

    checkpoints: dict[str, dict[str, Any]] = {}
    for label, path, step in (
        ("step20", step20_checkpoint, 20),
        ("step100", step100_checkpoint, 100),
    ):
        try:
            checkpoints[label] = _checkpoint_evidence(
                path, step, min_checkpoint_bytes
            )
        except Exception as exc:
            errors.append(str(exc))
            checkpoints[label] = {}

    log_result = _parse_logs((fresh_log, resume_log))
    rows = log_result["rows"]
    steps = [row["step"] for row in rows]
    steps_exact = steps == list(range(100))
    if not steps_exact:
        errors.append("optimizer steps are not exactly 0..99 without gaps/duplicates")
    counts = Counter(row["source"] for row in rows)
    if dict(counts) != EXPECTED_SOURCE_COUNTS:
        errors.append(
            f"source counts are {dict(counts)!r}, "
            f"expected {EXPECTED_SOURCE_COUNTS!r}"
        )
    if log_result["fatal_hits"]:
        errors.append(
            "training logs contain fatal patterns: "
            + repr(log_result["fatal_hits"][:5])
        )

    finite_summary: dict[str, bool] = {}
    for key in REQUIRED_FINITE_METRICS:
        valid = len(rows) == 100 and all(
            key in row["metrics"] and _finite(row["metrics"][key])
            for row in rows
        )
        finite_summary[key] = valid
        if not valid:
            errors.append(f"metric {key} is not finite at every optimizer step")
    positive_gradient_steps: dict[str, int] = {}
    for key in REQUIRED_POSITIVE_GRADIENTS:
        count = sum(
            _finite(row["metrics"].get(key))
            and float(row["metrics"][key]) > 0.0
            for row in rows
        )
        positive_gradient_steps[key] = count
        if count != 100:
            errors.append(
                f"gradient evidence {key} is positive at {count}/100 steps"
            )
    if any(
        row["metrics"].get("main_teacher_action_weight") != 0.0
        for row in rows
    ):
        errors.append("legacy teacher action objective became active")
    if any(row["metrics"].get("policy_flow_grip") != 0.0 for row in rows):
        errors.append("pose-only policy flow trained a gripper coordinate")

    resume_summary: dict[str, Any] = {"verified": False}
    try:
        telemetry_rows, telemetry_sha = _read_telemetry(telemetry)
        events = [
            row
            for row in telemetry_rows
            if row.get("schema") == RESUME_SCHEMA
            and row.get("event") == "exact_resume_restored"
        ]
        if len(events) != 1:
            raise ValueError(
                f"expected one exact-resume event, found {len(events)}"
            )
        event = events[0]
        step20 = checkpoints.get("step20") or {}
        step100 = checkpoints.get("step100") or {}
        required_equalities = {
            "checkpoint_step": (event.get("checkpoint_step"), 20),
            "checkpoint_basename": (
                event.get("checkpoint_basename"),
                step20_checkpoint.name,
            ),
            "checkpoint_sha256": (
                event.get("checkpoint_sha256"),
                step20.get("sha256"),
            ),
            "checkpoint_size_bytes": (
                event.get("checkpoint_size_bytes"),
                step20.get("size_bytes"),
            ),
            "run_lineage": (
                event.get("run_lineage"),
                step20.get("run_lineage"),
            ),
            "resolved_config_sha256": (
                event.get("resolved_config_sha256"),
                step20.get("resolved_config_sha256"),
            ),
            "resume_compat_sha256": (
                event.get("resume_compat_sha256"),
                step20.get("resume_compat_sha256"),
            ),
        }
        for key, (observed, expected) in required_equalities.items():
            if observed != expected:
                raise ValueError(
                    f"resume {key}={observed!r}, expected {expected!r}"
                )
        model_load = event.get("model_load") or {}
        if model_load.get("strict") is not True or any(
            model_load.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "skipped_keys",
                "expanded_keys",
            )
        ):
            raise ValueError("resume model load is not strict and exact")
        for component in ("optimizer", "scheduler"):
            evidence = event.get(component) or {}
            if (
                evidence.get("loaded") is not True
                or evidence.get("metadata_matches_checkpoint") is not True
            ):
                raise ValueError(f"resume {component} evidence is incomplete")
        sampler_restore = event.get("sampler_restore") or {}
        if (
            sampler_restore.get("verified") is not True
            or sampler_restore.get("fast_forward_applied") is not True
            or sampler_restore.get("fast_forward_without_dataset_io") is not True
            or int(sampler_restore.get("next_source_cycle_position", -1)) != 20
            or (
                rows
                and sampler_restore.get("next_batch_source")
                != rows[20]["source"]
            )
        ):
            raise ValueError("resume sampler fast-forward evidence is incomplete")
        if (event.get("rng_contract") or {}).get("verified") is not True:
            raise ValueError("resume RNG contract was not verified")
        for key in (
            "run_lineage",
            "resolved_config_sha256",
            "resume_compat_sha256",
        ):
            if step20.get(key) != step100.get(key):
                raise ValueError(f"step20/step100 checkpoint {key} differs")
        resume_summary = {
            "verified": True,
            "telemetry_path": str(telemetry.resolve()),
            "telemetry_sha256": telemetry_sha,
            "telemetry_rows": len(telemetry_rows),
            "event": event,
        }
    except Exception as exc:
        errors.append(str(exc))

    report = {
        "schema": SCHEMA,
        "passed": not errors,
        "decision": (
            "PASS_STAGE0_CAUSAL_DUAL_VIEW_CANARY"
            if not errors
            else "REJECT_STAGE0_CAUSAL_DUAL_VIEW_CANARY"
        ),
        "errors": errors,
        "inputs": {
            "runtime_config": {
                "path": str(runtime_config.resolve()),
                "sha256": sha256_file(runtime_config),
            },
            "seal_report": {
                "path": str(seal_report.resolve()),
                "sha256": sha256_file(seal_report),
            },
            "logs": log_result["logs"],
        },
        "training": {
            "optimizer_step_rows": len(rows),
            "steps_exact_0_to_99": steps_exact,
            "source_counts": dict(counts),
            "required_metrics_finite": finite_summary,
            "positive_gradient_steps": positive_gradient_steps,
            "fatal_hits": log_result["fatal_hits"],
        },
        "resume": resume_summary,
        "checkpoints": checkpoints,
        "scope": "bounded_objective_wiring_and_exact_resume_not_downstream_score",
    }
    return report


def _publish_no_clobber(path: Path, report: dict[str, Any]) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"existing report is non-identical: {path}")
        return sha256_file(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--seal-report", type=Path, required=True)
    parser.add_argument("--fresh-log", type=Path, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--step20-checkpoint", type=Path, required=True)
    parser.add_argument("--step100-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--min-checkpoint-bytes", type=int, default=15_000_000_000
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        runtime_config=args.runtime_config,
        seal_report=args.seal_report,
        fresh_log=args.fresh_log,
        resume_log=args.resume_log,
        telemetry=args.telemetry,
        step20_checkpoint=args.step20_checkpoint,
        step100_checkpoint=args.step100_checkpoint,
        min_checkpoint_bytes=args.min_checkpoint_bytes,
    )
    report_sha256 = _publish_no_clobber(args.out, report)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "decision": report["decision"],
                "report": str(args.out.resolve()),
                "report_sha256": report_sha256,
                "errors": report["errors"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
