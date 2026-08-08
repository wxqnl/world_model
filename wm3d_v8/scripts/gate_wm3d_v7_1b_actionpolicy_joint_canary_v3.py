#!/usr/bin/env python3
"""Issue the immutable 1K gate for full WM3D-V7 1B re-pretraining.

The gate consumes balanced, fixed-seed quality evaluations at an early and the
final canary checkpoint plus the rank-0 training log.  It proves objective
wiring and non-collapse; it is not presented as a downstream LIBERO score.
The output is a content-addressable receipt and this script never launches a
formal job.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from statistics import median
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preflight_wm3d_v7_stage0_actiondynamics import (  # noqa: E402
    load_config,
    resolved_config_sha256,
    sha256_file,
)


SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_canary_gate_v3"
CANARY_SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_canary_v3"
FORMAL_SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_formal_v3"
ERROR_PATTERNS = re.compile(
    r"Traceback|CUDA out of memory|OutOfMemoryError|NCCL.*(?:error|failed)|"
    r"non[-_ ]?finite|No space left|Input/output error|DataLoader worker.*exited",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"^\[rank0\] step (\d+) \(ep \d+\) (.*)$")
MIX_RE = re.compile(r"^\[rank0\] mixed_source_audit step=(\d+) (.*)$")
KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
EXPECTED_MIX = {
    "oxe_droid_action": 0.35,
    "oxe_bridge_action": 0.15,
    "robocasa_atomic": 0.10,
    "robocasa_composite": 0.20,
    "robocasa_mg": 0.20,
}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(report: dict[str, Any], key: str) -> float:
    value = ((report.get("metrics") or {}).get("ALL") or {}).get(key)
    if not isinstance(value, dict) or not _finite(value.get("mean")):
        raise ValueError(f"balanced eval is missing finite ALL.{key}")
    return float(value["mean"])


def _parse_kv(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, raw in KV_RE.findall(text):
        try:
            parsed[key] = float(raw)
        except ValueError:
            continue
    return parsed


def _parse_training_log(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    matches = ERROR_PATTERNS.findall(text)
    steps: list[tuple[int, dict[str, float]]] = []
    mixes: list[tuple[int, dict[str, float]]] = []
    for line in text.splitlines():
        step_match = STEP_RE.match(line)
        if step_match:
            steps.append((int(step_match.group(1)), _parse_kv(step_match.group(2))))
            continue
        mix_match = MIX_RE.match(line)
        if mix_match:
            mixes.append((int(mix_match.group(1)), _parse_kv(mix_match.group(2))))
    if not steps:
        raise ValueError("rank0 training log contains no optimizer-step telemetry")
    if not mixes:
        raise ValueError("rank0 training log contains no source-mix telemetry")
    return {
        "sha256": sha256_file(path),
        "path": str(path.resolve()),
        "error_matches": matches,
        "steps": steps,
        "mixes": mixes,
    }


def _window_median(
    rows: list[tuple[int, dict[str, float]]], key: str, *, first: bool
) -> float:
    selected = rows[:10] if first else rows[-10:]
    values = [payload[key] for _, payload in selected if key in payload and _finite(payload[key])]
    if len(values) < 5:
        raise ValueError(f"insufficient finite log samples for {key}: {len(values)}")
    return float(median(values))


def _load_quality(path: Path, expected_step: int) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("mode") != "balanced_v7_stage0_quality":
        raise ValueError(f"unexpected quality report mode in {path}")
    checkpoint = report.get("checkpoint") or {}
    if int(checkpoint.get("step", -1)) != expected_step:
        raise ValueError(
            f"quality report {path} binds step {checkpoint.get('step')}, expected {expected_step}"
        )
    sampling = report.get("sampling") or {}
    if int(sampling.get("samples_per_source", 0)) < 64:
        raise ValueError(f"quality report {path} uses fewer than 64 samples/source")
    if not bool(sampling.get("equal_source_weighting")):
        raise ValueError(f"quality report {path} is not source-balanced")
    for source, audit in (report.get("causal_runtime_audit") or {}).items():
        if float(audit.get("policy_pose_teacher_action_max_abs", math.inf)) > 1e-7:
            raise ValueError(f"{source}: serving pose leaked teacher action")
        if float(audit.get("policy_grip_teacher_action_max_abs", math.inf)) > 1e-7:
            raise ValueError(f"{source}: serving gripper leaked teacher action")
        if float(audit.get("policy_context_teacher_action_max_abs", math.inf)) > 1e-7:
            raise ValueError(f"{source}: policy context leaked teacher action")
        if float(audit.get("factual_world_teacher_action_mean_abs", 0.0)) <= 1e-5:
            raise ValueError(f"{source}: factual native world ignored teacher action")
    if len(report.get("causal_runtime_audit") or {}) != 5:
        raise ValueError(f"quality report {path} does not audit all five sources")
    return report


def _checkpoint_evidence(path: Path) -> dict[str, Any]:
    if path.name != "step_00001000.pt":
        raise ValueError(f"canary checkpoint is not step-addressed at 1000: {path.name}")
    if not path.is_file() or path.stat().st_size < 15_000_000_000:
        raise ValueError(f"canary checkpoint is missing or too small: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"canary checkpoint is not a torch ZIP: {path}")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    required = ("model", "opt", "sched", "step", "run_lineage", "sampler_state")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("canary checkpoint is missing: " + ", ".join(missing))
    if int(payload["step"]) != 1000:
        raise ValueError(f"checkpoint payload step is {payload['step']}, expected 1000")
    evidence = {
        "path": str(path.resolve()),
        "step": 1000,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "run_lineage": str(payload["run_lineage"]),
        "model_state_entries": len(payload["model"]),
        "optimizer_state_entries": len((payload["opt"] or {}).get("state") or {}),
        "scheduler_present": isinstance(payload["sched"], dict),
        "sampler_state_present": isinstance(payload["sampler_state"], dict),
        "zip_complete": True,
    }
    del payload
    return evidence


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    canary_cfg = load_config(args.canary_config)
    formal_cfg = load_config(args.formal_config)
    if (canary_cfg.get("contract") or {}).get("schema") != CANARY_SCHEMA:
        errors.append("canary config schema mismatch")
    if (formal_cfg.get("contract") or {}).get("schema") != FORMAL_SCHEMA:
        errors.append("formal config schema mismatch")
    train = canary_cfg.get("train") or {}
    model = canary_cfg.get("model") or {}
    exact_expectations = {
        "canary.max_steps": (train.get("max_steps"), 1000),
        "canary.direct_policy_head": (train.get("direct_policy_head"), "base"),
        "canary.grip_owner": (train.get("direct_policy_grip_owner"), "delta_composed"),
        "canary.flow_action_dim": (train.get("policy_flow_action_dim"), 6),
        "canary.flow_grip_weight": (float(train.get("policy_flow_grip_weight", -1)), 0.0),
        "canary.flow_serving": (model.get("policy_flow_use_as_policy"), False),
        "canary.fresh_init": (train.get("fresh_init_required"), True),
        "canary.pretrained": (train.get("pretrained_world_checkpoint"), None),
        "canary.resume": (train.get("resume_checkpoint"), None),
    }
    for label, (observed, expected) in exact_expectations.items():
        if observed != expected:
            errors.append(f"{label}: observed={observed!r}, expected={expected!r}")

    try:
        checkpoint = _checkpoint_evidence(args.checkpoint)
    except Exception as exc:  # gate must report every evidence failure atomically
        errors.append(str(exc))
        checkpoint = {}
    try:
        early = _load_quality(args.early_quality, 20)
        final = _load_quality(args.final_quality, 1000)
    except Exception as exc:
        errors.append(str(exc))
        early = final = {}
    try:
        telemetry = _parse_training_log(args.train_log)
    except Exception as exc:
        errors.append(str(exc))
        telemetry = {}

    metrics: dict[str, Any] = {}
    if early and final:
        try:
            keys = (
                "rgb_l1",
                "depth_relative_l1",
                "objective/L_point",
                "serving_pose_norm_l1",
                "serving_translation_mae_mm",
                "serving_rotation_mae_deg",
                "serving_grip_balanced_accuracy",
                "serving_grip_positive_recall",
                "serving_grip_negative_recall",
                "serving_grip_transition_recall",
                "native_no_teacher/translation_gain_vs_zero",
                "native_no_teacher/rotation_gain_vs_zero",
            )
            metrics = {
                key: {"step20": _mean(early, key), "step1000": _mean(final, key)}
                for key in keys
            }
            # Canary thresholds establish correct learning/serving wiring.  The
            # formal 100K run, not this 1K gate, is responsible for final skill.
            if metrics["serving_pose_norm_l1"]["step1000"] > min(
                0.80, 1.05 * metrics["serving_pose_norm_l1"]["step20"]
            ):
                errors.append("direct serving pose did not improve or remain within the safe 1K envelope")
            if metrics["serving_grip_balanced_accuracy"]["step1000"] < 0.52:
                errors.append("delta-composed gripper is below 0.52 balanced accuracy")
            for key in ("serving_grip_positive_recall", "serving_grip_negative_recall"):
                if metrics[key]["step1000"] < 0.20:
                    errors.append(f"{key} collapsed below 0.20")
            if metrics["serving_grip_transition_recall"]["step1000"] < 0.35:
                errors.append("gripper transition recall collapsed below 0.35")
            for key in ("rgb_l1", "depth_relative_l1", "objective/L_point"):
                if metrics[key]["step1000"] > 1.25 * metrics[key]["step20"]:
                    errors.append(f"native 3D quality regressed catastrophically for {key}")
        except (KeyError, ValueError) as exc:
            errors.append(str(exc))

    log_summary: dict[str, Any] = {}
    if telemetry:
        rows = telemetry["steps"]
        last_step = rows[-1][0]
        log_summary["first_step"] = rows[0][0]
        log_summary["last_step"] = last_step
        if last_step < 990:
            errors.append(f"rank0 log stops at step {last_step}, expected terminal 1000 telemetry")
        if telemetry["error_matches"]:
            errors.append(f"rank0 log contains fatal patterns: {telemetry['error_matches'][:5]}")
        try:
            for key in (
                "rgb_L1",
                "depth",
                "native_action",
                "native_future",
                "direct_pose",
                "policy_flow_pose",
            ):
                head = _window_median(rows, key, first=True)
                tail = _window_median(rows, key, first=False)
                log_summary[key] = {"head_median": head, "tail_median": tail}
                if key == "policy_flow_pose" and tail >= head:
                    errors.append("auxiliary pose flow did not improve across the canary")
            tail_payloads = [payload for _, payload in rows[-10:]]
            for key in (
                "factual_grad_action_proj",
                "factual_grad_state_dynamics",
                "factual_grad_no_teacher_head",
            ):
                if not any(payload.get(key, 0.0) > 0.0 for payload in tail_payloads):
                    errors.append(f"missing positive gradient evidence for {key}")
            if not any(payload.get("native_future_grad", 0.0) == 1.0 for payload in tail_payloads):
                errors.append("missing native future anchor gradient evidence")
            if any(payload.get("main_teacher_action_weight", 0.0) != 0.0 for payload in tail_payloads):
                errors.append("legacy teacher action objective became active")
            if any(payload.get("policy_flow_grip", 0.0) != 0.0 for payload in tail_payloads):
                errors.append("pose-only flow unexpectedly trained a gripper coordinate")
        except ValueError as exc:
            errors.append(str(exc))
        mix_step, mix = telemetry["mixes"][-1]
        log_summary["source_mix_step"] = mix_step
        log_summary["source_mix"] = mix
        for source, expected in EXPECTED_MIX.items():
            if abs(float(mix.get(source, -1.0)) - expected) > 0.035:
                errors.append(f"source mix {source} is outside ±0.035 at step {mix_step}")

    passed = not errors
    receipt = {
        "schema": SCHEMA,
        "passed": passed,
        "decision": "PASS_FORMAL_REPRETRAIN" if passed else "REJECT_FORMAL_REPRETRAIN",
        "errors": errors,
        "canary_config": {
            "path": str(args.canary_config.resolve()),
            "leaf_sha256": sha256_file(args.canary_config),
            "resolved_sha256": resolved_config_sha256(canary_cfg),
        },
        "formal_config_schema": FORMAL_SCHEMA,
        "formal_config_draft": {
            "path": str(args.formal_config.resolve()),
            "leaf_sha256": sha256_file(args.formal_config),
            "resolved_sha256": resolved_config_sha256(formal_cfg),
        },
        "checkpoint": checkpoint,
        "evidence": {
            "early_quality": {
                "path": str(args.early_quality.resolve()),
                "sha256": sha256_file(args.early_quality),
            },
            "final_quality": {
                "path": str(args.final_quality.resolve()),
                "sha256": sha256_file(args.final_quality),
            },
            "train_log": {
                "path": str(args.train_log.resolve()),
                "sha256": telemetry.get("sha256") if telemetry else None,
            },
        },
        "balanced_quality_metrics": metrics,
        "training_telemetry": log_summary,
        "interpretation": "objective_wiring_gate_not_downstream_libero_score",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-config", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--early-quality", type=Path, required=True)
    parser.add_argument("--final-quality", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = evaluate(args)
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
