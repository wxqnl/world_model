"""Compare two bound WM3D checkpoint-eval reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from wm3d.data.contracts import atomic_write_json, canonical_sha256
from wm3d.training.eval import EVAL_SCHEMA


COMPARISON_SCHEMA = "wm3d_v7_eval_comparison_v1"
LOWER_IS_BETTER = (
    "loss/total",
    "rgb/mse",
    "depth/mae",
    "point/mae",
    "geometry_confidence/mae",
    "camera_pose/mae",
    "action/mae",
)
HIGHER_IS_BETTER = (
    "rgb/psnr",
    "contact/accuracy",
)


def _load_report(path: Path) -> tuple[Path, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"eval report must be a regular file: {path}")
    resolved = path.resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"eval report is not a JSON object: {resolved}")
    if value.get("schema") != EVAL_SCHEMA:
        raise ValueError(f"eval report schema mismatch: {resolved}")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"eval metrics are missing: {resolved}")
    required = set(LOWER_IS_BETTER) | set(HIGHER_IS_BETTER)
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"eval metrics missing {sorted(missing)}: {resolved}")
    for name, raw in metrics.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"eval metric is not numeric: {name}")
        if not math.isfinite(float(raw)):
            raise ValueError(f"eval metric is not finite: {name}")
    return resolved, value


def _binding(report: Mapping[str, Any], name: str) -> Any:
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping) or name not in bindings:
        raise ValueError(f"eval binding is missing: {name}")
    return bindings[name]


def _lineage(report: Mapping[str, Any]) -> Any:
    metadata = _binding(report, "checkpoint_metadata")
    if not isinstance(metadata, Mapping) or "run_lineage" not in metadata:
        raise ValueError("checkpoint metadata has no run_lineage")
    return metadata["run_lineage"]


def compare_eval_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    max_relative_regression: float = 0.20,
    max_absolute_regression: float = 1.0e-6,
    max_psnr_drop: float = 1.0,
    max_contact_accuracy_drop: float = 0.05,
) -> dict[str, Any]:
    if not 0.0 <= max_relative_regression <= 1.0:
        raise ValueError("max_relative_regression must lie in [0,1]")
    if min(
        max_absolute_regression,
        max_psnr_drop,
        max_contact_accuracy_drop,
    ) < 0:
        raise ValueError("comparison tolerances must be non-negative")

    baseline_metrics = baseline["metrics"]
    candidate_metrics = candidate["metrics"]
    checks = {
        "baseline_eval_passed": baseline.get("pass") is True,
        "candidate_eval_passed": candidate.get("pass") is True,
        "candidate_is_later_checkpoint": int(candidate["checkpoint_step"])
        > int(baseline["checkpoint_step"]),
        "same_training_contract": candidate.get("config_sha256")
        == baseline.get("config_sha256"),
        "same_dataset_seal": _binding(candidate, "dataset_seal_sha256")
        == _binding(baseline, "dataset_seal_sha256"),
        "same_code_receipt": _binding(candidate, "code_receipt_sha256")
        == _binding(baseline, "code_receipt_sha256"),
        "same_parameter_count": _binding(candidate, "parameter_count")
        == _binding(baseline, "parameter_count"),
        "same_run_lineage": _lineage(candidate) == _lineage(baseline),
        "same_eval_world_size": candidate.get("world_size")
        == baseline.get("world_size"),
        "same_eval_steps_per_rank": candidate.get("eval_steps_per_rank")
        == baseline.get("eval_steps_per_rank"),
    }
    gates: dict[str, dict[str, float | bool]] = {}
    for name in LOWER_IS_BETTER:
        before = float(baseline_metrics[name])
        after = float(candidate_metrics[name])
        allowance = max(
            abs(before) * max_relative_regression,
            max_absolute_regression,
        )
        passed = after <= before + allowance
        gates[name] = {
            "baseline": before,
            "candidate": after,
            "delta": after - before,
            "maximum_allowed": before + allowance,
            "pass": passed,
        }
    for name, maximum_drop in (
        ("rgb/psnr", max_psnr_drop),
        ("contact/accuracy", max_contact_accuracy_drop),
    ):
        before = float(baseline_metrics[name])
        after = float(candidate_metrics[name])
        passed = after >= before - maximum_drop
        gates[name] = {
            "baseline": before,
            "candidate": after,
            "delta": after - before,
            "minimum_allowed": before - maximum_drop,
            "pass": passed,
        }
    checks["native_metrics_within_regression_budget"] = all(
        bool(item["pass"]) for item in gates.values()
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "pass": all(checks.values()),
        "baseline_step": int(baseline["checkpoint_step"]),
        "candidate_step": int(candidate["checkpoint_step"]),
        "checks": checks,
        "thresholds": {
            "max_relative_regression": max_relative_regression,
            "max_absolute_regression": max_absolute_regression,
            "max_psnr_drop": max_psnr_drop,
            "max_contact_accuracy_drop": max_contact_accuracy_drop,
        },
        "metric_gates": gates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-relative-regression", type=float, default=0.20)
    parser.add_argument("--max-absolute-regression", type=float, default=1.0e-6)
    parser.add_argument("--max-psnr-drop", type=float, default=1.0)
    parser.add_argument("--max-contact-accuracy-drop", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path, baseline = _load_report(args.baseline)
    candidate_path, candidate = _load_report(args.candidate)
    report = compare_eval_reports(
        baseline,
        candidate,
        max_relative_regression=args.max_relative_regression,
        max_absolute_regression=args.max_absolute_regression,
        max_psnr_drop=args.max_psnr_drop,
        max_contact_accuracy_drop=args.max_contact_accuracy_drop,
    )
    report["baseline_report"] = str(baseline_path)
    report["baseline_report_sha256"] = canonical_sha256(baseline)
    report["candidate_report"] = str(candidate_path)
    report["candidate_report_sha256"] = canonical_sha256(candidate)
    if args.output is not None:
        output = args.output
        if output.is_symlink() or output.exists():
            raise FileExistsError(f"refusing to overwrite comparison: {output}")
        atomic_write_json(output, report, exclusive=True)
    print(json.dumps(report, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
