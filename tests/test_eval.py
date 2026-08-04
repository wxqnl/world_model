from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts.tools.compare_eval_reports import compare_eval_reports
from wm3d.training.eval import (
    DIRECT_METRIC_PREFIXES,
    EVAL_SCHEMA,
    _masked_stats,
)


ROOT = Path(__file__).resolve().parents[1]


def _report(step: int) -> dict:
    return {
        "schema": EVAL_SCHEMA,
        "pass": True,
        "checkpoint_step": step,
        "config_sha256": "config",
        "world_size": 128,
        "eval_steps_per_rank": 64,
        "metrics": {
            "loss/total": 10.0,
            "rgb/mse": 0.04,
            "rgb/psnr": 13.9794,
            "depth/mae": 0.10,
            "point/mae": 0.20,
            "geometry_confidence/mae": 0.10,
            "camera_pose/mae": 0.20,
            "action/mae": 0.30,
            "contact/accuracy": 0.60,
        },
        "bindings": {
            "dataset_seal_sha256": "dataset",
            "code_receipt_sha256": "code",
            "parameter_count": 4_956_589_929,
            "checkpoint_metadata": {"run_lineage": "lineage"},
        },
    }


def test_eval_direct_metrics_cover_every_native_output_family() -> None:
    assert DIRECT_METRIC_PREFIXES == (
        "rgb",
        "depth",
        "point",
        "geometry_confidence",
        "camera_pose",
        "action",
        "contact",
    )


def test_masked_stats_counts_only_supervised_values() -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([True, False])
    absolute, squared, predicted, predicted_squared, count = _masked_stats(
        prediction,
        target,
        mask,
    )
    assert absolute.item() == pytest.approx(3.0)
    assert squared.item() == pytest.approx(5.0)
    assert predicted.item() == pytest.approx(3.0)
    assert predicted_squared.item() == pytest.approx(5.0)
    assert count.item() == pytest.approx(2.0)


def test_eval_comparison_accepts_bound_later_non_regressing_checkpoint() -> None:
    baseline = _report(1_000)
    candidate = _report(5_000)
    for name in (
        "loss/total",
        "rgb/mse",
        "depth/mae",
        "point/mae",
        "geometry_confidence/mae",
        "camera_pose/mae",
        "action/mae",
    ):
        candidate["metrics"][name] *= 0.9
    candidate["metrics"]["rgb/psnr"] += 0.5
    candidate["metrics"]["contact/accuracy"] += 0.05
    comparison = compare_eval_reports(baseline, candidate)
    assert comparison["pass"] is True
    assert all(comparison["checks"].values())
    assert all(
        gate["pass"] is True for gate in comparison["metric_gates"].values()
    )


def test_eval_comparison_fails_on_regression_or_lineage_drift() -> None:
    baseline = _report(1_000)
    candidate = deepcopy(_report(5_000))
    candidate["metrics"]["rgb/mse"] = baseline["metrics"]["rgb/mse"] * 2.0
    candidate["bindings"]["dataset_seal_sha256"] = "other-dataset"
    comparison = compare_eval_reports(baseline, candidate)
    assert comparison["pass"] is False
    assert comparison["checks"]["same_dataset_seal"] is False
    assert comparison["checks"]["native_metrics_within_regression_budget"] is False
    assert comparison["metric_gates"]["rgb/mse"]["pass"] is False


def test_eval_comparison_cli_writes_once_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    baseline_path.write_text(json.dumps(_report(1_000)), encoding="utf-8")
    candidate_path.write_text(json.dumps(_report(5_000)), encoding="utf-8")
    command = [
        sys.executable,
        str(ROOT / "scripts/tools/compare_eval_reports.py"),
        "--baseline",
        str(baseline_path),
        "--candidate",
        str(candidate_path),
        "--output",
        str(output_path),
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["pass"] is True
    second = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "refusing to overwrite comparison" in second.stderr
