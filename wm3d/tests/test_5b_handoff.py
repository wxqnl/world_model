from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from wm3d.models.model_factory import validate_model_profile
from wm3d.training.runtime_contract import validate_runtime_profile


ROOT = Path(__file__).resolve().parents[1]


def test_5b_validation_profile_matches_native_5b_and_128_h200() -> None:
    model = yaml.safe_load((ROOT / "configs/model/native_5b.yaml").read_text())
    runtime = yaml.safe_load(
        (ROOT / "configs/runtime/h200_128_fsdp2_validation10k.yaml").read_text()
    )
    validate_model_profile(model)
    validate_runtime_profile(runtime)
    assert model["expected_parameter_count"] == 5_108_342_963
    assert runtime["expected_world_size"] == 128
    assert runtime["distributed"]["shard_degree"] == 8
    assert runtime["resources"]["gpu_name_substring"] == "H200"
    assert runtime["resources"]["minimum_ib_rate_gbps"] == 400.0
    assert runtime["train"]["total_steps"] == 10_000
    assert runtime["train"]["checkpoint_steps"] == [100, 500]
    assert runtime["train"]["checkpoint_interval"] == 1000


def test_5b_site_init_is_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "site.env"
    command = [
        "bash",
        str(ROOT / "scripts/cluster/wm3d_5b.sh"),
        "init",
        str(destination),
    ]
    first = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert destination.is_file()
    assert destination.stat().st_mode & 0o777 == 0o600
    payload = destination.read_bytes()
    second = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert second.returncode == 2
    assert destination.read_bytes() == payload


def test_5b_report_accepts_complete_synthetic_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/step_00000010"
    checkpoint.mkdir(parents=True)
    metrics = {
        "step": 10,
        "lr": 1.0e-4,
        "source_id": 1,
        "grad_norm": 1.25,
        "seconds_per_log_interval": 2.0,
        "total": 4.0,
        "token_mse": 2.0,
    }
    (run / "train_metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    ownership = {
        "schema": "wm3d_v8_gradient_ownership_v2",
        "passed": True,
        "owners": {
            "native": {
                "required": True,
                "passed": True,
                "nonzero_elements": 2,
                "nonfinite_elements": 0,
            }
        },
    }
    (run / "gradient_ownership.json").write_text(json.dumps(ownership))
    metadata = {"schema": "wm3d_v8_distributed_checkpoint_v2", "step": 10, "world_size": 1}
    payload = checkpoint / "payload.bin"
    payload.write_bytes(b"sealed")
    (checkpoint / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
    manifest = {
        "schema": "wm3d_v8_distributed_checkpoint_v2",
        "step": 10,
        "files": {
            "metadata.json": {
                "size": (checkpoint / "metadata.json").stat().st_size,
                "sha256": _sha(checkpoint / "metadata.json"),
            },
            "payload.bin": {"size": payload.stat().st_size, "sha256": _sha(payload)},
        },
    }
    (checkpoint / "MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True))
    committed = {
        "schema": "wm3d_v8_distributed_checkpoint_commit_v2",
        "step": 10,
        "metadata_sha256": _sha(checkpoint / "metadata.json"),
        "manifest_sha256": _sha(checkpoint / "MANIFEST.json"),
        "manifest_content_sha256": _canonical_sha(manifest),
    }
    (checkpoint / "COMMITTED.json").write_text(json.dumps(committed, sort_keys=True))
    evaluation = {
        "schema": "wm3d_v8_unified_offline_eval_v2",
        "all_metrics_finite": True,
        "checkpoint_step": 10,
        "checkpoint_committed_sha256": _sha(checkpoint / "COMMITTED.json"),
        "metrics": {"total": 3.0},
        "coverage": {
            "native_supervised_elements": 12.0,
            "inactive_coarse_supervised_dimensions": 0.0,
        },
        "expected_coverage_lanes": ["native_supervised_elements"],
    }
    eval_path = run / "eval.json"
    eval_path.write_text(json.dumps(evaluation))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/tools/report_5b_run.py"),
            "--run-root",
            str(run),
            "--expected-step",
            "10",
            "--checkpoint",
            str(checkpoint),
            "--eval",
            str(eval_path),
            "--require-complete",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    # The strict complete report also requires the data/runtime closure.
    assert result.returncode == 1
    assert "checkpoint: PASS" in result.stdout
    assert "eval: PASS" in result.stdout
    assert "WM3D 5B pipeline: INCOMPLETE" in result.stdout


def test_5b_report_status_marks_missing_stages_pending(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/tools/report_5b_run.py"),
            "--run-root",
            str(tmp_path / "run"),
            "--data-profile",
            str(tmp_path / "missing.yaml"),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "WM3D 5B pipeline: INCOMPLETE" in result.stdout
    assert "pending:" in result.stdout


def test_5b_report_streams_large_jsonl_summary(tmp_path: Path) -> None:
    from scripts.tools.report_5b_run import _jsonl_summary

    path = tmp_path / "windows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(10_000):
            handle.write(
                json.dumps(
                    {
                        "sample_id": str(index),
                        "source": "alpha" if index % 2 else "beta",
                        "split": "train" if index % 10 else "val",
                    }
                )
                + "\n"
            )
    count, splits = _jsonl_summary(path)
    assert count == 10_000
    assert sum(splits.values()) == count
    assert splits["alpha:train"] == 5_000
    assert splits["beta:val"] == 1_000


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    import hashlib

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()
