from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.gate_wm3d_v8_stage0_causal_dual_view_canary import (
    GATE_SCHEMA,
    REVIEW_DECISION,
    REVIEW_SCHEMA,
    evaluate,
    sha256_file,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "step_00000100.pt"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("model", b"weights")
    review = {
        "schema": REVIEW_SCHEMA,
        "passed": True,
        "decision": REVIEW_DECISION,
        "errors": [],
        "scope": "bounded_objective_wiring_and_exact_resume_not_downstream_score",
        "checkpoints": {
            "step100": {
                "path": str(checkpoint),
                "step": 100,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "zip_complete": True,
            }
        },
        "training": {
            "required_metrics_finite": {"total": True, "action": True},
            "positive_gradient_steps": {"world": 100, "action": 100},
            "fatal_hits": [],
            "steps_exact_0_to_99": True,
        },
        "resume": {"verified": True},
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    return review_path, checkpoint


def test_gate_binds_review_and_checkpoint(tmp_path: Path) -> None:
    review, checkpoint = _inputs(tmp_path)
    gate = evaluate(review, checkpoint, min_checkpoint_bytes=1)
    assert gate["schema"] == GATE_SCHEMA
    assert gate["passed"] is True
    assert gate["checkpoint"]["step"] == 100
    assert gate["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert gate["canary_checkpoint_must_not_be_loaded"] is True


def test_gate_rejects_tampered_checkpoint(tmp_path: Path) -> None:
    review, checkpoint = _inputs(tmp_path)
    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        evaluate(review, checkpoint, min_checkpoint_bytes=1)


def test_gate_rejects_failed_review(tmp_path: Path) -> None:
    review, checkpoint = _inputs(tmp_path)
    payload = json.loads(review.read_text())
    payload["passed"] = False
    review.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="review did not pass"):
        evaluate(review, checkpoint, min_checkpoint_bytes=1)
