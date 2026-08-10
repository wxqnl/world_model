from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from scripts.review_wm3d_v8_stage0_causal_dual_view_canary import evaluate


SOURCES = (
    ["oxe_droid_action"] * 35
    + ["oxe_bridge_action"] * 15
    + ["robocasa_atomic"] * 10
    + ["robocasa_composite"] * 20
    + ["robocasa_mg"] * 20
)


def _checkpoint(path: Path, step: int) -> None:
    torch.save(
        {
            "step": step,
            "epoch": 0,
            "model": {"weight": torch.ones(1)},
            "opt": {"state": {0: {}}, "param_groups": [{}]},
            "sched": {"last_epoch": step, "_step_count": step + 1},
            "sampler_state": {
                "schema": "wm3d_v7_exact_source_cycle_v1",
                "epoch": 0,
                "micro_batches_consumed_in_epoch": step * 4,
                "gradient_accumulation_steps": 4,
                "sampler_num_replicas": 8,
                "sampler_seed": 1707,
                "source_cycle_optimizer_steps": 100,
                "source_cycle_position": step % 100,
            },
            "rng_contract_rank0": {
                "schema": "wm3d_v7_step_addressed_rng_v1",
                "base_seed": 1707,
                "rank": 0,
                "torch_cpu_state": torch.zeros(4, dtype=torch.uint8),
                "torch_cuda_state": torch.zeros(16, dtype=torch.uint8),
                "numpy_state": ("MT19937",),
                "python_state": (3,),
            },
            "run_lineage": "lineage-sha",
            "resolved_config_sha256": "resolved-sha",
            "resume_compat_sha256": "resume-sha",
        },
        path,
    )


def _step_line(step: int, source: str) -> str:
    return (
        f"[rank0] step {step} (ep 0) src={source} "
        "L_total=1.0 rgb_L1=0.1 lpips=0.2 depth=0.3 "
        "native_action=1.1 native_future=1.2 direct=1.3 direct_pose=0.4 "
        "policy_flow=1.4 policy_flow_pose=0.5 "
        "factual_grad_action_proj=1 factual_grad_state_dynamics=2 "
        "factual_grad_no_teacher_head=3 native_future_grad=1 "
        "main_teacher_action_weight=0 policy_flow_grip=0\n"
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "max_steps": 100,
                    "canary_initial_stop_step": 20,
                    "run_lineage": "human-readable-lineage",
                }
            }
        )
    )
    seal = tmp_path / "seal.json"
    seal.write_text(
        json.dumps(
            {
                "schema": "wm3d_v8_stage0_causal_dual_view_canary_seal_v1",
                "passed": True,
                "launch_ready": True,
                "errors": [],
                "blockers": [],
                "warnings": [],
            }
        )
    )
    fresh = tmp_path / "fresh.log"
    fresh.write_text("".join(_step_line(i, SOURCES[i]) for i in range(20)))
    resume = tmp_path / "resume.log"
    resume.write_text("".join(_step_line(i, SOURCES[i]) for i in range(20, 100)))
    step20 = tmp_path / "step_00000020.pt"
    step100 = tmp_path / "step_00000100.pt"
    _checkpoint(step20, 20)
    _checkpoint(step100, 100)
    import hashlib

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "schema": "wm3d_v7_action_dynamics_resume_telemetry_v1",
                "event": "exact_resume_restored",
                "checkpoint_step": 20,
                "checkpoint_basename": step20.name,
                "checkpoint_sha256": hashlib.sha256(step20.read_bytes()).hexdigest(),
                "checkpoint_size_bytes": step20.stat().st_size,
                "run_lineage": "lineage-sha",
                "resolved_config_sha256": "resolved-sha",
                "resume_compat_sha256": "resume-sha",
                "model_load": {
                    "strict": True,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "skipped_keys": [],
                    "expanded_keys": [],
                },
                "optimizer": {"loaded": True, "metadata_matches_checkpoint": True},
                "scheduler": {"loaded": True, "metadata_matches_checkpoint": True},
                "sampler_restore": {
                    "verified": True,
                    "fast_forward_applied": True,
                    "fast_forward_without_dataset_io": True,
                    "next_batch_source": SOURCES[20],
                    "next_source_cycle_position": 20,
                },
                "rng_contract": {"verified": True},
            }
        )
        + "\n"
    )
    return {
        "runtime_config": runtime,
        "seal_report": seal,
        "fresh_log": fresh,
        "resume_log": resume,
        "telemetry": telemetry,
        "step20_checkpoint": step20,
        "step100_checkpoint": step100,
    }


def test_review_accepts_exact_five_source_resume_run(tmp_path: Path) -> None:
    report = evaluate(**_inputs(tmp_path), min_checkpoint_bytes=0)

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["training"]["steps_exact_0_to_99"] is True
    assert report["training"]["source_counts"] == {
        "oxe_droid_action": 35,
        "oxe_bridge_action": 15,
        "robocasa_atomic": 10,
        "robocasa_composite": 20,
        "robocasa_mg": 20,
    }
    assert report["resume"]["verified"] is True
    assert report["checkpoints"]["step100"]["step"] == 100


def test_review_rejects_step_gap_or_duplicate(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    text = inputs["resume_log"].read_text()
    inputs["resume_log"].write_text(
        text.replace("[rank0] step 20 ", "[rank0] step 19 ", 1)
    )

    report = evaluate(**inputs, min_checkpoint_bytes=0)

    assert report["passed"] is False
    assert any(
        "optimizer steps are not exactly 0..99" in error
        for error in report["errors"]
    )
