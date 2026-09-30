"""New-run data-repair inheritance without any checkpoint or asset hash scan."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from wm3d.data.raw_asset_repairs import (
    ENVIRONMENT, SCHEMA, repair_resume_expectation,
)
from wm3d.training.runtime_contract import RuntimeContractError, validate_runtime_profile


@pytest.fixture
def approved(tmp_path, monkeypatch):
    # Contract-only tests: no video/model bytes or source checkpoint are created.
    value = {
        "schema": SCHEMA, "repair_id": "official_pr8",
        "upstream": "https://huggingface.co/datasets/cadene/droid/discussions/8",
        "user_approved": True, "adoption_run_root": str(tmp_path / "old_run"),
        "adoption_checkpoint_step": 2000,
        "assets": [{
            "source_root": str(tmp_path / "raw"),
            "relative_path": "videos/camera/episode.mp4",
            "sealed_digest": "a" * 64,
            "replacement_relative_path": ".wm3d_rgb_repairs/official_pr8/videos/camera/episode.mp4",
            "replacement_digest": "b" * 64,
            "decoded_frames": 10, "time_base": "1/15360",
            "timestamps_unchanged": True,
        }],
    }
    path = tmp_path / "approved.json"
    path.write_text(json.dumps(value))
    monkeypatch.setenv(ENVIRONMENT, str(path))
    return value


@pytest.mark.parametrize("step", [0, 1, 2000, 50000])
def test_fresh_run_binds_entire_existing_data_approval_from_step_zero(approved, tmp_path, step):
    expected = repair_resume_expectation(
        str(tmp_path / "fresh_run"), step, initial_raw_asset_repairs=deepcopy(approved)
    )
    assert expected == approved
    assert not (tmp_path / "old_run").exists()


def test_initial_inheritance_requires_the_active_complete_contract(approved, tmp_path, monkeypatch):
    changed = deepcopy(approved)
    changed["assets"][0]["decoded_frames"] += 1
    with pytest.raises(ValueError, match="differs"):
        repair_resume_expectation(str(tmp_path / "fresh"), 0, initial_raw_asset_repairs=changed)
    monkeypatch.delenv(ENVIRONMENT)
    with pytest.raises(ValueError, match="differs"):
        repair_resume_expectation(str(tmp_path / "fresh"), 0, initial_raw_asset_repairs=approved)


def test_initial_inheritance_cannot_relabel_the_original_run(approved):
    with pytest.raises(ValueError, match="distinct"):
        repair_resume_expectation(approved["adoption_run_root"], 0, initial_raw_asset_repairs=approved)


@pytest.mark.parametrize("step", [-1, True, 0.5])
def test_initial_inheritance_rejects_invalid_progress(approved, tmp_path, step):
    with pytest.raises(ValueError, match="non-negative"):
        repair_resume_expectation(str(tmp_path / "fresh"), step, initial_raw_asset_repairs=approved)


def test_original_adoption_exception_is_unchanged(approved, tmp_path):
    old = approved["adoption_run_root"]
    assert repair_resume_expectation(old, 2000) is None
    assert repair_resume_expectation(old, 2001) == approved
    with pytest.raises(ValueError, match="earlier"):
        repair_resume_expectation(old, 1999)
    with pytest.raises(ValueError, match="different formal run"):
        repair_resume_expectation(str(tmp_path / "fresh"), 5000)


def test_runtime_accepts_explicit_initial_data_revision_but_not_weight_warmstart(approved):
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load((root / "configs/runtime/h100_8_fsdp2.yaml").read_text())
    runtime["train"]["initial_raw_asset_repairs"] = deepcopy(approved)
    validate_runtime_profile(runtime)
    runtime["train"].update(model_warmstart_checkpoint="/data/Minko/old/checkpoints/step_00050000",
                            model_warmstart_new_parameter_prefixes=[])
    with pytest.raises(RuntimeContractError, match="fresh initialization"):
        validate_runtime_profile(runtime)


def test_runtime_rejects_unapproved_or_incomplete_inheritance(approved):
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load((root / "configs/runtime/h100_8_fsdp2.yaml").read_text())
    invalid = deepcopy(approved)
    invalid["user_approved"] = False
    runtime["train"]["initial_raw_asset_repairs"] = invalid
    with pytest.raises(RuntimeContractError, match="initial raw RGB repair"):
        validate_runtime_profile(runtime)
    invalid["user_approved"] = True
    invalid["assets"] = []
    with pytest.raises(RuntimeContractError, match="initial raw RGB repair"):
        validate_runtime_profile(runtime)
