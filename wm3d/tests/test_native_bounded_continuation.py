"""Bounded continuation regressions grounded in the actual frozen step783."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wm3d.data.step_sampler import FullEpochBatchSampler
from wm3d.training.native_bounded_continuation import (
    resolve_native_bounded_continuation,
    validate_native_bounded_declaration,
    validate_native_bounded_transition,
)
from wm3d.training.runtime_contract import (
    RuntimeContractError, canonical_sha256, load_materialized_runtime,
)

ROOT = Path("/data/Minko/wm3d_native_fresh_pretraining_20260930/small_data_20261002")
SOURCE = ROOT / "attempt_08_ur5_tcp_contract_2h_20261004"
DESTINATION = ROOT / "attempt_09_continue783_floor1e5_2h_20261005"


@pytest.fixture
def real_transition(monkeypatch):
    if not (SOURCE / "runtime.json").is_file():
        pytest.skip("requires the actual frozen native step783 source")
    monkeypatch.setenv("WM3D_RAW_ASSET_REPAIRS", str(SOURCE / "data_repairs/droid_official_pr8.json"))
    source, digest = load_materialized_runtime(SOURCE / "runtime.json")
    destination = deepcopy(source)
    destination["run"].update(
        name="native_bounded_continuation_contract_regression",
        lineage="native_bounded_continuation_contract_regression",
        output_root=str(DESTINATION / "training"),
        # This in-memory marker tests only the changed-code whitelist.
        # Launch provenance still requires its actual clean committed checkout.
        code_commit="0" * 40,
    )
    runtime = destination["runtime_profile"]
    runtime["name"] += "_bounded_continuation"
    runtime["train"].update(
        total_steps=1566, sampling_epochs=54,
        small_data_deadline_unix=source["runtime_profile"]["train"]["small_data_deadline_unix"] + 7200,
        native_bounded_continuation={
            "schema": "wm3d_native_bounded_continuation_v1",
            "source_runtime_path": str(SOURCE / "runtime.json"),
            "source_checkpoint_path": str(SOURCE / "training/checkpoints/step_00000783"),
            "source_step": 783,
        },
    )
    runtime["optimizer"].update(peak_lr=1e-5, min_lr=1e-5)
    runtime["schedule"]["warmup_steps"] = 0
    destination["bindings"]["runtime_profile_sha256"] = canonical_sha256(runtime)
    return source, digest, destination


def test_real_source_controls_resolve_without_distributed_init(real_transition):
    if not (SOURCE / "training/checkpoints/step_00000783").is_dir():
        pytest.skip("requires the actual frozen native step783 checkpoint")
    source, digest, destination = real_transition
    resolved, source_digest, audit = resolve_native_bounded_continuation(destination)
    assert resolved == source and source_digest == digest
    assert audit["source"]["checkpoint_inspection"] == {
        "path": str(SOURCE / "training/checkpoints/step_00000783"),
        "step": 783, "committed_sha256": audit["source"]["committed_sha256"],
        "saved_world_size": 24, "saved_shard_degree": 8, "resume_mode": "exact",
    }
    assert audit["destination"]["first_optimizer_update"] == 784
    assert audit["same_schedule_exact_trajectory"] is False
    assert "runtime_profile.train.checkpoint_interval" not in audit["allowed_change_paths"]
    assert destination["runtime_profile"]["train"]["checkpoint_interval"] == 783


@pytest.mark.parametrize("path,value", [
    (("runtime_profile", "train", "seed"), 3408),
    (("runtime_profile", "train", "checkpoint_interval"), 1566),
    (("runtime_profile", "optimizer", "video_backbone_lr_scale"), 1.0),
    (("runtime_profile", "optimizer", "betas"), [0.9, 0.999]),
    (("model_profile", "model", "native_3d_query_mode"), "zero"),
    (("runtime_profile", "train", "small_data_deadline_unix"), float("nan")),
    (("runtime_profile", "train", "small_data_deadline_unix"), float("inf")),
])
def test_real_source_transition_rejects_unapproved_changes(real_transition, path, value):
    source, digest, destination = real_transition
    target = destination
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    budget = json.loads((SOURCE / "training/BUDGET_STOP.json").read_text())
    with pytest.raises(RuntimeContractError):
        validate_native_bounded_transition(
            source, destination, source_runtime_config_sha256=digest,
            source_budget_stop=budget, source_committed_sha256="0" * 64,
        )


def test_declaration_has_exactly_four_fields(real_transition):
    _, _, destination = real_transition
    declaration = destination["runtime_profile"]["train"]["native_bounded_continuation"]
    validate_native_bounded_declaration(declaration)
    with pytest.raises(RuntimeContractError, match="fields"):
        validate_native_bounded_declaration({**declaration, "snapshot_root": "/tmp"})


def test_actual_24rank_saved_progress_and_rng_contract():
    checkpoint = SOURCE / "training/checkpoints/step_00000783"
    if not checkpoint.is_dir():
        pytest.skip("requires the actual frozen native step783 checkpoint")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["step"] == 783
    assert metadata["sampler_progress"] == {"next_optimizer_step": 783}
    assert metadata["world_size"] == 24 and metadata["shard_degree"] == 8
    for rank in range(24):
        state = torch.load(
            checkpoint / "rank_state" / f"rank_{rank:05d}.pt",
            map_location="cpu", weights_only=False,
        )
        assert state["progress"] == {"next_optimizer_step": 783}
        assert set(state["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert state["rng"]["torch_cuda"] is not None


def test_actual_source_count_sampler_does_not_restart_at_zero(real_transition):
    source, _, destination = real_transition
    train = source["runtime_profile"]["train"]
    source_name = train["small_data_selection"]["source"]
    n = train["small_data_selection"]["window_count_by_split"]["train"]
    # Counts are from the real selected source; no data/model/GPU probe.
    for rank in range(24):
        common = dict(
            source_spans={source_name: (0, n)}, source_order=[source_name],
            source_weights={source_name: n}, world_size=24, rank=rank,
            micro_batch_size=4, gradient_accumulation=1, seed=train["seed"],
        )
        original = FullEpochBatchSampler(
            **common, start_optimizer_step=782, num_optimizer_steps=1, epochs=27,
        )
        extended_prefix = FullEpochBatchSampler(
            **common, start_optimizer_step=782, num_optimizer_steps=1, epochs=54,
        )
        continued = FullEpochBatchSampler(
            **common, start_optimizer_step=783, num_optimizer_steps=1, epochs=54,
        )
        assert list(original) == list(extended_prefix)
        assert continued.describe_step(783)["epoch"] == 27
        assert continued.describe_step(783)["optimizer_step"] == 783
        assert destination["runtime_profile"]["train"]["sampling_epochs"] == 54
