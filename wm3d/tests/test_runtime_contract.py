from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

import pytest
import torch
import yaml

from wm3d.training.runtime_contract import (
    RuntimeContractError,
    _streaming_model_data_core,
    validate_direct_raw_data_closure,
    validate_runtime_profile,
)
from wm3d.training.pretrain import _collate_and_trim, _learning_rate
from wm3d.data.step_sampler import ExactSourceSchedule


def _load(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/runtime" / name).read_text())


def _load_data(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/data" / name).read_text())


def _load_source_lock(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/sources" / name).read_text())


@pytest.mark.parametrize(
    "name,world,shard",
    [
        ("smoke_2gpu_fsdp2.yaml", 2, 2),
        ("h100_8_fsdp2.yaml", 8, 8),
        ("h200_64_fsdp2.yaml", 64, 8),
        ("h200_64_fsdp2_canary1k.yaml", 64, 8),
        ("h200_64_fsdp2_validation100k.yaml", 64, 8),
        ("h200_128_fsdp2.yaml", 128, 8),
        ("h200_128_fsdp2_canary1k.yaml", 128, 8),
        ("h200_128_fsdp2_validation100k.yaml", 128, 8),
    ],
)
def test_same_runtime_contract_scales_across_topologies(
    name: str, world: int, shard: int
) -> None:
    value = _load(name)
    validate_runtime_profile(value)
    assert value["expected_world_size"] == world
    assert value["distributed"]["shard_degree"] == shard


def test_global_batch_is_derived_not_trusted() -> None:
    value = copy.deepcopy(_load("h100_8_fsdp2.yaml"))
    value["train"]["global_batch_size"] += 1
    with pytest.raises(RuntimeContractError, match="derived"):
        validate_runtime_profile(value)


def test_rgb_decode_chunk_size_is_an_optional_positive_execution_tuning() -> None:
    value = copy.deepcopy(_load("h100_8_fsdp2.yaml"))
    value["train"]["rgb_decode_chunk_size"] = 16
    validate_runtime_profile(value)

    for invalid in (0, -1, True, 1.5):
        invalid_value = copy.deepcopy(value)
        invalid_value["train"]["rgb_decode_chunk_size"] = invalid
        with pytest.raises(RuntimeContractError, match="rgb_decode_chunk_size"):
            validate_runtime_profile(invalid_value)


def test_rgb_perceptual_chunk_size_is_an_optional_positive_execution_tuning() -> None:
    value = copy.deepcopy(_load("h100_8_fsdp2.yaml"))
    value["train"]["rgb_perceptual_chunk_size"] = 16
    validate_runtime_profile(value)

    for invalid in (0, -1, True, 1.5):
        invalid_value = copy.deepcopy(value)
        invalid_value["train"]["rgb_perceptual_chunk_size"] = invalid
        with pytest.raises(RuntimeContractError, match="rgb_perceptual_chunk_size"):
            validate_runtime_profile(invalid_value)


def test_cudnn_benchmark_is_an_optional_boolean_execution_tuning() -> None:
    value = copy.deepcopy(_load("h100_8_fsdp2.yaml"))
    value["train"]["cudnn_benchmark"] = True
    validate_runtime_profile(value)

    for invalid in (0, 1, "true", None):
        invalid_value = copy.deepcopy(value)
        invalid_value["train"]["cudnn_benchmark"] = invalid
        with pytest.raises(RuntimeContractError, match="cudnn_benchmark"):
            validate_runtime_profile(invalid_value)


def test_nonzero_start_lr_survives_formal_warmup() -> None:
    value = _load("h100_16_fsdp2_dual_path_mb16_50k.yaml")
    validate_runtime_profile(value)
    start = value["optimizer"]["start_lr"]
    peak = value["optimizer"]["peak_lr"]
    warmup = value["schedule"]["warmup_steps"]
    assert _learning_rate(0, value) == pytest.approx(
        start + (peak - start) / warmup
    )
    assert _learning_rate(warmup - 1, value) == pytest.approx(peak)

    invalid = copy.deepcopy(value)
    invalid["optimizer"]["start_lr"] = peak * 2
    with pytest.raises(RuntimeContractError, match="learning rates"):
        validate_runtime_profile(invalid)


def test_runtime_does_not_contain_model_or_dataset_branch() -> None:
    value = _load("h200_128_fsdp2.yaml")
    serialized = yaml.safe_dump(value)
    assert "5b" not in serialized.lower()
    assert "agibot" not in serialized.lower()
    assert "robocasa" not in serialized.lower()


def test_h200_64_formal_profile_preserves_scaling_budget() -> None:
    value = _load("h200_64_fsdp2.yaml")
    train = value["train"]
    optimizer = value["optimizer"]
    schedule = value["schedule"]
    assert value["name"] == "h200_64_fsdp2_formal600k"
    assert train["total_steps"] == 600000
    assert train["micro_batch_size"] == 4
    assert train["global_batch_size"] == 256
    assert train["gradient_accumulation"] == 1
    assert train["seed"] == 271828
    assert train["validation_seed"] == 314159
    assert train["validate_every"] == 5000
    assert train["validation_steps"] == 100
    assert train["checkpoint_steps"] == [1000, 5000, 20000]
    assert train["checkpoint_interval"] == 20000
    assert optimizer["peak_lr"] == pytest.approx(0.00012)
    assert optimizer["min_lr"] == pytest.approx(0.00001)
    assert schedule["warmup_steps"] == 2000
    assert schedule["stable_fraction"] == pytest.approx(0.8)
    resources = value["resources"]
    assert resources["gpu_name_substring"] == "H200"
    assert resources["minimum_gpu_memory_mib"] == 135000
    assert resources["minimum_shm_bytes"] == 64_000_000_000
    assert resources["minimum_data_free_bytes"] == 10_000_000_000_000
    assert resources["minimum_output_free_bytes"] == 10_000_000_000_000
    assert resources["minimum_ib_rate_gbps"] == pytest.approx(400.0)
    assert resources["minimum_allreduce_gbps"] == pytest.approx(4.0)


def test_h200_64_canary1k_uses_the_same_cluster_gate() -> None:
    value = _load("h200_64_fsdp2_canary1k.yaml")
    validate_runtime_profile(value)
    train = value["train"]
    assert value["name"] == "h200_64_fsdp2_canary1k"
    assert train["total_steps"] == 1000
    assert train["micro_batch_size"] == 4
    assert train["gradient_accumulation"] == 1
    assert train["global_batch_size"] == 256
    assert train["seed"] == 271828
    assert train["validation_seed"] == 314159
    assert train["validate_every"] == 250
    assert train["validation_steps"] == 20
    assert train["checkpoint_steps"] == [100, 500]
    assert train["checkpoint_interval"] == 1000
    assert value["schedule"]["warmup_steps"] == 100
    assert value["schedule"]["stable_fraction"] == pytest.approx(0.8)


def test_resource_contract_is_strict_and_optional_for_smaller_topologies() -> None:
    smoke = _load("smoke_2gpu_fsdp2.yaml")
    assert "resources" not in smoke
    validate_runtime_profile(smoke)
    formal = copy.deepcopy(_load("h200_64_fsdp2.yaml"))
    formal["resources"]["minimum_shm_bytes"] = 0
    with pytest.raises(RuntimeContractError, match="minimum_shm_bytes"):
        validate_runtime_profile(formal)
    formal = copy.deepcopy(_load("h200_64_fsdp2.yaml"))
    formal["resources"]["unexpected"] = 1
    with pytest.raises(RuntimeContractError, match="resource fields mismatch"):
        validate_runtime_profile(formal)


def test_h200_validation_profile_is_explicitly_not_formal() -> None:
    value = _load("h200_64_fsdp2_validation100k.yaml")
    assert value["name"].endswith("validation100k")
    assert value["train"]["total_steps"] == 100000
    assert value["resources"]["minimum_ib_rate_gbps"] == pytest.approx(400.0)


def test_legacy_compatible_data_profile_preserves_families_weights_and_hours() -> None:
    value = _load_data("public_robot_5649h_legacy_compatible.template.yaml")
    sources = value["sources"]
    assert value["name"] == "public_robot_5649h_legacy_compatible"
    assert [row["name"] for row in sources] == [
        "legacy_v7_formal",
        "robocasa_full",
        "agibot_2026_imitation",
        "agibot_2026_rich",
        "agibot_2026_reinforcement",
        "agibot_beta",
    ]
    assert [row["weight"] for row in sources] == [10, 15, 10, 8, 12, 45]
    assert sum(row["weight"] for row in sources) == 100
    assert sum(float(row["nominal_hours"]) for row in sources) == pytest.approx(5649.4)
    assert value["notes"]["nominal_total_hours"] == pytest.approx(5649.4)
    assert "robocasa365_mg" in value["notes"]["overlap_policy"]
    legacy = next(row for row in value["embodiments"] if row["name"] == "legacy_v7_single_arm")
    assert len(legacy["groups"][0]["action_semantics"]) == 7
    assert len(legacy["groups"][0]["state_semantics"]) == 10


def test_expanded_profile_is_not_named_or_weighted_as_legacy_compatible() -> None:
    value = _load_data("public_robot_6106h.template.yaml")
    assert value["name"] == "public_robot_6106h_expanded"
    assert value["notes"]["nominal_total_hours"] == pytest.approx(6106.4)
    assert value["notes"]["profile_role"] == "expanded_optional_profile_not_legacy_compatible"
    assert len(value["sources"]) == 9


def test_legacy_compatible_source_lock_does_not_download_expanded_sources() -> None:
    value = _load_source_lock("public_sources_5649h_legacy_compatible.template.yaml")
    assert [row["name"] for row in value["sources"]] == [
        "robocasa_full",
        "agibot_world_2026",
        "agibot_beta",
        "agibot_alpha_converter",
    ]
    assert "legacy_v7_formal" not in {row["name"] for row in value["sources"]}
    serialized = yaml.safe_dump(value)
    for expanded_only in ("droid", "bridge", "atomic", "composite"):
        assert expanded_only not in serialized


def test_legacy_compatible_weights_are_the_actual_sampler_cycle() -> None:
    value = _load_data("public_robot_5649h_legacy_compatible.template.yaml")
    source_order = [row["name"] for row in value["sources"]]
    weights = {row["name"]: row["weight"] for row in value["sources"]}
    schedule = ExactSourceSchedule(source_order, weights, seed=271828)
    assert schedule.cycle_length == 100
    for cycle in range(3):
        observed = Counter(
            schedule.address(cycle * 100 + position).source_name
            for position in range(100)
        )
        assert observed == weights


def test_batch_collate_trims_storage_padding_without_dropping_real_queries() -> None:
    sample: dict[str, object] = {}
    sample["history_fine_action_values"] = torch.zeros(2, 1, 6, 3)
    sample["history_fine_action_mask"] = torch.zeros(2, 1, 6, 3, dtype=torch.bool)
    sample["history_fine_action_dt"] = torch.zeros(2, 1, 6)
    sample["history_fine_sample_mask"] = torch.tensor(
        [[[True, True, False, False, False, False]], [[True, False, False, False, False, False]]]
    )
    sample["future_factual_fine_action_values"] = torch.zeros(2, 1, 6, 3)
    sample["future_factual_fine_action_mask"] = torch.zeros(2, 1, 6, 3, dtype=torch.bool)
    sample["future_factual_fine_action_dt"] = torch.zeros(2, 1, 6)
    sample["future_factual_fine_sample_mask"] = torch.tensor(
        [[[True, True, True, False, False, False]], [[True, False, False, False, False, False]]]
    )
    sample["policy_query_dt"] = torch.zeros(2, 8)
    sample["policy_query_mask"] = torch.tensor(
        [[True, True, True, True, False, False, False, False], [True, True, False, False, False, False, False, False]]
    )
    sample["target_fine_action"] = torch.zeros(2, 8, 3)
    sample["target_fine_action_mask"] = torch.zeros(2, 8, 3, dtype=torch.bool)
    result = _collate_and_trim([sample, sample])
    assert result["history_fine_action_values"].shape[-2] == 3
    assert result["future_factual_fine_sample_mask"].shape[-1] == 3
    assert result["policy_query_dt"].shape[-1] == 4
    assert result["target_fine_action"].shape[-2] == 4
def test_dual_path_teacher_schedule_is_explicit_and_fail_closed() -> None:
    from wm3d.training.pretrain import (
        _appearance_teacher_ratio,
        _training_appearance_teacher_ratio,
    )

    value = _load("h100_3_fsdp2_dual_path_pilot.yaml")
    validate_runtime_profile(value)
    assert _appearance_teacher_ratio(0, value) == pytest.approx(1.0)
    assert _appearance_teacher_ratio(75, value) == pytest.approx(0.5)
    assert _appearance_teacher_ratio(150, value) == pytest.approx(0.0)
    assert _appearance_teacher_ratio(200, value) == pytest.approx(0.0)
    assert value["train"]["appearance_validation_three_way"] is True

    periodic = copy.deepcopy(value)
    periodic["train"]["appearance_teacher0_every_steps"] = 4
    validate_runtime_profile(periodic)
    assert _training_appearance_teacher_ratio(2, periodic) == pytest.approx(
        _appearance_teacher_ratio(2, periodic)
    )
    assert _training_appearance_teacher_ratio(3, periodic) == 0.0
    assert _training_appearance_teacher_ratio(4, periodic) == pytest.approx(
        _appearance_teacher_ratio(4, periodic)
    )

    invalid_periodic = copy.deepcopy(value)
    invalid_periodic["train"]["appearance_teacher0_every_steps"] = 1
    with pytest.raises(RuntimeContractError, match="integer >= 2"):
        validate_runtime_profile(invalid_periodic)

    partial = copy.deepcopy(value)
    partial["train"].pop("appearance_teacher_end_ratio")
    with pytest.raises(RuntimeContractError, match="provided together"):
        validate_runtime_profile(partial)

    reversed_schedule = copy.deepcopy(value)
    reversed_schedule["train"]["appearance_teacher_end_ratio"] = 1.1
    with pytest.raises(RuntimeContractError, match="0 <= end <= start <= 1"):
        validate_runtime_profile(reversed_schedule)

    invalid_validation = copy.deepcopy(value)
    invalid_validation["train"]["appearance_validation_three_way"] = 1
    with pytest.raises(RuntimeContractError, match="appearance_validation_three_way"):
        validate_runtime_profile(invalid_validation)

def _direct_closure_fixture() -> dict:
    digest = "a" * 64
    return {
        "schema": "wm3d_direct_raw_data_closure_v1",
        "name": "fixture",
        "data_profile_path": "/data/profile.yaml",
        "data_profile_sha256": digest,
        "metadata_seal_path": "/data/metadata.json",
        "metadata_seal_sha256": digest,
        "metadata_root": "/data/metadata",
        "episode_index_path": "/data/episodes.jsonl",
        "episode_index_sha256": digest,
        "cache_index_path": "/data/windows.jsonl",
        "cache_index_sha256": digest,
        "grouped_normalization_path": "/data/normalization.npz",
        "grouped_normalization_sha256": digest,
        "task_manifest_path": "/data/tasks.jsonl",
        "task_manifest_sha256": digest,
        "encoder_contract_path": "/data/encoder.yaml",
        "encoder_contract_sha256": digest,
        "task_bank_root": "/data/task_bank",
        "task_bank_index_sha256": digest,
        "source_manifest_sha256_by_name": {"source": digest},
        "adapter_contract_sha256_by_name": {"source": digest},
        "appearance_token_grid": 16,
        "appearance_feature_layer": 4,
        "direct_input_rgb_size": 518,
        "direct_decode_workers": 4,
        "direct_robot_cache_episodes": 8,
        "direct_prefetch_windows": 32,
        "direct_video_index_cache_assets": 64,
        "direct_encode_chunk_rows": 8,
        "direct_minimum_chunk_rows": 1,
    }


def test_direct_raw_closure_translates_to_the_sealed_metadata_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def capture(value: dict) -> None:
        observed.update(value)

    monkeypatch.setattr(
        "wm3d.training.runtime_contract.validate_streaming_data_closure",
        capture,
    )
    validate_direct_raw_data_closure(_direct_closure_fixture())

    assert observed["schema"] == "wm3d_streaming_raw_data_closure_v1"
    assert observed["lru_root"] == "/data/metadata"
    assert observed["encode_batch_frames"] == 8
    assert observed["decode_workers"] == 4
    assert "direct_prefetch_windows" not in observed


def test_streaming_data_contract_ignores_model_only_conditioning_scales() -> None:
    sealed = {
        "model": {
            "T": 16,
            "P": 256,
            "K": 8,
            "hidden_size": 1536,
        }
    }
    runtime = copy.deepcopy(sealed)
    runtime["model"].update(
        {
            "factual_dynamics_repeats": 3,
            "factual_action_residual_scale": 0.2,
            "render_factual_dynamics_repeats": 1,
            "render_factual_action_residual_scale": 0.0,
            "appearance_action_residual_scale": 0.5,
            "appearance_autoregressive_steps": 2,
            "policy_task_modulation": True,
            "rgb_context_action_scale": 1.0,
            "rgb_context_appearance_delta_scale": 1.0,
        }
    )

    assert _streaming_model_data_core(sealed) == _streaming_model_data_core(runtime)

    runtime["model"]["T"] = 32
    assert _streaming_model_data_core(sealed) != _streaming_model_data_core(runtime)


def test_direct_raw_closure_seals_ignored_action_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def capture(value: dict) -> None:
        observed.update(value)

    monkeypatch.setattr(
        "wm3d.training.runtime_contract.validate_streaming_data_closure",
        capture,
    )
    value = _direct_closure_fixture()
    value["direct_ignored_action_dimensions"] = [
        {"source": "source", "group": "controller", "dimensions": [7, 14]}
    ]
    validate_direct_raw_data_closure(value)

    assert "direct_ignored_action_dimensions" not in observed


def test_direct_raw_closure_rejects_duplicate_ignored_action_dimensions() -> None:
    value = _direct_closure_fixture()
    value["direct_ignored_action_dimensions"] = [
        {"source": "source", "group": "controller", "dimensions": [7, 7]}
    ]
    with pytest.raises(RuntimeContractError, match="sorted unique"):
        validate_direct_raw_data_closure(value)


def test_direct_raw_closure_rejects_an_invalid_oom_backoff_contract() -> None:
    value = _direct_closure_fixture()
    value["direct_minimum_chunk_rows"] = 16
    with pytest.raises(RuntimeContractError, match="minimum chunk"):
        validate_direct_raw_data_closure(value)
