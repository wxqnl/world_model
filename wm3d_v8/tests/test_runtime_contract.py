from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from wm3d_v3.training.runtime_contract import (
    RuntimeContractError,
    validate_runtime_profile,
)
from wm3d_v3.training.pretrain import _collate_and_trim


def _load(name: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "configs/runtime" / name).read_text())


@pytest.mark.parametrize(
    "name,world,shard",
    [
        ("smoke_2gpu_fsdp2.yaml", 2, 2),
        ("h100_8_fsdp2.yaml", 8, 8),
        ("h200_128_fsdp2.yaml", 128, 8),
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


def test_runtime_does_not_contain_model_or_dataset_branch() -> None:
    value = _load("h200_128_fsdp2.yaml")
    serialized = yaml.safe_dump(value)
    assert "5b" not in serialized.lower()
    assert "agibot" not in serialized.lower()
    assert "robocasa" not in serialized.lower()


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
