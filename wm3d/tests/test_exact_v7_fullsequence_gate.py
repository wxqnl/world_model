from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


TOOL = Path(__file__).parents[1] / "scripts/tools/run_exact_v7_fullsequence_gate.py"
SPEC = importlib.util.spec_from_file_location(TOOL.stem, TOOL)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class _ActionLikeBlock(torch.nn.Module):
    def forward(
        self,
        value: torch.Tensor,
        action_times: torch.Tensor,
        token_mask: torch.Tensor,
        task_condition: torch.Tensor,
        task_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        del action_times, task_condition, task_token_mask
        return value * token_mask[..., None]


def test_selects_full_token_mask_not_task_mask() -> None:
    block = _ActionLikeBlock()
    value = torch.randn(1, 9, 4)
    valid = torch.tensor([[True] * 6 + [False] * 3])
    task_mask = torch.zeros_like(valid)
    sequence, selected, semantics = gate._extract_sequence_and_valid(
        block,
        (value, torch.zeros(1, 9), valid, torch.zeros(1, 4), task_mask),
        {},
        expected_length=9,
    )
    assert sequence is value
    assert torch.equal(selected, valid)
    assert semantics == "true_is_valid"


def test_two_group_probe_does_not_mutate_source_batch() -> None:
    batch = {
        "action_group_ids": torch.tensor([[3, 0, 0]]),
        "action_group_mask": torch.tensor([[True, False, False]]),
        "future_factual_fine_action_values": torch.ones(1, 2, 3, 1, 2),
        "future_factual_fine_action_mask": torch.ones(
            1, 2, 3, 1, 2, dtype=torch.bool
        ),
        "future_factual_fine_action_dt": torch.ones(1, 2, 3, 1),
        "future_factual_fine_sample_mask": torch.ones(
            1, 2, 3, 1, dtype=torch.bool
        ),
        "future_factual_coarse_action_values": torch.ones(1, 2, 3, 2),
        "future_factual_coarse_action_mask": torch.ones(
            1, 2, 3, 2, dtype=torch.bool
        ),
    }
    original = batch["future_factual_fine_action_values"].clone()
    result = gate.make_two_group_variant(batch, max_group_id=16)
    assert torch.equal(batch["future_factual_fine_action_values"], original)
    assert bool(result["action_group_mask"][0, 1])
    assert int(result["action_group_ids"][0, 1]) == 4
    assert not torch.equal(
        result["future_factual_fine_action_values"][:, :, 0],
        result["future_factual_fine_action_values"][:, :, 1],
    )
