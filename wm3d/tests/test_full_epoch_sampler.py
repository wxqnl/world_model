"""Coverage/resume regression for fresh-run full-epoch scheduling."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wm3d.data.step_sampler import (
    FullEpochBatchSampler, SamplingContractError, complete_epoch_budget,
)
from wm3d.training.pretrain import _make_loader, _topology_contract_sha256, PretrainError
from wm3d.training.runtime_contract import RuntimeContractError, validate_runtime_profile


def arguments(world=2, accumulation=2):
    return dict(
        source_spans={"a": (0, 29), "b": (29, 69)},
        source_order=["a", "b"], source_weights={"a": 29, "b": 40},
        source_episode_spans={"a": [(0, 1), (1, 20), (20, 29)],
                              "b": [(29, 30), (30, 65), (65, 69)]},
        world_size=world, micro_batch_size=2, gradient_accumulation=accumulation,
        seed=3407, epochs=2,
    )


@pytest.mark.parametrize("world,accumulation", [(1, 1), (2, 2), (4, 1)])
def test_every_window_is_covered_in_each_epoch_and_padding_is_explicit(world, accumulation):
    args = arguments(world, accumulation)
    budget = complete_epoch_budget(args["source_weights"],
                                   global_batch=world * 2 * accumulation, epochs=2)
    per_epoch_batches = budget["steps_per_epoch"] * accumulation
    draws = []
    for rank in range(world):
        draws.append(list(FullEpochBatchSampler(
            **args, rank=rank, start_optimizer_step=0,
            num_optimizer_steps=budget["optimizer_steps"],
        )))
    for epoch in range(2):
        start = epoch * per_epoch_batches
        ranks = [[i for batch in batches[start:start+per_epoch_batches] for i in batch]
                 for batches in draws]
        counts = Counter(i for items in ranks for i in items)
        assert set(counts) == set(range(69))
        assert sum(counts.values()) - 69 == budget["padding_draws_per_epoch"]
        for left in range(world):
            for right in range(left+1, world):
                assert set(ranks[left]).isdisjoint(ranks[right])
        # Every optimizer step is source-homogeneous across ranks and accumulation.
        for offset in range(0, per_epoch_batches, accumulation):
            batch = [i for batches in draws
                     for micro in batches[start+offset:start+offset+accumulation] for i in micro]
            assert all(i < 29 for i in batch) or all(i >= 29 for i in batch)


def test_resume_before_tail_and_across_epoch_boundary_is_exact():
    args = arguments()
    budget = complete_epoch_budget(args["source_weights"], global_batch=8, epochs=2)
    for rank in range(2):
        full = list(FullEpochBatchSampler(
            **args, rank=rank, start_optimizer_step=0,
            num_optimizer_steps=budget["optimizer_steps"],
        ))
        for step in [1, budget["steps_per_epoch"]-1, budget["steps_per_epoch"],
                     budget["optimizer_steps"]-1]:
            resumed = list(FullEpochBatchSampler(
                **args, rank=rank, start_optimizer_step=step,
                num_optimizer_steps=budget["optimizer_steps"]-step,
            ))
            assert resumed == full[step * args["gradient_accumulation"]:]


def test_wrong_weights_or_out_of_budget_do_not_silently_skip_windows():
    args = arguments()
    wrong = dict(args, source_weights={"a": 1, "b": 1})
    with pytest.raises(SamplingContractError, match="actual selected-window"):
        FullEpochBatchSampler(**wrong, rank=0, start_optimizer_step=0, num_optimizer_steps=1)
    with pytest.raises(SamplingContractError, match="exceeds"):
        FullEpochBatchSampler(**args, rank=0, start_optimizer_step=0, num_optimizer_steps=999)
    with pytest.raises(SamplingContractError, match="at least one"):
        complete_epoch_budget({"a": 3}, global_batch=8, epochs=2)


@pytest.mark.parametrize("has_selected_weights", [False, True])
def test_loader_uses_new_mode_only_for_training_and_rejects_wrong_stop(has_selected_weights):
    args = arguments()
    dataset = SimpleNamespace(
        source_spans=args["source_spans"], source_names=args["source_order"],
        source_episode_spans=args["source_episode_spans"], requires_main_process=True,
    )
    if has_selected_weights:
        dataset.source_sampling_weights = {"a": 1, "b": 1}
    profile = SimpleNamespace(source_weights={"a": 3, "b": 8})
    runtime = {"train": {"micro_batch_size": 2, "num_workers": 0,
                        "persistent_workers": False, "sampling_mode": "source_count_epochs_v1",
                        "sampling_epochs": 2, "total_steps": 18}}
    common = dict(dataset=dataset, profile=profile, runtime_profile=runtime,
                  rank=0, world_size=2, start_step=0, num_steps=1, seed=3407,
                  gradient_accumulation=2)
    assert isinstance(_make_loader(**common, is_training=True).batch_sampler, FullEpochBatchSampler)
    assert not isinstance(_make_loader(**common).batch_sampler, FullEpochBatchSampler)
    runtime["train"]["total_steps"] = 17
    with pytest.raises(PretrainError, match="sealed full-epoch"):
        _make_loader(**common, is_training=True)


def test_runtime_requires_explicit_mode_and_epoch_count(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load((root / "configs/runtime/h100_8_fsdp2.yaml").read_text())
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    runtime["train"].update(train_episode_allowlist=str(selection),
                            sampling_mode="source_count_epochs_v1", sampling_epochs=2)
    validate_runtime_profile(runtime)
    for change in [{"sampling_epochs": 0}, {"sampling_epochs": True},
                   {"sampling_mode": "estimated_epochs"}]:
        wrong = deepcopy(runtime)
        wrong["train"].update(change)
        with pytest.raises(RuntimeContractError, match="complete epochs"):
            validate_runtime_profile(wrong)


def test_new_sampler_topology_is_bound_without_changing_legacy_contract():
    config = {
        "run": {"lineage": "run", "code_commit": "code", "environment_lock_sha256": "lock"},
        "bindings": {"model_contract_sha256": "model", "data_closure_sha256": "data",
                     "objective_profile_sha256": "objective"},
        "runtime_profile": {
            "expected_world_size": 16, "optimizer": {}, "schedule": {},
            "train": {"global_batch_size": 96, "total_steps": 18, "seed": 3407,
                      "gradient_clip": 1.0},
            "distributed": {"strategy": "fsdp2", "shard_degree": 8,
                            "param_dtype": "float32", "reduce_dtype": "float32",
                            "output_dtype": "bfloat16"},
        },
    }
    legacy = _topology_contract_sha256(config)
    config["runtime_profile"]["expected_world_size"] = 24
    assert _topology_contract_sha256(config) == legacy
    config["runtime_profile"]["train"].update(
        sampling_mode="source_count_epochs_v1", sampling_epochs=2)
    full24 = _topology_contract_sha256(config)
    config["runtime_profile"]["expected_world_size"] = 16
    assert _topology_contract_sha256(config) != full24
