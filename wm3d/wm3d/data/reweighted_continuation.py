"""Authorized source reweighting after the complete formal step30000 checkpoint.

This preserves each source's original rank-contiguous window cursor, including
the original tail-padding rule. Only the order/frequency of future source
batches changes. It is not an exact continuation of the old mixture trajectory.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from wm3d.data.step_sampler import (
    ExactSourceSchedule, FullEpochBatchSampler, SamplingContractError,
)

TRANSITION_STEP = 30000
TOTAL_STEPS = 167446
MULTIPLIERS = {
    "oxe_bridge": 4,
    "oxe_stanford_hydra": 4,
    "oxe_berkeley_autolab_ur5": 4,
    "oxe_bc_z": 2,
}


class ReweightedContinuationBatchSampler:
    """Same24/global96 source cursors with an explicitly new source schedule."""

    def __init__(
        self, source_spans, source_order, source_weights, *,
        world_size, rank, micro_batch_size, gradient_accumulation,
        start_optimizer_step, num_optimizer_steps, seed,
        source_episode_spans, multipliers, transition_step=TRANSITION_STEP,
        total_steps=TOTAL_STEPS,
    ):
        if (world_size, micro_batch_size, gradient_accumulation, seed) != (24, 4, 1, 3407):
            raise SamplingContractError("reweight continuation requires original24/global96/seed3407")
        if dict(multipliers) != MULTIPLIERS or any(type(v) is not int for v in multipliers.values()):
            raise SamplingContractError("source multipliers differ from the authorized four sources")
        if transition_step != TRANSITION_STEP or total_steps != TOTAL_STEPS:
            raise SamplingContractError("only the authorized30000->167446 transition is supported")
        if len(source_order) != 27 or not set(MULTIPLIERS).issubset(source_order):
            raise SamplingContractError("reweight continuation requires the original full27 sources")
        if source_episode_spans is None:
            raise SamplingContractError("real episode-local spans are required")
        if (type(start_optimizer_step) is not int or type(num_optimizer_steps) is not int
                or start_optimizer_step < transition_step or num_optimizer_steps <= 0
                or start_optimizer_step + num_optimizer_steps > total_steps):
            raise SamplingContractError("requested continuation interval is outside the approved budget")
        self.base = FullEpochBatchSampler(
            source_spans, source_order, source_weights, world_size=world_size,
            rank=rank, micro_batch_size=micro_batch_size,
            gradient_accumulation=gradient_accumulation,
            start_optimizer_step=0, num_optimizer_steps=1, seed=seed,
            source_episode_spans=source_episode_spans, epochs=2,
        )
        if (self.base.budget["selected_windows"], self.base.budget["steps_per_epoch"],
                self.base.budget["optimizer_steps"]) != (8036325, 83723, TOTAL_STEPS):
            raise SamplingContractError("actual original full27 window/batch budget differs")
        self.start_optimizer_step = start_optimizer_step
        self.num_optimizer_steps = num_optimizer_steps
        self.transition_step = transition_step
        self.total_steps = total_steps
        # The original source schedule is deterministic and independent of model RNG.
        # Count the exact consumed prefix, not a proportional approximation.
        old_schedule = self.base.schedule
        old_schedule.address(transition_step - 1)
        completed_cycles, position = divmod(transition_step, old_schedule.cycle_length)
        self.source_cursor_at_transition = {
            name: completed_cycles * old_schedule.weights[name] + bisect_left(
                old_schedule._cached_source_positions[i], position
            )
            for i, name in enumerate(old_schedule.source_order)
        }
        if sum(self.source_cursor_at_transition.values()) != transition_step:
            raise SamplingContractError("original source cursor migration lost optimizer batches")
        self.schedule = ExactSourceSchedule(
            source_order,
            {name: old_schedule.weights[name] * MULTIPLIERS.get(name, 1)
             for name in source_order},
            seed=seed,
        )
        self.migration = {
            "schema": "wm3d_formal30000_reweighted_sampler_v1",
            "transition_step": transition_step,
            "global_optimizer_budget": total_steps,
            "remaining_optimizer_updates": total_steps - transition_step,
            "source_weight_multipliers": dict(MULTIPLIERS),
            "other_source_multiplier": 1,
            "old_source_batch_cursor": dict(self.source_cursor_at_transition),
            "old_source_steps_per_epoch": dict(old_schedule.weights),
            "new_source_schedule_weights": dict(self.schedule.weights),
            "new_source_schedule_cycle_steps": self.schedule.cycle_length,
            "new_source_batch_fractions": self.schedule.fractions,
            "source_window_cursor_preserved": True,
            "source_rank_partition_and_permutation_unchanged": True,
            "source_tail_padding_rule_unchanged": True,
            "new_schedule_offset": 0,
            "global_step_reset": False,
            "model_rng_consumed_by_sampler": False,
            "all_sources_complete_two_epochs_claimed": False,
        }

    def __len__(self):
        return self.num_optimizer_steps * self.base.gradient_accumulation

    def describe_step(self, optimizer_step):
        if not TRANSITION_STEP <= optimizer_step < TOTAL_STEPS:
            raise SamplingContractError("step is outside the continuation")
        address = self.schedule.address(optimizer_step - self.transition_step)
        name = address.source_name
        cursor = self.source_cursor_at_transition[name] + address.source_occurrence
        source_epoch, batch_in_epoch = divmod(cursor, self.base.schedule.weights[name])
        return {
            "optimizer_step": optimizer_step,
            "source_name": name,
            "source_batch_cursor": cursor,
            "source_epoch": source_epoch,
            "source_batch_in_epoch": batch_in_epoch,
            "new_schedule_cycle": address.cycle,
            "new_schedule_position": address.cycle_position,
        }

    def batch_for_step(self, optimizer_step, micro_step=0):
        if not 0 <= micro_step < self.base.gradient_accumulation:
            raise SamplingContractError("invalid micro step")
        address = self.describe_step(optimizer_step)
        name = address["source_name"]
        source_start, source_stop = self.base.source_spans[name]
        rank_start, rank_length = self.base._rank_ranges[name]
        local_start = (address["source_batch_in_epoch"] * self.base.local_batch
                       + micro_step * self.base.micro_batch_size)
        epoch_start = address["source_epoch"] * (source_stop - source_start)
        permutation = self.base._permutations[name]
        return [
            source_start + permutation.at(
                epoch_start + rank_start + (local_start + i) % rank_length
            )
            for i in range(self.base.micro_batch_size)
        ]

    def __iter__(self):
        for step in range(self.start_optimizer_step,
                          self.start_optimizer_step + self.num_optimizer_steps):
            for micro in range(self.base.gradient_accumulation):
                yield self.batch_for_step(step, micro)

    def remaining_plan_counts(self):
        counts = Counter()
        for step in range(TRANSITION_STEP, TOTAL_STEPS):
            counts[self.describe_step(step)["source_name"]] += 1
        if sum(counts.values()) != TOTAL_STEPS - TRANSITION_STEP:
            raise SamplingContractError("remaining sampling plan lost updates")
        return dict(counts)
