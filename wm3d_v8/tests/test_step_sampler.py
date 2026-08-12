from __future__ import annotations

from collections import Counter

from wm3d_v3.data.step_sampler import ExactSourceSchedule, StepAddressedBatchSampler


def test_source_schedule_matches_integer_weights_exactly_per_cycle() -> None:
    schedule = ExactSourceSchedule(
        ["single", "bimanual", "wholebody"],
        {"single": 2, "bimanual": 3, "wholebody": 1},
        seed=19,
    )
    for cycle in range(4):
        counts = Counter(
            schedule.address(cycle * schedule.cycle_length + offset).source_name
            for offset in range(schedule.cycle_length)
        )
        assert counts == {"single": 2, "bimanual": 3, "wholebody": 1}


def test_resume_is_directly_addressed_by_optimizer_step() -> None:
    kwargs = dict(
        source_spans={"a": (0, 100), "b": (100, 220)},
        source_order=["a", "b"],
        source_weights={"a": 2, "b": 1},
        world_size=2,
        rank=1,
        micro_batch_size=3,
        gradient_accumulation=2,
        seed=41,
    )
    uninterrupted = StepAddressedBatchSampler(
        **kwargs, start_optimizer_step=0, num_optimizer_steps=7
    )
    resumed = StepAddressedBatchSampler(
        **kwargs, start_optimizer_step=4, num_optimizer_steps=3
    )
    full_batches = list(uninterrupted)
    assert list(resumed) == full_batches[4 * 2 :]


def test_ranks_receive_disjoint_samples_within_one_optimizer_step() -> None:
    base = dict(
        source_spans={"dual": (0, 64)},
        source_order=["dual"],
        source_weights={"dual": 1},
        world_size=2,
        micro_batch_size=4,
        gradient_accumulation=2,
        start_optimizer_step=0,
        num_optimizer_steps=1,
        seed=3,
    )
    rank0 = [item for batch in StepAddressedBatchSampler(**base, rank=0) for item in batch]
    rank1 = [item for batch in StepAddressedBatchSampler(**base, rank=1) for item in batch]
    assert len(set(rank0 + rank1)) == 16
    assert set(rank0).isdisjoint(rank1)


def _global_step_samples(
    *, world_size: int, micro_batch_size: int, gradient_accumulation: int
) -> list[int]:
    by_rank = [
        list(
            StepAddressedBatchSampler(
                source_spans={"dual": (0, 128)},
                source_order=["dual"],
                source_weights={"dual": 1},
                world_size=world_size,
                rank=rank,
                micro_batch_size=micro_batch_size,
                gradient_accumulation=gradient_accumulation,
                start_optimizer_step=7,
                num_optimizer_steps=1,
                seed=31,
            )
        )
        for rank in range(world_size)
    ]
    result: list[int] = []
    for micro_step in range(gradient_accumulation):
        for rank in range(world_size):
            result.extend(by_rank[rank][micro_step])
    return result


def test_topology_change_preserves_exact_global_sample_sequence() -> None:
    """A legal DCP reshard may repartition, never replace, a global batch."""

    two_rank = _global_step_samples(
        world_size=2, micro_batch_size=2, gradient_accumulation=2
    )
    four_rank = _global_step_samples(
        world_size=4, micro_batch_size=2, gradient_accumulation=1
    )
    eight_rank = _global_step_samples(
        world_size=8, micro_batch_size=1, gradient_accumulation=1
    )
    assert two_rank == four_rank == eight_rank
    assert len(two_rank) == len(set(two_rank)) == 8
