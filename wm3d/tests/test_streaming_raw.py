from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wm3d.data.step_sampler import EpisodeLocalPermutation, StepAddressedBatchSampler
from wm3d.data.streaming_raw import StreamingRawError, _StreamingEpisodeCache
from wm3d.data.unified_cache_dataset import CacheDataError, _ShardStore
from wm3d.training.pretrain import (
    PretrainError,
    _StreamingLookaheadBatchSampler,
    _relative_world_times_for_model,
)


def test_episode_local_permutation_is_bijective_and_local() -> None:
    spans = ((100, 106), (106, 111), (111, 120))
    permutation = EpisodeLocalPermutation(
        source_start=100,
        source_stop=120,
        episode_spans=spans,
        seed=17,
        source_name="robot",
    )
    first = [permutation.at(index) for index in range(20)]
    second = [permutation.at(index + 20) for index in range(20)]
    assert sorted(first) == list(range(20))
    assert sorted(second) == list(range(20))
    assert first != second
    episode = {}
    for episode_id, (start, stop) in enumerate(spans):
        for value in range(start - 100, stop - 100):
            episode[value] = episode_id
    transitions = sum(
        episode[left] != episode[right] for left, right in zip(first, first[1:])
    )
    assert transitions <= len(spans) - 1


def test_episode_local_sampler_resume_is_exact() -> None:
    kwargs = {
        "source_spans": {"robot": (0, 24)},
        "source_order": ("robot",),
        "source_weights": {"robot": 1},
        "world_size": 2,
        "rank": 1,
        "micro_batch_size": 2,
        "gradient_accumulation": 2,
        "num_optimizer_steps": 3,
        "seed": 991,
        "source_episode_spans": {
            "robot": ((0, 8), (8, 16), (16, 24))
        },
    }
    full = list(
        StepAddressedBatchSampler(start_optimizer_step=0, **kwargs)
    )
    resumed = list(
        StepAddressedBatchSampler(
            start_optimizer_step=1,
            **{**kwargs, "num_optimizer_steps": 2},
        )
    )
    assert resumed == full[2:]
    for offset in range(0, len(full), 2):
        flattened = full[offset] + full[offset + 1]
        assert len(flattened) == len(set(flattened))


def test_episode_local_sampler_partitions_whole_episodes_across_ranks() -> None:
    spans = tuple((start, start + 6) for start in range(0, 96, 6))
    kwargs = {
        "source_spans": {"robot": (0, 96)},
        "source_order": ("robot",),
        "source_weights": {"robot": 1},
        "world_size": 4,
        "micro_batch_size": 2,
        "gradient_accumulation": 1,
        "start_optimizer_step": 0,
        "num_optimizer_steps": 12,
        "seed": 812,
        "source_episode_spans": {"robot": spans},
    }
    episode_by_window = {
        window: episode
        for episode, (start, stop) in enumerate(spans)
        for window in range(start, stop)
    }
    samples_by_rank = [
        [
            item
            for batch in StepAddressedBatchSampler(rank=rank, **kwargs)
            for item in batch
        ]
        for rank in range(4)
    ]
    episodes_by_rank = [
        {episode_by_window[item] for item in samples}
        for samples in samples_by_rank
    ]

    assert all(len(samples) == len(set(samples)) == 24 for samples in samples_by_rank)
    assert len(set().union(*episodes_by_rank)) == 16
    for left in range(4):
        for right in range(left + 1, 4):
            assert episodes_by_rank[left].isdisjoint(episodes_by_rank[right])
    # Each rank consumes all six windows from its first episode before moving on.
    assert all(
        len({episode_by_window[item] for item in samples[:6]}) == 1
        for samples in samples_by_rank
    )


def test_episode_local_sampler_falls_back_for_tiny_sources() -> None:
    kwargs = {
        "source_spans": {"tiny": (0, 40)},
        "source_order": ("tiny",),
        "source_weights": {"tiny": 1},
        "world_size": 4,
        "micro_batch_size": 2,
        "gradient_accumulation": 1,
        "start_optimizer_step": 0,
        "num_optimizer_steps": 1,
        "seed": 44,
        "source_episode_spans": {"tiny": ((0, 20), (20, 40))},
    }
    by_rank = [
        [
            item
            for batch in StepAddressedBatchSampler(rank=rank, **kwargs)
            for item in batch
        ]
        for rank in range(4)
    ]
    flattened = [item for samples in by_rank for item in samples]
    assert len(flattened) == len(set(flattened)) == 8


def test_dynamic_shard_registration_rejects_digest_drift(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    store = _ShardStore(root, {}, verify_on_open=True)
    store.register("episode/features.safetensors", "a" * 64)
    store.register("episode/features.safetensors", "a" * 64)
    with pytest.raises(CacheDataError, match="changed digest"):
        store.register("episode/features.safetensors", "b" * 64)


def test_dynamic_shard_registration_accepts_verified_lru_replacement(
    tmp_path: Path,
) -> None:
    relative = "episode/features.safetensors"
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(b"first")
    store = _ShardStore(tmp_path.resolve(), {}, verify_on_open=True)
    store.register(relative, "a" * 64, verified=True)
    assert store.path(relative) == path

    path.unlink()
    path.write_bytes(b"replacement")
    store.register(
        relative,
        "b" * 64,
        verified=True,
        allow_verified_replacement=True,
    )

    assert store.expected_sha[relative] == "b" * 64
    assert relative not in store._resolved
    assert relative in store._verified
    assert store.path(relative) == path


def test_streaming_hot_hit_uses_verified_file_identity(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    payloads = ("feature.bin", "robot.bin", "rgb.bin")
    for name in payloads:
        (root / name).write_bytes(name.encode("utf-8"))
    entry = SimpleNamespace(
        feature_shard=payloads[0],
        robot_shard=payloads[1],
        rgb_pack=payloads[2],
    )
    task = SimpleNamespace(task_id="episode")
    cache = object.__new__(_StreamingEpisodeCache)
    cache.root = root
    cache._entries = OrderedDict(
        {task.task_id: (entry, sum((root / name).stat().st_size for name in payloads))}
    )
    cache._verified_payload_identity = {
        task.task_id: cache._payload_identity(entry)
    }
    cache.cache_hits = 0

    assert cache.ensure(task) is entry
    assert cache.cache_hits == 1

    (root / payloads[0]).write_bytes(b"different")
    with pytest.raises(StreamingRawError, match="changed after"):
        cache.ensure(task)


def test_streaming_batch_sampler_primes_two_future_batches() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.prefetched: list[tuple[int, ...]] = []

        def prefetch_indices(self, indices: list[int]) -> None:
            self.prefetched.append(tuple(indices))

    dataset = Dataset()
    sampler = _StreamingLookaheadBatchSampler(
        [[0, 1], [2, 3], [4, 5], [6, 7]], dataset, lookahead_batches=2
    )
    iterator = iter(sampler)

    assert next(iterator) == [0, 1]
    assert dataset.prefetched == [(0, 1), (2, 3), (4, 5)]
    assert list(iterator) == [[2, 3], [4, 5], [6, 7]]
    assert dataset.prefetched == [(0, 1), (2, 3), (4, 5), (6, 7)]


def test_world_times_are_recentered_before_bf16_model_input_cast() -> None:
    original = 252.0 + torch.arange(24, dtype=torch.float64) * 0.2
    world_times = original.unsqueeze(0)

    # At this episode offset, the root FSDP BF16 cast merges adjacent frames.
    assert not bool(torch.diff(world_times.to(torch.bfloat16), dim=1).gt(0).all())

    relative = _relative_world_times_for_model(world_times, context_length=16)

    assert relative.dtype == torch.float64
    assert relative[0, 15].item() == 0.0
    assert bool(torch.diff(relative.to(torch.bfloat16), dim=1).gt(0).all())
    torch.testing.assert_close(world_times, original.unsqueeze(0), rtol=0, atol=0)


def test_world_time_recentering_rejects_nonmonotonic_input() -> None:
    world_times = torch.tensor([[0.0, 0.2, 0.2, 0.4]], dtype=torch.float64)
    with pytest.raises(PretrainError, match="strictly increasing"):
        _relative_world_times_for_model(world_times, context_length=2)
