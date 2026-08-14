from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from wm3d.data.step_sampler import EpisodeLocalPermutation, StepAddressedBatchSampler
from wm3d.data.streaming_raw import StreamingRawError, _StreamingEpisodeCache
from wm3d.data.unified_cache_dataset import CacheDataError, _ShardStore


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


def test_dynamic_shard_registration_rejects_digest_drift(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    store = _ShardStore(root, {}, verify_on_open=True)
    store.register("episode/features.safetensors", "a" * 64)
    store.register("episode/features.safetensors", "a" * 64)
    with pytest.raises(CacheDataError, match="changed digest"):
        store.register("episode/features.safetensors", "b" * 64)


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
