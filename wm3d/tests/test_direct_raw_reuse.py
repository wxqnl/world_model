from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import torch

from wm3d.data.direct_raw import (
    DirectRawDataset,
    _DirectEpisode,
    _DirectWindowSource,
    _PreparedViewRowStore,
    _apply_ignored_action_dimensions,
)


def test_prepared_view_row_store_reuses_rows_and_evicts_by_bytes() -> None:
    images = torch.arange(
        2 * 2 * 3 * 4 * 4,
        dtype=torch.uint8,
    ).reshape(2, 2, 3, 4, 4)
    mask = torch.tensor([[True, True], [True, False]])
    row_bytes = images[0].numel() + mask[0].numel()
    store = _PreparedViewRowStore(maximum_bytes=row_bytes)

    store.put_many((10, 11), images, mask)
    cached = store.get_many((11, 12))

    assert set(cached) == {11}
    torch.testing.assert_close(cached[11][0], images[1])
    torch.testing.assert_close(cached[11][1], mask[1])
    assert store.metrics == {
        "prepared_row_cache_bytes": row_bytes,
        "prepared_row_cache_entries": 1,
        "prepared_row_cache_hits": 1,
        "prepared_row_cache_misses": 1,
        "prepared_row_cache_evictions": 1,
    }



def test_ignored_action_dimension_is_removed_from_every_training_lane() -> None:
    action = {
        "history_fine_action_mask": torch.ones(2, 1, 3, 4, dtype=torch.bool),
        "future_factual_fine_action_mask": torch.ones(2, 1, 3, 4, dtype=torch.bool),
        "target_fine_action_mask": torch.ones(1, 3, 4, dtype=torch.bool),
        "history_coarse_action_mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "future_factual_coarse_action_mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "target_coarse_action_mask": torch.ones(2, 1, 4, dtype=torch.bool),
        "action_semantic_ids": torch.ones(1, 4, dtype=torch.int64),
        "composition_operator_ids": torch.ones(1, 4, dtype=torch.int64),
    }

    _apply_ignored_action_dimensions(action, {0: (3,)})

    for name in (
        "history_fine_action_mask",
        "future_factual_fine_action_mask",
        "target_fine_action_mask",
        "history_coarse_action_mask",
        "future_factual_coarse_action_mask",
        "target_coarse_action_mask",
    ):
        assert not action[name][..., 3].any()
        assert action[name][..., :3].all()
    assert action["action_semantic_ids"][0, 3].item() == 0
    assert action["composition_operator_ids"][0, 3].item() == 0


def test_direct_window_source_coalesces_overlapping_episode_rows(monkeypatch) -> None:
    source = object.__new__(_DirectWindowSource)
    source.profile = SimpleNamespace(
        cache_representation={"view_slots": ("front",)}
    )
    source.sources = {"demo": SimpleNamespace(raw_root=Path("/unused"))}
    source.adapters = {
        "demo": SimpleNamespace(
            views=(SimpleNamespace(name="front", color_order="rgb"),)
        )
    }
    source.asset_verifier = object()
    source.video_indices = object()
    source.decode_workers = 1
    source.input_rgb_size = 14
    source.prepared_rows = _PreparedViewRowStore(maximum_bytes=1 << 20)
    source._metadata_lock = threading.RLock()
    source._decode_locks = tuple(threading.Lock() for _ in range(4))
    source._task_ordinal_by_id = {"task": 7}
    prepared_robot = object()
    episode = _DirectEpisode(
        selected_rows=np.arange(32, dtype=np.int64),
        prepared_robot=prepared_robot,
    )
    task = SimpleNamespace(source="demo")
    source._load_task = lambda _task_id: task
    source._load_episode = lambda _task: episode
    source.windows_decoded = 0
    source.decode_calls = 0
    source.coalesced_batches = 0
    source.coalesced_requested_rows = 0
    source.coalesced_unique_rows = 0
    source.decode_seconds = 0.0

    decoded_rows: list[tuple[int, ...]] = []

    def fake_decode_episode_window_views(**kwargs):
        rows = tuple(int(value) for value in kwargs["selected_observation_rows"])
        decoded_rows.append(rows)
        return rows, object()

    def fake_view_batch(*, decoded, **_kwargs):
        rows = torch.tensor(decoded, dtype=torch.float32)
        images = rows[:, None, None, None, None].expand(-1, 1, 3, 14, 14) / 255
        return images, torch.ones(len(decoded), 1, dtype=torch.bool)

    monkeypatch.setattr(
        "wm3d.data.direct_raw.decode_episode_window_views",
        fake_decode_episode_window_views,
    )
    monkeypatch.setattr(
        "scripts.data.run_cache_worker._view_batch",
        fake_view_batch,
    )
    first = SimpleNamespace(
        leading_feature_row=0,
        context_feature_rows=(1, 2),
        future_feature_rows=(3,),
    )
    second = SimpleNamespace(
        leading_feature_row=1,
        context_feature_rows=(2, 3),
        future_feature_rows=(4,),
    )

    result = source.decode_windows(((10, first, "task"), (11, second, "task")))

    assert decoded_rows == [(1, 2, 3, 4)]
    torch.testing.assert_close(
        result[10][0][:, 0, 0, 0, 0],
        torch.tensor([1, 2, 3], dtype=torch.uint8),
    )
    torch.testing.assert_close(
        result[11][0][:, 0, 0, 0, 0],
        torch.tensor([2, 3, 4], dtype=torch.uint8),
    )
    assert result[10][-1] is prepared_robot

    repeated = source.decode_windows(
        ((12, first, "task"), (13, second, "task"))
    )
    assert decoded_rows == [(1, 2, 3, 4)]
    torch.testing.assert_close(result[10][0], repeated[12][0])
    torch.testing.assert_close(result[11][0], repeated[13][0])
    assert source.windows_decoded == 4
    assert source.decode_calls == 1
    assert source.coalesced_requested_rows == 12
    assert source.coalesced_unique_rows == 8


def test_direct_dataset_prefetch_submits_one_coalesced_batch() -> None:
    dataset = object.__new__(DirectRawDataset)
    dataset.entries = [
        SimpleNamespace(feature_shard="a"),
        SimpleNamespace(feature_shard="b"),
    ]
    dataset._task_id_by_feature = {"a": "task-a", "b": "task-b"}
    calls: list[tuple[int, ...]] = []

    def decode_windows(requests):
        indices = tuple(index for index, _entry, _task_id in requests)
        calls.append(indices)
        return {index: (index,) for index in indices}

    dataset._source = SimpleNamespace(decode_windows=decode_windows)
    dataset.max_prefetch_windows = 32
    dataset._executor = ThreadPoolExecutor(max_workers=1)
    dataset._futures = {}
    dataset.prefetch_submitted = 0
    dataset.prefetch_consumed = 0
    dataset.prefetch_capacity_skips = 0
    dataset.prefetch_wait_seconds = 0.0
    try:
        dataset.prefetch_indices([0, 1])
        assert dataset._window(0) == (0,)
        assert dataset._window(1) == (1,)
    finally:
        dataset._executor.shutdown(wait=True)

    assert calls == [(0, 1)]
    assert dataset.prefetch_submitted == 2
    assert dataset.prefetch_consumed == 2
