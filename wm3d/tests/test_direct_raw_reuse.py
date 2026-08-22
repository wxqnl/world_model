from __future__ import annotations

import torch

from wm3d.data.direct_raw import (
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
