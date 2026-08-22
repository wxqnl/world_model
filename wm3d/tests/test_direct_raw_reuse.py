from __future__ import annotations

import torch

from wm3d.data.direct_raw import _PreparedViewRowStore


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
