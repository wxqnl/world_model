from __future__ import annotations

import numpy as np
import pytest

from wm3d_v3.data.window_selection import (
    WindowSelectionError,
    select_observed_world_window,
)


def test_window_uses_only_observed_irregular_times_and_real_leading_boundary() -> None:
    clock = np.asarray(
        [0.00, 0.03, 0.11, 0.19, 0.34, 0.51, 0.77, 1.05, 1.31, 1.62],
        dtype=np.float64,
    )
    result = select_observed_world_window(
        clock,
        anchor_index=6,
        context_samples=4,
        future_samples=2,
        context_horizon_s=0.65,
        future_horizon_s=0.85,
        minimum_horizon_coverage=0.6,
    )
    assert result.context_indices[-1] == 6
    assert result.leading_boundary_index == result.context_indices[0] - 1
    assert len(result.action_boundary_indices) == 4 + 2 + 1
    np.testing.assert_array_equal(result.world_times_s, clock[result.world_indices])
    np.testing.assert_array_equal(
        result.action_boundaries_s, clock[result.action_boundary_indices]
    )
    assert not np.allclose(np.diff(result.world_times_s), np.diff(result.world_times_s)[0])


def test_window_does_not_pad_when_future_horizon_or_count_is_unavailable() -> None:
    clock = np.arange(8, dtype=np.float64) * 0.1
    with pytest.raises(WindowSelectionError, match="observed states"):
        select_observed_world_window(
            clock,
            anchor_index=5,
            context_samples=3,
            future_samples=3,
            context_horizon_s=0.3,
            future_horizon_s=0.3,
            minimum_horizon_coverage=0.8,
        )


def test_window_rejects_fake_or_nonmonotonic_clock() -> None:
    with pytest.raises(WindowSelectionError, match="strictly increasing"):
        select_observed_world_window(
            [0.0, 0.1, 0.1, 0.3],
            anchor_index=2,
            context_samples=1,
            future_samples=1,
            context_horizon_s=0.1,
            future_horizon_s=0.1,
            minimum_horizon_coverage=0.5,
        )


def test_window_supports_explicit_real_future_offsets_without_interpolation() -> None:
    clock = np.arange(200, dtype=np.float64) * 0.05
    anchor = 70
    offsets = [0.6, 1.4, 2.0, 2.8, 3.4, 4.2, 4.8, 5.6]
    result = select_observed_world_window(
        clock,
        anchor_index=anchor,
        context_samples=16,
        future_samples=8,
        context_horizon_s=3.2,
        future_horizon_s=5.6,
        minimum_horizon_coverage=0.9,
        future_offsets_s=offsets,
    )
    np.testing.assert_array_equal(
        result.future_indices,
        anchor + np.asarray([12, 28, 40, 56, 68, 84, 96, 112]),
    )
    np.testing.assert_allclose(
        result.world_times_s[-8:] - clock[anchor], offsets, rtol=0, atol=1e-12
    )


def test_window_rejects_invalid_explicit_future_offsets() -> None:
    clock = np.arange(100, dtype=np.float64) * 0.1
    with pytest.raises(WindowSelectionError, match="last future offset"):
        select_observed_world_window(
            clock,
            anchor_index=40,
            context_samples=4,
            future_samples=2,
            context_horizon_s=0.4,
            future_horizon_s=1.0,
            minimum_horizon_coverage=0.8,
            future_offsets_s=[0.2, 0.9],
        )
