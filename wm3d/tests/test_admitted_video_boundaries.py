from __future__ import annotations

import numpy as np
import pytest

from wm3d.data.episode_io import EpisodeIOError
from wm3d.data.window_video import _segment_ordinals


def segment(timestamps, start, stop, count, strict=True):
    return _segment_ordinals(
        timestamps=timestamps, segment_kind="recorded_pts_range",
        start_s=start, stop_s=stop, observation_samples=count,
        view_name="left_wrist", strict_admitted=strict,
    )


def test_actual_robotwin_float64_boundary_roundoff():
    # Episode 14129, left-wrist file-059, native 30 Hz PTS.
    timestamps = np.arange(45232, dtype=np.float64) / 30.0
    args = (timestamps, 1248.4333333333343, 1256.5666666666677, 244)
    old = segment(*args, strict=False)
    assert old[2:] == (245, 1, 0)
    fixed = segment(*args)
    np.testing.assert_array_equal(fixed[0], np.arange(37453, 37697))
    assert fixed[2:] == (244, 0, 0)


@pytest.mark.parametrize("drift", [-5.0e-12, 5.0e-12])
def test_admitted_shared_boundaries_choose_same_ordinals(drift):
    timestamps = np.arange(40000, dtype=np.float64) / 30.0
    fixed = segment(timestamps, 1200.0 + drift, 1201.0 + drift, 30)
    np.testing.assert_array_equal(fixed[0], np.arange(36000, 36030))
    assert fixed[2:] == (30, 0, 0)


@pytest.mark.parametrize("count", [29, 31])
def test_admitted_real_cardinality_mismatch_still_rejected(count):
    with pytest.raises(EpisodeIOError, match="no dropped or synthetic"):
        segment(np.arange(90, dtype=np.float64) / 30.0, 1.0, 2.0, count)


def test_admitted_non_roundoff_offset_is_not_snapped():
    with pytest.raises(EpisodeIOError, match="no dropped or synthetic"):
        segment(np.arange(90, dtype=np.float64) / 30.0, 1.001, 2.0, 30)


def test_legacy_float32_hour_boundary_rule_unchanged():
    timestamps = (np.float32(6127.5)
                  + np.arange(20, dtype=np.float32) * np.float32(.05)).astype(np.float64)
    result = segment(timestamps, 6127.55, 6127.95, 8, strict=False)
    np.testing.assert_array_equal(result[0], np.arange(2, 9).tolist() + [8])
    assert result[2:] == (7, 0, 1)
