from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import wm3d.data.window_video as window_video
from wm3d.data.episode_io import VerifiedAssetStore
from wm3d.data.window_video import (
    VideoFrameIndex,
    VideoTimestampIndexStore,
    decode_episode_window_views,
)


class _Array:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def asnumpy(self) -> np.ndarray:
        return self.value


class _Reader:
    def __init__(
        self,
        timestamps: np.ndarray,
        *,
        timestamp_calls: list[int],
        batch_calls: list[list[int]],
    ) -> None:
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.timestamp_calls = timestamp_calls
        self.batch_calls = batch_calls

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_frame_timestamp(self, indices) -> _Array:
        values = np.asarray(list(indices), dtype=np.int64)
        self.timestamp_calls.append(len(values))
        starts = self.timestamps[values]
        return _Array(np.stack((starts, starts + 0.05), axis=-1))

    def get_batch(self, indices: list[int]) -> _Array:
        self.batch_calls.append(list(indices))
        frames = np.stack(
            [
                np.full((2, 3, 3), index, dtype=np.uint8)
                for index in indices
            ]
        )
        return _Array(frames)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(path: Path, *, observations: int, segment) -> SimpleNamespace:
    return SimpleNamespace(
        assets=(("rgb/head", path.name, _sha(path)),),
        views=(("head", "rgb/head", *segment),),
        observation_samples=observations,
    )




def _patch_timestamp_index(
    monkeypatch: pytest.MonkeyPatch,
    timestamps: np.ndarray,
) -> None:
    values = np.asarray(timestamps, dtype=np.float64)
    time_base_s = 1.0e-6
    pts = np.rint(values / time_base_s).astype(np.int64)
    monkeypatch.setattr(
        VideoTimestampIndexStore,
        "_build",
        staticmethod(
            lambda _path: VideoFrameIndex(
                pts=pts,
                time_base_s=time_base_s,
            )
        ),
    )

def test_window_decoder_reads_only_requested_entire_file_ordinals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "head.mp4"
    video.write_bytes(b"immutable fixture")
    timestamp_calls: list[int] = []
    batch_calls: list[list[int]] = []

    monkeypatch.setattr(
        window_video,
        "_open_video_reader",
        lambda _path: _Reader(
            np.arange(10, dtype=np.float64) * 0.1,
            timestamp_calls=timestamp_calls,
            batch_calls=batch_calls,
        ),
    )
    _patch_timestamp_index(
        monkeypatch,
        np.arange(10, dtype=np.float64) * 0.1,
    )
    store = VideoTimestampIndexStore(maximum_assets=2)
    decoded, evidence = decode_episode_window_views(
        task=_task(
            video,
            observations=10,
            segment=("entire_file", None, None),
        ),
        source_root=tmp_path,
        canonical_view_slots=("head",),
        selected_observation_rows=(1, 4, 8),
        asset_verifier=VerifiedAssetStore(),
        timestamp_indices=store,
        decode_workers=1,
    )

    assert batch_calls == [[1, 4, 8]]
    np.testing.assert_array_equal(
        decoded["head"].frames[:, 0, 0, 0],
        [1, 4, 8],
    )
    assert evidence["head"]["random_access_decoded_frame_count"] == 3
    assert timestamp_calls == []
    assert store.metrics["video_index_builds"] == 1


def test_window_decoder_maps_recorded_segment_rows_without_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "shared.mp4"
    video.write_bytes(b"shared immutable fixture")
    timestamp_calls: list[int] = []
    batch_calls: list[list[int]] = []
    timestamps = np.arange(20, dtype=np.float64) * 0.1

    monkeypatch.setattr(
        window_video,
        "_open_video_reader",
        lambda _path: _Reader(
            timestamps,
            timestamp_calls=timestamp_calls,
            batch_calls=batch_calls,
        ),
    )
    _patch_timestamp_index(monkeypatch, timestamps)
    store = VideoTimestampIndexStore(maximum_assets=2)
    task = _task(
        video,
        observations=8,
        segment=("recorded_pts_range", 0.5, 1.3),
    )
    decoded, _evidence = decode_episode_window_views(
        task=task,
        source_root=tmp_path,
        canonical_view_slots=("head",),
        selected_observation_rows=(0, 3, 7),
        asset_verifier=VerifiedAssetStore(),
        timestamp_indices=store,
        decode_workers=1,
    )
    assert batch_calls == [[5, 8, 12]]
    np.testing.assert_array_equal(
        decoded["head"].frames[:, 0, 0, 0],
        [5, 8, 12],
    )

    decode_episode_window_views(
        task=task,
        source_root=tmp_path,
        canonical_view_slots=("head",),
        selected_observation_rows=(1, 2, 6),
        asset_verifier=VerifiedAssetStore(),
        timestamp_indices=store,
        decode_workers=1,
    )
    assert batch_calls[-1] == [6, 7, 11]
    assert timestamp_calls == []
    assert store.metrics["video_index_hits"] == 1


def test_window_decoder_maps_float32_hour_scale_pts_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "long_shared.mp4"
    video.write_bytes(b"long shared immutable fixture")
    timestamp_calls: list[int] = []
    batch_calls: list[list[int]] = []
    timestamps = (
        np.float32(6127.5)
        + np.arange(20, dtype=np.float32) * np.float32(0.05)
    ).astype(np.float64)

    monkeypatch.setattr(
        window_video,
        "_open_video_reader",
        lambda _path: _Reader(
            timestamps,
            timestamp_calls=timestamp_calls,
            batch_calls=batch_calls,
        ),
    )
    _patch_timestamp_index(monkeypatch, timestamps)
    decoded, _evidence = decode_episode_window_views(
        task=_task(
            video,
            observations=8,
            segment=("recorded_pts_range", 6127.55, 6127.95),
        ),
        source_root=tmp_path,
        canonical_view_slots=("head",),
        selected_observation_rows=(0, 3, 7),
        asset_verifier=VerifiedAssetStore(),
        timestamp_indices=VideoTimestampIndexStore(maximum_assets=2),
        decode_workers=1,
    )

    assert batch_calls == [[2, 5, 8]]
    np.testing.assert_array_equal(
        decoded["head"].frames[:, 0, 0, 0],
        [2, 5, 8],
    )
