from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import wm3d.data.episode_io as episode_io
from wm3d.data.episode_io import (
    EpisodeIOError,
    VerifiedAssetStore,
    select_episode_cache_rows,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_asset_store_hashes_unchanged_shared_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = tmp_path / "shared.parquet"
    asset.write_bytes(b"immutable-source")
    expected = _sha(asset)
    calls = 0
    original = episode_io.sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(episode_io, "sha256_file", counted)
    store = VerifiedAssetStore()
    assert store.verify(tmp_path, asset.name, expected) == asset
    assert store.verify(tmp_path, asset.name, expected) == asset
    assert calls == 1


def test_verified_asset_store_rehashes_and_rejects_mutation(tmp_path: Path) -> None:
    asset = tmp_path / "shared.parquet"
    asset.write_bytes(b"immutable-source")
    expected = _sha(asset)
    store = VerifiedAssetStore()
    store.verify(tmp_path, asset.name, expected)
    # Use a different size because some network filesystems expose a coarse
    # mtime/ctime granularity.  Same-fingerprint replacement is prohibited by
    # the formal read-only raw-data mount requirement, while first use and
    # every observable mutation still receive a full SHA check.
    asset.write_bytes(b"mutated-source-with-different-size")
    with pytest.raises(EpisodeIOError, match="SHA mismatch"):
        store.verify(tmp_path, asset.name, expected)


def test_verified_asset_store_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with pytest.raises(EpisodeIOError, match="escapes source root"):
        VerifiedAssetStore().verify(root, "../outside.bin", _sha(outside))


def test_cache_row_selection_preserves_float32_five_hz_clock() -> None:
    clock = np.arange(117, dtype=np.float32) / np.float32(5.0)
    rows = select_episode_cache_rows(clock, minimum_separation_s=0.2)
    np.testing.assert_array_equal(rows, np.arange(clock.size, dtype=np.int64))


def test_cache_row_selection_still_rejects_materially_short_spacing() -> None:
    clock = np.arange(20, dtype=np.float64) * 0.199
    rows = select_episode_cache_rows(clock, minimum_separation_s=0.2)
    assert len(rows) < len(clock)


def test_decode_episode_views_parallelizes_real_cameras_without_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = []
    views = []
    for name in ("head", "left_wrist", "right_wrist"):
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(name.encode())
        role = f"rgb/{name}"
        assets.append((role, path.name, _sha(path)))
        views.append((name, role, "entire_file", None, None))
    task = SimpleNamespace(
        assets=tuple(assets),
        views=tuple(views),
        observation_samples=4,
    )
    calls: list[str] = []

    def fake_decode(path: Path, **_kwargs):
        calls.append(path.stem)
        value = ("head", "left_wrist", "right_wrist").index(path.stem)
        frames = np.full((4, 2, 2, 3), value, dtype=np.uint8)
        return frames, np.arange(4, dtype=np.float64)

    monkeypatch.setattr(episode_io, "_decode_segment", fake_decode)
    decoded, evidence = episode_io.decode_episode_views(
        task=task,
        source_root=tmp_path,
        canonical_view_slots=("head", "left_wrist", "right_wrist"),
        selected_observation_rows=[0, 2, 3],
        decode_workers=3,
    )
    assert set(calls) == {"head", "left_wrist", "right_wrist"}
    assert list(decoded) == ["head", "left_wrist", "right_wrist"]
    assert list(evidence) == ["head", "left_wrist", "right_wrist"]
    assert decoded["right_wrist"].frames.shape == (3, 2, 2, 3)


def test_decode_episode_views_drops_exactly_one_unaddressable_tail_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "head.mp4"
    path.write_bytes(b"head")
    task = SimpleNamespace(
        assets=(("rgb/head", path.name, _sha(path)),),
        views=(("head", "rgb/head", "entire_file", None, None),),
        observation_samples=4,
    )

    def fake_decode(_path: Path, **_kwargs):
        frames = np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1)
        frames = np.repeat(frames, 3, axis=3)
        return frames, np.arange(5, dtype=np.float64)

    monkeypatch.setattr(episode_io, "_decode_segment", fake_decode)
    decoded, evidence = episode_io.decode_episode_views(
        task=task,
        source_root=tmp_path,
        canonical_view_slots=("head",),
        selected_observation_rows=[0, 3],
        decode_workers=1,
    )
    np.testing.assert_array_equal(decoded["head"].frames[:, 0, 0, 0], [0, 3])
    np.testing.assert_array_equal(decoded["head"].recorded_pts_s, [0.0, 1.0, 2.0, 3.0])
    assert evidence["head"]["container_decoded_frame_count"] == 5
    assert evidence["head"]["decoded_frame_count"] == 4
    assert evidence["head"]["trailing_frames_dropped"] == 1


def test_decode_episode_views_still_rejects_larger_frame_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "head.mp4"
    path.write_bytes(b"head")
    task = SimpleNamespace(
        assets=(("rgb/head", path.name, _sha(path)),),
        views=(("head", "rgb/head", "entire_file", None, None),),
        observation_samples=4,
    )

    def fake_decode(_path: Path, **_kwargs):
        return np.zeros((6, 1, 1, 3), dtype=np.uint8), np.arange(6, dtype=np.float64)

    monkeypatch.setattr(episode_io, "_decode_segment", fake_decode)
    with pytest.raises(EpisodeIOError, match="ordinal binding failed"):
        episode_io.decode_episode_views(
            task=task,
            source_root=tmp_path,
            canonical_view_slots=("head",),
            selected_observation_rows=[0, 3],
            decode_workers=1,
        )


def test_decode_episode_views_rejects_nonpositive_worker_count(tmp_path: Path) -> None:
    task = SimpleNamespace(assets=(), views=(), observation_samples=2)
    with pytest.raises(EpisodeIOError, match="decode_workers"):
        episode_io.decode_episode_views(
            task=task,
            source_root=tmp_path,
            canonical_view_slots=("head",),
            selected_observation_rows=[0, 1],
            decode_workers=0,
        )
