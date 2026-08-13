from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import wm3d.data.episode_io as episode_io
from wm3d.data.episode_io import EpisodeIOError, VerifiedAssetStore


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
