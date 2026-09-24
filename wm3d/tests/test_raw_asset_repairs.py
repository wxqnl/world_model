from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from wm3d.data.episode_io import EpisodeIOError, VerifiedAssetStore
from wm3d.data.manifest_contract import sha256_file
from wm3d.data.raw_asset_repairs import (
    ENVIRONMENT, SCHEMA, active_raw_asset_repairs, repair_resume_expectation,
)
from wm3d.training.distributed_checkpoint import CheckpointIntegrityError, _validate_metadata
from tests.test_distributed_resume_contract import _metadata, _expected


def _install_contract(tmp_path: Path, monkeypatch):
    relative = "videos/camera/file.mp4"
    original = tmp_path / relative
    original.parent.mkdir(parents=True)
    original.write_bytes(b"sealed-original-video")
    replacement_relative = ".wm3d_rgb_repairs/official_v1/" + relative
    replacement = tmp_path / replacement_relative
    replacement.parent.mkdir(parents=True)
    replacement.write_bytes(b"approved-upstream-repaired-video")
    contract = {
        "schema": SCHEMA, "repair_id": "official_v1",
        "upstream": "https://example.org/official-repair", "user_approved": True,
        "adoption_run_root": str(tmp_path / "run"), "adoption_checkpoint_step": 2000,
        "assets": [{
            "source_root": str(tmp_path), "relative_path": relative,
            "sealed_digest": sha256_file(original),
            "replacement_relative_path": replacement_relative,
            "replacement_digest": sha256_file(replacement),
            "decoded_frames": 10, "time_base": "1/15360", "timestamps_unchanged": True,
        }],
    }
    path = tmp_path / "repair.json"
    path.write_text(json.dumps(contract))
    monkeypatch.setenv(ENVIRONMENT, str(path))
    return contract, path, original, replacement


def test_approved_repair_preserves_original_and_verifies_both_assets(tmp_path, monkeypatch):
    contract, _, original, replacement = _install_contract(tmp_path, monkeypatch)
    item = contract["assets"][0]
    store = VerifiedAssetStore()
    actual = store.verify(tmp_path, item["relative_path"], item["sealed_digest"])
    assert actual == replacement
    assert original.read_bytes() == b"sealed-original-video"
    assert store.content_digest(actual, item["sealed_digest"]) == item["replacement_digest"]
    replacement.write_bytes(b"unapproved-change")
    with pytest.raises(EpisodeIOError, match="mismatch"):
        store.verify(tmp_path, item["relative_path"], item["sealed_digest"])


def test_repair_does_not_bypass_original_seal(tmp_path, monkeypatch):
    contract, _, original, _ = _install_contract(tmp_path, monkeypatch)
    original.write_bytes(b"changed-original")
    item = contract["assets"][0]
    with pytest.raises(EpisodeIOError, match="mismatch"):
        VerifiedAssetStore().verify(tmp_path, item["relative_path"], item["sealed_digest"])


@pytest.mark.parametrize("mutation", [
    {"replacement_relative_path": "../escape.mp4"},
    {"relative_path": "data/actions.parquet"},
    {"timestamps_unchanged": False},
])
def test_repair_rejects_unscoped_or_unaligned_assets(tmp_path, monkeypatch, mutation):
    contract, path, _, _ = _install_contract(tmp_path, monkeypatch)
    contract["assets"][0].update(mutation)
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):
        active_raw_asset_repairs()


def test_checkpoint_binds_repair_and_allows_only_approved_adoption(tmp_path, monkeypatch):
    contract, _, _, _ = _install_contract(tmp_path, monkeypatch)
    root = contract["adoption_run_root"]
    assert repair_resume_expectation(root, 2000) is None
    assert repair_resume_expectation(root, 5000) == contract
    with pytest.raises(ValueError, match="earlier"):
        repair_resume_expectation(root, 1000)
    with pytest.raises(ValueError, match="different formal run"):
        repair_resume_expectation(str(tmp_path / "other"), 5000)
    metadata = dict(_metadata(), raw_asset_repairs=contract)
    with pytest.raises(CheckpointIntegrityError, match="raw_asset_repairs"):
        _validate_metadata(metadata, _expected())
    expected = replace(_expected(), raw_asset_repairs=contract)
    assert _validate_metadata(metadata, expected) == "exact"
    changed = deepcopy(metadata)
    changed["raw_asset_repairs"]["assets"][0]["decoded_frames"] += 1
    with pytest.raises(CheckpointIntegrityError, match="raw_asset_repairs"):
        _validate_metadata(changed, expected)
    with pytest.raises(CheckpointIntegrityError, match="raw_asset_repairs"):
        _validate_metadata(_metadata(), expected)
