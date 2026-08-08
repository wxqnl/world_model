from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cache_robocasa365_v7_compact.py"
SPEC = importlib.util.spec_from_file_location("cache_robocasa365_v7_compact", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(episode: int, clip_hash: str, frames: int):
    return SimpleNamespace(
        native_episode_id=f"episode_{episode}",
        clip_hash=clip_hash,
        native_start_frame=0,
        native_end_frame=frames,
    )


def _metadata(anchor_file: int, wrist_file: int):
    return {
        "videos/observation.images.robot0_agentview_left/chunk_index": 0,
        "videos/observation.images.robot0_agentview_left/file_index": anchor_file,
        "videos/observation.images.robot0_eye_in_hand/chunk_index": 0,
        "videos/observation.images.robot0_eye_in_hand/file_index": wrist_file,
    }


def test_whole_video_lpt_is_deterministic_and_never_splits_group(tmp_path):
    records = [
        _record(0, "a0", 400),
        _record(1, "a1", 200),
        _record(2, "b0", 320),
        _record(3, "c0", 160),
    ]
    metadata = {
        0: _metadata(0, 0),
        1: _metadata(0, 0),
        2: _metadata(1, 1),
        3: _metadata(2, 2),
    }
    first = MODULE._partition_by_video_group(records, tmp_path, metadata, 2)
    second = MODULE._partition_by_video_group(records, tmp_path, metadata, 2)
    assert [[row.clip_hash for row in shard] for shard in first[0]] == [
        [row.clip_hash for row in shard] for shard in second[0]
    ]
    owner = {
        row.clip_hash: shard_id
        for shard_id, shard in enumerate(first[0])
        for row in shard
    }
    assert owner["a0"] == owner["a1"]
    assert sorted(owner) == ["a0", "a1", "b0", "c0"]
    assert sum(first[2]) == 3


def test_selection_hash_rejects_duplicate_clip_hash():
    with pytest.raises(RuntimeError, match="duplicate clip_hash"):
        MODULE._selection_sha256([_record(0, "same", 96), _record(1, "same", 96)])


def test_shard_index_name_is_automatic():
    path = MODULE._shard_index_path(Path("index.jsonl"), 3, 8)
    assert path.name == "index.shard-00003-of-00008.jsonl"
