from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from wm3d.data.grouped_robot import bimanual_arm_spec
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.data.source_inventory import (
    SourceInventoryError,
    _task_text,
    scan_lerobot_source,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_fixture(root: Path, *, duplicate_time: bool = False) -> Path:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/chunk-000/observation.images.top").mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            }
        )
    )
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 4, "task_index": 0}) + "\n"
    )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "insert with both arms"}) + "\n"
    )
    time = [0.0, 0.031, 0.079, 0.142]
    if duplicate_time:
        time[2] = time[1]
    table = pa.table(
        {
            "timestamp": pa.array(time, type=pa.float64()),
            "action": pa.array(
                np.arange(56, dtype=np.float32).reshape(4, 14).tolist(),
                type=pa.list_(pa.float32(), 14),
            ),
            "observation.state": pa.array(
                np.arange(56, dtype=np.float32).reshape(4, 14).tolist(),
                type=pa.list_(pa.float32(), 14),
            ),
        }
    )
    pq.write_table(table, root / "data/chunk-000/episode_000000.parquet")
    (root / "videos/chunk-000/observation.images.top/episode_000000.mp4").write_bytes(
        b"real-container-fixture"
    )
    adapter = root / "adapter.yaml"
    groups = []
    for name, start in (("left_arm", 0), ("right_arm", 7)):
        columns = list(range(start, start + 7))
        groups.append(
            {
                "group": name,
                "supervision": "fine_command",
                "action": [
                    {"key": "action", "columns": columns, "scale": [1] * 7, "offset": [0] * 7}
                ],
                "state": [
                    {
                        "key": "observation.state",
                        "columns": list(range(start, start + 7)) + [start, start, start],
                        "scale": [1] * 10,
                        "offset": [0] * 10,
                    }
                ],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
            }
        )
    adapter.write_text(
        yaml.safe_dump(
            {
                "schema": "wm3d_v8_source_adapter_v3",
                "name": "aloha_fixture",
                "raw_format": "lerobot_parquet_video",
                "observation_time_key": "timestamp",
                "views": [{"name": "overhead", "key": "observation.images.top"}],
                "groups": groups,
            },
            sort_keys=False,
        )
    )
    return adapter


def test_inventory_preserves_both_arms_and_recorded_irregular_clock(tmp_path: Path) -> None:
    adapter_path = _raw_fixture(tmp_path)
    adapter = load_adapter_contract(adapter_path, expected_sha256=_sha(adapter_path))
    rows, receipt = scan_lerobot_source(
        root=tmp_path,
        source="aloha",
        embodiment=bimanual_arm_spec(),
        adapter=adapter,
        split_seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
        default_task="fallback",
    )
    assert len(rows) == 1
    assert set(rows[0]["robot_groups"]) == {"left_arm", "right_arm"}
    clock = rows[0]["robot_groups"]["left_arm"]["action_clock"]
    assert clock["min_dt_s"] == pytest.approx(0.031)
    assert clock["max_dt_s"] == pytest.approx(0.063)
    assert rows[0]["observation_clock"]["timestamp_sha256"] == clock["timestamp_sha256"]
    assert "hz" not in json.dumps(rows[0]).lower()
    assert rows[0]["task_text"] == "insert with both arms"
    assert rows[0]["views"][0]["segment_kind"] == "entire_file"
    assert receipt["episode_count"] == 1


def test_blank_inline_task_falls_back_to_explicit_default() -> None:
    assert _task_text({"tasks": [""]}, {}, "fallback") == "fallback"


def test_inventory_rejects_non_monotonic_recorded_clock(tmp_path: Path) -> None:
    adapter_path = _raw_fixture(tmp_path, duplicate_time=True)
    adapter = load_adapter_contract(adapter_path, expected_sha256=_sha(adapter_path))
    with pytest.raises(SourceInventoryError, match="strictly increasing"):
        scan_lerobot_source(
            root=tmp_path,
            source="aloha",
            embodiment=bimanual_arm_spec(),
            adapter=adapter,
            split_seed=7,
            train_fraction=0.8,
            validation_fraction=0.1,
            default_task="fallback",
        )


def test_inventory_explicit_episode_selection_is_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    adapter_path = _raw_fixture(tmp_path)
    adapter = load_adapter_contract(adapter_path, expected_sha256=_sha(adapter_path))
    rows, receipt = scan_lerobot_source(
        root=tmp_path,
        source="aloha",
        embodiment=bimanual_arm_spec(),
        adapter=adapter,
        split_seed=7,
        train_fraction=0.8,
        validation_fraction=0.1,
        default_task="fallback",
        episode_indices=[0],
    )
    assert len(rows) == 1
    assert receipt["selection"]["mode"] == "explicit_episode_indices"
    assert receipt["selection"]["episode_indices"] == [0]
    with pytest.raises(SourceInventoryError, match="absent"):
        scan_lerobot_source(
            root=tmp_path,
            source="aloha",
            embodiment=bimanual_arm_spec(),
            adapter=adapter,
            split_seed=7,
            train_fraction=0.8,
            validation_fraction=0.1,
            default_task="fallback",
            episode_indices=[99],
        )
