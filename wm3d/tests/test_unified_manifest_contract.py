from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from wm3d.data.manifest_contract import (
    CACHE_INDEX_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    ManifestContractError,
    load_cache_index,
    load_data_profile,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_row() -> dict[str, object]:
    def clock(key: str, samples: int, min_dt: float, max_dt: float) -> dict[str, object]:
        return {
            "key": key,
            "origin": "recorded_payload_timestamps",
            "unit": "seconds",
            "sample_count": samples,
            "start_s": 0.0,
            "end_s": 1.7,
            "min_dt_s": min_dt,
            "max_dt_s": max_dt,
            "timestamp_sha256": "b" * 64,
        }

    groups = {
        name: {
            "supervision": "fine_command",
            "action_samples": 47,
            "state_samples": 18,
            "action_clock": clock("action.timestamp", 47, 0.031, 0.079),
            "state_clock": clock("observation.timestamp", 18, 0.08, 0.12),
            "world_interval_index_key": None,
        }
        for name in ("left_arm", "right_arm")
    }
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "episode_id": "episode-0001",
        "source": "toy",
        "payload": "raw/episode-0001.parquet",
        "payload_sha256": "a" * 64,
        "payload_row_start": 10,
        "payload_row_stop": 57,
        "assets": [
            {
                "role": "primary_payload",
                "path": "raw/episode-0001.parquet",
                "sha256": "a" * 64,
            },
            {
                "role": "rgb/head",
                "path": "videos/episode-0001.mp4",
                "sha256": "c" * 64,
            },
        ],
        "views": [
            {
                "name": "head",
                "asset_role": "rgb/head",
                "segment_kind": "recorded_pts_range",
                "start_s": 0.0,
                "stop_s": 1.7,
            }
        ],
        "task_text": "coordinate both arms to insert the peg",
        "embodiment": "dual",
        "split": "train",
        "duration_s": 1.7,
        "observation_samples": 47,
        "observation_clock": clock("observation.rgb.timestamp", 47, 0.031, 0.079),
        "robot_groups": groups,
    }


def _profile(manifest: Path) -> dict[str, object]:
    adapter = manifest.with_name("adapter.yaml")
    if not adapter.exists():
        adapter.write_text("schema: fixture-only\nsource: toy\n")
    arm = {
        "action_semantics": ["joint_position_rad", "absolute_gripper_open01"],
        "state_semantics": ["joint_position_rad", "gripper_close01"],
        "action_frame": "joint_space",
        "state_frame": "joint_space",
        "composition_operators": ["last", "logical_last"],
    }
    return {
        "schema": "wm3d_v8_data_profile_v4",
        "name": "toy_dual",
        "cache_representation": {
            "schema": "wm3d_v8_episode_representation_v1",
            "token_grid": 2,
            "spatial_tokens": 4,
            "token_dim": 16,
            "num_views": 2,
            "view_slots": ["head", "left_wrist"],
            "rgb_size": 16,
            "time_binding": "episode_row_ordinal_with_pts_audit",
            "missing_view_policy": "mask_without_duplication",
            "state_frame_selection": {
                "mode": "observed_greedy_minimum_separation",
                "minimum_separation_seconds": 0.1,
                "preserve_observed_timestamps": True,
                "interpolation": "forbidden",
            },
        },
        "cache": {
            "schema": CACHE_INDEX_SCHEMA,
            "task_partition": "episode",
            "task_claim": "atomic_no_clobber",
            "resume": "receipt_and_sha",
        },
        "sources": [
            {
                "name": "toy",
                "adapter": "fixture",
                "raw_root": str(manifest.parent),
                "adapter_config": str(adapter),
                "adapter_contract_sha256": _sha(adapter),
                "manifest": str(manifest),
                "manifest_sha256": _sha(manifest),
                "embodiment": "dual",
                "weight": 1,
                "nominal_hours": 0.001,
                "license_id": "fixture",
            }
        ],
        "embodiments": [
            {
                "name": "dual",
                "embodiment_id": 4,
                "groups": [
                    {"name": "left_arm", "group_id": 1, **arm},
                    {"name": "right_arm", "group_id": 2, **arm},
                ],
            }
        ],
    }


def test_profile_keeps_bimanual_groups_and_does_not_require_fixed_hz(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(_source_row(), sort_keys=True) + "\n")
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(_profile(manifest), sort_keys=False))

    profile = load_data_profile(profile_path)
    assert [group.name for group in profile.embodiments["dual"].groups] == [
        "left_arm",
        "right_arm",
    ]
    assert "world_hz" not in profile.cache_representation
    assert "action_hz" not in profile.cache_representation


def test_profile_accepts_a_separate_higher_resolution_appearance_grid(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(_source_row(), sort_keys=True) + "\n")
    value = _profile(manifest)
    value["cache_representation"]["appearance_token_grid"] = 4  # type: ignore[index]
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(value, sort_keys=False))

    profile = load_data_profile(profile_path)
    assert profile.cache_representation["appearance_token_grid"] == 4

    value["cache_representation"]["appearance_token_grid"] = 1  # type: ignore[index]
    invalid_path = tmp_path / "invalid_profile.yaml"
    invalid_path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="at least token_grid"):
        load_data_profile(invalid_path)


def test_profile_rejects_a_global_resampling_mode(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(_source_row()) + "\n")
    value = _profile(manifest)
    value["cache_representation"]["state_frame_selection"]["mode"] = "fixed_5hz"  # type: ignore[index]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="selected from observations"):
        load_data_profile(path)


def test_profile_rejects_ambiguous_or_duplicate_view_slots(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(_source_row()) + "\n")
    value = _profile(manifest)
    value["cache_representation"]["view_slots"] = ["head", "head"]  # type: ignore[index]
    path = tmp_path / "bad_views.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="unique non-empty"):
        load_data_profile(path)


@pytest.mark.parametrize("forbidden", ["preferred_hz", "minimum_hz", "maximum_hz"])
def test_profile_rejects_global_world_rate_hints(
    tmp_path: Path, forbidden: str
) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(_source_row()) + "\n")
    value = _profile(manifest)
    value["cache_representation"]["state_frame_selection"][forbidden] = 20.0  # type: ignore[index]
    path = tmp_path / "bad_rate.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="fields must be exactly"):
        load_data_profile(path)


def test_profile_rejects_policy_times_derived_from_world_cadence(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    manifest.write_text(json.dumps(_source_row()) + "\n")
    value = _profile(manifest)
    value["cache_representation"]["policy_query"] = {"training_times": "world_rate_times_four"}  # type: ignore[index]
    path = tmp_path / "bad_action_rate.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="unknown=.*policy_query"):
        load_data_profile(path)


def test_source_manifest_rejects_declared_nominal_hz(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    row = _source_row()
    row["robot_groups"]["left_arm"]["action_hz"] = 20  # type: ignore[index]
    manifest.write_text(json.dumps(row) + "\n")
    value = _profile(manifest)
    value["sources"][0]["manifest_sha256"] = _sha(manifest)  # type: ignore[index]
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="group fields must be exactly"):
        load_data_profile(path)


def test_source_manifest_rejects_missing_bimanual_group(tmp_path: Path) -> None:
    manifest = tmp_path / "source.jsonl"
    row = _source_row()
    del row["robot_groups"]["right_arm"]  # type: ignore[index]
    manifest.write_text(json.dumps(row) + "\n")
    value = _profile(manifest)
    value["sources"][0]["manifest_sha256"] = _sha(manifest)  # type: ignore[index]
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="robot_groups"):
        load_data_profile(path)


def test_source_manifest_rejects_unbound_or_unknown_episode_inputs(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "source.jsonl"
    row = _source_row()
    row["assets"][0]["sha256"] = "d" * 64  # type: ignore[index]
    row["action_hz"] = 20
    manifest.write_text(json.dumps(row) + "\n")
    value = _profile(manifest)
    value["sources"][0]["manifest_sha256"] = _sha(manifest)  # type: ignore[index]
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ManifestContractError, match="unknown=.*action_hz"):
        load_data_profile(path)


def test_cache_index_uses_shared_episode_robot_shard_and_model_window_rows(tmp_path: Path) -> None:
    row = {
        "schema": CACHE_INDEX_SCHEMA,
        "sample_id": "sample-1",
        "source": "toy",
        "split": "train",
        "embodiment": "dual",
        "feature_shard": "features/000.safetensors",
        "feature_sha256": "b" * 64,
        "leading_feature_row": 1,
        "context_feature_rows": [2, 4, 7],
        "future_feature_rows": [9, 12],
        "robot_shard": "robot/003.safetensors",
        "robot_sha256": "c" * 64,
        "rgb_pack": "rgb/000.pack",
        "rgb_pack_sha256": "d" * 64,
    }
    path = tmp_path / "index.jsonl"
    path.write_text(json.dumps(row, sort_keys=True) + "\n")
    entries = load_cache_index(path, expected_sha256=_sha(path))
    assert entries[0].context_feature_rows == (2, 4, 7)
    assert entries[0].future_feature_rows == (9, 12)
    assert entries[0].leading_feature_row == 1
    assert entries[0].robot_shard == "robot/003.safetensors"


def test_cache_index_does_not_accept_legacy_shared_row_ambiguity(tmp_path: Path) -> None:
    row = {
        "schema": CACHE_INDEX_SCHEMA,
        "sample_id": "sample-1",
        "source": "toy",
        "split": "train",
        "embodiment": "dual",
        "feature_shard": "features/000.safetensors",
        "feature_sha256": "b" * 64,
        "robot_shard": "robot/003.safetensors",
        "robot_sha256": "c" * 64,
        "rgb_pack": "rgb/000.pack",
        "rgb_pack_sha256": "d" * 64,
        "row": 2,
    }
    path = tmp_path / "index.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ManifestContractError, match="context_feature_rows"):
        load_cache_index(path, expected_sha256=_sha(path))
