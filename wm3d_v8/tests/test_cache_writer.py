from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import wm3d_v3.data.grouped_normalization as grouped_normalization_module

from wm3d_v3.data.cache_tasks import plan_tasks
from wm3d_v3.data.cache_writer import UnifiedFrameCache, write_cache_task
from wm3d_v3.data.episode_robot import build_episode_robot_cache
from wm3d_v3.data.episode_robot import (
    assemble_robot_window_from_prepared_episode,
    prepare_episode_robot_tensors,
)
from wm3d_v3.data.grouped_robot import (
    GroupedRobotLimits,
    RawActionSeries,
    RawStateSeries,
    bimanual_arm_spec,
)
from wm3d_v3.data.grouped_normalization import (
    GroupedRobotNormalizer,
    build_grouped_normalization_artifact,
)
from wm3d_v3.data.manifest_contract import (
    DataProfile,
    SourceSpec,
    canonical_timestamp_sha256,
    canonical_sha256,
    load_cache_episode_index,
    sha256_file,
)
from wm3d_v3.data.unified_cache_dataset import UnifiedCacheDataset
from wm3d_v3.data.window_index import plan_window_index


def _clock() -> np.ndarray:
    return np.arange(20, dtype=np.float64) * 0.1


def _task(supervision: str = "fine_command"):
    clock = _clock()
    row = {
        "schema": "wm3d_v8_source_manifest_v4",
        "source": "dual",
        "episode_id": "episode-1",
        "payload": "raw/shared.parquet",
        "payload_sha256": "a" * 64,
        "payload_row_start": 10,
        "payload_row_stop": 30,
        "assets": [
            {
                "role": "primary_payload",
                "path": "raw/shared.parquet",
                "sha256": "a" * 64,
            },
            {"role": "rgb/head", "path": "video/head.mp4", "sha256": "b" * 64},
        ],
        "views": [
            {
                "name": "head",
                "asset_role": "rgb/head",
                "segment_kind": "entire_file",
                "start_s": None,
                "stop_s": None,
            }
        ],
        "task_text": "coordinate both arms",
        "embodiment": "bimanual_arm",
        "split": "train",
        "observation_samples": len(clock),
        "observation_clock": {
            "key": "timestamp",
            "origin": "recorded_payload_timestamps",
            "unit": "seconds",
            "sample_count": len(clock),
            "start_s": float(clock[0]),
            "end_s": float(clock[-1]),
            "min_dt_s": float(np.diff(clock).min()),
            "max_dt_s": float(np.diff(clock).max()),
            "timestamp_sha256": canonical_timestamp_sha256(clock),
        },
        "robot_groups": {
            "left_arm": {"supervision": supervision},
            "right_arm": {"supervision": supervision},
        },
    }
    return plan_tasks(
        [row],
        source_manifest_sha256="1" * 64,
        adapter_contract_sha256="c" * 64,
        encoder_contract_sha256="d" * 64,
        task_encoder_contract_sha256="f" * 64,
        task_bank_index_sha256="2" * 64,
        representation_contract_sha256="e" * 64,
        canonical_view_slots=("head", "left_wrist"),
    )[0]


def _frames() -> UnifiedFrameCache:
    torch.manual_seed(4)
    clock = torch.from_numpy(_clock())
    n, views, patches, dim = len(clock), 2, 4, 16
    confidence = torch.rand(n, views, patches).add_(0.1)
    return UnifiedFrameCache(
        source_observation_rows=torch.arange(n, dtype=torch.int64),
        frame_times_s=clock,
        view_tokens=torch.randn(n, views, patches, dim),
        rgb=torch.randint(0, 256, (n, views, 3, 16, 16), dtype=torch.uint8),
        view_mask=torch.ones(n, views, dtype=torch.bool),
        world_token_mask=torch.ones(n, patches, dtype=torch.bool),
        depth=torch.rand(n, views, patches).add_(0.1),
        depth_mask=torch.ones(n, views, patches, dtype=torch.bool),
        point=torch.randn(n, views, patches, 3),
        point_mask=torch.ones(n, views, patches, dtype=torch.bool),
        camera_pose=torch.randn(n, views, 9),
        camera_pose_mask=torch.ones(n, views, dtype=torch.bool),
        geometry_confidence=confidence,
    )


def _robot(supervision: str = "fine_command") -> dict[str, torch.Tensor]:
    clock = _clock()
    embodiment = bimanual_arm_spec()
    actions, states = [], []
    for slot, group in enumerate(embodiment.groups):
        action = np.zeros((len(clock) - 1, group.action_dim), dtype=np.float32)
        action[:, slot] = np.arange(len(action), dtype=np.float32) / 100.0
        action[:, -1] = float(slot)
        actions.append(
            RawActionSeries(
                group=group.name,
                supervision=supervision,
                values=action,
                timestamps_s=clock[:-1] if supervision == "fine_command" else None,
                world_interval_indices=(
                    np.arange(len(action), dtype=np.int64)
                    if supervision == "coarse_effect"
                    else None
                ),
            )
        )
        state = np.zeros((len(clock), group.state_dim), dtype=np.float32)
        state[:, slot] = np.arange(len(clock), dtype=np.float32) / 10.0
        states.append(
            RawStateSeries(
                group=group.name,
                values=state,
                timestamps_s=clock,
            )
        )
    return build_episode_robot_cache(
        embodiment=embodiment,
        action_series=actions,
        state_series=states,
        task_embedding=torch.arange(12, dtype=torch.float32),
        observation_times_s=clock,
        max_groups=2,
        max_action_dim=7,
        max_state_dim=10,
    ).as_tensors()


def _profile(root: Path) -> DataProfile:
    return DataProfile(
        path=root / "profile.yaml",
        profile_sha256="f" * 64,
        name="dual",
        sources=(
            SourceSpec(
                name="dual",
                adapter="fixture",
                raw_root=root,
                adapter_config_path=root / "adapter.yaml",
                adapter_contract_sha256="c" * 64,
                manifest_path=root / "source.jsonl",
                manifest_sha256="d" * 64,
                embodiment="bimanual_arm",
                weight=1,
                nominal_hours=None,
                license_id="fixture",
            ),
        ),
        embodiments={"bimanual_arm": bimanual_arm_spec()},
        cache_representation={
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
                "minimum_separation_seconds": 0.0,
                "preserve_observed_timestamps": True,
                "interpolation": "forbidden",
            },
        },
        cache={},
    )


def _model_profile() -> dict[str, object]:
    return {
        "schema": "wm3d_v8_model_profile_v1",
        "name": "tiny",
        "architecture": "native_world_model",
        "sampling": {
            "mode": "observed_monotonic_subsequence",
            "history_action_leading_boundary": "observed_previous_state",
            "context_horizon_seconds": 0.2,
            "future_horizon_seconds": 0.2,
            "minimum_horizon_coverage": 0.9,
            "minimum_anchor_separation_seconds": 0.1,
            "policy_target_horizon_seconds": 0.2,
            "policy_training_times": "observed_action_timestamps",
            "interpolation": "forbidden",
        },
        "model": {
            "T": 2,
            "K": 2,
            "P": 4,
            "token_dim": 16,
            "task_dim": 12,
            "num_views": 2,
            "rgb_size": 16,
            "rgb_decode_indices": [0, 1],
            "max_action_groups": 2,
            "max_action_substeps": 4,
            "max_action_dim": 7,
            "max_state_dim": 10,
            "max_policy_queries": 4,
            "max_aux_tokens": 2,
            "aux_dim": 8,
        },
    }


@pytest.mark.parametrize("supervision", ["fine_command", "coarse_effect"])
def test_episode_cache_is_shared_and_window_index_assembles_real_robot_times(
    tmp_path: Path,
    monkeypatch,
    supervision: str,
) -> None:
    task = _task(supervision)
    frames = _frames()
    robot = _robot(supervision)
    first = write_cache_task(
        task=task,
        cache_root=tmp_path,
        frames=frames,
        robot_tensors=robot,
        source_evidence={"fixture": True},
        jpeg_quality=95,
    )
    assert first["status"] == "published" and first["frames"] == len(_clock())
    assert write_cache_task(
        task=task,
        cache_root=tmp_path,
        frames=frames,
        robot_tensors=robot,
        source_evidence={"fixture": True},
        jpeg_quality=95,
    )["status"] == "already_complete"

    episode_index = tmp_path / "episode_index_fragments" / f"{task.task_id}.jsonl"
    episodes = load_cache_episode_index(
        episode_index, expected_sha256=sha256_file(episode_index)
    )
    rows = plan_window_index(
        episodes=episodes,
        cache_root=tmp_path,
        model_profile=_model_profile(),
        model_profile_sha256=canonical_sha256(_model_profile()),
        data_profile=_profile(tmp_path),
    )
    assert len(rows) > 2
    window_index = tmp_path / "window_index.jsonl"
    window_index.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    profile = _profile(tmp_path)
    model_profile = _model_profile()
    model_profile_sha256 = canonical_sha256(model_profile)
    feature_payloads = {
        (tmp_path / row["feature_shard"]).resolve() for row in rows
    }
    real_normalization_sha256_file = grouped_normalization_module.sha256_file

    def reject_feature_payload_hash(path: Path) -> str:
        if Path(path).resolve() in feature_payloads:
            raise AssertionError(
                "normalization must not re-hash the sealed feature payload"
            )
        return real_normalization_sha256_file(path)

    monkeypatch.setattr(
        grouped_normalization_module,
        "sha256_file",
        reject_feature_payload_hash,
    )
    normalization_artifact = build_grouped_normalization_artifact(
        data_profile=profile,
        model_profile=model_profile,
        model_profile_sha256=model_profile_sha256,
        window_index_path=window_index,
        window_index_sha256=sha256_file(window_index),
        cache_root=tmp_path,
    )
    normalizer = GroupedRobotNormalizer(normalization_artifact, data_profile=profile)
    dataset = UnifiedCacheDataset(
        cache_root=tmp_path,
        index_path=window_index,
        index_sha256=sha256_file(window_index),
        data_profile=profile,
        model_profile=model_profile,
        split="train",
        grouped_normalizer=normalizer,
    )
    loaded = dataset[0]
    assert loaded["world_tokens"].shape == (2, 2, 4, 16)
    assert loaded["target_tokens"].shape == (2, 4, 16)
    assert loaded["target_rgb"].shape == (2, 2, 3, 16, 16)
    assert loaded["action_group_mask"].tolist() == [True, True]
    assert loaded["current_state_mask"].all()
    assert loaded["policy_query_mask"].any(dim=-1).tolist() == [True, True]
    assert loaded["future_world_boundaries_dt"].shape == (3,)
    assert loaded["action_normalization_scale"].shape == (2, 7)
    assert loaded["action_normalization_scale"][:, -1].eq(1).all()
    assert loaded["action_normalization_offset"][:, -1].eq(0).all()
    assert {
        row["lane"]
        for row in normalization_artifact["rows"]
        if row["kind"] == "action"
    } == {supervision}
    if supervision == "fine_command":
        assert loaded["target_fine_action_mask"].any()
        assert not loaded["target_coarse_action_mask"].any()
    else:
        assert not loaded["target_fine_action_mask"].any()
        assert loaded["target_coarse_action_mask"].any()
    feature_relative = rows[0]["feature_shard"]
    dataset.shards._verified.discard(feature_relative)
    dataset.shards.expected_sha[feature_relative] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        dataset.shards.path(feature_relative)
    assert torch.isfinite(loaded["history_fine_action_values"]).all()
    assert torch.equal(
        loaded["target_fine_action"][..., -1],
        loaded["target_fine_action"][..., -1].round(),
    )


def test_policy_target_horizon_is_half_open() -> None:
    clock = _clock()
    prepared = prepare_episode_robot_tensors(
        _robot(), embodiment=bimanual_arm_spec()
    )
    result = assemble_robot_window_from_prepared_episode(
        prepared=prepared,
        embodiment=bimanual_arm_spec(),
        selected_source_boundary_indices=[0, 1, 2, 3, 4],
        limits=GroupedRobotLimits(
            max_groups=2, max_substeps=4, max_action_dim=7, max_state_dim=10
        ),
        context_samples=2,
        max_policy_queries=4,
        policy_target_horizon_s=0.2,
    )
    # policy_start=0.2 and the command at 0.4 belongs to the next chunk.
    expected = torch.tensor([0.0, 0.1], dtype=torch.float32)
    assert result["policy_query_mask"][:, :2].all()
    assert not result["policy_query_mask"][:, 2:].any()
    torch.testing.assert_close(result["policy_query_dt"][:, :2], expected.repeat(2, 1))
