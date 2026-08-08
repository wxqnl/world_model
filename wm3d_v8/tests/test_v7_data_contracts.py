from dataclasses import replace

import numpy as np
import pytest

from wm3d_v3.data.v7_action_contract import (
    ActionAdapter,
    audit_canonical_actions,
    canonicalize_dense_action,
    resample_canonical_actions,
)
from wm3d_v3.data.v7_contracts import (
    CANONICAL_ACTION_VERSION,
    V7ActionSpec,
    V7BranchSpec,
    V7ClipRecord,
    V7ViewSpec,
    canonical_clip_hash,
    enumerate_window_keys,
    stable_split,
    validate_record,
)


def make_record(*, with_branches: bool = False) -> V7ClipRecord:
    source = "robocasa365"
    episode = "episode_000001"
    adapter = CANONICAL_ACTION_VERSION
    group = "PickPlace/kitchen_01/seed_17"
    branches = ()
    if with_branches:
        branches = tuple(
            V7BranchSpec(
                root_id="root_17",
                branch_id=f"branch_{index}",
                root_seed=17,
                action_path=f"a{index}.npy",
                target_tokens_path=None,
                target_geometry_path=f"g{index}.npz",
                outcome_path=f"o{index}.json",
                simulator_state_path="state.npz",
                true_simulator_rollout=True,
            )
            for index in range(4)
        )
    return V7ClipRecord(
        source=source,
        native_episode_id=episode,
        native_start_frame=0,
        native_end_frame=200,
        native_fps=20.0,
        raw_path="episode.parquet",
        task_text="pick up the mug",
        task_class="PickPlace",
        scene_id="kitchen_01",
        robot="panda",
        embodiment_id="robocasa_panda",
        split_group=group,
        split=stable_split(group),
        views=(
            V7ViewSpec("external_anchor", "agentview_left", "cam0", "timestamp", calibrated=True),
            V7ViewSpec("wrist", "eye_in_hand", "cam2", "timestamp", calibrated=True),
        ),
        action=V7ActionSpec(
            adapter_version=adapter,
            raw_kind="dense7",
            source_frame="robot_base",
            rotation_repr="axis_angle",
            translation_unit="m",
            rotation_unit="rad",
            control_hz=20.0,
            is_delta=True,
            gripper_semantics="-1=open,+1=closed",
            action_key="action",
            observation_timestamp_key="timestamp",
            action_timestamp_key="timestamp",
            future_timestamp_key="timestamp_next",
            action_valid=True,
            audit_report="audits/robocasa.json",
        ),
        clip_hash=canonical_clip_hash(source, episode, 0, 200, adapter),
        branches=branches,
    )


def test_v7_record_and_windows_are_stable():
    record = make_record(with_branches=True)
    validate_record(record)
    assert record.has_true_counterfactual
    assert list(enumerate_window_keys(record)) == list(enumerate_window_keys(record))
    assert len(list(enumerate_window_keys(record))) > 0


def test_v7_rejects_pseudo_branch_and_split_leakage():
    record = make_record(with_branches=True)
    bad_branch = replace(record.branches[0], true_simulator_rollout=False)
    with pytest.raises(ValueError, match="pseudo"):
        validate_record(replace(record, branches=(bad_branch,) + record.branches[1:]))
    wrong_split = "test" if record.split != "test" else "train"
    with pytest.raises(ValueError, match="split leakage"):
        validate_record(replace(record, split=wrong_split))


def test_action_adapter_maps_frame_and_gripper():
    adapter = ActionAdapter(
        source="toy",
        source_frame="camera",
        translation_unit_scale=0.01,
        rotation_unit_scale=1.0,
        rotation_repr="axis_angle",
        gripper_open_value=1.0,
        gripper_closed_value=-1.0,
        base_from_source_rotation=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    )
    raw = np.array([[1.0, 0.0, 0.0, 0.1, 0.0, 0.0, -1.0]], dtype=np.float32)
    canonical = canonicalize_dense_action(raw, adapter)
    np.testing.assert_allclose(canonical[0, :3], [0.0, 0.01, 0.0], atol=1e-6)
    assert canonical[0, 6] == pytest.approx(1.0)


def test_action_audit_fails_wrong_control_rate():
    adapter = ActionAdapter(
        source="toy",
        source_frame="robot_base",
        translation_unit_scale=1.0,
        rotation_unit_scale=1.0,
        rotation_repr="axis_angle",
        nominal_hz=5.0,
    )
    actions = np.zeros((32, 7), dtype=np.float32)
    timestamps = np.arange(32, dtype=np.float64) / 20.0
    report = audit_canonical_actions(actions, source="toy", adapter=adapter, timestamps=timestamps)
    assert not report.passed
    assert "control_hz" in report.failures


def test_action_resampling_composes_translation_and_keeps_last_gripper():
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[:, 0] = 0.01
    actions[:, 5] = 0.01
    actions[:, 6] = np.arange(8) % 2
    downsampled = resample_canonical_actions(actions, source_hz=20.0, target_hz=5.0)
    np.testing.assert_allclose(downsampled[:, 0], 0.04, atol=1e-6)
    np.testing.assert_allclose(downsampled[:, 5], 0.04, atol=1e-5)
    np.testing.assert_array_equal(downsampled[:, 6], actions[[3, 7], 6])
