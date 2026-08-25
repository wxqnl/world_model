from __future__ import annotations

import numpy as np
import pytest

from wm3d.data.grouped_robot import (
    ActionGroupSpec,
    GroupedRobotContractError,
    GroupedRobotLimits,
    RawActionSeries,
    RawStateSnapshot,
    bimanual_arm_spec,
    compose_coarse_effects_for_observed_window,
    pack_grouped_robot_window,
    panda_single_arm_spec,
    slice_fine_action_series,
)


def _panda_state(timestamp: float = 0.0) -> RawStateSnapshot:
    return RawStateSnapshot("arm", timestamp, np.arange(10, dtype=np.float32))


def test_panda_policy_abi_uses_close_positive_gripper() -> None:
    group = panda_single_arm_spec().groups[0]
    assert group.action_semantics[-1] == "absolute_gripper_close01"
    assert group.state_semantics[-1] == "gripper_close01"


def test_native_20hz_commands_remain_four_real_substeps_per_world_interval() -> None:
    limits = GroupedRobotLimits(max_substeps=6)
    timestamps = np.arange(8, dtype=np.float64) / 20.0
    values = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
    packed = pack_grouped_robot_window(
        embodiment=panda_single_arm_spec(),
        limits=limits,
        world_boundaries_s=[0.0, 0.2, 0.4],
        action_series=[RawActionSeries("arm", "fine_command", values, timestamps)],
        current_state=[_panda_state()],
        policy_chunk_start_s=0.0,
    )

    assert packed.fine_sample_mask[:, 0].sum(axis=-1).tolist() == [4, 4]
    np.testing.assert_array_equal(packed.fine_action_values[0, 0, :4, :7], values[:4])
    np.testing.assert_array_equal(packed.fine_action_values[1, 0, :4, :7], values[4:])
    assert not packed.fine_sample_mask[:, 0, 4:].any()
    assert not packed.coarse_action_mask.any()


def test_world_state_cadence_is_timestamp_driven_not_fixed_to_five_hz() -> None:
    timestamps = np.array([0.0, 0.03, 0.08, 0.13, 0.22, 0.31], dtype=np.float64)
    values = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
    packed = pack_grouped_robot_window(
        embodiment=panda_single_arm_spec(),
        limits=GroupedRobotLimits(max_substeps=8),
        world_boundaries_s=[0.0, 0.1, 0.25, 0.4],
        action_series=[RawActionSeries("arm", "fine_command", values, timestamps)],
        current_state=[_panda_state()],
        policy_chunk_start_s=0.0,
    )

    np.testing.assert_allclose(packed.world_interval_dt, [0.1, 0.15, 0.15])
    assert packed.fine_sample_mask[:, 0].sum(axis=-1).tolist() == [3, 2, 1]


def test_coarse_effect_never_becomes_a_fine_policy_label() -> None:
    values = np.arange(14, dtype=np.float32).reshape(2, 7)
    packed = pack_grouped_robot_window(
        embodiment=panda_single_arm_spec(),
        limits=GroupedRobotLimits(),
        world_boundaries_s=[0.0, 0.2, 0.4],
        action_series=[
            RawActionSeries(
                "arm",
                "coarse_effect",
                values,
                world_interval_indices=np.array([0, 1]),
            )
        ],
        current_state=[_panda_state()],
        policy_chunk_start_s=0.0,
    )

    assert not packed.fine_action_mask.any()
    assert not packed.fine_sample_mask.any()
    np.testing.assert_array_equal(packed.coarse_action_values[:, 0, :7], values)
    assert packed.coarse_action_mask[:, 0, :7].all()


def test_bimanual_layout_preserves_independent_groups_and_current_state() -> None:
    embodiment = bimanual_arm_spec()
    timestamps = np.arange(6, dtype=np.float64) / 30.0
    left = np.full((6, 7), 1.0, dtype=np.float32)
    right = np.full((6, 7), 2.0, dtype=np.float32)
    packed = pack_grouped_robot_window(
        embodiment=embodiment,
        limits=GroupedRobotLimits(max_substeps=6),
        world_boundaries_s=[0.0, 0.2],
        action_series=[
            RawActionSeries("left_arm", "fine_command", left, timestamps),
            RawActionSeries("right_arm", "fine_command", right, timestamps),
        ],
        current_state=[
            RawStateSnapshot("left_arm", 0.0, np.full(10, 3.0, dtype=np.float32)),
            RawStateSnapshot("right_arm", 0.0, np.full(10, 4.0, dtype=np.float32)),
        ],
        policy_chunk_start_s=0.0,
    )

    assert packed.group_ids[:2].tolist() == [1, 2]
    assert packed.group_mask[:2].all()
    np.testing.assert_array_equal(packed.fine_action_values[0, 0, :, :7], left)
    np.testing.assert_array_equal(packed.fine_action_values[0, 1, :, :7], right)
    np.testing.assert_array_equal(packed.current_state_values[0, :10], 3.0)
    np.testing.assert_array_equal(packed.current_state_values[1, :10], 4.0)


def test_packer_rejects_over_capacity_instead_of_dropping_or_resampling() -> None:
    timestamps = np.arange(7, dtype=np.float64) / 40.0
    values = np.zeros((7, 7), dtype=np.float32)
    with pytest.raises(GroupedRobotContractError, match="more than 6 real commands"):
        pack_grouped_robot_window(
            embodiment=panda_single_arm_spec(),
            limits=GroupedRobotLimits(max_substeps=6),
            world_boundaries_s=[0.0, 0.2],
            action_series=[RawActionSeries("arm", "fine_command", values, timestamps)],
            current_state=[_panda_state()],
            policy_chunk_start_s=0.0,
        )


def test_packer_rejects_non_exact_current_state_instead_of_interpolation() -> None:
    timestamps = np.arange(4, dtype=np.float64) / 20.0
    values = np.zeros((4, 7), dtype=np.float32)
    with pytest.raises(
        GroupedRobotContractError, match="interpolation/fallback is forbidden"
    ):
        pack_grouped_robot_window(
            embodiment=panda_single_arm_spec(),
            limits=GroupedRobotLimits(timestamp_tolerance_s=1.0e-6),
            world_boundaries_s=[0.0, 0.2],
            action_series=[RawActionSeries("arm", "fine_command", values, timestamps)],
            current_state=[_panda_state(timestamp=0.01)],
            policy_chunk_start_s=0.0,
        )


def test_fine_timestamps_must_be_strictly_increasing() -> None:
    with pytest.raises(GroupedRobotContractError, match="strictly increasing"):
        pack_grouped_robot_window(
            embodiment=panda_single_arm_spec(),
            limits=GroupedRobotLimits(),
            world_boundaries_s=[0.0, 0.2],
            action_series=[
                RawActionSeries(
                    "arm",
                    "fine_command",
                    np.zeros((2, 7), dtype=np.float32),
                    np.array([0.0, 0.0]),
                )
            ],
            current_state=[_panda_state()],
            policy_chunk_start_s=0.0,
        )


def test_internal_world_boundary_does_not_snap_nearby_action_timestamps() -> None:
    boundary = np.float64(0.2)
    just_before = np.nextafter(boundary, np.float64(-np.inf))
    exact = boundary
    packed = pack_grouped_robot_window(
        embodiment=panda_single_arm_spec(),
        limits=GroupedRobotLimits(max_substeps=2, timestamp_tolerance_s=1.0e-6),
        world_boundaries_s=[0.0, boundary, 0.4],
        action_series=[
            RawActionSeries(
                "arm",
                "fine_command",
                np.stack(
                    (
                        np.full(7, 1.0, dtype=np.float32),
                        np.full(7, 2.0, dtype=np.float32),
                    )
                ),
                np.asarray([just_before, exact], dtype=np.float64),
            )
        ],
        current_state=[_panda_state()],
        policy_chunk_start_s=0.0,
    )
    assert packed.fine_sample_mask[:, 0].sum(axis=-1).tolist() == [1, 1]
    np.testing.assert_array_equal(packed.fine_action_values[0, 0, 0, :7], 1.0)
    np.testing.assert_array_equal(packed.fine_action_values[1, 0, 0, :7], 2.0)


def test_fine_series_slice_is_half_open_and_never_snaps() -> None:
    boundary = np.float64(0.2)
    before = np.nextafter(boundary, np.float64(-np.inf))
    after = np.nextafter(np.float64(0.4), np.float64(np.inf))
    source = RawActionSeries(
        "arm",
        "fine_command",
        np.arange(5, dtype=np.float32)[:, None],
        np.asarray([0.0, before, boundary, 0.399, after], dtype=np.float64),
    )
    selected = slice_fine_action_series(source, start_s=0.2, stop_s=0.4)
    np.testing.assert_array_equal(selected.values[:, 0], [2.0, 3.0])
    np.testing.assert_array_equal(selected.timestamps_s, [boundary, 0.399])


def test_coarse_effects_compose_over_selected_observed_intervals() -> None:
    group = ActionGroupSpec(
        name="mobile",
        group_id=1,
        action_semantics=(
            "delta_position_m",
            "delta_rotation_axis_angle_rad",
            "delta_rotation_axis_angle_rad",
            "delta_rotation_axis_angle_rad",
            "base_velocity_mps",
            "controller_mode",
        ),
        state_semantics=("controller_state",),
        action_frame="robot_base",
        state_frame="robot_base",
        composition_operators=(
            "sum",
            "so3_axis_angle_base_left",
            "so3_axis_angle_base_left",
            "so3_axis_angle_base_left",
            "time_weighted_mean",
            "logical_last",
        ),
    )
    values = np.asarray(
        [
            [1.0, np.pi / 2, 0.0, 0.0, 10.0, 0.0],
            [2.0, 0.0, np.pi / 2, 0.0, 20.0, 1.0],
            [4.0, 0.0, 0.0, np.pi / 2, 30.0, 0.0],
        ],
        dtype=np.float32,
    )
    composed = compose_coarse_effects_for_observed_window(
        RawActionSeries(
            "mobile",
            "coarse_effect",
            values,
            world_interval_indices=np.arange(3),
        ),
        group=group,
        source_world_times_s=[0.0, 0.1, 0.3, 0.6],
        selected_boundary_indices=[0, 2, 3],
    )
    assert composed.world_interval_indices.tolist() == [0, 1]
    np.testing.assert_allclose(composed.values[:, 0], [3.0, 4.0])
    np.testing.assert_allclose(composed.values[:, 4], [50.0 / 3.0, 30.0])
    np.testing.assert_array_equal(composed.values[:, 5], [1.0, 0.0])
    # Non-commuting x then y rotations are intentionally not vector-added.
    assert not np.allclose(composed.values[0, 1:4], [np.pi / 2, np.pi / 2, 0.0])
    assert composed.value_mask.all()


def test_coarse_composition_rejects_missing_raw_interval() -> None:
    group = panda_single_arm_spec().groups[0]
    with pytest.raises(GroupedRobotContractError, match="exactly one coarse effect"):
        compose_coarse_effects_for_observed_window(
            RawActionSeries(
                "arm",
                "coarse_effect",
                np.zeros((1, 7), np.float32),
                world_interval_indices=np.asarray([0]),
            ),
            group=group,
            source_world_times_s=[0.0, 0.1, 0.2],
            selected_boundary_indices=[0, 2],
        )
