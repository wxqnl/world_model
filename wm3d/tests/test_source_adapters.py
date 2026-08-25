from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from wm3d.data.grouped_robot import (
    ActionGroupSpec,
    EmbodimentSpec,
    bimanual_arm_spec,
)
from wm3d.data.source_adapters import (
    AdapterContractError,
    _canonical_action,
    _canonical_state7,
    adapt_action_series,
    adapt_current_state,
    adapt_robot_signals,
    load_adapter_contract,
)


class Accessor:
    def __init__(self, values: dict[str, np.ndarray]):
        self.values = values

    def array(self, key: str) -> np.ndarray:
        return self.values[key]


def _write_contract(path: Path) -> str:
    groups = []
    for name, prefix in (("left_arm", "left"), ("right_arm", "right")):
        groups.append(
            f"""  - group: {name}
    supervision: fine_command
    action:
      - key: {prefix}.action
        columns: [0, 1, 2, 3, 4, 5, 6]
        scale: [1, 1, 1, 1, 1, 1, 1]
        offset: [0, 0, 0, 0, 0, 0, 0]
    state:
      - key: {prefix}.state
        columns: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        scale: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        offset: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action_time_key: action.time
    state_time_key: state.time
    world_interval_index_key: null
"""
        )
    path.write_text(
        "schema: wm3d_v8_source_adapter_v3\n"
        "name: dual_fixture\n"
        "raw_format: npz\n"
        "observation_time_key: observation.time\n"
        "views:\n"
        "  - name: overhead\n"
        "    key: observation.rgb.overhead\n"
        "groups:\n" + "".join(groups)
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_config_driven_adapter_preserves_both_arms_and_native_times(
    tmp_path: Path,
) -> None:
    path = tmp_path / "adapter.yaml"
    contract = load_adapter_contract(path, expected_sha256=_write_contract(path))
    assert [(view.name, view.key) for view in contract.views] == [
        ("overhead", "observation.rgb.overhead")
    ]
    assert "left.action" in contract.required_array_keys
    time = np.array([0.0, 0.031, 0.079], dtype=np.float64)
    accessor = Accessor(
        {
            "left.action": np.full((3, 7), 1.0, np.float32),
            "right.action": np.full((3, 7), 2.0, np.float32),
            "left.state": np.full((3, 10), 3.0, np.float32),
            "right.state": np.full((3, 10), 4.0, np.float32),
            "action.time": time,
            "state.time": time,
            "observation.time": time,
        }
    )
    actions, states = adapt_robot_signals(
        accessor=accessor,
        contract=contract,
        embodiment=bimanual_arm_spec(),
        policy_chunk_start_s=0.0,
    )
    assert [item.group for item in actions] == ["left_arm", "right_arm"]
    np.testing.assert_array_equal(actions[0].timestamps_s, time)
    np.testing.assert_array_equal(actions[0].values, 1.0)
    np.testing.assert_array_equal(actions[1].values, 2.0)
    np.testing.assert_array_equal(states[0].values, 3.0)
    np.testing.assert_array_equal(states[1].values, 4.0)


def test_current_state_nearest_neighbor_fallback_is_forbidden(tmp_path: Path) -> None:
    path = tmp_path / "adapter.yaml"
    contract = load_adapter_contract(path, expected_sha256=_write_contract(path))
    accessor = Accessor(
        {
            "left.action": np.zeros((2, 7), np.float32),
            "right.action": np.zeros((2, 7), np.float32),
            "left.state": np.zeros((2, 10), np.float32),
            "right.state": np.zeros((2, 10), np.float32),
            "action.time": np.array([0.0, 0.05]),
            "state.time": np.array([0.001, 0.051]),
            "observation.time": np.array([0.0, 0.05]),
        }
    )
    with pytest.raises(AdapterContractError, match="fallback is forbidden"):
        adapt_robot_signals(
            accessor=accessor,
            contract=contract,
            embodiment=bimanual_arm_spec(),
            policy_chunk_start_s=0.0,
        )


def test_complete_action_decode_is_independent_of_policy_anchor(tmp_path: Path) -> None:
    path = tmp_path / "adapter.yaml"
    contract = load_adapter_contract(path, expected_sha256=_write_contract(path))
    time = np.asarray([0.0, 0.04, 0.09], dtype=np.float64)
    accessor = Accessor(
        {
            "left.action": np.ones((3, 7), np.float32),
            "right.action": np.full((3, 7), 2.0, np.float32),
            "left.state": np.full((3, 10), 3.0, np.float32),
            "right.state": np.full((3, 10), 4.0, np.float32),
            "action.time": time,
            "state.time": time,
            "observation.time": time,
        }
    )
    actions = adapt_action_series(
        accessor=accessor,
        contract=contract,
        embodiment=bimanual_arm_spec(),
    )
    assert len(actions) == 2
    np.testing.assert_array_equal(actions[0].timestamps_s, time)
    states = adapt_current_state(
        accessor=accessor,
        contract=contract,
        embodiment=bimanual_arm_spec(),
        policy_chunk_start_s=0.04,
    )
    assert [item.timestamp_s for item in states] == [0.04, 0.04]


def test_color_order_v4_accepts_bgr_and_rejects_unknown_order(tmp_path: Path) -> None:
    path = tmp_path / "adapter.yaml"
    _write_contract(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value["schema"] = "wm3d_source_adapter_v4"
    value["views"][0]["color_order"] = "bgr"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    contract = load_adapter_contract(
        path,
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert contract.views[0].color_order == "bgr"

    value["views"][0]["color_order"] = "yuv"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(AdapterContractError, match="color_order must be rgb/bgr"):
        load_adapter_contract(
            path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def _canonical_arm(*, state: bool = True) -> EmbodimentSpec:
    return EmbodimentSpec(
        "canonical_arm",
        200,
        (
            ActionGroupSpec(
                "arm",
                1,
                (
                    "delta_position_m",
                    "delta_position_m",
                    "delta_position_m",
                    "delta_rotation_axis_angle_rad",
                    "delta_rotation_axis_angle_rad",
                    "delta_rotation_axis_angle_rad",
                    "absolute_gripper_close01",
                ),
                (
                    (
                        "eef_position_m",
                        "eef_position_m",
                        "eef_position_m",
                        "eef_rotation_6d",
                        "eef_rotation_6d",
                        "eef_rotation_6d",
                        "eef_rotation_6d",
                        "eef_rotation_6d",
                        "eef_rotation_6d",
                        "controller_state",
                    )
                    if state
                    else ()
                ),
                "robot_base",
                "robot_base" if state else "not_applicable",
                (
                    "sum",
                    "sum",
                    "sum",
                    "so3_axis_angle_base_left",
                    "so3_axis_angle_base_left",
                    "so3_axis_angle_base_left",
                    "logical_last",
                ),
            ),
        ),
    )


def _write_canonical_contract(path: Path, *, droid: bool) -> str:
    action = (
        [
            {
                "key": "action.pose",
                "columns": list(range(6)),
                "scale": [1] * 6,
                "offset": [0] * 6,
            },
            {"key": "action.grip", "columns": [0], "scale": [1], "offset": [0]},
        ]
        if droid
        else [
            {
                "key": "action",
                "columns": list(range(7)),
                "scale": [1] * 7,
                "offset": [0] * 7,
            }
        ]
    )
    state_mapping = (
        [
            {
                "key": "state.pose",
                "columns": list(range(6)),
                "scale": [1] * 6,
                "offset": [0] * 6,
            },
            {"key": "state.grip", "columns": [0], "scale": [1], "offset": [0]},
        ]
        if droid
        else [
            {
                "key": "state",
                "columns": list(range(8)),
                "scale": [1] * 8,
                "offset": [0] * 8,
            }
        ]
    )
    value = {
        "schema": "wm3d_source_adapter_v5",
        "name": "canonical_fixture",
        "raw_format": "npz",
        "observation_time_key": "time",
        "views": [{"name": "head", "key": "rgb", "color_order": "rgb"}],
        "groups": [
            {
                "group": "arm",
                "supervision": "fine_command",
                "action": action,
                "state": state_mapping,
                "action_time_key": "time",
                "state_time_key": "time",
                "world_interval_index_key": None,
                "action_transform": "droid_target" if droid else "state_euler_residual",
                "state_transform": "droid_state" if droid else "pos_euler_8d",
                "action_value_mask": [True, True, True, True, True, False, True],
                "state_value_mask": [True] * 10,
            }
        ],
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v5_bridge_canonicalizes_rotation_gripper_and_masks(tmp_path: Path) -> None:
    path = tmp_path / "bridge.yaml"
    contract = load_adapter_contract(
        path,
        expected_sha256=_write_canonical_contract(path, droid=False),
    )
    time = np.asarray([0.0, 0.1], dtype=np.float64)
    state = np.zeros((2, 8), dtype=np.float32)
    state[:, 7] = 0.4
    action = np.zeros((2, 7), dtype=np.float32)
    action[:, :3] = (0.1, -0.2, 0.3)
    action[:, 5] = np.pi / 2.0
    action[:, 6] = 0.25
    accessor = Accessor({"action": action, "state": state, "time": time})

    actions = adapt_action_series(
        accessor=accessor, contract=contract, embodiment=_canonical_arm()
    )
    np.testing.assert_allclose(actions[0].values[:, :3], action[:, :3])
    np.testing.assert_allclose(
        actions[0].values[:, 3:6],
        np.asarray([[0.0, 0.0, np.pi / 2.0]] * 2),
        atol=1.0e-5,
    )
    np.testing.assert_allclose(actions[0].values[:, 6], 0.75)
    np.testing.assert_array_equal(
        actions[0].value_mask,
        np.asarray([[True, True, True, True, True, False, True]] * 2),
    )
    current = adapt_current_state(
        accessor=accessor,
        contract=contract,
        embodiment=_canonical_arm(),
        policy_chunk_start_s=0.0,
    )[0]
    np.testing.assert_allclose(current.values[3:9], [1, 0, 0, 0, 1, 0])
    np.testing.assert_allclose(current.values[9], 0.4)


def test_v5_droid_uses_target_minus_current_pose(tmp_path: Path) -> None:
    path = tmp_path / "droid.yaml"
    contract = load_adapter_contract(
        path,
        expected_sha256=_write_canonical_contract(path, droid=True),
    )
    time = np.asarray([0.0], dtype=np.float64)
    state_pose = np.asarray([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0]], np.float32)
    target_pose = np.asarray([[1.1, 2.2, 2.5, 0.0, 0.0, np.pi / 2.0]], np.float32)
    accessor = Accessor(
        {
            "action.pose": target_pose,
            "action.grip": np.asarray([[0.8]], np.float32),
            "state.pose": state_pose,
            "state.grip": np.asarray([[0.3]], np.float32),
            "time": time,
        }
    )
    action = adapt_action_series(
        accessor=accessor, contract=contract, embodiment=_canonical_arm()
    )[0].values[0]
    np.testing.assert_allclose(action[:3], [0.1, 0.2, -0.5], atol=1.0e-6)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, np.pi / 2.0], atol=1.0e-5)
    np.testing.assert_allclose(action[6], 0.2)


def test_nyu_signed_gripper_is_clipped_before_open_to_close() -> None:
    raw = np.zeros((2, 14), dtype=np.float32)
    raw[:, -2] = (-1.0, 1.0)
    action = _canonical_action(raw, "nyu_franka", state7=None)
    np.testing.assert_allclose(action[:, 6], [1.0, 0.0])


def test_robocasa_panda_uses_close_positive_action_and_state() -> None:
    state = np.zeros((2, 16), dtype=np.float32)
    state[:, 10:14] = (0.0, 0.0, 0.0, 1.0)
    state[0, 14:16] = (0.04, -0.04)
    state[1, 14:16] = (0.0, 0.0)
    canonical_state = _canonical_state7(state, "robocasa")
    np.testing.assert_allclose(canonical_state[:, 6], [0.0, 1.0])

    action = np.zeros((2, 7), dtype=np.float32)
    action[:, 6] = (-1.0, 1.0)
    canonical_action = _canonical_action(
        action,
        "robocasa",
        state7=canonical_state,
    )
    np.testing.assert_allclose(canonical_action[:, 6], [0.0, 1.0])


def test_robocasa_panda_rejects_impossible_gripper_aperture() -> None:
    state = np.zeros((1, 16), dtype=np.float32)
    state[:, 10:14] = (0.0, 0.0, 0.0, 1.0)
    state[0, 14:16] = (0.07, -0.07)
    with pytest.raises(AdapterContractError, match="0.12 m envelope"):
        _canonical_state7(state, "robocasa")
