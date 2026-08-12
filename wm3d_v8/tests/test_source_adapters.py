from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from wm3d_v3.data.grouped_robot import bimanual_arm_spec
from wm3d_v3.data.source_adapters import (
    AdapterContractError,
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
        "groups:\n"
        + "".join(groups)
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_config_driven_adapter_preserves_both_arms_and_native_times(tmp_path: Path) -> None:
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
