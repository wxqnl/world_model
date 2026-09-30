from dataclasses import replace

import numpy as np
import pytest

from wm3d.data.source_adapters import (
    AdapterContractError, GroupMapping, MappingTerm, align_coarse_action_rows,
)


class Arrays:
    def __init__(self, values):
        self.values = values

    def array(self, key):
        return self.values[key]


def mapping(**changes):
    base = GroupMapping(
        group="arm", supervision="coarse_effect",
        action=(MappingTerm("action", (0, 1), (1., 1.), (0., 0.)),),
        state=(MappingTerm("state", (0, 1), (1., 1.), (0., 0.)),),
        action_time_key=None, state_time_key="time",
        world_interval_index_key="frame_index",
        action_alignment="next_observation_drop_terminal",
    )
    return replace(base, **changes)


def inputs():
    state = np.asarray([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], np.float32)
    # Terminal label has no successor inside the episode, not a target to invent.
    action = np.concatenate([state[1:], [[99., 100.]]]).astype(np.float32)
    return Arrays({"state": state, "action": action,
                   "frame_index": np.arange(4), "time": np.asarray([0., .03, .08, .12])})


def run(a, m=None):
    return align_coarse_action_rows(
        accessor=a, mapping=m or mapping(), observation_time_key="time",
        action=a.array("action"),
    )


def test_keeps_all_real_intervals_without_interpolation():
    a = inputs()
    action, indices = run(a)
    np.testing.assert_array_equal(action, a.array("state")[1:])
    np.testing.assert_array_equal(indices, [0, 1, 2])
    assert a.array("state").shape == (4, 2)
    assert a.array("action")[-1, 0] == 99.


def test_old_as_recorded_mode_is_not_trimmed():
    a = inputs()
    action, indices = run(a, mapping(action_alignment="as_recorded"))
    np.testing.assert_array_equal(action, a.array("action"))
    np.testing.assert_array_equal(indices, a.array("frame_index"))


@pytest.mark.parametrize("change", [
    {"supervision": "fine_command"},
    {"action_time_key": "time"},
    {"state_time_key": None},
    {"state_transform": "zero"},
    {"action_transform": "open_to_close"},
    {"action_alignment": "guess_next_state"},
    {"action_value_mask": (True, False)},
])
def test_rejects_inconsistent_contract(change):
    with pytest.raises(AdapterContractError):
        run(inputs(), mapping(**change))


@pytest.mark.parametrize("field,value", [
    ("frame_index", np.asarray([0., 1.5, 2., 3.])),
    ("frame_index", np.asarray([1, 2, 3, 4])),
    ("frame_index", np.asarray([0, 1, 1, 3])),
    ("time", np.asarray([0., .03, .03, .12])),
    ("time", np.asarray([0., .03, np.nan, .12])),
])
def test_rejects_fabricated_or_invalid_interval_alignment(field, value):
    a = inputs()
    a.values[field] = value
    with pytest.raises(AdapterContractError):
        run(a)


def test_rejects_wrong_temporal_label():
    a = inputs()
    a.values["action"][0, 0] += 1.
    with pytest.raises(AdapterContractError, match="next observed state"):
        run(a)


def test_rejects_state_clock_shift():
    a = inputs()
    a.values["state_time"] = a.array("time") + .01
    with pytest.raises(AdapterContractError):
        run(a, mapping(state_time_key="state_time"))
