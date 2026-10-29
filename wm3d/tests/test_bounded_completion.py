"""Completion receipt must cover natural step limit as well as the wall clock."""
import pytest
from wm3d.training.pretrain import _bounded_completion_reason


@pytest.mark.parametrize("deadline,completed,total,reached,expected", [
    (100., 783, 783, False, "configured_steps_completed"),
    (100., 600, 783, True, "wall_clock_budget"),
    (100., 600, 783, False, None),
    (None, 783, 783, False, None),
    (100., 783, 783, True, "wall_clock_budget"),
])
def test_bounded_completion_reason(deadline, completed, total, reached, expected):
    assert _bounded_completion_reason(deadline=deadline, completed=completed,
                                     total_steps=total, deadline_reached=reached) == expected
