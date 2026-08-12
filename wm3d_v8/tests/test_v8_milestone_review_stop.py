from __future__ import annotations

import pytest

from wm3d_v3.training.train import resolve_milestone_review_stop_steps


def test_fail_closed_review_milestones_become_hard_stops() -> None:
    steps = resolve_milestone_review_stop_steps(
        {
            "milestone_reviews": {
                "fail_closed": True,
                "pause_on_missing_or_failed_review": True,
                "required_review_steps": [20, 100, 1000],
            }
        }
    )

    assert steps == frozenset({20, 100, 1000})


@pytest.mark.parametrize(
    "review_cfg",
    [
        None,
        {"fail_closed": False, "pause_on_missing_or_failed_review": True},
        {"fail_closed": True, "pause_on_missing_or_failed_review": False},
    ],
)
def test_review_stop_is_inactive_without_both_fail_closed_flags(
    review_cfg: dict | None,
) -> None:
    train_cfg = {} if review_cfg is None else {"milestone_reviews": review_cfg}

    assert resolve_milestone_review_stop_steps(train_cfg) == frozenset()


@pytest.mark.parametrize(
    "required_steps",
    [
        "1000",
        [0],
        [-1],
        [True],
        [1.5],
        [100, 100],
    ],
)
def test_invalid_review_milestones_fail_closed(required_steps: object) -> None:
    with pytest.raises(ValueError):
        resolve_milestone_review_stop_steps(
            {
                "milestone_reviews": {
                    "fail_closed": True,
                    "pause_on_missing_or_failed_review": True,
                    "required_review_steps": required_steps,
                }
            }
        )
