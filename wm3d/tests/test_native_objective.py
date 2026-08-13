from __future__ import annotations

import torch

from wm3d.data.grouped_robot import COMPOSITION_OPERATOR_IDS
from wm3d.training.native_objective import (
    NativeObjectiveConfig,
    compose_axis_angle_sequence,
    compose_policy_to_world_intervals,
    compute_native_objective,
)


def test_composition_uses_real_query_times_and_physical_operators() -> None:
    # One group: xyz sum, rotvec SO(3), gripper last.
    action = torch.zeros(1, 1, 5, 7)
    action[0, 0, :, 0] = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    action[0, 0, :, 3:6] = torch.tensor(
        [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3], [0, 0, 0.4], [0, 0, 0.5]]
    )
    action[0, 0, :, 6] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    mask = torch.ones_like(action, dtype=torch.bool)
    query_dt = torch.tensor([[[0.01, 0.07, 0.19, 0.21, 0.36]]])
    boundaries = torch.tensor([[0.0, 0.2, 0.4]])
    operators = torch.tensor(
        [[[
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["logical_last"],
        ]]]
    )
    composed, composed_mask = compose_policy_to_world_intervals(
        policy_action=action,
        policy_action_mask=mask,
        policy_query_dt=query_dt,
        future_world_boundaries_dt=boundaries,
        composition_operator_ids=operators,
    )

    torch.testing.assert_close(composed[0, :, 0, 0], torch.tensor([7.0, 24.0]))
    torch.testing.assert_close(composed[0, :, 0, 5], torch.tensor([0.6, 0.9]))
    torch.testing.assert_close(composed[0, :, 0, 6], torch.tensor([0.0, 0.0]))
    assert composed_mask.all()


def test_axis_angle_composition_is_differentiable() -> None:
    rotvec = torch.tensor(
        [[[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]],
        requires_grad=True,
    )
    composed, mask = compose_axis_angle_sequence(
        rotvec, torch.ones(1, 3, dtype=torch.bool)
    )
    composed.square().sum().backward()
    assert mask.item()
    assert rotvec.grad is not None
    assert torch.isfinite(rotvec.grad).all()
    assert rotvec.grad.abs().sum() > 0


def test_mixed_embodiment_batch_uses_per_sample_composition_operators() -> None:
    action = torch.tensor(
        [
            [[[[1.0], [2.0], [3.0]]]],
            [[[[1.0], [2.0], [3.0]]]],
        ]
    ).reshape(2, 1, 3, 1)
    mask = torch.ones_like(action, dtype=torch.bool)
    query_dt = torch.tensor([[[0.01, 0.05, 0.11]], [[0.01, 0.05, 0.11]]])
    boundaries = torch.tensor([[0.0, 0.2], [0.0, 0.2]])
    operators = torch.tensor(
        [
            [[COMPOSITION_OPERATOR_IDS["sum"]]],
            [[COMPOSITION_OPERATOR_IDS["last"]]],
        ]
    )
    composed, composed_mask = compose_policy_to_world_intervals(
        policy_action=action,
        policy_action_mask=mask,
        policy_query_dt=query_dt,
        future_world_boundaries_dt=boundaries,
        composition_operator_ids=operators,
    )
    torch.testing.assert_close(composed[:, 0, 0, 0], torch.tensor([6.0, 3.0]))
    assert composed_mask.all()


def test_time_weighted_mean_ignores_padded_query_clock_slots() -> None:
    action = torch.tensor([[[[2.0], [4.0], [99.0], [99.0]]]])
    mask = torch.tensor([[[[True], [True], [False], [False]]]])
    composed, composed_mask = compose_policy_to_world_intervals(
        policy_action=action,
        policy_action_mask=mask,
        policy_query_dt=torch.tensor([[[0.00, 0.05, 0.00, 0.00]]]),
        future_world_boundaries_dt=torch.tensor([[0.0, 0.2]]),
        composition_operator_ids=torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["time_weighted_mean"]]]]
        ),
    )
    # 2 holds for 0.05 s; the last real command 4 holds to the 0.2 s boundary.
    torch.testing.assert_close(composed[0, 0, 0, 0], torch.tensor(3.5))
    assert composed_mask[0, 0, 0, 0]


def test_coarse_only_batch_has_zero_fine_count_but_nonzero_policy_gradient() -> None:
    batch_size, groups, queries, action_dim, horizon, patches, token_dim = 1, 1, 4, 7, 2, 2, 4
    policy_raw = torch.zeros(
        batch_size, groups, queries, action_dim, requires_grad=True
    )
    policy_action = policy_raw.clone()
    output = {
        "pred_tokens": torch.zeros(batch_size, horizon, patches, token_dim, requires_grad=True),
        "rgb": torch.empty(batch_size, 0, 1, 3, 4, 4),
        "depth": torch.ones(batch_size, horizon, 1, patches),
        "point": torch.zeros(batch_size, horizon, 1, patches, 3),
        "camera_pose": torch.zeros(batch_size, horizon, 1, 9),
        "policy_action_raw": policy_raw,
        "policy_action_normalized": policy_raw,
        "policy_action": policy_action,
        "policy_action_mask": torch.ones_like(policy_raw, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy_raw, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy_raw, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.01, 0.08, 0.21, 0.31]]]),
    }
    operators = torch.tensor(
        [[[
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["sum"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["last"],
        ]]]
    )
    batch = {
        "target_tokens": torch.ones(batch_size, horizon, patches, token_dim),
        "target_fine_action": torch.zeros_like(policy_raw),
        "target_fine_action_mask": torch.zeros_like(policy_raw, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 0.2, 0.4]]),
        "composition_operator_ids": operators,
        "target_coarse_action": torch.ones(batch_size, horizon, groups, action_dim),
        "target_coarse_action_normalized": torch.ones(
            batch_size, horizon, groups, action_dim
        ),
        "target_coarse_action_mask": torch.ones(
            batch_size, horizon, groups, action_dim, dtype=torch.bool
        ),
        "action_normalization_offset": torch.zeros(batch_size, groups, action_dim),
        "action_normalization_scale": torch.ones(batch_size, groups, action_dim),
    }
    losses = compute_native_objective(
        output=output, batch=batch, config=NativeObjectiveConfig()
    )
    losses["total"].backward()
    assert losses["fine_supervised_dimensions"].item() == 0
    assert losses["coarse_supervised_dimensions"].item() > 0
    assert policy_raw.grad is not None
    assert torch.isfinite(policy_raw.grad).all()
    assert policy_raw.grad.abs().sum() > 0
