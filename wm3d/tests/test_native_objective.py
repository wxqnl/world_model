from __future__ import annotations

import pytest
import torch

from wm3d.data.grouped_robot import COMPOSITION_OPERATOR_IDS
from wm3d.training.native_objective import (
    NativeObjectiveConfig,
    NativeObjectiveError,
    _charbonnier,
    _factual_control_temporal_advantage,
    _factual_zero_advantage,
    _masked_rgb_perceptual,
    _warp_token_grid_with_pixel_flow,
    compose_axis_angle_sequence,
    compose_policy_to_world_intervals,
    compute_native_objective,
)


def test_temporal_action_rank_keeps_physical_noop_one_sided() -> None:
    target = torch.tensor([0.0, 1.0, 3.0]).view(1, 3, 1, 1)
    factual = torch.tensor([0.0, 0.5, 1.0]).view_as(target).requires_grad_()
    control = torch.zeros_like(target, requires_grad=True)
    mask = torch.ones(1, 3, 1, 1, dtype=torch.bool)

    control_error, gain, rank, response = _factual_control_temporal_advantage(
        factual=factual,
        control=control,
        target=target,
        mask=mask,
        margin=0.01,
        epsilon=1.0e-6,
        symmetric_control_gradient=False,
    )
    assert control_error.item() > 0
    assert gain.item() > 0
    assert rank.item() == pytest.approx(0.0)
    assert response.item() > 0

    worse_factual = torch.zeros_like(target, requires_grad=True)
    better_control = torch.tensor([0.0, 0.9, 2.8]).view_as(target).requires_grad_()
    _, worse_gain, active_rank, _ = _factual_control_temporal_advantage(
        factual=worse_factual,
        control=better_control,
        target=target,
        mask=mask,
        margin=0.01,
        epsilon=1.0e-6,
        symmetric_control_gradient=False,
    )
    assert worse_gain.item() < 0
    assert active_rank.item() > 0
    active_rank.backward()
    assert worse_factual.grad is not None
    assert worse_factual.grad.abs().sum() > 0
    assert better_control.grad is None


def test_temporal_action_rank_backpropagates_real_wrong_action_symmetrically() -> None:
    target = torch.tensor([0.0, 1.0, 3.0]).view(1, 3, 1, 1)
    factual = torch.tensor([0.0, 0.2, 0.4]).view_as(target).requires_grad_()
    wrong = torch.tensor([0.0, 0.9, 2.8]).view_as(target).requires_grad_()
    mask = torch.ones(1, 3, 1, 1, dtype=torch.bool)

    _, gain_before, rank, _ = _factual_control_temporal_advantage(
        factual=factual,
        control=wrong,
        target=target,
        mask=mask,
        margin=0.01,
        epsilon=1.0e-6,
        symmetric_control_gradient=True,
    )
    assert gain_before.item() < 0
    assert rank.item() > 0
    rank.backward()
    assert factual.grad is not None
    assert wrong.grad is not None
    assert torch.isfinite(factual.grad).all()
    assert torch.isfinite(wrong.grad).all()
    assert factual.grad.abs().sum() > 0
    assert wrong.grad.abs().sum() > 0

    learning_rate = 0.01
    factual_after = (factual - learning_rate * factual.grad).detach()
    wrong_after = (wrong - learning_rate * wrong.grad).detach()
    _, gain_after, _, _ = _factual_control_temporal_advantage(
        factual=factual_after,
        control=wrong_after,
        target=target,
        mask=mask,
        margin=0.01,
        epsilon=1.0e-6,
        symmetric_control_gradient=True,
    )
    assert gain_after > gain_before


def test_flow_aligned_p256_target_uses_backward_pixel_transport() -> None:
    source = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 1, 4, 1)
    zero_flow = torch.zeros(1, 1, 1, 2, 2, 2)
    torch.testing.assert_close(
        _warp_token_grid_with_pixel_flow(
            source, zero_flow, image_height=2, image_width=2
        ),
        source,
        rtol=0,
        atol=0,
    )

    right_flow = zero_flow.clone()
    right_flow[:, :, :, 0] = 1.0
    warped = _warp_token_grid_with_pixel_flow(
        source, right_flow, image_height=2, image_width=2
    )
    expected = torch.tensor([2.0, 0.0, 4.0, 0.0]).view_as(source)
    torch.testing.assert_close(warped, expected, rtol=0, atol=1.0e-6)


def test_v8_core_detail_supervises_only_spatial_high_frequency() -> None:
    prediction = torch.zeros(1, 1, 1, 4, 2, requires_grad=True)
    raw_target = torch.tensor(
        [
            [
                [
                    [3.0, 1.0, -1.0, -3.0],
                    [-3.0, -1.0, 1.0, 3.0],
                    [2.0, 1.0, -1.0, -2.0],
                    [-2.0, -1.0, 1.0, 2.0],
                ]
            ]
        ]
    ).view(1, 1, 1, 4, 4)
    policy = torch.zeros(1, 1, 1, 1)
    output = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "appearance_pred_tokens": prediction,
        "appearance_pred_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
        "appearance_state_detail": torch.ones(()),
        "rgb": torch.empty(1, 0, 1, 3, 2, 2),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.5]]]),
    }
    batch = {
        "target_tokens": torch.zeros(1, 1, 1, 2),
        "target_appearance_tokens": raw_target,
        "target_appearance_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 1.0]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.zeros(1, 1, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            appearance_l1=1.0,
            rgb_l1=0.0,
            rgb_charbonnier=0.0,
            rgb_gradient=0.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=0.0,
            action_coarse=0.0,
        ),
    )
    assert losses["appearance_l1"].item() > 0.0
    assert losses["appearance_supervised_elements"].item() == 8
    losses["total"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_rgb_charbonnier_uses_its_own_meaningful_epsilon() -> None:
    value = torch.tensor([0.0, 1.0e-4, 0.1])
    legacy = _charbonnier(value, 1.0e-6)
    configured = _charbonnier(value, 1.0e-3)
    assert legacy[0].item() == pytest.approx(1.0e-6)
    assert configured[0].item() == pytest.approx(1.0e-3)
    assert configured[1] > legacy[1]
    assert configured[2].item() == pytest.approx(legacy[2].item(), rel=1.0e-4)
    with pytest.raises(NativeObjectiveError, match="rgb_charbonnier_epsilon"):
        NativeObjectiveConfig(rgb_charbonnier_epsilon=0.0).validate()


def test_rgb_motion_objective_separates_static_and_changed_pixels() -> None:
    rgb = torch.zeros(1, 1, 1, 3, 2, 2, requires_grad=True)
    motion_logit = torch.zeros(1, 1, 1, 1, 2, 2, requires_grad=True)
    target_rgb = torch.zeros_like(rgb)
    target_rgb[..., 0, 0] = 1.0
    policy = torch.zeros(1, 1, 1, 1)
    output = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "rgb": rgb,
        "rgb_motion_logit": motion_logit,
        "depth": torch.ones(1, 1, 1, 1),
        "point": torch.zeros(1, 1, 1, 1, 3),
        "camera_pose": torch.zeros(1, 1, 1, 9),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.5]]]),
    }
    batch = {
        "target_tokens": torch.zeros(1, 1, 1, 2),
        "target_rgb": target_rgb,
        "target_rgb_mask": torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool),
        "context_rgb": torch.zeros(1, 1, 3, 2, 2),
        "context_rgb_mask": torch.ones(1, 1, dtype=torch.bool),
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 1.0]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.zeros(1, 1, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            rgb_l1=0.0,
            rgb_charbonnier=0.0,
            rgb_gradient=0.0,
            rgb_motion_l1=1.0,
            rgb_motion_bce=1.0,
            rgb_motion_dice=1.0,
            rgb_motion_pos_weight=2.0,
            rgb_motion_threshold=0.03,
            rgb_motion_gain=3.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=0.0,
            action_coarse=0.0,
        ),
    )

    assert losses["rgb_motion_fraction"].item() == pytest.approx(0.25)
    assert losses["rgb_motion_region_l1"].item() == pytest.approx(1.0)
    assert losses["rgb_static_region_l1"].item() == pytest.approx(0.0)
    assert losses["rgb_motion_l1"].item() == pytest.approx(1.0)
    assert losses["rgb_motion_bce"].item() == pytest.approx(
        1.25 * torch.log(torch.tensor(2.0)).item()
    )
    assert losses["rgb_motion_dice"].item() == pytest.approx(2.0 / 3.0)
    losses["total"].backward()
    assert rgb.grad is not None and rgb.grad.abs().sum() > 0
    assert motion_logit.grad is not None and motion_logit.grad.abs().sum() > 0


def test_v7_signed_temporal_geometry_is_relative_and_directional() -> None:
    target_depth = torch.tensor(
        [[[[1.0, 2.0, 3.0, 1000.0]], [[1.0, 3.0, 5.0, -1000.0]]]]
    )
    depth_mask = torch.tensor(
        [[[[True, True, True, False]], [[True, True, True, False]]]]
    )
    target_point = torch.zeros(1, 2, 1, 4, 3)
    target_point[0, 0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 1000.0])
    target_point[0, 1, 0, :, 0] = torch.tensor([1.0, 2.0, 4.0, -1000.0])
    point_mask = depth_mask.clone()
    policy = torch.zeros(1, 1, 1, 1)
    batch = {
        "target_tokens": torch.zeros(1, 2, 1, 2),
        "target_depth": target_depth,
        "target_depth_mask": depth_mask,
        "target_point": target_point,
        "target_point_mask": point_mask,
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 1.0, 2.0]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.zeros(1, 2, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 2, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    config = NativeObjectiveConfig(
        token_mse=0.0,
        token_cosine=0.0,
        rgb_l1=0.0,
        rgb_charbonnier=0.0,
        rgb_gradient=0.0,
        depth_log=0.0,
        depth_temporal=0.2,
        point=0.0,
        point_temporal=0.05,
        camera_pose=0.0,
        action_fine=0.0,
        action_coarse=0.0,
    )

    def evaluate(
        depth_prediction: torch.Tensor,
        point_prediction: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = {
            "pred_tokens": torch.zeros(1, 2, 1, 2),
            "rgb": torch.empty(1, 0, 1, 3, 2, 2),
            "depth": depth_prediction,
            "point": point_prediction,
            "camera_pose": torch.zeros(1, 2, 1, 9),
            "policy_action_raw": policy,
            "policy_action_normalized": policy,
            "policy_action": policy,
            "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
            "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
            "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
            "policy_query_dt": torch.tensor([[[0.5]]]),
        }
        return compute_native_objective(output=output, batch=batch, config=config)

    scaled_depth = target_depth.clone()
    scaled_depth[:, 0] *= 2.0
    scaled_depth[:, 1] *= 3.0
    scaled_point = target_point * 2.0 + 7.0
    matching = evaluate(scaled_depth, scaled_point)
    assert matching["depth_temporal"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert matching["point_temporal"].item() == pytest.approx(0.0, abs=1.0e-6)

    reversed_depth = target_depth.flip(1).clone().requires_grad_(True)
    reversed_point = target_point.flip(1).clone().requires_grad_(True)
    reversed_losses = evaluate(reversed_depth, reversed_point)
    assert reversed_losses["depth_temporal"].item() > 0.0
    assert reversed_losses["point_temporal"].item() > 0.0
    torch.testing.assert_close(
        reversed_losses["total"],
        0.2 * reversed_losses["depth_temporal"]
        + 0.05 * reversed_losses["point_temporal"],
    )
    reversed_losses["total"].backward()
    assert reversed_depth.grad is not None and reversed_depth.grad.abs().sum() > 0
    assert reversed_point.grad is not None and reversed_point.grad.abs().sum() > 0


@pytest.mark.parametrize("selected_valid", [True, False])
def test_original_v7_rgb_action_rank_trains_factual_and_wrong_graphs(
    selected_valid: bool,
) -> None:
    batch_size = 2
    factual = torch.full(
        (batch_size, 1, 1, 3, 2, 2), 0.8, requires_grad=True
    )
    wrong = torch.full((1, 1, 1, 3, 2, 2), 0.2, requires_grad=True)
    policy = torch.zeros(batch_size, 1, 1, 1)
    output = {
        "pred_tokens": torch.zeros(batch_size, 1, 1, 2),
        "rgb": factual,
        "shuffled_action_rgb": wrong,
        "shuffled_action_indices": torch.tensor([1]),
        "shuffled_action_valid": torch.tensor([selected_valid]),
        "shuffled_action_valid_fraction": torch.tensor(
            1.0 if selected_valid else 0.0
        ),
        "shuffled_action_distance": torch.tensor([0.5]),
        "context_pixel_action_rank_weight": torch.tensor(2.0),
        "context_pixel_action_separation_weight": torch.tensor(0.5),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.5]], [[0.5]]]),
    }
    batch = {
        "target_tokens": torch.zeros(batch_size, 1, 1, 2),
        "target_rgb": torch.ones(batch_size, 1, 1, 3, 2, 2),
        "target_rgb_mask": torch.ones(
            batch_size, 1, 1, 1, 1, 1, dtype=torch.bool
        ),
        "context_rgb": torch.zeros(batch_size, 1, 3, 2, 2),
        "context_rgb_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]]
        ),
        "composition_operator_ids": torch.tensor(
            [
                [[COMPOSITION_OPERATOR_IDS["last"]]],
                [[COMPOSITION_OPERATOR_IDS["last"]]],
            ]
        ),
        "target_coarse_action_normalized": torch.zeros(
            batch_size, 1, 1, 1
        ),
        "target_coarse_action_mask": torch.ones(
            batch_size, 1, 1, 1, dtype=torch.bool
        ),
        "action_normalization_offset": torch.zeros(batch_size, 1, 1),
        "action_normalization_scale": torch.ones(batch_size, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            rgb_l1=0.0,
            rgb_charbonnier=0.0,
            rgb_gradient=0.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=0.0,
            action_coarse=0.0,
            context_pixel_action_rank_weight=2.0,
            context_pixel_action_separation_weight=0.5,
            context_pixel_action_rank_batch_size=1,
            context_pixel_action_rank_margin=1.0,
            context_pixel_action_separation_margin=1.0,
        ),
    )
    expected = 1.0 if selected_valid else 0.0
    assert losses["context_pixel_action_gap"].item() == pytest.approx(0.6 * expected)
    assert losses["context_pixel_action_rgb_gap"].item() == pytest.approx(
        0.6 * expected
    )
    assert losses["context_pixel_action_rank"].item() == pytest.approx(
        0.4 * expected
    )
    assert losses["context_pixel_action_separation"].item() == pytest.approx(
        0.4 * expected
    )
    assert losses["context_pixel_action_valid_fraction"].item() == expected
    losses["total"].backward()
    assert factual.grad is not None
    assert wrong.grad is not None
    if selected_valid:
        assert factual.grad.abs().sum() > 0
        assert wrong.grad.abs().sum() > 0
    else:
        assert factual.grad.count_nonzero() == 0
        assert wrong.grad.count_nonzero() == 0


@pytest.mark.parametrize("pixel_units", [False, True])
@pytest.mark.parametrize("rgb_size", [4, 256])
def test_rgb_transport_objective_supervises_actual_flow_and_visibility(
    pixel_units: bool, rgb_size: int
) -> None:
    rgb = torch.zeros(1, 1, 1, 3, rgb_size, rgb_size)
    flow = torch.zeros(1, 1, 1, 2, 4, 4, requires_grad=True)
    disocclusion_logit = torch.zeros(
        1, 1, 1, 1, 4, 4, requires_grad=True
    )
    flow_target = torch.zeros(1, 1, 1, 2, 2, 2)
    flow_target[:, :, :, 0] = 4.0
    disocclusion_target = torch.zeros(1, 1, 1, 1, 2, 2)
    disocclusion_target[..., 0, 0] = 1.0
    policy = torch.zeros(1, 1, 1, 1)
    output = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "rgb": rgb,
        "rgb_flow_pixels": flow,
        "rgb_disocclusion_logit": disocclusion_logit,
        "depth": torch.ones(1, 1, 1, 1),
        "point": torch.zeros(1, 1, 1, 1, 3),
        "camera_pose": torch.zeros(1, 1, 1, 9),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.5]]]),
    }
    batch = {
        "target_tokens": torch.zeros(1, 1, 1, 2),
        "target_rgb": torch.zeros_like(rgb),
        "target_rgb_mask": torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool),
        "context_rgb": torch.zeros(1, 1, 3, rgb_size, rgb_size),
        "context_rgb_mask": torch.ones(1, 1, dtype=torch.bool),
        "rgb_flow_target_pixels": flow_target,
        "rgb_disocclusion_target": disocclusion_target,
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 1.0]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.zeros(1, 1, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            rgb_l1=0.0,
            rgb_charbonnier=0.0,
            rgb_gradient=0.0,
            rgb_flow_teacher=1.0,
            rgb_flow_teacher_pixel_units=pixel_units,
            rgb_disocclusion_bce=1.0,
            rgb_disocclusion_dice=1.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=0.0,
            action_coarse=0.0,
        ),
    )

    assert losses["rgb_flow_epe"].item() == pytest.approx(4.0, rel=1.0e-5)
    divisor = 1.0 if pixel_units else rgb_size / 2.0
    assert losses["rgb_flow_teacher"].item() == pytest.approx(
        4.0 / divisor, rel=1.0e-5
    )
    assert losses["rgb_flow_prediction_magnitude"].item() == pytest.approx(0.0)
    assert losses["rgb_flow_target_magnitude"].item() == pytest.approx(4.0)
    assert losses["rgb_flow_moving_epe"].item() == pytest.approx(4.0, rel=1.0e-5)
    assert losses["rgb_flow_static_epe"].item() == pytest.approx(0.0)
    assert losses["rgb_flow_magnitude_ratio"].item() == pytest.approx(0.0)
    assert losses["rgb_disocclusion_bce"].item() == pytest.approx(
        torch.log(torch.tensor(2.0)).item()
    )
    assert losses["rgb_disocclusion_dice"].item() == pytest.approx(2.0 / 3.0)
    assert losses["rgb_disocclusion_fraction"].item() == pytest.approx(0.25)
    losses["total"].backward()
    assert flow.grad is not None and flow.grad.abs().sum() > 0
    # The same 4px error gives the same pixel-space gradient at both image
    # sizes. Invalid/occluded teacher locations cannot contribute.
    assert float(flow.grad[:, :, :, 0].sum()) == pytest.approx(
        -1.0 / divisor, rel=1.0e-5
    )
    assert (
        disocclusion_logit.grad is not None
        and disocclusion_logit.grad.abs().sum() > 0
    )


def test_rgb_flow_teacher_balances_sparse_motion_and_static_regions() -> None:
    rgb = torch.zeros(1, 1, 1, 3, 4, 4)
    flow = torch.zeros(1, 1, 1, 2, 4, 4, requires_grad=True)
    flow_target = torch.zeros(1, 1, 1, 2, 2, 2)
    flow_target[..., 0, 0, 0] = 4.0
    policy = torch.zeros(1, 1, 1, 1)
    output = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "rgb": rgb,
        "rgb_flow_pixels": flow,
        "rgb_disocclusion_logit": torch.zeros(1, 1, 1, 1, 4, 4),
        "depth": torch.ones(1, 1, 1, 1),
        "point": torch.zeros(1, 1, 1, 1, 3),
        "camera_pose": torch.zeros(1, 1, 1, 9),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.5]]]),
    }
    batch = {
        "target_tokens": torch.zeros(1, 1, 1, 2),
        "target_rgb": torch.zeros_like(rgb),
        "target_rgb_mask": torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool),
        "context_rgb": torch.zeros(1, 1, 3, 4, 4),
        "context_rgb_mask": torch.ones(1, 1, dtype=torch.bool),
        "rgb_flow_target_pixels": flow_target,
        "rgb_disocclusion_target": torch.zeros(1, 1, 1, 1, 2, 2),
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 1.0]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.zeros(1, 1, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            rgb_l1=0.0,
            rgb_charbonnier=0.0,
            rgb_gradient=0.0,
            rgb_flow_teacher=1.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=0.0,
            action_coarse=0.0,
        ),
    )
    # One moving pixel and three static pixels each contribute half of the
    # per-sample objective. The old global mean returned approximately 2.0.
    assert losses["rgb_flow_teacher"].item() == pytest.approx(1.0, abs=5.0e-4)
    assert losses["rgb_flow_moving_epe"].item() == pytest.approx(4.0, rel=1.0e-5)
    assert losses["rgb_flow_static_epe"].item() == pytest.approx(0.001, rel=1.0e-5)
    losses["total"].backward()
    assert flow.grad is not None and flow.grad.abs().sum() > 0


def test_normalized_appearance_motion_l1_supervises_changed_patches() -> None:
    appearance_prediction = torch.zeros(1, 1, 1, 4, 2, requires_grad=True)
    with torch.no_grad():
        appearance_prediction[0, 0, 0, 0] = torch.tensor([-1.0, 1.0])
        # A large static-patch error must not leak into the motion-only metric.
        appearance_prediction[0, 0, 0, 1] = torch.tensor([10.0, 10.0])
    appearance_target = torch.zeros_like(appearance_prediction)
    appearance_target[0, 0, 0, 0] = torch.tensor([1.0, -1.0])
    target_rgb = torch.zeros(1, 1, 1, 3, 2, 2)
    target_rgb[..., 0, 0] = 1.0
    policy = torch.zeros(1, 1, 1, 1)
    output = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "appearance_pred_tokens": appearance_prediction,
        "appearance_pred_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
        "rgb": torch.zeros_like(target_rgb),
        "depth": torch.ones(1, 1, 1, 1),
        "point": torch.zeros(1, 1, 1, 1, 3),
        "camera_pose": torch.zeros(1, 1, 1, 9),
        "policy_action_raw": policy,
        "policy_action_normalized": policy,
        "policy_action": policy,
        "policy_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
        "policy_query_dt": torch.tensor([[[0.5]]]),
    }
    batch = {
        "target_tokens": torch.zeros(1, 1, 1, 2),
        "target_appearance_tokens": appearance_target,
        "target_appearance_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
        "appearance_context_tokens": torch.zeros(1, 1, 1, 4, 2),
        "appearance_context_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool),
        "target_rgb": target_rgb,
        "target_rgb_mask": torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool),
        "context_rgb": torch.zeros(1, 1, 3, 2, 2),
        "context_rgb_mask": torch.ones(1, 1, dtype=torch.bool),
        "target_fine_action": torch.zeros_like(policy),
        "target_fine_action_mask": torch.ones_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor([[0.0, 1.0]]),
        "composition_operator_ids": torch.tensor(
            [[[COMPOSITION_OPERATOR_IDS["last"]]]]
        ),
        "target_coarse_action_normalized": torch.zeros(1, 1, 1, 1),
        "target_coarse_action_mask": torch.ones(1, 1, 1, 1, dtype=torch.bool),
        "action_normalization_offset": torch.zeros(1, 1, 1),
        "action_normalization_scale": torch.ones(1, 1, 1),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            token_mse=0.0,
            token_cosine=0.0,
            appearance_motion_l1=1.0,
            rgb_l1=0.0,
            rgb_charbonnier=0.0,
            rgb_gradient=0.0,
            depth_log=0.0,
            point=0.0,
            camera_pose=0.0,
            action_fine=0.0,
            action_coarse=0.0,
        ),
    )

    expected = (
        appearance_prediction[0, 0, 0, 0]
        - torch.nn.functional.layer_norm(
            appearance_target[0, 0, 0, 0],
            (appearance_target.shape[-1],),
        )
    ).abs().mean()
    assert losses["appearance_motion_fraction"].item() == pytest.approx(0.25)
    assert losses["appearance_motion_l1"].item() == pytest.approx(expected.item())
    assert losses["total"].item() == pytest.approx(expected.item())
    losses["total"].backward()
    assert appearance_prediction.grad is not None
    assert appearance_prediction.grad[0, 0, 0, 0].abs().sum() > 0
    assert appearance_prediction.grad[0, 0, 0, 1].abs().sum() == 0


def test_factual_zero_advantage_updates_both_sides_of_the_ranking() -> None:
    factual = torch.full((2, 1, 1, 2), 0.2, requires_grad=True)
    zero_action = torch.full((2, 1, 1, 2), 0.2, requires_grad=True)
    target = torch.zeros_like(factual)
    mask = torch.ones(2, 1, 1, dtype=torch.bool)

    zero_error, gain, advantage, response_rms = _factual_zero_advantage(
        factual=factual,
        zero_action=zero_action,
        target=target,
        mask=mask[..., None],
        margin=0.01,
        epsilon=1.0e-6,
        absolute_error=False,
    )

    assert zero_error.item() == pytest.approx(0.04)
    assert gain.item() == pytest.approx(0.0)
    assert advantage.item() == pytest.approx(0.01)
    assert response_rms.item() == pytest.approx(0.0)
    advantage.backward()
    assert factual.grad is not None
    assert zero_action.grad is not None
    assert bool((factual.grad > 0).all())
    assert bool((zero_action.grad < 0).all())
    with torch.no_grad():
        updated_factual = factual - 0.1 * factual.grad
        updated_zero = zero_action - 0.1 * zero_action.grad
        updated_gain = updated_zero.square().mean() - updated_factual.square().mean()
    assert updated_gain > 0


def test_factual_zero_advantage_rewards_a_real_conditioning_gain() -> None:
    factual = torch.zeros(2, 1, 1, 2)
    zero_action = torch.ones_like(factual)
    target = torch.zeros_like(factual)
    mask = torch.tensor([[[True]], [[False]]])

    zero_error, gain, advantage, response_rms = _factual_zero_advantage(
        factual=factual,
        zero_action=zero_action,
        target=target,
        mask=mask[..., None],
        margin=0.01,
        epsilon=1.0e-6,
        absolute_error=False,
    )

    assert zero_error.item() == pytest.approx(1.0)
    assert gain.item() == pytest.approx(1.0)
    assert advantage.item() == pytest.approx(0.0)
    assert response_rms.item() == pytest.approx(1.0)
    with pytest.raises(
        NativeObjectiveError,
        match="action_counterfactual_token_margin",
    ):
        NativeObjectiveConfig(
            action_counterfactual_token_advantage=1.0
        ).validate()


def test_factual_zero_rgb_advantage_uses_l1_error() -> None:
    factual = torch.full((1, 1, 1, 1, 1, 1), 0.2)
    zero_action = torch.full_like(factual, 0.4)
    target = torch.zeros_like(factual)
    mask = torch.ones(1, 1, 1, 1, 1, 1, dtype=torch.bool)

    zero_error, gain, advantage, response_rms = _factual_zero_advantage(
        factual=factual,
        zero_action=zero_action,
        target=target,
        mask=mask,
        margin=0.1,
        epsilon=1.0e-6,
        absolute_error=True,
    )

    assert zero_error.item() == pytest.approx(0.4)
    assert gain.item() == pytest.approx(0.2)
    assert advantage.item() == pytest.approx(0.0)
    assert response_rms.item() == pytest.approx(0.2)


class _MeanFeatureDistance(torch.nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().mean(dim=(1, 2, 3), keepdim=True)


class _RecordingMeanFeatureDistance(_MeanFeatureDistance):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(prediction.shape[0]))
        return super().forward(prediction, target)


class _FrozenNonlinearFeatureDistance(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(0.7))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_features = torch.tanh(prediction * self.scale)
        target_features = torch.tanh(target * self.scale)
        return (prediction_features - target_features).square().mean(
            dim=(1, 2, 3), keepdim=True
        )


def test_perceptual_rgb_loss_masks_whole_views_and_backpropagates() -> None:
    prediction = torch.full((1, 2, 2, 3, 8, 8), 0.75, requires_grad=True)
    target = torch.full_like(prediction, 0.25)
    mask = torch.zeros(1, 2, 2, 1, 1, 1, dtype=torch.bool)
    mask[:, :, 0] = True

    loss = _masked_rgb_perceptual(
        prediction, target, mask, _MeanFeatureDistance()
    )
    loss.backward()
    assert loss.item() > 0
    assert prediction.grad is not None
    assert prediction.grad[:, :, 0].abs().sum() > 0
    assert prediction.grad[:, :, 1].count_nonzero() == 0

    partial = torch.zeros_like(prediction, dtype=torch.bool)
    partial[..., :4, :] = True
    with pytest.raises(NativeObjectiveError, match="whole-image"):
        _masked_rgb_perceptual(prediction, target, partial, _MeanFeatureDistance())


def test_perceptual_rgb_loss_uses_configured_execution_chunks() -> None:
    prediction = torch.full((1, 2, 2, 3, 8, 8), 0.75)
    target = torch.full_like(prediction, 0.25)
    mask = torch.ones(1, 2, 2, 1, 1, 1, dtype=torch.bool)
    model = _RecordingMeanFeatureDistance()

    with torch.no_grad():
        loss = _masked_rgb_perceptual(
            prediction,
            target,
            mask,
            model,
            chunk_size=3,
        )

    assert model.batch_sizes == [3, 1]
    assert loss.item() == pytest.approx(1.0)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(NativeObjectiveError, match="chunk size"):
            _masked_rgb_perceptual(
                prediction,
                target,
                mask,
                model,
                chunk_size=invalid,
            )


def test_frozen_perceptual_surrogate_preserves_scalar_and_first_order_gradient() -> None:
    generator = torch.Generator().manual_seed(712)
    prediction = torch.rand(
        (1, 2, 2, 3, 8, 8), generator=generator, requires_grad=True
    )
    reference_prediction = prediction.detach().clone().requires_grad_(True)
    target = torch.rand(prediction.shape, generator=generator)
    mask = torch.ones(1, 2, 2, 1, 1, 1, dtype=torch.bool)
    model = _FrozenNonlinearFeatureDistance().eval()

    optimized = _masked_rgb_perceptual(
        prediction,
        target,
        mask,
        model,
        chunk_size=3,
    )
    optimized.backward()
    reference = model(
        reference_prediction.reshape(-1, 3, 8, 8).mul(2.0).sub(1.0),
        target.reshape(-1, 3, 8, 8).mul(2.0).sub(1.0),
    ).float().sum() / 4
    reference.backward()

    # Chunked accumulation can differ from one monolithic reduction by one
    # fp32 rounding unit; the loss definition and image gradient are unchanged.
    torch.testing.assert_close(
        optimized.detach(), reference.detach(), rtol=1.0e-6, atol=1.0e-7
    )
    torch.testing.assert_close(
        prediction.grad,
        reference_prediction.grad,
        rtol=1.0e-6,
        atol=1.0e-7,
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


def test_composition_never_reads_per_dimension_tensor_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action = torch.arange(24, dtype=torch.float32).view(2, 2, 3, 2)
    mask = torch.ones_like(action, dtype=torch.bool)
    query_dt = torch.tensor(
        [[[0.01, 0.07, 0.15], [0.01, 0.07, 0.15]]] * 2
    )
    boundaries = torch.tensor([[0.0, 0.2], [0.0, 0.2]])
    operators = torch.full(
        (2, 2, 2), COMPOSITION_OPERATOR_IDS["sum"], dtype=torch.long
    )

    def reject_item(_value: torch.Tensor) -> float:
        raise AssertionError("composition attempted a per-dimension Tensor.item()")

    monkeypatch.setattr(torch.Tensor, "item", reject_item)
    composed, composed_mask = compose_policy_to_world_intervals(
        policy_action=action,
        policy_action_mask=mask,
        policy_query_dt=query_dt,
        future_world_boundaries_dt=boundaries,
        composition_operator_ids=operators,
    )

    torch.testing.assert_close(composed[:, 0], action.sum(dim=2))
    assert composed_mask.all()


def test_adjacent_left_and_right_so3_triplets_remain_independent() -> None:
    action = torch.tensor(
        [[[[0.10, 0.00, 0.00, 0.10, 0.00, 0.00],
           [0.00, 0.20, 0.00, 0.00, 0.20, 0.00],
           [0.00, 0.00, 0.30, 0.00, 0.00, 0.30]]]]
    )
    mask = torch.ones_like(action, dtype=torch.bool)
    operators = torch.tensor(
        [[[
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_body_right"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_body_right"],
            COMPOSITION_OPERATOR_IDS["so3_axis_angle_body_right"],
        ]]]
    )
    composed, composed_mask = compose_policy_to_world_intervals(
        policy_action=action,
        policy_action_mask=mask,
        policy_query_dt=torch.tensor([[[0.01, 0.07, 0.15]]]),
        future_world_boundaries_dt=torch.tensor([[0.0, 0.2]]),
        composition_operator_ids=operators,
    )
    left, _ = compose_axis_angle_sequence(
        action[0, 0, :, :3].unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        left_multiply=True,
    )
    right, _ = compose_axis_angle_sequence(
        action[0, 0, :, 3:].unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        left_multiply=False,
    )

    torch.testing.assert_close(composed[0, 0, 0, :3], left[0])
    torch.testing.assert_close(composed[0, 0, 0, 3:], right[0])
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
    appearance_pred = torch.zeros(
        batch_size, horizon, 1, patches, token_dim, requires_grad=True
    )
    output = {
        "pred_tokens": torch.zeros(batch_size, horizon, patches, token_dim, requires_grad=True),
        "zero_action_pred_tokens": torch.full(
            (batch_size, horizon, patches, token_dim), 2.0
        ),
        "appearance_pred_tokens": appearance_pred,
        "appearance_pred_mask": torch.ones_like(appearance_pred[..., 0], dtype=torch.bool),
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
        "target_appearance_tokens": torch.tensor(
            [1.0, -1.0, 1.0, -1.0]
        ).view(1, 1, 1, 1, token_dim).expand_as(appearance_pred).clone(),
        "target_appearance_mask": torch.ones_like(appearance_pred[..., 0], dtype=torch.bool),
    }
    losses = compute_native_objective(
        output=output,
        batch=batch,
        config=NativeObjectiveConfig(
            appearance_mse=1.0,
            action_counterfactual_token_advantage=1.0,
            action_counterfactual_token_margin=0.01,
        ),
    )
    losses["total"].backward()
    assert losses["fine_supervised_dimensions"].item() == 0
    assert losses["coarse_supervised_dimensions"].item() > 0
    assert policy_raw.grad is not None
    assert torch.isfinite(policy_raw.grad).all()
    assert losses["appearance_mse"].item() > 0
    assert losses["zero_action_token_mse"].item() == pytest.approx(1.0)
    assert losses["action_counterfactual_token_gain"].item() == pytest.approx(0.0)
    # A temporally stationary target has no action-direction signal and is
    # intentionally excluded from the trajectory rank.
    assert losses["action_counterfactual_token_advantage"].item() == 0.0
    assert losses["action_counterfactual_token_temporal_gain"].item() == 0.0
    assert losses["action_counterfactual_token_response_rms"].item() == pytest.approx(
        2.0
    )
    assert appearance_pred.grad is not None
    assert appearance_pred.grad.abs().sum() > 0
    assert policy_raw.grad.abs().sum() > 0
