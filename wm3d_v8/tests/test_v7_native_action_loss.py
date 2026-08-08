from __future__ import annotations

import torch

from wm3d_v3.training.v7_native_action_loss import (
    NativeActionLossConfig,
    native_action_loss,
)


def _inputs(batch: int = 2, horizon: int = 8):
    target_physical = torch.zeros(batch, horizon, 6)
    target_physical[..., 0] = 0.01
    target_physical[..., 4] = -0.02
    mean = torch.tensor(
        [[0.001, 0.0, 0.0, 0.0, -0.002, 0.0]] * batch
    )
    std = torch.tensor(
        [[0.02, 0.02, 0.02, 0.04, 0.04, 0.04]] * batch
    )
    target_norm = (target_physical - mean[:, None]) / std[:, None]
    grip = torch.zeros(batch, horizon)
    grip[:, 3:] = 1.0
    previous = torch.zeros(batch, 1)
    return target_physical, target_norm, mean, std, grip, previous


def test_native_action_loss_prefers_correct_physical_direction_and_magnitude():
    target_physical, target_norm, mean, std, grip, previous = _inputs()
    correct = native_action_loss(
        target_norm.clone(),
        torch.where(grip > 0.5, 8.0, -8.0),
        target_norm,
        target_physical,
        grip,
        mean,
        std,
        previous_grip_close01=previous,
    )
    collapsed = native_action_loss(
        torch.zeros_like(target_norm),
        torch.zeros_like(grip),
        target_norm,
        target_physical,
        grip,
        mean,
        std,
        previous_grip_close01=previous,
    )
    assert correct["loss"] < collapsed["loss"]
    assert correct["translation_cosine"] > 0.999
    assert correct["rotation_cosine"] > 0.999
    assert correct["translation_magnitude_ratio"] > 0.999
    assert correct["rotation_magnitude_ratio"] > 0.999
    assert correct["translation_gain_vs_zero"] > 0.999
    assert correct["rotation_gain_vs_zero"] > 0.999
    assert correct["grip_event_recall"] == 1.0


def test_native_action_loss_is_finite_for_all_zero_actions_and_has_gradients():
    batch, horizon = 2, 8
    predicted = torch.zeros(batch, horizon, 6, requires_grad=True)
    grip_logits = torch.zeros(batch, horizon, requires_grad=True)
    zeros = torch.zeros_like(predicted)
    result = native_action_loss(
        predicted,
        grip_logits,
        zeros,
        zeros,
        torch.zeros(batch, horizon),
        torch.zeros(batch, 6),
        torch.ones(batch, 6),
        cfg=NativeActionLossConfig(),
    )
    assert all(torch.isfinite(value) for value in result.values())
    result["loss"].backward()
    assert predicted.grad is not None
    assert grip_logits.grad is not None


def test_physical_metrics_use_each_samples_own_statistics():
    target_physical, target_norm, mean, std, grip, previous = _inputs()
    # Different source statistics can represent the same physical target with
    # different normalized values.  Exact per-row de-normalization must still
    # produce a perfect physical gain for both rows.
    mean[1] = torch.tensor([-0.004, 0.003, 0.0, 0.002, 0.0, 0.0])
    std[1] = torch.tensor([0.01, 0.03, 0.02, 0.08, 0.02, 0.05])
    target_norm = (target_physical - mean[:, None]) / std[:, None]
    result = native_action_loss(
        target_norm,
        torch.where(grip > 0.5, 8.0, -8.0),
        target_norm,
        target_physical,
        grip,
        mean,
        std,
        previous_grip_close01=previous,
    )
    assert result["translation_gain_vs_zero"] > 0.999
    assert result["rotation_gain_vs_zero"] > 0.999
