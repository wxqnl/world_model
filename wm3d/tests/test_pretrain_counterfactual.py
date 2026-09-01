from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from wm3d.training import pretrain
from wm3d.training.native_objective import NativeObjectiveConfig


def _objective(*, token: bool, rgb: bool) -> SimpleNamespace:
    return SimpleNamespace(
        action_counterfactual_token_advantage=1.0 if token else 0.0,
        action_counterfactual_rgb_advantage=1.0 if rgb else 0.0,
    )


def test_counterfactual_forward_uses_grad_token_control_and_detached_rgb_control(
    monkeypatch,
) -> None:
    fine = torch.randn(2, 3, requires_grad=True)
    coarse = torch.randn(2, 3, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    batch = {
        "future_factual_fine_action_values": fine,
        "future_factual_coarse_action_values": coarse,
        "future_factual_fine_action_mask": mask,
    }
    calls: list[dict[str, torch.Tensor]] = []
    controls: list[bool] = []

    def fake_forward(
        _model,
        value,
        *,
        appearance_teacher_ratio,
        compute_zero_action_control=False,
    ):
        assert appearance_teacher_ratio == 0.0
        calls.append(dict(value))
        controls.append(bool(compute_zero_action_control))
        signal = value["future_factual_fine_action_values"]
        result = {
            "pred_tokens": signal[:, None, None],
            "rgb": signal[:, None, None, :, None, None],
        }
        if compute_zero_action_control:
            result["zero_action_pred_tokens"] = result["pred_tokens"] * 0.0
        return result

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.0,
        objective=_objective(token=True, rgb=True),
    )

    assert controls == [True, False]
    assert len(calls) == 2
    torch.testing.assert_close(
        calls[0]["future_factual_fine_action_values"], fine
    )
    assert calls[1]["future_factual_fine_action_values"].count_nonzero() == 0
    assert calls[1]["future_factual_coarse_action_values"].count_nonzero() == 0
    assert calls[1]["future_factual_fine_action_mask"] is mask
    assert output["pred_tokens"].requires_grad
    assert output["zero_action_pred_tokens"].requires_grad
    assert not output["zero_action_rgb"].requires_grad


def test_counterfactual_forward_is_skipped_when_objective_is_disabled(
    monkeypatch,
) -> None:
    batch = {
        "future_factual_fine_action_values": torch.ones(1),
        "future_factual_coarse_action_values": torch.ones(1),
    }
    calls = 0

    def fake_forward(
        _model,
        _batch,
        *,
        appearance_teacher_ratio,
        compute_zero_action_control=False,
    ):
        nonlocal calls
        calls += 1
        assert appearance_teacher_ratio == 0.5
        assert not compute_zero_action_control
        return {"pred_tokens": torch.ones(1), "rgb": torch.ones(1)}

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.5,
        objective=_objective(token=False, rgb=False),
    )

    assert calls == 1
    assert "zero_action_pred_tokens" not in output
    assert "zero_action_rgb" not in output

def test_token_only_counterfactual_needs_one_model_forward(monkeypatch) -> None:
    fine = torch.randn(2, 3, requires_grad=True)
    batch = {
        "future_factual_fine_action_values": fine,
        "future_factual_coarse_action_values": torch.randn(2, 3),
    }
    controls: list[bool] = []

    def fake_forward(
        _model,
        value,
        *,
        appearance_teacher_ratio,
        compute_zero_action_control=False,
    ):
        controls.append(bool(compute_zero_action_control))
        signal = value["future_factual_fine_action_values"][:, None, None]
        result = {"pred_tokens": signal, "rgb": signal[..., None, None]}
        if compute_zero_action_control:
            result["zero_action_pred_tokens"] = signal * 0.0
        return result

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.0,
        objective=_objective(token=True, rgb=False),
    )

    assert controls == [True]
    assert output["zero_action_pred_tokens"].requires_grad
    assert "zero_action_rgb" not in output


def test_original_v7_rgb_action_schedule_matches_sparse_ramp() -> None:
    objective = NativeObjectiveConfig(
        context_pixel_action_rank_weight=2.0,
        context_pixel_action_separation_weight=0.5,
        context_pixel_action_rank_start_step=30_000,
        context_pixel_action_rank_ramp_steps=10_000,
        context_pixel_action_rank_every=8,
        context_pixel_action_rank_batch_size=1,
    )
    assert pretrain._scheduled_context_pixel_action_weights(
        objective, step=29_999
    ) == (0.0, 0.0)
    assert pretrain._scheduled_context_pixel_action_weights(
        objective, step=30_000
    ) == (0.0, 0.0)
    assert pretrain._scheduled_context_pixel_action_weights(
        objective, step=30_001
    ) == (0.0, 0.0)
    rank, separation = pretrain._scheduled_context_pixel_action_weights(
        objective, step=30_008
    )
    assert rank == pytest.approx(0.0016)
    assert separation == pytest.approx(0.0004)
    assert pretrain._scheduled_context_pixel_action_weights(
        objective, step=35_000
    ) == pytest.approx((1.0, 0.25))
    assert pretrain._scheduled_context_pixel_action_weights(
        objective, step=40_000
    ) == pytest.approx((2.0, 0.5))
    assert pretrain._scheduled_context_pixel_action_weights(
        objective, step=0, diagnostic_force=True
    ) == (2.0, 0.5)


def test_shuffled_rgb_action_forward_is_differentiable_and_subsampled(
    monkeypatch,
) -> None:
    batch_size = 3
    fine = torch.arange(
        batch_size * 2 * 1 * 1 * 2, dtype=torch.float32
    ).reshape(batch_size, 2, 1, 1, 2).requires_grad_()
    coarse = torch.arange(
        batch_size * 2 * 1 * 2, dtype=torch.float32
    ).reshape(batch_size, 2, 1, 2).requires_grad_()
    batch = {
        "future_factual_fine_action_values": fine,
        "future_factual_fine_action_mask": torch.ones_like(fine, dtype=torch.bool),
        "future_factual_fine_action_dt": torch.ones(batch_size, 2, 1, 1),
        "future_factual_fine_sample_mask": torch.ones(
            batch_size, 2, 1, 1, dtype=torch.bool
        ),
        "future_factual_coarse_action_values": coarse,
        "future_factual_coarse_action_mask": torch.ones_like(
            coarse, dtype=torch.bool
        ),
        "action_group_ids": torch.ones(batch_size, 1, dtype=torch.long),
        "action_group_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "action_semantic_ids": torch.ones(batch_size, 1, 2, dtype=torch.long),
        "embodiment_ids": torch.ones(batch_size, dtype=torch.long),
        "action_normalization_offset": torch.zeros(batch_size, 1, 2),
        "action_normalization_scale": torch.ones(batch_size, 1, 2),
    }
    calls: list[int] = []

    def fake_forward(
        _model,
        value,
        *,
        appearance_teacher_ratio,
        compute_zero_action_control=False,
    ):
        del appearance_teacher_ratio, compute_zero_action_control
        signal = value["future_factual_fine_action_values"].mean(
            dim=(1, 2, 3, 4)
        )
        calls.append(int(signal.shape[0]))
        return {
            "pred_tokens": signal[:, None, None],
            "rgb": signal[:, None, None, None, None, None].expand(
                -1, 1, 1, 3, 2, 2
            ),
        }

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    objective = NativeObjectiveConfig(
        context_pixel_action_rank_weight=2.0,
        context_pixel_action_separation_weight=0.5,
        context_pixel_action_rank_batch_size=1,
        context_pixel_action_negative_min_distance=0.0,
    )
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.0,
        objective=objective,
        step=0,
        diagnostic_force_context_pixel_action=True,
    )
    assert calls == [3, 1]
    assert output["shuffled_action_indices"].numel() == 1
    assert output["shuffled_action_valid"].item() is True
    assert output["shuffled_action_rgb"].requires_grad
    output["shuffled_action_rgb"].sum().backward()
    assert fine.grad is not None
    assert torch.isfinite(fine.grad).all()
    assert fine.grad.abs().sum() > 0


def test_invalid_shuffled_rgb_action_still_executes_fixed_shape_forward(
    monkeypatch,
) -> None:
    batch_size = 2
    fine = torch.zeros(batch_size, 1, 1, 1, 1)
    coarse = torch.zeros(batch_size, 1, 1, 1)
    batch = {
        "future_factual_fine_action_values": fine,
        "future_factual_fine_action_mask": torch.ones_like(fine, dtype=torch.bool),
        "future_factual_fine_action_dt": torch.ones(batch_size, 1, 1, 1),
        "future_factual_fine_sample_mask": torch.ones(
            batch_size, 1, 1, 1, dtype=torch.bool
        ),
        "future_factual_coarse_action_values": coarse,
        "future_factual_coarse_action_mask": torch.ones_like(
            coarse, dtype=torch.bool
        ),
        "action_group_ids": torch.ones(batch_size, 1, dtype=torch.long),
        "action_group_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "action_semantic_ids": torch.ones(batch_size, 1, 1, dtype=torch.long),
        "embodiment_ids": torch.ones(batch_size, dtype=torch.long),
        "action_normalization_offset": torch.zeros(batch_size, 1, 1),
        "action_normalization_scale": torch.ones(batch_size, 1, 1),
    }
    calls: list[int] = []

    def fake_forward(
        _model,
        value,
        *,
        appearance_teacher_ratio,
        compute_zero_action_control=False,
    ):
        del appearance_teacher_ratio, compute_zero_action_control
        size = int(value["future_factual_fine_action_values"].shape[0])
        calls.append(size)
        return {
            "pred_tokens": torch.zeros(size, 1, 1),
            "rgb": torch.zeros(size, 1, 1, 3, 2, 2),
        }

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.0,
        objective=NativeObjectiveConfig(
            context_pixel_action_rank_weight=2.0,
            context_pixel_action_separation_weight=0.5,
            context_pixel_action_rank_batch_size=1,
            context_pixel_action_negative_min_distance=0.05,
        ),
        step=0,
        diagnostic_force_context_pixel_action=True,
    )
    assert calls == [2, 1]
    assert output["shuffled_action_valid"].item() is False
    assert output["shuffled_action_valid_fraction"].item() == 0.0
