from __future__ import annotations

from types import SimpleNamespace

import torch

from wm3d.training import pretrain


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
