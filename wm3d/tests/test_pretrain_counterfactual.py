from __future__ import annotations

from types import SimpleNamespace

import torch

from wm3d.training import pretrain


def _objective(*, enabled: bool) -> SimpleNamespace:
    weight = 1.0 if enabled else 0.0
    return SimpleNamespace(
        action_counterfactual_token_advantage=weight,
        action_counterfactual_rgb_advantage=weight,
    )


def test_counterfactual_forward_zeros_values_and_detaches_control(
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

    def fake_forward(_model, value, *, appearance_teacher_ratio):
        assert appearance_teacher_ratio == 0.0
        calls.append(dict(value))
        signal = value["future_factual_fine_action_values"]
        return {
            "pred_tokens": signal[:, None, None],
            "rgb": signal[:, None, None, :, None, None],
        }

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.0,
        objective=_objective(enabled=True),
    )

    assert len(calls) == 2
    torch.testing.assert_close(
        calls[0]["future_factual_fine_action_values"], fine
    )
    assert calls[1]["future_factual_fine_action_values"].count_nonzero() == 0
    assert calls[1]["future_factual_coarse_action_values"].count_nonzero() == 0
    assert calls[1]["future_factual_fine_action_mask"] is mask
    assert output["pred_tokens"].requires_grad
    assert not output["zero_action_pred_tokens"].requires_grad
    assert not output["zero_action_rgb"].requires_grad


def test_counterfactual_forward_is_skipped_when_objective_is_disabled(
    monkeypatch,
) -> None:
    batch = {
        "future_factual_fine_action_values": torch.ones(1),
        "future_factual_coarse_action_values": torch.ones(1),
    }
    calls = 0

    def fake_forward(_model, _batch, *, appearance_teacher_ratio):
        nonlocal calls
        calls += 1
        assert appearance_teacher_ratio == 0.5
        return {"pred_tokens": torch.ones(1), "rgb": torch.ones(1)}

    monkeypatch.setattr(pretrain, "_forward", fake_forward)
    output = pretrain._forward_with_action_counterfactual(
        object(),
        batch,
        appearance_teacher_ratio=0.5,
        objective=_objective(enabled=False),
    )

    assert calls == 1
    assert "zero_action_pred_tokens" not in output
    assert "zero_action_rgb" not in output
