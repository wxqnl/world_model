"""Unit tests for VLA-fix building blocks."""
from __future__ import annotations
import torch
import pytest


def test_focal_bce_reduces_to_alpha_weighted_bce_when_gamma_zero():
    """With gamma=0 and alpha=0.5, focal = 0.5 * BCE per Lin et al. 2017."""
    from wm3d_v3.losses import focal_bce
    logits = torch.tensor([[-2.0, 0.0, 2.0]])
    targets = torch.tensor([[0.0, 1.0, 1.0]])
    f = focal_bce(logits, targets, alpha=0.5, gamma=0.0)
    b = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(f, 0.5 * b, atol=1e-5)


def test_focal_bce_downweights_easy_examples():
    from wm3d_v3.losses import focal_bce
    logits = torch.tensor([[10.0, -10.0]])
    targets = torch.tensor([[1.0, 0.0]])
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    foc = focal_bce(logits, targets, alpha=0.5, gamma=2.0)
    assert foc < bce


def test_huber_zero_on_exact_match():
    from wm3d_v3.losses import huber
    a = torch.randn(4, 6)
    assert huber(a, a, delta=1.0).item() == pytest.approx(0.0)


def test_huber_linear_far_from_target():
    from wm3d_v3.losses import huber
    a = torch.full((1, 1), 10.0)
    b = torch.zeros(1, 1)
    # delta=1 => |err|>1 => linear: |err|-0.5*delta = 10-0.5 = 9.5
    assert huber(a, b, delta=1.0).item() == pytest.approx(9.5, abs=1e-4)


def test_action_proj_head_shapes_and_denormalize():
    from wm3d_v3.models.action_proj import ActionProjHead
    mean = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    std = torch.tensor([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    head = ActionProjHead(z_dim=192, hidden=1024, n_layers=5,
                          stats_mean=mean, stats_std=std)
    z = torch.randn(2, 8, 192)
    out = head(z)
    assert out["pose_norm"].shape == (2, 8, 6)
    assert out["pose"].shape == (2, 8, 6)
    assert out["gripper_logit"].shape == (2, 8)
    denorm = out["pose_norm"] * std + mean
    assert torch.allclose(out["pose"], denorm, atol=1e-6)


def test_action_proj_head_unbounded_output():
    from wm3d_v3.models.action_proj import ActionProjHead
    mean = torch.zeros(6); std = torch.ones(6)
    head = ActionProjHead(z_dim=192, hidden=64, n_layers=2,
                          stats_mean=mean, stats_std=std)
    with torch.no_grad():
        head.pose_norm.weight.fill_(0.0)
        head.pose_norm.bias.copy_(torch.tensor([5.0] * 6))
    z = torch.zeros(1, 1, 192)
    out = head(z)
    # Old tanh*0.1 head would have clipped to 0.1; new head must exceed 1.0
    assert out["pose"].abs().max().item() > 1.0
