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
