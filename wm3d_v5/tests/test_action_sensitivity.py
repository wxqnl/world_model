from __future__ import annotations

import torch

from wm3d_v3.eval.action_sensitivity import (
    aggregate_metric_batches,
    compute_counterfactual_metrics,
    make_action_counterfactuals,
    parse_variants,
)


def test_make_action_counterfactuals_builds_expected_variants():
    actions = torch.tensor([
        [
            [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.20],
            [40.0, 50.0, 60.0, 4.0, 5.0, 6.0, 0.90],
        ]
    ])
    norm = torch.tensor([
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        ]
    ])

    variants = make_action_counterfactuals(
        actions,
        norm,
        variants=("zero", "sign_flip", "scaled", "grip_toggle"),
        scaled_factor=3.0,
    )

    assert set(variants) == {"real", "zero", "sign_flip", "scaled", "grip_toggle"}
    assert torch.allclose(variants["real"][..., :6], norm)
    assert torch.equal(variants["real"][..., 6], torch.tensor([[0.0, 1.0]]))
    assert torch.equal(variants["zero"], torch.zeros_like(variants["real"]))
    assert torch.allclose(variants["sign_flip"][..., :6], -norm)
    assert torch.equal(variants["sign_flip"][..., 6], variants["real"][..., 6])
    assert torch.allclose(variants["scaled"][..., :6], norm * 3.0)
    assert torch.equal(variants["scaled"][..., 6], variants["real"][..., 6])
    assert torch.allclose(variants["grip_toggle"][..., :6], norm)
    assert torch.equal(variants["grip_toggle"][..., 6], torch.tensor([[1.0, 0.0]]))


def test_make_action_counterfactuals_shuffles_deterministically_for_single_batch_item():
    actions = torch.tensor([[
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0],
    ]])
    gen = torch.Generator().manual_seed(0)

    variants = make_action_counterfactuals(actions, variants=("shuffled",), generator=gen)

    assert variants["shuffled"].shape == variants["real"].shape
    assert not torch.equal(variants["shuffled"], variants["real"])
    assert torch.equal(variants["shuffled"].sort(dim=1).values, variants["real"].sort(dim=1).values)


def test_parse_variants_accepts_comma_and_space_separated_values():
    assert parse_variants(["zero,scaled", "real", "grip_toggle"]) == ["zero", "scaled", "grip_toggle"]


def test_compute_counterfactual_metrics_and_aggregation():
    real_out = {
        "pred_tokens": torch.tensor([[[[1.0, 1.0]]], [[[2.0, 2.0]]]]),
        "depth": torch.tensor([[[[0.0, 1.0]]], [[[0.0, 1.0]]]]),
        "motion_hint": torch.tensor([[[[[0.0, 1.0]]]], [[[[1.0, 0.0]]]]]),
    }
    variant_out = {
        "pred_tokens": torch.tensor([[[[3.0, 3.0]]], [[[1.0, 1.0]]]]),
        "depth": torch.tensor([[[[1.0, 1.0]]], [[[0.5, 0.5]]]]),
        "motion_hint": torch.tensor([[[[[1.0, 1.0]]]], [[[[0.0, 0.0]]]]]),
    }
    targets = {
        "s_tgt": torch.tensor([[[[1.0, 1.0]]], [[[2.0, 2.0]]]]),
        "depth_tgt": torch.tensor([[[[0.0, 1.0]]], [[[0.0, 1.0]]]]),
        "motion_tgt": torch.tensor([[[[[0.0, 1.0]]]], [[[[1.0, 0.0]]]]]),
    }

    metrics = compute_counterfactual_metrics(real_out, variant_out, targets)

    assert torch.allclose(metrics["pred_tokens_mse_gap"], torch.tensor([4.0, 1.0]))
    assert torch.allclose(metrics["pred_tokens_gt_mse_gap"], torch.tensor([4.0, 1.0]))
    assert torch.equal(metrics["pred_tokens_gt_mse_acc"], torch.tensor([1.0, 1.0]))
    assert torch.equal(metrics["depth_gt_l1_acc"], torch.tensor([1.0, 1.0]))
    assert torch.equal(metrics["motion_hint_gt_l1_acc"], torch.tensor([1.0, 1.0]))

    summary = aggregate_metric_batches([
        {"x": torch.tensor([1.0, 3.0]), "acc": torch.tensor([1.0, 0.0])},
        {"x": torch.tensor([5.0]), "acc": torch.tensor([1.0])},
    ])

    assert summary == {"acc": 2.0 / 3.0, "x": 3.0}


def test_compute_counterfactual_metrics_resizes_motion_target_to_prediction():
    real_out = {
        "pred_tokens": torch.zeros(1, 1, 1, 2),
        "depth": torch.ones(1, 1, 1, 1),
        "motion_hint": torch.zeros(1, 1, 1, 1, 1),
    }
    variant_out = {
        "pred_tokens": torch.ones(1, 1, 1, 2),
        "depth": torch.full((1, 1, 1, 1), 2.0),
        "motion_hint": torch.ones(1, 1, 1, 1, 1),
    }
    targets = {
        "s_tgt": torch.zeros(1, 1, 1, 2),
        "depth_tgt": torch.ones(1, 1, 1, 1),
        "motion_tgt": torch.tensor([[[[[0.0, 1.0], [1.0, 1.0]]]]]),
    }

    metrics = compute_counterfactual_metrics(real_out, variant_out, targets)

    assert torch.equal(metrics["motion_hint_gt_l1_acc"], torch.tensor([1.0]))
