from __future__ import annotations

import torch

from wm3d_v3.eval.world3d_claim_eval import (
    _world_core_metrics,
    compute_3d_dynamics_metrics,
    motion_mask_from_rgb,
    summarize_counterfactual_showcase,
)


def test_motion_mask_and_3d_dynamics_focus_on_moving_region():
    context_rgb = torch.zeros(1, 3, 4, 4)
    rgb_tgt = torch.zeros(1, 2, 3, 4, 4)
    rgb_tgt[:, :, :, 0:2, 0:2] = 1.0
    context_depth = torch.ones(1, 4, 4)
    depth_tgt = torch.ones(1, 2, 4, 4)
    depth_tgt[:, :, 0:2, 0:2] = 3.0
    pred_depth = depth_tgt.clone()
    pred_depth[:, :, 2:, 2:] = 9.0

    mask = motion_mask_from_rgb(rgb_tgt, context_rgb, threshold=0.5)
    metrics = compute_3d_dynamics_metrics(pred_depth, depth_tgt, context_depth, rgb_tgt, context_rgb)

    assert mask.shape == (1, 2, 1, 4, 4)
    assert torch.allclose(metrics["motion_frac"], torch.tensor([0.25]))
    assert torch.allclose(metrics["motion_region_depth_l1"], torch.tensor([0.0]))
    assert metrics["global_depth_l1"].item() > metrics["motion_region_depth_l1"].item()
    assert torch.allclose(metrics["motion_region_depth_delta_l1"], torch.tensor([0.0]))


def test_counterfactual_showcase_summarizes_core_claim_rates():
    report = {
        "zero": {
            "pred_tokens_gt_mse_acc": 0.75,
            "depth_gt_l1_acc": 0.50,
            "motion_hint_gt_l1_acc": 0.25,
            "pred_tokens_gt_mse_gap": 0.02,
            "depth_gt_l1_gap": 0.03,
        },
        "sign_flip": {
            "pred_tokens_gt_mse_acc": 1.00,
            "depth_gt_l1_acc": 0.75,
            "motion_hint_gt_l1_acc": 0.50,
            "pred_tokens_gt_mse_gap": 0.04,
            "depth_gt_l1_gap": 0.01,
        },
    }

    summary = summarize_counterfactual_showcase(report)

    assert summary["mean_token_win_rate"] == 0.875
    assert summary["mean_depth_win_rate"] == 0.625
    assert summary["mean_motion_win_rate"] == 0.375
    assert summary["strongest_depth_variant"] == "zero"
    assert summary["claim"] == "real_action_beats_counterfactual_action"



def test_world_core_metrics_include_optional_point_and_pose_geometry():
    out = {
        "pred_tokens": torch.zeros(1, 2, 1, 3),
        "pose": torch.zeros(1, 2, 6),
        "gripper_logit": torch.ones(1, 2) * 10,
        "point": torch.zeros(1, 2, 4, 4, 3),
        "pose_geom": torch.zeros(1, 2, 9),
    }
    batch = {
        "s_tgt": torch.zeros(1, 2, 1, 3),
        "action_tgt": torch.ones(1, 2, 7),
        "point_tgt": torch.ones(1, 2, 2, 2, 3),
        "point_conf_tgt": torch.ones(1, 2, 2, 2),
        "pose_geom_tgt": torch.ones(1, 2, 9),
    }

    metrics = _world_core_metrics(out, batch)

    assert "world_point_l1" in metrics
    assert "camera_pose_enc_mse" in metrics
    assert torch.allclose(metrics["world_point_l1"], torch.tensor([1.0]))
    assert torch.allclose(metrics["camera_pose_enc_mse"], torch.tensor([1.0]))
