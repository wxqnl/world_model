from __future__ import annotations

import pytest
import torch


def _tiny_joint_config():
    from wm3d_v3.models.action_stream import ActionConfig
    from wm3d_v3.models.dual_stream import DualConfig
    from wm3d_v3.models.joint_model import JointConfig
    from wm3d_v3.models.state_stream import StateConfig

    sc = StateConfig(
        T=2, P=64, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
        cond_dim=16, action_cond_dim=7,
    )
    ac = ActionConfig(
        T=2, P=64, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
        z_dim=8, cond_dim=16, action_cond_dim=7,
    )
    return JointConfig(
        dual=DualConfig(state=sc, action=ac, xattn_layers_state=(), xattn_n_heads=4),
        action_proj_hidden=32,
        action_proj_layers=2,
        geom_hidden=16,
        pixel_hidden=16,
        pixel_n_res=1,
        enable_pixel=True,
        enable_context_pixel=True,
        context_pixel_hidden=32,
        context_pixel_residual_scale=0.75,
        context_pixel_predict_motion=True,
        context_pixel_motion_blend_gain=0.5,
        enable_bridging=False,
    )


def test_joint_model_context_renderer_requires_context_rgb():
    from wm3d_v3.models.joint_model import JointWorldModel

    model = JointWorldModel(_tiny_joint_config())
    s = torch.randn(1, 2, 64, 16)
    c = torch.randn(1, 16)
    action_cond = torch.randn(1, 2, 7)

    with pytest.raises(ValueError, match="context_rgb"):
        model(s, c, action_cond=action_cond, pixel=True)


def test_joint_model_context_renderer_outputs_rgb_shape_and_grad():
    from wm3d_v3.models.joint_model import JointWorldModel

    model = JointWorldModel(_tiny_joint_config())
    s = torch.randn(1, 2, 64, 16, requires_grad=True)
    c = torch.randn(1, 16)
    action_cond = torch.randn(1, 2, 7)
    context_rgb = torch.rand(1, 3, 256, 256)

    out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True)

    assert out["rgb"].shape == (1, 2, 3, 256, 256)
    assert out["motion_hint"].shape == (1, 2, 1, 256, 256)
    assert out["motion_logit"].shape == (1, 2, 1, 256, 256)
    assert out["motion_hint"].amin() >= 0
    assert out["motion_hint"].amax() <= 1
    loss = out["rgb"].mean() + out["motion_hint"].mean() + out["pred_tokens"].mean()
    loss.backward()
    assert s.grad is not None
    assert s.grad.abs().sum() > 0


def test_joint_model_can_disable_unsupervised_geometry_extras():
    from wm3d_v3.models.joint_model import JointWorldModel

    cfg = _tiny_joint_config()
    cfg.enable_geom_extra = False
    model = JointWorldModel(cfg)
    s = torch.randn(1, 2, 64, 16)
    c = torch.randn(1, 16)
    action_cond = torch.randn(1, 2, 7)
    context_rgb = torch.rand(1, 3, 256, 256)

    out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True)

    assert "depth" in out
    assert "point" not in out
    assert "pose_geom" not in out
    assert not any("point_head" in name or "pose_head" in name for name, _ in model.named_parameters())


def test_rgb_loss_supervises_motion_hint():
    from wm3d_v3.losses import LossWeights, compute_losses

    bsz, horizon, height, width = 1, 2, 16, 16
    motion_logit = torch.zeros(bsz, horizon, 1, height, width, requires_grad=True)
    rgb_pred = torch.zeros(bsz, horizon, 3, height, width, requires_grad=True)
    rgb_tgt = torch.zeros_like(rgb_pred)
    rgb_tgt[:, :, :, 4:12, 4:12] = 1.0
    rgb_ref = torch.zeros(bsz, 3, height, width)
    out = {
        "pred_tokens": torch.zeros(bsz, horizon, 64, 16),
        "depth": torch.ones(bsz, horizon, 4, 4),
        "pose": torch.zeros(bsz, horizon, 6),
        "gripper_logit": torch.zeros(bsz, horizon),
        "z_a": torch.zeros(bsz, horizon, 8),
        "rgb": rgb_pred,
        "motion_logit": motion_logit,
    }
    tgt = {
        "s_tgt": torch.zeros(bsz, horizon, 64, 16),
        "depth_tgt": torch.ones(bsz, horizon, 4, 4),
        "action_tgt": torch.zeros(bsz, horizon, 7),
        "rgb_tgt_p": rgb_tgt,
        "rgb_ref_p": rgb_ref,
    }
    weights = LossWeights(
        rgb_l1=0.0,
        rgb_lpips=0.0,
        rgb_motion_l1=0.0,
        rgb_edge=0.0,
        rgb_motion_bce=1.0,
        rgb_motion_dice=0.5,
    )

    losses = compute_losses(out, tgt, weights, lpips_fn=None)

    assert losses["L_rgb_motion_bce"] > 0
    assert losses["L_rgb_motion_dice"] > 0
    losses["L_total"].backward()
    assert motion_logit.grad is not None
    assert motion_logit.grad.abs().sum() > 0


def test_motion_loss_does_not_require_rgb_prediction():
    from wm3d_v3.losses import LossWeights, compute_losses

    bsz, horizon, height, width = 1, 2, 16, 16
    motion_logit = torch.zeros(bsz, horizon, 1, height, width, requires_grad=True)
    rgb_tgt = torch.zeros(bsz, horizon, 3, height, width)
    rgb_tgt[:, :, :, 4:12, 4:12] = 1.0
    rgb_ref = torch.zeros(bsz, 3, height, width)
    out = {
        "pred_tokens": torch.zeros(bsz, horizon, 64, 16),
        "depth": torch.ones(bsz, horizon, 4, 4),
        "pose": torch.zeros(bsz, horizon, 6),
        "gripper_logit": torch.zeros(bsz, horizon),
        "z_a": torch.zeros(bsz, horizon, 8),
        "motion_logit": motion_logit,
    }
    tgt = {
        "s_tgt": torch.zeros(bsz, horizon, 64, 16),
        "depth_tgt": torch.ones(bsz, horizon, 4, 4),
        "action_tgt": torch.zeros(bsz, horizon, 7),
        "rgb_tgt_p": rgb_tgt,
        "rgb_ref_p": rgb_ref,
    }
    weights = LossWeights(
        rgb_l1=0.0,
        rgb_lpips=0.0,
        rgb_motion_l1=0.0,
        rgb_edge=0.0,
        rgb_motion_bce=1.0,
        rgb_motion_dice=0.5,
    )

    losses = compute_losses(out, tgt, weights, lpips_fn=None)

    assert losses["L_rgb_motion_bce"] > 0
    assert losses["L_rgb_motion_dice"] > 0
    losses["L_total"].backward()
    assert motion_logit.grad is not None
    assert motion_logit.grad.abs().sum() > 0
