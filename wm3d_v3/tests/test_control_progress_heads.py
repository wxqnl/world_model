from __future__ import annotations

import torch


def test_control_head_default_256_shapes_and_grad_p64():
    from wm3d_v3.models.control_head import ControlHead, ControlHeadConfig

    torch.manual_seed(0)
    cfg = ControlHeadConfig(token_dim=32, hidden=8, task_dim=64)
    model = ControlHead(cfg)
    pred_tokens = torch.randn(1, 2, 64, 32, requires_grad=True)
    depth = torch.rand(1, 2, 12, 10)
    context_rgb = torch.rand(1, 3, 128, 160)
    action_cond = torch.randn(1, 2, 7)
    task_emb = torch.randn(1, 64)

    out = model(pred_tokens, depth, context_rgb, action_cond, task_emb)

    assert out["motion_logit"].shape == (1, 2, 1, 256, 256)
    assert out["motion_hint"].shape == (1, 2, 1, 256, 256)
    assert out["contact_logit"].shape == (1, 2, 1, 256, 256)
    assert out["contact_hint"].shape == (1, 2, 1, 256, 256)
    assert out["control_confidence"].shape == (1, 2)
    assert out["motion_hint"].amin() >= 0
    assert out["motion_hint"].amax() <= 1
    assert out["contact_hint"].amin() >= 0
    assert out["contact_hint"].amax() <= 1

    loss = (
        out["motion_logit"].mean()
        + out["contact_logit"].mean()
        + out["control_confidence"].mean()
    )
    loss.backward()
    assert pred_tokens.grad is not None
    assert pred_tokens.grad.abs().sum() > 0


def test_control_head_supports_p256_without_optional_context():
    from wm3d_v3.models.control_head import ControlHead, ControlHeadConfig

    torch.manual_seed(1)
    cfg = ControlHeadConfig(token_dim=24, hidden=8, output_size=64, task_dim=16)
    model = ControlHead(cfg)
    pred_tokens = torch.randn(2, 3, 256, 24, requires_grad=True)
    depth = torch.rand(2, 3, 8, 8)

    out = model(pred_tokens, depth)

    assert out["motion_logit"].shape == (2, 3, 1, 64, 64)
    assert out["contact_logit"].shape == (2, 3, 1, 64, 64)
    assert out["control_confidence"].shape == (2, 3)
    (out["motion_hint"].mean() + out["contact_hint"].mean()).backward()
    assert pred_tokens.grad is not None
    assert pred_tokens.grad.abs().sum() > 0


def test_progress_head_shapes_and_grad_p64_p256():
    from wm3d_v3.models.progress_head import ProgressHead, ProgressHeadConfig

    torch.manual_seed(2)
    cfg = ProgressHeadConfig(token_dim=32, hidden=16, n_layers=1, n_heads=4, task_dim=64)
    model = ProgressHead(cfg)

    for patches in (64, 256):
        future_tokens = torch.randn(2, 4, patches, 32, requires_grad=True)
        action_cond = torch.randn(2, 4, 7)
        task_emb = torch.randn(2, 64)

        out = model(future_tokens, action_cond=action_cond, task_emb=task_emb)

        assert out["progress"].shape == (2, 4)
        assert out["terminal_success_logit"].shape == (2,)
        assert out["plausibility_logit"].shape == (2,)
        loss = (
            out["progress"].mean()
            + out["terminal_success_logit"].mean()
            + out["plausibility_logit"].mean()
        )
        loss.backward()
        assert future_tokens.grad is not None
        assert future_tokens.grad.abs().sum() > 0


def test_joint_world_model_control_and_progress_outputs():
    from wm3d_v3.models.action_stream import ActionConfig
    from wm3d_v3.models.dual_stream import DualConfig
    from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
    from wm3d_v3.models.state_stream import StateConfig

    torch.manual_seed(3)
    sc = StateConfig(T=2, P=64, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                     cond_dim=16, action_cond_dim=7)
    ac = ActionConfig(T=2, P=64, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                      z_dim=8, cond_dim=16, action_cond_dim=7)
    cfg = JointConfig(
        dual=DualConfig(state=sc, action=ac, xattn_layers_state=(), xattn_n_heads=4),
        action_proj_hidden=32,
        action_proj_layers=2,
        geom_hidden=16,
        enable_geom_extra=False,
        enable_pixel=False,
        enable_bridging=False,
        enable_control_head=True,
        control_hidden=8,
        control_output_size=32,
        control_task_dim=16,
        enable_progress_head=True,
        progress_hidden=16,
        progress_layers=1,
        progress_heads=4,
        progress_task_dim=16,
    )
    model = JointWorldModel(cfg)
    s = torch.randn(1, 2, 64, 16)
    c = torch.randn(1, 16)
    action_cond = torch.randn(1, 2, 7)
    context_rgb = torch.rand(1, 3, 64, 64)

    out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=False)

    assert out["motion_hint"].shape == (1, 2, 1, 32, 32)
    assert out["contact_hint"].shape == (1, 2, 1, 32, 32)
    assert out["control_confidence"].shape == (1, 2)
    assert out["progress"].shape == (1, 2)
    assert out["terminal_success_logit"].shape == (1,)
    assert out["plausibility_logit"].shape == (1,)
