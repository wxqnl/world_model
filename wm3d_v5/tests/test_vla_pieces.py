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


def test_aux_idm_forward_shapes():
    from wm3d_v3.models.aux_idm import AuxIDM, AuxIDMConfig
    cfg = AuxIDMConfig(token_dim=2048, n_tokens=64, hidden=256, n_layers=2, k=8)
    m = AuxIDM(cfg)
    s_last = torch.randn(2, 64, 2048)
    s_fut_last = torch.randn(2, 64, 2048)
    out = m(s_last, s_fut_last)
    assert out["aux_pose_norm"].shape == (2, 8, 6)
    assert out["aux_grip"].shape == (2, 8)


def test_aux_idm_gradient_flows_to_both_inputs():
    from wm3d_v3.models.aux_idm import AuxIDM, AuxIDMConfig
    cfg = AuxIDMConfig(token_dim=32, n_tokens=8, hidden=32, n_layers=1, k=4)
    m = AuxIDM(cfg)
    s_last = torch.randn(1, 8, 32, requires_grad=True)
    s_fut_last = torch.randn(1, 8, 32, requires_grad=True)
    out = m(s_last, s_fut_last)
    out["aux_pose_norm"].sum().backward()
    assert s_last.grad is not None and s_last.grad.abs().sum() > 0
    assert s_fut_last.grad is not None and s_fut_last.grad.abs().sum() > 0


def test_joint_model_with_aux_idm_outputs_all_keys():
    from wm3d_v3.models.action_stream import ActionConfig
    from wm3d_v3.models.state_stream import StateConfig
    from wm3d_v3.models.dual_stream import DualConfig
    from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
    # P=64 so GeomDecoder's hardcoded 256->224 crop works (5 upsamples of grid=8)
    sc = StateConfig(T=4, P=64, D=64, hidden=64, n_layers=2, n_heads=4, k=2,
                     cond_dim=64)
    ac = ActionConfig(T=4, P=64, D=64, hidden=64, n_layers=2, n_heads=4, k=2,
                      z_dim=16, cond_dim=64)
    jc = JointConfig(
        dual=DualConfig(state=sc, action=ac,
                        xattn_layers_state=(1,), xattn_n_heads=4),
        action_proj_hidden=64, action_proj_layers=2,
        geom_hidden=32, pixel_hidden=32, pixel_n_res=1,
        enable_pixel=False, enable_bridging=False,
        enable_aux_idm=True, aux_idm_hidden=64, aux_idm_layers=1,
    )
    model = JointWorldModel(jc)
    s = torch.randn(1, 4, 64, 64)
    c = torch.randn(1, 64)
    out = model(s, c, pixel=False, bridging=False, aux_idm=True)
    for key in ("pose_norm", "pose", "gripper_logit",
                "aux_pose_norm", "aux_grip", "z_a", "pred_tokens"):
        assert key in out, f"missing {key}"
    assert out["aux_pose_norm"].shape == (1, 2, 6)


def test_compute_losses_vla_shape_and_keys():
    from wm3d_v3.losses import compute_losses_vla, VLALossWeights
    B, k = 2, 8
    out = {
        "pose_norm":     torch.randn(B, k, 6, requires_grad=True),
        "gripper_logit": torch.randn(B, k, requires_grad=True),
        "aux_pose_norm": torch.randn(B, k, 6, requires_grad=True),
        "aux_grip":      torch.randn(B, k, requires_grad=True),
    }
    tgt = {
        "action_tgt_norm": torch.randn(B, k, 6),
        "action_tgt":      torch.rand(B, k, 7),
    }
    w = VLALossWeights(action_pose=10.0, action_grip=2.0, aux_idm=5.0,
                       grip_pos_weight=1.5, grip_focal_gamma=2.0, grip_focal_alpha=0.25)
    losses = compute_losses_vla(out, tgt, w)
    for key in ("L_total", "L_pose", "L_grip", "L_aux_pose", "L_aux_grip"):
        assert key in losses, f"missing {key}"
    assert losses["L_total"].requires_grad


def test_compute_losses_vla_no_aux_idm_path():
    from wm3d_v3.losses import compute_losses_vla, VLALossWeights
    B, k = 1, 4
    out = {
        "pose_norm":     torch.randn(B, k, 6, requires_grad=True),
        "gripper_logit": torch.randn(B, k, requires_grad=True),
    }
    tgt = {
        "action_tgt_norm": torch.randn(B, k, 6),
        "action_tgt":      torch.rand(B, k, 7),
    }
    losses = compute_losses_vla(out, tgt, VLALossWeights())
    assert losses["L_aux_pose"].item() == 0.0
    assert losses["L_total"].requires_grad


def test_idm_stream_attends_over_input_and_future():
    from wm3d_v3.models.idm_stream import IDMStream, IDMStreamConfig
    cfg = IDMStreamConfig(T=4, k=2, P=8, D=32, hidden=32,
                          n_layers=2, n_heads=4, z_dim=8, cond_dim=32)
    m = IDMStream(cfg)
    s_in = torch.randn(2, 4, 8, 32)
    pred_tokens = torch.randn(2, 2, 8, 32)
    c = torch.randn(2, 32)
    out = m(s_in, pred_tokens, c)
    assert out["z_a"].shape == (2, 2, 8)


def test_idm_stream_gradient_flows_to_pred_tokens():
    from wm3d_v3.models.idm_stream import IDMStream, IDMStreamConfig
    cfg = IDMStreamConfig(T=4, k=2, P=8, D=32, hidden=32,
                          n_layers=1, n_heads=4, z_dim=8, cond_dim=32)
    m = IDMStream(cfg)
    s_in = torch.randn(2, 4, 8, 32, requires_grad=True)
    pred_tokens = torch.randn(2, 2, 8, 32, requires_grad=True)
    c = torch.randn(2, 32)
    out = m(s_in, pred_tokens, c)
    out["z_a"].sum().backward()
    assert pred_tokens.grad is not None and pred_tokens.grad.abs().sum() > 0
    assert s_in.grad is not None and s_in.grad.abs().sum() > 0


def test_depth_encoder_shapes():
    from wm3d_v3.models.depth_encoder import DepthEncoder, DepthEncoderConfig
    cfg = DepthEncoderConfig(in_size=224, out_grid=8, hidden_d=128, base_ch=8)
    m = DepthEncoder(cfg)
    depth = torch.rand(2, 4, 224, 224) * 2 + 0.5  # [0.5, 2.5] meters
    out = m(depth)
    assert out.shape == (2, 4, 64, 128)


def test_idm_stream_with_extra_tokens():
    from wm3d_v3.models.idm_stream import IDMStream, IDMStreamConfig
    cfg = IDMStreamConfig(T=4, k=2, P=8, D=32, hidden=32,
                          n_layers=1, n_heads=4, z_dim=8, cond_dim=32,
                          extra_token_dim=16)
    m = IDMStream(cfg)
    s_in = torch.randn(2, 4, 8, 32)
    pred = torch.randn(2, 2, 8, 32)
    c = torch.randn(2, 32)
    e_in = torch.randn(2, 4, 8, 16)
    e_pred = torch.randn(2, 2, 8, 16)
    out = m(s_in, pred, c, extra_in=e_in, extra_pred=e_pred)
    assert out["z_a"].shape == (2, 2, 8)


def test_joint_model_c_forward_and_grad():
    from wm3d_v3.models.depth_encoder import DepthEncoderConfig
    from wm3d_v3.models.idm_stream import IDMStreamConfig
    from wm3d_v3.models.joint_model_c import JointCConfig, JointWorldModelC
    from wm3d_v3.models.state_stream import StateConfig
    sc = StateConfig(T=4, P=64, D=64, hidden=64, n_layers=2, n_heads=4, k=2, cond_dim=64)
    ic = IDMStreamConfig(T=4, k=2, P=64, D=64, hidden=64,
                          n_layers=1, n_heads=4, z_dim=8, cond_dim=64,
                          extra_token_dim=32)
    dc = DepthEncoderConfig(in_size=224, out_grid=8, hidden_d=32, base_ch=4)
    cfg = JointCConfig(state=sc, idm=ic, depth_enc=dc,
                       action_proj_hidden=64, action_proj_layers=2,
                       geom_hidden=32, pixel_hidden=32, pixel_n_res=1,
                       enable_pixel=False)
    m = JointWorldModelC(cfg)
    s = torch.randn(1, 4, 64, 64, requires_grad=True)
    c = torch.randn(1, 64)
    depth_in = torch.rand(1, 4, 224, 224) + 0.5
    out = m(s, c, depth_in=depth_in, pixel=False)
    for k in ("pose_norm", "pose", "gripper_logit", "depth", "pred_tokens", "z_a"):
        assert k in out, f"missing {k}"
    assert out["pose_norm"].shape == (1, 2, 6)
    # gradient flow check: pose depends on s (via state -> pred -> idm) AND on depth (via depth_enc -> idm)
    loss = out["pose_norm"].sum() + out["depth"].sum() * 0.001
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
