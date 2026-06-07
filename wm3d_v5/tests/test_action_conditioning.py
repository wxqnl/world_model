from __future__ import annotations

import torch


def test_make_action_condition_uses_normalized_pose_and_binary_grip():
    from wm3d_v3.data.action_condition import make_action_condition

    actions = torch.tensor([[
        [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.20],
        [40.0, 50.0, 60.0, 4.0, 5.0, 6.0, 0.90],
    ]])
    norm = torch.tensor([[
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    ]])

    cond = make_action_condition(actions, norm)

    assert cond.shape == (1, 2, 7)
    assert torch.allclose(cond[..., :6], norm)
    assert torch.equal(cond[..., 6], torch.tensor([[0.0, 1.0]]))


def test_make_action_condition_falls_back_to_raw_pose_without_stats():
    from wm3d_v3.data.action_condition import make_action_condition

    actions = torch.tensor([[[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 0.51]]])
    cond = make_action_condition(actions)

    assert torch.allclose(cond[..., :6], actions[..., :6])
    assert torch.equal(cond[..., 6], torch.tensor([[1.0]]))


def test_state_stream_future_tokens_depend_on_action_condition():
    from wm3d_v3.models.state_stream import StateConfig, StateStream

    torch.manual_seed(0)
    cfg = StateConfig(
        T=2, P=4, D=8, hidden=16, n_layers=1, n_heads=4, k=2,
        cond_dim=8, action_cond_dim=7,
    )
    model = StateStream(cfg).eval()
    s = torch.randn(1, 2, 4, 8)
    c = torch.randn(1, 8)
    a0 = torch.zeros(1, 2, 7)
    a1 = a0.clone()
    a1[:, :, 0] = 3.0

    with torch.no_grad():
        _, pred0 = model(s, c, action_cond=a0)
        _, pred1 = model(s, c, action_cond=a1)

    assert pred0.shape == (1, 2, 4, 8)
    assert not torch.allclose(pred0, pred1)


def test_joint_world_model_accepts_action_condition_and_changes_tokens():
    from wm3d_v3.models.action_stream import ActionConfig
    from wm3d_v3.models.dual_stream import DualConfig
    from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
    from wm3d_v3.models.state_stream import StateConfig

    torch.manual_seed(1)
    sc = StateConfig(T=2, P=64, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                     cond_dim=16, action_cond_dim=7)
    ac = ActionConfig(T=2, P=64, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                      z_dim=8, cond_dim=16, action_cond_dim=7)
    jc = JointConfig(
        dual=DualConfig(state=sc, action=ac, xattn_layers_state=(), xattn_n_heads=4),
        action_proj_hidden=32, action_proj_layers=2,
        geom_hidden=16, pixel_hidden=16, pixel_n_res=1,
        enable_pixel=False, enable_bridging=False,
    )
