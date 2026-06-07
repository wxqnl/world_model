from __future__ import annotations

import json

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


def test_action_policy_progress_token_is_opt_in():
    from wm3d_v3.models.action_policy import ActionChunkPolicy, ActionChunkPolicyConfig

    base = ActionChunkPolicyConfig(
        token_dim=16,
        task_dim=16,
        hidden=32,
        n_layers=1,
        n_heads=4,
        chunk_layers=1,
        horizon=2,
        max_context=3,
        lowdim_dim=4,
        action_history_len=2,
        action_history_dim=7,
    )
    no_progress = ActionChunkPolicy(base)
    assert no_progress.pos_embed.shape[1] == base.max_context + 4

    cfg = ActionChunkPolicyConfig(**{**base.__dict__, "use_progress": True, "progress_dim": 1})
    model = ActionChunkPolicy(cfg)
    assert model.pos_embed.shape[1] == cfg.max_context + 5
    assert model.progress_proj is not None
    progress_emb = model.progress_proj(torch.tensor([[0.0], [1.0]]))
    assert not torch.allclose(progress_emb[0], progress_emb[1])

    summary_cfg = ActionChunkPolicyConfig(
        **{**base.__dict__, "use_progress": True, "progress_dim": 1, "progress_mode": "summary"}
    )
    summary_model = ActionChunkPolicy(summary_cfg)
    assert summary_model.pos_embed.shape[1] == summary_cfg.max_context + 4

    object_cfg = ActionChunkPolicyConfig(
        **{
            **base.__dict__,
            "object_state_dim": 5,
            "plan_state_dim": 8,
            "use_progress": True,
            "progress_dim": 1,
            "progress_mode": "summary",
        }
    )
    object_model = ActionChunkPolicy(object_cfg)
    assert object_model.pos_embed.shape[1] == object_cfg.max_context + 4

    out = model(
        torch.randn(2, 3, 64, 16),
        task_emb=torch.randn(2, 16),
        lowdim_state=torch.randn(2, 4),
        action_history=torch.randn(2, 2, 7),
        progress_state=torch.tensor([0.25, 0.75]),
    )
    assert out["policy_action_cond"].shape == (2, 2, 7)


def test_action_policy_object_state_shapes_and_grad():
    from wm3d_v3.models.action_policy import ActionChunkPolicy, ActionChunkPolicyConfig

    torch.manual_seed(4)
    cfg = ActionChunkPolicyConfig(
        token_dim=16,
        task_dim=16,
        hidden=32,
        n_layers=1,
        n_heads=4,
        chunk_layers=1,
        horizon=2,
        max_context=3,
        lowdim_dim=4,
        object_state_dim=6,
        plan_state_dim=8,
        action_history_len=2,
        action_history_dim=7,
        use_progress=True,
        progress_mode="summary",
    )
    model = ActionChunkPolicy(cfg)
    context = torch.randn(2, 3, 64, 16)
    task = torch.randn(2, 16)
    lowdim = torch.randn(2, 4)
    obj = torch.randn(2, 6, requires_grad=True)
    plan = torch.randn(2, 8, requires_grad=True)
    hist = torch.randn(2, 2, 7)
    progress = torch.tensor([[0.2], [0.7]])

    out = model(
        context,
        task_emb=task,
        lowdim_state=lowdim,
        object_state=obj,
        plan_state=plan,
        action_history=hist,
        progress_state=progress,
    )

    assert out["policy_action_cond"].shape == (2, 2, 7)
    out["policy_pose_norm"].sum().backward()
    object_final = model.object_state_proj[-1]
    assert object_final.weight.grad is not None
    assert object_final.weight.grad.abs().sum() > 0
    plan_final = model.plan_state_proj[-1]
    assert plan_final.weight.grad is not None
    assert plan_final.weight.grad.abs().sum() > 0


def test_action_policy_local_residual_is_zero_init_and_trainable():
    from wm3d_v3.models.action_policy import ActionChunkPolicy, ActionChunkPolicyConfig

    torch.manual_seed(5)
    base_kwargs = dict(
        token_dim=16,
        task_dim=16,
        hidden=32,
        n_layers=1,
        n_heads=4,
        chunk_layers=1,
        horizon=2,
        max_context=3,
        lowdim_dim=4,
        plan_state_dim=8,
        action_history_len=2,
        action_history_dim=7,
        use_progress=True,
        progress_mode="summary",
    )
    base = ActionChunkPolicy(ActionChunkPolicyConfig(**base_kwargs))
    local = ActionChunkPolicy(
        ActionChunkPolicyConfig(
            **base_kwargs,
            enable_local_residual=True,
            local_hidden=16,
            local_layers=1,
        )
    )
    local.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    local.eval()

    context = torch.randn(2, 3, 64, 16)
    task = torch.randn(2, 16)
    lowdim = torch.randn(2, 4)
    plan = torch.randn(2, 8)
    hist = torch.randn(2, 2, 7)
    progress = torch.tensor([[0.1], [0.8]])

    base_out = base(context, task_emb=task, lowdim_state=lowdim, plan_state=plan, action_history=hist, progress_state=progress)
    local_out = local(context, task_emb=task, lowdim_state=lowdim, plan_state=plan, action_history=hist, progress_state=progress)

    assert torch.allclose(local_out["policy_pose_norm"], base_out["policy_pose_norm"])
    assert torch.allclose(local_out["policy_gripper_logit"], base_out["policy_gripper_logit"])
    assert local_out["local_pose_residual"].shape == (2, 2, 6)
    assert local_out["local_gripper_logit_residual"].shape == (2, 2)

    local_out["policy_pose_norm"].sum().backward()
    final = local.local_residual_head[-1]
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum() > 0


def test_action_policy_waypoint_head_is_stage_routed_zero_init_and_trainable():
    from wm3d_v3.models.action_policy import ActionChunkPolicy, ActionChunkPolicyConfig

    torch.manual_seed(6)
    base_kwargs = dict(
        token_dim=16,
        task_dim=16,
        hidden=32,
        n_layers=1,
        n_heads=4,
        chunk_layers=1,
        horizon=2,
        max_context=3,
        lowdim_dim=4,
        plan_state_dim=17,
        action_history_len=2,
        action_history_dim=7,
        use_progress=True,
        progress_mode="summary",
    )
    base = ActionChunkPolicy(ActionChunkPolicyConfig(**base_kwargs))
    waypoint = ActionChunkPolicy(
        ActionChunkPolicyConfig(
            **base_kwargs,
            enable_waypoint_head=True,
            waypoint_hidden=16,
            waypoint_layers=1,
            waypoint_num_stages=4,
            waypoint_stage_dim=4,
        )
    )
    waypoint.load_state_dict(base.state_dict(), strict=False)
    base.eval()
    waypoint.eval()

    context = torch.randn(2, 3, 64, 16)
    task = torch.randn(2, 16)
    lowdim = torch.randn(2, 4)
    plan = torch.zeros(2, 17)
    plan[0, 0] = 1.0
    plan[1, 3] = 1.0
    hist = torch.randn(2, 2, 7)
    progress = torch.tensor([[0.1], [0.8]])

    base_out = base(context, task_emb=task, lowdim_state=lowdim, plan_state=plan, action_history=hist, progress_state=progress)
    waypoint_out = waypoint(context, task_emb=task, lowdim_state=lowdim, plan_state=plan, action_history=hist, progress_state=progress)

    assert torch.allclose(waypoint_out["policy_pose_norm"], base_out["policy_pose_norm"])
    assert torch.allclose(waypoint_out["policy_gripper_logit"], base_out["policy_gripper_logit"])
    assert waypoint_out["waypoint_pose"].shape == (2, 2, 6)
    assert waypoint_out["waypoint_gripper_logit"].shape == (2, 2)
    assert waypoint_out["waypoint_stage_weights"].shape == (2, 4)
    assert torch.allclose(waypoint_out["waypoint_stage_weights"], plan[:, :4])

    waypoint_out["policy_pose_norm"].sum().backward()
    final = waypoint.waypoint_head[-1]
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum() > 0

    stage3_only = ActionChunkPolicy(
        ActionChunkPolicyConfig(
            **base_kwargs,
            enable_waypoint_head=True,
            waypoint_hidden=16,
            waypoint_layers=1,
            waypoint_num_stages=4,
            waypoint_stage_dim=4,
            waypoint_active_stages=(3,),
        )
    ).eval()
    masked_out = stage3_only(
        context,
        task_emb=task,
        lowdim_state=lowdim,
        plan_state=plan,
        action_history=hist,
        progress_state=progress,
    )
    assert torch.allclose(masked_out["waypoint_stage_weights"][0], torch.zeros(4))
    assert torch.allclose(masked_out["waypoint_stage_weights"][1], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_plan_waypoint_selection_outputs_raw_action_chunk_without_video():
    from types import SimpleNamespace

    from wm3d_v3.policy.world_model_policy import select_action_chunk

    model = SimpleNamespace(cfg=SimpleNamespace(dual=SimpleNamespace(state=SimpleNamespace(k=3))))
    s = torch.zeros(2, 2, 64, 16)
    task = torch.zeros(2, 16)
    plan = torch.zeros(2, 17)
    plan[0, 0] = 1.0
    plan[0, 8:11] = torch.tensor([0.1, -0.2, 0.3])
    plan[1, 3] = 1.0
    plan[1, 11:14] = torch.tensor([0.2, 0.0, 0.0])
    plan[1, 14:17] = torch.tensor([0.05, 0.01, -0.02])

    out = select_action_chunk(model, s, task, plan_state=plan, selection_mode="plan_waypoint")

    assert out["selected_action_raw"].shape == (2, 3, 7)
    assert torch.all(out["selected_idx"] == -3)
    assert torch.allclose(out["selected_action_raw"][0, 0, :3], torch.tensor([0.4, -0.8, 0.9]))
    assert out["selected_action_raw"][0, 0, 6] == 0.0
    assert out["selected_action_raw"][1, 0, 6] == 1.0
    assert out["candidate_scores"].shape == (2, 1)


def test_stage3_place_overlay_only_modifies_terminal_stage():
    from wm3d_v3.policy.world_model_policy import _apply_stage3_place_overlay

    action = torch.zeros(3, 2, 7)
    action[..., 0] = 0.12
    action[..., 6] = 0.0
    plan = torch.zeros(3, 17)
    plan[0, 2] = 1.0
    plan[1, 3] = 1.0
    plan[1, 8:11] = torch.tensor([0.02, 0.01, 0.02])
    plan[1, 11:14] = torch.tensor([-0.08, 0.21, -0.02])
    plan[1, 14:17] = torch.tensor([-0.07, 0.20, -0.08])
    plan[2, 3] = 1.0
    plan[2, 8:11] = torch.tensor([0.01, 0.01, 0.01])
    plan[2, 11:14] = torch.tensor([0.01, 0.02, -0.20])
    plan[2, 14:17] = torch.tensor([0.01, 0.02, -0.22])

    out = _apply_stage3_place_overlay(action, plan)

    assert torch.allclose(out[0], action[0])
    assert out[1, 0, 6] == 1.0
    assert out[1, 0, 1] > 0.5
    assert out[2, 0, 6] == 0.0


def test_terminal_reference_loader_and_overlay(tmp_path):
    from wm3d_v3.policy.token_policy import load_terminal_reference
    from wm3d_v3.policy.world_model_policy import (
        _apply_terminal_linear_overlay,
        _apply_terminal_reference_overlay,
        _apply_trace_linear_overlay,
    )

    trace = {
        "results": [
            {
                "step_trace": [
                    {"plan_stage": 2, "plan_state": [0.0] * 17, "policy_action": [9.0] * 7},
                    {
                        "plan_stage": 3,
                        "plan_state": [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0],
                        "policy_action": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
                    },
                    {
                        "plan_stage": 3,
                        "plan_state": [0.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, -0.1, 0.0, 0.0, -0.2, 0.0, 0.0, -0.3, 0.0, 0.0],
                        "policy_action": [-0.4, -0.5, -0.6, 0.0, 0.0, 0.0, -1.0],
                    },
                ]
            }
        ]
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace))
    ref = load_terminal_reference(path)
    assert ref["features"].shape == (2, 9)
    assert ref["actions"].shape == (2, 7)
    assert ref["linear_weights"].shape == (10, 7)
    assert ref["trace_features"].shape == (3, 17)
    assert ref["trace_actions"].shape == (3, 7)
    assert ref["trace_linear_weights"].shape == (18, 7)
    assert ref["actions"][1, 6] == 0.0

    action = torch.zeros(2, 2, 7)
    plan = torch.zeros(2, 17)
    plan[0, 0] = 1.0
    plan[1, 3] = 1.0
    plan[1, 8:17] = ref["features"][0]
    out = _apply_terminal_reference_overlay(action, plan, ref)
    assert torch.allclose(out[0], action[0])
    assert torch.allclose(out[1, 0], ref["actions"][0])

    linear = _apply_terminal_linear_overlay(action, plan, ref)
    assert torch.allclose(linear[0], action[0])
    assert linear[1, 0, 0] > 0.0
    assert linear[1, 0, 6] == 1.0

    trace_linear = _apply_trace_linear_overlay(action, plan, ref)
    assert trace_linear.shape == action.shape
    assert not torch.allclose(trace_linear[0], action[0])
    assert trace_linear[1, 0, 6] == 1.0
