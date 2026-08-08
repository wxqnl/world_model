import pytest
import torch

from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


def _loss_inputs(*, grip_target: torch.Tensor | None = None):
    pose_norm = torch.ones(1, 2, 6, requires_grad=True)
    physical_scale = 0.02
    if grip_target is None:
        grip_target = torch.zeros(1, 2)
    action_tgt = torch.cat(
        (torch.zeros(1, 2, 6), grip_target[..., None]),
        dim=-1,
    )
    out = {
        "pred_tokens": torch.zeros(1, 2, 1, 1),
        "pose_norm": pose_norm,
        "pose": pose_norm * physical_scale,
        "gripper_logit": torch.zeros(1, 2, requires_grad=True),
        "z_a": torch.zeros(1, 2, 1),
    }
    target = {
        "s_tgt": torch.zeros(1, 2, 1, 1),
        "action_tgt": action_tgt,
        "action_tgt_norm": torch.zeros(1, 2, 6),
    }
    return out, target


def _weights(**overrides):
    values = {
        "geom_depth": 0.0,
        "geom_point": 0.0,
        "geom_pose": 0.0,
        "action": 1.0,
        "grip": 0.0,
        "idm_reg": 0.0,
    }
    values.update(overrides)
    return LossWeights(**values)


def test_normalized_action_pose_loss_is_not_suppressed_by_physical_scale():
    out, target = _loss_inputs()
    losses = compute_losses(
        out,
        target,
        _weights(action_pose_space="normalized"),
    )

    assert losses["L_pose_action"].item() == pytest.approx(0.5)
    assert losses["L_pose_action_normalized"].item() == pytest.approx(0.5)
    assert losses["L_pose_action_physical"].item() == pytest.approx(0.0004)
    losses["L_total"].backward()
    assert out["pose_norm"].grad is not None
    assert out["pose_norm"].grad.abs().sum().item() > 0


def test_physical_action_pose_space_remains_legacy_default():
    out, target = _loss_inputs()
    losses = compute_losses(out, target, _weights())

    assert losses["L_pose_action"].item() == pytest.approx(0.0004)
    assert losses["L_pose_action_normalized"].item() == pytest.approx(0.0)


def test_normalized_action_pose_space_requires_normalized_contract():
    out, target = _loss_inputs()
    del target["action_tgt_norm"]

    with pytest.raises(ValueError, match="action_tgt_norm"):
        compute_losses(
            out,
            target,
            _weights(action_pose_space="normalized"),
        )


def test_action_grip_positive_weight_balances_positive_targets():
    out, target = _loss_inputs(grip_target=torch.tensor([[1.0, 0.0]]))
    base = compute_losses(out, target, _weights(grip=1.0))["L_grip"]
    balanced = compute_losses(
        out,
        target,
        _weights(grip=1.0, action_grip_positive_weight=2.0),
    )["L_grip"]

    assert base.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert balanced.item() == pytest.approx(1.5 * base.item())


def _tiny_joint_model() -> JointWorldModel:
    state = StateConfig(
        T=2,
        P=4,
        D=16,
        hidden=16,
        n_layers=1,
        n_heads=4,
        k=2,
        cond_dim=8,
        action_cond_dim=7,
    )
    action = ActionConfig(
        T=2,
        P=4,
        D=16,
        hidden=16,
        n_layers=1,
        n_heads=4,
        k=2,
        z_dim=8,
        cond_dim=8,
        action_cond_dim=7,
    )
    return JointWorldModel(
        JointConfig(
            dual=DualConfig(
                state=state,
                action=action,
                xattn_layers_state=(),
                xattn_n_heads=4,
            ),
            action_proj_hidden=16,
            action_proj_layers=2,
            geom_hidden=32,
            enable_geom_extra=False,
            enable_pixel=False,
            enable_bridging=False,
        )
    )


def test_internal_no_teacher_action_pass_preserves_conditioned_world_rollout():
    torch.manual_seed(7)
    model = _tiny_joint_model().eval()
    state = torch.randn(1, 2, 4, 16)
    task = torch.randn(1, 8)
    action = torch.randn(1, 2, 7)

    conditioned = model(state, task, action_cond=action, pixel=False)
    combined = model(
        state,
        task,
        action_cond=action,
        pixel=False,
        native_action_no_teacher=True,
    )
    no_teacher = model(state, task, action_cond=None, pixel=False)

    assert torch.equal(combined["pred_tokens"], conditioned["pred_tokens"])
    assert torch.equal(
        combined["native_action_no_teacher_pose_norm"],
        no_teacher["pose_norm"],
    )
    assert torch.equal(
        combined["native_action_no_teacher_gripper_logit"],
        no_teacher["gripper_logit"],
    )
