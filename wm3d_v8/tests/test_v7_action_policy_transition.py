from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wm3d_v3.data.window_dataset import _action_history_before_index
from wm3d_v3.training.train import (
    _direct_policy_only_forward,
    _main_teacher_action_weight_for_batch,
    apply_direct_policy_oxe_overrides,
    build_model,
    compute_direct_policy_loss,
    validate_action_pretraining_preflight,
    validate_action_policy_resume_load,
    validate_future_value_resume_load,
)


def _separated_action_pretrain_config() -> dict:
    return {
        "model": {
            "enable_action_policy": True,
            "policy_enable_flow_head": True,
            "policy_flow_use_as_policy": False,
        },
        "train": {
            "direct_policy_only": True,
            "direct_policy_weight": 1.0,
            "direct_policy_head": "base",
            "policy_flow_weight": 0.5,
            "enforce_separate_direct_flow_heads": True,
            "direct_policy_context_source": "input",
            "trainable_prefixes": ["action_policy."],
        },
    }


def test_action_pretraining_preflight_accepts_separated_heads() -> None:
    assert validate_action_pretraining_preflight(_separated_action_pretrain_config())


def _joint_native_action_pretrain_config() -> dict:
    return {
        "model": {
            "enable_action_policy": True,
            "policy_context_source": "core_pred",
            "policy_core_action_cond": "none",
            "policy_action_history_len": 1,
            "policy_flow_use_as_policy": False,
            "policy_grip_owner": "absolute",
        },
        "data": {"policy_action_history_len": 1},
        "train": {
            "joint_native_action_pretraining": True,
            "direct_policy_only": False,
            "direct_policy_weight": 1.0,
            "direct_policy_head": "base",
            "policy_flow_weight": 0.0,
            "direct_policy_grip_partition_contract": True,
            "direct_policy_grip_owner": "absolute",
            "direct_policy_require_action_prev_grip": True,
            "direct_policy_grip_natural_bce_weight": 1.0,
            "native_future_no_teacher_weight": 0.1,
            "factual_action_conditioning": {"enabled": True, "start_step": 0},
            "trainable_prefixes": [],
        },
    }


def test_action_pretraining_preflight_accepts_joint_native_policy() -> None:
    assert validate_action_pretraining_preflight(
        _joint_native_action_pretrain_config()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "teacher_core",
        "frozen_core",
        "no_history",
        "flow_owner",
        "no_natural_calibration",
        "no_future_anchor",
    ],
)
def test_action_pretraining_preflight_rejects_noncausal_joint_policy(
    mutation: str,
) -> None:
    cfg = _joint_native_action_pretrain_config()
    if mutation == "teacher_core":
        cfg["model"]["policy_core_action_cond"] = "same"
    if mutation == "frozen_core":
        cfg["model"]["policy_context_source"] = "core_pred_detach"
    if mutation == "no_history":
        cfg["model"]["policy_action_history_len"] = 0
        cfg["data"]["policy_action_history_len"] = 0
    if mutation == "flow_owner":
        cfg["model"]["policy_flow_use_as_policy"] = True
    if mutation == "no_natural_calibration":
        cfg["train"]["direct_policy_grip_natural_bce_weight"] = 0.0
    if mutation == "no_future_anchor":
        cfg["train"]["native_future_no_teacher_weight"] = 0.0
    with pytest.raises(RuntimeError):
        validate_action_pretraining_preflight(cfg)


def _joint_delta_grip_config() -> dict:
    cfg = _joint_native_action_pretrain_config()
    cfg["model"].update(
        {
            "policy_grip_owner": "delta_composed",
            "policy_enable_grip_delta_head": True,
            "policy_grip_delta_use_composed_action_cond": True,
            "policy_grip_delta_soft_compose_action_cond": True,
            "policy_grip_delta_straight_through_action_cond": True,
        }
    )
    cfg["train"].update(
        {
            "direct_policy_grip_owner": "delta_composed",
            "direct_policy_grip_weight": 0.0,
            "direct_policy_first_grip_weight": 0.0,
            "direct_policy_grip_natural_bce_weight": 0.0,
            "direct_policy_first_grip_natural_bce_weight": 0.0,
            "direct_policy_grip_delta_ce_weight": 1.0,
            "direct_policy_grip_delta_natural_ce_weight": 0.25,
        }
    )
    return cfg


def test_action_pretraining_preflight_accepts_delta_composed_owner() -> None:
    assert validate_action_pretraining_preflight(_joint_delta_grip_config())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "policy_grip_delta_use_composed_action_cond", False),
        ("model", "policy_grip_delta_soft_compose_action_cond", False),
        ("model", "policy_grip_delta_straight_through_action_cond", False),
        ("train", "direct_policy_grip_delta_ce_weight", 0.0),
        ("train", "direct_policy_grip_delta_natural_ce_weight", 0.0),
        ("train", "direct_policy_grip_natural_bce_weight", 1.0),
    ],
)
def test_action_pretraining_preflight_rejects_invalid_delta_composed_owner(
    section: str, key: str, value,
) -> None:
    cfg = _joint_delta_grip_config()
    cfg[section][key] = value
    with pytest.raises(RuntimeError):
        validate_action_pretraining_preflight(cfg)


def test_natural_gripper_bce_preserves_calibrated_token_prior() -> None:
    batch, horizon = 2, 4
    action = torch.zeros(batch, horizon, 7)
    action[..., 6] = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    out = {
        "base_policy_pose_norm": torch.zeros(batch, horizon, 6),
        "base_policy_gripper_logit": torch.zeros(batch, horizon),
    }
    losses = compute_direct_policy_loss(
        out,
        action,
        torch.zeros(batch, horizon, 6),
        {
            "direct_policy_weight": 1.0,
            "direct_policy_head": "base",
            "direct_policy_pose_weight": 0.0,
            "direct_policy_first_pose_weight": 0.0,
            "direct_policy_delta_weight": 0.0,
            "direct_policy_grip_weight": 0.0,
            "direct_policy_first_grip_weight": 0.0,
            "direct_policy_grip_natural_bce_weight": 2.5,
        },
    )
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        out["base_policy_gripper_logit"], action[..., 6]
    )
    assert torch.allclose(losses["direct_policy_grip_natural_bce"], expected_bce)
    assert torch.allclose(losses["L_direct_policy"], 2.5 * expected_bce)


def test_delta_event_natural_ce_preserves_hold_up_down_prior() -> None:
    batch, horizon = 2, 4
    action = torch.zeros(batch, horizon, 7)
    action[..., 6] = torch.tensor(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    )
    delta_logits = torch.tensor(
        [
            [[2.0, -1.0, -1.0], [0.0, 2.0, -1.0], [2.0, -1.0, -1.0], [0.0, -1.0, 2.0]],
            [[0.0, 2.0, -1.0], [2.0, -1.0, -1.0], [0.0, -1.0, 2.0], [2.0, -1.0, -1.0]],
        ]
    )
    out = {
        "base_policy_pose_norm": torch.zeros(batch, horizon, 6),
        "base_policy_gripper_logit": torch.zeros(batch, horizon),
        "policy_grip_delta_logits": delta_logits,
    }
    losses = compute_direct_policy_loss(
        out,
        action,
        torch.zeros(batch, horizon, 6),
        {
            "direct_policy_weight": 1.0,
            "direct_policy_head": "base",
            "direct_policy_grip_partition_contract": True,
            "direct_policy_grip_owner": "delta_composed",
            "direct_policy_pose_weight": 0.0,
            "direct_policy_first_pose_weight": 0.0,
            "direct_policy_delta_weight": 0.0,
            "direct_policy_grip_weight": 0.0,
            "direct_policy_first_grip_weight": 0.0,
            "direct_policy_grip_delta_ce_weight": 0.0,
            "direct_policy_grip_delta_natural_ce_weight": 0.4,
        },
        action_prev_grip=torch.tensor([0.0, 0.0]),
    )
    event_target = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    expected = torch.nn.functional.cross_entropy(
        delta_logits.reshape(-1, 3), event_target.reshape(-1)
    )
    assert torch.allclose(losses["direct_policy_grip_delta_natural_ce"], expected)
    assert torch.allclose(losses["L_direct_policy"], 0.4 * expected)


def test_canonical_history_ends_before_offset_action_target() -> None:
    actions = np.arange(12 * 7, dtype=np.float32).reshape(12, 7)
    history = _action_history_before_index(
        actions,
        end_exclusive=7,
        hist_len=4,
        action_dim=7,
    )
    np.testing.assert_array_equal(history, actions[3:7])
    np.testing.assert_array_equal(history[-1], actions[6])


def test_joint_policy_is_teacher_action_free_and_shapes_native_core() -> None:
    torch.manual_seed(1707)
    cfg = {
        "model": {
            "state": {
                "T": 4,
                "P": 4,
                "D": 32,
                "hidden": 32,
                "n_layers": 2,
                "n_heads": 4,
                "k": 2,
                "action_cond_dim": 7,
            },
            "action": {
                "T": 4,
                "P": 4,
                "D": 32,
                "hidden": 32,
                "n_layers": 2,
                "n_heads": 4,
                "k": 2,
                "z_dim": 16,
                "action_cond_dim": 7,
            },
            "xattn_layers_state": [0, 1],
            "xattn_n_heads": 4,
            "enable_multiview_fuser": False,
            "enable_token_codec": False,
            "action_proj_hidden": 32,
            "action_proj_layers": 2,
            "geom_hidden": 16,
            "enable_geom_extra": False,
            "pixel_hidden": 16,
            "pixel_n_res": 1,
            "enable_pixel": False,
            "enable_context_pixel": False,
            "enable_action_policy": True,
            "policy_hidden": 32,
            "policy_layers": 2,
            "policy_heads": 4,
            "policy_chunk_layers": 1,
            "policy_horizon": 2,
            "policy_task_dim": 2048,
            "policy_max_context": 4,
            "policy_dropout": 0.0,
            "policy_patch_pool": "last_patches",
            "policy_max_spatial_tokens": 4,
            "policy_context_source": "core_pred",
            "policy_core_action_cond": "none",
            "policy_action_history_len": 2,
            "policy_action_history_dim": 7,
            "policy_action_history_as_token": True,
            "policy_enable_flow_head": False,
            "policy_flow_use_as_policy": False,
            "enable_bridging": False,
            "enable_world_prior": False,
        }
    }
    model = build_model(cfg).eval()
    state = torch.randn(2, 4, 4, 32)
    task = torch.randn(2, 2048)
    history = torch.randn(2, 2, 7, requires_grad=True)
    history.data[..., 6].sigmoid_()
    teacher_a = torch.randn(2, 2, 7, requires_grad=True)
    teacher_b = teacher_a.detach() + 3.0

    out_a = model(
        state,
        task,
        action_cond=teacher_a,
        action_history=history,
        pixel=False,
        skip_native_prediction_heads=True,
    )
    with torch.no_grad():
        out_b = model(
            state,
            task,
            action_cond=teacher_b,
            action_history=history.detach(),
            pixel=False,
            skip_native_prediction_heads=True,
        )
    assert torch.allclose(
        out_a["base_policy_pose_norm"], out_b["base_policy_pose_norm"], atol=1e-6
    )
    assert torch.allclose(
        out_a["base_policy_gripper_logit"],
        out_b["base_policy_gripper_logit"],
        atol=1e-6,
    )

    target = torch.zeros(2, 2, 7)
    target[..., :6] = torch.randn(2, 2, 6)
    target[..., 6] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    losses = compute_direct_policy_loss(
        out_a,
        target,
        target[..., :6],
        {
            "direct_policy_weight": 1.0,
            "direct_policy_head": "base",
            "direct_policy_grip_partition_contract": True,
            "direct_policy_grip_owner": "absolute",
            "direct_policy_grip_weight": 0.5,
            "direct_policy_first_grip_weight": 0.5,
            "direct_policy_grip_natural_bce_weight": 1.0,
            "direct_policy_first_grip_natural_bce_weight": 0.5,
        },
        action_prev_grip=history[..., -1, 6],
    )
    losses["L_direct_policy"].backward()

    def grad_sum(prefix: str) -> float:
        return sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        )

    assert grad_sum("action_policy.") > 0.0
    assert grad_sum("dual.state.") > 0.0
    assert history.grad is not None and float(history.grad.abs().sum()) > 0.0
    assert teacher_a.grad is None or float(teacher_a.grad.abs().sum()) == 0.0


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("train", "direct_policy_head", "policy", "direct_policy_head=base"),
        ("model", "policy_flow_use_as_policy", True, "policy_flow_use_as_policy=false"),
        ("train", "direct_policy_only", False, "direct_policy_only=true"),
        ("train", "direct_policy_core_action_cond", "teacher", "must be action-free"),
    ],
)
def test_action_pretraining_preflight_rejects_ambiguous_contract(
    section: str, key: str, value, message: str
) -> None:
    cfg = _separated_action_pretrain_config()
    if key == "direct_policy_core_action_cond":
        cfg["train"]["direct_policy_context_source"] = "core_pred"
    cfg[section][key] = value
    with pytest.raises(RuntimeError, match=message):
        validate_action_pretraining_preflight(cfg)


def test_direct_policy_oxe_overrides_are_loader_only() -> None:
    source = {
        "manifest": "/immutable/source.jsonl",
        "canonical_action_enabled": True,
        "k": 8,
        "load_state_tgt": True,
    }
    resolved = apply_direct_policy_oxe_overrides(
        source,
        {
            "direct_policy_oxe_overrides": {
                "k": 32,
                "load_state_tgt": False,
                "load_geom": False,
                "window_geom_shard_index": "/cache/local/index.tsv",
                "window_geom_shard_root": "/cache/local/shards",
            }
        },
    )
    assert source["k"] == 8
    assert resolved["k"] == 32
    assert resolved["load_state_tgt"] is False
    assert resolved["load_geom"] is False
    assert resolved["window_geom_shard_index"] == "/cache/local/index.tsv"
    assert resolved["window_geom_shard_root"] == "/cache/local/shards"
    assert resolved["manifest"] == source["manifest"]
    assert resolved["canonical_action_enabled"] is True


def test_direct_policy_oxe_overrides_reject_identity_mutation() -> None:
    with pytest.raises(ValueError, match="non-loader keys"):
        apply_direct_policy_oxe_overrides(
            {"manifest": "/immutable/source.jsonl"},
            {"direct_policy_oxe_overrides": {"manifest": "/wrong.jsonl"}},
        )


def test_direct_policy_multiview_adapts_forward_keyword_to_fuser_contract() -> None:
    class Policy(torch.nn.Module):
        def forward(self, tokens, *, task_emb, **kwargs):
            return {"tokens": tokens, "task_emb": task_emb, "kwargs": kwargs}

    class Model:
        def __init__(self) -> None:
            self.action_policy = Policy()

        def fuse_views(
            self,
            anchor_tokens,
            wrist_tokens=None,
            *,
            view_mask=None,
            anchor_camera_pose=None,
            wrist_camera_pose=None,
        ):
            assert wrist_tokens is not None
            assert view_mask is not None
            return anchor_tokens + wrist_tokens

    anchor = torch.ones(2, 3, 4, 5)
    wrist = torch.full_like(anchor, 2.0)
    task = torch.randn(2, 7)
    output = _direct_policy_only_forward(
        Model(),
        anchor,
        task,
        action_cond=None,
        context_rgb=None,
        policy_kwargs={},
        train_cfg={"direct_policy_context_source": "input"},
        multiview_kwargs={
            "wrist_s": wrist,
            "view_mask": torch.ones(2, 3, 2, dtype=torch.bool),
        },
    )
    assert torch.equal(output["tokens"], anchor + wrist)
    assert torch.equal(output["task_emb"], task)


def test_direct_policy_telemetry_does_not_claim_wm_teacher_action_loss() -> None:
    factual = SimpleNamespace(action=1.25)
    representation = SimpleNamespace(action=0.0)
    assert _main_teacher_action_weight_for_batch(
        direct_policy_only=True,
        representation_only_batch=False,
        factual_weights=factual,
        representation_weights=representation,
    ) == 0.0
    assert _main_teacher_action_weight_for_batch(
        direct_policy_only=False,
        representation_only_batch=False,
        factual_weights=factual,
        representation_weights=representation,
    ) == 1.25


def test_stage0_to_policy_accepts_only_new_policy_tensors() -> None:
    validate_action_policy_resume_load(
        SimpleNamespace(
            missing_keys=[
                "action_policy.query",
                "action_policy.flow_head.out.weight",
            ],
            unexpected_keys=[],
            skipped_keys=[],
            expanded_keys=[],
        )
    )


def test_stage0_to_value_policy_accepts_only_the_two_new_heads() -> None:
    validate_future_value_resume_load(
        SimpleNamespace(
            missing_keys=[
                "future_value_head.backbone.pos",
                "action_policy.query",
                "action_policy.flow_head.out.weight",
            ],
            unexpected_keys=[],
            skipped_keys=[],
            expanded_keys=[],
        ),
        allowed_missing_prefixes=("future_value_head.", "action_policy."),
    )


def test_stage0_to_value_policy_rejects_world_tensor_mismatch() -> None:
    with pytest.raises(RuntimeError, match="strict Stage 0-compatible"):
        validate_future_value_resume_load(
            SimpleNamespace(
                missing_keys=[
                    "future_value_head.backbone.pos",
                    "dual.state.out_proj.weight",
                ],
                unexpected_keys=[],
                skipped_keys=[],
                expanded_keys=[],
            ),
            allowed_missing_prefixes=("future_value_head.", "action_policy."),
        )


@pytest.mark.parametrize(
    "load_result",
    [
        SimpleNamespace(
            missing_keys=["dual.state.out_proj.weight"],
            unexpected_keys=[],
            skipped_keys=[],
            expanded_keys=[],
        ),
        SimpleNamespace(
            missing_keys=["action_policy.query"],
            unexpected_keys=["old_policy.query"],
            skipped_keys=[],
            expanded_keys=[],
        ),
        SimpleNamespace(
            missing_keys=[],
            unexpected_keys=[],
            skipped_keys=[],
            expanded_keys=[],
        ),
    ],
)
def test_stage0_to_policy_rejects_non_policy_mismatch(load_result) -> None:
    with pytest.raises(RuntimeError, match="strict Stage 0-compatible"):
        validate_action_policy_resume_load(load_result)
