from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.wm3d_v8_preflight_common import _Checks
from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    _validate_contract_and_objective,
    validate_preflight,
)
from wm3d_v3.data.v8_action_contract import (
    V8_DUAL_RATE_ACTION_SCHEMA,
    V8_POLICY_HISTORY_SCHEMA,
)
from wm3d_v3.training.train import (
    _window_config,
    apply_direct_policy_oxe_overrides,
    batch_to_device,
    build_model,
    build_v8_action_policy_contract,
    compute_v8_unified_action_policy_loss,
    load_train_config,
    validate_action_pretraining_preflight,
)


def _targets(batch: int = 2) -> dict[str, torch.Tensor]:
    fine = torch.zeros(batch, 8, 7)
    fine[..., 6] = 1.0
    coarse = torch.zeros(batch, 2, 7)
    coarse[..., 6] = 1.0
    fine_valid = torch.ones(batch, 8, dtype=torch.bool)
    fine_valid[-1] = False
    return {
        "policy_action_tgt": fine,
        "policy_action_tgt_norm": torch.zeros(batch, 8, 6),
        "policy_action_valid_mask": fine_valid,
        "policy_action_coarse_tgt": coarse,
        "policy_action_coarse_tgt_norm": torch.zeros(batch, 2, 6),
        "policy_action_coarse_valid_mask": torch.ones(batch, 2, dtype=torch.bool),
        "policy_action_pose_mean": torch.zeros(batch, 6),
        "policy_action_pose_std": torch.ones(batch, 6),
        "policy_action_coarse_pose_mean": torch.zeros(batch, 6),
        "policy_action_coarse_pose_std": torch.ones(batch, 6),
    }


def test_unified_policy_loss_is_finite_and_backpropagates_through_composition() -> None:
    pose = torch.randn(2, 8, 6, requires_grad=True) * 0.01
    pose.retain_grad()
    grip = torch.zeros(2, 8, requires_grad=True)
    losses = compute_v8_unified_action_policy_loss(
        {
            "base_policy_pose_norm": pose,
            "base_policy_gripper_logit": grip,
        },
        _targets(),
        {
            "v8_policy_fine_pose_weight": 1.0,
            "v8_policy_fine_grip_weight": 0.3,
            "v8_policy_coarse_pose_weight": 1.0,
            "v8_policy_coarse_grip_weight": 0.3,
        },
    )
    assert torch.isfinite(losses["L_direct_policy"])
    losses["L_direct_policy"].backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()
    assert float(pose.grad.abs().sum()) > 0.0
    assert grip.grad is not None and float(grip.grad.abs().sum()) > 0.0
    assert losses["v8_policy_fine_label_fraction"] == pytest.approx(0.5)


def test_unified_policy_loss_reaches_policy_and_action_free_native_core() -> None:
    torch.manual_seed(1808)
    model = build_model(
        {
            "model": {
                "state": {
                    "T": 4,
                    "P": 4,
                    "D": 32,
                    "hidden": 32,
                    "n_layers": 2,
                    "n_heads": 4,
                    "k": 2,
                    "action_cond_dim": 36,
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
                    "action_cond_dim": 36,
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
                "policy_horizon": 8,
                "policy_task_dim": 2048,
                "policy_max_context": 4,
                "policy_dropout": 0.0,
                "policy_patch_pool": "last_patches",
                "policy_max_spatial_tokens": 4,
                "policy_context_source": "core_pred",
                "policy_core_action_cond": "none",
                "policy_action_history_len": 16,
                "policy_action_history_dim": 9,
                "policy_action_history_as_token": True,
                "policy_enable_flow_head": False,
                "policy_flow_use_as_policy": False,
                "enable_bridging": False,
                "enable_world_prior": False,
            }
        }
    )
    state = torch.randn(2, 4, 4, 32)
    task = torch.randn(2, 2048)
    teacher_action = torch.randn(2, 2, 36, requires_grad=True)
    history = torch.randn(2, 16, 9)
    history[..., 7] = 0.05
    history[..., 8] = 1.0
    out = model(
        state,
        task,
        action_cond=teacher_action,
        action_history=history,
        pixel=False,
        skip_native_prediction_heads=True,
    )
    losses = compute_v8_unified_action_policy_loss(out, _targets(), {})
    losses["L_direct_policy"].backward()

    def grad_sum(prefix: str) -> float:
        return sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        )

    assert grad_sum("action_policy.") > 0.0
    assert grad_sum("dual.state.") > 0.0
    assert teacher_action.grad is None or float(teacher_action.grad.abs().sum()) == 0.0


def test_coarse_only_sample_does_not_consume_fake_fine_target() -> None:
    out = {
        "base_policy_pose_norm": torch.zeros(2, 8, 6),
        "base_policy_gripper_logit": torch.zeros(2, 8),
    }
    first = _targets()
    second = {key: value.clone() for key, value in first.items()}
    second["policy_action_tgt"][-1, :, :6] = 1000.0
    second["policy_action_tgt_norm"][-1] = -1000.0
    loss_a = compute_v8_unified_action_policy_loss(out, first, {})["L_direct_policy"]
    loss_b = compute_v8_unified_action_policy_loss(out, second, {})["L_direct_policy"]
    assert torch.equal(loss_a, loss_b)


def test_batch_to_device_replaces_legacy_7d_condition_with_v8_36d() -> None:
    batch_size = 1
    batch = {
        "s_in": torch.zeros(batch_size, 16, 1, 2),
        "c": torch.zeros(batch_size, 2),
        "action_tgt": torch.zeros(batch_size, 8, 7),
        "action_tgt_norm": torch.zeros(batch_size, 8, 6),
        "v8_dynamics_action_cond": torch.zeros(batch_size, 8, 36),
        "v8_action_contract_version": [V8_DUAL_RATE_ACTION_SCHEMA],
        "v8_action_history_schema": [V8_POLICY_HISTORY_SCHEMA],
        "v8_action_stats_key": ["fine|coarse"],
    }
    _s, _c, action_cond, _rgb, tgt = batch_to_device(
        batch,
        torch.device("cpu"),
        8,
        direct_policy_only=True,
        action_grip_contract="close01",
    )
    assert action_cond.shape == (1, 8, 36)
    assert tgt["v8_action_contract_version"] == V8_DUAL_RATE_ACTION_SCHEMA


def test_v8_config_resolves_unique_owner_and_checkpoint_abi() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    assert validate_action_pretraining_preflight(config) is True
    contract = build_v8_action_policy_contract(config)
    assert contract is not None
    assert contract["policy_output_shape"] == [8, 7]
    assert contract["flow_owner"] is False
    assert contract["delta_event_owner"] is False
    assert contract["policy_context_source"] == "core_pred"
    assert contract["policy_core_action_cond"] == "none"
    assert "context_source" not in contract
    assert "core_action_condition" not in contract
    assert set(contract["normalizers"]["oxe"]) == {
        "oxe_bridge_action",
        "oxe_droid_action",
    }
    assert contract["normalizers"]["oxe"]["oxe_bridge_action"][
        "stats_sha256_by_source"
    ] == {
        "bridge": "0b96402a464d9708ba654c48057b5f49f90acdb297a277fbbd10a422ca33d711"
    }
    assert len(contract["contract_sha256"]) == 64


def test_v8_checkpoint_contract_hash_binds_oxe_normalizer_identity() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    original = build_v8_action_policy_contract(config)
    config["data"]["oxe_sources"][0]["canonical_action_stats_sha256_by_source"][
        "droid"
    ] = "0" * 64
    changed = build_v8_action_policy_contract(config)
    assert original is not None and changed is not None
    assert original["contract_sha256"] != changed["contract_sha256"]


def test_v8_oxe_window_config_carries_sealed_stats_sha_to_runtime() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    source = apply_direct_policy_oxe_overrides(
        config["data"]["oxe_sources"][0],
        config["data"],
    )
    window = _window_config(source, config["model"])
    assert window.v8_dual_rate_action_enabled is True
    assert window.canonical_action_stats_sha256_by_source == source[
        "canonical_action_stats_sha256_by_source"
    ]


def test_v8_config_rejects_reintroduced_flow_owner() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    config["model"]["policy_enable_flow_head"] = True
    with pytest.raises(RuntimeError, match="forbids the flow action head"):
        validate_action_pretraining_preflight(config)


def test_v8_config_rejects_reintroduced_trunk_action_owner() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    config["model"]["policy_action_add_trunk"] = True
    with pytest.raises(RuntimeError, match="duplicate native trunk action"):
        validate_action_pretraining_preflight(config)


def test_v8_script_preflight_accepts_unified_action_v2_contract() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    checks = _Checks("structure")
    sources = _validate_contract_and_objective(checks, config)
    assert checks.errors == []
    assert set(sources) == {"oxe_bridge_action", "oxe_droid_action"}


def test_v8_script_preflight_rejects_wrong_dual_rate_dynamics_width() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    config["model"]["state"]["action_cond_dim"] = 7
    checks = _Checks("structure")
    _validate_contract_and_objective(checks, config)
    assert any("model.state.action_cond_dim" in error for error in checks.errors)
    assert any("state action_cond_dim must be 36" in error for error in checks.errors)


def test_v8_static_preflight_reports_new_owner_and_zero_legacy_losses() -> None:
    config = load_train_config(
        Path("configs/wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml")
    )
    report = validate_preflight(
        config,
        mode="static",
        verify_training_assets=False,
        verify_local_resources=False,
    )
    objective = report["action_objective"]
    assert objective["v8_dual_rate_action_enabled"] is True
    assert objective["serving_owner"] == "base"
    assert objective["gripper_owner"] == "absolute"
    assert objective["policy_flow_weight"] == 0.0
    assert objective["native_action_no_teacher_weight"] == 0.0
