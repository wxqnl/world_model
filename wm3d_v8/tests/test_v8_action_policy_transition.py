from __future__ import annotations

import copy
import json

import pytest
import torch

import scripts.audit_wm3d_v8_stage0_libero_transition as transition_audit

from wm3d_v3.training.v8_action_policy_transition import (
    V8ActionPolicyTransitionError,
    action_contract_sha256,
    checkpoint_config_sha256,
    decode_v8_executable_action_chunk,
    load_v8_stage0_for_libero_strict,
    validate_v8_stage0_checkpoint_payload,
)


def _contract() -> dict:
    value = {
        "schema": "wm3d_v8_stage0_action_policy_contract_v2",
        "world_state_hz": 5,
        "policy_hz": 20,
        "policy_horizon": 8,
        "policy_chunk_seconds": 0.4,
        "policy_output_shape": [8, 7],
        "pose": {
            "frame": "base",
            "translation_unit": "meter",
            "translation_semantics": "per_controller_step_delta",
            "rotation": "axis_angle_rotvec_radian_delta",
            "normalization": "source_bound_affine_pose6",
        },
        "gripper": {
            "semantics": "absolute_close01",
            "threshold": 0.5,
            "environment_polarity_conversion": "execution_boundary_only",
        },
        "serving_owner": "action_policy.base_policy.[pose_norm,gripper_logit]",
        "serving_decoder": "decode_v8_executable_action_chunk_v1",
        "flow_owner": False,
        "delta_event_owner": False,
        "policy_context_source": "core_pred",
        "policy_core_action_cond": "none",
        "history": {
            "schema": "wm3d_v8_action_history_20hz_dt_valid_v1",
            "length": 16,
            "dim": 9,
        },
        "dynamics_condition": {
            "schema": "wm3d_v8_dual_rate_action_v1",
            "dim": 36,
            "substeps_per_world_interval": 4,
            "layout": "4x(action7)+valid4+dt_seconds4",
        },
        "normalizers": {
            "robocasa_fine_stats_sha256": "1" * 64,
            "robocasa_coarse_stats_sha256": "2" * 64,
            "oxe": {
                "oxe_bridge_action": {
                    "cache_manifest_sha256": "3" * 64,
                    "audit_gate_sha256": "4" * 64,
                    "evidence_sha256": "5" * 64,
                    "stats_sha256_by_source": {"bridge": "6" * 64},
                },
                "oxe_droid_action": {
                    "cache_manifest_sha256": "7" * 64,
                    "audit_gate_sha256": "8" * 64,
                    "evidence_sha256": "9" * 64,
                    "stats_sha256_by_source": {"droid": "a" * 64},
                },
            },
        },
        "stage0_native3d_owner": True,
    }
    value["contract_sha256"] = action_contract_sha256(value)
    return value


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.core = torch.nn.Linear(2, 2)
        self.action_policy = torch.nn.Linear(2, 7)


def _payload(model: torch.nn.Module) -> dict:
    cfg = {
        "contract": {
            "schema": "wm3d_v8_stage0_causal_dual_view_unified_action_formal_v2"
        },
        "train": {"stage_transition": None},
    }
    resolved_config_sha256 = checkpoint_config_sha256(cfg)
    cfg["train"]["resolved_config_sha256"] = resolved_config_sha256
    return {
        "model": copy.deepcopy(model.state_dict()),
        "step": 100,
        "resolved_config_sha256": resolved_config_sha256,
        "action_policy_contract": _contract(),
        "cfg": cfg,
    }


def test_strict_stage0_to_libero_loads_every_tensor() -> None:
    source = _Tiny()
    target = _Tiny()
    report = load_v8_stage0_for_libero_strict(
        target, _payload(source), expected_contract=_contract()
    )
    assert report["strict"] is True
    assert report["loaded_tensor_count"] == len(source.state_dict())
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_cli_audit_materializes_target_and_uses_strict_loader(monkeypatch) -> None:
    source = _Tiny()
    monkeypatch.setattr(transition_audit, "build_model", lambda _config: _Tiny())
    report = transition_audit._materialize_and_strict_load_target(
        {"model": {}},
        _payload(source),
        _contract(),
    )
    assert report["strict"] is True
    assert report["loaded_tensor_count"] == len(source.state_dict())

    incomplete = _payload(source)
    incomplete["model"].pop("core.bias")
    with pytest.raises(V8ActionPolicyTransitionError, match="model identity mismatch"):
        transition_audit._materialize_and_strict_load_target(
            {"model": {}},
            incomplete,
            _contract(),
        )


@pytest.mark.parametrize("mutation", ["contract", "missing", "extra", "shape"])
def test_strict_stage0_to_libero_fails_closed(mutation: str) -> None:
    source = _Tiny()
    payload = _payload(source)
    if mutation == "contract":
        payload["action_policy_contract"]["policy_hz"] = 5
    elif mutation == "missing":
        payload["model"].pop("action_policy.bias")
    elif mutation == "extra":
        payload["model"]["replacement_head.weight"] = torch.zeros(1)
    elif mutation == "shape":
        payload["model"]["action_policy.weight"] = torch.zeros(8, 2)
    with pytest.raises(V8ActionPolicyTransitionError):
        load_v8_stage0_for_libero_strict(
            _Tiny(), payload, expected_contract=_contract()
        )


def test_transition_rejects_non_stage0_lineage() -> None:
    payload = _payload(_Tiny())
    payload["cfg"]["train"]["stage_transition"] = {"mode": "stage1"}
    with pytest.raises(V8ActionPolicyTransitionError, match="fresh Stage0"):
        validate_v8_stage0_checkpoint_payload(payload)


def test_transition_rejects_nonpositive_checkpoint_step() -> None:
    payload = _payload(_Tiny())
    payload["step"] = 0
    with pytest.raises(V8ActionPolicyTransitionError, match="positive numbered"):
        validate_v8_stage0_checkpoint_payload(payload)


def test_transition_rejects_checkpoint_config_identity_mismatch() -> None:
    payload = _payload(_Tiny())
    payload["cfg"]["train"]["new_semantics"] = True
    with pytest.raises(V8ActionPolicyTransitionError, match="config identity mismatch"):
        validate_v8_stage0_checkpoint_payload(payload)


def test_transition_rejects_self_consistent_contract_without_normalizer_identity() -> None:
    payload = _payload(_Tiny())
    contract = payload["action_policy_contract"]
    contract["normalizers"].pop("robocasa_fine_stats_sha256")
    contract["contract_sha256"] = action_contract_sha256(contract)
    with pytest.raises(V8ActionPolicyTransitionError, match="robocasa_fine_stats_sha256"):
        validate_v8_stage0_checkpoint_payload(payload)


def test_serving_decoder_denormalizes_pose_and_converts_gripper_only_at_boundary() -> None:
    pose_norm = torch.ones(2, 8, 6)
    grip_logit = torch.tensor([[20.0] * 8, [-20.0] * 8])
    action = decode_v8_executable_action_chunk(
        {
            "base_policy_pose_norm": pose_norm,
            "base_policy_gripper_logit": grip_logit,
        },
        pose_mean=torch.arange(6, dtype=torch.float32),
        pose_std=torch.full((6,), 2.0),
        gripper_output="signed_open_positive",
        hard_gripper=True,
    )
    assert action.shape == (2, 8, 7)
    assert torch.equal(action[..., :6], pose_norm * 2.0 + torch.arange(6))
    assert torch.equal(action[0, :, 6], torch.full((8,), -1.0))
    assert torch.equal(action[1, :, 6], torch.full((8,), 1.0))


def test_serving_decoder_rejects_bad_normalizer_and_implicit_polarity() -> None:
    outputs = {
        "base_policy_pose_norm": torch.zeros(1, 8, 6),
        "base_policy_gripper_logit": torch.zeros(1, 8),
    }
    with pytest.raises(V8ActionPolicyTransitionError, match="std must be positive"):
        decode_v8_executable_action_chunk(
            outputs,
            pose_mean=torch.zeros(6),
            pose_std=torch.zeros(6),
        )
    with pytest.raises(V8ActionPolicyTransitionError, match="gripper_output must"):
        decode_v8_executable_action_chunk(
            outputs,
            pose_mean=torch.zeros(6),
            pose_std=torch.ones(6),
            gripper_output="guess_libero",
        )
