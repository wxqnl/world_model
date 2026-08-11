"""Strict WM3D-V8 Stage0 to downstream action-policy inheritance.

LIBERO and other downstream robot tasks must reuse the executable Stage0
``action_policy`` tensors and ABI.  There is no legacy inference fallback,
partial policy load, horizon expansion, or silent head replacement here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import torch

from wm3d_v3.data.v8_proprio_contract import (
    V8_EMBODIMENT_VOCAB,
    V8_EMBODIMENT_VOCAB_SHA256,
    V8_PROPRIO_ANCHOR,
    V8_PROPRIO_DIM,
    V8_PROPRIO_LAYOUT,
    V8_PROPRIO_SCHEMA,
)


ACTION_POLICY_PREFIX = "action_policy."
CONTRACT_SCHEMA_V2 = "wm3d_v8_stage0_action_policy_contract_v2"
CONTRACT_SCHEMA_V3 = "wm3d_v8_stage0_action_policy_contract_v3"
CONTRACT_SCHEMA = CONTRACT_SCHEMA_V2
CONTRACT_SCHEMAS = {CONTRACT_SCHEMA_V2, CONTRACT_SCHEMA_V3}
CONFIG_SCHEMA_TO_CONTRACT_SCHEMA = {
    "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2": CONTRACT_SCHEMA_V2,
    "wm3d_v8_stage0_causal_dual_view_unified_action_formal_v2": CONTRACT_SCHEMA_V2,
    "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v3": CONTRACT_SCHEMA_V3,
    "wm3d_v8_stage0_causal_dual_view_unified_action_formal_v3": CONTRACT_SCHEMA_V3,
}
LOWER_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class V8ActionPolicyTransitionError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def action_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_config_sha256(config: Mapping[str, Any]) -> str:
    """Recompute the runtime config digest stored before its self field."""

    try:
        payload = json.loads(json.dumps(config))
    except (TypeError, ValueError) as exc:
        raise V8ActionPolicyTransitionError(
            f"Stage0 checkpoint cfg is not JSON-serializable: {exc}"
        ) from exc
    train = payload.get("train") or {}
    train.pop("resolved_config_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def validate_v8_action_policy_contract(contract: Any) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise V8ActionPolicyTransitionError("Stage0 checkpoint has no V8 action contract")
    schema = str(contract.get("schema") or "")
    expected = {
        "world_state_hz": 5,
        "policy_hz": 20,
        "policy_horizon": 8,
        "policy_chunk_seconds": 0.4,
        "policy_output_shape": [8, 7],
        "serving_owner": "action_policy.base_policy.[pose_norm,gripper_logit]",
        "serving_decoder": "decode_v8_executable_action_chunk_v1",
        "flow_owner": False,
        "delta_event_owner": False,
        "policy_context_source": "core_pred",
        "policy_core_action_cond": "none",
        "stage0_native3d_owner": True,
    }
    mismatches: dict[str, dict[str, Any]] = {}
    if schema not in CONTRACT_SCHEMAS:
        mismatches["schema"] = {
            "actual": schema,
            "expected": sorted(CONTRACT_SCHEMAS),
        }
    mismatches.update({
        key: {"actual": contract.get(key), "expected": value}
        for key, value in expected.items()
        if contract.get(key) != value
    })
    pose = contract.get("pose") or {}
    gripper = contract.get("gripper") or {}
    history = contract.get("history") or {}
    dynamics = contract.get("dynamics_condition") or {}
    nested_mismatches = {
        "pose.frame": (pose.get("frame"), "base"),
        "pose.translation_unit": (pose.get("translation_unit"), "meter"),
        "pose.translation_semantics": (
            pose.get("translation_semantics"),
            "per_controller_step_delta",
        ),
        "pose.rotation": (pose.get("rotation"), "axis_angle_rotvec_radian_delta"),
        "pose.normalization": (
            pose.get("normalization"),
            "source_bound_affine_pose6",
        ),
        "gripper.semantics": (gripper.get("semantics"), "absolute_close01"),
        "gripper.threshold": (gripper.get("threshold"), 0.5),
        "gripper.environment_polarity_conversion": (
            gripper.get("environment_polarity_conversion"),
            "execution_boundary_only",
        ),
        "history.schema": (
            history.get("schema"),
            "wm3d_v8_action_history_20hz_dt_valid_v1",
        ),
        "history.length": (history.get("length"), 16),
        "history.dim": (history.get("dim"), 9),
        "dynamics.schema": (
            dynamics.get("schema"),
            "wm3d_v8_dual_rate_action_v1",
        ),
        "dynamics.dim": (dynamics.get("dim"), 36),
        "dynamics.substeps": (dynamics.get("substeps_per_world_interval"), 4),
        "dynamics.layout": (
            dynamics.get("layout"),
            "4x(action7)+valid4+dt_seconds4",
        ),
    }
    for key, (actual, expected_value) in nested_mismatches.items():
        if actual != expected_value:
            mismatches[key] = {"actual": actual, "expected": expected_value}
    normalizers = contract.get("normalizers") or {}
    for key in ("robocasa_fine_stats_sha256", "robocasa_coarse_stats_sha256"):
        value = str(normalizers.get(key) or "")
        if LOWER_HEX64.fullmatch(value) is None:
            mismatches[f"normalizers.{key}"] = {
                "actual": value,
                "expected": "lowercase SHA256",
            }
    oxe = normalizers.get("oxe") or {}
    expected_oxe = {"oxe_bridge_action", "oxe_droid_action"}
    if set(oxe) != expected_oxe:
        mismatches["normalizers.oxe.sources"] = {
            "actual": sorted(oxe),
            "expected": sorted(expected_oxe),
        }
    for source_name, source in oxe.items():
        if not isinstance(source, dict):
            mismatches[f"normalizers.oxe.{source_name}"] = {
                "actual": type(source).__name__,
                "expected": "mapping",
            }
            continue
        for key in (
            "cache_manifest_sha256",
            "audit_gate_sha256",
            "evidence_sha256",
        ):
            value = str(source.get(key) or "")
            if LOWER_HEX64.fullmatch(value) is None:
                mismatches[f"normalizers.oxe.{source_name}.{key}"] = {
                    "actual": value,
                    "expected": "lowercase SHA256",
                }
        stats_by_source = source.get("stats_sha256_by_source") or {}
        if not stats_by_source or any(
            LOWER_HEX64.fullmatch(str(value)) is None
            for value in stats_by_source.values()
        ):
            mismatches[
                f"normalizers.oxe.{source_name}.stats_sha256_by_source"
            ] = {
                "actual": stats_by_source,
                "expected": "non-empty source-to-SHA256 mapping",
            }
    proprio = contract.get("proprio")
    if schema == CONTRACT_SCHEMA_V2 and proprio is not None:
        mismatches["proprio"] = {
            "actual": "present",
            "expected": "absent from v2",
        }
    if schema == CONTRACT_SCHEMA_V3:
        if not isinstance(proprio, dict):
            mismatches["proprio"] = {
                "actual": type(proprio).__name__,
                "expected": "mapping",
            }
            proprio = {}
        expected_proprio = {
            "schema": V8_PROPRIO_SCHEMA,
            "required": True,
            "dim": V8_PROPRIO_DIM,
            "layout": list(V8_PROPRIO_LAYOUT),
            "anchor": V8_PROPRIO_ANCHOR,
            "normalization": "source_bound_train_only_affine",
            "embodiment_vocab": dict(V8_EMBODIMENT_VOCAB),
            "embodiment_vocab_sha256": V8_EMBODIMENT_VOCAB_SHA256,
        }
        for key, expected_value in expected_proprio.items():
            actual = proprio.get(key)
            if actual != expected_value:
                mismatches[f"proprio.{key}"] = {
                    "actual": actual,
                    "expected": expected_value,
                }
        sources = proprio.get("sources") or {}
        expected_sources = {
            "robocasa": ("panda_robocasa_libero", 2),
            "droid": ("franka_droid", 0),
            "bridge": ("widowx_bridge", 1),
        }
        if not isinstance(sources, dict) or set(sources) != set(expected_sources):
            mismatches["proprio.sources"] = {
                "actual": (
                    sorted(sources)
                    if isinstance(sources, dict)
                    else type(sources).__name__
                ),
                "expected": sorted(expected_sources),
            }
            sources = sources if isinstance(sources, dict) else {}
        for source, (embodiment, embodiment_id) in expected_sources.items():
            value = sources.get(source)
            if not isinstance(value, dict):
                mismatches[f"proprio.sources.{source}"] = {
                    "actual": type(value).__name__,
                    "expected": "mapping",
                }
                continue
            for key, expected_value in (
                ("embodiment", embodiment),
                ("embodiment_id", embodiment_id),
            ):
                if value.get(key) != expected_value:
                    mismatches[f"proprio.sources.{source}.{key}"] = {
                        "actual": value.get(key),
                        "expected": expected_value,
                    }
            for key in ("index_path", "stats_path"):
                if not str(value.get(key) or ""):
                    mismatches[f"proprio.sources.{source}.{key}"] = {
                        "actual": value.get(key),
                        "expected": "non-empty path",
                    }
            for key in ("index_sha256", "stats_sha256"):
                digest = str(value.get(key) or "")
                if LOWER_HEX64.fullmatch(digest) is None:
                    mismatches[f"proprio.sources.{source}.{key}"] = {
                        "actual": digest,
                        "expected": "lowercase SHA256",
                    }
    embedded_sha = str(contract.get("contract_sha256") or "")
    observed_sha = action_contract_sha256(contract)
    if embedded_sha != observed_sha:
        mismatches["contract_sha256"] = {
            "actual": embedded_sha,
            "expected": observed_sha,
        }
    if mismatches:
        raise V8ActionPolicyTransitionError(
            "V8 action ABI mismatch: " + json.dumps(mismatches, sort_keys=True)
        )
    return dict(contract)


def decode_v8_executable_action_chunk(
    outputs: Mapping[str, torch.Tensor],
    *,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    gripper_output: str = "close01",
    hard_gripper: bool = False,
) -> torch.Tensor:
    """Decode the sole normalized Stage0 owner into physical ``[B,8,7]``.

    Pose denormalization is source-bound.  Gripper polarity conversion is
    explicit and occurs only at the robot/environment execution boundary.
    """

    if "base_policy_pose_norm" not in outputs or "base_policy_gripper_logit" not in outputs:
        raise V8ActionPolicyTransitionError(
            "V8 serving requires base_policy_pose_norm and base_policy_gripper_logit"
        )
    pose_norm = outputs["base_policy_pose_norm"]
    grip_logit = outputs["base_policy_gripper_logit"]
    if tuple(pose_norm.shape[-2:]) != (8, 6):
        raise V8ActionPolicyTransitionError(
            f"V8 normalized pose owner must end in [8,6], got {tuple(pose_norm.shape)}"
        )
    if tuple(grip_logit.shape) != tuple(pose_norm.shape[:-1]):
        raise V8ActionPolicyTransitionError(
            "V8 gripper logit shape must match the pose [B,8] prefix"
        )
    mean = torch.as_tensor(pose_mean, device=pose_norm.device, dtype=pose_norm.dtype)
    std = torch.as_tensor(pose_std, device=pose_norm.device, dtype=pose_norm.dtype)
    if mean.ndim == 1:
        mean = mean.reshape(*([1] * (pose_norm.ndim - 1)), 6)
    elif mean.ndim == 2 and pose_norm.ndim == 3:
        mean = mean[:, None, :]
    if std.ndim == 1:
        std = std.reshape(*([1] * (pose_norm.ndim - 1)), 6)
    elif std.ndim == 2 and pose_norm.ndim == 3:
        std = std[:, None, :]
    try:
        broadcast_shape = torch.broadcast_shapes(
            tuple(pose_norm.shape), tuple(mean.shape), tuple(std.shape)
        )
    except RuntimeError as exc:
        raise V8ActionPolicyTransitionError(
            f"pose normalizer cannot broadcast to {tuple(pose_norm.shape)}"
        ) from exc
    if broadcast_shape != tuple(pose_norm.shape):
        raise V8ActionPolicyTransitionError(
            f"pose normalizer broadcasts to {broadcast_shape}, expected {tuple(pose_norm.shape)}"
        )
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise V8ActionPolicyTransitionError("pose normalizer contains non-finite values")
    if bool((std <= 0.0).any()):
        raise V8ActionPolicyTransitionError("pose normalizer std must be positive")
    physical_pose = pose_norm * std + mean
    close01 = torch.sigmoid(grip_logit)
    if hard_gripper:
        close01 = (close01 >= 0.5).to(dtype=physical_pose.dtype)
    else:
        close01 = close01.to(dtype=physical_pose.dtype)
    if gripper_output == "close01":
        grip = close01
    elif gripper_output == "signed_close_positive":
        grip = close01 * 2.0 - 1.0
    elif gripper_output == "signed_open_positive":
        grip = 1.0 - close01 * 2.0
    else:
        raise V8ActionPolicyTransitionError(
            "gripper_output must be close01, signed_close_positive, or signed_open_positive"
        )
    action = torch.cat((physical_pose, grip.unsqueeze(-1)), dim=-1)
    if tuple(action.shape[-2:]) != (8, 7) or not bool(torch.isfinite(action).all()):
        raise V8ActionPolicyTransitionError(
            f"decoded V8 action must be finite [...,8,7], got {tuple(action.shape)}"
        )
    return action


def checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise V8ActionPolicyTransitionError("Stage0 checkpoint model state is missing")
    bad_values = [key for key, value in state.items() if not torch.is_tensor(value)]
    if bad_values:
        raise V8ActionPolicyTransitionError(
            f"Stage0 model state contains non-tensors: {bad_values[:16]}"
        )
    if any(str(key).startswith("module.") for key in state):
        raise V8ActionPolicyTransitionError(
            "Stage0 state unexpectedly contains a DDP module. prefix"
        )
    return state


def validate_v8_stage0_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict):
        raise V8ActionPolicyTransitionError("Stage0 checkpoint cfg is missing")
    config_schema = str(((cfg.get("contract") or {}).get("schema")) or "")
    if config_schema not in CONFIG_SCHEMA_TO_CONTRACT_SCHEMA:
        raise V8ActionPolicyTransitionError(
            f"checkpoint is not a V8 unified-action Stage0 artifact: {config_schema!r}"
        )
    train_cfg = cfg.get("train") or {}
    if train_cfg.get("stage_transition") is not None:
        raise V8ActionPolicyTransitionError("checkpoint is not a fresh Stage0 lineage")
    resolved_sha = str(payload.get("resolved_config_sha256") or "")
    if LOWER_HEX64.fullmatch(resolved_sha) is None:
        raise V8ActionPolicyTransitionError(
            "Stage0 checkpoint resolved_config_sha256 is missing/invalid"
        )
    embedded_cfg_sha = str((train_cfg or {}).get("resolved_config_sha256") or "")
    recomputed_cfg_sha = checkpoint_config_sha256(cfg)
    if resolved_sha != embedded_cfg_sha or resolved_sha != recomputed_cfg_sha:
        raise V8ActionPolicyTransitionError(
            "Stage0 checkpoint config identity mismatch: "
            + json.dumps(
                {
                    "payload": resolved_sha,
                    "cfg_embedded": embedded_cfg_sha,
                    "cfg_recomputed": recomputed_cfg_sha,
                },
                sort_keys=True,
            )
        )
    step = int(payload.get("step", -1))
    if step <= 0:
        raise V8ActionPolicyTransitionError(
            f"Stage0 checkpoint step must be a positive numbered milestone, got {step}"
        )
    contract = validate_v8_action_policy_contract(payload.get("action_policy_contract"))
    expected_contract_schema = CONFIG_SCHEMA_TO_CONTRACT_SCHEMA[config_schema]
    if contract["schema"] != expected_contract_schema:
        raise V8ActionPolicyTransitionError(
            "Stage0 config/action contract schema mismatch: "
            + json.dumps(
                {
                    "config_schema": config_schema,
                    "actual_contract_schema": contract["schema"],
                    "expected_contract_schema": expected_contract_schema,
                },
                sort_keys=True,
            )
        )
    if expected_contract is not None and contract != dict(expected_contract):
        raise V8ActionPolicyTransitionError(
            "downstream expected action ABI differs from Stage0: "
            + json.dumps(
                {"actual": contract, "expected": dict(expected_contract)},
                sort_keys=True,
            )
        )
    state = checkpoint_state(payload)
    policy_keys = sorted(key for key in state if str(key).startswith(ACTION_POLICY_PREFIX))
    if not policy_keys:
        raise V8ActionPolicyTransitionError("Stage0 checkpoint contains no action_policy tensors")
    return {
        "step": step,
        "config_schema": config_schema,
        "resolved_config_sha256": resolved_sha,
        "action_policy_contract": contract,
        "action_policy_contract_sha256": contract["contract_sha256"],
        "model_tensor_count": len(state),
        "action_policy_tensor_count": len(policy_keys),
        "action_policy_keys_sha256": hashlib.sha256(
            "\n".join(policy_keys).encode("utf-8")
        ).hexdigest(),
    }


def load_v8_stage0_for_libero_strict(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the complete Stage0 world+policy state with exact key/shape identity."""

    report = validate_v8_stage0_checkpoint_payload(
        payload, expected_contract=expected_contract
    )
    target = model.module if hasattr(model, "module") else model
    source = checkpoint_state(payload)
    target_state = target.state_dict()
    missing = sorted(set(target_state) - set(source))
    unexpected = sorted(set(source) - set(target_state))
    shape_mismatch = sorted(
        key
        for key in set(source).intersection(target_state)
        if tuple(source[key].shape) != tuple(target_state[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise V8ActionPolicyTransitionError(
            "Stage0→LIBERO model identity mismatch: "
            + json.dumps(
                {
                    "missing": missing[:32],
                    "unexpected": unexpected[:32],
                    "shape_mismatch": shape_mismatch[:32],
                },
                sort_keys=True,
            )
        )
    target.load_state_dict(dict(source), strict=True)
    return {**report, "loaded_tensor_count": len(source), "strict": True}
