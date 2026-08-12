#!/usr/bin/env python3
"""Fail-closed preflight for WM3D-V8 Stage0 causal dual-view training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.wm3d_v8_preflight_common import (  # noqa: E402
    CANONICAL_LAYOUT,
    CANONICAL_VERSION,
    EXPECTED_OXE_ACTION_KINDS,
    EXPECTED_OXE_DATASETS,
    EXPECTED_OXE_OFFSETS,
    EXPECTED_ROTATION_CONVERSIONS,
    EXPECTED_SOURCES,
    FORBIDDEN_DATASETS,
    _Checks,
    _scalar,
    _validate_action_cache_split,
    _validate_canonical_gate,
    _validate_checkpoint_lineage,
    _validate_local_resources,
    _validate_stats,
    load_config,
    resolved_config_sha256,
    sha256_file,
)
from wm3d_v3.data.v8_causal_dual_view import (  # noqa: E402
    CAUSAL_DUAL_VIEW_REPRESENTATION,
    CAUSAL_DUAL_VIEW_SCHEMA,
    GEOMETRY_COORDINATE_FRAME,
    TARGET_USAGE,
    validate_causal_dual_view_archive,
)
from wm3d_v3.data.v8_action_contract import (  # noqa: E402
    DYNAMICS_ACTION_DIM,
    POLICY_HISTORY_DIM,
    POLICY_HISTORY_LEN,
    POLICY_HORIZON,
    V8_ACTION_SIDECAR_INDEX_SCHEMA,
    V8_ACTION_SIDECAR_SCHEMA,
    V8_ACTION_STATS_SCHEMA,
    V8_DUAL_RATE_ACTION_SCHEMA,
    V8_POLICY_HISTORY_SCHEMA,
)
from wm3d_v3.data.v8_proprio_contract import (  # noqa: E402
    V8_EMBODIMENT_VOCAB,
    V8_EMBODIMENT_VOCAB_SHA256,
    V8_PROPRIO_ANCHOR,
    V8_PROPRIO_DIM,
    V8_PROPRIO_LAYOUT,
    V8_PROPRIO_SCHEMA,
    V8ProprioStore,
)
from wm3d_v3.training.train import (  # noqa: E402
    apply_direct_policy_oxe_overrides,
    build_v8_action_policy_contract,
    build_datasets,
    validate_action_pretraining_preflight,
)


REPORT_SCHEMA = "wm3d_v8_stage0_causal_dual_view_preflight_report_v1"
CANARY_SCHEMA = "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2"
FORMAL_SCHEMA = "wm3d_v8_stage0_causal_dual_view_unified_action_formal_v2"
CANARY_SCHEMA_V3 = "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v3"
FORMAL_SCHEMA_V3 = "wm3d_v8_stage0_causal_dual_view_unified_action_formal_v3"
CANARY_GATE_SCHEMA = "wm3d_v8_stage0_causal_dual_view_canary_gate_v1"
EXPECTED_PROFILES = {
    CANARY_SCHEMA: "canary_non_inheritable",
    FORMAL_SCHEMA: "formal",
    CANARY_SCHEMA_V3: "canary_non_inheritable",
    FORMAL_SCHEMA_V3: "formal",
}
EXPECTED_PARTITIONS = {
    "robocasa_atomic": "atomic",
    "robocasa_composite": "composite",
    "robocasa_mg": "mg",
}
RUNTIME_FILES = (
    PROJECT_ROOT / "scripts/preflight_wm3d_v8_stage0_causal_dual_view.py",
    PROJECT_ROOT / "scripts/wm3d_v8_preflight_common.py",
    PROJECT_ROOT / "wm3d_v3/data/v8_causal_dual_view.py",
    PROJECT_ROOT / "wm3d_v3/data/v8_action_contract.py",
    PROJECT_ROOT / "wm3d_v3/data/v8_proprio_contract.py",
    PROJECT_ROOT / "wm3d_v3/data/window_dataset.py",
    PROJECT_ROOT / "wm3d_v3/data/v7_compact_dataset.py",
    PROJECT_ROOT / "wm3d_v3/training/train.py",
)
LOWER_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CausalDualViewPreflightError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_runtime_dependencies(
    checks: _Checks, config: dict[str, Any]
) -> dict[str, bool]:
    model = config.get("model") or {}
    loss = config.get("loss") or {}
    try:
        rgb_lpips = float(loss.get("rgb_lpips", 0.0))
    except (TypeError, ValueError):
        rgb_lpips = 0.0
    required = bool(model.get("enable_pixel")) and rgb_lpips > 0.0
    report: dict[str, bool] = {}
    if required:
        available = importlib.util.find_spec("lpips") is not None
        report["lpips"] = available
        checks.expect(
            available,
            "runtime dependency lpips is required when loss.rgb_lpips > 0",
        )
    return report


def _np_scalar(archive: Any, name: str) -> Any:
    if name not in archive:
        raise KeyError(name)
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar, got {value.shape}")
    return value.item()


def _effective_oxe_sources(
    checks: _Checks, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    data = config.get("data") or {}
    raw_sources = data.get("oxe_sources")
    checks.expect(isinstance(raw_sources, list), "data.oxe_sources must be a list")
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_sources, list):
        return result
    for raw in raw_sources:
        if not isinstance(raw, dict) or not raw.get("source_name"):
            checks.errors.append("every OXE source requires source_name")
            continue
        name = str(raw["source_name"])
        checks.expect(name not in result, f"duplicate OXE source: {name}")
        try:
            result[name] = apply_direct_policy_oxe_overrides(raw, data)
        except (TypeError, ValueError) as exc:
            checks.errors.append(f"{name}: invalid OXE override: {exc}")
    return result


def _validate_contract_and_objective(
    checks: _Checks, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    contract = config.get("contract") or {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    train = config.get("train") or {}
    loss = config.get("loss") or {}
    schema = contract.get("schema")
    unified_action = schema in EXPECTED_PROFILES
    proprio_v3 = schema in {CANARY_SCHEMA_V3, FORMAL_SCHEMA_V3}

    checks.expect(schema in EXPECTED_PROFILES, f"unsupported contract schema: {schema!r}")
    checks.equal(
        bool(data.get("v8_proprio_enabled", False)),
        proprio_v3,
        "contract.schema/data.v8_proprio_enabled",
    )
    if not proprio_v3:
        checks.equal(
            bool(model.get("policy_require_lowdim_state", False)),
            False,
            "v2.model.policy_require_lowdim_state",
        )
        checks.equal(
            bool(model.get("policy_require_embodiment", False)),
            False,
            "v2.model.policy_require_embodiment",
        )
    if schema in EXPECTED_PROFILES:
        checks.equal(contract.get("profile"), EXPECTED_PROFILES[schema], "contract.profile")
    checks.equal(
        contract.get("causal_dual_view_schema"),
        CAUSAL_DUAL_VIEW_SCHEMA,
        "contract.causal_dual_view_schema",
    )
    checks.equal(
        contract.get("causal_dual_view_representation"),
        CAUSAL_DUAL_VIEW_REPRESENTATION,
        "contract.causal_dual_view_representation",
    )
    checks.equal(contract.get("context_future_leakage"), False, "contract.leakage")
    checks.equal(contract.get("target_usage"), TARGET_USAGE, "contract.target_usage")
    checks.equal(
        contract.get("geometry_coordinate_frame"),
        GEOMETRY_COORDINATE_FRAME,
        "contract.geometry_coordinate_frame",
    )
    checks.equal(
        contract.get("future_observation_inputs_forbidden"),
        True,
        "contract.future_observation_inputs_forbidden",
    )
    checks.equal(
        contract.get("native_3d_outputs_required"),
        ["rgb", "depth", "point", "pose"],
        "contract.native_3d_outputs_required",
    )
    canonical = contract.get("canonical_action") or {}
    checks.equal(canonical.get("version"), CANONICAL_VERSION, "canonical.version")
    checks.equal(canonical.get("layout"), CANONICAL_LAYOUT, "canonical.layout")
    checks.equal(canonical.get("source_frame"), "base", "canonical.source_frame")
    checks.equal(canonical.get("translation_unit"), "meter", "canonical.translation_unit")
    checks.equal(canonical.get("rotation_representation"), "axis_angle", "canonical.rotation")
    checks.equal(
        canonical.get("gripper_semantics"),
        "signed_close_positive_continuous",
        "canonical.gripper",
    )

    checks.equal(model.get("enable_action_policy"), True, "model.enable_action_policy")
    checks.equal(model.get("policy_flow_use_as_policy"), False, "model.flow_serving")
    built_action_contract: dict[str, Any] | None = None
    if unified_action:
        checks.equal(model.get("policy_enable_flow_head"), False, "model.flow_head")
        checks.equal(model.get("policy_enable_grip_delta_head"), False, "model.delta_head")
        checks.equal(model.get("policy_grip_owner"), "absolute", "model.grip_owner")
        checks.equal(model.get("policy_action_add_trunk"), False, "model.trunk_action_owner")
        checks.equal(model.get("policy_enable_local_residual"), False, "model.local_residual_owner")
        checks.equal(model.get("policy_enable_waypoint_head"), False, "model.waypoint_owner")
        checks.equal(model.get("policy_horizon"), POLICY_HORIZON, "model.policy_horizon")
        checks.equal(
            model.get("policy_action_history_len"),
            POLICY_HISTORY_LEN,
            "model.policy_action_history_len",
        )
        checks.equal(
            model.get("policy_action_history_dim"),
            POLICY_HISTORY_DIM,
            "model.policy_action_history_dim",
        )
        checks.equal(
            model.get("context_pixel_action_dim"),
            DYNAMICS_ACTION_DIM,
            "model.context_pixel_action_dim",
        )
        checks.equal(
            contract.get("action_policy_contract_schema"),
            (
                "wm3d_v8_stage0_action_policy_contract_v3"
                if proprio_v3
                else "wm3d_v8_stage0_action_policy_contract_v2"
            ),
            "contract.action_policy_contract_schema",
        )
        try:
            built_action_contract = build_v8_action_policy_contract(config)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            message = f"action contract build failed: {exc}"
            unresolved_template = checks.mode == "static" and "PENDING_" in str(exc)
            (checks.blockers if unresolved_template else checks.errors).append(message)
        else:
            checks.equal(
                (built_action_contract or {}).get("schema"),
                (
                    "wm3d_v8_stage0_action_policy_contract_v3"
                    if proprio_v3
                    else "wm3d_v8_stage0_action_policy_contract_v2"
                ),
                "built action contract schema",
            )
        checks.equal(contract.get("world_state_hz"), 5, "contract.world_state_hz")
        checks.equal(contract.get("policy_hz"), 20, "contract.policy_hz")
        checks.equal(contract.get("policy_horizon"), 8, "contract.policy_horizon")
    if proprio_v3:
        checks.equal(
            model.get("policy_head_type"),
            "native",
            "model.policy_head_type",
        )
        checks.equal(
            model.get("policy_lowdim_dim"),
            V8_PROPRIO_DIM,
            "model.policy_lowdim_dim",
        )
        checks.equal(
            model.get("policy_require_lowdim_state"),
            True,
            "model.policy_require_lowdim_state",
        )
        checks.equal(
            model.get("policy_embodiment_vocab_size"),
            len(V8_EMBODIMENT_VOCAB),
            "model.policy_embodiment_vocab_size",
        )
        checks.equal(
            model.get("policy_require_embodiment"),
            True,
            "model.policy_require_embodiment",
        )
    checks.equal(model.get("enable_pixel"), True, "model.enable_pixel")
    checks.equal(model.get("enable_geom_extra"), True, "model.enable_geom_extra")
    checks.equal(model.get("token_codec_frozen"), True, "model.token_codec_frozen")
    for key in ("state", "action"):
        section = model.get(key) or {}
        checks.equal(section.get("T"), 16, f"model.{key}.T")
        checks.equal(section.get("P"), 64, f"model.{key}.P")
        checks.equal(section.get("D"), 2048, f"model.{key}.D")
        checks.equal(section.get("k"), 8, f"model.{key}.k")
        if unified_action:
            checks.equal(
                section.get("action_cond_dim"),
                DYNAMICS_ACTION_DIM,
                f"model.{key}.action_cond_dim",
            )

    checks.equal(data.get("dataset_type"), "v7_mixed", "data.dataset_type")
    checks.equal(data.get("T"), 16, "data.T")
    checks.equal(data.get("k"), 8, "data.k")
    checks.equal(
        data.get("compact_causal_dual_view_required"),
        True,
        "data.compact_causal_dual_view_required",
    )
    checks.equal(
        data.get("compact_causal_dual_view_representation"),
        CAUSAL_DUAL_VIEW_REPRESENTATION,
        "data.compact_causal_dual_view_representation",
    )
    checks.equal(
        data.get("robocasa_partitions"),
        ["atomic", "composite", "mg"],
        "data.robocasa_partitions",
    )
    if unified_action:
        checks.equal(
            data.get("v8_dual_rate_action_enabled"),
            True,
            "data.v8_dual_rate_action_enabled",
        )
        checks.equal(
            data.get("policy_action_history_len"),
            POLICY_HISTORY_LEN,
            "data.policy_action_history_len",
        )
        checks.equal(
            data.get("policy_action_history_dim"),
            POLICY_HISTORY_DIM,
            "data.policy_action_history_dim",
        )
    if proprio_v3:
        checks.equal(data.get("v8_proprio_enabled"), True, "data.v8_proprio_enabled")
        checks.equal(
            data.get("load_policy_state", False),
            False,
            "data.load_policy_state",
        )
    checks.equal(
        train.get("joint_native_action_pretraining"),
        True,
        "train.joint_native_action_pretraining",
    )
    checks.equal(train.get("direct_policy_weight"), 1.0, "train.direct_policy_weight")
    checks.equal(
        train.get("policy_flow_weight"),
        0.0,
        "train.policy_flow_weight",
    )
    checks.equal(
        train.get("native_action_no_teacher_weight"),
        0.0,
        "train.native_action_no_teacher_weight",
    )
    checks.equal(
        train.get("native_action_no_teacher_start_step"),
        0,
        "train.native_action_no_teacher_start_step",
    )
    checks.equal(
        train.get("native_action_no_teacher_every"),
        1,
        "train.native_action_no_teacher_every",
    )
    checks.equal(
        train.get("native_future_no_teacher_weight"),
        0.20,
        "train.native_future_no_teacher_weight",
    )
    checks.equal(
        train.get("native_future_no_teacher_weight_start_step"),
        0,
        "train.native_future_no_teacher_weight_start_step",
    )
    checks.equal(
        train.get("direct_policy_grip_owner"),
        "absolute",
        "train.grip_owner",
    )
    if unified_action:
        checks.equal(
            train.get("v8_dual_rate_action_enabled"),
            True,
            "train.v8_dual_rate_action_enabled",
        )
        checks.equal(train.get("direct_policy_head"), "base", "train.direct_policy_head")
        for key in (
            "v8_policy_fine_pose_weight",
            "v8_policy_fine_grip_weight",
            "v8_policy_coarse_pose_weight",
            "v8_policy_coarse_grip_weight",
        ):
            try:
                positive = float(train.get(key, 0.0)) > 0.0
            except (TypeError, ValueError):
                positive = False
            checks.expect(positive, f"train.{key} must be positive")
        if built_action_contract is not None:
            try:
                validate_action_pretraining_preflight(config)
            except (TypeError, ValueError, RuntimeError) as exc:
                message = f"V8 unified action preflight failed: {exc}"
                unresolved_template = checks.mode == "static" and "PENDING_" in str(exc)
                (checks.blockers if unresolved_template else checks.errors).append(message)
    factual = train.get("factual_action_conditioning") or {}
    checks.equal(factual.get("enabled"), True, "train.factual.enabled")
    checks.equal(factual.get("start_step"), 0, "train.factual.start_step")
    checks.equal(factual.get("detach_action_condition"), False, "train.factual.detach")
    checks.equal(
        set(factual.get("required_sources") or ()),
        set(EXPECTED_SOURCES),
        "train.factual.required_sources",
    )
    sampler = train.get("mixed_batch_sampler") or {}
    checks.equal(
        sampler.get("source_cycle_counts_exact"),
        EXPECTED_SOURCES,
        "train.mixed_batch_sampler.source_cycle_counts_exact",
    )
    checks.equal(sampler.get("cycle_optimizer_steps"), 100, "train.sampler.cycle")
    checks.equal(sampler.get("synchronized_across_ranks"), True, "train.sampler.sync")
    checks.equal(
        set(sampler.get("forbidden_sources") or ()),
        FORBIDDEN_DATASETS,
        "train.sampler.forbidden_sources",
    )

    checks.equal(train.get("fresh_initialization_required"), True, "train.fresh_initialization")
    checks.equal(train.get("fresh_init_required"), True, "train.fresh_init")
    checks.equal(train.get("forbid_warm_start"), True, "train.forbid_warm_start")
    checks.equal(train.get("forbid_cross_run_resume"), True, "train.forbid_cross_run_resume")
    checks.equal(train.get("stage_transition"), None, "train.stage_transition")
    checks.equal(train.get("resume_checkpoint"), None, "train.resume_checkpoint")
    checks.equal(train.get("pretrained_world_checkpoint"), None, "train.pretrained_world_checkpoint")
    checks.equal(
        train.get("model_initialization"),
        "random_world_weights_with_frozen_pinned_codec",
        "train.model_initialization",
    )
    lineage = str(train.get("run_lineage") or "")
    checks.expect(
        bool(lineage) and "causal_dual_view" in lineage,
        f"fresh causal run_lineage is missing: {lineage!r}",
    )
    for key in ("rgb_l1", "rgb_lpips", "geom_depth", "geom_point", "geom_pose"):
        try:
            valid = float(loss.get(key, 0.0)) > 0.0
        except (TypeError, ValueError):
            valid = False
        checks.expect(valid, f"loss.{key} must be positive")

    sources = _effective_oxe_sources(checks, config)
    checks.equal(set(sources), set(EXPECTED_OXE_DATASETS), "data.oxe_sources")
    for name, source in sources.items():
        checks.equal(
            source.get("include_datasets"),
            [EXPECTED_OXE_DATASETS[name]],
            f"{name}.include_datasets",
        )
        checks.equal(source.get("T"), 16, f"{name}.T")
        checks.equal(source.get("k"), 8, f"{name}.k")
        if unified_action:
            checks.equal(
                source.get("v8_dual_rate_action_enabled"),
                True,
                f"{name}.v8_dual_rate_action_enabled",
            )
            checks.equal(
                source.get("policy_action_history_len"),
                POLICY_HISTORY_LEN,
                f"{name}.policy_action_history_len",
            )
            checks.equal(
                source.get("policy_action_history_dim"),
                POLICY_HISTORY_DIM,
                f"{name}.policy_action_history_dim",
            )
        if proprio_v3:
            checks.equal(
                source.get("v8_proprio_enabled"),
                True,
                f"{name}.v8_proprio_enabled",
            )
            checks.equal(
                source.get("load_policy_state", False),
                False,
                f"{name}.load_policy_state",
            )
            for key in (
                "v8_proprio_index",
                "v8_proprio_index_sha256",
                "v8_proprio_stats",
                "v8_proprio_stats_sha256",
            ):
                checks.expect(
                    bool(source.get(key)),
                    f"{name}.{key} is missing",
                )
        checks.equal(source.get("causal_dual_view_required"), True, f"{name}.causal_required")
        checks.equal(
            source.get("causal_dual_view_representation"),
            CAUSAL_DUAL_VIEW_REPRESENTATION,
            f"{name}.causal_representation",
        )
        for key in (
            "load_rgb",
            "load_geom",
            "load_state_tgt",
            "load_geom_extra",
            "require_geom_extra",
            "use_window_tokens",
            "trust_window_geom_cache",
        ):
            checks.equal(source.get(key), True, f"{name}.{key}")
        checks.equal(source.get("window_geom_shard_index"), None, f"{name}.shard_index")
        checks.equal(source.get("window_geom_shard_root"), None, f"{name}.shard_root")
        checks.expect(bool(source.get("cache_root")), f"{name}.cache_root is missing")
        checks.expect(
            bool(source.get("window_geom_cache_root")),
            f"{name}.window_geom_cache_root is missing",
        )
        checks.expect(bool(source.get("window_geom_subdir")), f"{name}.window_geom_subdir is missing")
    return sources


def _validate_v8_action_sidecars(
    checks: _Checks, data: dict[str, Any]
) -> dict[str, Any]:
    if not bool(data.get("v8_dual_rate_action_enabled", False)):
        return {"enabled": False}
    index_path = checks.pinned_file(
        data.get("v8_action_sidecar_index"),
        data.get("v8_action_sidecar_index_sha256"),
        "robocasa.action20_index",
    )
    stats_path = checks.pinned_file(
        data.get("v8_action_sidecar_stats"),
        data.get("v8_action_sidecar_stats_sha256"),
        "robocasa.action20_stats",
    )
    report: dict[str, Any] = {"enabled": True, "rows": 0}
    if checks.mode != "full" or index_path is None or stats_path is None:
        return report
    compact_path = Path(str(data.get("compact_index") or ""))
    compact_keys: set[tuple[str, str]] = set()
    compact_rows: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for line in compact_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("split")), str(row.get("clip_hash")))
            checks.expect(key not in compact_rows, f"robocasa compact duplicate: {key}")
            compact_keys.add(key)
            compact_rows[key] = row
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"robocasa compact index cannot be audited for action20: {exc}")
    sidecar_keys: set[tuple[str, str]] = set()
    sidecar_adapter_digests: set[str] = set()
    train_fine_action_count = 0
    try:
        for line_number, line in enumerate(
            index_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            label = f"robocasa.action20_index[{line_number}]"
            checks.equal(row.get("schema"), V8_ACTION_SIDECAR_INDEX_SCHEMA, f"{label}.schema")
            key = (str(row.get("split")), str(row.get("clip_hash")))
            checks.expect(key not in sidecar_keys, f"{label} duplicate identity: {key}")
            sidecar_keys.add(key)
            compact_row = compact_rows.get(key)
            checks.expect(
                compact_row is not None,
                f"{label} has no matching compact row: {key}",
            )
            if compact_row is not None:
                for identity_key in (
                    "source",
                    "v7_source",
                    "action_audit_sha256",
                ):
                    checks.equal(
                        row.get(identity_key),
                        compact_row.get(identity_key),
                        f"{label}.{identity_key}",
                    )
                checks.equal(
                    row.get("source_archive_path"),
                    str(Path(str(compact_row.get("path") or "")).resolve()),
                    f"{label}.source_archive_path",
                )
                checks.equal(
                    row.get("source_archive_sha256"),
                    compact_row.get("artifact_sha256"),
                    f"{label}.source_archive_sha256",
                )
            adapter_digest = str(row.get("action_audit_sha256") or "")
            checks.expect(
                LOWER_HEX64.fullmatch(adapter_digest) is not None,
                f"{label}.action_audit_sha256 is invalid",
            )
            if LOWER_HEX64.fullmatch(adapter_digest) is not None:
                sidecar_adapter_digests.add(adapter_digest)
            if key[0] == "train":
                train_fine_action_count += int(row.get("fine_action_count", 0) or 0)
            artifact = Path(str(row.get("path") or ""))
            expected_sha = str(row.get("artifact_sha256") or "")
            checks.expect(artifact.is_file(), f"{label} artifact is missing: {artifact}")
            checks.expect(
                LOWER_HEX64.fullmatch(expected_sha) is not None,
                f"{label} artifact SHA256 is invalid",
            )
            if not artifact.is_file() or LOWER_HEX64.fullmatch(expected_sha) is None:
                continue
            observed = sha256_file(artifact)
            checks.equal(observed, expected_sha, f"{label}.artifact_sha256")
            try:
                with np.load(artifact, allow_pickle=False) as archive:
                    checks.equal(
                        _np_scalar(archive, "schema"),
                        V8_ACTION_SIDECAR_SCHEMA,
                        f"{label}.payload.schema",
                    )
                    checks.equal(
                        _np_scalar(archive, "clip_hash"),
                        key[1],
                        f"{label}.payload.clip_hash",
                    )
                    for identity_key in (
                        "split",
                        "source",
                        "v7_source",
                        "source_archive_path",
                        "source_archive_sha256",
                        "action_audit_sha256",
                    ):
                        checks.equal(
                            _np_scalar(archive, identity_key),
                            row.get(identity_key),
                            f"{label}.payload.{identity_key}",
                        )
                    checks.equal(
                        int(_np_scalar(archive, "policy_hz")),
                        20,
                        f"{label}.payload.policy_hz",
                    )
                    checks.equal(
                        int(_np_scalar(archive, "world_hz")),
                        5,
                        f"{label}.payload.world_hz",
                    )
                    actions = np.asarray(archive["fine_actions"], dtype=np.float32)
                    checks.equal(actions.ndim, 2, f"{label}.fine_actions.ndim")
                    if actions.ndim == 2:
                        checks.equal(actions.shape[1], 7, f"{label}.fine_actions.dim")
                        checks.equal(
                            actions.shape[0],
                            int(row.get("fine_action_count", -1)),
                            f"{label}.fine_action_count",
                        )
                    checks.expect(
                        bool(np.isfinite(actions).all()),
                        f"{label}.fine_actions contains non-finite values",
                    )
                    native_indices = np.asarray(
                        archive["native_frame_indices"], dtype=np.int64
                    )
                    checks.equal(
                        native_indices.shape,
                        (int(row.get("fine_action_count", -1)),),
                        f"{label}.native_frame_indices",
                    )
                    composition_error = np.asarray(
                        archive["composition_max_abs_by_dim"], dtype=np.float32
                    )
                    checks.equal(
                        composition_error.shape,
                        (7,),
                        f"{label}.composition_error.shape",
                    )
                    checks.expect(
                        bool(np.isfinite(composition_error).all())
                        and bool((composition_error <= 2.0e-5).all()),
                        f"{label}.composition error exceeds 2e-5",
                    )
                    checks.equal(
                        int(_np_scalar(archive, "dropped_tail_actions")),
                        int(row.get("dropped_tail_actions", -1)),
                        f"{label}.dropped_tail_actions",
                    )
                    checks.expect(
                        0 <= int(row.get("dropped_tail_actions", -1)) < 4,
                        f"{label}.dropped_tail_actions must be in [0,3]",
                    )
            except (OSError, KeyError, ValueError) as exc:
                checks.errors.append(f"{label} payload is invalid: {exc}")
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"robocasa action20 index is unreadable: {exc}")
    checks.equal(sidecar_keys, compact_keys, "robocasa.action20 exact compact coverage")
    report["rows"] = len(sidecar_keys)
    try:
        with np.load(stats_path, allow_pickle=False) as archive:
            checks.equal(
                _np_scalar(archive, "schema"),
                V8_ACTION_STATS_SCHEMA,
                "robocasa.action20_stats.schema",
            )
            checks.equal(
                _np_scalar(archive, "split"),
                "train",
                "robocasa.action20_stats.split",
            )
            mean = np.asarray(archive["mean"], dtype=np.float32)
            std = np.asarray(archive["std"], dtype=np.float32)
            checks.equal(mean.shape, (6,), "robocasa.action20_stats.mean")
            checks.equal(std.shape, (6,), "robocasa.action20_stats.std")
            checks.expect(bool(np.isfinite(mean).all()), "action20 mean is non-finite")
            checks.expect(
                bool(np.isfinite(std).all()) and bool((std > 0.0).all()),
                "action20 std must be finite and positive",
            )
            checks.expect(
                int(_np_scalar(archive, "count")) > 0,
                "action20 train stats count must be positive",
            )
            checks.equal(
                int(_np_scalar(archive, "count")),
                train_fine_action_count,
                "robocasa.action20_stats.train_action_count",
            )
            input_indices = json.loads(
                str(_np_scalar(archive, "input_indices_json"))
            )
            compact_resolved = str(compact_path.resolve())
            checks.equal(
                input_indices.get(compact_resolved),
                data.get("compact_index_sha256"),
                "robocasa.action20_stats.compact_index_sha256",
            )
            adapter_audits = json.loads(
                str(_np_scalar(archive, "adapter_audits_json"))
            )
            checks.equal(
                set(adapter_audits.values()),
                sidecar_adapter_digests,
                "robocasa.action20_stats.adapter_audits",
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        checks.errors.append(f"robocasa action20 stats are invalid: {exc}")
    return report


def _validate_v8_proprio_assets(
    checks: _Checks,
    data: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not bool(data.get("v8_proprio_enabled", False)):
        return {"enabled": False, "sources": {}}
    specs: dict[str, dict[str, Any]] = {"robocasa": data}
    for source_name, source_cfg in sources.items():
        canonical_sources = list(source_cfg.get("canonical_action_sources") or ())
        if len(canonical_sources) != 1:
            checks.errors.append(
                f"{source_name}: V8 proprio canonical source is ambiguous: "
                f"{canonical_sources}"
            )
            continue
        canonical_source = str(canonical_sources[0])
        checks.expect(
            canonical_source not in specs,
            f"duplicate V8 proprio source: {canonical_source}",
        )
        specs[canonical_source] = source_cfg
    checks.equal(
        set(specs),
        {"robocasa", "droid", "bridge"},
        "V8 proprio source coverage",
    )
    report: dict[str, Any] = {"enabled": True, "sources": {}}
    for source, source_cfg in sorted(specs.items()):
        index_path = checks.pinned_file(
            source_cfg.get("v8_proprio_index"),
            source_cfg.get("v8_proprio_index_sha256"),
            f"{source}.proprio_index",
        )
        stats_path = checks.pinned_file(
            source_cfg.get("v8_proprio_stats"),
            source_cfg.get("v8_proprio_stats_sha256"),
            f"{source}.proprio_stats",
        )
        source_report: dict[str, Any] = {
            "index": str(index_path) if index_path is not None else None,
            "stats": str(stats_path) if stats_path is not None else None,
        }
        if checks.mode == "full" and index_path is not None and stats_path is not None:
            try:
                store = V8ProprioStore(
                    index_path=index_path,
                    index_sha256=str(source_cfg.get("v8_proprio_index_sha256")),
                    stats_path=stats_path,
                    stats_sha256=str(source_cfg.get("v8_proprio_stats_sha256")),
                    source=source,
                    split=None,
                )
                source_report["records"] = len(store.records)
                checks.expect(
                    len(store.records) > 0,
                    f"{source}.proprio_index contains no records",
                )
            except Exception as exc:
                checks.errors.append(
                    f"{source} V8 proprio assets are invalid: "
                    f"{type(exc).__name__}: {exc}"
                )
        report["sources"][source] = source_report
    return report


def _validate_training_assets(
    checks: _Checks,
    config: dict[str, Any],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not sources:
        return {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    contract = config.get("contract") or {}
    action20_report = _validate_v8_action_sidecars(checks, data)
    proprio_report = _validate_v8_proprio_assets(checks, data, sources)
    checks.pinned_file(
        model.get("token_codec_checkpoint"),
        model.get("token_codec_checkpoint_sha256"),
        "model.token_codec",
    )
    stats = checks.pinned_file(
        data.get("action_stats"),
        data.get("action_stats_sha256"),
        "robocasa.action_stats",
    )
    sidecar_paths = list(data.get("rgb_sidecar_indices") or ())
    sidecar_pins = data.get("rgb_sidecar_sha256") or {}
    checks.equal(set(sidecar_paths), set(sidecar_pins), "robocasa.rgb_sidecar pins")
    for path in sidecar_paths:
        checks.pinned_file(path, sidecar_pins.get(path), f"robocasa.rgb_sidecar:{path}")
    if checks.mode == "full" and stats is not None:
        try:
            with np.load(stats, allow_pickle=False) as archive:
                checks.equal(_scalar(archive, "split"), "train", "robocasa.stats.split")
                checks.equal(
                    _scalar(archive, "index_sha256"),
                    data.get("action_stats_source_index_sha256"),
                    "robocasa.stats.source_index_sha256",
                )
                checks.equal(np.asarray(archive["mean"]).shape, (6,), "robocasa.stats.mean")
                checks.equal(np.asarray(archive["std"]).shape, (6,), "robocasa.stats.std")
        except (OSError, KeyError, ValueError) as exc:
            checks.errors.append(f"robocasa action stats are invalid: {exc}")

    for name, source in sources.items():
        dataset = EXPECTED_OXE_DATASETS[name]
        checks.equal(
            source.get("allowed_action_kinds"),
            [EXPECTED_OXE_ACTION_KINDS[name]],
            f"{name}.action_kind",
        )
        checks.equal(
            source.get("default_action_frame_offset"),
            EXPECTED_OXE_OFFSETS[name],
            f"{name}.action_frame_offset",
        )
        adapter = source.get("action_adapter") or {}
        checks.equal(adapter.get("version"), CANONICAL_VERSION, f"{name}.adapter.version")
        checks.equal(
            adapter.get("rotation_conversion"),
            EXPECTED_ROTATION_CONVERSIONS[name],
            f"{name}.adapter.rotation",
        )
        checks.equal(adapter.get("source_frame"), "base", f"{name}.adapter.frame")
        checks.equal(
            adapter.get("gripper_semantics"),
            "signed_close_positive_continuous",
            f"{name}.adapter.gripper",
        )
        manifest = checks.pinned_file(
            source.get("manifest"), source.get("manifest_sha256"), f"{name}.manifest"
        )
        checks.pinned_file(
            source.get("action_contract_evidence_path"),
            source.get("action_contract_evidence_sha256"),
            f"{name}.temporal_contract",
        )
        if name == "oxe_droid_action":
            checks.pinned_file(
                source.get("droid_provenance_index"),
                source.get("droid_provenance_index_sha256"),
                f"{name}.provenance",
            )
        cache_manifest = checks.pinned_file(
            source.get("canonical_action_cache_manifest"),
            source.get("canonical_action_cache_manifest_sha256"),
            f"{name}.canonical_action_cache",
        )
        gate = checks.pinned_file(
            source.get("action_audit_gate"),
            source.get("action_audit_gate_sha256"),
            f"{name}.action_audit_gate",
        )
        by_source = source.get("canonical_action_stats_by_source") or {}
        by_source_sha = source.get("canonical_action_stats_sha256_by_source") or {}
        stat_path = checks.pinned_file(
            by_source.get(dataset),
            by_source_sha.get(dataset),
            f"{name}.canonical_stats",
        )
        _validate_canonical_gate(checks, source, gate)
        _validate_action_cache_split(checks, source, cache_manifest)
        _validate_stats(checks, source, stat_path)
        if checks.mode == "full":
            root = Path(str(source.get("cache_root") or ""))
            checks.expect(root.is_dir(), f"{name}.cache_root is missing: {root}")
            window_root = Path(
                str(source.get("window_geom_cache_root") or "")
            )
            checks.expect(
                window_root.is_dir(),
                f"{name}.window_geom_cache_root is missing: {window_root}",
            )
            window_subdir = str(source.get("window_geom_subdir") or "")
            checks.expect(
                (window_root / window_subdir).is_dir(),
                f"{name}.window geometry directory is missing: {window_root / window_subdir}",
            )
            checks.expect(manifest is not None, f"{name}.manifest is unavailable")

    gate_report = None
    if contract.get("canary_required") is True:
        gate_path = checks.pinned_file(
            contract.get("canary_gate_receipt"),
            contract.get("canary_gate_receipt_sha256"),
            "contract.canary_gate_receipt",
        )
        if checks.mode == "full" and gate_path is not None:
            try:
                gate_report = json.loads(gate_path.read_text())
                checks.equal(gate_report.get("schema"), CANARY_GATE_SCHEMA, "canary_gate.schema")
                checks.equal(gate_report.get("passed"), True, "canary_gate.passed")
                checks.equal(
                    (gate_report.get("checkpoint") or {}).get("step"),
                    100,
                    "canary_gate.checkpoint.step",
                )
                checks.expect(
                    LOWER_HEX64.fullmatch(
                        str((gate_report.get("checkpoint") or {}).get("sha256") or "")
                    ) is not None,
                    "canary_gate checkpoint SHA256 is missing",
                )
            except (OSError, json.JSONDecodeError) as exc:
                checks.errors.append(f"canary gate is unreadable: {exc}")
    return {
        "canary_gate": gate_report,
        "action20": action20_report,
        "proprio": proprio_report,
    }


def _row_matches_source(source: str, spec: dict[str, Any], row: dict[str, Any]) -> bool:
    if spec.get("kind") == "oxe":
        return row.get("source") == source
    partition = str(spec.get("partition") or "")
    return str(row.get("v7_source", row.get("source", ""))) == partition


def _validate_row_identity(
    checks: _Checks,
    *,
    source: str,
    spec: dict[str, Any],
    row: dict[str, Any],
    line_label: str,
) -> tuple[str, str] | None:
    checks.equal(row.get("schema"), CAUSAL_DUAL_VIEW_SCHEMA, f"{line_label}.schema")
    checks.equal(
        row.get("representation"),
        CAUSAL_DUAL_VIEW_REPRESENTATION,
        f"{line_label}.representation",
    )
    checks.equal(row.get("context_future_leakage"), False, f"{line_label}.leakage")
    checks.equal(row.get("target_usage"), TARGET_USAGE, f"{line_label}.target_usage")
    checks.equal(
        row.get("geometry_coordinate_frame"),
        GEOMETRY_COORDINATE_FRAME,
        f"{line_label}.geometry_coordinate_frame",
    )
    checks.equal(row.get("T"), 16, f"{line_label}.T")
    checks.equal(row.get("k"), 8, f"{line_label}.k")
    checks.equal(row.get("P"), 64, f"{line_label}.P")
    checks.equal(row.get("token_D"), 2048, f"{line_label}.token_D")
    checks.equal(row.get("paired_views"), bool(spec.get("paired_views")), f"{line_label}.paired_views")
    split = str(row.get("split") or "")
    checks.expect(split in {"train", "val", "test"}, f"{line_label}.split is invalid: {split!r}")
    if spec.get("kind") == "oxe":
        identity = str(row.get("clip_id") or "")
        start = row.get("start")
        checks.expect(bool(identity), f"{line_label}.clip_id is missing")
        checks.expect(isinstance(start, int) and start >= 0, f"{line_label}.start is invalid")
        unique = f"{identity}\0{start}"
    else:
        identity = str(row.get("clip_hash") or "")
        starts = row.get("window_starts")
        checks.expect(bool(identity), f"{line_label}.clip_hash is missing")
        checks.expect(
            isinstance(starts, list)
            and bool(starts)
            and starts == sorted(set(starts)),
            f"{line_label}.window_starts is not sorted unique",
        )
        checks.equal(row.get("windows"), len(starts or ()), f"{line_label}.windows")
        unique = identity
    if not identity or not split:
        return None
    return split, unique


def _verify_artifact(
    checks: _Checks,
    *,
    source: str,
    spec: dict[str, Any],
    row: dict[str, Any],
    label: str,
) -> str | None:
    path = Path(str(row.get("path") or ""))
    expected_sha = str(row.get("artifact_sha256") or "")
    if not LOWER_HEX64.fullmatch(expected_sha):
        checks.errors.append(f"{label}.artifact_sha256 is invalid: {expected_sha!r}")
        return None
    if not path.is_file():
        checks.errors.append(f"{label}.artifact is missing: {path}")
        return None
    observed = sha256_file(path)
    if observed != expected_sha:
        checks.errors.append(
            f"{label}.artifact digest mismatch: observed={observed} expected={expected_sha}"
        )
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            summary = validate_causal_dual_view_archive(
                archive,
                T=16,
                k=8,
                paired_views=bool(spec.get("paired_views")),
            )
            checks.equal(summary.get("token_count"), 64, f"{label}.archive.P")
            checks.equal(_np_scalar(archive, "token_dim"), 2048, f"{label}.archive.token_D")
            checks.equal(
                _np_scalar(archive, "latent_dim"),
                row.get("latent_dim"),
                f"{label}.archive.latent_dim",
            )
            checks.equal(_np_scalar(archive, "split"), row.get("split"), f"{label}.archive.split")
            if spec.get("kind") == "oxe":
                checks.equal(
                    _np_scalar(archive, "clip_id"),
                    row.get("clip_id"),
                    f"{label}.archive.clip_id",
                )
                checks.equal(
                    _np_scalar(archive, "start"),
                    row.get("start"),
                    f"{label}.archive.start",
                )
            else:
                checks.equal(
                    _np_scalar(archive, "clip_hash"),
                    row.get("clip_hash"),
                    f"{label}.archive.clip_hash",
                )
                checks.equal(
                    np.asarray(archive["window_starts"], dtype=np.int64).tolist(),
                    row.get("window_starts"),
                    f"{label}.archive.window_starts",
                )
    except (OSError, KeyError, ValueError) as exc:
        checks.errors.append(f"{label}.artifact contract failed: {type(exc).__name__}: {exc}")
        return None
    return observed


def _validate_causal_indices(
    checks: _Checks, config: dict[str, Any]
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    data = config.get("data") or {}
    specs = data.get("causal_dual_view_indices")
    checks.expect(isinstance(specs, dict), "data.causal_dual_view_indices must be a mapping")
    if not isinstance(specs, dict):
        return {}, {}
    checks.equal(set(specs), set(EXPECTED_SOURCES), "causal_dual_view_indices.sources")
    coverage: dict[str, dict[str, int]] = {}
    contract_hashes: dict[str, str] = {}
    for source in EXPECTED_SOURCES:
        spec = specs.get(source)
        if not isinstance(spec, dict):
            checks.errors.append(f"causal index spec is missing for {source}")
            continue
        expected_kind = "oxe" if source.startswith("oxe_") else "compact"
        checks.equal(spec.get("kind"), expected_kind, f"{source}.index.kind")
        checks.equal(
            spec.get("paired_views"),
            expected_kind == "compact",
            f"{source}.index.paired_views",
        )
        if expected_kind == "compact":
            checks.equal(
                spec.get("partition"),
                EXPECTED_PARTITIONS[source],
                f"{source}.index.partition",
            )
        paths = list(spec.get("paths") or ())
        digests = list(spec.get("sha256") or ())
        checks.expect(bool(paths), f"{source}.index.paths is empty")
        checks.equal(len(paths), len(digests), f"{source}.index pin count")
        index_paths: list[Path] = []
        for number, path in enumerate(paths):
            digest = digests[number] if number < len(digests) else None
            pinned = checks.pinned_file(path, digest, f"{source}.index[{number}]")
            if pinned is not None:
                index_paths.append(pinned)
        contract_hashes[source] = _json_sha256(
            {"paths": paths, "sha256": digests, "spec": spec}
        )
        if checks.mode != "full":
            continue
        counts = {"artifacts": 0, "train": 0, "val": 0}
        identities: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        artifacts: list[str] = []
        seen_rows: set[str] = set()
        for index_path in index_paths:
            try:
                lines = index_path.read_text().splitlines()
            except OSError as exc:
                checks.errors.append(f"{source}.index is unreadable: {exc}")
                continue
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    checks.errors.append(
                        f"{source}.index[{line_number}] invalid JSON: {exc}"
                    )
                    continue
                if not _row_matches_source(source, spec, row):
                    continue
                label = f"{source}.index[{line_number}]"
                identity = _validate_row_identity(
                    checks,
                    source=source,
                    spec=spec,
                    row=row,
                    line_label=label,
                )
                if identity is None:
                    continue
                split, unique = identity
                row_key = f"{split}\0{unique}"
                checks.expect(row_key not in seen_rows, f"{source} duplicate row identity: {row_key!r}")
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                identities.setdefault(split, set()).add(unique)
                digest = _verify_artifact(
                    checks, source=source, spec=spec, row=row, label=label
                )
                if digest is not None:
                    artifacts.append(digest)
                    counts["artifacts"] += 1
                    if split in ("train", "val"):
                        counts[split] += 1
        checks.expect(counts["train"] > 0, f"{source} has no train artifacts")
        checks.expect(counts["val"] > 0, f"{source} has no val artifacts")
        overlap = identities["train"] & identities["val"]
        checks.expect(not overlap, f"{source} train/val identity overlap: {sorted(overlap)[:3]}")
        coverage[source] = counts
        contract_hashes[source] = _json_sha256(
            {
                "spec": spec,
                "index_sha256": digests,
                "artifact_sha256": sorted(artifacts),
                "coverage": counts,
            }
        )
    return coverage, contract_hashes


def _validate_dataset_probe_v8(
    checks: _Checks, config: dict[str, Any]
) -> dict[str, Any]:
    """Construct all five sources without importing the optional eval package."""

    if checks.mode != "full":
        return {}
    train_cfg = config.get("train") or {}
    try:
        global_batch = (
            int(train_cfg["batch_size_per_gpu"])
            * int(train_cfg["gpus_per_node"])
            * int(train_cfg["num_nodes"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        checks.errors.append(f"dataset_probe global batch is invalid: {exc}")
        return {
            "global_batch": None,
            "train_source_lengths": {},
            "validation_source_lengths": {},
            "source_lengths": {},
            "samples": {},
        }
    checks.expect(global_batch > 0, "dataset_probe global batch must be positive")
    report: dict[str, Any] = {
        "global_batch": global_batch,
        "train_source_lengths": {},
        "validation_source_lengths": {},
        "source_lengths": {},
        "samples": {},
    }
    try:
        train, validation = build_datasets(config)
        for split_name, mixed in (("train", train), ("validation", validation)):
            if not hasattr(mixed, "source_names") or not hasattr(
                mixed, "datasets"
            ):
                raise RuntimeError(
                    f"expected formal V8 mixed {split_name} dataset"
                )
        train_sources = {
            str(name): dataset
            for name, dataset in zip(
                train.source_names, train.datasets, strict=True
            )
        }
        validation_sources = {
            str(name): dataset
            for name, dataset in zip(
                validation.source_names, validation.datasets, strict=True
            )
        }
        train_lengths = {
            name: len(dataset) for name, dataset in train_sources.items()
        }
        validation_lengths = {
            name: len(dataset) for name, dataset in validation_sources.items()
        }
    except Exception as exc:
        checks.errors.append(
            "formal dataset construction failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return report

    report["train_source_lengths"] = {
        str(name): int(length) for name, length in train_lengths.items()
    }
    report["validation_source_lengths"] = {
        str(name): int(length) for name, length in validation_lengths.items()
    }
    # Keep the legacy alias for existing report consumers.
    report["source_lengths"] = dict(report["validation_source_lengths"])
    checks.equal(
        set(train_sources), set(EXPECTED_SOURCES),
        "dataset_probe.train.source_names",
    )
    checks.equal(
        set(validation_sources), set(EXPECTED_SOURCES),
        "dataset_probe.validation.source_names",
    )
    for split_name, lengths in (
        ("train", train_lengths),
        ("validation", validation_lengths),
    ):
        for name in EXPECTED_SOURCES:
            length = int(lengths.get(name, 0) or 0)
            checks.expect(
                length >= global_batch,
                f"dataset_probe.{name} {split_name} has {length} samples; "
                f"requires global batch {global_batch}",
            )
    required = {
        "s_in": None,
        "s_tgt_codec": (8, 64, 384),
        "action_tgt": (8, 7),
        "action_tgt_norm": (8, 6),
        "c": (2048,),
        "rgb_tgt": None,
        "depth_tgt": None,
        "point_tgt": None,
        "pose_geom_tgt": None,
    }
    unified_action = bool(
        (config.get("data") or {}).get("v8_dual_rate_action_enabled", False)
    )
    proprio_enabled = bool(
        (config.get("data") or {}).get("v8_proprio_enabled", False)
    )
    if unified_action:
        required.update(
            {
                "v8_dynamics_action_cond": (8, DYNAMICS_ACTION_DIM),
                "policy_action_tgt": (POLICY_HORIZON, 7),
                "policy_action_tgt_norm": (POLICY_HORIZON, 6),
                "policy_action_valid_mask": (POLICY_HORIZON,),
                "policy_action_coarse_tgt": (
                    POLICY_HORIZON // 4,
                    7,
                ),
                "policy_action_coarse_tgt_norm": (
                    POLICY_HORIZON // 4,
                    6,
                ),
                "policy_action_coarse_valid_mask": (POLICY_HORIZON // 4,),
                "action_history": (POLICY_HISTORY_LEN, POLICY_HISTORY_DIM),
            }
        )
    if proprio_enabled:
        required.update(
            {
                "lowdim_state": (V8_PROPRIO_DIM,),
                "policy_proprio_raw": (V8_PROPRIO_DIM,),
                "embodiment_id": (),
            }
        )
    for name in EXPECTED_SOURCES:
        dataset = validation_sources.get(name)
        length = int(validation_lengths.get(name, 0) or 0)
        checks.expect(length > 0, f"dataset_probe.{name} is empty")
        if dataset is None or length <= 0:
            continue
        try:
            sample = dataset[0]
        except Exception as exc:
            checks.errors.append(
                f"dataset_probe.{name}[0] failed: {type(exc).__name__}: {exc}"
            )
            continue
        checks.expect(
            isinstance(sample, dict),
            f"dataset_probe.{name}[0] is not a mapping",
        )
        if not isinstance(sample, dict):
            continue
        sample_report: dict[str, Any] = {"keys": sorted(sample)}
        tensors: dict[str, Any] = {}
        for key, expected_shape in required.items():
            value = sample.get(key)
            checks.expect(
                value is not None, f"dataset_probe.{name}.{key} is missing"
            )
            if value is None:
                continue
            try:
                tensor = torch.as_tensor(value)
            except (TypeError, ValueError) as exc:
                checks.errors.append(
                    f"dataset_probe.{name}.{key} is not tensor-like: {exc}"
                )
                continue
            shape = tuple(int(dim) for dim in tensor.shape)
            tensors[key] = {"shape": list(shape), "dtype": str(tensor.dtype)}
            if expected_shape is not None:
                checks.equal(
                    shape,
                    expected_shape,
                    f"dataset_probe.{name}.{key}.shape",
                )
            checks.expect(
                tensor.numel() > 0, f"dataset_probe.{name}.{key} is empty"
            )
            if tensor.is_floating_point():
                checks.expect(
                    bool(torch.isfinite(tensor).all().item()),
                    f"dataset_probe.{name}.{key} contains nonfinite values",
                )
        sample_report["tensors"] = tensors
        if proprio_enabled:
            checks.equal(
                sample.get("policy_proprio_anchor"),
                V8_PROPRIO_ANCHOR,
                f"dataset_probe.{name}.policy_proprio_anchor",
            )
            checks.expect(
                bool(sample.get("policy_proprio_stats_key")),
                f"dataset_probe.{name}.policy_proprio_stats_key is missing",
            )
            embodiment_id = int(torch.as_tensor(sample["embodiment_id"]).item())
            checks.expect(
                embodiment_id in set(V8_EMBODIMENT_VOCAB.values()),
                f"dataset_probe.{name}.embodiment_id is invalid: {embodiment_id}",
            )
            normalized = torch.as_tensor(sample["lowdim_state"]).float()
            raw = torch.as_tensor(sample["policy_proprio_raw"]).float()
            checks.expect(
                bool(torch.isfinite(normalized).all())
                and bool(torch.isfinite(raw).all()),
                f"dataset_probe.{name}.policy_proprio is non-finite",
            )
            proprio_frame = int(
                torch.as_tensor(sample["policy_proprio_frame_index"]).item()
            )
            action_frames = torch.as_tensor(sample["action_frame_indices"])
            checks.expect(
                action_frames.ndim == 1 and len(action_frames) >= 1,
                f"dataset_probe.{name}.action_frame_indices is invalid",
            )
            if action_frames.ndim == 1 and len(action_frames) >= 1:
                checks.equal(
                    proprio_frame,
                    int(action_frames[0].item()),
                    f"dataset_probe.{name}.proprio/action anchor frame",
                )
        if unified_action:
            checks.equal(
                sample.get("v8_action_contract_version"),
                V8_DUAL_RATE_ACTION_SCHEMA,
                f"dataset_probe.{name}.v8_action_contract_version",
            )
            checks.equal(
                sample.get("v8_action_history_schema"),
                V8_POLICY_HISTORY_SCHEMA,
                f"dataset_probe.{name}.v8_action_history_schema",
            )
            checks.expect(
                bool(sample.get("v8_action_stats_key")),
                f"dataset_probe.{name}.v8_action_stats_key is missing",
            )
            fine_mask = torch.as_tensor(sample["policy_action_valid_mask"]).bool()
            coarse_mask = torch.as_tensor(
                sample["policy_action_coarse_valid_mask"]
            ).bool()
            dynamics = torch.as_tensor(sample["v8_dynamics_action_cond"]).float()
            checks.expect(
                bool(coarse_mask.all()),
                f"dataset_probe.{name} lacks factual 5 Hz composition labels",
            )
            if name.startswith("robocasa_"):
                checks.expect(
                    bool(fine_mask.all()),
                    f"dataset_probe.{name} lacks real 20 Hz labels",
                )
                checks.expect(
                    bool((dynamics[:, 28:32] == 1.0).all()),
                    f"dataset_probe.{name} has invalid 20 Hz substep mask",
                )
                checks.expect(
                    bool(torch.allclose(
                        dynamics[:, 32:36],
                        torch.full_like(dynamics[:, 32:36], 0.05),
                        rtol=0.0,
                        atol=1.0e-8,
                    )),
                    f"dataset_probe.{name} has invalid 20 Hz dt",
                )
            else:
                checks.expect(
                    not bool(fine_mask.any()),
                    f"dataset_probe.{name} fabricated 20 Hz labels",
                )
                checks.expect(
                    bool((dynamics[:, 28] == 1.0).all())
                    and not bool((dynamics[:, 29:32] != 0.0).any()),
                    f"dataset_probe.{name} has invalid coarse-only substep mask",
                )
                checks.expect(
                    bool(torch.allclose(
                        dynamics[:, 32],
                        torch.full_like(dynamics[:, 32], 0.2),
                        rtol=0.0,
                        atol=1.0e-8,
                    ))
                    and not bool((dynamics[:, 33:36] != 0.0).any()),
                    f"dataset_probe.{name} has invalid coarse-only dt",
                )
        report["samples"][name] = sample_report
    return report


def validate_preflight(
    config: dict[str, Any],
    mode: str = "full",
    *,
    verify_training_assets: bool = True,
    verify_local_resources: bool = True,
    exact_resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"static", "full"}:
        raise ValueError(f"unsupported mode: {mode}")
    checks = _Checks(mode)
    sources = _validate_contract_and_objective(checks, config)
    runtime_dependencies = _validate_runtime_dependencies(checks, config)
    asset_report = (
        _validate_training_assets(checks, config, sources)
        if verify_training_assets
        else {}
    )
    coverage, contract_hashes = _validate_causal_indices(checks, config)
    dataset_probe = (
        _validate_dataset_probe_v8(checks, config)
        if verify_training_assets
        else {}
    )
    health = (
        _validate_local_resources(
            checks, config, exact_resume_checkpoint=exact_resume_checkpoint
        )
        if verify_local_resources
        else {}
    )
    report = {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "passed": not checks.errors,
        "launch_ready": mode == "full" and not checks.errors and not checks.blockers,
        "errors": checks.errors,
        "warnings": [],
        "blockers": checks.blockers,
        "resolved_config_sha256": resolved_config_sha256(config),
        "run_lineage": (config.get("train") or {}).get("run_lineage"),
        "source_coverage": coverage,
        "cache_contract_hashes": contract_hashes,
        "runtime_dependencies": runtime_dependencies,
        "action_objective": {
            "contract_schema": (config.get("contract") or {}).get("schema"),
            "v8_dual_rate_action_enabled": bool(
                (config.get("train") or {}).get(
                    "v8_dual_rate_action_enabled", False
                )
            ),
            "serving_owner": (config.get("train") or {}).get(
                "direct_policy_head"
            ),
            "gripper_owner": (config.get("train") or {}).get(
                "direct_policy_grip_owner"
            ),
            "direct_policy_weight": float(
                (config.get("train") or {}).get("direct_policy_weight", 0.0)
            ),
            "policy_flow_weight": float(
                (config.get("train") or {}).get("policy_flow_weight", 0.0)
            ),
            "native_action_no_teacher_weight": float(
                (config.get("train") or {}).get(
                    "native_action_no_teacher_weight", 0.0
                )
            ),
            "native_future_no_teacher_weight": float(
                (config.get("train") or {}).get(
                    "native_future_no_teacher_weight", 0.0
                )
            ),
            "native_action_start_step": int(
                (config.get("train") or {}).get(
                    "native_action_no_teacher_start_step", 0
                )
            ),
            "native_action_every": int(
                (config.get("train") or {}).get(
                    "native_action_no_teacher_every", 1
                )
            ),
            "factual_sources": list(EXPECTED_SOURCES),
        },
        "verified_artifacts": checks.verified_artifacts,
        "training_assets": asset_report,
        "dataset_probe": dataset_probe,
        "health": health,
        "runtime": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in RUNTIME_FILES
        },
    }
    if checks.errors:
        raise CausalDualViewPreflightError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "full"), default="full")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--exact-resume-checkpoint", type=Path)
    args = parser.parse_args()
    try:
        report = validate_preflight(
            load_config(args.config),
            mode=args.mode,
            exact_resume_checkpoint=args.exact_resume_checkpoint,
        )
        exit_code = 0
    except (OSError, ValueError, yaml.YAMLError, CausalDualViewPreflightError) as exc:
        if isinstance(exc, CausalDualViewPreflightError):
            report = exc.report
        else:
            report = {
                "schema": REPORT_SCHEMA,
                "mode": args.mode,
                "passed": False,
                "launch_ready": False,
                "errors": [str(exc)],
                "warnings": [],
                "blockers": [],
            }
        exit_code = 1
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text, end="")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
