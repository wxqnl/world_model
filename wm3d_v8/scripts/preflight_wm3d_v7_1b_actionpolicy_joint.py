#!/usr/bin/env python3
"""Fail-closed preflight for the repaired WM3D-V7 1B joint pretraining run.

This is deliberately separate from the historical two-node stage-0 preflight.
The repaired run is a fresh 3x8-GPU lineage that jointly trains the native 3D
world model and an executable action policy.  The deterministic base head owns
pose serving, the delta-composed head owns gripper serving, and a six-DoF flow
head is an auxiliary pose objective only.

``--mode static`` checks the resolved contract and reports not-yet-produced
canary evidence as a blocker.  ``--mode full`` additionally verifies all pinned
data artifacts, the canary receipt, an empty output lineage, local GPU/ECC
health, and disk headroom.  This script never starts training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preflight_wm3d_v7_stage0_actiondynamics import (  # noqa: E402
    CANONICAL_LAYOUT,
    CANONICAL_VERSION,
    EXPECTED_OXE_ACTION_KINDS,
    EXPECTED_OXE_DATASETS,
    EXPECTED_OXE_OFFSETS,
    EXPECTED_ROTATION_CONVERSIONS,
    EXPECTED_SOURCES,
    FORBIDDEN_DATASETS,
    RUNTIME_CANONICAL_ACTION_MODULE_PATH,
    RUNTIME_TRAIN_MODULE_PATH,
    _Checks,
    _scalar,
    _validate_action_cache_split,
    _validate_canonical_gate,
    _validate_robocasa_index,
    _validate_stats,
    load_config,
    resolved_config_sha256,
    sha256_file,
)
from wm3d_v3.training.train import validate_action_pretraining_preflight  # noqa: E402


SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_preflight_report_v3"
FORMAL_SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_formal_v3"
FORMAL_PREFLIGHT = "scripts/preflight_wm3d_v7_1b_actionpolicy_joint.py"
FORMAL_NODE_LAUNCHER = (
    PROJECT_ROOT / "scripts/launch_wm3d_v7_1b_actionpolicy_joint_formal100k_node_v3.sh"
)
FORMAL_DISTRIBUTED_AUDIT_SMOKE = (
    PROJECT_ROOT / "scripts/smoke_wm3d_v7_1b_actionpolicy_distributed_audit_v3.py"
)
FORMAL_ORCHESTRATOR = (
    PROJECT_ROOT / "scripts/start_wm3d_v7_1b_actionpolicy_joint_formal100k_3node24_v3.sh"
)
FORMAL_IB_ALLREDUCE_SMOKE = (
    PROJECT_ROOT / "scripts/smoke_wm3d_v7_1b_actionpolicy_ib_allreduce_v1.py"
)
FORMAL_IB_TRANSPORT_MATRIX_RUNNER = (
    PROJECT_ROOT / "scripts/run_wm3d_v7_1b_actionpolicy_ib_transport_matrix_v1.sh"
)
FORMAL_IB_FULL_AUDIT_RUNNER = (
    PROJECT_ROOT / "scripts/run_wm3d_v7_1b_actionpolicy_ib_full_audit_v1.sh"
)
FORMAL_DISTRIBUTED_AUDIT_RECEIPT = (
    PROJECT_ROOT
    / "audits/actionrepair_1b_20260807/formal_transport_audit_smoke_v6_ib_receipt.json"
)
FORMAL_DISTRIBUTED_AUDIT_RECEIPT_SHA256 = (
    "36d2c3390fe572441b01f20ca89d32a4bd7fe847c58b617106199a24daac679f"
)
FORMAL_DISTRIBUTED_AUDIT_RECEIPT_SCHEMA = (
    "wm3d_v7_1b_formal_distributed_audit_smoke_receipt_v3"
)
CANARY_SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_canary_v3"
CANARY_GATE_SCHEMA = "wm3d_v7_1b_native_actionpolicy_joint_canary_gate_v3"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MIN_DATA_FREE_BYTES = 200_000_000_000
REQUIRED_NCCL_EXPORTS = (
    "NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10",
    "NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8",
    "export NCCL_IB_DISABLE=0",
    "export NCCL_NET=IB",
    'export NCCL_IB_HCA="${NCCL_IB_HCA_ALLOWLIST}"',
    "export NCCL_NET_GDR_LEVEL=2",
    "export NCCL_SOCKET_IFNAME=bond0.1411",
    "export NCCL_SOCKET_FAMILY=AF_INET",
    "export GLOO_SOCKET_IFNAME=bond0.1411",
)
FORBIDDEN_NCCL_EXPORTS = (
    "export NCCL_IB_DISABLE=1",
    "export NCCL_NET=Socket",
    "export NCCL_IB_HCA=^mlx5_bond_0",
    "export NCCL_SOCKET_NTHREADS=",
    "export NCCL_NSOCKS_PERTHREAD=",
)


class ActionPreflightError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("; ".join(report["errors"]))


def _expect_positive(checks: _Checks, value: Any, label: str) -> None:
    try:
        passed = math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        passed = False
    checks.expect(passed, f"{label} must be finite and positive, got {value!r}")


def _validate_launcher_contract(checks: _Checks) -> None:
    checks.expect(FORMAL_NODE_LAUNCHER.is_file(), f"formal launcher is missing: {FORMAL_NODE_LAUNCHER}")
    if not FORMAL_NODE_LAUNCHER.is_file():
        return
    launcher_text = FORMAL_NODE_LAUNCHER.read_text()
    for export_line in REQUIRED_NCCL_EXPORTS:
        checks.expect(
            export_line in launcher_text,
            f"formal launcher is missing required distributed transport contract: {export_line}",
        )
    for export_line in FORBIDDEN_NCCL_EXPORTS:
        checks.expect(
            export_line not in launcher_text,
            f"formal launcher retains forbidden distributed transport contract: {export_line}",
        )


def _validate_distributed_transport_receipt(checks: _Checks) -> dict[str, Any] | None:
    receipt_path = checks.pinned_file(
        FORMAL_DISTRIBUTED_AUDIT_RECEIPT,
        FORMAL_DISTRIBUTED_AUDIT_RECEIPT_SHA256,
        "distributed.transport_audit_smoke_receipt",
    )
    if checks.mode != "full" or receipt_path is None:
        return None
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"distributed transport receipt is unreadable: {exc}")
        return None
    checks.equal(
        receipt.get("schema"),
        FORMAL_DISTRIBUTED_AUDIT_RECEIPT_SCHEMA,
        "distributed.transport_receipt.schema",
    )
    checks.equal(receipt.get("passed"), True, "distributed.transport_receipt.passed")
    checks.equal(receipt.get("world_size"), 24, "distributed.transport_receipt.world_size")
    transport = receipt.get("transport") or {}
    checks.equal(
        transport.get("eager_before_dataset_io"),
        True,
        "distributed.transport_receipt.eager_before_dataset_io",
    )
    checks.equal(transport.get("ib_disabled"), False, "distributed.transport_receipt.ib_disabled")
    checks.equal(transport.get("net"), "IB", "distributed.transport_receipt.net")
    checks.equal(
        transport.get("inter_node_network"),
        "RDMA",
        "distributed.transport_receipt.inter_node_network",
    )
    checks.equal(
        transport.get("net_gdr_level"),
        2,
        "distributed.transport_receipt.net_gdr_level",
    )
    checks.equal(
        transport.get("socket_ifname"),
        "bond0.1411",
        "distributed.transport_receipt.socket_ifname",
    )
    checks.equal(
        transport.get("socket_family"),
        "AF_INET",
        "distributed.transport_receipt.socket_family",
    )
    checks.equal(
        transport.get("hca_by_node_rank"),
        {
            "0": "mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10",
            "1": "mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10",
            "2": "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8",
        },
        "distributed.transport_receipt.hca_by_node_rank",
    )
    runtime = receipt.get("runtime") or {}
    runtime_closure = (
        ("train_sha256", RUNTIME_TRAIN_MODULE_PATH),
        ("formal_launcher_sha256", FORMAL_NODE_LAUNCHER),
        ("formal_orchestrator_sha256", FORMAL_ORCHESTRATOR),
        ("full_audit_smoke_sha256", FORMAL_DISTRIBUTED_AUDIT_SMOKE),
        ("allreduce_smoke_sha256", FORMAL_IB_ALLREDUCE_SMOKE),
        ("transport_matrix_runner_sha256", FORMAL_IB_TRANSPORT_MATRIX_RUNNER),
        ("full_audit_runner_sha256", FORMAL_IB_FULL_AUDIT_RUNNER),
    )
    for key, path in runtime_closure:
        checks.equal(
            runtime.get(key),
            sha256_file(path),
            f"distributed.transport_receipt.{key}",
        )
    audit = receipt.get("global_mixed_source_audit") or {}
    checks.equal(audit.get("enabled"), True, "distributed.transport_receipt.audit.enabled")
    checks.equal(
        audit.get("mode"),
        "global-identical",
        "distributed.transport_receipt.audit.mode",
    )
    checks.equal(audit.get("world_size"), 24, "distributed.transport_receipt.audit.world_size")
    matrix = receipt.get("transport_matrix") or {}
    checks.expect(
        int(matrix.get("ib_minimal_full24_passes", 0) or 0) >= 3,
        "distributed transport receipt requires at least three IB 24-rank passes",
    )
    checks.equal(
        matrix.get("all_rank_successes_each_pass"),
        24,
        "distributed.transport_receipt.matrix.all_rank_successes_each_pass",
    )
    checks.equal(
        matrix.get("socket_fallback_occurrences"),
        0,
        "distributed.transport_receipt.matrix.socket_fallback_occurrences",
    )
    checks.equal(
        matrix.get("ib_full_five_source_audit_passed"),
        True,
        "distributed.transport_receipt.matrix.full_five_source_audit",
    )
    passes = matrix.get("passes") or []
    checks.equal(len(passes), 3, "distributed.transport_receipt.matrix.pass_count")
    for index, item in enumerate(passes):
        checks.equal(item.get("passed"), True, f"distributed.transport_receipt.pass[{index}].passed")
        checks.equal(
            item.get("all_rank_successes"),
            24,
            f"distributed.transport_receipt.pass[{index}].all_rank_successes",
        )
        checks.equal(
            item.get("socket_fallback_count"),
            0,
            f"distributed.transport_receipt.pass[{index}].socket_fallback_count",
        )
    logs = receipt.get("logs") or []
    checks.equal(len(logs), 12, "distributed.transport_receipt.log_count")
    checks.equal(
        sum(int(item.get("success_token_count", 0) or 0) for item in logs),
        72,
        "distributed.transport_receipt.allreduce_success_token_count",
    )
    for index, item in enumerate(logs):
        checks.equal(
            item.get("ib_selection_count"),
            8,
            f"distributed.transport_receipt.log[{index}].ib_selection_count",
        )
        checks.equal(
            item.get("socket_selection_count"),
            0,
            f"distributed.transport_receipt.log[{index}].socket_selection_count",
        )
        checks.equal(
            item.get("error_marker_count"),
            0,
            f"distributed.transport_receipt.log[{index}].error_marker_count",
        )
        raw_path = item.get("path")
        raw_sha = item.get("sha256")
        if isinstance(raw_path, str) and isinstance(raw_sha, str):
            checks.pinned_file(
                Path(raw_path),
                raw_sha,
                f"distributed.transport_receipt.log[{index}]",
            )
        else:
            checks.errors.append(
                f"distributed transport receipt log[{index}] lacks path/SHA"
            )
    return receipt


def _pinned_canary_receipt(
    checks: _Checks, contract: dict[str, Any]
) -> tuple[Path | None, dict[str, Any] | None]:
    raw_path = contract.get("canary_gate_receipt")
    raw_sha = str(contract.get("canary_gate_receipt_sha256") or "")
    pending = (
        not raw_path
        or str(raw_path).startswith("PENDING_")
        or raw_sha.startswith("PENDING_")
    )
    if pending:
        message = "formal launch awaits an immutable 1K canary gate receipt"
        if checks.mode == "full":
            checks.errors.append(message)
        else:
            checks.blockers.append(message)
        return None, None
    if not HEX64.fullmatch(raw_sha):
        checks.errors.append("contract.canary_gate_receipt_sha256 is not a SHA256")
        return None, None
    receipt_path = checks.pinned_file(raw_path, raw_sha, "canary.gate_receipt")
    if checks.mode != "full" or receipt_path is None:
        return receipt_path, None
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        checks.errors.append(f"canary gate receipt is unreadable: {exc}")
        return receipt_path, None
    checks.equal(receipt.get("schema"), CANARY_GATE_SCHEMA, "canary.gate.schema")
    checks.equal(receipt.get("passed"), True, "canary.gate.passed")
    checks.equal(receipt.get("decision"), "PASS_FORMAL_REPRETRAIN", "canary.gate.decision")
    checks.equal(receipt.get("formal_config_schema"), FORMAL_SCHEMA, "canary.gate.formal_schema")
    checkpoint = receipt.get("checkpoint") or {}
    checks.equal(checkpoint.get("step"), 1000, "canary.gate.checkpoint.step")
    checks.expect(
        HEX64.fullmatch(str(checkpoint.get("sha256") or "")) is not None,
        "canary gate checkpoint SHA256 is missing",
    )
    checks.expect(
        int(checkpoint.get("size_bytes", 0) or 0) >= 15_000_000_000,
        "canary gate checkpoint is below 15GB",
    )
    return receipt_path, receipt


def _validate_data(checks: _Checks, config: dict[str, Any]) -> None:
    contract = config.get("contract") or {}
    data = config.get("data") or {}
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

    checks.equal(data.get("dataset_type"), "v7_mixed", "data.dataset_type")
    checks.equal(data.get("T"), 16, "data.T")
    checks.equal(data.get("k"), 8, "data.k")
    checks.equal(data.get("robocasa_partitions"), ["atomic", "composite", "mg"], "data.robocasa_partitions")
    checks.equal(data.get("require_action_stats"), True, "data.require_action_stats")
    checks.equal(data.get("require_rgb_sidecar"), True, "data.require_rgb_sidecar")
    checks.equal(data.get("require_task_emb"), True, "data.require_task_emb")
    checks.equal(data.get("policy_action_history_len"), 4, "data.policy_action_history_len")
    checks.equal(data.get("policy_action_history_dim"), 7, "data.policy_action_history_dim")

    compact = checks.pinned_file(
        data.get("compact_index"), data.get("compact_index_sha256"), "robocasa.compact_index"
    )
    stats = checks.pinned_file(
        data.get("action_stats"), data.get("action_stats_sha256"), "robocasa.action_stats"
    )
    sidecar_pins = data.get("rgb_sidecar_sha256") or {}
    sidecar_paths = list(data.get("rgb_sidecar_indices") or ())
    checks.equal(set(sidecar_pins), set(sidecar_paths), "robocasa.rgb_sidecar pins")
    for path in sidecar_paths:
        checks.pinned_file(path, sidecar_pins.get(path), f"robocasa.rgb_sidecar:{Path(path).parent.name}")
    _validate_robocasa_index(checks, compact)
    if checks.mode == "full" and stats is not None:
        try:
            with np.load(stats, allow_pickle=False) as archive:
                checks.equal(_scalar(archive, "split"), "train", "robocasa.stats.split")
                checks.equal(
                    _scalar(archive, "index_sha256"),
                    data.get("compact_index_sha256"),
                    "robocasa.stats.index_sha256",
                )
                checks.equal(np.asarray(archive["mean"]).shape, (6,), "robocasa.stats.mean.shape")
                checks.equal(np.asarray(archive["std"]).shape, (6,), "robocasa.stats.std.shape")
        except (OSError, KeyError, ValueError) as exc:
            checks.errors.append(f"robocasa action stats are invalid: {exc}")

    sources = data.get("oxe_sources")
    checks.expect(isinstance(sources, list), "data.oxe_sources must be a list")
    by_name: dict[str, dict[str, Any]] = {}
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict) or not source.get("source_name"):
                checks.errors.append("every OXE source needs source_name")
                continue
            name = str(source["source_name"])
            checks.expect(name not in by_name, f"duplicate OXE source {name}")
            by_name[name] = source
    checks.equal(set(by_name), set(EXPECTED_OXE_DATASETS), "data.oxe_sources names")

    for name, dataset in EXPECTED_OXE_DATASETS.items():
        source = by_name.get(name)
        if source is None:
            continue
        checks.equal(source.get("include_datasets"), [dataset], f"{name}.include_datasets")
        checks.expect(
            not (set(source.get("include_datasets") or ()) & FORBIDDEN_DATASETS),
            f"{name} includes a forbidden source",
        )
        checks.equal(source.get("T"), 16, f"{name}.T")
        checks.equal(source.get("k"), 8, f"{name}.k")
        checks.equal(source.get("load_rgb"), True, f"{name}.load_rgb")
        checks.equal(source.get("load_geom"), True, f"{name}.load_geom")
        checks.equal(source.get("load_geom_extra"), True, f"{name}.load_geom_extra")
        checks.equal(source.get("load_state_tgt"), True, f"{name}.load_state_tgt")
        checks.equal(source.get("require_task_emb"), True, f"{name}.require_task_emb")
        checks.equal(source.get("require_geom_extra"), True, f"{name}.require_geom_extra")
        checks.equal(source.get("require_train_val_disjoint"), True, f"{name}.train_val_disjoint")
        checks.equal(source.get("require_canonical_action_contract"), True, f"{name}.canonical_required")
        checks.equal(source.get("canonical_action_enabled"), True, f"{name}.canonical_enabled")
        checks.equal(source.get("allowed_action_kinds"), [EXPECTED_OXE_ACTION_KINDS[name]], f"{name}.action_kind")
        checks.equal(source.get("default_action_frame_offset"), EXPECTED_OXE_OFFSETS[name], f"{name}.frame_offset")
        checks.equal(source.get("split"), {"mode": "episode", "val_frac": 0.03, "seed": 909}, f"{name}.split")
        adapter = source.get("action_adapter") or {}
        checks.equal(adapter.get("version"), CANONICAL_VERSION, f"{name}.adapter.version")
        checks.equal(adapter.get("rotation_conversion"), EXPECTED_ROTATION_CONVERSIONS[name], f"{name}.adapter.rotation")
        checks.equal(adapter.get("source_frame"), "base", f"{name}.adapter.frame")
        checks.equal(adapter.get("translation_unit"), "meter", f"{name}.adapter.unit")
        checks.equal(adapter.get("gripper_semantics"), "signed_close_positive_continuous", f"{name}.adapter.gripper")

        canonical_stats = source.get("canonical_action_stats_by_source") or {}
        canonical_stats_sha = source.get("canonical_action_stats_sha256_by_source") or {}
        checks.equal(set(canonical_stats), {dataset}, f"{name}.stats keys")
        checks.equal(set(canonical_stats_sha), {dataset}, f"{name}.stats SHA keys")
        checks.pinned_file(source.get("manifest"), source.get("manifest_sha256"), f"{name}.manifest")
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
        audit_gate = checks.pinned_file(
            source.get("action_audit_gate"),
            source.get("action_audit_gate_sha256"),
            f"{name}.action_audit_gate",
        )
        stats_path = checks.pinned_file(
            canonical_stats.get(dataset),
            canonical_stats_sha.get(dataset),
            f"{name}.canonical_stats",
        )
        _validate_canonical_gate(checks, source, audit_gate)
        _validate_action_cache_split(checks, source, cache_manifest)
        _validate_stats(checks, source, stats_path)
        for label in ("cache_root", "window_geom_shard_root"):
            raw = source.get(label)
            if checks.mode == "full":
                checks.expect(bool(raw) and Path(str(raw)).is_dir(), f"{name}.{label} is missing: {raw}")
        index = source.get("window_geom_shard_index")
        index_sha = source.get("window_geom_shard_index_sha256")
        checks.expect(
            isinstance(index_sha, str) and HEX64.fullmatch(index_sha) is not None,
            f"{name}.window_geom_shard_index_sha256 is missing or invalid",
        )
        checks.pinned_file(index, index_sha, f"{name}.window_geom_shard_index")


def _validate_dataset_probe(
    checks: _Checks, config: dict[str, Any]
) -> dict[str, Any]:
    """Construct every formal source and read one real world/action sample.

    Path existence alone did not catch incomplete cross-host cache views in an
    earlier dry run.  A full preflight therefore exercises the same dataset
    factory as evaluation and requires finite native-3D plus action tensors
    from all five sources before a launcher may be authorized.
    """

    if checks.mode != "full":
        return {}
    report: dict[str, Any] = {"source_lengths": {}, "samples": {}}
    try:
        from scripts.eval_v7_stage0_compare import _source_datasets

        sources, lengths = _source_datasets(config)
    except Exception as exc:  # Dataset/cache errors must become preflight evidence.
        checks.errors.append(f"formal dataset construction failed: {type(exc).__name__}: {exc}")
        return report

    report["source_lengths"] = {str(name): int(length) for name, length in lengths.items()}
    checks.equal(set(sources), set(EXPECTED_SOURCES), "dataset_probe.source_names")
    required = {
        "s_in": None,
        "action_tgt": (8, 7),
        "action_tgt_norm": (8, 6),
        "c": (2048,),
        "rgb_tgt": None,
        "depth_tgt": None,
        "point_tgt": None,
        "pose_geom_tgt": None,
    }
    for name in EXPECTED_SOURCES:
        dataset = sources.get(name)
        length = int(lengths.get(name, 0) or 0)
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
        checks.expect(isinstance(sample, dict), f"dataset_probe.{name}[0] is not a mapping")
        if not isinstance(sample, dict):
            continue
        sample_report: dict[str, Any] = {"keys": sorted(sample)}
        tensors: dict[str, Any] = {}
        for key, expected_shape in required.items():
            value = sample.get(key)
            checks.expect(value is not None, f"dataset_probe.{name}.{key} is missing")
            if value is None:
                continue
            try:
                tensor = torch.as_tensor(value)
            except (TypeError, ValueError) as exc:
                checks.errors.append(f"dataset_probe.{name}.{key} is not tensor-like: {exc}")
                continue
            shape = tuple(int(dim) for dim in tensor.shape)
            tensors[key] = {"shape": list(shape), "dtype": str(tensor.dtype)}
            if expected_shape is not None:
                checks.equal(shape, expected_shape, f"dataset_probe.{name}.{key}.shape")
            checks.expect(tensor.numel() > 0, f"dataset_probe.{name}.{key} is empty")
            if tensor.is_floating_point():
                checks.expect(
                    bool(torch.isfinite(tensor).all().item()),
                    f"dataset_probe.{name}.{key} contains nonfinite values",
                )
        sample_report["tensors"] = tensors
        report["samples"][name] = sample_report
    return report


def _validate_objective(checks: _Checks, config: dict[str, Any]) -> int:
    contract = config.get("contract") or {}
    model = config.get("model") or {}
    train = config.get("train") or {}
    loss = config.get("loss") or {}
    optimizer = config.get("optimizer") or {}
    schedule = config.get("lr_schedule") or {}
    sampler = train.get("mixed_batch_sampler") or {}

    checks.equal(contract.get("schema"), FORMAL_SCHEMA, "contract.schema")
    checks.equal(contract.get("profile"), "formal", "contract.profile")
    checks.equal(contract.get("preflight_required"), True, "contract.preflight_required")
    checks.equal(contract.get("preflight_script"), FORMAL_PREFLIGHT, "contract.preflight_script")
    checks.equal(contract.get("canary_required"), True, "contract.canary_required")
    checks.equal(contract.get("auto_promote"), False, "contract.auto_promote")
    checks.equal(contract.get("future_observation_inputs_forbidden"), True, "contract.future_observation_inputs_forbidden")
    checks.equal(contract.get("native_3d_outputs_required"), ["rgb", "depth", "point", "pose"], "contract.native_outputs")
    checks.equal(contract.get("serving_action_owner"), "direct_pose_plus_delta_composed_gripper", "contract.serving_owner")
    checks.equal(contract.get("auxiliary_action_owner"), "pose_only_flow_matching", "contract.aux_owner")
    checks.equal(contract.get("flow_policy_enabled"), True, "contract.flow_policy_enabled")
    checks.equal(contract.get("flow_serving_enabled"), False, "contract.flow_serving_enabled")

    try:
        checks.equal(validate_action_pretraining_preflight(config), True, "runtime.action_preflight")
    except (RuntimeError, TypeError, ValueError) as exc:
        checks.errors.append(f"runtime action pretraining preflight failed: {exc}")

    state = model.get("state") or {}
    action = model.get("action") or {}
    for label, section, hidden, layers in (
        ("state", state, 1600, 18),
        ("action", action, 1280, 14),
    ):
        checks.equal(section.get("T"), 16, f"model.{label}.T")
        checks.equal(section.get("P"), 64, f"model.{label}.P")
        checks.equal(section.get("D"), 2048, f"model.{label}.D")
        checks.equal(section.get("k"), 8, f"model.{label}.k")
        checks.equal(section.get("hidden"), hidden, f"model.{label}.hidden")
        checks.equal(section.get("n_layers"), layers, f"model.{label}.layers")
        checks.equal(section.get("n_heads"), 16, f"model.{label}.heads")
        checks.equal(section.get("action_cond_dim"), 7, f"model.{label}.action_cond_dim")
    checks.equal(model.get("enable_pixel"), True, "model.enable_pixel")
    checks.equal(model.get("enable_context_pixel"), True, "model.enable_context_pixel")
    checks.equal(model.get("enable_geom_extra"), True, "model.enable_geom_extra")
    checks.equal(model.get("enable_action_policy"), True, "model.enable_action_policy")
    checks.equal(model.get("enable_world_prior"), False, "model.enable_world_prior")
    checks.equal(model.get("policy_context_source"), "core_pred", "model.policy_context_source")
    checks.equal(model.get("policy_core_action_cond"), "none", "model.policy_core_action_cond")
    checks.equal(model.get("policy_action_history_len"), 4, "model.policy_action_history_len")
    checks.equal(model.get("policy_action_history_dim"), 7, "model.policy_action_history_dim")
    checks.equal(model.get("policy_enable_grip_delta_head"), True, "model.grip_delta_head")
    checks.equal(model.get("policy_grip_owner"), "delta_composed", "model.grip_owner")
    checks.equal(model.get("policy_grip_delta_use_composed_action_cond"), True, "model.grip_compose")
    checks.equal(model.get("policy_grip_delta_soft_compose_action_cond"), True, "model.grip_soft_compose")
    checks.equal(model.get("policy_grip_delta_straight_through_action_cond"), True, "model.grip_straight_through")
    checks.equal(model.get("policy_enable_flow_head"), True, "model.flow_head")
    checks.equal(model.get("policy_flow_use_as_policy"), False, "model.flow_serving")
    checks.equal(model.get("policy_flow_action_dim"), 6, "model.flow_action_dim")
    checks.equal(model.get("token_codec_frozen"), True, "model.token_codec_frozen")
    checks.pinned_file(
        model.get("token_codec_checkpoint"),
        model.get("token_codec_checkpoint_sha256"),
        "model.token_codec",
    )

    checks.equal(train.get("joint_native_action_pretraining"), True, "train.joint_native_action_pretraining")
    checks.equal(train.get("fresh_initialization_required"), True, "train.fresh_initialization_required")
    checks.equal(train.get("fresh_init_required"), True, "train.fresh_init_required")
    checks.equal(train.get("forbid_warm_start"), True, "train.forbid_warm_start")
    checks.equal(train.get("stage_transition"), None, "train.stage_transition")
    checks.equal(train.get("resume_checkpoint"), None, "train.resume_checkpoint")
    checks.equal(train.get("pretrained_world_checkpoint"), None, "train.pretrained_world_checkpoint")
    checks.equal(train.get("model_initialization"), "random_world_weights_with_frozen_pinned_codec", "train.model_initialization")
    checks.equal(train.get("trainable_prefixes"), [], "train.trainable_prefixes")
    checks.equal(train.get("freeze_prefixes"), [], "train.freeze_prefixes")
    checks.equal(train.get("direct_policy_head"), "base", "train.direct_policy_head")
    checks.equal(train.get("direct_policy_grip_owner"), "delta_composed", "train.grip_owner")
    checks.equal(train.get("direct_policy_require_action_prev_grip"), True, "train.require_prev_grip")
    checks.equal(train.get("direct_policy_grip_partition_contract"), True, "train.grip_partition_contract")
    _expect_positive(checks, train.get("direct_policy_weight"), "train.direct_policy_weight")
    _expect_positive(checks, train.get("direct_policy_pose_weight"), "train.direct_policy_pose_weight")
    _expect_positive(checks, train.get("direct_policy_grip_delta_ce_weight"), "train.grip_delta_ce")
    _expect_positive(checks, train.get("direct_policy_grip_delta_natural_ce_weight"), "train.grip_delta_natural_ce")
    _expect_positive(checks, train.get("native_action_no_teacher_weight"), "train.native_action_no_teacher_weight")
    _expect_positive(checks, train.get("native_future_no_teacher_weight"), "train.native_future_no_teacher_weight")
    checks.equal(train.get("native_action_no_teacher_start_step"), 0, "train.native_action_start")
    checks.equal(train.get("native_action_no_teacher_every"), 1, "train.native_action_every")
    checks.approx(train.get("policy_flow_weight"), 0.25, "train.policy_flow_weight")
    checks.equal(train.get("policy_flow_action_dim"), 6, "train.policy_flow_action_dim")
    checks.approx(train.get("policy_flow_pose_weight"), 1.0, "train.policy_flow_pose_weight")
    checks.approx(train.get("policy_flow_grip_weight"), 0.0, "train.policy_flow_grip_weight")
    factual = train.get("factual_action_conditioning") or {}
    checks.equal(factual.get("enabled"), True, "train.factual.enabled")
    checks.equal(factual.get("start_step"), 0, "train.factual.start")
    checks.equal(factual.get("detach_action_condition"), False, "train.factual.detach")
    checks.equal(set(factual.get("required_sources") or ()), set(EXPECTED_SOURCES), "train.factual.sources")
    checks.equal(set(train.get("action_aux_sources") or ()), set(EXPECTED_SOURCES), "train.action_aux_sources")

    checks.equal(sampler.get("enabled"), True, "sampler.enabled")
    checks.equal(sampler.get("source_cycle_counts_exact"), EXPECTED_SOURCES, "sampler.source_mix")
    checks.equal(sampler.get("cycle_optimizer_steps"), 100, "sampler.cycle")
    checks.equal(sampler.get("synchronized_across_ranks"), True, "sampler.synchronized")
    checks.equal(sampler.get("accumulation_group_same_source"), True, "sampler.accumulation_group_same_source")
    checks.equal(set(sampler.get("forbidden_sources") or ()), FORBIDDEN_DATASETS, "sampler.forbidden")

    checks.equal(train.get("num_nodes"), 3, "train.num_nodes")
    checks.equal(train.get("gpus_per_node"), 8, "train.gpus_per_node")
    checks.equal(train.get("batch_size_per_gpu"), 2, "train.batch_size_per_gpu")
    checks.equal(train.get("gradient_accumulation_steps"), 2, "train.grad_accum")
    computed_global = (
        int(train.get("num_nodes", 0))
        * int(train.get("gpus_per_node", 0))
        * int(train.get("batch_size_per_gpu", 0))
        * int(train.get("gradient_accumulation_steps", 0))
    )
    checks.equal(computed_global, 96, "computed_global_batch")
    checks.equal(train.get("effective_global_batch"), 96, "train.effective_global_batch")
    checks.equal(train.get("max_steps"), 100000, "train.max_steps")
    checks.equal(train.get("warmup_steps"), 2000, "train.warmup_steps")
    checks.equal(train.get("ckpt_every_steps"), 5000, "train.ckpt_every_steps")
    checks.equal(train.get("precision"), "bf16", "train.precision")
    checks.equal(train.get("abort_on_nonfinite"), True, "train.abort_on_nonfinite")

    checks.equal(optimizer.get("type"), "adamw", "optimizer.type")
    checks.approx(optimizer.get("peak_lr"), 1.0e-5, "optimizer.peak_lr")
    checks.approx(optimizer.get("weight_decay"), 0.02, "optimizer.weight_decay")
    checks.approx(train.get("lr_multipliers", {}).get("action_policy"), 3.0, "train.action_policy_lr_multiplier")
    checks.equal(schedule.get("type"), "wsd", "lr_schedule.type")
    checks.approx(schedule.get("peak_lr"), 1.0e-5, "lr_schedule.peak_lr")
    checks.equal(schedule.get("warmup_steps"), 2000, "lr_schedule.warmup")
    checks.approx(schedule.get("stable_frac"), 0.78, "lr_schedule.stable_frac")
    checks.approx(schedule.get("decay_frac"), 0.20, "lr_schedule.decay_frac")

    checks.approx(loss.get("action"), 0.0, "loss.legacy_teacher_action")
    for key in ("rgb_l1", "rgb_lpips", "geom_depth", "geom_point", "geom_pose"):
        _expect_positive(checks, loss.get(key), f"loss.{key}")
    checks.equal(train.get("enable_hunyuan_latent_loss"), False, "train.enable_hunyuan_latent_loss")
    checks.approx(train.get("hunyuan_latent_weight"), 0.0, "train.hunyuan_latent_weight")
    checks.approx(train.get("prior_hunyuan_latent_weight"), 0.0, "train.prior_hunyuan_latent_weight")
    return computed_global


def _validate_local_resources(checks: _Checks, config: dict[str, Any]) -> dict[str, Any]:
    health: dict[str, Any] = {}
    if checks.mode != "full":
        return health
    try:
        usage = shutil.disk_usage("/data")
        health["data_free_bytes"] = usage.free
        checks.expect(
            usage.free >= MIN_DATA_FREE_BYTES,
            f"/data free space is below {MIN_DATA_FREE_BYTES}: {usage.free}",
        )
    except OSError as exc:
        checks.errors.append(f"cannot inspect /data free space: {exc}")

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        health["gpu_ecc_rows"] = rows
        checks.equal(len(rows), 8, "local GPU count")
        for row in rows:
            fields = [part.strip() for part in row.split(",")]
            checks.expect(
                len(fields) == 3 and fields[1:] == ["0", "0"],
                f"uncorrected ECC is nonzero or unreadable: {row}",
            )
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        app_rows = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
        health["compute_apps"] = app_rows
        checks.expect(not app_rows, f"GPU compute applications are already active: {app_rows[:4]}")
    except (OSError, subprocess.CalledProcessError) as exc:
        checks.errors.append(f"cannot inspect GPU health: {exc}")

    out = config.get("out") or {}
    root = Path(str(out.get("root") or ""))
    checks.expect(str(root) not in {"", "."}, "out.root is missing")
    checks.equal(out.get("require_empty_checkpoint_dir"), True, "out.require_empty_checkpoint_dir")
    ckpt_dir = root / str(out.get("ckpt_dir", "ckpt"))
    existing = []
    if ckpt_dir.exists():
        existing = sorted(ckpt_dir.glob("step_*.pt")) + sorted(ckpt_dir.glob("latest.pt"))
    checks.expect(not existing, f"formal output checkpoint lineage is not empty: {existing[:3]}")
    health["checkpoint_dir"] = str(ckpt_dir)
    health["checkpoint_files"] = [str(path) for path in existing]
    return health


def validate_preflight(config: dict[str, Any], mode: str = "full") -> dict[str, Any]:
    if mode not in {"static", "full"}:
        raise ValueError(f"unsupported mode: {mode}")
    checks = _Checks(mode)
    global_batch = _validate_objective(checks, config)
    _validate_data(checks, config)
    dataset_probe = _validate_dataset_probe(checks, config)
    _validate_launcher_contract(checks)
    transport_receipt = _validate_distributed_transport_receipt(checks)
    _, canary_receipt = _pinned_canary_receipt(checks, config.get("contract") or {})
    health = _validate_local_resources(checks, config)
    report = {
        "schema": SCHEMA,
        "mode": mode,
        "passed": not checks.errors,
        "launch_ready": mode == "full" and not checks.errors and not checks.blockers,
        "errors": checks.errors,
        "blockers": checks.blockers,
        "resolved_config_sha256": resolved_config_sha256(config),
        "global_batch": global_batch,
        "source_cycle_counts": EXPECTED_SOURCES,
        "dataset_probe": dataset_probe,
        "initialization_mode": "fresh_random_world_with_frozen_pinned_codec",
        "serving_action_owner": "direct_pose_plus_delta_composed_gripper",
        "auxiliary_action_owner": "pose_only_flow_matching",
        "distributed_transport_gate": transport_receipt,
        "verified_artifacts": checks.verified_artifacts,
        "canary_gate": canary_receipt,
        "health": health,
        "runtime": {
            "project_root": str(PROJECT_ROOT),
            "train_path": str(RUNTIME_TRAIN_MODULE_PATH),
            "train_sha256": sha256_file(RUNTIME_TRAIN_MODULE_PATH),
            "canonical_action_path": str(RUNTIME_CANONICAL_ACTION_MODULE_PATH),
            "canonical_action_sha256": sha256_file(RUNTIME_CANONICAL_ACTION_MODULE_PATH),
            "action_policy_path": str(PROJECT_ROOT / "wm3d_v3/models/action_policy.py"),
            "action_policy_sha256": sha256_file(PROJECT_ROOT / "wm3d_v3/models/action_policy.py"),
            "node_launcher_path": str(FORMAL_NODE_LAUNCHER),
            "node_launcher_sha256": sha256_file(FORMAL_NODE_LAUNCHER),
            "formal_orchestrator_path": str(FORMAL_ORCHESTRATOR),
            "formal_orchestrator_sha256": sha256_file(FORMAL_ORCHESTRATOR),
            "required_nccl_exports": list(REQUIRED_NCCL_EXPORTS),
            "distributed_audit_smoke_path": str(FORMAL_DISTRIBUTED_AUDIT_SMOKE),
            "distributed_audit_smoke_sha256": sha256_file(FORMAL_DISTRIBUTED_AUDIT_SMOKE),
            "ib_allreduce_smoke_path": str(FORMAL_IB_ALLREDUCE_SMOKE),
            "ib_allreduce_smoke_sha256": sha256_file(FORMAL_IB_ALLREDUCE_SMOKE),
            "ib_transport_matrix_runner_path": str(FORMAL_IB_TRANSPORT_MATRIX_RUNNER),
            "ib_transport_matrix_runner_sha256": sha256_file(
                FORMAL_IB_TRANSPORT_MATRIX_RUNNER
            ),
            "ib_full_audit_runner_path": str(FORMAL_IB_FULL_AUDIT_RUNNER),
            "ib_full_audit_runner_sha256": sha256_file(FORMAL_IB_FULL_AUDIT_RUNNER),
            "distributed_audit_receipt_path": str(FORMAL_DISTRIBUTED_AUDIT_RECEIPT),
            "distributed_audit_receipt_sha256": sha256_file(
                FORMAL_DISTRIBUTED_AUDIT_RECEIPT
            ),
            "preflight_path": str(Path(__file__).resolve()),
            "preflight_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    if checks.errors:
        raise ActionPreflightError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "full"), default="full")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    try:
        report = validate_preflight(load_config(args.config), mode=args.mode)
    except (OSError, ValueError, yaml.YAMLError, ActionPreflightError) as exc:
        if isinstance(exc, ActionPreflightError):
            report = exc.report
        else:
            report = {
                "schema": SCHEMA,
                "mode": args.mode,
                "passed": False,
                "launch_ready": False,
                "errors": [str(exc)],
                "blockers": [],
            }
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(1)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
