#!/usr/bin/env python3
"""Fail-closed preflight for the isolated native V7 Stage1-P pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from wm3d_v3.stage1_planner.dataset import SCHEMA, Stage1BranchDatasetConfig  # noqa: E402
from wm3d_v3.stage1_planner.train import (  # noqa: E402
    DYNAMICS_PREFIXES,
    PLANNER_PREFIXES,
    SERVING_GUARD_PREFIXES,
)
from wm3d_v3.training.train import config_sha256, load_train_config  # noqa: E402


REPORT_SCHEMA = "wm3d_v7_stage1_planner_preflight_v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_ROLES = Stage1BranchDatasetConfig.expected_roles
MIN_DATA_FREE_BYTES = 160_000_000_000
STAGE0_RUN_TOKEN = "wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3"


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.evidence: dict[str, Any] = {}

    def expect(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def pinned(self, path_value: Any, sha_value: Any, label: str) -> Path | None:
        path = Path(str(path_value or ""))
        expected = str(sha_value or "")
        self.expect(bool(HEX64.fullmatch(expected)), f"{label} SHA256 is not pinned")
        self.expect(path.is_file(), f"{label} file is missing: {path}")
        if not path.is_file() or not HEX64.fullmatch(expected):
            return None
        actual = sha256_file(path)
        self.expect(actual == expected, f"{label} SHA256 mismatch: {actual} != {expected}")
        self.evidence[f"{label}_sha256"] = actual
        return path


def _validate_model_contract(checks: Checks, cfg: dict[str, Any]) -> None:
    model = cfg.get("model") or {}
    state = model.get("state") or {}
    action = model.get("action") or {}
    expected = {"T": 16, "P": 64, "D": 2048, "k": 8, "action_cond_dim": 7}
    for name, value in expected.items():
        checks.expect(state.get(name) == value, f"model.state.{name} must remain {value}")
        checks.expect(action.get(name) == value, f"model.action.{name} must remain {value}")
    checks.expect(bool(model.get("enable_geom_extra")), "native depth/point/pose heads are required")
    checks.expect(bool(model.get("enable_action_policy")), "Stage0 direct action policy is required")
    checks.expect(bool(model.get("policy_enable_flow_head")), "pose-flow proposer is required")
    checks.expect(model.get("policy_flow_action_dim") == 6, "flow proposer must remain pose-only 6D")
    checks.expect(model.get("policy_flow_use_as_policy") is False, "flow must never own serving")
    checks.expect(model.get("policy_action_history_len") == 4, "Stage0 H4 action history must remain")
    checks.expect(model.get("policy_grip_owner") == "delta_composed", "gripper owner changed")
    checks.expect(model.get("token_codec_latent_dim") == 384, "pinned V7 token codec changed")


def _validate_index(checks: Checks, index: Path, *, sample_payloads: bool) -> None:
    rows: list[dict[str, Any]] = []
    roots: set[str] = set()
    split_groups: dict[str, set[str]] = {name: set() for name in ("train", "val", "test")}
    counts = {name: 0 for name in split_groups}
    with index.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            root_id = str(row.get("root_id", ""))
            checks.expect(row.get("schema") == SCHEMA, f"index line {line_number}: schema mismatch")
            checks.expect(bool(root_id) and root_id not in roots, f"index line {line_number}: duplicate root")
            roots.add(root_id)
            split = str(row.get("split", ""))
            checks.expect(split in split_groups, f"index line {line_number}: invalid split")
            if split in split_groups:
                counts[split] += 1
                split_groups[split].add(str(row.get("split_group", "")))
            checks.expect(tuple(row.get("branch_roles") or ()) == EXPECTED_ROLES, f"{root_id}: roles")
            checks.expect(row.get("context_frames") == 16, f"{root_id}: T16 contract")
            checks.expect(row.get("future_frames") == 32, f"{root_id}: H32 contract")
            checks.expect(row.get("action_history_len") == 4, f"{root_id}: H4 history contract")
            checks.expect(row.get("factual_index") == 0, f"{root_id}: factual index")
            checks.expect(row.get("direct_index") == 1, f"{root_id}: direct index")
            checks.expect(row.get("same_root_current_runtime_exact") is True, f"{root_id}: exact root")
            checks.expect(row.get("pseudo_outcomes") is False, f"{root_id}: pseudo outcome")
            checks.expect(row.get("future_observation_leakage") is False, f"{root_id}: leakage")
            checks.expect(
                row.get("context_source")
                == "current_pinned_robocasa_runtime_causal_replay",
                f"{root_id}: real causal T16 context",
            )
            checks.expect(
                bool(HEX64.fullmatch(str(row.get("root_context_sha256", "")))),
                f"{root_id}: root-context SHA",
            )
            checks.expect(row.get("all_branch_native_geometry") is True, f"{root_id}: branch geometry")
            checks.expect(row.get("single_vggt_gauge_per_branch") is True, f"{root_id}: VGGT gauge")
            path = Path(str(row.get("path", "")))
            checks.expect(path.is_file(), f"{root_id}: payload missing: {path}")
            rows.append(row)
    checks.expect(bool(rows), "Stage1-P index is empty")
    for split, count in counts.items():
        checks.expect(count > 0, f"Stage1-P split is empty: {split}")
        checks.expect("" not in split_groups[split], f"blank split_group in {split}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_groups[left] & split_groups[right]
        checks.expect(not overlap, f"split_group leakage {left}/{right}: {sorted(overlap)[:4]}")
    checks.evidence["index_rows"] = len(rows)
    checks.evidence["split_counts"] = counts
    if not sample_payloads or not rows:
        return
    samples: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row.get("split") == split]
        samples.extend(selected[:1] + selected[-1:] if len(selected) > 1 else selected)
    required = {
        "anchor_codes",
        "anchor_scale",
        "wrist_codes",
        "wrist_scale",
        "task_emb",
        "root_context_sha256",
        "branch_codes",
        "branch_scales",
        "branch_depth_tgt",
        "branch_depth_conf_tgt",
        "branch_point_tgt",
        "branch_point_conf_tgt",
        "branch_pose_geom_tgt",
        "branch_actions_physical",
        "action_history_physical",
        "branch_rewards",
        "branch_dones",
        "branch_success",
    }
    for row in samples:
        with np.load(row["path"], allow_pickle=False) as archive:
            checks.expect(required <= set(archive.files), f"{row['root_id']}: missing payload keys")
            if not required <= set(archive.files):
                continue
            checks.expect(str(archive["schema"].item()) == SCHEMA, f"{row['root_id']}: payload schema")
            checks.expect(tuple(str(v) for v in archive["branch_roles"].tolist()) == EXPECTED_ROLES, f"{row['root_id']}: payload roles")
            checks.expect(archive["anchor_codes"].shape == (16, 64, 384), f"{row['root_id']}: anchor context")
            checks.expect(archive["wrist_codes"].shape == (16, 64, 384), f"{row['root_id']}: wrist context")
            checks.expect(archive["task_emb"].shape == (2048,), f"{row['root_id']}: task embedding")
            checks.expect(str(archive["root_context_sha256"].item()) == row.get("root_context_sha256"), f"{row['root_id']}: context provenance")
            checks.expect(archive["branch_codes"].shape[:3] == (11, 32, 64), f"{row['root_id']}: branch tokens")
            checks.expect(archive["branch_depth_tgt"].shape == (11, 32, 8, 8), f"{row['root_id']}: depth")
            checks.expect(archive["branch_point_tgt"].shape == (11, 32, 8, 8, 3), f"{row['root_id']}: point")
            checks.expect(archive["branch_pose_geom_tgt"].shape == (11, 32, 9), f"{row['root_id']}: pose")
            checks.expect(archive["branch_actions_physical"].shape == (11, 32, 7), f"{row['root_id']}: actions")
            checks.expect(archive["action_history_physical"].shape == (4, 7), f"{row['root_id']}: history")


def _validate_predecessor(
    checks: Checks,
    phase_cfg: dict[str, Any],
    source_sha: str | None,
) -> None:
    phase = str(phase_cfg.get("phase"))
    predecessor = phase_cfg.get("predecessor_overlay")
    if phase == "dynamics":
        checks.expect(predecessor in (None, ""), "dynamics phase must start without a Stage1 overlay")
        return
    expected_phase = {"planner": "dynamics", "joint": "planner"}.get(phase)
    checks.expect(expected_phase is not None, f"unsupported phase: {phase}")
    path = checks.pinned(predecessor, phase_cfg.get("predecessor_overlay_sha256"), "predecessor_overlay")
    if path is None:
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        checks.errors.append(f"predecessor overlay is unreadable: {exc}")
        return
    checks.expect(payload.get("schema") == "wm3d_v7_stage1_planner_overlay_v1", "predecessor schema")
    checks.expect(payload.get("phase") == expected_phase, "predecessor phase mismatch")
    checks.expect(payload.get("source_checkpoint_sha256") == source_sha, "predecessor Stage0 mismatch")
    checks.expect(int(payload.get("step", -1)) > 0, "predecessor endpoint step is invalid")


def _validate_runtime(checks: Checks, phase_cfg: dict[str, Any]) -> None:
    data_free = shutil.disk_usage("/data").free
    checks.evidence["data_free_bytes"] = data_free
    checks.expect(data_free >= MIN_DATA_FREE_BYTES, f"/data free {data_free} < {MIN_DATA_FREE_BYTES}")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    checks.expect(result.returncode == 0, f"nvidia-smi failed: {result.stderr.strip()}")
    gpu_rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    checks.expect(len(gpu_rows) == int(phase_cfg.get("gpus_per_node", -1)), "GPU count differs from config")
    for row in gpu_rows:
        fields = [field.strip() for field in row.split(",")]
        checks.expect(len(fields) == 3 and fields[1:] == ["0", "0"], f"uncorrected ECC is nonzero: {row}")
    process = subprocess.run(["pgrep", "-af", STAGE0_RUN_TOKEN], text=True, capture_output=True, check=False)
    active = [line for line in process.stdout.splitlines() if "preflight_wm3d_v7_stage1_planner" not in line]
    checks.expect(not active, "Stage0 formal training is still active; Stage1 launch is forbidden")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "data", "train"), default="static")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    checks = Checks()
    try:
        cfg = load_train_config(args.cfg)
    except Exception as exc:
        cfg = {}
        checks.errors.append(f"config resolution failed: {exc}")
    phase_cfg = dict(cfg.get("planner_stage") or {})
    data_cfg = dict(cfg.get("planner_data") or {})
    _validate_model_contract(checks, cfg)
    checks.expect(phase_cfg.get("phase") in {"dynamics", "planner", "joint"}, "invalid planner phase")
    checks.expect(phase_cfg.get("num_nodes") == 3, "formal Stage1-P requires three nodes")
    checks.expect(phase_cfg.get("gpus_per_node") == 8, "formal Stage1-P requires eight GPUs/node")
    checks.expect(phase_cfg.get("activation_checkpointing") is True, "activation checkpointing required")
    checks.expect(phase_cfg.get("candidate_microbatch") == 1, "candidate microbatch must be one")
    checks.expect(data_cfg.get("context_frames") == 16, "planner data must be T16")
    checks.expect(
        data_cfg.get("context_source")
        == "current_pinned_robocasa_runtime_causal_replay",
        "planner data must use real causal runtime T16 context",
    )
    checks.expect(data_cfg.get("future_frames") == 32, "planner data must be H32")
    checks.expect(data_cfg.get("action_history_len") == 4, "planner data must preserve H4 history")
    checks.expect(bool(phase_cfg.get("run_lineage")), "run_lineage is required")
    out_root = Path(str((cfg.get("out") or {}).get("root", "")))
    stage0_root = Path(
        "/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806/results/"
        + STAGE0_RUN_TOKEN
    )
    checks.expect(out_root != stage0_root and stage0_root not in out_root.parents, "Stage1 output overlaps Stage0")
    checks.expect(all(not prefix.startswith(SERVING_GUARD_PREFIXES) for prefix in (*DYNAMICS_PREFIXES, *PLANNER_PREFIXES)), "trainable allowlist touches serving action")

    source_path = checks.pinned(
        phase_cfg.get("source_checkpoint"),
        phase_cfg.get("source_checkpoint_sha256"),
        "source_checkpoint",
    )
    source_sha = sha256_file(source_path) if source_path is not None else None
    index_path = checks.pinned(data_cfg.get("index"), data_cfg.get("index_sha256"), "branch_index")
    checks.pinned(data_cfg.get("action_stats"), data_cfg.get("action_stats_sha256"), "action_stats")
    if args.mode in {"data", "train"} and index_path is not None:
        _validate_index(checks, index_path, sample_payloads=True)
    _validate_predecessor(checks, phase_cfg, source_sha)
    if args.mode == "train":
        _validate_runtime(checks, phase_cfg)
        checkpoint_dir = out_root / "ckpt"
        if args.resume is None and checkpoint_dir.exists():
            checks.expect(not any(checkpoint_dir.iterdir()), f"fresh phase checkpoint dir is not empty: {checkpoint_dir}")
        if args.resume is not None:
            resume_path = checks.pinned(args.resume, args.resume_sha256, "resume_checkpoint")
            if resume_path is not None:
                try:
                    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
                except Exception as exc:
                    checks.errors.append(f"resume checkpoint is unreadable: {exc}")
                else:
                    checks.expect(resume.get("schema") == "wm3d_v7_stage1_planner_overlay_v1", "resume schema")
                    checks.expect(resume.get("phase") == phase_cfg.get("phase"), "resume phase mismatch")
                    checks.expect(resume.get("run_lineage") == phase_cfg.get("run_lineage"), "resume lineage mismatch")
                    checks.expect(resume.get("source_checkpoint_sha256") == source_sha, "resume Stage0 mismatch")
                    checks.expect(int(resume.get("step", -1)) < int(phase_cfg.get("max_steps", -1)), "resume is not before hard stop")
        elif args.resume_sha256:
            checks.errors.append("resume SHA256 was provided without a resume path")

    report = {
        "schema": REPORT_SCHEMA,
        "mode": args.mode,
        "passed": not checks.errors,
        "errors": checks.errors,
        "warnings": checks.warnings,
        "config": str(args.cfg.resolve()),
        "config_sha256": config_sha256(cfg) if cfg else None,
        "phase": phase_cfg.get("phase"),
        "run_lineage": phase_cfg.get("run_lineage"),
        "evidence": checks.evidence,
        "contracts": {
            "native_3d": True,
            "planner_action_blind": True,
            "future_observation_leakage": False,
            "real_causal_t16_context": True,
            "serving_direct_policy_frozen": True,
            "flow_pose_only_auxiliary": True,
            "automatic_phase_promotion": False,
        },
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
        temporary.write_text(output)
        os.replace(temporary, args.report)
    print(output, end="")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
