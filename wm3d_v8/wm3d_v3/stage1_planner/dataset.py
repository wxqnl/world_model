"""Strict real-branch dataset for the WM3D-V8 Stage1 planning validation.

The expensive simulator branches were collected previously, but this loader
does not reuse their V7 action tensor as a dynamics condition.  It replays the
exact 20 Hz simulator commands through the audited canonical adapter, proves
that four controller substeps compose to the stored 5 Hz physical effect, and
then packs the native V8 36D causal action condition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wm3d_v3.data.v7_action_contract import ActionAdapter, canonicalize_dense_action
from wm3d_v3.data.v8_action_contract import (
    DYNAMICS_ACTION_DIM,
    POLICY_DT_SECONDS,
    PoseStats,
    SUBSTEPS_PER_WORLD,
    compose_base_delta_actions_np,
    pack_dynamics_action_condition,
    require_v8_pinned_file,
)


BRANCH_SCHEMA = "wm3d_v7_stage1_planner_branch_compact_v2"
RUNTIME_SCHEMA = "wm3d_v7_stage1_planner_same_root_runtime_v3"
DATASET_SCHEMA = "wm3d_v8_stage1_real_branch_adapter_v1"
EXPECTED_ROLES = (
    "factual_teacher",
    "direct",
    "flow_0",
    "flow_1",
    "flow_2",
    "flow_3",
    "grip_open",
    "grip_close",
    "arm_hold",
    "pose_reverse",
    "pose_half",
)


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


@dataclass(frozen=True)
class Stage1BranchDatasetConfig:
    branch_index: Path
    branch_index_sha256: str
    branch_payload_sha256_manifest: Path
    branch_payload_sha256_manifest_sha256: str
    runtime_index: Path
    runtime_index_sha256: str
    action_stats: Path
    action_stats_sha256: str
    action_adapter_audit: Path
    action_adapter_audit_sha256: str
    split: str
    context_frames: int = 16
    future_frames: int = 32
    verify_runtime_payload_sha256: bool = True
    require_task_emb: bool = True


class Stage1BranchDataset(Dataset):
    """Return true native futures and V8-compatible candidate conditions."""

    def __init__(self, cfg: Stage1BranchDatasetConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {cfg.split}")
        if cfg.context_frames != 16 or cfg.future_frames != 32:
            raise ValueError("WM3D-V8 Stage1 validation requires T16/H32")

        branch_index = require_v8_pinned_file(
            cfg.branch_index,
            cfg.branch_index_sha256,
            label="Stage1 branch index",
        )
        runtime_index = require_v8_pinned_file(
            cfg.runtime_index,
            cfg.runtime_index_sha256,
            label="Stage1 runtime index",
        )
        branch_payload_manifest = require_v8_pinned_file(
            cfg.branch_payload_sha256_manifest,
            cfg.branch_payload_sha256_manifest_sha256,
            label="Stage1 branch payload SHA256 manifest",
        )
        stats_path = require_v8_pinned_file(
            cfg.action_stats,
            cfg.action_stats_sha256,
            label="Stage1 V8 action statistics",
        )
        audit_path = require_v8_pinned_file(
            cfg.action_adapter_audit,
            cfg.action_adapter_audit_sha256,
            label="Stage1 action adapter audit",
        )

        with np.load(stats_path, allow_pickle=False) as stats:
            if str(stats["schema"].item()) != "wm3d_v8_action20_stats_v1":
                raise ValueError("Stage1 requires formal V8 fine-action statistics")
            if str(stats["split"].item()) != "train":
                raise ValueError("Stage1 action statistics must be train-only")
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        self.pose_stats = PoseStats(mean=mean, std=std, key=cfg.action_stats_sha256)

        audit = json.loads(audit_path.read_text())
        factual_audit = audit.get("factual_action_audit")
        if not isinstance(factual_audit, dict) or factual_audit.get("passed") is not True:
            raise ValueError("Stage1 action adapter audit did not pass")
        adapter_payload = audit.get("adapter")
        if not isinstance(adapter_payload, dict):
            raise ValueError("Stage1 action adapter audit lacks adapter payload")
        self.adapter = ActionAdapter(**adapter_payload)
        if abs(float(self.adapter.nominal_hz) - 20.0) > 1.0e-9:
            raise ValueError("Stage1 real runtime actions must be exactly 20 Hz")

        branch_payload_sha256: dict[str, str] = {}
        for line_number, line in enumerate(
            branch_payload_manifest.read_text().splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            root_id = str(row.get("root_id", ""))
            digest = str(row.get("payload_sha256", ""))
            if not _hex64(root_id) or root_id in branch_payload_sha256:
                raise ValueError(
                    f"blank/duplicate branch payload root_id: {root_id!r}"
                )
            if not _hex64(digest):
                raise ValueError(
                    f"invalid branch payload SHA at line {line_number}"
                )
            branch_payload_sha256[root_id] = digest

        runtime_rows: dict[str, dict] = {}
        for line_number, line in enumerate(runtime_index.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != RUNTIME_SCHEMA:
                raise ValueError(f"unexpected runtime schema at line {line_number}")
            root_id = str(row.get("root_id", ""))
            if not _hex64(root_id) or root_id in runtime_rows:
                raise ValueError(f"blank/duplicate runtime root_id: {root_id!r}")
            self._validate_exact_row(row, label=f"runtime {root_id}")
            if row.get("action_audit_sha256") != cfg.action_adapter_audit_sha256:
                raise ValueError(f"runtime action audit mismatch: {root_id}")
            if not _hex64(row.get("payload_sha256")):
                raise ValueError(f"runtime payload SHA is missing: {root_id}")
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            runtime_rows[root_id] = row

        records: list[tuple[dict, dict]] = []
        seen_roots: set[str] = set()
        for line_number, line in enumerate(branch_index.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != BRANCH_SCHEMA:
                raise ValueError(f"unexpected branch schema at line {line_number}")
            if row.get("split") != cfg.split:
                continue
            root_id = str(row.get("root_id", ""))
            if not _hex64(root_id) or root_id in seen_roots:
                raise ValueError(f"blank/duplicate branch root_id: {root_id!r}")
            seen_roots.add(root_id)
            self._validate_exact_row(row, label=f"branch {root_id}")
            if int(row.get("context_frames", -1)) != cfg.context_frames:
                raise ValueError(f"context horizon mismatch: {root_id}")
            if int(row.get("future_frames", -1)) != cfg.future_frames:
                raise ValueError(f"future horizon mismatch: {root_id}")
            if tuple(row.get("branch_roles") or ()) != EXPECTED_ROLES:
                raise ValueError(f"candidate role contract mismatch: {root_id}")
            if row.get("runtime_index_sha256") != cfg.runtime_index_sha256:
                raise ValueError(f"branch/runtime index lineage mismatch: {root_id}")
            runtime = runtime_rows.get(root_id)
            if runtime is None:
                raise ValueError(f"branch has no exact runtime row: {root_id}")
            if row.get("runtime_payload_sha256") != runtime.get("payload_sha256"):
                raise ValueError(f"branch/runtime payload lineage mismatch: {root_id}")
            if row.get("split") != runtime.get("split"):
                raise ValueError(f"branch/runtime split mismatch: {root_id}")
            if tuple(runtime.get("branch_roles") or ()) != EXPECTED_ROLES:
                raise ValueError(f"runtime role contract mismatch: {root_id}")
            path = Path(row["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_payload_sha256 = branch_payload_sha256.get(root_id)
            if expected_payload_sha256 is None:
                raise ValueError(f"branch payload SHA is missing: {root_id}")
            if sha256_file(path) != expected_payload_sha256:
                raise ValueError(f"branch payload SHA mismatch: {root_id}")
            records.append((row, runtime))
        if not records:
            raise RuntimeError(f"Stage1 split {cfg.split} is empty")
        self.records = records

        indexed_roots = {
            json.loads(line)["root_id"]
            for line in branch_index.read_text().splitlines()
            if line.strip()
        }
        if set(branch_payload_sha256) != indexed_roots:
            missing = sorted(indexed_roots - set(branch_payload_sha256))
            extra = sorted(set(branch_payload_sha256) - indexed_roots)
            raise ValueError(
                f"branch payload SHA manifest closure mismatch "
                f"missing={missing[:4]} extra={extra[:4]}"
            )

        if cfg.verify_runtime_payload_sha256:
            for _, runtime in self.records:
                observed = sha256_file(Path(runtime["path"]))
                if observed != runtime["payload_sha256"]:
                    raise ValueError(
                        f"runtime payload SHA mismatch: {runtime['root_id']}"
                    )

    @staticmethod
    def _validate_exact_row(row: dict, *, label: str) -> None:
        required_true = (
            "same_root_current_runtime_exact",
            "same_root_simulator_state_exact",
            "same_root_render_state_exact",
            "root_rgb_equivalence_all_passed",
            "same_root_rgb_canonicalized",
        )
        if any(row.get(key) is not True for key in required_true):
            raise ValueError(f"{label} is not exact same-root evidence")
        if row.get("pseudo_outcomes") is not False:
            raise ValueError(f"{label} uses pseudo outcomes")
        if row.get("future_observation_leakage") is not False:
            raise ValueError(f"{label} violates causal observation contract")
        if row.get("outcome_source") != "current_pinned_robocasa_simulator":
            raise ValueError(f"{label} is not grounded in the pinned simulator")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
        result = np.asarray(codes, dtype=np.int8).astype(np.float32) * np.asarray(
            scale, dtype=np.float32
        )
        if not np.isfinite(result).all():
            raise ValueError("non-finite dequantized token cache")
        return result

    def _v8_candidate_conditions(
        self,
        simulator_actions: np.ndarray,
        physical_actions: np.ndarray,
        *,
        root_id: str,
    ) -> np.ndarray:
        candidates, controller_horizon, raw_dim = simulator_actions.shape
        if controller_horizon != self.cfg.future_frames * SUBSTEPS_PER_WORLD:
            raise RuntimeError(f"20 Hz action horizon mismatch: {root_id}")
        fine = canonicalize_dense_action(
            simulator_actions.reshape(-1, raw_dim), self.adapter
        ).reshape(candidates, self.cfg.future_frames, SUBSTEPS_PER_WORLD, 7)
        composed = compose_base_delta_actions_np(fine)
        if not np.allclose(composed, physical_actions, rtol=0.0, atol=2.0e-5):
            error = float(np.max(np.abs(composed - physical_actions)))
            raise RuntimeError(
                f"20 Hz to 5 Hz physical composition mismatch {error}: {root_id}"
            )
        normalized = fine.copy()
        normalized[..., :6] = self.pose_stats.normalize(fine[..., :6])
        valid = np.ones(normalized.shape[:-1], dtype=np.bool_)
        dt = np.full(normalized.shape[:-1], POLICY_DT_SECONDS, dtype=np.float32)
        packed = np.stack(
            [pack_dynamics_action_condition(value, valid[i], dt[i]) for i, value in enumerate(normalized)],
            axis=0,
        )
        if packed.shape != (
            candidates,
            self.cfg.future_frames,
            DYNAMICS_ACTION_DIM,
        ):
            raise AssertionError(packed.shape)
        return packed

    def __getitem__(self, index: int) -> dict[str, object]:
        row, runtime_row = self.records[index]
        root_id = row["root_id"]
        with np.load(row["path"], allow_pickle=False) as archive:
            if str(archive["schema"].item()) != BRANCH_SCHEMA:
                raise ValueError(f"branch payload schema mismatch: {root_id}")
            if str(archive["root_id"].item()) != root_id:
                raise ValueError(f"branch payload identity mismatch: {root_id}")
            roles = tuple(str(value) for value in archive["branch_roles"].tolist())
            anchor_codes = np.asarray(archive["anchor_codes"], dtype=np.int8)
            anchor_scale = np.asarray(archive["anchor_scale"], dtype=np.float32)
            wrist_codes = np.asarray(archive["wrist_codes"], dtype=np.int8)
            wrist_scale = np.asarray(archive["wrist_scale"], dtype=np.float32)
            context = self._dequantize(anchor_codes, anchor_scale)
            wrist = self._dequantize(wrist_codes, wrist_scale)
            branches = self._dequantize(archive["branch_codes"], archive["branch_scales"])
            physical = np.asarray(archive["branch_actions_physical"], dtype=np.float32)
            branch_valid = np.asarray(archive["branch_valid"], dtype=np.bool_)
            rewards = np.asarray(archive["branch_rewards"], dtype=np.float32)
            dones = np.asarray(archive["branch_dones"], dtype=np.bool_)
            success = np.asarray(archive["branch_success"], dtype=np.bool_)
            task = np.asarray(archive["task_emb"], dtype=np.float32)
            depth = np.asarray(archive["branch_depth_tgt"], dtype=np.float32)
            depth_conf = np.asarray(archive["branch_depth_conf_tgt"], dtype=np.float32)
            point = np.asarray(archive["branch_point_tgt"], dtype=np.float32)
            point_conf = np.asarray(archive["branch_point_conf_tgt"], dtype=np.float32)
            pose = np.asarray(archive["branch_pose_geom_tgt"], dtype=np.float32)
            factual_index = int(archive["factual_index"].item())
            direct_index = int(archive["direct_index"].item())

        with np.load(runtime_row["path"], allow_pickle=False) as runtime:
            if str(runtime["schema"].item()) != RUNTIME_SCHEMA:
                raise ValueError(f"runtime payload schema mismatch: {root_id}")
            if str(runtime["root_id"].item()) != root_id:
                raise ValueError(f"runtime payload identity mismatch: {root_id}")
            runtime_roles = tuple(str(value) for value in runtime["branch_roles"].tolist())
            simulator_actions = np.asarray(runtime["simulator_actions"], dtype=np.float32)
            runtime_physical = np.asarray(runtime["branch_actions_physical"], dtype=np.float32)
            runtime_rewards = np.asarray(runtime["branch_rewards"], dtype=np.float32)
            runtime_dones = np.asarray(runtime["branch_dones"], dtype=np.bool_)
            runtime_success = np.asarray(runtime["branch_success"], dtype=np.bool_)

        candidates = len(EXPECTED_ROLES)
        horizon = self.cfg.future_frames
        if roles != EXPECTED_ROLES or runtime_roles != EXPECTED_ROLES:
            raise RuntimeError(f"payload role mismatch: {root_id}")
        if anchor_codes.shape != (16, 64, 384) or anchor_scale.shape != (16, 1, 1):
            raise RuntimeError(f"anchor context cache shape mismatch: {root_id}")
        if wrist_codes.shape != anchor_codes.shape or wrist_scale.shape != anchor_scale.shape:
            raise RuntimeError(f"wrist context cache shape mismatch: {root_id}")
        if context.shape != (16, 64, 384) or wrist.shape != context.shape:
            raise RuntimeError(f"context token shape mismatch: {root_id}")
        if branches.shape != (candidates, horizon, 64, 384):
            raise RuntimeError(f"branch token shape mismatch: {root_id}")
        if physical.shape != (candidates, horizon, 7):
            raise RuntimeError(f"physical action shape mismatch: {root_id}")
        if runtime_physical.shape != physical.shape or not np.array_equal(runtime_physical, physical):
            raise RuntimeError(f"branch/runtime physical actions differ: {root_id}")
        if not np.array_equal(runtime_rewards, rewards) or not np.array_equal(runtime_dones, dones) or not np.array_equal(runtime_success, success):
            raise RuntimeError(f"branch/runtime outcomes differ: {root_id}")
        if branch_valid.shape != (candidates,) or not branch_valid.all():
            raise RuntimeError(f"branch validity mismatch: {root_id}")
        if any(value.shape != (candidates, horizon) for value in (rewards, dones, success)):
            raise RuntimeError(f"outcome shape mismatch: {root_id}")
        if depth.ndim != 4 or depth.shape[:2] != (candidates, horizon) or depth_conf.shape != depth.shape:
            raise RuntimeError(f"depth shape mismatch: {root_id}")
        if point.ndim != 5 or point.shape[:2] != (candidates, horizon) or point.shape[-1] != 3:
            raise RuntimeError(f"point shape mismatch: {root_id}")
        if point_conf.shape != point.shape[:-1] or pose.shape != (candidates, horizon, 9):
            raise RuntimeError(f"point/pose shape mismatch: {root_id}")
        if not (factual_index == 0 and roles[factual_index] == "factual_teacher"):
            raise RuntimeError(f"factual teacher must be branch zero: {root_id}")
        if not (direct_index == 1 and roles[direct_index] == "direct"):
            raise RuntimeError(f"direct candidate index mismatch: {root_id}")
        finite = (physical, simulator_actions, rewards, task, depth, depth_conf, point, point_conf, pose)
        if any(not np.isfinite(value).all() for value in finite):
            raise RuntimeError(f"non-finite payload: {root_id}")
        if self.cfg.require_task_emb and (task.shape != (2048,) or not np.any(task)):
            raise RuntimeError(f"missing real task embedding: {root_id}")

        conditions = self._v8_candidate_conditions(
            simulator_actions,
            physical,
            root_id=root_id,
        )
        planning_mask = branch_valid.copy()
        planning_mask[factual_index] = False
        return {
            "s_in": torch.from_numpy(context.copy()),
            "s_wrist": torch.from_numpy(wrist.copy()),
            "view_mask": torch.ones((self.cfg.context_frames, 2), dtype=torch.bool),
            "c": torch.from_numpy(task.copy()),
            "candidate_actions": torch.from_numpy(conditions.copy()),
            "branch_actions_physical": torch.from_numpy(physical.copy()),
            "branch_s_tgt_codec": torch.from_numpy(branches.copy()),
            "branch_depth_tgt": torch.from_numpy(depth.copy()),
            "branch_depth_conf_tgt": torch.from_numpy(depth_conf.copy()),
            "branch_point_tgt": torch.from_numpy(point.copy()),
            "branch_point_conf_tgt": torch.from_numpy(point_conf.copy()),
            "branch_pose_geom_tgt": torch.from_numpy(pose.copy()),
            "branch_rewards": torch.from_numpy(rewards.copy()),
            "branch_dones": torch.from_numpy(dones.copy()),
            "branch_success": torch.from_numpy(success.copy()),
            "branch_valid": torch.from_numpy(branch_valid.copy()),
            "planning_mask": torch.from_numpy(planning_mask),
            "factual_index": factual_index,
            "direct_index": direct_index,
            "branch_roles": roles,
            "root_id": root_id,
            "task": row.get("task", row.get("task_text", "")),
            "source_candidate_checkpoint_sha256": row.get("stage0_checkpoint_sha256"),
        }
