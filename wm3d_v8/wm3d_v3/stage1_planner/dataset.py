"""Strict v2 same-root branch dataset for native V7 planning.

Unlike the legacy K8 cache, every branch carries its own H32 native token,
depth, point and pose evidence.  The loader rejects pseudo outcomes, missing
root history and any mismatch between the JSONL identity and NPZ payload.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA = "wm3d_v7_stage1_planner_branch_compact_v2"


@dataclass(frozen=True)
class Stage1BranchDatasetConfig:
    index_path: Path
    split: str
    action_stats: Path
    context_frames: int = 16
    future_frames: int = 32
    action_history_len: int = 4
    expected_roles: tuple[str, ...] = (
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
    require_task_emb: bool = True


class Stage1BranchDataset(Dataset):
    def __init__(self, cfg: Stage1BranchDatasetConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {cfg.split}")
        if cfg.context_frames != 16 or cfg.future_frames != 32:
            raise ValueError("V7 Stage1-P requires T16/H32")
        if cfg.action_history_len != 4:
            raise ValueError("V7 Stage1-P must preserve the Stage0 H4 action history")
        with np.load(cfg.action_stats, allow_pickle=False) as stats:
            if str(stats["split"].item()) != "train":
                raise ValueError("action statistics must be train-only")
            self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
            self.action_std = np.asarray(stats["std"], dtype=np.float32)
        if (
            self.action_mean.shape != (6,)
            or self.action_std.shape != (6,)
            or not np.isfinite(self.action_mean).all()
            or not np.isfinite(self.action_std).all()
            or np.any(self.action_std <= 0)
        ):
            raise ValueError("invalid canonical action statistics")

        records: list[dict] = []
        seen_roots: set[str] = set()
        with Path(cfg.index_path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != SCHEMA:
                    raise ValueError(f"unexpected schema at line {line_number}")
                if row.get("split") != cfg.split:
                    continue
                root_id = str(row.get("root_id", ""))
                if not root_id or root_id in seen_roots:
                    raise ValueError(f"blank/duplicate root_id: {root_id!r}")
                seen_roots.add(root_id)
                if not row.get("same_root_current_runtime_exact"):
                    raise ValueError(f"root is not current-runtime exact: {root_id}")
                if row.get("pseudo_outcomes") is not False:
                    raise ValueError(f"pseudo outcomes are forbidden: {root_id}")
                if row.get("future_observation_leakage") is not False:
                    raise ValueError(f"future observation leakage contract missing: {root_id}")
                if row.get("context_source") != (
                    "current_pinned_robocasa_runtime_causal_replay"
                ):
                    raise ValueError(f"real causal T16 context missing: {root_id}")
                root_context_sha = str(row.get("root_context_sha256", ""))
                if len(root_context_sha) != 64 or any(
                    char not in "0123456789abcdef" for char in root_context_sha
                ):
                    raise ValueError(f"root-context SHA missing: {root_id}")
                if int(row.get("context_frames", -1)) != cfg.context_frames:
                    raise ValueError(f"context horizon mismatch: {root_id}")
                if int(row.get("future_frames", -1)) != cfg.future_frames:
                    raise ValueError(f"future horizon mismatch: {root_id}")
                if tuple(row.get("branch_roles") or ()) != cfg.expected_roles:
                    raise ValueError(f"candidate role contract mismatch: {root_id}")
                path = Path(row["path"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                records.append(row)
        if not records:
            raise RuntimeError(f"Stage1-P split {cfg.split} is empty")
        self.records = records

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

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records[index]
        with np.load(row["path"], allow_pickle=False) as archive:
            if str(archive["schema"].item()) != SCHEMA:
                raise ValueError(f"payload schema mismatch: {row['root_id']}")
            if str(archive["root_id"].item()) != row["root_id"]:
                raise ValueError(f"payload identity mismatch: {row['root_id']}")
            roles = tuple(str(value) for value in archive["branch_roles"].tolist())
            if roles != self.cfg.expected_roles:
                raise ValueError(f"payload role mismatch: {row['root_id']}")
            root_context_sha = str(archive["root_context_sha256"].item())
            anchor_codes = np.asarray(archive["anchor_codes"], dtype=np.int8)
            anchor_scale = np.asarray(archive["anchor_scale"], dtype=np.float32)
            wrist_codes = np.asarray(archive["wrist_codes"], dtype=np.int8)
            wrist_scale = np.asarray(archive["wrist_scale"], dtype=np.float32)
            context = self._dequantize(anchor_codes, anchor_scale)
            wrist = self._dequantize(wrist_codes, wrist_scale)
            branches = self._dequantize(archive["branch_codes"], archive["branch_scales"])
            actions = np.asarray(archive["branch_actions_physical"], dtype=np.float32)
            action_history = np.asarray(archive["action_history_physical"], dtype=np.float32)
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

        c = len(roles)
        h = self.cfg.future_frames
        if root_context_sha != row["root_context_sha256"]:
            raise ValueError(f"payload root-context provenance mismatch: {row['root_id']}")
        if anchor_codes.shape != (16, 64, 384) or anchor_scale.shape != (16, 1, 1):
            raise RuntimeError(f"anchor context cache shape mismatch: {row['root_id']}")
        if wrist_codes.shape != anchor_codes.shape or wrist_scale.shape != anchor_scale.shape:
            raise RuntimeError(f"wrist context cache shape mismatch: {row['root_id']}")
        if context.shape[:2] != (self.cfg.context_frames, 64) or wrist.shape != context.shape:
            raise RuntimeError(f"context token shape mismatch: {row['root_id']}")
        if branches.shape[:3] != (c, h, 64):
            raise RuntimeError(f"branch token shape mismatch: {row['root_id']}")
        if actions.shape != (c, h, 7) or action_history.shape != (4, 7):
            raise RuntimeError(f"action/history shape mismatch: {row['root_id']}")
        if branch_valid.shape != (c,) or not branch_valid.all():
            raise RuntimeError(f"branch validity mismatch: {row['root_id']}")
        if any(value.shape != (c, h) for value in (rewards, dones, success)):
            raise RuntimeError(f"outcome shape mismatch: {row['root_id']}")
        if depth.ndim != 4 or depth.shape[:2] != (c, h) or depth_conf.shape != depth.shape:
            raise RuntimeError(f"depth shape mismatch: {row['root_id']}")
        if point.ndim != 5 or point.shape[:2] != (c, h) or point.shape[-1] != 3:
            raise RuntimeError(f"point shape mismatch: {row['root_id']}")
        if point_conf.shape != point.shape[:-1] or pose.shape != (c, h, 9):
            raise RuntimeError(f"point/pose shape mismatch: {row['root_id']}")
        if not (factual_index == 0 and roles[factual_index] == "factual_teacher"):
            raise RuntimeError(f"factual teacher must be branch zero: {row['root_id']}")
        if not (0 <= direct_index < c and roles[direct_index] == "direct"):
            raise RuntimeError(f"direct candidate index mismatch: {row['root_id']}")
        tensors = (actions, action_history, rewards, task, depth, depth_conf, point, point_conf, pose)
        if any(not np.isfinite(value).all() for value in tensors):
            raise RuntimeError(f"non-finite payload: {row['root_id']}")
        if self.cfg.require_task_emb and (task.shape != (2048,) or not np.any(task)):
            raise RuntimeError(f"missing real task embedding: {row['root_id']}")

        action_cond = actions.copy()
        action_cond[..., :6] = (
            action_cond[..., :6] - self.action_mean[None, None]
        ) / self.action_std[None, None]
        action_cond[..., 6] = np.clip((actions[..., 6] + 1.0) * 0.5, 0.0, 1.0)
        history = action_history.copy()
        history[..., 6] = np.clip((history[..., 6] + 1.0) * 0.5, 0.0, 1.0)
        planning_mask = branch_valid.copy()
        planning_mask[factual_index] = False
        return {
            "s_in": torch.from_numpy(context.copy()),
            "s_wrist": torch.from_numpy(wrist.copy()),
            "view_mask": torch.ones((self.cfg.context_frames, 2), dtype=torch.bool),
            "c": torch.from_numpy(task.copy()),
            "candidate_actions": torch.from_numpy(action_cond.copy()),
            "branch_actions_physical": torch.from_numpy(actions.copy()),
            "action_history": torch.from_numpy(history.copy()),
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
            "root_id": row["root_id"],
            "task": row.get("task", row.get("task_text", "")),
        }
