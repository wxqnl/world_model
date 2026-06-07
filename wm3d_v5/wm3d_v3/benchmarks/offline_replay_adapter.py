"""Offline OXE replay adapter for benchmark-runner contract tests.

This is not a substitute for LIBERO/CALVIN/SimplerEnv success-rate evaluation.
It uses the same `BenchmarkAdapter` interface on cached OXE windows so the
policy loop can be exercised on real WM3D tokens, Qwen task embeddings, and
demonstration actions before an external simulator is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from wm3d_v3.benchmarks.adapter import BenchmarkAdapter, BenchmarkTask, TokenizedObservation
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.run_eval import build_dataset_for_split


@dataclass
class OfflineReplayState:
    sample: dict[str, Any]
    step: int = 0


class OfflineReplayAdapter(BenchmarkAdapter):
    """Replay cached validation windows as a lightweight policy-loop benchmark."""

    name = "offline_replay"

    def __init__(
        self,
        cfg_path: str | Path,
        *,
        split: str = "val",
        success_pose_l1_threshold: float = 0.5,
    ) -> None:
        self.cfg_path = Path(cfg_path)
        self.cfg = yaml.safe_load(self.cfg_path.read_text())
        self.split = split
        self.success_pose_l1_threshold = float(success_pose_l1_threshold)
        records = read_manifest(self.cfg["data"]["manifest"])
        self.dataset = build_dataset_for_split(records, self.cfg, split=split)
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        action_stats = self.cfg["data"].get("action_stats")
        if action_stats:
            stats_path = Path(action_stats)
            if stats_path.exists():
                import numpy as np

                stats = np.load(stats_path)
                self.action_mean = torch.from_numpy(stats["mean"][:6].astype("float32"))
                self.action_std = torch.from_numpy(stats["std"][:6].astype("float32")).clamp_min(1e-6)

    def iter_tasks(self, *, limit: int | None = None) -> list[BenchmarkTask]:
        n = min(len(self.dataset), limit or len(self.dataset))
        tasks: list[BenchmarkTask] = []
        for index in range(n):
            sample = self.dataset[index]
            tasks.append(
                BenchmarkTask(
                    name=f"{sample['dataset']}::{sample['clip_id']}::{int(sample['start'])}",
                    instruction="offline replay of cached demonstration window",
                    metadata={
                        "dataset_index": index,
                        "dataset": str(sample["dataset"]),
                        "clip_id": str(sample["clip_id"]),
                        "start": int(sample["start"]),
                    },
                )
            )
        return tasks

    def reset(self, task: BenchmarkTask) -> OfflineReplayState:
        index = int(task.metadata["dataset_index"])
        return OfflineReplayState(sample=self.dataset[index])

    def observe(self, env_state: OfflineReplayState, task: BenchmarkTask) -> TokenizedObservation:
        sample = env_state.sample
        context_rgb = sample["rgb_in"][-1].permute(2, 0, 1).contiguous().unsqueeze(0)
        return TokenizedObservation(
            context_tokens=sample["s_in"].unsqueeze(0),
            task_emb=sample["c"].unsqueeze(0),
            context_rgb=context_rgb,
            metadata={
                "dataset": str(sample["dataset"]),
                "clip_id": str(sample["clip_id"]),
                "start": int(sample["start"]),
            },
        )

    def to_env_action(
        self,
        raw_action: torch.Tensor,
        env_state: OfflineReplayState,
        task: BenchmarkTask,
    ) -> torch.Tensor:
        return raw_action.detach().float().cpu()

    def step(
        self,
        env_state: OfflineReplayState,
        env_action: torch.Tensor,
    ) -> tuple[OfflineReplayState, bool, dict[str, Any]]:
        sample = env_state.sample
        target_raw = sample["action_tgt"][0].detach().float().cpu()
        pred_raw = env_action.detach().float().cpu()
        pose_l1_raw = (pred_raw[:6] - target_raw[:6]).abs().mean()
        grip_target = float(target_raw[6] > 0.5)
        grip_pred = float(pred_raw[6] > 0.5)
        info: dict[str, Any] = {
            "pose_l1_raw": float(pose_l1_raw),
            "grip_match": float(grip_pred == grip_target),
            "target_first_action_raw": target_raw.tolist(),
            "pred_first_action_raw": pred_raw.tolist(),
        }
        if self.action_mean is not None and self.action_std is not None:
            pred_norm = (pred_raw[:6] - self.action_mean) / self.action_std
            target_norm = sample["action_tgt_norm"][0].detach().float().cpu()
            pose_l1_norm = (pred_norm - target_norm).abs().mean()
            info["pose_l1_norm"] = float(pose_l1_norm)
            info["success"] = float(pose_l1_norm <= self.success_pose_l1_threshold)
        else:
            info["success"] = float(pose_l1_raw <= self.success_pose_l1_threshold)
        env_state.step += 1
        return env_state, True, info

    def is_success(
        self,
        env_state: OfflineReplayState,
        info: dict[str, Any],
        task: BenchmarkTask,
    ) -> bool:
        return bool(info.get("success", 0.0))
