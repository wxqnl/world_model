"""Dataset reader for the episode-shared WM3D-v7 compact cache."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class V7CompactDatasetConfig:
    index_path: Path
    split: str
    T: int = 16
    k: int = 8
    stride: int = 2
    view_dropout: float = 0.0
    seed: int = 0
    require_task_emb: bool = True
    action_stats: Path | None = None
    require_action_stats: bool = True
    rgb_sidecar_indices: tuple[Path, ...] = ()
    require_rgb_sidecar: bool = False


@dataclass
class V7SameRootBranchDatasetConfig:
    index_path: Path
    split: str
    T: int = 16
    k: int = 8
    require_task_emb: bool = True
    action_stats: Path | None = None
    require_action_stats: bool = True


class V7CompactWindowDataset(Dataset):
    """Return compressed tokens; the fixed codec is decoded on GPU by the model."""

    def __init__(self, cfg: V7CompactDatasetConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {cfg.split}")
        if min(cfg.T, cfg.k, cfg.stride) <= 0:
            raise ValueError("T, k, and stride must be positive")
        rows: list[dict] = []
        with Path(cfg.index_path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["split"] == cfg.split:
                        if not row.get("action_valid") or row.get("pseudo_outcomes") is not False:
                            raise ValueError(f"invalid formal cache row: {row.get('clip_hash')}")
                        teacher = row.get("geometry_teacher") or {}
                        if not teacher.get("pseudo_teacher") or not teacher.get("confidence_stored"):
                            raise ValueError(f"missing geometry pseudo-teacher provenance: {row.get('clip_hash')}")
                        rows.append(row)
        self.records = rows
        self.rgb_records: dict[str, dict] = {}
        for index_path in cfg.rgb_sidecar_indices:
            with Path(index_path).open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    rgb_row = json.loads(line)
                    if rgb_row.get("schema") != "wm3d_v7_rgb_sidecar_v1":
                        raise ValueError(
                            f"unexpected RGB sidecar schema {index_path}:{line_number}"
                        )
                    clip_hash = str(rgb_row.get("clip_hash", ""))
                    if not clip_hash or clip_hash in self.rgb_records:
                        raise ValueError(
                            f"blank or duplicate RGB sidecar clip_hash: {clip_hash}"
                        )
                    self.rgb_records[clip_hash] = rgb_row
        if cfg.require_rgb_sidecar and not cfg.rgb_sidecar_indices:
            raise ValueError("require_rgb_sidecar=true requires rgb_sidecar_indices")
        if cfg.require_rgb_sidecar:
            missing = [row["clip_hash"] for row in rows if row["clip_hash"] not in self.rgb_records]
            if missing:
                raise ValueError(
                    f"RGB sidecar is missing {len(missing)} compact clips; first={missing[:8]}"
                )
        for row in rows:
            rgb_row = self.rgb_records.get(row["clip_hash"])
            if rgb_row is None:
                continue
            if str(rgb_row.get("split")) != str(row["split"]):
                raise ValueError(f"RGB sidecar split mismatch: {row['clip_hash']}")
            if int(rgb_row.get("model_frames", -1)) != int(row["model_frames"]):
                raise ValueError(f"RGB sidecar frame-count mismatch: {row['clip_hash']}")
            if not Path(rgb_row["path"]).is_file():
                raise FileNotFoundError(rgb_row["path"])
        if cfg.action_stats is None:
            if cfg.require_action_stats:
                raise ValueError("formal compact training requires train-only action_stats")
            self.action_mean = np.zeros(6, dtype=np.float32)
            self.action_std = np.ones(6, dtype=np.float32)
        else:
            with np.load(cfg.action_stats, allow_pickle=False) as stats:
                if str(stats["split"].item()) != "train":
                    raise ValueError("action statistics must be fit on train split only")
                self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
                self.action_std = np.asarray(stats["std"], dtype=np.float32)
            if self.action_mean.shape != (6,) or self.action_std.shape != (6,) or np.any(self.action_std <= 0):
                raise ValueError("invalid canonical action statistics")
        self.index: list[tuple[int, int]] = []
        required = cfg.T + cfg.k
        for record_index, row in enumerate(rows):
            segments = row.get("geometry_segments")
            if not segments:
                raise ValueError(f"formal cache has no VGGT gauge segments: {row.get('clip_hash')}")
            for segment_start, segment_stop in segments:
                segment_start, segment_stop = int(segment_start), int(segment_stop)
                if not (0 <= segment_start < segment_stop <= int(row["model_frames"])):
                    raise ValueError(f"invalid geometry segment: {row.get('clip_hash')}")
                self.index.extend(
                    (record_index, start)
                    for start in range(
                        segment_start,
                        max(segment_start, segment_stop - required + 1),
                        cfg.stride,
                    )
                )
        if not self.index:
            raise RuntimeError(f"compact cache split {cfg.split} contains no valid windows")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _latent(archive, prefix: str) -> np.ndarray:
        codes = np.asarray(archive[f"{prefix}_codes"], dtype=np.int8)
        scale = np.asarray(archive[f"{prefix}_scale"], dtype=np.float32)
        return codes.astype(np.float32) * scale

    def _drop_wrist(self, sample_index: int) -> bool:
        if self.cfg.split != "train" or self.cfg.view_dropout <= 0:
            return False
        generator = np.random.default_rng(
            np.random.SeedSequence((int(self.cfg.seed), int(self.epoch), int(sample_index)))
        )
        return bool(generator.random() < self.cfg.view_dropout)

    def __getitem__(self, sample_index: int) -> dict:
        record_index, start = self.index[sample_index]
        row = self.records[record_index]
        T, k = self.cfg.T, self.cfg.k
        with np.load(row["path"], allow_pickle=False) as archive:
            if str(archive["schema"].item()) != "wm3d_v7_compact_geom_v3":
                raise ValueError(f"unexpected compact cache schema: {row['path']}")
            for key, expected in (
                ("clip_hash", row["clip_hash"]),
                ("split", row["split"]),
                ("source", row["source"]),
                ("action_adapter_version", row["action_adapter_version"]),
                ("action_audit_sha256", row["action_audit_sha256"]),
            ):
                if str(archive[key].item()) != str(expected):
                    raise ValueError(f"compact cache identity mismatch for {key}: {row['clip_hash']}")
            anchor = self._latent(archive, "anchor")
            paired_views = bool(row.get("paired_views", False))
            wrist = self._latent(archive, "wrist") if paired_views else np.zeros_like(anchor)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            action_valid_mask = np.asarray(archive["action_valid_mask"], dtype=np.bool_)
            task = np.asarray(archive["task_emb"], dtype=np.float32)
            depth = np.asarray(archive["depth_patch"], dtype=np.float32)
            depth_conf = np.asarray(archive["depth_conf_patch"], dtype=np.float32)
            points = np.asarray(archive["point_patch"], dtype=np.float32)
            point_conf = np.asarray(archive["point_conf_patch"], dtype=np.float32)
            pose = np.asarray(archive["pose_enc"], dtype=np.float32)
            geometry_segment_id = np.asarray(archive["geometry_segment_id"], dtype=np.int16)
        if self.cfg.require_task_emb and (task.shape != (2048,) or not np.any(task)):
            raise RuntimeError(f"missing real task embedding: {row['clip_hash']}")
        end = start + T + k
        if min(
            len(anchor), len(wrist), len(actions), len(depth), len(depth_conf),
            len(points), len(point_conf), len(pose),
        ) < end:
            raise RuntimeError(f"short compact cache record: {row['clip_hash']}")
        if len(action_valid_mask) < end or not action_valid_mask[start:end].all():
            raise RuntimeError(f"invalid action interval: {row['clip_hash']}")
        if len(np.unique(geometry_segment_id[start:end])) != 1:
            raise RuntimeError(f"window crossed a VGGT geometry gauge boundary: {row['clip_hash']}")
        # The action at model index t drives the transition from frame t to t+1.
        action_start = start + T - 1
        action_window = actions[action_start : action_start + k]
        previous_grip = actions[action_start - 1 : action_start, 6]
        wrist_dropped = (not paired_views) or self._drop_wrist(sample_index)
        view_mask = np.ones((T, 2), dtype=np.bool_)
        if wrist_dropped:
            view_mask[:, 1] = False
        sample = {
            "s_in": torch.from_numpy(anchor[start : start + T].copy()),
            "s_wrist": torch.from_numpy(wrist[start : start + T].copy()),
            "view_mask": torch.from_numpy(view_mask),
            "s_tgt_codec": torch.from_numpy(anchor[start + T : end].copy()),
            "depth_tgt": torch.from_numpy(depth[start + T : end].copy()),
            "depth_conf_tgt": torch.from_numpy(depth_conf[start + T : end].copy()),
            "point_tgt": torch.from_numpy(points[start + T : end].copy()),
            "point_conf_tgt": torch.from_numpy(point_conf[start + T : end].copy()),
            "pose_geom_tgt": torch.from_numpy(pose[start + T : end].copy()),
            "action_tgt": torch.from_numpy(action_window.copy()),
            "action_tgt_norm": torch.from_numpy(
                ((action_window[:, :6] - self.action_mean) / self.action_std).copy()
            ),
            "action_prev_grip": torch.from_numpy(previous_grip.copy()),
            "c": torch.from_numpy(task.copy()),
            "clip_id": row["clip_hash"],
            "start": start,
            "dataset": row.get("v7_source", row["source"]),
            "action_frame_indices": torch.arange(action_start, action_start + k),
            "action_valid_count": len(actions),
            "action_contract_key": "robocasa365|5|wm3d_v7_base_delta_axisangle_gripclose_v1",
            "action_frame_offset": -1,
        }
        rgb_row = self.rgb_records.get(row["clip_hash"])
        if rgb_row is not None:
            with np.load(rgb_row["path"], allow_pickle=False) as rgb_archive:
                if str(rgb_archive["schema"].item()) != "wm3d_v7_rgb_sidecar_v1":
                    raise ValueError(f"unexpected RGB payload schema: {row['clip_hash']}")
                for key, expected in (
                    ("clip_hash", row["clip_hash"]),
                    ("split", row["split"]),
                    ("source", row["source"]),
                ):
                    if str(rgb_archive[key].item()) != str(expected):
                        raise ValueError(
                            f"RGB sidecar identity mismatch for {key}: {row['clip_hash']}"
                        )
                rgb = np.asarray(rgb_archive["rgb_anchor"], dtype=np.uint8)
            if rgb.shape != (int(row["model_frames"]), 256, 256, 3):
                raise RuntimeError(
                    f"RGB sidecar shape mismatch for {row['clip_hash']}: {rgb.shape}"
                )
            sample["rgb_in"] = torch.from_numpy(rgb[start : start + T].copy()).float().div_(255.0)
            sample["rgb_tgt"] = torch.from_numpy(rgb[start + T : end].copy()).float().div_(255.0)
        return sample


class V7SameRootBranchDataset(Dataset):
    """One true same-root K-branch calibration sample per simulator root."""

    def __init__(self, cfg: V7SameRootBranchDatasetConfig):
        self.cfg = cfg
        if cfg.split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported split: {cfg.split}")
        self.records = []
        with Path(cfg.index_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] != cfg.split:
                    continue
                if row.get("schema") != "wm3d_v7_same_root_branch_compact_v1":
                    raise ValueError(f"unexpected same-root schema: {row.get('root_id')}")
                if not row.get("same_root_current_runtime_exact"):
                    raise ValueError(f"same-root exactness missing: {row.get('root_id')}")
                if row.get("historical_runtime_reconstruction_exact") is not False:
                    raise ValueError(f"historical runtime provenance missing: {row.get('root_id')}")
                if row.get("pseudo_outcomes") is not False:
                    raise ValueError(f"pseudo same-root outcome forbidden: {row.get('root_id')}")
                if int(row["context_frames"]) != cfg.T or int(row["future_frames"]) != cfg.k:
                    raise ValueError(f"same-root horizon mismatch: {row.get('root_id')}")
                self.records.append(row)
        if not self.records:
            raise RuntimeError(f"same-root compact split {cfg.split} is empty")
        if cfg.action_stats is None:
            if cfg.require_action_stats:
                raise ValueError("same-root training requires train-only action_stats")
            self.action_mean = np.zeros(6, dtype=np.float32)
            self.action_std = np.ones(6, dtype=np.float32)
        else:
            with np.load(cfg.action_stats, allow_pickle=False) as stats:
                if str(stats["split"].item()) != "train":
                    raise ValueError("action statistics must be fit on train split only")
                self.action_mean = np.asarray(stats["mean"], dtype=np.float32)
                self.action_std = np.asarray(stats["std"], dtype=np.float32)
            if self.action_mean.shape != (6,) or self.action_std.shape != (6,):
                raise ValueError("invalid same-root action statistics")
            if np.any(self.action_std <= 0):
                raise ValueError("same-root action std must be positive")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return np.asarray(codes, dtype=np.int8).astype(np.float32) * np.asarray(
            scale, dtype=np.float32
        )

    def __getitem__(self, sample_index: int) -> dict:
        row = self.records[sample_index]
        with np.load(row["path"], allow_pickle=False) as archive:
            if str(archive["schema"].item()) != "wm3d_v7_same_root_branch_compact_v1":
                raise ValueError(f"same-root payload schema mismatch: {row['root_id']}")
            if str(archive["root_id"].item()) != row["root_id"]:
                raise ValueError(f"same-root payload identity mismatch: {row['root_id']}")
            context = self._dequantize(archive["anchor_codes"], archive["anchor_scale"])
            wrist = self._dequantize(archive["wrist_codes"], archive["wrist_scale"])
            factual = self._dequantize(archive["factual_codes"], archive["factual_scale"])
            branches = self._dequantize(archive["branch_codes"], archive["branch_scales"])
            branch_actions = np.asarray(archive["branch_actions"], dtype=np.float32)
            branch_valid = np.asarray(archive["branch_valid"], dtype=np.bool_)
            task = np.asarray(archive["task_emb"], dtype=np.float32)
            depth = np.asarray(archive["depth_tgt"], dtype=np.float32)
            depth_conf = np.asarray(archive["depth_conf_tgt"], dtype=np.float32)
            points = np.asarray(archive["point_tgt"], dtype=np.float32)
            point_conf = np.asarray(archive["point_conf_tgt"], dtype=np.float32)
            pose = np.asarray(archive["pose_geom_tgt"], dtype=np.float32)
            branch_rewards = np.asarray(archive["branch_rewards"], dtype=np.float32)
            branch_dones = np.asarray(archive["branch_dones"], dtype=np.bool_)
            branch_success = np.asarray(archive["branch_success"], dtype=np.bool_)
        if context.shape[0] != self.cfg.T or wrist.shape[0] != self.cfg.T:
            raise RuntimeError(f"same-root context length mismatch: {row['root_id']}")
        if factual.shape[0] != self.cfg.k or branches.shape[1] != self.cfg.k:
            raise RuntimeError(f"same-root target length mismatch: {row['root_id']}")
        if branch_actions.shape[:2] != branches.shape[:2] or branch_actions.shape[-1] != 7:
            raise RuntimeError(f"same-root action/target mismatch: {row['root_id']}")
        outcome_shape = branch_actions.shape[:2]
        if any(
            outcome.shape != outcome_shape
            for outcome in (branch_rewards, branch_dones, branch_success)
        ):
            raise RuntimeError(f"same-root outcome/action mismatch: {row['root_id']}")
        if branch_valid.shape != (branches.shape[0],) or not branch_valid.all():
            raise RuntimeError(f"same-root branch validity mismatch: {row['root_id']}")
        if self.cfg.require_task_emb and (task.shape != (2048,) or not np.any(task)):
            raise RuntimeError(f"missing same-root task embedding: {row['root_id']}")
        action_tgt = branch_actions[0]
        branch_action_cond = branch_actions.copy()
        branch_action_cond[:, :, :6] = (
            branch_action_cond[:, :, :6] - self.action_mean[None, None]
        ) / self.action_std[None, None]
        branch_action_cond[:, :, 6] = (branch_actions[:, :, 6] > 0.5).astype(np.float32)
        return {
            "s_in": torch.from_numpy(context.copy()),
            "s_wrist": torch.from_numpy(wrist.copy()),
            "view_mask": torch.ones((self.cfg.T, 2), dtype=torch.bool),
            "s_tgt_codec": torch.from_numpy(factual.copy()),
            "depth_tgt": torch.from_numpy(depth.copy()),
            "depth_conf_tgt": torch.from_numpy(depth_conf.copy()),
            "point_tgt": torch.from_numpy(points.copy()),
            "point_conf_tgt": torch.from_numpy(point_conf.copy()),
            "pose_geom_tgt": torch.from_numpy(pose.copy()),
            "action_tgt": torch.from_numpy(action_tgt.copy()),
            "action_tgt_norm": torch.from_numpy(
                ((action_tgt[:, :6] - self.action_mean) / self.action_std).copy()
            ),
            "action_prev_grip": torch.from_numpy(action_tgt[:1, 6].copy()),
            "branch_actions": torch.from_numpy(branch_action_cond),
            "branch_s_tgt_codec": torch.from_numpy(branches.copy()),
            "branch_valid": torch.from_numpy(branch_valid.copy()),
            "branch_rewards": torch.from_numpy(branch_rewards.copy()),
            "branch_dones": torch.from_numpy(branch_dones.copy()),
            "branch_success": torch.from_numpy(branch_success.copy()),
            "c": torch.from_numpy(task.copy()),
            "clip_id": row["root_id"],
            "start": 0,
            "dataset": "robocasa_same_root_current_runtime",
            "action_frame_indices": torch.arange(self.cfg.k),
            "action_valid_count": self.cfg.k,
            "action_contract_key": "robocasa365|5|wm3d_v7_base_delta_axisangle_gripclose_v1",
            "action_frame_offset": -1,
        }
