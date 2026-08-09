from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.mixed_source_dataset import partition_v7_compact_dataset
from wm3d_v3.data.v7_compact_dataset import (
    V7CompactDatasetConfig,
    V7CompactWindowDataset,
    V7SameRootBranchDataset,
    V7SameRootBranchDatasetConfig,
)


def test_compact_partition_uses_v7_source_when_dataset_is_absent() -> None:
    class CompactRecords(Dataset):
        records = [
            {"v7_source": "atomic", "source": "robocasa365"},
            {"v7_source": "composite", "source": "robocasa365"},
            {"v7_source": "mg", "source": "robocasa365"},
        ]
        index = [(0, 0), (1, 0), (2, 0)]

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, index: int) -> int:
            return index

    partitions = partition_v7_compact_dataset(
        CompactRecords(), ("atomic", "composite", "mg")
    )
    assert {name: list(part.indices) for name, part in partitions.items()} == {
        "atomic": [0], "composite": [1], "mg": [2]
    }


def _write_clip(path, split: str, frames: int = 26) -> None:
    codes = np.zeros((frames, 64, 384), dtype=np.int8)
    codes[..., 0] = np.arange(frames, dtype=np.int8)[:, None]
    scale = np.ones((frames, 1, 1), dtype=np.float16)
    actions = np.zeros((frames, 7), dtype=np.float32)
    actions[:, 0] = np.arange(frames)
    actions[:, 6] = 1.0
    np.savez_compressed(
        path,
        schema=np.asarray("wm3d_v7_compact_geom_v3"),
        clip_hash=np.asarray(f"{split}_clip"),
        split=np.asarray(split),
        source=np.asarray("robocasa365"),
        action_adapter_version=np.asarray("wm3d_v7_base_delta_axisangle_gripclose_v1"),
        action_audit_sha256=np.asarray("audit_sha"),
        anchor_codes=codes,
        anchor_scale=scale,
        wrist_codes=codes,
        wrist_scale=scale,
        task_emb=np.ones(2048, dtype=np.float16),
        depth_patch=np.ones((frames, 8, 8), dtype=np.float16),
        depth_conf_patch=np.ones((frames, 8, 8), dtype=np.float16),
        point_patch=np.ones((frames, 8, 8, 3), dtype=np.float16),
        point_conf_patch=np.ones((frames, 8, 8), dtype=np.float16),
        pose_enc=np.ones((frames, 9), dtype=np.float16),
        geometry_segment_id=np.zeros(frames, dtype=np.int16),
        action_valid_mask=np.ones(frames, dtype=np.bool_),
        actions=actions,
    )


def _write_rgb_sidecar(path, split: str) -> None:
    rgb = np.zeros((26, 256, 256, 3), dtype=np.uint8)
    rgb[:, 0, 0, 0] = np.arange(26, dtype=np.uint8)
    np.savez_compressed(
        path,
        schema=np.asarray("wm3d_v7_rgb_sidecar_v1"),
        clip_hash=np.asarray(f"{split}_clip"),
        split=np.asarray(split),
        source=np.asarray("robocasa365"),
        rgb_anchor=rgb,
    )


def test_compact_loader_keeps_future_action_alignment_and_paired_views(tmp_path):
    train = tmp_path / "train.npz"
    val = tmp_path / "val.npz"
    _write_clip(train, "train")
    _write_clip(val, "val")
    index = tmp_path / "index.jsonl"
    rows = [
        {
            "clip_hash": "train_clip",
            "split": "train",
            "source": "robocasa365",
            "path": str(train),
            "model_frames": 26,
            "geometry_segments": [[0, 26]],
            "paired_views": True,
            "action_valid": True,
            "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
            "action_audit_sha256": "audit_sha",
            "pseudo_outcomes": False,
            "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
        },
        {
            "clip_hash": "val_clip",
            "split": "val",
            "source": "robocasa365",
            "path": str(val),
            "model_frames": 26,
            "geometry_segments": [[0, 26]],
            "paired_views": True,
            "action_valid": True,
            "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
            "action_audit_sha256": "audit_sha",
            "pseudo_outcomes": False,
            "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
        },
    ]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = V7CompactWindowDataset(
        V7CompactDatasetConfig(
            index_path=index, split="train", T=16, k=8, stride=2,
            require_action_stats=False,
            policy_action_history_len=1,
        )
    )
    sample = dataset[0]
    assert sample["s_in"].shape == (16, 64, 384)
    assert sample["s_wrist"].shape == (16, 64, 384)
    assert sample["s_tgt_codec"].shape == (8, 64, 384)
    assert sample["depth_tgt"].shape == (8, 8, 8)
    assert sample["point_tgt"].shape == (8, 8, 8, 3)
    assert sample["pose_geom_tgt"].shape == (8, 9)
    assert sample["view_mask"].all()
    # Context ends at token 15, therefore action 15 drives future token 16.
    assert sample["action_tgt"][:, 0].tolist() == list(range(15, 23))
    assert sample["action_prev_grip"].tolist() == [1.0]
    assert sample["action_pose_mean"].tolist() == [0.0] * 6
    assert sample["action_pose_std"].tolist() == [1.0] * 6
    assert sample["action_history"].shape == (1, 7)
    assert sample["action_history"][0, 0].item() == 14.0
    assert sample["action_history"][0, 6].item() == 1.0


def test_compact_loader_view_dropout_is_epoch_deterministic(tmp_path):
    train = tmp_path / "train.npz"
    _write_clip(train, "train")
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "clip_hash": "train_clip",
                "split": "train",
                "source": "robocasa365",
                "path": str(train),
                "model_frames": 26,
                "geometry_segments": [[0, 26]],
                "paired_views": True,
                "action_valid": True,
                "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
                "action_audit_sha256": "audit_sha",
                "pseudo_outcomes": False,
                "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
            }
        )
        + "\n"
    )
    dataset = V7CompactWindowDataset(
        V7CompactDatasetConfig(
            index_path=index,
            split="train",
            T=16,
            k=8,
            stride=2,
            view_dropout=1.0,
            require_action_stats=False,
        )
    )
    first = dataset[0]["view_mask"]
    second = dataset[0]["view_mask"]
    assert first[:, 0].all() and not first[:, 1].any()
    assert np.array_equal(first.numpy(), second.numpy())


def test_compact_loader_joins_aligned_rgb_sidecar(tmp_path):
    train = tmp_path / "train.npz"
    rgb = tmp_path / "train_rgb.npz"
    _write_clip(train, "train")
    _write_rgb_sidecar(rgb, "train")
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "clip_hash": "train_clip",
                "split": "train",
                "source": "robocasa365",
                "path": str(train),
                "model_frames": 26,
                "geometry_segments": [[0, 26]],
                "paired_views": True,
                "action_valid": True,
                "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
                "action_audit_sha256": "audit_sha",
                "pseudo_outcomes": False,
                "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
            }
        )
        + "\n"
    )
    rgb_index = tmp_path / "rgb_index.jsonl"
    rgb_index.write_text(
        json.dumps(
            {
                "schema": "wm3d_v7_rgb_sidecar_v1",
                "clip_hash": "train_clip",
                "split": "train",
                "source": "robocasa365",
                "path": str(rgb),
                "model_frames": 26,
            }
        )
        + "\n"
    )
    dataset = V7CompactWindowDataset(
        V7CompactDatasetConfig(
            index_path=index,
            split="train",
            T=16,
            k=8,
            stride=2,
            require_action_stats=False,
            rgb_sidecar_indices=(rgb_index,),
            require_rgb_sidecar=True,
        )
    )
    sample = dataset[0]
    assert sample["rgb_in"].shape == (16, 256, 256, 3)
    assert sample["rgb_tgt"].shape == (8, 256, 256, 3)
    assert torch.isclose(sample["rgb_in"][15, 0, 0, 0], torch.tensor(15.0 / 255.0))
    assert torch.isclose(sample["rgb_tgt"][0, 0, 0, 0], torch.tensor(16.0 / 255.0))


def test_compact_loader_filters_gauge_crossing_windows_and_supports_mono(tmp_path):
    train = tmp_path / "train.npz"
    _write_clip(train, "train")
    row = {
        "clip_hash": "train_clip",
        "split": "train",
        "source": "robocasa365",
        "path": str(train),
        "model_frames": 26,
        "geometry_segments": [[0, 24], [24, 26]],
        "paired_views": False,
        "action_valid": True,
        "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
        "action_audit_sha256": "audit_sha",
        "pseudo_outcomes": False,
        "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
    }
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps(row) + "\n")
    dataset = V7CompactWindowDataset(
        V7CompactDatasetConfig(
            index_path=index,
            split="train",
            T=16,
            k=8,
            stride=2,
            require_action_stats=False,
        )
    )
    assert len(dataset) == 1
    sample = dataset[0]
    assert not sample["view_mask"][:, 1].any()
    assert not sample["s_wrist"].any()


def test_compact_action_only_supports_k32_beyond_future_geometry_segment(tmp_path):
    train = tmp_path / "train_k32.npz"
    _write_clip(train, "train", frames=64)
    row = {
        "clip_hash": "train_clip",
        "split": "train",
        "source": "robocasa365",
        "path": str(train),
        "model_frames": 64,
        # S0 geometry supervision ends after T+8, but the episode action
        # stream continues and is valid for a K=32 policy target.
        "geometry_segments": [[0, 24]],
        "paired_views": True,
        "action_valid": True,
        "action_adapter_version": "wm3d_v7_base_delta_axisangle_gripclose_v1",
        "action_audit_sha256": "audit_sha",
        "pseudo_outcomes": False,
        "geometry_teacher": {"pseudo_teacher": True, "confidence_stored": True},
    }
    index = tmp_path / "index_k32.jsonl"
    index.write_text(json.dumps(row) + "\n")
    dataset = V7CompactWindowDataset(
        V7CompactDatasetConfig(
            index_path=index,
            split="train",
            T=16,
            k=32,
            stride=2,
            require_action_stats=False,
            action_only=True,
        )
    )
    assert len(dataset) == 5
    sample = dataset[0]
    assert sample["s_in"].shape == (16, 64, 384)
    assert sample["action_tgt"].shape == (32, 7)
    assert sample["action_tgt"][:, 0].tolist() == list(range(15, 47))
    assert "s_tgt_codec" not in sample
    assert "depth_tgt" not in sample
    assert "point_tgt" not in sample


def test_same_root_loader_returns_k_true_branches(tmp_path):
    path = tmp_path / "root.npz"
    context_codes = np.ones((16, 64, 384), dtype=np.int8)
    target_codes = np.ones((8, 64, 384), dtype=np.int8)
    branch_codes = np.stack([target_codes * value for value in (1, 2, 3, 4)])
    context_scale = np.ones((16, 1, 1), dtype=np.float16)
    target_scale = np.ones((8, 1, 1), dtype=np.float16)
    branch_scale = np.ones((4, 8, 1, 1), dtype=np.float16)
    branch_actions = np.zeros((4, 8, 7), dtype=np.float32)
    branch_actions[:, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    branch_actions[0, :, 6] = -1.0
    branch_actions[1:, :, 6] = 1.0
    np.savez_compressed(
        path,
        schema=np.asarray("wm3d_v7_same_root_branch_compact_v1"),
        root_id=np.asarray("root0"),
        anchor_codes=context_codes,
        anchor_scale=context_scale,
        wrist_codes=context_codes,
        wrist_scale=context_scale,
        factual_codes=target_codes,
        factual_scale=target_scale,
        branch_codes=branch_codes,
        branch_scales=branch_scale,
        branch_actions=branch_actions,
        branch_valid=np.ones(4, dtype=np.bool_),
        task_emb=np.ones(2048, dtype=np.float16),
        depth_tgt=np.ones((8, 8, 8), dtype=np.float16),
        depth_conf_tgt=np.ones((8, 8, 8), dtype=np.float16),
        point_tgt=np.ones((8, 8, 8, 3), dtype=np.float16),
        point_conf_tgt=np.ones((8, 8, 8), dtype=np.float16),
        pose_geom_tgt=np.ones((8, 9), dtype=np.float16),
        branch_rewards=np.zeros((4, 8), dtype=np.float32),
        branch_dones=np.zeros((4, 8), dtype=np.bool_),
        branch_success=np.zeros((4, 8), dtype=np.bool_),
    )
    index = tmp_path / "branches.jsonl"
    index.write_text(
        json.dumps(
            {
                "schema": "wm3d_v7_same_root_branch_compact_v1",
                "root_id": "root0",
                "path": str(path),
                "split": "train",
                "branches": 4,
                "context_frames": 16,
                "future_frames": 8,
                "same_root_current_runtime_exact": True,
                "historical_runtime_reconstruction_exact": False,
                "pseudo_outcomes": False,
            }
        )
        + "\n"
    )
    dataset = V7SameRootBranchDataset(
        V7SameRootBranchDatasetConfig(
            index_path=index,
            split="train",
            require_action_stats=False,
        )
    )
    sample = dataset[0]
    assert sample["s_in"].shape == (16, 64, 384)
    assert sample["s_tgt_codec"].shape == (8, 64, 384)
    assert sample["branch_s_tgt_codec"].shape == (4, 8, 64, 384)
    assert sample["branch_actions"].shape == (4, 8, 7)
    assert sample["branch_valid"].all()
    assert sample["branch_s_tgt_codec"][:, 0, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert torch.equal(
        sample["branch_actions"][0],
        make_action_condition(sample["action_tgt"], sample["action_tgt_norm"]),
    )
    assert sample["branch_actions"][0, :, 6].tolist() == [0.0] * 8
    assert sample["branch_actions"][1, :, 6].tolist() == [1.0] * 8
    assert "action_prev_grip" not in sample
    assert sample["action_pose_mean"].tolist() == [0.0] * 6
    assert sample["action_pose_std"].tolist() == [1.0] * 6
