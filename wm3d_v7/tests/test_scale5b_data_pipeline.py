from __future__ import annotations

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from safetensors.torch import save_file
import torch

from wm3d_v3.data.scale5b_action import (
    ActionNormalization,
    RawActionSeries,
    align_auxiliary_tokens,
    align_grouped_actions,
)
from wm3d_v3.data.scale5b_codec import (
    JpegPackReader,
    JpegPackWriter,
    dequantize_per_vector,
    quantize_per_vector,
)
from wm3d_v3.data.scale5b_contracts import (
    ContractError,
    DatasetContract,
    canonical_sha256,
    sha256_file,
)
from wm3d_v3.data.scale5b_dataset import (
    Native5BSourceDataset,
    WindowLoaderConfig,
)
from wm3d_v3.data.scale5b_sources import (
    ActionColumnSpec,
    EpisodeDescriptor,
    ViewSegment,
    deterministic_split,
    plan_shard,
    validate_episode_inputs,
)


def _contract() -> DatasetContract:
    return DatasetContract.from_mapping(
        {
            "name": "synthetic_native5b",
            "feature_fps": 5.0,
            "action_fps": 30.0,
            "T": 24,
            "P": 144,
            "K": 16,
            "token_dim": 2048,
            "task_dim": 8,
            "num_views": 3,
            "max_action_groups": 2,
            "max_action_dim": 4,
            "action_substeps": 6,
            "max_group_id": 8,
            "max_embodiments": 4,
            "max_aux_tokens": 8,
            "aux_dim": 256,
            "max_aux_type_id": 64,
            "source_order": ["droid"],
            "sources": [
                {
                    "name": "droid",
                    "adapter": "lerobot",
                    "raw_root": "/synthetic",
                    "license_id": "synthetic-test",
                    "nominal_hours": 1.0,
                    "weight": 1,
                    "embodiment_names": ["arm"],
                    "split_seed": 7,
                    "train_fraction": 0.8,
                }
            ],
            "embodiments": [
                {
                    "name": "arm",
                    "embodiment_id": 0,
                    "views": ["head", "left_hand", "right_hand"],
                    "action_groups": [
                        {
                            "name": "arm",
                            "group_id": 0,
                            "dimensions": ["x", "y", "z", "yaw"],
                            "rate_hz": 30.0,
                            "control_mode": "delta_pose",
                        },
                        {
                            "name": "gripper",
                            "group_id": 1,
                            "dimensions": ["close"],
                            "rate_hz": 30.0,
                            "control_mode": "discrete_gripper",
                        },
                    ],
                    "auxiliary_modalities": [
                        {
                            "name": "force",
                            "type_id": 2,
                            "dimensions": ["fx", "fy"],
                            "rate_hz": 30.0,
                        }
                    ],
                }
            ],
        }
    )


def test_token_quantization_and_jpeg_pack_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(2)
    value = torch.randn(5, 3, 7, 32)
    quantized, scale = quantize_per_vector(value)
    restored = dequantize_per_vector(quantized, scale, dtype=torch.float32)
    assert float((restored - value).abs().max()) <= float(scale.max()) / 2 + 1e-6

    path = tmp_path / "rgb.jpgpack"
    writer = JpegPackWriter(path, quality=95)
    first = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
    second = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
    writer.append(first)
    writer.append(second)
    offsets, lengths = writer.close()
    reader = JpegPackReader(path)
    decoded = reader.decode(offsets[1].tolist(), lengths[1].tolist())
    reader.close()
    assert decoded.shape == second.shape
    assert float((decoded.float() - second.float()).abs().mean()) < 70.0


def test_grouped_action_alignment_and_deterministic_partition() -> None:
    contract = _contract()
    embodiment = contract.embodiments[0]
    timestamps = np.arange(300, dtype=np.float64) / 30.0
    series = {
        "arm": RawActionSeries(
            timestamps,
            np.stack(
                [
                    np.sin(timestamps),
                    np.cos(timestamps),
                    timestamps / 10.0,
                    timestamps * 0.0,
                ],
                axis=1,
            ).astype(np.float32),
        ),
        "gripper": RawActionSeries(
            timestamps,
            (np.arange(300) % 20 < 10).astype(np.float32)[:, None],
        ),
    }
    aligned = align_grouped_actions(
        visual_timestamps=np.arange(24, dtype=np.float64) / 5.0,
        group_series=series,
        embodiment=embodiment,
        normalizations={
            "arm": ActionNormalization(np.zeros(4), np.ones(4)),
            "gripper": ActionNormalization(np.zeros(1), np.ones(1)),
        },
        max_groups=2,
        max_action_dim=4,
        action_substeps=6,
    )
    assert aligned["action_values"].shape == (24, 2, 6, 4)
    assert aligned["action_dim_mask"][:, 0].all()
    assert aligned["action_dim_mask"][:, 1, :, 0].all()
    assert not aligned["action_dim_mask"][:, 1, :, 1:].any()
    assert not aligned["contact_mask"][:, 0].any()
    assert aligned["contact_mask"][:, 1].all()
    auxiliary = align_auxiliary_tokens(
        visual_timestamps=np.arange(24, dtype=np.float64) / 5.0,
        modality_series={
            "force": RawActionSeries(
                timestamps,
                np.stack((np.sin(timestamps), np.cos(timestamps)), axis=1),
            )
        },
        embodiment=embodiment,
        normalizations={
            "force": ActionNormalization(np.zeros(2), np.ones(2))
        },
        max_aux_tokens=8,
        aux_dim=256,
        max_aux_type_id=64,
    )
    assert auxiliary["aux_tokens"].shape == (24, 8, 256)
    assert auxiliary["aux_mask"][:, 0].all()
    assert auxiliary["aux_tokens"][:, 0, 2].eq(1).all()
    missing_values = np.stack(
        (np.sin(timestamps), np.cos(timestamps)),
        axis=1,
    ).astype(np.float32)
    missing_valid = np.ones_like(missing_values, dtype=bool)
    missing_values[30:35, 0] = np.nan
    missing_valid[30:35, 0] = False
    missing_values = np.where(missing_valid, missing_values, 0.0)
    with_missing = align_auxiliary_tokens(
        visual_timestamps=np.arange(24, dtype=np.float64) / 5.0,
        modality_series={
            "force": RawActionSeries(
                timestamps,
                missing_values,
                valid=missing_valid,
            )
        },
        embodiment=embodiment,
        normalizations={
            "force": ActionNormalization(np.zeros(2), np.ones(2))
        },
        max_aux_tokens=8,
        aux_dim=256,
        max_aux_type_id=64,
    )
    assert torch.isfinite(with_missing["aux_tokens"]).all()
    assert with_missing["aux_tokens"][5, 0, 66].item() == 0.0
    assert deterministic_split("droid", "episode-7", seed=3, train_fraction=0.9) == (
        deterministic_split("droid", "episode-7", seed=3, train_fraction=0.9)
    )
    assert plan_shard("episode-7", 128) == plan_shard("episode-7", 128)


def test_quantized_random_access_dataset(tmp_path: Path) -> None:
    contract = _contract()
    (tmp_path / "payload").mkdir()
    (tmp_path / "control").mkdir()
    index_dir = tmp_path / "indexes" / "val" / "droid"
    index_dir.mkdir(parents=True)
    torch.manual_seed(5)
    tokens = torch.randn(40, 3, 144, 2048, dtype=torch.float16)
    tokens[:, 2] = 500.0
    tokens_q, tokens_scale = quantize_per_vector(tokens)
    summary_q, summary_scale = quantize_per_vector(
        tokens.float().mean(dim=(1, 2))
    )
    jpeg_writer = JpegPackWriter(tmp_path / "payload" / "rgb.jpgpack", quality=95)
    for frame in range(40):
        image = torch.full((3, 3, 32, 32), frame * 5, dtype=torch.uint8)
        jpeg_writer.append(image)
    offsets, lengths = jpeg_writer.close()
    save_file(
        {
            "view_tokens_q": tokens_q,
            "view_tokens_scale": tokens_scale,
                "view_mask": torch.tensor(
                    [[True, True, False]], dtype=torch.bool
                ).expand(40, -1).contiguous(),
            "rgb_offsets": offsets,
            "rgb_lengths": lengths,
            "depth": torch.ones(40, 3, 144, dtype=torch.float16),
            "point": torch.zeros(40, 3, 144, 3, dtype=torch.float16),
                "geometry_confidence": torch.tensor(
                    [[[1.0], [1.0], [0.0]]], dtype=torch.float16
                ).expand(40, 3, 144).contiguous(),
            "camera_pose": torch.zeros(40, 3, 9),
            "frame_summary_q": summary_q,
            "frame_summary_scale": summary_scale,
            "aux_tokens": torch.zeros(40, 8, 256, dtype=torch.float16),
            "aux_mask": torch.ones(40, 8, dtype=torch.bool),
        },
        tmp_path / "payload" / "features.safetensors",
    )
    save_file(
        {
            "action_values": torch.zeros(40, 2, 6, 4),
            "action_dim_mask": torch.ones(40, 2, 6, 4, dtype=torch.bool),
            "contact": torch.zeros(40, 2, 6),
            "contact_mask": torch.tensor(
                [[[False] * 6, [True] * 6]], dtype=torch.bool
            ).expand(40, -1, -1).contiguous(),
        },
        tmp_path / "payload" / "actions.safetensors",
    )
    save_file(
        {"embeddings": torch.randn(1, 8, dtype=torch.bfloat16)},
        tmp_path / "control" / "task_embeddings.safetensors",
    )
    row = {
        "window_id": "0" * 64,
        "episode_id": "droid:0",
        "source": "droid",
        "feature_shard": "payload/features.safetensors",
        "action_shard": "payload/actions.safetensors",
        "rgb_pack": "payload/rgb.jpgpack",
        "frame_offset": 0,
        "action_offset": 0,
        "frame_count": 40,
        "episode_frame_start": 0,
        "episode_frame_stop": 40,
        "task_id": 0,
        "embodiment_id": 0,
        "action_group_ids": [0, 1],
        "action_group_mask": [True, True],
    }
    pq.write_table(pa.Table.from_pylist([row]), index_dir / "part-000000.parquet")
    dataset = Native5BSourceDataset(
        tmp_path,
        contract,
        source_name="droid",
        split="val",
        config=WindowLoaderConfig(
            rgb_decode_indices=(3, 7, 11, 15),
            memory_slots=2,
            memory_stride_frames=2,
            row_group_cache_size=1,
            task_cache_size=2,
            strict_shapes=True,
        ),
    )
    sample = dataset[0]
    restored = dequantize_per_vector(tokens_q, tokens_scale)
    assert torch.equal(sample["world_tokens"], restored[:24])
    assert torch.allclose(
        sample["target_tokens"].float(),
        restored[24:, :2].float().mean(dim=1),
        atol=2e-2,
        rtol=1e-3,
    )
    assert sample["target_rgb"].shape == (4, 3, 3, 32, 32)
    assert sample["target_action_values"].shape == (16, 2, 6, 4)
    assert not sample["target_contact_mask"][:, 0].any()
    assert sample["target_contact_mask"][:, 1].all()


def test_episode_input_validation_checks_parquet_width_and_video(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array([0.0, 0.1, 0.2, 0.3]),
                "episode_index": pa.array([7, 7, 7, 7]),
                "action": pa.array(
                    [[0.0] * 7, [0.1] * 7, [0.2] * 7, [0.3] * 7]
                ),
            }
        ),
        raw / "episode.parquet",
    )
    (raw / "head.mp4").write_bytes(b"not-decoded-by-metadata-validator")
    action_columns = (
        ActionColumnSpec("arm", "action", (0, 1, 2, 3, 4, 5)),
        ActionColumnSpec("gripper", "action", (6,), discrete=True),
    )
    episode = EpisodeDescriptor(
        source="legacy",
        episode_id="legacy:7",
        episode_index=7,
        embodiment="single_arm",
        split="train",
        task_text="pick",
        raw_root=str(raw),
        data_relative_path="episode.parquet",
        data_row_start=0,
        data_row_stop=4,
        timestamp_column="timestamp",
        episode_column="episode_index",
        source_fps=10.0,
        duration_seconds=0.4,
        views=(
            ViewSegment("head", "head", "head.mp4", 0.0, 0.4),
            ViewSegment("left_hand", None, None, 0.0, 0.4),
            ViewSegment("right_hand", None, None, 0.0, 0.4),
        ),
        action_columns=action_columns,
    )
    report = validate_episode_inputs((episode,))
    assert report["episodes"] == 1
    assert report["unique_data_files"] == 1
    assert report["unique_video_files"] == 1
    invalid_mapping = episode.as_dict()
    invalid_mapping["action_columns"][0]["indices"] = [0, 1, 2, 3, 4, 9]
    invalid = EpisodeDescriptor.from_mapping(invalid_mapping)
    with pytest.raises(ContractError, match="does not cover"):
        validate_episode_inputs((invalid,))


def test_action_statistics_global_budget_is_bounded_and_deterministic() -> None:
    module = runpy.run_path("scripts/scale5b/build_action_stats.py")
    allocate = module["_episode_sample_positions"]
    episodes = [
        SimpleNamespace(
            episode_id="source:000",
            data_row_start=0,
            data_row_stop=11,
        ),
        SimpleNamespace(
            episode_id="source:001",
            data_row_start=20,
            data_row_stop=37,
        ),
        SimpleNamespace(
            episode_id="source:002",
            data_row_start=0,
            data_row_stop=5,
        ),
    ]
    first = allocate(episodes, sample_budget=13, seed_text="sealed-plan")
    second = allocate(episodes, sample_budget=13, seed_text="sealed-plan")
    assert sum(len(value) for value in first.values()) == 13
    assert first.keys() == second.keys()
    for episode in episodes:
        left = first.get(episode.episode_id, np.empty(0, dtype=np.int64))
        right = second.get(episode.episode_id, np.empty(0, dtype=np.int64))
        assert np.array_equal(left, right)
        assert len(left) == len(np.unique(left))
        assert np.all(left >= 0)
        assert np.all(
            left < episode.data_row_stop - episode.data_row_start
        )
    all_rows = allocate(episodes, sample_budget=10_000, seed_text="all")
    assert sum(len(value) for value in all_rows.values()) == 33


def test_encoded_part_reentry_binds_complete_lineage(tmp_path: Path) -> None:
    module = runpy.run_path("scripts/scale5b/encode_shard.py")
    verify = module["_verify_existing_part"]
    part = tmp_path / "part-00003-000007"
    part.mkdir()
    payload_names = (
        "features.safetensors",
        "actions.safetensors",
        "rgb.jpgpack",
        "windows.parquet",
    )
    for index, name in enumerate(payload_names):
        (part / name).write_bytes(bytes([index + 1]) * (index + 2))
    lineage = {
        "episode_plan_sha256": "1" * 64,
        "dataset_contract_sha256": "2" * 64,
        "action_stats_sha256": "3" * 64,
        "task_index_sha256": "4" * 64,
        "encoder_asset_receipt_sha256": "5" * 64,
    }
    encoder_identity = {
        "vggt_model": "facebook/VGGT-1B",
        "vggt_revision": "a" * 40,
    }
    manifest = {
        "schema": "wm3d_v7_native5b_encoded_part_v2",
        "part_name": part.name,
        "part_index": 7,
        "worker_shard_id": 3,
        "worker_num_shards": 16,
        **lineage,
        **encoder_identity,
        "frames": 40,
        "windows": 1,
        "files": {
            name: {
                "size": (part / name).stat().st_size,
                "sha256": sha256_file(part / name),
            }
            for name in payload_names
        },
    }
    manifest_path = part / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    (part / "COMMITTED.json").write_text(
        json.dumps(
            {
                "schema": "wm3d_v7_native5b_encoded_part_commit_v2",
                "part_name": part.name,
                "manifest_sha256": sha256_file(manifest_path),
                "manifest_content_sha256": canonical_sha256(manifest),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    arguments = {
        "expected_part_name": part.name,
        "expected_part_index": 7,
        "expected_shard_id": 3,
        "expected_num_shards": 16,
        "expected_lineage": lineage,
    }
    assert verify(part, **arguments)
    merge_module = runpy.run_path("scripts/scale5b/merge_and_seal.py")
    merged = merge_module["_verify_part"](
        part,
        {**lineage, **encoder_identity},
    )
    assert merged == manifest
    drifted = {**lineage, "encoder_asset_receipt_sha256": "6" * 64}
    assert not verify(part, **{**arguments, "expected_lineage": drifted})
    (part / "unexpected.bin").write_bytes(b"forbidden")
    assert not verify(part, **arguments)


def test_merge_binds_worker_summaries_and_rejects_payload_aliases(
    tmp_path: Path,
) -> None:
    module = runpy.run_path("scripts/scale5b/merge_and_seal.py")
    load_receipts = module["_load_worker_receipts"]
    committed_names = module["_committed_part_names"]
    receipt_root = tmp_path / "dataset"
    receipt_directory = receipt_root / "receipts" / "encode_workers"
    receipt_directory.mkdir(parents=True)
    part_name = "part-00000-000000"
    receipt = {
        "schema": "wm3d_v7_native5b_encode_worker_receipt_v1",
        "shard_id": 0,
        "num_shards": 1,
        "episode_plan_sha256": "1" * 64,
        "dataset_contract_sha256": "2" * 64,
        "action_stats_sha256": "3" * 64,
        "task_index_sha256": "4" * 64,
        "encoder_asset_receipt_sha256": "5" * 64,
        "vggt_model": "facebook/VGGT-1B",
        "vggt_revision": "a" * 40,
        "parts": [{"part": part_name, "frames": 512, "windows": 473}],
    }
    receipt_path = receipt_directory / "worker_00000.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    lineage, summaries = load_receipts(receipt_root, 1)
    assert lineage["worker_num_shards"] == 1
    assert summaries == {part_name: {"frames": 512, "windows": 473}}

    receipt["parts"][0]["part"] = "part-00000-000002"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="not contiguous"):
        load_receipts(receipt_root, 1)

    parts_root = receipt_root / "payload" / "parts"
    parts_root.mkdir(parents=True)
    (parts_root / part_name).mkdir()
    (parts_root / ".part-incomplete-evidence").mkdir()
    assert committed_names(receipt_root) == {part_name}
    outside = tmp_path / "outside"
    outside.mkdir()
    (parts_root / "part-00000-000001").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="unexpected non-committed"):
        committed_names(receipt_root)
