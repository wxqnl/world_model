from __future__ import annotations

import io
import json
from pathlib import Path
import platform
import runpy
import shutil
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from safetensors.torch import save_file
import torch

from wm3d.data.action import (
    ActionNormalization,
    RawActionSeries,
    align_auxiliary_tokens,
    align_grouped_actions,
)
from wm3d.data.codec import (
    JpegPackReader,
    JpegPackWriter,
    dequantize_per_vector,
    quantize_per_vector,
)
from wm3d.data.contracts import (
    ContractError,
    DatasetContract,
    canonical_sha256,
    sha256_file,
)
from wm3d.data.dataset import (
    SourceDataset,
    WindowLoaderConfig,
)
from wm3d.data.sources import (
    ActionColumnSpec,
    EpisodeDescriptor,
    SourceLayout,
    ViewSegment,
    deterministic_split,
    plan_shard,
    scan_lerobot,
    scan_lerobot_collection,
    validate_episode_inputs,
)


def _contract() -> DatasetContract:
    return DatasetContract.from_mapping(
        {
            "name": "synthetic_wm3d",
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


def test_encoder_resize_preserves_exact_unit_interval() -> None:
    resize = runpy.run_path("scripts/data/cache_vggt_shard.py")["_resize_views"]
    white = torch.full(
        (2, 3, 3, 480, 640),
        255,
        dtype=torch.uint8,
    )
    resized = resize(white, 384)
    assert float(resized.min()) >= 0.0
    assert float(resized.max()) <= 1.0


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
        normalizations={"force": ActionNormalization(np.zeros(2), np.ones(2))},
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
        normalizations={"force": ActionNormalization(np.zeros(2), np.ones(2))},
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
    summary_q, summary_scale = quantize_per_vector(tokens.float().mean(dim=(1, 2)))
    jpeg_writer = JpegPackWriter(tmp_path / "payload" / "rgb.jpgpack", quality=95)
    for frame in range(40):
        image = torch.full((3, 3, 32, 32), frame * 5, dtype=torch.uint8)
        jpeg_writer.append(image)
    offsets, lengths = jpeg_writer.close()
    save_file(
        {
            "view_tokens_q": tokens_q,
            "view_tokens_scale": tokens_scale,
            "view_mask": torch.tensor([[True, True, False]], dtype=torch.bool)
            .expand(40, -1)
            .contiguous(),
            "rgb_offsets": offsets,
            "rgb_lengths": lengths,
            "depth": torch.ones(40, 3, 144, dtype=torch.float16),
            "point": torch.zeros(40, 3, 144, 3, dtype=torch.float16),
            "geometry_confidence": torch.tensor(
                [[[1.0], [1.0], [0.0]]], dtype=torch.float16
            )
            .expand(40, 3, 144)
            .contiguous(),
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
            "contact_mask": torch.tensor([[[False] * 6, [True] * 6]], dtype=torch.bool)
            .expand(40, -1, -1)
            .contiguous(),
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
    dataset = SourceDataset(
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


def test_lerobot_collection_namespaces_restarted_episode_indices(
    tmp_path: Path,
) -> None:
    collection = tmp_path / "collection"

    def make_nested(relative: str, value: float) -> Path:
        root = collection / relative
        (root / "meta").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "videos" / "chunk-000" / "rgb").mkdir(parents=True)
        (root / "meta" / "info.json").write_text(
            json.dumps({"fps": 10.0}), encoding="utf-8"
        )
        (root / "meta" / "episodes.jsonl").write_text(
            json.dumps(
                {
                    "episode_index": 0,
                    "length": 4,
                    "task": f"task-{relative}",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pq.write_table(
            pa.table(
                {
                    "timestamp": [0.0, 0.1, 0.2, 0.3],
                    "episode_index": [0, 0, 0, 0],
                    "action": [[value] * 7] * 4,
                }
            ),
            root / "data" / "chunk-000" / "episode_000000.parquet",
        )
        (root / "videos" / "chunk-000" / "rgb" / "episode_000000.mp4").write_bytes(
            b"metadata-only-video"
        )
        return root

    first = make_nested("task-a/dataset", 0.0)
    second = make_nested("task-b/dataset", 1.0)
    layout = SourceLayout.from_mapping(
        {
            "source": "agibot",
            "adapter": "lerobot_collection",
            "embodiment": "wholebody",
            "view_keys": {
                "head": "rgb",
                "left_hand": None,
                "right_hand": None,
            },
            "action_columns": [
                {
                    "group_name": "arm",
                    "column": "action",
                    "indices": [0, 1, 2, 3, 4, 5],
                },
                {
                    "group_name": "gripper",
                    "column": "action",
                    "indices": [6],
                    "discrete": True,
                },
            ],
        }
    )
    episodes = scan_lerobot_collection(
        collection,
        layout,
        split_seed=17,
        train_fraction=0.8,
    )
    assert len(episodes) == 2
    assert len({episode.episode_id for episode in episodes}) == 2
    assert {Path(episode.raw_root) for episode in episodes} == {first, second}
    assert all(episode.episode_index == 0 for episode in episodes)
    assert all(episode.episode_id.startswith("agibot:") for episode in episodes)


def test_lerobot_v3_global_rows_are_rebased_per_shared_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lerobot"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "rgb" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 10.0,
                "data_path": (
                    "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                ),
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
            }
        ),
        encoding="utf-8",
    )
    metadata = []
    for episode_index, file_index, dataset_start, video_start in (
        (0, 0, 0, 0.0),
        (1, 0, 4, 0.4),
        (2, 1, 8, 0.0),
    ):
        metadata.append(
            {
                "episode_index": episode_index,
                "length": 4,
                "data/chunk_index": 0,
                "data/file_index": file_index,
                "dataset_from_index": dataset_start,
                "dataset_to_index": dataset_start + 4,
                "videos/rgb/chunk_index": 0,
                "videos/rgb/file_index": file_index,
                "videos/rgb/from_timestamp": video_start,
                "videos/rgb/to_timestamp": video_start + 0.4,
                "tasks": ["move the arm"],
            }
        )
    pq.write_table(
        pa.Table.from_pylist(metadata),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    for file_index, episode_indices in ((0, (0, 1)), (1, (2,))):
        timestamps: list[float] = []
        episodes: list[int] = []
        actions: list[list[float]] = []
        for episode_index in episode_indices:
            timestamps.extend([0.0, 0.1, 0.2, 0.3])
            episodes.extend([episode_index] * 4)
            actions.extend([[float(episode_index)] * 7] * 4)
        pq.write_table(
            pa.table(
                {
                    "timestamp": timestamps,
                    "episode_index": episodes,
                    "action": actions,
                }
            ),
            root / "data" / "chunk-000" / f"file-{file_index:03d}.parquet",
        )
        (root / "videos" / "rgb" / "chunk-000" / f"file-{file_index:03d}.mp4").write_bytes(
            b"metadata-only-video"
        )
    layout = SourceLayout.from_mapping(
        {
            "source": "aloha",
            "adapter": "lerobot",
            "embodiment": "bimanual",
            "view_keys": {
                "head": "rgb",
                "left_hand": None,
                "right_hand": None,
            },
            "action_columns": [
                {
                    "group_name": "arm",
                    "column": "action",
                    "indices": [0, 1, 2, 3, 4, 5],
                },
                {
                    "group_name": "gripper",
                    "column": "action",
                    "indices": [6],
                    "discrete": True,
                },
            ],
        }
    )
    episodes = scan_lerobot(
        root,
        layout,
        split_seed=7,
        train_fraction=0.8,
    )
    assert [
        (episode.data_relative_path, episode.data_row_start, episode.data_row_stop)
        for episode in episodes
    ] == [
        ("data/chunk-000/file-000.parquet", 0, 4),
        ("data/chunk-000/file-000.parquet", 4, 8),
        ("data/chunk-000/file-001.parquet", 0, 4),
    ]
    assert validate_episode_inputs(episodes)["episodes"] == 3


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
                "action": pa.array([[0.0] * 7, [0.1] * 7, [0.2] * 7, [0.3] * 7]),
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
    module = runpy.run_path("scripts/data/build_action_stats.py")
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
        assert np.all(left < episode.data_row_stop - episode.data_row_start)
    all_rows = allocate(episodes, sample_budget=10_000, seed_text="all")
    assert sum(len(value) for value in all_rows.values()) == 33


def test_encoded_part_reentry_binds_complete_lineage(tmp_path: Path) -> None:
    module = runpy.run_path("scripts/data/cache_vggt_shard.py")
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
        "schema": "wm3d_v7_encoded_part_v2",
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
                "schema": "wm3d_v7_encoded_part_commit_v2",
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
    merge_module = runpy.run_path("scripts/data/seal_dataset.py")
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
    module = runpy.run_path("scripts/data/seal_dataset.py")
    load_receipts = module["_load_worker_receipts"]
    committed_names = module["_committed_part_names"]
    receipt_root = tmp_path / "dataset"
    receipt_directory = receipt_root / "receipts" / "encode_workers"
    receipt_directory.mkdir(parents=True)
    part_name = "part-00000-000000"
    receipt = {
        "schema": "wm3d_v7_encode_worker_receipt_v1",
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


def test_raw_source_lock_is_immutable_and_gated_dry_run_needs_no_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = runpy.run_path("scripts/data/download_raw_snapshots.py")
    lock = {
        "schema": "wm3d_v7_raw_sources_lock_v1",
        "sources": {
            "beta": {
                "repo_id": "agibot-world/AgiBotWorld-Beta",
                "repo_type": "dataset",
                "revision": "a" * 40,
                "target_subdir": "beta",
                "gated": True,
                "allow_patterns": [],
                "ignore_patterns": [],
            }
        },
    }
    lock_path = tmp_path / "raw.lock.yaml"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    assert module["_load_lock"](lock_path)["sources"]["beta"]["gated"]

    tool_root = tmp_path / "tool"
    (tool_root / "scripts").mkdir(parents=True)
    tool = tool_root / "scripts" / "convert_to_lerobot.py"
    tool.write_text("# official converter\n", encoding="utf-8")
    files, total_bytes, payload_sha256 = module["_payload_inventory"](
        tool_root,
        hash_content=True,
    )
    assert files == 1
    assert total_bytes == tool.stat().st_size
    assert payload_sha256 == {"scripts/convert_to_lerobot.py": sha256_file(tool)}
    receipt = {
        "schema": module["RECEIPT_SCHEMA"],
        "complete": True,
        "source": "tool",
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "resolved_revision": "a" * 40,
        "target": str(tool_root),
        "allow_patterns": [],
        "ignore_patterns": [],
        "payload_files": files,
        "payload_bytes": total_bytes,
        "payload_sha256": payload_sha256,
    }
    receipt_path = tool_root / ".wm3d_v7_download_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    source = {
        "repo_id": "owner/repo",
        "revision": "a" * 40,
        "allow_patterns": [],
        "ignore_patterns": [],
        "materialization": "vendor_tool_bundle",
    }
    assert module["_receipt_matches"](receipt_path, "tool", source, tool_root)
    tool.write_text("# modified converter\n", encoding="utf-8")
    assert not module["_receipt_matches"](
        receipt_path,
        "tool",
        source,
        tool_root,
    )

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_raw_snapshots.py",
            "--lock",
            str(lock_path),
            "--raw-root",
            str(raw_root),
            "--dry-run",
        ],
    )
    module["main"]()
    result = json.loads(capsys.readouterr().out)
    assert result["pass"] is True
    assert result["results"][0]["status"] == "dry_run"
    assert not (raw_root / "beta").exists()

    lock["sources"]["beta"]["revision"] = "main"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="40 位小写提交 SHA"):
        module["_load_lock"](lock_path)


def test_archive_extractor_rejects_path_escape_and_special_names() -> None:
    module = runpy.run_path("scripts/data/safe_extract_lerobot_collection.py")
    member_relative = module["_member_relative"]
    assert member_relative("task/meta/info.json") == Path("task/meta/info.json")
    for unsafe in ("", "/absolute/file", "../escape", "task/../../escape"):
        with pytest.raises(ValueError, match="归档成员路径不安全"):
            member_relative(unsafe)


def test_archive_collection_finalize_binds_download_and_exact_archive_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = runpy.run_path("scripts/data/safe_extract_lerobot_collection.py")
    snapshot = tmp_path / "snapshot"
    archive_root = snapshot / "ImitationLearning"
    archive_root.mkdir(parents=True)
    archive = archive_root / "part-000.tar"
    with tarfile.open(archive, mode="w") as handle:
        payload = b'{"fps": 5}'
        info = tarfile.TarInfo("dataset/meta/info.json")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))
    download_receipt = snapshot / ".wm3d_v7_download_receipt.json"
    download_receipt.write_text(
        json.dumps(
            {
                "schema": "wm3d_v7_raw_download_receipt_v1",
                "complete": True,
                "target": str(snapshot),
                "revision": "a" * 40,
                "resolved_revision": "a" * 40,
                "payload_files": 1,
                "payload_bytes": archive.stat().st_size,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "collection"
    output.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safe_extract_lerobot_collection.py",
            "--archive-root",
            str(archive_root),
            "--output-root",
            str(output),
        ],
    )
    module["main"]()
    assert json.loads(capsys.readouterr().out)["pass"] is True
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safe_extract_lerobot_collection.py",
            "--archive-root",
            str(archive_root),
            "--output-root",
            str(output),
            "--finalize",
            "--download-receipt",
            str(download_receipt),
        ],
    )
    module["main"]()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "finalized"
    assert result["archives"] == 1
    assert result["lerobot_roots"] == 1
    final = output / ".wm3d_v7_collection_materialization_receipt.json"
    assert final.is_file()
    archive_bytes = bytearray(archive.read_bytes())
    archive_bytes[-1] ^= 1
    archive.write_bytes(archive_bytes)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safe_extract_lerobot_collection.py",
            "--archive-root",
            str(archive_root),
            "--output-root",
            str(output),
        ],
    )
    with pytest.raises(FileExistsError, match="无匹配完成 receipt"):
        module["main"]()


def test_agibot_beta_materialization_is_exact_and_reentrant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("scripts/data/safe_materialize_agibot_beta.py")
    snapshot = tmp_path / "agibot_beta_snapshot"
    (snapshot / "task_info").mkdir(parents=True)
    (snapshot / "observations" / "327").mkdir(parents=True)
    (snapshot / "parameters").mkdir()
    (snapshot / "proprio_stats").mkdir()
    episodes = (648642, 648649)
    (snapshot / "task_info" / "task_327.json").write_text(
        json.dumps(
            [{"task_id": 327, "episode_id": episode_id} for episode_id in episodes]
        ),
        encoding="utf-8",
    )
    revision = "a" * 40
    (snapshot / ".wm3d_v7_download_receipt.json").write_text(
        json.dumps(
            {
                "schema": "wm3d_v7_raw_download_receipt_v1",
                "complete": True,
                "source": "agibot_beta_snapshot",
                "repo_id": "agibot-world/AgiBotWorld-Beta",
                "revision": revision,
                "resolved_revision": revision,
                "target": str(snapshot),
                "payload_files": 4,
                "payload_bytes": 1,
            }
        ),
        encoding="utf-8",
    )

    def write_tar(path: Path, members: dict[str, bytes]) -> None:
        with tarfile.open(path, mode="w") as archive:
            directory = tarfile.TarInfo("327/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

    write_tar(
        snapshot / "observations" / "327" / "648642-648649.tar",
        {
            f"{episode_id}/videos/head.mp4": f"video-{episode_id}".encode()
            for episode_id in episodes
        },
    )
    write_tar(
        snapshot / "parameters" / "648642-648649.tar",
        {f"327/{episode_id}/camera/intrinsics.json": b"{}" for episode_id in episodes},
    )
    write_tar(
        snapshot / "proprio_stats" / "648642-648649.tar",
        {
            f"proprio_stats/327/{episode_id}/proprio_stats.h5": b"stats"
            for episode_id in episodes
        },
    )

    output_parent = tmp_path / "materialized"
    output_parent.mkdir()
    task_list_module = runpy.run_path("scripts/data/list_agibot_beta_tasks.py")
    task_list_path = output_parent / "task_ids.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "list_agibot_beta_tasks.py",
            "--raw-root",
            str(snapshot),
            "--output",
            str(task_list_path),
        ],
    )
    task_list_module["main"]()
    assert task_list_path.read_text(encoding="utf-8") == "327\n"
    output = output_parent / "agibot_beta_raw"
    prepared = module["_prepare"](snapshot, output)
    assert prepared["tasks"] == 1
    assert prepared["episodes"] == 2
    extracted = module["_extract"](
        snapshot,
        output,
        shard_id=0,
        num_shards=1,
    )
    assert extracted["archives"] == 3
    assert {item["status"] for item in extracted["results"]} == {"extracted"}
    repeated = module["_extract"](
        snapshot,
        output,
        shard_id=0,
        num_shards=1,
    )
    assert {item["status"] for item in repeated["results"]} == {"already_complete"}
    finalized = module["_finalize"](snapshot, output)
    assert finalized["complete"] is True
    assert finalized["archives"] == 3
    assert finalized["episodes"] == 2
    converter_module = runpy.run_path("scripts/data/convert_agibot_beta_task.py")
    receipt_path, receipt = converter_module["_materialization_receipt"](output)
    assert receipt_path.name == ".wm3d_v7_beta_materialization_receipt.json"
    assert (
        receipt["materialization_plan_sha256"]
        == finalized["materialization_plan_sha256"]
    )
    converted = tmp_path / "converted"
    converted.mkdir()
    task_root = converted / "task_000327"
    (task_root / "meta").mkdir(parents=True)
    (task_root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    converter_snapshot = tmp_path / "agibot_alpha_converter_snapshot"
    (converter_snapshot / "scripts").mkdir(parents=True)
    converter = converter_snapshot / "scripts" / "convert_to_lerobot.py"
    converter.write_text("# frozen converter\n", encoding="utf-8")
    converter_revision = "b" * 40
    converter_download_receipt = converter_snapshot / ".wm3d_v7_download_receipt.json"
    converter_download_receipt.write_text(
        json.dumps(
            {
                "schema": converter_module["DOWNLOAD_RECEIPT_SCHEMA"],
                "complete": True,
                "source": converter_module["CONVERTER_SOURCE"],
                "repo_id": converter_module["CONVERTER_REPO_ID"],
                "revision": converter_revision,
                "resolved_revision": converter_revision,
                "target": str(converter_snapshot),
                "payload_files": 1,
                "payload_bytes": converter.stat().st_size,
                "payload_sha256": {
                    "scripts/convert_to_lerobot.py": sha256_file(converter)
                },
            }
        ),
        encoding="utf-8",
    )
    frozen_receipt_path, frozen_receipt = converter_module["_converter_receipt"](
        converter,
        converter_download_receipt,
    )
    assert frozen_receipt_path == converter_download_receipt
    assert frozen_receipt["revision"] == converter_revision
    environment_module = runpy.run_path(
        "environments/verify_agibot_converter_environment.py"
    )
    environment_root = tmp_path / "agibot_converter_environment"
    environment_root.mkdir()
    formal_environment_contract = json.loads(
        Path(
            "environments/agibot_converter_environment_contract.json"
        ).read_text(encoding="utf-8")
    )
    environment_contract = {
        "schema": environment_module["CONTRACT_SCHEMA"],
        "python_major_minor": ".".join(platform.python_version_tuple()[:2]),
        "lerobot": {
            "version": "0.1.0",
            "revision": formal_environment_contract["lerobot"]["revision"],
        },
        "packages": {},
        "required_imports": [],
        "required_commands": [],
    }
    environment_contract_path = environment_root / "environment_contract.json"
    environment_contract_path.write_text(
        json.dumps(environment_contract),
        encoding="utf-8",
    )
    revision_file = environment_root / "LEROBOT_REVISION"
    revision_file.write_text(
        formal_environment_contract["lerobot"]["revision"] + "\n",
        encoding="utf-8",
    )
    environment_receipt_path = environment_root / "environment_receipt.json"
    environment_receipt = environment_module["current_environment"](
        contract_path=environment_contract_path,
        revision_file=revision_file,
    )
    environment_receipt["created_at_utc"] = "2026-07-29T00:00:00+00:00"
    environment_receipt_path.write_text(
        json.dumps(environment_receipt),
        encoding="utf-8",
    )
    portable_bundle = tmp_path / "portable_converter_environment"
    portable_bundle.mkdir()
    for source in (
        environment_contract_path,
        revision_file,
        environment_receipt_path,
    ):
        shutil.copyfile(source, portable_bundle / source.name)
    portable_receipt = portable_bundle / environment_receipt_path.name
    portable_value = environment_module["validate_receipt"](
        portable_receipt,
        check_current=False,
    )
    assert portable_value["lerobot_revision"] == environment_receipt["lerobot_revision"]
    (portable_bundle / "LEROBOT_REVISION").write_text(
        "0" * 40 + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="内容与绑定文件不匹配"):
        environment_module["validate_receipt"](
            portable_receipt,
            check_current=False,
        )
    frozen_environment_path, frozen_environment = converter_module[
        "_converter_environment_receipt"
    ](environment_receipt_path)
    assert frozen_environment_path == environment_receipt_path
    assert (
        frozen_environment["lerobot_revision"]
        == formal_environment_contract["lerobot"]["revision"]
    )
    task_list = tmp_path / "task_ids.txt"
    task_list.write_text("327\n", encoding="utf-8")
    conversion_receipt = {
        "schema": converter_module["RECEIPT_SCHEMA"],
        "complete": True,
        "task_id": 327,
        "converter_sha256": sha256_file(converter),
        "converter_download_receipt_sha256": sha256_file(converter_download_receipt),
        "converter_environment_receipt_sha256": sha256_file(environment_receipt_path),
        "lerobot_revision": frozen_environment["lerobot_revision"],
        "raw_root": str(output),
        "materialization_receipt_sha256": sha256_file(receipt_path),
        "lerobot_roots": ["."],
    }
    (task_root / ".wm3d_v7_conversion_receipt.json").write_text(
        json.dumps(conversion_receipt),
        encoding="utf-8",
    )
    conversion_final = converter_module["_finalize_collection"](
        raw_root=output,
        output_root=converted,
        converter=converter,
        converter_download_receipt=converter_download_receipt,
        converter_environment_receipt=environment_receipt_path,
        task_list=task_list,
        materialization_receipt=receipt_path,
    )
    assert conversion_final["status"] == "finalized"
    assert conversion_final["tasks"] == 1
    assert (converted / ".wm3d_v7_beta_conversion_collection_receipt.json").is_file()
    for category in ("observations", "parameters", "proprio_stats"):
        assert {
            int(path.name) for path in (output / category / "327").iterdir()
        } == set(episodes)
    with pytest.raises(ValueError, match="归档成员路径不安全"):
        module["_normalized_member"](
            "parameters/example.tar",
            "../../escape",
            task_ids={327},
            episode_to_task={648642: 327},
        )


def test_planning_templates_compile_and_bind_grouped_action_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("scripts/data/compile_dataset_contract.py")
    variable_names = (
        "DROID_ROOT",
        "BRIDGE_ROOT",
        "ATOMIC_ROOT",
        "COMPOSITE_ROOT",
        "MG_ROOT",
        "AGIBOT_2026_IMITATION_ROOT",
        "AGIBOT_2026_RICH_ROOT",
        "AGIBOT_2026_REINFORCEMENT_ROOT",
        "AGIBOT_BETA_ROOT",
    )
    for index, name in enumerate(variable_names):
        monkeypatch.setenv(name, str(tmp_path / f"source-{index}"))
    inventory = module["yaml"].safe_load(
        Path("configs/data/public_6106h.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = DatasetContract.from_mapping(module["_expand"](inventory))
    assert len(contract.sha256) == 64
    assert contract.source_weights == {
        "droid": 14,
        "bridge": 6,
        "atomic": 4,
        "composite": 8,
        "mg": 8,
        "agibot_2026_imitation": 10,
        "agibot_2026_rich": 8,
        "agibot_2026_reinforcement": 12,
        "agibot_beta": 30,
    }
    assert sum(source.nominal_hours for source in contract.sources) == pytest.approx(
        6106.4
    )
    assert {source.adapter for source in contract.sources[:5]} == {"lerobot"}
    assert {source.adapter for source in contract.sources[5:]} == {
        "lerobot_collection"
    }

    raw_layouts = json.loads(
        Path("configs/data/public_6106h_layouts.json").read_text(
            encoding="utf-8"
        )
    )
    layouts = {item["source"]: item for item in raw_layouts["layouts"]}
    assert set(layouts) == set(contract.source_order)
    assert {
        item["column"] for item in layouts["droid"]["action_columns"]
    } == {"action.original"}
    assert sum(
        len(item["indices"]) for item in layouts["bridge"]["action_columns"]
    ) == 7
    for source in ("atomic", "composite", "mg"):
        assert sum(
            len(item["indices"]) for item in layouts[source]["action_columns"]
        ) == 12
    assert sum(
        len(item["indices"]) for item in layouts["agibot_beta"]["action_columns"]
    ) == 22

    raw_lock = module["yaml"].safe_load(
        Path("configs/data/raw_sources.lock.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(raw_lock["sources"]) == (
        "droid",
        "bridge",
        "atomic",
        "composite",
        "mg",
        "agibot_world_2026_snapshot",
        "agibot_beta_snapshot",
        "agibot_alpha_converter_snapshot",
    )
    assert raw_lock["sources"]["droid"]["repo_id"] == "lerobot/droid_1.0.1"
    assert (
        raw_lock["sources"]["bridge"]["repo_id"]
        == "ember-lab-berkeley/bridge_v2"
    )
    converter_source = raw_lock["sources"]["agibot_alpha_converter_snapshot"]
    assert converter_source["repo_id"] == "agibot-world/AgiBotWorld-Alpha"
    assert converter_source["gated"] is True
    assert converter_source["allow_patterns"] == [
        "README.md",
        "scripts/convert_to_lerobot.py",
    ]
    converter_environment = json.loads(
        Path(
            "environments/agibot_converter_environment_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        converter_environment["schema"]
        == "wm3d_v7_agibot_converter_environment_contract_v1"
    )
    assert converter_environment["python_major_minor"] == "3.10"
    assert converter_environment["lerobot"] == {
        "version": "0.1.0",
        "revision": "8e7d6970eaf5a64b8af6ec45586d201b8ca9ef16",
        "source_url": (
            "https://github.com/huggingface/lerobot/archive/"
            "8e7d6970eaf5a64b8af6ec45586d201b8ca9ef16.tar.gz"
        ),
        "source_sha256": (
            "51e51e7e2d91c46db3bb4ccb9604d55776d9f9f90389465ed2603fe9f9bbc702"
        ),
        "pyproject_sha256": (
            "34a923b9d6739c52d63af14d20282d5cbebbc78a46a81d76600ad33ae4057d66"
        ),
        "poetry_lock_sha256": (
            "fabc7c9544e0073cabc2cf351c38b92a6f6578f343320be91a65c4050a7a10d3"
        ),
    }
