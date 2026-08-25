from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from scripts.data import materialize_existing_robot_mix
from scripts.data.materialize_existing_robot_mix import (
    ROBOCASA,
    ROBOCASA_ACTION_SCALES,
    _episode_candidates,
    _robocasa_adapter,
    _segment_has_video_coverage,
    _window_evidence,
)
from scripts.data.run_cache_worker import _encode, _view_batch
from wm3d.data.step_sampler import SamplingContractError
from wm3d.training.pretrain import _require_sampling_capacity


ROOT = Path(__file__).resolve().parents[1]


class _OOMBackoffEncoder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(
        self, images: torch.Tensor, view_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        del view_mask
        frames = int(images.shape[1])
        self.batch_sizes.append(frames)
        if frames > 2:
            raise torch.OutOfMemoryError("fixture encoder peak")
        return {"tokens": torch.arange(frames).reshape(1, frames, 1)}


def test_cache_encoder_retries_a_smaller_chunk_after_oom(
    capsys: pytest.CaptureFixture[str],
) -> None:
    encoder = _OOMBackoffEncoder()
    encoded = _encode(
        encoder=encoder,
        images=torch.zeros(5, 1, 3, 2, 2),
        view_mask=torch.ones(5, 1, dtype=torch.bool),
        device=torch.device("cpu"),
        batch_frames=4,
    )

    assert encoder.batch_sizes == [4, 2, 2, 1]
    assert encoded["tokens"].shape == (5, 1)
    event = json.loads(capsys.readouterr().out)
    assert event["streaming_raw_encoder"] == "oom_backoff"
    assert event["attempted_batch_frames"] == 4
    assert event["retry_batch_frames"] == 2


class _CapacityDataset:
    requires_main_process = True
    source_names = ("fixture",)
    source_episode_spans = None

    def __init__(self, windows: int) -> None:
        self.source_spans = {"fixture": (0, windows)}

    def __len__(self) -> int:
        return self.source_spans["fixture"][1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        raise AssertionError(f"capacity preflight must not read sample {index}")


class _CapacityProfile:
    source_weights = {"fixture": 1}


def test_preflight_checks_validation_global_micro_batch_capacity() -> None:
    runtime = {
        "train": {
            "micro_batch_size": 8,
            "validation_micro_batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": 2,
        }
    }
    with pytest.raises(SamplingContractError, match="below global batch 32"):
        _require_sampling_capacity(
            _CapacityDataset(31),
            _CapacityProfile(),
            runtime,
            rank=0,
            world_size=8,
            gradient_accumulation=1,
            seed=7340,
            micro_batch_size=runtime["train"]["validation_micro_batch_size"],
        )
    _require_sampling_capacity(
        _CapacityDataset(32),
        _CapacityProfile(),
        runtime,
        rank=0,
        world_size=8,
        gradient_accumulation=1,
        seed=7340,
        micro_batch_size=runtime["train"]["validation_micro_batch_size"],
    )


def test_existing_robot_mix_counts_every_physically_valid_window() -> None:
    clock = np.arange(20, dtype=np.float64) * 0.25
    model_profile = {
        "model": {"T": 4, "K": 2},
        "sampling": {
            "minimum_anchor_separation_seconds": 0.25,
            "context_horizon_seconds": 0.75,
            "future_horizon_seconds": 0.5,
            "minimum_horizon_coverage": 1.0,
            "future_offsets_seconds": [0.25, 0.5],
        },
    }
    evidence = _window_evidence(clock, model_profile)
    assert evidence is not None
    assert evidence["first_valid_anchor_index"] == 4
    assert evidence["valid_window_count"] == 14
    fast_evidence = _window_evidence(clock, model_profile, count_all=False)
    assert fast_evidence is not None
    assert fast_evidence["first_valid_anchor_index"] == 4
    assert fast_evidence["valid_window_count"] == 1


def test_existing_robot_mix_rejects_episode_segment_beyond_video() -> None:
    assert _segment_has_video_coverage(
        requested_start_s=10.0,
        requested_stop_s=20.0,
        available_start_s=0.0,
        available_stop_s=20.0,
        frame_count=400,
    )
    assert not _segment_has_video_coverage(
        requested_start_s=10.0,
        requested_stop_s=20.1,
        available_start_s=0.0,
        available_stop_s=20.0,
        frame_count=400,
    )
    assert not _segment_has_video_coverage(
        requested_start_s=None,
        requested_stop_s=None,
        available_start_s=0.0,
        available_stop_s=20.0,
        frame_count=1,
    )


def test_existing_robot_mix_binds_video_template_to_camera_file_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_candidates: list[str] = []

    def choose(_root: Path, candidates: object) -> str:
        observed_candidates.extend(str(candidate) for candidate in candidates)
        return observed_candidates[0]

    monkeypatch.setattr(materialize_existing_robot_mix, "_existing_relative", choose)
    monkeypatch.setattr(
        materialize_existing_robot_mix,
        "_video_bounds",
        lambda _path, _cache: (0.0, 20.0, 200),
    )
    assert materialize_existing_robot_mix._episode_video_coverage(
        root=tmp_path,
        row={
            "data/chunk_index": 0,
            "data/file_index": 0,
            "videos/observation.images.head/chunk_index": 0,
            "videos/observation.images.head/file_index": 3,
            "videos/observation.images.head/from_timestamp": 10.0,
            "videos/observation.images.head/to_timestamp": 11.0,
        },
        episode_index=7,
        view_keys=("observation.images.head",),
        video_template=(
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        cache={},
    )
    assert observed_candidates[0] == (
        "videos/observation.images.head/chunk-000/file-003.mp4"
    )


def test_robocasa_partitions_keep_their_audited_controller_scales() -> None:
    features = {
        "action": {"shape": [12]},
        "observation.state": {"shape": [16]},
        "observation.images.robot0_agentview_left": {},
        "observation.images.robot0_eye_in_hand": {},
    }
    observed_scales: set[tuple[float, ...]] = set()
    for plan in ROBOCASA:
        adapter, embodiment = _robocasa_adapter(plan, {"features": features})
        arm = next(group for group in adapter["groups"] if group["group"] == "arm")
        translation, rotation = ROBOCASA_ACTION_SCALES[plan.name]
        assert tuple(arm["action"][0]["scale"]) == translation
        assert tuple(arm["action"][1]["scale"]) == rotation
        assert arm["action"][2]["scale"] == [1.0]
        assert embodiment["name"] == "panda_robocasa_libero"
        assert embodiment["embodiment_id"] == 2
        arm_group = next(
            group for group in embodiment["groups"] if group["name"] == "arm"
        )
        assert arm_group["group_id"] == 12
        assert arm_group["action_semantics"][-1] == "absolute_gripper_close01"
        observed_scales.add(tuple(translation + rotation))
    assert len(observed_scales) == len(ROBOCASA)


def test_existing_robot_mix_reads_episode_video_segments(tmp_path: Path) -> None:
    episodes = tmp_path / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [7],
                "length": [32],
                "videos/observation.images.head/from_timestamp": [10443.1],
                "videos/observation.images.head/to_timestamp": [10484.6],
            }
        ),
        episodes / "file-000.parquet",
    )

    rows = list(_episode_candidates(tmp_path))
    assert rows == [
        {
            "episode_index": 7,
            "length": 32,
            "videos/observation.images.head/from_timestamp": 10443.1,
            "videos/observation.images.head/to_timestamp": 10484.6,
        }
    ]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _lerobot(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    video = root / "videos/chunk-000/observation.images.top/episode_000000.mp4"
    video.parent.mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "fixture",
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.top": {"dtype": "video"},
                    "timestamp": {"dtype": "float64"},
                    "action": {"dtype": "float32", "shape": [14]},
                    "observation.state": {"dtype": "float32", "shape": [14]},
                },
            }
        )
    )
    (root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 4, "task_index": 0}) + "\n"
    )
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "bimanual fixture"}) + "\n"
    )
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array([0.0, 0.04, 0.09, 0.15], type=pa.float64()),
                "action": pa.array(
                    np.zeros((4, 14), np.float32).tolist(),
                    type=pa.list_(pa.float32(), 14),
                ),
                "observation.state": pa.array(
                    np.zeros((4, 14), np.float32).tolist(),
                    type=pa.list_(pa.float32(), 14),
                ),
            }
        ),
        root / "data/chunk-000/episode_000000.parquet",
    )
    video.write_bytes(b"container-fixture")


def _adapter(path: Path) -> None:
    groups = []
    for name, start in (("left_arm", 0), ("right_arm", 7)):
        groups.append(
            {
                "group": name,
                "supervision": "fine_command",
                "action": [
                    {
                        "key": "action",
                        "columns": list(range(start, start + 7)),
                        "scale": [1.0] * 7,
                        "offset": [0.0] * 7,
                    }
                ],
                "state": [
                    {
                        "key": "observation.state",
                        "columns": list(range(start, start + 7)),
                        "scale": [1.0] * 7,
                        "offset": [0.0] * 7,
                    }
                ],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "wm3d_v8_source_adapter_v3",
                "name": "fixture",
                "raw_format": "lerobot_parquet_video",
                "observation_time_key": "timestamp",
                "views": [{"name": "head", "key": "observation.images.top"}],
                "groups": groups,
            },
            sort_keys=False,
        )
    )


def _template(path: Path) -> None:
    groups = []
    for group_id, name in enumerate(("left_arm", "right_arm"), 1):
        groups.append(
            {
                "name": name,
                "group_id": group_id,
                "action_semantics": ["controller_command"] * 7,
                "state_semantics": ["joint_position_rad"] * 7,
                "action_frame": f"{name}_native",
                "state_frame": f"{name}_native",
                "composition_operators": ["last"] * 7,
            }
        )
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "wm3d_v8_data_profile_v4",
                "name": "fixture",
                "cache_representation": {"view_slots": ["head"]},
                "sources": [{"name": "fixture", "embodiment": "fixture_bot"}],
                "embodiments": [
                    {"name": "fixture_bot", "embodiment_id": 1, "groups": groups}
                ],
            },
            sort_keys=False,
        )
    )


def test_archive_schema_adapter_collection_inventory_roundtrip(tmp_path: Path) -> None:
    raw_episode = tmp_path / "episode"
    _lerobot(raw_episode)
    snapshot = tmp_path / "snapshot"
    subset = snapshot / "ImitationLearning"
    subset.mkdir(parents=True)
    archive = subset / "one.tar"
    with tarfile.open(archive, "w") as handle:
        handle.add(raw_episode, arcname="lerobot")
    download = tmp_path / "download.json"
    download.write_text(
        json.dumps(
            {
                "schema": "wm3d_v8_raw_snapshot_receipt_v1",
                "source": "agibot_world_2026",
                "snapshot_path": str(snapshot.resolve()),
                "snapshot_file_count": 1,
                "snapshot_total_bytes": archive.stat().st_size,
            }
        )
    )
    collection = tmp_path / "collection"
    _run(
        "scripts/data/materialize_archive_collection.py",
        "--snapshot-root", snapshot,
        "--download-receipt", download,
        "--download-source", "agibot_world_2026",
        "--source-prefix", "ImitationLearning",
        "--output-root", collection,
    )
    _run(
        "scripts/data/materialize_archive_collection.py",
        "--snapshot-root", snapshot,
        "--download-receipt", download,
        "--download-source", "agibot_world_2026",
        "--source-prefix", "ImitationLearning",
        "--output-root", collection,
        "--finalize",
    )
    receipt = collection / "collection_receipt.json"
    schema = tmp_path / "schema.json"
    candidate = tmp_path / "candidate.json"
    _run(
        "scripts/data/inspect_source_schema.py",
        "--root", collection,
        "--collection",
        "--require-homogeneous",
        "--upstream-receipt", receipt,
        "--candidate-output", candidate,
        "--output", schema,
    )
    assert json.loads(schema.read_text())["formal_inventory_ready"] is False

    adapter = tmp_path / "adapter.yaml"
    template = tmp_path / "template.yaml"
    _adapter(adapter)
    _template(template)
    audit = tmp_path / "adapter_receipt.json"
    _run(
        "scripts/data/audit_adapter_contract.py",
        "--schema-audit", schema,
        "--adapter-candidate", candidate,
        "--adapter-contract", adapter,
        "--adapter-contract-sha256", _sha(adapter),
        "--data-template", template,
        "--source", "fixture",
        "--operator", "test",
        "--confirm", "I_VERIFIED_FIELDS_UNITS_FRAMES_GRIPPER_GROUPS_AND_NATIVE_CLOCKS",
        "--output", audit,
    )
    manifest = tmp_path / "manifest.jsonl"
    inventory = tmp_path / "inventory.json"
    _run(
        "scripts/data/materialize_collection_inventory.py",
        "--data-template", template,
        "--source", "fixture",
        "--collection-root", collection,
        "--collection-receipt", receipt,
        "--adapter-contract", adapter,
        "--adapter-contract-sha256", _sha(adapter),
        "--adapter-audit-receipt", audit,
        "--output-manifest", manifest,
        "--output-receipt", inventory,
    )
    row = json.loads(manifest.read_text().strip())
    assert row["source"] == "fixture"
    assert set(row["robot_groups"]) == {"left_arm", "right_arm"}
    assert len(row["episode_id"].split(":")) == 3
    assert not Path(row["payload"]).is_absolute()


def test_schema_audit_records_observed_width_for_variable_arrow_list(
    tmp_path: Path,
) -> None:
    root = tmp_path / "variable-list"
    _lerobot(root)
    payload = root / "data/chunk-000/episode_000000.parquet"
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array([0.0, 0.04, 0.09, 0.15]),
                "action": pa.array(np.zeros((4, 14), np.float32).tolist()),
                "observation.state": pa.array(
                    np.zeros((4, 14), np.float32).tolist()
                ),
            }
        ),
        payload,
    )
    upstream = tmp_path / "download.json"
    upstream.write_text("{}\n")
    schema = tmp_path / "schema.json"
    candidate = tmp_path / "candidate.json"
    _run(
        "scripts/data/inspect_source_schema.py",
        "--root", root,
        "--upstream-receipt", upstream,
        "--candidate-output", candidate,
        "--output", schema,
    )
    action = json.loads(schema.read_text())["roots"][0]["sample_data"][0][
        "columns"
    ]["action"]
    assert action["arrow_type"].startswith("list<")
    assert action["observed_list_widths"] == [14]
    assert action["observed_list_rows"] == 4
    assert action["observed_list_null_rows"] == 0


def test_cache_rgb_bicubic_resize_preserves_normalized_input_range() -> None:
    class Decoded:
        pass

    decoded = Decoded()
    # A hard edge makes bicubic interpolation overshoot without an explicit
    # post-resize clamp, which then violates the sealed VGGT [0,1] ABI.
    decoded.frames = np.zeros((2, 17, 31, 3), dtype=np.uint8)
    decoded.frames[:, :, 16:] = 255
    images, mask = _view_batch(
        decoded={"head": decoded},
        slots=("head", "left_wrist", "right_wrist"),
        input_size=518,
    )
    assert torch.isfinite(images).all()
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0
    assert mask.tolist() == [[True, False, False], [True, False, False]]


def test_cache_rgb_restores_declared_bgr_source_order() -> None:
    class Decoded:
        pass

    decoded = Decoded()
    decoded.frames = np.zeros((1, 14, 14, 3), dtype=np.uint8)
    decoded.frames[..., 0] = 10
    decoded.frames[..., 1] = 20
    decoded.frames[..., 2] = 30
    images, mask = _view_batch(
        decoded={"head": decoded},
        slots=("head",),
        input_size=14,
        color_order_by_view={"head": "bgr"},
    )
    assert mask.tolist() == [[True]]
    expected = torch.tensor([30.0, 20.0, 10.0]) / 255.0
    torch.testing.assert_close(images[0, 0, :, 0, 0], expected)


def test_external_converter_is_receipt_bound_and_closes_all_jobs(tmp_path: Path) -> None:
    import sys

    input_root = tmp_path / "beta"
    converter_root = tmp_path / "converter"
    input_root.mkdir()
    (converter_root / "scripts").mkdir(parents=True)
    converter = converter_root / "scripts/convert_to_lerobot.py"
    converter.write_text(
        """\
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--src_path'); p.add_argument('--task_id'); p.add_argument('--tgt_path')
a=p.parse_args(); out=Path(a.tgt_path)/('task_'+a.task_id); (out/'meta').mkdir(parents=True)
(out/'meta/info.json').write_text(json.dumps({'task': a.task_id}))
"""
    )
    input_receipt = tmp_path / "beta_download.json"
    converter_receipt = tmp_path / "converter_download.json"
    for path, source, root in (
        (input_receipt, "agibot_beta", input_root),
        (converter_receipt, "agibot_alpha_converter", converter_root),
    ):
        path.write_text(
            json.dumps(
                {
                    "schema": "wm3d_v8_raw_snapshot_receipt_v1",
                    "source": source,
                    "snapshot_path": str(root.resolve()),
                    "snapshot_file_count": 1,
                    "snapshot_total_bytes": 1,
                }
            )
        )
    environment = tmp_path / "environment.json"
    environment.write_text(json.dumps({"fixture": True}))
    task_ids = tmp_path / "tasks.txt"
    task_ids.write_text("3\n9\n")
    contract = tmp_path / "converter.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "schema": "wm3d_v8_external_converter_contract_v1",
                "name": "fixture",
                "input_source": "agibot_beta",
                "converter_source": "agibot_alpha_converter",
                "converter_relative_path": "scripts/convert_to_lerobot.py",
                "environment_receipt_sha256": _sha(environment),
                "required_bindings": ["task_id"],
                "argv": [
                    "{python}",
                    "{converter}",
                    "--src_path",
                    "{input_root}",
                    "--task_id",
                    "{task_id}",
                    "--tgt_path",
                    "{output_root}",
                ],
                "output_kind": "lerobot_collection",
                "lerobot_root_glob": "**/meta/info.json",
            },
            sort_keys=False,
        )
    )
    output = tmp_path / "converted"
    _run(
        "scripts/data/run_external_converter.py",
        "--contract",
        contract,
        "--input-root",
        input_root,
        "--input-download-receipt",
        input_receipt,
        "--converter-root",
        converter_root,
        "--converter-download-receipt",
        converter_receipt,
        "--environment-receipt",
        environment,
        "--python-bin",
        Path(sys.executable),
        "--binding-file",
        f"task_id={task_ids}",
        "--output-root",
        output,
    )
    receipt = json.loads((output / "collection_receipt.json").read_text())
    conversion = json.loads((output / "conversion_receipt.json").read_text())
    assert receipt["lerobot_root_count"] == 2
    assert conversion["binding_files"]["task_id"]["sha256"] == _sha(task_ids)
    assert all(str(output) in command[-1] for command in conversion["commands"])
