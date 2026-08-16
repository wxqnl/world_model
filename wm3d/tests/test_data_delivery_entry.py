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
import torch
import yaml

from scripts.data.materialize_existing_robot_mix import _window_evidence
from scripts.data.run_cache_worker import _view_batch


ROOT = Path(__file__).resolve().parents[1]


def test_existing_robot_mix_counts_every_physically_valid_window() -> None:
    evidence = _window_evidence(
        np.arange(20, dtype=np.float64) * 0.25,
        {
            "model": {"T": 4, "K": 2},
            "sampling": {
                "minimum_anchor_separation_seconds": 0.25,
                "context_horizon_seconds": 0.75,
                "future_horizon_seconds": 0.5,
                "minimum_horizon_coverage": 1.0,
                "future_offsets_seconds": [0.25, 0.5],
            },
        },
    )
    assert evidence is not None
    assert evidence["first_valid_anchor_index"] == 4
    assert evidence["valid_window_count"] == 14


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
