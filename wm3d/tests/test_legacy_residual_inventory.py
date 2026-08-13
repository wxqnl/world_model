from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from wm3d.data.grouped_robot import ActionGroupSpec, EmbodimentSpec
from wm3d.data.legacy_residual_inventory import (
    LEGACY_FORMAL_SOURCE,
    LegacyResidualImportError,
    import_legacy_residual_plan,
)
from wm3d.data.manifest_contract import sha256_file
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.data.source_inventory import deterministic_split, validate_written_inventory
from scripts.data import materialize_legacy_residual_inventory as materializer


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_video(path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is required for the real-video importer fixture")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=16x16:r=20:d=0.20",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(path),
        ],
        check=True,
    )
    assert path.stat().st_size > 0


def _write_payload(
    path: Path,
    *,
    timestamps: tuple[float, ...] = (0.0, 0.04, 0.09, 0.15),
    action_width: int = 7,
    state_width: int = 10,
    nonfinite_state: bool = False,
) -> None:
    rows = len(timestamps)
    action = np.arange(rows * action_width, dtype=np.float64).reshape(rows, action_width)
    state = np.arange(rows * state_width, dtype=np.float64).reshape(rows, state_width) / 100.0
    if nonfinite_state:
        state[2, 3] = np.nan
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.float64()),
                "episode_index": pa.array([7] * rows, type=pa.int64()),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float64(), action_width)),
                "state": pa.array(
                    state.tolist(), type=pa.list_(pa.float64(), state_width)
                ),
            }
        ),
        path,
        row_group_size=2,
    )


def _embodiment() -> EmbodimentSpec:
    return EmbodimentSpec(
        name="legacy_arm_wm3d",
        embodiment_id=91,
        groups=(
            ActionGroupSpec(
                name="arm",
                group_id=1,
                action_semantics=(
                    "delta_position_m",
                    "delta_position_m",
                    "delta_position_m",
                    "delta_rotation_axis_angle_rad",
                    "delta_rotation_axis_angle_rad",
                    "delta_rotation_axis_angle_rad",
                    "absolute_gripper_open01",
                ),
                state_semantics=(
                    "eef_position_m",
                    "eef_position_m",
                    "eef_position_m",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "eef_rotation_6d",
                    "gripper_open01",
                ),
                action_frame="robot_base",
                state_frame="robot_base",
                composition_operators=(
                    "sum",
                    "sum",
                    "sum",
                    "so3_axis_angle_base_left",
                    "so3_axis_angle_base_left",
                    "so3_axis_angle_base_left",
                    "last",
                ),
            ),
        ),
    )


def _adapter(path: Path):
    value = {
        "schema": "wm3d_v8_source_adapter_v3",
        "name": "legacy_v7_residual_parquet",
        "raw_format": "lerobot_parquet_video",
        "observation_time_key": "timestamp",
        "views": [{"name": "head", "key": "observation.images.head"}],
        "groups": [
            {
                "group": "arm",
                "supervision": "fine_command",
                "action": [
                    {
                        "key": "action",
                        "columns": list(range(7)),
                        "scale": [1.0] * 7,
                        "offset": [0.0] * 7,
                    }
                ],
                "state": [
                    {
                        "key": "state",
                        "columns": list(range(10)),
                        "scale": [1.0] * 10,
                        "offset": [0.0] * 10,
                    }
                ],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
            }
        ],
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return load_adapter_contract(path, expected_sha256=sha256_file(path))


def _legacy_row(root: Path) -> dict[str, object]:
    return {
        "schema": "wm3d_v7_native5b_episode_plan_v1",
        "source": LEGACY_FORMAL_SOURCE,
        "episode_id": "legacy_v7_formal:000000007",
        "episode_index": 7,
        "embodiment": "legacy_arm_wm3d",
        "split": "test",
        "task_text": "pick up the blue block",
        "raw_root": str(root),
        "data_relative_path": "episode.parquet",
        "data_row_start": 0,
        "data_row_stop": 4,
        "timestamp_column": "timestamp",
        "episode_column": "episode_index",
        "source_fps": 999.0,
        "duration_seconds": 999.0,
        "views": [
            {
                "canonical_name": "head",
                "feature_key": "observation.images.head",
                "relative_path": "head.mp4",
                "start_seconds": 0.0,
                "stop_seconds": 0.2,
            },
            {
                "canonical_name": "left_hand",
                "feature_key": None,
                "relative_path": None,
                "start_seconds": 0.0,
                "stop_seconds": 0.0,
            },
            {
                "canonical_name": "right_hand",
                "feature_key": None,
                "relative_path": None,
                "start_seconds": 0.0,
                "stop_seconds": 0.0,
            },
        ],
        # V7 split arm6/gripper1 is accepted only because the exact raw
        # coordinates form one non-overlapping cover of the audited WM3D arm7.
        "action_columns": [
            {
                "group_name": "arm_pose",
                "column": "action",
                "indices": list(range(6)),
                "discrete": False,
            },
            {
                "group_name": "gripper",
                "column": "action",
                "indices": [6],
                "discrete": False,
            },
        ],
        "auxiliary_columns": [],
        "provenance_dataset": "audited_legacy_robot_residual",
    }


@pytest.fixture()
def legacy_fixture(tmp_path: Path):
    root = tmp_path / "raw"
    root.mkdir()
    _write_payload(root / "episode.parquet")
    _valid_video(root / "head.mp4")
    adapter = _adapter(tmp_path / "adapter.yaml")
    row = _legacy_row(root)
    plan = tmp_path / "legacy.jsonl"
    _write_jsonl(plan, [row])
    return root, plan, row, adapter


def _import(root: Path, plan: Path, adapter):
    return import_legacy_residual_plan(
        plan_path=plan,
        raw_root=root,
        source=LEGACY_FORMAL_SOURCE,
        embodiment=_embodiment(),
        adapter=adapter,
        view_slots=("head", "left_hand", "right_hand"),
        split_seed=3407,
        train_fraction=0.98,
        validation_fraction=0.01,
    )


def test_real_parquet_video_import_rebuilds_wm3d_evidence(
    legacy_fixture, tmp_path: Path
):
    root, plan, _legacy, adapter = legacy_fixture
    rows, receipt = _import(root, plan, adapter)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == LEGACY_FORMAL_SOURCE
    assert row["payload_sha256"] == sha256_file(root / "episode.parquet")
    assert row["assets"][1]["sha256"] == sha256_file(root / "head.mp4")
    assert [item["name"] for item in row["views"]] == ["head"]
    assert row["duration_s"] == pytest.approx(0.21)
    assert row["duration_s"] != 999.0
    assert row["split"] == deterministic_split(
        LEGACY_FORMAL_SOURCE,
        row["episode_id"],
        seed=3407,
        train_fraction=0.98,
        validation_fraction=0.01,
    )
    assert row["robot_groups"]["arm"]["state_samples"] == 4
    assert row["robot_groups"]["arm"]["state_clock"]["sample_count"] == 4
    assert receipt["selection"]["legacy_plan_sha256"] == sha256_file(plan)
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, list(rows))
    assert validate_written_inventory(
        manifest, source=LEGACY_FORMAL_SOURCE, embodiment=_embodiment()
    )["episodes"] == 1


def test_rejects_forbidden_mg_provenance(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["provenance_dataset"] = "RoboCasa365_MG"
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="forbidden provenance"):
        _import(root, plan, adapter)


@pytest.mark.parametrize("escape", ["../outside.parquet", "/tmp/outside.parquet"])
def test_rejects_path_escape(legacy_fixture, escape: str):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["data_relative_path"] = escape
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="unsafe relative path"):
        _import(root, plan, adapter)


def test_rejects_intermediate_symlink_escape(legacy_fixture, tmp_path: Path):
    root, plan, row, adapter = legacy_fixture
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil.copy2(root / "episode.parquet", outside / "episode.parquet")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    changed = copy.deepcopy(row)
    changed["data_relative_path"] = "linked/episode.parquet"
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="symlink components"):
        _import(root, plan, adapter)


def test_safe_cluster_relocation_uses_only_current_raw_root(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["raw_root"] = "/old/cluster/archive/legacy_v7_formal"
    _write_jsonl(plan, [changed])
    rows, receipt = _import(root, plan, adapter)
    assert len(rows) == 1
    selection = receipt["selection"]
    assert selection["legacy_provenance_raw_roots"] == [changed["raw_root"]]
    assert receipt["raw_root"] == str(root.resolve())


def test_wm3d_adapter_owns_legacy_camera_rename(legacy_fixture, tmp_path: Path):
    root, plan, row, _unused_adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["views"][0]["canonical_name"] = "old_head_camera_name"
    _write_jsonl(plan, [changed])
    adapter = _adapter(tmp_path / "renamed_adapter.yaml")
    rows, _receipt = import_legacy_residual_plan(
        plan_path=plan,
        raw_root=root,
        source=LEGACY_FORMAL_SOURCE,
        embodiment=_embodiment(),
        adapter=adapter,
        view_slots=("head", "left_wrist", "right_wrist"),
        split_seed=3407,
        train_fraction=0.98,
        validation_fraction=0.01,
    )
    assert [view["name"] for view in rows[0]["views"]] == ["head"]


def test_rejects_relative_legacy_raw_root_provenance(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["raw_root"] = "old/relative/root"
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="non-empty/absolute"):
        _import(root, plan, adapter)


def test_rejects_payload_field_width_drift(legacy_fixture):
    root, plan, _row, adapter = legacy_fixture
    _write_payload(root / "episode.parquet", action_width=6)
    with pytest.raises(LegacyResidualImportError, match="exceed width"):
        _import(root, plan, adapter)


@pytest.mark.parametrize(
    ("state_width", "nonfinite_state", "pattern"),
    [(9, False, "exceed width"), (10, True, "NaN/Inf")],
)
def test_requires_real_finite_10d_current_state(
    legacy_fixture, state_width: int, nonfinite_state: bool, pattern: str
):
    root, plan, _row, adapter = legacy_fixture
    _write_payload(
        root / "episode.parquet",
        state_width=state_width,
        nonfinite_state=nonfinite_state,
    )
    with pytest.raises(LegacyResidualImportError, match=pattern):
        _import(root, plan, adapter)


def test_legacy_action_groups_are_exact_coordinate_attestation(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["action_columns"][1]["indices"] = [5]
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="overlapping legacy action"):
        _import(root, plan, adapter)


def test_rejects_wrong_legacy_source_even_when_payload_is_valid(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["source"] = "other_source"
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="source must be"):
        _import(root, plan, adapter)


def test_rejects_missing_provenance_and_embodiment_drift(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["provenance_dataset"] = ""
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="must be non-empty"):
        _import(root, plan, adapter)
    changed = copy.deepcopy(row)
    changed["embodiment"] = "wrong"
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="embodiment does not match"):
        _import(root, plan, adapter)


def test_rejects_clock_drift(legacy_fixture):
    root, plan, _row, adapter = legacy_fixture
    _write_payload(root / "episode.parquet", timestamps=(0.0, 0.04, 0.04, 0.15))
    with pytest.raises(LegacyResidualImportError, match="strictly increasing"):
        _import(root, plan, adapter)


def test_rejects_invalid_or_out_of_range_video_segment(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    changed = copy.deepcopy(row)
    changed["views"][0]["stop_seconds"] = 10.0
    _write_jsonl(plan, [changed])
    with pytest.raises(LegacyResidualImportError, match="exceeds duration"):
        _import(root, plan, adapter)
    _write_jsonl(plan, [row])
    (root / "head.mp4").write_bytes(b"not-a-video")
    with pytest.raises(LegacyResidualImportError, match="cannot be decoded/probed"):
        _import(root, plan, adapter)


def test_rejects_duplicate_episode_identity(legacy_fixture):
    root, plan, row, adapter = legacy_fixture
    _write_jsonl(plan, [row, copy.deepcopy(row)])
    with pytest.raises(LegacyResidualImportError, match="duplicate/empty episode_id"):
        _import(root, plan, adapter)


def _template(path: Path) -> None:
    group = _embodiment().groups[0]
    value = {
        "schema": "wm3d_v8_data_profile_v4",
        "sources": [
            {"name": LEGACY_FORMAL_SOURCE, "embodiment": "legacy_arm_wm3d"}
        ],
        "embodiments": [
            {
                "name": "legacy_arm_wm3d",
                "embodiment_id": 91,
                "groups": [
                    {
                        "name": group.name,
                        "group_id": group.group_id,
                        "action_semantics": list(group.action_semantics),
                        "state_semantics": list(group.state_semantics),
                        "action_frame": group.action_frame,
                        "state_frame": group.state_frame,
                        "composition_operators": list(group.composition_operators),
                    }
                ],
            }
        ],
        "cache_representation": {
            "num_views": 3,
            "view_slots": ["head", "left_hand", "right_hand"],
            "missing_view_policy": "mask_without_duplication",
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_cli_sha_binding_and_no_clobber(legacy_fixture, tmp_path: Path, monkeypatch):
    root, plan, _row, adapter = legacy_fixture
    template = tmp_path / "profile.yaml"
    _template(template)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "wm3d_v8_source_adapter_audit_receipt_v1",
                "source": LEGACY_FORMAL_SOURCE,
                "adapter_contract_sha256": adapter.sha256,
                "data_template_sha256": sha256_file(template),
                "structural_checks": "pass",
                "semantic_review": "operator_confirmed_fail_closed",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "formal.jsonl"
    receipt = tmp_path / "receipt.json"
    argv = [
        "materialize_legacy_residual_inventory.py",
        "--legacy-plan",
        str(plan),
        "--data-template",
        str(template),
        "--source",
        LEGACY_FORMAL_SOURCE,
        "--raw-root",
        str(root),
        "--adapter-contract",
        str(adapter.path),
        "--adapter-contract-sha256",
        adapter.sha256,
        "--adapter-audit-receipt",
        str(audit),
        "--output-manifest",
        str(manifest),
        "--output-receipt",
        str(receipt),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    materializer.main()
    first_manifest = manifest.read_bytes()
    first_receipt = receipt.read_bytes()
    materializer.main()  # byte-identical idempotent rerun
    assert manifest.read_bytes() == first_manifest
    assert receipt.read_bytes() == first_receipt

    # A real payload SHA drift changes the rebuilt manifest and may never
    # overwrite the already-published formal inventory.
    with (root / "head.mp4").open("ab") as handle:
        handle.write(b"sha-drift")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materializer.main()
    assert manifest.read_bytes() == first_manifest
    assert receipt.read_bytes() == first_receipt


def test_cli_rejects_audit_sha_drift(legacy_fixture, tmp_path: Path):
    _root, _plan, _row, adapter = legacy_fixture
    template = tmp_path / "profile.yaml"
    _template(template)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema": "wm3d_v8_source_adapter_audit_receipt_v1",
                "source": LEGACY_FORMAL_SOURCE,
                "adapter_contract_sha256": "0" * 64,
                "data_template_sha256": sha256_file(template),
                "structural_checks": "pass",
                "semantic_review": "operator_confirmed_fail_closed",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not authorize"):
        materializer._audit_receipt(
            audit,
            source=LEGACY_FORMAL_SOURCE,
            adapter_sha=adapter.sha256,
            template_sha=sha256_file(template),
        )


def test_pair_preflight_does_not_publish_partial_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"stale\n")
    materializer._assert_publishable(manifest, b"new-manifest\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materializer._assert_publishable(receipt, b"new-receipt\n")
    assert not manifest.exists()
    assert receipt.read_bytes() == b"stale\n"
