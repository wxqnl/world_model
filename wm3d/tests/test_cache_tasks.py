from __future__ import annotations

from pathlib import Path

from wm3d.data.cache_tasks import AtomicTaskClaim, plan_tasks
from wm3d.data.manifest_contract import sha256_file


def _source_row():
    return {
        "schema": "wm3d_v8_source_manifest_v4",
        "source": "dual",
        "episode_id": "ep-1",
        "payload": "raw/shared.parquet",
        "payload_sha256": "a" * 64,
        "payload_row_start": 100,
        "payload_row_stop": 200,
        "assets": [
            {
                "role": "primary_payload",
                "path": "raw/shared.parquet",
                "sha256": "a" * 64,
            },
            {"role": "rgb/head", "path": "video/head.mp4", "sha256": "f" * 64},
        ],
        "views": [
            {
                "name": "head",
                "asset_role": "rgb/head",
                "segment_kind": "entire_file",
                "start_s": None,
                "stop_s": None,
            }
        ],
        "task_text": "coordinate both arms",
        "embodiment": "dual",
        "split": "train",
        "observation_samples": 100,
        "observation_clock": {
            "key": "timestamp", "origin": "recorded_payload_timestamps",
            "unit": "seconds", "sample_count": 100, "start_s": 0.0,
            "end_s": 3.3, "min_dt_s": 0.03, "max_dt_s": 0.04,
            "timestamp_sha256": "9" * 64,
        },
        "robot_groups": {"left_arm": {"supervision": "fine_command"}},
    }


def _task():
    return plan_tasks(
        [_source_row()],
        source_manifest_sha256="1" * 64,
        adapter_contract_sha256="b" * 64,
        encoder_contract_sha256="c" * 64,
        task_encoder_contract_sha256="f" * 64,
        task_bank_index_sha256="2" * 64,
        representation_contract_sha256="d" * 64,
        canonical_view_slots=("head", "left_wrist"),
    )[0]


def test_task_identity_changes_when_any_contract_changes() -> None:
    first = _task()
    common = {
        "source_manifest_sha256": "1" * 64,
        "adapter_contract_sha256": "b" * 64,
        "encoder_contract_sha256": "c" * 64,
        "task_encoder_contract_sha256": "f" * 64,
        "task_bank_index_sha256": "2" * 64,
        "representation_contract_sha256": "d" * 64,
        "canonical_view_slots": ("head", "left_wrist"),
    }
    for field, replacement in (
        ("source_manifest_sha256", "3" * 64),
        ("adapter_contract_sha256", "4" * 64),
        ("encoder_contract_sha256", "5" * 64),
        ("task_encoder_contract_sha256", "6" * 64),
        ("task_bank_index_sha256", "7" * 64),
        ("representation_contract_sha256", "8" * 64),
    ):
        changed = dict(common)
        changed[field] = replacement
        second = plan_tasks([_source_row()], **changed)[0]
        assert first.task_id != second.task_id, field


def test_task_identity_binds_episode_slice_and_every_source_asset() -> None:
    first = _task()
    changed_slice = _source_row()
    changed_slice["payload_row_start"] = 101
    changed_slice["payload_row_stop"] = 201
    changed_asset = _source_row()
    changed_asset["assets"][1]["sha256"] = "e" * 64
    common = dict(
        source_manifest_sha256="1" * 64,
        adapter_contract_sha256="b" * 64,
        encoder_contract_sha256="c" * 64,
        task_encoder_contract_sha256="f" * 64,
        task_bank_index_sha256="2" * 64,
        representation_contract_sha256="d" * 64,
        canonical_view_slots=("head", "left_wrist"),
    )
    assert plan_tasks([changed_slice], **common)[0].task_id != first.task_id
    assert plan_tasks([changed_asset], **common)[0].task_id != first.task_id


def test_task_planner_rejects_view_outside_profile_slots() -> None:
    row = _source_row()
    row["views"][0]["name"] = "mystery_camera"
    try:
        plan_tasks(
            [row],
            source_manifest_sha256="1" * 64,
            adapter_contract_sha256="b" * 64,
            encoder_contract_sha256="c" * 64,
            task_encoder_contract_sha256="f" * 64,
            task_bank_index_sha256="2" * 64,
            representation_contract_sha256="d" * 64,
            canonical_view_slots=("head", "left_wrist"),
        )
    except Exception as exc:
        assert "canonical slots" in str(exc)
    else:
        raise AssertionError("unknown view must fail closed")


def test_completed_task_is_verified_and_skipped(tmp_path: Path) -> None:
    task = _task()
    output = tmp_path / "out.bin"
    output.write_bytes(b"sealed")
    with AtomicTaskClaim(tmp_path, task) as claim:
        claim.publish_receipt({output: sha256_file(output)})
    assert AtomicTaskClaim(tmp_path, task).completed()
    assert not (tmp_path / "claims" / f"{task.task_id}.claim").exists()
