from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.wm3d_v8_preflight_common import _Checks
from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    _validate_v8_action_sidecars,
)
from scripts.build_wm3d_v8_dual_rate_action_sidecars import (
    _RunningPoseStats,
    _build_one,
    _load_index_rows,
    _publish_bytes_no_clobber,
    _publish_npz_no_clobber,
    sha256_file,
)
from wm3d_v3.data.v7_action_contract import (
    ActionAdapter,
    canonicalize_dense_action,
    resample_canonical_actions,
)
from wm3d_v3.data.v8_action_contract import (
    V8_ACTION_STATS_SCHEMA,
)


def test_action_sidecar_publish_is_no_clobber_and_idempotent(tmp_path: Path) -> None:
    payload = {"value": np.arange(8, dtype=np.float32)}
    npz_path = tmp_path / "sidecar.npz"
    first = _publish_npz_no_clobber(npz_path, payload)
    second = _publish_npz_no_clobber(npz_path, payload)
    assert first == second
    text_path = tmp_path / "index.jsonl"
    first_text = _publish_bytes_no_clobber(text_path, b"{}\n")
    second_text = _publish_bytes_no_clobber(text_path, b"{}\n")
    assert first_text == second_text
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_action20_stats_are_train_streaming_statistics() -> None:
    stats = _RunningPoseStats()
    values = np.arange(60, dtype=np.float32).reshape(10, 6) / 100.0
    stats.update(values[:4])
    stats.update(values[4:])
    mean, std = stats.arrays()
    assert stats.count == 10
    assert np.allclose(mean, values.mean(axis=0), atol=1e-7)
    assert np.allclose(std, values.std(axis=0), atol=1e-7)


def test_builder_accepts_production_archive_without_redundant_partition_key(
    tmp_path: Path,
) -> None:
    adapter = ActionAdapter(
        source="robocasa365",
        source_frame="robot_base",
        translation_unit_scale=1.0,
        rotation_unit_scale=1.0,
        rotation_repr="axis_angle",
        nominal_hz=20.0,
    )
    raw = np.zeros((8, 12), dtype=np.float32)
    raw[:, 5] = 0.01
    raw[:, 8] = 0.005
    raw[:, 11] = np.asarray([-1, -1, -1, 1, 1, 1, 1, -1], dtype=np.float32)
    dense = np.concatenate((raw[:, 5:11], raw[:, 11:12]), axis=1)
    fine = canonicalize_dense_action(dense, adapter)
    world = resample_canonical_actions(fine, source_hz=20.0, target_hz=5.0)
    digest = "a" * 64
    clip_hash = "b" * 64
    archive_path = tmp_path / "production_shape.npz"
    np.savez(
        archive_path,
        clip_hash=np.asarray(clip_hash),
        split=np.asarray("train"),
        source=np.asarray("robocasa365"),
        action_audit_sha256=np.asarray(digest),
        source_control_hz=np.asarray(20.0),
        model_control_hz=np.asarray(5.0),
        raw_actions=raw,
        actions=world,
        native_frame_indices=np.arange(len(raw), dtype=np.int64),
    )
    row, observed_fine_pose = _build_one(
        {
            "clip_hash": clip_hash,
            "split": "train",
            "source": "robocasa365",
            "v7_source": "atomic",
            "path": str(archive_path),
            "artifact_sha256": sha256_file(archive_path),
            "action_audit_sha256": digest,
        },
        adapter=adapter,
        adapter_sha256=digest,
        output_root=tmp_path / "output",
        composition_atol=2.0e-5,
        dry_run=True,
    )
    assert row["v7_source"] == "atomic"
    assert row["fine_action_count"] == 8
    assert observed_fine_pose.shape == (8, 6)


def test_builder_rejects_source_archive_digest_mismatch(tmp_path: Path) -> None:
    adapter = ActionAdapter(
        source="robocasa365",
        source_frame="robot_base",
        translation_unit_scale=1.0,
        rotation_unit_scale=1.0,
        rotation_repr="axis_angle",
        nominal_hz=20.0,
    )
    raw = np.zeros((8, 12), dtype=np.float32)
    raw[:, 11] = -1.0
    dense = np.concatenate((raw[:, 5:11], raw[:, 11:12]), axis=1)
    world = resample_canonical_actions(
        canonicalize_dense_action(dense, adapter),
        source_hz=20.0,
        target_hz=5.0,
    )
    digest = "a" * 64
    clip_hash = "b" * 64
    archive_path = tmp_path / "source.npz"
    np.savez(
        archive_path,
        clip_hash=np.asarray(clip_hash),
        split=np.asarray("train"),
        source=np.asarray("robocasa365"),
        action_audit_sha256=np.asarray(digest),
        source_control_hz=np.asarray(20.0),
        model_control_hz=np.asarray(5.0),
        raw_actions=raw,
        actions=world,
        native_frame_indices=np.arange(len(raw), dtype=np.int64),
    )
    with pytest.raises(RuntimeError, match="source archive digest mismatch"):
        _build_one(
            {
                "clip_hash": clip_hash,
                "split": "train",
                "source": "robocasa365",
                "v7_source": "atomic",
                "path": str(archive_path),
                "artifact_sha256": "c" * 64,
                "action_audit_sha256": digest,
            },
            adapter=adapter,
            adapter_sha256=digest,
            output_root=tmp_path / "output",
            composition_atol=2.0e-5,
            dry_run=True,
        )


def test_builder_rejects_even_identical_duplicate_clip_rows(tmp_path: Path) -> None:
    row = {
        "clip_hash": "d" * 64,
        "split": "train",
        "source": "robocasa365",
    }
    payload = json.dumps(row, sort_keys=True) + "\n"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate clip is forbidden"):
        _load_index_rows([first, second])


def test_full_preflight_audits_exact_sidecar_identity_and_statistics(
    tmp_path: Path,
) -> None:
    adapter = ActionAdapter(
        source="robocasa365",
        source_frame="robot_base",
        translation_unit_scale=1.0,
        rotation_unit_scale=1.0,
        rotation_repr="axis_angle",
        nominal_hz=20.0,
    )
    raw = np.zeros((8, 12), dtype=np.float32)
    raw[:, 5] = 0.01
    raw[:, 8] = 0.005
    raw[:, 11] = np.asarray([-1, -1, -1, 1, 1, 1, 1, -1], dtype=np.float32)
    dense = np.concatenate((raw[:, 5:11], raw[:, 11:12]), axis=1)
    fine = canonicalize_dense_action(dense, adapter)
    world = resample_canonical_actions(fine, source_hz=20.0, target_hz=5.0)
    digest = "a" * 64
    clip_hash = "b" * 64
    archive_path = tmp_path / "source.npz"
    np.savez(
        archive_path,
        clip_hash=np.asarray(clip_hash),
        split=np.asarray("train"),
        source=np.asarray("robocasa365"),
        action_audit_sha256=np.asarray(digest),
        source_control_hz=np.asarray(20.0),
        model_control_hz=np.asarray(5.0),
        raw_actions=raw,
        actions=world,
        native_frame_indices=np.arange(len(raw), dtype=np.int64),
    )
    compact_row = {
        "clip_hash": clip_hash,
        "split": "train",
        "source": "robocasa365",
        "v7_source": "atomic",
        "path": str(archive_path.resolve()),
        "artifact_sha256": sha256_file(archive_path),
        "action_audit_sha256": digest,
    }
    compact_index = tmp_path / "compact.jsonl"
    compact_payload = (
        json.dumps(compact_row, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    compact_sha = _publish_bytes_no_clobber(compact_index, compact_payload)
    sidecar_row, fine_pose = _build_one(
        compact_row,
        adapter=adapter,
        adapter_sha256=digest,
        output_root=tmp_path / "sidecars",
        composition_atol=2.0e-5,
        dry_run=False,
    )
    sidecar_index = tmp_path / "sidecar_index.jsonl"
    sidecar_index_sha = _publish_bytes_no_clobber(
        sidecar_index,
        (
            json.dumps(sidecar_row, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode(),
    )
    stats = _RunningPoseStats()
    stats.update(fine_pose)
    mean, std = stats.arrays()
    stats_path = tmp_path / "stats.npz"
    stats_sha = _publish_npz_no_clobber(
        stats_path,
        {
            "schema": np.asarray(V8_ACTION_STATS_SCHEMA),
            "split": np.asarray("train"),
            "mean": mean,
            "std": std,
            "count": np.asarray(stats.count, dtype=np.int64),
            "world_hz": np.asarray(5, dtype=np.int64),
            "policy_hz": np.asarray(20, dtype=np.int64),
            "input_indices_json": np.asarray(
                json.dumps(
                    {str(compact_index.resolve()): compact_sha},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "adapter_audits_json": np.asarray(
                json.dumps(
                    {"atomic": digest}, sort_keys=True, separators=(",", ":")
                )
            ),
        },
    )
    checks = _Checks("full")
    report = _validate_v8_action_sidecars(
        checks,
        {
            "v8_dual_rate_action_enabled": True,
            "compact_index": str(compact_index),
            "compact_index_sha256": compact_sha,
            "v8_action_sidecar_index": str(sidecar_index),
            "v8_action_sidecar_index_sha256": sidecar_index_sha,
            "v8_action_sidecar_stats": str(stats_path),
            "v8_action_sidecar_stats_sha256": stats_sha,
        },
    )
    assert checks.errors == []
    assert report == {"enabled": True, "rows": 1}
