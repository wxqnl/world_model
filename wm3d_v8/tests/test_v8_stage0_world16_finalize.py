from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_wm3d_v8_stage0_causal_dual_view_world16 import (
    CODEC_GATE_SHA256,
    CODEC_SHA256,
    OXE_SCHEMA,
    ROBO_SCHEMA,
    REPRESENTATION,
    ROW_SCHEMA,
    VGGT_REVISION,
    FamilySpec,
    collect_family,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_shard(root: Path, spec: FamilySpec, shard: int, identity: str) -> None:
    suffix = f"shard-{shard:05d}-of-{spec.shards:05d}"
    index = root / f"{spec.name}.{suffix}.jsonl"
    row = {
        "schema": ROW_SCHEMA,
        "representation": REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        "T": 16,
        "k": 8,
        "P": 64,
        "token_D": 2048,
        "latent_dim": 384,
        "source": spec.source,
        "split": spec.split,
        "paired_views": False,
        "clip_id": identity,
        "start": shard,
        "path": str(root / f"{identity}.npz"),
        "artifact_sha256": "a" * 64,
    }
    index.write_text(json.dumps(row) + "\n")
    report = {
        "schema": OXE_SCHEMA,
        "pass": True,
        "representation": REPRESENTATION,
        "source": spec.source,
        "split": spec.split,
        "shard_id": shard,
        "num_shards": spec.shards,
        "selected_global": spec.global_count,
        "selected_shard": 1,
        "encoded": 1,
        "selection_sha256": "b" * 64,
        "manifest_sha256": spec.manifest_sha256,
        "codec_sha256": CODEC_SHA256,
        "config_sha256": "d" * 64,
        "vggt_revision": VGGT_REVISION,
        "index_sha256": _sha(index),
    }
    index.with_suffix(".report.json").write_text(json.dumps(report))


def _spec() -> FamilySpec:
    return FamilySpec(
        name="oxe_unit_train",
        kind="oxe",
        shards=2,
        global_count=2,
        manifest_sha256="c" * 64,
        source="oxe_unit",
        split="train",
    )


def _write_robo_shard(root: Path, spec: FamilySpec, shard: int, identity: str) -> None:
    suffix = f"shard-{shard:05d}-of-{spec.shards:05d}"
    index = root / f"{spec.name}.{suffix}.jsonl"
    row = {
        "schema": ROW_SCHEMA,
        "representation": REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        "T": 16,
        "k": 8,
        "P": 64,
        "token_D": 2048,
        "source": "robocasa365",
        "v7_source": spec.partition,
        "split": "train" if shard == 0 else "val",
        "paired_views": True,
        "clip_hash": identity,
        "path": str(root / f"{identity}.npz"),
        "artifact_sha256": "e" * 64,
    }
    index.write_text(json.dumps(row) + "\n")
    report = {
        "schema": ROBO_SCHEMA,
        "pass": True,
        "causal_dual_view": True,
        "representation": REPRESENTATION,
        "v7_source": spec.partition,
        "global_selected_clips": spec.global_count,
        "clips": 1,
        "paired_views": True,
        "task_embedding_real": True,
        "rgb_sidecar_coverage_passed": True,
        "factual_action_audit_passed": True,
        "selection_sha256": "f" * 64,
        "manifest_sha256": spec.manifest_sha256,
        "codec_sha256": CODEC_SHA256,
        "codec_downstream_report_sha256": CODEC_GATE_SHA256,
        "config_sha256": "1" * 64,
        "geometry_teacher": {
            "revision": VGGT_REVISION,
            "pseudo_teacher": True,
        },
        "sharding": {
            "shard_id": shard,
            "num_shards": spec.shards,
            "assigned_clips": 1,
        },
        "index_sha256": _sha(index),
    }
    index.with_suffix(".report.json").write_text(json.dumps(report))


def _robo_spec() -> FamilySpec:
    return FamilySpec(
        name="robocasa_unit",
        kind="robocasa",
        shards=2,
        global_count=2,
        manifest_sha256="2" * 64,
        partition="unit",
    )


def test_collect_family_merges_in_identity_order(tmp_path: Path) -> None:
    spec = _spec()
    _write_shard(tmp_path, spec, 0, "z")
    _write_shard(tmp_path, spec, 1, "a")
    data, evidence, artifacts = collect_family(tmp_path, spec)
    rows = [json.loads(line) for line in data.decode().splitlines()]
    assert [row["clip_id"] for row in rows] == ["a", "z"]
    assert evidence["actual_count"] == 2
    assert evidence["merged_index_sha256"] == hashlib.sha256(data).hexdigest()
    assert len(artifacts) == 2


def test_collect_family_rejects_duplicate_identity(tmp_path: Path) -> None:
    spec = _spec()
    _write_shard(tmp_path, spec, 0, "same")
    _write_shard(tmp_path, spec, 1, "same")
    with pytest.raises(ValueError, match="duplicate (artifact path|row identity)"):
        collect_family(tmp_path, spec)


def test_collect_family_rejects_tampered_index(tmp_path: Path) -> None:
    spec = _spec()
    _write_shard(tmp_path, spec, 0, "x")
    _write_shard(tmp_path, spec, 1, "y")
    index = tmp_path / "oxe_unit_train.shard-00001-of-00002.jsonl"
    index.write_text(index.read_text() + "{}\n")
    with pytest.raises(ValueError, match="index SHA mismatch"):
        collect_family(tmp_path, spec)


def test_collect_robocasa_family_validates_causal_receipts(tmp_path: Path) -> None:
    spec = _robo_spec()
    _write_robo_shard(tmp_path, spec, 0, "b")
    _write_robo_shard(tmp_path, spec, 1, "a")
    data, evidence, artifacts = collect_family(tmp_path, spec)
    rows = [json.loads(line) for line in data.decode().splitlines()]
    assert [row["clip_hash"] for row in rows] == ["b", "a"]
    assert evidence["actual_count"] == 2
    assert evidence["producer_config_sha256"] == "1" * 64
    assert len(artifacts) == 2


def test_collect_robocasa_rejects_noncausal_receipt(tmp_path: Path) -> None:
    spec = _robo_spec()
    _write_robo_shard(tmp_path, spec, 0, "a")
    _write_robo_shard(tmp_path, spec, 1, "b")
    report = tmp_path / "robocasa_unit.shard-00001-of-00002.report.json"
    payload = json.loads(report.read_text())
    payload["causal_dual_view"] = False
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="causal dual-view receipt is false"):
        collect_family(tmp_path, spec)
