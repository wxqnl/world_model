from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.preflight_wm3d_v8_stage0_causal_dual_view as preflight_module
from scripts.seal_wm3d_v8_stage0_causal_dual_view_canary import (
    _merge_robocasa_indices,
    _runtime_overlay,
)
from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    _Checks,
    _validate_dataset_probe_v8,
    CausalDualViewPreflightError,
    load_config,
    validate_preflight,
)
from wm3d_v3.training.train import (
    capture_rng_contract,
    validate_empty_checkpoint_dir_preflight,
)
from wm3d_v3.data.v8_causal_dual_view import (
    CAUSAL_DUAL_VIEW_REPRESENTATION as REPRESENTATION,
    CAUSAL_DUAL_VIEW_SCHEMA as SCHEMA,
    causal_dual_view_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
MIX = {
    "oxe_droid_action": 35,
    "oxe_bridge_action": 15,
    "robocasa_atomic": 10,
    "robocasa_composite": 20,
    "robocasa_mg": 20,
}


@pytest.mark.parametrize(
    "name,profile,max_steps",
    [
        ("wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml",
         "canary_non_inheritable", 100),
        ("wm3d_v8_stage0_causal_dual_view_unified_action_formal_v2.yaml",
         "formal", 100000),
    ],
)
def test_resolved_configs_lock_causal_action_policy_contract(
    name: str, profile: str, max_steps: int
) -> None:
    cfg = load_config(ROOT / "configs" / name)
    contract, model = cfg["contract"], cfg["model"]
    data, train = cfg["data"], cfg["train"]

    assert contract["profile"] == profile
    assert contract["causal_dual_view_schema"] == SCHEMA
    assert contract["causal_dual_view_representation"] == REPRESENTATION
    assert contract["context_future_leakage"] is False
    assert contract["target_usage"] == "supervision_only"
    assert contract["geometry_coordinate_frame"] == "first_observed_camera"
    assert contract["native_3d_outputs_required"] == [
        "rgb", "depth", "point", "pose"
    ]
    assert model["enable_action_policy"] is True
    assert model["policy_enable_flow_head"] is False
    assert model["policy_flow_use_as_policy"] is False
    assert model["policy_grip_owner"] == "absolute"
    assert model["policy_horizon"] == 8
    assert model["state"]["action_cond_dim"] == 36
    assert model["action"]["action_cond_dim"] == 36

    assert train["joint_native_action_pretraining"] is True
    assert train["direct_policy_weight"] == 1.0
    assert train["policy_flow_weight"] == 0.0
    assert train["native_action_no_teacher_weight"] == 0.0
    assert train["native_action_no_teacher_start_step"] == 0
    assert train["native_action_no_teacher_every"] == 1
    assert train["native_future_no_teacher_weight"] == 0.20
    assert train["native_future_no_teacher_weight_start_step"] == 0
    assert train["factual_action_conditioning"]["start_step"] == 0
    assert set(train["factual_action_conditioning"]["required_sources"]) == set(MIX)
    assert train["mixed_batch_sampler"]["source_cycle_counts_exact"] == MIX
    assert train["max_steps"] == max_steps
    assert train["fresh_initialization_required"] is True
    assert train["fresh_init_required"] is True
    assert train["forbid_warm_start"] is True
    assert train["stage_transition"] is None
    assert train["resume_checkpoint"] is None
    assert train["pretrained_world_checkpoint"] is None
    assert "causal_dual_view" in train["run_lineage"]

    assert data["T"] == 16 and data["k"] == 8
    assert data["v8_dual_rate_action_enabled"] is True
    assert data["compact_causal_dual_view_required"] is True
    assert data["compact_causal_dual_view_representation"] == REPRESENTATION
    assert set(data["causal_dual_view_indices"]) == set(MIX)
    override = data["direct_policy_oxe_overrides"]
    assert "cache_root" not in override
    assert override["window_geom_cache_root"] == (
        "/data/Minko/world_model/"
        "wm3d_v8_stage0_causal_dual_view_20260809/cache/oxe"
    )
    assert override["causal_dual_view_required"] is True
    assert override["causal_dual_view_representation"] == REPRESENTATION
    assert override["use_window_tokens"] is True
    assert override["load_state_tgt"] is True
    assert override["require_geom_extra"] is True


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(compact, split, identity, source, legacy=False):
    T, k, P, C = 16, 8, 64, 4
    lead = (1,) if compact else ()
    result = {
        **causal_dual_view_metadata(T=T, k=k),
        "context_codes": np.zeros(lead + (T, P, C), dtype=np.int8),
        "context_scale": np.ones(lead + (T, 1, 1), dtype=np.float16),
        "future_codes": np.ones(lead + (k, P, C), dtype=np.int8),
        "future_scale": np.ones(lead + (k, 1, 1), dtype=np.float16),
        "future_depth_patch": np.ones(lead + (k, 8, 8), dtype=np.float16),
        "future_depth_conf_patch": np.ones(lead + (k, 8, 8), dtype=np.float16),
        "future_point_patch": np.ones(lead + (k, 8, 8, 3), dtype=np.float16),
        "future_point_conf_patch": np.ones(lead + (k, 8, 8), dtype=np.float16),
        "future_pose_enc": np.ones(lead + (k, 9), dtype=np.float16),
        "split": np.asarray(split),
        "source": np.asarray(source),
        "token_count": np.asarray(P, dtype=np.int64),
        "token_dim": np.asarray(2048, dtype=np.int64),
        "latent_dim": np.asarray(C, dtype=np.int64),
    }
    if compact:
        result.update({
            "clip_hash": np.asarray(identity),
            "window_starts": np.asarray([0], dtype=np.int64),
            "wrist_context_codes": np.zeros((1, T, P, C), dtype=np.int8),
            "wrist_context_scale": np.ones((1, T, 1, 1), dtype=np.float16),
        })
    else:
        result.update({
            "clip_id": np.asarray(identity),
            "start": np.asarray(0, dtype=np.int64),
        })
    if legacy:
        result["schema"] = np.asarray("wm3d_v7_compact_geom_v3")
    return result


def _artifact(root, source, split, compact, legacy=False):
    identity = f"{source}_{split}"
    path = root / f"{identity}.npz"
    np.savez(path, **_payload(compact, split, identity, source, legacy))
    row = {
        "schema": "wm3d_v7_compact_geom_v3" if legacy else SCHEMA,
        "representation": REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": "supervision_only",
        "geometry_coordinate_frame": "first_observed_camera",
        "split": split,
        "source": source,
        "path": str(path),
        "artifact_sha256": _sha(path),
        "T": 16, "k": 8, "P": 64, "token_D": 2048, "latent_dim": 4,
    }
    if compact:
        row.update({
            "clip_hash": identity, "paired_views": True,
            "windows": 1, "window_starts": [0],
        })
    else:
        row.update({"clip_id": identity, "start": 0, "paired_views": False})
    return row


def _index(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return _sha(path)


def test_merge_robocasa_indices_binds_partition_identity(tmp_path: Path) -> None:
    inputs = {}
    for partition in ("atomic", "composite", "mg"):
        rows = []
        for split in ("train", "val"):
            row = _artifact(tmp_path, partition, split, True)
            row["source"] = "robocasa365"
            rows.append(row)
        path = tmp_path / f"{partition}.jsonl"
        _index(path, rows)
        inputs[partition] = path

    output = tmp_path / "combined.jsonl"
    digest, merged = _merge_robocasa_indices(inputs, output)
    assert digest == _sha(output)
    assert len(merged) == 6
    assert {row["v7_source"] for row in merged} == {
        "atomic", "composite", "mg"
    }
    assert {row["source"] for row in merged} == {"robocasa365"}
    assert all(row["schema"] == SCHEMA for row in merged)
    assert all(row["representation"] == REPRESENTATION for row in merged)

    replay_digest, replay_rows = _merge_robocasa_indices(inputs, output)
    assert replay_digest == digest
    assert replay_rows == merged


def test_runtime_overlay_preserves_exact_resume_training_schedule(
    tmp_path: Path,
) -> None:
    """The first hard stop must not shorten the optimizer/LR schedule."""

    base = tmp_path / "base.yaml"
    base.write_text("{}\n")
    windows = tmp_path / "causal_windows"
    oxe_paths = {}
    for source in ("oxe_droid_action", "oxe_bridge_action"):
        path = tmp_path / f"{source}.jsonl"
        path.write_text(json.dumps({
            "path": str(windows / f"{source}__start_000000.npz"),
        }) + "\n")
        oxe_paths[source] = [path]
    robocasa = tmp_path / "robocasa.jsonl"
    robocasa.write_text("{}\n")
    action_index = tmp_path / "action20.jsonl"
    action_index.write_text("{}\n")
    action_stats = tmp_path / "action20_stats.npz"
    action_stats.write_bytes(b"sealed stats")
    out_root = tmp_path / "results"

    overlay = _runtime_overlay(
        base_config=base,
        oxe_paths=oxe_paths,
        combined_robocasa_index=robocasa,
        action_sidecar_index=action_index,
        action_sidecar_stats=action_stats,
        max_steps=100,
        initial_stop_step=20,
        out_root=out_root,
        run_lineage="wm3d_v8_stage0_causal_dual_view_canary100_test",
    )

    assert overlay["train"]["max_steps"] == 100
    assert overlay["train"]["canary_initial_stop_step"] == 20
    assert overlay["train"]["planned_review_stop_step"] == 100
    assert overlay["train"]["ckpt_every_steps"] == 20
    assert overlay["train"]["checkpoint_milestone_steps"] == [20, 100]
    assert overlay["train"]["run_lineage"].endswith("canary100_test")
    assert overlay["out"] == {
        "root": str(out_root.resolve()),
        "require_empty_checkpoint_dir": True,
    }
    assert overlay["data"]["direct_policy_oxe_overrides"] == {
        "window_geom_cache_root": str(tmp_path.resolve()),
        "window_geom_subdir": "causal_windows",
        "window_geom_shard_index": None,
        "window_geom_shard_root": None,
    }
    assert overlay["data"]["v8_action_sidecar_index"] == str(
        action_index.resolve()
    )
    assert overlay["data"]["v8_action_sidecar_index_sha256"] == _sha(
        action_index
    )
    assert overlay["data"]["v8_action_sidecar_stats"] == str(
        action_stats.resolve()
    )
    assert overlay["data"]["v8_action_sidecar_stats_sha256"] == _sha(
        action_stats
    )


def test_runtime_overlay_rejects_split_oxe_window_directories(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("{}\n")
    oxe_paths = {}
    for source in ("oxe_droid_action", "oxe_bridge_action"):
        path = tmp_path / f"{source}.jsonl"
        artifact = tmp_path / source / "episode__start_000000.npz"
        path.write_text(json.dumps({"path": str(artifact)}) + "\n")
        oxe_paths[source] = [path]
    robocasa = tmp_path / "robocasa.jsonl"
    robocasa.write_text("{}\n")
    action_index = tmp_path / "action20.jsonl"
    action_index.write_text("{}\n")
    action_stats = tmp_path / "action20_stats.npz"
    action_stats.write_bytes(b"sealed stats")

    with pytest.raises(ValueError, match="one shared window directory"):
        _runtime_overlay(
            base_config=base,
            oxe_paths=oxe_paths,
            combined_robocasa_index=robocasa,
            action_sidecar_index=action_index,
            action_sidecar_stats=action_stats,
        )


def _config(tmp_path: Path, legacy_source=None):
    cfg = load_config(
        ROOT / "configs" /
        "wm3d_v8_stage0_causal_dual_view_unified_action_canary_v2.yaml"
    )
    indices = {}
    for source in ("oxe_droid_action", "oxe_bridge_action"):
        rows = [
            _artifact(tmp_path, source, split, False, source == legacy_source)
            for split in ("train", "val")
        ]
        path = tmp_path / f"{source}.jsonl"
        indices[source] = {
            "kind": "oxe", "paths": [str(path)],
            "sha256": [_index(path, rows)], "paired_views": False,
        }

    rows = []
    for partition in ("atomic", "composite", "mg"):
        source = f"robocasa_{partition}"
        rows.extend(
            _artifact(tmp_path, partition, split, True, source == legacy_source)
            for split in ("train", "val")
        )
    path = tmp_path / "robocasa.jsonl"
    digest = _index(path, rows)
    for partition in ("atomic", "composite", "mg"):
        indices[f"robocasa_{partition}"] = {
            "kind": "compact", "paths": [str(path)], "sha256": [digest],
            "partition": partition, "paired_views": True,
        }
    cfg["data"]["causal_dual_view_indices"] = indices
    cfg["data"]["compact_index"] = str(path)
    cfg["data"]["compact_index_sha256"] = digest
    cfg["out"]["root"] = str(tmp_path / "out")
    return cfg


def test_full_preflight_scans_all_five_sources(tmp_path: Path) -> None:
    report = validate_preflight(
        _config(tmp_path), mode="full",
        verify_training_assets=False, verify_local_resources=False,
    )
    assert report["passed"] is True and report["launch_ready"] is True
    assert report["source_coverage"] == {
        source: {"artifacts": 2, "train": 1, "val": 1} for source in MIX
    }
    assert set(report["cache_contract_hashes"]) == set(MIX)


def test_full_preflight_rejects_legacy_cache(tmp_path: Path) -> None:
    with pytest.raises(CausalDualViewPreflightError) as info:
        validate_preflight(
            _config(tmp_path, "oxe_droid_action"), mode="full",
            verify_training_assets=False, verify_local_resources=False,
        )
    assert any("schema" in error for error in info.value.report["errors"])


class _ProbeDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        return {
            "s_in": np.zeros((1,), dtype=np.float32),
            "s_tgt": np.zeros((8, 64, 384), dtype=np.float32),
            "action_tgt": np.zeros((8, 7), dtype=np.float32),
            "action_tgt_norm": np.zeros((8, 6), dtype=np.float32),
            "c": np.zeros((2048,), dtype=np.float32),
            "rgb_tgt": np.zeros((1,), dtype=np.float32),
            "depth_tgt": np.zeros((1,), dtype=np.float32),
            "point_tgt": np.zeros((1,), dtype=np.float32),
            "pose_geom_tgt": np.zeros((1,), dtype=np.float32),
        }


class _ProbeMix:
    def __init__(self, lengths: dict[str, int]) -> None:
        self.source_names = list(MIX)
        self.datasets = [_ProbeDataset(lengths[name]) for name in self.source_names]


def test_full_dataset_probe_rejects_legacy_unmarked_codec_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths = {source: 16 for source in MIX}
    monkeypatch.setattr(
        preflight_module,
        "build_datasets",
        lambda _config: (_ProbeMix(lengths), _ProbeMix(lengths)),
    )
    checks = _Checks("full")

    _validate_dataset_probe_v8(
        checks,
        {
            "train": {
                "batch_size_per_gpu": 2,
                "gpus_per_node": 8,
                "num_nodes": 1,
            }
        },
    )

    assert any(
        "dataset_probe.oxe_droid_action.s_tgt_codec is missing" in error
        for error in checks.errors
    )


def test_full_dataset_probe_requires_each_source_to_fill_global_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_lengths = {source: 16 for source in MIX}
    validation_lengths = {source: 16 for source in MIX}
    train_lengths["oxe_droid_action"] = 15
    validation_lengths["oxe_bridge_action"] = 15
    monkeypatch.setattr(
        preflight_module,
        "build_datasets",
        lambda _config: (_ProbeMix(train_lengths), _ProbeMix(validation_lengths)),
    )
    checks = _Checks("full")

    report = _validate_dataset_probe_v8(
        checks,
        {
            "train": {
                "batch_size_per_gpu": 2,
                "gpus_per_node": 8,
                "num_nodes": 1,
            }
        },
    )

    assert report["global_batch"] == 16
    assert report["train_source_lengths"]["oxe_droid_action"] == 15
    assert report["validation_source_lengths"]["oxe_bridge_action"] == 15
    assert any(
        "dataset_probe.oxe_droid_action train has 15 samples; "
        "requires global batch 16" in error
        for error in checks.errors
    )
    assert any(
        "dataset_probe.oxe_bridge_action validation has 15 samples; "
        "requires global batch 16" in error
        for error in checks.errors
    )


def test_world16_v2_config_is_global64_and_unified_action() -> None:
    cfg = load_config(
        ROOT
        / "configs"
        / "wm3d_v8_stage0_causal_dual_view_unified_action_formal100k_world16_node43_node44_v2.yaml"
    )
    assert cfg["contract"]["schema"] == (
        "wm3d_v8_stage0_causal_dual_view_unified_action_formal_v2"
    )
    assert cfg["contract"]["serving_action_owner"] == (
        "unified_base_policy_pose6_absolute_gripper"
    )
    assert cfg["train"]["num_nodes"] == 2
    assert cfg["train"]["gpus_per_node"] == 8
    assert cfg["train"]["batch_size_per_gpu"] == 2
    assert cfg["train"]["gradient_accumulation_steps"] == 2
    assert cfg["train"]["effective_global_batch"] == 64
    assert cfg["train"]["max_steps"] == 100000

def test_fresh_checkpoint_guard_allows_explicit_exact_resume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    checkpoint = root / "ckpt" / "step_00000020.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"numbered checkpoint")
    cfg = {
        "out": {
            "root": str(root),
            "ckpt_dir": "ckpt",
            "require_empty_checkpoint_dir": True,
        }
    }

    with pytest.raises(RuntimeError, match="checkpoint directory is not empty"):
        validate_empty_checkpoint_dir_preflight(
            cfg, resume_checkpoint=None
        )

    validate_empty_checkpoint_dir_preflight(
        cfg, resume_checkpoint=checkpoint
    )


def test_checkpoint_rng_contract_is_not_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight_module.torch.cuda, "is_available", lambda: False)

    contract = capture_rng_contract(base_seed=1707, rank=0)

    assert contract["schema"] == "wm3d_v7_step_addressed_rng_v1"
    assert contract["base_seed"] == 1707
    assert contract["rank"] == 0
    assert contract["rank_stride"] == 100_003
    assert contract["step_offset"] == 10_000_019
    assert contract["torch_cpu_state"].numel() > 0


def test_exact_resume_preflight_accepts_only_bound_checkpoint_lineage(
    tmp_path: Path,
) -> None:
    out_root = tmp_path / "out"
    ckpt_dir = out_root / "ckpt"
    ckpt_dir.mkdir(parents=True)
    resume = ckpt_dir / "step_00000020.pt"
    resume.write_bytes(b"checkpoint")
    (ckpt_dir / "latest.pt").symlink_to(resume.name)
    config = {
        "out": {
            "root": str(out_root),
            "require_empty_checkpoint_dir": True,
        }
    }
    validator = getattr(preflight_module, "_validate_checkpoint_lineage")

    checks = _Checks("full")
    health = validator(checks, config, exact_resume_checkpoint=resume)
    assert checks.errors == []
    assert health["checkpoint_files"] == [
        str(ckpt_dir / "latest.pt"),
        str(resume),
    ]

    (ckpt_dir / "step_00000100.pt").write_bytes(b"unexpected")
    checks = _Checks("full")
    validator(checks, config, exact_resume_checkpoint=resume)
    assert any("exact resume checkpoint lineage mismatch" in error for error in checks.errors)


def test_runtime_dependency_gate_requires_lpips_for_pixel_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight_module.importlib.util,
        "find_spec",
        lambda name: None if name == "lpips" else object(),
    )
    checks = _Checks("full")

    report = preflight_module._validate_runtime_dependencies(
        checks,
        {
            "model": {"enable_pixel": True},
            "loss": {"rgb_lpips": 0.55},
        },
    )

    assert report == {"lpips": False}
    assert checks.errors == [
        "runtime dependency lpips is required when loss.rgb_lpips > 0"
    ]
