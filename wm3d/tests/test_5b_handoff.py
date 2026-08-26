from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml
import pytest

from scripts.data.materialize_oxe_default import build_templates
from wm3d.models.model_factory import validate_model_profile
from wm3d.training.runtime_contract import validate_runtime_profile


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "profile,total_steps,checkpoint_steps,checkpoint_interval,teacher_decay_steps",
    [
        ("h200_64_fsdp2_canary1k.yaml", 1_000, [100, 500], 1_000, 750),
        ("h200_64_fsdp2_validation100k.yaml", 100_000, [], 1_000, 10_000),
        (
            "h200_64_fsdp2.yaml",
            600_000,
            [1_000, 5_000, 20_000],
            20_000,
            10_000,
        ),
    ],
)
def test_5b_presets_match_dual_path_5b_and_64_h200(
    profile: str,
    total_steps: int,
    checkpoint_steps: list[int],
    checkpoint_interval: int,
    teacher_decay_steps: int,
) -> None:
    model = yaml.safe_load(
        (ROOT / "configs/model/native_5b_dual_path.yaml").read_text()
    )
    objective = yaml.safe_load(
        (ROOT / "configs/objective/stage0_native_dual_path.yaml").read_text()
    )
    runtime = yaml.safe_load((ROOT / "configs/runtime" / profile).read_text())
    validate_model_profile(model)
    validate_runtime_profile(runtime)
    assert model["expected_parameter_count"] == 5_548_022_136
    assert model["model"]["schema"] == "wm3d_native_world_model_v2"
    assert model["model"]["P"] == 144
    assert model["model"]["appearance_P"] == 256
    assert model["model"]["rgb_hidden"] == 1536
    assert model["model"]["rgb_res_blocks"] == 2
    assert model["model"]["rgb_decode_chunk_size"] == 2
    assert model["model"]["rgb_decode_indices"] == list(range(16))
    assert model["model"]["factual_dynamics_repeats"] == 2
    assert model["model"]["factual_action_residual_scale"] == 0.3
    assert "render_factual_dynamics_repeats" not in model["model"]
    assert "render_factual_action_residual_scale" not in model["model"]
    assert model["model"]["appearance_action_residual_scale"] == 0.0
    assert model["model"]["rgb_context_enabled"] is True
    assert model["model"]["rgb_context_residual_scale"] == 0.75
    assert model["model"]["rgb_context_motion_blend_gain"] == 0.5
    assert objective["objective"]["rgb_l1"] == 0.5
    assert objective["objective"]["rgb_charbonnier"] == 1.0
    assert objective["objective"]["rgb_charbonnier_epsilon"] == 0.000001
    assert objective["objective"]["action_counterfactual_token_advantage"] == 1.0
    assert objective["objective"]["action_counterfactual_token_margin"] == 0.005
    assert objective["objective"]["action_counterfactual_rgb_advantage"] == 0.0
    assert objective["objective"]["action_counterfactual_rgb_margin"] == 0.002
    assert objective["objective"]["rgb_perceptual"] == 0.1
    assert objective["objective"]["rgb_motion_l1"] == 1.0
    assert objective["objective"]["rgb_motion_bce"] == 0.03
    assert objective["objective"]["rgb_motion_dice"] == 0.03
    assert objective["objective"]["appearance_mse"] == 1.0
    assert objective["objective"]["appearance_cosine"] == 0.1
    assert runtime["expected_world_size"] == 64
    assert runtime["distributed"]["shard_degree"] == 8
    assert runtime["resources"]["gpu_name_substring"] == "H200"
    assert runtime["resources"]["minimum_ib_rate_gbps"] == 400.0
    assert runtime["train"]["micro_batch_size"] == 4
    assert runtime["train"]["validation_micro_batch_size"] == 1
    assert runtime["train"]["gradient_accumulation"] == 1
    assert runtime["train"]["global_batch_size"] == 256
    assert runtime["train"]["total_steps"] == total_steps
    assert runtime["train"]["checkpoint_steps"] == checkpoint_steps
    assert runtime["train"]["checkpoint_interval"] == checkpoint_interval
    assert runtime["train"]["rgb_decode_chunk_size"] == 2
    assert runtime["train"]["rgb_perceptual_chunk_size"] == 8
    assert runtime["train"]["appearance_teacher_start_ratio"] == 1.0
    assert runtime["train"]["appearance_teacher_end_ratio"] == 0.0
    assert runtime["train"]["appearance_validation_three_way"] is True
    assert (
        runtime["train"]["appearance_teacher_decay_steps"]
        == teacher_decay_steps
    )


def test_5b_site_init_is_no_clobber(tmp_path: Path) -> None:
    destination = tmp_path / "site.env"
    command = [
        "bash",
        str(ROOT / "scripts/cluster/wm3d_5b.sh"),
        "init",
        "canary1k",
        str(destination),
    ]
    first = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert first.returncode == 0, first.stderr
    assert destination.is_file()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert "WM3D_5B_PRESET=canary1k" in destination.read_text()
    payload = destination.read_bytes()
    second = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    assert second.returncode == 2
    assert destination.read_bytes() == payload


@pytest.mark.parametrize(
    "data_mode,detail",
    [
        ("direct_raw", "direct VGGT"),
        ("streaming_raw", "stream LRU"),
        ("episode_cache", "episode cache"),
    ],
)
def test_5b_init_selects_data_mode(
    tmp_path: Path, data_mode: str, detail: str
) -> None:
    site = tmp_path / f"{data_mode}.env"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/cluster/wm3d_5b.sh"),
            "init",
            "canary1k",
            str(site),
            data_mode,
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = site.read_text()
    assert f"WM3D_DATA_MODE={data_mode}" in payload
    payload = payload.replace(
        "PYTHON_BIN=${ENV_DIR}/bin/python", f"PYTHON_BIN={sys.executable}"
    )
    site.write_text(payload)
    plan = subprocess.run(
        ["bash", str(ROOT / "scripts/cluster/wm3d_5b.sh"), "plan", str(site)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert plan.returncode == 0, plan.stderr
    assert f"data mode:  {data_mode}" in plan.stdout
    assert detail in plan.stdout


@pytest.mark.parametrize(
    "preset,runtime,steps",
    [
        ("canary1k", "h200_64_fsdp2_canary1k.yaml", "1000"),
        ("validation100k", "h200_64_fsdp2_validation100k.yaml", "100000"),
        ("formal600k", "h200_64_fsdp2.yaml", "600000"),
    ],
)
def test_5b_init_selects_a_complete_preset(
    tmp_path: Path, preset: str, runtime: str, steps: str
) -> None:
    site = tmp_path / f"{preset}.env"
    init = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/cluster/wm3d_5b.sh"),
            "init",
            preset,
            str(site),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert init.returncode == 0, init.stderr
    payload = site.read_text()
    payload = payload.replace("PYTHON_BIN=${ENV_DIR}/bin/python", f"PYTHON_BIN={sys.executable}")
    payload = payload.replace(
        "HF_TOKEN_FILE=/data/secrets/huggingface_token",
        f"HF_TOKEN_FILE={tmp_path / 'token'}",
    )
    site.write_text(payload)
    (tmp_path / "token").write_text("test")
    plan = subprocess.run(
        ["bash", str(ROOT / "scripts/cluster/wm3d_5b.sh"), "plan", str(site)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert plan.returncode == 0, plan.stderr
    assert f"preset:     {preset}" in plan.stdout
    assert runtime in plan.stdout
    assert f"final step: {steps}" in plan.stdout
    assert f"runs/5b_{preset}" in plan.stdout


def test_5b_init_rejects_unknown_preset(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/cluster/wm3d_5b.sh"),
            "init",
            "not-a-preset",
            str(tmp_path / "site.env"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "未知 5B preset" in result.stderr


def test_5b_init_rejects_retired_validation10k_preset(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/cluster/wm3d_5b.sh"),
            "init",
            "validation10k",
            str(tmp_path / "site.env"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "未知 5B preset" in result.stderr


def test_5b_lock_passes_exact_license_confirmation() -> None:
    wrapper = (ROOT / "scripts/cluster/wm3d_5b.sh").read_text()
    assert "ACCEPT_DATA_LICENSES=YES" in wrapper
    assert "--confirm-licenses YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES" in wrapper


def test_5b_site_defaults_to_oxe_and_direct_p144_p256() -> None:
    site = (ROOT / "configs/cluster/h200_5b.env.example").read_text()
    assert "NNODES=8" in site
    assert "GPUS_PER_NODE=8" in site
    assert "CACHE_WORKER_COUNT=64" in site
    assert "CACHE_BATCH_FRAMES=16" in site
    assert "CACHE_DECODE_WORKERS=4" in site
    assert "CACHE_WRITER_THREADS=2" in site
    assert "DATA_FAMILY=public_robot_oxe" in site
    assert "INCLUDE_AGIBOT_BETA=NO" in site
    assert "WM3D_DATA_MODE=direct_raw" in site
    assert "DIRECT_INPUT_RGB_SIZE=518" in site
    assert "DIRECT_DECODE_WORKERS=1" in site
    assert "DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK=1" in site
    assert "DIRECT_PREFETCH_WINDOWS=8" in site
    assert "DIRECT_VIDEO_INDEX_CACHE_ASSETS=128" in site
    assert "DIRECT_ENCODE_CHUNK_ROWS=32" in site
    assert "DIRECT_MINIMUM_CHUNK_ROWS=4" in site
    assert "DIRECT_APPEARANCE_FEATURE_LAYER=4" in site
    assert "STREAMING_LRU_ROOT=" not in site
    assert "STREAMING_LRU_GIB_PER_RANK=" not in site
    assert "MODEL_PROFILE=configs/model/native_5b_dual_path.yaml" in site
    assert (
        "ENCODER_CONTRACT=configs/encoder/vggt_native_p144_appearance_p256.yaml"
        in site
    )
    assert "OBJECTIVE_PROFILE=configs/objective/stage0_native_dual_path.yaml" in site
    assert "SOURCE_TEMPLATE=${CONTROL_ROOT}/public_sources_oxe.template.yaml" in site
    assert "DATA_TEMPLATE=${CONTROL_ROOT}/public_robot_oxe.template.yaml" in site


def test_5b_direct_template_has_optimized_runtime_defaults() -> None:
    site = (ROOT / "configs/cluster/h200_5b_direct.env.example").read_text()
    assert "WM3D_DATA_MODE=direct_raw" in site
    assert "DIRECT_DECODE_WORKERS=1" in site
    assert "DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK=1" in site
    assert "MODEL_PROFILE=configs/model/native_5b_dual_path.yaml" in site
    assert "ENCODER_CONTRACT=configs/encoder/vggt_native_p144_appearance_p256.yaml" in site


def _oxe_info(action_dim: int, state_dim: int, views: int = 1) -> dict:
    features = {
        "action": {"dtype": "float32", "shape": [action_dim]},
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
    }
    for index in range(views):
        features[f"observation.images.camera_{index}"] = {
            "dtype": "video",
            "shape": [128, 128, 3],
        }
    return {"features": features}


def test_5b_oxe_generator_adds_sources_without_a_fixed_pool_share() -> None:
    default = yaml.safe_load(
        (ROOT / "configs/data/public_robot_6106h.template.yaml").read_text()
    )
    source = yaml.safe_load(
        (ROOT / "configs/sources/public_sources.template.yaml").read_text()
    )
    repos = ("lerobot/droid_1.0.1",) + tuple(
        f"lerobot/fixture_{index:02d}" for index in range(55)
    )
    generated_sources, generated_data, adapters = build_templates(
        base_source=source,
        base_data=default,
        repo_ids=repos,
        metadata_by_repo={repo: _oxe_info(7, 7, 2) for repo in repos[1:]},
    )
    default_weights = {row["name"]: row["weight"] for row in default["sources"]}
    generated_weights = {
        row["name"]: row["weight"] for row in generated_data["sources"]
    }

    assert default_weights["agibot_beta"] == 30
    assert "agibot_beta" not in generated_weights
    main = {
        name: value
        for name, value in generated_weights.items()
        if not name.startswith("oxe_")
    }
    assert main == {
        name: value
        for name, value in default_weights.items()
        if name != "agibot_beta"
    }
    oxe = {
        name: value
        for name, value in generated_weights.items()
        if name.startswith("oxe_")
    }
    assert len(oxe) == 55
    assert set(oxe.values()) == {1}
    assert sum(default_weights.values()) == 100
    assert sum(main.values()) == 70
    assert sum(generated_weights.values()) == 125
    assert len(generated_sources["sources"]) == 61
    assert len(generated_data["sources"]) == 63
    assert generated_data["name"] == "public_robot_oxe"
    assert generated_data["notes"]["agibot_beta_enabled"] is False
    assert generated_data["notes"]["oxe_dataset_count_including_droid"] == 56
    assert generated_data["notes"]["oxe_new_source_count"] == 55
    assert generated_data["notes"]["oxe_added_source_weight"] == 1
    assert len(adapters) == 55
    assert all(
        adapter["groups"][0]["action"][0]["columns"] == list(range(7))
        for adapter in adapters.values()
    )


def test_5b_oxe_generator_keeps_beta_only_when_explicitly_enabled() -> None:
    default = yaml.safe_load(
        (ROOT / "configs/data/public_robot_6106h.template.yaml").read_text()
    )
    source = yaml.safe_load(
        (ROOT / "configs/sources/public_sources.template.yaml").read_text()
    )
    repos = ("lerobot/droid_1.0.1", "lerobot/fixture")
    generated_sources, generated_data, _ = build_templates(
        base_source=source,
        base_data=default,
        repo_ids=repos,
        metadata_by_repo={"lerobot/fixture": _oxe_info(7, 7, 2)},
        include_agibot_beta=True,
    )
    source_names = {row["name"] for row in generated_sources["sources"]}
    weights = {row["name"]: row["weight"] for row in generated_data["sources"]}

    assert {"agibot_beta", "agibot_alpha_converter"} <= source_names
    assert weights["agibot_beta"] == 30
    assert weights["oxe_fixture"] == 1
    assert sum(weights.values()) == 101
    assert generated_data["name"] == "public_robot_oxe_with_agibot_beta"
    assert generated_data["notes"]["agibot_beta_enabled"] is True


def test_5b_oxe_generator_rejects_capacity_overflow() -> None:
    default = yaml.safe_load(
        (ROOT / "configs/data/public_robot_6106h.template.yaml").read_text()
    )
    source = yaml.safe_load(
        (ROOT / "configs/sources/public_sources.template.yaml").read_text()
    )
    with pytest.raises(ValueError, match="outside WM3D capacity"):
        build_templates(
            base_source=source,
            base_data=default,
            repo_ids=("lerobot/droid_1.0.1", "lerobot/too_wide"),
            metadata_by_repo={"lerobot/too_wide": _oxe_info(17, 7)},
        )


def test_5b_cache_wrapper_wires_decode_encode_write_parallelism() -> None:
    wrapper = (ROOT / "scripts/cluster/wm3d_5b.sh").read_text()
    assert '--decode-workers "${CACHE_DECODE_WORKERS:-4}"' in wrapper
    assert '--writer-threads "${CACHE_WRITER_THREADS:-2}"' in wrapper
    assert '--batch-frames "${CACHE_BATCH_FRAMES:-16}"' in wrapper
    assert "--fail-fast" in wrapper


def test_5b_report_accepts_complete_synthetic_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/step_00000010"
    checkpoint.mkdir(parents=True)
    metrics = {
        "step": 10,
        "lr": 1.0e-4,
        "source_id": 1,
        "grad_norm": 1.25,
        "seconds_per_log_interval": 2.0,
        "total": 4.0,
        "token_mse": 2.0,
    }
    (run / "train_metrics.jsonl").write_text(json.dumps(metrics) + "\n")
    ownership = {
        "schema": "wm3d_v8_gradient_ownership_v2",
        "passed": True,
        "owners": {
            "native": {
                "required": True,
                "passed": True,
                "nonzero_elements": 2,
                "nonfinite_elements": 0,
            }
        },
    }
    (run / "gradient_ownership.json").write_text(json.dumps(ownership))
    metadata = {"schema": "wm3d_v8_distributed_checkpoint_v2", "step": 10, "world_size": 1}
    payload = checkpoint / "payload.bin"
    payload.write_bytes(b"sealed")
    (checkpoint / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
    manifest = {
        "schema": "wm3d_v8_distributed_checkpoint_v2",
        "step": 10,
        "files": {
            "metadata.json": {
                "size": (checkpoint / "metadata.json").stat().st_size,
                "sha256": _sha(checkpoint / "metadata.json"),
            },
            "payload.bin": {"size": payload.stat().st_size, "sha256": _sha(payload)},
        },
    }
    (checkpoint / "MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True))
    committed = {
        "schema": "wm3d_v8_distributed_checkpoint_commit_v2",
        "step": 10,
        "metadata_sha256": _sha(checkpoint / "metadata.json"),
        "manifest_sha256": _sha(checkpoint / "MANIFEST.json"),
        "manifest_content_sha256": _canonical_sha(manifest),
    }
    (checkpoint / "COMMITTED.json").write_text(json.dumps(committed, sort_keys=True))
    evaluation = {
        "schema": "wm3d_v8_unified_offline_eval_v2",
        "all_metrics_finite": True,
        "checkpoint_step": 10,
        "checkpoint_committed_sha256": _sha(checkpoint / "COMMITTED.json"),
        "metrics": {"total": 3.0},
        "coverage": {
            "native_supervised_elements": 12.0,
            "inactive_coarse_supervised_dimensions": 0.0,
        },
        "expected_coverage_lanes": ["native_supervised_elements"],
    }
    eval_path = run / "eval.json"
    eval_path.write_text(json.dumps(evaluation))

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/tools/report_5b_run.py"),
            "--run-root",
            str(run),
            "--expected-step",
            "10",
            "--checkpoint",
            str(checkpoint),
            "--eval",
            str(eval_path),
            "--require-complete",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    # The strict complete report also requires the data/runtime closure.
    assert result.returncode == 1
    assert "checkpoint: PASS" in result.stdout
    assert "eval: PASS" in result.stdout
    assert "WM3D 5B pipeline: INCOMPLETE" in result.stdout


def test_5b_report_status_marks_missing_stages_pending(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/tools/report_5b_run.py"),
            "--run-root",
            str(tmp_path / "run"),
            "--data-profile",
            str(tmp_path / "missing.yaml"),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "WM3D 5B pipeline: INCOMPLETE" in result.stdout
    assert "pending:" in result.stdout


def test_5b_report_streams_large_jsonl_summary(tmp_path: Path) -> None:
    from scripts.tools.report_5b_run import _jsonl_summary

    path = tmp_path / "windows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(10_000):
            handle.write(
                json.dumps(
                    {
                        "sample_id": str(index),
                        "source": "alpha" if index % 2 else "beta",
                        "split": "train" if index % 10 else "val",
                    }
                )
                + "\n"
            )
    count, splits = _jsonl_summary(path)
    assert count == 10_000
    assert sum(splits.values()) == count
    assert splits["alpha:train"] == 5_000
    assert splits["beta:val"] == 1_000


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    import hashlib

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()
