from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.data.materialize_oxe_default import (
    _representation_from_profiles,
    build_templates,
)
from wm3d.models.model_factory import validate_model_profile
from wm3d.training.runtime_contract import validate_runtime_profile


ROOT = Path(__file__).resolve().parents[1]


def _oxe_info(action_dim: int = 7, state_dim: int = 7) -> dict:
    return {
        "features": {
            "action": {"dtype": "float32", "shape": [action_dim]},
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "observation.images.camera": {
                "dtype": "video",
                "shape": [256, 256, 3],
            },
        }
    }


def test_1b_streaming_presets_use_dual_path_1b_and_saturating_batch() -> None:
    model = yaml.safe_load(
        (ROOT / "configs/model/native_1b_dual_path.yaml").read_text()
    )
    validate_model_profile(model)
    assert model["expected_parameter_count"] == 1_489_091_608
    assert model["model"]["P"] == 64
    assert model["model"]["appearance_P"] == 256
    assert model["model"]["rgb_decode_indices"] == list(range(8))
    assert model["model"]["rgb_context_enabled"] is True
    assert model["model"]["rgb_context_residual_scale"] == 0.75
    assert model["model"]["rgb_context_motion_blend_gain"] == 0.5
    assert model["model"]["rgb_context_appearance_delta_scale"] == 1.0

    expected = {
        "h100_8_fsdp2_streaming_canary1k.yaml": 1_000,
        "h100_8_fsdp2_streaming_formal100k.yaml": 100_000,
    }
    for filename, steps in expected.items():
        runtime = yaml.safe_load((ROOT / "configs/runtime" / filename).read_text())
        validate_runtime_profile(runtime)
        assert runtime["expected_world_size"] == 8
        assert runtime["resources"]["gpu_name_substring"] == "H100"
        assert runtime["train"]["micro_batch_size"] == 8
        assert runtime["train"]["validation_micro_batch_size"] == 2
        assert runtime["train"]["gradient_accumulation"] == 1
        assert runtime["train"]["global_batch_size"] == 64
        assert runtime["train"]["total_steps"] == steps


def test_1b_site_init_and_plan_are_scale_specific(tmp_path: Path) -> None:
    site = tmp_path / "site.env"
    init = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/cluster/wm3d_1b.sh"),
            "init",
            "formal100k",
            str(site),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert init.returncode == 0, init.stderr
    payload = site.read_text()
    assert "WM3D_1B_PRESET=formal100k" in payload
    assert "MODEL_PROFILE=configs/model/native_1b_dual_path.yaml" in payload
    assert (
        "ENCODER_CONTRACT=configs/encoder/vggt_native_p64_appearance_p256.yaml"
        in payload
    )
    assert "OBJECTIVE_PROFILE=configs/objective/stage0_native_dual_path.yaml" in payload
    assert "WM3D_DATA_MODE=direct_raw" in payload
    assert "MINIMUM_RAW_FILESYSTEM_BYTES=0" in payload
    assert "DIRECT_PREFETCH_WINDOWS=16" in payload
    assert "DIRECT_DECODE_WORKERS=1" in payload
    assert "DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK=1" in payload
    assert "STREAMING_LRU_ROOT=" not in payload
    site.write_text(
        payload.replace(
            "PYTHON_BIN=${ENV_DIR}/bin/python", f"PYTHON_BIN={sys.executable}"
        ).replace(
            "HF_TOKEN_FILE=/data/secrets/huggingface_token",
            f"HF_TOKEN_FILE={tmp_path / 'token'}",
        )
    )
    (tmp_path / "token").write_text("test")
    plan = subprocess.run(
        ["bash", str(ROOT / "scripts/cluster/wm3d_1b.sh"), "plan", str(site)],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert plan.returncode == 0, plan.stderr
    assert "WM3D 1B site plan" in plan.stdout
    assert "preset:     formal100k" in plan.stdout
    assert "h100_8_fsdp2_streaming_formal100k.yaml" in plan.stdout
    assert "final step: 100000" in plan.stdout
    assert "runs/1b_formal100k" in plan.stdout


def test_1b_rejects_5b_only_preset(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/cluster/wm3d_1b.sh"),
            "init",
            "formal600k",
            str(tmp_path / "site.env"),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "未知 1B preset" in result.stderr


def test_1b_oxe_mix_is_60_sources_without_agibot_or_pca() -> None:
    base_data = yaml.safe_load(
        (ROOT / "configs/data/public_robot_6106h.template.yaml").read_text()
    )
    base_source = yaml.safe_load(
        (ROOT / "configs/sources/public_sources.template.yaml").read_text()
    )
    representation = _representation_from_profiles(
        base=base_data["cache_representation"],
        model_path=ROOT / "configs/model/native_1b_dual_path.yaml",
        encoder_path=ROOT / "configs/encoder/vggt_native_p64_appearance_p256.yaml",
    )
    repos = ("lerobot/droid_1.0.1",) + tuple(
        f"lerobot/fixture_{index:02d}" for index in range(55)
    )
    generated_sources, generated_data, adapters = build_templates(
        base_source=base_source,
        base_data=base_data,
        repo_ids=repos,
        metadata_by_repo={repo: _oxe_info() for repo in repos[1:]},
        include_agibot_2026=False,
        representation_override=representation,
        profile_name="public_robot_1b_oxe",
        profile_role="default_1b_public_profile",
    )
    source_names = {row["name"] for row in generated_sources["sources"]}
    rows = {row["name"]: row for row in generated_data["sources"]}
    assert len(source_names) == 60
    assert len(rows) == 60
    assert len(adapters) == 55
    assert "agibot_world_2026" not in source_names
    assert "agibot_beta" not in source_names
    assert not any(name.startswith("agibot_2026") for name in rows)
    assert {"droid", "bridge", "atomic", "composite", "mg"} <= rows.keys()
    assert sum(int(row["weight"]) for row in rows.values()) == 95
    assert generated_data["name"] == "public_robot_1b_oxe"
    assert generated_data["notes"]["agibot_world_2026_enabled"] is False
    assert generated_data["notes"]["agibot_beta_enabled"] is False
    assert generated_data["cache_representation"]["spatial_tokens"] == 64
    assert generated_data["cache_representation"]["appearance_token_grid"] == 16
    assert generated_data["cache_representation"]["appearance_feature_layer"] == 4
    assert generated_data["cache_representation"]["token_dim"] == 2048
    assert generated_data["cache_representation"]["rgb_size"] == 256
    assert "pca" not in json.dumps(generated_data["cache_representation"]).lower()


def test_1b_docs_make_raw_storage_and_canary_explicit() -> None:
    document = (ROOT / "docs/WM3D_1B_STREAMING.md").read_text()
    assert "60 个 source" in document
    assert "15–20TB" in document
    assert "2048→384 PCA" in document
    assert "./run_wm3d.sh 1b streaming-prepare" in document
    assert "./run_wm3d.sh 1b train" in document
    assert "./run_wm3d.sh 1b eval" in document
    assert "./run_wm3d.sh 1b verify" in document
