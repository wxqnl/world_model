from __future__ import annotations

import json

import pytest

from wm3d_v3.eval.formal_world_model_benchmark import (
    build_benchmark_card,
    build_markdown_table,
    build_our_rows,
    load_literature_baselines,
    validate_literature_baseline,
)


def _video_report() -> dict:
    return {
        "mode": {
            "metric_protocol": "future_rgb_video_quality",
            "fvd_protocol": "i3d_torchscript_kinetics400_features_frechet_distance",
            "fvd_context_frames": 0,
            "split": "val",
        },
        "checkpoint": {"path": "results/stage2/ckpt/best.pt", "step": 14547},
        "subset": {"selected_total_windows": 120},
        "counts": {"ALL": 120},
        "metrics": {
            "ALL": {
                "model_psnr": 24.30,
                "model_ssim": 0.8355,
                "model_lpips": 0.1544,
                "model_rgb_l1": 0.051,
                "model_motion_rgb_l1": 0.122,
                "last_frame_psnr": 20.0,
                "last_frame_ssim": 0.7001,
                "last_frame_lpips": 0.2301,
            }
        },
        "fvd": {"ALL": {"model_fvd": 674.1, "last_frame_fvd": 900.0, "num_videos": 120}},
    }


def _native_report() -> dict:
    return {
        "counts": {"ALL": 120},
        "core_contribution": {
            "evidence": {
                "motion_region_depth_l1": 0.12,
                "motion_region_depth_change_l1": 0.08,
                "depth_change_l1": 0.07,
                "depth_change_cos": 0.61,
                "mean_depth_win_rate_vs_counterfactual": 0.71,
                "mean_motion_region_depth_win_rate_vs_counterfactual": 0.69,
                "mean_token_win_rate_vs_counterfactual": 0.66,
            }
        },
    }


def test_build_our_rows_scales_worldvla_style_metrics():
    rows = build_our_rows(_video_report(), _native_report(), model_name="WM3D-1B-stage2")

    assert rows[0]["model"] == "WM3D-1B-stage2"
    assert rows[0]["comparison_type"] == "same_harness"
    assert rows[0]["fvd"] == pytest.approx(674.1)
    assert rows[0]["psnr"] == pytest.approx(24.30)
    assert rows[0]["ssim_raw"] == pytest.approx(0.8355)
    assert rows[0]["ssim_x100"] == pytest.approx(83.55)
    assert rows[0]["lpips_raw"] == pytest.approx(0.1544)
    assert rows[0]["lpips_x100"] == pytest.approx(15.44)
    assert rows[0]["motion_region_depth_l1"] == pytest.approx(0.12)
    assert rows[0]["real_action_depth_win_rate"] == pytest.approx(0.71)


def test_build_our_rows_adds_last_frame_internal_baseline():
    rows = build_our_rows(_video_report(), _native_report(), model_name="WM3D-1B-stage2")
    last = rows[1]

    assert last["model"] == "Last-frame repeat"
    assert last["comparison_type"] == "internal_baseline"
    assert last["fvd"] == pytest.approx(900.0)
    assert last["ssim_x100"] == pytest.approx(70.01)
    assert last["lpips_x100"] == pytest.approx(23.01)
    assert last["native3d_protocol"] == "not_applicable"


def test_literature_baseline_validation_and_loading(tmp_path):
    data = [
        {
            "model": "RynnVLA-002 Action World Model",
            "suite": "LIBERO-Long",
            "comparison_type": "literature_protocol",
            "source_url": "https://github.com/alibaba-damo-academy/RynnVLA-002",
            "protocol": "official_rnnvla_002_readme_world_model_512",
            "metrics": {"fvd": 427.86, "psnr": 19.36, "ssim_x100": 72.19, "lpips_x100": 27.78},
        }
    ]
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps(data))

    rows = load_literature_baselines(path)

    assert rows[0]["model"] == "RynnVLA-002 Action World Model"
    assert rows[0]["comparison_type"] == "literature_protocol"
    assert rows[0]["fvd"] == pytest.approx(427.86)
    assert rows[0]["ssim_raw"] == pytest.approx(0.7219)


def test_literature_baseline_requires_source_url():
    with pytest.raises(ValueError, match="source_url"):
        validate_literature_baseline(
            {
                "model": "Missing Source",
                "suite": "LIBERO-Long",
                "comparison_type": "literature_protocol",
                "protocol": "paper",
                "metrics": {"fvd": 1.0},
            }
        )


def test_benchmark_card_separates_same_harness_from_literature_rows():
    own = build_our_rows(_video_report(), _native_report(), model_name="WM3D-1B-stage2")
    literature = [
        {
            "model": "RynnVLA-002 Action World Model",
            "suite": "LIBERO-Long",
            "comparison_type": "literature_protocol",
            "protocol": "official_rnnvla_002_readme_world_model_512",
            "source_url": "https://github.com/alibaba-damo-academy/RynnVLA-002",
            "fvd": 427.86,
            "psnr": 19.36,
            "ssim_raw": 0.7219,
            "ssim_x100": 72.19,
            "lpips_raw": 0.2778,
            "lpips_x100": 27.78,
        }
    ]

    card = build_benchmark_card(
        model_name="WM3D-1B-stage2",
        cfg="configs/stage2.yaml",
        ckpt="results/stage2/ckpt/best.pt",
        video_quality=_video_report(),
        native3d=_native_report(),
        rows=own + literature,
    )

    assert card["claim_boundary"]["same_harness_comparison"] == "paper-strong"
    assert card["claim_boundary"]["literature_protocol_comparison"] == "context-only"
    assert card["protocol"]["video_quality"]["fvd_protocol"] == "i3d_torchscript_kinetics400_features_frechet_distance"
    assert len(card["leaderboard_rows"]) == 3


def test_markdown_table_contains_comparison_type_and_scaled_metrics():
    rows = build_our_rows(_video_report(), _native_report(), model_name="WM3D-1B-stage2")
    table = build_markdown_table(rows)

    assert "| Model | Suite | Compare | FVD↓ | PSNR↑ | SSIM↑ (x100) | LPIPS↓ (x100) |" in table
    assert "WM3D-1B-stage2" in table
    assert "83.55" in table
    assert "same_harness" in table
