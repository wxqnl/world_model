"""Regression for demo measurements, independent of model quality claims."""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from scripts.eval_foundation_joint_demo import (
    _action_metrics, _observation_inputs, _save_outputs, _temporal_metrics, parse_args,
    _evaluation_output_directory,
)


def test_demo_defaults_to_joint_and_strips_future_labels(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "foundation-demo", "--runtime", "runtime.yaml", "--checkpoint", "step_00000020",
        "--output", "unused",
    ])
    assert parse_args().mode == "joint"
    batch = {
        "video": torch.randn(1, 3, 9, 12, 8),
        "action": torch.full((1, 4, 6), float("nan")),
        "action_time_s": torch.arange(4)[None, :, None].expand(1, -1, 6),
        "action_element_mask": torch.ones(1, 4, 6, dtype=torch.bool),
        "vggt_current_tokens": torch.randn(1, 3, 4, 8),
        "vggt_future_tokens": torch.tensor(float("nan")),
        "vggt_rollout_target_tokens": torch.tensor(float("nan")),
        "future_3d_images": torch.tensor(float("nan")),
        "native_3d_trajectory_target": torch.tensor(float("nan")),
        "native_3d_trajectory_mask": torch.tensor(False),
    }
    inputs = _observation_inputs(batch)
    assert "action" not in inputs
    assert inputs["video"].shape[2] == 1
    assert not any("future" in key or "target" in key for key in inputs)
    assert inputs["action_time_s"] is batch["action_time_s"]


def test_action_metrics_use_masks_and_report_zero_and_physical_errors():
    batch = {
        "action": torch.tensor([[[1., 2., 0.], [3., 4., 0.]]]),
        "action_element_mask": torch.tensor([[[True, True, False], [True, True, False]]]),
        "action_normalization_scale": torch.tensor([[[.1, .2, 1.]]]),
    }
    metrics = _action_metrics("policy", torch.zeros(1, 2, 3), batch)
    assert metrics["policy_action_normalized_mae"] == 2.5
    assert metrics["policy_action_zero_baseline_mae"] == 2.5
    assert metrics["policy_action_query_mae"] == [1.5, 3.5]
    assert np.allclose(metrics["policy_action_physical_mae_by_dim"][:2], [.2, .6])
    assert metrics["policy_action_physical_mae_by_dim"][2] is None


def test_demo_motion_excludes_observation_reconstruction_jump():
    first = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    future = Image.fromarray(np.full((4, 4, 3), 100, dtype=np.uint8))
    video = [first] + [future] * 8
    metrics = _temporal_metrics(video, video, video)
    assert metrics["target_frame_delta_rms"] == 0
    assert metrics["factual_frame_delta_rms"] == 0
    assert metrics["factual_is_temporally_identical"]
    assert metrics["factual_zero_response_rms"] == 0


def test_demo_detects_uniform_color_even_when_rgb_channels_differ():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:] = (200, 50, 100)
    video = [Image.fromarray(pixels)] * 9
    metrics = _temporal_metrics(video, video, None)
    assert metrics["factual_is_single_color"]


def test_demo_export_keeps_lossless_frames(tmp_path):
    frames = [
        Image.fromarray(np.full((12, 8, 3), step * 17, dtype=np.uint8))
        for step in range(9)
    ]
    output = tmp_path / "demo"
    _save_outputs(output, target=frames, factual=frames, zero=frames, duration_ms=100)
    for index, expected in enumerate(frames):
        path = output / "frames" / "factual_action" / f"frame_{index:03d}.png"
        with Image.open(path) as actual:
            assert np.array_equal(np.asarray(actual), np.asarray(expected))
    assert (output / "comparison.gif").is_file()


def test_demo_motion_ignores_missing_views():
    target = [Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))] * 9
    prediction = []
    for step in range(9):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[2:] = step * 20
        prediction.append(Image.fromarray(image))
    valid = np.zeros((4, 4), dtype=bool)
    valid[:2] = True
    metrics = _temporal_metrics(target, prediction, target, valid)
    assert metrics["factual_frame_delta_rms"] == 0
    assert metrics["factual_target_l1"] == 0
    assert metrics["factual_zero_response_rms"] == 0


def test_native_demo_preserves_checkpoint_topology_and_rank_output_paths():
    root = Path("demo")
    assert _evaluation_output_directory(root, native_sharded=True,
        expected_world_size=8, world_size=8, rank=3) == root / "rank_00003"
    with pytest.raises(RuntimeError, match="training topology"):
        _evaluation_output_directory(root, native_sharded=True,
            expected_world_size=8, world_size=1, rank=0)
    assert _evaluation_output_directory(root, native_sharded=False,
        expected_world_size=8, world_size=1, rank=0) == root
