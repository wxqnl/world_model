"""Regression for demo measurements, independent of model quality claims."""

import numpy as np
from PIL import Image

from scripts.eval_foundation_joint_demo import _save_outputs, _temporal_metrics


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
