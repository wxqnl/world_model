import numpy as np
import pytest
import torch

from wm3d_v3.eval.video_quality_eval import (
    batch_sample_rows,
    ensure_min_video_frames,
    fvd_protocol_name,
    frechet_distance,
    motion_region_rgb_l1,
    psnr_video,
    ssim_video,
)


def test_psnr_and_ssim_prefer_identical_video_over_shifted_video():
    target = torch.full((2, 3, 3, 16, 16), 0.5)
    identical = target.clone()
    shifted = torch.clamp(target + 0.2, 0.0, 1.0)

    assert torch.all(psnr_video(identical, target) > 70.0)
    assert torch.all(psnr_video(shifted, target) < 15.0)
    assert torch.allclose(ssim_video(identical, target), torch.ones(2), atol=1e-4)
    assert torch.all(ssim_video(shifted, target) < 0.95)


def test_frechet_distance_is_zero_for_identical_features_and_positive_for_shift():
    features = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    assert frechet_distance(features, features) == pytest.approx(0.0, abs=1e-8)
    assert frechet_distance(features, features + 1.0) > 1.0


def test_motion_region_rgb_l1_only_scores_pixels_that_changed_from_context():
    context = torch.zeros((1, 3, 4, 4))
    target = torch.zeros((1, 2, 3, 4, 4))
    prediction = target.clone()
    target[:, :, :, :2, :] = 1.0
    prediction[:, :, :, :2, :] = 0.75
    prediction[:, :, :, 2:, :] = 1.0

    score = motion_region_rgb_l1(prediction, target, context, threshold=0.03)

    assert score.shape == (1,)
    assert score.item() == pytest.approx(0.25)


def test_fvd_protocol_name_distinguishes_i3d_from_r3d_proxy():
    assert fvd_protocol_name("r3d18") == "r3d18_kinetics400_features_frechet_distance"
    assert fvd_protocol_name("i3d_torchscript") == "i3d_torchscript_kinetics400_features_frechet_distance"


def test_batch_sample_rows_records_reproducible_window_identity():
    batch = {
        "dataset": ["bridge", "kuka"],
        "clip_id": ["clip-a", "clip-b"],
        "start": torch.tensor([12, 20]),
    }

    rows = batch_sample_rows(batch, batch_index=3, global_offset=10)

    assert rows == [
        {"sample_index": 10, "batch_index": 3, "batch_sample_index": 0, "dataset": "bridge", "clip_id": "clip-a", "start": 12},
        {"sample_index": 11, "batch_index": 3, "batch_sample_index": 1, "dataset": "kuka", "clip_id": "clip-b", "start": 20},
    ]


def test_ensure_min_video_frames_repeats_last_frame():
    video = torch.arange(2 * 3 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 3, 4, 4)

    padded = ensure_min_video_frames(video, min_frames=5)

    assert padded.shape == (2, 5, 3, 4, 4)
    assert torch.equal(padded[:, :3], video)
    assert torch.equal(padded[:, 3], video[:, -1])
    assert torch.equal(padded[:, 4], video[:, -1])
