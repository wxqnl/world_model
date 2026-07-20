from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.worldarena_context_pyramid_val import (
    ProtocolError,
    RenderConfig,
    aligned_video_psnr,
    blend_context_residual,
    locked_grid,
    parse_variant_name,
    render_baseline,
    render_context_pyramid,
    select_candidate,
    select_locked_panel,
    variant_name,
)


def _fake_rows() -> list[dict[str, object]]:
    return [
        {"id": f"{task}:episode{episode}", "task": task, "episode": episode}
        for task in reversed([f"task_{index:02d}" for index in range(50)])
        for episode in range(50)
    ]


def test_locked_panel_is_deterministic_and_val_only() -> None:
    panel = select_locked_panel(_fake_rows())

    assert [(row["task"], row["episode"]) for row in panel] == [
        ("task_00", 36),
        ("task_12", 37),
        ("task_24", 38),
        ("task_36", 39),
        ("task_49", 36),
    ]


def test_locked_panel_rejects_incomplete_task_set() -> None:
    with pytest.raises(ProtocolError, match="expected 50 tasks"):
        select_locked_panel(_fake_rows()[:-50])


def test_locked_grid_has_exactly_six_global_configs() -> None:
    assert [
        (config.alpha, config.low, config.high, config.sigma, config.native_size)
        for config in locked_grid()
    ] == [
        (0.50, 0.02, 0.08, 1.0, 64),
        (0.50, 0.04, 0.12, 1.0, 64),
        (0.75, 0.02, 0.08, 1.0, 64),
        (0.75, 0.04, 0.12, 1.0, 64),
        (1.00, 0.02, 0.08, 1.0, 64),
        (1.00, 0.04, 0.12, 1.0, 64),
    ]


def test_render_config_rejects_values_outside_locked_grid() -> None:
    with pytest.raises(ProtocolError, match="outside the locked grid"):
        RenderConfig(alpha=0.25, low=0.02, high=0.08)
    with pytest.raises(ProtocolError, match="outside the locked grid"):
        RenderConfig(alpha=0.50, low=0.01, high=0.08)


def _checkerboard(height: int, width: int) -> np.ndarray:
    values = np.indices((height, width)).sum(axis=0) % 2
    return np.repeat(values[..., None], 3, axis=-1).astype(np.float32)


def test_zero_motion_injects_context_residual_and_full_motion_injects_none() -> None:
    initial = _checkerboard(96, 128)
    context64 = cv2.resize(initial, (64, 64), interpolation=cv2.INTER_AREA)
    static = context64[None]
    moved = np.zeros((1, 64, 64, 3), dtype=np.float32)
    config = RenderConfig(alpha=1.0, low=0.02, high=0.08)

    static_output = render_context_pyramid(
        initial, static, config, output_size=(128, 96)
    )
    moved_output = render_context_pyramid(
        initial, moved, config, output_size=(128, 96)
    )
    context_high = cv2.resize(initial, (128, 96), interpolation=cv2.INTER_CUBIC)
    moved_low = cv2.resize(moved[0], (128, 96), interpolation=cv2.INTER_CUBIC)

    assert np.allclose(static_output[0], context_high, atol=1e-5)
    assert np.allclose(moved_output[0], moved_low, atol=1e-5)


def test_alpha_zero_blend_is_identity() -> None:
    rng = np.random.default_rng(3)
    low_prediction = rng.random((2, 16, 20, 3), dtype=np.float32)
    residual = rng.normal(size=(16, 20, 3)).astype(np.float32)
    motion_mask = rng.random((2, 16, 20, 1), dtype=np.float32)

    output = blend_context_residual(
        low_prediction, residual, motion_mask, alpha=0.0
    )

    assert np.array_equal(output, low_prediction)


def test_renderer_is_finite_bounded_and_does_not_mutate_inputs() -> None:
    rng = np.random.default_rng(7)
    initial = rng.random((91, 117, 3), dtype=np.float32)
    native = rng.random((3, 3, 73, 85), dtype=np.float32)
    initial_copy = initial.copy()
    native_copy = native.copy()

    output = render_context_pyramid(
        initial, native, RenderConfig(0.75, 0.04, 0.12)
    )

    assert output.shape == (3, 480, 640, 3)
    assert np.isfinite(output).all()
    assert output.min() >= 0.0 and output.max() <= 1.0
    assert np.array_equal(initial, initial_copy)
    assert np.array_equal(native, native_copy)


def test_baseline_matches_current_linear_resize_contract() -> None:
    rng = np.random.default_rng(11)
    initial = rng.random((64, 64, 3), dtype=np.float32)
    native = rng.random((2, 3, 64, 64), dtype=np.float32)

    output = render_baseline(initial, native, output_size=(80, 60))
    expected = np.stack(
        [
            cv2.resize(
                np.moveaxis(frame, 0, -1),
                (80, 60),
                interpolation=cv2.INTER_LINEAR,
            )
            for frame in native
        ]
    )

    assert np.allclose(output, expected, atol=1e-7)


def _write_video(path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    assert writer.isOpened()
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def test_psnr_rejects_frame_count_mismatch(tmp_path) -> None:
    pred = tmp_path / "pred.mp4"
    gt = tmp_path / "gt.mp4"
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    _write_video(pred, [black] * 3)
    _write_video(gt, [black] * 4)

    with pytest.raises(ProtocolError, match="frame count mismatch"):
        aligned_video_psnr(pred, gt)


def test_psnr_reports_finite_aligned_mean(tmp_path) -> None:
    pred = tmp_path / "pred.mp4"
    gt = tmp_path / "gt.mp4"
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    white = np.full((32, 24, 3), 255, dtype=np.uint8)
    _write_video(pred, [black, black])
    _write_video(gt, [white, white])

    result = aligned_video_psnr(pred, gt)

    assert result["frames"] == 2
    assert np.isfinite(result["mean"])
    assert len(result["per_frame"]) == 2


def _baseline_metrics() -> dict[str, float | int]:
    return {
        "psnr": 20.0,
        "image_quality": 0.5,
        "jepa_similarity": 0.8,
        "dynamic_degree": 0.4,
        "motion_smoothness": 0.6,
        "coverage": 5,
    }


def test_variant_name_round_trip_uses_locked_config() -> None:
    for config in locked_grid():
        assert parse_variant_name(variant_name(config)) == config


def test_select_candidate_uses_aggregate_gates_and_tie_breaks() -> None:
    baseline = _baseline_metrics()
    candidates = {
        "a050_l002_h008": {**baseline, "psnr": 20.30},
        "a075_l002_h008": {**baseline, "psnr": 20.31},
        "a100_l002_h008": {
            **baseline,
            "psnr": 21.00,
            "jepa_similarity": 0.70,
        },
    }

    result = select_candidate(baseline, candidates)

    assert result["decision"] == "GO"
    assert result["selected"] == "a050_l002_h008"
    assert result["checks"]["a100_l002_h008"]["jepa_similarity"] is False


def test_select_candidate_reports_no_go_when_quality_falls() -> None:
    baseline = _baseline_metrics()
    result = select_candidate(
        baseline,
        {
            "a050_l002_h008": {
                **baseline,
                "psnr": 20.5,
                "image_quality": 0.49,
            }
        },
    )

    assert result["decision"] == "NO-GO"
    assert result["selected"] is None


def test_select_candidate_requires_exact_five_video_coverage() -> None:
    baseline = _baseline_metrics()
    with pytest.raises(ProtocolError, match="coverage must equal five"):
        select_candidate({**baseline, "coverage": 4}, {})
