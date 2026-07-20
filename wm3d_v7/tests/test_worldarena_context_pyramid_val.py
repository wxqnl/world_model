from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.worldarena_context_pyramid_val import (
    ProtocolError,
    RenderConfig,
    blend_context_residual,
    locked_grid,
    render_baseline,
    render_context_pyramid,
    select_locked_panel,
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
