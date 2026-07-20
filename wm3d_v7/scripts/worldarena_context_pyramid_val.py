#!/usr/bin/env python3
"""Protocol-locked WorldArena validation renderer and scoring primitives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


class ProtocolError(RuntimeError):
    """Raised when the validation-only diagnostic contract is violated."""


@dataclass(frozen=True, order=True)
class RenderConfig:
    alpha: float
    low: float
    high: float
    sigma: float = 1.0
    native_size: int = 64

    def __post_init__(self) -> None:
        if self.alpha not in (0.5, 0.75, 1.0):
            raise ProtocolError("alpha is outside the locked grid")
        if (self.low, self.high) not in ((0.02, 0.08), (0.04, 0.12)):
            raise ProtocolError("motion ramp is outside the locked grid")
        if self.sigma != 1.0 or self.native_size != 64:
            raise ProtocolError("sigma/native size must remain 1.0/64")


def locked_grid() -> tuple[RenderConfig, ...]:
    """Return the exact six globally allowed renderer configurations."""
    return tuple(
        RenderConfig(alpha=alpha, low=low, high=high)
        for alpha in (0.5, 0.75, 1.0)
        for low, high in ((0.02, 0.08), (0.04, 0.12))
    )


def select_locked_panel(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select the deterministic five-record validation panel before any decoding."""
    tasks = sorted({str(row.get("task", "")) for row in rows})
    if len(tasks) != 50:
        raise ProtocolError(f"expected 50 tasks, got {len(tasks)}")
    identities: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        task = str(row.get("task", ""))
        episode = int(row.get("episode", -1))
        key = (task, episode)
        if key in identities:
            raise ProtocolError(f"duplicate task/episode identity: {key}")
        identities[key] = dict(row)

    chosen = [
        (tasks[index], episode)
        for index, episode in zip((0, 12, 24, 36, 49), (36, 37, 38, 39, 36), strict=True)
    ]
    if any(episode not in (36, 37, 38, 39) for _, episode in chosen):
        raise ProtocolError("selected panel escaped episodes 36-39")
    missing = [identity for identity in chosen if identity not in identities]
    if missing:
        raise ProtocolError(f"missing selected identities: {missing}")
    panel = [identities[identity] for identity in chosen]
    ids = [str(row.get("id", "")) for row in panel]
    if len(panel) != 5 or len(set(ids)) != 5 or any(not value for value in ids):
        raise ProtocolError("panel must contain five unique non-empty ids")
    return panel


def _float_rgb(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ProtocolError(f"{name} must be finite")
    if float(array.min(initial=0.0)) < 0.0 or float(array.max(initial=0.0)) > 1.0:
        raise ProtocolError(f"{name} must be in RGB [0,1]")
    return array


def _native_thwc(native_rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(native_rgb)
    if value.ndim != 4:
        raise ProtocolError("native RGB must be rank four")
    if value.shape[1] == 3:
        value = np.moveaxis(value, 1, -1)
    if value.shape[-1] != 3:
        raise ProtocolError("native RGB must be T,H,W,3 or T,3,H,W")
    return _float_rgb(value, name="native RGB")


def _initial_hwc(initial_rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(initial_rgb)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ProtocolError("initial RGB must be H,W,3")
    return _float_rgb(value, name="initial RGB")


def _validate_output_size(output_size: tuple[int, int]) -> tuple[int, int]:
    if (
        len(output_size) != 2
        or not all(isinstance(value, int) for value in output_size)
        or output_size[0] <= 0
        or output_size[1] <= 0
    ):
        raise ProtocolError("output_size must be positive integer (width,height)")
    return output_size


def render_baseline(
    initial_rgb: np.ndarray,
    native_rgb: np.ndarray,
    *,
    output_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Match the existing renderer's per-prediction INTER_LINEAR resize."""
    _initial_hwc(initial_rgb)
    native = _native_thwc(native_rgb)
    size = _validate_output_size(output_size)
    output = np.stack(
        [cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) for frame in native]
    ).astype(np.float32)
    if not np.isfinite(output).all():
        raise ProtocolError("baseline renderer produced NaN/Inf")
    return np.clip(output, 0.0, 1.0)


def blend_context_residual(
    low_prediction: np.ndarray,
    residual: np.ndarray,
    motion_mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Blend one fixed context residual band outside the soft motion mask."""
    low = np.asarray(low_prediction, dtype=np.float32)
    high = np.asarray(residual, dtype=np.float32)
    mask = np.asarray(motion_mask, dtype=np.float32)
    if low.ndim != 4 or low.shape[-1] != 3:
        raise ProtocolError("low prediction must be T,H,W,3")
    if high.shape != low.shape[1:]:
        raise ProtocolError("context residual must be H,W,3")
    if mask.shape != (*low.shape[:3], 1):
        raise ProtocolError("motion mask must be T,H,W,1")
    if not np.isfinite(low).all() or not np.isfinite(high).all() or not np.isfinite(mask).all():
        raise ProtocolError("blend inputs must be finite")
    if float(mask.min(initial=0.0)) < 0.0 or float(mask.max(initial=0.0)) > 1.0:
        raise ProtocolError("motion mask must be in [0,1]")
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ProtocolError("alpha must be finite and in [0,1]")
    return np.clip(low + float(alpha) * (1.0 - mask) * high[None], 0.0, 1.0)


def render_context_pyramid(
    initial_rgb: np.ndarray,
    native_rgb: np.ndarray,
    config: RenderConfig,
    *,
    output_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Render native predictions with a validation-locked context Laplacian band."""
    if config not in locked_grid():
        raise ProtocolError("renderer config is not in the locked grid")
    initial = _initial_hwc(initial_rgb)
    native = _native_thwc(native_rgb)
    size = _validate_output_size(output_size)
    native_size = (config.native_size, config.native_size)
    native64 = np.stack(
        [cv2.resize(frame, native_size, interpolation=cv2.INTER_AREA) for frame in native]
    ).astype(np.float32)
    context64 = cv2.resize(initial, native_size, interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    context_high = cv2.resize(initial, size, interpolation=cv2.INTER_CUBIC).astype(
        np.float32
    )
    context_low_up = cv2.resize(
        context64, size, interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)
    residual = context_high - context_low_up

    low_predictions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame in native64:
        distance = np.mean(np.abs(frame - context64), axis=-1)
        mask = np.clip(
            (distance - config.low) / (config.high - config.low), 0.0, 1.0
        )
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=config.sigma, sigmaY=config.sigma)
        masks.append(
            cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)[..., None].astype(
                np.float32
            )
        )
        low_predictions.append(
            cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        )
    output = blend_context_residual(
        np.stack(low_predictions),
        residual,
        np.stack(masks),
        alpha=config.alpha,
    )
    if not np.isfinite(output).all():
        raise ProtocolError("context-pyramid renderer produced NaN/Inf")
    return output.astype(np.float32)
