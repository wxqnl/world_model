"""Causal dual-view cache contract for WM3D-V8 Stage0.

The observed context and the future supervision are produced by independent
VGGT forwards.  Only the T-frame forward may become a model input.  The
T+K-frame forward is sliced to its final K frames and is supervision-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


CAUSAL_DUAL_VIEW_SCHEMA = "wm3d_v8_stage0_causal_dual_view_v1"
CAUSAL_DUAL_VIEW_REPRESENTATION = (
    "wm3d_v8_vggt_observed_context_target_split_v1"
)
GEOMETRY_COORDINATE_FRAME = "first_observed_camera"
TARGET_USAGE = "supervision_only"


def causal_dual_view_metadata(*, T: int, k: int) -> dict[str, np.ndarray]:
    """Return the exact scalar identity stored in every causal cache archive."""

    if T <= 0 or k <= 0:
        raise ValueError("T and k must be positive")
    return {
        "schema": np.asarray(CAUSAL_DUAL_VIEW_SCHEMA),
        "representation": np.asarray(CAUSAL_DUAL_VIEW_REPRESENTATION),
        "context_future_leakage": np.asarray(False),
        "target_usage": np.asarray(TARGET_USAGE),
        "geometry_coordinate_frame": np.asarray(GEOMETRY_COORDINATE_FRAME),
        "context_frames": np.asarray(T, dtype=np.int64),
        "future_frames": np.asarray(k, dtype=np.int64),
        "context_forward_frames": np.asarray(T, dtype=np.int64),
        "target_forward_frames": np.asarray(T + k, dtype=np.int64),
        "target_observed_outputs_discarded": np.asarray(T, dtype=np.int64),
    }


def _one_batch(output: Mapping[str, Any], key: str) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"VGGT output missing tensor {key}")
    if value.ndim < 2 or value.shape[0] != 1:
        raise ValueError(f"VGGT output {key} must have batch size one: {tuple(value.shape)}")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"VGGT output {key} contains non-finite values")
    return value[0]


def _quantize(
    tokens: torch.Tensor,
    *,
    codec: Any,
) -> tuple[np.ndarray, np.ndarray]:
    latent = codec.encode(tokens.to(codec.mean.device) if hasattr(codec, "mean") else tokens)
    codes, scale = codec.quantize(latent)
    if codes.dtype != torch.int8:
        raise ValueError(f"codec codes must be int8, got {codes.dtype}")
    if not torch.isfinite(scale.float()).all() or not bool((scale > 0).all()):
        raise ValueError("codec scale must be finite and positive")
    return codes.cpu().numpy(), scale.cpu().numpy()


def _future_geometry(
    output: Mapping[str, Any],
    *,
    T: int,
    k: int,
    patch_grid: int,
) -> dict[str, np.ndarray]:
    depth = _one_batch(output, "depth")[T : T + k].float()
    depth_conf = _one_batch(output, "depth_conf")[T : T + k].float()
    points = _one_batch(output, "world_points")[T : T + k].float()
    point_conf = _one_batch(output, "world_points_conf")[T : T + k].float()
    pose = _one_batch(output, "pose_enc")[T : T + k]
    if min(depth.shape[0], depth_conf.shape[0], points.shape[0], point_conf.shape[0], pose.shape[0]) != k:
        raise ValueError("target VGGT forward is shorter than K")
    if depth.ndim != 3 or depth_conf.ndim != 3:
        raise ValueError("VGGT depth and depth confidence must be [K,H,W]")
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("VGGT world points must be [K,H,W,3]")
    if point_conf.ndim != 3:
        raise ValueError("VGGT point confidence must be [K,H,W]")

    depth_patch = F.adaptive_avg_pool2d(
        depth[:, None], (patch_grid, patch_grid)
    ).squeeze(1)
    depth_conf_patch = F.adaptive_avg_pool2d(
        depth_conf[:, None], (patch_grid, patch_grid)
    ).squeeze(1)
    point_patch = F.adaptive_avg_pool2d(
        points.permute(0, 3, 1, 2), (patch_grid, patch_grid)
    ).permute(0, 2, 3, 1)
    point_conf_patch = F.adaptive_avg_pool2d(
        point_conf[:, None], (patch_grid, patch_grid)
    ).squeeze(1)
    return {
        "future_depth_patch": depth_patch.to(torch.float16).cpu().numpy(),
        "future_depth_conf_patch": depth_conf_patch.to(torch.float16).cpu().numpy(),
        "future_point_patch": point_patch.to(torch.float16).cpu().numpy(),
        "future_point_conf_patch": point_conf_patch.to(torch.float16).cpu().numpy(),
        "future_pose_enc": pose.to(torch.float16).cpu().numpy(),
    }


@torch.inference_mode()
def encode_causal_dual_view(
    images: torch.Tensor,
    *,
    encoder: torch.nn.Module,
    codec: Any,
    T: int,
    k: int,
    patch_grid: int = 8,
) -> dict[str, np.ndarray]:
    """Encode one T+K window without allowing future-conditioned context.

    VGGT uses the first frame of each independent forward as its camera gauge.
    Because both forwards begin with the same first observed frame, their
    geometry is expressed in the required first-observed-camera gauge.
    """

    if T <= 0 or k <= 0 or patch_grid <= 0:
        raise ValueError("T, k, and patch_grid must be positive")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a [T+K,3,H,W] tensor")
    if images.shape[0] != T + k or images.shape[1] != 3:
        raise ValueError(
            f"expected images [{T + k},3,H,W], got {tuple(images.shape)}"
        )
    if not torch.isfinite(images.float()).all():
        raise ValueError("images contain non-finite values")

    flag_names = ("return_depth", "return_depth_conf", "return_geom_extra")
    previous = {name: getattr(encoder, name) for name in flag_names if hasattr(encoder, name)}
    try:
        for name in flag_names:
            if hasattr(encoder, name):
                setattr(encoder, name, True)
        context_output = encoder(images[:T].unsqueeze(0))
        target_output = encoder(images.unsqueeze(0))
    finally:
        for name, value in previous.items():
            setattr(encoder, name, value)

    context_tokens = _one_batch(context_output, "pooled")
    target_tokens = _one_batch(target_output, "pooled")
    if context_tokens.shape[0] != T:
        raise ValueError(
            f"context VGGT forward returned {context_tokens.shape[0]} frames, expected {T}"
        )
    future_tokens = target_tokens[T : T + k]
    if future_tokens.shape[0] != k:
        raise ValueError(
            f"target VGGT forward returned {target_tokens.shape[0]} frames, expected {T + k}"
        )
    if context_tokens.shape[1:] != future_tokens.shape[1:]:
        raise ValueError("context and target VGGT token shapes differ")

    context_codes, context_scale = _quantize(context_tokens, codec=codec)
    future_codes, future_scale = _quantize(future_tokens, codec=codec)
    result = {
        "context_codes": context_codes,
        "context_scale": context_scale,
        "future_codes": future_codes,
        "future_scale": future_scale,
    }
    result.update(
        _future_geometry(
            target_output,
            T=T,
            k=k,
            patch_grid=patch_grid,
        )
    )
    return result


def _scalar(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise ValueError(f"causal dual-view archive missing {name}")
    value = np.asarray(payload[name])
    if value.shape != ():
        raise ValueError(f"causal dual-view scalar {name} has shape {value.shape}")
    return value.item()


def _require_exact_metadata(payload: Mapping[str, Any], *, T: int, k: int) -> None:
    expected = {
        "schema": CAUSAL_DUAL_VIEW_SCHEMA,
        "representation": CAUSAL_DUAL_VIEW_REPRESENTATION,
        "context_future_leakage": False,
        "target_usage": TARGET_USAGE,
        "geometry_coordinate_frame": GEOMETRY_COORDINATE_FRAME,
        "context_frames": T,
        "future_frames": k,
        "context_forward_frames": T,
        "target_forward_frames": T + k,
        "target_observed_outputs_discarded": T,
    }
    for name, wanted in expected.items():
        actual = _scalar(payload, name)
        if actual != wanted:
            if name == "schema":
                raise ValueError(
                    f"unexpected causal dual-view schema: {actual!r}; expected {wanted!r}"
                )
            raise ValueError(
                f"causal dual-view identity mismatch for {name}: {actual!r} != {wanted!r}"
            )


def _array(payload: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"causal dual-view archive missing {name}")
    return np.asarray(payload[name])


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"causal dual-view array {name} is not numeric")
    if not np.isfinite(value.astype(np.float32, copy=False)).all():
        raise ValueError(f"causal dual-view array {name} contains non-finite values")


def validate_causal_dual_view_archive(
    payload: Mapping[str, Any],
    *,
    T: int,
    k: int,
    paired_views: bool,
) -> dict[str, int | bool]:
    """Validate a per-window OXE or per-clip compact causal archive."""

    _require_exact_metadata(payload, T=T, k=k)
    context_codes = _array(payload, "context_codes")
    future_codes = _array(payload, "future_codes")
    context_scale = _array(payload, "context_scale")
    future_scale = _array(payload, "future_scale")
    if context_codes.dtype != np.int8 or future_codes.dtype != np.int8:
        raise ValueError("causal dual-view token codes must be int8")

    compact = context_codes.ndim == 4
    if compact:
        windows, context_frames, token_count, latent_dim = context_codes.shape
        if windows <= 0:
            raise ValueError("compact causal dual-view archive has no windows")
        expected_future_shape = (windows, k, token_count, latent_dim)
        expected_context_scale = (windows, T, 1, 1)
        expected_future_scale = (windows, k, 1, 1)
        starts = _array(payload, "window_starts")
        if starts.shape != (windows,) or not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("window_starts must be an integer [W] array")
        if len(np.unique(starts)) != windows:
            raise ValueError("window_starts contains duplicates")
    elif context_codes.ndim == 3:
        windows = 1
        context_frames, token_count, latent_dim = context_codes.shape
        expected_future_shape = (k, token_count, latent_dim)
        expected_context_scale = (T, 1, 1)
        expected_future_scale = (k, 1, 1)
    else:
        raise ValueError(
            f"context_codes must be [T,P,C] or [W,T,P,C], got {context_codes.shape}"
        )

    if context_frames != T:
        raise ValueError(f"context_codes has {context_frames} frames, expected {T}")
    if token_count <= 0 or latent_dim <= 0:
        raise ValueError("causal dual-view token shape must be non-empty")
    if future_codes.shape != expected_future_shape:
        raise ValueError(
            f"future_codes shape {future_codes.shape} != {expected_future_shape}"
        )
    if context_scale.shape != expected_context_scale:
        raise ValueError(
            f"context_scale shape {context_scale.shape} != {expected_context_scale}"
        )
    if future_scale.shape != expected_future_scale:
        raise ValueError(
            f"future_scale shape {future_scale.shape} != {expected_future_scale}"
        )

    leading = (windows,) if compact else ()
    required_shapes = {
        "future_depth_patch": leading + (k, 8, 8),
        "future_depth_conf_patch": leading + (k, 8, 8),
        "future_point_patch": leading + (k, 8, 8, 3),
        "future_point_conf_patch": leading + (k, 8, 8),
    }
    for name, shape in required_shapes.items():
        value = _array(payload, name)
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        _require_finite(name, value)

    pose = _array(payload, "future_pose_enc")
    if pose.ndim < len(leading) + 2 or pose.shape[: len(leading) + 1] != leading + (k,):
        raise ValueError(
            f"future_pose_enc must begin with {leading + (k,)}, got {pose.shape}"
        )
    _require_finite("future_pose_enc", pose)

    if paired_views:
        wrist_codes = _array(payload, "wrist_context_codes")
        wrist_scale = _array(payload, "wrist_context_scale")
        if wrist_codes.shape != context_codes.shape:
            raise ValueError("wrist_context_codes shape differs from context_codes")
        if wrist_codes.dtype != np.int8:
            raise ValueError("wrist_context_codes must be int8")
        if wrist_scale.shape != context_scale.shape:
            raise ValueError("wrist_context_scale shape differs from context_scale")
        _require_finite("wrist_context_scale", wrist_scale)

    for name, value in (
        ("context_scale", context_scale),
        ("future_scale", future_scale),
    ):
        _require_finite(name, value)
        if not np.all(value > 0):
            raise ValueError(f"{name} must be positive")

    return {
        "compact": compact,
        "windows": windows,
        "context_frames": T,
        "future_frames": k,
        "token_count": token_count,
        "latent_dim": latent_dim,
    }
