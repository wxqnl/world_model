"""Leak-free, variable-view VGGT encoder for every WM3D model profile."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .vggt_encoder import VGGTEncoder


NATIVE_VGGT_SCHEMA = "wm3d_native_vggt_encoder_v1"
VGGT_CACHED_FEATURE_LAYERS = (4, 11, 17, 23)


@dataclass(frozen=True)
class NativeVGGTConfig:
    schema: str = NATIVE_VGGT_SCHEMA
    model_name: str = "facebook/VGGT-1B"
    model_revision: str = ""
    token_grid: int = 12
    appearance_token_grid: int = 0
    appearance_feature_layer: int = -1
    # VGGT input resolution and WM3D's stored RGB target resolution are two
    # different contracts.  518 is 37 patches at VGGT's patch size 14.
    input_rgb_size: int = 518
    input_preprocess: str = "aspect_pad_white"
    target_rgb_size: int = 384
    token_dim: int = 2048
    max_views: int = 3
    dtype: str = "bf16"
    time_isolation: str = "fold_time_into_batch"
    missing_view_policy: str = "exclude_from_geometry"

    def validate(self) -> None:
        if self.schema != NATIVE_VGGT_SCHEMA:
            raise ValueError(f"encoder schema must be {NATIVE_VGGT_SCHEMA}")
        if not self.model_revision:
            raise ValueError("VGGT model revision must be pinned")
        if (
            self.token_grid <= 0
            or self.input_rgb_size <= 0
            or self.target_rgb_size <= 0
            or self.max_views <= 0
        ):
            raise ValueError("token grid/input RGB/target RGB/max_views must be positive")
        if self.input_rgb_size % 14:
            raise ValueError("VGGT input_rgb_size must be divisible by patch size 14")
        if self.appearance_token_grid < 0:
            raise ValueError("appearance_token_grid cannot be negative")
        if self.appearance_feature_layer < -1:
            raise ValueError(
                "appearance_feature_layer must be -1 or a non-negative cached layer"
            )
        if (
            self.appearance_token_grid > 0
            and self.appearance_feature_layer not in (-1, *VGGT_CACHED_FEATURE_LAYERS)
        ):
            raise ValueError(
                "appearance_feature_layer must select a VGGT cached layer: "
                f"{VGGT_CACHED_FEATURE_LAYERS}"
            )
        if self.appearance_token_grid == 0 and self.appearance_feature_layer != -1:
            raise ValueError(
                "appearance_feature_layer requires appearance_token_grid"
            )
        if self.appearance_token_grid > self.input_rgb_size // 14:
            raise ValueError(
                "appearance_token_grid exceeds the native VGGT patch grid"
            )
        if self.input_preprocess != "aspect_pad_white":
            raise ValueError(
                "VGGT input preprocessing must preserve aspect ratio and pad white"
            )
        if self.token_dim != 2048:
            raise ValueError("current sealed VGGT external token dimension must be 2048")
        if self.dtype not in {"bf16", "fp16"}:
            raise ValueError("encoder dtype must be bf16 or fp16")
        if self.time_isolation != "fold_time_into_batch":
            raise ValueError("time must be folded into batch to prevent future leakage")
        if self.missing_view_policy != "exclude_from_geometry":
            raise ValueError("missing views cannot be replaced with geometry evidence")


def _pool_scalar(value: torch.Tensor, grid: int) -> torch.Tensor:
    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 4:
        raise ValueError(f"expected scalar image grid, got {tuple(value.shape)}")
    batch, frames, height, width = value.shape
    return F.adaptive_avg_pool2d(
        value.reshape(batch * frames, 1, height, width).float(), (grid, grid)
    ).reshape(batch, frames, grid * grid)


def _pool_vector(value: torch.Tensor, grid: int) -> torch.Tensor:
    if value.ndim != 5:
        raise ValueError(f"expected vector image grid, got {tuple(value.shape)}")
    batch, frames, height, width, channels = value.shape
    return F.adaptive_avg_pool2d(
        value.permute(0, 1, 4, 2, 3).reshape(
            batch * frames, channels, height, width
        ).float(),
        (grid, grid),
    ).reshape(batch, frames, channels, grid * grid).permute(0, 1, 3, 2)


class NativeVGGTEncoder(torch.nn.Module):
    """Encode simultaneous cameras jointly and every timestamp independently.

    Input ``[B,T,V,3,H,W]`` is partitioned by the actual camera availability
    pattern.  For each partition, time is folded into the batch dimension and
    only the simultaneous active views enter VGGT's sequence dimension.  This
    supports one, two or many cameras without black-image placeholders and
    makes future-frame leakage structurally impossible.
    """

    def __init__(
        self,
        config: NativeVGGTConfig,
        *,
        device: str = "cuda",
        local_files_only: bool = True,
        encoder: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
        self.encoder = encoder or VGGTEncoder(
            device=device,
            model_name=config.model_name,
            model_revision=config.model_revision,
            local_files_only=local_files_only,
            token_grid=config.token_grid,
            appearance_token_grid=config.appearance_token_grid or None,
            appearance_feature_layer=(
                config.appearance_feature_layer
                if config.appearance_token_grid
                else None
            ),
            return_depth=True,
            return_depth_conf=True,
            return_geom_extra=True,
            dtype=dtype,
        )

    def _encode_pattern(
        self,
        images: torch.Tensor,
        *,
        geometry_row_mask: torch.Tensor | None,
        appearance_row_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        selective_forward = getattr(self.encoder, "forward_selective", None)
        if callable(selective_forward):
            encoded: dict[str, Any] = selective_forward(
                images,
                geometry_batch_mask=geometry_row_mask,
                appearance_batch_mask=appearance_row_mask,
            )
        else:
            encoded = self.encoder(images)
        if encoded.get("geom_extra_missing"):
            raise RuntimeError(
                "formal native cache requires VGGT depth/point/camera heads: "
                f"{encoded['geom_extra_missing']}"
            )
        grid = self.config.token_grid
        tokens = encoded["pooled"]
        if tokens.ndim != 4 or tokens.shape[-2:] != (
            grid * grid,
            self.config.token_dim,
        ):
            raise RuntimeError(f"VGGT token ABI drifted to {tuple(tokens.shape)}")
        appearance_tokens = encoded.get("appearance_pooled")
        appearance_grid = int(self.config.appearance_token_grid)
        if appearance_grid:
            if appearance_tokens is None or appearance_tokens.ndim != 4:
                raise RuntimeError("VGGT appearance token output is missing")
            if appearance_tokens.shape[-2:] != (
                appearance_grid * appearance_grid,
                self.config.token_dim,
            ):
                raise RuntimeError(
                    f"VGGT appearance token ABI drifted to {tuple(appearance_tokens.shape)}"
                )
        depth = _pool_scalar(encoded["depth"], grid)
        depth_conf = _pool_scalar(encoded["depth_conf"], grid)
        point = _pool_vector(encoded["world_points"], grid)
        point_conf = _pool_scalar(encoded["world_points_conf"], grid)
        confidence = torch.sqrt(
            depth_conf.float().clamp_min(0.0) * point_conf.float().clamp_min(0.0)
        )
        confidence = confidence / confidence.amax(
            dim=(-1, -2), keepdim=True
        ).clamp_min(1.0e-6)
        pose = encoded["pose_enc"]
        if pose.shape[-1] < 9:
            raise RuntimeError(f"VGGT pose dim {pose.shape[-1]} is below 9")
        result = {
            "tokens": tokens,
            "depth": depth,
            "point": point,
            "confidence": confidence,
            "pose": pose[..., :9],
        }
        if appearance_tokens is not None:
            result["appearance_tokens"] = appearance_tokens
        return result

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        *,
        geometry_row_mask: torch.Tensor | None = None,
        appearance_row_mask: torch.Tensor | None = None,
        rgb_row_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 6:
            raise ValueError("images must be [B,T,V,3,H,W]")
        batch, times, views, channels, height, width = images.shape
        if channels != 3 or not 0 < views <= self.config.max_views:
            raise ValueError("RGB channel/view count is incompatible with encoder")
        if tuple(view_mask.shape) != (batch, times, views):
            raise ValueError("view_mask must be [B,T,V]")
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every timestamp requires at least one real camera")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("RGB input contains NaN/Inf")
        if float(images.min()) < 0 or float(images.max()) > 1:
            raise ValueError("RGB input must be normalized to [0,1]")
        if (height, width) != (
            self.config.input_rgb_size,
            self.config.input_rgb_size,
        ):
            raise ValueError(
                "RGB input size differs from the sealed VGGT input contract: "
                f"{(height, width)} != "
                f"{(self.config.input_rgb_size, self.config.input_rgb_size)}"
            )

        def normalize_row_mask(
            value: torch.Tensor | None, name: str
        ) -> torch.Tensor:
            if value is None:
                return torch.ones(
                    (batch, times), dtype=torch.bool, device=images.device
                )
            if tuple(value.shape) != (batch, times):
                raise ValueError(
                    f"{name} must be [B,T], got {tuple(value.shape)}"
                )
            return value.to(device=images.device, dtype=torch.bool)

        geometry_rows = normalize_row_mask(geometry_row_mask, "geometry_row_mask")
        appearance_rows = normalize_row_mask(
            appearance_row_mask, "appearance_row_mask"
        )
        rgb_rows = normalize_row_mask(rgb_row_mask, "rgb_row_mask")
        flat_images = images.reshape(batch * times, views, 3, height, width)
        flat_mask = view_mask.reshape(batch * times, views).bool()
        flat_geometry_rows = geometry_rows.reshape(batch * times)
        flat_appearance_rows = appearance_rows.reshape(batch * times)
        flat_rgb_rows = rgb_rows.reshape(batch * times)
        partitions: dict[tuple[bool, ...], list[int]] = defaultdict(list)
        for row, pattern in enumerate(flat_mask.cpu().tolist()):
            partitions[tuple(bool(item) for item in pattern)].append(row)

        patches = self.config.token_grid * self.config.token_grid
        tokens = images.new_zeros(batch * times, views, patches, self.config.token_dim)
        appearance_grid = int(self.config.appearance_token_grid)
        appearance_tokens = None
        if appearance_grid:
            appearance_tokens = images.new_zeros(
                batch * times,
                views,
                appearance_grid * appearance_grid,
                self.config.token_dim,
            )
        depth = images.new_zeros(batch * times, views, patches)
        point = images.new_zeros(batch * times, views, patches, 3)
        confidence = images.new_zeros(batch * times, views, patches)
        pose = images.new_zeros(batch * times, views, 9)
        for pattern, row_list in sorted(partitions.items()):
            active = torch.tensor(
                [index for index, present in enumerate(pattern) if present],
                dtype=torch.long,
                device=images.device,
            )
            rows = torch.tensor(row_list, dtype=torch.long, device=images.device)
            selected = flat_images.index_select(0, rows).index_select(1, active)
            # The second dimension is simultaneous views only.  Different
            # times remain independent batch elements.
            encoded = self._encode_pattern(
                selected,
                geometry_row_mask=flat_geometry_rows.index_select(0, rows),
                appearance_row_mask=flat_appearance_rows.index_select(0, rows),
            )
            outputs = [
                (tokens, "tokens"),
                (depth, "depth"),
                (point, "point"),
                (confidence, "confidence"),
                (pose, "pose"),
            ]
            if appearance_tokens is not None:
                outputs.insert(1, (appearance_tokens, "appearance_tokens"))
            for output, name in outputs:
                value = encoded[name].to(dtype=output.dtype)
                for local_view, global_view in enumerate(active.tolist()):
                    output[rows, global_view] = value[:, local_view]

        tokens = tokens.reshape(batch, times, views, patches, -1)
        if appearance_tokens is not None:
            appearance_tokens = appearance_tokens.reshape(
                batch, times, views, appearance_grid * appearance_grid, -1
            )
        depth = depth.reshape(batch, times, views, patches)
        point = point.reshape(batch, times, views, patches, 3)
        confidence = confidence.reshape(batch, times, views, patches)
        pose = pose.reshape(batch, times, views, 9)
        target_size = self.config.target_rgb_size
        rgb = images.new_zeros(
            batch * times, views, 3, target_size, target_size
        )
        rgb_indices = torch.nonzero(flat_rgb_rows, as_tuple=False).flatten()
        if int(rgb_indices.numel()):
            selected_rgb = flat_images.index_select(0, rgb_indices)
            resized_rgb = F.interpolate(
                selected_rgb.reshape(-1, 3, height, width),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(-1, views, 3, target_size, target_size)
            rgb.index_copy_(0, rgb_indices, resized_rgb)
        rgb = rgb.reshape(
            batch,
            times,
            views,
            3,
            target_size,
            target_size,
        )
        result = {
            "view_tokens": tokens.to(torch.bfloat16),
            "view_mask": view_mask.bool(),
            "rgb": rgb.mul(255).round().clamp(0, 255).to(torch.uint8),
            "depth": depth.to(torch.float16),
            "point": point.to(torch.float16),
            "geometry_confidence": confidence.to(torch.float16),
            "camera_pose": pose.to(torch.float32),
        }
        if appearance_tokens is not None:
            result["appearance_tokens"] = appearance_tokens.to(torch.bfloat16)
        return result
