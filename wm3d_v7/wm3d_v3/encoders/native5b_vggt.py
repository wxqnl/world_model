"""Leak-free three-view VGGT cache encoder for native WM3D-V7 5B."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .vggt_encoder import VGGTEncoder


def _pool_scalar_grid(value: torch.Tensor, grid: int) -> torch.Tensor:
    """Pool [B,T,H,W] or [B,T,H,W,1] to [B,T,P]."""

    if value.ndim == 5 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.ndim != 4:
        raise ValueError(f"expected scalar image grid, got {tuple(value.shape)}")
    batch, frames, height, width = value.shape
    pooled = F.adaptive_avg_pool2d(
        value.reshape(batch * frames, 1, height, width).float(),
        (grid, grid),
    )
    return pooled.reshape(batch, frames, grid * grid)


def _pool_vector_grid(value: torch.Tensor, grid: int) -> torch.Tensor:
    """Pool [B,T,H,W,C] to [B,T,P,C]."""

    if value.ndim != 5:
        raise ValueError(f"expected vector image grid, got {tuple(value.shape)}")
    batch, frames, height, width, channels = value.shape
    pooled = F.adaptive_avg_pool2d(
        value.permute(0, 1, 4, 2, 3).reshape(
            batch * frames, channels, height, width
        ).float(),
        (grid, grid),
    )
    return pooled.reshape(batch, frames, channels, grid * grid).permute(
        0, 1, 3, 2
    )


class Native5BVGGTEncoder(torch.nn.Module):
    """Encode each visual time independently while jointly using three views.

    Input is [B,T,V,3,H,W].  It is reshaped to [B*T,V,3,H,W] before VGGT, so
    head/left/right cameras at one time can establish 3D geometry, but no
    context feature can attend to a future time.  This is the cache-level
    future-leak guard.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        model_name: str = "facebook/VGGT-1B",
        model_revision: str,
        local_files_only: bool = True,
        token_grid: int = 12,
        target_rgb_size: int = 384,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if int(token_grid) != 12:
            raise ValueError("native5b production token grid must be 12")
        self.grid = int(token_grid)
        self.target_rgb_size = int(target_rgb_size)
        self.encoder = VGGTEncoder(
            device=device,
            model_name=model_name,
            token_grid=token_grid,
            return_depth=True,
            return_depth_conf=True,
            return_geom_extra=True,
            dtype=dtype,
            model_revision=model_revision,
            local_files_only=local_files_only,
        )

    @torch.inference_mode()
    def forward(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 6:
            raise ValueError("native5b images must be [B,T,V,3,H,W]")
        batch, times, views, channels, height, width = images.shape
        if views != 3 or channels != 3:
            raise ValueError("native5b requires exactly three RGB views")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("non-finite RGB input")
        if float(images.min()) < 0.0 or float(images.max()) > 1.0:
            raise ValueError("RGB input must be normalized to [0,1]")
        if view_mask is None:
            view_mask = torch.ones(
                (batch, times, views),
                dtype=torch.bool,
                device=images.device,
            )
        if tuple(view_mask.shape) != (batch, times, views):
            raise ValueError("view_mask must be [B,T,V]")
        view_mask = view_mask.to(device=images.device, dtype=torch.bool)
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every encoded frame requires at least one RGB view")
        availability = view_mask[0, 0]
        if not bool((view_mask == availability[None, None]).all()):
            raise ValueError(
                "one encoder batch must use a stable camera-availability layout"
            )
        active_indices = torch.nonzero(availability, as_tuple=False).flatten()
        active_views = int(active_indices.numel())

        # Time is folded into the independent batch dimension, never the VGGT
        # sequence dimension.  Only simultaneous *available* views interact;
        # missing cameras never become black-image geometry evidence.
        active_images = images.index_select(2, active_indices)
        flat = active_images.reshape(
            batch * times,
            active_views,
            channels,
            height,
            width,
        )
        encoded: dict[str, Any] = self.encoder(flat)
        if encoded.get("geom_extra_missing"):
            raise RuntimeError(
                "formal native5b requires all VGGT geometry heads: "
                f"{encoded['geom_extra_missing']}"
            )
        active_tokens = encoded["pooled"].reshape(
            batch,
            times,
            active_views,
            self.grid * self.grid,
            -1,
        )
        if active_tokens.shape[-1] != 2048:
            raise RuntimeError(
                f"VGGT token dim drifted to {active_tokens.shape[-1]}"
            )
        active_depth = _pool_scalar_grid(encoded["depth"], self.grid).reshape(
            batch,
            times,
            active_views,
            self.grid * self.grid,
        )
        active_depth_confidence = _pool_scalar_grid(
            encoded["depth_conf"], self.grid
        ).reshape(batch, times, active_views, self.grid * self.grid)
        active_point = _pool_vector_grid(
            encoded["world_points"],
            self.grid,
        ).reshape(
            batch,
            times,
            active_views,
            self.grid * self.grid,
            3,
        )
        active_point_confidence = _pool_scalar_grid(
            encoded["world_points_conf"], self.grid
        ).reshape(batch, times, active_views, self.grid * self.grid)
        active_confidence = torch.sqrt(
            active_depth_confidence.float().clamp_min(0.0)
            * active_point_confidence.float().clamp_min(0.0)
        )
        active_confidence = active_confidence / active_confidence.amax(
            dim=(-1, -2), keepdim=True
        ).clamp_min(1.0e-6)
        active_pose = encoded["pose_enc"].reshape(
            batch,
            times,
            active_views,
            -1,
        )
        if active_pose.shape[-1] < 9:
            raise RuntimeError(
                f"VGGT pose dim {active_pose.shape[-1]} is below 9"
            )
        active_pose = active_pose[..., :9]

        view_tokens = active_tokens.new_zeros(
            batch,
            times,
            views,
            self.grid * self.grid,
            active_tokens.shape[-1],
        )
        depth = active_depth.new_zeros(
            batch, times, views, self.grid * self.grid
        )
        point = active_point.new_zeros(
            batch, times, views, self.grid * self.grid, 3
        )
        confidence = active_confidence.new_zeros(
            batch, times, views, self.grid * self.grid
        )
        pose = active_pose.new_zeros(batch, times, views, 9)
        view_tokens[:, :, active_indices] = active_tokens
        depth[:, :, active_indices] = active_depth
        point[:, :, active_indices] = active_point
        confidence[:, :, active_indices] = active_confidence
        pose[:, :, active_indices] = active_pose
        weights = confidence[..., None]
        world_tokens = (view_tokens.float() * weights).sum(dim=2) / weights.sum(
            dim=2
        ).clamp_min(1.0e-6)

        rgb = F.interpolate(
            images.reshape(batch * times * views, 3, height, width),
            size=(self.target_rgb_size, self.target_rgb_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(
            batch,
            times,
            views,
            3,
            self.target_rgb_size,
            self.target_rgb_size,
        )
        return {
            "view_tokens": view_tokens.to(torch.bfloat16),
            "view_mask": view_mask,
            "world_tokens": world_tokens.to(torch.bfloat16),
            "frame_summary": world_tokens.mean(dim=3).to(torch.bfloat16),
            "rgb": rgb.mul(255.0).round().clamp(0, 255).to(torch.uint8),
            "depth": depth.to(torch.float16),
            "point": point.to(torch.float16),
            "geometry_confidence": confidence.to(torch.float16),
            "camera_pose": pose.to(torch.float32),
        }
