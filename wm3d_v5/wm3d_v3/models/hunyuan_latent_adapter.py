"""Adapters from wm3d structured controls to Hunyuan VAE latents."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class Conv3dBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class HunyuanLatentAdapterConfig:
    token_dim: int = 2048
    token_grid: int = 8
    hidden: int = 192
    latent_channels: int = 16
    latent_time: int = 3
    latent_hw: int = 32
    action_dim: int = 7
    task_dim: int = 2048
    n_blocks: int = 4
    use_motion: bool = True
    use_rough_rgb: bool = True
    use_context: bool = True
    use_action: bool = True
    use_task: bool = True


class HunyuanLatentAdapter(nn.Module):
    """Predict scaled Hunyuan VAE latents from frozen wm3d controls.

    The target latent corresponds to a 9-frame clip: current context frame plus
    the 8 predicted future frames. For the 884 Hunyuan VAE this compresses to
    three latent time steps at 256 -> 32 spatial resolution.
    """

    def __init__(self, cfg: HunyuanLatentAdapterConfig | None = None):
        super().__init__()
        self.cfg = cfg or HunyuanLatentAdapterConfig()
        h = self.cfg.hidden

        self.token_proj = nn.Sequential(
            nn.Conv2d(self.cfg.token_dim, h, kernel_size=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv3d(1, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.motion_proj = nn.Sequential(
            nn.Conv3d(1, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.rough_proj = nn.Sequential(
            nn.Conv3d(3, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(3, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(self.cfg.action_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.blocks = nn.Sequential(*[Conv3dBlock(h) for _ in range(self.cfg.n_blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels, kernel_size=3, padding=1),
        )

    def zero_init_output(self) -> None:
        """Start residual prediction as an identity passthrough."""
        conv = self.out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final adapter layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    @staticmethod
    def _grid_size(patches: int) -> int:
        grid = int(math.isqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"P must be a square token grid, got P={patches}")
        return grid

    def _target_shape(self, target_latents: torch.Tensor | None) -> tuple[int, int, int]:
        if target_latents is None:
            return self.cfg.latent_time, self.cfg.latent_hw, self.cfg.latent_hw
        if target_latents.ndim != 5:
            raise ValueError(f"target_latents must be [B,C,T,H,W], got {tuple(target_latents.shape)}")
        return target_latents.shape[2], target_latents.shape[3], target_latents.shape[4]

    def _resize_video(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)

    def forward(
        self,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor,
        *,
        context_rgb: torch.Tensor | None = None,
        motion_hint: torch.Tensor | None = None,
        rough_rgb: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
        target_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred_tokens.ndim != 4:
            raise ValueError(f"pred_tokens must be [B,k,P,D], got {tuple(pred_tokens.shape)}")
        if depth.ndim != 4:
            raise ValueError(f"depth must be [B,k,H,W], got {tuple(depth.shape)}")

        bsz, horizon, patches, dim = pred_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        if depth.shape[:2] != (bsz, horizon):
            raise ValueError(f"depth leading dims must be {(bsz, horizon)}, got {tuple(depth.shape[:2])}")
        latent_t, latent_h, latent_w = self._target_shape(target_latents)

        grid = self._grid_size(patches)
        x = pred_tokens.reshape(bsz * horizon, patches, dim).transpose(1, 2)
        x = x.reshape(bsz * horizon, dim, grid, grid)
        x = self.token_proj(x)
        x = F.interpolate(x, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        x = x.reshape(bsz, horizon, self.cfg.hidden, latent_h, latent_w).permute(0, 2, 1, 3, 4)

        depth_v = depth[:, :, None]
        depth_v = self._resize_video(depth_v.permute(0, 2, 1, 3, 4), horizon, latent_h, latent_w)
        x = x + self.depth_proj(depth_v.to(dtype=x.dtype))

        if self.cfg.use_motion and motion_hint is not None:
            motion_v = motion_hint.permute(0, 2, 1, 3, 4).contiguous()
            motion_v = self._resize_video(motion_v, horizon, latent_h, latent_w)
            x = x + self.motion_proj(motion_v.to(dtype=x.dtype))

        if self.cfg.use_rough_rgb and rough_rgb is not None:
            rough_v = self._resize_video(rough_rgb.permute(0, 2, 1, 3, 4), horizon, latent_h, latent_w)
            x = x + self.rough_proj(rough_v.to(dtype=x.dtype))

        if self.cfg.use_context and context_rgb is not None:
            ctx = F.interpolate(context_rgb, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
            ctx = self.context_proj(ctx.to(dtype=x.dtype))[:, :, None]
            x = x + ctx.expand(-1, -1, horizon, -1, -1)

        if self.cfg.use_action:
            if action_cond is None:
                action_cond = pred_tokens.new_zeros(bsz, horizon, self.cfg.action_dim)
            action = self.action_proj(action_cond.to(dtype=x.dtype)).permute(0, 2, 1)[:, :, :, None, None]
            x = x + action

        if self.cfg.use_task:
            if task_emb is None:
                task_emb = pred_tokens.new_zeros(bsz, self.cfg.task_dim)
            task = self.task_proj(task_emb.to(dtype=x.dtype))[:, :, None, None, None]
            x = x + task

        x = self._resize_video(x, latent_t, latent_h, latent_w)
        x = self.blocks(x)
        return self.out(x)
