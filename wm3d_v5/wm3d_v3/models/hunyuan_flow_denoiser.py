"""Flow-matching denoiser for Hunyuan VAE latents conditioned on wm3d controls."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm3d_v3.models.hunyuan_latent_adapter import Conv3dBlock, _norm_groups


def timestep_embedding(x: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding for continuous flow sigmas in [0, 1]."""
    if x.ndim != 1:
        x = x.reshape(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=x.device) / max(1, half)
    )
    args = x.float()[:, None] * freqs[None] * max_period
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


@dataclass
class HunyuanFlowDenoiserConfig:
    token_dim: int = 2048
    hidden: int = 192
    latent_channels: int = 16
    action_dim: int = 7
    task_dim: int = 2048
    n_blocks: int = 4
    use_motion: bool = True
    use_rough_rgb: bool = True
    use_rough_latents: bool = True
    use_context: bool = True
    use_action: bool = True
    use_task: bool = True


class HunyuanFlowDenoiser(nn.Module):
    """Predict flow velocity `noise - clean_latent` from noisy latents and wm3d controls."""

    def __init__(self, cfg: HunyuanFlowDenoiserConfig | None = None):
        super().__init__()
        self.cfg = cfg or HunyuanFlowDenoiserConfig()
        h = self.cfg.hidden

        self.noisy_proj = nn.Sequential(
            nn.Conv3d(self.cfg.latent_channels, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.rough_latent_proj = nn.Sequential(
            nn.Conv3d(self.cfg.latent_channels, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
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
        self.rough_rgb_proj = nn.Sequential(
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
        self.sigma_proj = nn.Sequential(
            nn.Linear(h, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.blocks = nn.Sequential(*[Conv3dBlock(h) for _ in range(self.cfg.n_blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels, kernel_size=3, padding=1),
        )
        self.zero_init_output()

    def zero_init_output(self) -> None:
        conv = self.out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final denoiser layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    @staticmethod
    def _grid_size(patches: int) -> int:
        grid = int(math.isqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"P must be a square token grid, got P={patches}")
        return grid

    @staticmethod
    def _resize_video(x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        sigma: torch.Tensor,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor,
        *,
        context_rgb: torch.Tensor | None = None,
        motion_hint: torch.Tensor | None = None,
        rough_rgb: torch.Tensor | None = None,
        rough_latents: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_latents.ndim != 5:
            raise ValueError(f"noisy_latents must be [B,C,T,H,W], got {tuple(noisy_latents.shape)}")
        if pred_tokens.ndim != 4:
            raise ValueError(f"pred_tokens must be [B,k,P,D], got {tuple(pred_tokens.shape)}")
        if depth.ndim != 4:
            raise ValueError(f"depth must be [B,k,H,W], got {tuple(depth.shape)}")

        bsz, _, latent_t, latent_h, latent_w = noisy_latents.shape
        horizon, patches, dim = pred_tokens.shape[1], pred_tokens.shape[2], pred_tokens.shape[3]
        if pred_tokens.shape[0] != bsz or depth.shape[:2] != (bsz, horizon):
            raise ValueError("control batch/horizon dimensions do not match")
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")

        x = self.noisy_proj(noisy_latents)

        if self.cfg.use_rough_latents and rough_latents is not None:
            x = x + self.rough_latent_proj(rough_latents.to(dtype=x.dtype))

        grid = self._grid_size(patches)
        tok = pred_tokens.reshape(bsz * horizon, patches, dim).transpose(1, 2)
        tok = tok.reshape(bsz * horizon, dim, grid, grid)
        tok = self.token_proj(tok)
        tok = F.interpolate(tok, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        tok = tok.reshape(bsz, horizon, self.cfg.hidden, latent_h, latent_w).permute(0, 2, 1, 3, 4)
        x = x + self._resize_video(tok, latent_t, latent_h, latent_w)

        depth_v = depth[:, :, None].permute(0, 2, 1, 3, 4)
        depth_v = self._resize_video(depth_v, latent_t, latent_h, latent_w)
        x = x + self.depth_proj(depth_v.to(dtype=x.dtype))

        if self.cfg.use_motion and motion_hint is not None:
            motion_v = motion_hint.permute(0, 2, 1, 3, 4).contiguous()
            motion_v = self._resize_video(motion_v, latent_t, latent_h, latent_w)
            x = x + self.motion_proj(motion_v.to(dtype=x.dtype))

        if self.cfg.use_rough_rgb and rough_rgb is not None:
            rough_v = rough_rgb.permute(0, 2, 1, 3, 4).contiguous()
            rough_v = self._resize_video(rough_v, latent_t, latent_h, latent_w)
            x = x + self.rough_rgb_proj(rough_v.to(dtype=x.dtype))

        if self.cfg.use_context and context_rgb is not None:
            ctx = F.interpolate(context_rgb, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
            ctx = self.context_proj(ctx.to(dtype=x.dtype))[:, :, None]
            x = x + ctx.expand(-1, -1, latent_t, -1, -1)

        if self.cfg.use_action:
            if action_cond is None:
                action_cond = pred_tokens.new_zeros(bsz, horizon, self.cfg.action_dim)
            action = self.action_proj(action_cond.to(dtype=x.dtype)).permute(0, 2, 1)[:, :, :, None, None]
            action = self._resize_video(action, latent_t, latent_h, latent_w)
            x = x + action

        if self.cfg.use_task:
            if task_emb is None:
                task_emb = pred_tokens.new_zeros(bsz, self.cfg.task_dim)
            task = self.task_proj(task_emb.to(dtype=x.dtype))[:, :, None, None, None]
            x = x + task

        sigma_emb = timestep_embedding(sigma, self.cfg.hidden).to(dtype=x.dtype)
        sigma_emb = self.sigma_proj(sigma_emb)[:, :, None, None, None]
        x = x + sigma_emb

        x = self.blocks(x)
        return self.out(x)
