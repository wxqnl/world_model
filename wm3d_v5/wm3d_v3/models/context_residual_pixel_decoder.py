"""Context-conditioned pixel decoder for GT-token upper-bound experiments."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_norm_groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


@dataclass
class ContextResidualPixelDecoderConfig:
    token_dim: int = 2048
    token_grid: int = 16
    hidden: int = 384
    action_dim: int = 7
    task_dim: int = 2048
    residual_scale: float = 0.75
    use_action: bool = True
    use_task: bool = True
    predict_motion: bool = False
    motion_blend_gain: float = 0.0


class ContextResidualPixelDecoder(nn.Module):
    """Decode future RGB from GT VGGT tokens while borrowing context detail.

    This is intentionally not a world model. It tests whether P256 VGGT tokens
    contain enough information for crisp RGB when the renderer can reuse the
    latest observed frame through U-Net skips and residual prediction.
    """

    def __init__(self, cfg: ContextResidualPixelDecoderConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        c1 = max(32, h // 8)
        c2 = max(64, h // 4)
        c3 = max(128, h // 2)

        self.ctx256 = ConvBlock(3, c1)
        self.ctx128 = DownBlock(c1, c1)
        self.ctx64 = DownBlock(c1, c2)
        self.ctx32 = DownBlock(c2, c3)
        self.ctx16 = DownBlock(c3, h)

        self.token_proj = nn.Sequential(
            nn.Conv2d(cfg.token_dim, h, kernel_size=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            ConvBlock(h, h),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(cfg.action_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(cfg.task_dim),
            nn.Linear(cfg.task_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )

        self.fuse16 = ConvBlock(h + h, h)
        self.up32 = UpBlock(h, c3)
        self.fuse32 = ConvBlock(c3 + c3, c3)
        self.up64 = UpBlock(c3, c2)
        self.fuse64 = ConvBlock(c2 + c2, c2)
        self.up128 = UpBlock(c2, c1)
        self.fuse128 = ConvBlock(c1 + c1, c1)
        self.up256 = UpBlock(c1, c1)
        self.fuse256 = ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 7, kernel_size=3, padding=1)
        self.motion_head = nn.Conv2d(c1, 1, kernel_size=3, padding=1) if cfg.predict_motion else None
        if self.motion_head is not None:
            nn.init.zeros_(self.motion_head.weight)
            nn.init.constant_(self.motion_head.bias, -4.0)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _repeat_for_horizon(x: torch.Tensor, horizon: int) -> torch.Tensor:
        return x[:, None].expand(-1, horizon, -1, -1, -1).reshape(
            x.shape[0] * horizon, x.shape[1], x.shape[2], x.shape[3]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        context_rgb: torch.Tensor,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return RGB predictions in [0, 1], optionally with motion-control maps.

        Args:
            tokens: [B, k, P, D] future GT state tokens.
            context_rgb: [B, 3, H, W] latest observed RGB frame.
            action_cond: optional [B, k, action_dim] future action condition.
            task_emb: optional [B, task_dim] Qwen task embedding.
        """
        if tokens.ndim != 4:
            raise ValueError(f"tokens must be [B,k,P,D], got {tuple(tokens.shape)}")
        if context_rgb.ndim != 4 or context_rgb.shape[1] != 3:
            raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context_rgb.shape)}")

        bsz, horizon, patches, dim = tokens.shape
        grid = self.cfg.token_grid
        if patches != grid * grid:
            raise ValueError(f"expected P={grid * grid}, got {patches}")
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected D={self.cfg.token_dim}, got {dim}")

        s256 = self.ctx256(context_rgb)
        s128 = self.ctx128(s256)
        s64 = self.ctx64(s128)
        s32 = self.ctx32(s64)
        s16 = self.ctx16(s32)

        x = tokens.reshape(bsz * horizon, patches, dim).transpose(1, 2)
        x = x.reshape(bsz * horizon, dim, grid, grid)
        x = self.token_proj(x)
        if x.shape[-2:] != s16.shape[-2:]:
            x = F.interpolate(x, size=s16.shape[-2:], mode="bilinear", align_corners=False)

        if self.cfg.use_action:
            if action_cond is None:
                raise ValueError("action_cond is required when use_action=True")
            act = action_cond.reshape(bsz * horizon, -1)
            x = x + self.action_proj(act).to(dtype=x.dtype).view(bsz * horizon, -1, 1, 1)
        if self.cfg.use_task:
            if task_emb is None:
                raise ValueError("task_emb is required when use_task=True")
            task = task_emb[:, None].expand(-1, horizon, -1).reshape(bsz * horizon, -1)
            x = x + self.task_proj(task).to(dtype=x.dtype).view(bsz * horizon, -1, 1, 1)

        x = self.fuse16(torch.cat([x, self._repeat_for_horizon(s16, horizon)], dim=1))
        x = self.up32(x)
        x = self.fuse32(torch.cat([x, self._repeat_for_horizon(s32, horizon)], dim=1))
        x = self.up64(x)
        x = self.fuse64(torch.cat([x, self._repeat_for_horizon(s64, horizon)], dim=1))
        x = self.up128(x)
        x = self.fuse128(torch.cat([x, self._repeat_for_horizon(s128, horizon)], dim=1))
        x = self.up256(x)
        x = self.fuse256(torch.cat([x, self._repeat_for_horizon(s256, horizon)], dim=1))

        motion_logit = self.motion_head(x) if self.motion_head is not None else None
        motion_hint = torch.sigmoid(motion_logit) if motion_logit is not None else None
        raw = self.head(x)
        direct = torch.sigmoid(raw[:, 0:3])
        residual = torch.tanh(raw[:, 3:6]) * float(self.cfg.residual_scale)
        blend = torch.sigmoid(raw[:, 6:7])
        if motion_hint is not None and self.cfg.motion_blend_gain > 0:
            blend = torch.clamp(blend + motion_hint.to(dtype=blend.dtype) * self.cfg.motion_blend_gain, 0.0, 1.0)
        base = self._repeat_for_horizon(context_rgb, horizon)
        residual_rgb = torch.clamp(base + residual, 0.0, 1.0)
        rgb = blend * direct + (1.0 - blend) * residual_rgb
        rgb = rgb.reshape(bsz, horizon, 3, context_rgb.shape[-2], context_rgb.shape[-1])
        if not return_aux:
            return rgb
        out = {"rgb": rgb}
        if motion_logit is not None and motion_hint is not None:
            out["motion_logit"] = motion_logit.reshape(
                bsz, horizon, 1, context_rgb.shape[-2], context_rgb.shape[-1]
            )
            out["motion_hint"] = motion_hint.reshape(
                bsz, horizon, 1, context_rgb.shape[-2], context_rgb.shape[-1]
            )
            out["rgb_blend"] = blend.reshape(
                bsz, horizon, 1, context_rgb.shape[-2], context_rgb.shape[-1]
            )
        return out
