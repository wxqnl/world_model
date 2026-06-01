"""Independent control prediction head for future state tokens."""
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


@dataclass
class ControlHeadConfig:
    token_dim: int = 2048
    hidden: int = 128
    output_size: int = 256
    fuse_size: int = 64
    refine_channels: int = 16
    use_refine: bool = True
    action_dim: int = 7
    task_dim: int = 2048
    use_context: bool = True
    use_action: bool = True
    use_task: bool = True


class ControlHead(nn.Module):
    """Predict motion/contact control maps from future tokens and geometry.

    The head is intentionally standalone: it consumes already-predicted future
    tokens and optional conditioning, and returns dense 2D maps at a fixed output
    size. Patch layout is inferred from P, so both P64 (8x8) and P256 (16x16)
    tokens are supported without changing the module.
    """

    def __init__(self, cfg: ControlHeadConfig | None = None):
        super().__init__()
        self.cfg = cfg or ControlHeadConfig()
        h = self.cfg.hidden
        r = self.cfg.refine_channels

        self.token_proj = nn.Sequential(
            nn.Conv2d(self.cfg.token_dim, h, kernel_size=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            ConvBlock(h, h),
        )
        self.depth_proj = nn.Sequential(
            nn.Conv2d(1, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(3, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            ConvBlock(h, h),
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
        self.fuse = nn.Sequential(
            ConvBlock(h * 3, h),
            ConvBlock(h, h),
        )
        self.map_head = nn.Conv2d(h, 2, kernel_size=3, padding=1)
        refine_in = 2 + 1 + (3 if self.cfg.use_context else 0)
        self.refine = nn.Sequential(
            nn.Conv2d(refine_in, r, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(r), r),
            nn.SiLU(inplace=True),
            ConvBlock(r, r),
            nn.Conv2d(r, 2, kernel_size=3, padding=1),
        ) if self.cfg.use_refine and self.cfg.output_size > self.cfg.fuse_size else None
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(h, 1),
        )

        nn.init.constant_(self.map_head.bias, -4.0)
        if self.refine is not None:
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    @staticmethod
    def _grid_size(patches: int) -> int:
        grid = int(math.isqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"P must be a square token grid, got P={patches}")
        return grid

    @staticmethod
    def _repeat_for_horizon(x: torch.Tensor, horizon: int) -> torch.Tensor:
        return x[:, None].expand(-1, horizon, -1, -1, -1).reshape(
            x.shape[0] * horizon, x.shape[1], x.shape[2], x.shape[3]
        )

    def _conditioning(
        self,
        bsz: int,
        horizon: int,
        device: torch.device,
        dtype: torch.dtype,
        action_cond: torch.Tensor | None,
        task_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        cond = torch.zeros(bsz * horizon, self.cfg.hidden, 1, 1, device=device, dtype=dtype)
        if self.cfg.use_action:
            if action_cond is None:
                action_cond = torch.zeros(
                    bsz, horizon, self.cfg.action_dim, device=device, dtype=dtype
                )
            if action_cond.shape != (bsz, horizon, self.cfg.action_dim):
                raise ValueError(
                    f"action_cond must be {(bsz, horizon, self.cfg.action_dim)}, "
                    f"got {tuple(action_cond.shape)}"
                )
            act = action_cond.to(device=device, dtype=dtype).reshape(bsz * horizon, -1)
            cond = cond + self.action_proj(act).view(bsz * horizon, -1, 1, 1)
        if self.cfg.use_task:
            if task_emb is None:
                task_emb = torch.zeros(bsz, self.cfg.task_dim, device=device, dtype=dtype)
            if task_emb.shape != (bsz, self.cfg.task_dim):
                raise ValueError(
                    f"task_emb must be {(bsz, self.cfg.task_dim)}, got {tuple(task_emb.shape)}"
                )
            task = task_emb.to(device=device, dtype=dtype)
            task = task[:, None].expand(-1, horizon, -1).reshape(bsz * horizon, -1)
            cond = cond + self.task_proj(task).view(bsz * horizon, -1, 1, 1)
        return cond

    def forward(
        self,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor,
        context_rgb: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return motion/contact logits, hints, and per-step confidence.

        Args:
            pred_tokens: [B, k, P, D] future tokens.
            depth: [B, k, Hd, Wd] predicted or observed depth.
            context_rgb: optional [B, 3, H, W] context frame.
            action_cond: optional [B, k, 7] action condition.
            task_emb: optional [B, 2048] task embedding.
        """
        if pred_tokens.ndim != 4:
            raise ValueError(f"pred_tokens must be [B,k,P,D], got {tuple(pred_tokens.shape)}")
        if depth.ndim != 4:
            raise ValueError(f"depth must be [B,k,Hd,Wd], got {tuple(depth.shape)}")

        bsz, horizon, patches, dim = pred_tokens.shape
        if depth.shape[:2] != (bsz, horizon):
            raise ValueError(
                f"depth leading dims must be {(bsz, horizon)}, got {tuple(depth.shape[:2])}"
            )
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected D={self.cfg.token_dim}, got {dim}")
        if context_rgb is not None and (context_rgb.ndim != 4 or context_rgb.shape[:2] != (bsz, 3)):
            raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context_rgb.shape)}")

        grid = self._grid_size(patches)
        fuse_hw = (self.cfg.fuse_size, self.cfg.fuse_size)
        out_hw = (self.cfg.output_size, self.cfg.output_size)
        tokens = pred_tokens.reshape(bsz * horizon, patches, dim).transpose(1, 2)
        tokens = tokens.reshape(bsz * horizon, dim, grid, grid)
        token_feat = self.token_proj(tokens)
        token_feat = F.interpolate(token_feat, size=fuse_hw, mode="bilinear", align_corners=False)

        depth_feat = depth.reshape(bsz * horizon, 1, depth.shape[-2], depth.shape[-1])
        depth_feat = F.interpolate(depth_feat, size=fuse_hw, mode="bilinear", align_corners=False)
        depth_feat = self.depth_proj(depth_feat.to(dtype=token_feat.dtype))

        if self.cfg.use_context and context_rgb is not None:
            context = F.interpolate(context_rgb, size=fuse_hw, mode="bilinear", align_corners=False)
            context_feat = self.context_proj(context.to(dtype=token_feat.dtype))
            context_feat = self._repeat_for_horizon(context_feat, horizon)
        else:
            context_feat = torch.zeros_like(token_feat)

        token_feat = token_feat + self._conditioning(
            bsz,
            horizon,
            pred_tokens.device,
            token_feat.dtype,
            action_cond,
            task_emb,
        )
        feat = self.fuse(torch.cat([token_feat, depth_feat, context_feat], dim=1))
        logits = self.map_head(feat)
        if logits.shape[-2:] != out_hw:
            logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        if self.refine is not None:
            refine_inputs = [logits]
            depth_ref = depth.reshape(bsz * horizon, 1, depth.shape[-2], depth.shape[-1])
            depth_ref = F.interpolate(depth_ref, size=out_hw, mode="bilinear", align_corners=False)
            refine_inputs.append(depth_ref.to(dtype=logits.dtype))
            if self.cfg.use_context and context_rgb is not None:
                ctx_ref = F.interpolate(context_rgb, size=out_hw, mode="bilinear", align_corners=False)
                ctx_ref = self._repeat_for_horizon(ctx_ref.to(dtype=logits.dtype), horizon)
                refine_inputs.append(ctx_ref)
            elif self.cfg.use_context:
                refine_inputs.append(logits.new_zeros(bsz * horizon, 3, *out_hw))
            logits = logits + self.refine(torch.cat(refine_inputs, dim=1))
        motion_logit = logits[:, 0:1].reshape(bsz, horizon, 1, *out_hw)
        contact_logit = logits[:, 1:2].reshape(bsz, horizon, 1, *out_hw)
        confidence = torch.sigmoid(self.confidence_head(feat).reshape(bsz, horizon))

        return {
            "motion_logit": motion_logit,
            "motion_hint": torch.sigmoid(motion_logit),
            "contact_logit": contact_logit,
            "contact_hint": torch.sigmoid(contact_logit),
            "control_confidence": confidence,
        }
