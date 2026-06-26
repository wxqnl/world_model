"""Wan2.2 TI2V control adapter for WM3D native state predictions.

This mirrors the Hunyuan stage125 boundary but targets the official WanModel
shape: video latents are patchified by Wan with ``patch_size=(1, 2, 2)`` and
then passed through a flat ``blocks`` ModuleList of WanAttentionBlock modules.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


CONTROL_CHECKPOINT_KIND = "wan_ti2v_control_adapter_v1"


def _norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _Conv3dBlock(nn.Module):
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


class _ActionTokenBlock(nn.Module):
    """Small action stream that produces token residuals for Wan blocks."""

    def __init__(self, action_hidden: int, video_hidden: int, heads: int, mlp_mult: float = 2.0):
        super().__init__()
        self.action_hidden = int(action_hidden)
        self.video_hidden = int(video_hidden)
        self.heads = max(1, int(heads))
        if self.action_hidden % self.heads != 0:
            self.action_hidden = int(math.ceil(self.action_hidden / self.heads) * self.heads)
        mlp_hidden = max(self.action_hidden, int(round(self.action_hidden * float(mlp_mult))))

        self.action_norm = nn.LayerNorm(self.action_hidden)
        self.action_qkv = nn.Linear(self.action_hidden, 3 * self.action_hidden)
        self.action_out = nn.Linear(self.action_hidden, self.action_hidden)
        self.video_norm = nn.LayerNorm(self.video_hidden)
        self.cross_q = nn.Linear(self.video_hidden, self.action_hidden)
        self.cross_k = nn.Linear(self.action_hidden, self.action_hidden)
        self.cross_v = nn.Linear(self.action_hidden, self.action_hidden)
        self.video_out = nn.Linear(self.action_hidden, self.video_hidden)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.action_hidden),
            nn.Linear(self.action_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.action_hidden),
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        d = self.action_hidden // self.heads
        return x.view(int(x.shape[0]), int(x.shape[1]), self.heads, d).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).reshape(int(x.shape[0]), int(x.shape[2]), self.action_hidden)

    def forward(self, action_tokens: torch.Tensor, video_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.action_qkv(self.action_norm(action_tokens)).chunk(3, dim=-1)
        action_tokens = action_tokens + self.action_out(
            self._merge(
                F.scaled_dot_product_attention(
                    self._split(q),
                    self._split(k),
                    self._split(v),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )
        action_tokens = action_tokens + self.mlp(action_tokens)

        qv = self.cross_q(self.video_norm(video_tokens))
        ka = self.cross_k(action_tokens)
        va = self.cross_v(action_tokens)
        video_delta = self.video_out(
            self._merge(
                F.scaled_dot_product_attention(
                    self._split(qv),
                    self._split(ka),
                    self._split(va),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )
        return action_tokens, video_delta


class _ParallelActionVideoBlock(nn.Module):
    """Bidirectional action/video interaction used inside each Wan block.

    The action stream first updates from current video tokens, then video tokens
    cross-attend the updated action stream.  This makes actions part of the
    denoising block computation instead of an output-only residual.
    """

    def __init__(self, action_hidden: int, video_hidden: int, heads: int, mlp_mult: float = 2.0):
        super().__init__()
        self.action_hidden = int(action_hidden)
        self.video_hidden = int(video_hidden)
        self.heads = max(1, int(heads))
        if self.action_hidden % self.heads != 0:
            self.action_hidden = int(math.ceil(self.action_hidden / self.heads) * self.heads)
        mlp_hidden = max(self.action_hidden, int(round(self.action_hidden * float(mlp_mult))))

        self.action_norm = nn.LayerNorm(self.action_hidden)
        self.action_qkv = nn.Linear(self.action_hidden, 3 * self.action_hidden)
        self.action_self_out = nn.Linear(self.action_hidden, self.action_hidden)

        self.action_cross_norm = nn.LayerNorm(self.action_hidden)
        self.video_key_norm = nn.LayerNorm(self.video_hidden)
        self.action_from_video_q = nn.Linear(self.action_hidden, self.action_hidden)
        self.video_to_action_k = nn.Linear(self.video_hidden, self.action_hidden)
        self.video_to_action_v = nn.Linear(self.video_hidden, self.action_hidden)
        self.action_from_video_out = nn.Linear(self.action_hidden, self.action_hidden)

        self.video_norm = nn.LayerNorm(self.video_hidden)
        self.video_q = nn.Linear(self.video_hidden, self.action_hidden)
        self.action_k = nn.Linear(self.action_hidden, self.action_hidden)
        self.action_v = nn.Linear(self.action_hidden, self.action_hidden)
        self.video_out = nn.Linear(self.action_hidden, self.video_hidden)

        self.action_mlp = nn.Sequential(
            nn.LayerNorm(self.action_hidden),
            nn.Linear(self.action_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.action_hidden),
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        d = self.action_hidden // self.heads
        return x.view(int(x.shape[0]), int(x.shape[1]), self.heads, d).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).reshape(int(x.shape[0]), int(x.shape[2]), self.action_hidden)

    def forward(self, action_tokens: torch.Tensor, video_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.action_qkv(self.action_norm(action_tokens)).chunk(3, dim=-1)
        action_tokens = action_tokens + self.action_self_out(
            self._merge(
                F.scaled_dot_product_attention(
                    self._split(q),
                    self._split(k),
                    self._split(v),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )

        qa = self.action_from_video_q(self.action_cross_norm(action_tokens))
        kv = self.video_key_norm(video_tokens)
        action_tokens = action_tokens + self.action_from_video_out(
            self._merge(
                F.scaled_dot_product_attention(
                    self._split(qa),
                    self._split(self.video_to_action_k(kv)),
                    self._split(self.video_to_action_v(kv)),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )
        action_tokens = action_tokens + self.action_mlp(action_tokens)

        qv = self.video_q(self.video_norm(video_tokens))
        an = self.action_cross_norm(action_tokens)
        video_delta = self.video_out(
            self._merge(
                F.scaled_dot_product_attention(
                    self._split(qv),
                    self._split(self.action_k(an)),
                    self._split(self.action_v(an)),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )
        return action_tokens, video_delta


@dataclass
class WanTI2VControlConfig:
    token_dim: int = 2048
    token_grid: int = 8
    latent_channels: int = 48
    hidden: int = 192
    dit_hidden: int = 3072
    action_dim: int = 7
    task_dim: int = 2048
    num_layers: int = 30
    patch_size: tuple[int, int, int] = (1, 2, 2)
    vae_stride: tuple[int, int, int] = (4, 16, 16)
    use_depth: bool = True
    use_motion: bool = True
    use_contact: bool = True
    use_context: bool = True
    use_action: bool = True
    use_task: bool = True
    use_point: bool = True
    use_pose: bool = True
    point_dim: int = 3
    pose_dim: int = 9
    use_noisy_latents: bool = True
    use_source_latents: bool = True
    use_sigma_embed: bool = True
    action_residual_scale: float = 0.75
    action_direct_scale: float = 0.25
    action_latent_scale: float = 0.20
    use_action_token_block: bool = True
    action_token_block_scale: float = 0.75
    action_token_hidden: int = 256
    action_token_heads: int = 4
    action_token_mlp_mult: float = 2.0
    use_parallel_action_video_blocks: bool = False
    parallel_action_video_scale: float = 1.0
    parallel_action_video_hidden: int = 256
    parallel_action_video_heads: int = 4
    parallel_action_video_mlp_mult: float = 2.0
    parallel_action_video_gate_source: str = "none"
    parallel_action_video_gate_min: float = 0.0
    parallel_action_video_gate_threshold: float = 0.05
    parallel_action_video_gate_dilate: int = 0
    parallel_action_video_gate_power: float = 1.0
    parallel_action_video_gate_detach: bool = True
    use_action_context_tokens: bool = False
    action_context_dim: int = 4096
    action_context_hidden: int = 512
    action_context_pos_scale: float = 0.05
    use_vam_action_expert: bool = False
    vam_action_freq_dim: int = 256
    vam_action_video_delta_scale: float = 1.0
    vam_action_policy_cond_scale: float = 0.0
    vam_video_use_clean_action: bool = False
    vam_video_action_source: str = "clean"
    vam_video_policy_blend: float = 0.0
    vam_update_noisy_action_stream: bool = True
    layer_gain_start: float = 0.85
    layer_gain_end: float = 1.35
    layer_gain_power: float = 1.0


@dataclass
class WanTI2VControlState:
    features: torch.Tensor
    scale: float = 1.0
    latent_shape: tuple[int, int, int, int, int] | None = None
    action_tokens: torch.Tensor | None = None
    action_stream: torch.Tensor | None = None
    video_action_tokens: torch.Tensor | None = None
    video_action_stream: torch.Tensor | None = None
    video_action_gate_tokens: torch.Tensor | None = None
    action_time_emb: torch.Tensor | None = None
    action_pred_velocity: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


class WanTI2VControlAdapter(nn.Module):
    """Encode WM3D controls into Wan token and latent residuals."""

    def __init__(self, cfg: WanTI2VControlConfig | None = None):
        super().__init__()
        self.cfg = cfg or WanTI2VControlConfig()
        h = self.cfg.hidden

        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.depth_proj = nn.Sequential(nn.Conv3d(1, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.motion_proj = nn.Sequential(nn.Conv3d(1, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.contact_proj = nn.Sequential(nn.Conv3d(1, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.context_proj = nn.Sequential(nn.Conv2d(3, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.action_proj = nn.Sequential(nn.LayerNorm(self.cfg.action_dim), nn.Linear(self.cfg.action_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.task_proj = nn.Sequential(nn.LayerNorm(self.cfg.task_dim), nn.Linear(self.cfg.task_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.point_proj = nn.Sequential(nn.Conv3d(self.cfg.point_dim, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.pose_proj = nn.Sequential(nn.Linear(self.cfg.pose_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.noisy_proj = nn.Sequential(nn.Conv3d(self.cfg.latent_channels, h, 1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.source_proj = nn.Sequential(nn.Conv3d(self.cfg.latent_channels, h, 1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.sigma_proj = nn.Sequential(nn.Linear(1, h), nn.SiLU(inplace=True), nn.Linear(h, h))

        self.fuse = nn.Sequential(_Conv3dBlock(h), _Conv3dBlock(h))
        self.to_dit = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, self.cfg.dit_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.dit_hidden, self.cfg.dit_hidden),
        )
        self.action_token_in = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, self.cfg.action_token_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.action_token_hidden, self.cfg.action_token_hidden),
        )
        self.action_context_in = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, self.cfg.action_context_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.action_context_hidden, self.cfg.action_context_dim),
        )
        self.vam_action_proj_in = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, self.cfg.parallel_action_video_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.parallel_action_video_hidden, self.cfg.parallel_action_video_hidden),
        )
        self.vam_policy_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, self.cfg.parallel_action_video_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.parallel_action_video_hidden, self.cfg.parallel_action_video_hidden),
        )
        self.vam_action_time_embedding = nn.Sequential(
            nn.Linear(self.cfg.vam_action_freq_dim, self.cfg.parallel_action_video_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.parallel_action_video_hidden, self.cfg.parallel_action_video_hidden),
        )
        self.vam_action_time_projection = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Linear(self.cfg.parallel_action_video_hidden, self.cfg.parallel_action_video_hidden * 2),
        )
        self.action_blocks = nn.ModuleList(
            [
                _ActionTokenBlock(
                    self.cfg.action_token_hidden,
                    self.cfg.dit_hidden,
                    self.cfg.action_token_heads,
                    self.cfg.action_token_mlp_mult,
                )
                for _ in range(max(1, int(self.cfg.num_layers)))
            ]
        )
        self.parallel_action_blocks = nn.ModuleList(
            [
                _ParallelActionVideoBlock(
                    self.cfg.parallel_action_video_hidden,
                    self.cfg.dit_hidden,
                    self.cfg.parallel_action_video_heads,
                    self.cfg.parallel_action_video_mlp_mult,
                )
                for _ in range(max(1, int(self.cfg.num_layers)))
            ]
        )
        self.latent_head = nn.Sequential(
            _Conv3dBlock(h),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels, 3, padding=1),
        )
        self.vam_action_head_norm = nn.LayerNorm(self.cfg.parallel_action_video_hidden)
        self.vam_action_head = nn.Linear(self.cfg.parallel_action_video_hidden, self.cfg.action_dim)
        self.vam_action_head_modulation = nn.Parameter(
            torch.randn(1, 2, self.cfg.parallel_action_video_hidden) / max(1, self.cfg.parallel_action_video_hidden) ** 0.5
        )
        self._control_state: WanTI2VControlState | None = None
        self.zero_init_output()

    @staticmethod
    def _action_position_encoding(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= 0 or dim <= 0:
            return torch.zeros(1, max(0, length), max(0, dim), device=device, dtype=dtype)
        pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        half = max(1, dim // 2)
        freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0 : 2 * half : 2] = torch.sin(pos * freq[: pe[:, 0 : 2 * half : 2].shape[1]])
        pe[:, 1 : 2 * half : 2] = torch.cos(pos * freq[: pe[:, 1 : 2 * half : 2].shape[1]])
        return pe.to(dtype=dtype)[None]

    @staticmethod
    def _sinusoidal_embedding(dim: int, position: torch.Tensor) -> torch.Tensor:
        if dim % 2 != 0:
            dim += 1
        half = dim // 2
        pos = position.float().reshape(-1, 1)
        freq = torch.exp(
            torch.arange(half, device=position.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        emb = torch.cat([torch.cos(pos * freq), torch.sin(pos * freq)], dim=1)
        return emb[:, :dim]

    @property
    def control_state(self) -> WanTI2VControlState | None:
        return self._control_state

    def zero_init_output(self) -> None:
        final_token = self.to_dit[-1]
        if isinstance(final_token, nn.Linear):
            nn.init.zeros_(final_token.weight)
            if final_token.bias is not None:
                nn.init.zeros_(final_token.bias)
        for block in self.action_blocks:
            nn.init.zeros_(block.video_out.weight)
            if block.video_out.bias is not None:
                nn.init.zeros_(block.video_out.bias)
        for block in self.parallel_action_blocks:
            nn.init.zeros_(block.video_out.weight)
            if block.video_out.bias is not None:
                nn.init.zeros_(block.video_out.bias)
        final = self.latent_head[-1]
        if isinstance(final, nn.Conv3d):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        nn.init.zeros_(self.vam_action_head.weight)
        if self.vam_action_head.bias is not None:
            nn.init.zeros_(self.vam_action_head.bias)

    def _param_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        p = next(self.parameters())
        return p.dtype, p.device

    @staticmethod
    def _resize_video(x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        return F.interpolate(x, size=(int(t), int(h), int(w)), mode="trilinear", align_corners=False)

    def _infer_batch_horizon(self, pred_tokens: torch.Tensor) -> tuple[int, int]:
        if pred_tokens.ndim == 4:
            return int(pred_tokens.shape[0]), int(pred_tokens.shape[1])
        if pred_tokens.ndim == 3:
            return int(pred_tokens.shape[0]), int(pred_tokens.shape[1])
        raise ValueError(f"pred_tokens must be [B,T,P,D] or [B,P,D], got {tuple(pred_tokens.shape)}")

    def _tokens_to_bcthw(self, pred_tokens: torch.Tensor, *, latent_t: int, latent_h: int, latent_w: int) -> torch.Tensor:
        b, horizon = self._infer_batch_horizon(pred_tokens)
        if pred_tokens.ndim == 3:
            pred_tokens = pred_tokens[:, None]
            horizon = 1
        token = self.token_proj(pred_tokens.to(device=self._param_dtype_device()[1], dtype=self._param_dtype_device()[0]))
        p = int(token.shape[2])
        g = int(round(math.sqrt(p)))
        if g * g != p:
            g = int(self.cfg.token_grid)
            token = token[:, :, : g * g]
        token = token.view(b, horizon, g, g, self.cfg.hidden).permute(0, 4, 1, 2, 3).contiguous()
        return self._resize_video(token, latent_t, latent_h, latent_w)

    def _hint_to_bcthw(self, x: torch.Tensor | None, *, batch: int, latent_t: int, latent_h: int, latent_w: int, name: str) -> torch.Tensor | None:
        if x is None:
            return None
        dtype, device = self._param_dtype_device()
        x = x.to(device=device, dtype=dtype)
        if x.ndim == 5:
            if x.shape[1] == 1:
                y = x
            else:
                y = x.mean(dim=2, keepdim=False)[:, None]
        elif x.ndim == 4:
            y = x[:, None]
        else:
            raise ValueError(f"{name} must be 4D/5D, got {tuple(x.shape)}")
        if int(y.shape[0]) != int(batch):
            raise ValueError(f"{name} batch mismatch {tuple(y.shape)} vs {batch}")
        return self._resize_video(y, latent_t, latent_h, latent_w)

    def _rgb_to_bcthw(self, x: torch.Tensor | None, *, batch: int, latent_t: int, latent_h: int, latent_w: int, name: str) -> torch.Tensor | None:
        if x is None:
            return None
        dtype, device = self._param_dtype_device()
        x = x.to(device=device, dtype=dtype)
        if x.ndim == 4:
            if int(x.shape[0]) != int(batch) or int(x.shape[1]) != 3:
                raise ValueError(f"{name} must be [B,3,H,W], got {tuple(x.shape)}")
            ctx = self.context_proj(x)
            ctx = ctx[:, :, None].expand(-1, -1, int(latent_t), -1, -1).contiguous()
            return self._resize_video(ctx, latent_t, latent_h, latent_w)
        if x.ndim == 5:
            if int(x.shape[0]) != int(batch):
                raise ValueError(f"{name} batch mismatch {tuple(x.shape)} vs {batch}")
            if int(x.shape[2]) == 3:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            if int(x.shape[1]) != 3:
                raise ValueError(f"{name} must have RGB channels, got {tuple(x.shape)}")
            b, c, t, h, w = x.shape
            flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            feat = self.context_proj(flat).view(b, t, self.cfg.hidden, h, w).permute(0, 2, 1, 3, 4).contiguous()
            return self._resize_video(feat, latent_t, latent_h, latent_w)
        raise ValueError(f"{name} must be 4D/5D, got {tuple(x.shape)}")

    def _point_to_bcthw(self, x: torch.Tensor | None, *, batch: int, latent_t: int, latent_h: int, latent_w: int) -> torch.Tensor | None:
        if x is None:
            return None
        dtype, device = self._param_dtype_device()
        x = x.to(device=device, dtype=dtype)
        if x.ndim == 5 and int(x.shape[-1]) == self.cfg.point_dim:
            y = x.permute(0, 4, 1, 2, 3).contiguous()
        elif x.ndim == 5 and int(x.shape[2]) == self.cfg.point_dim:
            y = x.permute(0, 2, 1, 3, 4).contiguous()
        elif x.ndim == 5 and int(x.shape[1]) == self.cfg.point_dim:
            y = x
        elif x.ndim == 4 and int(x.shape[1]) == self.cfg.point_dim:
            y = x[:, :, None]
        else:
            raise ValueError(f"point must be [B,T,3,H,W] or [B,3,T,H,W], got {tuple(x.shape)}")
        if int(y.shape[0]) != int(batch):
            raise ValueError(f"point batch mismatch {tuple(y.shape)} vs {batch}")
        return self._resize_video(y, latent_t, latent_h, latent_w)

    def _pose_to_bcthw(self, x: torch.Tensor | None, *, batch: int, latent_t: int, latent_h: int, latent_w: int) -> torch.Tensor | None:
        if x is None:
            return None
        dtype, device = self._param_dtype_device()
        x = x.to(device=device, dtype=dtype)
        if x.ndim > 3:
            x = x.flatten(2).mean(dim=-1)
        if x.ndim == 2:
            x = x[:, None]
        if int(x.shape[0]) != int(batch):
            raise ValueError(f"pose batch mismatch {tuple(x.shape)} vs {batch}")
        if int(x.shape[-1]) < self.cfg.pose_dim:
            x = F.pad(x, (0, self.cfg.pose_dim - int(x.shape[-1])))
        x = x[..., : self.cfg.pose_dim]
        feat = self.pose_proj(x).mean(dim=1)
        return feat[:, :, None, None, None].expand(-1, -1, int(latent_t), int(latent_h), int(latent_w)).contiguous()

    def _latents_to_bcthw(self, x: torch.Tensor | None, *, batch: int, latent_t: int, latent_h: int, latent_w: int, name: str) -> torch.Tensor | None:
        if x is None:
            return None
        dtype, device = self._param_dtype_device()
        x = x.to(device=device, dtype=dtype)
        if x.ndim != 5 or int(x.shape[0]) != int(batch) or int(x.shape[1]) != self.cfg.latent_channels:
            raise ValueError(f"{name} must be [B,{self.cfg.latent_channels},T,H,W], got {tuple(x.shape)}")
        return self._resize_video(x, latent_t, latent_h, latent_w)

    def _build_video_action_gate(self, motion: torch.Tensor | None, contact: torch.Tensor | None) -> torch.Tensor | None:
        source = str(getattr(self.cfg, "parallel_action_video_gate_source", "none")).strip().lower()
        if source in {"", "none", "off", "false", "0"}:
            return None
        parts: list[torch.Tensor] = []
        if "motion" in source and motion is not None:
            parts.append(motion.detach() if bool(getattr(self.cfg, "parallel_action_video_gate_detach", True)) else motion)
        if "contact" in source and contact is not None:
            parts.append(contact.detach() if bool(getattr(self.cfg, "parallel_action_video_gate_detach", True)) else contact)
        if not parts:
            return None
        gate = None
        for item in parts:
            item = item.float().abs().mean(dim=1, keepdim=True)
            den = item.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
            item = (item / den).clamp(0.0, 1.0)
            gate = item if gate is None else torch.maximum(gate, item)
        if gate is None:
            return None
        threshold = max(0.0, min(0.95, float(getattr(self.cfg, "parallel_action_video_gate_threshold", 0.05))))
        gate = ((gate - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
        dilate = max(0, int(getattr(self.cfg, "parallel_action_video_gate_dilate", 0)))
        if dilate > 0:
            k = 2 * dilate + 1
            gate = F.max_pool3d(gate, kernel_size=(1, k, k), stride=1, padding=(0, dilate, dilate))
        power = max(1e-6, float(getattr(self.cfg, "parallel_action_video_gate_power", 1.0)))
        if power != 1.0:
            gate = gate.clamp(0.0, 1.0).pow(power)
        gate_min = max(0.0, min(1.0, float(getattr(self.cfg, "parallel_action_video_gate_min", 0.0))))
        if gate_min > 0.0:
            gate = gate * (1.0 - gate_min) + gate_min
        return gate.clamp(0.0, 1.0)

    @staticmethod
    def _gate_to_token_sequence(gate: torch.Tensor | None, tokens: torch.Tensor) -> torch.Tensor | None:
        if gate is None or tokens.ndim != 3:
            return None
        b, length = int(tokens.shape[0]), int(tokens.shape[1])
        if int(gate.shape[0]) != b or length <= 0:
            return None
        gate_f = gate.to(device=tokens.device, dtype=tokens.dtype).clamp(0.0, 1.0)
        if gate_f.ndim == 3:
            flat = gate_f
        else:
            flat = gate_f.flatten(2).transpose(1, 2).contiguous()
        if int(flat.shape[1]) != length:
            flat = F.interpolate(flat.transpose(1, 2).float(), size=length, mode="linear", align_corners=False).transpose(1, 2)
            flat = flat.to(device=tokens.device, dtype=tokens.dtype)
        return flat[:, :length, :1].clamp(0.0, 1.0)

    def build_control_state(
        self,
        *,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor | None,
        motion_hint: torch.Tensor | None = None,
        contact_hint: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
        point: torch.Tensor | None = None,
        pose_geom: torch.Tensor | None = None,
        noisy_latents: torch.Tensor | None = None,
        source_latents: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        action_noisy: torch.Tensor | None = None,
        action_sigma: torch.Tensor | None = None,
        policy_action_cond: torch.Tensor | None = None,
        latent_shape: tuple[int, int, int, int, int] | None = None,
        scale: float = 1.0,
    ) -> WanTI2VControlState:
        dtype, device = self._param_dtype_device()
        pred_tokens = pred_tokens.to(device=device, dtype=dtype)
        batch, _ = self._infer_batch_horizon(pred_tokens)
        if latent_shape is None:
            latent_t = int(pred_tokens.shape[1]) if pred_tokens.ndim == 4 else 1
            latent_h = latent_w = int(self.cfg.token_grid * self.cfg.patch_size[1])
            latent_shape = (batch, self.cfg.latent_channels, latent_t, latent_h, latent_w)
        latent_t, latent_h, latent_w = int(latent_shape[2]), int(latent_shape[3]), int(latent_shape[4])

        features = self._tokens_to_bcthw(pred_tokens, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
        motion_gate_hint = None
        contact_gate_hint = None
        if self.cfg.use_depth:
            d = self._hint_to_bcthw(depth, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, name="depth")
            if d is not None:
                features = features + self.depth_proj(d)
        if self.cfg.use_motion:
            m = self._hint_to_bcthw(motion_hint, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, name="motion_hint")
            if m is not None:
                motion_gate_hint = m
                features = features + self.motion_proj(m)
        if self.cfg.use_contact:
            c = self._hint_to_bcthw(contact_hint, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, name="contact_hint")
            if c is not None:
                contact_gate_hint = c
                features = features + self.contact_proj(c)
        if self.cfg.use_context:
            ctx = self._rgb_to_bcthw(context_rgb, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, name="context_rgb")
            if ctx is not None:
                features = features + ctx
        if self.cfg.use_point:
            p = self._point_to_bcthw(point, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
            if p is not None:
                features = features + self.point_proj(p)
        if self.cfg.use_pose:
            pose = self._pose_to_bcthw(pose_geom, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w)
            if pose is not None:
                features = features + pose
        if self.cfg.use_task and task_emb is not None:
            task = task_emb.to(device=device, dtype=dtype)
            if task.ndim == 3:
                task = task.mean(dim=1)
            task_feat = self.task_proj(task)
            features = features + task_feat[:, :, None, None, None]
        action_tokens = None
        video_action_tokens = None
        time_emb = None
        if self.cfg.use_action and action_cond is not None:
            action = action_cond.to(device=device, dtype=dtype)
            if action.ndim == 2:
                action = action[:, None]
            action_feat = self.action_proj(action).mean(dim=1)
            features = features + action_feat[:, :, None, None, None] * float(self.cfg.action_residual_scale)
            action_tokens = self.action_token_in(action)
            if int(action_tokens.shape[-1]) != int(self.cfg.parallel_action_video_hidden):
                # The parallel blocks default to action_token_hidden.  Keep the
                # state width explicit if a legacy checkpoint used a different
                # value and parallel mode was later disabled.
                action_tokens = action_tokens[..., : int(self.cfg.parallel_action_video_hidden)]
            action_tokens = action_tokens + self._action_position_encoding(
                int(action_tokens.shape[1]),
                int(action_tokens.shape[2]),
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )
        if bool(self.cfg.use_vam_action_expert):
            action_input = action_noisy if action_noisy is not None else action_cond
            if action_input is None:
                action_input = torch.zeros(batch, 1, self.cfg.action_dim, device=device, dtype=dtype)
            action_input = action_input.to(device=device, dtype=dtype)
            if action_input.ndim == 2:
                action_input = action_input[:, None]
            action_tokens = self.vam_action_proj_in(action_input)
            if policy_action_cond is not None and float(self.cfg.vam_action_policy_cond_scale) != 0.0:
                policy = policy_action_cond.to(device=device, dtype=dtype)
                if policy.ndim == 2:
                    policy = policy[:, None]
                if int(policy.shape[1]) != int(action_input.shape[1]):
                    if int(policy.shape[1]) > int(action_input.shape[1]):
                        policy = policy[:, : int(action_input.shape[1])]
                    else:
                        pad = policy[:, -1:].expand(-1, int(action_input.shape[1]) - int(policy.shape[1]), -1)
                        policy = torch.cat([policy, pad], dim=1)
                action_tokens = action_tokens + self.vam_policy_proj(policy) * float(self.cfg.vam_action_policy_cond_scale)
            action_tokens = action_tokens + self._action_position_encoding(
                int(action_tokens.shape[1]),
                int(action_tokens.shape[2]),
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )
            sigma_for_action = action_sigma if action_sigma is not None else sigma
            if sigma_for_action is None:
                sigma_for_action = torch.ones(batch, device=device, dtype=dtype)
            sigma_for_action = sigma_for_action.to(device=device, dtype=dtype).reshape(batch)
            time_emb = self.vam_action_time_embedding(
                self._sinusoidal_embedding(int(self.cfg.vam_action_freq_dim), sigma_for_action * 1000.0)
                .to(device=device, dtype=dtype)
            )
            action_tokens = action_tokens + time_emb[:, None]
            video_action_source = str(getattr(self.cfg, "vam_video_action_source", "clean")).strip().lower()
            if bool(self.cfg.vam_video_use_clean_action) and video_action_source == "noisy":
                video_action_source = "clean"
            video_action = None
            if video_action_source in {"policy", "teacher"} and policy_action_cond is not None:
                video_action = policy_action_cond.to(device=device, dtype=dtype)
            elif video_action_source in {"policy_blend", "teacher_blend"} and action_cond is not None:
                clean_action = action_cond.to(device=device, dtype=dtype)
                if clean_action.ndim == 2:
                    clean_action = clean_action[:, None]
                if policy_action_cond is not None:
                    policy = policy_action_cond.to(device=device, dtype=dtype)
                    if policy.ndim == 2:
                        policy = policy[:, None]
                    if int(policy.shape[1]) != int(clean_action.shape[1]):
                        if int(policy.shape[1]) > int(clean_action.shape[1]):
                            policy = policy[:, : int(clean_action.shape[1])]
                        else:
                            pad = policy[:, -1:].expand(-1, int(clean_action.shape[1]) - int(policy.shape[1]), -1)
                            policy = torch.cat([policy, pad], dim=1)
                    blend = max(0.0, min(1.0, float(getattr(self.cfg, "vam_video_policy_blend", 0.5))))
                    video_action = clean_action * (1.0 - blend) + policy * blend
                else:
                    video_action = clean_action
            elif bool(self.cfg.vam_video_use_clean_action) and action_cond is not None:
                video_action = action_cond.to(device=device, dtype=dtype)
            if video_action is not None:
                if video_action.ndim == 2:
                    video_action = video_action[:, None]
                video_action_tokens = self.vam_action_proj_in(video_action)
                video_action_tokens = video_action_tokens + self._action_position_encoding(
                    int(video_action_tokens.shape[1]),
                    int(video_action_tokens.shape[2]),
                    device=video_action_tokens.device,
                    dtype=video_action_tokens.dtype,
                )
            else:
                video_action_tokens = action_tokens
        if self.cfg.use_noisy_latents:
            noisy = self._latents_to_bcthw(noisy_latents, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, name="noisy_latents")
            if noisy is not None:
                features = features + self.noisy_proj(noisy)
        if self.cfg.use_source_latents:
            src = self._latents_to_bcthw(source_latents, batch=batch, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, name="source_latents")
            if src is not None:
                features = features + self.source_proj(src)
        if self.cfg.use_sigma_embed and sigma is not None:
            sig = sigma.to(device=device, dtype=dtype).reshape(batch, 1)
            sig_feat = self.sigma_proj(sig)
            features = features + sig_feat[:, :, None, None, None]

        features = self.fuse(features)
        video_action_gate = self._build_video_action_gate(motion_gate_hint, contact_gate_hint)
        return WanTI2VControlState(
            features=features,
            scale=float(scale),
            latent_shape=tuple(int(v) for v in latent_shape),
            action_tokens=action_tokens,
            video_action_tokens=video_action_tokens,
            video_action_gate_tokens=self._gate_to_token_sequence(video_action_gate, features.flatten(2).transpose(1, 2)),
            action_time_emb=time_emb if bool(self.cfg.use_vam_action_expert) else None,
        )

    def prepare_controls(self, *args: Any, **kwargs: Any) -> WanTI2VControlState:
        state = self.build_control_state(*args, **kwargs)
        self._control_state = state
        return state

    def set_control_state(self, state: WanTI2VControlState) -> None:
        self._control_state = state

    def clear_control_state(self) -> None:
        self._control_state = None

    def reset_action_stream(self) -> None:
        if self._control_state is not None:
            self._control_state.action_stream = None
            self._control_state.video_action_stream = None

    def action_context_tokens(self, action_cond: torch.Tensor | None) -> torch.Tensor | None:
        """Project action chunks into Wan text-context-width tokens.

        Wan video blocks already cross-attend every layer to the text context.
        Appending learned action tokens to that native context gives actions a
        direct path into video denoising instead of relying only on external
        residual hooks after each block.
        """

        if not bool(self.cfg.use_action_context_tokens) or action_cond is None:
            return None
        dtype, device = self._param_dtype_device()
        action = action_cond.to(device=device, dtype=dtype)
        if action.ndim == 2:
            action = action[:, None]
        tokens = self.action_context_in(action)
        tokens = tokens + self._action_position_encoding(
            int(tokens.shape[1]),
            int(tokens.shape[2]),
            device=tokens.device,
            dtype=tokens.dtype,
        ) * float(self.cfg.action_context_pos_scale)
        return tokens

    def _layer_gain(self, layer_idx: int) -> float:
        n = max(1, int(self.cfg.num_layers) - 1)
        x = max(0.0, min(1.0, float(layer_idx) / float(n)))
        x = x ** float(self.cfg.layer_gain_power)
        return float(self.cfg.layer_gain_start) * (1.0 - x) + float(self.cfg.layer_gain_end) * x

    def _sequence_features(self, token_len: int, latent_shape: tuple[int, int, int, int, int] | None) -> torch.Tensor:
        state = self._control_state
        if state is None:
            raise RuntimeError("Wan control state is not prepared")
        b, h = int(state.features.shape[0]), int(state.features.shape[1])
        if latent_shape is None:
            latent_shape = state.latent_shape
        if latent_shape is None:
            side = int(round(math.sqrt(max(1, token_len))))
            grid = (1, side, max(1, token_len // max(1, side)))
        else:
            pt, ph, pw = self.cfg.patch_size
            grid = (
                max(1, int(math.ceil(int(latent_shape[2]) / int(pt)))),
                max(1, int(math.ceil(int(latent_shape[3]) / int(ph)))),
                max(1, int(math.ceil(int(latent_shape[4]) / int(pw)))),
            )
        x = F.interpolate(state.features, size=grid, mode="trilinear", align_corners=False)
        x = x.permute(0, 2, 3, 4, 1).reshape(b, -1, h)
        if int(x.shape[1]) < int(token_len):
            pad = x.new_zeros(b, int(token_len) - int(x.shape[1]), h)
            x = torch.cat([x, pad], dim=1)
        return x[:, : int(token_len)]

    def token_residual(self, tokens: torch.Tensor, layer_idx: int, *, latent_shape: tuple[int, int, int, int, int] | None = None) -> torch.Tensor:
        state = self._control_state
        if state is None:
            return torch.zeros_like(tokens)
        seq = self._sequence_features(int(tokens.shape[1]), latent_shape).to(device=tokens.device, dtype=tokens.dtype)
        residual = self.to_dit(seq)
        if (
            self.cfg.use_action_token_block
            and not bool(self.cfg.use_parallel_action_video_blocks)
            and state.action_tokens is not None
            and int(layer_idx) < len(self.action_blocks)
        ):
            stream = state.action_stream
            if stream is None:
                stream = state.action_tokens.to(device=tokens.device, dtype=tokens.dtype)
            stream, action_delta = self.action_blocks[int(layer_idx)](stream, tokens)
            state.action_stream = stream
            residual = residual + action_delta * float(self.cfg.action_token_block_scale)
        return residual * float(state.scale) * self._layer_gain(int(layer_idx))

    def parallel_action_video_delta(self, tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        state = self._control_state
        if (
            state is None
            or (state.video_action_tokens is None and state.action_tokens is None)
            or not bool(self.cfg.use_parallel_action_video_blocks)
            or int(layer_idx) >= len(self.parallel_action_blocks)
        ):
            return torch.zeros_like(tokens)
        video_tokens = state.video_action_tokens if state.video_action_tokens is not None else state.action_tokens
        if video_tokens is None:
            return torch.zeros_like(tokens)
        stream = state.video_action_stream
        if stream is None:
            stream = video_tokens.to(device=tokens.device, dtype=tokens.dtype)
        block = self.parallel_action_blocks[int(layer_idx)]
        stream, video_delta = block(stream, tokens)
        state.video_action_stream = stream
        if state.action_tokens is video_tokens:
            state.action_stream = stream
        elif bool(self.cfg.vam_update_noisy_action_stream) and state.action_tokens is not None:
            flow_stream = state.action_stream
            if flow_stream is None:
                flow_stream = state.action_tokens.to(device=tokens.device, dtype=tokens.dtype)
            flow_stream, _ = block(flow_stream, tokens)
            state.action_stream = flow_stream
        scale = float(state.scale) * float(self.cfg.parallel_action_video_scale) * self._layer_gain(int(layer_idx))
        if bool(self.cfg.use_vam_action_expert):
            scale *= float(self.cfg.vam_action_video_delta_scale)
        gate = state.video_action_gate_tokens
        if gate is not None:
            gate = self._gate_to_token_sequence(gate, video_delta)
            if gate is not None:
                video_delta = video_delta * gate.to(device=video_delta.device, dtype=video_delta.dtype)
        return video_delta * scale

    def action_velocity_prediction(self) -> torch.Tensor | None:
        state = self._control_state
        if state is None or state.action_tokens is None or not bool(self.cfg.use_vam_action_expert):
            return None
        stream = state.action_stream
        if stream is None:
            stream = state.action_tokens
        if state.action_time_emb is None:
            mod = self.vam_action_head_modulation.to(device=stream.device, dtype=stream.dtype)
            shift, scale = mod[:, :1], mod[:, 1:]
        else:
            emb = self.vam_action_time_projection(state.action_time_emb.to(device=stream.device, dtype=stream.dtype))
            shift, scale = emb[:, None].chunk(2, dim=-1)
            mod = self.vam_action_head_modulation.to(device=stream.device, dtype=stream.dtype)
            shift = shift + mod[:, :1]
            scale = scale + mod[:, 1:]
        pred = self.vam_action_head(self.vam_action_head_norm(stream) * (1.0 + scale) + shift)
        state.action_pred_velocity = pred
        return pred

    def latent_residual(self, target_latents: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        state = self._control_state
        if state is None:
            if isinstance(target_latents, torch.Tensor):
                return torch.zeros_like(target_latents)
            dtype, device = self._param_dtype_device()
            return torch.zeros(tuple(int(v) for v in target_latents), dtype=dtype, device=device)
        if isinstance(target_latents, torch.Tensor):
            shape = tuple(int(v) for v in target_latents.shape)
            dtype, device = target_latents.dtype, target_latents.device
        else:
            shape = tuple(int(v) for v in target_latents)
            dtype, device = self._param_dtype_device()
        x = self.latent_head(state.features)
        x = F.interpolate(x, size=shape[2:], mode="trilinear", align_corners=False)
        return x.to(device=device, dtype=dtype) * float(state.scale) * float(self.cfg.action_latent_scale)

    def source_latent_delta(self, source_latents: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        return self.latent_residual(source_latents)


class WanTI2VControlInjector:
    """Forward hooks that add adapter residuals to WanAttentionBlock outputs."""

    def __init__(self, transformer: nn.Module, adapter: WanTI2VControlAdapter):
        self.transformer = transformer
        self.adapter = adapter
        self._handles: list[Any] = []
        self._original_forwards: list[tuple[nn.Module, Callable[..., Any]]] = []
        self._installed = False
        self._latent_shape: tuple[int, int, int, int, int] | None = None

    @staticmethod
    def _iter_blocks(transformer: nn.Module) -> list[nn.Module]:
        blocks = getattr(transformer, "blocks", None)
        if isinstance(blocks, nn.ModuleList):
            return list(blocks)
        if isinstance(blocks, nn.Sequential):
            return list(blocks)
        if isinstance(blocks, (list, tuple)):
            return [b for b in blocks if isinstance(b, nn.Module)]
        return []

    @staticmethod
    def _extract_tensor(output: Any) -> tuple[torch.Tensor | None, Callable[[torch.Tensor], Any]]:
        if torch.is_tensor(output):
            return output, lambda x: x
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return output[0], lambda x: (x, *output[1:])
        if isinstance(output, list) and output and torch.is_tensor(output[0]):
            return output[0], lambda x: [x, *output[1:]]
        return None, lambda x: output

    def _make_block_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            x, rebuild = self._extract_tensor(output)
            if x is None or self.adapter.control_state is None:
                return output
            residual = self.adapter.token_residual(x, layer_idx, latent_shape=self._latent_shape)
            return rebuild(x + residual.to(device=x.device, dtype=x.dtype))

        return hook

    def _make_transformer_pre_hook(self):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...]) -> None:
            self.adapter.reset_action_stream()

        return hook

    def _make_parallel_forward(self, layer_idx: int):
        injector = self

        def forward(block: nn.Module, x: torch.Tensor, e: torch.Tensor, seq_lens: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, context: torch.Tensor, context_lens: torch.Tensor | None):
            assert e.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.float32):
                emod = (block.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            assert emod[0].dtype == torch.float32

            y = block.self_attn(
                block.norm1(x).float() * (1 + emod[1].squeeze(2)) + emod[0].squeeze(2),
                seq_lens,
                grid_sizes,
                freqs,
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * emod[2].squeeze(2)

            if injector.adapter.control_state is not None:
                control_delta = injector.adapter.token_residual(x, layer_idx, latent_shape=injector._latent_shape)
                action_delta = injector.adapter.parallel_action_video_delta(x, layer_idx)
                x = x + control_delta.to(device=x.device, dtype=x.dtype) + action_delta.to(device=x.device, dtype=x.dtype)

            x = x + block.cross_attn(block.norm3(x), context, context_lens)
            y = block.ffn(block.norm2(x).float() * (1 + emod[4].squeeze(2)) + emod[3].squeeze(2))
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * emod[5].squeeze(2)
            return x

        return forward

    def install(self) -> None:
        if self._installed:
            return
        blocks = self._iter_blocks(self.transformer)
        if not blocks:
            raise RuntimeError("Could not locate Wan transformer blocks; expected transformer.blocks")
        self._handles.append(self.transformer.register_forward_pre_hook(self._make_transformer_pre_hook()))
        if bool(self.adapter.cfg.use_parallel_action_video_blocks):
            for idx, block in enumerate(blocks):
                original = block.forward
                self._original_forwards.append((block, original))
                block.forward = MethodType(self._make_parallel_forward(idx), block)
        else:
            for idx, block in enumerate(blocks):
                self._handles.append(block.register_forward_hook(self._make_block_hook(idx)))
        self._installed = True

    def remove(self) -> None:
        while self._original_forwards:
            block, original = self._original_forwards.pop()
            block.forward = original
        while self._handles:
            self._handles.pop().remove()
        self._installed = False

    @contextmanager
    def use_controls(self, *args: Any, latent_shape: tuple[int, int, int, int, int] | None = None, **kwargs: Any):
        state = self.adapter.prepare_controls(*args, latent_shape=latent_shape, **kwargs)
        self._latent_shape = state.latent_shape
        self.install()
        try:
            yield state
        finally:
            self.adapter.clear_control_state()
            self.remove()

    @contextmanager
    def use_control_state(self, state: WanTI2VControlState, *, latent_shape: tuple[int, int, int, int, int] | None = None):
        self.adapter.set_control_state(state)
        self._latent_shape = latent_shape or state.latent_shape
        self.install()
        try:
            yield state
        finally:
            self.adapter.clear_control_state()
            self.remove()


def _cfg_from_payload(value: Any, *, ckpt_path: Path) -> WanTI2VControlConfig:
    if value is None:
        return WanTI2VControlConfig()
    if isinstance(value, WanTI2VControlConfig):
        return value
    if isinstance(value, dict):
        allowed = WanTI2VControlConfig.__dataclass_fields__.keys()
        return WanTI2VControlConfig(**{k: v for k, v in value.items() if k in allowed})
    raise RuntimeError(f"{ckpt_path} has invalid Wan control cfg payload {type(value)!r}")


def load_wan_ti2v_control_checkpoint(path: str | Path, device: str | torch.device | None = None) -> tuple[WanTI2VControlAdapter, dict[str, Any]]:
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ckpt_path} must contain a dict payload")
    if payload.get("kind") not in {CONTROL_CHECKPOINT_KIND, None}:
        raise RuntimeError(f"{ckpt_path} kind={payload.get('kind')!r} is not a Wan TI2V control checkpoint")
    cfg = _cfg_from_payload(payload.get("cfg") or payload.get("adapter_cfg"), ckpt_path=ckpt_path)
    adapter = WanTI2VControlAdapter(cfg)
    state = payload.get("state_dict") or payload.get("adapter") or payload.get("wan_control_adapter")
    if state is None:
        raise RuntimeError(f"{ckpt_path} missing adapter state_dict")
    report = adapter.load_state_dict(state, strict=False)
    if device is not None:
        adapter.to(device)
    return adapter, {
        "path": str(ckpt_path),
        "missing": list(report.missing_keys),
        "unexpected": list(report.unexpected_keys),
        "step": payload.get("step"),
    }


def save_wan_ti2v_control_checkpoint(
    path: str | Path,
    adapter: WanTI2VControlAdapter,
    *,
    wm_ckpt: str | Path | None = None,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    target = adapter.module if hasattr(adapter, "module") else adapter
    payload = {
        "kind": CONTROL_CHECKPOINT_KIND,
        "cfg": asdict(target.cfg),
        "state_dict": target.state_dict(),
        "wm_ckpt": str(wm_ckpt) if wm_ckpt is not None else None,
        "step": step,
        "metrics": metrics or {},
    }
    tmp = path.with_name("." + path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
