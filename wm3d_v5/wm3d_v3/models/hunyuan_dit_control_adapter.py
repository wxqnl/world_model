"""Control adapter that injects wm3d controls into Hunyuan DiT image tokens.

The adapter is intentionally isolated from the core world-model stages. It is
zero initialized, so installing it on a frozen Hunyuan transformer is an exact
no-op until a control checkpoint is trained and loaded.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import types
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _activation_checkpoint


CONTROL_CHECKPOINT_KIND = "hunyuan_dit_control_adapter_v1"


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


class _ParallelActionDiTBlock(nn.Module):
    """Small action-stream block: action self-attn, action->video cross-attn, then video residual.

    This mirrors the useful part of tau-0-WM's VAM action branch inside the
    adapter: action tokens evolve as their own stream and read current video
    hidden states at every Hunyuan layer.
    """

    def __init__(self, action_hidden: int, video_hidden: int, heads: int, mlp_mult: float = 2.0):
        super().__init__()
        self.action_hidden = int(action_hidden)
        self.video_hidden = int(video_hidden)
        self.heads = max(1, int(heads))
        if self.action_hidden % self.heads != 0:
            self.action_hidden = int(math.ceil(self.action_hidden / self.heads) * self.heads)
        mlp_hidden = max(self.action_hidden, int(round(self.action_hidden * float(mlp_mult))))

        self.action_self_norm = nn.LayerNorm(self.action_hidden)
        self.action_self_qkv = nn.Linear(self.action_hidden, 3 * self.action_hidden)
        self.action_self_out = nn.Linear(self.action_hidden, self.action_hidden)

        self.action_cross_norm = nn.LayerNorm(self.action_hidden)
        self.video_cross_norm = nn.LayerNorm(self.video_hidden)
        self.action_cross_q = nn.Linear(self.action_hidden, self.action_hidden)
        self.video_cross_k = nn.Linear(self.video_hidden, self.action_hidden)
        self.video_cross_v = nn.Linear(self.video_hidden, self.action_hidden)
        self.action_cross_out = nn.Linear(self.action_hidden, self.action_hidden)

        self.action_mlp_norm = nn.LayerNorm(self.action_hidden)
        self.action_mlp = nn.Sequential(
            nn.Linear(self.action_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.action_hidden),
        )

        self.video_res_norm = nn.LayerNorm(self.video_hidden)
        self.action_res_norm = nn.LayerNorm(self.action_hidden)
        self.video_res_q = nn.Linear(self.video_hidden, self.action_hidden)
        self.action_res_k = nn.Linear(self.action_hidden, self.action_hidden)
        self.action_res_v = nn.Linear(self.action_hidden, self.action_hidden)
        self.video_res_out = nn.Linear(self.action_hidden, self.video_hidden)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        head_dim = self.action_hidden // self.heads
        return x.view(int(x.shape[0]), int(x.shape[1]), self.heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).reshape(int(x.shape[0]), int(x.shape[2]), self.action_hidden)

    def forward(self, action_tokens: torch.Tensor, video_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.action_self_qkv(self.action_self_norm(action_tokens)).chunk(3, dim=-1)
        action_tokens = action_tokens + self.action_self_out(
            self._merge_heads(
                F.scaled_dot_product_attention(
                    self._split_heads(q),
                    self._split_heads(k),
                    self._split_heads(v),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )

        aq = self.action_cross_q(self.action_cross_norm(action_tokens))
        vk = self.video_cross_k(self.video_cross_norm(video_tokens))
        vv = self.video_cross_v(self.video_cross_norm(video_tokens))
        action_tokens = action_tokens + self.action_cross_out(
            self._merge_heads(
                F.scaled_dot_product_attention(
                    self._split_heads(aq),
                    self._split_heads(vk),
                    self._split_heads(vv),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )
        action_tokens = action_tokens + self.action_mlp(self.action_mlp_norm(action_tokens))

        vq = self.video_res_q(self.video_res_norm(video_tokens))
        ak = self.action_res_k(self.action_res_norm(action_tokens))
        av = self.action_res_v(self.action_res_norm(action_tokens))
        video_residual = self.video_res_out(
            self._merge_heads(
                F.scaled_dot_product_attention(
                    self._split_heads(vq),
                    self._split_heads(ak),
                    self._split_heads(av),
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
        )
        return action_tokens, video_residual


@dataclass
class HunyuanDiTControlConfig:
    token_dim: int = 2048
    token_grid: int = 8
    latent_channels: int = 16
    hidden: int = 192
    dit_hidden: int = 3072
    action_dim: int = 7
    task_dim: int = 2048
    double_blocks: int = 20
    single_blocks: int = 40
    use_depth: bool = True
    use_motion: bool = True
    use_contact: bool = True
    use_rough: bool = True
    use_context: bool = True
    use_action: bool = True
    use_task: bool = True
    use_point: bool = True
    use_pose: bool = True
    use_rgb_features: bool = False
    rgb_feature_dim: int = 0
    rgb_feature_gain: float = 1.0
    use_noisy_latents: bool = False
    use_source_latents: bool = False
    use_sigma_embed: bool = False
    point_dim: int = 3
    pose_dim: int = 9
    action_residual_scale: float = 1.0
    action_token_scale: float = 1.0
    action_direct_scale: float = 0.0
    action_latent_scale: float = 0.0
    use_action_cross_attn: bool = False
    action_cross_attn_scale: float = 0.0
    action_cross_attn_hidden: int = 192
    action_cross_attn_heads: int = 4
    action_cross_attn_time_scale: float = 1.0
    use_temporal_action_summary: bool = False
    temporal_action_summary_scale: float = 1.0
    use_parallel_action_dit: bool = False
    parallel_action_dit_scale: float = 0.0
    parallel_action_dit_hidden: int = 256
    parallel_action_dit_heads: int = 4
    parallel_action_dit_mlp_mult: float = 2.0
    native_parallel_action_forward: bool = False
    use_block_action_film: bool = False
    block_action_film_scale: float = 1.0
    block_action_film_hidden: int = 192
    double_control_gain_start: float = 1.0
    double_control_gain_end: float = 1.0
    double_control_gain_power: float = 1.0
    single_control_gain_start: float = 1.0
    single_control_gain_end: float = 1.0
    single_control_gain_power: float = 1.0
    cfg_branch: str = "conditional_only"


@dataclass
class HunyuanDiTControlState:
    features: torch.Tensor
    scale: float = 1.0
    img_token_len: int | None = None
    action_film: torch.Tensor | None = None
    action_latent: torch.Tensor | None = None
    action_tokens: torch.Tensor | None = None
    parallel_action_stream: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])


class HunyuanDiTControlAdapter(nn.Module):
    """Encode wm3d controls into residuals for Hunyuan DiT image tokens."""

    def __init__(self, cfg: HunyuanDiTControlConfig | None = None):
        super().__init__()
        self.cfg = cfg or HunyuanDiTControlConfig()
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
        self.rough_proj = nn.Sequential(nn.Conv3d(3, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.context_proj = nn.Sequential(nn.Conv2d(3, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.action_proj = nn.Sequential(nn.Linear(self.cfg.action_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.action_token_fuse = nn.Sequential(
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.action_direct_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.action_latent_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, self.cfg.latent_channels),
        )
        cross_h = max(16, int(self.cfg.action_cross_attn_hidden))
        cross_heads = max(1, int(self.cfg.action_cross_attn_heads))
        if cross_h % cross_heads != 0:
            cross_h = int(math.ceil(cross_h / cross_heads) * cross_heads)
        self.action_cross_hidden = cross_h
        self.action_cross_heads = cross_heads
        self.action_cross_token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, cross_h),
            nn.SiLU(inplace=True),
            nn.Linear(cross_h, cross_h),
        )
        self.action_cross_time_proj = nn.Sequential(nn.Linear(1, cross_h), nn.SiLU(inplace=True), nn.Linear(cross_h, cross_h))
        self.action_cross_q = nn.Linear(self.cfg.dit_hidden, cross_h)
        self.action_cross_k = nn.Linear(cross_h, cross_h)
        self.action_cross_v = nn.Linear(cross_h, cross_h)
        self.action_cross_out = nn.Linear(cross_h, self.cfg.dit_hidden)
        par_h = max(16, int(self.cfg.parallel_action_dit_hidden))
        par_heads = max(1, int(self.cfg.parallel_action_dit_heads))
        if par_h % par_heads != 0:
            par_h = int(math.ceil(par_h / par_heads) * par_heads)
        self.parallel_action_hidden = par_h
        self.parallel_action_heads = par_heads
        self.parallel_action_token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, par_h),
            nn.SiLU(inplace=True),
            nn.Linear(par_h, par_h),
        )
        self.parallel_action_time_proj = nn.Sequential(
            nn.Linear(1, par_h),
            nn.SiLU(inplace=True),
            nn.Linear(par_h, par_h),
        )
        self.double_parallel_action_blocks = nn.ModuleList(
            [
                _ParallelActionDiTBlock(
                    par_h,
                    self.cfg.dit_hidden,
                    par_heads,
                    mlp_mult=float(self.cfg.parallel_action_dit_mlp_mult),
                )
                for _ in range(self.cfg.double_blocks)
            ]
        )
        self.single_parallel_action_blocks = nn.ModuleList(
            [
                _ParallelActionDiTBlock(
                    par_h,
                    self.cfg.dit_hidden,
                    par_heads,
                    mlp_mult=float(self.cfg.parallel_action_dit_mlp_mult),
                )
                for _ in range(self.cfg.single_blocks)
            ]
        )
        film_h = max(16, int(self.cfg.block_action_film_hidden))
        self.action_summary_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, film_h),
            nn.SiLU(inplace=True),
            nn.Linear(film_h, h),
            nn.SiLU(inplace=True),
        )
        self.temporal_action_summary_token_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.action_dim),
            nn.Linear(self.cfg.action_dim, film_h),
            nn.SiLU(inplace=True),
            nn.Linear(film_h, film_h),
        )
        self.temporal_action_summary_time_proj = nn.Sequential(
            nn.Linear(1, film_h),
            nn.SiLU(inplace=True),
            nn.Linear(film_h, film_h),
        )
        self.temporal_action_summary_out = nn.Sequential(
            nn.LayerNorm(film_h),
            nn.Linear(film_h, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.task_proj = nn.Sequential(nn.LayerNorm(self.cfg.task_dim), nn.Linear(self.cfg.task_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.point_proj = nn.Sequential(nn.Conv3d(self.cfg.point_dim, h, 3, padding=1), nn.GroupNorm(_norm_groups(h), h), nn.SiLU(inplace=True))
        self.pose_proj = nn.Sequential(nn.Linear(self.cfg.pose_dim, h), nn.SiLU(inplace=True), nn.Linear(h, h))
        self.rgb_feature_proj = (
            self._make_rgb_feature_proj(int(self.cfg.rgb_feature_dim))
            if bool(self.cfg.use_rgb_features) and int(self.cfg.rgb_feature_dim) > 0
            else None
        )
        self.noisy_latent_proj = (
            nn.Sequential(
                nn.Conv3d(self.cfg.latent_channels, h, kernel_size=3, padding=1),
                nn.GroupNorm(_norm_groups(h), h),
                nn.SiLU(inplace=True),
                nn.Conv3d(h, h, kernel_size=3, padding=1),
            )
            if bool(self.cfg.use_noisy_latents)
            else None
        )
        self.source_latent_condition_proj = (
            nn.Sequential(
                nn.Conv3d(self.cfg.latent_channels, h, kernel_size=3, padding=1),
                nn.GroupNorm(_norm_groups(h), h),
                nn.SiLU(inplace=True),
                nn.Conv3d(h, h, kernel_size=3, padding=1),
            )
            if bool(self.cfg.use_source_latents)
            else None
        )
        self.sigma_proj = (
            nn.Sequential(nn.Linear(1, h), nn.SiLU(inplace=True), nn.Linear(h, h))
            if bool(self.cfg.use_sigma_embed)
            else None
        )
        self.fuse = nn.Sequential(_Conv3dBlock(h), _Conv3dBlock(h))

        self.double_projections = nn.ModuleList([nn.Linear(h, self.cfg.dit_hidden) for _ in range(self.cfg.double_blocks)])
        self.single_projections = nn.ModuleList([nn.Linear(h, self.cfg.dit_hidden) for _ in range(self.cfg.single_blocks)])
        self.double_action_film = nn.ModuleList([nn.Linear(h, 2 * self.cfg.dit_hidden) for _ in range(self.cfg.double_blocks)])
        self.single_action_film = nn.ModuleList([nn.Linear(h, 2 * self.cfg.dit_hidden) for _ in range(self.cfg.single_blocks)])
        self.double_gates = nn.Parameter(torch.zeros(self.cfg.double_blocks))
        self.single_gates = nn.Parameter(torch.zeros(self.cfg.single_blocks))
        self.double_action_cross_gates = nn.Parameter(torch.ones(self.cfg.double_blocks))
        self.single_action_cross_gates = nn.Parameter(torch.ones(self.cfg.single_blocks))
        self.double_parallel_action_gates = nn.Parameter(torch.ones(self.cfg.double_blocks))
        self.single_parallel_action_gates = nn.Parameter(torch.ones(self.cfg.single_blocks))
        self.latent_projection = nn.Conv3d(h, self.cfg.latent_channels, kernel_size=1)
        self.source_latent_projection = nn.Conv3d(h, self.cfg.latent_channels, kernel_size=1)
        self._control_state: HunyuanDiTControlState | None = None
        self.zero_init_output()

    @property
    def control_state(self) -> HunyuanDiTControlState | None:
        return self._control_state

    def zero_init_output(self) -> None:
        """Make all residual branches an exact no-op at initialization."""
        for layer in [*self.double_projections, *self.single_projections]:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for layer in [*self.double_action_film, *self.single_action_film]:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.ones_(self.double_gates)
        nn.init.ones_(self.single_gates)
        nn.init.zeros_(self.latent_projection.weight)
        if self.latent_projection.bias is not None:
            nn.init.zeros_(self.latent_projection.bias)
        nn.init.zeros_(self.source_latent_projection.weight)
        if self.source_latent_projection.bias is not None:
            nn.init.zeros_(self.source_latent_projection.bias)
        final = self.action_token_fuse[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        final = self.action_direct_proj[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        final = self.action_latent_proj[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        nn.init.zeros_(self.action_cross_out.weight)
        if self.action_cross_out.bias is not None:
            nn.init.zeros_(self.action_cross_out.bias)
        for block in [*self.double_parallel_action_blocks, *self.single_parallel_action_blocks]:
            nn.init.zeros_(block.video_res_out.weight)
            if block.video_res_out.bias is not None:
                nn.init.zeros_(block.video_res_out.bias)
        final = self.temporal_action_summary_out[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            if final.bias is not None:
                nn.init.zeros_(final.bias)
        if self.noisy_latent_proj is not None:
            final_conv = self.noisy_latent_proj[-1]
            if isinstance(final_conv, nn.Conv3d):
                nn.init.zeros_(final_conv.weight)
                if final_conv.bias is not None:
                    nn.init.zeros_(final_conv.bias)
        if self.source_latent_condition_proj is not None:
            final_conv = self.source_latent_condition_proj[-1]
            if isinstance(final_conv, nn.Conv3d):
                nn.init.zeros_(final_conv.weight)
                if final_conv.bias is not None:
                    nn.init.zeros_(final_conv.bias)
        if self.sigma_proj is not None:
            final_linear = self.sigma_proj[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                if final_linear.bias is not None:
                    nn.init.zeros_(final_linear.bias)

    def _make_rgb_feature_proj(self, in_channels: int) -> nn.Sequential:
        h = int(self.cfg.hidden)
        return nn.Sequential(
            nn.Conv3d(int(in_channels), h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )

    def enable_rgb_features(self, *, dim: int, gain: float = 1.0) -> None:
        dim = int(dim)
        if dim <= 0:
            raise ValueError(f"rgb feature dim must be positive, got {dim}")
        self.cfg.use_rgb_features = True
        self.cfg.rgb_feature_dim = dim
        self.cfg.rgb_feature_gain = float(gain)
        current_dim = None
        if self.rgb_feature_proj is not None and isinstance(self.rgb_feature_proj[0], nn.Conv3d):
            current_dim = int(self.rgb_feature_proj[0].in_channels)
        if self.rgb_feature_proj is None or current_dim != dim:
            dtype, device = self._param_dtype_device()
            self.rgb_feature_proj = self._make_rgb_feature_proj(dim).to(device=device, dtype=dtype)

    @staticmethod
    def _grid_size(patches: int) -> int:
        grid = int(math.isqrt(patches))
        if grid * grid != patches:
            raise ValueError(f"P must be a square token grid, got P={patches}")
        return grid

    def _param_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        p = next(self.parameters())
        return p.dtype, p.device

    @staticmethod
    def _resize_video(x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        return F.interpolate(x, size=(t, h, w), mode="trilinear", align_corners=False)

    @staticmethod
    def _hint_to_bcthw(x: torch.Tensor, *, batch: int, horizon: int, name: str) -> torch.Tensor:
        if x.ndim == 4:
            x = x[:, :, None]
        if x.ndim != 5:
            raise ValueError(f"{name} must be [B,T,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        if x.shape[0] != batch or x.shape[1] != horizon:
            raise ValueError(f"{name} leading dims must be {(batch, horizon)}, got {tuple(x.shape[:2])}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        return x

    @staticmethod
    def _rgb_to_bcthw(x: torch.Tensor, *, batch: int, horizon: int, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"{name} must be [B,T,3,H,W], got {tuple(x.shape)}")
        if x.shape[0] != batch or x.shape[1] != horizon:
            raise ValueError(f"{name} leading dims must be {(batch, horizon)}, got {tuple(x.shape[:2])}")
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] > 3:
            x = x[:, :3]
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1, -1)
        if x.shape[1] != 3:
            raise ValueError(f"{name} must have 1 or 3 channels, got {x.shape[1]}")
        return x

    @staticmethod
    def _features_to_bcthw(x: torch.Tensor, *, batch: int, horizon: int, channels: int, name: str) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"{name} must be [B,T,C,H,W] or [B,C,T,H,W], got {tuple(x.shape)}")
        if x.shape[0] != batch:
            raise ValueError(f"{name} batch must be {batch}, got {x.shape[0]}")
        if x.shape[1] == horizon and x.shape[2] == channels:
            return x.permute(0, 2, 1, 3, 4).contiguous()
        if x.shape[1] == channels:
            out = x.contiguous()
            if out.shape[2] != horizon:
                out = F.interpolate(out, size=(horizon, out.shape[-2], out.shape[-1]), mode="trilinear", align_corners=False)
            return out
        raise ValueError(
            f"{name} expected channels={channels} in [B,T,C,H,W] or [B,C,T,H,W], got {tuple(x.shape)}"
        )

    def build_control_state(
        self,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor | None = None,
        *,
        motion_hint: torch.Tensor | None = None,
        contact_hint: torch.Tensor | None = None,
        rough_rgb: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
        point: torch.Tensor | None = None,
        pose_geom: torch.Tensor | None = None,
        rgb_motion_features: torch.Tensor | None = None,
        noisy_latents: torch.Tensor | None = None,
        source_latents: torch.Tensor | None = None,
        sigma: torch.Tensor | None = None,
        scale: float = 1.0,
        img_token_len: int | None = None,
    ) -> HunyuanDiTControlState:
        if pred_tokens.ndim != 4:
            raise ValueError(f"pred_tokens must be [B,T,P,D], got {tuple(pred_tokens.shape)}")
        batch, horizon, patches, dim = pred_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        grid = self._grid_size(patches)
        dtype, device = self._param_dtype_device()
        pred_tokens = pred_tokens.to(device=device, dtype=dtype)

        tok = self.token_proj(pred_tokens)
        feat = tok.reshape(batch, horizon, grid, grid, self.cfg.hidden).permute(0, 4, 1, 2, 3).contiguous()

        if self.cfg.use_depth:
            if depth is None:
                depth_v = feat.new_zeros(batch, 1, horizon, grid, grid)
            else:
                depth_v = self._hint_to_bcthw(depth.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="depth")
                depth_v = self._resize_video(depth_v, horizon, grid, grid)
            feat = feat + self.depth_proj(depth_v)

        if self.cfg.use_motion and motion_hint is not None:
            motion_v = self._hint_to_bcthw(motion_hint.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="motion_hint")
            feat = feat + self.motion_proj(self._resize_video(motion_v, horizon, grid, grid))

        if self.cfg.use_contact and contact_hint is not None:
            contact_v = self._hint_to_bcthw(contact_hint.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="contact_hint")
            feat = feat + self.contact_proj(self._resize_video(contact_v, horizon, grid, grid))

        if self.cfg.use_rough and rough_rgb is not None:
            rough_v = self._rgb_to_bcthw(rough_rgb.to(device=device, dtype=dtype), batch=batch, horizon=horizon, name="rough_rgb")
            feat = feat + self.rough_proj(self._resize_video(rough_v, horizon, grid, grid))

        if self.cfg.use_context and context_rgb is not None:
            if context_rgb.ndim != 4 or context_rgb.shape[0] != batch:
                raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context_rgb.shape)}")
            ctx = context_rgb.to(device=device, dtype=dtype)
            if ctx.shape[1] == 1:
                ctx = ctx.expand(-1, 3, -1, -1)
            if ctx.shape[1] != 3:
                raise ValueError(f"context_rgb must have 1 or 3 channels, got {ctx.shape[1]}")
            ctx = F.interpolate(ctx, size=(grid, grid), mode="bilinear", align_corners=False)
            ctx = self.context_proj(ctx)[:, :, None]
            feat = feat + ctx.expand(-1, -1, horizon, -1, -1)

        if self.rgb_feature_proj is not None and rgb_motion_features is not None:
            rgb_feat = self._features_to_bcthw(
                rgb_motion_features.to(device=device, dtype=dtype),
                batch=batch,
                horizon=horizon,
                channels=int(self.cfg.rgb_feature_dim),
                name="rgb_motion_features",
            )
            rgb_feat = self._resize_video(rgb_feat, horizon, grid, grid)
            feat = feat + float(self.cfg.rgb_feature_gain) * self.rgb_feature_proj(rgb_feat)

        if self.noisy_latent_proj is not None and noisy_latents is not None:
            z = noisy_latents.to(device=device, dtype=dtype)
            if z.ndim != 5:
                raise ValueError(f"noisy_latents must be [B,C,T,H,W], got {tuple(z.shape)}")
            if z.shape[0] != batch:
                raise ValueError(f"noisy_latents batch must be {batch}, got {z.shape[0]}")
            if z.shape[1] != self.cfg.latent_channels:
                raise ValueError(f"noisy_latents channels must be {self.cfg.latent_channels}, got {z.shape[1]}")
            feat = feat + self.noisy_latent_proj(self._resize_video(z, horizon, grid, grid))

        if self.source_latent_condition_proj is not None and source_latents is not None:
            src = source_latents.to(device=device, dtype=dtype)
            if src.ndim != 5:
                raise ValueError(f"source_latents must be [B,C,T,H,W], got {tuple(src.shape)}")
            if src.shape[0] != batch:
                raise ValueError(f"source_latents batch must be {batch}, got {src.shape[0]}")
            if src.shape[1] != self.cfg.latent_channels:
                raise ValueError(f"source_latents channels must be {self.cfg.latent_channels}, got {src.shape[1]}")
            feat = feat + self.source_latent_condition_proj(self._resize_video(src, horizon, grid, grid))

        if self.sigma_proj is not None:
            if sigma is None:
                sigma_v = feat.new_zeros(batch, 1)
            else:
                sigma_v = sigma.to(device=device, dtype=dtype)
                if sigma_v.ndim == 0:
                    sigma_v = sigma_v.expand(batch)
                sigma_v = sigma_v.reshape(batch, -1)
                if sigma_v.shape[1] != 1:
                    sigma_v = sigma_v[:, :1]
            if sigma_v.shape != (batch, 1):
                raise ValueError(f"sigma must broadcast to [B,1] with B={batch}, got {tuple(sigma_v.shape)}")
            feat = feat + self.sigma_proj(sigma_v)[:, :, None, None, None]

        action_residual = None
        action_film = None
        action_latent = None
        action_tokens = None
        parallel_action_stream = None
        if (
            self.cfg.use_action
            or self.cfg.use_block_action_film
            or abs(float(self.cfg.action_direct_scale)) > 0.0
            or abs(float(self.cfg.action_latent_scale)) > 0.0
            or bool(self.cfg.use_action_cross_attn)
            or bool(self.cfg.use_parallel_action_dit)
        ):
            if action_cond is None:
                action_cond = pred_tokens.new_zeros(batch, horizon, self.cfg.action_dim)
            action_cond = action_cond.to(device=device, dtype=dtype)
            if action_cond.ndim == 2:
                action_cond = action_cond[:, None].expand(-1, horizon, -1)
            if action_cond.shape != (batch, horizon, self.cfg.action_dim):
                raise ValueError(f"action_cond must be [B,T,{self.cfg.action_dim}], got {tuple(action_cond.shape)}")
            if abs(float(self.cfg.action_direct_scale)) > 0.0:
                direct_action = self.action_direct_proj(action_cond).permute(0, 2, 1)[:, :, :, None, None]
                feat = feat + float(self.cfg.action_direct_scale) * direct_action
            if abs(float(self.cfg.action_latent_scale)) > 0.0:
                action_latent = self.action_latent_proj(action_cond).permute(0, 2, 1)[:, :, :, None, None].contiguous()
            if bool(self.cfg.use_action_cross_attn):
                action_tokens = self.action_cross_token_proj(action_cond)
                pos = torch.linspace(0.0, 1.0, steps=horizon, device=device, dtype=dtype)[None, :, None].expand(batch, -1, -1)
                action_tokens = action_tokens + float(self.cfg.action_cross_attn_time_scale) * self.action_cross_time_proj(pos)
            if bool(self.cfg.use_parallel_action_dit):
                parallel_action_stream = self.parallel_action_token_proj(action_cond)
                pos = torch.linspace(0.0, 1.0, steps=horizon, device=device, dtype=dtype)[None, :, None].expand(batch, -1, -1)
                parallel_action_stream = parallel_action_stream + self.parallel_action_time_proj(pos)
            if self.cfg.use_block_action_film:
                action_film = self.action_summary_proj(action_cond.float().mean(dim=1).to(device=device, dtype=dtype))
                if bool(self.cfg.use_temporal_action_summary):
                    pos = torch.linspace(0.0, 1.0, steps=horizon, device=device, dtype=dtype)[None, :, None].expand(batch, -1, -1)
                    action_seq = self.temporal_action_summary_token_proj(action_cond)
                    action_seq = action_seq + self.temporal_action_summary_time_proj(pos)
                    action_film = action_film + float(self.cfg.temporal_action_summary_scale) * self.temporal_action_summary_out(
                        action_seq.mean(dim=1)
                    )
            if self.cfg.use_action:
                action = self.action_proj(action_cond).permute(0, 2, 1)[:, :, :, None, None]
                feat = feat + action
                action_grid = action.expand(-1, -1, -1, grid, grid)
                token_action = torch.cat([feat, action_grid], dim=1)
                token_action = token_action.permute(0, 2, 3, 4, 1).contiguous()
                token_action = self.action_token_fuse(token_action)
                token_action = token_action.permute(0, 4, 1, 2, 3).contiguous()
                feat = feat + float(self.cfg.action_token_scale) * token_action
                action_residual = action

        if self.cfg.use_task:
            if task_emb is None:
                task_emb = pred_tokens.new_zeros(batch, self.cfg.task_dim)
            task_emb = task_emb.to(device=device, dtype=dtype)
            if task_emb.shape != (batch, self.cfg.task_dim):
                raise ValueError(f"task_emb must be [B,{self.cfg.task_dim}], got {tuple(task_emb.shape)}")
            task = self.task_proj(task_emb)[:, :, None, None, None]
            feat = feat + task

        if self.cfg.use_point and point is not None:
            pts = point.to(device=device, dtype=dtype)
            if pts.ndim == 5 and pts.shape[-1] == self.cfg.point_dim:
                pts = pts.permute(0, 4, 1, 2, 3).contiguous()  # [B,T,H,W,3] -> [B,3,T,H,W]
            elif not (pts.ndim == 5 and pts.shape[1] == self.cfg.point_dim):
                raise ValueError(f"point must be [B,T,H,W,{self.cfg.point_dim}] or [B,{self.cfg.point_dim},T,H,W], got {tuple(point.shape)}")
            feat = feat + self.point_proj(self._resize_video(pts, horizon, grid, grid))

        if self.cfg.use_pose and pose_geom is not None:
            pose_g = pose_geom.to(device=device, dtype=dtype)
            if pose_g.ndim == 2:
                pose_g = pose_g[:, None].expand(-1, horizon, -1)
            if pose_g.shape != (batch, horizon, self.cfg.pose_dim):
                raise ValueError(f"pose_geom must be [B,T,{self.cfg.pose_dim}], got {tuple(pose_geom.shape)}")
            pose_feat = self.pose_proj(pose_g).permute(0, 2, 1)[:, :, :, None, None]
            feat = feat + pose_feat

        fused = self.fuse(feat)
        if action_residual is not None:
            fused = fused + float(self.cfg.action_residual_scale) * action_residual
        state = HunyuanDiTControlState(
            features=fused,
            scale=float(scale),
            img_token_len=img_token_len,
            action_film=action_film,
            action_latent=action_latent,
            action_tokens=action_tokens,
            parallel_action_stream=parallel_action_stream,
        )
        return state

    def prepare_controls(self, *args: Any, **kwargs: Any) -> HunyuanDiTControlState:
        state = self.build_control_state(*args, **kwargs)
        self._control_state = state
        return state

    def set_control_state(self, state: HunyuanDiTControlState) -> None:
        self._control_state = state

    def clear_control_state(self) -> None:
        self._control_state = None

    @staticmethod
    def _latent_thw(latent_shape: Any, state: HunyuanDiTControlState) -> tuple[int, int, int]:
        if latent_shape is None:
            return int(state.features.shape[2]), int(state.features.shape[3]), int(state.features.shape[4])
        if isinstance(latent_shape, torch.Tensor):
            latent_shape = tuple(latent_shape.shape)
        latent_shape = tuple(latent_shape)
        if len(latent_shape) >= 5:
            return int(latent_shape[-3]), int(latent_shape[-2]), int(latent_shape[-1])
        if len(latent_shape) == 4:
            return int(latent_shape[-3]), int(latent_shape[-2]), int(latent_shape[-1])
        if len(latent_shape) == 3:
            return int(latent_shape[0]), int(latent_shape[1]), int(latent_shape[2])
        raise ValueError(f"latent_shape must have 3, 4, or 5 dims, got {latent_shape}")

    def _sequence_features(self, token_len: int, latent_shape: Any) -> torch.Tensor:
        state = self._control_state
        if state is None:
            raise RuntimeError("control state is not prepared; call prepare_controls/build_control_state first")
        t, h, w = self._latent_thw(latent_shape, state)
        feat = self._resize_video(state.features, t, h, w)
        seq = feat.flatten(2).transpose(1, 2).contiguous()
        if seq.shape[1] != token_len:
            seq = F.interpolate(seq.transpose(1, 2), size=token_len, mode="linear", align_corners=False).transpose(1, 2)
        return seq

    @staticmethod
    def _require_channels(x: torch.Tensor, channels: int, *, name: str) -> torch.Tensor:
        if x.shape[-1] != channels:
            raise RuntimeError(
                f"{name} residual hidden dim {x.shape[-1]} does not match Hunyuan DiT hidden dim {channels}; "
                "train/load the control adapter with cfg.dit_hidden set to the target transformer hidden_size"
            )
        return x

    def _expand_for_cfg(self, residual: torch.Tensor, batch_size: int) -> torch.Tensor:
        state = self._control_state
        if state is None:
            raise RuntimeError("control state is not prepared")
        control_batch = state.batch_size
        if batch_size == control_batch:
            return residual
        if batch_size == 2 * control_batch and self.cfg.cfg_branch == "conditional_only":
            # HunyuanVideoPipeline concatenates classifier-free batches as
            # [negative/unconditional, prompt/conditional]. Keep wm3d control
            # on the conditional half so text guidance remains meaningful.
            zeros = residual.new_zeros(control_batch, *residual.shape[1:])
            return torch.cat([zeros, residual], dim=0)
        if control_batch == 1:
            return residual.expand(batch_size, *residual.shape[1:])
        raise ValueError(
            f"cannot map control batch {control_batch} to transformer batch {batch_size}; "
            "supported cases are equal batch or CFG batch 2x controls"
        )

    def _expand_action_tokens_for_cfg(self, action_tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        state = self._control_state
        if state is None:
            raise RuntimeError("control state is not prepared")
        control_batch = state.batch_size
        if batch_size == control_batch:
            return action_tokens
        if batch_size == 2 * control_batch and self.cfg.cfg_branch == "conditional_only":
            zeros = action_tokens.new_zeros(control_batch, *action_tokens.shape[1:])
            return torch.cat([zeros, action_tokens], dim=0)
        if control_batch == 1:
            return action_tokens.expand(batch_size, *action_tokens.shape[1:])
        raise ValueError(
            f"cannot map action-token batch {control_batch} to transformer batch {batch_size}; "
            "supported cases are equal batch or CFG batch 2x controls"
        )

    def _action_cross_residual(self, tokens: torch.Tensor, layer_idx: int, *, stream: str) -> torch.Tensor:
        state = self._control_state
        if (
            state is None
            or state.action_tokens is None
            or not bool(self.cfg.use_action_cross_attn)
            or abs(float(self.cfg.action_cross_attn_scale)) <= 0.0
            or tokens.numel() == 0
        ):
            return torch.zeros_like(tokens)
        action_tokens = state.action_tokens.to(device=tokens.device, dtype=tokens.dtype)
        action_tokens = self._expand_action_tokens_for_cfg(action_tokens, int(tokens.shape[0]))
        q = self.action_cross_q(tokens)
        k = self.action_cross_k(action_tokens)
        v = self.action_cross_v(action_tokens)
        heads = int(self.action_cross_heads)
        head_dim = int(self.action_cross_hidden) // heads

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(int(x.shape[0]), int(x.shape[1]), heads, head_dim).transpose(1, 2)

        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)
        attended = F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(int(tokens.shape[0]), int(tokens.shape[1]), int(self.action_cross_hidden))
        residual = self.action_cross_out(attended)
        if stream == "double":
            gate = self.double_action_cross_gates[layer_idx] if 0 <= layer_idx < len(self.double_action_cross_gates) else 1.0
            gain = self._layer_gain(layer_idx, len(self.double_action_cross_gates), stream="double")
        elif stream == "single":
            gate = self.single_action_cross_gates[layer_idx] if 0 <= layer_idx < len(self.single_action_cross_gates) else 1.0
            gain = self._layer_gain(layer_idx, len(self.single_action_cross_gates), stream="single")
        else:
            raise ValueError(f"unknown stream={stream!r}")
        gate_t = gate.to(device=tokens.device, dtype=tokens.dtype) if torch.is_tensor(gate) else tokens.new_tensor(float(gate))
        return residual * gate_t * float(gain) * float(self.cfg.action_cross_attn_scale) * float(state.scale)

    def _parallel_action_dit_residual(self, tokens: torch.Tensor, layer_idx: int, *, stream: str) -> torch.Tensor:
        state = self._control_state
        if (
            state is None
            or state.parallel_action_stream is None
            or not bool(self.cfg.use_parallel_action_dit)
            or abs(float(self.cfg.parallel_action_dit_scale)) <= 0.0
            or tokens.numel() == 0
        ):
            return torch.zeros_like(tokens)
        if stream == "double":
            if layer_idx < 0 or layer_idx >= len(self.double_parallel_action_blocks):
                return torch.zeros_like(tokens)
            block = self.double_parallel_action_blocks[layer_idx]
            gate = self.double_parallel_action_gates[layer_idx]
            gain = self._layer_gain(layer_idx, len(self.double_parallel_action_blocks), stream="double")
        elif stream == "single":
            if layer_idx < 0 or layer_idx >= len(self.single_parallel_action_blocks):
                return torch.zeros_like(tokens)
            block = self.single_parallel_action_blocks[layer_idx]
            gate = self.single_parallel_action_gates[layer_idx]
            gain = self._layer_gain(layer_idx, len(self.single_parallel_action_blocks), stream="single")
        else:
            raise ValueError(f"unknown stream={stream!r}")

        control_batch = int(state.batch_size)
        action_stream = state.parallel_action_stream.to(device=tokens.device, dtype=tokens.dtype)
        action_stream = self._expand_action_tokens_for_cfg(action_stream, int(tokens.shape[0]))
        updated_action, video_residual = block(action_stream, tokens)

        if int(tokens.shape[0]) == control_batch:
            state.parallel_action_stream = updated_action
        elif int(tokens.shape[0]) == 2 * control_batch and self.cfg.cfg_branch == "conditional_only":
            state.parallel_action_stream = updated_action[control_batch:]
        elif control_batch == 1 and int(tokens.shape[0]) != 1:
            state.parallel_action_stream = updated_action[:1]

        gate_t = gate.to(device=tokens.device, dtype=tokens.dtype)
        return video_residual * gate_t * float(gain) * float(self.cfg.parallel_action_dit_scale) * float(state.scale)

    def parallel_action_dit_step(
        self,
        tokens: torch.Tensor,
        action_stream: torch.Tensor | None,
        layer_idx: int,
        *,
        stream: str,
        control_batch: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Checkpoint-safe parallel action step with explicit action-stream threading.

        Hook-based injection mutates ``control_state.parallel_action_stream`` inside
        each block hook. That is not safe when Hunyuan activation checkpointing
        recomputes blocks. This helper mirrors the same action-DiT update, but
        returns the updated action stream to a native forward loop so the stream is
        local to the forward graph, like tau-0-WM's action branch.
        """
        state = self._control_state
        if (
            state is None
            or action_stream is None
            or not bool(self.cfg.use_parallel_action_dit)
            or abs(float(self.cfg.parallel_action_dit_scale)) <= 0.0
            or tokens.numel() == 0
        ):
            return action_stream, torch.zeros_like(tokens)
        if stream == "double":
            if layer_idx < 0 or layer_idx >= len(self.double_parallel_action_blocks):
                return action_stream, torch.zeros_like(tokens)
            block = self.double_parallel_action_blocks[layer_idx]
            gate = self.double_parallel_action_gates[layer_idx]
            gain = self._layer_gain(layer_idx, len(self.double_parallel_action_blocks), stream="double")
        elif stream == "single":
            if layer_idx < 0 or layer_idx >= len(self.single_parallel_action_blocks):
                return action_stream, torch.zeros_like(tokens)
            block = self.single_parallel_action_blocks[layer_idx]
            gate = self.single_parallel_action_gates[layer_idx]
            gain = self._layer_gain(layer_idx, len(self.single_parallel_action_blocks), stream="single")
        else:
            raise ValueError(f"unknown stream={stream!r}")

        action_for_tokens = action_stream.to(device=tokens.device, dtype=tokens.dtype)
        action_for_tokens = self._expand_action_tokens_for_cfg(action_for_tokens, int(tokens.shape[0]))
        updated_action, video_residual = block(action_for_tokens, tokens)

        control_batch = max(1, int(control_batch))
        if int(tokens.shape[0]) == control_batch:
            next_action = updated_action
        elif int(tokens.shape[0]) == 2 * control_batch and self.cfg.cfg_branch == "conditional_only":
            next_action = updated_action[control_batch:]
        elif control_batch == 1 and int(tokens.shape[0]) != 1:
            next_action = updated_action[:1]
        else:
            next_action = updated_action[:control_batch]

        gate_t = gate.to(device=tokens.device, dtype=tokens.dtype)
        residual = video_residual * gate_t * float(gain) * float(self.cfg.parallel_action_dit_scale) * float(state.scale)
        return next_action, residual.to(device=tokens.device, dtype=tokens.dtype)

    def _apply_action_film(self, residual: torch.Tensor, layer_idx: int, *, stream: str) -> torch.Tensor:
        state = self._control_state
        if (
            state is None
            or not bool(self.cfg.use_block_action_film)
            or state.action_film is None
            or abs(float(self.cfg.block_action_film_scale)) <= 0
        ):
            return residual
        if stream == "double":
            if layer_idx < 0 or layer_idx >= len(self.double_action_film):
                return residual
            mod = self.double_action_film[layer_idx](state.action_film)
        elif stream == "single":
            if layer_idx < 0 or layer_idx >= len(self.single_action_film):
                return residual
            mod = self.single_action_film[layer_idx](state.action_film)
        else:
            raise ValueError(f"unknown stream={stream!r}")
        mod = mod.to(device=residual.device, dtype=residual.dtype)
        scale, shift = mod.chunk(2, dim=-1)
        strength = float(self.cfg.block_action_film_scale)
        return residual * (1.0 + strength * scale[:, None, :]) + strength * shift[:, None, :]

    def _layer_gain(self, layer_idx: int, total_layers: int, *, stream: str) -> float:
        total_layers = max(1, int(total_layers))
        if stream == "double":
            start = float(self.cfg.double_control_gain_start)
            end = float(self.cfg.double_control_gain_end)
            power = max(1e-6, float(self.cfg.double_control_gain_power))
        elif stream == "single":
            start = float(self.cfg.single_control_gain_start)
            end = float(self.cfg.single_control_gain_end)
            power = max(1e-6, float(self.cfg.single_control_gain_power))
        else:
            raise ValueError(f"unknown stream={stream!r}")
        if total_layers == 1:
            frac = 1.0
        else:
            frac = max(0.0, min(1.0, float(layer_idx) / float(total_layers - 1)))
        frac = frac ** power
        return start + (end - start) * frac

    def double_residual(
        self,
        layer_idx: int,
        img_tokens: torch.Tensor,
        latent_shape: Any,
        batch_size: int,
        *,
        include_parallel_action: bool = True,
    ) -> torch.Tensor:
        if self._control_state is None:
            return torch.zeros_like(img_tokens)
        if layer_idx < 0 or layer_idx >= len(self.double_projections):
            return torch.zeros_like(img_tokens)
        seq = self._sequence_features(int(img_tokens.shape[1]), latent_shape)
        residual = self.double_projections[layer_idx](seq) * self.double_gates[layer_idx].to(dtype=seq.dtype)
        residual = self._apply_action_film(residual, layer_idx, stream="double")
        residual = residual * self._layer_gain(layer_idx, len(self.double_projections), stream="double")
        residual = self._require_channels(residual, int(img_tokens.shape[-1]), name="double_block") * float(self._control_state.scale)
        residual = self._expand_for_cfg(residual, int(batch_size))
        residual = residual + self._action_cross_residual(img_tokens, layer_idx, stream="double")
        if include_parallel_action:
            residual = residual + self._parallel_action_dit_residual(img_tokens, layer_idx, stream="double")
        return residual.to(device=img_tokens.device, dtype=img_tokens.dtype)

    def single_residual(
        self,
        layer_idx: int,
        x_tokens: torch.Tensor,
        img_token_len: int,
        latent_shape: Any,
        batch_size: int,
        *,
        include_parallel_action: bool = True,
    ) -> torch.Tensor:
        if self._control_state is None:
            return torch.zeros_like(x_tokens)
        if layer_idx < 0 or layer_idx >= len(self.single_projections):
            return torch.zeros_like(x_tokens)
        img_token_len = max(0, min(int(img_token_len), int(x_tokens.shape[1])))
        if img_token_len == 0:
            return torch.zeros_like(x_tokens)
        seq = self._sequence_features(img_token_len, latent_shape)
        img_residual = self.single_projections[layer_idx](seq) * self.single_gates[layer_idx].to(dtype=seq.dtype)
        img_residual = self._apply_action_film(img_residual, layer_idx, stream="single")
        img_residual = img_residual * self._layer_gain(layer_idx, len(self.single_projections), stream="single")
        img_residual = self._require_channels(img_residual, int(x_tokens.shape[-1]), name="single_block") * float(self._control_state.scale)
        img_residual = self._expand_for_cfg(img_residual, int(batch_size))
        img_residual = img_residual + self._action_cross_residual(x_tokens[:, :img_token_len], layer_idx, stream="single")
        if include_parallel_action:
            img_residual = img_residual + self._parallel_action_dit_residual(x_tokens[:, :img_token_len], layer_idx, stream="single")
        out = torch.zeros_like(x_tokens)
        out[:, :img_token_len] = img_residual.to(device=x_tokens.device, dtype=x_tokens.dtype)
        return out

    def _latent_project(
        self,
        target_latents: torch.Tensor | tuple[int, ...],
        projection: nn.Conv3d,
        *,
        include_action_latent: bool = True,
    ) -> torch.Tensor:
        if self._control_state is None:
            raise RuntimeError("control state is not prepared")
        shape = tuple(target_latents.shape) if isinstance(target_latents, torch.Tensor) else tuple(target_latents)
        if len(shape) != 5:
            raise ValueError(f"target_latents must be [B,C,T,H,W], got {shape}")
        t, h, w = int(shape[2]), int(shape[3]), int(shape[4])
        feat = self._resize_video(self._control_state.features, t, h, w)
        residual = projection(feat) * float(self._control_state.scale)
        if include_action_latent and self._control_state.action_latent is not None and abs(float(self.cfg.action_latent_scale)) > 0.0:
            action_residual = self._resize_video(self._control_state.action_latent, t, h, w)
            residual = residual + action_residual * float(self.cfg.action_latent_scale) * float(self._control_state.scale)
        residual = self._expand_for_cfg(residual, int(shape[0]))
        if isinstance(target_latents, torch.Tensor):
            residual = residual.to(device=target_latents.device, dtype=target_latents.dtype)
        return residual

    def latent_residual(self, target_latents: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        """Training helper that maps the same control state to latent velocity shape."""
        return self._latent_project(target_latents, self.latent_projection)

    def source_latent_delta(self, target_latents: torch.Tensor | tuple[int, ...]) -> torch.Tensor:
        """Predict a WM-conditioned Hunyuan latent delta from the context latent source."""
        return self._latent_project(target_latents, self.source_latent_projection)


class HunyuanDiTControlInjector:
    """Install temporary forward hooks on Hunyuan double/single DiT blocks."""

    def __init__(self, transformer: nn.Module, adapter: HunyuanDiTControlAdapter):
        self.transformer = transformer
        self.adapter = adapter
        self._handles: list[Any] = []
        self._img_token_len: int | None = None
        self._latent_shape: Any = None
        self._double_pre_control_scale: float = 0.0
        self._single_pre_control_scale: float = 0.0
        self._final_velocity_residual_scale: float = 0.0
        self._final_velocity_residual_mask: torch.Tensor | None = None
        self._base_control_args: tuple[Any, ...] = ()
        self._base_control_kwargs: dict[str, Any] = {}
        self._dynamic_control_from_forward: bool = False
        self._native_parallel_action_forward: bool = bool(adapter.cfg.native_parallel_action_forward)
        self._original_forward: Callable[..., Any] | None = None

    @staticmethod
    def _iter_blocks(owner: Any, name: str) -> list[nn.Module]:
        blocks = getattr(owner, name, None)
        if blocks is None:
            return []
        if isinstance(blocks, nn.ModuleList):
            return list(blocks)
        if isinstance(blocks, nn.Sequential):
            return list(blocks)
        if isinstance(blocks, (list, tuple)):
            return [b for b in blocks if isinstance(b, nn.Module)]
        return []

    @staticmethod
    def _extract_tensor(output: Any) -> tuple[torch.Tensor | None, Callable[[torch.Tensor], Any]]:
        if isinstance(output, torch.Tensor):
            return output, lambda x: x
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda x: (x, *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            return output[0], lambda x: [x, *output[1:]]
        return None, lambda _x: output

    def _make_transformer_pre_hook(self):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not self._dynamic_control_from_forward:
                return None
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            noisy_latents = inputs[0]
            if noisy_latents.ndim != 5:
                return None
            sigma = None
            if len(inputs) > 1 and torch.is_tensor(inputs[1]):
                sigma = inputs[1].float() / 1000.0
            kwargs = dict(self._base_control_kwargs)
            kwargs["noisy_latents"] = noisy_latents
            if sigma is not None:
                kwargs["sigma"] = sigma
            self._latent_shape = tuple(noisy_latents.shape)
            self.adapter.prepare_controls(*self._base_control_args, **kwargs)
            return None

        return hook

    def _make_double_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not inputs or self.adapter.control_state is None:
                return None
            img = inputs[0]
            if not torch.is_tensor(img) or abs(float(self._double_pre_control_scale)) <= 0:
                return None
            self._img_token_len = int(img.shape[1])
            residual = self.adapter.double_residual(layer_idx, img, self._latent_shape, int(img.shape[0]))
            img = img + residual * float(self._double_pre_control_scale)
            return (img, *inputs[1:])
        return hook

    def _make_double_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor, rebuild = self._extract_tensor(output)
            if tensor is None or self.adapter.control_state is None:
                return output
            self._img_token_len = int(tensor.shape[1])
            residual = self.adapter.double_residual(layer_idx, tensor, self._latent_shape, int(tensor.shape[0]))
            return rebuild(tensor + residual)
        return hook

    @staticmethod
    def _single_img_token_len(inputs: tuple[Any, ...], x: torch.Tensor, fallback: int | None) -> int:
        img_len = fallback
        if len(inputs) > 2:
            txt_len = inputs[2]
            try:
                if torch.is_tensor(txt_len):
                    txt_len = int(txt_len.detach().item())
                else:
                    txt_len = int(txt_len)
                img_len = int(x.shape[1]) - max(0, txt_len)
            except (TypeError, ValueError, RuntimeError):
                pass
        if img_len is None:
            img_len = int(x.shape[1])
        return max(0, min(int(img_len), int(x.shape[1])))

    def _make_single_pre_hook(self, layer_idx: int):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not inputs or self.adapter.control_state is None:
                return None
            x = inputs[0]
            if not torch.is_tensor(x) or abs(float(self._single_pre_control_scale)) <= 0:
                return None
            img_len = self._single_img_token_len(inputs, x, self._img_token_len or self.adapter.control_state.img_token_len)
            residual = self.adapter.single_residual(layer_idx, x, img_len, self._latent_shape, int(x.shape[0]))
            x = x + residual * float(self._single_pre_control_scale)
            return (x, *inputs[1:])
        return hook

    def _make_single_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            tensor, rebuild = self._extract_tensor(output)
            if tensor is None or self.adapter.control_state is None:
                return output
            # Hunyuan HYVideoDiffusionTransformer enters single_blocks as
            # torch.cat((img, txt), 1), so the image stream is the prefix.
            img_len = self._img_token_len or self.adapter.control_state.img_token_len or int(tensor.shape[1])
            residual = self.adapter.single_residual(layer_idx, tensor, img_len, self._latent_shape, int(tensor.shape[0]))
            return rebuild(tensor + residual)
        return hook

    def _make_transformer_output_hook(self):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            if self.adapter.control_state is None or abs(float(self._final_velocity_residual_scale)) <= 0:
                return output
            if not inputs or not torch.is_tensor(inputs[0]):
                return output
            noisy_latents = inputs[0]
            if noisy_latents.ndim != 5:
                return output
            residual = self.adapter.latent_residual(noisy_latents) * float(self._final_velocity_residual_scale)
            if self._final_velocity_residual_mask is not None:
                residual = residual * self._final_velocity_residual_mask.to(device=residual.device, dtype=residual.dtype)

            def add_residual(x: torch.Tensor) -> torch.Tensor:
                return x + residual.to(device=x.device, dtype=x.dtype)

            if torch.is_tensor(output):
                return add_residual(output)
            if isinstance(output, dict) and torch.is_tensor(output.get("x")):
                out = dict(output)
                out["x"] = add_residual(out["x"])
                return out
            if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                return (add_residual(output[0]), *output[1:])
            return output

        return hook

    def _make_native_forward(self, original_forward: Callable[..., Any]) -> Callable[..., Any]:
        func = original_forward.__func__ if hasattr(original_forward, "__func__") else original_forward
        globals_dict = getattr(func, "__globals__", {})
        get_cu_seqlens = globals_dict["get_cu_seqlens"]
        activation_checkpoint = globals_dict.get("activation_checkpoint", _activation_checkpoint)

        injector = self

        def native_forward(
            module: nn.Module,
            x: torch.Tensor,
            t: torch.Tensor,
            text_states: torch.Tensor = None,
            text_mask: torch.Tensor = None,
            text_states_2: torch.Tensor | None = None,
            freqs_cos: torch.Tensor | None = None,
            freqs_sin: torch.Tensor | None = None,
            guidance: torch.Tensor = None,
            return_dict: bool = True,
        ) -> Any:
            out: dict[str, torch.Tensor] = {}
            noisy_latents = x
            if injector._dynamic_control_from_forward and noisy_latents.ndim == 5:
                sigma = t.float() / 1000.0
                kwargs = dict(injector._base_control_kwargs)
                kwargs["noisy_latents"] = noisy_latents
                kwargs["sigma"] = sigma
                injector._latent_shape = tuple(noisy_latents.shape)
                injector.adapter.prepare_controls(*injector._base_control_args, **kwargs)

            img = x
            txt = text_states
            _, _, ot, oh, ow = x.shape
            tt, th, tw = (
                ot // module.patch_size[0],
                oh // module.patch_size[1],
                ow // module.patch_size[2],
            )

            vec = module.time_in(t)
            vec = vec + module.vector_in(text_states_2)
            if module.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + module.guidance_in(guidance)

            img = module.img_in(img)
            if module.text_projection == "linear":
                txt = module.txt_in(txt)
            elif module.text_projection == "single_refiner":
                txt = module.txt_in(txt, t, text_mask if module.use_attention_mask else None)
            else:
                raise NotImplementedError(f"Unsupported text_projection: {module.text_projection}")

            txt_seq_len = int(txt.shape[1])
            img_seq_len = int(img.shape[1])
            injector._img_token_len = img_seq_len
            cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
            cu_seqlens_kv = cu_seqlens_q
            max_seqlen_q = img_seq_len + txt_seq_len
            max_seqlen_kv = max_seqlen_q
            freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
            use_activation_checkpoint = bool(getattr(module, "_wm3d_activation_checkpoint", False)) and module.training and torch.is_grad_enabled()
            activation_checkpoint_use_reentrant = bool(getattr(module, "_wm3d_activation_checkpoint_use_reentrant", False))

            state = injector.adapter.control_state
            action_stream = None if state is None else state.parallel_action_stream
            control_batch = 1 if state is None else int(state.batch_size)

            for layer_idx, block in enumerate(module.double_blocks):
                if use_activation_checkpoint:
                    def _double_forward(img_arg, txt_arg, vec_arg, _block=block):
                        return _block(
                            img_arg,
                            txt_arg,
                            vec_arg,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            freqs_cis,
                        )

                    img, txt = activation_checkpoint(
                        _double_forward,
                        img,
                        txt,
                        vec,
                        use_reentrant=activation_checkpoint_use_reentrant,
                    )
                else:
                    img, txt = block(
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                    )
                if injector.adapter.control_state is not None:
                    base = injector.adapter.double_residual(
                        layer_idx,
                        img,
                        injector._latent_shape,
                        int(img.shape[0]),
                        include_parallel_action=False,
                    )
                    action_stream, parallel = injector.adapter.parallel_action_dit_step(
                        img,
                        action_stream,
                        layer_idx,
                        stream="double",
                        control_batch=control_batch,
                    )
                    img = img + base + parallel

            x_tokens = torch.cat((img, txt), 1)
            for layer_idx, block in enumerate(module.single_blocks):
                if use_activation_checkpoint:
                    def _single_forward(x_arg, vec_arg, _block=block):
                        return _block(
                            x_arg,
                            vec_arg,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                        )

                    x_tokens = activation_checkpoint(
                        _single_forward,
                        x_tokens,
                        vec,
                        use_reentrant=activation_checkpoint_use_reentrant,
                    )
                else:
                    x_tokens = block(
                        x_tokens,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                    )
                if injector.adapter.control_state is not None:
                    base = injector.adapter.single_residual(
                        layer_idx,
                        x_tokens,
                        img_seq_len,
                        injector._latent_shape,
                        int(x_tokens.shape[0]),
                        include_parallel_action=False,
                    )
                    action_stream, parallel = injector.adapter.parallel_action_dit_step(
                        x_tokens[:, :img_seq_len],
                        action_stream,
                        layer_idx,
                        stream="single",
                        control_batch=control_batch,
                    )
                    merged = torch.zeros_like(x_tokens)
                    merged[:, :img_seq_len] = parallel
                    x_tokens = x_tokens + base + merged

            img = x_tokens[:, :img_seq_len, ...]
            img = module.final_layer(img, vec)
            img = module.unpatchify(img, tt, th, tw)
            if injector.adapter.control_state is not None and abs(float(injector._final_velocity_residual_scale)) > 0:
                residual = injector.adapter.latent_residual(noisy_latents) * float(injector._final_velocity_residual_scale)
                if injector._final_velocity_residual_mask is not None:
                    residual = residual * injector._final_velocity_residual_mask.to(device=residual.device, dtype=residual.dtype)
                img = img + residual.to(device=img.device, dtype=img.dtype)
            if return_dict:
                out["x"] = img
                return out
            return img

        return native_forward

    def install(self) -> None:
        if self._handles:
            return
        if self._native_parallel_action_forward:
            if self._original_forward is None:
                self._original_forward = self.transformer.forward
                self.transformer.forward = types.MethodType(self._make_native_forward(self._original_forward), self.transformer)
            return
        self._handles.append(self.transformer.register_forward_pre_hook(self._make_transformer_pre_hook()))
        self._handles.append(self.transformer.register_forward_hook(self._make_transformer_output_hook()))
        for idx, block in enumerate(self._iter_blocks(self.transformer, "double_blocks")):
            self._handles.append(block.register_forward_pre_hook(self._make_double_pre_hook(idx)))
            self._handles.append(block.register_forward_hook(self._make_double_hook(idx)))
        for idx, block in enumerate(self._iter_blocks(self.transformer, "single_blocks")):
            self._handles.append(block.register_forward_pre_hook(self._make_single_pre_hook(idx)))
            self._handles.append(block.register_forward_hook(self._make_single_hook(idx)))

    def remove(self) -> None:
        if self._original_forward is not None:
            self.transformer.forward = self._original_forward
            self._original_forward = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._img_token_len = None
        self._latent_shape = None
        self._double_pre_control_scale = 0.0
        self._single_pre_control_scale = 0.0
        self._final_velocity_residual_scale = 0.0
        self._final_velocity_residual_mask = None
        self._base_control_args = ()
        self._base_control_kwargs = {}
        self._dynamic_control_from_forward = False

    @contextmanager
    def use_controls(
        self,
        *args: Any,
        latent_shape: Any = None,
        double_pre_control_scale: float = 0.0,
        single_pre_control_scale: float = 0.0,
        final_velocity_residual_scale: float = 0.0,
        final_velocity_residual_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        self.install()
        self._latent_shape = latent_shape
        self._double_pre_control_scale = float(double_pre_control_scale)
        self._single_pre_control_scale = float(single_pre_control_scale)
        self._final_velocity_residual_scale = float(final_velocity_residual_scale)
        self._final_velocity_residual_mask = final_velocity_residual_mask
        self._base_control_args = tuple(args)
        self._base_control_kwargs = dict(kwargs)
        self._dynamic_control_from_forward = bool(self.adapter.cfg.use_noisy_latents or self.adapter.cfg.use_sigma_embed)
        try:
            self.adapter.prepare_controls(*args, **kwargs)
            yield self
        finally:
            self.adapter.clear_control_state()
            self.remove()

    @contextmanager
    def use_control_state(
        self,
        state: HunyuanDiTControlState,
        *,
        latent_shape: Any = None,
        double_pre_control_scale: float = 0.0,
        single_pre_control_scale: float = 0.0,
        final_velocity_residual_scale: float = 0.0,
        final_velocity_residual_mask: torch.Tensor | None = None,
    ):
        self.install()
        self._latent_shape = latent_shape
        self._double_pre_control_scale = float(double_pre_control_scale)
        self._single_pre_control_scale = float(single_pre_control_scale)
        self._final_velocity_residual_scale = float(final_velocity_residual_scale)
        self._final_velocity_residual_mask = final_velocity_residual_mask
        try:
            self.adapter.set_control_state(state)
            yield self
        finally:
            self.adapter.clear_control_state()
            self.remove()


def _cfg_from_payload(value: Any, *, ckpt_path: Path) -> HunyuanDiTControlConfig:
    if isinstance(value, HunyuanDiTControlConfig):
        return value
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        value = vars(value)
    if not isinstance(value, dict):
        raise RuntimeError(f"{ckpt_path} cfg must be a dict, got {type(value).__name__}")
    try:
        return HunyuanDiTControlConfig(**value)
    except TypeError as exc:
        raise RuntimeError(f"{ckpt_path} cfg does not match HunyuanDiTControlConfig: {exc}") from exc


def load_hunyuan_dit_control_checkpoint(path: str | Path, device: str | torch.device | None = None) -> tuple[HunyuanDiTControlAdapter, dict[str, Any]]:
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ckpt_path} checkpoint must be a dict")
    if "kind" not in payload:
        raise RuntimeError(f"{ckpt_path} checkpoint is missing kind={CONTROL_CHECKPOINT_KIND}")
    actual_kind = payload.get("kind")
    if actual_kind != CONTROL_CHECKPOINT_KIND:
        raise RuntimeError(f"{ckpt_path} kind must be {CONTROL_CHECKPOINT_KIND}, got {actual_kind!r}")
    if "model" not in payload or "cfg" not in payload:
        raise RuntimeError(f"{ckpt_path} must contain model and cfg")
    cfg = _cfg_from_payload(payload["cfg"], ckpt_path=ckpt_path)
    adapter = HunyuanDiTControlAdapter(cfg)
    result = adapter.load_state_dict(payload["model"], strict=False)
    allowed_missing = tuple(
        k
        for k in result.missing_keys
        if k.startswith("action_token_fuse.")
        or k.startswith("action_summary_proj.")
        or k.startswith("action_direct_proj.")
        or k.startswith("action_latent_proj.")
        or k.startswith("action_cross_token_proj.")
        or k.startswith("action_cross_time_proj.")
        or k.startswith("action_cross_q.")
        or k.startswith("action_cross_k.")
        or k.startswith("action_cross_v.")
        or k.startswith("action_cross_out.")
        or k.startswith("double_action_cross_gates")
        or k.startswith("single_action_cross_gates")
        or k.startswith("temporal_action_summary_token_proj.")
        or k.startswith("temporal_action_summary_time_proj.")
        or k.startswith("temporal_action_summary_out.")
        or k.startswith("parallel_action_token_proj.")
        or k.startswith("parallel_action_time_proj.")
        or k.startswith("double_parallel_action_blocks.")
        or k.startswith("single_parallel_action_blocks.")
        or k.startswith("double_parallel_action_gates")
        or k.startswith("single_parallel_action_gates")
        or k.startswith("double_action_film.")
        or k.startswith("single_action_film.")
        or k.startswith("rgb_feature_proj.")
        or k.startswith("source_latent_projection.")
        or k.startswith("noisy_latent_proj.")
        or k.startswith("source_latent_condition_proj.")
        or k.startswith("sigma_proj.")
    )
    unexpected = tuple(result.unexpected_keys)
    missing = tuple(k for k in result.missing_keys if k not in allowed_missing)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path} is incompatible with HunyuanDiTControlAdapter: "
            f"missing={missing[:20]} unexpected={unexpected[:20]}"
        )
    if allowed_missing:
        payload = dict(payload)
        payload["load_missing_initialized"] = list(allowed_missing)
    if device is not None:
        adapter = adapter.to(device)
    adapter.eval()
    return adapter, payload


def save_hunyuan_dit_control_checkpoint(
    path: str | Path,
    adapter: HunyuanDiTControlAdapter,
    *,
    metrics: dict[str, Any] | None = None,
    wm_ckpt: str | Path | None = None,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = adapter.module if hasattr(adapter, "module") else adapter
    if not isinstance(target, HunyuanDiTControlAdapter):
        raise TypeError(f"expected HunyuanDiTControlAdapter, got {type(target).__name__}")
    payload: dict[str, Any] = {
        "kind": CONTROL_CHECKPOINT_KIND,
        "model": target.state_dict(),
        "cfg": asdict(target.cfg),
        "metrics": metrics or {},
        "wm_ckpt": str(wm_ckpt) if wm_ckpt is not None else None,
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))
    return payload


__all__ = [
    "CONTROL_CHECKPOINT_KIND",
    "HunyuanDiTControlAdapter",
    "HunyuanDiTControlConfig",
    "HunyuanDiTControlInjector",
    "HunyuanDiTControlState",
    "load_hunyuan_dit_control_checkpoint",
    "save_hunyuan_dit_control_checkpoint",
]
