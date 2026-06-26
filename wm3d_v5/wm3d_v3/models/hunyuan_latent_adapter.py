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
    use_point: bool = False
    use_pose: bool = False
    point_dim: int = 3
    pose_dim: int = 9
    motion_gain_init: float = 1.0
    output_mode: str = "direct"
    residual_scale: float = 1.0
    mask_bias_init: float = -2.0
    mask_temperature: float = 1.0
    mask_min: float = 0.0
    mask_max: float = 1.0
    motion_mask_prior_weight: float = 0.0
    motion_residual_boost: float = 0.0
    velocity_scale: float = 1.0
    velocity_blocks: int = 0
    velocity_motion_prior_weight: float = 0.0
    velocity_motion_prior_power: float = 1.0
    velocity_mask_floor: float = 0.0
    temporal_resampler: str = "interp"
    temporal_resampler_horizon: int = 8
    temporal_resampler_sigma: float = 1.15
    temporal_resampler_temperature: float = 1.0
    rough_latent_delta_scale: float = 0.0
    rough_latent_delta_mask_source: str = "prior"
    rough_latent_delta_mask_power: float = 1.0
    rough_latent_delta_mask_floor: float = 0.0
    rough_latent_delta_mask_topk: float = 0.0
    use_rough_latent_delta_condition: bool = False
    use_rgb_scaffold_mask: bool = False
    rgb_scaffold_mask_use_rough: bool = True
    rgb_scaffold_mask_hidden: int = 64
    rgb_scaffold_mask_bias_init: float = -4.0
    use_rgb_features: bool = False
    rgb_feature_dim: int = 0
    rgb_feature_gain: float = 1.0
    use_temporal_memory: bool = False
    temporal_memory_heads: int = 4
    temporal_memory_layers: int = 1
    temporal_memory_mlp_mult: float = 2.0
    temporal_memory_gate_init: float = 0.35
    motion_region_threshold: float = 0.20
    motion_region_softness: float = 0.08
    motion_region_power: float = 1.0
    motion_region_dilate: int = 1
    motion_region_temporal_dilate: int = 1
    motion_region_topk: float = 0.0
    motion_region_floor: float = 0.02
    motion_region_prior_weight: float = 0.75
    motion_region_bg_ceiling: float = 0.06
    motion_region_mask_mode: str = "max"
    direct_delta_static_center_weight: float = 0.0
    direct_delta_temporal_center_weight: float = 0.0
    direct_delta_spatial_highpass_weight: float = 0.0
    direct_delta_spatial_highpass_kernel: int = 1
    direct_delta_static_floor: float = 1.0
    direct_delta_static_energy_limit: float = 0.0
    carrier_delta_scale: float = 1.0
    carrier_delta_static_center_weight: float = 0.0
    carrier_delta_temporal_center_weight: float = 0.0
    carrier_delta_static_floor: float = 1.0
    carrier_delta_spatial_highpass_weight: float = 0.0
    carrier_delta_spatial_highpass_kernel: int = 1
    carrier_delta_energy_limit: float = 0.0
    carrier_delta_static_energy_limit: float = 0.0
    carrier_mask_source: str = "mask"
    carrier_foreground_topk: float = 0.0
    carrier_foreground_min_score: float = 0.0
    carrier_foreground_prior_score_weight: float = 0.5
    carrier_foreground_soft_residual: float = 0.0
    carrier_foreground_hard_scale: float = 1.0
    carrier_foreground_prior_combine: str = "max"
    background_residual_scale: float = 0.0
    action_velocity_scale: float = 1.0
    action_velocity_direct_delta_scale: float = 1.0
    action_velocity_motion_prior_weight: float = 0.0
    action_velocity_motion_prior_floor: float = 0.0
    action_velocity_static_center_weight: float = 0.0
    action_velocity_static_floor: float = 1.0
    action_velocity_static_mask_source: str = "motion_prior"
    action_velocity_static_mask_topk: float = 0.0
    action_velocity_static_mask_threshold: float = 0.0
    action_velocity_static_mask_softness: float = 0.05
    action_velocity_action_gate_weight: float = 0.0
    action_velocity_action_gate_floor: float = 0.0
    action_velocity_action_gate_power: float = 1.0
    action_velocity_action_gate_normalizer: float = 0.20
    action_basis_residual_scale: float = 0.0
    action_basis_normalizer: float = 0.20
    action_basis_blocks: int = 1
    action_basis_residual_mode: str = "free"
    action_basis_projection_clip: float = 1.0
    action_basis_input_mode: str = "mixed"
    wm_delta_residual_scale: float = 0.0
    wm_delta_blocks: int = 1
    wm_delta_source: str = "feature_delta"
    foreground_blocks: int = 2
    foreground_delta_source: str = "residual_delta"
    foreground_residual_scale: float = 1.0
    foreground_delta_clip: float = 0.0
    foreground_write_bias_init: float = -2.0
    foreground_visible_bias_init: float = -2.0
    foreground_alpha_temperature: float = 1.0
    foreground_alpha_min: float = 0.0
    foreground_alpha_max: float = 1.0
    foreground_motion_prior_weight: float = 0.50
    foreground_motion_prior_floor: float = 0.02


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
        self.point_proj = nn.Sequential(
            nn.Conv3d(self.cfg.point_dim, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.pose_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.pose_dim),
            nn.Linear(self.cfg.pose_dim, h),
            nn.SiLU(inplace=True),
            nn.Linear(h, h),
        )
        self.motion_proj = nn.Sequential(
            nn.Conv3d(1, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.motion_gain = nn.Parameter(torch.tensor(float(self.cfg.motion_gain_init), dtype=torch.float32))
        self.rough_proj = nn.Sequential(
            nn.Conv3d(3, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.rough_latent_proj = nn.Sequential(
            nn.Conv3d(self.cfg.latent_channels, h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
        )
        self.rgb_feature_proj = None
        if bool(self.cfg.use_rgb_features) and int(self.cfg.rgb_feature_dim) > 0:
            self.rgb_feature_proj = nn.Sequential(
                nn.Conv3d(int(self.cfg.rgb_feature_dim), h, kernel_size=3, padding=1),
                nn.GroupNorm(_norm_groups(h), h),
                nn.SiLU(inplace=True),
                Conv3dBlock(h),
            )
        rgb_h = max(16, int(self.cfg.rgb_scaffold_mask_hidden))
        self.rgb_scaffold_mask_head = nn.Sequential(
            nn.Conv2d(8, rgb_h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(rgb_h), rgb_h),
            nn.SiLU(inplace=True),
            nn.Conv2d(rgb_h, rgb_h, kernel_size=3, padding=1),
            nn.GroupNorm(_norm_groups(rgb_h), rgb_h),
            nn.SiLU(inplace=True),
            nn.Conv2d(rgb_h, 1, kernel_size=3, padding=1),
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
        self.mask_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, 1, kernel_size=3, padding=1),
        )
        self.velocity_blocks = nn.Sequential(*[Conv3dBlock(h) for _ in range(max(0, int(self.cfg.velocity_blocks)))])
        self.velocity_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels, kernel_size=3, padding=1),
        )
        self.action_basis_blocks = nn.Sequential(*[Conv3dBlock(h) for _ in range(max(0, int(self.cfg.action_basis_blocks)))])
        self.action_basis_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels * self.cfg.action_dim, kernel_size=3, padding=1),
        )
        self.wm_delta_blocks = nn.Sequential(*[Conv3dBlock(h) for _ in range(max(0, int(self.cfg.wm_delta_blocks)))])
        self.wm_delta_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels, kernel_size=3, padding=1),
        )
        self.foreground_blocks = nn.Sequential(*[Conv3dBlock(h) for _ in range(max(0, int(self.cfg.foreground_blocks)))])
        self.foreground_delta_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, self.cfg.latent_channels, kernel_size=3, padding=1),
        )
        self.foreground_write_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, 1, kernel_size=3, padding=1),
        )
        self.foreground_visible_out = nn.Sequential(
            nn.GroupNorm(_norm_groups(h), h),
            nn.SiLU(inplace=True),
            nn.Conv3d(h, 1, kernel_size=3, padding=1),
        )
        self.temporal_logits = nn.Parameter(
            self._init_temporal_logits(
                int(self.cfg.latent_time),
                int(self.cfg.temporal_resampler_horizon),
                float(self.cfg.temporal_resampler_sigma),
            )
        )
        self.temporal_memory_query = nn.Parameter(torch.zeros(int(self.cfg.latent_time), h))
        self.temporal_memory_layers = nn.ModuleList()
        heads = max(1, int(self.cfg.temporal_memory_heads))
        if h % heads != 0:
            heads = 1
        mlp_hidden = max(h, int(round(float(self.cfg.temporal_memory_mlp_mult) * h)))
        for _ in range(max(0, int(self.cfg.temporal_memory_layers))):
            self.temporal_memory_layers.append(
                nn.ModuleDict(
                    {
                        "q_norm": nn.LayerNorm(h),
                        "m_norm": nn.LayerNorm(h),
                        "attn": nn.MultiheadAttention(h, heads, batch_first=True),
                        "ffn_norm": nn.LayerNorm(h),
                        "ffn": nn.Sequential(
                            nn.Linear(h, mlp_hidden),
                            nn.SiLU(inplace=True),
                            nn.Linear(mlp_hidden, h),
                        ),
                    }
                )
            )
        self.temporal_memory_refine = Conv3dBlock(h)
        gate = min(max(float(self.cfg.temporal_memory_gate_init), 1e-4), 1.0 - 1e-4)
        self.temporal_memory_gate_logit = nn.Parameter(torch.tensor(math.log(gate / (1.0 - gate)), dtype=torch.float32))
        self.reset_mask_output()
        self.reset_rgb_scaffold_mask_output()
        self.zero_init_velocity_output()
        self.zero_init_action_basis_output()
        self.zero_init_wm_delta_output()
        self.reset_foreground_output()
        self.reset_temporal_memory()

    @staticmethod
    def _init_temporal_logits(latent_t: int, horizon: int, sigma: float) -> torch.Tensor:
        """Gaussian 8-frame -> 3-latent prior, trainable after checkpoint load."""
        latent_t = max(1, int(latent_t))
        horizon = max(1, int(horizon))
        sigma = max(float(sigma), 1e-3)
        src = torch.arange(horizon, dtype=torch.float32)
        scale = float(horizon) / float(latent_t)
        centers = (torch.arange(latent_t, dtype=torch.float32) + 0.5) * scale - 0.5
        centers = centers.clamp(0.0, float(horizon - 1))
        logits = -0.5 * ((src[None, :] - centers[:, None]) / sigma) ** 2
        return logits

    def zero_init_output(self) -> None:
        """Start residual prediction as an identity passthrough."""
        conv = self.out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final adapter layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def reset_mask_output(self) -> None:
        """Bias the dynamic mask toward a small area before learning."""
        conv = self.mask_out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final mask layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, float(self.cfg.mask_bias_init))

    def reset_rgb_scaffold_mask_output(self) -> None:
        conv = self.rgb_scaffold_mask_head[-1]
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("expected final RGB scaffold mask layer to be nn.Conv2d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, float(self.cfg.rgb_scaffold_mask_bias_init))

    def zero_init_velocity_output(self) -> None:
        """Keep newly-added temporal velocity branch as a no-op for old checkpoints."""
        conv = self.velocity_out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final velocity layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def zero_init_action_basis_output(self) -> None:
        """Keep the signed action-basis residual as a no-op until explicitly trained."""
        conv = self.action_basis_out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final action-basis layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def zero_init_wm_delta_output(self) -> None:
        """Keep the WM action-delta branch as a no-op for old checkpoints."""
        conv = self.wm_delta_out[-1]
        if not isinstance(conv, nn.Conv3d):
            raise TypeError("expected final WM delta layer to be nn.Conv3d")
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    def reset_foreground_output(self) -> None:
        """Initialize foreground residual as a conservative context-preserving layer."""
        delta = self.foreground_delta_out[-1]
        if not isinstance(delta, nn.Conv3d):
            raise TypeError("expected final foreground delta layer to be nn.Conv3d")
        nn.init.zeros_(delta.weight)
        if delta.bias is not None:
            nn.init.zeros_(delta.bias)
        write = self.foreground_write_out[-1]
        visible = self.foreground_visible_out[-1]
        if not isinstance(write, nn.Conv3d) or not isinstance(visible, nn.Conv3d):
            raise TypeError("expected final foreground alpha layers to be nn.Conv3d")
        nn.init.zeros_(write.weight)
        nn.init.zeros_(visible.weight)
        if write.bias is not None:
            nn.init.constant_(write.bias, float(self.cfg.foreground_write_bias_init))
        if visible.bias is not None:
            nn.init.constant_(visible.bias, float(self.cfg.foreground_visible_bias_init))

    def reset_temporal_memory(self) -> None:
        """Initialize latent-time queries around the same 8->3 temporal prior."""
        nn.init.normal_(self.temporal_memory_query, std=0.02)
        logits = self._init_temporal_logits(
            int(self.cfg.latent_time),
            int(self.cfg.temporal_resampler_horizon),
            float(self.cfg.temporal_resampler_sigma),
        )
        weights = torch.softmax(logits, dim=1)
        centers = torch.matmul(weights, torch.linspace(-1.0, 1.0, weights.shape[1]))
        with torch.no_grad():
            self.temporal_memory_query[:, 0].copy_(centers)

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

    def _token_depth_features(
        self,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor,
        *,
        latent_h: int,
        latent_w: int,
        motion_hint: torch.Tensor | None = None,
        point: torch.Tensor | None = None,
        pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, horizon, patches, dim = pred_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        grid = self._grid_size(patches)
        x = pred_tokens.reshape(bsz * horizon, patches, dim).transpose(1, 2)
        x = x.reshape(bsz * horizon, dim, grid, grid)
        x = self.token_proj(x)
        x = F.interpolate(x, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        x = x.reshape(bsz, horizon, self.cfg.hidden, latent_h, latent_w).permute(0, 2, 1, 3, 4)

        depth_v = depth[:, :, None]
        depth_v = self._resize_video(depth_v.permute(0, 2, 1, 3, 4), horizon, latent_h, latent_w)
        x = x + self.depth_proj(depth_v.to(dtype=x.dtype))

        if self.cfg.use_point and point is not None:
            pts = point.to(device=x.device, dtype=x.dtype)
            if pts.ndim == 5 and pts.shape[-1] == self.cfg.point_dim:
                pts = pts.permute(0, 4, 1, 2, 3).contiguous()
            elif not (pts.ndim == 5 and pts.shape[1] == self.cfg.point_dim):
                raise ValueError(
                    "point must be [B,T,H,W,C] or [B,C,T,H,W], "
                    f"got {tuple(point.shape)}"
                )
            if pts.shape[0] != bsz or pts.shape[2] != horizon:
                raise ValueError(
                    f"point leading dims must be [B={bsz}, T={horizon}], "
                    f"got {tuple(point.shape)}"
                )
            pts = self._resize_video(pts, horizon, latent_h, latent_w)
            x = x + self.point_proj(pts)

        if self.cfg.use_pose and pose is not None:
            pose_t = pose.to(device=x.device, dtype=x.dtype)
            if pose_t.ndim == 2:
                pose_t = pose_t[:, None].expand(-1, horizon, -1)
            if pose_t.ndim != 3 or pose_t.shape[0] != bsz or pose_t.shape[1] != horizon:
                raise ValueError(
                    f"pose must be [B,T,{self.cfg.pose_dim}] or [B,{self.cfg.pose_dim}], "
                    f"got {tuple(pose.shape)}"
                )
            if pose_t.shape[-1] != self.cfg.pose_dim:
                if pose_t.shape[-1] > self.cfg.pose_dim:
                    pose_t = pose_t[..., : self.cfg.pose_dim]
                else:
                    pad = pose_t.new_zeros(*pose_t.shape[:-1], self.cfg.pose_dim - pose_t.shape[-1])
                    pose_t = torch.cat([pose_t, pad], dim=-1)
            pose_feat = self.pose_proj(pose_t).permute(0, 2, 1)[:, :, :, None, None]
            x = x + pose_feat

        motion_prior = None
        if self.cfg.use_motion and motion_hint is not None:
            motion_v = motion_hint.permute(0, 2, 1, 3, 4).contiguous()
            motion_v = self._resize_video(motion_v, horizon, latent_h, latent_w)
            motion_prior = motion_v.clamp(0.0, 1.0)
            motion_gain = self.motion_gain.to(device=x.device, dtype=x.dtype)
            x = x + motion_gain * self.motion_proj(motion_v.to(dtype=x.dtype))
        return x, motion_prior

    def _rgb_scaffold_mask(
        self,
        context_rgb: torch.Tensor | None,
        rough_rgb: torch.Tensor | None,
        motion_hint: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not bool(self.cfg.use_rgb_scaffold_mask):
            return None
        if context_rgb is None:
            return None
        use_rough = bool(self.cfg.rgb_scaffold_mask_use_rough) and rough_rgb is not None
        if use_rough:
            bsz, horizon, _, height, width = rough_rgb.shape
            device = rough_rgb.device
            dtype = rough_rgb.dtype
        elif motion_hint is not None:
            bsz, horizon, _, height, width = motion_hint.shape
            device = motion_hint.device
            dtype = motion_hint.dtype
        else:
            return None
        ctx = context_rgb[:, None].expand(-1, horizon, -1, -1, -1)
        if ctx.shape[-2:] != (height, width):
            ctx = F.interpolate(
                ctx.reshape(bsz * horizon, 3, ctx.shape[-2], ctx.shape[-1]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).reshape(bsz, horizon, 3, height, width)
        if motion_hint is not None:
            geom = motion_hint.to(device=device, dtype=dtype)
            if geom.shape[1] != horizon or geom.shape[-2:] != (height, width):
                geom = F.interpolate(
                    geom.permute(0, 2, 1, 3, 4),
                    size=(horizon, height, width),
                    mode="trilinear",
                    align_corners=False,
                ).permute(0, 2, 1, 3, 4)
            geom = geom.clamp(0.0, 1.0)
        else:
            geom = None
        if use_rough:
            rough_feat = rough_rgb.to(device=device, dtype=dtype)
            rough_mag = (rough_feat.float() - ctx.to(device=device).float()).abs().mean(dim=2, keepdim=True)
            rough_prior = rough_mag
            denom = rough_prior.flatten(2).amax(dim=-1).clamp_min(1e-6).view(bsz, horizon, 1, 1, 1)
            rough_prior = (rough_prior / denom).clamp(0.0, 1.0).to(device=device, dtype=dtype)
            if geom is None:
                geom = rough_prior
        else:
            rough_feat = ctx.to(device=device, dtype=dtype)
            if geom is None:
                return None
            rough_prior = geom.to(device=device, dtype=dtype)
        feat = torch.cat(
            [
                rough_feat,
                ctx.to(device=device, dtype=dtype),
                rough_prior.to(device=device, dtype=dtype),
                geom.to(device=device, dtype=dtype),
            ],
            dim=2,
        )
        logits = self.rgb_scaffold_mask_head(feat.reshape(bsz * horizon, 8, height, width))
        logits = logits.reshape(bsz, horizon, 1, height, width)
        return torch.sigmoid(logits), logits

    def _temporal_resample(self, x: torch.Tensor, t: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(x.shape[2], h, w), mode="trilinear", align_corners=False)
        if x.shape[2] == t:
            return x, None
        if self.cfg.temporal_resampler != "learned":
            return self._resize_video(x, t, h, w), None
        if t > self.temporal_logits.shape[0] or x.shape[2] > self.temporal_logits.shape[1]:
            return self._resize_video(x, t, h, w), None
        temperature = max(float(self.cfg.temporal_resampler_temperature), 1e-6)
        logits = self.temporal_logits[:t, : x.shape[2]] / temperature
        weights = torch.softmax(logits, dim=1).to(device=x.device, dtype=x.dtype)
        x = torch.einsum("bcthw,lt->bclhw", x, weights)
        return x.contiguous(), weights

    def _temporal_memory_resample(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor | None:
        if not bool(self.cfg.use_temporal_memory) or len(self.temporal_memory_layers) == 0:
            return None
        if x.shape[-2:] != (h, w):
            x = F.interpolate(x, size=(x.shape[2], h, w), mode="trilinear", align_corners=False)
        bsz, channels, horizon, height, width = x.shape
        memory = x.permute(0, 3, 4, 2, 1).reshape(bsz * height * width, horizon, channels)
        query = self.temporal_memory_query
        if query.shape[0] != t:
            query = F.interpolate(
                query.transpose(0, 1)[None],
                size=t,
                mode="linear",
                align_corners=False,
            ).squeeze(0).transpose(0, 1)
        query = query.to(device=x.device, dtype=x.dtype)[None].expand(memory.shape[0], -1, -1)
        for layer in self.temporal_memory_layers:
            q_norm = layer["q_norm"](query)
            m_norm = layer["m_norm"](memory)
            attn, _ = layer["attn"](q_norm, m_norm, m_norm, need_weights=False)
            query = query + attn
            query = query + layer["ffn"](layer["ffn_norm"](query))
        memory_video = query.reshape(bsz, height, width, t, channels).permute(0, 4, 3, 1, 2).contiguous()
        memory_video = self.temporal_memory_refine(memory_video)
        gate = torch.sigmoid(self.temporal_memory_gate_logit).to(device=x.device, dtype=x.dtype)
        return gate * memory_video

    def _motion_region_prior(self, motion_prior: torch.Tensor | None, t: int, h: int, w: int) -> torch.Tensor | None:
        if motion_prior is None:
            return None
        prior = self._resize_video(motion_prior, t, h, w).clamp(0.0, 1.0)
        threshold = float(self.cfg.motion_region_threshold)
        softness = float(self.cfg.motion_region_softness)
        if softness > 0:
            prior = torch.sigmoid((prior - threshold) / softness)
        else:
            prior = (prior > threshold).to(dtype=prior.dtype)
        power = max(float(self.cfg.motion_region_power), 1e-6)
        if abs(power - 1.0) > 1e-6:
            prior = prior.pow(power)
        spatial_radius = max(0, int(self.cfg.motion_region_dilate))
        temporal_radius = max(0, int(self.cfg.motion_region_temporal_dilate))
        if spatial_radius > 0 or temporal_radius > 0:
            kt = 2 * temporal_radius + 1
            khw = 2 * spatial_radius + 1
            prior = F.max_pool3d(
                prior,
                kernel_size=(kt, khw, khw),
                stride=1,
                padding=(temporal_radius, spatial_radius, spatial_radius),
            )
        topk_frac = float(self.cfg.motion_region_topk)
        if 0.0 < topk_frac < 1.0:
            flat = prior.flatten(3)
            k = max(1, int(math.ceil(topk_frac * float(flat.shape[-1]))))
            idx = torch.topk(flat, k=k, dim=-1).indices
            hard = torch.zeros_like(flat).scatter_(-1, idx, 1.0)
            prior = (flat * hard).reshape_as(prior)
        floor = min(max(float(self.cfg.motion_region_floor), 0.0), 1.0)
        if floor > 0:
            prior = floor + (1.0 - floor) * prior
        return prior.clamp(0.0, 1.0)

    def _delta_dynamic_prior(self, delta: torch.Tensor) -> torch.Tensor:
        """Estimate dynamic latent regions from delta energy without RGB compositing."""
        score = delta.float().abs().mean(dim=1, keepdim=True)
        denom = score.flatten(3).amax(dim=-1, keepdim=True).view(*score.shape[:3], 1, 1).clamp_min(1e-6)
        score = (score / denom).clamp(0.0, 1.0)
        threshold = float(getattr(self.cfg, "action_velocity_static_mask_threshold", 0.0))
        softness = max(float(getattr(self.cfg, "action_velocity_static_mask_softness", 0.05)), 1e-6)
        if threshold > 0:
            prior = torch.sigmoid((score - threshold) / softness)
        else:
            prior = score
        topk_frac = float(getattr(self.cfg, "action_velocity_static_mask_topk", 0.0))
        if 0.0 < topk_frac < 1.0:
            flat_score = score.flatten(3)
            k = max(1, int(math.ceil(float(flat_score.shape[-1]) * topk_frac)))
            idx = torch.topk(flat_score, k=k, dim=-1).indices
            hard = torch.zeros_like(flat_score).scatter_(-1, idx, 1.0).reshape_as(score)
            prior = prior * hard
        return prior.clamp(0.0, 1.0)

    def forward(
        self,
        pred_tokens: torch.Tensor,
        depth: torch.Tensor,
        *,
        context_rgb: torch.Tensor | None = None,
        motion_hint: torch.Tensor | None = None,
        rough_rgb: torch.Tensor | None = None,
        rgb_motion_features: torch.Tensor | None = None,
        rough_latent_delta: torch.Tensor | None = None,
        rough_delta_mask: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        task_emb: torch.Tensor | None = None,
        point: torch.Tensor | None = None,
        pose: torch.Tensor | None = None,
        target_latents: torch.Tensor | None = None,
        base_latents: torch.Tensor | None = None,
        reference_pred_tokens: torch.Tensor | None = None,
        reference_depth: torch.Tensor | None = None,
        reference_motion_hint: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
        rgb_scaffold_mask = self._rgb_scaffold_mask(context_rgb, rough_rgb, motion_hint)

        x, motion_prior = self._token_depth_features(
            pred_tokens,
            depth,
            latent_h=latent_h,
            latent_w=latent_w,
            motion_hint=motion_hint,
            point=point,
            pose=pose,
        )
        wm_delta_residual = None
        wm_delta_residual_horizon = None
        wm_delta_temporal_weights = None
        basis_mode_for_ref = str(getattr(self.cfg, "action_basis_residual_mode", "free")).lower()
        needs_wm_delta_carrier = basis_mode_for_ref in {
            "wm_delta_project",
            "wm_action_delta_project",
            "project_wm_delta",
        }
        if (
            (abs(float(getattr(self.cfg, "wm_delta_residual_scale", 0.0))) > 0 or needs_wm_delta_carrier)
            and reference_pred_tokens is not None
            and reference_depth is not None
        ):
            if reference_pred_tokens.shape[:3] != pred_tokens.shape[:3] or reference_pred_tokens.shape[-1] != pred_tokens.shape[-1]:
                raise ValueError(
                    "reference_pred_tokens must match pred_tokens shape, "
                    f"got {tuple(reference_pred_tokens.shape)} vs {tuple(pred_tokens.shape)}"
                )
            if reference_depth.shape != depth.shape:
                raise ValueError(f"reference_depth must match depth shape, got {tuple(reference_depth.shape)} vs {tuple(depth.shape)}")
            ref_x, _ = self._token_depth_features(
                reference_pred_tokens.to(device=pred_tokens.device, dtype=pred_tokens.dtype),
                reference_depth.to(device=depth.device, dtype=depth.dtype),
                latent_h=latent_h,
                latent_w=latent_w,
                motion_hint=reference_motion_hint,
            )
            wm_delta_x = x - ref_x.to(device=x.device, dtype=x.dtype)
            wm_delta_source = str(getattr(self.cfg, "wm_delta_source", "feature_delta")).lower()
            if wm_delta_source in {"feature_delta", "wm_delta_head"}:
                wm_delta_residual_horizon = self.wm_delta_out(self.wm_delta_blocks(wm_delta_x)).float()
            elif wm_delta_source in {"shared_head", "shared_residual_head", "residual_head"}:
                wm_delta_residual_horizon = self.out(self.blocks(wm_delta_x.to(dtype=x.dtype))).float()
            else:
                raise ValueError(f"unknown wm_delta_source={wm_delta_source!r}")
            if wm_delta_residual_horizon.shape[2] > 0:
                wm_delta_residual_horizon = wm_delta_residual_horizon.clone()
                wm_delta_residual_horizon[:, :, 0] = 0
            wm_delta_residual, wm_delta_temporal_weights = self._temporal_resample(
                wm_delta_residual_horizon.to(dtype=x.dtype),
                latent_t,
                latent_h,
                latent_w,
            )

        if self.cfg.use_rough_rgb and rough_rgb is not None:
            rough_v = self._resize_video(rough_rgb.permute(0, 2, 1, 3, 4), horizon, latent_h, latent_w)
            x = x + self.rough_proj(rough_v.to(dtype=x.dtype))

        if self.rgb_feature_proj is not None and rgb_motion_features is not None:
            if rgb_motion_features.ndim != 5:
                raise ValueError(
                    "rgb_motion_features must be [B,T,C,H,W] or [B,C,T,H,W], "
                    f"got {tuple(rgb_motion_features.shape)}"
                )
            if rgb_motion_features.shape[0] != bsz:
                raise ValueError(
                    f"rgb_motion_features batch must be {bsz}, got {rgb_motion_features.shape[0]}"
                )
            if rgb_motion_features.shape[1] == horizon:
                rgb_feat = rgb_motion_features.permute(0, 2, 1, 3, 4).contiguous()
            else:
                rgb_feat = rgb_motion_features.contiguous()
            expected_c = int(self.cfg.rgb_feature_dim)
            if rgb_feat.shape[1] != expected_c:
                raise ValueError(f"expected rgb feature dim {expected_c}, got {rgb_feat.shape[1]}")
            rgb_feat = self._resize_video(rgb_feat.to(device=x.device, dtype=x.dtype), horizon, latent_h, latent_w)
            x = x + float(self.cfg.rgb_feature_gain) * self.rgb_feature_proj(rgb_feat)

        if self.cfg.use_context and context_rgb is not None:
            ctx = F.interpolate(context_rgb, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
            ctx = self.context_proj(ctx.to(dtype=x.dtype))[:, :, None]
            x = x + ctx.expand(-1, -1, horizon, -1, -1)

        if self.cfg.use_task:
            if task_emb is None:
                task_emb = pred_tokens.new_zeros(bsz, self.cfg.task_dim)
            task = self.task_proj(task_emb.to(dtype=x.dtype))[:, :, None, None, None]
            x = x + task

        x_action_free = x
        if self.cfg.use_action:
            if action_cond is None:
                action_cond = pred_tokens.new_zeros(bsz, horizon, self.cfg.action_dim)
            action = self.action_proj(action_cond.to(dtype=x.dtype)).permute(0, 2, 1)[:, :, :, None, None]
            x = x + action

        x_pre_temporal = x
        action_velocity = None
        action_velocity_residual_horizon = None
        action_latent_velocity_residual = None
        action_velocity_temporal_weights = None
        action_velocity_gate = None
        action_velocity_action_gate = None
        action_velocity_static_mask = None
        action_basis_velocity = None
        action_basis_residual_horizon = None
        action_basis_residual = None
        action_basis_residual_projected = None
        action_basis_projection_coeff = None
        action_basis_temporal_weights = None
        if self.cfg.output_mode == "action_latent_velocity":
            action_velocity = self.velocity_out(self.velocity_blocks(x_pre_temporal)).float()
            if action_velocity.shape[2] > 0:
                action_velocity = action_velocity.clone()
                action_velocity[:, :, 0] = 0
            motion_gate = None
            if motion_prior is not None:
                motion_gate = self._resize_video(
                    motion_prior.to(device=action_velocity.device, dtype=torch.float32),
                    action_velocity.shape[2],
                    latent_h,
                    latent_w,
                ).clamp(0.0, 1.0)
            static_center_weight = float(getattr(self.cfg, "action_velocity_static_center_weight", 0.0))
            if abs(static_center_weight) > 0:
                if motion_gate is not None:
                    static_mask = (1.0 - motion_gate).clamp(0.0, 1.0)
                else:
                    static_mask = torch.ones(
                        action_velocity.shape[0],
                        1,
                        action_velocity.shape[2],
                        action_velocity.shape[3],
                        action_velocity.shape[4],
                        device=action_velocity.device,
                        dtype=torch.float32,
                    )
                denom = static_mask.sum(dim=(3, 4), keepdim=True).clamp_min(1e-6)
                static_mean = (action_velocity * static_mask).sum(dim=(3, 4), keepdim=True) / denom
                action_velocity = action_velocity - static_center_weight * static_mean
                action_velocity_static_mask = static_mask
            static_floor = min(max(float(getattr(self.cfg, "action_velocity_static_floor", 1.0)), 0.0), 1.0)
            if static_floor < 1.0 and motion_gate is not None:
                action_velocity = action_velocity * (motion_gate + static_floor * (1.0 - motion_gate))
            prior_weight = min(max(float(getattr(self.cfg, "action_velocity_motion_prior_weight", 0.0)), 0.0), 1.0)
            if prior_weight > 0 and motion_gate is not None:
                floor = min(max(float(getattr(self.cfg, "action_velocity_motion_prior_floor", 0.0)), 0.0), 1.0)
                prior_gate = floor + (1.0 - floor) * motion_gate
                action_velocity_gate = (1.0 - prior_weight) + prior_weight * prior_gate
                action_velocity = action_velocity * action_velocity_gate
            action_velocity_residual_horizon = torch.cumsum(action_velocity, dim=2)
            action_latent_velocity_residual, action_velocity_temporal_weights = self._temporal_resample(
                action_velocity_residual_horizon.to(dtype=x.dtype),
                latent_t,
                latent_h,
                latent_w,
            )
            action_gate_weight = min(max(float(getattr(self.cfg, "action_velocity_action_gate_weight", 0.0)), 0.0), 1.0)
            if action_gate_weight > 0:
                if action_cond is None:
                    action_for_gate = pred_tokens.new_zeros(bsz, horizon, self.cfg.action_dim)
                else:
                    action_for_gate = action_cond.to(device=x.device, dtype=torch.float32)
                action_mag = action_for_gate.abs().mean(dim=-1, keepdim=True)
                normalizer = max(float(getattr(self.cfg, "action_velocity_action_gate_normalizer", 0.20)), 1e-6)
                gate = (action_mag / normalizer).clamp(0.0, 1.0)
                power = max(float(getattr(self.cfg, "action_velocity_action_gate_power", 1.0)), 1e-6)
                if abs(power - 1.0) > 1e-6:
                    gate = gate.pow(power)
                floor = min(max(float(getattr(self.cfg, "action_velocity_action_gate_floor", 0.0)), 0.0), 1.0)
                gate = floor + (1.0 - floor) * gate
                gate_video = gate.permute(0, 2, 1)[:, :, :, None, None].to(dtype=x.dtype)
                action_velocity_action_gate, _ = self._temporal_resample(gate_video, latent_t, 1, 1)
                action_velocity_action_gate = action_velocity_action_gate.to(dtype=torch.float32).clamp(0.0, 1.0)
            if abs(float(getattr(self.cfg, "action_basis_residual_scale", 0.0))) > 0:
                basis_input_mode = str(getattr(self.cfg, "action_basis_input_mode", "mixed")).lower()
                if basis_input_mode in {"scene", "scene_only", "action_free"}:
                    basis_input = x_action_free
                elif basis_input_mode == "mixed":
                    basis_input = x_pre_temporal
                else:
                    raise ValueError(f"unknown action_basis_input_mode={basis_input_mode!r}")
                basis_raw = self.action_basis_out(self.action_basis_blocks(basis_input)).float()
                basis_raw = basis_raw.reshape(
                    bsz,
                    int(self.cfg.latent_channels),
                    int(self.cfg.action_dim),
                    basis_raw.shape[2],
                    basis_raw.shape[3],
                    basis_raw.shape[4],
                )
                if action_cond is None:
                    action_for_basis = pred_tokens.new_zeros(bsz, horizon, self.cfg.action_dim)
                else:
                    action_for_basis = action_cond.to(device=x.device, dtype=torch.float32)
                normalizer = max(float(getattr(self.cfg, "action_basis_normalizer", 0.20)), 1e-6)
                action_signed = (action_for_basis / normalizer).clamp(-1.0, 1.0)
                action_signed = action_signed.permute(0, 2, 1)[:, None, :, :, None, None]
                action_basis_velocity = (basis_raw * action_signed).sum(dim=2)
                if action_basis_velocity.shape[2] > 0:
                    action_basis_velocity = action_basis_velocity.clone()
                    action_basis_velocity[:, :, 0] = 0
                action_basis_residual_horizon = torch.cumsum(action_basis_velocity, dim=2)
                action_basis_residual, action_basis_temporal_weights = self._temporal_resample(
                    action_basis_residual_horizon.to(dtype=x.dtype),
                    latent_t,
                    latent_h,
                    latent_w,
                )
        x, temporal_weights = self._temporal_resample(x, latent_t, latent_h, latent_w)
        temporal_memory = self._temporal_memory_resample(x_pre_temporal, latent_t, latent_h, latent_w)
        if temporal_memory is not None:
            x = x + temporal_memory
        if bool(self.cfg.use_rough_latent_delta_condition) and rough_latent_delta is not None:
            rough_delta_cond = self._resize_video(
                rough_latent_delta.to(device=x.device, dtype=x.dtype),
                x.shape[2],
                latent_h,
                latent_w,
            )
            x = x + self.rough_latent_proj(rough_delta_cond)
        x = self.blocks(x)
        residual = self.out(x)
        if self.cfg.output_mode == "direct":
            if return_components:
                out = {"latents": residual, "residual": residual}
                if temporal_weights is not None:
                    out["temporal_weights"] = temporal_weights
                if temporal_memory is not None:
                    out["temporal_memory"] = temporal_memory
                    out["temporal_memory_gate"] = torch.sigmoid(self.temporal_memory_gate_logit)
                if rgb_scaffold_mask is not None:
                    out["rgb_scaffold_mask"], out["rgb_scaffold_mask_logits"] = rgb_scaffold_mask
                return out
            return residual
        if self.cfg.output_mode not in {
            "context_residual_mask",
            "context_residual_mask_velocity",
            "rough_motion_refine",
            "motion_carrier_anchor",
            "action_latent_velocity",
            "direct_context_blend",
            "direct_motion_region_blend",
            "direct_temporal_delta_blend",
            "direct_temporal_delta_motion_region_blend",
            "direct_temporal_delta_bgprotect",
            "foreground_context_residual",
        }:
            raise ValueError(f"unknown output_mode={self.cfg.output_mode!r}")
        if base_latents is None:
            raise ValueError(f"output_mode={self.cfg.output_mode!r} requires base_latents")
        logits = self.mask_out(x)
        temperature = max(float(self.cfg.mask_temperature), 1e-6)
        mask = torch.sigmoid(logits / temperature)
        mask = mask * (float(self.cfg.mask_max) - float(self.cfg.mask_min)) + float(self.cfg.mask_min)
        if motion_prior is not None and float(self.cfg.motion_mask_prior_weight) > 0:
            prior = self._resize_video(motion_prior, mask.shape[2], mask.shape[3], mask.shape[4])
            prior = (float(self.cfg.motion_mask_prior_weight) * prior.to(dtype=mask.dtype)).clamp(0.0, 1.0)
            mask = 1.0 - (1.0 - mask) * (1.0 - prior)
        base_full = base_latents.to(device=residual.device)
        base = base_full.to(dtype=residual.dtype)
        residual_scale = float(self.cfg.residual_scale)
        rough_delta_added = None
        rough_delta_gate = None
        carrier_delta = None
        centered_carrier_delta = None
        carrier_delta_applied_scale = None
        foreground_write_alpha = None
        foreground_visible_alpha = None
        foreground_prior = None
        foreground_write_logits = None
        foreground_visible_logits = None
        if self.cfg.output_mode == "foreground_context_residual":
            fg_h = self.foreground_blocks(x)
            head_delta = self.foreground_delta_out(fg_h)
            delta_source = str(self.cfg.foreground_delta_source)
            if delta_source == "head":
                foreground_delta = head_delta
            elif delta_source == "residual":
                foreground_delta = residual
            elif delta_source == "residual_delta":
                foreground_delta = residual.float() - base_full.float()
            elif delta_source == "temporal_delta":
                foreground_delta = residual.float() - residual[:, :, :1].float().expand_as(residual)
            elif delta_source == "residual_delta_head":
                foreground_delta = residual.float() - base_full.float() + head_delta.float()
            else:
                raise ValueError(f"unknown foreground_delta_source={delta_source!r}")
            clip = float(self.cfg.foreground_delta_clip)
            if clip > 0:
                foreground_delta = foreground_delta.clamp(-clip, clip)
            alpha_temp = max(float(self.cfg.foreground_alpha_temperature), 1e-6)
            alpha_min = float(self.cfg.foreground_alpha_min)
            alpha_max = float(self.cfg.foreground_alpha_max)
            write_logits = self.foreground_write_out(fg_h)
            visible_logits = self.foreground_visible_out(fg_h)
            foreground_write_alpha = torch.sigmoid(write_logits / alpha_temp)
            foreground_visible_alpha = torch.sigmoid(visible_logits / alpha_temp)
            foreground_write_alpha = foreground_write_alpha * (alpha_max - alpha_min) + alpha_min
            foreground_visible_alpha = foreground_visible_alpha * (alpha_max - alpha_min) + alpha_min
            region_prior = self._motion_region_prior(motion_prior, foreground_write_alpha.shape[2], foreground_write_alpha.shape[3], foreground_write_alpha.shape[4])
            if region_prior is not None:
                region_prior = region_prior.to(dtype=foreground_write_alpha.dtype, device=foreground_write_alpha.device)
                prior_floor = min(max(float(self.cfg.foreground_motion_prior_floor), 0.0), 1.0)
                prior = prior_floor + (1.0 - prior_floor) * region_prior.clamp(0.0, 1.0)
                forced = min(max(float(self.cfg.foreground_motion_prior_weight), 0.0), 1.0) * prior
                foreground_write_alpha = torch.maximum(foreground_write_alpha, forced)
                foreground_visible_alpha = torch.maximum(foreground_visible_alpha, forced)
            pred = base_full.float() + float(self.cfg.foreground_residual_scale) * foreground_write_alpha.float() * foreground_delta.float()
            if rough_delta_added is not None:
                pred = pred + rough_delta_added.float()
            if return_components:
                out = {
                    "latents": pred,
                    "base_latents": base,
                    "residual": residual,
                    "foreground_delta": foreground_delta,
                    "foreground_write_alpha": foreground_write_alpha,
                    "foreground_visible_alpha": foreground_visible_alpha,
                    "foreground_write_alpha_rgb": foreground_write_alpha.permute(0, 2, 1, 3, 4).contiguous(),
                    "foreground_visible_alpha_rgb": foreground_visible_alpha.permute(0, 2, 1, 3, 4).contiguous(),
                    "foreground_write_logits": write_logits,
                    "foreground_visible_logits": visible_logits,
                    "mask": foreground_write_alpha,
                    "mask_logits": write_logits,
                }
                if temporal_weights is not None:
                    out["temporal_weights"] = temporal_weights
                if temporal_memory is not None:
                    out["temporal_memory"] = temporal_memory
                    out["temporal_memory_gate"] = torch.sigmoid(self.temporal_memory_gate_logit)
                if rgb_scaffold_mask is not None:
                    out["rgb_scaffold_mask"], out["rgb_scaffold_mask_logits"] = rgb_scaffold_mask
                if region_prior is not None:
                    out["motion_region_prior_latent"] = region_prior
                    out["motion_region_prior"] = region_prior.permute(0, 2, 1, 3, 4).contiguous()
                return out
            return pred
        if rough_latent_delta is not None and abs(float(self.cfg.rough_latent_delta_scale)) > 0:
            rough_delta = self._resize_video(
                rough_latent_delta.to(device=residual.device, dtype=residual.dtype),
                base.shape[2],
                base.shape[3],
                base.shape[4],
            )
            source = str(self.cfg.rough_latent_delta_mask_source)
            if rough_delta_mask is not None:
                prior_gate = self._resize_video(
                    rough_delta_mask.to(device=residual.device, dtype=residual.dtype),
                    base.shape[2],
                    base.shape[3],
                    base.shape[4],
                ).clamp(0.0, 1.0)
            else:
                prior_gate = torch.ones(
                    base.shape[0],
                    1,
                    base.shape[2],
                    base.shape[3],
                    base.shape[4],
                    dtype=residual.dtype,
                    device=residual.device,
                )
            if source == "none":
                rough_delta_gate = torch.ones_like(prior_gate)
            elif source == "mask":
                rough_delta_gate = mask
            elif source == "prior":
                rough_delta_gate = prior_gate
            elif source == "max":
                rough_delta_gate = torch.maximum(mask, prior_gate)
            elif source == "min":
                rough_delta_gate = torch.minimum(mask, prior_gate)
            else:
                raise ValueError(f"unknown rough_latent_delta_mask_source={source!r}")
            power = max(float(self.cfg.rough_latent_delta_mask_power), 1e-6)
            if abs(power - 1.0) > 1e-6:
                rough_delta_gate = rough_delta_gate.pow(power)
            floor = min(max(float(self.cfg.rough_latent_delta_mask_floor), 0.0), 1.0)
            if floor > 0:
                rough_delta_gate = floor + (1.0 - floor) * rough_delta_gate
            topk_frac = float(self.cfg.rough_latent_delta_mask_topk)
            if 0.0 < topk_frac < 1.0:
                flat = rough_delta_gate.flatten(3)
                k = max(1, int(math.ceil(topk_frac * float(flat.shape[-1]))))
                idx = torch.topk(flat, k=k, dim=-1).indices
                hard = torch.zeros_like(flat).scatter_(-1, idx, 1.0)
                rough_delta_gate = (flat * hard).reshape_as(rough_delta_gate)
            rough_delta_added = float(self.cfg.rough_latent_delta_scale) * rough_delta_gate * rough_delta

        velocity = None
        velocity_residual = None
        residual_anchor = None
        region_prior = None
        learned_mask = mask
        action_velocity_direct_delta = None
        action_velocity_combined_delta = None
        action_velocity_delta_centered = None
        action_velocity_static_prior = None
        direct_blend_modes = {
            "direct_context_blend",
            "direct_motion_region_blend",
            "direct_temporal_delta_blend",
            "direct_temporal_delta_motion_region_blend",
            "direct_temporal_delta_bgprotect",
        }
        motion_region_modes = {"direct_motion_region_blend", "direct_temporal_delta_motion_region_blend"}
        bgprotect_modes = {"direct_temporal_delta_bgprotect"}
        temporal_delta_modes = {
            "direct_temporal_delta_blend",
            "direct_temporal_delta_motion_region_blend",
            "direct_temporal_delta_bgprotect",
        }
        direct_temporal_delta = None
        if self.cfg.output_mode in direct_blend_modes:
            direct_latents = residual
            blend_mask = mask
            if self.cfg.output_mode in motion_region_modes or self.cfg.output_mode in bgprotect_modes:
                region_prior = self._motion_region_prior(motion_prior, mask.shape[2], mask.shape[3], mask.shape[4])
                if region_prior is not None:
                    region_prior = region_prior.to(dtype=mask.dtype, device=mask.device)
                    if self.cfg.output_mode in motion_region_modes:
                        prior_weight = min(max(float(self.cfg.motion_region_prior_weight), 0.0), 1.0)
                        mask_mode = str(getattr(self.cfg, "motion_region_mask_mode", "max"))
                        if mask_mode == "max":
                            gated = mask * region_prior
                            forced = prior_weight * region_prior
                            blend_mask = torch.maximum(gated, forced.clamp(0.0, 1.0))
                        elif mask_mode == "floor_blend":
                            blend_mask = region_prior * (prior_weight + (1.0 - prior_weight) * mask)
                        elif mask_mode in {"multiply", "product"}:
                            blend_mask = mask * (prior_weight + (1.0 - prior_weight) * region_prior)
                        elif mask_mode in {"min", "intersect"}:
                            blend_mask = torch.minimum(mask, region_prior)
                        elif mask_mode == "learned":
                            blend_mask = mask
                        elif mask_mode == "prior":
                            blend_mask = region_prior
                        else:
                            raise ValueError(f"unknown motion_region_mask_mode={mask_mode!r}")
                        bg_ceiling = min(max(float(self.cfg.motion_region_bg_ceiling), 0.0), 1.0)
                        if bg_ceiling < 1.0:
                            ceiling = region_prior + bg_ceiling * (1.0 - region_prior)
                            blend_mask = torch.minimum(blend_mask, ceiling.to(dtype=blend_mask.dtype))
                        blend_mask = blend_mask.clamp(0.0, 1.0)
                    else:
                        static_floor = min(max(float(getattr(self.cfg, "direct_delta_static_floor", 1.0)), 0.0), 1.0)
                        blend_mask = region_prior + static_floor * (1.0 - region_prior)
                        blend_mask = blend_mask.clamp(static_floor, 1.0)
            mask = blend_mask
            if self.cfg.output_mode in temporal_delta_modes:
                direct_temporal_delta = direct_latents.float() - direct_latents[:, :, :1].float().expand_as(direct_latents)
                if self.cfg.output_mode in bgprotect_modes and region_prior is not None:
                    static_mask = (1.0 - region_prior.float()).clamp(0.0, 1.0)
                else:
                    static_mask = (1.0 - mask.float()).clamp(0.0, 1.0)
                static_center_weight = float(getattr(self.cfg, "direct_delta_static_center_weight", 0.0))
                if abs(static_center_weight) > 0:
                    denom = static_mask.sum(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
                    static_mean = (direct_temporal_delta * static_mask).sum(dim=(2, 3, 4), keepdim=True) / denom
                    direct_temporal_delta = direct_temporal_delta - static_center_weight * static_mean
                temporal_center_weight = float(getattr(self.cfg, "direct_delta_temporal_center_weight", 0.0))
                if abs(temporal_center_weight) > 0 and direct_temporal_delta.shape[2] > 1:
                    direct_temporal_delta = direct_temporal_delta - temporal_center_weight * direct_temporal_delta.mean(dim=2, keepdim=True)
                highpass_weight = float(getattr(self.cfg, "direct_delta_spatial_highpass_weight", 0.0))
                highpass_kernel = int(getattr(self.cfg, "direct_delta_spatial_highpass_kernel", 1))
                if abs(highpass_weight) > 0 and highpass_kernel > 1:
                    if highpass_kernel % 2 == 0:
                        highpass_kernel += 1
                    bsz, channels, time, height, width = direct_temporal_delta.shape
                    flat = direct_temporal_delta.permute(0, 2, 1, 3, 4).reshape(bsz * time, channels, height, width)
                    lowpass = F.avg_pool2d(
                        flat,
                        kernel_size=highpass_kernel,
                        stride=1,
                        padding=highpass_kernel // 2,
                        count_include_pad=False,
                    )
                    lowpass = lowpass.reshape(bsz, time, channels, height, width).permute(0, 2, 1, 3, 4)
                    direct_temporal_delta = direct_temporal_delta - highpass_weight * lowpass
                if self.cfg.output_mode in bgprotect_modes:
                    if region_prior is not None:
                        static_floor = min(max(float(getattr(self.cfg, "direct_delta_static_floor", 1.0)), 0.0), 1.0)
                        delta_gate = region_prior.float() + static_floor * (1.0 - region_prior.float())
                        direct_temporal_delta = direct_temporal_delta * delta_gate.clamp(static_floor, 1.0)
                    static_limit = float(getattr(self.cfg, "direct_delta_static_energy_limit", 0.0))
                    if static_limit > 0 and static_mask is not None:
                        static_weight = static_mask.expand(-1, direct_temporal_delta.shape[1], -1, -1, -1)
                        denom = static_weight.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
                        static_energy = (direct_temporal_delta.abs() * static_weight).sum(
                            dim=(1, 2, 3, 4),
                            keepdim=True,
                        ) / denom
                        static_scale = (static_limit / static_energy.clamp_min(1e-6)).clamp(max=1.0)
                        dynamic_part = direct_temporal_delta * (1.0 - static_weight)
                        static_part = direct_temporal_delta * static_weight * static_scale
                        direct_temporal_delta = dynamic_part + static_part
                    pred = base_full.float() + residual_scale * direct_temporal_delta
                else:
                    pred = base_full.float() + residual_scale * mask.float() * direct_temporal_delta
            else:
                pred = base * (1.0 - mask.to(dtype=base.dtype)) + direct_latents * mask.to(dtype=direct_latents.dtype)
        elif self.cfg.output_mode == "rough_motion_refine":
            # The base latents carry motion from a rough video. The learned path
            # only refines detail/background, so a closed mask cannot erase motion.
            pred = base + residual_scale * mask * residual
        elif self.cfg.output_mode == "motion_carrier_anchor":
            if rough_latent_delta is None:
                raise ValueError("output_mode='motion_carrier_anchor' requires rough_latent_delta")
            carrier_delta = self._resize_video(
                rough_latent_delta.to(device=residual.device, dtype=residual.dtype),
                base.shape[2],
                base.shape[3],
                base.shape[4],
            )
            region_prior = self._motion_region_prior(motion_prior, mask.shape[2], mask.shape[3], mask.shape[4])
            learned_mask = mask
            write_mask = mask
            carrier_mask_source = str(getattr(self.cfg, "carrier_mask_source", "mask"))
            if carrier_mask_source == "foreground":
                fg_h = self.foreground_blocks(x)
                foreground_write_logits = self.foreground_write_out(fg_h)
                foreground_visible_logits = self.foreground_visible_out(fg_h)
                alpha_temp = max(float(self.cfg.foreground_alpha_temperature), 1e-6)
                alpha_min = float(self.cfg.foreground_alpha_min)
                alpha_max = float(self.cfg.foreground_alpha_max)
                foreground_write_alpha = torch.sigmoid(foreground_write_logits / alpha_temp)
                foreground_visible_alpha = torch.sigmoid(foreground_visible_logits / alpha_temp)
                foreground_write_alpha = foreground_write_alpha * (alpha_max - alpha_min) + alpha_min
                foreground_visible_alpha = foreground_visible_alpha * (alpha_max - alpha_min) + alpha_min
                prior_terms = []
                if region_prior is not None:
                    prior_terms.append(region_prior.to(dtype=foreground_write_alpha.dtype, device=foreground_write_alpha.device))
                if rough_delta_mask is not None:
                    rough_prior = self._resize_video(
                        rough_delta_mask.to(device=foreground_write_alpha.device, dtype=foreground_write_alpha.dtype),
                        foreground_write_alpha.shape[2],
                        foreground_write_alpha.shape[3],
                        foreground_write_alpha.shape[4],
                    ).clamp(0.0, 1.0)
                    prior_terms.append(rough_prior)
                if rgb_scaffold_mask is not None:
                    scaffold_mask, _ = rgb_scaffold_mask
                    scaffold_prior = scaffold_mask.permute(0, 2, 1, 3, 4).contiguous()
                    scaffold_prior = self._resize_video(
                        scaffold_prior.to(device=foreground_write_alpha.device, dtype=foreground_write_alpha.dtype),
                        foreground_write_alpha.shape[2],
                        foreground_write_alpha.shape[3],
                        foreground_write_alpha.shape[4],
                    ).clamp(0.0, 1.0)
                    prior_terms.append(scaffold_prior)
                if prior_terms:
                    prior_stack = torch.stack(prior_terms, dim=0)
                    prior_combine = str(getattr(self.cfg, "carrier_foreground_prior_combine", "max"))
                    if prior_combine == "max":
                        foreground_prior = prior_stack.amax(dim=0)
                    elif prior_combine == "min":
                        foreground_prior = prior_stack.amin(dim=0)
                    elif prior_combine == "product":
                        foreground_prior = prior_stack.prod(dim=0)
                    else:
                        raise ValueError(f"unknown carrier_foreground_prior_combine={prior_combine!r}")
                    foreground_prior = foreground_prior.clamp(0.0, 1.0)
                topk_frac = float(getattr(self.cfg, "carrier_foreground_topk", 0.0))
                if topk_frac > 0:
                    topk_frac = min(max(topk_frac, 0.0), 1.0)
                    score = foreground_write_alpha
                    prior_score_weight = min(max(float(getattr(self.cfg, "carrier_foreground_prior_score_weight", 0.0)), 0.0), 1.0)
                    if foreground_prior is not None and prior_score_weight > 0:
                        prior_score = foreground_prior.to(dtype=score.dtype, device=score.device)
                        score = score * ((1.0 - prior_score_weight) + prior_score_weight * prior_score)
                    flat_score = score.flatten(-2)
                    k = max(1, int(math.ceil(flat_score.shape[-1] * topk_frac)))
                    threshold = flat_score.topk(k, dim=-1).values[..., -1].view(*score.shape[:-2], 1, 1)
                    hard_mask = score >= threshold
                    min_score = float(getattr(self.cfg, "carrier_foreground_min_score", 0.0))
                    if min_score > 0:
                        hard_mask = hard_mask & (score >= min_score)
                    hard_mask = hard_mask.to(dtype=foreground_write_alpha.dtype)
                    # Straight-through hard foreground selection: the forward mask is
                    # binary, but gradients still train the foreground write head.
                    hard_mask = hard_mask + score - score.detach()
                    hard_scale = float(getattr(self.cfg, "carrier_foreground_hard_scale", 1.0))
                    soft_residual = min(max(float(getattr(self.cfg, "carrier_foreground_soft_residual", 0.0)), 0.0), 1.0)
                    write_mask = (hard_scale * hard_mask + soft_residual * foreground_write_alpha).clamp(0.0, 1.0)
                    foreground_visible_alpha = torch.maximum(foreground_visible_alpha, write_mask)
                elif foreground_prior is not None:
                    prior_floor = min(max(float(self.cfg.foreground_motion_prior_floor), 0.0), 1.0)
                    prior = prior_floor + (1.0 - prior_floor) * foreground_prior
                    forced = min(max(float(self.cfg.foreground_motion_prior_weight), 0.0), 1.0) * prior
                    forced = forced.to(dtype=foreground_write_alpha.dtype)
                    write_mask = torch.maximum(foreground_write_alpha, forced)
                    foreground_visible_alpha = torch.maximum(foreground_visible_alpha, forced)
                else:
                    write_mask = foreground_write_alpha
                learned_mask = foreground_write_alpha
                region_prior = write_mask.clamp(0.0, 1.0)
            elif carrier_mask_source == "mask" and region_prior is not None:
                region_prior = region_prior.to(dtype=mask.dtype, device=mask.device)
                floor = min(max(float(self.cfg.motion_region_prior_weight), 0.0), 1.0)
                mask_mode = str(getattr(self.cfg, "motion_region_mask_mode", "max"))
                if mask_mode == "floor_blend":
                    write_mask = region_prior * (floor + (1.0 - floor) * mask)
                    bg_ceiling = min(max(float(self.cfg.motion_region_bg_ceiling), 0.0), 1.0)
                    if bg_ceiling > 0:
                        write_mask = write_mask + bg_ceiling * (1.0 - region_prior) * mask
                else:
                    forced = (floor * region_prior).clamp(0.0, 1.0)
                    write_mask = torch.maximum(mask * region_prior, forced)
                bg_ceiling = min(max(float(self.cfg.motion_region_bg_ceiling), 0.0), 1.0)
                if bg_ceiling < 1.0:
                    ceiling = region_prior + bg_ceiling * (1.0 - region_prior)
                    write_mask = torch.minimum(write_mask, ceiling.to(dtype=write_mask.dtype))
            elif carrier_mask_source != "mask":
                raise ValueError(f"unknown carrier_mask_source={carrier_mask_source!r}")
            mask = write_mask.clamp(0.0, 1.0)
            carrier_delta_f = carrier_delta.float()
            static_center_weight = float(getattr(self.cfg, "carrier_delta_static_center_weight", 0.0))
            if abs(static_center_weight) > 0:
                if region_prior is not None:
                    static_mask = (1.0 - region_prior.float()).clamp(0.0, 1.0)
                else:
                    static_mask = (1.0 - mask.float()).clamp(0.0, 1.0)
                static_denom = static_mask.sum(dim=(3, 4), keepdim=True).clamp_min(1e-6)
                static_mean = (carrier_delta_f * static_mask).sum(dim=(3, 4), keepdim=True) / static_denom
                carrier_delta_f = carrier_delta_f - static_center_weight * static_mean
            temporal_center_weight = float(getattr(self.cfg, "carrier_delta_temporal_center_weight", 0.0))
            if abs(temporal_center_weight) > 0 and carrier_delta_f.shape[2] > 1:
                temporal_mean = carrier_delta_f.mean(dim=2, keepdim=True)
                carrier_delta_f = carrier_delta_f - temporal_center_weight * temporal_mean
            highpass_weight = float(getattr(self.cfg, "carrier_delta_spatial_highpass_weight", 0.0))
            highpass_kernel = int(getattr(self.cfg, "carrier_delta_spatial_highpass_kernel", 1))
            if abs(highpass_weight) > 0 and highpass_kernel > 1:
                if highpass_kernel % 2 == 0:
                    highpass_kernel += 1
                bsz, channels, time, height, width = carrier_delta_f.shape
                flat = carrier_delta_f.permute(0, 2, 1, 3, 4).reshape(bsz * time, channels, height, width)
                lowpass = F.avg_pool2d(
                    flat,
                    kernel_size=highpass_kernel,
                    stride=1,
                    padding=highpass_kernel // 2,
                    count_include_pad=False,
                )
                lowpass = lowpass.reshape(bsz, time, channels, height, width).permute(0, 2, 1, 3, 4)
                carrier_delta_f = carrier_delta_f - highpass_weight * lowpass
            static_floor = min(max(float(getattr(self.cfg, "carrier_delta_static_floor", 1.0)), 0.0), 1.0)
            if static_floor < 1.0:
                if region_prior is not None:
                    carrier_gate = region_prior.float() + static_floor * (1.0 - region_prior.float())
                else:
                    carrier_gate = mask.float() + static_floor * (1.0 - mask.float())
                carrier_delta_f = carrier_delta_f * carrier_gate.clamp(static_floor, 1.0)
            centered_carrier_delta = carrier_delta_f
            carrier_write = mask.float() * carrier_delta_f
            overall_limit = float(getattr(self.cfg, "carrier_delta_energy_limit", 0.0))
            static_limit = float(getattr(self.cfg, "carrier_delta_static_energy_limit", 0.0))
            if overall_limit > 0 or static_limit > 0:
                scale_terms = []
                eps = 1e-6
                if overall_limit > 0:
                    energy = carrier_write.abs().mean(dim=(1, 2, 3, 4), keepdim=True)
                    scale_terms.append((overall_limit / energy.clamp_min(eps)).clamp(max=1.0))
                if static_limit > 0:
                    if region_prior is not None:
                        static_mask = (1.0 - region_prior.float()).clamp(0.0, 1.0)
                    else:
                        static_mask = (1.0 - mask.float()).clamp(0.0, 1.0)
                    static_weight = static_mask.expand(-1, carrier_write.shape[1], -1, -1, -1)
                    denom = static_weight.sum(dim=(1, 2, 3, 4), keepdim=True).clamp_min(eps)
                    static_energy = (carrier_write.abs() * static_weight).sum(dim=(1, 2, 3, 4), keepdim=True) / denom
                    scale_terms.append((static_limit / static_energy.clamp_min(eps)).clamp(max=1.0))
                carrier_delta_applied_scale = torch.stack(scale_terms, dim=0).amin(dim=0)
                carrier_write = carrier_write * carrier_delta_applied_scale
            else:
                carrier_delta_applied_scale = torch.ones(
                    carrier_write.shape[0],
                    1,
                    1,
                    1,
                    1,
                    device=carrier_write.device,
                    dtype=carrier_write.dtype,
                )
            pred = base_full.float() + float(self.cfg.carrier_delta_scale) * carrier_write
            pred = pred + residual_scale * mask.float() * residual.float()
            bg_residual_scale = float(self.cfg.background_residual_scale)
            if abs(bg_residual_scale) > 0:
                pred = pred + bg_residual_scale * (1.0 - mask.float()) * residual.float()
        elif self.cfg.output_mode == "action_latent_velocity":
            if action_latent_velocity_residual is None:
                raise RuntimeError("action_latent_velocity branch did not build a horizon velocity residual")
            action_velocity_direct_delta = residual.float() - residual[:, :, :1].float().expand_as(residual)
            action_velocity_combined_delta = (
                float(getattr(self.cfg, "action_velocity_direct_delta_scale", 1.0)) * action_velocity_direct_delta
                + float(getattr(self.cfg, "action_velocity_scale", 1.0)) * action_latent_velocity_residual.float()
            )
            if action_basis_residual is not None:
                basis_delta = action_basis_residual.float()
                basis_mode = str(getattr(self.cfg, "action_basis_residual_mode", "free")).lower()
                if basis_mode in {
                    "direct_delta_project",
                    "direct_delta_projection",
                    "project_direct_delta",
                    "wm_delta_project",
                    "wm_action_delta_project",
                    "project_wm_delta",
                }:
                    if basis_mode in {"wm_delta_project", "wm_action_delta_project", "project_wm_delta"}:
                        if wm_delta_residual is None:
                            raise RuntimeError(
                                "action_basis_residual_mode=wm_delta_project requires reference WM outputs "
                                "and wm_delta_residual_scale != 0"
                            )
                        carrier_delta_f = wm_delta_residual.float()
                    else:
                        carrier_delta_f = action_velocity_direct_delta.float()
                    denom = carrier_delta_f.square().sum(dim=1, keepdim=True).clamp_min(1e-6)
                    action_basis_projection_coeff = (basis_delta * carrier_delta_f).sum(dim=1, keepdim=True) / denom
                    clip = float(getattr(self.cfg, "action_basis_projection_clip", 1.0))
                    if clip > 0:
                        action_basis_projection_coeff = clip * torch.tanh(action_basis_projection_coeff / clip)
                    action_basis_residual_projected = action_basis_projection_coeff * carrier_delta_f
                    basis_delta = action_basis_residual_projected
                elif basis_mode != "free":
                    raise ValueError(f"unknown action_basis_residual_mode={basis_mode!r}")
                action_velocity_combined_delta = action_velocity_combined_delta + float(
                    getattr(self.cfg, "action_basis_residual_scale", 0.0)
                ) * basis_delta
            if wm_delta_residual is not None:
                action_velocity_combined_delta = action_velocity_combined_delta + float(
                    getattr(self.cfg, "wm_delta_residual_scale", 0.0)
                ) * wm_delta_residual.float()
            action_gate_weight = min(max(float(getattr(self.cfg, "action_velocity_action_gate_weight", 0.0)), 0.0), 1.0)
            if action_gate_weight > 0 and action_velocity_action_gate is not None:
                action_gate = (1.0 - action_gate_weight) + action_gate_weight * action_velocity_action_gate
                action_velocity_combined_delta = action_velocity_combined_delta * action_gate
            region_prior = self._motion_region_prior(motion_prior, mask.shape[2], mask.shape[3], mask.shape[4])
            static_mask_source = str(getattr(self.cfg, "action_velocity_static_mask_source", "motion_prior"))
            if static_mask_source == "direct_delta":
                action_velocity_static_prior = self._delta_dynamic_prior(action_velocity_direct_delta)
            elif static_mask_source == "combined_delta":
                action_velocity_static_prior = self._delta_dynamic_prior(action_velocity_combined_delta)
            elif static_mask_source == "motion_prior":
                action_velocity_static_prior = region_prior
            else:
                raise ValueError(f"unknown action_velocity_static_mask_source={static_mask_source!r}")
            static_center_weight = float(getattr(self.cfg, "action_velocity_static_center_weight", 0.0))
            if abs(static_center_weight) > 0:
                if action_velocity_static_prior is not None:
                    static_mask = (1.0 - action_velocity_static_prior.float()).clamp(0.0, 1.0)
                else:
                    static_mask = (1.0 - mask.float()).clamp(0.0, 1.0)
                denom = static_mask.sum(dim=(3, 4), keepdim=True).clamp_min(1e-6)
                static_mean = (action_velocity_combined_delta * static_mask).sum(dim=(3, 4), keepdim=True) / denom
                action_velocity_combined_delta = action_velocity_combined_delta - static_center_weight * static_mean
                action_velocity_static_mask = static_mask
            static_floor = min(max(float(getattr(self.cfg, "action_velocity_static_floor", 1.0)), 0.0), 1.0)
            if static_floor < 1.0 and action_velocity_static_prior is not None:
                action_velocity_combined_delta = action_velocity_combined_delta * (
                    action_velocity_static_prior.float() + static_floor * (1.0 - action_velocity_static_prior.float())
                )
            action_velocity_delta_centered = action_velocity_combined_delta
            pred = base_full.float() + action_velocity_combined_delta
        elif self.cfg.output_mode == "context_residual_mask_velocity":
            velocity_x = self.velocity_blocks(x)
            velocity = self.velocity_out(velocity_x)
            velocity_delta = velocity
            if velocity_delta.shape[2] > 0:
                velocity_delta = velocity_delta.clone()
                velocity_delta[:, :, 0] = 0
            velocity_residual = torch.cumsum(velocity_delta, dim=2)
            if motion_prior is not None and float(self.cfg.motion_residual_boost) > 0:
                boost = self._resize_video(motion_prior, residual.shape[2], residual.shape[3], residual.shape[4])
                boost = 1.0 + float(self.cfg.motion_residual_boost) * boost.to(dtype=residual.dtype)
                velocity_residual = boost * velocity_residual
            residual_anchor = residual[:, :, :1].expand_as(residual)
            anchor_mask = mask[:, :, :1].expand_as(mask)
            velocity_mask = mask
            if motion_prior is not None and float(self.cfg.velocity_motion_prior_weight) > 0:
                prior = self._resize_video(motion_prior, mask.shape[2], mask.shape[3], mask.shape[4])
                prior = prior.to(dtype=mask.dtype).clamp(0.0, 1.0)
                power = max(float(self.cfg.velocity_motion_prior_power), 1e-6)
                if abs(power - 1.0) > 1e-6:
                    prior = prior.pow(power)
                prior = (float(self.cfg.velocity_motion_prior_weight) * prior).clamp(0.0, 1.0)
                floor = float(self.cfg.velocity_mask_floor)
                floor = min(max(floor, 0.0), 1.0)
                velocity_mask = mask * (floor + (1.0 - floor) * prior)
            pred = base + residual_scale * (
                anchor_mask * residual_anchor
                + velocity_mask * float(self.cfg.velocity_scale) * velocity_residual
            )
        else:
            if motion_prior is not None and float(self.cfg.motion_residual_boost) > 0:
                boost = self._resize_video(motion_prior, residual.shape[2], residual.shape[3], residual.shape[4])
                boost = 1.0 + float(self.cfg.motion_residual_boost) * boost.to(dtype=residual.dtype)
                pred = base + residual_scale * boost * mask * residual
            else:
                pred = base + residual_scale * mask * residual
        if rough_delta_added is not None:
            pred = pred + rough_delta_added
        if return_components:
            out = {
                "latents": pred,
                "base_latents": base,
                "residual": residual,
                "mask": mask,
                "mask_logits": logits,
            }
            if learned_mask is not mask:
                out["learned_mask"] = learned_mask
                if foreground_write_alpha is not None:
                    out["foreground_write_alpha"] = foreground_write_alpha
                    out["foreground_visible_alpha"] = foreground_visible_alpha
                    out["foreground_write_alpha_rgb"] = foreground_write_alpha.permute(0, 2, 1, 3, 4).contiguous()
                    out["foreground_visible_alpha_rgb"] = foreground_visible_alpha.permute(0, 2, 1, 3, 4).contiguous()
                    out["foreground_write_logits"] = foreground_write_logits
                    out["foreground_visible_logits"] = foreground_visible_logits
                    out["mask_logits"] = foreground_write_logits
                    out["carrier_write_mask"] = mask
                    if foreground_prior is not None:
                        out["foreground_prior"] = foreground_prior
            if region_prior is not None:
                out["motion_region_prior_latent"] = region_prior
                out["motion_region_prior"] = region_prior.permute(0, 2, 1, 3, 4).contiguous()
            if temporal_weights is not None:
                out["temporal_weights"] = temporal_weights
            if temporal_memory is not None:
                out["temporal_memory"] = temporal_memory
                out["temporal_memory_gate"] = torch.sigmoid(self.temporal_memory_gate_logit)
            if rgb_scaffold_mask is not None:
                out["rgb_scaffold_mask"], out["rgb_scaffold_mask_logits"] = rgb_scaffold_mask
            if rough_delta_added is not None:
                out["rough_latent_delta"] = rough_latent_delta
                out["rough_delta_gate"] = rough_delta_gate
                out["rough_delta_added"] = rough_delta_added
            if carrier_delta is not None:
                out["carrier_delta"] = carrier_delta
            if centered_carrier_delta is not None:
                out["carrier_delta_centered"] = centered_carrier_delta
            if carrier_delta_applied_scale is not None:
                out["carrier_delta_applied_scale"] = carrier_delta_applied_scale
            if velocity is not None:
                out["velocity"] = velocity
                out["velocity_residual"] = velocity_residual
                out["residual_anchor"] = residual_anchor
                out["velocity_mask"] = velocity_mask
            if action_velocity is not None:
                out["action_velocity"] = action_velocity
                out["action_velocity_residual_horizon"] = action_velocity_residual_horizon
                out["action_latent_velocity_residual"] = action_latent_velocity_residual
                out["action_velocity_direct_delta"] = action_velocity_direct_delta
                out["action_velocity_combined_delta"] = action_velocity_combined_delta
                out["action_velocity_delta_centered"] = action_velocity_delta_centered
                if action_velocity_gate is not None:
                    out["action_velocity_gate"] = action_velocity_gate
                if action_velocity_action_gate is not None:
                    out["action_velocity_action_gate"] = action_velocity_action_gate
                if action_velocity_static_mask is not None:
                    out["action_velocity_static_mask"] = action_velocity_static_mask
                if action_velocity_static_prior is not None:
                    out["action_velocity_static_prior"] = action_velocity_static_prior
                if action_velocity_temporal_weights is not None:
                    out["action_velocity_temporal_weights"] = action_velocity_temporal_weights
                if action_basis_residual is not None:
                    out["action_basis_velocity"] = action_basis_velocity
                    out["action_basis_residual_horizon"] = action_basis_residual_horizon
                    out["action_basis_residual"] = action_basis_residual
                    if action_basis_residual_projected is not None:
                        out["action_basis_residual_projected"] = action_basis_residual_projected
                    if action_basis_projection_coeff is not None:
                        out["action_basis_projection_coeff"] = action_basis_projection_coeff
                    if action_basis_temporal_weights is not None:
                        out["action_basis_temporal_weights"] = action_basis_temporal_weights
            if wm_delta_residual is not None:
                out["wm_delta_residual"] = wm_delta_residual
                out["wm_delta_residual_horizon"] = wm_delta_residual_horizon
                if wm_delta_temporal_weights is not None:
                    out["wm_delta_temporal_weights"] = wm_delta_temporal_weights
            if self.cfg.output_mode in direct_blend_modes:
                out["direct_latents"] = residual
            if direct_temporal_delta is not None:
                out["direct_temporal_delta"] = direct_temporal_delta
            return out
        return pred
