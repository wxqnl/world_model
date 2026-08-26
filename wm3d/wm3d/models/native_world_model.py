"""Scalable native WM3D core shared by every model-size profile.

The implementation has no model-size, dataset, robot, or fixed-rate branch.
It operates on two timestamped lanes:

* an action-free native state prior used by the policy; and
* a factual-action dynamics refinement used by explicit world prediction.

This separation makes future-action leakage structurally impossible while
retaining action-conditioned RGB/depth/point prediction.  One grouped action
head serves single-arm, bimanual, mobile-manipulator, and whole-body profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite, isqrt, log
from typing import Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS


NATIVE_WORLD_MODEL_SCHEMA = "wm3d_native_world_model_v2"


def _round_multiple(value: float, multiple: int = 256) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


@dataclass(frozen=True)
class NativeWorldModelConfig:
    """Architecture-only configuration; scale is selected by YAML values."""

    schema: str = NATIVE_WORLD_MODEL_SCHEMA

    # External native representation.
    T: int = 24
    P: int = 144
    K: int = 16
    token_dim: int = 2048
    task_dim: int = 2048
    num_views: int = 3

    # State and policy trunks.
    state_hidden: int = 2560
    state_layers: int = 32
    state_heads: int = 20
    state_ff_mult: float = 2.5
    action_hidden: int = 2048
    action_layers: int = 24
    action_heads: int = 16
    action_ff_mult: float = 8.0 / 3.0
    # A pooled task embedding already enters the policy input. This bounded,
    # zero-initialized FiLM route keeps the task observable at every action
    # layer and at the final visual read without introducing a competing loss.
    policy_task_modulation: bool = False
    # Normalization is a training coordinate system, not dataset identity.  A
    # shared policy that mixes several sources for one embodiment must know
    # which action/state coordinate transform its normalized inputs and target
    # use.  This query-only route makes that calibration explicit without
    # exposing a source id, adding an auxiliary loss, or changing world/RGB.
    policy_calibration_conditioning: bool = False
    bridge_layers_state: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29)
    bridge_heads: int = 16
    dynamics_layers: int = 4
    # Reuse the factual-only refinement stack several times. This deepens
    # action conditioning without adding parameters or exposing future action
    # to the action-free state/policy trunks.
    factual_dynamics_repeats: int = 1
    # A parameter-free, per-horizon residual that preserves the factual action
    # value before cross-attention.  It is applied only to the world branch;
    # the action-free prior and policy branch remain structurally isolated.
    factual_action_residual_scale: float = 0.0
    # Rendering can use a shallower, lower-amplitude pass through the same
    # factual stack. This keeps the token lane strongly action-causal without
    # forcing RGB/geometry to consume an over-refined state. None preserves
    # the historical shared-lane behavior.
    render_factual_dynamics_repeats: Optional[int] = None
    render_factual_action_residual_scale: Optional[float] = None
    # A second parameter-free skip reaches the future appearance query just
    # before token projection.  This prevents the RGB path from erasing a
    # valid factual-action response learned by the geometry state.
    appearance_action_residual_scale: float = 0.0

    # Multi-view input.
    view_hidden: int = 1024
    view_heads: int = 8
    view_ff_mult: float = 2.5

    # Embodiment-aware state/action capacity.  These are padding capacities,
    # not declarations of a single robot or controller rate.
    max_action_groups: int = 8
    max_action_dim: int = 16
    max_state_dim: int = 32
    # Capacity ceilings only.  A cache/batch may use any smaller real S/C;
    # forward slices the learned queries and never computes padded rate slots
    # merely to emulate a global controller frequency.
    max_action_substeps: int = 128
    max_policy_queries: int = 256
    max_group_id: int = 128
    max_embodiments: int = 512
    max_action_semantic_id: int = 64
    max_state_semantic_id: int = 64

    # Continuous-time representation in seconds.
    time_fourier_dim: int = 128
    time_min_period_s: float = 0.01
    time_max_period_s: float = 120.0

    # Optional non-visual observations (force, tactile, lidar, etc.).
    max_aux_tokens: int = 16
    aux_dim: int = 256
    max_aux_type_id: int = 128

    # Explicit native prediction heads.
    rgb_hidden: int = 1280
    rgb_res_blocks: int = 2
    rgb_decode_chunk_size: int = 4
    rgb_size: int = 384
    rgb_decode_indices: tuple[int, ...] = tuple(range(16))
    rgb_context_enabled: bool = False
    rgb_context_residual_scale: float = 0.75
    rgb_context_motion_blend_gain: float = 0.5
    # V7 conditions the context-residual renderer directly on the future
    # command. Keep that route renderer-only: the canonical grouped factual
    # action summary reaches the RGB bottleneck but never the action-free
    # state or policy trunks.
    rgb_context_action_scale: float = 0.0
    # Preserve the last observed RGB as the static carrier, while giving the
    # future-minus-last P256 appearance residual an explicit spatial path at
    # every decoder scale.  This prevents the full-resolution context skips
    # from becoming a complete shortcut around future appearance dynamics.
    rgb_context_appearance_delta_scale: float = 0.0
    geom_hidden: int = 768

    # Optional high-resolution, per-view appearance lane.  The geometry/action
    # trunk keeps using P fused native tokens; this lane retains the frozen
    # encoder's unfused view tokens for rendering only.
    appearance_enabled: bool = False
    appearance_P: int = 256
    appearance_context_frames: int = 4
    appearance_hidden: int = 512
    appearance_layers: int = 2
    appearance_heads: int = 8
    appearance_ff_mult: float = 2.0

    dropout: float = 0.0
    activation_checkpointing: bool = True

    def validate(self) -> None:
        if self.schema != NATIVE_WORLD_MODEL_SCHEMA:
            raise ValueError(
                f"unsupported native model schema {self.schema!r}; "
                f"expected {NATIVE_WORLD_MODEL_SCHEMA!r}"
            )
        if self.T <= 0 or self.K <= 0 or self.P <= 0:
            raise ValueError("T, K, and P must be positive")
        grid = isqrt(self.P)
        if grid * grid != self.P:
            raise ValueError(f"P must be a square token grid, got {self.P}")
        for hidden, heads, name in (
            (self.state_hidden, self.state_heads, "state"),
            (self.action_hidden, self.action_heads, "action"),
            (self.state_hidden, self.bridge_heads, "bridge/state"),
            (self.action_hidden, self.bridge_heads, "bridge/action"),
            (self.view_hidden, self.view_heads, "view"),
        ):
            if hidden <= 0 or heads <= 0 or hidden % heads:
                raise ValueError(
                    f"{name} hidden={hidden} must be positive and divisible by heads={heads}"
                )
        if len(set(self.bridge_layers_state)) != len(self.bridge_layers_state):
            raise ValueError("bridge_layers_state contains duplicates")
        if not isinstance(self.policy_task_modulation, bool):
            raise ValueError("policy_task_modulation must be boolean")
        if not isinstance(self.policy_calibration_conditioning, bool):
            raise ValueError("policy_calibration_conditioning must be boolean")
        if any(
            index < 0 or index >= self.state_layers
            for index in self.bridge_layers_state
        ):
            raise ValueError("bridge layer index is outside state trunk")
        if self.dynamics_layers <= 0:
            raise ValueError("dynamics_layers must be positive")
        if self.factual_dynamics_repeats <= 0:
            raise ValueError("factual_dynamics_repeats must be positive")
        if (
            self.render_factual_dynamics_repeats is not None
            and self.render_factual_dynamics_repeats <= 0
        ):
            raise ValueError("render_factual_dynamics_repeats must be positive")
        if (
            not isfinite(self.factual_action_residual_scale)
            or self.factual_action_residual_scale < 0.0
        ):
            raise ValueError(
                "factual_action_residual_scale must be finite and non-negative"
            )
        if self.render_factual_action_residual_scale is not None and (
            not isfinite(self.render_factual_action_residual_scale)
            or self.render_factual_action_residual_scale < 0.0
        ):
            raise ValueError(
                "render_factual_action_residual_scale must be finite and non-negative"
            )
        if (
            not isfinite(self.appearance_action_residual_scale)
            or self.appearance_action_residual_scale < 0.0
        ):
            raise ValueError(
                "appearance_action_residual_scale must be finite and non-negative"
            )
        if (
            self.rgb_hidden <= 0
            or self.rgb_res_blocks <= 0
            or self.rgb_decode_chunk_size <= 0
        ):
            raise ValueError(
                "rgb_hidden, rgb_res_blocks and rgb_decode_chunk_size must be positive"
            )
        if not self.rgb_decode_indices:
            raise ValueError("rgb_decode_indices cannot be empty")
        if tuple(sorted(set(self.rgb_decode_indices))) != self.rgb_decode_indices:
            raise ValueError("rgb_decode_indices must be unique and increasing")
        if any(index < 0 or index >= self.K for index in self.rgb_decode_indices):
            raise ValueError(
                "rgb_decode_indices must refer to future world-state steps"
            )
        if not isinstance(self.rgb_context_enabled, bool):
            raise ValueError("rgb_context_enabled must be boolean")
        if (
            not isfinite(self.rgb_context_residual_scale)
            or self.rgb_context_residual_scale <= 0.0
        ):
            raise ValueError("rgb_context_residual_scale must be finite and positive")
        if (
            not isfinite(self.rgb_context_motion_blend_gain)
            or self.rgb_context_motion_blend_gain < 0.0
        ):
            raise ValueError(
                "rgb_context_motion_blend_gain must be finite and non-negative"
            )
        if (
            not isfinite(self.rgb_context_action_scale)
            or self.rgb_context_action_scale < 0.0
        ):
            raise ValueError("rgb_context_action_scale must be finite and non-negative")
        if self.rgb_context_action_scale > 0.0 and not self.rgb_context_enabled:
            raise ValueError("rgb_context_action_scale requires rgb_context_enabled")
        if (
            not isfinite(self.rgb_context_appearance_delta_scale)
            or self.rgb_context_appearance_delta_scale < 0.0
        ):
            raise ValueError(
                "rgb_context_appearance_delta_scale must be finite and non-negative"
            )
        if self.rgb_context_appearance_delta_scale > 0.0 and (
            not self.rgb_context_enabled or not self.appearance_enabled
        ):
            raise ValueError(
                "rgb_context_appearance_delta_scale requires context RGB and appearance"
            )
        if not isinstance(self.appearance_enabled, bool):
            raise ValueError("appearance_enabled must be boolean")
        if self.appearance_enabled:
            appearance_grid = isqrt(self.appearance_P)
            if appearance_grid * appearance_grid != self.appearance_P:
                raise ValueError("appearance_P must be a square token grid")
            if not 0 < self.appearance_context_frames <= self.T:
                raise ValueError("appearance_context_frames must lie within T")
            if (
                self.appearance_hidden <= 0
                or self.appearance_layers <= 0
                or self.appearance_heads <= 0
                or self.appearance_hidden % self.appearance_heads
                or self.appearance_ff_mult <= 0
            ):
                raise ValueError("appearance capacity/head fields are invalid")
            if self.appearance_P < self.P:
                raise ValueError("appearance_P cannot be smaller than geometry P")
            if appearance_grid > self.rgb_size:
                raise ValueError("appearance grid cannot exceed RGB output size")
        for name in (
            "max_action_groups",
            "max_action_dim",
            "max_state_dim",
            "max_action_substeps",
            "max_policy_queries",
            "max_group_id",
            "max_embodiments",
            "max_action_semantic_id",
            "max_state_semantic_id",
            "max_aux_tokens",
            "aux_dim",
            "max_aux_type_id",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.time_fourier_dim <= 0 or self.time_fourier_dim % 2:
            raise ValueError("time_fourier_dim must be a positive even integer")
        if not 0 < self.time_min_period_s < self.time_max_period_s:
            raise ValueError("time period range must satisfy 0 < min < max")
        if self.rgb_size % grid:
            raise ValueError("rgb_size must be divisible by sqrt(P)")
        ratio = self.rgb_size // grid
        if ratio <= 0 or ratio & (ratio - 1):
            raise ValueError("rgb_size / sqrt(P) must be a power of two")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value_fp32 = value.float()
        normalized = value_fp32 * torch.rsqrt(
            value_fp32.square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=value.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float, dropout: float):
        super().__init__()
        inner = _round_multiple(dim * mult)
        self.gate_up = nn.Linear(dim, inner * 2, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(value).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * up))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        value: torch.Tensor,
        *,
        is_causal: bool = False,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, length, _ = value.shape
        query, key, item = self.qkv(value).chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        item = item.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        result = F.scaled_dot_product_attention(
            query,
            key,
            item,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and allowed_mask is None,
        )
        return self.out(result.transpose(1, 2).reshape(batch, length, self.dim))


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.head_dim = query_dim // heads
        self.query = nn.Linear(query_dim, query_dim, bias=False)
        self.key_value = nn.Linear(context_dim, query_dim * 2, bias=False)
        self.out = nn.Linear(query_dim, query_dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query_value: torch.Tensor,
        context: torch.Tensor,
        *,
        allowed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_length, _ = query_value.shape
        context_length = context.shape[1]
        query = (
            self.query(query_value)
            .view(batch, query_length, self.heads, self.head_dim)
            .transpose(1, 2)
        )
        key, value = self.key_value(context).chunk(2, dim=-1)
        key = key.view(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, context_length, self.heads, self.head_dim).transpose(
            1, 2
        )
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(
            result.transpose(1, 2).reshape(batch, query_length, self.query_dim)
        )


class ContinuousTimeEmbedding(nn.Module):
    """Fourier time encoding parameterized in physical seconds."""

    def __init__(self, output_dim: int, cfg: NativeWorldModelConfig):
        super().__init__()
        half = cfg.time_fourier_dim // 2
        log_min_frequency = log(1.0 / cfg.time_max_period_s)
        log_max_frequency = log(1.0 / cfg.time_min_period_s)
        self._log_min_frequency = log_min_frequency
        self._log_max_frequency = log_max_frequency
        self._half = half
        frequencies = torch.exp(
            torch.linspace(log_min_frequency, log_max_frequency, half)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.proj = nn.Sequential(
            nn.Linear(cfg.time_fourier_dim + 1, output_dim, bias=False),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim, bias=False),
        )

    def reset_parameters(self) -> None:
        # Linear children initialize their own parameters.  This method owns
        # only the persistent Fourier buffer, which must be reconstructed after
        # FSDP2 meta-device materialization.
        values = torch.exp(
            torch.linspace(
                self._log_min_frequency,
                self._log_max_frequency,
                self._half,
                device=self.frequencies.device,
                dtype=self.frequencies.dtype,
            )
        )
        self.frequencies.copy_(values)

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(seconds):
            seconds = seconds.float()
        angles = seconds[..., None] * self.frequencies.to(dtype=seconds.dtype)
        features = torch.cat(
            (seconds[..., None], torch.sin(angles), torch.cos(angles)), dim=-1
        )
        return self.proj(features)


class MultiViewTokenFuser(nn.Module):
    """Fuse cameras only along the view axis at each time/patch coordinate."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.num_views = cfg.num_views
        self.in_proj = nn.Linear(cfg.token_dim, cfg.view_hidden, bias=False)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.view_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        self.attn_norm = RMSNorm(cfg.view_hidden)
        self.attn = SelfAttention(cfg.view_hidden, cfg.view_heads, cfg.dropout)
        self.ff_norm = RMSNorm(cfg.view_hidden)
        self.ff = SwiGLU(cfg.view_hidden, cfg.view_ff_mult, cfg.dropout)
        self.gate = nn.Linear(cfg.view_hidden, 1, bias=False)
        self.out_proj = nn.Linear(cfg.view_hidden, cfg.state_hidden, bias=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.view_embed, std=0.02)

    def forward(self, tokens: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, views, patches, _ = tokens.shape
        if views != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {views}")
        if tuple(view_mask.shape) != (batch, frames, views):
            raise ValueError("view_mask must be [B,T,V]")
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every context frame must contain at least one real view")
        value = self.in_proj(tokens) + self.view_embed
        value = value.permute(0, 1, 3, 2, 4).reshape(
            batch * frames * patches, views, -1
        )
        valid = view_mask[:, :, None, :].expand(batch, frames, patches, views)
        valid = valid.reshape(batch * frames * patches, views)
        value = value + self.attn(
            self.attn_norm(value), allowed_mask=valid[:, None, None, :]
        )
        value = value + self.ff(self.ff_norm(value))
        logits = self.gate(value).squeeze(-1).masked_fill(~valid, float("-inf"))
        fused = (value * logits.softmax(dim=-1)[..., None]).sum(dim=1)
        return self.out_proj(fused.view(batch, frames, patches, -1))


class FactorizedStateBlock(nn.Module):
    """Spatial attention per state time and causal temporal attention per patch."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        dim = cfg.state_hidden
        self.spatial_norm = RMSNorm(dim)
        self.spatial = SelfAttention(dim, cfg.state_heads, cfg.dropout)
        self.temporal_norm = RMSNorm(dim)
        self.temporal = SelfAttention(dim, cfg.state_heads, cfg.dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, cfg.state_ff_mult, cfg.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, dim = value.shape
        spatial = value.reshape(batch * frames, patches, dim)
        spatial = spatial + self.spatial(self.spatial_norm(spatial))
        value = spatial.view(batch, frames, patches, dim)
        temporal = value.transpose(1, 2).reshape(batch * patches, frames, dim)
        temporal = temporal + self.temporal(
            self.temporal_norm(temporal), is_causal=True
        )
        value = temporal.view(batch, patches, frames, dim).transpose(1, 2)
        return value + self.ff(self.ff_norm(value))


class GroupedSignalEncoder(nn.Module):
    """Encode source-native fine commands and explicit coarse effects."""

    def __init__(self, output_dim: int, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.fine_value = nn.Linear(cfg.max_action_dim * 2, output_dim, bias=False)
        self.coarse_value = nn.Linear(cfg.max_action_dim * 2, output_dim, bias=False)
        self.time = ContinuousTimeEmbedding(output_dim, cfg)
        # The non-linearity is deliberately applied *after* joining each
        # command with its recorded timestamp.  An additive value/time
        # embedding followed immediately by a mean would lose their pairing:
        # swapping two commands between two timestamps would produce the same
        # summary.  This residual joint encoder makes the set of
        # (timestamp, command) pairs observable without inventing a fixed-rate
        # action grid.
        self.fine_joint_norm = RMSNorm(output_dim)
        self.fine_joint = nn.Linear(output_dim, output_dim, bias=False)
        self.action_semantic = nn.Embedding(cfg.max_action_semantic_id, output_dim)
        self.group = nn.Embedding(cfg.max_group_id, output_dim)
        self.embodiment = nn.Embedding(cfg.max_embodiments, output_dim)
        self.output_norm = RMSNorm(output_dim)

    def forward(
        self,
        *,
        fine_values: torch.Tensor,
        fine_dim_mask: torch.Tensor,
        fine_dt: torch.Tensor,
        fine_sample_mask: torch.Tensor,
        coarse_values: torch.Tensor,
        coarse_dim_mask: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        group_ids: torch.Tensor,
        group_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if fine_values.shape != fine_dim_mask.shape:
            raise ValueError("fine action values and dimension mask must match")
        groups, substeps, action_dim = fine_values.shape[-3:]
        if groups != cfg.max_action_groups or action_dim != cfg.max_action_dim:
            raise ValueError(
                "fine action group/dimension capacity is incompatible with model"
            )
        if not 0 < substeps <= cfg.max_action_substeps:
            raise ValueError(
                f"fine action substeps {substeps} exceed capacity "
                f"{cfg.max_action_substeps}"
            )
        if tuple(fine_dt.shape) != tuple(fine_values.shape[:-1]):
            raise ValueError("fine_action_dt must be [B,F,G,S]")
        if fine_sample_mask.shape != fine_dt.shape:
            raise ValueError("fine_sample_mask must match fine_action_dt")
        if coarse_values.shape != coarse_dim_mask.shape:
            raise ValueError("coarse action values and dimension mask must match")
        if tuple(coarse_values.shape[-2:]) != (
            cfg.max_action_groups,
            cfg.max_action_dim,
        ):
            raise ValueError("coarse action suffix is incompatible with model capacity")
        batch, frames = fine_values.shape[:2]
        if tuple(action_semantic_ids.shape) != (
            batch,
            cfg.max_action_groups,
            cfg.max_action_dim,
        ):
            raise ValueError("action_semantic_ids must be [B,G,A]")

        fine_pair = torch.cat(
            (
                fine_values * fine_dim_mask.to(dtype=fine_values.dtype),
                fine_dim_mask.to(dtype=fine_values.dtype),
            ),
            dim=-1,
        )
        fine_tokens = self.fine_value(fine_pair) + self.time(fine_dt)
        fine_tokens = fine_tokens + F.silu(
            self.fine_joint(self.fine_joint_norm(fine_tokens))
        )
        real_fine = fine_sample_mask & fine_dim_mask.any(dim=-1)
        fine_weight = real_fine[..., None].to(dtype=fine_tokens.dtype)
        fine_summary = (fine_tokens * fine_weight).sum(dim=3)
        fine_count = fine_weight.sum(dim=3).clamp_min(1.0)
        fine_summary = fine_summary / fine_count

        coarse_pair = torch.cat(
            (
                coarse_values * coarse_dim_mask.to(dtype=coarse_values.dtype),
                coarse_dim_mask.to(dtype=coarse_values.dtype),
            ),
            dim=-1,
        )
        coarse_summary = self.coarse_value(coarse_pair)
        real_coarse = coarse_dim_mask.any(dim=-1)
        fine_present = real_fine.any(dim=3)
        source_count = fine_present.to(torch.int32) + real_coarse.to(torch.int32)
        signal = (
            fine_summary * fine_present[..., None].to(dtype=fine_summary.dtype)
            + coarse_summary * real_coarse[..., None].to(dtype=coarse_summary.dtype)
        ) / source_count.clamp_min(1)[..., None].to(dtype=fine_summary.dtype)

        semantic = self.action_semantic(action_semantic_ids)
        semantic_valid = action_semantic_ids.ne(0)
        semantic = (semantic * semantic_valid[..., None].to(dtype=semantic.dtype)).sum(
            dim=2
        ) / semantic_valid.sum(dim=2).clamp_min(1)[..., None]
        signal = signal + semantic[:, None]
        signal = signal + self.group(group_ids)[:, None]
        signal = signal + self.embodiment(embodiment_ids)[:, None, None]
        valid = group_mask[:, None, :] & source_count.gt(0)
        signal = self.output_norm(signal)
        return signal * group_mask[:, None, :, None].to(signal.dtype), valid


class CurrentStateEncoder(nn.Module):
    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.value = nn.Linear(cfg.max_state_dim * 2, cfg.action_hidden, bias=False)
        self.semantic = nn.Embedding(cfg.max_state_semantic_id, cfg.action_hidden)
        self.group = nn.Embedding(cfg.max_group_id, cfg.action_hidden)
        self.embodiment = nn.Embedding(cfg.max_embodiments, cfg.action_hidden)
        self.norm = RMSNorm(cfg.action_hidden)

    def forward(
        self,
        values: torch.Tensor,
        dim_mask: torch.Tensor,
        semantic_ids: torch.Tensor,
        group_ids: torch.Tensor,
        group_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        expected = (values.shape[0], cfg.max_action_groups, cfg.max_state_dim)
        if tuple(values.shape) != expected or dim_mask.shape != values.shape:
            raise ValueError(f"current state must have shape {expected}")
        if semantic_ids.shape != values.shape:
            raise ValueError("state semantic ids must align to current state")
        group_declares_state = semantic_ids.ne(0).any(dim=-1)
        if not bool((dim_mask.any(dim=-1) | ~group_declares_state | ~group_mask).all()):
            raise ValueError(
                "every group with state semantics requires measured current state"
            )
        if not bool((dim_mask & group_mask[..., None]).any(dim=(1, 2)).all()):
            raise ValueError(
                "every sample requires at least one measured current-state value"
            )
        pair = torch.cat(
            (values * dim_mask.to(dtype=values.dtype), dim_mask.to(dtype=values.dtype)),
            dim=-1,
        )
        token = self.value(pair)
        semantic = self.semantic(semantic_ids)
        semantic_valid = semantic_ids.ne(0)
        semantic = (semantic * semantic_valid[..., None].to(dtype=semantic.dtype)).sum(
            dim=2
        ) / semantic_valid.sum(dim=2).clamp_min(1)[..., None]
        token = (
            token
            + semantic
            + self.group(group_ids)
            + self.embodiment(embodiment_ids)[:, None]
        )
        return self.norm(token) * group_mask[..., None].to(dtype=token.dtype)


class PolicyCalibrationEncoder(nn.Module):
    """Encode the exact control coordinate transform for each policy group.

    The data pipeline z-scores action history, current state and continuous
    action targets with source-specific statistics.  Embodiment/group/semantic
    ids alone cannot identify that numerical coordinate system when several
    sources share the same robot.  This encoder therefore exposes only the
    transform itself: offset, log-scale and real-dimension mask for action and
    state.  It never receives a source id or a future action value.

    A single zero-initialized linear map is sufficient and intentional.  The
    old forward is exactly preserved at initialization, while the existing
    action objective gives the map a non-zero gradient on the first step.  No
    competing calibration loss or tunable loss weight is introduced.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        feature_dim = 3 * (cfg.max_action_dim + cfg.max_state_dim)
        self.weight = nn.Parameter(torch.zeros(cfg.action_hidden, feature_dim))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    @staticmethod
    def _features(
        offset: torch.Tensor,
        scale: torch.Tensor,
        valid: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        if offset.shape != scale.shape or valid.shape != offset.shape:
            raise ValueError(f"{name} calibration tensors must have identical shapes")
        if (
            not bool(torch.isfinite(offset).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
        ):
            raise ValueError(f"{name} calibration statistics are invalid")
        mask = valid.to(dtype=torch.float32)
        # asinh is linear near zero and only logarithmic for large offsets;
        # log-scale represents multiplicative unit changes directly.
        offset_feature = torch.asinh(offset.float()) * mask
        scale_feature = torch.log(scale.float()) * mask
        return torch.cat((offset_feature, scale_feature, mask), dim=-1)

    def forward(
        self,
        action_offset: torch.Tensor,
        action_scale: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        state_offset: torch.Tensor,
        state_scale: torch.Tensor,
        state_semantic_ids: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        batch = action_offset.shape[0]
        if tuple(action_offset.shape) != (
            batch,
            cfg.max_action_groups,
            cfg.max_action_dim,
        ):
            raise ValueError("action calibration must be [B,G,A]")
        if tuple(state_offset.shape) != (
            batch,
            cfg.max_action_groups,
            cfg.max_state_dim,
        ):
            raise ValueError("state calibration must be [B,G,D]")
        if action_semantic_ids.shape != action_offset.shape:
            raise ValueError("action semantics must align with action calibration")
        if state_semantic_ids.shape != state_offset.shape:
            raise ValueError("state semantics must align with state calibration")
        if tuple(group_mask.shape) != (batch, cfg.max_action_groups):
            raise ValueError("group mask must align with policy calibration")
        action_valid = action_semantic_ids.ne(0) & group_mask[..., None]
        state_valid = state_semantic_ids.ne(0) & group_mask[..., None]
        features = torch.cat(
            (
                self._features(
                    action_offset,
                    action_scale,
                    action_valid,
                    name="action",
                ),
                self._features(
                    state_offset,
                    state_scale,
                    state_valid,
                    name="state",
                ),
            ),
            dim=-1,
        )
        encoded = F.linear(features.to(dtype=self.weight.dtype), self.weight)
        return encoded * group_mask[..., None].to(dtype=encoded.dtype)


class TaskFeatureModulation(nn.Module):
    """Bounded FiLM whose exact initialization is the identity function.

    The shared task projection establishes the semantic basis once. Each
    policy layer learns only bounded feature-wise scale/shift gates, avoiding
    both a large per-layer projection and an extra objective weight. A token
    mask restricts the route to policy queries; history remains physical
    evidence selected by a task-conditioned query rather than being relabeled
    with the task at every layer.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.task_norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.scale)
        nn.init.zeros_(self.shift)

    def forward(
        self,
        value: torch.Tensor,
        task_condition: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, dim = task_condition.shape
        if value.shape[0] != batch or value.shape[-1] != dim:
            raise ValueError("task condition must align with feature batch/dimension")
        if token_mask.shape != value.shape[:-1]:
            raise ValueError("task modulation mask must align with feature tokens")
        task = torch.tanh(self.task_norm(task_condition))
        task = task.view(batch, *((1,) * (value.ndim - 2)), dim)
        scale = torch.tanh(self.scale).view(*((1,) * (value.ndim - 1)), dim)
        shift = torch.tanh(self.shift).view(*((1,) * (value.ndim - 1)), dim)
        conditioned = value * (1.0 + task * scale) + task * shift
        return torch.where(token_mask[..., None], conditioned, value)


class ActionBlock(nn.Module):
    """Factorized temporal/group attention for variable-rate embodiments.

    Reusing one attention projection for both axes keeps the parameter budget
    stable while changing compute from ``O((steps*groups)^2)`` to
    ``O(groups*steps^2 + steps*groups^2)``.  Timestamp masks, rather than a
    nominal controller frequency, define causality on both axes.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        dim = cfg.action_hidden
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, cfg.action_heads, cfg.dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, cfg.action_ff_mult, cfg.dropout)
        self.attn_task_modulation: Optional[TaskFeatureModulation] = (
            TaskFeatureModulation(dim) if cfg.policy_task_modulation else None
        )
        self.ff_task_modulation: Optional[TaskFeatureModulation] = (
            TaskFeatureModulation(dim) if cfg.policy_task_modulation else None
        )

    def forward(
        self,
        value: torch.Tensor,
        action_times: torch.Tensor,
        token_mask: torch.Tensor,
        task_condition: torch.Tensor,
        task_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, groups, dim = value.shape
        if tuple(action_times.shape) != (batch, steps, groups):
            raise ValueError("action_times must align with [B,S,G] action tokens")
        if token_mask.shape != action_times.shape:
            raise ValueError("action token mask must align with action times")
        if tuple(task_condition.shape) != (batch, dim):
            raise ValueError("task_condition must be [B,D] for action tokens")
        if task_token_mask.shape != token_mask.shape:
            raise ValueError("task token mask must align with action token mask")
        if bool((task_token_mask & ~token_mask).any()):
            raise ValueError("task modulation cannot enable an invalid action token")

        # Each group first follows its own physical timeline.
        temporal = value.transpose(1, 2).reshape(batch * groups, steps, dim)
        temporal_input = self.attn_norm(value)
        if self.attn_task_modulation is not None:
            temporal_input = self.attn_task_modulation(
                temporal_input, task_condition, task_token_mask
            )
        temporal_input = temporal_input.transpose(1, 2).reshape(
            batch * groups, steps, dim
        )
        temporal_times = action_times.transpose(1, 2).reshape(batch * groups, steps)
        temporal_valid = token_mask.transpose(1, 2).reshape(batch * groups, steps)
        temporal_allowed = (
            temporal_times[:, :, None] + 1.0e-7 >= temporal_times[:, None, :]
        ) & temporal_valid[:, None, :]
        temporal = temporal + self.attn(
            temporal_input,
            allowed_mask=temporal_allowed[:, None],
        )
        value = temporal.view(batch, groups, steps, dim).transpose(1, 2)

        # At a common sequence slot, groups exchange only information whose
        # physical timestamp is not later than the receiving token.
        grouped = value.reshape(batch * steps, groups, dim)
        group_times = action_times.reshape(batch * steps, groups)
        group_valid = token_mask.reshape(batch * steps, groups)
        group_allowed = (
            group_times[:, :, None] + 1.0e-7 >= group_times[:, None, :]
        ) & group_valid[:, None, :]
        group_input = self.attn_norm(value)
        if self.attn_task_modulation is not None:
            group_input = self.attn_task_modulation(
                group_input, task_condition, task_token_mask
            )
        grouped = grouped + self.attn(
            group_input.reshape(batch * steps, groups, dim),
            allowed_mask=group_allowed[:, None],
        )
        value = grouped.view(batch, steps, groups, dim)
        ff_input = self.ff_norm(value)
        if self.ff_task_modulation is not None:
            ff_input = self.ff_task_modulation(
                ff_input, task_condition, task_token_mask
            )
        value = value + self.ff(ff_input)
        return value * token_mask[..., None].to(dtype=value.dtype)


class StateActionBridge(nn.Module):
    """Bridge action-free native state and policy latents without labels."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.history_steps = cfg.T
        self.state_norm = RMSNorm(cfg.state_hidden)
        self.action_norm = RMSNorm(cfg.action_hidden)
        self.action_reads_state = CrossAttention(
            cfg.action_hidden, cfg.state_hidden, cfg.bridge_heads, cfg.dropout
        )
        self.state_reads_action = CrossAttention(
            cfg.state_hidden, cfg.action_hidden, cfg.bridge_heads, cfg.dropout
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, state_steps, patches, _ = state.shape
        action_steps, groups = action.shape[1:3]
        state_summary = state.mean(dim=2)
        action_flat = action.reshape(batch, action_steps * groups, -1)
        valid_flat = action_mask.reshape(batch, action_steps * groups)
        action_update = self.action_reads_state(
            self.action_norm(action_flat), self.state_norm(state_summary)
        )
        action_flat = (action_flat + action_update) * valid_flat[..., None].to(
            dtype=action_flat.dtype
        )
        action = action_flat.view(batch, action_steps, groups, -1)
        # World prior may read already executed history, but learned future
        # policy queries and current-state-conditioned query latents cannot
        # write back into the action-free state branch.
        history_action = action[:, : self.history_steps].reshape(
            batch, self.history_steps * groups, -1
        )
        history_valid = action_mask[:, : self.history_steps].reshape(
            batch, self.history_steps * groups
        )
        state_update = self.state_reads_action(
            self.state_norm(state_summary),
            self.action_norm(history_action),
            allowed_mask=history_valid[:, None, None, :],
        )
        state = state + state_update[:, :state_steps, None, :] / patches**0.5
        return state, action


class DynamicsConditionBlock(nn.Module):
    """Apply factual action effects only after the action-free state prior."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.null_action = nn.Parameter(torch.empty(1, 1, 1, cfg.state_hidden))
        nn.init.normal_(self.null_action, std=0.02)
        self.state_norm = RMSNorm(cfg.state_hidden)
        self.action_norm = RMSNorm(cfg.state_hidden)
        self.cross = CrossAttention(
            cfg.state_hidden, cfg.state_hidden, cfg.state_heads, cfg.dropout
        )
        self.factorized = FactorizedStateBlock(cfg)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.null_action, std=0.02)

    def forward(
        self,
        future_state: torch.Tensor,
        factual_action: torch.Tensor,
        factual_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, patches, dim = future_state.shape
        groups = factual_action.shape[2]
        null = self.null_action.expand(batch, horizon, -1, -1)
        context = torch.cat((null, factual_action), dim=2)
        valid = torch.cat(
            (
                torch.ones(
                    batch, horizon, 1, dtype=torch.bool, device=factual_mask.device
                ),
                factual_mask,
            ),
            dim=2,
        )
        query = future_state.reshape(batch * horizon, patches, dim)
        context = context.reshape(batch * horizon, groups + 1, dim)
        valid = valid.reshape(batch * horizon, groups + 1)
        update = self.cross(
            self.state_norm(query),
            self.action_norm(context),
            allowed_mask=valid[:, None, None, :],
        )
        future_state = (query + update).view(batch, horizon, patches, dim)
        return self.factorized(future_state)


class FactorizedAppearanceBlock(nn.Module):
    """Preserve view identity while mixing spatial and causal temporal context."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        dim = cfg.appearance_hidden
        self.spatial_norm = RMSNorm(dim)
        self.spatial = SelfAttention(dim, cfg.appearance_heads, cfg.dropout)
        self.temporal_norm = RMSNorm(dim)
        self.temporal = SelfAttention(dim, cfg.appearance_heads, cfg.dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, cfg.appearance_ff_mult, cfg.dropout)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, views, patches, dim = value.shape
        if tuple(mask.shape) != (batch, steps, views, patches):
            raise ValueError("appearance mask must be [B,S,V,P]")

        spatial = value.reshape(batch * steps * views, patches, dim)
        spatial_valid = mask.reshape(batch * steps * views, patches)
        spatial = spatial + self.spatial(
            self.spatial_norm(spatial),
            allowed_mask=spatial_valid[:, None, None, :],
        )
        value = spatial.view(batch, steps, views, patches, dim)
        value = value * mask[..., None].to(dtype=value.dtype)

        temporal = value.permute(0, 2, 3, 1, 4).reshape(
            batch * views * patches, steps, dim
        )
        temporal_valid = mask.permute(0, 2, 3, 1).reshape(
            batch * views * patches, steps
        )
        causal = torch.ones(steps, steps, dtype=torch.bool, device=value.device).tril()
        allowed = causal[None, None] & temporal_valid[:, None, None, :]
        temporal = temporal + self.temporal(
            self.temporal_norm(temporal), allowed_mask=allowed
        )
        temporal = temporal + self.ff(self.ff_norm(temporal))
        value = temporal.view(batch, views, patches, steps, dim).permute(0, 3, 1, 2, 4)
        return value * mask[..., None].to(dtype=value.dtype)


class PerViewAppearanceDynamics(nn.Module):
    """Predict unfused future view latents from view history and 3D dynamics."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.geometry_grid = isqrt(cfg.P)
        self.appearance_grid = isqrt(cfg.appearance_P)
        self.input = nn.Linear(cfg.token_dim, cfg.appearance_hidden, bias=False)
        self.geometry = nn.Linear(cfg.state_hidden, cfg.appearance_hidden, bias=False)
        self.time = ContinuousTimeEmbedding(cfg.appearance_hidden, cfg)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.appearance_hidden)
        )
        self.patch_embed = nn.Parameter(
            torch.empty(1, 1, 1, cfg.appearance_P, cfg.appearance_hidden)
        )
        self.future_seed = nn.Parameter(
            torch.empty(1, cfg.K, 1, cfg.appearance_P, cfg.appearance_hidden)
        )
        for parameter in (self.view_embed, self.patch_embed, self.future_seed):
            nn.init.normal_(parameter, std=0.02)
        blocks: tuple[nn.Module, ...] = tuple(
            FactorizedAppearanceBlock(cfg) for _ in range(cfg.appearance_layers)
        )
        if cfg.activation_checkpointing:
            blocks = tuple(checkpoint_wrapper(block) for block in blocks)
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.appearance_hidden)
        self.output = nn.Linear(cfg.appearance_hidden, cfg.token_dim, bias=False)

    def reset_parameters(self) -> None:
        for parameter in (self.view_embed, self.patch_embed, self.future_seed):
            nn.init.normal_(parameter, std=0.02)

    def _upsample_geometry(self, future_state: torch.Tensor) -> torch.Tensor:
        batch, horizon, patches, _ = future_state.shape
        if (horizon, patches) != (self.cfg.K, self.cfg.P):
            raise ValueError(
                "future geometry shape is incompatible with appearance lane"
            )
        value = self.geometry(future_state)
        value = value.reshape(
            batch * horizon,
            self.geometry_grid,
            self.geometry_grid,
            self.cfg.appearance_hidden,
        ).permute(0, 3, 1, 2)
        value = F.interpolate(
            value.float(),
            size=(self.appearance_grid, self.appearance_grid),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=future_state.dtype)
        return value.permute(0, 2, 3, 1).reshape(
            batch, horizon, self.cfg.appearance_P, self.cfg.appearance_hidden
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        future_state: torch.Tensor,
        world_times_s: torch.Tensor,
        future_mask: Optional[torch.Tensor] = None,
        factual_action_summary: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        expected = (
            context_tokens.shape[0],
            cfg.appearance_context_frames,
            cfg.num_views,
            cfg.appearance_P,
            cfg.token_dim,
        )
        if tuple(context_tokens.shape) != expected:
            raise ValueError(f"appearance context must be {expected}")
        if context_mask.shape != context_tokens.shape[:-1]:
            raise ValueError("appearance context mask must align to tokens")
        batch = context_tokens.shape[0]
        if tuple(world_times_s.shape) != (batch, cfg.T + cfg.K):
            raise ValueError("appearance world times are incompatible with T/K")
        if future_mask is None:
            future_mask = (
                context_mask.any(dim=1)[:, None].expand(-1, cfg.K, -1, -1).clone()
            )
        elif tuple(future_mask.shape) != (
            batch,
            cfg.K,
            cfg.num_views,
            cfg.appearance_P,
        ):
            raise ValueError("appearance future mask must be [B,K,V,P]")

        context_time = world_times_s[:, cfg.T - cfg.appearance_context_frames : cfg.T]
        future_time = world_times_s[:, cfg.T :]
        action_skip: Optional[torch.Tensor] = None
        if cfg.appearance_action_residual_scale > 0.0:
            if factual_action_summary is None or tuple(
                factual_action_summary.shape
            ) != (batch, cfg.K, cfg.state_hidden):
                raise ValueError(
                    "appearance action residual requires factual [B,K,state_hidden]"
                )
            action_skip = self.geometry(factual_action_summary)[:, :, None, None]
        context = self.input(context_tokens)
        context = context + self.time(context_time)[:, :, None, None]
        context = context + self.view_embed + self.patch_embed

        geometry = self._upsample_geometry(future_state)[:, :, None]
        future = self.future_seed.expand(batch, -1, cfg.num_views, -1, -1)
        future = future + geometry + self.time(future_time)[:, :, None, None]
        future = future + self.view_embed + self.patch_embed
        value = torch.cat((context, future), dim=1)
        mask = torch.cat((context_mask.bool(), future_mask.bool()), dim=1)
        value = value * mask[..., None].to(dtype=value.dtype)
        for block in self.blocks:
            value = block(value, mask)

        last_context = torch.zeros_like(context_tokens[:, 0])
        for index in range(cfg.appearance_context_frames):
            valid = context_mask[:, index, ..., None]
            last_context = torch.where(valid, context_tokens[:, index], last_context)
        future_value = value[:, -cfg.K :]
        if action_skip is not None:
            future_value = future_value + (
                cfg.appearance_action_residual_scale * action_skip
            )
        predicted = self.output(self.norm(future_value)) + last_context[:, None]
        predicted = predicted * future_mask[..., None].to(dtype=predicted.dtype)
        return predicted, future_mask.bool()


class UnifiedActionHead(nn.Module):
    """The sole policy owner; semantic decoding is a deterministic transform."""

    _GRIPPER_IDS = (
        ACTION_SEMANTIC_IDS["absolute_gripper_open01"],
        ACTION_SEMANTIC_IDS["absolute_gripper_close01"],
    )
    _BINARY_IDS = _GRIPPER_IDS + (
        ACTION_SEMANTIC_IDS["binary_contact"],
        ACTION_SEMANTIC_IDS["controller_mode"],
    )

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.norm = RMSNorm(cfg.action_hidden)
        self.output = nn.Linear(cfg.action_hidden, cfg.max_action_dim)

    def forward(
        self,
        query: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        query_mask: torch.Tensor,
        normalization_offset: torch.Tensor,
        normalization_scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Query is [B,C,G,H]; public ABI is group-major [B,G,C,A].
        raw = self.output(self.norm(query)).transpose(1, 2)
        if (
            normalization_offset.shape != action_semantic_ids.shape
            or normalization_scale.shape != action_semantic_ids.shape
            or not bool(torch.isfinite(normalization_offset).all())
            or not bool(torch.isfinite(normalization_scale).all())
            or bool((normalization_scale <= 0).any())
        ):
            raise ValueError("action normalization statistics are invalid")
        semantic = action_semantic_ids[:, :, None, :]
        gripper = torch.zeros_like(semantic, dtype=torch.bool)
        for semantic_id in self._GRIPPER_IDS:
            gripper = gripper | semantic.eq(semantic_id)
        binary = torch.zeros_like(semantic, dtype=torch.bool)
        for semantic_id in self._BINARY_IDS:
            binary = binary | semantic.eq(semantic_id)
        if bool(
            (
                binary
                & (
                    normalization_offset[:, :, None].ne(0)
                    | normalization_scale[:, :, None].ne(1)
                )
            ).any()
        ):
            raise ValueError(
                "gripper/binary/discrete action normalization must be identity"
            )
        physical = (
            raw * normalization_scale[:, :, None] + normalization_offset[:, :, None]
        )
        decoded = torch.where(binary, torch.sigmoid(raw), physical)
        output_mask = query_mask[..., None] & action_semantic_ids[:, :, None, :].ne(0)
        return {
            "policy_action_raw": raw * output_mask.to(dtype=raw.dtype),
            "policy_action_normalized": torch.where(binary, torch.sigmoid(raw), raw)
            * output_mask.to(dtype=raw.dtype),
            "policy_action": decoded * output_mask.to(dtype=decoded.dtype),
            "policy_action_mask": output_mask,
            "policy_gripper_mask": gripper & output_mask,
            "policy_binary_mask": binary & output_mask,
        }


class NativeGeometryHead(nn.Module):
    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.num_views = cfg.num_views
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.geom_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        self.input = nn.Linear(cfg.state_hidden, cfg.geom_hidden, bias=False)
        self.norm = RMSNorm(cfg.geom_hidden)
        self.depth = nn.Linear(cfg.geom_hidden, 1)
        self.point = nn.Linear(cfg.geom_hidden, 3)
        self.confidence = nn.Linear(cfg.geom_hidden, 1)
        self.camera = nn.Sequential(
            nn.Linear(cfg.state_hidden, cfg.geom_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.geom_hidden, cfg.num_views * 9),
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.view_embed, std=0.02)

    def forward(self, future_state: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, horizon, _, _ = future_state.shape
        value = self.input(future_state)[:, :, None] + self.view_embed
        value = self.norm(value)
        return {
            "depth": F.softplus(self.depth(value).squeeze(-1)),
            "point": self.point(value),
            "geometry_confidence": torch.sigmoid(self.confidence(value).squeeze(-1)),
            "camera_pose": self.camera(future_state.mean(dim=2)).view(
                batch, horizon, self.num_views, 9
            ),
        }


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.gelu(self.norm1(value)))
        residual = self.conv2(F.gelu(self.norm2(residual)))
        return value + residual


class NativeRGBImageDecoder(nn.Module):
    """Decode one bounded image chunk from native tokens."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = isqrt(cfg.appearance_P if cfg.appearance_enabled else cfg.P)
        self.geometry_grid = isqrt(cfg.P)
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.token_dim, cfg.rgb_hidden, 1),
            nn.GroupNorm(min(8, cfg.rgb_hidden), cfg.rgb_hidden),
            nn.GELU(),
        )
        self.geometry_stem = (
            nn.Conv2d(cfg.state_hidden, cfg.rgb_hidden, 1, bias=False)
            if cfg.appearance_enabled
            else None
        )
        stages = (cfg.rgb_size // self.grid).bit_length() - 1
        self.decode_grid = cfg.rgb_size // (1 << stages)
        if stages == 5:
            channels = (
                cfg.rgb_hidden,
                cfg.rgb_hidden,
                cfg.rgb_hidden // 2,
                cfg.rgb_hidden // 4,
                cfg.rgb_hidden // 4,
                cfg.rgb_hidden // 8,
            )
        elif stages == 4:
            channels = (
                cfg.rgb_hidden,
                cfg.rgb_hidden,
                cfg.rgb_hidden // 2,
                cfg.rgb_hidden // 4,
                cfg.rgb_hidden // 8,
            )
        else:
            channels = tuple(
                [cfg.rgb_hidden]
                + [
                    max(cfg.rgb_hidden >> index, cfg.rgb_hidden // 8)
                    for index in range(stages)
                ]
            )
        ups: list[nn.Module] = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            stage: list[nn.Module] = [
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.GroupNorm(min(8, output_channels), output_channels),
                nn.GELU(),
            ]
            stage.extend(
                ResidualConvBlock(output_channels) for _ in range(cfg.rgb_res_blocks)
            )
            ups.append(nn.Sequential(*stage))
        self.ups = nn.ModuleList(ups)
        self.output = nn.Conv2d(channels[-1], 3, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        view_embedding: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
        )
        value = self.stem(value) + view_embedding
        if self.geometry_stem is not None:
            if geometry_tokens is None or tuple(geometry_tokens.shape[1:]) != (
                self.cfg.P,
                self.cfg.state_hidden,
            ):
                raise ValueError(
                    "dual-path RGB decoder requires native geometry tokens"
                )
            geometry = geometry_tokens.transpose(1, 2).reshape(
                geometry_tokens.shape[0],
                self.cfg.state_hidden,
                self.geometry_grid,
                self.geometry_grid,
            )
            geometry = self.geometry_stem(geometry)
            if self.geometry_grid != self.grid:
                geometry = F.interpolate(
                    geometry.float(),
                    size=(self.grid, self.grid),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=value.dtype)
            value = value + geometry
        if self.decode_grid != self.grid:
            value = F.interpolate(
                value.float(),
                size=(self.decode_grid, self.decode_grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=tokens.dtype)
        for upsample in self.ups:
            value = upsample(value)
        if tuple(value.shape[-2:]) != (self.cfg.rgb_size, self.cfg.rgb_size):
            value = F.interpolate(
                value.float(),
                size=(self.cfg.rgb_size, self.cfg.rgb_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(dtype=tokens.dtype)
        return torch.sigmoid(self.output(value))


def _rgb_norm_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _RGBConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_rgb_norm_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(_rgb_norm_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _RGBDownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
            nn.GroupNorm(_rgb_norm_groups(output_channels), output_channels),
            nn.SiLU(inplace=True),
            _RGBConvBlock(output_channels, output_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class _RGBUpBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = _RGBConvBlock(input_channels, output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return self.conv(value)


class NativeContextRGBImageDecoder(nn.Module):
    """Preserve observed detail and learn only future RGB changes."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = isqrt(cfg.appearance_P if cfg.appearance_enabled else cfg.P)
        self.geometry_grid = isqrt(cfg.P)
        stages = (cfg.rgb_size // self.grid).bit_length() - 1
        self.decode_grid = cfg.rgb_size // (1 << stages)
        if stages == 5:
            channels = (
                cfg.rgb_hidden,
                cfg.rgb_hidden,
                cfg.rgb_hidden // 2,
                cfg.rgb_hidden // 4,
                cfg.rgb_hidden // 4,
                cfg.rgb_hidden // 8,
            )
        elif stages == 4:
            channels = (
                cfg.rgb_hidden,
                cfg.rgb_hidden,
                cfg.rgb_hidden // 2,
                cfg.rgb_hidden // 4,
                cfg.rgb_hidden // 8,
            )
        else:
            channels = tuple(
                [cfg.rgb_hidden]
                + [
                    max(cfg.rgb_hidden >> index, cfg.rgb_hidden // 8)
                    for index in range(stages)
                ]
            )
        self.token_stem = nn.Sequential(
            nn.Conv2d(cfg.token_dim, channels[0], 1),
            nn.GroupNorm(_rgb_norm_groups(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _RGBConvBlock(channels[0], channels[0]),
        )
        self.appearance_delta_stem = (
            nn.Sequential(
                nn.Conv2d(cfg.token_dim, channels[0], 1, bias=False),
                nn.GroupNorm(_rgb_norm_groups(channels[0]), channels[0], affine=False),
                nn.SiLU(inplace=True),
            )
            if cfg.rgb_context_appearance_delta_scale > 0.0
            else None
        )
        self.appearance_delta_projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels[0], output_channels, 1, bias=False),
                nn.GroupNorm(
                    _rgb_norm_groups(output_channels), output_channels, affine=False
                ),
                nn.SiLU(inplace=True),
            )
            for output_channels in (
                channels[1:] if cfg.rgb_context_appearance_delta_scale > 0.0 else ()
            )
        )
        self.geometry_stem = (
            nn.Conv2d(cfg.state_hidden, channels[0], 1, bias=False)
            if cfg.appearance_enabled
            else None
        )
        self.action_proj = (
            nn.Sequential(
                nn.Linear(cfg.state_hidden, channels[0]),
                nn.SiLU(inplace=True),
                nn.Linear(channels[0], channels[0]),
            )
            if cfg.rgb_context_action_scale > 0.0
            else None
        )
        # V7 conditions the renderer bottleneck on the task directly.  The
        # world/token path also sees the task, but asking that lower-resolution
        # path to preserve all task information was the V8 RGB regression we
        # are explicitly avoiding here.
        self.task_proj = nn.Sequential(
            nn.LayerNorm(cfg.task_dim),
            nn.Linear(cfg.task_dim, channels[0]),
            nn.SiLU(inplace=True),
            nn.Linear(channels[0], channels[0]),
        )
        context_channels = tuple(reversed(channels))
        self.context_stem = _RGBConvBlock(3, context_channels[0])
        self.context_downs = nn.ModuleList(
            _RGBDownBlock(input_channels, output_channels)
            for input_channels, output_channels in zip(
                context_channels, context_channels[1:]
            )
        )
        self.bottleneck_fuse = _RGBConvBlock(
            channels[0] + context_channels[-1], channels[0]
        )
        self.ups = nn.ModuleList(
            _RGBUpBlock(input_channels, output_channels)
            for input_channels, output_channels in zip(channels, channels[1:])
        )
        self.skip_fuses = nn.ModuleList(
            _RGBConvBlock(output_channels * 2, output_channels)
            for output_channels in channels[1:]
        )
        self.head = nn.Conv2d(channels[-1], 7, 3, padding=1)
        self.motion_head = nn.Conv2d(channels[-1], 1, 3, padding=1)
        nn.init.zeros_(self.motion_head.weight)
        nn.init.constant_(self.motion_head.bias, -4.0)

    def forward(
        self,
        tokens: torch.Tensor,
        view_embedding: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor],
        appearance_context_tokens: Optional[torch.Tensor],
        factual_action_summary: Optional[torch.Tensor],
        task_embedding: torch.Tensor,
        context_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(context_rgb.shape[1:]) != (
            3,
            self.cfg.rgb_size,
            self.cfg.rgb_size,
        ):
            raise ValueError("context RGB must be [N,3,rgb_size,rgb_size]")
        value = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
        )
        value = self.token_stem(value) + view_embedding
        appearance_delta: Optional[torch.Tensor] = None
        if self.appearance_delta_stem is not None:
            if appearance_context_tokens is None or tuple(
                appearance_context_tokens.shape
            ) != tuple(tokens.shape):
                raise ValueError(
                    "context RGB appearance tokens must align to future tokens"
                )
            delta = tokens - appearance_context_tokens.to(dtype=tokens.dtype)
            delta = delta.transpose(1, 2).reshape(
                tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
            )
            appearance_delta = self.appearance_delta_stem(delta).to(dtype=value.dtype)
        elif appearance_context_tokens is not None:
            raise ValueError(
                "appearance context was supplied while RGB delta conditioning is disabled"
            )
        if tuple(task_embedding.shape) != (tokens.shape[0], self.cfg.task_dim):
            raise ValueError("context RGB task embedding must be [N,task_dim]")
        task = self.task_proj(task_embedding).to(dtype=value.dtype)
        value = value + task[:, :, None, None]
        if self.action_proj is not None:
            if factual_action_summary is None or tuple(
                factual_action_summary.shape
            ) != (tokens.shape[0], self.cfg.state_hidden):
                raise ValueError(
                    "context RGB factual action summary must be [N,state_hidden]"
                )
            action = self.action_proj(factual_action_summary).to(dtype=value.dtype)
            value = value + (
                float(self.cfg.rgb_context_action_scale) * action[:, :, None, None]
            )
        elif factual_action_summary is not None:
            raise ValueError(
                "factual action summary was supplied while RGB action conditioning is disabled"
            )
        if self.geometry_stem is not None:
            if geometry_tokens is None or tuple(geometry_tokens.shape[1:]) != (
                self.cfg.P,
                self.cfg.state_hidden,
            ):
                raise ValueError(
                    "dual-path RGB decoder requires native geometry tokens"
                )
            geometry = geometry_tokens.transpose(1, 2).reshape(
                geometry_tokens.shape[0],
                self.cfg.state_hidden,
                self.geometry_grid,
                self.geometry_grid,
            )
            geometry = self.geometry_stem(geometry)
            if self.geometry_grid != self.grid:
                geometry = F.interpolate(
                    geometry.float(),
                    size=(self.grid, self.grid),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=value.dtype)
            value = value + geometry
        if self.decode_grid != self.grid:
            value = F.interpolate(
                value.float(),
                size=(self.decode_grid, self.decode_grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=tokens.dtype)
            if appearance_delta is not None:
                appearance_delta = F.interpolate(
                    appearance_delta.float(),
                    size=(self.decode_grid, self.decode_grid),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=tokens.dtype)

        context = context_rgb.to(dtype=value.dtype)
        skips = [self.context_stem(context)]
        for downsample in self.context_downs:
            skips.append(downsample(skips[-1]))
        if skips[-1].shape[-2:] != value.shape[-2:]:
            raise ValueError("context pyramid does not align with RGB token grid")
        value = self.bottleneck_fuse(torch.cat((value, skips[-1]), dim=1))
        delta_scale = float(self.cfg.rgb_context_appearance_delta_scale)
        if appearance_delta is not None:
            value = value + delta_scale * torch.tanh(appearance_delta)
        for stage_index, (upsample, fuse, skip) in enumerate(
            zip(self.ups, self.skip_fuses, reversed(skips[:-1]))
        ):
            value = upsample(value)
            value = fuse(torch.cat((value, skip), dim=1))
            if appearance_delta is not None:
                delta_at_scale = F.interpolate(
                    appearance_delta.float(),
                    size=value.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=value.dtype)
                delta_at_scale = self.appearance_delta_projections[stage_index](
                    delta_at_scale
                )
                value = value + delta_scale * torch.tanh(delta_at_scale)

        raw = self.head(value)
        direct = torch.sigmoid(raw[:, :3])
        residual = torch.tanh(raw[:, 3:6]) * float(self.cfg.rgb_context_residual_scale)
        motion_logit = self.motion_head(value)
        motion_hint = torch.sigmoid(motion_logit)
        blend = torch.sigmoid(raw[:, 6:7])
        if self.cfg.rgb_context_motion_blend_gain > 0.0:
            blend = torch.clamp(
                blend
                + motion_hint.to(dtype=blend.dtype)
                * float(self.cfg.rgb_context_motion_blend_gain),
                0.0,
                1.0,
            )
        residual_rgb = torch.clamp(context + residual, 0.0, 1.0)
        rgb = blend * direct + (1.0 - blend) * residual_rgb
        return rgb, motion_logit, blend


class NativeRGBDecoder(nn.Module):
    """Restore the V7 native token-to-pixel path with bounded image chunks."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.view_embed = nn.Parameter(torch.empty(cfg.num_views, cfg.rgb_hidden, 1, 1))
        nn.init.normal_(self.view_embed, std=0.02)
        image_decoder: nn.Module
        if cfg.rgb_context_enabled:
            image_decoder = NativeContextRGBImageDecoder(cfg)
        else:
            image_decoder = NativeRGBImageDecoder(cfg)
        if cfg.activation_checkpointing:
            image_decoder = checkpoint_wrapper(image_decoder)
        self.image_decoder = image_decoder

    def reset_parameters(self) -> None:
        nn.init.normal_(self.view_embed, std=0.02)

    def forward(
        self,
        future_tokens: torch.Tensor,
        frame_indices: Optional[Sequence[int]],
        target_view_mask: Optional[torch.Tensor] = None,
        appearance_tokens: Optional[torch.Tensor] = None,
        appearance_context_tokens: Optional[torch.Tensor] = None,
        geometry_state: Optional[torch.Tensor] = None,
        factual_action_summary: Optional[torch.Tensor] = None,
        task_embedding: Optional[torch.Tensor] = None,
        context_rgb: Optional[torch.Tensor] = None,
        context_rgb_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = tuple(
            self.cfg.rgb_decode_indices if frame_indices is None else frame_indices
        )
        if any(index < 0 or index >= future_tokens.shape[1] for index in indices):
            raise ValueError("RGB decode index is outside the future horizon")
        if tuple(future_tokens.shape[2:]) != (self.cfg.P, self.cfg.token_dim):
            raise ValueError("future RGB tokens must end in [P,token_dim]")
        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=future_tokens.device
        )
        if not indices:
            empty = future_tokens.new_empty(
                future_tokens.shape[0],
                0,
                self.cfg.num_views,
                3,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            empty_aux = future_tokens.new_empty(
                future_tokens.shape[0],
                0,
                self.cfg.num_views,
                1,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            return empty, index_tensor, empty_aux, empty_aux
        views = self.cfg.num_views
        expanded_geometry: Optional[torch.Tensor] = None
        expanded_appearance_context: Optional[torch.Tensor] = None
        if self.cfg.appearance_enabled:
            expected = (
                future_tokens.shape[0],
                self.cfg.K,
                views,
                self.cfg.appearance_P,
                self.cfg.token_dim,
            )
            if appearance_tokens is None or tuple(appearance_tokens.shape) != expected:
                raise ValueError(f"appearance RGB tokens must be {expected}")
            if geometry_state is None or tuple(geometry_state.shape[1:]) != (
                self.cfg.K,
                self.cfg.P,
                self.cfg.state_hidden,
            ):
                raise ValueError("dual-path RGB geometry state is incompatible")
            selected = appearance_tokens.index_select(1, index_tensor)
            batch, frames, _, patches, token_dim = selected.shape
            expanded = selected.reshape(batch * frames * views, patches, token_dim)
            geometry = geometry_state.index_select(1, index_tensor)
            geometry = geometry[:, :, None].expand(-1, -1, views, -1, -1)
            expanded_geometry = geometry.reshape(
                batch * frames * views, self.cfg.P, self.cfg.state_hidden
            )
            if self.cfg.rgb_context_appearance_delta_scale > 0.0:
                expected_context_appearance = (
                    batch,
                    views,
                    self.cfg.appearance_P,
                    self.cfg.token_dim,
                )
                if (
                    appearance_context_tokens is None
                    or tuple(appearance_context_tokens.shape)
                    != expected_context_appearance
                ):
                    raise ValueError(
                        "appearance_context_tokens must be "
                        f"{expected_context_appearance}"
                    )
                expanded_appearance_context = (
                    appearance_context_tokens[:, None]
                    .expand(-1, frames, -1, -1, -1)
                    .reshape(batch * frames * views, patches, token_dim)
                )
            elif appearance_context_tokens is not None:
                raise ValueError(
                    "appearance context was supplied while RGB delta conditioning is disabled"
                )
        else:
            if appearance_context_tokens is not None:
                raise ValueError(
                    "appearance context was supplied to a fused-only RGB decoder"
                )
            selected = future_tokens.index_select(1, index_tensor)
            batch, frames, patches, token_dim = selected.shape
            expanded = selected[:, :, None].expand(-1, -1, views, -1, -1)
            expanded = expanded.reshape(batch * frames * views, patches, token_dim)
        view_ids = torch.arange(views, device=future_tokens.device)
        view_ids = view_ids.view(1, 1, views).expand(batch, frames, -1).reshape(-1)
        expanded_action: Optional[torch.Tensor] = None
        if self.cfg.rgb_context_action_scale > 0.0:
            expected_action = (
                batch,
                self.cfg.K,
                self.cfg.state_hidden,
            )
            if (
                factual_action_summary is None
                or tuple(factual_action_summary.shape) != expected_action
            ):
                raise ValueError(f"factual_action_summary must be {expected_action}")
            selected_action = factual_action_summary.index_select(1, index_tensor)
            expanded_action = (
                selected_action[:, :, None]
                .expand(-1, -1, views, -1)
                .reshape(batch * frames * views, self.cfg.state_hidden)
            )
        elif factual_action_summary is not None:
            raise ValueError(
                "factual action summary was supplied while RGB action conditioning is disabled"
            )
        expanded_task: Optional[torch.Tensor] = None
        if self.cfg.rgb_context_enabled:
            if task_embedding is None or tuple(task_embedding.shape) != (
                batch,
                self.cfg.task_dim,
            ):
                raise ValueError(f"task_embedding must be {(batch, self.cfg.task_dim)}")
            expanded_task = (
                task_embedding[:, None, None]
                .expand(-1, frames, views, -1)
                .reshape(batch * frames * views, self.cfg.task_dim)
            )
        elif task_embedding is not None:
            raise ValueError("task embedding was supplied to a non-context renderer")
        expanded_context: Optional[torch.Tensor] = None
        context_valid: Optional[torch.Tensor] = None
        if self.cfg.rgb_context_enabled:
            expected_context = (
                batch,
                views,
                3,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            if context_rgb is None or tuple(context_rgb.shape) != expected_context:
                raise ValueError(f"context_rgb must be {expected_context}")
            if context_rgb_mask is None or tuple(context_rgb_mask.shape) != (
                batch,
                views,
            ):
                raise ValueError("context_rgb_mask must be [B,V]")
            expanded_context = (
                context_rgb[:, None]
                .expand(-1, frames, -1, -1, -1, -1)
                .reshape(
                    batch * frames * views,
                    3,
                    self.cfg.rgb_size,
                    self.cfg.rgb_size,
                )
            )
            context_valid = (
                context_rgb_mask[:, None].expand(-1, frames, -1).reshape(-1).bool()
            )
        elif context_rgb is not None or context_rgb_mask is not None:
            raise ValueError("context RGB was supplied to a non-context renderer")
        if target_view_mask is None:
            valid = torch.ones(
                batch * frames * views, dtype=torch.bool, device=future_tokens.device
            )
        else:
            if tuple(target_view_mask.shape) != (batch, frames, views):
                raise ValueError("target_view_mask must be [B,F,V]")
            valid = target_view_mask.reshape(-1).bool()
        if context_valid is not None:
            valid = valid & context_valid
        # ``image_decoder`` is an FSDP unit.  Every rank must therefore call it
        # the same number of times in the same order.  Chunking only the valid
        # views made that call count data-dependent: a rank with one additional
        # visible camera entered an extra all-gather while its peers had already
        # moved on to gradient reduction, eventually deadlocking NCCL.
        #
        # Decode the fixed dense slot layout and mask invalid slots afterwards.
        # The layout depends only on the sealed batch/model shape, so collective
        # ordering is identical across ranks while invalid RGB outputs and their
        # gradients remain exactly zero.
        dense_indices = torch.arange(
            batch * frames * views, device=future_tokens.device
        )
        decoded_chunks: list[torch.Tensor] = []
        motion_chunks: list[torch.Tensor] = []
        blend_chunks: list[torch.Tensor] = []
        for start in range(
            0, int(dense_indices.numel()), self.cfg.rgb_decode_chunk_size
        ):
            chunk_indices = dense_indices[
                start : start + self.cfg.rgb_decode_chunk_size
            ]
            decoder_inputs = (
                expanded.index_select(0, chunk_indices),
                self.view_embed.index_select(
                    0, view_ids.index_select(0, chunk_indices)
                ),
                (
                    None
                    if expanded_geometry is None
                    else expanded_geometry.index_select(0, chunk_indices)
                ),
            )
            if self.cfg.rgb_context_enabled:
                assert expanded_context is not None
                assert expanded_task is not None
                decoded, motion_logit, blend = self.image_decoder(
                    *decoder_inputs,
                    (
                        None
                        if expanded_appearance_context is None
                        else expanded_appearance_context.index_select(0, chunk_indices)
                    ),
                    (
                        None
                        if expanded_action is None
                        else expanded_action.index_select(0, chunk_indices)
                    ),
                    expanded_task.index_select(0, chunk_indices),
                    expanded_context.index_select(0, chunk_indices),
                )
            else:
                decoded = self.image_decoder(*decoder_inputs)
                motion_logit = decoded.new_zeros(
                    decoded.shape[0], 1, decoded.shape[-2], decoded.shape[-1]
                )
                blend = torch.zeros_like(motion_logit)
            chunk_valid = valid.index_select(0, chunk_indices)[:, None, None, None]
            decoded_chunks.append(decoded * chunk_valid.to(dtype=decoded.dtype))
            motion_chunks.append(
                motion_logit * chunk_valid.to(dtype=motion_logit.dtype)
            )
            blend_chunks.append(blend * chunk_valid.to(dtype=blend.dtype))
        dense = torch.cat(decoded_chunks, dim=0)
        dense_motion = torch.cat(motion_chunks, dim=0)
        dense_blend = torch.cat(blend_chunks, dim=0)
        return (
            dense.view(batch, frames, views, 3, self.cfg.rgb_size, self.cfg.rgb_size),
            index_tensor,
            dense_motion.view(
                batch, frames, views, 1, self.cfg.rgb_size, self.cfg.rgb_size
            ),
            dense_blend.view(
                batch, frames, views, 1, self.cfg.rgb_size, self.cfg.rgb_size
            ),
        )


class NativeWorldModel(nn.Module):
    """One scalable implementation used by both 1B and 5B profiles."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.view_fuser = MultiViewTokenFuser(cfg)
        self.state_time = ContinuousTimeEmbedding(cfg.state_hidden, cfg)
        self.action_time = ContinuousTimeEmbedding(cfg.action_hidden, cfg)
        self.state_space = nn.Parameter(torch.empty(1, 1, cfg.P, cfg.state_hidden))
        self.future_queries = nn.Parameter(
            torch.empty(1, cfg.K, cfg.P, cfg.state_hidden)
        )
        # Query identity comes from physical time, group/embodiment and current
        # state.  A shared seed avoids learning discrete 20Hz-style position
        # slots and makes the capacity ceiling parameter-count independent.
        self.policy_query_seed = nn.Parameter(torch.empty(1, 1, 1, cfg.action_hidden))
        for parameter in (
            self.state_space,
            self.future_queries,
            self.policy_query_seed,
        ):
            nn.init.normal_(parameter, std=0.02)
        self.task_state = nn.Linear(cfg.task_dim, cfg.state_hidden, bias=False)
        self.task_action = nn.Linear(cfg.task_dim, cfg.action_hidden, bias=False)
        self.state_input_norm = RMSNorm(cfg.state_hidden)

        self.history_action = GroupedSignalEncoder(cfg.action_hidden, cfg)
        self.factual_action = GroupedSignalEncoder(cfg.state_hidden, cfg)
        self.current_state = CurrentStateEncoder(cfg)
        self.policy_calibration: Optional[PolicyCalibrationEncoder] = (
            PolicyCalibrationEncoder(cfg)
            if cfg.policy_calibration_conditioning
            else None
        )
        self.aux_value = nn.Linear(cfg.aux_dim, cfg.state_hidden, bias=False)
        self.aux_type = nn.Embedding(cfg.max_aux_type_id, cfg.state_hidden)

        self.state_blocks = self._checkpoint_module_list(
            (FactorizedStateBlock(cfg) for _ in range(cfg.state_layers)),
            enabled=cfg.activation_checkpointing,
        )
        self.action_blocks = self._checkpoint_module_list(
            (ActionBlock(cfg) for _ in range(cfg.action_layers)),
            enabled=cfg.activation_checkpointing,
        )
        self.bridges = self._checkpoint_module_list(
            (StateActionBridge(cfg) for _ in cfg.bridge_layers_state),
            enabled=cfg.activation_checkpointing,
        )
        self.policy_spatial_norm = RMSNorm(cfg.action_hidden)
        self.policy_spatial_cross = CrossAttention(
            cfg.action_hidden, cfg.state_hidden, cfg.action_heads, cfg.dropout
        )
        self.policy_spatial_task_modulation: Optional[TaskFeatureModulation] = (
            TaskFeatureModulation(cfg.action_hidden)
            if cfg.policy_task_modulation
            else None
        )
        self.dynamics_blocks = self._checkpoint_module_list(
            (DynamicsConditionBlock(cfg) for _ in range(cfg.dynamics_layers)),
            enabled=cfg.activation_checkpointing,
        )
        self.state_norm = RMSNorm(cfg.state_hidden)
        self.action_norm = RMSNorm(cfg.action_hidden)
        self.token_output = nn.Linear(cfg.state_hidden, cfg.token_dim, bias=False)
        self.appearance_dynamics: Optional[PerViewAppearanceDynamics] = (
            PerViewAppearanceDynamics(cfg) if cfg.appearance_enabled else None
        )
        self.action_head = UnifiedActionHead(cfg)
        self.geometry_head = NativeGeometryHead(cfg)
        self.rgb_head = NativeRGBDecoder(cfg)

        self._action_steps = [
            (cfg.action_layers * (index + 1) // cfg.state_layers)
            - (cfg.action_layers * index // cfg.state_layers)
            for index in range(cfg.state_layers)
        ]
        self._bridge_by_state_layer = {
            state_index: bridge_index
            for bridge_index, state_index in enumerate(cfg.bridge_layers_state)
        }

    def reset_parameters(self) -> None:
        """Initialize parameters directly owned by the root module.

        Child modules are reset independently by the FSDP2 meta materializer;
        recursively resetting them here would initialize some tensors twice
        and make rank determinism depend on wrapping order.
        """

        for parameter in (
            self.state_space,
            self.future_queries,
            self.policy_query_seed,
        ):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def _checkpoint_module_list(
        modules: Iterable[nn.Module], *, enabled: bool
    ) -> nn.ModuleList:
        """Install non-reentrant checkpoint wrappers before any DP transform.

        FSDP2 applies its input mixed-precision cast at the forward boundary of
        each fully-sharded unit.  The checkpoint wrapper must therefore be the
        unit passed to ``fully_shard``: its saved inputs and recomputed inputs
        are both the already-cast tensors.  Functionally checkpointing an
        already-sharded child instead bypasses that boundary during backward
        recomputation and can compare BF16 saved metadata with FP32 recomputed
        metadata.

        ``checkpoint_wrapper`` keeps state-dict names transparent, so DCP and
        DDP use the identical model/checkpoint contract.  The default strict
        non-reentrant determinism check remains enabled.
        """

        values = tuple(modules)
        if enabled:
            values = tuple(checkpoint_wrapper(module) for module in values)
        return nn.ModuleList(values)

    @staticmethod
    def _run(module: nn.Module, *args: torch.Tensor, enabled: bool):
        # Activation checkpointing is installed structurally in __init__ so
        # this call enters the same wrapper under unwrapped, DDP, and FSDP2
        # execution.  ``enabled`` is retained at call sites as an explicit
        # architecture contract and to avoid two model-size-specific paths.
        del enabled
        return module(*args)

    def _validate_world_times(
        self, world_times_s: torch.Tensor, batch: int
    ) -> torch.Tensor:
        expected = (batch, self.cfg.T + self.cfg.K)
        if tuple(world_times_s.shape) != expected:
            raise ValueError(f"world_times_s must be {expected}")
        if not bool(torch.isfinite(world_times_s).all()):
            raise ValueError("world_times_s contains non-finite values")
        if not bool(torch.diff(world_times_s, dim=1).gt(0).all()):
            raise ValueError("world_times_s must be strictly increasing per sample")
        return world_times_s - world_times_s[:, self.cfg.T - 1 : self.cfg.T]

    def _encode_aux(
        self,
        state: torch.Tensor,
        aux_values: Optional[torch.Tensor],
        aux_mask: Optional[torch.Tensor],
        aux_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if aux_values is None:
            if aux_mask is not None or aux_type_ids is not None:
                raise ValueError("aux mask/type cannot be supplied without aux values")
            return state
        cfg = self.cfg
        expected = (state.shape[0], cfg.T, cfg.max_aux_tokens, cfg.aux_dim)
        if tuple(aux_values.shape) != expected:
            raise ValueError(f"aux_values must be {expected}")
        if aux_mask is None or tuple(aux_mask.shape) != expected[:-1]:
            raise ValueError("aux_mask must align to aux_values")
        if aux_type_ids is None or tuple(aux_type_ids.shape) != expected[:-1]:
            raise ValueError("aux_type_ids must align to aux_values")
        token = self.aux_value(aux_values) + self.aux_type(aux_type_ids)
        weight = aux_mask[..., None].to(dtype=token.dtype)
        summary = (token * weight).sum(dim=2) / weight.sum(dim=2).clamp_min(1.0)
        return torch.cat(
            (state[:, : cfg.T] + summary[:, :, None], state[:, cfg.T :]), dim=1
        )

    def forward(
        self,
        *,
        world_tokens: torch.Tensor,
        view_mask: torch.Tensor,
        world_times_s: torch.Tensor,
        task_embedding: torch.Tensor,
        history_fine_action_values: torch.Tensor,
        history_fine_action_mask: torch.Tensor,
        history_fine_action_dt: torch.Tensor,
        history_fine_sample_mask: torch.Tensor,
        history_coarse_action_values: torch.Tensor,
        history_coarse_action_mask: torch.Tensor,
        future_factual_fine_action_values: torch.Tensor,
        future_factual_fine_action_mask: torch.Tensor,
        future_factual_fine_action_dt: torch.Tensor,
        future_factual_fine_sample_mask: torch.Tensor,
        future_factual_coarse_action_values: torch.Tensor,
        future_factual_coarse_action_mask: torch.Tensor,
        action_group_ids: torch.Tensor,
        action_group_mask: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        current_state_values: torch.Tensor,
        current_state_mask: torch.Tensor,
        state_semantic_ids: torch.Tensor,
        embodiment_ids: torch.Tensor,
        policy_query_dt: torch.Tensor,
        policy_query_mask: torch.Tensor,
        action_normalization_offset: torch.Tensor,
        action_normalization_scale: torch.Tensor,
        state_normalization_offset: Optional[torch.Tensor] = None,
        state_normalization_scale: Optional[torch.Tensor] = None,
        aux_values: Optional[torch.Tensor] = None,
        aux_mask: Optional[torch.Tensor] = None,
        aux_type_ids: Optional[torch.Tensor] = None,
        rgb_frame_indices: Optional[Sequence[int]] = None,
        rgb_view_mask: Optional[torch.Tensor] = None,
        context_rgb: Optional[torch.Tensor] = None,
        context_rgb_mask: Optional[torch.Tensor] = None,
        appearance_context_tokens: Optional[torch.Tensor] = None,
        appearance_context_mask: Optional[torch.Tensor] = None,
        target_appearance_tokens: Optional[torch.Tensor] = None,
        target_appearance_mask: Optional[torch.Tensor] = None,
        appearance_teacher_ratio: float | torch.Tensor = 0.0,
        compute_zero_action_control: bool = False,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        expected_world = (cfg.T, cfg.num_views, cfg.P, cfg.token_dim)
        if tuple(world_tokens.shape[1:]) != expected_world:
            raise ValueError(f"world_tokens suffix must be {expected_world}")
        batch = world_tokens.shape[0]
        relative_world_time = self._validate_world_times(world_times_s, batch)
        if tuple(action_group_ids.shape) != (batch, cfg.max_action_groups):
            raise ValueError("action_group_ids must be [B,G]")
        if action_group_mask.shape != action_group_ids.shape:
            raise ValueError("action_group_mask must align to group ids")
        if not bool(action_group_mask.any(dim=-1).all()):
            raise ValueError("every sample must contain at least one action group")
        if policy_query_dt.ndim != 3 or tuple(policy_query_dt.shape[:2]) != (
            batch,
            cfg.max_action_groups,
        ):
            raise ValueError("policy_query_dt must be [B,G,C]")
        query_count = int(policy_query_dt.shape[2])
        if not 0 < query_count <= cfg.max_policy_queries:
            raise ValueError(
                f"policy query count {query_count} exceeds capacity "
                f"{cfg.max_policy_queries}"
            )
        if policy_query_mask.shape != policy_query_dt.shape:
            raise ValueError("policy_query_mask must align to policy_query_dt")
        if not bool(torch.isfinite(policy_query_dt).all()):
            raise ValueError("policy query times contain non-finite values")
        if bool((policy_query_dt[policy_query_mask] < 0).any()):
            raise ValueError("policy query times must be non-negative from chunk start")
        query_pair_mask = policy_query_mask[:, :, 1:] & policy_query_mask[:, :, :-1]
        query_delta = torch.diff(policy_query_dt, dim=-1)
        if bool((query_delta[query_pair_mask] <= 0).any()):
            raise ValueError(
                "valid policy query times must be strictly increasing per action group"
            )

        context = self.view_fuser(world_tokens, view_mask)
        future = self.future_queries.expand(batch, -1, -1, -1)
        state = torch.cat((context, future), dim=1)
        state = self.state_input_norm(state)
        state = (
            state + self.state_space + self.state_time(relative_world_time)[:, :, None]
        )
        state = state + self.task_state(task_embedding)[:, None, None]
        state = self._encode_aux(state, aux_values, aux_mask, aux_type_ids)

        history, history_valid = self.history_action(
            fine_values=history_fine_action_values,
            fine_dim_mask=history_fine_action_mask,
            fine_dt=history_fine_action_dt,
            fine_sample_mask=history_fine_sample_mask,
            coarse_values=history_coarse_action_values,
            coarse_dim_mask=history_coarse_action_mask,
            action_semantic_ids=action_semantic_ids,
            group_ids=action_group_ids,
            group_mask=action_group_mask,
            embodiment_ids=embodiment_ids,
        )
        current = self.current_state(
            current_state_values,
            current_state_mask,
            state_semantic_ids,
            action_group_ids,
            action_group_mask,
            embodiment_ids,
        )
        calibration: Optional[torch.Tensor] = None
        if self.policy_calibration is not None:
            if (
                state_normalization_offset is None
                or state_normalization_scale is None
            ):
                raise ValueError(
                    "policy calibration requires state normalization statistics"
                )
            calibration = self.policy_calibration(
                action_normalization_offset,
                action_normalization_scale,
                action_semantic_ids,
                state_normalization_offset,
                state_normalization_scale,
                state_semantic_ids,
                action_group_mask,
            )
        query = self.policy_query_seed.expand(
            batch, query_count, cfg.max_action_groups, -1
        )
        query_time = policy_query_dt.transpose(1, 2)
        query = query + self.action_time(query_time)
        action_task = self.task_action(task_embedding)
        query = query + current[:, None] + action_task[:, None, None]
        if calibration is not None:
            query = query + calibration[:, None]
        history = (
            history + self.action_time(relative_world_time[:, : cfg.T])[:, :, None]
        )
        history = history + action_task[:, None, None]
        action = torch.cat((history, query), dim=1)
        history_mask = history_valid & action_group_mask[:, None, :]
        query_token_mask = (
            policy_query_mask.transpose(1, 2) & action_group_mask[:, None]
        )
        action_mask = torch.cat((history_mask, query_token_mask), dim=1)
        task_token_mask = torch.cat(
            (torch.zeros_like(history_mask), query_token_mask), dim=1
        )
        action_times = torch.cat(
            (
                relative_world_time[:, : cfg.T, None].expand(
                    -1, -1, cfg.max_action_groups
                ),
                query_time,
            ),
            dim=1,
        )
        action_index = 0
        for state_index, state_block in enumerate(self.state_blocks):
            state = self._run(state_block, state, enabled=cfg.activation_checkpointing)
            for _ in range(self._action_steps[state_index]):
                action = self._run(
                    self.action_blocks[action_index],
                    action,
                    action_times,
                    action_mask,
                    action_task,
                    task_token_mask,
                    enabled=cfg.activation_checkpointing,
                )
                action_index += 1
            bridge_index = self._bridge_by_state_layer.get(state_index)
            if bridge_index is not None:
                state, action = self._run(
                    self.bridges[bridge_index],
                    state,
                    action,
                    action_mask,
                    enabled=cfg.activation_checkpointing,
                )
        while action_index < len(self.action_blocks):
            action = self._run(
                self.action_blocks[action_index],
                action,
                action_times,
                action_mask,
                action_task,
                task_token_mask,
                enabled=cfg.activation_checkpointing,
            )
            action_index += 1

        # One full-spatial read gives policy queries direct access to predicted
        # native geometry while still reading the action-free branch only.
        prior_state = self.state_norm(state)
        policy_query = action[:, cfg.T :]
        query_flat = policy_query.reshape(
            batch, query_count * cfg.max_action_groups, cfg.action_hidden
        )
        spatial_query = self.policy_spatial_norm(query_flat)
        if self.policy_spatial_task_modulation is not None:
            spatial_query = self.policy_spatial_task_modulation(
                spatial_query,
                action_task,
                query_token_mask.reshape(
                    batch, query_count * cfg.max_action_groups
                ),
            )
        spatial_update = self.policy_spatial_cross(
            spatial_query,
            prior_state.reshape(batch, (cfg.T + cfg.K) * cfg.P, cfg.state_hidden),
        )
        policy_query = (query_flat + spatial_update).view(
            batch, query_count, cfg.max_action_groups, cfg.action_hidden
        )
        policy_query = self.action_norm(policy_query)
        policy_query = policy_query * query_token_mask[..., None].to(policy_query.dtype)

        action_free_future = prior_state[:, cfg.T :]

        def encode_factual(
            fine_values: torch.Tensor,
            coarse_values: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            encoded, encoded_mask = self.factual_action(
                fine_values=fine_values,
                fine_dim_mask=future_factual_fine_action_mask,
                fine_dt=future_factual_fine_action_dt,
                fine_sample_mask=future_factual_fine_sample_mask,
                coarse_values=coarse_values,
                coarse_dim_mask=future_factual_coarse_action_mask,
                action_semantic_ids=action_semantic_ids,
                group_ids=action_group_ids,
                group_mask=action_group_mask,
                embodiment_ids=embodiment_ids,
            )
            summary: Optional[torch.Tensor] = None
            if (
                cfg.factual_action_residual_scale > 0.0
                or cfg.render_factual_action_residual_scale is not None
                or cfg.appearance_action_residual_scale > 0.0
                or cfg.rgb_context_action_scale > 0.0
            ):
                weight = encoded_mask[..., None].to(dtype=encoded.dtype)
                summary = (encoded * weight).sum(dim=2)
                summary = summary / weight.sum(dim=2).clamp_min(1.0)
            return encoded, encoded_mask, summary

        def refine_factual(
            encoded: torch.Tensor,
            encoded_mask: torch.Tensor,
            summary: Optional[torch.Tensor],
            *,
            repeats: int,
            residual_scale: float,
        ) -> torch.Tensor:
            refined = action_free_future
            if residual_scale > 0.0:
                assert summary is not None
                refined = refined + (residual_scale * summary[:, :, None, :])
            for _ in range(repeats):
                for dynamics_block in self.dynamics_blocks:
                    refined = self._run(
                        dynamics_block,
                        refined,
                        encoded,
                        encoded_mask,
                        enabled=cfg.activation_checkpointing,
                    )
            return self.state_norm(refined)

        factual_encoded, factual_encoded_mask, factual_summary = encode_factual(
            future_factual_fine_action_values,
            future_factual_coarse_action_values,
        )
        factual_future = refine_factual(
            factual_encoded,
            factual_encoded_mask,
            factual_summary,
            repeats=cfg.factual_dynamics_repeats,
            residual_scale=cfg.factual_action_residual_scale,
        )
        render_repeats = (
            cfg.factual_dynamics_repeats
            if cfg.render_factual_dynamics_repeats is None
            else cfg.render_factual_dynamics_repeats
        )
        render_residual_scale = (
            cfg.factual_action_residual_scale
            if cfg.render_factual_action_residual_scale is None
            else cfg.render_factual_action_residual_scale
        )
        if (
            render_repeats == cfg.factual_dynamics_repeats
            and render_residual_scale == cfg.factual_action_residual_scale
        ):
            render_future = factual_future
        else:
            render_future = refine_factual(
                factual_encoded,
                factual_encoded_mask,
                factual_summary,
                repeats=render_repeats,
                residual_scale=render_residual_scale,
            )
        zero_action_pred_tokens: Optional[torch.Tensor] = None
        zero_encoded: Optional[torch.Tensor] = None
        zero_encoded_mask: Optional[torch.Tensor] = None
        zero_summary: Optional[torch.Tensor] = None
        if compute_zero_action_control or cfg.rgb_context_action_scale > 0.0:
            zero_encoded, zero_encoded_mask, zero_summary = encode_factual(
                torch.zeros_like(future_factual_fine_action_values),
                torch.zeros_like(future_factual_coarse_action_values),
            )
        rgb_action_summary: Optional[torch.Tensor] = None
        if cfg.rgb_context_action_scale > 0.0:
            assert factual_summary is not None and zero_summary is not None
            # The renderer needs the action value itself, as in V7, rather
            # than the action encoder's large action-independent mixture of
            # mask/time/semantic/group/embodiment context.  Centering against
            # the same-mask zero command preserves those semantics in the
            # world lane while making the direct RGB route exactly zero for a
            # neutral future action.
            rgb_action_summary = factual_summary - zero_summary
        if compute_zero_action_control:
            assert zero_encoded is not None and zero_encoded_mask is not None
            zero_action_future = refine_factual(
                zero_encoded,
                zero_encoded_mask,
                zero_summary,
                repeats=cfg.factual_dynamics_repeats,
                residual_scale=cfg.factual_action_residual_scale,
            )
            zero_action_pred_tokens = self.token_output(zero_action_future)

        action_free_pred_tokens = self.token_output(action_free_future)
        pred_tokens = self.token_output(factual_future)
        render_pred_tokens = (
            pred_tokens
            if render_future is factual_future
            else self.token_output(render_future)
        )
        appearance_for_rgb: Optional[torch.Tensor] = None
        appearance_context_for_rgb: Optional[torch.Tensor] = None
        appearance_ratio = pred_tokens.new_zeros(())
        appearance_pred: Optional[torch.Tensor] = None
        appearance_pred_mask: Optional[torch.Tensor] = None
        if cfg.appearance_enabled:
            if (
                self.appearance_dynamics is None
                or appearance_context_tokens is None
                or appearance_context_mask is None
            ):
                raise ValueError(
                    "dual-path model requires appearance context tokens and mask"
                )
            appearance_context_for_rgb = torch.zeros_like(
                appearance_context_tokens[:, 0]
            )
            for context_index in range(int(appearance_context_tokens.shape[1])):
                appearance_context_for_rgb = torch.where(
                    appearance_context_mask[:, context_index, ..., None].bool(),
                    appearance_context_tokens[:, context_index],
                    appearance_context_for_rgb,
                )
            appearance_pred, appearance_pred_mask = self.appearance_dynamics(
                appearance_context_tokens,
                appearance_context_mask,
                render_future,
                relative_world_time,
                target_appearance_mask,
                factual_summary,
            )
            appearance_ratio = torch.as_tensor(
                appearance_teacher_ratio,
                dtype=appearance_pred.dtype,
                device=appearance_pred.device,
            )
            if appearance_ratio.numel() != 1 or not bool(
                ((appearance_ratio >= 0) & (appearance_ratio <= 1)).all()
            ):
                raise ValueError("appearance teacher ratio must be a scalar in [0,1]")
            if target_appearance_tokens is None:
                if bool(appearance_ratio > 0):
                    raise ValueError(
                        "teacher forcing requires target appearance tokens"
                    )
                appearance_for_rgb = appearance_pred
            else:
                if target_appearance_tokens.shape != appearance_pred.shape:
                    raise ValueError(
                        "target appearance tokens must align to predictions"
                    )
                if (
                    target_appearance_mask is None
                    or target_appearance_mask.shape != appearance_pred.shape[:-1]
                ):
                    raise ValueError("target appearance mask must align to predictions")
                appearance_for_rgb = torch.lerp(
                    appearance_pred,
                    target_appearance_tokens.detach().to(dtype=appearance_pred.dtype),
                    appearance_ratio,
                )
                appearance_for_rgb = appearance_for_rgb * target_appearance_mask[
                    ..., None
                ].to(dtype=appearance_for_rgb.dtype)
        elif any(
            value is not None
            for value in (
                appearance_context_tokens,
                appearance_context_mask,
                target_appearance_tokens,
                target_appearance_mask,
            )
        ):
            raise ValueError("appearance tensors were supplied to a fused-only model")
        output: dict[str, torch.Tensor] = {
            "action_free_native_state": action_free_future,
            "action_free_pred_tokens": action_free_pred_tokens,
            "native_state": factual_future,
            "pred_tokens": pred_tokens,
            "policy_latent": policy_query.transpose(1, 2),
            "world_times_s": world_times_s,
            "policy_query_dt": policy_query_dt,
            "appearance_teacher_ratio": appearance_ratio,
        }
        if zero_action_pred_tokens is not None:
            output["zero_action_pred_tokens"] = zero_action_pred_tokens
        if appearance_pred is not None and appearance_pred_mask is not None:
            output["appearance_pred_tokens"] = appearance_pred
            output["appearance_pred_mask"] = appearance_pred_mask
        output.update(
            self.action_head(
                policy_query,
                action_semantic_ids,
                policy_query_mask,
                action_normalization_offset,
                action_normalization_scale,
            )
        )
        output.update(self.geometry_head(render_future))
        rgb, rgb_indices, rgb_motion_logit, rgb_blend = self._run(
            self.rgb_head,
            render_pred_tokens,
            rgb_frame_indices,
            rgb_view_mask,
            appearance_for_rgb,
            (
                appearance_context_for_rgb
                if cfg.rgb_context_appearance_delta_scale > 0.0
                else None
            ),
            render_future if cfg.appearance_enabled else None,
            rgb_action_summary,
            task_embedding if cfg.rgb_context_enabled else None,
            context_rgb,
            context_rgb_mask,
            enabled=cfg.activation_checkpointing,
        )
        output["rgb"] = rgb
        output["rgb_frame_indices"] = rgb_indices
        if cfg.rgb_context_enabled:
            output["rgb_motion_logit"] = rgb_motion_logit
            output["rgb_blend"] = rgb_blend
        return output

    def iter_fsdp_units(self) -> Iterable[nn.Module]:
        """Yield communication-sized modules for bottom-up FSDP2 wrapping."""

        yield self.view_fuser
        yield from self.state_blocks
        yield from self.action_blocks
        yield from self.bridges
        yield from self.dynamics_blocks
        if self.appearance_dynamics is not None:
            yield from self.appearance_dynamics.blocks
        yield self.rgb_head.image_decoder
        yield self.geometry_head

    def iter_activation_checkpoint_units(self) -> Iterable[nn.Module]:
        """Yield the exact units checkpointed before DDP/FSDP2 wrapping."""

        if not self.cfg.activation_checkpointing:
            return
        yield from self.state_blocks
        yield from self.action_blocks
        yield from self.bridges
        yield from self.dynamics_blocks
        if self.appearance_dynamics is not None:
            yield from self.appearance_dynamics.blocks
        yield self.rgb_head.image_decoder

    def parameter_counts(self) -> dict[str, int]:
        groups: dict[str, nn.Module] = {
            "multiview_fuser": self.view_fuser,
            "state_trunk": self.state_blocks,
            "action_trunk": self.action_blocks,
            "state_action_bridges": self.bridges,
            "dynamics_refinement": self.dynamics_blocks,
            "rgb_head": self.rgb_head,
            "geometry_head": self.geometry_head,
            "action_head": self.action_head,
        }
        if self.appearance_dynamics is not None:
            groups["appearance_dynamics"] = self.appearance_dynamics
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counts["other"] = sum(
            parameter.numel() for parameter in self.parameters()
        ) - sum(counts.values())
        counts["total"] = sum(parameter.numel() for parameter in self.parameters())
        return counts

    def num_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def native_config_from_mapping(mapping: Mapping[str, object]) -> NativeWorldModelConfig:
    """Construct a strict config while accepting YAML lists for tuple fields."""

    valid = {item.name for item in fields(NativeWorldModelConfig)}
    unknown = sorted(set(mapping) - valid)
    if unknown:
        raise ValueError(f"unknown native model config keys: {unknown}")
    values = dict(mapping)
    for key in ("bridge_layers_state", "rgb_decode_indices"):
        if key in values:
            values[key] = tuple(int(item) for item in values[key])
    cfg = NativeWorldModelConfig(**values)
    cfg.validate()
    return cfg
