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
from wm3d.training.native_objective import compose_axis_angle_sequence


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
    # Factual-only bridge placement.  V7 counted encoder layers differently
    # from the factorized policy trunk, so its proven schedule must not mutate
    # the action-free policy execution above.
    factual_v7_bridge_layers_state: tuple[int, ...] = ()
    bridge_heads: int = 16
    dynamics_layers: int = 4
    # Compatibility knob for non-V7 profiles. The proven V7 path uses two
    # independently parameterized layers and exactly one pass; sharing one
    # layer across repeats is not an architectural substitute.
    factual_dynamics_repeats: int = 1
    # A parameter-free, per-horizon residual that preserves the factual action
    # value before cross-attention.  It is applied only to the world branch;
    # the action-free prior and policy branch remain structurally isolated.
    factual_action_residual_scale: float = 0.0
    # V7 exposed the future command before every state encoder layer, then
    # injected it again into the two-layer future decoder.  V8 evaluates that
    # factual path separately so the policy prior remains action-free.
    factual_v7_early_action_conditioning: bool = False
    factual_v7_early_action_scale: float = 0.0
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
    # Select the renderer that is structurally identical to the original V7
    # 60K context-residual U-Net.  The only V8 adaptation is the per-view
    # embedding and the grouped-action summary width at the bottleneck.
    rgb_original_v7_context: bool = False
    # Action-owned RGB transport. The factual P64 state predicts dense motion,
    # the changed-pixel support and a bounded correction. The observed image
    # may reach a future frame only through that predicted transport; it is
    # never exposed as an unwarped U-Net skip or an unconditional copy path.
    rgb_action_owned_transport: bool = False
    # Optional V8 clarity head on top of the proven V7 renderer. It consumes
    # only factual P64 and the final 256x256 decoder feature; a fixed zero-DC
    # high-pass operator removes low frequency before the bounded correction
    # reaches RGB, so it cannot own motion, blend, flow, or context transport.
    rgb_v7_high_frequency_refiner: bool = False
    rgb_v7_high_frequency_channels: int = 16
    rgb_v7_high_frequency_scale: float = 0.0

    # Align the V7 observed frame with motion from the factual P64 dynamics
    # before any context RGB/feature skip enters the future renderer.
    rgb_context_alignment_enabled: bool = False
    # Keep the V7 action-free temporal prior as the RGB transport owner.  The
    # factual world lane remains action-causal, while centered physical action
    # reaches RGB through the renderer-only action conditioner below.  This
    # prevents repeated factual refinement from homogenizing RGB horizons.
    rgb_render_action_free_prior: bool = False
    rgb_context_residual_scale: float = 0.75
    rgb_context_motion_blend_gain: float = 0.5
    # V7 conditions the context-residual renderer directly on the future
    # command. Keep that route renderer-only: the canonical grouped factual
    # action summary reaches the RGB bottleneck but never the action-free
    # state or policy trunks.
    rgb_context_action_scale: float = 0.0
    # Preserve the last observed RGB as the static carrier. P256 contributes a
    # post-transport, high-frequency RGB correction and cannot change the V7
    # motion, flow, visibility or blend owners.
    rgb_context_appearance_delta_scale: float = 0.0
    # V8 core keeps the factual P64 state as the sole motion owner.  A bounded
    # P256 high-frequency residual may sharpen that base image, but it cannot
    # carry a full frame, context copy, flow field, or a second motion path.
    rgb_detail_residual_scale: float = 0.0
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
    appearance_autoregressive_steps: int = 2
    # Keep V7/P64 as the motion owner. The P256 lane predicts only the future
    # detail left after transporting the last observed P256 feature with the
    # teacher backward flow. Zero detail is therefore an exact V7 fallback.
    appearance_flow_aligned_detail: bool = False
    # Predict only the spatial high-pass component of future P256 features
    # directly from the factual future state.  This path is train/serve
    # identical, has no future-target or last-frame input, and is deliberately
    # mutually exclusive with the legacy appearance dynamics paths.
    appearance_state_detail: bool = False
    appearance_detail_dim: int = 256

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
        if len(set(self.factual_v7_bridge_layers_state)) != len(
            self.factual_v7_bridge_layers_state
        ):
            raise ValueError("factual_v7_bridge_layers_state contains duplicates")
        if not isinstance(self.policy_task_modulation, bool):
            raise ValueError("policy_task_modulation must be boolean")
        if not isinstance(self.policy_calibration_conditioning, bool):
            raise ValueError("policy_calibration_conditioning must be boolean")
        if any(
            index < 0 or index >= self.state_layers
            for index in self.bridge_layers_state
        ):
            raise ValueError("bridge layer index is outside state trunk")
        if any(
            index < 0 or index >= self.state_layers
            for index in self.factual_v7_bridge_layers_state
        ):
            raise ValueError("factual V7 bridge layer index is outside state trunk")
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
        if not isinstance(self.factual_v7_early_action_conditioning, bool):
            raise ValueError("factual_v7_early_action_conditioning must be boolean")
        if (
            not isfinite(self.factual_v7_early_action_scale)
            or self.factual_v7_early_action_scale < 0.0
        ):
            raise ValueError(
                "factual_v7_early_action_scale must be finite and non-negative"
            )
        if self.factual_v7_early_action_conditioning:
            if self.factual_v7_early_action_scale <= 0.0:
                raise ValueError(
                    "early factual action conditioning requires a positive scale"
                )
            if not self.bridge_layers_state:
                raise ValueError("V7 factual execution requires at least one bridge")
            if self.factual_v7_bridge_layers_state and len(
                self.factual_v7_bridge_layers_state
            ) != len(self.bridge_layers_state):
                raise ValueError(
                    "V7 factual execution needs one exact layer index per shared bridge"
                )
        elif self.factual_v7_early_action_scale != 0.0:
            raise ValueError(
                "early factual action scale must be zero when conditioning is disabled"
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
        if not isinstance(self.rgb_original_v7_context, bool):
            raise ValueError("rgb_original_v7_context must be boolean")
        if not isinstance(self.rgb_action_owned_transport, bool):
            raise ValueError("rgb_action_owned_transport must be boolean")
        if self.rgb_original_v7_context and self.rgb_action_owned_transport:
            raise ValueError(
                "original V7 RGB and action-owned transport are mutually exclusive"
            )
        if not isinstance(self.rgb_v7_high_frequency_refiner, bool):
            raise ValueError("rgb_v7_high_frequency_refiner must be boolean")
        if self.rgb_v7_high_frequency_channels <= 0:
            raise ValueError("rgb_v7_high_frequency_channels must be positive")
        if (
            not isfinite(self.rgb_v7_high_frequency_scale)
            or self.rgb_v7_high_frequency_scale < 0.0
            or self.rgb_v7_high_frequency_scale > 0.125
        ):
            raise ValueError(
                "rgb_v7_high_frequency_scale must be finite and lie in [0,0.125]"
            )
        if self.rgb_v7_high_frequency_refiner:
            if not (
                self.rgb_original_v7_context or self.rgb_action_owned_transport
            ):
                raise ValueError(
                    "bounded high-frequency refinement requires a factual P64 RGB path"
                )
            if self.rgb_v7_high_frequency_scale <= 0.0:
                raise ValueError(
                    "V7 high-frequency refinement requires a positive bound"
                )
            final_channels = max(32, self.rgb_hidden // 8)
            if (
                self.rgb_v7_high_frequency_channels > final_channels
                or final_channels % self.rgb_v7_high_frequency_channels
            ):
                raise ValueError(
                    "V7 high-frequency channels must divide the final RGB width"
                )
        elif self.rgb_v7_high_frequency_scale != 0.0:
            raise ValueError(
                "V7 high-frequency scale must be zero when the refiner is disabled"
            )

        if self.rgb_original_v7_context:
            if not self.rgb_context_enabled:
                raise ValueError("original V7 RGB requires context RGB")
            if self.rgb_context_alignment_enabled:
                raise ValueError(
                    "original V7 RGB is mutually exclusive with learned alignment"
                )
            if self.appearance_enabled:
                raise ValueError(
                    "original V7 RGB consumes factual P64 directly and has no appearance lane"
                )
            if self.rgb_context_appearance_delta_scale != 0.0:
                raise ValueError(
                    "original V7 RGB cannot use a second appearance correction"
                )
            if self.rgb_detail_residual_scale != 0.0:
                raise ValueError(
                    "original V7 RGB cannot use the direct-decoder detail branch"
                )
            if self.dynamics_layers != 2 or self.factual_dynamics_repeats != 1:
                raise ValueError(
                    "original V7 RGB requires two distinct factual decoder layers"
                )
            if self.factual_action_residual_scale != 1.0:
                raise ValueError(
                    "original V7 RGB requires one unit-scale action query injection"
                )
            if (
                not self.factual_v7_early_action_conditioning
                or self.factual_v7_early_action_scale != 1.0
            ):
                raise ValueError(
                    "original V7 RGB requires unit-scale action conditioning before the state trunk"
                )
            if (
                self.render_factual_dynamics_repeats is not None
                or self.render_factual_action_residual_scale is not None
            ):
                raise ValueError(
                    "original V7 RGB cannot split token and render dynamics"
                )
        if self.rgb_action_owned_transport:
            if not self.rgb_context_enabled:
                raise ValueError("action-owned RGB transport requires context RGB")
            if self.rgb_context_alignment_enabled:
                raise ValueError(
                    "action-owned RGB transport already owns alignment"
                )
            if self.rgb_render_action_free_prior:
                raise ValueError(
                    "action-owned RGB transport requires the factual future state"
                )
            if self.appearance_enabled:
                raise ValueError(
                    "action-owned RGB transport has no independent appearance lane"
                )
            if self.rgb_context_action_scale != 0.0:
                raise ValueError(
                    "action-owned RGB transport must receive future action only through factual P64"
                )
            if self.rgb_context_appearance_delta_scale != 0.0:
                raise ValueError(
                    "action-owned RGB transport forbids a second appearance correction"
                )
            if self.rgb_detail_residual_scale != 0.0:
                raise ValueError(
                    "action-owned RGB transport forbids a second direct detail lane"
                )
            if self.factual_dynamics_repeats != 1:
                raise ValueError(
                    "action-owned RGB transport requires one causal factual StateStream pass"
                )
            if self.factual_action_residual_scale != 1.0:
                raise ValueError(
                    "action-owned RGB transport requires unit factual action injection"
                )
            if (
                self.factual_v7_early_action_conditioning
                or self.factual_v7_early_action_scale != 0.0
                or self.factual_v7_bridge_layers_state
            ):
                raise ValueError(
                    "action-owned RGB transport forbids legacy V7 factual action conditioning"
                )
            if (
                self.render_factual_dynamics_repeats is not None
                or self.render_factual_action_residual_scale is not None
            ):
                raise ValueError(
                    "action-owned RGB transport cannot split token and render dynamics"
                )
        if not isinstance(self.rgb_context_alignment_enabled, bool):
            raise ValueError("rgb_context_alignment_enabled must be boolean")
        if self.rgb_context_alignment_enabled and not self.rgb_context_enabled:
            raise ValueError("RGB context alignment requires context RGB")
        if not isinstance(self.rgb_render_action_free_prior, bool):
            raise ValueError("rgb_render_action_free_prior must be boolean")
        if self.rgb_render_action_free_prior and not self.rgb_context_enabled:
            raise ValueError("action-free RGB prior requires context RGB")
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
        if (
            not isfinite(self.rgb_detail_residual_scale)
            or self.rgb_detail_residual_scale < 0.0
        ):
            raise ValueError(
                "rgb_detail_residual_scale must be finite and non-negative"
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
            if not 0 < self.appearance_autoregressive_steps <= self.K:
                raise ValueError("appearance_autoregressive_steps must lie within K")
            if not isinstance(self.appearance_flow_aligned_detail, bool):
                raise ValueError("appearance_flow_aligned_detail must be boolean")
            if not isinstance(self.appearance_state_detail, bool):
                raise ValueError("appearance_state_detail must be boolean")
            if self.appearance_flow_aligned_detail and (
                not self.rgb_context_enabled
                or not self.rgb_context_alignment_enabled
                or self.rgb_context_appearance_delta_scale <= 0.0
            ):
                raise ValueError(
                    "appearance_flow_aligned_detail requires aligned context RGB "
                    "and positive appearance detail conditioning"
                )
            if self.appearance_state_detail:
                if self.appearance_detail_dim <= 0 or (
                    self.token_dim % self.appearance_detail_dim
                ):
                    raise ValueError(
                        "appearance_detail_dim must positively divide token_dim"
                    )
                if self.appearance_flow_aligned_detail:
                    raise ValueError(
                        "state detail and flow-aligned appearance are mutually exclusive"
                    )
                if self.rgb_context_enabled:
                    raise ValueError(
                        "state detail uses the direct RGB decoder, not context RGB"
                    )
                if self.rgb_render_action_free_prior:
                    raise ValueError(
                        "state detail requires the factual RGB future state"
                    )
                if self.rgb_detail_residual_scale <= 0.0:
                    raise ValueError(
                        "state detail requires a positive RGB detail residual scale"
                    )
                if self.appearance_action_residual_scale != 0.0:
                    raise ValueError(
                        "state detail already consumes the factual state; a second action skip is forbidden"
                    )
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


class _ZeroInitializedLinear(nn.Linear):
    """Linear projection that remains zero after meta-shard materialization."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _ZeroInitializedEmbedding(nn.Embedding):
    """Embedding that remains zero after meta-shard materialization."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class MultiViewTokenFuser(nn.Module):
    """Fuse auxiliary cameras into the head-camera coordinate.

    View zero is the sealed ``head``/anchor view. Each anchor patch queries
    every patch from every valid auxiliary view, matching the original V7
    spatial contract. The output therefore remains an anchor-coordinate P64
    grid instead of averaging unrelated rays with the same patch index.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.num_views = cfg.num_views
        self.original_v7_anchor = (
            cfg.rgb_original_v7_context or cfg.rgb_action_owned_transport
        )
        if self.original_v7_anchor:
            # Preserve the canonical V7 anchor path exactly: the head-camera
            # P64 token reaches the state stream without first being compressed
            # through the auxiliary-view attention width. Auxiliary views are
            # a zero-init residual on top of it, never a replacement for it.
            self.anchor_projection = nn.Linear(
                cfg.token_dim, cfg.state_hidden, bias=True
            )
            self.auxiliary_projection = nn.Linear(
                cfg.token_dim, cfg.view_hidden, bias=False
            )
        else:
            self.in_proj = nn.Linear(cfg.token_dim, cfg.view_hidden, bias=False)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.view_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        self.attn_norm = RMSNorm(cfg.view_hidden)
        self.cross = CrossAttention(
            cfg.view_hidden, cfg.view_hidden, cfg.view_heads, cfg.dropout
        )
        self.ff_norm = RMSNorm(cfg.view_hidden)
        self.ff = SwiGLU(cfg.view_hidden, cfg.view_ff_mult, cfg.dropout)
        if self.original_v7_anchor:
            self.output_projection = _ZeroInitializedLinear(
                cfg.view_hidden, cfg.state_hidden, bias=False
            )
            self.residual_gate = nn.Parameter(torch.ones(1))
            nn.init.zeros_(self.output_projection.weight)
        else:
            self.gate = _ZeroInitializedLinear(cfg.view_hidden, 1, bias=False)
            nn.init.zeros_(self.gate.weight)
            self.out_proj = nn.Linear(
                cfg.view_hidden, cfg.state_hidden, bias=False
            )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.view_embed, std=0.02)
        if self.original_v7_anchor:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.ones_(self.residual_gate)
        else:
            nn.init.zeros_(self.gate.weight)

    def forward(self, tokens: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        batch, frames, views, patches, _ = tokens.shape
        if views != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {views}")
        if tuple(view_mask.shape) != (batch, frames, views):
            raise ValueError("view_mask must be [B,T,V]")
        if not bool(view_mask[:, :, 0].all()):
            raise ValueError("every context frame must contain the head anchor view")
        if self.original_v7_anchor:
            anchor_state = self.anchor_projection(tokens[:, :, 0])
            value = self.auxiliary_projection(tokens) + self.view_embed
        else:
            value = self.in_proj(tokens) + self.view_embed
        anchor = value[:, :, 0].reshape(batch * frames, patches, -1)
        if views == 1:
            dependency_modules = [self.cross, self.ff]
            if self.original_v7_anchor:
                dependency_modules.extend(
                    [self.auxiliary_projection, self.output_projection]
                )
            dependency = sum(
                parameter.reshape(-1)[0] * 0.0
                for module in dependency_modules
                for parameter in module.parameters()
            )
            if self.original_v7_anchor:
                return anchor_state + dependency.to(dtype=anchor_state.dtype)
            fused = anchor + dependency.to(dtype=anchor.dtype)
        else:
            auxiliary = value[:, :, 1:].reshape(
                batch * frames, (views - 1) * patches, -1
            )
            auxiliary_valid = (
                view_mask[:, :, 1:, None]
                .expand(batch, frames, views - 1, patches)
                .reshape(batch * frames, (views - 1) * patches)
            )
            has_auxiliary = auxiliary_valid.any(dim=-1)
            # SDPA cannot consume an all-masked key row. Supply one zero
            # fallback key, then mask its residual back to exact zero.
            safe_valid = auxiliary_valid.clone()
            safe_auxiliary = auxiliary.clone()
            missing = ~has_auxiliary
            if bool(missing.any()):
                safe_valid[missing, 0] = True
                safe_auxiliary[missing, 0] = 0
            residual = self.cross(
                self.attn_norm(anchor),
                self.attn_norm(safe_auxiliary),
                allowed_mask=safe_valid[:, None, None, :],
            )
            residual = residual + self.ff(self.ff_norm(residual))
            residual = residual * has_auxiliary[:, None, None].to(residual.dtype)
            if self.original_v7_anchor:
                residual = self.output_projection(residual).view(
                    batch, frames, patches, -1
                )
                return anchor_state + torch.tanh(self.residual_gate) * residual
            gate = torch.tanh(self.gate.weight.mean())
            fused = anchor + gate * residual
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

    def forward(
        self,
        value: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if value.ndim == 3:
            if token_mask is None or tuple(token_mask.shape) != tuple(value.shape[:2]):
                raise ValueError("full state sequence mask must be [B,L]")
            value = value * token_mask[..., None].to(dtype=value.dtype)
            value = value + self.spatial(
                self.spatial_norm(value),
                allowed_mask=token_mask[:, None, None, :],
            )
            value = value * token_mask[..., None].to(dtype=value.dtype)
            value = value + self.ff(self.ff_norm(value))
            return value * token_mask[..., None].to(dtype=value.dtype)
        if value.ndim != 4:
            raise ValueError("state block expects [B,F,P,D] or [B,L,D]")
        if token_mask is not None:
            raise ValueError("factorized state execution does not take a token mask")
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

    def __init__(
        self,
        output_dim: int,
        cfg: NativeWorldModelConfig,
        *,
        condition_on_normalization: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.condition_on_normalization = bool(condition_on_normalization)
        self.fine_value = nn.Linear(cfg.max_action_dim * 2, output_dim, bias=False)
        # Preserve the information V7 obtained by flattening every fixed-rate
        # substep while still supporting V8's variable-rate windows.  Mean
        # alone makes one repeated delta command indistinguishable from many.
        self.fine_aggregate = (
            nn.Linear(cfg.max_action_dim * 4, output_dim, bias=False)
            if self.condition_on_normalization
            else None
        )
        self.coarse_value = nn.Linear(cfg.max_action_dim * 2, output_dim, bias=False)
        self.normalization = (
            nn.Linear(cfg.max_action_dim * 3, output_dim, bias=False)
            if self.condition_on_normalization
            else None
        )
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
        # The V7 direct anchor was sufficient for its predominantly
        # single-owner action layout.  V8 permits several simultaneous
        # physical owners, so route each group's linear action feature before
        # pooling.  A zero-initialized diagonal gain starts as the exact V7
        # unit map, remains linear in the physical value, and can separate
        # arm/base/controller commands without adding another action head.
        self.direct_group_gain = (
            _ZeroInitializedEmbedding(cfg.max_group_id, output_dim)
            if self.condition_on_normalization
            else None
        )
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
        normalization_offset: Optional[torch.Tensor] = None,
        normalization_scale: Optional[torch.Tensor] = None,
        include_direct_physical: bool = False,
    ) -> tuple[torch.Tensor, ...]:
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

        normalization_token: Optional[torch.Tensor] = None
        if self.condition_on_normalization:
            expected_normalization = (
                batch,
                cfg.max_action_groups,
                cfg.max_action_dim,
            )
            if (
                normalization_offset is None
                or normalization_scale is None
                or tuple(normalization_offset.shape) != expected_normalization
                or tuple(normalization_scale.shape) != expected_normalization
            ):
                raise ValueError(
                    "calibrated action encoding requires per-group offset/scale"
                )
            if not bool(torch.isfinite(normalization_offset).all()) or not bool(
                torch.isfinite(normalization_scale).all()
            ):
                raise ValueError("action normalization contains non-finite values")
            if bool((normalization_scale <= 0).any()):
                raise ValueError("action normalization scale must be positive")
            semantic_valid = action_semantic_ids.ne(0)
            calibration = torch.cat(
                (
                    normalization_offset
                    * semantic_valid.to(dtype=normalization_offset.dtype),
                    normalization_scale.log()
                    * semantic_valid.to(dtype=normalization_scale.dtype),
                    semantic_valid.to(dtype=normalization_offset.dtype),
                ),
                dim=-1,
            )
            assert self.normalization is not None
            normalization_token = self.normalization(calibration)
        elif normalization_offset is not None or normalization_scale is not None:
            raise ValueError(
                "normalization statistics were supplied to an uncalibrated encoder"
            )

        fine_pair = torch.cat(
            (
                fine_values * fine_dim_mask.to(dtype=fine_values.dtype),
                fine_dim_mask.to(dtype=fine_values.dtype),
            ),
            dim=-1,
        )
        # Preserve a source-normalized, linear physical-action feature before
        # time/identity mixing, nonlinearities, and RMS normalization.  The
        # factual branch centers this feature against physical zero and uses
        # it as the V7-compatible same-horizon block-0 anchor.
        direct_fine_tokens = self.fine_value(fine_pair)
        fine_tokens = direct_fine_tokens + self.time(fine_dt)
        if normalization_token is not None:
            fine_tokens = fine_tokens + normalization_token[:, None, :, None]
        fine_tokens = fine_tokens + F.silu(
            self.fine_joint(self.fine_joint_norm(fine_tokens))
        )
        real_fine = fine_sample_mask & fine_dim_mask.any(dim=-1)
        fine_weight = real_fine[..., None].to(dtype=fine_tokens.dtype)
        fine_summary = (fine_tokens * fine_weight).sum(dim=3)
        fine_count = fine_weight.sum(dim=3).clamp_min(1.0)
        fine_summary = fine_summary / fine_count
        direct_fine_summary = (direct_fine_tokens * fine_weight).sum(dim=3)
        direct_fine_summary = direct_fine_summary / fine_count
        if self.fine_aggregate is not None:
            fine_dim_valid = fine_dim_mask & fine_sample_mask[..., None]
            fine_dim_weight = fine_dim_valid.to(dtype=fine_values.dtype)
            per_dim_count = fine_dim_weight.sum(dim=3)
            per_dim_sum = (fine_values * fine_dim_weight).sum(dim=3)
            per_dim_mean = per_dim_sum / per_dim_count.clamp_min(1.0)
            substep_index = torch.arange(substeps, device=fine_values.device).view(
                1, 1, 1, substeps, 1
            )
            last_index = torch.where(
                fine_dim_valid,
                substep_index,
                torch.full_like(substep_index, -1),
            ).amax(dim=3)
            gathered_last = fine_values.gather(
                3,
                last_index.clamp_min(0).unsqueeze(3).expand(-1, -1, -1, 1, -1),
            ).squeeze(3)
            per_dim_last = gathered_last * last_index.ge(0).to(gathered_last.dtype)
            aggregate = torch.cat(
                (
                    per_dim_mean,
                    per_dim_sum,
                    per_dim_last,
                    per_dim_count / float(substeps),
                ),
                dim=-1,
            )
            aggregate_feature = self.fine_aggregate(aggregate)
            fine_summary = fine_summary + aggregate_feature
            direct_fine_summary = direct_fine_summary + aggregate_feature

        coarse_pair = torch.cat(
            (
                coarse_values * coarse_dim_mask.to(dtype=coarse_values.dtype),
                coarse_dim_mask.to(dtype=coarse_values.dtype),
            ),
            dim=-1,
        )
        direct_coarse_summary = self.coarse_value(coarse_pair)
        coarse_summary = direct_coarse_summary
        if normalization_token is not None:
            coarse_summary = coarse_summary + normalization_token[:, None]
        real_coarse = coarse_dim_mask.any(dim=-1)
        fine_present = real_fine.any(dim=3)
        if include_direct_physical and bool((fine_present & real_coarse).any()):
            raise ValueError(
                "direct physical action requires exactly one fine/coarse lane "
                "per group and horizon"
            )
        source_count = fine_present.to(torch.int32) + real_coarse.to(torch.int32)
        direct_signal = (
            direct_fine_summary
            * fine_present[..., None].to(dtype=direct_fine_summary.dtype)
            + direct_coarse_summary
            * real_coarse[..., None].to(dtype=direct_coarse_summary.dtype)
        ) / source_count.clamp_min(1)[..., None].to(
            dtype=direct_fine_summary.dtype
        )
        if include_direct_physical:
            if self.direct_group_gain is None:
                raise RuntimeError("direct physical action group router is unavailable")
            group_gain = 1.0 + torch.tanh(self.direct_group_gain(group_ids))
            direct_signal = direct_signal * group_gain[:, None]
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
        encoded = signal * group_mask[:, None, :, None].to(signal.dtype)
        if include_direct_physical:
            direct_signal = direct_signal * valid[..., None].to(direct_signal.dtype)
            return encoded, valid, direct_signal
        return encoded, valid


class CanonicalV7ActionTokenEncoder(nn.Module):
    """Project the unique canonical Kx7 physical command into one token/K.

    The grouped adapter remains the owner of source-specific normalization,
    substep composition and semantic validation.  This module starts only
    after that adapter has produced the proven V7
    [dx,dy,dz,rx,ry,rz,close01] ABI.  It intentionally mirrors V7's biased
    two-layer action projection and learned horizon identity, while adding no
    state/action trunk.
    """

    def __init__(self, hidden: int, horizon: int):
        super().__init__()
        self.horizon = int(horizon)
        self.proj = nn.Sequential(
            nn.Linear(7, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=True),
        )
        self.horizon_pos = nn.Parameter(torch.empty(1, horizon, hidden))
        nn.init.normal_(self.horizon_pos, std=0.02)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.horizon_pos, std=0.02)

    def forward(self, action_cond: torch.Tensor) -> torch.Tensor:
        if action_cond.ndim == 3:
            if tuple(action_cond.shape[1:]) != (self.horizon, 7):
                raise ValueError("canonical V7 action must be [B,K,7]")
            return self.proj(action_cond) + self.horizon_pos
        if action_cond.ndim == 4:
            if action_cond.shape[1] != self.horizon or action_cond.shape[-1] != 7:
                raise ValueError("grouped canonical V7 action must be [B,K,G,7]")
            return self.proj(action_cond) + self.horizon_pos[:, :, None]
        raise ValueError("canonical V7 action must be [B,K,7] or [B,K,G,7]")


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
        if value.ndim == 3:
            if tuple(token_mask.shape) != tuple(value.shape[:2]):
                raise ValueError("full action sequence mask must be [B,L]")
            value = value * token_mask[..., None].to(dtype=value.dtype)
            value = value + self.attn(
                self.attn_norm(value),
                allowed_mask=token_mask[:, None, None, :],
            )
            value = value * token_mask[..., None].to(dtype=value.dtype)
            value = value + self.ff(self.ff_norm(value))
            return value * token_mask[..., None].to(dtype=value.dtype)
        if value.ndim != 4:
            raise ValueError("action block expects [B,S,G,D] or [B,L,D]")
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
        *,
        factual: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The factual V7 topology bridges complete token sequences in both
        # directions.  No patch mean or frame-wise broadcast is permitted:
        # those operations erase the spatial identity needed to predict the
        # direction of robot/object motion.  The existing projection weights
        # are reused, so this branch adds no bridge parameters.
        if state.ndim == 3 and action.ndim == 3:
            if tuple(state.shape[:2]) != tuple(action.shape[:2]):
                raise ValueError("full-token bridge streams must share token length")
            if tuple(action_mask.shape) != tuple(action.shape[:2]):
                raise ValueError("full-token action mask must be [B,L]")
            state_normalized = self.state_norm(state)
            action_normalized = self.action_norm(action)
            action_update = self.action_reads_state(
                action_normalized,
                state_normalized,
                allowed_mask=action_mask[:, None, None, :],
            )
            state_update = self.state_reads_action(
                state_normalized,
                action_normalized,
                allowed_mask=action_mask[:, None, None, :],
            )
            state = (state + state_update) * action_mask[..., None].to(
                dtype=state.dtype
            )
            action = (action + action_update) * action_mask[..., None].to(
                dtype=action.dtype
            )
            return state, action
        if state.ndim != 4 or action.ndim != 4:
            raise ValueError("bridge inputs must both be factorized or full-token")
        batch, state_steps, patches, _ = state.shape
        action_steps, groups = action.shape[1:3]
        state_summary = state.mean(dim=2)
        action_flat = action.reshape(batch, action_steps * groups, -1)
        valid_flat = action_mask.reshape(batch, action_steps * groups)
        # The factual pass follows the original V7 guard: its ActionStream may
        # read only observed state.  Otherwise a late candidate command can
        # travel future-state -> history-action -> earlier-future-state through
        # the bridge and silently violate horizon causality.  The policy pass
        # keeps its existing action-free behavior.
        state_memory = (
            state_summary[:, : self.history_steps] if factual else state_summary
        )
        action_update = self.action_reads_state(
            self.action_norm(action_flat), self.state_norm(state_memory)
        )
        action_flat = (action_flat + action_update) * valid_flat[..., None].to(
            dtype=action_flat.dtype
        )
        action = action_flat.view(batch, action_steps, groups, -1)
        # In the action-free policy pass, learned future queries must not write
        # back into the world prior.  In the separate factual pass, matching
        # V7 means the state can read the complete candidate-free ActionStream;
        # candidate information itself remains owned by StateStream.
        if factual:
            state_action = action_flat
            state_action_valid = valid_flat
        else:
            state_action = action[:, : self.history_steps].reshape(
                batch, self.history_steps * groups, -1
            )
            state_action_valid = action_mask[:, : self.history_steps].reshape(
                batch, self.history_steps * groups
            )
        state_update = self.state_reads_action(
            self.state_norm(state_summary),
            self.action_norm(state_action),
            allowed_mask=state_action_valid[:, None, None, :],
        )
        state = state + state_update[:, :state_steps, None, :] / patches**0.5
        return state, action


class OriginalV7FactualDecoderLayer(nn.Module):
    """One independently parameterized layer of the original V7 decoder.

    V7 did not refine an already-predicted action-free future.  It started from
    learned future queries, added the physical action to those queries, and ran
    two *distinct* pre-norm Transformer decoder layers over the complete
    future grid.  Each layer read a memory containing task, observed state and
    future-action tokens.  Keeping this layer factual-only preserves V8's
    policy/action-free isolation while restoring the proven V7 world path.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.layer = nn.TransformerDecoderLayer(
            d_model=cfg.state_hidden,
            nhead=cfg.state_heads,
            dim_feedforward=cfg.state_hidden * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self,
        future_query: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
    ) -> torch.Tensor:
        if future_query.ndim != 4:
            raise ValueError("future query must be [B,K,P,D]")
        batch, horizon, patches, dim = future_query.shape
        if memory.ndim != 3 or memory.shape[0] != batch or memory.shape[-1] != dim:
            raise ValueError("decoder memory must be [B,M,D]")
        if tuple(memory_valid.shape) != tuple(memory.shape[:2]):
            raise ValueError("decoder memory mask must be [B,M]")
        if not bool(memory_valid.any(dim=1).all()):
            raise ValueError("every factual decoder sample needs valid memory")
        decoded = self.layer(
            future_query.reshape(batch, horizon * patches, dim),
            memory,
            memory_key_padding_mask=~memory_valid,
        )
        return decoded.view(batch, horizon, patches, dim)


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


class AppearanceActionConditioner(nn.Module):
    """Condition future appearance patches on centered physical actions.

    Grouped action tokens are read before spatial/temporal appearance
    reasoning. Cross-attention preserves group semantics when an embodiment
    has multiple action owners; bounded feature modulation keeps the update
    patch-dependent even for the common single-arm/single-group case. Every
    projection is bias-free, so a same-mask zero command produces an exactly
    zero update without introducing a second objective.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.scale = cfg.appearance_action_residual_scale
        self.query_norm = RMSNorm(cfg.appearance_hidden)
        self.action_input = nn.Linear(
            cfg.state_hidden, cfg.appearance_hidden, bias=False
        )
        self.action_norm = RMSNorm(cfg.appearance_hidden)
        self.cross = CrossAttention(
            cfg.appearance_hidden,
            cfg.appearance_hidden,
            cfg.appearance_heads,
            cfg.dropout,
        )

    def forward(
        self,
        future: torch.Tensor,
        centered_action: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, views, patches, dim = future.shape
        if centered_action.ndim != 4:
            raise ValueError("centered appearance action must be [B,K,G,D]")
        if centered_action.shape[:2] != (batch, horizon):
            raise ValueError("centered appearance action must align to B/K")
        if tuple(action_mask.shape) != tuple(centered_action.shape[:-1]):
            raise ValueError("centered appearance action mask must be [B,K,G]")

        groups = centered_action.shape[2]
        valid = action_mask.bool().reshape(batch * horizon, groups)
        action = centered_action * action_mask[..., None].to(
            dtype=centered_action.dtype
        )
        action = self.action_input(action).reshape(batch * horizon, groups, dim)

        # SDPA requires at least one allowed key. A row without a valid group
        # receives a visible all-zero token, which remains an exact zero update
        # because every conditioner projection is bias-free.
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~safe_valid.any(dim=-1)

        query = future.reshape(batch * horizon, views * patches, dim)
        normalized_query = self.query_norm(query)
        condition = self.cross(
            normalized_query,
            self.action_norm(action),
            allowed_mask=safe_valid[:, None, None, :],
        )
        # With one action group, plain cross-attention returns the same value
        # for every patch. Feature modulation makes that physical command act
        # through each patch/view representation while remaining continuous
        # and exactly neutral at zero action.
        update = condition * (1.0 + torch.tanh(normalized_query))
        return (query + self.scale * update).view(batch, horizon, views, patches, dim)


class PerViewAppearanceDynamics(nn.Module):
    """Causal one-step appearance predictor with teacher-forced and AR paths."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.geometry_grid = isqrt(cfg.P)
        self.appearance_grid = isqrt(cfg.appearance_P)
        self.input = nn.Linear(cfg.token_dim, cfg.appearance_hidden, bias=False)
        self.geometry = nn.Linear(cfg.state_hidden, cfg.appearance_hidden, bias=False)
        action_conditioner: Optional[nn.Module] = None
        if cfg.appearance_action_residual_scale > 0.0:
            action_conditioner = AppearanceActionConditioner(cfg)
            if cfg.activation_checkpointing:
                action_conditioner = checkpoint_wrapper(action_conditioner)
        self.action_conditioner = action_conditioner
        self.time = ContinuousTimeEmbedding(cfg.appearance_hidden, cfg)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.appearance_hidden)
        )
        self.patch_embed = nn.Parameter(
            torch.empty(1, 1, 1, cfg.appearance_P, cfg.appearance_hidden)
        )
        for parameter in (self.view_embed, self.patch_embed):
            nn.init.normal_(parameter, std=0.02)
        blocks: tuple[nn.Module, ...] = tuple(
            FactorizedAppearanceBlock(cfg) for _ in range(cfg.appearance_layers)
        )
        if cfg.activation_checkpointing:
            blocks = tuple(checkpoint_wrapper(block) for block in blocks)
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.appearance_hidden)
        output_type = (
            _ZeroInitializedLinear
            if cfg.appearance_flow_aligned_detail
            else nn.Linear
        )
        self.output = output_type(cfg.appearance_hidden, cfg.token_dim, bias=False)

    def reset_parameters(self) -> None:
        for parameter in (self.view_embed, self.patch_embed):
            nn.init.normal_(parameter, std=0.02)

    @staticmethod
    def _normalize_tokens(value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(value.float(), (value.shape[-1],)).to(dtype=value.dtype)

    def _upsample_geometry(self, future_state: torch.Tensor) -> torch.Tensor:
        batch, horizon, patches, _ = future_state.shape
        if patches != self.cfg.P:
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

    def _conditioned_sequence(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        *,
        transition_start: int,
        future_state: torch.Tensor,
        future_time: torch.Tensor,
        centered_action_tokens: Optional[torch.Tensor],
        centered_action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, steps, _views, _patches, _token_dim = tokens.shape
        transitions = steps - transition_start
        if transitions <= 0 or tuple(future_state.shape[:2]) != (
            batch,
            transitions,
        ):
            raise ValueError("appearance transition sequence is misaligned")
        if tuple(future_time.shape) != (batch, transitions):
            raise ValueError("appearance transition times are misaligned")

        value = self.input(tokens) + self.view_embed + self.patch_embed
        prefix = value[:, :transition_start]
        if transition_start:
            prefix = prefix + self.time(times[:, :transition_start])[:, :, None, None]
        transition = value[:, transition_start:]
        transition = transition + self.time(future_time)[:, :, None, None]
        transition = transition + self._upsample_geometry(future_state)[:, :, None]
        if self.action_conditioner is not None:
            if centered_action_tokens is None or centered_action_mask is None:
                raise ValueError(
                    "appearance action conditioning requires centered grouped tokens"
                )
            transition = self.action_conditioner(
                transition, centered_action_tokens, centered_action_mask
            )
        value = torch.cat((prefix, transition), dim=1)
        value = value * mask[..., None].to(dtype=value.dtype)
        for block in self.blocks:
            value = block(value, mask)
        return value

    def _project(self, value: torch.Tensor) -> torch.Tensor:
        return self._normalize_tokens(self.output(self.norm(value)))

    def _flow_aligned_detail(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        context_time: torch.Tensor,
        future_state: torch.Tensor,
        future_time: torch.Tensor,
        future_mask: torch.Tensor,
        centered_action_tokens: Optional[torch.Tensor],
        centered_action_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict all K detail residuals without an absolute-token feedback loop."""

        cfg = self.cfg
        batch = context_tokens.shape[0]
        queries = context_tokens.new_zeros(
            batch,
            cfg.K,
            cfg.num_views,
            cfg.appearance_P,
            cfg.token_dim,
        )
        source = torch.cat((context_tokens, queries), dim=1)
        source_mask = torch.cat((context_mask, future_mask), dim=1)
        source_time = torch.cat((context_time, future_time), dim=1)
        value = self._conditioned_sequence(
            source,
            source_mask,
            source_time,
            transition_start=cfg.appearance_context_frames,
            future_state=future_state,
            future_time=future_time,
            centered_action_tokens=centered_action_tokens,
            centered_action_mask=centered_action_mask,
        )
        prediction_mask = future_mask & context_mask.any(dim=1)[:, None]
        prediction = self.output(self.norm(value[:, cfg.appearance_context_frames :]))
        prediction = prediction * prediction_mask[..., None].to(dtype=prediction.dtype)
        return prediction, prediction_mask

    def _teacher_forced(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        context_time: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
        future_state: torch.Tensor,
        future_time: torch.Tensor,
        centered_action_tokens: Optional[torch.Tensor],
        centered_action_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        source_tokens = torch.cat((context_tokens, target_tokens[:, :-1]), dim=1)
        source_mask = torch.cat((context_mask, target_mask[:, :-1]), dim=1)
        source_time = torch.cat((context_time, future_time[:, :-1]), dim=1)
        transition_start = cfg.appearance_context_frames - 1
        value = self._conditioned_sequence(
            source_tokens,
            source_mask,
            source_time,
            transition_start=transition_start,
            future_state=future_state,
            future_time=future_time,
            centered_action_tokens=centered_action_tokens,
            centered_action_mask=centered_action_mask,
        )
        prediction_mask = target_mask & source_mask[:, transition_start:]
        prediction = self._project(value[:, transition_start:])
        prediction = prediction * prediction_mask[..., None].to(dtype=prediction.dtype)
        return prediction, prediction_mask

    def _autoregressive(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        context_time: torch.Tensor,
        future_state: torch.Tensor,
        future_time: torch.Tensor,
        future_mask: torch.Tensor,
        centered_action_tokens: Optional[torch.Tensor],
        centered_action_mask: Optional[torch.Tensor],
        *,
        steps: int,
        first_prediction: Optional[torch.Tensor] = None,
        first_prediction_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_tokens = context_tokens
        history_mask = context_mask
        history_time = context_time
        predictions: list[torch.Tensor] = []
        prediction_masks: list[torch.Tensor] = []
        start_index = 0
        if first_prediction is not None:
            expected_prediction = (
                context_tokens.shape[0],
                self.cfg.num_views,
                self.cfg.appearance_P,
                self.cfg.token_dim,
            )
            if tuple(first_prediction.shape) != expected_prediction:
                raise ValueError("reused appearance prediction has incompatible shape")
            if (
                first_prediction_mask is None
                or tuple(first_prediction_mask.shape) != expected_prediction[:-1]
            ):
                raise ValueError("reused appearance mask has incompatible shape")
            predictions.append(first_prediction)
            prediction_masks.append(first_prediction_mask)
            history_tokens = torch.cat(
                (history_tokens, first_prediction[:, None]), dim=1
            )
            history_mask = torch.cat(
                (history_mask, first_prediction_mask[:, None]), dim=1
            )
            history_time = torch.cat((history_time, future_time[:, :1]), dim=1)
            start_index = 1
        elif first_prediction_mask is not None:
            raise ValueError("reused appearance mask requires a prediction")

        for index in range(start_index, steps):
            value = self._conditioned_sequence(
                history_tokens,
                history_mask,
                history_time,
                transition_start=int(history_tokens.shape[1]) - 1,
                future_state=future_state[:, index : index + 1],
                future_time=future_time[:, index : index + 1],
                centered_action_tokens=(
                    None
                    if centered_action_tokens is None
                    else centered_action_tokens[:, index : index + 1]
                ),
                centered_action_mask=(
                    None
                    if centered_action_mask is None
                    else centered_action_mask[:, index : index + 1]
                ),
            )
            prediction_mask = future_mask[:, index] & history_mask[:, -1]
            prediction = self._project(value[:, -1])
            prediction = prediction * prediction_mask[..., None].to(
                dtype=prediction.dtype
            )
            predictions.append(prediction)
            prediction_masks.append(prediction_mask)
            history_tokens = torch.cat((history_tokens, prediction[:, None]), dim=1)
            history_mask = torch.cat((history_mask, prediction_mask[:, None]), dim=1)
            history_time = torch.cat(
                (history_time, future_time[:, index : index + 1]), dim=1
            )
        return torch.stack(predictions, dim=1), torch.stack(prediction_masks, dim=1)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        future_state: torch.Tensor,
        world_times_s: torch.Tensor,
        future_mask: Optional[torch.Tensor] = None,
        centered_action_tokens: Optional[torch.Tensor] = None,
        centered_action_mask: Optional[torch.Tensor] = None,
        target_tokens: Optional[torch.Tensor] = None,
        rollout_steps: Optional[int] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
        context_tokens = self._normalize_tokens(context_tokens)
        context_tokens = context_tokens * context_mask[..., None].to(
            dtype=context_tokens.dtype
        )
        if centered_action_tokens is not None and tuple(
            centered_action_tokens.shape
        ) != (
            batch,
            cfg.K,
            cfg.max_action_groups,
            cfg.state_hidden,
        ):
            raise ValueError("centered appearance action has incompatible shape")
        steps = (
            cfg.appearance_autoregressive_steps
            if rollout_steps is None
            else int(rollout_steps)
        )
        if target_tokens is None:
            steps = cfg.K
        if not 0 < steps <= cfg.K:
            raise ValueError("appearance rollout steps must lie within K")

        if cfg.appearance_flow_aligned_detail:
            if steps != cfg.K:
                raise ValueError(
                    "flow-aligned appearance detail requires full K rollout"
                )
            prediction, prediction_mask = self._flow_aligned_detail(
                context_tokens,
                context_mask.bool(),
                context_time,
                future_state,
                future_time,
                future_mask.bool(),
                centered_action_tokens,
                centered_action_mask,
            )
            empty_prediction = prediction[:, :0]
            empty_mask = prediction_mask[:, :0]
            return (
                prediction,
                prediction_mask,
                empty_prediction,
                empty_mask,
                empty_prediction,
                empty_mask,
            )

        teacher_prediction = context_tokens.new_empty(
            batch, 0, cfg.num_views, cfg.appearance_P, cfg.token_dim
        )
        teacher_mask = context_mask.new_empty(batch, 0, cfg.num_views, cfg.appearance_P)
        if target_tokens is not None:
            expected_target = (
                batch,
                cfg.K,
                cfg.num_views,
                cfg.appearance_P,
                cfg.token_dim,
            )
            if tuple(target_tokens.shape) != expected_target:
                raise ValueError(f"appearance target must be {expected_target}")
            target_tokens = self._normalize_tokens(target_tokens)
            target_tokens = target_tokens * future_mask[..., None].to(
                dtype=target_tokens.dtype
            )
            teacher_prediction, teacher_mask = self._teacher_forced(
                context_tokens,
                context_mask.bool(),
                context_time,
                target_tokens,
                future_mask.bool(),
                future_state,
                future_time,
                centered_action_tokens,
                centered_action_mask,
            )

        # Teacher position zero and AR position zero have exactly the same
        # causal inputs.  Reuse that result instead of running the full
        # appearance stack a second time.  Training dropout would make the two
        # old passes intentionally stochastic, so preserve that behavior for
        # nonzero-dropout profiles.
        reuse_teacher_first = target_tokens is not None and (
            not self.training or float(cfg.dropout) == 0.0
        )
        autoregressive_prediction, autoregressive_mask = self._autoregressive(
            context_tokens,
            context_mask.bool(),
            context_time,
            future_state,
            future_time,
            future_mask.bool(),
            centered_action_tokens,
            centered_action_mask,
            steps=steps,
            first_prediction=(
                teacher_prediction[:, 0] if reuse_teacher_first else None
            ),
            first_prediction_mask=(teacher_mask[:, 0] if reuse_teacher_first else None),
        )
        if steps == cfg.K:
            predicted = autoregressive_prediction
            predicted_mask = autoregressive_mask
        elif target_tokens is not None:
            predicted = teacher_prediction
            predicted_mask = teacher_mask
        else:
            raise RuntimeError("partial target-free appearance rollout is invalid")
        return (
            predicted,
            predicted_mask,
            teacher_prediction,
            teacher_mask,
            autoregressive_prediction,
            autoregressive_mask,
        )


class _SpatialDetailBlock(nn.Module):
    """Cheap local refinement for a future high-frequency feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(_rgb_norm_groups(channels), channels)
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = F.silu(self.norm(value))
        update = self.pointwise(F.silu(self.depthwise(update)))
        return value + update


class FutureSpatialDetailPredictor(nn.Module):
    """Predict future P256 high-frequency detail from the factual P64 state.

    The module deliberately has no observed-P256, target-P256, flow, or
    autoregressive input.  Motion and horizon semantics therefore stay owned
    by the factual future state, while this small spatial head only restores
    detail that is absent at the P64 grid.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.state_grid = isqrt(cfg.P)
        self.detail_grid = isqrt(cfg.appearance_P)
        channels = cfg.appearance_detail_dim
        self.input = nn.Conv2d(cfg.state_hidden, channels, 1, bias=False)
        self.view_embed = nn.Parameter(torch.empty(1, 1, cfg.num_views, channels, 1, 1))
        nn.init.normal_(self.view_embed, std=0.02)
        self.blocks = nn.Sequential(
            _SpatialDetailBlock(channels),
            _SpatialDetailBlock(channels),
        )
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.view_embed, std=0.02)

    def forward(
        self,
        future_state: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        batch = future_state.shape[0]
        if tuple(future_state.shape[1:]) != (
            cfg.K,
            cfg.P,
            cfg.state_hidden,
        ):
            raise ValueError("future detail state must be [B,K,P,state_hidden]")
        if tuple(future_mask.shape) != (
            batch,
            cfg.K,
            cfg.num_views,
            cfg.appearance_P,
        ):
            raise ValueError("future detail mask must be [B,K,V,appearance_P]")
        value = future_state.reshape(batch * cfg.K, cfg.P, cfg.state_hidden).transpose(
            1, 2
        )
        value = value.reshape(
            batch * cfg.K,
            cfg.state_hidden,
            self.state_grid,
            self.state_grid,
        )
        value = self.input(value)
        if self.state_grid != self.detail_grid:
            value = F.interpolate(
                value.float(),
                size=(self.detail_grid, self.detail_grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=future_state.dtype)
        value = value.view(
            batch,
            cfg.K,
            1,
            cfg.appearance_detail_dim,
            self.detail_grid,
            self.detail_grid,
        )
        value = value.expand(-1, -1, cfg.num_views, -1, -1, -1)
        value = value + self.view_embed.to(dtype=value.dtype)
        value = value.reshape(
            batch * cfg.K * cfg.num_views,
            cfg.appearance_detail_dim,
            self.detail_grid,
            self.detail_grid,
        )
        value = self.output(self.blocks(value))
        value = (
            value.flatten(2)
            .transpose(1, 2)
            .reshape(
                batch,
                cfg.K,
                cfg.num_views,
                cfg.appearance_P,
                cfg.appearance_detail_dim,
            )
        )
        mask = future_mask.bool()
        return value * mask[..., None].to(dtype=value.dtype), mask


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
        self.grid = isqrt(
            cfg.P
            if cfg.appearance_state_detail
            else (cfg.appearance_P if cfg.appearance_enabled else cfg.P)
        )
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
        self.detail_grid = isqrt(cfg.appearance_P)
        detail_channels = max(32, min(128, cfg.rgb_hidden // 8))
        self.detail_stem = (
            nn.Sequential(
                nn.Conv2d(
                    cfg.appearance_detail_dim,
                    detail_channels,
                    1,
                    bias=False,
                ),
                nn.SiLU(inplace=True),
            )
            if cfg.appearance_state_detail
            else None
        )
        detail_stages = (
            (cfg.rgb_size // self.detail_grid).bit_length() - 1
            if cfg.appearance_state_detail
            else 0
        )
        self.detail_ups = nn.ModuleList(
            _RGBZeroPreservingDetailUpBlock(detail_channels, detail_channels)
            for _ in range(detail_stages)
        )
        self.detail_output = (
            _RGBDetailHead(detail_channels, 3, 3, padding=1, bias=False)
            if cfg.appearance_state_detail
            else None
        )

    def forward(
        self,
        tokens: torch.Tensor,
        view_embedding: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor] = None,
        appearance_detail_tokens: Optional[torch.Tensor] = None,
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
        rgb = torch.sigmoid(self.output(value))
        if self.detail_stem is not None:
            if appearance_detail_tokens is None or tuple(
                appearance_detail_tokens.shape[1:]
            ) != (
                self.cfg.appearance_P,
                self.cfg.appearance_detail_dim,
            ):
                raise ValueError(
                    "state detail RGB input must end in [appearance_P,appearance_detail_dim]"
                )
            detail = appearance_detail_tokens.transpose(1, 2).reshape(
                appearance_detail_tokens.shape[0],
                self.cfg.appearance_detail_dim,
                self.detail_grid,
                self.detail_grid,
            )
            detail = self.detail_stem(detail).to(dtype=rgb.dtype)
            for upsample in self.detail_ups:
                detail = upsample(detail)
            if detail.shape[-2:] != (self.cfg.rgb_size, self.cfg.rgb_size):
                detail = F.interpolate(
                    detail.float(),
                    size=(self.cfg.rgb_size, self.cfg.rgb_size),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=rgb.dtype)
            assert self.detail_output is not None
            detail_logits = self.detail_output(detail).float()
            low_frequency = F.avg_pool2d(
                detail_logits,
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False,
            )
            correction = torch.tanh(detail_logits - low_frequency).to(dtype=rgb.dtype)
            rgb = torch.clamp(
                rgb + float(self.cfg.rgb_detail_residual_scale) * correction,
                0.0,
                1.0,
            )
        elif appearance_detail_tokens is not None:
            raise ValueError(
                "appearance detail was supplied while state detail is disabled"
            )
        return rgb


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


class _RGBZeroPreservingDetailUpBlock(nn.Module):
    """Upsample P256 detail without creating content from a zero residual."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        return self.conv(value)


class _RGBDetailHead(nn.Conv2d):
    """Start near zero while keeping detail gradients live."""

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=1.0e-4)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _RGBFlowHead(nn.Conv2d):
    """Keep a newly materialized flow field at the identity transform."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _RGBDisocclusionHead(nn.Conv2d):
    """Begin with observed pixels visible and learn only genuine redraws."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, -2.0)


class _RGBMotionHead(nn.Conv2d):
    """Preserve V7's initially closed motion gate under FSDP2 materialization."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, -4.0)


def _warp_rgb_feature_with_pixel_flow(
    source: torch.Tensor,
    flow_pixels: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-warp an RGB feature using flow measured in RGB pixels."""

    if source.ndim != 4 or flow_pixels.ndim != 4:
        raise ValueError("RGB feature and pixel flow must be rank four")
    if source.shape[0] != flow_pixels.shape[0] or flow_pixels.shape[1] != 2:
        raise ValueError("RGB feature and [N,2,H,W] flow must share a batch")
    if source.shape[-2:] != flow_pixels.shape[-2:]:
        flow_pixels = F.interpolate(
            flow_pixels.float(),
            size=source.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    height, width = source.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=source.device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=source.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    flow = flow_pixels.float()
    displacement = torch.stack(
        (
            2.0 * flow[:, 0] / float(max(1, image_width - 1)),
            2.0 * flow[:, 1] / float(max(1, image_height - 1)),
        ),
        dim=-1,
    )
    sampling_grid = base_grid + displacement
    valid = (
        (sampling_grid[..., 0] >= -1.0)
        & (sampling_grid[..., 0] <= 1.0)
        & (sampling_grid[..., 1] >= -1.0)
        & (sampling_grid[..., 1] <= 1.0)
    )[:, None]
    warped = F.grid_sample(
        source.float(),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.to(dtype=source.dtype), valid


class _GroupedAverageProjection(nn.Conv2d):
    """Grouped 1x1 average whose initialization survives meta materialization."""

    def reset_parameters(self) -> None:
        nn.init.constant_(
            self.weight,
            1.0 / float(self.in_channels // self.groups),
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _ZeroProjection(nn.Conv2d):
    """Zero projection whose initialization survives meta materialization."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def normalized_physical_noop_action(
    values: torch.Tensor,
    dim_mask: torch.Tensor,
    action_semantic_ids: torch.Tensor,
    normalization_offset: torch.Tensor,
    normalization_scale: torch.Tensor,
    *,
    group_axis: int,
) -> torch.Tensor:
    """Encode a physical no-op control in a source-normalized tensor.

    Dataset tensors use ``(physical - offset) / scale``.  Filling those tensors
    with numeric zero therefore means the source mean, not a physical no-op.
    Counterfactual controls must use this helper so that incremental motion
    means zero metres/radians/rates for every source.  Absolute gripper,
    absolute joint and opaque controller channels are preserved: numeric zero
    is not a meaningful no-op for those semantics.
    """

    if values.shape != dim_mask.shape or dim_mask.dtype != torch.bool:
        raise ValueError("physical-noop values and masks must align")
    group_axis = int(group_axis) % values.ndim
    expected_stats = (values.shape[0], values.shape[group_axis], values.shape[-1])
    if (
        tuple(action_semantic_ids.shape) != expected_stats
        or tuple(normalization_offset.shape) != expected_stats
        or normalization_scale.shape != normalization_offset.shape
        or not bool(torch.isfinite(normalization_offset).all())
        or not bool(torch.isfinite(normalization_scale).all())
        or bool((normalization_scale <= 0).any())
    ):
        raise ValueError("physical-noop normalization statistics are invalid")
    broadcast = [1] * values.ndim
    broadcast[0] = values.shape[0]
    broadcast[group_axis] = values.shape[group_axis]
    broadcast[-1] = values.shape[-1]
    offset = normalization_offset.to(dtype=values.dtype).reshape(broadcast)
    scale = normalization_scale.to(dtype=values.dtype).reshape(broadcast)
    encoded_zero = -offset / scale
    zeroable_ids = torch.as_tensor(
        (
            ACTION_SEMANTIC_IDS["delta_position_m"],
            ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
            ACTION_SEMANTIC_IDS["joint_delta_rad"],
            ACTION_SEMANTIC_IDS["base_velocity_mps"],
            ACTION_SEMANTIC_IDS["base_yaw_rate_rps"],
            ACTION_SEMANTIC_IDS["joint_velocity_rps"],
        ),
        dtype=action_semantic_ids.dtype,
        device=action_semantic_ids.device,
    )
    zeroable = action_semantic_ids[..., None].eq(zeroable_ids).any(dim=-1)
    semantic_shape = [1] * values.ndim
    semantic_shape[0] = values.shape[0]
    semantic_shape[group_axis] = values.shape[group_axis]
    semantic_shape[-1] = values.shape[-1]
    zeroable = zeroable.reshape(semantic_shape)
    preserved = torch.where(dim_mask, values, torch.zeros_like(values))
    return torch.where(dim_mask & zeroable, encoded_zero, preserved)


def normalized_physical_zero_action(
    values: torch.Tensor,
    dim_mask: torch.Tensor,
    action_semantic_ids: torch.Tensor,
    normalization_offset: torch.Tensor,
    normalization_scale: torch.Tensor,
    *,
    group_axis: int,
) -> torch.Tensor:
    """Encode numeric physical zero for every valid action dimension.

    This is the model-internal counterfactual anchor.  Unlike the pose no-op
    helper above, it also maps absolute/binary channels to physical zero so
    factual gripper or controller values remain observable after centering.
    """

    # Reuse all shape/statistics validation from the no-op helper.
    normalized_physical_noop_action(
        values,
        dim_mask,
        action_semantic_ids,
        normalization_offset,
        normalization_scale,
        group_axis=group_axis,
    )
    group_axis = int(group_axis) % values.ndim
    broadcast = [1] * values.ndim
    broadcast[0] = values.shape[0]
    broadcast[group_axis] = values.shape[group_axis]
    broadcast[-1] = values.shape[-1]
    offset = normalization_offset.to(dtype=values.dtype).reshape(broadcast)
    scale = normalization_scale.to(dtype=values.dtype).reshape(broadcast)
    return torch.where(dim_mask, -offset / scale, torch.zeros_like(values))


class OriginalV7RGBActionAdapter(nn.Module):
    """Compose source-normalized commands into one physical Kx7 action ABI.

    The input retains source-rate, masked, normalized and grouped tensors.  The
    output is source-independent physical motion:
    ``[metres, base-frame SO(3) rotvec radians, absolute close01]``.  Fine
    controller substeps are de-normalized before translation summation and
    rotation composition; coarse effects are de-normalized once.  This keeps
    identical physical commands identical across data sources.
    """

    _EXPECTED_SEMANTICS = (
        ACTION_SEMANTIC_IDS["delta_position_m"],
        ACTION_SEMANTIC_IDS["delta_position_m"],
        ACTION_SEMANTIC_IDS["delta_position_m"],
        ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
        ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
        ACTION_SEMANTIC_IDS["delta_rotation_axis_angle_rad"],
        ACTION_SEMANTIC_IDS["absolute_gripper_close01"],
    )

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _select_future_group(
        value: torch.Tensor, group_index: torch.Tensor
    ) -> torch.Tensor:
        shape = list(value.shape)
        index = group_index.view(shape[0], 1, 1, *([1] * (value.ndim - 3)))
        expand = list(shape)
        expand[2] = 1
        return value.gather(2, index.expand(expand)).squeeze(2)

    @staticmethod
    def _select_static_group(
        value: torch.Tensor, group_index: torch.Tensor
    ) -> torch.Tensor:
        shape = list(value.shape)
        index = group_index.view(shape[0], 1, *([1] * (value.ndim - 2)))
        expand = list(shape)
        expand[1] = 1
        return value.gather(1, index.expand(expand)).squeeze(1)

    def forward(
        self,
        *,
        fine_values: torch.Tensor,
        fine_dim_mask: torch.Tensor,
        fine_sample_mask: torch.Tensor,
        coarse_values: torch.Tensor,
        coarse_dim_mask: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        group_mask: torch.Tensor,
        normalization_offset: torch.Tensor,
        normalization_scale: torch.Tensor,
        return_grouped: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        batch = fine_values.shape[0]
        if tuple(fine_values.shape[1:3]) != (cfg.K, cfg.max_action_groups):
            raise ValueError("V7 RGB fine action must align to [B,K,G,S,A]")
        if fine_values.shape != fine_dim_mask.shape:
            raise ValueError("V7 RGB fine action values/mask must match")
        if fine_sample_mask.shape != fine_values.shape[:-1]:
            raise ValueError("V7 RGB fine sample mask must align")
        if tuple(coarse_values.shape[1:]) != (
            cfg.K,
            cfg.max_action_groups,
            cfg.max_action_dim,
        ) or coarse_dim_mask.shape != coarse_values.shape:
            raise ValueError("V7 RGB coarse action must align to [B,K,G,A]")
        if tuple(action_semantic_ids.shape) != (
            batch,
            cfg.max_action_groups,
            cfg.max_action_dim,
        ):
            raise ValueError("V7 RGB action semantics must align to [B,G,A]")
        if tuple(group_mask.shape) != (batch, cfg.max_action_groups):
            raise ValueError("V7 RGB group mask must align to [B,G]")
        if (
            normalization_offset.shape != action_semantic_ids.shape
            or normalization_scale.shape != action_semantic_ids.shape
            or bool((normalization_scale <= 0).any())
        ):
            raise ValueError("V7 RGB action normalization is invalid")

        expected = torch.as_tensor(
            self._EXPECTED_SEMANTICS,
            dtype=action_semantic_ids.dtype,
            device=action_semantic_ids.device,
        )
        canonical_group = (
            action_semantic_ids[..., :7].eq(expected).all(dim=-1) & group_mask
        )
        if not bool(canonical_group.any(dim=1).all()):
            raise ValueError(
                "V7 factual action requires at least one canonical seven-dimensional arm group"
            )
        canonical = canonical_group[:, None, :, None, None]
        fine = fine_values
        fine_valid = (
            fine_dim_mask[..., :7]
            & fine_sample_mask[..., None]
            & canonical
        )
        coarse = coarse_values
        coarse_mask = coarse_dim_mask[..., :7] & canonical_group[:, None, :, None]
        offset = normalization_offset
        scale = normalization_scale

        fine_present = fine_valid.any(dim=(-1, -2))
        coarse_present = coarse_mask.any(dim=-1)
        if bool((fine_present & coarse_present).any()):
            raise ValueError("one V7 RGB horizon cannot mix fine and coarse arm actions")
        canonical_horizon = canonical_group[:, None, :].expand(-1, cfg.K, -1)
        if not bool(((fine_present | coarse_present) | ~canonical_horizon).all()):
            raise ValueError(
                "every canonical V7 arm group/horizon requires a real command"
            )

        fine_offset = offset[:, None, :, None, :7].to(dtype=fine.dtype)
        fine_scale = scale[:, None, :, None, :7].to(dtype=fine.dtype)
        physical_fine = torch.where(
            fine_valid,
            fine[..., :7] * fine_scale + fine_offset,
            torch.zeros_like(fine[..., :7]),
        )
        translation = physical_fine[..., :3].sum(dim=3)
        physical_rotation = physical_fine[..., 3:6]
        rotation_valid = fine_valid[..., 3:6].any(dim=-1)
        rotation, rotation_present = compose_axis_angle_sequence(
            physical_rotation, rotation_valid, left_multiply=True
        )
        rotation = rotation * rotation_present[..., None].to(rotation.dtype)

        substeps = fine.shape[3]
        substep_index = torch.arange(
            substeps, device=fine.device, dtype=torch.long
        ).view(1, 1, 1, substeps)
        grip_valid = fine_valid[..., 6]
        last_index = torch.where(grip_valid, substep_index, -1).amax(dim=3)
        grip = physical_fine[..., 6].gather(
            3, last_index.clamp_min(0).unsqueeze(3)
        )
        grip = grip.squeeze(3)
        grip = torch.where(last_index >= 0, grip, torch.zeros_like(grip))
        fine_action = torch.cat(
            (translation, rotation, grip[..., None]),
            dim=-1,
        )

        coarse_action = torch.where(
            coarse_mask,
            coarse[..., :7] * scale[:, None, :, :7].to(dtype=coarse.dtype)
            + offset[:, None, :, :7].to(dtype=coarse.dtype),
            torch.zeros_like(coarse[..., :7]),
        )
        grouped_action = torch.where(
            fine_present[..., None], fine_action, coarse_action
        )
        grouped_action = grouped_action * canonical_group[:, None, :, None].to(
            dtype=grouped_action.dtype
        )
        if return_grouped:
            return grouped_action, canonical_group
        if not bool(canonical_group.sum(dim=1).eq(1).all()):
            raise ValueError(
                "a direct V7 Kx7 action skip requires exactly one canonical arm group"
            )
        group_index = canonical_group.to(dtype=torch.long).argmax(dim=1)
        action = self._select_future_group(grouped_action, group_index)
        if tuple(action.shape) != (batch, cfg.K, 7):
            raise RuntimeError("V7 RGB action adapter produced an invalid shape")
        return action


class NativeV7BoundedHighFrequencyRefiner(nn.Module):
    """Tiny factual-P64 detail head with a fixed zero-DC output contract."""

    _HIGH_PASS_KERNEL = 5

    def __init__(
        self,
        cfg: NativeWorldModelConfig,
        *,
        feature_channels: int,
    ) -> None:
        super().__init__()
        detail_channels = int(cfg.rgb_v7_high_frequency_channels)
        if feature_channels % detail_channels:
            raise ValueError("detail channels must divide decoder feature channels")
        self.cfg = cfg
        self.feature_proj = _GroupedAverageProjection(
            feature_channels,
            detail_channels,
            kernel_size=1,
            groups=detail_channels,
            bias=False,
        )
        self.token_proj = nn.Conv2d(
            cfg.token_dim,
            detail_channels,
            kernel_size=1,
            bias=False,
        )
        mixed_channels = 2 * detail_channels
        self.spatial_filter = nn.Conv2d(
            mixed_channels,
            mixed_channels,
            kernel_size=3,
            padding=1,
            groups=mixed_channels,
            padding_mode="replicate",
            bias=False,
        )
        self.output_proj = _ZeroProjection(
            mixed_channels,
            3,
            kernel_size=1,
            bias=False,
        )

        # Dedicated projections preserve average/zero initialization when FSDP2
        # materializes and resets their meta-owned parameter shards.

    def forward(
        self,
        factual_tokens: torch.Tensor,
        decoder_features: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(factual_tokens.shape[1:]) != (
            self.cfg.P,
            self.cfg.token_dim,
        ):
            raise ValueError("detail factual tokens must end in [P,token_dim]")
        if factual_tokens.shape[0] != decoder_features.shape[0]:
            raise ValueError("detail factual tokens and RGB features must align")
        grid = isqrt(self.cfg.P)
        token_map = factual_tokens.transpose(1, 2).reshape(
            factual_tokens.shape[0], self.cfg.token_dim, grid, grid
        )
        token_detail = self.token_proj(token_map)
        token_detail = F.interpolate(
            token_detail,
            size=decoder_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature_detail = self.feature_proj(decoder_features)
        basis = self.output_proj(
            F.silu(
                self.spatial_filter(torch.cat((feature_detail, token_detail), dim=1))
            )
        )
        basis = torch.tanh(basis.float())
        radius = self._HIGH_PASS_KERNEL // 2
        low_frequency = F.avg_pool2d(
            F.pad(basis, (radius, radius, radius, radius), mode="replicate"),
            kernel_size=self._HIGH_PASS_KERNEL,
            stride=1,
        )
        high_frequency = basis - low_frequency
        # Remove finite-image DC remainder from boundary handling. Both terms
        # lie in [-1,1], so the centered difference lies in [-4,4]; multiplying
        # by 0.25 makes the configured scale a strict per-pixel bound.
        high_frequency = high_frequency - high_frequency.mean(
            dim=(-2, -1), keepdim=True
        )
        correction = 0.25 * float(self.cfg.rgb_v7_high_frequency_scale) * high_frequency
        return correction.to(dtype=decoder_features.dtype)


class _RGBTransportMotionHead(nn.Conv2d):
    """Start near the empirical moving-pixel prior without closing gradients."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, -2.0)


class _RGBTransportFlowHead(nn.Conv2d):
    """Near-identity transport with a live gradient to the factual trunk."""

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, std=1.0e-5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class NativeActionOwnedTransportRGBImageDecoder(nn.Module):
    """Render future RGB with factual P64 as the only motion owner.

    The observed image is an appearance carrier, not a future-state input.  It
    reaches every output pixel only through a backward flow predicted from the
    factual future state. The separately supervised change mask is auxiliary
    and cannot attenuate that flow. There is no full-frequency redraw path;
    the optional bounded zero-DC refiner can add only high-frequency detail.
    Static identity is represented by zero flow rather than a copy bypass.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = isqrt(cfg.P)
        hidden = cfg.rgb_hidden
        self.decode_grid = min(16, cfg.rgb_size)
        stages = (cfg.rgb_size // self.decode_grid).bit_length() - 1
        channels = tuple(
            [hidden]
            + [max(32, hidden >> min(stage, 3)) for stage in range(1, stages + 1)]
        )
        final_channels = channels[-1]
        # Predict transport on a smooth grid, then upsample the displacement
        # in full-image pixel units. A dense unconstrained 256x256 field can
        # lower RGB loss by stretching a rigid robot instead of moving it.
        flow_limit = min(32, cfg.rgb_size)
        self.flow_grid = self.decode_grid
        while self.flow_grid * 2 <= flow_limit:
            self.flow_grid *= 2
        flow_stage = (self.flow_grid // self.decode_grid).bit_length() - 1
        flow_channels = channels[flow_stage]

        self.token_proj = nn.Sequential(
            nn.Conv2d(cfg.token_dim, hidden, 1),
            nn.GroupNorm(_rgb_norm_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            _RGBConvBlock(hidden, hidden),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(cfg.task_dim),
            nn.Linear(cfg.task_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.ups = nn.ModuleList(
            _RGBUpBlock(input_channels, output_channels)
            for input_channels, output_channels in zip(channels, channels[1:])
        )

        self.flow_head = _RGBTransportFlowHead(flow_channels, 2, 3, padding=1)
        self.motion_head = _RGBTransportMotionHead(
            final_channels, 1, 3, padding=1
        )
        self.high_frequency_refiner: Optional[NativeV7BoundedHighFrequencyRefiner] = (
            None
        )
        if cfg.rgb_v7_high_frequency_refiner:
            self.high_frequency_refiner = NativeV7BoundedHighFrequencyRefiner(
                cfg,
                feature_channels=final_channels,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        view_embedding: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor],
        appearance_context_tokens: Optional[torch.Tensor],
        factual_action_summary: Optional[torch.Tensor],
        task_embedding: torch.Tensor,
        context_rgb: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        motion_tokens: Optional[torch.Tensor] = None,
        appearance_detail_residual_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if tuple(tokens.shape[1:]) != (self.cfg.P, self.cfg.token_dim):
            raise ValueError("transport RGB tokens must end in [P,token_dim]")
        if tuple(view_embedding.shape) != (
            tokens.shape[0],
            self.cfg.rgb_hidden,
            1,
            1,
        ):
            raise ValueError("view embedding must align to transport RGB slots")
        if any(
            value is not None
            for value in (
                geometry_tokens,
                appearance_context_tokens,
                motion_tokens,
                appearance_detail_residual_tokens,
            )
        ):
            raise ValueError(
                "action-owned transport has no geometry, appearance or second motion lane"
            )
        if tuple(context_rgb.shape[1:]) != (
            3,
            self.cfg.rgb_size,
            self.cfg.rgb_size,
        ):
            raise ValueError("transport context RGB must be [N,3,rgb_size,rgb_size]")
        if context_indices is None:
            if context_rgb.shape[0] != tokens.shape[0]:
                raise ValueError("transport context RGB must align to decoder slots")
            context_indices = torch.arange(
                tokens.shape[0], device=context_rgb.device, dtype=torch.long
            )
        elif (
            tuple(context_indices.shape) != (tokens.shape[0],)
            or context_indices.dtype != torch.long
            or context_indices.device != context_rgb.device
        ):
            raise ValueError("transport context indices must be aligned int64")
        if factual_action_summary is not None and factual_action_summary.shape != (
            tokens.shape[0], self.cfg.state_hidden
        ):
            raise ValueError(
                "factual action summary must align to transport RGB slots"
            )
        if task_embedding.shape != (tokens.shape[0], self.cfg.task_dim):
            raise ValueError("task embedding must align to transport RGB slots")

        value = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
        )
        value = self.token_proj(value)
        if value.shape[-2:] != (self.decode_grid, self.decode_grid):
            value = F.interpolate(
                value.float(),
                size=(self.decode_grid, self.decode_grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=tokens.dtype)
        value = value + view_embedding.to(dtype=value.dtype)
        task = self.task_proj(task_embedding).to(dtype=value.dtype)
        value = value + task[:, :, None, None]

        flow_value = (
            value
            if value.shape[-2:] == (self.flow_grid, self.flow_grid)
            else None
        )
        for upsample in self.ups:
            value = upsample(value)
            if value.shape[-2:] == (self.flow_grid, self.flow_grid):
                flow_value = value
        if value.shape[-2:] != (self.cfg.rgb_size, self.cfg.rgb_size):
            # Model profiles are not restricted to powers of two (the 5B
            # profile renders 384x384).  Keep the learned tower on efficient
            # power-of-two grids and resize its final feature map exactly once.
            value = F.interpolate(
                value.float(),
                size=(self.cfg.rgb_size, self.cfg.rgb_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=tokens.dtype)
        if flow_value is None:
            raise RuntimeError("transport decoder did not produce its smooth flow grid")

        context_base = context_rgb.index_select(0, context_indices).to(
            dtype=value.dtype
        )
        max_flow_pixels = 0.5 * float(self.cfg.rgb_size)
        raw_flow_pixels = max_flow_pixels * torch.tanh(
            self.flow_head(flow_value).float()
        )
        if raw_flow_pixels.shape[-2:] != (self.cfg.rgb_size, self.cfg.rgb_size):
            raw_flow_pixels = F.interpolate(
                raw_flow_pixels,
                size=(self.cfg.rgb_size, self.cfg.rgb_size),
                mode="bilinear",
                align_corners=True,
            )
        motion_logit = self.motion_head(value)
        motion = torch.sigmoid(motion_logit)
        # Flow is the sole low-frequency motion owner.  Do not multiply it by
        # the learned motion probability: doing so creates a copy-last local
        # optimum and attenuates both RGB and flow-teacher gradients whenever
        # the gate is small.  Static identity is learned as zero flow.
        flow_pixels = raw_flow_pixels
        transported, warp_valid = _warp_rgb_feature_with_pixel_flow(
            context_base,
            flow_pixels,
            image_height=self.cfg.rgb_size,
            image_width=self.cfg.rgb_size,
        )
        # The transported image is the only appearance base.  In particular,
        # there is no full-frequency residual that can redraw the moving
        # region instead of learning transport.  Warp validity remains a
        # diagnostic and never opens another rendering path.
        warp_invalid = (~warp_valid).to(dtype=torch.float32)
        rgb = transported
        if self.high_frequency_refiner is not None:
            rgb = torch.clamp(
                rgb.float() + self.high_frequency_refiner(tokens, value),
                0.0,
                1.0,
            ).to(dtype=value.dtype)
        disocclusion_logit = torch.where(
            warp_valid,
            torch.full_like(warp_invalid, -8.0),
            torch.full_like(warp_invalid, 8.0),
        )
        return (
            rgb,
            motion_logit,
            motion,
            flow_pixels,
            disocclusion_logit.to(dtype=value.dtype),
        )


class NativeOriginalV7ContextRGBImageDecoder(nn.Module):
    """Original V7 P64 + observed-RGB context-residual renderer.

    The spatial path and output parameterization match the decoder used by the
    original V7 60K checkpoint: a full-resolution observed RGB pyramid, P64
    factual tokens interpolated into the 16x16 bottleneck, direct RGB plus a
    bounded residual, and learned blend/motion maps. Its renderer input remains
    the exact canonical seven-dimensional physical action used by V7.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = isqrt(cfg.P)
        hidden = cfg.rgb_hidden
        channels_256 = max(32, hidden // 8)
        channels_128 = channels_256
        channels_64 = max(64, hidden // 4)
        channels_32 = max(128, hidden // 2)

        self.ctx256 = _RGBConvBlock(3, channels_256)
        self.ctx128 = _RGBDownBlock(channels_256, channels_128)
        self.ctx64 = _RGBDownBlock(channels_128, channels_64)
        self.ctx32 = _RGBDownBlock(channels_64, channels_32)
        self.ctx16 = _RGBDownBlock(channels_32, hidden)

        self.token_proj = nn.Sequential(
            nn.Conv2d(cfg.token_dim, hidden, 1),
            nn.GroupNorm(_rgb_norm_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            _RGBConvBlock(hidden, hidden),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.task_proj = nn.Sequential(
            nn.LayerNorm(cfg.task_dim),
            nn.Linear(cfg.task_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
        )

        self.fuse16 = _RGBConvBlock(hidden + hidden, hidden)
        self.up32 = _RGBUpBlock(hidden, channels_32)
        self.fuse32 = _RGBConvBlock(channels_32 + channels_32, channels_32)
        self.up64 = _RGBUpBlock(channels_32, channels_64)
        self.fuse64 = _RGBConvBlock(channels_64 + channels_64, channels_64)
        self.up128 = _RGBUpBlock(channels_64, channels_128)
        self.fuse128 = _RGBConvBlock(channels_128 + channels_128, channels_128)
        self.up256 = _RGBUpBlock(channels_128, channels_256)
        self.fuse256 = _RGBConvBlock(channels_256 + channels_256, channels_256)
        self.head = nn.Conv2d(channels_256, 7, 3, padding=1)
        self.motion_head = _RGBMotionHead(channels_256, 1, 3, padding=1)
        self.high_frequency_refiner: Optional[NativeV7BoundedHighFrequencyRefiner] = (
            None
        )
        if cfg.rgb_v7_high_frequency_refiner:
            self.high_frequency_refiner = NativeV7BoundedHighFrequencyRefiner(
                cfg,
                feature_channels=channels_256,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        view_embedding: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor],
        appearance_context_tokens: Optional[torch.Tensor],
        factual_action_summary: Optional[torch.Tensor],
        task_embedding: torch.Tensor,
        context_rgb: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        motion_tokens: Optional[torch.Tensor] = None,
        appearance_detail_residual_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if tuple(tokens.shape[1:]) != (self.cfg.P, self.cfg.token_dim):
            raise ValueError("original V7 RGB tokens must end in [P,token_dim]")
        if tuple(view_embedding.shape) != (
            tokens.shape[0],
            self.cfg.rgb_hidden,
            1,
            1,
        ):
            raise ValueError("view embedding must align to original V7 RGB slots")
        if any(
            value is not None
            for value in (
                geometry_tokens,
                appearance_context_tokens,
                motion_tokens,
                appearance_detail_residual_tokens,
            )
        ):
            raise ValueError(
                "original V7 RGB has no geometry, appearance, flow, or detail side lane"
            )
        if tuple(context_rgb.shape[1:]) != (
            3,
            self.cfg.rgb_size,
            self.cfg.rgb_size,
        ):
            raise ValueError("context RGB must be [N,3,rgb_size,rgb_size]")
        if context_indices is None:
            if context_rgb.shape[0] != tokens.shape[0]:
                raise ValueError("context RGB must align to decoder slots")
            context_indices = torch.arange(
                tokens.shape[0], device=context_rgb.device, dtype=torch.long
            )
        elif (
            tuple(context_indices.shape) != (tokens.shape[0],)
            or context_indices.dtype != torch.long
            or context_indices.device != context_rgb.device
        ):
            raise ValueError("context RGB indices must be aligned device-local int64")
        if task_embedding.shape != (tokens.shape[0], self.cfg.task_dim):
            raise ValueError("task embedding must align to original V7 RGB slots")
        if self.cfg.rgb_context_action_scale > 0.0:
            if factual_action_summary is None or factual_action_summary.shape != (
                tokens.shape[0],
                7,
            ):
                raise ValueError(
                    "canonical physical action must align to original V7 RGB slots"
                )
        elif factual_action_summary is not None:
            raise ValueError("RGB action was supplied while its scale is zero")

        # Build each observed-view pyramid once, then gather it for the selected
        # future horizons.  This preserves the original computation while
        # avoiding repeated full-resolution context work inside one dense chunk.
        context_256 = self.ctx256(context_rgb)
        context_128 = self.ctx128(context_256)
        context_64 = self.ctx64(context_128)
        context_32 = self.ctx32(context_64)
        context_16 = self.ctx16(context_32)
        context_256 = context_256.index_select(0, context_indices)
        context_128 = context_128.index_select(0, context_indices)
        context_64 = context_64.index_select(0, context_indices)
        context_32 = context_32.index_select(0, context_indices)
        context_16 = context_16.index_select(0, context_indices)
        context_base = context_rgb.index_select(0, context_indices)

        value = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
        )
        value = self.token_proj(value)
        if value.shape[-2:] != context_16.shape[-2:]:
            value = F.interpolate(
                value,
                size=context_16.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        value = value + view_embedding.to(dtype=value.dtype)
        if factual_action_summary is not None:
            action = self.action_proj(factual_action_summary).to(dtype=value.dtype)
            value = value + float(self.cfg.rgb_context_action_scale) * action.view(
                tokens.shape[0], -1, 1, 1
            )
        task = self.task_proj(task_embedding).to(dtype=value.dtype)
        value = value + task.view(tokens.shape[0], -1, 1, 1)

        value = self.fuse16(torch.cat((value, context_16), dim=1))
        value = self.up32(value)
        value = self.fuse32(torch.cat((value, context_32), dim=1))
        value = self.up64(value)
        value = self.fuse64(torch.cat((value, context_64), dim=1))
        value = self.up128(value)
        value = self.fuse128(torch.cat((value, context_128), dim=1))
        value = self.up256(value)
        value = self.fuse256(torch.cat((value, context_256), dim=1))

        motion_logit = self.motion_head(value)
        motion_hint = torch.sigmoid(motion_logit)
        raw = self.head(value)
        direct = torch.sigmoid(raw[:, 0:3])
        residual = torch.tanh(raw[:, 3:6]) * float(self.cfg.rgb_context_residual_scale)
        blend = torch.sigmoid(raw[:, 6:7])
        if self.cfg.rgb_context_motion_blend_gain > 0.0:
            blend = torch.clamp(
                blend
                + motion_hint.to(dtype=blend.dtype)
                * float(self.cfg.rgb_context_motion_blend_gain),
                0.0,
                1.0,
            )
        residual_rgb = torch.clamp(context_base + residual, 0.0, 1.0)
        rgb = blend * direct + (1.0 - blend) * residual_rgb
        if self.high_frequency_refiner is not None:
            detail_correction = self.high_frequency_refiner(tokens, value)
            rgb = torch.clamp(rgb + detail_correction, 0.0, 1.0)
        flow = rgb.new_zeros(rgb.shape[0], 2, *rgb.shape[-2:])
        disocclusion_logit = rgb.new_zeros(rgb.shape[0], 1, *rgb.shape[-2:])
        return rgb, motion_logit, blend, flow, disocclusion_logit


class NativeContextRGBImageDecoder(nn.Module):
    """Align V7 observed detail before synthesizing future RGB changes."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = isqrt(
            cfg.P
            if cfg.appearance_flow_aligned_detail
            else (cfg.appearance_P if cfg.appearance_enabled else cfg.P)
        )
        self.appearance_grid = isqrt(cfg.appearance_P)
        self.motion_grid = isqrt(cfg.P)
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
        motion_floor = min(32, channels[-1])
        motion_channels = tuple(max(motion_floor, channel // 4) for channel in channels)
        self.token_stem = nn.Sequential(
            nn.Conv2d(cfg.token_dim, channels[0], 1),
            nn.GroupNorm(_rgb_norm_groups(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _RGBConvBlock(channels[0], channels[0]),
        )
        # P256 remains an appearance/synthesis condition. The native P64 RGB
        # prior and centered physical action are the sole future-position
        # authorities for flow, motion support and visibility.
        self.motion_token_stem = (
            nn.Sequential(
                nn.Conv2d(cfg.token_dim, motion_channels[0], 1),
                nn.GroupNorm(_rgb_norm_groups(motion_channels[0]), motion_channels[0]),
                nn.SiLU(inplace=True),
                _RGBConvBlock(motion_channels[0], motion_channels[0]),
            )
            if cfg.rgb_context_alignment_enabled
            else None
        )
        self.motion_view_proj = (
            nn.Conv2d(channels[0], motion_channels[0], 1, bias=False)
            if cfg.rgb_context_alignment_enabled
            else None
        )
        self.motion_geometry_stem = (
            nn.Conv2d(cfg.state_hidden, motion_channels[0], 1, bias=False)
            if cfg.rgb_context_alignment_enabled and cfg.appearance_enabled
            else None
        )
        self.motion_action_proj = (
            nn.Sequential(
                nn.Linear(cfg.state_hidden, motion_channels[0]),
                nn.SiLU(inplace=True),
                nn.Linear(motion_channels[0], motion_channels[0]),
            )
            if cfg.rgb_context_alignment_enabled and cfg.rgb_context_action_scale > 0.0
            else None
        )
        self.motion_task_proj = (
            nn.Sequential(
                nn.LayerNorm(cfg.task_dim),
                nn.Linear(cfg.task_dim, motion_channels[0]),
                nn.SiLU(inplace=True),
                nn.Linear(motion_channels[0], motion_channels[0]),
            )
            if cfg.rgb_context_alignment_enabled
            else None
        )
        self.motion_to_synthesis = (
            nn.Conv2d(motion_channels[0], channels[0], 1, bias=False)
            if cfg.rgb_context_alignment_enabled
            else None
        )
        # P256 is a post-transport detail lane, never a second motion owner.
        # A narrow zero-preserving decoder is sufficient because P64/V7 owns
        # geometry, flow, visibility and blend.  Keeping this branch out of the
        # main U-Net also removes the former wide multiscale compute regression.
        detail_base_channels = max(64, cfg.rgb_hidden // 8)
        detail_stages = (
            (cfg.rgb_size // self.appearance_grid).bit_length() - 1
            if cfg.rgb_context_appearance_delta_scale > 0.0
            else 0
        )
        self.appearance_detail_decode_grid = (
            cfg.rgb_size // (1 << detail_stages)
            if cfg.rgb_context_appearance_delta_scale > 0.0
            else self.appearance_grid
        )
        detail_channels = tuple(
            max(32, detail_base_channels >> min(stage, 2))
            for stage in range(detail_stages + 1)
        )
        self.appearance_delta_stem = (
            nn.Sequential(
                nn.Conv2d(cfg.token_dim, detail_channels[0], 1, bias=False),
                nn.SiLU(inplace=True),
            )
            if cfg.rgb_context_appearance_delta_scale > 0.0
            else None
        )
        self.appearance_detail_ups = nn.ModuleList(
            _RGBZeroPreservingDetailUpBlock(input_channels, output_channels)
            for input_channels, output_channels in zip(
                detail_channels, detail_channels[1:]
            )
        )
        self.appearance_detail_head = (
            _RGBDetailHead(detail_channels[-1], 3, 3, padding=1, bias=False)
            if cfg.rgb_context_appearance_delta_scale > 0.0
            else None
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
        self.motion_ups = nn.ModuleList(
            (
                _RGBUpBlock(input_channels, output_channels)
                for input_channels, output_channels in zip(
                    motion_channels, motion_channels[1:]
                )
            )
            if cfg.rgb_context_alignment_enabled
            else ()
        )
        self.head = nn.Conv2d(channels[-1], 7, 3, padding=1)
        motion_output_channels = (
            motion_channels[-1] if cfg.rgb_context_alignment_enabled else channels[-1]
        )
        # Preserve the V7 learned motion/blend path on the synthesis features.
        # The transport tower predicts only alignment and visibility; it must
        # not replace the renderer path that already produces moving RGB.
        self.motion_head = _RGBMotionHead(channels[-1], 1, 3, padding=1)
        self.flow_head = (
            _RGBFlowHead(motion_output_channels, 2, 3, padding=1)
            if cfg.rgb_context_alignment_enabled
            else None
        )
        self.disocclusion_head = (
            _RGBDisocclusionHead(motion_output_channels, 1, 3, padding=1)
            if cfg.rgb_context_alignment_enabled
            else None
        )

    def forward(
        self,
        tokens: torch.Tensor,
        view_embedding: torch.Tensor,
        geometry_tokens: Optional[torch.Tensor],
        appearance_context_tokens: Optional[torch.Tensor],
        factual_action_summary: Optional[torch.Tensor],
        task_embedding: torch.Tensor,
        context_rgb: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
        motion_tokens: Optional[torch.Tensor] = None,
        appearance_detail_residual_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if tuple(context_rgb.shape[1:]) != (
            3,
            self.cfg.rgb_size,
            self.cfg.rgb_size,
        ):
            raise ValueError("context RGB must be [N,3,rgb_size,rgb_size]")
        if context_indices is None:
            if context_rgb.shape[0] != tokens.shape[0]:
                raise ValueError("context RGB must align to decoder tokens")
        else:
            if tuple(context_indices.shape) != (tokens.shape[0],):
                raise ValueError("context RGB indices must be [N]")
            if context_indices.dtype != torch.long:
                raise ValueError("context RGB indices must use torch.long")
            if context_indices.device != context_rgb.device:
                raise ValueError("context RGB indices must share the RGB device")
        value = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
        )
        value = self.token_stem(value) + view_embedding
        motion_value: Optional[torch.Tensor] = None
        if self.cfg.rgb_context_alignment_enabled:
            if (
                motion_tokens is None
                or tuple(motion_tokens.shape[1:]) != (self.cfg.P, self.cfg.token_dim)
                or motion_tokens.shape[0] != tokens.shape[0]
            ):
                raise ValueError(
                    "context RGB motion tokens must align and end in [P,token_dim]"
                )
            assert self.motion_token_stem is not None
            assert self.motion_view_proj is not None
            motion_value = motion_tokens.transpose(1, 2).reshape(
                motion_tokens.shape[0],
                self.cfg.token_dim,
                self.motion_grid,
                self.motion_grid,
            )
            motion_value = self.motion_token_stem(motion_value) + self.motion_view_proj(
                view_embedding
            )
        elif motion_tokens is not None:
            raise ValueError("motion tokens require aligned context RGB")
        appearance_delta: Optional[torch.Tensor] = None
        if self.cfg.appearance_flow_aligned_detail:
            if appearance_context_tokens is not None:
                raise ValueError(
                    "flow-aligned detail mode does not consume absolute P256 context"
                )
            if (
                appearance_detail_residual_tokens is None
                or tuple(appearance_detail_residual_tokens.shape[1:])
                != (self.cfg.appearance_P, self.cfg.token_dim)
                or appearance_detail_residual_tokens.shape[0] != tokens.shape[0]
            ):
                raise ValueError(
                    "flow-aligned P256 detail residuals must end in "
                    "[appearance_P,token_dim]"
                )
            assert self.appearance_delta_stem is not None
            delta = appearance_detail_residual_tokens.transpose(1, 2).reshape(
                tokens.shape[0],
                self.cfg.token_dim,
                self.appearance_grid,
                self.appearance_grid,
            )
            appearance_delta = self.appearance_delta_stem(delta).to(dtype=value.dtype)
        elif self.appearance_delta_stem is not None:
            if appearance_detail_residual_tokens is not None:
                raise ValueError(
                    "appearance detail residuals require flow-aligned mode"
                )
            if appearance_context_tokens is None or tuple(
                appearance_context_tokens.shape
            ) != tuple(tokens.shape):
                raise ValueError(
                    "context RGB appearance tokens must align to future tokens"
                )
            delta = tokens - appearance_context_tokens.to(dtype=tokens.dtype)
            delta = delta.transpose(1, 2).reshape(
                tokens.shape[0],
                self.cfg.token_dim,
                self.appearance_grid,
                self.appearance_grid,
            )
            appearance_delta = self.appearance_delta_stem(delta).to(dtype=value.dtype)
        elif (
            appearance_context_tokens is not None
            or appearance_detail_residual_tokens is not None
        ):
            raise ValueError(
                "appearance context was supplied while RGB delta conditioning is disabled"
            )
        if tuple(task_embedding.shape) != (tokens.shape[0], self.cfg.task_dim):
            raise ValueError("context RGB task embedding must be [N,task_dim]")
        task = self.task_proj(task_embedding).to(dtype=value.dtype)
        value = value + task[:, :, None, None]
        if motion_value is not None:
            assert self.motion_task_proj is not None
            motion_task = self.motion_task_proj(task_embedding).to(
                dtype=motion_value.dtype
            )
            motion_value = motion_value + motion_task[:, :, None, None]
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
            if motion_value is not None:
                assert self.motion_action_proj is not None
                motion_action = self.motion_action_proj(factual_action_summary).to(
                    dtype=motion_value.dtype
                )
                motion_value = motion_value + (
                    float(self.cfg.rgb_context_action_scale)
                    * motion_action[:, :, None, None]
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
            synthesis_geometry = self.geometry_stem(geometry)
            if self.geometry_grid != self.grid:
                synthesis_geometry = F.interpolate(
                    synthesis_geometry.float(),
                    size=(self.grid, self.grid),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=value.dtype)
            value = value + synthesis_geometry
            if motion_value is not None:
                assert self.motion_geometry_stem is not None
                motion_geometry = self.motion_geometry_stem(geometry)
                if self.geometry_grid != self.motion_grid:
                    motion_geometry = F.interpolate(
                        motion_geometry.float(),
                        size=(self.motion_grid, self.motion_grid),
                        mode="bilinear",
                        align_corners=False,
                    ).to(dtype=motion_value.dtype)
                motion_value = motion_value + motion_geometry
        if self.decode_grid != self.grid:
            value = F.interpolate(
                value.float(),
                size=(self.decode_grid, self.decode_grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=tokens.dtype)
        if motion_value is not None and self.decode_grid != self.motion_grid:
            assert motion_tokens is not None
            motion_value = F.interpolate(
                motion_value.float(),
                size=(self.decode_grid, self.decode_grid),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=motion_tokens.dtype)

        # Motion/action information enters synthesis, but the P256 appearance
        # lane never enters the transport/visibility tower below.
        if motion_value is not None:
            assert self.motion_to_synthesis is not None
            value = value + self.motion_to_synthesis(motion_value).to(dtype=value.dtype)

        # One observed image conditions every decoded future for the same
        # batch/view pair. Legacy mode can reuse its raw context pyramid;
        # aligned mode must first move the RGB and derives every skip from that
        # aligned image, so no unwarped high-dimensional feature is materialized.
        def expand_context(value_to_expand: torch.Tensor) -> torch.Tensor:
            if context_indices is None:
                return value_to_expand
            return value_to_expand.index_select(0, context_indices)

        context = context_rgb.to(dtype=value.dtype)
        if self.cfg.rgb_context_alignment_enabled:
            # Predict transport before the appearance branch sees any context
            # feature. This tower is driven only by the P64 RGB prior,
            # geometry, centered action, task and view identity; unreliable
            # P256 detail cannot shrink motion or move visibility boundaries.
            assert motion_value is not None
            assert self.flow_head is not None
            assert self.disocclusion_head is not None
            motion_features = motion_value
            for motion_upsample in self.motion_ups:
                motion_features = motion_upsample(motion_features)
            if motion_features.shape[-2:] != (
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            ):
                raise ValueError(
                    "RGB motion tower does not reach the output resolution"
                )
            max_flow_pixels = 0.5 * float(self.cfg.rgb_size)
            raw_flow = self.flow_head(motion_features).float()
            # The head predicts normalized displacement. Scaling raw logits
            # down by the image radius would make a unit activation equal one
            # pixel and delay visible transport by roughly rgb_size / 2.
            flow_pixels = max_flow_pixels * torch.tanh(raw_flow)
            disocclusion_logit = self.disocclusion_head(motion_features)
            disocclusion = torch.sigmoid(disocclusion_logit)

            # Warp three RGB channels once, then derive the entire feature
            # pyramid from the aligned image. This is both stricter and much
            # cheaper than grid-sampling every wide U-Net skip separately.
            warped_context, warp_valid = _warp_rgb_feature_with_pixel_flow(
                expand_context(context),
                flow_pixels,
                image_height=self.cfg.rgb_size,
                image_width=self.cfg.rgb_size,
            )
            aligned_skips = [self.context_stem(warped_context)]
            for downsample in self.context_downs:
                aligned_skips.append(downsample(aligned_skips[-1]))
            warped_skips = tuple(aligned_skips)
        else:
            skips = [self.context_stem(context)]
            for downsample in self.context_downs:
                skips.append(downsample(skips[-1]))
            warped_context = expand_context(context)
            warped_skips = tuple(expand_context(skip) for skip in skips)
            warp_valid = None
            disocclusion = None
            disocclusion_logit = value.new_zeros(
                value.shape[0], 1, self.cfg.rgb_size, self.cfg.rgb_size
            )
            flow_pixels = value.new_zeros(
                value.shape[0], 2, self.cfg.rgb_size, self.cfg.rgb_size
            )
            motion_logit = None
        if warped_skips[-1].shape[-2:] != value.shape[-2:]:
            raise ValueError("context pyramid does not align with RGB token grid")

        value = self.bottleneck_fuse(torch.cat((value, warped_skips[-1]), dim=1))
        for upsample, fuse, context_skip in zip(
            self.ups, self.skip_fuses, reversed(warped_skips[:-1])
        ):
            value = upsample(value)
            value = fuse(torch.cat((value, context_skip), dim=1))

        motion_logit = self.motion_head(value)
        motion_hint = torch.sigmoid(motion_logit)
        raw = self.head(value)
        direct = torch.sigmoid(raw[:, :3])
        residual = torch.tanh(raw[:, 3:6]) * float(self.cfg.rgb_context_residual_scale)
        residual_rgb = torch.clamp(warped_context + residual, 0.0, 1.0)
        legacy_blend = torch.sigmoid(raw[:, 6:7])
        if self.cfg.rgb_context_motion_blend_gain > 0.0:
            legacy_blend = torch.clamp(
                legacy_blend
                + motion_hint.to(dtype=legacy_blend.dtype)
                * float(self.cfg.rgb_context_motion_blend_gain),
                0.0,
                1.0,
            )
        if self.cfg.rgb_context_alignment_enabled:
            assert disocclusion is not None
            assert warp_valid is not None
            transport = (1.0 - disocclusion.to(dtype=value.dtype)) * warp_valid.to(
                dtype=value.dtype
            )
            # V7's direct/context learned blend remains the base renderer.
            # Alignment replaces only the context-detail share that V7 would
            # otherwise take from the unwarped observation.  Therefore an
            # identity flow exactly falls back to V7 instead of copy-last.
            context_detail_weight = (1.0 - legacy_blend) * transport
            blend = 1.0 - context_detail_weight
            rgb = context_detail_weight * residual_rgb + blend * direct
        else:
            blend = legacy_blend
            rgb = blend * direct + (1.0 - blend) * residual_rgb
        if appearance_delta is not None:
            assert self.appearance_detail_head is not None
            detail = appearance_delta
            if detail.shape[-2:] != (
                self.appearance_detail_decode_grid,
                self.appearance_detail_decode_grid,
            ):
                detail = F.interpolate(
                    detail.float(),
                    size=(
                        self.appearance_detail_decode_grid,
                        self.appearance_detail_decode_grid,
                    ),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=value.dtype)
            for detail_upsample in self.appearance_detail_ups:
                detail = detail_upsample(detail)
            if detail.shape[-2:] != (self.cfg.rgb_size, self.cfg.rgb_size):
                raise ValueError("P256 detail decoder does not reach RGB resolution")
            detail_logits = self.appearance_detail_head(detail).float()
            low_frequency = F.avg_pool2d(
                detail_logits,
                kernel_size=5,
                stride=1,
                padding=2,
                count_include_pad=False,
            )
            detail_correction = torch.tanh(detail_logits - low_frequency).to(
                dtype=rgb.dtype
            )
            rgb = torch.clamp(
                rgb
                + float(self.cfg.rgb_context_appearance_delta_scale)
                * detail_correction,
                0.0,
                1.0,
            )
        return rgb, motion_logit, blend, flow_pixels, disocclusion_logit


class NativeRGBDecoder(nn.Module):
    """Restore the V7 native token-to-pixel path with bounded image chunks."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.rgb_original_v7_context:
            self.register_buffer(
                "view_embed",
                torch.zeros(cfg.num_views, cfg.rgb_hidden, 1, 1),
                persistent=False,
            )
        else:
            self.view_embed = nn.Parameter(
                torch.empty(cfg.num_views, cfg.rgb_hidden, 1, 1)
            )
            nn.init.normal_(self.view_embed, std=0.02)
        image_decoder: nn.Module
        if cfg.rgb_action_owned_transport:
            image_decoder = NativeActionOwnedTransportRGBImageDecoder(cfg)
        elif cfg.rgb_original_v7_context:
            image_decoder = NativeOriginalV7ContextRGBImageDecoder(cfg)
        elif cfg.rgb_context_enabled:
            image_decoder = NativeContextRGBImageDecoder(cfg)
        else:
            image_decoder = NativeRGBImageDecoder(cfg)
        if cfg.activation_checkpointing:
            image_decoder = checkpoint_wrapper(image_decoder)
        self.image_decoder = image_decoder

    def reset_parameters(self) -> None:
        if isinstance(self.view_embed, nn.Parameter):
            nn.init.normal_(self.view_embed, std=0.02)
        else:
            # ``view_embed`` is a non-persistent buffer on the exact V7 RGB
            # path. FSDP2 first allocates its meta storage with ``to_empty``;
            # without writing it here, arbitrary allocator contents enter
            # every RGB slot even though ordinary construction produced zero.
            self.view_embed.zero_()

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
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
            empty_flow = future_tokens.new_empty(
                future_tokens.shape[0],
                0,
                self.cfg.num_views,
                2,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            return empty, index_tensor, empty_aux, empty_aux, empty_flow, empty_aux
        views = self.cfg.num_views
        selected_motion = future_tokens.index_select(1, index_tensor)
        batch, frames, motion_patches, motion_token_dim = selected_motion.shape
        expanded_motion = (
            selected_motion[:, :, None]
            .expand(-1, -1, views, -1, -1)
            .reshape(
                batch * frames * views,
                motion_patches,
                motion_token_dim,
            )
        )
        expanded_geometry: Optional[torch.Tensor] = None
        expanded_appearance_context: Optional[torch.Tensor] = None
        expanded_appearance_detail: Optional[torch.Tensor] = None
        if self.cfg.appearance_enabled:
            appearance_dim = (
                self.cfg.appearance_detail_dim
                if self.cfg.appearance_state_detail
                else self.cfg.token_dim
            )
            expected = (
                future_tokens.shape[0],
                self.cfg.K,
                views,
                self.cfg.appearance_P,
                appearance_dim,
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
            appearance_batch, appearance_frames, _, patches, token_dim = selected.shape
            if (appearance_batch, appearance_frames) != (batch, frames):
                raise ValueError("appearance and motion RGB horizons must align")
            selected_appearance = selected.reshape(
                batch * frames * views, patches, token_dim
            )
            if (
                self.cfg.appearance_flow_aligned_detail
                or self.cfg.appearance_state_detail
            ):
                expanded = expanded_motion
                expanded_appearance_detail = selected_appearance
            else:
                expanded = selected_appearance
            geometry = geometry_state.index_select(1, index_tensor)
            geometry = geometry[:, :, None].expand(-1, -1, views, -1, -1)
            expanded_geometry = geometry.reshape(
                batch * frames * views, self.cfg.P, self.cfg.state_hidden
            )
            if (
                self.cfg.rgb_context_appearance_delta_scale > 0.0
                and not self.cfg.appearance_flow_aligned_detail
                and not self.cfg.appearance_state_detail
            ):
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
            elif (
                appearance_context_tokens is not None
                and not self.cfg.appearance_flow_aligned_detail
                and not self.cfg.appearance_state_detail
            ):
                raise ValueError(
                    "appearance context was supplied while RGB delta conditioning is disabled"
                )
        else:
            if appearance_context_tokens is not None:
                raise ValueError(
                    "appearance context was supplied to a fused-only RGB decoder"
                )
            patches = motion_patches
            token_dim = motion_token_dim
            expanded = expanded_motion
        view_ids = torch.arange(views, device=future_tokens.device)
        view_ids = view_ids.view(1, 1, views).expand(batch, frames, -1).reshape(-1)
        expanded_action: Optional[torch.Tensor] = None
        if self.cfg.rgb_context_action_scale > 0.0:
            action_dim = (
                7 if self.cfg.rgb_original_v7_context else self.cfg.state_hidden
            )
            expected_action = (batch, self.cfg.K, action_dim)
            if (
                factual_action_summary is None
                or tuple(factual_action_summary.shape) != expected_action
            ):
                raise ValueError(f"factual_action_summary must be {expected_action}")
            selected_action = factual_action_summary.index_select(1, index_tensor)
            expanded_action = (
                selected_action[:, :, None]
                .expand(-1, -1, views, -1)
                .reshape(batch * frames * views, action_dim)
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
        context_bank: Optional[torch.Tensor] = None
        context_slot_ids: Optional[torch.Tensor] = None
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
            context_bank = context_rgb.reshape(
                batch * views,
                3,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            context_slot_ids = (
                torch.arange(batch * views, device=future_tokens.device)
                .view(batch, 1, views)
                .expand(-1, frames, -1)
                .reshape(-1)
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
        source_indices = torch.arange(
            batch * frames * views, device=future_tokens.device
        )
        if self.cfg.rgb_context_enabled:
            # Group all horizons for one batch/view so a decoder chunk can
            # reuse the observed context pyramid across K instead of rebuilding
            # it once per image slot.  Outputs are transposed back below.
            dense_indices = (
                source_indices.view(batch, frames, views).permute(0, 2, 1).reshape(-1)
            )
        else:
            dense_indices = source_indices
        decoded_chunks: list[torch.Tensor] = []
        motion_chunks: list[torch.Tensor] = []
        blend_chunks: list[torch.Tensor] = []
        flow_chunks: list[torch.Tensor] = []
        disocclusion_chunks: list[torch.Tensor] = []
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
                assert context_bank is not None
                assert context_slot_ids is not None
                assert expanded_task is not None
                chunk_context_ids = context_slot_ids.index_select(0, chunk_indices)
                unique_context_ids, local_context_indices = torch.unique_consecutive(
                    chunk_context_ids, return_inverse=True
                )
                (
                    decoded,
                    motion_logit,
                    blend,
                    flow_pixels,
                    disocclusion_logit,
                ) = self.image_decoder(
                    decoder_inputs[0],
                    decoder_inputs[1],
                    decoder_inputs[2],
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
                    context_bank.index_select(0, unique_context_ids),
                    local_context_indices,
                    motion_tokens=(
                        expanded_motion.index_select(0, chunk_indices)
                        if self.cfg.rgb_context_alignment_enabled
                        else None
                    ),
                    appearance_detail_residual_tokens=(
                        None
                        if expanded_appearance_detail is None
                        else expanded_appearance_detail.index_select(0, chunk_indices)
                    ),
                )
            else:
                decoded = self.image_decoder(
                    *decoder_inputs,
                    (
                        None
                        if expanded_appearance_detail is None
                        else expanded_appearance_detail.index_select(0, chunk_indices)
                    ),
                )
                motion_logit = decoded.new_zeros(
                    decoded.shape[0], 1, decoded.shape[-2], decoded.shape[-1]
                )
                blend = torch.zeros_like(motion_logit)
                flow_pixels = decoded.new_zeros(
                    decoded.shape[0], 2, decoded.shape[-2], decoded.shape[-1]
                )
                disocclusion_logit = torch.zeros_like(motion_logit)
            chunk_valid = valid.index_select(0, chunk_indices)[:, None, None, None]
            decoded_chunks.append(decoded * chunk_valid.to(dtype=decoded.dtype))
            motion_chunks.append(
                motion_logit * chunk_valid.to(dtype=motion_logit.dtype)
            )
            blend_chunks.append(blend * chunk_valid.to(dtype=blend.dtype))
            flow_chunks.append(flow_pixels * chunk_valid.to(dtype=flow_pixels.dtype))
            disocclusion_chunks.append(
                disocclusion_logit * chunk_valid.to(dtype=disocclusion_logit.dtype)
            )
        dense = torch.cat(decoded_chunks, dim=0)
        dense_motion = torch.cat(motion_chunks, dim=0)
        dense_blend = torch.cat(blend_chunks, dim=0)
        dense_flow = torch.cat(flow_chunks, dim=0)
        dense_disocclusion = torch.cat(disocclusion_chunks, dim=0)
        if self.cfg.rgb_context_enabled:
            rgb = dense.view(
                batch, views, frames, 3, self.cfg.rgb_size, self.cfg.rgb_size
            ).permute(0, 2, 1, 3, 4, 5)
            motion = dense_motion.view(
                batch, views, frames, 1, self.cfg.rgb_size, self.cfg.rgb_size
            ).permute(0, 2, 1, 3, 4, 5)
            blend = dense_blend.view(
                batch, views, frames, 1, self.cfg.rgb_size, self.cfg.rgb_size
            ).permute(0, 2, 1, 3, 4, 5)
            flow = dense_flow.view(
                batch, views, frames, 2, self.cfg.rgb_size, self.cfg.rgb_size
            ).permute(0, 2, 1, 3, 4, 5)
            disocclusion_logit = dense_disocclusion.view(
                batch, views, frames, 1, self.cfg.rgb_size, self.cfg.rgb_size
            ).permute(0, 2, 1, 3, 4, 5)
        else:
            rgb = dense.view(
                batch, frames, views, 3, self.cfg.rgb_size, self.cfg.rgb_size
            )
            motion = dense_motion.view(
                batch, frames, views, 1, self.cfg.rgb_size, self.cfg.rgb_size
            )
            blend = dense_blend.view(
                batch, frames, views, 1, self.cfg.rgb_size, self.cfg.rgb_size
            )
            flow = dense_flow.view(
                batch, frames, views, 2, self.cfg.rgb_size, self.cfg.rgb_size
            )
            disocclusion_logit = dense_disocclusion.view(
                batch, frames, views, 1, self.cfg.rgb_size, self.cfg.rgb_size
            )
        return (
            rgb,
            index_tensor,
            motion,
            blend,
            flow,
            disocclusion_logit,
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
        # Legacy factual-decoder profiles keep their independent query basis.
        # Action-owned transport instead follows the verified V7/OXE order:
        # normalized future commands condition the future StateStream before
        # the first shared state block.  It therefore needs no second decoder
        # or another learned future prior.
        if cfg.rgb_action_owned_transport:
            self.factual_decoder_queries = None
            self.factual_decoder_space = None
            self.factual_decoder_time = None
        else:
            self.factual_decoder_queries = nn.Parameter(
                torch.empty(1, cfg.K, cfg.P, cfg.state_hidden)
            )
            self.factual_decoder_space = nn.Parameter(
                torch.empty(1, 1, cfg.P, cfg.state_hidden)
            )
            self.factual_decoder_time = ContinuousTimeEmbedding(
                cfg.state_hidden, cfg
            )
        # Query identity comes from physical time, group/embodiment and current
        # state.  A shared seed avoids learning discrete 20Hz-style position
        # slots and makes the capacity ceiling parameter-count independent.
        self.policy_query_seed = nn.Parameter(torch.empty(1, 1, 1, cfg.action_hidden))
        for parameter in (
            self.state_space,
            self.future_queries,
            self.factual_decoder_queries,
            self.factual_decoder_space,
            self.policy_query_seed,
        ):
            if parameter is not None:
                nn.init.normal_(parameter, std=0.02)
        self.task_state = nn.Linear(cfg.task_dim, cfg.state_hidden, bias=False)
        self.factual_task: Optional[nn.Linear] = (
            None
            if cfg.rgb_action_owned_transport
            else nn.Linear(cfg.task_dim, cfg.state_hidden, bias=True)
        )
        self.task_action = nn.Linear(cfg.task_dim, cfg.action_hidden, bias=False)
        self.state_input_norm = RMSNorm(cfg.state_hidden)

        self.history_action = GroupedSignalEncoder(cfg.action_hidden, cfg)
        self.factual_action: Optional[GroupedSignalEncoder] = (
            GroupedSignalEncoder(
                cfg.state_hidden,
                cfg,
                condition_on_normalization=cfg.rgb_action_owned_transport,
            )
        )
        self.factual_state_action_query_norm: Optional[RMSNorm] = (
            RMSNorm(cfg.state_hidden) if cfg.rgb_action_owned_transport else None
        )
        # Do not normalize the centered action context here.  Its norm carries
        # the physical command magnitude (after source calibration); an
        # RMSNorm would make a 0.25x and 2x command nearly indistinguishable.
        self.factual_state_action_context_norm: Optional[RMSNorm] = None
        self.factual_state_action_cross: Optional[CrossAttention] = (
            CrossAttention(
                cfg.state_hidden,
                cfg.state_hidden,
                cfg.state_heads,
                cfg.dropout,
            )
            if cfg.rgb_action_owned_transport
            else None
        )
        legacy_v7_factual = (
            cfg.factual_v7_early_action_conditioning
            and not cfg.rgb_action_owned_transport
        )
        self.factual_v7_query_action: Optional[CanonicalV7ActionTokenEncoder] = (
            CanonicalV7ActionTokenEncoder(cfg.state_hidden, cfg.K)
            if legacy_v7_factual
            else None
        )
        self.factual_v7_stream_action: Optional[CanonicalV7ActionTokenEncoder] = (
            CanonicalV7ActionTokenEncoder(cfg.action_hidden, cfg.K)
            if legacy_v7_factual
            else None
        )
        self.factual_v7_action_memory: Optional[nn.Linear] = (
            nn.Linear(cfg.action_hidden, cfg.state_hidden, bias=True)
            if legacy_v7_factual
            else None
        )
        self.factual_v7_state_to_action: Optional[nn.Linear] = (
            nn.Linear(cfg.state_hidden, cfg.action_hidden, bias=True)
            if legacy_v7_factual
            else None
        )
        self.factual_v7_group_query_norm: Optional[RMSNorm] = (
            RMSNorm(cfg.state_hidden)
            if legacy_v7_factual
            else None
        )
        self.factual_v7_group_action_norm: Optional[RMSNorm] = (
            RMSNorm(cfg.action_hidden)
            if legacy_v7_factual
            else None
        )
        self.factual_v7_group_query_cross: Optional[CrossAttention] = (
            CrossAttention(
                cfg.state_hidden,
                cfg.action_hidden,
                cfg.state_heads,
                cfg.dropout,
            )
            if legacy_v7_factual
            else None
        )
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
            (
                OriginalV7FactualDecoderLayer(cfg)
                for _ in (
                    () if cfg.rgb_action_owned_transport else range(cfg.dynamics_layers)
                )
            ),
            enabled=cfg.activation_checkpointing,
        )
        self.state_norm = RMSNorm(cfg.state_hidden)
        self.action_norm = RMSNorm(cfg.action_hidden)
        self.token_output = nn.Linear(cfg.state_hidden, cfg.token_dim, bias=False)
        # V7's factual decoder owned its own biased output projection.  Sharing
        # the action-free token head made the factual branch compete with the
        # policy prior and removed another part of the proven V7 contract.
        self.factual_token_output = nn.Linear(
            cfg.state_hidden, cfg.token_dim, bias=True
        )
        self.appearance_dynamics: Optional[nn.Module] = None
        if cfg.appearance_enabled:
            self.appearance_dynamics = (
                FutureSpatialDetailPredictor(cfg)
                if cfg.appearance_state_detail
                else PerViewAppearanceDynamics(cfg)
            )
        self.action_head = UnifiedActionHead(cfg)
        self.geometry_head = NativeGeometryHead(cfg)
        self.rgb_head = NativeRGBDecoder(cfg)
        self.original_v7_rgb_action: Optional[OriginalV7RGBActionAdapter] = (
            OriginalV7RGBActionAdapter(cfg)
            if (
                legacy_v7_factual
                or (
                    cfg.rgb_original_v7_context
                    and cfg.rgb_context_action_scale > 0.0
                )
            )
            else None
        )

        self._action_steps = [
            (cfg.action_layers * (index + 1) // cfg.state_layers)
            - (cfg.action_layers * index // cfg.state_layers)
            for index in range(cfg.state_layers)
        ]
        self._bridge_by_state_layer = {
            state_index: bridge_index
            for bridge_index, state_index in enumerate(cfg.bridge_layers_state)
        }
        factual_bridge_layers = (
            cfg.factual_v7_bridge_layers_state
            if cfg.factual_v7_bridge_layers_state
            else cfg.bridge_layers_state
        )
        self._factual_bridge_by_state_layer = {
            state_index: bridge_index
            for bridge_index, state_index in enumerate(factual_bridge_layers)
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
            self.factual_decoder_queries,
            self.factual_decoder_space,
            self.policy_query_seed,
        ):
            if parameter is not None:
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
    def _run(
        module: nn.Module,
        *args: torch.Tensor,
        enabled: bool,
        **kwargs: object,
    ):
        # Activation checkpointing is installed structurally in __init__ so
        # this call enters the same wrapper under unwrapped, DDP, and FSDP2
        # execution.  ``enabled`` is retained at call sites as an explicit
        # architecture contract and to avoid two model-size-specific paths.
        del enabled
        return module(*args, **kwargs)

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

    def _apply_factual_frame_action_modulation(
        self,
        state: torch.Tensor,
        action_gate: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Refresh future P64 with its same-horizon physical command.

        The initial action-owned injection is an additive seed. With one
        active group, ordinary cross-attention reduces that seed to the same
        channel vector at every patch. Reusing its per-horizon bounded gate
        after each factual state block and multiplying by the current patch
        representation keeps the command present while making its effect
        spatially non-homogeneous. The centered action encoder and bias-free
        projections make a zero command an exact no-op. Observed slots are
        never modified here.
        """

        cfg = self.cfg
        batch = state.shape[0]
        expected_state = (
            batch,
            cfg.T + cfg.K,
            cfg.P,
            cfg.state_hidden,
        )
        if tuple(state.shape) != expected_state:
            raise ValueError(f"factual state must be {expected_state}")
        expected_gate = (batch, cfg.K, cfg.P, cfg.state_hidden)
        if tuple(action_gate.shape) != expected_gate:
            raise ValueError(f"factual frame-action gate must be {expected_gate}")
        if not isfinite(scale) or scale < 0.0:
            raise ValueError("factual frame-action modulation scale is invalid")
        if scale == 0.0:
            return state
        if self.factual_state_action_query_norm is None:
            raise RuntimeError(
                "factual frame-action modulation norm is unavailable"
            )

        future = state[:, cfg.T :]
        normalized_future = self.factual_state_action_query_norm(future)
        spatial_update = action_gate * normalized_future
        return torch.cat(
            (
                state[:, : cfg.T],
                future + float(scale) * spatial_update,
            ),
            dim=1,
        )

    def _apply_factual_direct_action(
        self,
        state: torch.Tensor,
        direct_action: torch.Tensor,
    ) -> torch.Tensor:
        """Add the physical command to its matching future slot before block 0.

        This is the original V7 causal anchor: no scene-query attention,
        nonlinearity, normalization, or horizon pooling is allowed between the
        source-normalized command projection and its future state slot.
        """

        cfg = self.cfg
        batch = state.shape[0]
        expected_state = (batch, cfg.T + cfg.K, cfg.P, cfg.state_hidden)
        if tuple(state.shape) != expected_state:
            raise ValueError(f"factual state must be {expected_state}")
        expected_action = (batch, cfg.K, cfg.state_hidden)
        if tuple(direct_action.shape) != expected_action:
            raise ValueError(
                f"direct factual action must be {expected_action}"
            )
        return torch.cat(
            (
                state[:, : cfg.T],
                state[:, cfg.T :] + direct_action[:, :, None, :],
            ),
            dim=1,
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
        policy_only: bool = False,
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
        # V7 factual streams carry task as one explicit sequence token. Keep a
        # task-free observed P64 seed for those streams while leaving the
        # action-free policy calculation below byte-for-byte unchanged.
        factual_state_seed = self._encode_aux(
            state, aux_values, aux_mask, aux_type_ids
        )
        state = state + self.task_state(task_embedding)[:, None, None]
        state = self._encode_aux(state, aux_values, aux_mask, aux_type_ids)
        # Preserve the exact pre-block input for the factual pass.  The policy
        # pass below remains action-free; the same shared blocks are then run
        # with only future StateStream slots conditioned by the candidate.
        factual_state_input = state

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
            if state_normalization_offset is None or state_normalization_scale is None:
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
        factual_action_input = action
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
                query_token_mask.reshape(batch, query_count * cfg.max_action_groups),
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

        # Deployment does not know a future factual candidate: the policy is
        # the component that proposes it.  Keep that boundary explicit instead
        # of fabricating a future command or running the world/RGB branch with
        # an empty action mask.  Training retains the full forward by default.
        if policy_only:
            if compute_zero_action_control:
                raise ValueError(
                    "policy_only cannot request zero-action world control"
                )
            policy_output: dict[str, torch.Tensor] = {
                "policy_latent": policy_query.transpose(1, 2),
                "world_times_s": world_times_s,
                "policy_query_dt": policy_query_dt,
            }
            policy_output.update(
                self.action_head(
                    policy_query,
                    action_semantic_ids,
                    policy_query_mask,
                    action_normalization_offset,
                    action_normalization_scale,
                )
            )
            return policy_output

        action_free_future = prior_state[:, cfg.T :]
        factual_query: Optional[torch.Tensor] = None
        task_memory: Optional[torch.Tensor] = None
        decoder_prefix_valid: Optional[torch.Tensor] = None
        if not cfg.rgb_action_owned_transport:
            if (
                self.factual_decoder_queries is None
                or self.factual_decoder_space is None
                or self.factual_decoder_time is None
                or self.factual_task is None
            ):
                raise RuntimeError("legacy factual decoder modules are unavailable")
            factual_query = (
                self.factual_decoder_queries.expand(batch, -1, -1, -1)
                + self.factual_decoder_space
                + self.factual_decoder_time(relative_world_time[:, cfg.T :])[
                    :, :, None
                ]
            )
            task_memory = self.state_norm(self.factual_task(task_embedding))[:, None]
            decoder_prefix_valid = torch.ones(
                batch,
                1 + cfg.T * cfg.P,
                dtype=torch.bool,
                device=task_memory.device,
            )

        def encode_factual(
            fine_values: torch.Tensor,
            coarse_values: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]:
            if self.factual_action is None:
                raise RuntimeError("grouped factual summary encoder is unavailable")
            encoded_result = self.factual_action(
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
                normalization_offset=(
                    action_normalization_offset
                    if self.factual_action.condition_on_normalization
                    else None
                ),
                normalization_scale=(
                    action_normalization_scale
                    if self.factual_action.condition_on_normalization
                    else None
                ),
                include_direct_physical=cfg.rgb_action_owned_transport,
            )
            direct: Optional[torch.Tensor] = None
            if cfg.rgb_action_owned_transport:
                encoded, encoded_mask, direct = encoded_result
            else:
                encoded, encoded_mask = encoded_result
            summary: Optional[torch.Tensor] = None
            if (
                cfg.factual_action_residual_scale > 0.0
                or cfg.render_factual_action_residual_scale is not None
                or cfg.rgb_context_action_scale > 0.0
            ):
                weight = encoded_mask[..., None].to(dtype=encoded.dtype)
                summary = (encoded * weight).sum(dim=2)
                summary = summary / weight.sum(dim=2).clamp_min(1.0)
            return encoded, encoded_mask, summary, direct

        def refine_factual(
            encoded: torch.Tensor,
            encoded_mask: torch.Tensor,
            summary: Optional[torch.Tensor],
            direct_summary: Optional[torch.Tensor],
            observed_state: torch.Tensor,
            *,
            repeats: int,
            residual_scale: float,
        ) -> torch.Tensor:
            if tuple(encoded.shape[:3]) != (
                batch,
                cfg.K,
                cfg.max_action_groups,
            ):
                raise ValueError("factual action memory must align to [B,K,G]")
            if tuple(encoded_mask.shape) != tuple(encoded.shape[:3]):
                raise ValueError("factual action mask must align to action memory")
            if cfg.rgb_action_owned_transport:
                if repeats != 1:
                    raise ValueError(
                        "causal factual StateStream is executed exactly once"
                    )
                if (
                    self.factual_state_action_query_norm is None
                    or self.factual_state_action_cross is None
                ):
                    raise RuntimeError(
                        "group-preserving factual state conditioner is unavailable"
                    )
                if direct_summary is None or tuple(direct_summary.shape) != (
                    batch,
                    cfg.K,
                    cfg.state_hidden,
                ):
                    raise ValueError(
                        "direct physical action must align to [B,K,state_hidden]"
                    )

                # Preserve each physical owner until it reaches every future
                # patch. This is the V7/OXE first-conditioning point, extended
                # without averaging base/arm/mode owners into one token. The
                # resulting per-horizon gate is reused after every state block;
                # otherwise a one-group command remains a common channel bias
                # that the following shared trunk can rotate away.
                future_seed = factual_state_input[:, cfg.T :]
                grouped_action = encoded * encoded_mask[..., None].to(
                    dtype=encoded.dtype
                )
                flat_grouped_action = grouped_action.reshape(
                    batch * cfg.K,
                    cfg.max_action_groups,
                    cfg.state_hidden,
                )
                group_valid = encoded_mask.reshape(
                    batch * cfg.K, cfg.max_action_groups
                )
                safe_group_valid = group_valid.clone()
                safe_group_valid[:, 0] |= ~safe_group_valid.any(dim=1)
                action_update = self.factual_state_action_cross(
                    self.factual_state_action_query_norm(future_seed).reshape(
                        batch * cfg.K, cfg.P, cfg.state_hidden
                    ),
                    flat_grouped_action,
                    allowed_mask=safe_group_valid[:, None, None, :],
                ).reshape(batch, cfg.K, cfg.P, cfg.state_hidden)
                action_gate = torch.tanh(action_update)
                candidate_state = self._apply_factual_direct_action(
                    factual_state_input,
                    direct_summary,
                )
                candidate_action = factual_action_input
                candidate_action_index = 0
                per_block_action_scale = residual_scale / float(
                    max(1, len(self.state_blocks))
                )
                for state_index, state_block in enumerate(self.state_blocks):
                    candidate_state = self._run(
                        state_block,
                        candidate_state,
                        enabled=cfg.activation_checkpointing,
                    )
                    candidate_state = self._apply_factual_frame_action_modulation(
                        candidate_state,
                        action_gate,
                        scale=per_block_action_scale,
                    )
                    for _ in range(self._action_steps[state_index]):
                        candidate_action = self._run(
                            self.action_blocks[candidate_action_index],
                            candidate_action,
                            action_times,
                            action_mask,
                            action_task,
                            task_token_mask,
                            enabled=cfg.activation_checkpointing,
                        )
                        candidate_action_index += 1
                    bridge_index = self._bridge_by_state_layer.get(state_index)
                    if bridge_index is not None:
                        candidate_state, candidate_action = self._run(
                            self.bridges[bridge_index],
                            candidate_state,
                            candidate_action,
                            action_mask,
                            factual=True,
                            enabled=cfg.activation_checkpointing,
                        )
                while candidate_action_index < len(self.action_blocks):
                    candidate_action = self._run(
                        self.action_blocks[candidate_action_index],
                        candidate_action,
                        action_times,
                        action_mask,
                        action_task,
                        task_token_mask,
                        enabled=cfg.activation_checkpointing,
                    )
                    candidate_action_index += 1
                return self.state_norm(candidate_state)[:, cfg.T :]

            if tuple(observed_state.shape) != (
                batch,
                cfg.T,
                cfg.P,
                cfg.state_hidden,
            ):
                raise ValueError("factual state memory must align to [B,T,P,D]")
            observed_memory = observed_state.reshape(
                batch, cfg.T * cfg.P, cfg.state_hidden
            )
            if (
                task_memory is None
                or decoder_prefix_valid is None
                or factual_query is None
            ):
                raise RuntimeError("legacy factual decoder inputs are unavailable")
            decoder_memory_prefix = torch.cat((task_memory, observed_memory), dim=1)
            action_memory = encoded.reshape(
                batch, cfg.K * cfg.max_action_groups, cfg.state_hidden
            )
            action_valid = encoded_mask.reshape(batch, cfg.K * cfg.max_action_groups)
            memory = torch.cat((decoder_memory_prefix, action_memory), dim=1)
            memory_valid = torch.cat((decoder_prefix_valid, action_valid), dim=1)
            refined = factual_query
            if residual_scale > 0.0:
                assert summary is not None
                refined = refined + (residual_scale * summary[:, :, None, :])
            for _ in range(repeats):
                for dynamics_block in self.dynamics_blocks:
                    refined = self._run(
                        dynamics_block,
                        refined,
                        memory,
                        memory_valid,
                        enabled=cfg.activation_checkpointing,
                    )
            return refined

        factual_encoded: Optional[torch.Tensor] = None
        factual_encoded_mask: Optional[torch.Tensor] = None
        zero_encoded_mask: Optional[torch.Tensor] = None
        centered_encoded: Optional[torch.Tensor] = None
        noop_centered_encoded: Optional[torch.Tensor] = None
        centered_summary: Optional[torch.Tensor] = None
        noop_centered_summary: Optional[torch.Tensor] = None
        centered_direct_summary: Optional[torch.Tensor] = None
        noop_centered_direct_summary: Optional[torch.Tensor] = None
        zero_physical_fine_action = normalized_physical_zero_action(
            future_factual_fine_action_values,
            future_factual_fine_action_mask,
            action_semantic_ids,
            action_normalization_offset,
            action_normalization_scale,
            group_axis=2,
        )
        zero_physical_coarse_action = normalized_physical_zero_action(
            future_factual_coarse_action_values,
            future_factual_coarse_action_mask,
            action_semantic_ids,
            action_normalization_offset,
            action_normalization_scale,
            group_axis=2,
        )
        noop_physical_fine_action = normalized_physical_noop_action(
            future_factual_fine_action_values,
            future_factual_fine_action_mask,
            action_semantic_ids,
            action_normalization_offset,
            action_normalization_scale,
            group_axis=2,
        )
        noop_physical_coarse_action = normalized_physical_noop_action(
            future_factual_coarse_action_values,
            future_factual_coarse_action_mask,
            action_semantic_ids,
            action_normalization_offset,
            action_normalization_scale,
            group_axis=2,
        )
        if self.factual_action is not None:
            (
                factual_encoded,
                factual_encoded_mask,
                factual_summary,
                factual_direct,
            ) = encode_factual(
                future_factual_fine_action_values,
                future_factual_coarse_action_values,
            )
            # Center against the exact same masked encoding of the zero
            # physical command. This removes time/semantic/group/embodiment
            # constants while preserving every real grouped action token.
            (
                zero_encoded,
                zero_encoded_mask,
                zero_summary,
                zero_direct,
            ) = encode_factual(
                zero_physical_fine_action,
                zero_physical_coarse_action,
            )
            if not torch.equal(factual_encoded_mask, zero_encoded_mask):
                raise RuntimeError("factual and zero action masks must be identical")
            # Numeric zero is a coordinate origin, not a frozen control
            # branch.  Keep both encoder evaluations differentiable so the
            # parameter gradient is taken with respect to the physical
            # displacement (factual - zero), matching the centered forward
            # value.  Detaching this anchor leaves the forward value correct
            # but trains the projection on the source z-score itself.
            zero_encoded_anchor = zero_encoded
            centered_encoded = factual_encoded - zero_encoded_anchor
            (
                noop_encoded,
                noop_encoded_mask,
                noop_summary,
                noop_direct,
            ) = encode_factual(
                noop_physical_fine_action,
                noop_physical_coarse_action,
            )
            if not torch.equal(factual_encoded_mask, noop_encoded_mask):
                raise RuntimeError("factual and no-op action masks must be identical")
            noop_centered_encoded = noop_encoded - zero_encoded_anchor
            if factual_summary is not None:
                if zero_summary is None or noop_summary is None:
                    raise RuntimeError("zero/no-op action summaries are unavailable")
                zero_summary_anchor = zero_summary
                centered_summary = factual_summary - zero_summary_anchor
                noop_centered_summary = noop_summary - zero_summary_anchor
            if cfg.rgb_action_owned_transport:
                if factual_direct is None or zero_direct is None or noop_direct is None:
                    raise RuntimeError("direct physical action features are unavailable")
                direct_anchor = zero_direct
                centered_direct = factual_direct - direct_anchor
                noop_centered_direct = noop_direct - direct_anchor
                direct_weight = factual_encoded_mask[..., None].to(
                    dtype=centered_direct.dtype
                )
                direct_denom = direct_weight.sum(dim=2).clamp_min(1.0)
                centered_direct_summary = (
                    centered_direct * direct_weight
                ).sum(dim=2) / direct_denom
                noop_centered_direct_summary = (
                    noop_centered_direct * direct_weight
                ).sum(dim=2) / direct_denom

        canonical_action_cond: Optional[torch.Tensor] = None
        canonical_grouped_action: Optional[torch.Tensor] = None
        canonical_group_mask: Optional[torch.Tensor] = None
        canonical_single_group: Optional[torch.Tensor] = None
        canonical_noop_action_cond: Optional[torch.Tensor] = None
        canonical_noop_grouped_action: Optional[torch.Tensor] = None
        legacy_v7_factual = (
            cfg.factual_v7_early_action_conditioning
            and not cfg.rgb_action_owned_transport
        )
        if legacy_v7_factual or (
            cfg.rgb_original_v7_context and cfg.rgb_context_action_scale > 0.0
        ):
            if self.original_v7_rgb_action is None:
                raise RuntimeError("canonical V7 physical action adapter is unavailable")
            canonical_grouped_action, canonical_group_mask = self.original_v7_rgb_action(
                fine_values=future_factual_fine_action_values,
                fine_dim_mask=future_factual_fine_action_mask,
                fine_sample_mask=future_factual_fine_sample_mask,
                coarse_values=future_factual_coarse_action_values,
                coarse_dim_mask=future_factual_coarse_action_mask,
                action_semantic_ids=action_semantic_ids,
                group_mask=action_group_mask,
                normalization_offset=action_normalization_offset,
                normalization_scale=action_normalization_scale,
                return_grouped=True,
            )
            canonical_single_group = canonical_group_mask.sum(dim=1).eq(1)
            canonical_group_index = canonical_group_mask.to(dtype=torch.long).argmax(
                dim=1
            )
            selected_action = OriginalV7RGBActionAdapter._select_future_group(
                canonical_grouped_action, canonical_group_index
            )
            # The original V7 direct query/RGB skip is exact only for one arm
            # owner. Multi-group samples stay fully conditioned through the
            # factual streams below; a raw group mean would cancel commands
            # from different physical owners, so the direct skip is disabled.
            canonical_action_cond = torch.where(
                canonical_single_group[:, None, None],
                selected_action,
                torch.zeros_like(selected_action),
            )
            if legacy_v7_factual:
                (
                    canonical_noop_grouped_action,
                    canonical_noop_group_mask,
                ) = self.original_v7_rgb_action(
                    fine_values=noop_physical_fine_action,
                    fine_dim_mask=future_factual_fine_action_mask,
                    fine_sample_mask=future_factual_fine_sample_mask,
                    coarse_values=noop_physical_coarse_action,
                    coarse_dim_mask=future_factual_coarse_action_mask,
                    action_semantic_ids=action_semantic_ids,
                    group_mask=action_group_mask,
                    normalization_offset=action_normalization_offset,
                    normalization_scale=action_normalization_scale,
                    return_grouped=True,
                )
                if not torch.equal(canonical_group_mask, canonical_noop_group_mask):
                    raise RuntimeError("factual and no-op canonical groups must match")
                selected_noop_action = OriginalV7RGBActionAdapter._select_future_group(
                    canonical_noop_grouped_action, canonical_group_index
                )
                canonical_noop_action_cond = torch.where(
                    canonical_single_group[:, None, None],
                    selected_noop_action,
                    torch.zeros_like(selected_noop_action),
                )

        def encode_factual_action_stream(
            fine_values: torch.Tensor,
            coarse_values: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.history_action(
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

        def run_v7_fullsequence_factual(
            physical_action: torch.Tensor,
            physical_action_mask: torch.Tensor,
            canonical_action: torch.Tensor,
            grouped_canonical_action: torch.Tensor,
            canonical_groups: torch.Tensor,
            single_group: torch.Tensor,
            *,
            repeats: int,
            residual_scale: float,
        ) -> torch.Tensor:
            """Run the exact V7 dual full-sequence factual topology.

            Both streams contain ``task + observed TP + future KG`` from
            block zero onward.  State/action blocks are shared with the
            policy pass but run non-causally over these explicit sequences;
            every configured bridge is applied once and synchronously.  This
            adds no duplicate trunk and never exposes the future candidate to
            the already-computed policy/action-free result.
            """

            expected_action = (
                batch,
                cfg.K,
                cfg.max_action_groups,
                cfg.action_hidden,
            )
            if tuple(physical_action.shape) != expected_action:
                raise ValueError("factual ActionStream must be [B,K,G,A]")
            if tuple(physical_action_mask.shape) != expected_action[:3]:
                raise ValueError("factual action mask must be [B,K,G]")
            if tuple(canonical_action.shape) != (batch, cfg.K, 7):
                raise ValueError("canonical factual action must be [B,K,7]")
            if tuple(grouped_canonical_action.shape) != (
                batch,
                cfg.K,
                cfg.max_action_groups,
                7,
            ):
                raise ValueError("grouped canonical action must be [B,K,G,7]")
            if tuple(canonical_groups.shape) != (
                batch,
                cfg.max_action_groups,
            ):
                raise ValueError("canonical group mask must be [B,G]")
            if tuple(single_group.shape) != (batch,):
                raise ValueError("single-group selector must be [B]")
            required = (
                self.factual_v7_query_action,
                self.factual_v7_stream_action,
                self.factual_v7_action_memory,
                self.factual_v7_state_to_action,
                self.factual_v7_group_query_norm,
                self.factual_v7_group_action_norm,
                self.factual_v7_group_query_cross,
            )
            if any(module is None for module in required):
                raise RuntimeError("full-sequence V7 factual modules are unavailable")
            assert self.factual_v7_query_action is not None
            assert self.factual_v7_stream_action is not None
            assert self.factual_v7_action_memory is not None
            assert self.factual_v7_state_to_action is not None
            assert self.factual_v7_group_query_norm is not None
            assert self.factual_v7_group_action_norm is not None
            assert self.factual_v7_group_query_cross is not None

            observed_state = factual_state_seed[:, : cfg.T]
            observed_state_flat = observed_state.reshape(
                batch, cfg.T * cfg.P, cfg.state_hidden
            )
            factual_action_times = relative_world_time[:, cfg.T :, None].expand(
                -1, -1, cfg.max_action_groups
            )
            # Keep group/semantic/embodiment identity in the token.  Centering
            # the entire token would erase group identity; only the canonical
            # physical values are centered upstream by the adapter.
            factual_action = physical_action + self.action_time(
                factual_action_times
            )
            factual_action = factual_action * physical_action_mask[..., None].to(
                dtype=factual_action.dtype
            )
            grouped_state_action = self.factual_v7_query_action(
                grouped_canonical_action
            )
            grouped_stream_action = self.factual_v7_stream_action(
                grouped_canonical_action
            )
            physical_group = canonical_groups[:, None, :, None]
            state_action = torch.where(
                physical_group,
                grouped_state_action,
                self.factual_v7_action_memory(factual_action),
            ).reshape(batch, cfg.K * cfg.max_action_groups, cfg.state_hidden)
            action_observed = self.factual_v7_state_to_action(
                observed_state_flat
            )
            action_future = torch.where(
                physical_group,
                grouped_stream_action,
                factual_action,
            ).reshape(
                batch, cfg.K * cfg.max_action_groups, cfg.action_hidden
            )
            state_sequence = torch.cat(
                (task_memory, observed_state_flat, state_action), dim=1
            )
            action_sequence = torch.cat(
                (action_task[:, None], action_observed, action_future), dim=1
            )
            action_valid = physical_action_mask.reshape(
                batch, cfg.K * cfg.max_action_groups
            )
            sequence_valid = torch.cat(
                (
                    torch.ones(
                        batch,
                        1 + cfg.T * cfg.P,
                        dtype=torch.bool,
                        device=physical_action.device,
                    ),
                    action_valid,
                ),
                dim=1,
            )
            expected_length = 1 + cfg.T * cfg.P + cfg.K * cfg.max_action_groups
            if state_sequence.shape[1] != expected_length:
                raise RuntimeError("state factual stream lost full TP/KG topology")
            if action_sequence.shape[1] != expected_length:
                raise RuntimeError("action factual stream lost full TP/KG topology")
            state_sequence = state_sequence * sequence_valid[..., None].to(
                dtype=state_sequence.dtype
            )
            action_sequence = action_sequence * sequence_valid[..., None].to(
                dtype=action_sequence.dtype
            )

            dummy_times = torch.zeros(
                batch,
                expected_length,
                dtype=relative_world_time.dtype,
                device=relative_world_time.device,
            )
            dummy_task_mask = torch.zeros_like(sequence_valid)
            action_index = 0
            bridge_calls = 0
            for state_index, state_block in enumerate(self.state_blocks):
                state_sequence = self._run(
                    state_block,
                    state_sequence,
                    sequence_valid,
                    enabled=cfg.activation_checkpointing,
                )
                for _ in range(self._action_steps[state_index]):
                    action_sequence = self._run(
                        self.action_blocks[action_index],
                        action_sequence,
                        dummy_times,
                        sequence_valid,
                        action_task,
                        dummy_task_mask,
                        enabled=cfg.activation_checkpointing,
                    )
                    action_index += 1
                bridge_index = self._factual_bridge_by_state_layer.get(state_index)
                if bridge_index is not None:
                    state_sequence, action_sequence = self._run(
                        self.bridges[bridge_index],
                        state_sequence,
                        action_sequence,
                        sequence_valid,
                        enabled=cfg.activation_checkpointing,
                    )
                    bridge_calls += 1
            while action_index < len(self.action_blocks):
                action_sequence = self._run(
                    self.action_blocks[action_index],
                    action_sequence,
                    dummy_times,
                    sequence_valid,
                    action_task,
                    dummy_task_mask,
                    enabled=cfg.activation_checkpointing,
                )
                action_index += 1
            if bridge_calls != len(self._factual_bridge_by_state_layer):
                raise RuntimeError("factual V7 bridge schedule was not applied once")

            state_sequence = self.state_norm(state_sequence)
            action_sequence = self.action_norm(action_sequence)
            state_sequence = state_sequence * sequence_valid[..., None].to(
                dtype=state_sequence.dtype
            )
            action_sequence = action_sequence * sequence_valid[..., None].to(
                dtype=action_sequence.dtype
            )
            # Exact V7 decoder memory is the complete post-block StateStream.
            # It already contains transformed KG tokens; concatenating the
            # ActionStream here would duplicate the candidate.
            memory = state_sequence
            memory_valid = sequence_valid

            exact_query_update = self.factual_v7_query_action(canonical_action)
            post_group_action = action_sequence[:, 1 + cfg.T * cfg.P :].reshape(
                batch * cfg.K,
                cfg.max_action_groups,
                cfg.action_hidden,
            )
            query_by_horizon = self.factual_v7_group_query_norm(
                factual_query
            ).reshape(batch * cfg.K, cfg.P, cfg.state_hidden)
            group_valid = physical_action_mask.reshape(
                batch * cfg.K, cfg.max_action_groups
            )
            safe_group_valid = group_valid.clone()
            safe_group_valid[:, 0] |= ~safe_group_valid.any(dim=1)
            grouped_query_update = self.factual_v7_group_query_cross(
                query_by_horizon,
                self.factual_v7_group_action_norm(post_group_action),
                allowed_mask=safe_group_valid[:, None, None, :],
            ).reshape(batch, cfg.K, cfg.P, cfg.state_hidden)
            query_update = torch.where(
                single_group[:, None, None, None],
                exact_query_update[:, :, None, :],
                grouped_query_update,
            )
            refined = factual_query
            if residual_scale > 0.0:
                refined = refined + residual_scale * query_update
            for _ in range(repeats):
                for dynamics_block in self.dynamics_blocks:
                    refined = self._run(
                        dynamics_block,
                        refined,
                        memory,
                        memory_valid,
                        enabled=cfg.activation_checkpointing,
                    )
            return refined

        def encode_v7_factual_state(
            action_summary: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if cfg.rgb_action_owned_transport:
                return factual_state_input[:, : cfg.T]
            if not cfg.factual_v7_early_action_conditioning:
                return prior_state[:, : cfg.T]
            if action_summary is None or tuple(action_summary.shape) != (
                batch,
                cfg.K,
                cfg.state_hidden,
            ):
                raise ValueError(
                    "early factual action summary must align to [B,K,D]"
                )
            # Exact V7 appends one physical-action token per future horizon to
            # the observed state sequence before every encoder layer.  The V8
            # trunk is factorized, so the equivalent shared-parameter form is
            # to append the K centered command tokens to the spatial token set
            # of every frame.  Spatial attention then lets every world patch
            # read the command before the first block; the action tokens are
            # removed again before the factual decoder.  This is materially
            # different from adding a channel bias after the state trunk.
            action_tokens = (
                float(cfg.factual_v7_early_action_scale) * action_summary
            )
            factual_state = torch.cat(
                (
                    factual_state_seed[:, : cfg.T],
                    action_tokens[:, None].expand(-1, cfg.T, -1, -1),
                ),
                dim=2,
            )
            for state_block in self.state_blocks:
                factual_state = self._run(
                    state_block,
                    factual_state,
                    enabled=cfg.activation_checkpointing,
                )
            return self.state_norm(factual_state[:, :, : cfg.P])

        factual_action_stream: Optional[torch.Tensor] = None
        zero_action_stream: Optional[torch.Tensor] = None
        factual_action_stream_mask: Optional[torch.Tensor] = None
        zero_action_stream_mask: Optional[torch.Tensor] = None
        if legacy_v7_factual:
            if canonical_action_cond is None:
                raise RuntimeError("canonical V7 action is unavailable")
            if canonical_single_group is None:
                raise RuntimeError("canonical V7 group ownership is unavailable")
            if canonical_grouped_action is None or canonical_group_mask is None:
                raise RuntimeError("grouped canonical V7 action is unavailable")
            factual_action_stream, factual_action_stream_mask = (
                encode_factual_action_stream(
                    future_factual_fine_action_values,
                    future_factual_coarse_action_values,
                )
            )
            zero_action_stream, zero_action_stream_mask = (
                encode_factual_action_stream(
                    noop_physical_fine_action,
                    noop_physical_coarse_action,
                )
            )
            if not torch.equal(
                factual_action_stream_mask, zero_action_stream_mask
            ):
                raise RuntimeError("factual ActionStream masks must match")
            factual_future = run_v7_fullsequence_factual(
                factual_action_stream,
                factual_action_stream_mask,
                canonical_action_cond,
                canonical_grouped_action,
                canonical_group_mask,
                canonical_single_group,
                repeats=cfg.factual_dynamics_repeats,
                residual_scale=cfg.factual_action_residual_scale,
            )
            factual_observed_state = None
        else:
            if centered_encoded is None or factual_encoded_mask is None:
                raise RuntimeError("centered factual action tokens are unavailable")
            factual_observed_state = encode_v7_factual_state(centered_summary)
            factual_future = refine_factual(
                centered_encoded,
                factual_encoded_mask,
                centered_summary,
                centered_direct_summary,
                factual_observed_state,
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
        elif legacy_v7_factual:
            assert factual_action_stream is not None
            assert factual_action_stream_mask is not None
            assert canonical_action_cond is not None
            assert canonical_grouped_action is not None
            assert canonical_group_mask is not None
            assert canonical_single_group is not None
            render_future = run_v7_fullsequence_factual(
                factual_action_stream,
                factual_action_stream_mask,
                canonical_action_cond,
                canonical_grouped_action,
                canonical_group_mask,
                canonical_single_group,
                repeats=render_repeats,
                residual_scale=render_residual_scale,
            )
        else:
            assert factual_observed_state is not None
            if centered_encoded is None or factual_encoded_mask is None:
                raise RuntimeError("centered render action tokens are unavailable")
            render_future = refine_factual(
                centered_encoded,
                factual_encoded_mask,
                centered_summary,
                centered_direct_summary,
                factual_observed_state,
                repeats=render_repeats,
                residual_scale=render_residual_scale,
            )
        rgb_render_future = (
            action_free_future if cfg.rgb_render_action_free_prior else render_future
        )
        zero_action_pred_tokens: Optional[torch.Tensor] = None
        rgb_action_summary: Optional[torch.Tensor] = None
        if cfg.rgb_context_action_scale > 0.0:
            if cfg.rgb_action_owned_transport:
                if centered_summary is None:
                    raise RuntimeError(
                        "normalized factual RGB action effect is unavailable"
                    )
                # P64 is the causal anchor-to-horizon motion owner.  Keep this
                # direct renderer hint local: summing a mixed learned latent
                # would incorrectly accumulate absolute gripper/mode values
                # and rotations.  Temporal composition happens in StateStream.
                rgb_action_summary = centered_summary
            elif cfg.rgb_original_v7_context:
                if canonical_action_cond is None:
                    raise RuntimeError("canonical V7 RGB action is unavailable")
                rgb_action_summary = canonical_action_cond
            else:
                if centered_summary is None:
                    raise RuntimeError("centered RGB action summary is unavailable")
                rgb_action_summary = centered_summary
        appearance_action_tokens: Optional[torch.Tensor] = None
        appearance_action_mask: Optional[torch.Tensor] = None
        if cfg.appearance_action_residual_scale > 0.0:
            # Keep every grouped action token and remove only the same-mask
            # action-independent encoder component. The appearance lane can
            # then resolve physical direction per patch before its own
            # spatial/temporal reasoning, while a zero command stays exact 0.
            if centered_encoded is None or factual_encoded_mask is None:
                raise RuntimeError("centered appearance action tokens are unavailable")
            appearance_action_tokens = centered_encoded
            appearance_action_mask = factual_encoded_mask
        if compute_zero_action_control:
            if legacy_v7_factual:
                assert zero_action_stream is not None
                assert zero_action_stream_mask is not None
                assert canonical_grouped_action is not None
                assert canonical_group_mask is not None
                assert canonical_single_group is not None
                assert canonical_noop_action_cond is not None
                assert canonical_noop_grouped_action is not None
                zero_action_future = run_v7_fullsequence_factual(
                    zero_action_stream,
                    zero_action_stream_mask,
                    canonical_noop_action_cond,
                    canonical_noop_grouped_action,
                    canonical_group_mask,
                    canonical_single_group,
                    repeats=cfg.factual_dynamics_repeats,
                    residual_scale=cfg.factual_action_residual_scale,
                )
            else:
                if noop_centered_encoded is None or zero_encoded_mask is None:
                    raise RuntimeError("centered no-op action tokens are unavailable")
                zero_factual_observed_state = encode_v7_factual_state(
                    noop_centered_summary
                )
                zero_action_future = refine_factual(
                    noop_centered_encoded,
                    zero_encoded_mask,
                    noop_centered_summary,
                    noop_centered_direct_summary,
                    zero_factual_observed_state,
                    repeats=cfg.factual_dynamics_repeats,
                    residual_scale=cfg.factual_action_residual_scale,
                )
            zero_action_pred_tokens = self.factual_token_output(zero_action_future)

        action_free_pred_tokens = self.token_output(action_free_future)
        pred_tokens = self.factual_token_output(factual_future)
        render_pred_tokens = (
            pred_tokens
            if render_future is factual_future
            else self.factual_token_output(render_future)
        )
        rgb_render_pred_tokens = (
            action_free_pred_tokens
            if rgb_render_future is action_free_future
            else render_pred_tokens
        )
        appearance_for_rgb: Optional[torch.Tensor] = None
        appearance_context_for_rgb: Optional[torch.Tensor] = None
        appearance_ratio = pred_tokens.new_zeros(())
        appearance_pred: Optional[torch.Tensor] = None
        appearance_pred_mask: Optional[torch.Tensor] = None
        appearance_teacher_pred: Optional[torch.Tensor] = None
        appearance_teacher_mask: Optional[torch.Tensor] = None
        appearance_autoregressive_pred: Optional[torch.Tensor] = None
        appearance_autoregressive_mask: Optional[torch.Tensor] = None
        if cfg.appearance_enabled:
            if cfg.appearance_state_detail:
                if self.appearance_dynamics is None:
                    raise RuntimeError("state detail predictor is unavailable")
                view_present = view_mask.bool().any(dim=1)
                detail_mask = view_present[:, None, :, None].expand(
                    -1, cfg.K, -1, cfg.appearance_P
                )
                if target_appearance_mask is not None:
                    if tuple(target_appearance_mask.shape) != tuple(detail_mask.shape):
                        raise ValueError(
                            "target appearance mask must align to state detail"
                        )
                    detail_mask = detail_mask & target_appearance_mask.bool()
                appearance_pred, appearance_pred_mask = self.appearance_dynamics(
                    rgb_render_future,
                    detail_mask,
                )
                appearance_for_rgb = appearance_pred
            else:
                if (
                    self.appearance_dynamics is None
                    or appearance_context_tokens is None
                    or appearance_context_mask is None
                ):
                    raise ValueError(
                        "dual-path model requires appearance context tokens and mask"
                    )
                appearance_ratio = torch.as_tensor(
                    appearance_teacher_ratio,
                    dtype=pred_tokens.dtype,
                    device=pred_tokens.device,
                )
                if appearance_ratio.numel() != 1 or not bool(
                    ((appearance_ratio >= 0) & (appearance_ratio <= 1)).all()
                ):
                    raise ValueError(
                        "appearance teacher ratio must be a scalar in [0,1]"
                    )
                if cfg.appearance_flow_aligned_detail:
                    appearance_ratio = appearance_ratio * 0.0
                if target_appearance_tokens is None:
                    if bool(appearance_ratio > 0):
                        raise ValueError(
                            "teacher forcing requires target appearance tokens"
                        )
                elif (
                    target_appearance_mask is None
                    or target_appearance_mask.shape
                    != target_appearance_tokens.shape[:-1]
                ):
                    raise ValueError("target appearance mask must align to targets")

                appearance_context_for_rgb = torch.zeros_like(
                    appearance_context_tokens[:, 0]
                )
                for context_index in range(int(appearance_context_tokens.shape[1])):
                    appearance_context_for_rgb = torch.where(
                        appearance_context_mask[:, context_index, ..., None].bool(),
                        appearance_context_tokens[:, context_index],
                        appearance_context_for_rgb,
                    )
                appearance_context_for_rgb = F.layer_norm(
                    appearance_context_for_rgb.float(),
                    (appearance_context_for_rgb.shape[-1],),
                ).to(dtype=appearance_context_tokens.dtype)

                # Legacy appearance profiles remain readable for old evidence,
                # but V8 core never selects this target/context-dependent path.
                rollout_steps = cfg.K
                (
                    appearance_pred,
                    appearance_pred_mask,
                    appearance_teacher_pred,
                    appearance_teacher_mask,
                    appearance_autoregressive_pred,
                    appearance_autoregressive_mask,
                ) = self.appearance_dynamics(
                    appearance_context_tokens,
                    appearance_context_mask,
                    rgb_render_future,
                    relative_world_time,
                    target_appearance_mask,
                    appearance_action_tokens,
                    appearance_action_mask,
                    target_appearance_tokens,
                    rollout_steps,
                )
                appearance_for_rgb = appearance_pred * appearance_pred_mask[
                    ..., None
                ].to(dtype=appearance_pred.dtype)
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
            if cfg.appearance_flow_aligned_detail:
                output["appearance_flow_aligned_detail"] = appearance_pred.new_ones(())
            if cfg.appearance_state_detail:
                output["appearance_state_detail"] = appearance_pred.new_ones(())
        if (
            appearance_teacher_pred is not None
            and appearance_teacher_mask is not None
            and appearance_teacher_pred.shape[1]
        ):
            output["appearance_teacher_pred_tokens"] = appearance_teacher_pred
            output["appearance_teacher_pred_mask"] = appearance_teacher_mask
        if (
            appearance_autoregressive_pred is not None
            and appearance_autoregressive_mask is not None
            and appearance_autoregressive_pred.shape[1]
        ):
            output["appearance_autoregressive_pred_tokens"] = (
                appearance_autoregressive_pred
            )
            output["appearance_autoregressive_pred_mask"] = (
                appearance_autoregressive_mask
            )
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
        (
            rgb,
            rgb_indices,
            rgb_motion_logit,
            rgb_blend,
            rgb_flow_pixels,
            rgb_disocclusion_logit,
        ) = self._run(
            self.rgb_head,
            rgb_render_pred_tokens,
            rgb_frame_indices,
            rgb_view_mask,
            appearance_for_rgb,
            (
                appearance_context_for_rgb
                if cfg.rgb_context_appearance_delta_scale > 0.0
                and not cfg.appearance_flow_aligned_detail
                and not cfg.appearance_state_detail
                else None
            ),
            rgb_render_future if cfg.appearance_enabled else None,
            rgb_action_summary,
            task_embedding if cfg.rgb_context_enabled else None,
            context_rgb if cfg.rgb_context_enabled else None,
            context_rgb_mask if cfg.rgb_context_enabled else None,
            enabled=cfg.activation_checkpointing,
        )
        output["rgb"] = rgb
        output["rgb_frame_indices"] = rgb_indices
        if cfg.rgb_context_enabled:
            output["rgb_motion_logit"] = rgb_motion_logit
            output["rgb_blend"] = rgb_blend
        if cfg.rgb_context_alignment_enabled or cfg.rgb_action_owned_transport:
            output["rgb_flow_pixels"] = rgb_flow_pixels
            output["rgb_disocclusion_logit"] = rgb_disocclusion_logit
        return output

    def iter_fsdp_units(self) -> Iterable[nn.Module]:
        """Yield communication-sized modules for bottom-up FSDP2 wrapping."""

        yield self.view_fuser
        yield from self.state_blocks
        yield from self.action_blocks
        yield from self.bridges
        yield from self.dynamics_blocks
        if self.appearance_dynamics is not None:
            if self.cfg.appearance_state_detail:
                yield self.appearance_dynamics
            else:
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
        if (
            self.appearance_dynamics is not None
            and not self.cfg.appearance_state_detail
        ):
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
    for key in (
        "bridge_layers_state",
        "factual_v7_bridge_layers_state",
        "rgb_decode_indices",
    ):
        if key in values:
            values[key] = tuple(int(item) for item in values[key])
    cfg = NativeWorldModelConfig(**values)
    cfg.validate()
    return cfg
