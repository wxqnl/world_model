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
from math import isqrt, log
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
    bridge_layers_state: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29)
    bridge_heads: int = 16
    dynamics_layers: int = 4

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
    geom_hidden: int = 768

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
        if any(index < 0 or index >= self.state_layers for index in self.bridge_layers_state):
            raise ValueError("bridge layer index is outside state trunk")
        if self.dynamics_layers <= 0:
            raise ValueError("dynamics_layers must be positive")
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
            raise ValueError("rgb_decode_indices must refer to future world-state steps")
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
        query = self.query(query_value).view(
            batch, query_length, self.heads, self.head_dim
        ).transpose(1, 2)
        key, value = self.key_value(context).chunk(2, dim=-1)
        key = key.view(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, context_length, self.heads, self.head_dim).transpose(1, 2)
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
        frequencies = torch.exp(torch.linspace(log_min_frequency, log_max_frequency, half))
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
        temporal = temporal + self.temporal(self.temporal_norm(temporal), is_causal=True)
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
        semantic = (
            semantic * semantic_valid[..., None].to(dtype=semantic.dtype)
        ).sum(dim=2) / semantic_valid.sum(dim=2).clamp_min(1)[..., None]
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
            raise ValueError("every group with state semantics requires measured current state")
        if not bool((dim_mask & group_mask[..., None]).any(dim=(1, 2)).all()):
            raise ValueError("every sample requires at least one measured current-state value")
        pair = torch.cat(
            (values * dim_mask.to(dtype=values.dtype), dim_mask.to(dtype=values.dtype)),
            dim=-1,
        )
        token = self.value(pair)
        semantic = self.semantic(semantic_ids)
        semantic_valid = semantic_ids.ne(0)
        semantic = (
            semantic * semantic_valid[..., None].to(dtype=semantic.dtype)
        ).sum(dim=2) / semantic_valid.sum(dim=2).clamp_min(1)[..., None]
        token = token + semantic + self.group(group_ids) + self.embodiment(embodiment_ids)[:, None]
        return self.norm(token) * group_mask[..., None].to(dtype=token.dtype)


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

    def forward(
        self,
        value: torch.Tensor,
        action_times: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, groups, dim = value.shape
        if tuple(action_times.shape) != (batch, steps, groups):
            raise ValueError("action_times must align with [B,S,G] action tokens")
        if token_mask.shape != action_times.shape:
            raise ValueError("action token mask must align with action times")

        # Each group first follows its own physical timeline.
        temporal = value.transpose(1, 2).reshape(batch * groups, steps, dim)
        temporal_times = action_times.transpose(1, 2).reshape(batch * groups, steps)
        temporal_valid = token_mask.transpose(1, 2).reshape(batch * groups, steps)
        temporal_allowed = (
            temporal_times[:, :, None] + 1.0e-7
            >= temporal_times[:, None, :]
        ) & temporal_valid[:, None, :]
        temporal = temporal + self.attn(
            self.attn_norm(temporal),
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
        grouped = grouped + self.attn(
            self.attn_norm(grouped),
            allowed_mask=group_allowed[:, None],
        )
        grouped = grouped + self.ff(self.ff_norm(grouped))
        value = grouped.view(batch, steps, groups, dim)
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
                torch.ones(batch, horizon, 1, dtype=torch.bool, device=factual_mask.device),
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
        physical = raw * normalization_scale[:, :, None] + normalization_offset[:, :, None]
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
        self.grid = isqrt(cfg.P)
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.token_dim, cfg.rgb_hidden, 1),
            nn.GroupNorm(min(8, cfg.rgb_hidden), cfg.rgb_hidden),
            nn.GELU(),
        )
        stages = (cfg.rgb_size // self.grid).bit_length() - 1
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
                ResidualConvBlock(output_channels)
                for _ in range(cfg.rgb_res_blocks)
            )
            ups.append(nn.Sequential(*stage))
        self.ups = nn.ModuleList(ups)
        self.output = nn.Conv2d(channels[-1], 3, 1)

    def forward(
        self, tokens: torch.Tensor, view_embedding: torch.Tensor
    ) -> torch.Tensor:
        value = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.cfg.token_dim, self.grid, self.grid
        )
        value = self.stem(value) + view_embedding
        for upsample in self.ups:
            value = upsample(value)
        return torch.sigmoid(self.output(value))


class NativeRGBDecoder(nn.Module):
    """Restore the V7 native token-to-pixel path with bounded image chunks."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.view_embed = nn.Parameter(
            torch.empty(cfg.num_views, cfg.rgb_hidden, 1, 1)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        image_decoder: nn.Module = NativeRGBImageDecoder(cfg)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = tuple(
            self.cfg.rgb_decode_indices if frame_indices is None else frame_indices
        )
        if any(index < 0 or index >= future_tokens.shape[1] for index in indices):
            raise ValueError("RGB decode index is outside the future horizon")
        if tuple(future_tokens.shape[2:]) != (self.cfg.P, self.cfg.token_dim):
            raise ValueError("future RGB tokens must end in [P,token_dim]")
        index_tensor = torch.tensor(indices, dtype=torch.long, device=future_tokens.device)
        if not indices:
            empty = future_tokens.new_empty(
                future_tokens.shape[0],
                0,
                self.cfg.num_views,
                3,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            return empty, index_tensor
        selected = future_tokens.index_select(1, index_tensor)
        batch, frames, patches, token_dim = selected.shape
        views = self.cfg.num_views
        expanded = selected[:, :, None].expand(-1, -1, views, -1, -1)
        expanded = expanded.reshape(batch * frames * views, patches, token_dim)
        view_ids = torch.arange(views, device=future_tokens.device)
        view_ids = view_ids.view(1, 1, views).expand(batch, frames, -1).reshape(-1)
        if target_view_mask is None:
            valid = torch.ones(
                batch * frames * views, dtype=torch.bool, device=future_tokens.device
            )
        else:
            if tuple(target_view_mask.shape) != (batch, frames, views):
                raise ValueError("target_view_mask must be [B,F,V]")
            valid = target_view_mask.reshape(-1).bool()
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            empty = future_tokens.new_zeros(
                batch, frames, views, 3, self.cfg.rgb_size, self.cfg.rgb_size
            )
            return empty, index_tensor
        decoded_chunks: list[torch.Tensor] = []
        for start in range(
            0, int(valid_indices.numel()), self.cfg.rgb_decode_chunk_size
        ):
            chunk_indices = valid_indices[
                start : start + self.cfg.rgb_decode_chunk_size
            ]
            decoded_chunks.append(
                self.image_decoder(
                    expanded.index_select(0, chunk_indices),
                    self.view_embed.index_select(
                        0, view_ids.index_select(0, chunk_indices)
                    ),
                )
            )
        decoded = torch.cat(decoded_chunks, dim=0)
        dense = decoded.new_zeros(
            batch * frames * views, 3, self.cfg.rgb_size, self.cfg.rgb_size
        )
        dense = dense.index_copy(0, valid_indices, decoded)
        return (
            dense.view(
                batch, frames, views, 3, self.cfg.rgb_size, self.cfg.rgb_size
            ),
            index_tensor,
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
        self.future_queries = nn.Parameter(torch.empty(1, cfg.K, cfg.P, cfg.state_hidden))
        # Query identity comes from physical time, group/embodiment and current
        # state.  A shared seed avoids learning discrete 20Hz-style position
        # slots and makes the capacity ceiling parameter-count independent.
        self.policy_query_seed = nn.Parameter(
            torch.empty(1, 1, 1, cfg.action_hidden)
        )
        for parameter in (self.state_space, self.future_queries, self.policy_query_seed):
            nn.init.normal_(parameter, std=0.02)
        self.task_state = nn.Linear(cfg.task_dim, cfg.state_hidden, bias=False)
        self.task_action = nn.Linear(cfg.task_dim, cfg.action_hidden, bias=False)
        self.state_input_norm = RMSNorm(cfg.state_hidden)

        self.history_action = GroupedSignalEncoder(cfg.action_hidden, cfg)
        self.factual_action = GroupedSignalEncoder(cfg.state_hidden, cfg)
        self.current_state = CurrentStateEncoder(cfg)
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
        self.dynamics_blocks = self._checkpoint_module_list(
            (DynamicsConditionBlock(cfg) for _ in range(cfg.dynamics_layers)),
            enabled=cfg.activation_checkpointing,
        )
        self.state_norm = RMSNorm(cfg.state_hidden)
        self.action_norm = RMSNorm(cfg.action_hidden)
        self.token_output = nn.Linear(cfg.state_hidden, cfg.token_dim, bias=False)
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

    def _validate_world_times(self, world_times_s: torch.Tensor, batch: int) -> torch.Tensor:
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
        aux_values: Optional[torch.Tensor] = None,
        aux_mask: Optional[torch.Tensor] = None,
        aux_type_ids: Optional[torch.Tensor] = None,
        rgb_frame_indices: Optional[Sequence[int]] = None,
        rgb_view_mask: Optional[torch.Tensor] = None,
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
        state = state + self.state_space + self.state_time(relative_world_time)[:, :, None]
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
        query = self.policy_query_seed.expand(
            batch, query_count, cfg.max_action_groups, -1
        )
        query_time = policy_query_dt.transpose(1, 2)
        query = query + self.action_time(query_time)
        query = query + current[:, None] + self.task_action(task_embedding)[:, None, None]
        history = history + self.action_time(relative_world_time[:, : cfg.T])[:, :, None]
        history = history + self.task_action(task_embedding)[:, None, None]
        action = torch.cat((history, query), dim=1)
        history_mask = history_valid & action_group_mask[:, None, :]
        query_token_mask = policy_query_mask.transpose(1, 2) & action_group_mask[:, None]
        action_mask = torch.cat((history_mask, query_token_mask), dim=1)
        action_times = torch.cat(
            (
                relative_world_time[:, : cfg.T, None].expand(-1, -1, cfg.max_action_groups),
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
        spatial_update = self.policy_spatial_cross(
            self.policy_spatial_norm(query_flat),
            prior_state.reshape(batch, (cfg.T + cfg.K) * cfg.P, cfg.state_hidden),
        )
        policy_query = (query_flat + spatial_update).view(
            batch, query_count, cfg.max_action_groups, cfg.action_hidden
        )
        policy_query = self.action_norm(policy_query)
        policy_query = policy_query * query_token_mask[..., None].to(policy_query.dtype)

        factual, factual_mask = self.factual_action(
            fine_values=future_factual_fine_action_values,
            fine_dim_mask=future_factual_fine_action_mask,
            fine_dt=future_factual_fine_action_dt,
            fine_sample_mask=future_factual_fine_sample_mask,
            coarse_values=future_factual_coarse_action_values,
            coarse_dim_mask=future_factual_coarse_action_mask,
            action_semantic_ids=action_semantic_ids,
            group_ids=action_group_ids,
            group_mask=action_group_mask,
            embodiment_ids=embodiment_ids,
        )
        action_free_future = prior_state[:, cfg.T :]
        factual_future = action_free_future
        for dynamics_block in self.dynamics_blocks:
            factual_future = self._run(
                dynamics_block,
                factual_future,
                factual,
                factual_mask,
                enabled=cfg.activation_checkpointing,
            )
        factual_future = self.state_norm(factual_future)

        action_free_pred_tokens = self.token_output(action_free_future)
        pred_tokens = self.token_output(factual_future)
        output: dict[str, torch.Tensor] = {
            "action_free_native_state": action_free_future,
            "action_free_pred_tokens": action_free_pred_tokens,
            "native_state": factual_future,
            "pred_tokens": pred_tokens,
            "policy_latent": policy_query.transpose(1, 2),
            "world_times_s": world_times_s,
            "policy_query_dt": policy_query_dt,
        }
        output.update(
            self.action_head(
                policy_query,
                action_semantic_ids,
                policy_query_mask,
                action_normalization_offset,
                action_normalization_scale,
            )
        )
        output.update(self.geometry_head(factual_future))
        rgb, rgb_indices = self._run(
            self.rgb_head,
            pred_tokens,
            rgb_frame_indices,
            rgb_view_mask,
            enabled=cfg.activation_checkpointing,
        )
        output["rgb"] = rgb
        output["rgb_frame_indices"] = rgb_indices
        return output

    def iter_fsdp_units(self) -> Iterable[nn.Module]:
        """Yield communication-sized modules for bottom-up FSDP2 wrapping."""

        yield self.view_fuser
        yield from self.state_blocks
        yield from self.action_blocks
        yield from self.bridges
        yield from self.dynamics_blocks
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
        yield self.rgb_head.image_decoder

    def parameter_counts(self) -> dict[str, int]:
        groups: Mapping[str, nn.Module] = {
            "multiview_fuser": self.view_fuser,
            "state_trunk": self.state_blocks,
            "action_trunk": self.action_blocks,
            "state_action_bridges": self.bridges,
            "dynamics_refinement": self.dynamics_blocks,
            "rgb_head": self.rgb_head,
            "geometry_head": self.geometry_head,
            "action_head": self.action_head,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counts["other"] = sum(parameter.numel() for parameter in self.parameters()) - sum(
            counts.values()
        )
        counts["total"] = sum(parameter.numel() for parameter in self.parameters())
        return counts

    def num_trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


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
