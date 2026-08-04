"""WM3D core.

This module deliberately does not import any Wan or VLA implementation.  The
state trunk owns the future world and emits explicit token, RGB, depth, point,
camera, and confidence predictions.  The action trunk is a native, grouped
robot-action model trained from step zero.

The no-leak contract is structural:

* future factual actions are injected only into future state query tokens;
* temporal state attention is causal, so those queries cannot write backwards
  into context state tokens;
* the state-to-action bridge reads context state tokens only; and
* future action-trunk inputs are learned queries, never target actions.

That lets factual actions condition world dynamics without turning action
prediction into a teacher-forced identity map.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _round_multiple(value: float, multiple: int = 256) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


@dataclass(frozen=True)
class WM3DConfig:
    # External representation contract.
    T: int = 24
    P: int = 144
    K: int = 16
    token_dim: int = 2048
    task_dim: int = 2048
    num_views: int = 3

    # Approximately 5B trainable parameters with the default block counts.
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

    view_hidden: int = 1024
    view_heads: int = 8
    view_ff_mult: float = 2.5

    # Embodiment-aware, grouped, high-rate action interface.
    max_action_groups: int = 8
    max_action_dim: int = 16
    action_substeps: int = 6  # 30 Hz actions aligned to a 5 Hz visual frame.
    max_group_id: int = 64
    max_embodiments: int = 256

    # Optional low-frequency memory and auxiliary observations.
    memory_dim: int = 2048
    memory_every_state_layers: int = 4
    max_aux_tokens: int = 8
    aux_dim: int = 256
    max_aux_type_id: int = 64

    # Native explicit prediction heads.
    rgb_hidden: int = 512
    rgb_size: int = 384
    rgb_decode_indices: tuple[int, ...] = (3, 7, 11, 15)
    geom_hidden: int = 768

    dropout: float = 0.0
    activation_checkpointing: bool = True

    def validate(self) -> None:
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
            if hidden % heads:
                raise ValueError(
                    f"{name} hidden={hidden} is not divisible by heads={heads}"
                )
        if len(set(self.bridge_layers_state)) != len(self.bridge_layers_state):
            raise ValueError("bridge_layers_state contains duplicates")
        if any(i < 0 or i >= self.state_layers for i in self.bridge_layers_state):
            raise ValueError("bridge layer index is outside state trunk")
        if any(i < 0 or i >= self.K for i in self.rgb_decode_indices):
            raise ValueError("rgb_decode_indices must refer to future steps")
        if (
            self.max_aux_tokens <= 0
            or self.aux_dim <= 0
            or not 0 < self.max_aux_type_id < self.aux_dim
        ):
            raise ValueError("invalid auxiliary token contract")
        if self.rgb_size % grid:
            raise ValueError("rgb_size must be divisible by the spatial token grid")
        ratio = self.rgb_size // grid
        if ratio <= 0 or ratio & (ratio - 1):
            raise ValueError("rgb_size / sqrt(P) must be a power of two")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        x_norm = x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + self.eps)
        return x_norm.to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mult: float, dropout: float = 0.0):
        super().__init__()
        inner = _round_multiple(dim * mult)
        self.gate_up = nn.Linear(dim, inner * 2, bias=False)
        self.down = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * value))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_causal: bool = False,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and allowed_mask is None,
        )
        return self.out(y.transpose(1, 2).reshape(batch, length, self.dim))


class CrossAttention(nn.Module):
    def __init__(
        self, query_dim: int, context_dim: int, heads: int, dropout: float = 0.0
    ):
        super().__init__()
        if query_dim % heads:
            raise ValueError("cross-attention query dim must be divisible by heads")
        self.query_dim = query_dim
        self.heads = heads
        self.head_dim = query_dim // heads
        self.q = nn.Linear(query_dim, query_dim, bias=False)
        self.kv = nn.Linear(context_dim, query_dim * 2, bias=False)
        self.out = nn.Linear(query_dim, query_dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, q_len, _ = query.shape
        k_len = context.shape[1]
        q = self.q(query).view(batch, q_len, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(context).chunk(2, dim=-1)
        k = k.view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, k_len, self.heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(batch, q_len, self.query_dim))


class MultiViewTokenFuser(nn.Module):
    """Fuse head/left-hand/right-hand observations without changing P or D.

    Attention is restricted to the view axis for every time/patch coordinate.
    This preserves the explicit 12x12 native spatial lattice while allowing
    each fused token to use all available cameras.
    """

    def __init__(self, cfg: WM3DConfig):
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

    def forward(
        self,
        tokens: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, views, patches, _ = tokens.shape
        if views != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {views}")
        if tuple(view_mask.shape) != (batch, frames, views):
            raise ValueError("view_mask must be [B,T,V]")
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every context frame must contain at least one view")
        x = self.in_proj(tokens) + self.view_embed
        x = x.permute(0, 1, 3, 2, 4).reshape(batch * frames * patches, views, -1)
        valid = view_mask[:, :, None, :].expand(batch, frames, patches, views)
        valid = valid.reshape(batch * frames * patches, views)
        allowed = valid[:, None, None, :]
        x = x + self.attn(self.attn_norm(x), allowed_mask=allowed)
        x = x + self.ff(self.ff_norm(x))
        logits = self.gate(x).squeeze(-1).masked_fill(~valid, float("-inf"))
        fused = (x * logits.softmax(dim=-1)[..., None]).sum(dim=1)
        fused = fused.view(batch, frames, patches, -1)
        return self.out_proj(fused)


class FactorizedStateBlock(nn.Module):
    """Spatial attention per frame plus causal temporal attention per patch."""

    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        dim = cfg.state_hidden
        self.spatial_norm = RMSNorm(dim)
        self.spatial = SelfAttention(dim, cfg.state_heads, cfg.dropout)
        self.temporal_norm = RMSNorm(dim)
        self.temporal = SelfAttention(dim, cfg.state_heads, cfg.dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, cfg.state_ff_mult, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, frames, patches, dim = x.shape
        spatial = x.reshape(batch * frames, patches, dim)
        spatial = spatial + self.spatial(self.spatial_norm(spatial))
        x = spatial.view(batch, frames, patches, dim)
        temporal = x.transpose(1, 2).reshape(batch * patches, frames, dim)
        temporal = temporal + self.temporal(
            self.temporal_norm(temporal), is_causal=True
        )
        x = temporal.view(batch, patches, frames, dim).transpose(1, 2)
        return x + self.ff(self.ff_norm(x))


class ActionBlock(nn.Module):
    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        dim = cfg.action_hidden
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, cfg.action_heads, cfg.dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, cfg.action_ff_mult, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        allowed_mask: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, groups, dim = x.shape
        flat = x.reshape(batch, frames * groups, dim)
        flat = flat + self.attn(self.attn_norm(flat), allowed_mask=allowed_mask)
        flat = flat + self.ff(self.ff_norm(flat))
        x = flat.view(batch, frames, groups, dim)
        return x * group_mask[:, None, :, None].to(dtype=x.dtype)


class StateActionBridge(nn.Module):
    """Bidirectional latent bridge using frame/group summaries.

    State-to-action reads only the first ``T`` frame summaries.  This is the
    key guard preventing factual future action conditioning from reaching the
    native action head.
    """

    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        self.T = cfg.T
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
        group_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, frames, patches, _ = state.shape
        groups = action.shape[2]
        state_summary = state.mean(dim=2)
        action_flat = action.reshape(batch, frames * groups, -1)
        action_update = self.action_reads_state(
            self.action_norm(action_flat),
            self.state_norm(state_summary[:, : self.T]),
        )
        action = action + action_update.view(batch, frames, groups, -1)
        action = action * group_mask[:, None, :, None].to(dtype=action.dtype)

        state_update = self.state_reads_action(
            self.state_norm(state_summary),
            self.action_norm(action.reshape(batch, frames * groups, -1)),
        )
        state = state + state_update[:, :, None, :] / patches**0.5
        return state, action


class GroupedActionTokenizer(nn.Module):
    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        flat = cfg.action_substeps * cfg.max_action_dim
        self.cfg = cfg
        self.value_proj = nn.Linear(flat * 2, cfg.action_hidden, bias=False)
        self.factual_proj = nn.Linear(flat * 2, cfg.state_hidden, bias=False)
        self.group_embed = nn.Embedding(cfg.max_group_id, cfg.action_hidden)
        self.embodiment_embed = nn.Embedding(cfg.max_embodiments, cfg.action_hidden)
        self.future_queries = nn.Parameter(
            torch.empty(1, cfg.K, cfg.max_action_groups, cfg.action_hidden)
        )
        nn.init.normal_(self.future_queries, std=0.02)

    def _flat_pair(
        self,
        values: torch.Tensor,
        dim_mask: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != dim_mask.shape:
            raise ValueError(
                "action values and dimension mask must have identical shape"
            )
        expected = (
            self.cfg.max_action_groups,
            self.cfg.action_substeps,
            self.cfg.max_action_dim,
        )
        if tuple(values.shape[-3:]) != expected:
            raise ValueError(
                f"action suffix must be {expected}, got {tuple(values.shape[-3:])}"
            )
        masked = values * dim_mask.to(dtype=values.dtype)
        return torch.cat(
            (masked.flatten(-2), dim_mask.to(values.dtype).flatten(-2)), dim=-1
        )

    def policy_tokens(
        self,
        context_values: torch.Tensor,
        context_dim_mask: torch.Tensor,
        group_ids: torch.Tensor,
        embodiment_ids: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch = context_values.shape[0]
        if context_values.shape[1] != self.cfg.T:
            raise ValueError("policy action input must contain context actions only")
        context = self.value_proj(self._flat_pair(context_values, context_dim_mask))
        future = self.future_queries.expand(batch, -1, -1, -1)
        tokens = torch.cat((context, future), dim=1)
        group = self.group_embed(group_ids)[:, None]
        embodiment = self.embodiment_embed(embodiment_ids)[:, None, None]
        tokens = tokens + group + embodiment
        return tokens * group_mask[:, None, :, None].to(dtype=tokens.dtype)

    def factual_future_condition(
        self,
        future_values: torch.Tensor,
        future_dim_mask: torch.Tensor,
        group_mask: torch.Tensor,
    ) -> torch.Tensor:
        if future_values.shape[1] != self.cfg.K:
            raise ValueError("factual world action must have K future visual steps")
        group_features = self.factual_proj(
            self._flat_pair(future_values, future_dim_mask)
        )
        weights = group_mask[:, None, :, None].to(dtype=group_features.dtype)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (group_features * weights).sum(dim=2) / denom


class GroupedActionHead(nn.Module):
    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.action_substeps * cfg.max_action_dim
        self.norm = RMSNorm(cfg.action_hidden)
        self.mean = nn.Linear(cfg.action_hidden, width)
        self.log_scale = nn.Linear(cfg.action_hidden, width)
        self.contact = nn.Linear(cfg.action_hidden, cfg.action_substeps)

    def forward(self, future_action: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, horizon, groups, _ = future_action.shape
        x = self.norm(future_action)
        shape = (
            batch,
            horizon,
            groups,
            self.cfg.action_substeps,
            self.cfg.max_action_dim,
        )
        return {
            "action_mean": self.mean(x).view(shape),
            "action_log_scale": self.log_scale(x).clamp(-7.0, 3.0).view(shape),
            "contact_logit": self.contact(x),
        }


class NativeGeometryHead(nn.Module):
    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        self.num_views = cfg.num_views
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.geom_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        self.in_proj = nn.Linear(cfg.state_hidden, cfg.geom_hidden, bias=False)
        self.norm = RMSNorm(cfg.geom_hidden)
        self.depth = nn.Linear(cfg.geom_hidden, 1)
        self.point = nn.Linear(cfg.geom_hidden, 3)
        self.confidence = nn.Linear(cfg.geom_hidden, 1)
        self.camera = nn.Sequential(
            nn.Linear(cfg.state_hidden, cfg.geom_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.geom_hidden, cfg.num_views * 9),
        )

    def forward(self, future_state: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, horizon, patches, _ = future_state.shape
        x = self.in_proj(future_state)[:, :, None] + self.view_embed
        x = self.norm(x)
        camera = self.camera(future_state.mean(dim=2)).view(
            batch, horizon, self.num_views, 9
        )
        return {
            "depth": F.softplus(self.depth(x).squeeze(-1)),
            "point": self.point(x),
            "geometry_confidence": torch.sigmoid(self.confidence(x).squeeze(-1)),
            "camera_pose": camera,
        }


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class NativeRGBDecoder(nn.Module):
    """Multi-view native decoder with learned residual upsampling.

    Only selected future frames are decoded during pretraining to bound
    activation memory.  Evaluation may request all K frames explicitly.
    """

    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        self.cfg = cfg
        self.grid = isqrt(cfg.P)
        self.in_proj = nn.Linear(cfg.state_hidden, cfg.rgb_hidden, bias=False)
        self.view_embed = nn.Parameter(
            torch.empty(1, 1, cfg.num_views, 1, cfg.rgb_hidden)
        )
        nn.init.normal_(self.view_embed, std=0.02)
        stages = (cfg.rgb_size // self.grid).bit_length() - 1
        modules: list[nn.Module] = [ResidualConvBlock(cfg.rgb_hidden)]
        channels = cfg.rgb_hidden
        for _ in range(stages):
            next_channels = max(64, channels // 2)
            modules.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(channels, next_channels, 3, padding=1),
                    ResidualConvBlock(next_channels),
                ]
            )
            channels = next_channels
        self.decoder = nn.Sequential(*modules)
        self.out = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(
        self,
        future_state: torch.Tensor,
        frame_indices: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = tuple(
            self.cfg.rgb_decode_indices if frame_indices is None else frame_indices
        )
        if not indices:
            empty = future_state.new_empty(
                future_state.shape[0],
                0,
                self.cfg.num_views,
                3,
                self.cfg.rgb_size,
                self.cfg.rgb_size,
            )
            return empty, torch.empty(0, dtype=torch.long, device=future_state.device)
        if any(index < 0 or index >= future_state.shape[1] for index in indices):
            raise ValueError("RGB decode index is outside the future horizon")
        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=future_state.device
        )
        selected = future_state.index_select(1, index_tensor)
        x = self.in_proj(selected)[:, :, None] + self.view_embed
        batch, frames, views, patches, channels = x.shape
        x = x.permute(0, 1, 2, 4, 3).reshape(
            batch * frames * views, channels, self.grid, self.grid
        )
        rgb = torch.sigmoid(self.out(self.decoder(x)))
        rgb = rgb.view(batch, frames, views, 3, self.cfg.rgb_size, self.cfg.rgb_size)
        return rgb, index_tensor


class WM3D(nn.Module):
    def __init__(self, cfg: WM3DConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        frames = cfg.T + cfg.K
        self.view_fuser = MultiViewTokenFuser(cfg)
        self.fused_input_norm = RMSNorm(cfg.state_hidden)
        self.state_out = nn.Linear(cfg.state_hidden, cfg.token_dim, bias=False)
        self.task_state = nn.Linear(cfg.task_dim, cfg.state_hidden, bias=False)
        self.task_action = nn.Linear(cfg.task_dim, cfg.action_hidden, bias=False)
        self.state_time = nn.Parameter(torch.empty(1, frames, 1, cfg.state_hidden))
        self.state_space = nn.Parameter(torch.empty(1, 1, cfg.P, cfg.state_hidden))
        self.future_state_queries = nn.Parameter(
            torch.empty(1, cfg.K, cfg.P, cfg.state_hidden)
        )
        self.action_time = nn.Parameter(torch.empty(1, frames, 1, cfg.action_hidden))
        for parameter in (
            self.state_time,
            self.state_space,
            self.future_state_queries,
            self.action_time,
        ):
            nn.init.normal_(parameter, std=0.02)

        self.action_tokenizer = GroupedActionTokenizer(cfg)
        self.aux_proj = nn.Linear(cfg.aux_dim, cfg.state_hidden, bias=False)
        self.memory_proj = nn.Linear(cfg.memory_dim, cfg.state_hidden, bias=False)
        self.memory_norm = RMSNorm(cfg.state_hidden)
        self.memory_cross = CrossAttention(
            cfg.state_hidden, cfg.state_hidden, cfg.state_heads, cfg.dropout
        )

        self.state_blocks = nn.ModuleList(
            FactorizedStateBlock(cfg) for _ in range(cfg.state_layers)
        )
        self.action_blocks = nn.ModuleList(
            ActionBlock(cfg) for _ in range(cfg.action_layers)
        )
        self.bridges = nn.ModuleList(
            StateActionBridge(cfg) for _ in cfg.bridge_layers_state
        )
        self.state_norm = RMSNorm(cfg.state_hidden)
        self.action_norm = RMSNorm(cfg.action_hidden)
        self.action_head = GroupedActionHead(cfg)
        self.geometry_head = NativeGeometryHead(cfg)
        self.rgb_head = NativeRGBDecoder(cfg)

        self._action_steps = [
            (cfg.action_layers * (i + 1) // cfg.state_layers)
            - (cfg.action_layers * i // cfg.state_layers)
            for i in range(cfg.state_layers)
        ]
        self._bridge_by_state_layer = {
            state_layer: bridge_i
            for bridge_i, state_layer in enumerate(cfg.bridge_layers_state)
        }

    @staticmethod
    def _run(module: nn.Module, *args: torch.Tensor, enabled: bool):
        if enabled and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _action_allowed_mask(
        self,
        batch: int,
        group_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        del dtype
        frames = self.cfg.T + self.cfg.K
        groups = self.cfg.max_action_groups
        time_ids = torch.arange(frames, device=group_mask.device).repeat_interleave(
            groups
        )
        causal = time_ids[None, :] <= time_ids[:, None]
        valid = (
            group_mask[:, None, :]
            .expand(batch, frames, groups)
            .reshape(batch, frames * groups)
        )
        allowed = causal[None, None] & valid[:, None, None, :]
        return allowed

    def _add_memory(
        self,
        state: torch.Tensor,
        memory_tokens: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if memory_tokens is None:
            return state
        memory = self.memory_proj(memory_tokens)
        summary = state.mean(dim=2)
        allowed = None
        if memory_mask is not None:
            allowed = memory_mask[:, None, None, :].to(dtype=torch.bool)
        update = self.memory_cross(
            self.memory_norm(summary), self.memory_norm(memory), allowed_mask=allowed
        )
        return state + update[:, :, None, :] / state.shape[2] ** 0.5

    def forward(
        self,
        *,
        world_tokens: torch.Tensor,
        view_mask: torch.Tensor | None = None,
        task_embedding: torch.Tensor,
        context_action_values: torch.Tensor,
        context_action_dim_mask: torch.Tensor,
        future_factual_action_values: torch.Tensor,
        future_factual_action_dim_mask: torch.Tensor,
        action_group_ids: torch.Tensor,
        action_group_mask: torch.Tensor,
        embodiment_ids: torch.Tensor,
        aux_tokens: torch.Tensor | None = None,
        aux_mask: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        rgb_frame_indices: Sequence[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        expected_views = (cfg.T, cfg.num_views, cfg.P, cfg.token_dim)
        if tuple(world_tokens.shape[1:]) != expected_views:
            raise ValueError(
                "production world_tokens must be [B,T,V,P,D]="
                f"[B,{cfg.T},{cfg.num_views},{cfg.P},{cfg.token_dim}], got "
                f"{tuple(world_tokens.shape)}"
            )
        batch = world_tokens.shape[0]
        if view_mask is None:
            raise ValueError(
                "production multiview input requires an explicit view_mask"
            )
        context_state = self.view_fuser(world_tokens, view_mask)
        context_state = self.fused_input_norm(context_state)
        if tuple(action_group_ids.shape) != (batch, cfg.max_action_groups):
            raise ValueError("action_group_ids has the wrong shape")
        if tuple(action_group_mask.shape) != (batch, cfg.max_action_groups):
            raise ValueError("action_group_mask has the wrong shape")
        if not bool(action_group_mask.any(dim=-1).all()):
            raise ValueError("every sample must contain at least one action group")

        future_state = self.future_state_queries.expand(batch, -1, -1, -1)
        state = torch.cat((context_state, future_state), dim=1)
        state = state + self.state_time + self.state_space
        state = state + self.task_state(task_embedding)[:, None, None]
        factual = self.action_tokenizer.factual_future_condition(
            future_factual_action_values,
            future_factual_action_dim_mask,
            action_group_mask,
        )
        # Future factual actions never touch context frames.
        state = torch.cat(
            (state[:, : cfg.T], state[:, cfg.T :] + factual[:, :, None, :]), dim=1
        )
        if aux_tokens is not None:
            if tuple(aux_tokens.shape) != (
                batch,
                cfg.T,
                cfg.max_aux_tokens,
                cfg.aux_dim,
            ):
                raise ValueError("aux_tokens must be [B,T,max_aux_tokens,aux_dim]")
            if aux_mask is None or tuple(aux_mask.shape) != (
                batch,
                cfg.T,
                cfg.max_aux_tokens,
            ):
                raise ValueError("aux_mask must align to auxiliary tokens")
            aux = self.aux_proj(aux_tokens)
            weights = aux_mask[..., None].to(dtype=aux.dtype)
            aux = (aux * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
            state = torch.cat(
                (state[:, : cfg.T] + aux[:, :, None, :], state[:, cfg.T :]),
                dim=1,
            )

        action = self.action_tokenizer.policy_tokens(
            context_action_values,
            context_action_dim_mask,
            action_group_ids,
            embodiment_ids,
            action_group_mask,
        )
        action = (
            action + self.action_time + self.task_action(task_embedding)[:, None, None]
        )
        allowed = self._action_allowed_mask(batch, action_group_mask, action.dtype)

        action_i = 0
        for state_i, state_block in enumerate(self.state_blocks):
            state = self._run(state_block, state, enabled=cfg.activation_checkpointing)
            for _ in range(self._action_steps[state_i]):
                action = self._run(
                    self.action_blocks[action_i],
                    action,
                    allowed,
                    action_group_mask,
                    enabled=cfg.activation_checkpointing,
                )
                action_i += 1
            bridge_i = self._bridge_by_state_layer.get(state_i)
            if bridge_i is not None:
                state, action = self._run(
                    self.bridges[bridge_i],
                    state,
                    action,
                    action_group_mask,
                    enabled=cfg.activation_checkpointing,
                )
            if (
                memory_tokens is not None
                and cfg.memory_every_state_layers > 0
                and (state_i + 1) % cfg.memory_every_state_layers == 0
            ):
                state = self._add_memory(state, memory_tokens, memory_mask)
        while action_i < len(self.action_blocks):
            action = self._run(
                self.action_blocks[action_i],
                action,
                allowed,
                action_group_mask,
                enabled=cfg.activation_checkpointing,
            )
            action_i += 1

        state = self.state_norm(state)
        action = self.action_norm(action)
        future_state = state[:, cfg.T :]
        future_action = action[:, cfg.T :]
        output: dict[str, torch.Tensor] = {
            "pred_tokens": self.state_out(future_state),
            "native_state": future_state,
            "native_action_latent": future_action,
        }
        output.update(self.action_head(future_action))
        output.update(self.geometry_head(future_state))
        rgb, rgb_indices = self.rgb_head(future_state, rgb_frame_indices)
        output["rgb"] = rgb
        output["rgb_frame_indices"] = rgb_indices
        return output

    def iter_transformer_units(self) -> Iterable[nn.Module]:
        """Yield every communication-sized unit for bottom-up FSDP2 wrapping."""

        yield self.view_fuser
        yield from self.state_blocks
        yield from self.action_blocks
        yield from self.bridges

    def parameter_counts(self) -> dict[str, int]:
        groups = {
            "multiview_fuser": self.view_fuser,
            "state_trunk": self.state_blocks,
            "action_trunk": self.action_blocks,
            "bridges": self.bridges,
            "rgb_head": self.rgb_head,
            "geometry_head": self.geometry_head,
            "action_head": self.action_head,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counts["total"] = sum(parameter.numel() for parameter in self.parameters())
        return counts


def config_from_mapping(mapping: dict) -> WM3DConfig:
    """Construct a strict config while accepting YAML lists for tuple fields."""
    values = dict(mapping)
    for key in ("bridge_layers_state", "rgb_decode_indices"):
        if key in values:
            values[key] = tuple(int(item) for item in values[key])
    return WM3DConfig(**values)
