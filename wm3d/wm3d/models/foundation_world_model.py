"""WSA-Large foundation backend with WM3D robot and native-3D contracts."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
import importlib
import importlib.machinery
from math import isqrt
from pathlib import Path
import sys
import types
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


FOUNDATION_WORLD_MODEL_SCHEMA = "wm3d_foundation_world_model_v1"
FOUNDATION_OBJECTIVE_SCHEMA = "wm3d_foundation_joint_objective_v1"


class FoundationModelError(ValueError):
    pass


def validate_foundation_objective(
    mapping: Mapping[str, object], *, initialization: str = "wsa_checkpoint"
) -> None:
    expected = {
        "schema": FOUNDATION_OBJECTIVE_SCHEMA,
        "owner": "wm3d_native_joint_training_loss" if initialization == "wan_video" else "wsa_joint_training_loss",
    }
    if dict(mapping) != expected:
        raise FoundationModelError(
            "foundation objective owner disagrees with the initialization/training route"
        )


@dataclass(frozen=True)
class FoundationWorldModelConfig:
    schema: str = FOUNDATION_WORLD_MODEL_SCHEMA

    # Existing WM3D direct-raw window and grouped-robot ABI.
    T: int = 16
    P: int = 64
    K: int = 8
    token_dim: int = 2048
    task_dim: int = 2048
    num_views: int = 3
    max_action_groups: int = 3
    max_action_dim: int = 14
    max_state_dim: int = 10
    max_action_substeps: int = 128
    max_policy_queries: int = 256
    max_group_id: int = 128
    max_embodiments: int = 512
    max_action_semantic_id: int = 64
    max_state_semantic_id: int = 64
    max_aux_tokens: int = 1
    aux_dim: int = 1
    rgb_size: int = 256
    rgb_decode_indices: tuple[int, ...] = tuple(range(8))
    # Native-3D uses a hierarchical token budget. P64 is the inexpensive
    # temporal rollout grid, while P144 matches the pretrained WSA per-view
    # Future3D query grid. The official high-resolution P1296 decoder remains
    # owned by the WSA checkpoint.
    native_3d_context_tokens: int = 144
    native_3d_rollout_tokens: int = 64
    native_3d_target_tokens: int = 144

    # WSA/Wan canvas and context widths.
    video_height: int = 384
    video_width: int = 320
    wsa_text_dim: int = 4096
    context_fusion_heads: int = 16
    robot_conditioning_hidden: int = 256
    action_time_fourier_dim: int = 64
    wsa_action_horizon: int = 32

    # Frozen external sources and weights. All are local, explicit inputs.
    upstream_source_root: str = ""
    wsa_pretrained_path: str = ""
    wan_model_path: str = ""
    wan_tokenizer_model_path: str = ""
    wan_context_store_path: str = ""

    # Joint objectives.
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    native_3d_loss_weight: float = 0.1
    native_3d_rollout_loss_weight: float = 0.05
    activation_checkpointing: bool = True
    dtype: str = "bf16"
    # Explicit stage contract; loading an old profile preserves its behavior.
    # adapters freezes the released experts, not the gradient path through them.
    finetune_mode: str = "full"
    # Explicit pretraining boundary; old checkpoint profiles remain readable.
    initialization: str = "wsa_checkpoint"
    native_3d_mode: str = "legacy_readout"
    # Generated from local generic Wan weights by the upstream pretraining tool.
    wan_action_backbone_path: str = ""
    wan_3d_backbone_path: str = ""
    # Explicit fresh-start policy; old sealed profiles remain reproducible.
    pretrained_interface_init: str = "legacy_skip"
    # Preserve old runtimes; fp32 only promotes the small trainable interfaces.
    trainable_interface_dtype: str = "model"

    @property
    def action_width(self) -> int:
        return self.max_action_groups * self.max_action_dim

    @property
    def state_width(self) -> int:
        return self.max_action_groups * self.max_state_dim

    def validate(self, *, require_paths: bool = False) -> None:
        if self.schema != FOUNDATION_WORLD_MODEL_SCHEMA:
            raise FoundationModelError(
                f"foundation model schema must be {FOUNDATION_WORLD_MODEL_SCHEMA}"
            )
        for name in (
            "T",
            "P",
            "K",
            "token_dim",
            "task_dim",
            "num_views",
            "max_action_groups",
            "max_action_dim",
            "max_state_dim",
            "max_action_substeps",
            "max_policy_queries",
            "video_height",
            "video_width",
            "wsa_text_dim",
            "context_fusion_heads",
            "robot_conditioning_hidden",
            "action_time_fourier_dim",
            "wsa_action_horizon",
            "native_3d_context_tokens",
            "native_3d_rollout_tokens",
            "native_3d_target_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise FoundationModelError(f"{name} must be positive")
        if self.T != 16 or self.K != 8 or self.num_views != 3:
            raise FoundationModelError(
                "the first foundation profile requires T16/K8/three views"
            )
        if self.P != 64 or self.token_dim != 2048:
            raise FoundationModelError("foundation data contract must be P64x2048")
        for name in (
            "native_3d_context_tokens",
            "native_3d_rollout_tokens",
            "native_3d_target_tokens",
        ):
            value = int(getattr(self, name))
            grid = isqrt(value)
            if grid**2 != value:
                raise FoundationModelError(f"{name} must form a square")
        if self.native_3d_rollout_tokens != self.P:
            raise FoundationModelError(
                "native-3D rollout tokens must match the P64 data contract"
            )
        if self.native_3d_context_tokens != self.native_3d_target_tokens:
            raise FoundationModelError(
                "the first native-3D foundation profile uses one P144 VGGT grid "
                "for current-state input and final-state supervision"
            )
        if self.K % 4 or self.video_height % 16 or self.video_width % 16:
            raise FoundationModelError("Wan video shape is not VAE compatible")
        if self.video_height % 3 or self.video_width % 2:
            raise FoundationModelError("three-view canvas geometry is invalid")
        if self.wsa_text_dim % self.context_fusion_heads:
            raise FoundationModelError("WSA context width must divide fusion heads")
        if self.action_time_fourier_dim % 2:
            raise FoundationModelError("action time Fourier width must be even")
        if (
            self.wsa_action_horizon % self.K
            or self.wsa_action_horizon > self.K * self.max_action_substeps
        ):
            raise FoundationModelError(
                "WSA action horizon must be divisible by K and fit storage capacity"
            )
        if tuple(self.rgb_decode_indices) != tuple(range(self.K)):
            raise FoundationModelError("foundation video loss must supervise all K frames")
        if self.initialization not in {"wsa_checkpoint", "wan_video"}:
            raise FoundationModelError("unsupported foundation initialization")
        if self.native_3d_mode not in {"legacy_readout", "trajectory"}:
            raise FoundationModelError("unsupported native-3D mode")
        if self.initialization == "wan_video":
            if self.wsa_pretrained_path or self.pretrained_interface_init != "legacy_skip":
                raise FoundationModelError("Wan-only initialization forbids WSA checkpoint/prior transfer")
            if not self.wan_action_backbone_path or not self.wan_3d_backbone_path:
                raise FoundationModelError("native pretraining requires explicit Wan-derived expert backbones")
            if self.finetune_mode not in {"full", "native_bootstrap"}:
                raise FoundationModelError("native experts must be trained, not frozen adapters")
            if self.native_3d_mode != "trajectory" or self.native_3d_context_tokens != self.P:
                raise FoundationModelError("native pretraining requires K8 per-view P64 trajectory states")
            if self.native_3d_rollout_loss_weight != 0:
                raise FoundationModelError("trajectory states replace the old post-hoc rollout loss")
        elif self.native_3d_mode != "legacy_readout" or self.finetune_mode == "native_bootstrap":
            raise FoundationModelError("trajectory/bootstrap mode requires Wan-only initialization")
        if self.native_3d_mode == "legacy_readout" and self.native_3d_rollout_loss_weight <= 0:
            raise FoundationModelError("legacy rollout weight must be positive")
        for name in ("video_loss_weight", "action_loss_weight", "native_3d_loss_weight"):
            if float(getattr(self, name)) <= 0:
                raise FoundationModelError(f"{name} must be positive")
        if self.dtype != "bf16":
            raise FoundationModelError("the WSA foundation profile requires bf16")
        if self.finetune_mode not in {"full", "adapters", "native_bootstrap"}:
            raise FoundationModelError("unsupported foundation training stage")
        if self.trainable_interface_dtype not in {"model", "fp32"}:
            raise FoundationModelError("unsupported trainable interface precision")
        if self.trainable_interface_dtype == "fp32" and self.finetune_mode != "adapters":
            raise FoundationModelError("FP32 interfaces require adapters mode")
        if self.pretrained_interface_init not in {
            "legacy_skip", "pretrained_action_prefix",
        }:
            raise FoundationModelError("unsupported pretrained interface initialization")
        if require_paths:
            for name in (
                "upstream_source_root",
                "wan_model_path",
                "wan_tokenizer_model_path",
                "wan_context_store_path",
            ) + (("wsa_pretrained_path",) if self.initialization == "wsa_checkpoint"
                 else ("wan_action_backbone_path", "wan_3d_backbone_path")):
                if not str(getattr(self, name)).strip():
                    raise FoundationModelError(f"{name} must be explicit")
                value = Path(str(getattr(self, name))).expanduser()
                if not value.exists():
                    raise FoundationModelError(f"{name} does not exist: {value}")


def foundation_config_from_mapping(
    mapping: Mapping[str, object],
) -> FoundationWorldModelConfig:
    values = dict(mapping)
    valid = {item.name for item in fields(FoundationWorldModelConfig)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise FoundationModelError(f"unknown foundation model fields: {unknown}")
    for name in ("rgb_decode_indices",):
        if name in values:
            values[name] = tuple(int(item) for item in values[name])
    config = FoundationWorldModelConfig(**values)
    config.validate()
    return config


class ResidualContextFusion(nn.Module):
    """Zero-gated token cross-attention that preserves the WSA initialization."""

    def __init__(self, base_dim: int, extra_dim: int, heads: int) -> None:
        super().__init__()
        self.base_norm = nn.LayerNorm(base_dim)
        self.extra_norm = nn.LayerNorm(extra_dim)
        self.extra_projection = nn.Linear(extra_dim, base_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            base_dim, heads, dropout=0.0, batch_first=True, bias=False
        )
        self.gate = nn.Parameter(torch.zeros(base_dim))

    def forward(
        self,
        base: torch.Tensor,
        base_mask: torch.Tensor,
        extra: torch.Tensor,
        extra_mask: torch.Tensor,
    ) -> torch.Tensor:
        if base.ndim != 3 or extra.ndim != 3:
            raise FoundationModelError("context fusion tensors must be sequences")
        if base_mask.shape != base.shape[:2] or extra_mask.shape != extra.shape[:2]:
            raise FoundationModelError("context fusion masks disagree with tokens")
        if not bool(extra_mask.any(dim=1).all()):
            raise FoundationModelError("every sample needs real conditioning tokens")
        query = self.base_norm(base)
        memory = self.extra_projection(self.extra_norm(extra))
        update, _ = self.attention(
            query,
            memory,
            memory,
            key_padding_mask=~extra_mask,
            need_weights=False,
        )
        gated = torch.tanh(self.gate).to(dtype=update.dtype)[None, None]
        return base + update * gated * base_mask[..., None].to(update.dtype)


class GroupedRobotConditioner(nn.Module):
    """Encode physical semantics, calibration and observed action history."""

    def __init__(self, config: FoundationWorldModelConfig) -> None:
        super().__init__()
        hidden = config.robot_conditioning_hidden
        self.config = config
        self.group = nn.Embedding(config.max_group_id, hidden)
        self.action_semantic = nn.Embedding(config.max_action_semantic_id, hidden)
        self.state_semantic = nn.Embedding(config.max_state_semantic_id, hidden)
        self.embodiment = nn.Embedding(config.max_embodiments, hidden)
        self.action_numeric = nn.Linear(2, hidden, bias=False)
        self.state_numeric = nn.Linear(2, hidden, bias=False)
        self.dimension_output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, config.wsa_text_dim, bias=False),
        )
        self.history_value = nn.Linear(
            config.action_width, config.wsa_text_dim, bias=False
        )
        self.history_time = nn.Sequential(
            nn.Linear(config.action_time_fourier_dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, config.wsa_text_dim, bias=False),
        )
        frequencies = torch.logspace(-2, 2, config.action_time_fourier_dim // 2)
        self.register_buffer("time_frequencies", frequencies, persistent=False)

    def _time_features(self, time_s: torch.Tensor) -> torch.Tensor:
        phase = time_s[..., None].float() * self.time_frequencies.float()
        return torch.cat((phase.sin(), phase.cos()), dim=-1)

    @staticmethod
    def _valid_command_time(
        time_s: torch.Tensor, element_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = element_mask.sum(dim=-1)
        valid = count > 0
        masked_time = time_s.float().masked_fill(~element_mask, 0.0)
        mean = masked_time.sum(dim=-1) / count.clamp_min(1)
        spread = torch.where(
            element_mask,
            (time_s.float() - mean[..., None]).abs(),
            torch.zeros_like(time_s, dtype=torch.float32),
        ).amax(dim=-1)
        if bool((spread[valid] > 1.0e-5).any()):
            raise FoundationModelError(
                "one command has inconsistent per-dimension timestamps"
            )
        if not bool(torch.isfinite(mean[valid]).all()):
            raise FoundationModelError("robot command time must be finite")
        for row_mean, row_valid in zip(mean, valid, strict=True):
            ordered = row_mean[row_valid]
            if ordered.numel() > 1 and not bool(torch.diff(ordered).gt(0).all()):
                raise FoundationModelError(
                    "robot command timestamps must strictly increase"
                )
        return mean, valid

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        group_ids = batch["action_group_ids"].long()
        group_mask = batch["action_group_mask"].bool()
        action_semantic = batch["action_semantic_ids"].long()
        state_semantic = batch["state_semantic_ids"].long()
        embodiment_ids = batch["embodiment_ids"].long()
        batch_size, groups = group_ids.shape
        if groups != cfg.max_action_groups:
            raise FoundationModelError("grouped robot capacity changed")
        if action_semantic.shape != (batch_size, groups, cfg.max_action_dim):
            raise FoundationModelError("action semantic shape changed")
        if state_semantic.shape != (batch_size, groups, cfg.max_state_dim):
            raise FoundationModelError("state semantic shape changed")
        embodiment = self.embodiment(embodiment_ids)[:, None, None]
        group = self.group(group_ids)[:, :, None]

        action_dim_valid = (~batch["action_dim_is_pad"].bool()).reshape(
            batch_size, groups, cfg.max_action_dim
        ) & group_mask[:, :, None]
        action_scale = batch["action_normalization_scale"].float()
        action_offset = batch["action_normalization_offset"].float()
        if bool((action_scale[action_dim_valid] <= 0).any()):
            raise FoundationModelError("valid action normalization scale must be positive")
        action_numeric = torch.stack(
            (action_offset, action_scale.clamp_min(1.0e-12).log()), dim=-1
        )
        action_tokens = (
            self.action_semantic(action_semantic)
            + group
            + embodiment
            + self.action_numeric(action_numeric)
        )

        state_dim_valid = (~batch["proprio_dim_is_pad"].bool()).reshape(
            batch_size, groups, cfg.max_state_dim
        ) & group_mask[:, :, None]
        state_scale = batch["state_normalization_scale"].float()
        state_offset = batch["state_normalization_offset"].float()
        if bool((state_scale[state_dim_valid] <= 0).any()):
            raise FoundationModelError("valid state normalization scale must be positive")
        state_numeric = torch.stack(
            (state_offset, state_scale.clamp_min(1.0e-12).log()), dim=-1
        )
        state_tokens = (
            self.state_semantic(state_semantic)
            + group
            + embodiment
            + self.state_numeric(state_numeric)
        )

        action_tokens = self.dimension_output(action_tokens).reshape(
            batch_size, cfg.action_width, cfg.wsa_text_dim
        )
        state_tokens = self.dimension_output(state_tokens).reshape(
            batch_size, cfg.state_width, cfg.wsa_text_dim
        )
        history = batch["history_action"].float()
        history_mask = batch["history_action_element_mask"].bool()
        history_time = batch["history_action_time_s"].float()
        command_time, history_valid = self._valid_command_time(
            history_time, history_mask
        )
        if not bool((command_time[history_valid] <= 0).all()):
            raise FoundationModelError("action history extends beyond policy time zero")
        history_tokens = self.history_value(history)
        history_tokens = history_tokens + self.history_time(
            self._time_features(command_time)
        )
        embodiment_token = self.dimension_output(
            self.embodiment(embodiment_ids)[:, None]
        )
        tokens = torch.cat(
            (embodiment_token, action_tokens, state_tokens, history_tokens), dim=1
        )
        mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device),
                action_dim_valid.reshape(batch_size, cfg.action_width),
                state_dim_valid.reshape(batch_size, cfg.state_width),
                history_valid,
            ),
            dim=1,
        )
        return tokens, mask


class ActionTimeConditioner(nn.Module):
    """Add exact source-native physical command times to ActionDiT tokens."""

    def __init__(self, hidden_dim: int, fourier_dim: int) -> None:
        super().__init__()
        self.fourier_dim = int(fourier_dim)
        frequencies = torch.logspace(-2, 2, fourier_dim // 2)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        nn.init.zeros_(self.mlp[-1].weight)

    def forward(self, time_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        command_time, valid = GroupedRobotConditioner._valid_command_time(
            time_s, mask
        )
        if not bool(valid.any(dim=-1).all()):
            raise FoundationModelError("WSA action horizon needs a valid command per sample")
        if not bool((command_time[valid] >= 0).all()):
            raise FoundationModelError("future action command time must be non-negative")
        phase = command_time[..., None].float() * self.frequencies.float()
        features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.mlp(features).masked_fill(~valid[..., None], 0.0)


class BoundVGGTTeacher(nn.Module):
    """Expose precomputed frozen VGGT tokens through WSA's teacher interface."""

    def __init__(self, output_tokens_per_view: int, layers: int) -> None:
        super().__init__()
        self.output_tokens_per_view = int(output_tokens_per_view)
        self.layers = int(layers)
        self._target: torch.Tensor | None = None

    def bind(self, target: torch.Tensor) -> None:
        if self._target is not None:
            raise RuntimeError("VGGT teacher target is already bound")
        if target.ndim != 4:
            raise FoundationModelError("VGGT target must be [B,V,P,D]")
        self._target = target.detach()

    def clear(self) -> None:
        self._target = None

    def forward(self, _images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._target is None:
            raise RuntimeError("VGGT target was not bound for this WSA forward")
        target = self._target
        batch, views, patches, width = target.shape
        source_grid = isqrt(patches)
        target_grid = isqrt(self.output_tokens_per_view)
        if source_grid**2 != patches or target_grid**2 != self.output_tokens_per_view:
            raise FoundationModelError("native-3D token grids must be square")
        if source_grid != target_grid:
            target = F.interpolate(
                target.permute(0, 1, 3, 2).reshape(
                    batch * views, width, source_grid, source_grid
                ).float(),
                size=(target_grid, target_grid),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, views, width, -1).permute(0, 1, 3, 2)
        flattened = target.reshape(batch, views * self.output_tokens_per_view, width)
        return tuple(flattened for _ in range(self.layers))


def _resize_spatial_tokens(
    tokens: torch.Tensor, output_tokens: int
) -> torch.Tensor:
    """Resize a per-view square token grid without mixing cameras."""

    if tokens.ndim != 4:
        raise FoundationModelError("native-3D tokens must be [B,V,P,D]")
    batch, views, patches, width = tokens.shape
    source_grid = isqrt(patches)
    target_grid = isqrt(int(output_tokens))
    if source_grid**2 != patches or target_grid**2 != int(output_tokens):
        raise FoundationModelError("native-3D token grids must be square")
    if source_grid == target_grid:
        return tokens
    resized = F.interpolate(
        tokens.permute(0, 1, 3, 2)
        .reshape(batch * views, width, source_grid, source_grid)
        .float(),
        size=(target_grid, target_grid),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, views, width, int(output_tokens)).permute(
        0, 1, 3, 2
    ).to(dtype=tokens.dtype)


class Native3DStateConditioner(nn.Module):
    """Inject the observed P144 native state into WSA Future3D queries.

    The gate starts at exact zero, so loading this adapter leaves the released
    WSA checkpoint numerically unchanged.  Missing cameras are zeroed before
    the update, and views are never pooled together.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        num_views: int,
        tokens_per_view: int,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.tokens_per_view = int(tokens_per_view)
        self.norm = nn.LayerNorm(token_dim)
        self.projection = nn.Linear(token_dim, hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self, tokens: torch.Tensor, view_mask: torch.Tensor
    ) -> torch.Tensor:
        expected = (self.num_views, self.tokens_per_view)
        if tokens.ndim != 4 or tuple(tokens.shape[1:3]) != expected:
            raise FoundationModelError(
                "observed native-3D state does not match the WSA P144 view grid"
            )
        if tuple(view_mask.shape) != tuple(tokens.shape[:2]):
            raise FoundationModelError("observed native-3D view mask changed")
        if not bool(view_mask.any(dim=1).all()):
            raise FoundationModelError("every sample requires an observed 3D view")
        update = self.projection(self.norm(tokens))
        update = update * view_mask[..., None, None].to(dtype=update.dtype)
        update = update * torch.tanh(self.gate)[None, None, None]
        return update.reshape(tokens.shape[0], -1, update.shape[-1])


class Native3DRolloutHead(nn.Module):
    """Causally decode K future P64 states from WSA and physical actions.

    The MoT sequence remains P144/view.  A factorized temporal readout adds
    only O(K*P64) work and avoids expanding the 30-layer joint attention to
    K*P144 tokens.  Source-native fine commands are grouped by world interval
    without interpolation.  Each recurrent update depends on the preceding
    state, the shared WSA Future3D representation and that interval's clean
    action/time/mask tuple.  This is a world-only path and cannot affect the
    ActionDiT policy prediction.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        token_dim: int,
        num_views: int,
        context_tokens_per_view: int,
        rollout_tokens_per_view: int,
        horizons: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.num_views = int(num_views)
        self.context_tokens_per_view = int(context_tokens_per_view)
        self.rollout_tokens_per_view = int(rollout_tokens_per_view)
        self.horizons = int(horizons)
        bottleneck = max(64, min(256, hidden_dim // 4))
        self.current_norm = nn.LayerNorm(token_dim)
        self.current_projection = nn.Linear(token_dim, hidden_dim, bias=False)
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.state_projection = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.shared_projection = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.action_projection = nn.Linear(
            3 * int(action_dim), bottleneck, bias=False
        )
        self.horizon_embedding = nn.Parameter(
            torch.randn(self.horizons, bottleneck) * 0.02
        )
        self.state_update = nn.Linear(bottleneck, hidden_dim, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, token_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        shared_tokens: torch.Tensor,
        current: torch.Tensor,
        action: torch.Tensor,
        action_time_s: torch.Tensor,
        action_element_mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.num_views * self.context_tokens_per_view
        if shared_tokens.ndim != 3 or shared_tokens.shape[1] != expected:
            raise FoundationModelError(
                "shared Future3D tokens do not match the P144/view contract"
            )
        if (
            action.ndim != 3
            or action_time_s.shape != action.shape
            or action_element_mask.shape != action.shape
        ):
            raise FoundationModelError(
                "native-3D rollout action/time/mask shapes disagree"
            )
        if action.shape[1] % self.horizons:
            raise FoundationModelError(
                "source-native future commands must divide into K intervals"
            )
        if action_element_mask.dtype != torch.bool:
            raise FoundationModelError("native-3D rollout action mask must be boolean")
        batch, _sequence, hidden = shared_tokens.shape
        if current.ndim != 4 or tuple(current.shape[:3]) != (
            batch,
            self.num_views,
            self.context_tokens_per_view,
        ):
            raise FoundationModelError(
                "current native-3D state does not match the P144/view contract"
            )
        spatial = shared_tokens.reshape(
            batch,
            self.num_views,
            self.context_tokens_per_view,
            hidden,
        )
        spatial = _resize_spatial_tokens(spatial, self.rollout_tokens_per_view)
        current_p64 = _resize_spatial_tokens(
            current.to(device=spatial.device, dtype=spatial.dtype),
            self.rollout_tokens_per_view,
        )
        state = self.current_projection(self.current_norm(current_p64))
        shared_drive = self.shared_projection(self.shared_norm(spatial))

        mask = action_element_mask.to(device=action.device)
        command = torch.cat(
            (
                action.masked_fill(~mask, 0.0),
                action_time_s.masked_fill(~mask, 0.0),
                mask.to(dtype=action.dtype),
            ),
            dim=-1,
        )
        command = self.action_projection(command.to(dtype=spatial.dtype))
        substeps = action.shape[1] // self.horizons
        command = command.reshape(batch, self.horizons, substeps, -1)
        command_valid = mask.any(dim=-1).reshape(batch, self.horizons, substeps)
        # Padding is storage, not an extra zero-valued physical command.
        # A sparse interval must not have its action drive diluted by padding.
        command = command.masked_fill(~command_valid[..., None], 0.0).sum(dim=2)
        command = command / command_valid.sum(dim=2, keepdim=True).clamp_min(1)

        predictions: list[torch.Tensor] = []
        for index in range(self.horizons):
            drive = (
                self.state_projection(self.state_norm(state))
                + shared_drive
                + command[:, index, None, None]
                + self.horizon_embedding[index][None, None, None].to(
                    dtype=spatial.dtype
                )
            )
            state = state + self.state_update(F.silu(drive))
            residual = self.output(self.output_norm(state))
            predictions.append(current_p64 + residual)
        return torch.stack(predictions, dim=1)

    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        horizon: torch.Tensor,
        view_mask: torch.Tensor,
        current: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (
            prediction.shape[0],
            self.horizons,
            self.num_views,
            self.rollout_tokens_per_view,
        )
        if prediction.ndim != 5 or tuple(prediction.shape[:4]) != expected:
            raise FoundationModelError("native-3D rollout prediction shape changed")
        if tuple(target.shape) != tuple(prediction.shape[:1] + prediction.shape[2:]):
            raise FoundationModelError("native-3D rollout target shape changed")
        if horizon.shape != prediction.shape[:1] or horizon.dtype != torch.long:
            raise FoundationModelError("native-3D rollout horizon must be int64 [B]")
        if bool(((horizon < 0) | (horizon >= self.horizons)).any()):
            raise FoundationModelError("native-3D rollout horizon is out of range")
        if tuple(view_mask.shape) != (prediction.shape[0], self.num_views):
            raise FoundationModelError("native-3D rollout view mask changed")
        rows = torch.arange(prediction.shape[0], device=prediction.device)
        selected = prediction[rows, horizon]
        valid = view_mask.to(device=selected.device, dtype=torch.bool)
        if not bool(valid.any(dim=1).all()):
            raise FoundationModelError("every rollout target requires a real view")
        current_p64 = _resize_spatial_tokens(
            current.to(device=selected.device, dtype=selected.dtype),
            self.rollout_tokens_per_view,
        )
        target_valid = target.to(device=selected.device, dtype=selected.dtype)[valid]
        pred_delta = (selected - current_p64)[valid].float()
        target_delta = (target_valid - current_p64[valid]).float()
        direction_per_view = F.cosine_similarity(
            pred_delta.flatten(1), target_delta.flatten(1), dim=-1
        )
        target_scale = target_delta.detach().square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt().clamp_min(1.0e-3)
        smooth_l1 = F.smooth_l1_loss(
            pred_delta / target_scale,
            target_delta / target_scale,
        )
        # A single normalized robust regression already constrains both
        # direction and amplitude.  Do not optimize cosine at exact zero
        # prediction: its denominator creates a very large first-step
        # gradient.  Direction remains an explicit detached acceptance metric.
        loss = smooth_l1
        direction = direction_per_view.mean()
        amplitude_ratio = (
            pred_delta.square().mean().sqrt()
            / target_delta.square().mean().sqrt().clamp_min(1.0e-8)
        )
        return loss, {
            "loss_native_3d_rollout_delta_smooth_l1": smooth_l1.detach(),
            "native_3d_rollout_direction": direction.detach(),
            "native_3d_rollout_amplitude_ratio": amplitude_ratio.detach(),
        }


class WSABackendAdapter(nn.Module):
    """Add WM3D time and native-3D contracts without forking WSA's MoT core."""

    def __init__(
        self,
        wsa_model: nn.Module,
        *,
        config: FoundationWorldModelConfig,
    ) -> None:
        super().__init__()
        self.wsa_model = wsa_model
        action_expert = self.wsa_model.action_expert
        self.action_time = ActionTimeConditioner(
            int(action_expert.hidden_dim), config.action_time_fourier_dim
        )
        future_3d = self.wsa_model.future_3d_expert
        query_layers = tuple(future_3d.query_layer_indices)
        if int(future_3d.da3_num_views) != config.num_views:
            raise FoundationModelError("WSA Future3D view count changed")
        if int(future_3d.query_tokens_per_view) != config.native_3d_context_tokens:
            raise FoundationModelError(
                "P144 current-state grid must match WSA Future3D queries/view"
            )
        self.vggt_teacher = BoundVGGTTeacher(
            int(future_3d.da3_tokens_per_view),
            len(query_layers),
        )
        self.wsa_model.da3_teacher = self.vggt_teacher
        self.wsa_model.da3_teacher_layers = tuple(range(len(query_layers)))
        self.wsa_model.da3_layer_weights = tuple(1.0 for _ in query_layers)
        self.wsa_model.loss_lambda_3d = float(config.native_3d_loss_weight)
        self.native_3d_state = Native3DStateConditioner(
            token_dim=config.token_dim,
            hidden_dim=int(future_3d.hidden_dim),
            num_views=config.num_views,
            tokens_per_view=config.native_3d_context_tokens,
        )
        self.native_3d_rollout = Native3DRolloutHead(
            hidden_dim=int(future_3d.hidden_dim),
            token_dim=config.token_dim,
            num_views=config.num_views,
            context_tokens_per_view=config.native_3d_context_tokens,
            rollout_tokens_per_view=config.native_3d_rollout_tokens,
            horizons=config.K,
            action_dim=config.action_width,
        )
        self.native_3d_rollout_loss_weight = float(
            config.native_3d_rollout_loss_weight
        )
        self._active_time_update: torch.Tensor | None = None
        self._active_native_3d_update: torch.Tensor | None = None
        self._capture_rollout = False
        self._captured_rollout_tokens: torch.Tensor | None = None
        self._time_hook = action_expert.action_encoder.register_forward_hook(
            self._inject_action_time
        )
        self._base_future_3d_pre_dit = future_3d.pre_dit

        def conditioned_pre_dit(
            _future_3d: nn.Module, *args: Any, **kwargs: Any
        ) -> Mapping[str, Any]:
            pre = self._base_future_3d_pre_dit(*args, **kwargs)
            if self._active_native_3d_update is None:
                raise RuntimeError("observed native-3D state was not bound")
            update = self._active_native_3d_update
            if pre["tokens"].shape != update.shape:
                raise FoundationModelError(
                    "observed native-3D update shape changed inside WSA"
                )
            pre = dict(pre)
            pre["tokens"] = pre["tokens"] + update.to(
                device=pre["tokens"].device, dtype=pre["tokens"].dtype
            )
            return pre

        future_3d.pre_dit = types.MethodType(conditioned_pre_dit, future_3d)

        def capture_rollout(
            _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            if self._capture_rollout:
                self._captured_rollout_tokens = output

        self._rollout_hook = future_3d.layer_norms[-1].register_forward_hook(
            capture_rollout
        )

    def _inject_action_time(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if self._active_time_update is None:
            raise RuntimeError("ActionDiT time update was not bound")
        if output.shape != self._active_time_update.shape:
            raise FoundationModelError("ActionDiT time update shape changed")
        return output + self._active_time_update.to(dtype=output.dtype)

    @contextmanager
    def _masked_action_sampling(self, element_mask: torch.Tensor, *, joint: bool):
        """Keep known-zero padding on the same flow path used in training.

        WM3D masks padded coordinates out of WSA's action loss. Their unconstrained
        head outputs must not feed back as inputs on later sampling steps.
        Training uses x_sigma = sigma * epsilon there because x_clean is zero.
        Reuse the original sampling noise (no new RNG draw) and its known
        velocity epsilon. Valid coordinates and the official sampler are
        unchanged; this wrapper is never used by training.
        """
        if bool(element_mask.all()):
            yield
            return
        method_name = "_predict_joint_noise" if joint else "_predict_action_noise"
        original = getattr(self.wsa_model, method_name)
        had_override = method_name in vars(self.wsa_model)
        initial_noise = None
        num_timesteps = float(
            self.wsa_model.train_action_scheduler.num_train_timesteps
        )

        def predict(**kwargs: Any) -> Any:
            nonlocal initial_noise
            latent = kwargs["latents_action"]
            timestep = kwargs["timestep_action"]
            if latent.shape != element_mask.shape:
                raise FoundationModelError("sampling action padding shape changed")
            mask = element_mask.to(device=latent.device, dtype=torch.bool)
            if initial_noise is None:
                if not bool((timestep == num_timesteps).all()):
                    raise FoundationModelError(
                        "masked action sampling must start from full noise"
                    )
                initial_noise = latent.detach().clone()
            sigma = (timestep / num_timesteps).to(
                device=latent.device, dtype=latent.dtype
            ).reshape(-1, 1, 1)
            kwargs["latents_action"] = torch.where(
                mask, latent, sigma * initial_noise
            )
            result = original(**kwargs)
            if joint:
                video_velocity, action_velocity = result
                return video_velocity, torch.where(
                    mask, action_velocity, initial_noise
                )
            return torch.where(mask, result, initial_noise)

        setattr(self.wsa_model, method_name, predict)
        try:
            yield
        finally:
            if had_override:
                setattr(self.wsa_model, method_name, original)
            else:
                delattr(self.wsa_model, method_name)

    def forward(
        self,
        sample: Mapping[str, torch.Tensor],
        vggt_target: torch.Tensor,
        current_native_3d: torch.Tensor,
        current_native_3d_view_mask: torch.Tensor,
        rollout_target: torch.Tensor,
        rollout_horizon: torch.Tensor,
        rollout_view_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            self._active_time_update is not None
            or self._active_native_3d_update is not None
        ):
            raise RuntimeError("concurrent WSA backend forwards are forbidden")
        self._active_time_update = self.action_time(
            sample["action_time_s"], sample["action_element_mask"]
        )
        self._active_native_3d_update = self.native_3d_state(
            current_native_3d, current_native_3d_view_mask
        )
        self._capture_rollout = True
        self._captured_rollout_tokens = None
        self.vggt_teacher.bind(vggt_target)
        try:
            total, metrics = self.wsa_model.training_loss(
                sample,
                collect_loss_dict=True,
                collect_detailed_loss_dict=True,
                loss_dict_as_tensors=True,
            )
            captured = self._captured_rollout_tokens
            if captured is None:
                raise RuntimeError("WSA Future3D rollout state was not captured")
            rollout_prediction = self.native_3d_rollout(
                captured,
                current_native_3d,
                sample["action"],
                sample["action_time_s"],
                sample["action_element_mask"],
            )
            rollout_loss, rollout_metrics = self.native_3d_rollout.loss(
                rollout_prediction,
                rollout_target,
                rollout_horizon,
                rollout_view_mask,
                current_native_3d,
            )
            weighted_rollout = self.native_3d_rollout_loss_weight * rollout_loss
            total = total + weighted_rollout
        finally:
            self.vggt_teacher.clear()
            self._active_time_update = None
            self._active_native_3d_update = None
            self._capture_rollout = False
            self._captured_rollout_tokens = None
        result = {"total": total}
        result.update(
            {
                str(name): (
                    value
                    if isinstance(value, torch.Tensor)
                    else total.detach().new_tensor(float(value))
                )
                for name, value in metrics.items()
            }
        )
        result.update(rollout_metrics)
        result["loss_native_3d_rollout"] = rollout_loss.detach()
        result["loss_native_3d_rollout_w"] = weighted_rollout.detach()
        result["native_3d_context_gate_rms"] = self.native_3d_state.gate.float().square().mean().sqrt().detach()
        return result

    @torch.inference_mode()
    def infer_action(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        current_native_3d: torch.Tensor,
        current_native_3d_view_mask: torch.Tensor,
        action_time_s: torch.Tensor,
        action_element_mask: torch.Tensor,
        num_inference_steps: int,
        seed: int | None,
    ) -> torch.Tensor:
        """Run ActionDiT with the same source-native time update used in training."""

        if (
            self._active_time_update is not None
            or self._active_native_3d_update is not None
        ):
            raise RuntimeError("concurrent WSA backend forwards are forbidden")
        if input_image.ndim != 4 or input_image.shape[0] != 1:
            raise FoundationModelError(
                "foundation ActionDiT inference currently requires batch size one"
            )
        if action_time_s.shape != action_element_mask.shape:
            raise FoundationModelError("serving action time/mask shapes disagree")
        if action_time_s.ndim != 3 or action_time_s.shape[0] != 1:
            raise FoundationModelError("serving action time must be [1,H,D]")
        if proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0]
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise FoundationModelError("serving proprio must resolve to [1,D]")
        self._active_time_update = self.action_time(
            action_time_s, action_element_mask.bool()
        )
        self._active_native_3d_update = self.native_3d_state(
            current_native_3d, current_native_3d_view_mask
        )
        try:
            with self._masked_action_sampling(action_element_mask, joint=False):
                result = self.wsa_model.infer_action(
                    prompt=None,
                    input_image=input_image,
                    action_horizon=int(action_time_s.shape[1]),
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=int(num_inference_steps),
                    seed=seed,
                )
        finally:
            self._active_time_update = None
            self._active_native_3d_update = None
        action = result.get("action")
        expected = (int(action_time_s.shape[1]), int(action_time_s.shape[2]))
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected:
            raise FoundationModelError(
                f"WSA ActionDiT returned an invalid action shape: {getattr(action, 'shape', None)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError("foundation action inference is non-finite")
        return action.unsqueeze(0).masked_fill(
            ~action_element_mask.to(device=action.device, dtype=torch.bool), 0.0
        )

    @torch.inference_mode()
    def infer_world(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        current_native_3d: torch.Tensor,
        current_native_3d_view_mask: torch.Tensor,
        factual_action: torch.Tensor | None,
        action_time_s: torch.Tensor,
        action_element_mask: torch.Tensor,
        num_video_frames: int,
        num_inference_steps: int,
        seed: int | None,
    ) -> Mapping[str, Any]:
        """Use official joint sampling, optionally condition on a known candidate.

        With factual_action=None, both video and action are generated from noise.
        No candidate or future target is supplied. With a known candidate, this
        is world inference only, and its action output must not be returned.
        Released WSA-Large ignores video_expert's optional clean action input.
        Keep the actual ActionDiT input clean at action timestep zero throughout
        video denoising. Known-action forward dynamics is not joint action/video
        sampling: noising the known candidate weakens its early conditioning.
        Reuse the official video sampler; infer_action stays untouched.
        """

        if (
            self._active_time_update is not None
            or self._active_native_3d_update is not None
        ):
            raise RuntimeError("concurrent WSA backend forwards are forbidden")
        if input_image.ndim != 4 or input_image.shape[0] != 1:
            raise FoundationModelError(
                "foundation joint inference currently requires batch size one"
            )
        if action_time_s.shape != action_element_mask.shape or (
            factual_action is not None and factual_action.shape != action_time_s.shape
        ):
            raise FoundationModelError(
                "joint action, time, and element-mask shapes disagree"
            )
        if action_time_s.ndim != 3 or action_time_s.shape[0] != 1:
            raise FoundationModelError("joint action tensors must be [1,H,D]")
        if proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0]
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise FoundationModelError("joint proprio must resolve to [1,D]")
        mask = action_element_mask.bool()
        if factual_action is not None:
            factual_action = factual_action.masked_fill(~mask, 0.0)
            if not bool(torch.isfinite(factual_action).all()):
                raise FloatingPointError("world candidate action is non-finite")
        self._active_time_update = self.action_time(action_time_s, mask)
        self._active_native_3d_update = self.native_3d_state(
            current_native_3d, current_native_3d_view_mask
        )
        original_predict = self.wsa_model._predict_joint_noise
        had_override = "_predict_joint_noise" in vars(self.wsa_model)
        conditioned_steps = 0

        def predict_candidate(**kwargs: Any) -> Any:
            nonlocal conditioned_steps
            latents = kwargs["latents_action"]
            # WSA trains video/action noise levels independently. Conditional
            # forward dynamics fixes the known modality at noise level zero;
            # only the unknown future video is denoised. Padded coordinates
            # are also clean zeros, matching the masked training action at t=0.
            kwargs["latents_action"] = factual_action.to(
                device=latents.device, dtype=latents.dtype
            )
            kwargs["timestep_action"] = torch.zeros_like(kwargs["timestep_action"])
            kwargs["gt_action"] = None
            conditioned_steps += 1
            return original_predict(**kwargs)

        if factual_action is not None:
            self.wsa_model._predict_joint_noise = predict_candidate
        try:
            sampling = (
                self._masked_action_sampling(mask, joint=True)
                if factual_action is None else nullcontext()
            )
            with sampling:
                result = self.wsa_model.infer_joint(
                    prompt=None,
                    input_image=input_image,
                    num_video_frames=int(num_video_frames),
                    action_horizon=int(action_time_s.shape[1]),
                    action=None,
                    proprio=proprio,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=int(num_inference_steps),
                    seed=seed,
                    test_action_with_infer_action=False,
                )
        finally:
            if factual_action is not None:
                if had_override:
                    self.wsa_model._predict_joint_noise = original_predict
                else:
                    del self.wsa_model._predict_joint_noise
            self._active_time_update = None
            self._active_native_3d_update = None
        if not isinstance(result, Mapping) or "video" not in result:
            raise FoundationModelError("WSA joint inference returned no video")
        if factual_action is None:
            action = result.get("action")
            if not isinstance(action, torch.Tensor) or tuple(action.shape) != tuple(
                action_time_s.shape[1:]
            ):
                raise FoundationModelError("WSA joint sampling returned invalid action shape")
            if not bool(torch.isfinite(action).all()):
                raise FloatingPointError("WSA joint sampling returned non-finite action")
            return {
                "video": result["video"],
                "action_normalized": action.unsqueeze(0).masked_fill(
                    ~mask.to(device=action.device), 0.0
                ),
                "conditioning": {"mode": "joint_from_observation_no_candidate"},
            }
        if conditioned_steps != int(num_inference_steps):
            raise FoundationModelError("WSA candidate conditioning missed a sampling step")
        # The clamped branch is not a policy prediction. Do not expose its
        # action output; that would confuse conditioning with action ability.
        return {
            "video": result["video"],
            "conditioning": {
                "mode": "action_dit_clean_candidate_t0",
                "conditioned_steps": conditioned_steps,
            },
        }


class FoundationWorldModel(nn.Module):
    """Joint WSA-Large video, action and native-3D model."""

    def __init__(
        self,
        config: FoundationWorldModelConfig,
        *,
        backend: nn.Module,
    ) -> None:
        super().__init__()
        config.validate()
        self.cfg = config
        self.backend = backend
        self.robot_conditioner = GroupedRobotConditioner(config)
        self.robot_fusion = ResidualContextFusion(
            config.wsa_text_dim,
            config.wsa_text_dim,
            config.context_fusion_heads,
        )
        if config.finetune_mode == "native_bootstrap":
            self.backend.wsa_model.video_expert.requires_grad_(False)
        if config.finetune_mode == "adapters":
            # No no_grad/detach: new input adapters still learn through the
            # frozen experts. Keep all pretrained values exactly as loaded.
            self.backend.wsa_model.requires_grad_(False)
            if config.trainable_interface_dtype == "fp32":
                from .foundation_precision import promote_trainable_interfaces
                promote_trainable_interfaces(self.backend.wsa_model)
            for modules in self._pretrained_interface_modules().values():
                for module in modules:
                    module.requires_grad_(True)

    def _pretrained_interface_modules(self) -> Mapping[str, tuple[nn.Module, ...]]:
        wsa = self.backend.wsa_model
        return {
            "foundation_action_interface": (
                wsa.action_expert.action_encoder,
                wsa.action_expert.head,
            ),
            # The released readout predicted DA3; our teacher supplies VGGT.
            # Adapt only its final mapping, not its attention/FFN backbone.
            "foundation_native_3d_interface": (
                wsa.future_3d_expert.output_decoder.output_norm,
                wsa.future_3d_expert.output_decoder.output_proj,
            ),
            "foundation_proprio": (wsa.proprio_encoder,),
        }

    def train(self, mode: bool = True) -> "FoundationWorldModel":
        """Preserve WSA's official DiT-only fine-tuning boundary."""

        super().train(mode)
        if mode:
            wsa_model = self.backend.wsa_model
            wsa_model.eval()
            wsa_model.dit.train()
            proprio = getattr(wsa_model, "proprio_encoder", None)
            if proprio is not None:
                proprio.train()
        return self

    def iter_fsdp_units(self):
        # Expert block.forward is not the MoT execution boundary. Root-only
        # sharding keeps parameters live through pre_dit/attention/post_dit.
        return ()

    @property
    def fsdp_root_only(self):
        return self.cfg.initialization == "wan_video"

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context, context_mask = self._fused_context(batch)
        sample = {
            "video": batch["video"],
            "image_is_pad": batch["image_is_pad"],
            "context": context,
            "context_mask": context_mask,
            "action": batch["action"],
            "action_is_pad": batch["action_is_pad"],
            "action_dim_is_pad": batch["action_dim_is_pad"],
            "action_time_s": batch["action_time_s"],
            "action_element_mask": batch["action_element_mask"],
            "proprio": batch["proprio"],
            "future_3d_images": batch["future_3d_images"],
            "future_3d_img_masks": batch["future_3d_img_masks"],
        }
        if self.cfg.native_3d_mode == "trajectory":
            for name in ("native_3d_trajectory_target", "native_3d_trajectory_mask"):
                sample[name] = batch[name]
        result = self.backend(
            sample,
            batch["vggt_future_tokens"],
            batch["vggt_current_tokens"],
            batch["vggt_current_view_mask"],
            batch["vggt_rollout_target_tokens"],
            batch["vggt_rollout_target_horizon"],
            batch["vggt_rollout_target_view_mask"],
        )
        if "total" not in result or result["total"].ndim != 0:
            raise FoundationModelError("foundation backend must return scalar total")
        if not bool(torch.isfinite(result["total"])):
            raise FloatingPointError("foundation loss is non-finite")
        return result

    def _fused_context(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = batch["wan_context"]
        context_mask = batch["wan_context_mask"].bool()
        if context.ndim != 3 or context.shape[-1] != self.cfg.wsa_text_dim:
            raise FoundationModelError("Wan context ABI changed")
        # Training already runs inside BF16 autocast, while serving calls this
        # method directly. Keep the shared adapter path dtype-identical in both
        # cases instead of relying on an outer training-only context.
        with torch.autocast(
            device_type=context.device.type,
            dtype=torch.bfloat16,
            enabled=context.device.type == "cuda",
        ):
            robot_tokens, robot_mask = self.robot_conditioner(batch)
            fused = self.robot_fusion(
                context, context_mask, robot_tokens, robot_mask
            )
        return fused, context_mask

    @torch.inference_mode()
    def predict_action(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        num_inference_steps: int = 20,
        seed: int | None = None,
    ) -> Mapping[str, torch.Tensor]:
        """Predict one source-native action chunk and return normalized/physical ABI."""

        video = batch["video"]
        action_time_s = batch["action_time_s"]
        element_mask = batch["action_element_mask"].bool()
        if video.ndim != 5 or video.shape[0] != 1:
            raise FoundationModelError(
                "foundation action serving currently requires one video sample"
            )
        if action_time_s.shape != element_mask.shape:
            raise FoundationModelError("serving action time/mask shapes disagree")
        if action_time_s.shape[-1] != self.cfg.action_width:
            raise FoundationModelError("serving action width changed")
        if num_inference_steps <= 0:
            raise FoundationModelError("num_inference_steps must be positive")
        proprio = batch["proprio"]
        if proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0]
        if proprio.ndim != 2 or proprio.shape != (1, self.cfg.state_width):
            raise FoundationModelError("serving proprio must resolve to [1,D]")
        context, context_mask = self._fused_context(batch)
        normalized = self.backend.infer_action(
            input_image=video[:, :, 0],
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            current_native_3d=batch["vggt_current_tokens"],
            current_native_3d_view_mask=batch["vggt_current_view_mask"],
            action_time_s=action_time_s,
            action_element_mask=element_mask,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return self._action_outputs(normalized, batch)

    def _action_outputs(
        self, normalized: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        element_mask = batch["action_element_mask"].bool()
        action_time_s = batch["action_time_s"]
        offset = batch["action_normalization_offset"].reshape(
            1, 1, self.cfg.action_width
        )
        scale = batch["action_normalization_scale"].reshape(
            1, 1, self.cfg.action_width
        )
        offset = offset.to(device=normalized.device, dtype=normalized.dtype)
        scale = scale.to(device=normalized.device, dtype=normalized.dtype)
        mask = element_mask.to(device=normalized.device)
        normalized = normalized.masked_fill(~mask, 0.0)
        physical = torch.where(mask, normalized * scale + offset, 0.0)
        return {
            "action_normalized": normalized,
            "action_physical": physical.reshape(
                1,
                int(normalized.shape[1]),
                self.cfg.max_action_groups,
                self.cfg.max_action_dim,
            ),
            "action_element_mask": mask.reshape(
                1,
                int(normalized.shape[1]),
                self.cfg.max_action_groups,
                self.cfg.max_action_dim,
            ),
            "action_time_s": action_time_s.to(device=normalized.device).reshape(
                1,
                int(normalized.shape[1]),
                self.cfg.max_action_groups,
                self.cfg.max_action_dim,
            ),
        }

    @torch.inference_mode()
    def predict_joint(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        num_inference_steps: int = 20,
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        """Generate video and action from observation; future labels are not inputs."""
        video = batch["video"]
        time_s = batch["action_time_s"]
        mask = batch["action_element_mask"].bool()
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[2] < 1:
            raise FoundationModelError("joint serving requires one observed video frame")
        if time_s.ndim != 3 or time_s.shape != mask.shape or (
            time_s.shape[-1] != self.cfg.action_width
        ):
            raise FoundationModelError("joint serving action request ABI changed")
        if num_inference_steps <= 0:
            raise FoundationModelError("num_inference_steps must be positive")
        proprio = batch["proprio"]
        if proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0]
        if proprio.shape != (1, self.cfg.state_width):
            raise FoundationModelError("joint proprio must resolve to [1,D]")
        context, context_mask = self._fused_context(batch)
        result = self.backend.infer_world(
            input_image=video[:, :, 0],
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            current_native_3d=batch["vggt_current_tokens"],
            current_native_3d_view_mask=batch["vggt_current_view_mask"],
            factual_action=None,
            action_time_s=time_s,
            action_element_mask=mask,
            num_video_frames=int(self.cfg.K) + 1,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return {
            **self._action_outputs(result["action_normalized"], batch),
            "video": result["video"],
            "conditioning": result["conditioning"],
            **{key: result[key] for key in ("native_3d_future_tokens", "native_3d_view_mask") if key in result},
        }

    @torch.inference_mode()
    def predict_world(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        num_inference_steps: int = 20,
        seed: int | None = None,
        zero_future_action: bool = False,
    ) -> Mapping[str, Any]:
        """Generate candidate-conditioned K8 video; predict_action owns policy.

        zero_future_action is zero in normalized coordinates: the source
        profile's offset in physical units, not a physical no-op. Future
        video/native targets are not read.
        """

        video = batch["video"]
        action = batch["action"]
        action_time_s = batch["action_time_s"]
        element_mask = batch["action_element_mask"].bool()
        if video.ndim != 5 or video.shape[0] != 1:
            raise FoundationModelError(
                "foundation joint serving currently requires one video sample"
            )
        if action.shape != action_time_s.shape or action.shape != element_mask.shape:
            raise FoundationModelError("joint serving action ABI changed")
        if action.shape[-1] != self.cfg.action_width:
            raise FoundationModelError("joint serving action width changed")
        if num_inference_steps <= 0:
            raise FoundationModelError("num_inference_steps must be positive")
        proprio = batch["proprio"]
        if proprio.ndim == 3 and proprio.shape[1] == 1:
            proprio = proprio[:, 0]
        if proprio.ndim != 2 or proprio.shape != (1, self.cfg.state_width):
            raise FoundationModelError("joint proprio must resolve to [1,D]")
        context, context_mask = self._fused_context(batch)
        factual_action = torch.zeros_like(action) if zero_future_action else action
        result = self.backend.infer_world(
            input_image=video[:, :, 0],
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            current_native_3d=batch["vggt_current_tokens"],
            current_native_3d_view_mask=batch["vggt_current_view_mask"],
            factual_action=factual_action,
            action_time_s=action_time_s,
            action_element_mask=element_mask,
            num_video_frames=int(self.cfg.K) + 1,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        return {
            "video": result["video"],
            "conditioning": result.get("conditioning"),
            **{key: result[key] for key in ("native_3d_future_tokens", "native_3d_view_mask") if key in result},
            "factual_action_normalized": factual_action,
            "action_element_mask": element_mask,
        }

    def parameter_counts(self) -> Mapping[str, int]:
        def count(module: nn.Module, trainable: bool | None = None) -> int:
            values = module.parameters()
            if trainable is not None:
                values = (p for p in values if p.requires_grad is trainable)
            return sum(parameter.numel() for parameter in values)

        return {
            "total": count(self),
            "trainable": count(self, True),
            "wsa_backend": count(self.backend),
            "wm3d_conditioning": count(self.robot_conditioner)
            + count(self.robot_fusion)
            + count(self.backend.native_3d_state)
            + count(self.backend.native_3d_rollout),
        }

    def gradient_owner_modules(self) -> Mapping[str, tuple[nn.Module, ...]]:
        """Disjoint trainable owners used by the common Stage0 audit."""

        wsa_model = self.backend.wsa_model
        owners: dict[str, tuple[nn.Module, ...]] = {
            "foundation_action_time": (self.backend.action_time,),
            "foundation_robot_conditioning": (
                self.robot_conditioner,
                self.robot_fusion,
            ),
            "foundation_native_3d_state": (self.backend.native_3d_state,),
            "foundation_native_3d_rollout": (self.backend.native_3d_rollout,),
        }
        if self.cfg.native_3d_mode == "trajectory":
            owners.pop("foundation_native_3d_rollout")
        if self.cfg.finetune_mode == "adapters":
            owners.update(self._pretrained_interface_modules())
            return owners
        owners["foundation_mixture_of_transformers"] = (wsa_model.dit,)
        proprio = getattr(wsa_model, "proprio_encoder", None)
        if proprio is not None and any(p.requires_grad for p in proprio.parameters()):
            owners["foundation_proprio"] = (proprio,)
        return owners

    @property
    def gradient_audit_start_step(self) -> int:
        # The residual robot-context gate starts at exact zero. Step one learns
        # the gate; step two proves gradient flow through the conditioner.
        return 1


def _load_upstream_symbols(source_root: Path | str) -> tuple[Any, Any]:
    root = Path(source_root).expanduser().resolve(strict=True)
    src = root / "src"
    marker = src / "lerobot" / "policies" / "WSA_Large" / "modeling_wsa_large.py"
    if not marker.is_file():
        raise FoundationModelError("upstream WSA source tree is incomplete")
    text = str(src)
    if text not in sys.path:
        sys.path.insert(0, text)

    # Upstream's ``lerobot.policies`` initializer eagerly imports every policy
    # (including unrelated Cosmos/teleoperation dependencies).  WSA-Large does
    # not need those modules.  Install narrow namespace packages so importing
    # the official WSA-Large files remains dependency-local and reproducible.
    lerobot = importlib.import_module("lerobot")

    def ensure_namespace(name: str, directory: Path, parent: object) -> object:
        existing = sys.modules.get(name)
        if existing is not None:
            return existing
        module = types.ModuleType(name)
        module.__file__ = str(directory / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(directory)]
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
        module.__spec__ = spec
        sys.modules[name] = module
        setattr(parent, name.rsplit(".", 1)[-1], module)
        return module

    policies = ensure_namespace("lerobot.policies", src / "lerobot" / "policies", lerobot)
    ensure_namespace(
        "lerobot.policies.WSA_Base",
        src / "lerobot" / "policies" / "WSA_Base",
        policies,
    )
    ensure_namespace(
        "lerobot.policies.WSA_Large",
        src / "lerobot" / "policies" / "WSA_Large",
        policies,
    )
    from lerobot.policies.WSA_Large.configuration_wsa_large import WSALargeConfig
    from lerobot.policies.WSA_Large.modeling_wsa_large import WSALargePolicy
    from lerobot.policies.WSA_Large.core.models.wan22 import wsa_large

    if not callable(getattr(wsa_large, "masked_action_time_mean", None)):
        raise FoundationModelError(
            "WSA action time-padding loss fix is missing; run foundation-env"
        )
    return WSALargeConfig, WSALargePolicy


def _retain_expanded_wsa_action_prior(
    policy: nn.Module, pretrained_root: Path | str,
) -> list[dict[str, object]]:
    """Preserve released action coordinates when extending storage capacity.

    This is a parameter prior, not a remapping of robot physical semantics.
    Existing rows/columns keep their released values; new coordinates start
    at zero and remain trainable through the existing action objective.
    Proprio is deliberately not copied: its new physical-state layout is
    independently adapted. Old materialized runtimes retain legacy loading.
    """
    from safetensors import safe_open

    expert = policy.model.action_expert
    targets = {
        "model.mot.mixtures.action.action_encoder.weight": expert.action_encoder.weight,
        "model.mot.mixtures.action.action_encoder.bias": expert.action_encoder.bias,
        "model.mot.mixtures.action.head.weight": expert.head.weight,
        "model.mot.mixtures.action.head.bias": expert.head.bias,
    }
    receipt: list[dict[str, object]] = []
    path = Path(pretrained_root) / "model.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as source:
        # Validate all shapes before touching any parameter.
        values = {}
        for name, target in targets.items():
            value = source.get_tensor(name)
            if target is None or value.ndim != target.ndim or any(
                old > new for old, new in zip(value.shape, target.shape, strict=True)
            ):
                raise FoundationModelError(
                    f"pretrained action prior requires expansion, not truncation: {name}"
                )
            if not bool(torch.isfinite(value).all()):
                raise FoundationModelError(f"non-finite pretrained action prior: {name}")
            values[name] = value
        with torch.no_grad():
            for name, target in targets.items():
                value = values[name]
                overlap = tuple(slice(0, int(size)) for size in value.shape)
                target.zero_()
                target[overlap].copy_(value)
                receipt.append({
                    "name": name,
                    "released_shape": list(value.shape),
                    "model_shape": list(target.shape),
                    "new_coordinates": "zero_initialized_trainable",
                })
    return receipt


def build_foundation_world_model(
    config: FoundationWorldModelConfig,
    *,
    device: torch.device | str,
) -> FoundationWorldModel:
    """Load the official WSA-Large weights and install WM3D-only adapters."""

    config.validate(require_paths=True)
    WSALargeConfig, WSALargePolicy = _load_upstream_symbols(
        config.upstream_source_root
    )
    if config.initialization == "wan_video":
        from .foundation_native import build_native_pretraining_model
        return build_native_pretraining_model(config, device=device)
    # The released config carries a registered ``type`` discriminator, so it
    # must be decoded through the upstream base registry rather than through
    # the concrete dataclass (which correctly rejects an unknown ``type``
    # field when addressed directly).
    from lerobot.configs.policies import PreTrainedConfig

    wsa_config = PreTrainedConfig.from_pretrained(
        config.wsa_pretrained_path,
        local_files_only=True,
    )
    if not isinstance(wsa_config, WSALargeConfig):
        raise FoundationModelError(
            "the pretrained source did not resolve to WSA-Large"
        )
    wsa_config.model_id = config.wan_model_path
    wsa_config.tokenizer_model_id = config.wan_tokenizer_model_path
    # The released WSA checkpoint already contains all three MoT experts.
    # Build their exact structures without separate backbone loads, then load
    # the single joint checkpoint once below.
    wsa_config.skip_dit_load_from_pretrain = True
    wsa_config.action_dit_pretrained_path = None
    wsa_config.future_3d_pretrained_path = None
    wsa_config.load_text_encoder = False
    wsa_config.redirect_common_files = False
    wsa_config.native_checkpoint_path = None
    wsa_config.device = str(device)
    wsa_config.dtype = "bfloat16"
    wsa_config.action_dim = config.action_width
    wsa_config.proprio_dim = config.state_width
    # This is WSA's default inference/noise horizon, not the raw-data storage
    # ceiling. Training remains source-native and accepts any K-divisible
    # command count carried by the current source-homogeneous batch.
    wsa_config.action_horizon = config.wsa_action_horizon
    wsa_config.lambda_video = config.video_loss_weight
    wsa_config.lambda_action = config.action_loss_weight
    # Avoid constructing DA3. The adapter installs frozen VGGT targets after load.
    wsa_config.lambda_3d = 0.0
    wsa_config.mask_action_dim_padding_loss = True
    wsa_config.mot_checkpoint_mixed_attn = config.activation_checkpointing
    wsa_config.video_dit_config = dict(wsa_config.video_dit_config)
    wsa_config.video_dit_config["action_dim"] = config.action_width
    wsa_config.video_dit_config["use_gradient_checkpointing"] = bool(
        config.activation_checkpointing
    )
    wsa_config.action_dit_config = dict(wsa_config.action_dit_config)
    wsa_config.action_dit_config["action_dim"] = config.action_width
    wsa_config.action_dit_config["use_gradient_checkpointing"] = bool(
        config.activation_checkpointing
    )
    wsa_config.future_3d_config = dict(wsa_config.future_3d_config)
    wsa_config.future_3d_config["da3_num_views"] = config.num_views
    wsa_config.future_3d_config["use_gradient_checkpointing"] = bool(
        config.activation_checkpointing
    )
    policy = WSALargePolicy.from_pretrained(
        config.wsa_pretrained_path,
        config=wsa_config,
        local_files_only=True,
        strict=False,
    )
    if config.pretrained_interface_init == "pretrained_action_prefix":
        policy._wm3d_action_prior_receipt = _retain_expanded_wsa_action_prior(
            policy, config.wsa_pretrained_path
        )
    # Follow the official fine-tuning boundary: train the MoT experts and
    # proprio projection, while keeping Wan VAE and other fixed components
    # frozen. The WM3D robot/time adapters added below remain trainable.
    policy.train(True)
    backend = WSABackendAdapter(
        policy.model,
        config=config,
    )
    # WSA loads itself onto the requested device, while the WM3D fusion/robot
    # adapters above are newly initialized modules. Keep the
    # factory contract device-complete before the distributed wrapper sees it.
    return FoundationWorldModel(config, backend=backend).to(device)
