"""WSA-Large foundation backend with WM3D robot and native-3D contracts."""

from __future__ import annotations

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


def validate_foundation_objective(mapping: Mapping[str, object]) -> None:
    expected = {
        "schema": FOUNDATION_OBJECTIVE_SCHEMA,
        "owner": "wsa_joint_training_loss",
    }
    if dict(mapping) != expected:
        raise FoundationModelError(
            "foundation objective must be owned exclusively by WSA joint training"
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
    activation_checkpointing: bool = True
    dtype: str = "bf16"

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
        target_grid = isqrt(self.native_3d_target_tokens)
        if target_grid**2 != self.native_3d_target_tokens:
            raise FoundationModelError("native_3d_target_tokens must form a square")
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
        for name in (
            "video_loss_weight",
            "action_loss_weight",
            "native_3d_loss_weight",
        ):
            if float(getattr(self, name)) <= 0:
                raise FoundationModelError(f"{name} must be positive")
        if self.dtype != "bf16":
            raise FoundationModelError("the WSA foundation profile requires bf16")
        if require_paths:
            for name in (
                "upstream_source_root",
                "wsa_pretrained_path",
                "wan_model_path",
                "wan_tokenizer_model_path",
                "wan_context_store_path",
            ):
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
        mean = (time_s.float() * element_mask).sum(dim=-1) / count.clamp_min(1)
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
        if not bool(valid.all()):
            raise FoundationModelError("WSA action horizon cannot contain padded commands")
        if not bool((command_time >= 0).all()):
            raise FoundationModelError("future action command time must be non-negative")
        phase = command_time[..., None].float() * self.frequencies.float()
        features = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.mlp(features)


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


class WSABackendAdapter(nn.Module):
    """Add WM3D time and VGGT contracts without forking WSA's MoT core."""

    def __init__(
        self,
        wsa_model: nn.Module,
        *,
        native_3d_loss_weight: float,
        action_time_fourier_dim: int,
    ) -> None:
        super().__init__()
        self.wsa_model = wsa_model
        action_expert = self.wsa_model.action_expert
        self.action_time = ActionTimeConditioner(
            int(action_expert.hidden_dim), action_time_fourier_dim
        )
        query_layers = tuple(self.wsa_model.future_3d_expert.query_layer_indices)
        self.vggt_teacher = BoundVGGTTeacher(
            int(self.wsa_model.future_3d_expert.da3_tokens_per_view),
            len(query_layers),
        )
        self.wsa_model.da3_teacher = self.vggt_teacher
        self.wsa_model.da3_teacher_layers = tuple(range(len(query_layers)))
        self.wsa_model.da3_layer_weights = tuple(1.0 for _ in query_layers)
        self.wsa_model.loss_lambda_3d = float(native_3d_loss_weight)
        self._active_time_update: torch.Tensor | None = None
        self._time_hook = action_expert.action_encoder.register_forward_hook(
            self._inject_action_time
        )

    def _inject_action_time(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if self._active_time_update is None:
            raise RuntimeError("ActionDiT time update was not bound")
        if output.shape != self._active_time_update.shape:
            raise FoundationModelError("ActionDiT time update shape changed")
        return output + self._active_time_update.to(dtype=output.dtype)

    def forward(
        self,
        sample: Mapping[str, torch.Tensor],
        vggt_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self._active_time_update is not None:
            raise RuntimeError("concurrent WSA backend forwards are forbidden")
        self._active_time_update = self.action_time(
            sample["action_time_s"], sample["action_element_mask"]
        )
        self.vggt_teacher.bind(vggt_target)
        try:
            total, metrics = self.wsa_model.training_loss(
                sample,
                collect_loss_dict=True,
                collect_detailed_loss_dict=True,
                loss_dict_as_tensors=True,
            )
        finally:
            self.vggt_teacher.clear()
            self._active_time_update = None
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
        return result

    @torch.inference_mode()
    def infer_action(
        self,
        *,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
        action_time_s: torch.Tensor,
        action_element_mask: torch.Tensor,
        num_inference_steps: int,
        seed: int | None,
    ) -> torch.Tensor:
        """Run ActionDiT with the same source-native time update used in training."""

        if self._active_time_update is not None:
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
        try:
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
        action = result.get("action")
        expected = (int(action_time_s.shape[1]), int(action_time_s.shape[2]))
        if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected:
            raise FoundationModelError(
                f"WSA ActionDiT returned an invalid action shape: {getattr(action, 'shape', None)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError("foundation action inference is non-finite")
        return action.unsqueeze(0)


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
        result = self.backend(sample, batch["vggt_future_tokens"])
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
            action_time_s=action_time_s,
            action_element_mask=element_mask,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )
        offset = batch["action_normalization_offset"].reshape(
            1, 1, self.cfg.action_width
        )
        scale = batch["action_normalization_scale"].reshape(
            1, 1, self.cfg.action_width
        )
        offset = offset.to(device=normalized.device, dtype=normalized.dtype)
        scale = scale.to(device=normalized.device, dtype=normalized.dtype)
        mask = element_mask.to(device=normalized.device)
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
            + count(self.robot_fusion),
        }

    def gradient_owner_modules(self) -> Mapping[str, tuple[nn.Module, ...]]:
        """Disjoint trainable owners used by the common Stage0 audit."""

        wsa_model = self.backend.wsa_model
        owners: dict[str, tuple[nn.Module, ...]] = {
            "foundation_mixture_of_transformers": (wsa_model.dit,),
            "foundation_action_time": (self.backend.action_time,),
            "foundation_robot_conditioning": (
                self.robot_conditioner,
                self.robot_fusion,
            ),
        }
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

    return WSALargeConfig, WSALargePolicy


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
    # Follow the official fine-tuning boundary: train the MoT experts and
    # proprio projection, while keeping Wan VAE and other fixed components
    # frozen. The WM3D robot/time adapters added below remain trainable.
    policy.train(True)
    backend = WSABackendAdapter(
        policy.model,
        native_3d_loss_weight=config.native_3d_loss_weight,
        action_time_fourier_dim=config.action_time_fourier_dim,
    )
    # WSA loads itself onto the requested device, while the WM3D fusion/robot
    # adapters above are newly initialized modules. Keep the
    # factory contract device-complete before the distributed wrapper sees it.
    return FoundationWorldModel(config, backend=backend).to(device)
