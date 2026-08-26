"""WSA-style flow-matching policy for WM3D V9.

The module is intentionally separate from the V8 regression owner.  It keeps
WM3D's group-major public ABI, semantic masks, physical normalization and
variable action timestamps while replacing continuous one-shot regression by
velocity flow matching and iterative integration.  Binary controller fields
remain logits because diffusing Bernoulli targets would destroy their exact
semantics.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS

from .native_world_model import (
    CrossAttention,
    NativeWorldModelConfig,
    RMSNorm,
    SelfAttention,
    SwiGLU,
)


class ContinuousFlowMatchScheduler:
    """Exact shifted continuous-flow schedule used by WSA Large.

    Weight statistics are computed with Python scalars so model construction
    remains safe inside the FSDP meta-device context.
    """

    def __init__(self, num_train_timesteps: int, shift: float, eps: float = 1.0e-10):
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if shift <= 0:
            raise ValueError("flow shift must be positive")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        values = []
        steps = float(self.num_train_timesteps)
        for index in range(self.num_train_timesteps):
            u = 1.0 - float(index) / steps
            sigma = self.shift * u / (1.0 + (self.shift - 1.0) * u)
            timestep = sigma * steps
            values.append(exp(-2.0 * ((timestep - steps / 2.0) / steps) ** 2))
        self._y_min = min(values)
        self._weight_norm_const = sum(value - self._y_min for value in values) / len(
            values
        )

    @staticmethod
    def phi(value: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * value / (1.0 + (shift - 1.0) * value)

    def sample_training_timestep(
        self, batch: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if batch <= 0:
            raise ValueError("flow batch must be positive")
        uniform = torch.rand(batch, device=device, dtype=torch.float32)
        sigma = self.phi(uniform, self.shift)
        return (sigma * float(self.num_train_timesteps)).to(dtype=dtype)

    def sigma(self, timestep: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return (timestep.float() / float(self.num_train_timesteps)).to(dtype=dtype)

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.sigma(timestep, dtype=sample.dtype).view(
            -1, *((1,) * (sample.ndim - 1))
        )
        return (1.0 - sigma) * sample + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        value = timestep.float()
        steps = float(self.num_train_timesteps)
        weight = torch.exp(-2.0 * ((value - steps / 2.0) / steps) ** 2)
        return (weight - self._y_min) / (self._weight_norm_const + self.eps)

    def build_inference_schedule(
        self,
        num_steps: int,
        *,
        shift: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_steps <= 0:
            raise ValueError("flow inference steps must be positive")
        if shift <= 0:
            raise ValueError("flow inference shift must be positive")
        uniform = torch.linspace(
            1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32
        )
        sigma = self.phi(uniform, shift)
        timesteps = sigma[:-1] * float(self.num_train_timesteps)
        deltas = sigma[1:] - sigma[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(
        velocity: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        return sample + velocity * delta.to(device=sample.device, dtype=sample.dtype)


class FlowTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion-time embedding followed by WSA's two-layer MLP."""

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        half = cfg.time_fourier_dim // 2
        self._half = half
        self._log_max_period = log(10_000.0)
        frequencies = torch.exp(
            -self._log_max_period
            * torch.arange(half, dtype=torch.float32)
            / max(1, half)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.proj = nn.Sequential(
            nn.Linear(cfg.time_fourier_dim, cfg.action_hidden),
            nn.SiLU(),
            nn.Linear(cfg.action_hidden, cfg.action_hidden),
        )

    def reset_parameters(self) -> None:
        values = torch.exp(
            -self._log_max_period
            * torch.arange(
                self._half,
                device=self.frequencies.device,
                dtype=self.frequencies.dtype,
            )
            / max(1, self._half)
        )
        self.frequencies.copy_(values)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError("flow timestep must be [B]")
        angles = timestep.float()[:, None] * self.frequencies.float()[None]
        features = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return self.proj(features.to(dtype=timestep.dtype))


def _modulate(
    value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return value * (1.0 + scale[:, None, None]) + shift[:, None, None]


class FlowActionBlock(nn.Module):
    """Grouped ActionDiT block with persistent policy-context conditioning.

    WSA applies self-attention, context cross-attention and time-modulated FFN
    in every block.  WM3D factorizes the two attentions over physical time and
    action group so the same contract scales to whole-body action layouts.
    """

    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        dim = cfg.action_hidden
        self.self_norm = nn.LayerNorm(dim, eps=1.0e-6, elementwise_affine=False)
        self.self_attn = SelfAttention(dim, cfg.action_heads, cfg.dropout)
        self.cross_norm = nn.LayerNorm(dim, eps=1.0e-6, elementwise_affine=False)
        self.cross_attn = CrossAttention(dim, dim, cfg.action_heads, cfg.dropout)
        self.ff_norm = nn.LayerNorm(dim, eps=1.0e-6, elementwise_affine=False)
        self.ff = SwiGLU(dim, cfg.policy_flow_ff_mult, cfg.dropout)
        self.modulation = nn.Parameter(torch.empty(1, 6, dim))
        nn.init.normal_(self.modulation, std=1.0 / sqrt(dim))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.modulation, std=1.0 / sqrt(self.modulation.shape[-1]))

    @staticmethod
    def _key_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask[:, None, None, :]

    def _factorized_self(
        self,
        value: torch.Tensor,
        normalized: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, groups, dim = value.shape
        temporal = value.transpose(1, 2).reshape(batch * groups, steps, dim)
        temporal_input = normalized.transpose(1, 2).reshape(batch * groups, steps, dim)
        temporal_mask = token_mask.transpose(1, 2).reshape(batch * groups, steps)
        temporal = temporal + self.self_attn(
            temporal_input, allowed_mask=self._key_mask(temporal_mask)
        )
        value = temporal.view(batch, groups, steps, dim).transpose(1, 2)

        grouped = value.reshape(batch * steps, groups, dim)
        group_input = normalized.reshape(batch * steps, groups, dim)
        group_mask = token_mask.reshape(batch * steps, groups)
        grouped = grouped + self.self_attn(
            group_input, allowed_mask=self._key_mask(group_mask)
        )
        return grouped.view(batch, steps, groups, dim)

    def _factorized_cross(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, groups, dim = value.shape
        temporal_query = (
            self.cross_norm(value).transpose(1, 2).reshape(batch * groups, steps, dim)
        )
        temporal_context = condition.transpose(1, 2).reshape(batch * groups, steps, dim)
        temporal_mask = token_mask.transpose(1, 2).reshape(batch * groups, steps)
        temporal = (
            self.cross_attn(
                temporal_query,
                temporal_context,
                allowed_mask=self._key_mask(temporal_mask),
            )
            .view(batch, groups, steps, dim)
            .transpose(1, 2)
        )

        group_query = self.cross_norm(value).reshape(batch * steps, groups, dim)
        group_context = condition.reshape(batch * steps, groups, dim)
        group_mask = token_mask.reshape(batch * steps, groups)
        grouped = self.cross_attn(
            group_query,
            group_context,
            allowed_mask=self._key_mask(group_mask),
        ).view(batch, steps, groups, dim)
        return value + 0.5 * (temporal + grouped)

    def forward(
        self,
        value: torch.Tensor,
        condition: torch.Tensor,
        time_modulation: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if value.shape != condition.shape:
            raise ValueError("flow value and policy condition must align")
        if token_mask.shape != value.shape[:-1]:
            raise ValueError("flow token mask must align to values")
        if tuple(time_modulation.shape) != (
            value.shape[0],
            6,
            value.shape[-1],
        ):
            raise ValueError("flow time modulation must be [B,6,H]")
        modulation = self.modulation.to(dtype=time_modulation.dtype) + time_modulation
        shift_self, scale_self, gate_self, shift_ff, scale_ff, gate_ff = (
            modulation.unbind(dim=1)
        )
        normalized = _modulate(self.self_norm(value), shift_self, scale_self)
        attended = self._factorized_self(value, normalized, token_mask)
        value = value + gate_self[:, None, None] * (attended - value)
        value = self._factorized_cross(value, condition, token_mask)
        ff_input = _modulate(self.ff_norm(value), shift_ff, scale_ff)
        value = value + gate_ff[:, None, None] * self.ff(ff_input)
        return value * token_mask[..., None].to(dtype=value.dtype)


class FlowActionOutputHead(nn.Module):
    def __init__(self, cfg: NativeWorldModelConfig):
        super().__init__()
        dim = cfg.action_hidden
        self.norm = nn.LayerNorm(dim, eps=1.0e-6, elementwise_affine=False)
        self.output = nn.Linear(dim, cfg.max_action_dim)
        self.modulation = nn.Parameter(torch.empty(1, 2, dim))
        nn.init.normal_(self.modulation, std=1.0 / sqrt(dim))

    def reset_parameters(self) -> None:
        nn.init.normal_(self.modulation, std=1.0 / sqrt(self.modulation.shape[-1]))

    def forward(
        self, value: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        modulation = (
            self.modulation.to(dtype=time_embedding.dtype) + time_embedding[:, None]
        )
        shift, scale = modulation.unbind(dim=1)
        value = _modulate(self.norm(value), shift, scale)
        return self.output(value)


class GroupedFlowActionHead(nn.Module):
    """Continuous flow policy with an unchanged grouped physical-action ABI."""

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
        self.scheduler = ContinuousFlowMatchScheduler(
            cfg.policy_flow_train_timesteps, cfg.policy_flow_train_shift
        )
        self.action_input = nn.Linear(cfg.max_action_dim, cfg.action_hidden)
        self.condition_input = nn.Linear(
            cfg.action_hidden, cfg.action_hidden, bias=False
        )
        self.time_embedding = FlowTimestepEmbedding(cfg)
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(cfg.action_hidden, cfg.action_hidden * 6)
        )
        blocks: tuple[nn.Module, ...] = tuple(
            FlowActionBlock(cfg) for _ in range(cfg.policy_flow_layers)
        )
        if cfg.activation_checkpointing:
            blocks = tuple(checkpoint_wrapper(block) for block in blocks)
        self.blocks = nn.ModuleList(blocks)
        self.velocity_head = FlowActionOutputHead(cfg)
        self.binary_norm = RMSNorm(cfg.action_hidden)
        self.binary_output = nn.Linear(cfg.action_hidden, cfg.max_action_dim)

    def _semantic_masks(
        self,
        action_semantic_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic = action_semantic_ids[:, :, None]
        output_mask = query_mask[..., None] & semantic.ne(0)
        gripper = torch.zeros_like(output_mask)
        for semantic_id in self._GRIPPER_IDS:
            gripper = gripper | semantic.eq(semantic_id)
        binary = torch.zeros_like(output_mask)
        for semantic_id in self._BINARY_IDS:
            binary = binary | semantic.eq(semantic_id)
        binary = binary & output_mask
        return output_mask, gripper & output_mask, binary, output_mask & ~binary

    @staticmethod
    def _validate_normalization(
        action_semantic_ids: torch.Tensor,
        offset: torch.Tensor,
        scale: torch.Tensor,
        binary: torch.Tensor,
    ) -> None:
        if (
            offset.shape != action_semantic_ids.shape
            or scale.shape != action_semantic_ids.shape
            or not bool(torch.isfinite(offset).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0).any())
        ):
            raise ValueError("action normalization statistics are invalid")
        if bool((binary & (offset[:, :, None].ne(0) | scale[:, :, None].ne(1))).any()):
            raise ValueError(
                "gripper/binary/discrete action normalization must be identity"
            )

    def _predict_velocity(
        self,
        noisy_action: torch.Tensor,
        query: torch.Tensor,
        token_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Public action is [B,G,C,A], policy context is [B,C,G,H].
        action = noisy_action.transpose(1, 2)
        condition = self.condition_input(query)
        value = self.action_input(action) + condition
        time = self.time_embedding(timestep)
        time_modulation = self.time_projection(time).view(
            query.shape[0], 6, self.cfg.action_hidden
        )
        for block in self.blocks:
            value = block(value, condition, time_modulation, token_mask.transpose(1, 2))
        return self.velocity_head(value, time).transpose(1, 2)

    @staticmethod
    def _validate_optional_tensor(
        value: torch.Tensor,
        expected: tuple[int, ...],
        *,
        name: str,
    ) -> None:
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must be {expected}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")

    def forward(
        self,
        query: torch.Tensor,
        action_semantic_ids: torch.Tensor,
        query_mask: torch.Tensor,
        normalization_offset: torch.Tensor,
        normalization_scale: torch.Tensor,
        *,
        target_action: Optional[torch.Tensor] = None,
        target_action_mask: Optional[torch.Tensor] = None,
        flow_noise: Optional[torch.Tensor] = None,
        flow_timestep: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch, queries, groups, _ = query.shape
        expected = (batch, groups, queries, self.cfg.max_action_dim)
        if tuple(query_mask.shape) != (batch, groups, queries):
            raise ValueError("policy query mask must be [B,G,C]")
        if tuple(action_semantic_ids.shape) != (
            batch,
            groups,
            self.cfg.max_action_dim,
        ):
            raise ValueError("action semantics must be [B,G,A]")
        output_mask, gripper_mask, binary_mask, continuous_mask = self._semantic_masks(
            action_semantic_ids, query_mask
        )
        self._validate_normalization(
            action_semantic_ids,
            normalization_offset,
            normalization_scale,
            binary_mask,
        )
        binary_logits = self.binary_output(self.binary_norm(query)).transpose(1, 2)

        flow: dict[str, torch.Tensor] = {}
        if self.training:
            if target_action is None or target_action_mask is None:
                raise ValueError(
                    "flow-matching training requires target action and mask"
                )
            self._validate_optional_tensor(
                target_action, expected, name="target_action"
            )
            if tuple(target_action_mask.shape) != expected:
                raise ValueError(f"target_action_mask must be {expected}")
            if flow_noise is None:
                flow_noise = torch.randn_like(target_action)
            else:
                self._validate_optional_tensor(flow_noise, expected, name="flow_noise")
            if flow_timestep is None:
                flow_timestep = self.scheduler.sample_training_timestep(
                    batch, device=query.device, dtype=query.dtype
                )
            else:
                self._validate_optional_tensor(
                    flow_timestep, (batch,), name="flow_timestep"
                )
                if bool(
                    (
                        (flow_timestep < 0)
                        | (flow_timestep > self.cfg.policy_flow_train_timesteps)
                    ).any()
                ):
                    raise ValueError("flow_timestep lies outside the training schedule")
            supervised_continuous = continuous_mask & target_action_mask.bool()
            target_continuous = torch.where(
                continuous_mask, target_action, torch.zeros_like(target_action)
            )
            noise = torch.where(
                continuous_mask, flow_noise, torch.zeros_like(flow_noise)
            )
            noisy = self.scheduler.add_noise(target_continuous, noise, flow_timestep)
            velocity = self._predict_velocity(
                noisy, query, query_mask, flow_timestep
            ) * continuous_mask.to(dtype=query.dtype)
            target_velocity = self.scheduler.training_target(
                target_continuous, noise
            ) * supervised_continuous.to(dtype=query.dtype)
            sigma = self.scheduler.sigma(flow_timestep, dtype=query.dtype).view(
                batch, 1, 1, 1
            )
            continuous = noisy - sigma * velocity
            flow = {
                "policy_flow_velocity": velocity,
                "policy_flow_target_velocity": target_velocity,
                "policy_flow_continuous_mask": supervised_continuous,
                "policy_flow_timestep": flow_timestep,
                "policy_flow_sigma": sigma.reshape(batch),
                "policy_flow_weight": self.scheduler.training_weight(flow_timestep),
                "policy_flow_noisy_action": noisy,
            }
        else:
            # Validation batches carry labels through the shared pretrain
            # forward adapter, but sampling must never read them. Serving omits
            # them entirely; both paths therefore execute the same integrator.
            del target_action, target_action_mask
            if flow_timestep is not None:
                raise ValueError("flow inference uses the sealed integration schedule")
            if flow_noise is None:
                sample = torch.randn(expected, device=query.device, dtype=query.dtype)
            else:
                self._validate_optional_tensor(flow_noise, expected, name="flow_noise")
                sample = flow_noise.to(dtype=query.dtype)
            sample = sample * continuous_mask.to(dtype=sample.dtype)
            timesteps, deltas = self.scheduler.build_inference_schedule(
                self.cfg.policy_flow_inference_steps,
                shift=self.cfg.policy_flow_infer_shift,
                device=query.device,
                dtype=query.dtype,
            )
            for timestep, delta in zip(timesteps, deltas):
                velocity = self._predict_velocity(
                    sample,
                    query,
                    query_mask,
                    timestep.expand(batch),
                ) * continuous_mask.to(dtype=sample.dtype)
                sample = self.scheduler.step(velocity, delta, sample)
                sample = sample * continuous_mask.to(dtype=sample.dtype)
            continuous = sample

        raw = torch.where(binary_mask, binary_logits, continuous)
        normalized = torch.where(binary_mask, torch.sigmoid(binary_logits), continuous)
        physical = (
            continuous * normalization_scale[:, :, None]
            + normalization_offset[:, :, None]
        )
        decoded = torch.where(binary_mask, torch.sigmoid(binary_logits), physical)
        mask_float = output_mask.to(dtype=raw.dtype)
        output = {
            "policy_action_raw": raw * mask_float,
            "policy_action_normalized": normalized * mask_float,
            "policy_action": decoded * mask_float,
            "policy_action_mask": output_mask,
            "policy_gripper_mask": gripper_mask,
            "policy_binary_mask": binary_mask,
        }
        output.update(flow)
        return output
