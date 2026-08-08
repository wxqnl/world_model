"""Transferable OFT action queries and thin benchmark adapters for WM3D."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OFTAdapterSpec:
    name: str
    action_dim: int
    grip_indices: tuple[int, ...] = ()
    state_dim: int = 0
    history_dim: int = 0
    history_len: int = 0
    normalization_version: str = "identity_v1"
    grip_loss: str = "bce_logits"
    grip_threshold: float = 0.5

    def validate(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError("OFT adapter name must be non-empty and cannot contain '.'")
        if self.action_dim <= 0:
            raise ValueError("OFT adapter action_dim must be positive")
        if len(set(self.grip_indices)) != len(self.grip_indices):
            raise ValueError("OFT adapter grip_indices must be unique")
        if any(index < 0 or index >= self.action_dim for index in self.grip_indices):
            raise ValueError(
                f"OFT adapter grip indices {self.grip_indices} exceed action_dim={self.action_dim}"
            )
        if self.state_dim < 0 or self.history_dim < 0 or self.history_len < 0:
            raise ValueError("OFT adapter state/history dimensions cannot be negative")
        if (self.history_dim > 0) != (self.history_len > 0):
            raise ValueError("OFT adapter history_dim and history_len must both be zero or both be positive")
        if not self.normalization_version:
            raise ValueError("OFT adapter normalization_version must be non-empty")
        if self.grip_loss != "bce_logits":
            raise ValueError(f"unsupported OFT grip loss {self.grip_loss!r}")
        if not 0.0 < float(self.grip_threshold) < 1.0:
            raise ValueError("OFT adapter grip_threshold must be in (0,1)")


class OFTResidualBlock(nn.Module):
    """StarVLA-style pre-norm residual MLP block."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.fc(self.layer_norm(x)))


class OFTMLPTrunk(nn.Module):
    """Shared portion of the StarVLA residual OFT regression head."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.activation = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [OFTResidualBlock(hidden), OFTResidualBlock(hidden)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(self.layer_norm1(x)))
        for block in self.res_blocks:
            x = block(x)
        return self.layer_norm2(x)


class OFTActionAdapter(nn.Module):
    """A benchmark-specific final projection over shared OFT features."""

    def __init__(self, hidden: int, spec: OFTAdapterSpec) -> None:
        super().__init__()
        spec.validate()
        self.spec = spec
        self.projection = nn.Linear(hidden, spec.action_dim)
        self.state_projection = (
            nn.Sequential(nn.LayerNorm(spec.state_dim), nn.Linear(spec.state_dim, hidden))
            if spec.state_dim > 0
            else None
        )
        self.history_projection = (
            nn.Sequential(
                nn.LayerNorm(spec.history_len * spec.history_dim),
                nn.Linear(spec.history_len * spec.history_dim, hidden),
            )
            if spec.history_dim > 0
            else None
        )
        self.condition_fusion = (
            nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            if self.state_projection is not None or self.history_projection is not None
            else None
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = torch.zeros_like(features[:, 0])
        if self.state_projection is not None:
            expected = (features.shape[0], self.spec.state_dim)
            if state is None or state.shape != expected:
                actual = None if state is None else tuple(state.shape)
                raise ValueError(f"OFT adapter state must be {expected}, got {actual}")
            condition = condition + self.state_projection(
                state.to(device=features.device, dtype=features.dtype)
            )
        elif state is not None:
            raise ValueError(f"OFT adapter {self.spec.name!r} does not declare a state input")
        if self.history_projection is not None:
            if (
                action_history is None
                or action_history.ndim != 3
                or action_history.shape[0] != features.shape[0]
                or action_history.shape[1] != self.spec.history_len
                or action_history.shape[-1] != self.spec.history_dim
            ):
                actual = None if action_history is None else tuple(action_history.shape)
                raise ValueError(
                    "OFT adapter history must be [B,history_len="
                    f"{self.spec.history_len},history_dim={self.spec.history_dim}], got {actual}"
                )
            history = action_history.to(device=features.device, dtype=features.dtype).reshape(
                features.shape[0],
                -1,
            )
            condition = condition + self.history_projection(history)
        elif action_history is not None:
            raise ValueError(f"OFT adapter {self.spec.name!r} does not declare a history input")
        if self.condition_fusion is not None:
            expanded = condition[:, None].expand(-1, features.shape[1], -1)
            features = features + self.condition_fusion(torch.cat([features, expanded], dim=-1))
        return self.projection(features)

    def decode_actions(self, actions: torch.Tensor, *, hard_grip: bool = False) -> torch.Tensor:
        if actions.shape[-1] != self.spec.action_dim:
            raise ValueError(
                f"OFT actions must end in action_dim={self.spec.action_dim}, got {tuple(actions.shape)}"
            )
        decoded = actions.clone()
        if self.spec.grip_indices:
            indices = list(self.spec.grip_indices)
            grip_prob = torch.sigmoid(actions[..., indices])
            decoded[..., indices] = (
                (grip_prob >= self.spec.grip_threshold).to(dtype=actions.dtype)
                if hard_grip
                else grip_prob
            )
        return decoded


class WM3DOFTQueryTrunk(nn.Module):
    """Generate parallel action-query features from encoded WM3D context.

    StarVLA WanOFT expands one pooled video feature into K action queries with a
    K-specific linear layer. WM3D uses deterministic temporal positions and a
    shared cross-attention decoder so Stage2 K8 weights remain defined at K16.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        max_horizon: int,
        n_heads: int,
        n_layers: int,
        mlp_hidden: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if max_horizon <= 0:
            raise ValueError("OFT max_horizon must be positive")
        if context_dim % n_heads != 0:
            raise ValueError("OFT context_dim must be divisible by n_heads")
        self.context_dim = int(context_dim)
        self.max_horizon = int(max_horizon)
        self.feature_dim = int(mlp_hidden)
        self.query_seed = nn.Parameter(torch.zeros(1, 1, context_dim))
        self.summary_proj = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, context_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=context_dim,
            nhead=n_heads,
            dim_feedforward=context_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(layer, num_layers=max(1, int(n_layers)))
        self.query_norm = nn.LayerNorm(context_dim)
        self.mlp = OFTMLPTrunk(context_dim, mlp_hidden)
        self.register_buffer(
            "temporal_position",
            self._sinusoidal_position(max_horizon, context_dim),
            persistent=True,
        )
        nn.init.normal_(self.query_seed, std=0.02)

    @staticmethod
    def _sinusoidal_position(length: int, hidden: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, hidden, 2, dtype=torch.float32)
            * (-math.log(10000.0) / float(hidden))
        )
        encoding = torch.zeros(length, hidden, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div)
        if hidden > 1:
            encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(
        self,
        memory: torch.Tensor,
        summary: torch.Tensor,
        *,
        horizon: int,
    ) -> torch.Tensor:
        if memory.ndim != 3:
            raise ValueError(f"OFT memory must be [B,S,H], got {tuple(memory.shape)}")
        if summary.shape != (memory.shape[0], self.context_dim):
            raise ValueError(
                f"OFT summary must be {(memory.shape[0], self.context_dim)}, got {tuple(summary.shape)}"
            )
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError(
                f"OFT horizon must be in [1,max_horizon={self.max_horizon}], got {horizon}"
            )
        temporal = self.temporal_position[:, :horizon].to(device=summary.device, dtype=summary.dtype)
        queries = (
            self.query_seed.to(device=summary.device, dtype=summary.dtype)
            + temporal
            + self.summary_proj(summary)[:, None]
        )
        causal_mask = torch.triu(
            torch.ones(horizon, horizon, device=queries.device, dtype=torch.bool),
            diagonal=1,
        )
        decoded = self.query_decoder(queries, memory, tgt_mask=causal_mask)
        return self.mlp(self.query_norm(decoded))


class WM3DOFTHead(nn.Module):
    """Shared variable-horizon OFT trunk with thin action-schema adapters."""

    def __init__(
        self,
        *,
        context_dim: int,
        max_horizon: int,
        n_heads: int,
        n_layers: int,
        mlp_hidden: int,
        dropout: float,
        default_adapter: OFTAdapterSpec,
    ) -> None:
        super().__init__()
        self.trunk = WM3DOFTQueryTrunk(
            context_dim=context_dim,
            max_horizon=max_horizon,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
        self.adapters = nn.ModuleDict()
        self.adapter_specs: dict[str, OFTAdapterSpec] = {}
        self.default_adapter_name = default_adapter.name
        self.register_adapter(default_adapter)

    def register_adapter(self, spec: OFTAdapterSpec) -> None:
        spec.validate()
        if spec.name in self.adapters:
            existing = self.adapter_specs[spec.name]
            if existing != spec:
                raise ValueError(f"OFT adapter {spec.name!r} already exists with spec {existing}")
            return
        adapter = OFTActionAdapter(self.trunk.feature_dim, spec).to(
            device=self.trunk.query_seed.device,
            dtype=self.trunk.query_seed.dtype,
        )
        self.adapters[spec.name] = adapter
        self.adapter_specs[spec.name] = spec

    def forward(
        self,
        memory: torch.Tensor,
        summary: torch.Tensor,
        *,
        adapter_name: str | None = None,
        horizon: int,
        adapter_state: torch.Tensor | None = None,
        adapter_action_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        name = adapter_name or self.default_adapter_name
        if name not in self.adapters:
            raise KeyError(f"unknown OFT adapter {name!r}; available={tuple(self.adapters.keys())}")
        features = self.trunk(memory, summary, horizon=horizon)
        return self.adapters[name](
            features,
            state=adapter_state,
            action_history=adapter_action_history,
        ), features

    def decode_actions(
        self,
        actions: torch.Tensor,
        *,
        adapter_name: str | None = None,
        hard_grip: bool = False,
    ) -> torch.Tensor:
        name = adapter_name or self.default_adapter_name
        if name not in self.adapters:
            raise KeyError(f"unknown OFT adapter {name!r}; available={tuple(self.adapters.keys())}")
        return self.adapters[name].decode_actions(actions, hard_grip=hard_grip)

    def load_shared_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, list[str]]:
        """Load shared OFT weights and shape-compatible adapters from any policy checkpoint."""
        target = self.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        shape_mismatch: list[str] = []
        skipped: list[str] = []
        for raw_key, value in state_dict.items():
            marker = "oft_head."
            if marker in raw_key:
                key = raw_key.split(marker, 1)[1]
            elif raw_key.startswith(("trunk.", "adapters.")):
                key = raw_key
            else:
                skipped.append(raw_key)
                continue
            if key not in target:
                skipped.append(raw_key)
            elif target[key].shape != value.shape:
                shape_mismatch.append(raw_key)
            else:
                compatible[key] = value
        self.load_state_dict(compatible, strict=False)
        missing_shared = sorted(key for key in target if key.startswith("trunk.") and key not in compatible)
        return {
            "loaded": sorted(compatible),
            "missing_shared": missing_shared,
            "shape_mismatch": sorted(shape_mismatch),
            "skipped": sorted(skipped),
        }
