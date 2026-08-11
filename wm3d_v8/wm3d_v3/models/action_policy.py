"""Direct action chunk policy head for closed-loop benchmark control.

This head is intentionally separate from the tau0-style proposer/ranker path.
The proposer is useful for test-time search; this policy is the behavior-cloning
path we can train and evaluate directly in LIBERO/VLA rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .wm3d_oft import OFTAdapterSpec, WM3DOFTHead


@dataclass
class ActionChunkPolicyConfig:
    token_dim: int = 2048
    task_dim: int = 2048
    hidden: int = 768
    n_layers: int = 6
    n_heads: int = 8
    chunk_layers: int = 2
    horizon: int = 8
    max_context: int = 16
    dropout: float = 0.1
    use_task: bool = True
    patch_pool: str = "mean"
    max_spatial_tokens: int = 64
    use_context_rgb: bool = False
    rgb_spatial_tokens: int = 64
    lowdim_dim: int = 0
    require_lowdim_state: bool = False
    embodiment_vocab_size: int = 0
    require_embodiment: bool = False
    object_state_dim: int = 0
    plan_state_dim: int = 0
    action_history_len: int = 0
    action_history_dim: int = 7
    action_history_as_token: bool = True
    grip_history_adapter: bool = False
    grip_history_hidden: int = 128
    grip_history_zero_init: bool = True
    enable_grip_delta_head: bool = False
    grip_delta_hidden: int = 256
    grip_delta_zero_init: bool = True
    grip_delta_use_composed_action_cond: bool = False
    grip_delta_soft_compose_action_cond: bool = False
    grip_delta_straight_through_action_cond: bool = False
    grip_owner: str = "auto"
    use_progress: bool = False
    progress_dim: int = 1
    progress_mode: str = "token"
    enable_local_residual: bool = False
    local_hidden: int = 256
    local_layers: int = 2
    local_residual_scale: float = 1.0
    local_use_lowdim: bool = True
    local_use_plan_state: bool = True
    local_use_progress: bool = True
    local_use_action_history: bool = True
    enable_waypoint_head: bool = False
    waypoint_hidden: int = 256
    waypoint_layers: int = 2
    waypoint_num_stages: int = 4
    waypoint_stage_dim: int = 4
    waypoint_active_stages: tuple[int, ...] = ()
    waypoint_residual_scale: float = 1.0
    waypoint_mode: str = "residual"
    waypoint_use_summary: bool = True
    waypoint_use_lowdim: bool = True
    waypoint_use_plan_state: bool = True
    waypoint_use_progress: bool = True
    waypoint_use_action_history: bool = True
    enable_prior_policy: bool = False
    prior_chunk_layers: int = 1
    zero_init_output: bool = False
    enable_flow_head: bool = False
    flow_use_as_policy: bool = False
    flow_layers: int = 2
    flow_hidden: int = 768
    flow_action_dim: int = 7
    flow_default_steps: int = 8
    flow_noise_scale: float = 1.0
    flow_zero_init_output: bool = False
    head_type: str = "native"
    oft_max_horizon: int = 16
    oft_query_layers: int = 2
    oft_mlp_hidden: int = 0
    oft_adapter_name: str = "canonical_7d"
    oft_action_dim: int = 7
    oft_grip_indices: tuple[int, ...] = (6,)
    oft_normalization_version: str = "wm3d_d7_norm_v1"
    oft_grip_loss: str = "bce_logits"
    oft_grip_threshold: float = 0.5
    oft_adapters: tuple[dict[str, object], ...] = ()


class ActionChunkPolicy(nn.Module):
    """Predict one executable action chunk from current WM3D context tokens."""

    def __init__(self, cfg: ActionChunkPolicyConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ActionChunkPolicyConfig()
        if self.cfg.progress_mode not in ("token", "summary"):
            raise ValueError(f"progress_mode must be 'token' or 'summary', got {self.cfg.progress_mode!r}")
        if self.cfg.waypoint_mode not in ("residual", "direct", "aux"):
            raise ValueError(f"waypoint_mode must be residual/direct/aux, got {self.cfg.waypoint_mode!r}")
        if self.cfg.patch_pool not in ("mean", "task_attn", "last_patches"):
            raise ValueError(f"patch_pool must be mean/task_attn/last_patches, got {self.cfg.patch_pool!r}")
        if self.cfg.grip_owner not in ("auto", "absolute", "delta_composed"):
            raise ValueError(f"grip_owner must be auto/absolute/delta_composed, got {self.cfg.grip_owner!r}")
        if self.cfg.grip_owner == "absolute" and self.cfg.grip_delta_use_composed_action_cond:
            raise ValueError("absolute grip owner cannot enable composed delta execution")
        if self.cfg.grip_owner == "delta_composed" and not self.cfg.grip_delta_use_composed_action_cond:
            raise ValueError(
                "delta_composed grip owner requires grip_delta_use_composed_action_cond=true"
            )
        if self.cfg.head_type not in ("native", "oft"):
            raise ValueError(f"head_type must be native/oft, got {self.cfg.head_type!r}")
        if self.cfg.head_type == "oft" and self.cfg.enable_flow_head:
            raise ValueError("OFT and flow action heads cannot be active in the same policy")
        if self.cfg.head_type == "oft" and self.cfg.horizon > self.cfg.oft_max_horizon:
            raise ValueError(
                f"policy horizon={self.cfg.horizon} exceeds OFT max_horizon={self.cfg.oft_max_horizon}"
            )
        if self.cfg.require_lowdim_state and self.cfg.lowdim_dim <= 0:
            raise ValueError("required lowdim state needs a positive lowdim_dim")
        if self.cfg.require_embodiment and self.cfg.embodiment_vocab_size <= 0:
            raise ValueError(
                "required embodiment token needs a positive vocabulary size"
            )
        self.policy_feature_dim = int(
            self.cfg.oft_mlp_hidden or self.cfg.hidden
            if self.cfg.head_type == "oft"
            else self.cfg.hidden
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        if self.cfg.patch_pool == "task_attn":
            self.patch_query = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden))
            self.patch_task_proj = nn.Sequential(
                nn.LayerNorm(self.cfg.task_dim),
                nn.Linear(self.cfg.task_dim, self.cfg.hidden),
            )
        else:
            self.patch_query = None
            self.patch_task_proj = None
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        if self.cfg.lowdim_dim > 0:
            self.lowdim_proj = nn.Sequential(
                nn.LayerNorm(self.cfg.lowdim_dim),
                nn.Linear(self.cfg.lowdim_dim, self.cfg.hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
            )
        else:
            self.lowdim_proj = None
        self.embodiment_embed = (
            nn.Embedding(self.cfg.embodiment_vocab_size, self.cfg.hidden)
            if self.cfg.embodiment_vocab_size > 0
            else None
        )
        if self.cfg.object_state_dim > 0:
            self.object_state_proj = nn.Sequential(
                nn.LayerNorm(self.cfg.object_state_dim),
                nn.Linear(self.cfg.object_state_dim, self.cfg.hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
            )
        else:
            self.object_state_proj = None
        if self.cfg.plan_state_dim > 0:
            self.plan_state_proj = nn.Sequential(
                nn.LayerNorm(self.cfg.plan_state_dim),
                nn.Linear(self.cfg.plan_state_dim, self.cfg.hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
            )
        else:
            self.plan_state_proj = None
        if self.cfg.action_history_len > 0 and self.cfg.action_history_as_token:
            self.action_history_proj = nn.Sequential(
                nn.LayerNorm(self.cfg.action_history_len * self.cfg.action_history_dim),
                nn.Linear(self.cfg.action_history_len * self.cfg.action_history_dim, self.cfg.hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
            )
        else:
            self.action_history_proj = None
        hist_adapter_dim = self.cfg.action_history_len * self.cfg.action_history_dim
        if self.cfg.grip_history_adapter and hist_adapter_dim > 0:
            self.grip_history_adapter = nn.Sequential(
                nn.LayerNorm(self.cfg.hidden + hist_adapter_dim),
                nn.Linear(self.cfg.hidden + hist_adapter_dim, self.cfg.grip_history_hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.grip_history_hidden, self.cfg.horizon),
            )
            if self.cfg.grip_history_zero_init:
                nn.init.zeros_(self.grip_history_adapter[-1].weight)
                nn.init.zeros_(self.grip_history_adapter[-1].bias)
        else:
            self.grip_history_adapter = None
        if self.cfg.enable_grip_delta_head:
            self.grip_delta_head = nn.Sequential(
                nn.LayerNorm(self.policy_feature_dim),
                nn.Linear(self.policy_feature_dim, self.cfg.grip_delta_hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.grip_delta_hidden, 3),
            )
            if self.cfg.grip_delta_zero_init:
                nn.init.zeros_(self.grip_delta_head[-1].weight)
                nn.init.zeros_(self.grip_delta_head[-1].bias)
        else:
            self.grip_delta_head = None
        if self.cfg.use_progress:
            progress_layers: list[nn.Module] = []
            if self.cfg.progress_dim > 1:
                progress_layers.append(nn.LayerNorm(self.cfg.progress_dim))
            progress_layers.extend(
                [
                    nn.Linear(self.cfg.progress_dim, self.cfg.hidden),
                    nn.GELU(),
                    nn.Dropout(self.cfg.dropout),
                    nn.Linear(self.cfg.hidden, self.cfg.hidden),
                ]
            )
            self.progress_proj = nn.Sequential(*progress_layers)
        else:
            self.progress_proj = None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden))
        pos_extra = 5 if self.cfg.use_progress and self.cfg.progress_mode == "token" else 4
        self.pos_embed = nn.Parameter(torch.zeros(1, self.cfg.max_context + pos_extra, self.cfg.hidden))
        if self.cfg.patch_pool == "last_patches":
            self.spatial_pos_embed = nn.Parameter(torch.zeros(1, int(self.cfg.max_spatial_tokens), self.cfg.hidden))
        else:
            self.spatial_pos_embed = None
        if self.cfg.use_context_rgb:
            self.rgb_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 256),
                nn.GELU(),
                nn.Conv2d(256, self.cfg.hidden, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            self.rgb_grid = int(round(float(self.cfg.rgb_spatial_tokens) ** 0.5))
            if self.rgb_grid * self.rgb_grid != int(self.cfg.rgb_spatial_tokens):
                raise ValueError("rgb_spatial_tokens must be a square number")
            self.rgb_pool = nn.AdaptiveAvgPool2d((self.rgb_grid, self.rgb_grid))
            self.rgb_pos_embed = nn.Parameter(torch.zeros(1, int(self.cfg.rgb_spatial_tokens), self.cfg.hidden))
        else:
            self.rgb_encoder = None
            self.rgb_grid = 0
            self.rgb_pool = None
            self.rgb_pos_embed = None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.hidden * 4,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers=self.cfg.n_layers)
        self.context_norm = nn.LayerNorm(self.cfg.hidden)

        if self.cfg.head_type == "oft":
            self.horizon_embed = None
            self.chunk_decoder = None
            self.chunk_norm = None
            self.pose_norm_head = None
            self.gripper_head = None
            self.oft_head = WM3DOFTHead(
                context_dim=self.cfg.hidden,
                max_horizon=self.cfg.oft_max_horizon,
                n_heads=self.cfg.n_heads,
                n_layers=self.cfg.oft_query_layers,
                mlp_hidden=self.policy_feature_dim,
                dropout=self.cfg.dropout,
                default_adapter=OFTAdapterSpec(
                    name=self.cfg.oft_adapter_name,
                    action_dim=self.cfg.oft_action_dim,
                    grip_indices=tuple(self.cfg.oft_grip_indices),
                    normalization_version=self.cfg.oft_normalization_version,
                    grip_loss=self.cfg.oft_grip_loss,
                    grip_threshold=self.cfg.oft_grip_threshold,
                ),
            )
            for adapter_cfg in self.cfg.oft_adapters:
                self.oft_head.register_adapter(
                    OFTAdapterSpec(
                        name=str(adapter_cfg["name"]),
                        action_dim=int(adapter_cfg["action_dim"]),
                        grip_indices=tuple(int(value) for value in adapter_cfg.get("grip_indices", ())),
                        state_dim=int(adapter_cfg.get("state_dim", 0)),
                        history_dim=int(adapter_cfg.get("history_dim", 0)),
                        history_len=int(adapter_cfg.get("history_len", 0)),
                        normalization_version=str(
                            adapter_cfg.get("normalization_version", "identity_v1")
                        ),
                        grip_loss=str(adapter_cfg.get("grip_loss", "bce_logits")),
                        grip_threshold=float(adapter_cfg.get("grip_threshold", 0.5)),
                    )
                )
        else:
            self.horizon_embed = nn.Parameter(torch.zeros(1, self.cfg.horizon, self.cfg.hidden))
            chunk_layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.hidden,
                nhead=self.cfg.n_heads,
                dim_feedforward=self.cfg.hidden * 4,
                dropout=self.cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.chunk_decoder = nn.TransformerEncoder(chunk_layer, num_layers=max(1, self.cfg.chunk_layers))
            self.chunk_norm = nn.LayerNorm(self.cfg.hidden)
            self.pose_norm_head = nn.Sequential(
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
                nn.GELU(),
                nn.Linear(self.cfg.hidden, 6),
            )
            self.gripper_head = nn.Sequential(
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
                nn.GELU(),
                nn.Linear(self.cfg.hidden, 1),
            )
            self.oft_head = None
        if self.cfg.enable_flow_head:
            flow_action_dim = int(self.cfg.flow_action_dim)
            if flow_action_dim not in (6, 7):
                raise ValueError(
                    "flow_action_dim must be 6 for pose-only flow or 7 for "
                    f"[pose6,grip_logit], got {self.cfg.flow_action_dim}"
                )
            flow_hidden = int(self.cfg.flow_hidden or self.cfg.hidden)
            self.flow_action_proj = nn.Sequential(
                nn.LayerNorm(flow_action_dim),
                nn.Linear(flow_action_dim, flow_hidden),
                nn.GELU(),
                nn.Linear(flow_hidden, self.cfg.hidden),
            )
            self.flow_time_proj = nn.Sequential(
                nn.Linear(1, flow_hidden),
                nn.SiLU(),
                nn.Linear(flow_hidden, self.cfg.hidden),
            )
            self.flow_horizon_embed = nn.Parameter(torch.zeros(1, self.cfg.horizon, self.cfg.hidden))
            flow_layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.hidden,
                nhead=self.cfg.n_heads,
                dim_feedforward=self.cfg.hidden * 4,
                dropout=self.cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.flow_decoder = nn.TransformerEncoder(flow_layer, num_layers=max(1, int(self.cfg.flow_layers)))
            self.flow_norm = nn.LayerNorm(self.cfg.hidden)
            self.flow_head = nn.Sequential(
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
                nn.GELU(),
                nn.Linear(self.cfg.hidden, flow_action_dim),
            )
            if self.cfg.flow_zero_init_output:
                nn.init.zeros_(self.flow_head[-1].weight)
                nn.init.zeros_(self.flow_head[-1].bias)
        else:
            self.flow_action_proj = None
            self.flow_time_proj = None
            self.flow_horizon_embed = None
            self.flow_decoder = None
            self.flow_norm = None
            self.flow_head = None
        if self.cfg.enable_prior_policy:
            self.prior_horizon_embed = nn.Parameter(torch.zeros(1, self.cfg.horizon, self.cfg.hidden))
            prior_layer = nn.TransformerEncoderLayer(
                d_model=self.cfg.hidden,
                nhead=self.cfg.n_heads,
                dim_feedforward=self.cfg.hidden * 4,
                dropout=self.cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.prior_chunk_decoder = nn.TransformerEncoder(
                prior_layer,
                num_layers=max(1, int(self.cfg.prior_chunk_layers)),
            )
            self.prior_chunk_norm = nn.LayerNorm(self.cfg.hidden)
            self.prior_pose_norm_head = nn.Sequential(
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
                nn.GELU(),
                nn.Linear(self.cfg.hidden, 6),
            )
            self.prior_gripper_head = nn.Sequential(
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
                nn.GELU(),
                nn.Linear(self.cfg.hidden, 1),
            )
        else:
            self.prior_horizon_embed = None
            self.prior_chunk_decoder = None
            self.prior_chunk_norm = None
            self.prior_pose_norm_head = None
            self.prior_gripper_head = None
        self.local_input_dim = self._local_input_dim()
        self.local_residual_head = self._build_local_residual_head() if self.cfg.enable_local_residual else None
        self.waypoint_input_dim = self._waypoint_input_dim()
        self.waypoint_head = self._build_waypoint_head() if self.cfg.enable_waypoint_head else None
        self._reset_parameters()

    def _local_input_dim(self) -> int:
        if not self.cfg.enable_local_residual:
            return 0
        dim = 0
        if self.cfg.local_use_plan_state:
            dim += self.cfg.plan_state_dim
        if self.cfg.local_use_lowdim:
            dim += self.cfg.lowdim_dim
        if self.cfg.local_use_progress and self.cfg.use_progress:
            dim += self.cfg.progress_dim
        if self.cfg.local_use_action_history:
            dim += self.cfg.action_history_len * self.cfg.action_history_dim
        if dim <= 0:
            raise ValueError("enable_local_residual requires at least one local input feature")
        return dim

    def _build_local_residual_head(self) -> nn.Sequential:
        layers: list[nn.Module] = [nn.LayerNorm(self.local_input_dim)]
        hidden = int(self.cfg.local_hidden)
        for li in range(max(1, int(self.cfg.local_layers))):
            layers.append(nn.Linear(self.local_input_dim if li == 0 else hidden, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.cfg.dropout))
        layers.append(nn.Linear(hidden, self.cfg.horizon * 7))
        return nn.Sequential(*layers)

    def _waypoint_input_dim(self) -> int:
        if not self.cfg.enable_waypoint_head:
            return 0
        dim = self.cfg.hidden if self.cfg.waypoint_use_summary else 0
        if self.cfg.waypoint_use_plan_state:
            dim += self.cfg.plan_state_dim
        if self.cfg.waypoint_use_lowdim:
            dim += self.cfg.lowdim_dim
        if self.cfg.waypoint_use_progress and self.cfg.use_progress:
            dim += self.cfg.progress_dim
        if self.cfg.waypoint_use_action_history:
            dim += self.cfg.action_history_len * self.cfg.action_history_dim
        if dim <= 0:
            raise ValueError("enable_waypoint_head requires at least one waypoint input feature")
        if self.cfg.waypoint_num_stages <= 0:
            raise ValueError("waypoint_num_stages must be positive")
        if self.cfg.waypoint_stage_dim <= 0:
            raise ValueError("waypoint_stage_dim must be positive")
        return dim

    def _build_waypoint_head(self) -> nn.Sequential:
        layers: list[nn.Module] = [nn.LayerNorm(self.waypoint_input_dim)]
        hidden = int(self.cfg.waypoint_hidden)
        for li in range(max(1, int(self.cfg.waypoint_layers))):
            layers.append(nn.Linear(self.waypoint_input_dim if li == 0 else hidden, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.cfg.dropout))
        layers.append(nn.Linear(hidden, self.cfg.waypoint_num_stages * self.cfg.horizon * 7))
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        if self.spatial_pos_embed is not None:
            nn.init.normal_(self.spatial_pos_embed, std=0.02)
        if self.rgb_pos_embed is not None:
            nn.init.normal_(self.rgb_pos_embed, std=0.02)
        if self.horizon_embed is not None:
            nn.init.normal_(self.horizon_embed, std=0.02)
        if self.patch_query is not None:
            nn.init.normal_(self.patch_query, std=0.02)
        if self.prior_horizon_embed is not None:
            nn.init.normal_(self.prior_horizon_embed, std=0.02)
        if self.flow_horizon_embed is not None:
            nn.init.normal_(self.flow_horizon_embed, std=0.02)
        if self.cfg.zero_init_output:
            if self.oft_head is not None:
                projection = self.oft_head.adapters[self.cfg.oft_adapter_name].projection
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            else:
                if self.pose_norm_head is not None:
                    nn.init.zeros_(self.pose_norm_head[-1].weight)
                    nn.init.zeros_(self.pose_norm_head[-1].bias)
                if self.gripper_head is not None:
                    nn.init.zeros_(self.gripper_head[-1].weight)
                    nn.init.zeros_(self.gripper_head[-1].bias)
        if self.progress_proj is not None and self.cfg.progress_mode == "summary":
            final = self.progress_proj[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if self.object_state_proj is not None:
            final = self.object_state_proj[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if self.plan_state_proj is not None:
            final = self.plan_state_proj[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if self.local_residual_head is not None:
            final = self.local_residual_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        if self.waypoint_head is not None:
            final = self.waypoint_head[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    def _local_feature(
        self,
        *,
        bsz: int,
        dtype: torch.dtype,
        device: torch.device,
        lowdim_state: torch.Tensor | None,
        action_history: torch.Tensor | None,
        progress_state: torch.Tensor | None,
        plan_state: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.local_residual_head is None:
            return None
        parts: list[torch.Tensor] = []
        if self.cfg.local_use_plan_state:
            if plan_state is None:
                plan_state = torch.zeros(bsz, self.cfg.plan_state_dim, device=device, dtype=dtype)
            if plan_state.shape != (bsz, self.cfg.plan_state_dim):
                raise ValueError(f"plan_state must be {(bsz, self.cfg.plan_state_dim)}, got {tuple(plan_state.shape)}")
            parts.append(plan_state.to(device=device, dtype=dtype))
        if self.cfg.local_use_lowdim:
            if lowdim_state is None:
                lowdim_state = torch.zeros(bsz, self.cfg.lowdim_dim, device=device, dtype=dtype)
            if lowdim_state.shape != (bsz, self.cfg.lowdim_dim):
                raise ValueError(f"lowdim_state must be {(bsz, self.cfg.lowdim_dim)}, got {tuple(lowdim_state.shape)}")
            parts.append(lowdim_state.to(device=device, dtype=dtype))
        if self.cfg.local_use_progress and self.cfg.use_progress:
            if progress_state is None:
                progress_state = torch.zeros(bsz, self.cfg.progress_dim, device=device, dtype=dtype)
            if progress_state.ndim == 1:
                progress_state = progress_state[:, None]
            if progress_state.shape != (bsz, self.cfg.progress_dim):
                raise ValueError(f"progress_state must be {(bsz, self.cfg.progress_dim)}, got {tuple(progress_state.shape)}")
            parts.append(progress_state.to(device=device, dtype=dtype))
        if self.cfg.local_use_action_history:
            if action_history is None:
                action_history = torch.zeros(
                    bsz,
                    self.cfg.action_history_len,
                    self.cfg.action_history_dim,
                    device=device,
                    dtype=dtype,
                )
            expected = (bsz, self.cfg.action_history_len, self.cfg.action_history_dim)
            if action_history.shape != expected:
                raise ValueError(f"action_history must be {expected}, got {tuple(action_history.shape)}")
            parts.append(action_history.to(device=device, dtype=dtype).reshape(bsz, -1))
        return torch.cat(parts, dim=1)

    def _waypoint_feature(
        self,
        *,
        bsz: int,
        dtype: torch.dtype,
        device: torch.device,
        summary: torch.Tensor,
        lowdim_state: torch.Tensor | None,
        action_history: torch.Tensor | None,
        progress_state: torch.Tensor | None,
        plan_state: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.waypoint_head is None:
            return None
        parts: list[torch.Tensor] = []
        if self.cfg.waypoint_use_summary:
            parts.append(summary.to(device=device, dtype=dtype))
        if self.cfg.waypoint_use_plan_state:
            if plan_state is None:
                plan_state = torch.zeros(bsz, self.cfg.plan_state_dim, device=device, dtype=dtype)
            if plan_state.shape != (bsz, self.cfg.plan_state_dim):
                raise ValueError(f"plan_state must be {(bsz, self.cfg.plan_state_dim)}, got {tuple(plan_state.shape)}")
            parts.append(plan_state.to(device=device, dtype=dtype))
        if self.cfg.waypoint_use_lowdim:
            if lowdim_state is None:
                lowdim_state = torch.zeros(bsz, self.cfg.lowdim_dim, device=device, dtype=dtype)
            if lowdim_state.shape != (bsz, self.cfg.lowdim_dim):
                raise ValueError(f"lowdim_state must be {(bsz, self.cfg.lowdim_dim)}, got {tuple(lowdim_state.shape)}")
            parts.append(lowdim_state.to(device=device, dtype=dtype))
        if self.cfg.waypoint_use_progress and self.cfg.use_progress:
            if progress_state is None:
                progress_state = torch.zeros(bsz, self.cfg.progress_dim, device=device, dtype=dtype)
            if progress_state.ndim == 1:
                progress_state = progress_state[:, None]
            if progress_state.shape != (bsz, self.cfg.progress_dim):
                raise ValueError(f"progress_state must be {(bsz, self.cfg.progress_dim)}, got {tuple(progress_state.shape)}")
            parts.append(progress_state.to(device=device, dtype=dtype))
        if self.cfg.waypoint_use_action_history:
            if action_history is None:
                action_history = torch.zeros(
                    bsz,
                    self.cfg.action_history_len,
                    self.cfg.action_history_dim,
                    device=device,
                    dtype=dtype,
                )
            expected = (bsz, self.cfg.action_history_len, self.cfg.action_history_dim)
            if action_history.shape != expected:
                raise ValueError(f"action_history must be {expected}, got {tuple(action_history.shape)}")
            parts.append(action_history.to(device=device, dtype=dtype).reshape(bsz, -1))
        return torch.cat(parts, dim=1)

    def _waypoint_stage_weights(
        self,
        *,
        bsz: int,
        dtype: torch.dtype,
        device: torch.device,
        plan_state: torch.Tensor | None,
    ) -> torch.Tensor:
        n_stages = int(self.cfg.waypoint_num_stages)
        stage_dim = min(int(self.cfg.waypoint_stage_dim), n_stages)
        if plan_state is None or plan_state.shape[-1] < stage_dim:
            return torch.full((bsz, n_stages), 1.0 / n_stages, device=device, dtype=dtype)
        raw = plan_state[:, :stage_dim].to(device=device, dtype=dtype).clamp_min(0.0)
        if stage_dim < n_stages:
            raw = torch.cat(
                [raw, torch.zeros(bsz, n_stages - stage_dim, device=device, dtype=dtype)],
                dim=1,
            )
        denom = raw.sum(dim=1, keepdim=True)
        fallback = torch.full_like(raw, 1.0 / n_stages)
        weights = torch.where(denom > 1e-6, raw / denom.clamp_min(1e-6), fallback)
        active_stages = tuple(int(stage) for stage in (self.cfg.waypoint_active_stages or ()))
        if active_stages:
            mask = torch.zeros(n_stages, device=device, dtype=dtype)
            for stage in active_stages:
                if 0 <= stage < n_stages:
                    mask[stage] = 1.0
            weights = weights * mask[None]
        return weights

    def _normalize_flow_t(
        self,
        flow_t: torch.Tensor,
        *,
        bsz: int,
        horizon: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if flow_t.ndim == 0:
            flow_t = flow_t.reshape(1, 1, 1).expand(bsz, horizon, 1)
        elif flow_t.ndim == 1:
            if flow_t.shape[0] == bsz:
                flow_t = flow_t[:, None, None].expand(-1, horizon, -1)
            elif flow_t.shape[0] == horizon:
                flow_t = flow_t[None, :, None].expand(bsz, -1, -1)
            else:
                raise ValueError(f"flow_t rank1 must have B or H entries, got {tuple(flow_t.shape)}")
        elif flow_t.ndim == 2:
            if flow_t.shape == (bsz, 1):
                flow_t = flow_t[:, None, :].expand(-1, horizon, -1)
            elif flow_t.shape == (bsz, horizon):
                flow_t = flow_t[:, :, None]
            else:
                raise ValueError(f"flow_t rank2 must be [B,1] or [B,H], got {tuple(flow_t.shape)}")
        elif flow_t.ndim == 3:
            if flow_t.shape == (bsz, 1, 1):
                flow_t = flow_t.expand(-1, horizon, -1)
            elif flow_t.shape != (bsz, horizon, 1):
                raise ValueError(f"flow_t rank3 must be [B,1,1] or [B,H,1], got {tuple(flow_t.shape)}")
        else:
            raise ValueError(f"flow_t must have rank <=3, got {tuple(flow_t.shape)}")
        return flow_t.to(device=device, dtype=dtype)

    def _flow_decode(self, q: torch.Tensor, flow_action: torch.Tensor, flow_t: torch.Tensor) -> torch.Tensor:
        if (
            self.flow_action_proj is None
            or self.flow_time_proj is None
            or self.flow_horizon_embed is None
            or self.flow_decoder is None
            or self.flow_norm is None
            or self.flow_head is None
        ):
            raise RuntimeError("flow head is not enabled")
        bsz, horizon, _hidden = q.shape
        expected = (bsz, horizon, int(self.cfg.flow_action_dim))
        if flow_action.shape != expected:
            raise ValueError(f"flow_action must be {expected}, got {tuple(flow_action.shape)}")
        flow_action = flow_action.to(device=q.device, dtype=q.dtype)
        flow_t = self._normalize_flow_t(
            flow_t,
            bsz=bsz,
            horizon=horizon,
            dtype=q.dtype,
            device=q.device,
        )
        x = (
            q
            + self.flow_action_proj(flow_action)
            + self.flow_time_proj(flow_t)
            + self.flow_horizon_embed.to(device=q.device, dtype=q.dtype)
        )
        x = self.flow_norm(self.flow_decoder(x))
        return self.flow_head(x)

    def _sample_flow_action(
        self,
        q: torch.Tensor,
        *,
        steps: int,
        noise: torch.Tensor | None = None,
        noise_scale: float | None = None,
    ) -> torch.Tensor:
        if self.flow_head is None:
            raise RuntimeError("flow sampling requires enable_flow_head=True")
        bsz, horizon, _hidden = q.shape
        flow_action_dim = int(self.cfg.flow_action_dim)
        if noise is None:
            scale = float(self.cfg.flow_noise_scale if noise_scale is None else noise_scale)
            x = torch.randn(bsz, horizon, flow_action_dim, device=q.device, dtype=q.dtype) * scale
        else:
            expected = (bsz, horizon, flow_action_dim)
            if noise.shape != expected:
                raise ValueError(f"flow_noise must be {expected}, got {tuple(noise.shape)}")
            x = noise.to(device=q.device, dtype=q.dtype)
        n_steps = max(1, int(steps))
        dt = 1.0 / float(n_steps)
        for si in range(n_steps):
            t_val = (float(si) + 0.5) / float(n_steps)
            t = x.new_full((bsz, 1, 1), t_val)
            v = self._flow_decode(q, x, t)
            x = x + dt * v
        return x

    def forward(
        self,
        context_tokens: torch.Tensor,
        task_emb: torch.Tensor | None = None,
        *,
        lowdim_state: torch.Tensor | None = None,
        embodiment_id: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        progress_state: torch.Tensor | None = None,
        object_state: torch.Tensor | None = None,
        plan_state: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        flow_action: torch.Tensor | None = None,
        flow_t: torch.Tensor | None = None,
        flow_sample: bool | None = None,
        flow_sample_steps: int | None = None,
        flow_noise: torch.Tensor | None = None,
        flow_noise_scale: float | None = None,
        oft_adapter_name: str | None = None,
        oft_horizon: int | None = None,
        oft_state: torch.Tensor | None = None,
        oft_action_history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context_tokens.ndim != 4:
            raise ValueError(f"context_tokens must be [B,T,P,D], got {tuple(context_tokens.shape)}")
        bsz, time, _patches, dim = context_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        if time > self.cfg.max_context:
            raise ValueError(f"context T={time} exceeds max_context={self.cfg.max_context}")
        selected_oft_adapter = oft_adapter_name or self.cfg.oft_adapter_name
        generic_oft = self.oft_head is not None and selected_oft_adapter != self.cfg.oft_adapter_name
        if generic_oft and lowdim_state is not None:
            raise ValueError("noncanonical OFT adapters must pass benchmark state through oft_state")
        if generic_oft and action_history is not None:
            raise ValueError("noncanonical OFT adapters must pass benchmark history through oft_action_history")

        if self.cfg.use_task:
            if task_emb is None:
                task_emb = torch.zeros(bsz, self.cfg.task_dim, device=context_tokens.device, dtype=context_tokens.dtype)
            if task_emb.shape != (bsz, self.cfg.task_dim):
                raise ValueError(f"task_emb must be {(bsz, self.cfg.task_dim)}, got {tuple(task_emb.shape)}")
            task_emb_in = task_emb.to(device=context_tokens.device, dtype=context_tokens.dtype)
        else:
            task_emb_in = torch.zeros(bsz, self.cfg.task_dim, device=context_tokens.device, dtype=context_tokens.dtype)

        patch_h = self.context_proj(context_tokens.reshape(bsz * time, _patches, dim)).view(
            bsz,
            time,
            _patches,
            self.cfg.hidden,
        )
        extra_spatial_tokens: list[torch.Tensor] = []
        if self.cfg.patch_pool == "task_attn":
            if self.patch_query is None or self.patch_task_proj is None:
                raise RuntimeError("task_attn patch pool is not initialized")
            query = self.patch_query.to(device=context_tokens.device, dtype=patch_h.dtype)
            if self.cfg.use_task:
                query = query + self.patch_task_proj(task_emb_in.to(dtype=patch_h.dtype))[:, None]
            logits = (patch_h * query[:, None]).sum(dim=-1) * (self.cfg.hidden ** -0.5)
            attn = logits.softmax(dim=2)
            frame_h = (patch_h * attn[..., None]).sum(dim=2)
        elif self.cfg.patch_pool == "last_patches":
            frame_h = patch_h.mean(dim=2)
            max_spatial = int(self.cfg.max_spatial_tokens)
            if _patches > max_spatial:
                raise ValueError(f"context patches={_patches} exceeds max_spatial_tokens={max_spatial}")
            spatial_tokens = patch_h[:, -1]
            if self.spatial_pos_embed is None:
                raise RuntimeError("last_patches patch pool is missing spatial_pos_embed")
            spatial_tokens = spatial_tokens + self.spatial_pos_embed[:, :_patches].to(
                device=context_tokens.device,
                dtype=spatial_tokens.dtype,
            )
            extra_spatial_tokens.append(spatial_tokens)
        else:
            frame_h = patch_h.mean(dim=2)
        if self.rgb_encoder is not None:
            if self.rgb_pool is None or self.rgb_pos_embed is None:
                raise RuntimeError("context RGB branch is not initialized")
            if context_rgb is None:
                raise ValueError(
                    "context_rgb is required when use_context_rgb=true; "
                    "training and serving must use the same visual-context contract"
                )
            context_rgb = context_rgb.to(device=context_tokens.device, dtype=frame_h.dtype)
            if context_rgb.ndim != 4 or context_rgb.shape[:2] != (bsz, 3):
                raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context_rgb.shape)}")
            rgb_feat = self.rgb_pool(self.rgb_encoder(context_rgb))
            rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)
            rgb_tokens = rgb_tokens + self.rgb_pos_embed[:, : rgb_tokens.shape[1]].to(
                device=rgb_tokens.device,
                dtype=rgb_tokens.dtype,
            )
            extra_spatial_tokens.append(rgb_tokens)
        cls = self.cls_token.expand(bsz, -1, -1).to(dtype=frame_h.dtype)
        if self.cfg.use_task:
            task_h = self.task_proj(task_emb_in.to(dtype=frame_h.dtype))[:, None]
        else:
            task_h = torch.zeros_like(cls)
        aux_tokens = [cls, task_h]
        progress_h: torch.Tensor | None = None
        object_h: torch.Tensor | None = None
        plan_h: torch.Tensor | None = None
        if self.lowdim_proj is not None and not generic_oft:
            if lowdim_state is None:
                if self.cfg.require_lowdim_state:
                    raise ValueError("lowdim_state is required by the policy ABI")
                lowdim_state = torch.zeros(
                    bsz,
                    self.cfg.lowdim_dim,
                    device=context_tokens.device,
                    dtype=context_tokens.dtype,
                )
            if lowdim_state.shape != (bsz, self.cfg.lowdim_dim):
                raise ValueError(
                    f"lowdim_state must be {(bsz, self.cfg.lowdim_dim)}, "
                    f"got {tuple(lowdim_state.shape)}"
                )
            lowdim_state = lowdim_state.to(
                device=context_tokens.device, dtype=frame_h.dtype
            )
            if not torch.isfinite(lowdim_state).all():
                raise ValueError("lowdim_state contains non-finite values")
            aux_tokens.append(self.lowdim_proj(lowdim_state)[:, None])
        if self.embodiment_embed is not None and not generic_oft:
            if embodiment_id is None:
                if self.cfg.require_embodiment:
                    raise ValueError("embodiment_id is required by the policy ABI")
                embodiment_id = torch.zeros(
                    bsz, device=context_tokens.device, dtype=torch.long
                )
            if embodiment_id.shape != (bsz,):
                raise ValueError(
                    f"embodiment_id must be {(bsz,)}, got {tuple(embodiment_id.shape)}"
                )
            if embodiment_id.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
            ):
                raise ValueError("embodiment_id must use an integer dtype")
            embodiment_id = embodiment_id.to(
                device=context_tokens.device, dtype=torch.long
            )
            if bool((
                (embodiment_id < 0)
                | (embodiment_id >= self.cfg.embodiment_vocab_size)
            ).any()):
                raise ValueError("embodiment_id lies outside the sealed vocabulary")
            aux_tokens.append(self.embodiment_embed(embodiment_id)[:, None])
        if self.object_state_proj is not None:
            if object_state is None:
                object_state = torch.zeros(
                    bsz,
                    self.cfg.object_state_dim,
                    device=context_tokens.device,
                    dtype=context_tokens.dtype,
                )
            if object_state.shape != (bsz, self.cfg.object_state_dim):
                raise ValueError(f"object_state must be {(bsz, self.cfg.object_state_dim)}, got {tuple(object_state.shape)}")
            object_h = self.object_state_proj(object_state.to(device=context_tokens.device, dtype=frame_h.dtype))
        if self.plan_state_proj is not None:
            if plan_state is None:
                plan_state = torch.zeros(
                    bsz,
                    self.cfg.plan_state_dim,
                    device=context_tokens.device,
                    dtype=context_tokens.dtype,
                )
            if plan_state.shape != (bsz, self.cfg.plan_state_dim):
                raise ValueError(f"plan_state must be {(bsz, self.cfg.plan_state_dim)}, got {tuple(plan_state.shape)}")
            plan_h = self.plan_state_proj(plan_state.to(device=context_tokens.device, dtype=frame_h.dtype))
        history_flat: torch.Tensor | None = None
        if self.cfg.action_history_len > 0 and not generic_oft:
            if action_history is None:
                raise ValueError(
                    "action_history is required when action_history_len > 0; "
                    "serving clients must pass the previously executed canonical action"
                )
            expected = (bsz, self.cfg.action_history_len, self.cfg.action_history_dim)
            if action_history.shape != expected:
                raise ValueError(f"action_history must be {expected}, got {tuple(action_history.shape)}")
            history_flat = action_history.to(device=context_tokens.device, dtype=frame_h.dtype).reshape(bsz, -1)
            if self.action_history_proj is not None:
                aux_tokens.append(self.action_history_proj(history_flat)[:, None])
        if self.progress_proj is not None:
            if progress_state is None:
                progress_state = torch.zeros(
                    bsz,
                    self.cfg.progress_dim,
                    device=context_tokens.device,
                    dtype=context_tokens.dtype,
                )
            if progress_state.ndim == 1:
                progress_state = progress_state[:, None]
            expected = (bsz, self.cfg.progress_dim)
            if progress_state.shape != expected:
                raise ValueError(f"progress_state must be {expected}, got {tuple(progress_state.shape)}")
            progress_h = self.progress_proj(progress_state.to(device=context_tokens.device, dtype=frame_h.dtype))
            if self.cfg.progress_mode == "token":
                aux_tokens.append(progress_h[:, None])
        n_aux = sum(token.shape[1] for token in aux_tokens)
        x = torch.cat([*aux_tokens, frame_h], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]].to(device=x.device, dtype=x.dtype)
        n_spatial = 0
        if extra_spatial_tokens:
            spatial_cat = torch.cat(extra_spatial_tokens, dim=1)
            n_spatial = int(spatial_cat.shape[1])
            x = torch.cat([x, spatial_cat], dim=1)
        ctx = self.context_norm(self.context_encoder(x))
        frame_ctx = ctx[:, n_aux: n_aux + time]
        aux_ctx = ctx[:, 1:n_aux]
        aux_summary = aux_ctx.mean(dim=1) if aux_ctx.shape[1] > 0 else torch.zeros_like(ctx[:, 0])
        summary = ctx[:, 0] + 0.5 * aux_summary + 0.5 * frame_ctx[:, -1]
        if n_spatial > 0:
            spatial_ctx = ctx[:, n_aux + time: n_aux + time + n_spatial]
            summary = summary + 0.75 * spatial_ctx.mean(dim=1)
        if progress_h is not None and self.cfg.progress_mode == "summary":
            summary = summary + progress_h
        if object_h is not None:
            summary = summary + object_h
        if plan_h is not None:
            summary = summary + plan_h

        oft_actions: torch.Tensor | None = None
        oft_features: torch.Tensor | None = None
        if self.oft_head is not None:
            adapter_name = selected_oft_adapter
            horizon = int(oft_horizon or self.cfg.horizon)
            oft_actions, oft_features = self.oft_head(
                ctx,
                summary,
                adapter_name=adapter_name,
                horizon=horizon,
                adapter_state=oft_state,
                adapter_action_history=oft_action_history,
            )
            spec = self.oft_head.adapter_specs[adapter_name]
            is_canonical = (
                adapter_name == self.cfg.oft_adapter_name
                and spec.action_dim == 7
                and spec.grip_indices == (6,)
            )
            if not is_canonical:
                decoded = self.oft_head.decode_actions(
                    oft_actions,
                    adapter_name=adapter_name,
                    hard_grip=False,
                )
                executable = self.oft_head.decode_actions(
                    oft_actions,
                    adapter_name=adapter_name,
                    hard_grip=True,
                )
                return {
                    "policy_oft_actions": oft_actions,
                    "policy_oft_decoded_actions": decoded,
                    "policy_oft_executable_actions": executable,
                    "policy_oft_query_features": oft_features,
                    "policy_oft_adapter_name": adapter_name,
                }
            if horizon != self.cfg.horizon:
                raise ValueError(
                    "canonical Stage2 OFT adapter must use configured policy horizon "
                    f"{self.cfg.horizon}, got {horizon}"
                )
            q = oft_features
            pose_norm = oft_actions[..., :6]
            gripper_logit = oft_actions[..., 6]
        else:
            if any(value is not None for value in (oft_adapter_name, oft_horizon, oft_state, oft_action_history)):
                raise ValueError("OFT adapter arguments require head_type='oft'")
            if self.horizon_embed is None or self.chunk_decoder is None or self.chunk_norm is None:
                raise RuntimeError("native policy head is not initialized")
            if self.pose_norm_head is None or self.gripper_head is None:
                raise RuntimeError("native policy output heads are not initialized")
            q = summary[:, None] + self.horizon_embed.to(device=summary.device, dtype=summary.dtype)
            q = self.chunk_norm(self.chunk_decoder(q))
            pose_norm = self.pose_norm_head(q)
            gripper_logit = self.gripper_head(q).squeeze(-1)
        flow_policy_action: torch.Tensor | None = None
        flow_velocity: torch.Tensor | None = None
        flow_t_norm: torch.Tensor | None = None
        if self.cfg.enable_flow_head:
            sample_requested = bool(flow_sample) if flow_sample is not None else (
                self.cfg.flow_use_as_policy and (flow_action is None or flow_t is None)
            )
            if sample_requested:
                flow_policy_action = self._sample_flow_action(
                    q,
                    steps=int(flow_sample_steps or self.cfg.flow_default_steps),
                    noise=flow_noise,
                    noise_scale=flow_noise_scale,
                )
            elif flow_action is not None and flow_t is not None:
                flow_action_dim = int(self.cfg.flow_action_dim)
                if flow_action.shape[:2] != q.shape[:2] or flow_action.shape[-1] != flow_action_dim:
                    raise ValueError(
                        f"flow_action must be [B,H,{flow_action_dim}] matching policy horizon, got {tuple(flow_action.shape)}"
                    )
                flow_t_norm = self._normalize_flow_t(
                    flow_t,
                    bsz=bsz,
                    horizon=int(q.shape[1]),
                    dtype=q.dtype,
                    device=q.device,
                )
                flow_velocity = self._flow_decode(q, flow_action.to(device=q.device, dtype=q.dtype), flow_t_norm)
                flow_policy_action = flow_action.to(device=q.device, dtype=q.dtype) + (1.0 - flow_t_norm) * flow_velocity
        grip_delta_logits = self.grip_delta_head(q) if self.grip_delta_head is not None else None
        if self.grip_history_adapter is not None:
            if history_flat is None:
                history_flat = torch.zeros(
                    bsz,
                    self.cfg.action_history_len * self.cfg.action_history_dim,
                    device=summary.device,
                    dtype=summary.dtype,
                )
            grip_hist_delta = self.grip_history_adapter(torch.cat([summary, history_flat.to(dtype=summary.dtype)], dim=-1))
            gripper_logit = gripper_logit + grip_hist_delta
        else:
            grip_hist_delta = None
        base_pose_norm = pose_norm
        base_gripper_logit = gripper_logit
        flow_pose_norm: torch.Tensor | None = None
        flow_gripper_logit: torch.Tensor | None = None
        if flow_policy_action is not None:
            flow_pose_norm = flow_policy_action[..., :6]
            if flow_policy_action.shape[-1] > 6:
                flow_gripper_logit = flow_policy_action[..., 6]
            else:
                flow_gripper_logit = base_gripper_logit
        if flow_policy_action is not None and self.cfg.flow_use_as_policy:
            pose_norm = flow_policy_action[..., :6]
            if flow_policy_action.shape[-1] > 6:
                gripper_logit = flow_policy_action[..., 6]
        out: dict[str, torch.Tensor] = {
            "base_policy_pose_norm": base_pose_norm,
            "base_policy_gripper_logit": base_gripper_logit,
        }
        if oft_actions is not None and oft_features is not None:
            out["policy_oft_actions"] = oft_actions
            out["policy_oft_decoded_actions"] = self.oft_head.decode_actions(
                oft_actions,
                adapter_name=self.cfg.oft_adapter_name,
                hard_grip=False,
            )
            out["policy_oft_executable_actions"] = self.oft_head.decode_actions(
                oft_actions,
                adapter_name=self.cfg.oft_adapter_name,
                hard_grip=True,
            )
            out["policy_oft_query_features"] = oft_features
        if flow_policy_action is not None and flow_pose_norm is not None and flow_gripper_logit is not None:
            flow_gripper_prob = torch.sigmoid(flow_gripper_logit)
            out.update(
                {
                    "policy_flow_pose_norm": flow_pose_norm,
                    "policy_flow_gripper_logit": flow_gripper_logit,
                    "policy_flow_gripper_prob": flow_gripper_prob,
                    "policy_flow_action": flow_policy_action,
                    "policy_flow_action_cond": torch.cat(
                        [flow_pose_norm, flow_gripper_prob.unsqueeze(-1)],
                        dim=-1,
                    ),
                }
            )
        if flow_velocity is not None:
            out["policy_flow_velocity"] = flow_velocity
            out["policy_flow_input"] = flow_action.to(device=q.device)
            if flow_t_norm is not None:
                out["policy_flow_t"] = flow_t.to(device=q.device)
        if grip_delta_logits is not None:
            out["policy_grip_delta_logits"] = grip_delta_logits
        if grip_hist_delta is not None:
            out["grip_history_logit_delta"] = grip_hist_delta
        if (
            self.prior_horizon_embed is not None
            and self.prior_chunk_decoder is not None
            and self.prior_chunk_norm is not None
            and self.prior_pose_norm_head is not None
            and self.prior_gripper_head is not None
        ):
            prior_q = summary[:, None] + self.prior_horizon_embed.to(device=summary.device, dtype=summary.dtype)
            prior_q = self.prior_chunk_norm(self.prior_chunk_decoder(prior_q))
            prior_pose_norm = self.prior_pose_norm_head(prior_q)
            prior_gripper_logit = self.prior_gripper_head(prior_q).squeeze(-1)
            out.update(
                {
                    "prior_policy_pose_norm": prior_pose_norm,
                    "prior_policy_gripper_logit": prior_gripper_logit,
                    "prior_policy_action_cond": torch.cat(
                        [prior_pose_norm, torch.sigmoid(prior_gripper_logit)[..., None]],
                        dim=-1,
                    ),
                }
            )
        local_feature = self._local_feature(
            bsz=bsz,
            dtype=pose_norm.dtype,
            device=context_tokens.device,
            lowdim_state=lowdim_state,
            action_history=action_history,
            progress_state=progress_state,
            plan_state=plan_state,
        )
        flow_overrode_policy = flow_policy_action is not None and self.cfg.flow_use_as_policy
        if self.local_residual_head is not None and local_feature is not None and not flow_overrode_policy:
            local = self.local_residual_head(local_feature).view(bsz, self.cfg.horizon, 7)
            scale = float(self.cfg.local_residual_scale)
            pose_residual = local[..., :6]
            gripper_residual = local[..., 6]
            pose_norm = pose_norm + scale * pose_residual
            gripper_logit = gripper_logit + scale * gripper_residual
            out.update(
                {
                    "local_pose_residual": pose_residual,
                    "local_gripper_logit_residual": gripper_residual,
                }
            )
        waypoint_feature = self._waypoint_feature(
            bsz=bsz,
            dtype=pose_norm.dtype,
            device=context_tokens.device,
            summary=summary,
            lowdim_state=lowdim_state,
            action_history=action_history,
            progress_state=progress_state,
            plan_state=plan_state,
        )
        if self.waypoint_head is not None and waypoint_feature is not None and not flow_overrode_policy:
            waypoint = self.waypoint_head(waypoint_feature).view(
                bsz,
                self.cfg.waypoint_num_stages,
                self.cfg.horizon,
                7,
            )
            weights = self._waypoint_stage_weights(
                bsz=bsz,
                dtype=waypoint.dtype,
                device=context_tokens.device,
                plan_state=plan_state,
            )
            waypoint = (waypoint * weights[:, :, None, None]).sum(dim=1)
            waypoint_pose = waypoint[..., :6]
            waypoint_gripper = waypoint[..., 6]
            scale = float(self.cfg.waypoint_residual_scale)
            if self.cfg.waypoint_mode == "residual":
                pose_norm = pose_norm + scale * waypoint_pose
                gripper_logit = gripper_logit + scale * waypoint_gripper
            elif self.cfg.waypoint_mode == "direct":
                pose_norm = scale * waypoint_pose
                gripper_logit = scale * waypoint_gripper
            out.update(
                {
                    "waypoint_pose": waypoint_pose,
                    "waypoint_gripper_logit": waypoint_gripper,
                    "waypoint_stage_weights": weights,
                }
            )
        grip_prob = torch.sigmoid(gripper_logit)
        if (
            self.cfg.grip_delta_use_composed_action_cond
            and grip_delta_logits is not None
            and action_history is not None
            and action_history.shape[1] > 0
            and action_history.shape[2] > 6
        ):
            prev = action_history[:, -1, 6].to(device=grip_delta_logits.device, dtype=grip_delta_logits.dtype)
            hard_current = prev > 0.5
            events = grip_delta_logits.argmax(dim=-1)
            hard_steps: list[torch.Tensor] = []
            for ti in range(events.shape[1]):
                hard_current = torch.where(
                    events[:, ti] == 1,
                    torch.ones_like(hard_current),
                    torch.where(events[:, ti] == 2, torch.zeros_like(hard_current), hard_current),
                )
                hard_steps.append(hard_current.to(dtype=grip_delta_logits.dtype))
            grip_composed_hard = torch.stack(hard_steps, dim=1)
            if self.cfg.grip_delta_soft_compose_action_cond:
                delta_prob = torch.softmax(grip_delta_logits.float(), dim=-1).to(dtype=grip_delta_logits.dtype)
                soft_current = prev.clamp(0.0, 1.0)
                soft_steps: list[torch.Tensor] = []
                for ti in range(delta_prob.shape[1]):
                    hold_prob = delta_prob[:, ti, 0]
                    up_prob = delta_prob[:, ti, 1]
                    # A down event sets the state to 0; its effect is represented
                    # by the softmax mass not assigned to hold/up.
                    soft_current = up_prob + hold_prob * soft_current
                    soft_steps.append(soft_current.clamp(0.0, 1.0))
                grip_composed_soft = torch.stack(soft_steps, dim=1)
                out["policy_gripper_composed_hard"] = grip_composed_hard
                out["policy_gripper_composed_soft"] = grip_composed_soft
                if self.cfg.grip_delta_straight_through_action_cond:
                    grip_composed = grip_composed_hard + (grip_composed_soft - grip_composed_soft.detach())
                else:
                    grip_composed = grip_composed_soft
            else:
                grip_composed = grip_composed_hard
            grip_for_action = grip_composed
            out["policy_gripper_composed"] = grip_composed
        else:
            grip_for_action = grip_prob
        action_cond = torch.cat([pose_norm, grip_for_action[..., None]], dim=-1)
        out.update(
            {
                "policy_pose_norm": pose_norm,
                "policy_gripper_logit": gripper_logit,
                "policy_gripper_prob": grip_prob,
                "policy_action_cond": action_cond,
            }
        )
        return out

    def register_oft_adapter(
        self,
        name: str,
        *,
        action_dim: int,
        grip_indices: tuple[int, ...] = (),
        state_dim: int = 0,
        history_dim: int = 0,
        history_len: int = 0,
        normalization_version: str = "identity_v1",
        grip_loss: str = "bce_logits",
        grip_threshold: float = 0.5,
    ) -> None:
        if self.oft_head is None:
            raise RuntimeError("register_oft_adapter requires head_type='oft'")
        self.oft_head.register_adapter(
            OFTAdapterSpec(
                name=name,
                action_dim=action_dim,
                grip_indices=tuple(grip_indices),
                state_dim=state_dim,
                history_dim=history_dim,
                history_len=history_len,
                normalization_version=normalization_version,
                grip_loss=grip_loss,
                grip_threshold=grip_threshold,
            )
        )

    def load_oft_shared_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, list[str]]:
        if self.oft_head is None:
            raise RuntimeError("load_oft_shared_state_dict requires head_type='oft'")
        return self.oft_head.load_shared_state_dict(state_dict)

    def checkpoint_contract(self) -> dict[str, object]:
        contract: dict[str, object] = {
            "version": "wm3d_action_policy_v1",
            "head_type": self.cfg.head_type,
            "horizon": int(self.cfg.horizon),
            "context_schema": {
                "lowdim_dim": int(self.cfg.lowdim_dim),
                "action_history_len": int(self.cfg.action_history_len),
                "action_history_dim": int(self.cfg.action_history_dim),
                "use_context_rgb": bool(self.cfg.use_context_rgb),
                "use_task": bool(self.cfg.use_task),
                "task_dim": int(self.cfg.task_dim),
                "patch_pool": self.cfg.patch_pool,
                "max_spatial_tokens": int(self.cfg.max_spatial_tokens),
            },
        }
        if self.oft_head is None:
            return contract
        contract.update(
            {
                "version": "wm3d_oft_v1",
                "max_horizon": int(self.cfg.oft_max_horizon),
                "default_adapter": self.cfg.oft_adapter_name,
                "trunk_feature_dim": int(self.oft_head.trunk.feature_dim),
                "adapters": {
                    name: {
                        "action_dim": int(spec.action_dim),
                        "grip_indices": list(spec.grip_indices),
                        "state_dim": int(spec.state_dim),
                        "history_dim": int(spec.history_dim),
                        "history_len": int(spec.history_len),
                        "normalization_version": spec.normalization_version,
                        "grip_loss": spec.grip_loss,
                        "grip_threshold": float(spec.grip_threshold),
                    }
                    for name, spec in sorted(self.oft_head.adapter_specs.items())
                },
            }
        )
        return contract
