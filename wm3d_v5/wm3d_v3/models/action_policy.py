"""Direct action chunk policy head for closed-loop benchmark control.

This head is intentionally separate from the tau0-style proposer/ranker path.
The proposer is useful for test-time search; this policy is the behavior-cloning
path we can train and evaluate directly in LIBERO/VLA rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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
    object_state_dim: int = 0
    plan_state_dim: int = 0
    action_history_len: int = 0
    action_history_dim: int = 7
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
        if self.cfg.action_history_len > 0:
            self.action_history_proj = nn.Sequential(
                nn.LayerNorm(self.cfg.action_history_len * self.cfg.action_history_dim),
                nn.Linear(self.cfg.action_history_len * self.cfg.action_history_dim, self.cfg.hidden),
                nn.GELU(),
                nn.Dropout(self.cfg.dropout),
                nn.Linear(self.cfg.hidden, self.cfg.hidden),
            )
        else:
            self.action_history_proj = None
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
        nn.init.normal_(self.horizon_embed, std=0.02)
        if self.patch_query is not None:
            nn.init.normal_(self.patch_query, std=0.02)
        if self.prior_horizon_embed is not None:
            nn.init.normal_(self.prior_horizon_embed, std=0.02)
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


    def forward(
        self,
        context_tokens: torch.Tensor,
        task_emb: torch.Tensor | None = None,
        *,
        lowdim_state: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        progress_state: torch.Tensor | None = None,
        object_state: torch.Tensor | None = None,
        plan_state: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context_tokens.ndim != 4:
            raise ValueError(f"context_tokens must be [B,T,P,D], got {tuple(context_tokens.shape)}")
        bsz, time, _patches, dim = context_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        if time > self.cfg.max_context:
            raise ValueError(f"context T={time} exceeds max_context={self.cfg.max_context}")

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
                context_rgb = torch.zeros(bsz, 3, 256, 256, device=context_tokens.device, dtype=context_tokens.dtype)
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
        if self.lowdim_proj is not None:
            if lowdim_state is None:
                lowdim_state = torch.zeros(
                    bsz,
                    self.cfg.lowdim_dim,
                    device=context_tokens.device,
                    dtype=context_tokens.dtype,
                )
            if lowdim_state.shape != (bsz, self.cfg.lowdim_dim):
                raise ValueError(f"lowdim_state must be {(bsz, self.cfg.lowdim_dim)}, got {tuple(lowdim_state.shape)}")
            aux_tokens.append(self.lowdim_proj(lowdim_state.to(device=context_tokens.device, dtype=frame_h.dtype))[:, None])
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
        if self.action_history_proj is not None:
            if action_history is None:
                action_history = torch.zeros(
                    bsz,
                    self.cfg.action_history_len,
                    self.cfg.action_history_dim,
                    device=context_tokens.device,
                    dtype=context_tokens.dtype,
                )
            expected = (bsz, self.cfg.action_history_len, self.cfg.action_history_dim)
            if action_history.shape != expected:
                raise ValueError(f"action_history must be {expected}, got {tuple(action_history.shape)}")
            hist = action_history.to(device=context_tokens.device, dtype=frame_h.dtype).reshape(bsz, -1)
            aux_tokens.append(self.action_history_proj(hist)[:, None])
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

        q = summary[:, None] + self.horizon_embed.to(device=summary.device, dtype=summary.dtype)
        q = self.chunk_norm(self.chunk_decoder(q))
        pose_norm = self.pose_norm_head(q)
        gripper_logit = self.gripper_head(q).squeeze(-1)
        out: dict[str, torch.Tensor] = {
            "base_policy_pose_norm": pose_norm,
            "base_policy_gripper_logit": gripper_logit,
        }
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
        if self.local_residual_head is not None and local_feature is not None:
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
        if self.waypoint_head is not None and waypoint_feature is not None:
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
        action_cond = torch.cat([pose_norm, torch.sigmoid(gripper_logit)[..., None]], dim=-1)
        out.update(
            {
                "policy_pose_norm": pose_norm,
                "policy_gripper_logit": gripper_logit,
                "policy_action_cond": action_cond,
            }
        )
        return out
