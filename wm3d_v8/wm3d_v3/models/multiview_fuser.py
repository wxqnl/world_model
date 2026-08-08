"""View-aware token fusion that preserves the exact v6 mono path at init."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MultiViewFuserConfig:
    token_dim: int = 2048
    n_heads: int = 16
    dropout: float = 0.0
    use_camera_pose: bool = True
    pose_dim: int = 16


class MultiViewTokenFuser(nn.Module):
    """Fuse an optional wrist view into external-anchor tokens.

    The output shape is always ``[B,T,P,D]``.  A missing wrist view is an
    exact identity operation.  The output projection is zero-initialized, so
    loading v6 and enabling this module cannot perturb mono or paired
    predictions before the fuser learns.
    """

    def __init__(self, cfg: MultiViewFuserConfig | None = None):
        super().__init__()
        self.cfg = cfg or MultiViewFuserConfig()
        if self.cfg.token_dim % self.cfg.n_heads:
            raise ValueError("token_dim must be divisible by n_heads")
        self.anchor_norm = nn.LayerNorm(self.cfg.token_dim)
        self.wrist_norm = nn.LayerNorm(self.cfg.token_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.cfg.token_dim,
            self.cfg.n_heads,
            dropout=self.cfg.dropout,
            batch_first=True,
        )
        self.role_embedding = nn.Parameter(torch.zeros(2, self.cfg.token_dim))
        nn.init.normal_(self.role_embedding, std=0.02)
        self.pose_projection = (
            nn.Sequential(
                nn.LayerNorm(self.cfg.pose_dim),
                nn.Linear(self.cfg.pose_dim, self.cfg.token_dim),
            )
            if self.cfg.use_camera_pose
            else None
        )
        self.output_projection = nn.Linear(self.cfg.token_dim, self.cfg.token_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        # Exact identity is already guaranteed by the zero output projection.
        # A second multiplicative zero here would give both factors zero
        # gradient forever, so the gate must start non-zero.
        self.residual_gate = nn.Parameter(torch.ones(()))

    @staticmethod
    def _flatten_pose(pose: torch.Tensor, bsz: int, frames: int) -> torch.Tensor:
        if pose.shape[:2] != (bsz, frames):
            raise ValueError(f"camera pose must start with [B,T], got {tuple(pose.shape)}")
        return pose.reshape(bsz * frames, -1)

    def forward(
        self,
        anchor_tokens: torch.Tensor,
        wrist_tokens: torch.Tensor | None = None,
        *,
        view_mask: torch.Tensor | None = None,
        anchor_camera_pose: torch.Tensor | None = None,
        wrist_camera_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if anchor_tokens.ndim != 4:
            raise ValueError(f"anchor_tokens must be [B,T,P,D], got {tuple(anchor_tokens.shape)}")
        bsz, frames, patches, dim = anchor_tokens.shape
        if dim != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}, got {dim}")
        if wrist_tokens is None:
            return anchor_tokens
        if wrist_tokens.shape != anchor_tokens.shape:
            raise ValueError("anchor and wrist token shapes must match")
        if view_mask is None:
            wrist_valid = torch.ones(bsz, frames, device=anchor_tokens.device, dtype=torch.bool)
        else:
            if view_mask.shape == (bsz, frames, 2):
                if not bool(torch.all(view_mask[..., 0])):
                    raise ValueError("external anchor cannot be masked")
                wrist_valid = view_mask[..., 1].bool()
            elif view_mask.shape == (bsz, frames):
                wrist_valid = view_mask.bool()
            else:
                raise ValueError("view_mask must be [B,T] wrist mask or [B,T,2]")
        if not bool(torch.any(wrist_valid)):
            # Mono OXE batches deliberately supply a dummy wrist tensor so all
            # fuser parameters remain part of the DDP graph. Avoid paying for
            # a 2048-D cross-attention whose residual will be masked to zero.
            dependency = anchor_tokens.new_zeros(())
            for parameter in self.parameters():
                dependency = dependency + parameter.reshape(-1)[0] * 0.0
            return anchor_tokens + dependency.to(anchor_tokens.dtype)
        query = self.anchor_norm(anchor_tokens).reshape(bsz * frames, patches, dim)
        key_value = self.wrist_norm(wrist_tokens).reshape(bsz * frames, patches, dim)
        query = query + self.role_embedding[0].to(query.dtype)
        key_value = key_value + self.role_embedding[1].to(key_value.dtype)
        if self.pose_projection is not None:
            if (anchor_camera_pose is None) != (wrist_camera_pose is None):
                raise ValueError("anchor and wrist camera poses must be provided together")
            if anchor_camera_pose is not None:
                anchor_pose = self._flatten_pose(anchor_camera_pose, bsz, frames)
                wrist_pose = self._flatten_pose(wrist_camera_pose, bsz, frames)
                if anchor_pose.shape[-1] != self.cfg.pose_dim:
                    raise ValueError(f"expected flattened pose dim {self.cfg.pose_dim}")
                query = query + self.pose_projection(anchor_pose.to(query.dtype))[:, None]
                key_value = key_value + self.pose_projection(wrist_pose.to(key_value.dtype))[:, None]

        fused, _ = self.cross_attention(query, key_value, key_value, need_weights=False)
        residual = self.output_projection(fused).reshape(bsz, frames, patches, dim)
        residual = residual * wrist_valid[:, :, None, None].to(residual.dtype)
        return anchor_tokens + torch.tanh(self.residual_gate) * residual
