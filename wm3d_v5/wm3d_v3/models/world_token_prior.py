"""Text/context/action conditioned generator for VGGT-style world tokens."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class WorldTokenPriorConfig:
    token_dim: int = 2048
    n_tokens: int = 64
    horizon: int = 8
    context_frames: int = 16
    hidden: int = 768
    n_layers: int = 4
    n_heads: int = 8
    mlp_mult: int = 4
    dropout: float = 0.0
    task_dim: int = 2048
    action_dim: int = 7
    use_context: bool = True
    use_action: bool = True
    predict_initial: bool = True


class WorldTokenPrior(nn.Module):
    """Generate or denoise 3D world tokens from task text plus optional conditions.

    The learned query is used in both the direct generation path and the
    flow-denoising path, so training with clean token targets still updates the
    parameters needed by text-to-world inference.
    """

    def __init__(self, cfg: WorldTokenPriorConfig | None = None):
        super().__init__()
        self.cfg = cfg or WorldTokenPriorConfig()
        h = self.cfg.hidden
        self.total_frames = self.cfg.horizon + (1 if self.cfg.predict_initial else 0)
        self.task_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.task_dim),
            nn.Linear(self.cfg.task_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.cfg.token_dim),
            nn.Linear(self.cfg.token_dim, h),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(self.cfg.action_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.noisy_proj = nn.Linear(self.cfg.token_dim, h)
        self.out_proj = nn.Linear(h, self.cfg.token_dim)
        self.time_proj = nn.Sequential(nn.Linear(1, h), nn.GELU(), nn.Linear(h, h))

        self.query = nn.Parameter(torch.zeros(1, self.total_frames, self.cfg.n_tokens, h))
        self.frame_pos = nn.Parameter(torch.zeros(1, self.total_frames, 1, h))
        self.patch_pos = nn.Parameter(torch.zeros(1, 1, self.cfg.n_tokens, h))
        self.context_pos = nn.Parameter(torch.zeros(1, self.cfg.context_frames, h))
        self.action_pos = nn.Parameter(torch.zeros(1, self.cfg.horizon, h))
        for param in (self.query, self.frame_pos, self.patch_pos, self.context_pos, self.action_pos):
            nn.init.normal_(param, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=h,
            nhead=self.cfg.n_heads,
            dim_feedforward=h * self.cfg.mlp_mult,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=self.cfg.n_layers)
        self.norm = nn.LayerNorm(h)

    def _batch_size(
        self,
        task_emb: torch.Tensor,
        context_tokens: torch.Tensor | None,
        action_cond: torch.Tensor | None,
        clean_tokens: torch.Tensor | None = None,
    ) -> int:
        for x in (task_emb, context_tokens, action_cond, clean_tokens):
            if x is not None:
                return int(x.shape[0])
        raise ValueError("WorldTokenPrior needs at least task_emb to infer batch size")

    def _memory(
        self,
        task_emb: torch.Tensor,
        context_tokens: torch.Tensor | None,
        action_cond: torch.Tensor | None,
        bsz: int,
    ) -> torch.Tensor:
        device = task_emb.device
        dtype = self.task_proj[1].weight.dtype
        mem = [self.task_proj(task_emb.to(device=device, dtype=dtype)).unsqueeze(1)]
        if self.cfg.use_context and context_tokens is not None:
            if context_tokens.ndim != 4:
                raise ValueError(f"context_tokens must be [B,T,P,D], got {tuple(context_tokens.shape)}")
            ctx = context_tokens.to(device=device, dtype=dtype).mean(dim=2)
            if ctx.shape[-1] != self.cfg.token_dim:
                raise ValueError(f"expected context token dim {self.cfg.token_dim}, got {ctx.shape[-1]}")
            pos = self.context_pos[:, : ctx.shape[1]].to(device=device, dtype=dtype)
            mem.append(self.context_proj(ctx) + pos)
        if self.cfg.use_action:
            if action_cond is None:
                action_cond = torch.zeros(bsz, self.cfg.horizon, self.cfg.action_dim, device=device, dtype=dtype)
            if action_cond.shape != (bsz, self.cfg.horizon, self.cfg.action_dim):
                raise ValueError(
                    f"action_cond must be {(bsz, self.cfg.horizon, self.cfg.action_dim)}, "
                    f"got {tuple(action_cond.shape)}"
                )
            action = action_cond.to(device=device, dtype=dtype)
            mem.append(self.action_proj(action) + self.action_pos.to(device=device, dtype=dtype))
        return torch.cat(mem, dim=1)

    def _query_base(self, bsz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = self.query + self.frame_pos + self.patch_pos
        return q.to(device=device, dtype=dtype).expand(bsz, -1, -1, -1).reshape(
            bsz, self.total_frames * self.cfg.n_tokens, self.cfg.hidden
        )

    def _split_tokens(self, all_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {"prior_all_tokens": all_tokens}
        if self.cfg.predict_initial:
            out["prior_initial_tokens"] = all_tokens[:, 0]
            out["prior_future_tokens"] = all_tokens[:, 1:]
        else:
            out["prior_future_tokens"] = all_tokens
        return out

    def _decode_velocity(self, noised: torch.Tensor, t: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        bsz = noised.shape[0]
        q = self.noisy_proj(noised).reshape(bsz, self.total_frames * self.cfg.n_tokens, self.cfg.hidden)
        q = q + self._query_base(bsz, memory.device, memory.dtype)
        q = q + self.time_proj(t.reshape(bsz, 1).to(device=memory.device, dtype=memory.dtype)).unsqueeze(1)
        decoded = self.norm(self.decoder(q, memory))
        return self.out_proj(decoded).reshape_as(noised)

    def forward(
        self,
        task_emb: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        clean_tokens: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz = self._batch_size(task_emb, context_tokens, action_cond, clean_tokens)
        memory = self._memory(task_emb, context_tokens, action_cond, bsz)
        dtype = memory.dtype
        device = memory.device
        out: dict[str, torch.Tensor] = {}

        if clean_tokens is not None:
            expected = (bsz, self.total_frames, self.cfg.n_tokens, self.cfg.token_dim)
            if tuple(clean_tokens.shape) != expected:
                raise ValueError(f"clean_tokens must be {expected}, got {tuple(clean_tokens.shape)}")
            clean = clean_tokens.to(device=device, dtype=dtype)
            noise = torch.randn(clean.shape, device=device, dtype=dtype, generator=generator)
            t = torch.rand((bsz, 1, 1, 1), device=device, dtype=dtype, generator=generator)
            noised = (1.0 - t) * noise + t * clean
            velocity = self._decode_velocity(noised, t, memory)
            velocity_target = clean - noise
            all_tokens = noised + (1.0 - t) * velocity
            out.update({
                "prior_velocity": velocity,
                "prior_velocity_target": velocity_target,
                "prior_flow_t": t.reshape(bsz),
                "prior_noised_tokens": noised,
            })
        else:
            q = self._query_base(bsz, device, dtype)
            decoded = self.norm(self.decoder(q, memory))
            all_tokens = self.out_proj(decoded).reshape(
                bsz, self.total_frames, self.cfg.n_tokens, self.cfg.token_dim
            )

        out.update(self._split_tokens(all_tokens))
        return out

    @torch.no_grad()
    def sample_flow(
        self,
        task_emb: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Lightweight Euler sampler for token-space flow inference."""
        steps = max(1, int(steps))
        bsz = self._batch_size(task_emb, context_tokens, action_cond)
        memory = self._memory(task_emb, context_tokens, action_cond, bsz)
        shape = (bsz, self.total_frames, self.cfg.n_tokens, self.cfg.token_dim)
        x = torch.randn(shape, device=memory.device, dtype=memory.dtype, generator=generator)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t_val = (i + 0.5) / float(steps)
            t = torch.full((bsz, 1, 1, 1), t_val, device=memory.device, dtype=memory.dtype)
            x = x + dt * self._decode_velocity(x, t, memory)
        return self._split_tokens(x)
