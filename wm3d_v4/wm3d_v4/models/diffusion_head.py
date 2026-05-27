"""DiT-style latent diffusion head conditioned on v3 pred_tokens via cross-attention.

Input :  noisy_latent [B, k, 4, 32, 32], timesteps [B*k] (or [B,k]),
         cond_tokens  [B, k, 64, D_cond]   (== v3 pred_tokens)
Output:  predicted_noise [B, k, 4, 32, 32]
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiffusionHeadConfig:
    latent_channels: int = 4
    latent_size: int = 32
    patch_size: int = 2
    hidden: int = 768
    n_layers: int = 14
    n_heads: int = 12
    mlp_ratio: float = 4.0
    cond_dim: int = 2048      # v3 pred_tokens D
    cond_seq_len: int = 64    # tokens per future frame
    timestep_dim: int = 256
    dropout: float = 0.0


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, mlp_ratio: float,
                 cond_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm_x = nn.LayerNorm(hidden, elementwise_affine=False)
        self.norm_c = nn.LayerNorm(cond_hidden)
        self.cross = nn.MultiheadAttention(hidden, n_heads, kdim=cond_hidden,
                                            vdim=cond_hidden, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, int(hidden * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden * mlp_ratio), hidden),
        )
        # AdaLN modulation from timestep emb: 6 params per block (shift/scale × 3 sub-layers)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s_msa, sh_msa, g_msa, s_mlp, sh_mlp, g_mlp = self.ada(t_emb).chunk(6, dim=-1)
        h = modulate(self.norm1(x), sh_msa, s_msa)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + g_msa.unsqueeze(1) * a
        # Cross-attn (no modulation on cross — keep simple)
        h2 = self.norm_x(x)
        c = self.norm_c(cond)
        ca, _ = self.cross(h2, c, c, need_weights=False)
        x = x + ca
        # MLP
        h3 = modulate(self.norm2(x), sh_mlp, s_mlp)
        m = self.mlp(h3)
        x = x + g_mlp.unsqueeze(1) * m
        return x


class DiffusionHead(nn.Module):
    def __init__(self, cfg: DiffusionHeadConfig):
        super().__init__()
        self.cfg = cfg
        ps = cfg.patch_size
        n_patch = (cfg.latent_size // ps) ** 2
        patch_dim = ps * ps * cfg.latent_channels   # 2*2*4 = 16

        self.patch_embed = nn.Linear(patch_dim, cfg.hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patch, cfg.hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.t_proj = nn.Sequential(
            nn.Linear(cfg.timestep_dim, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden, cfg.n_heads, cfg.mlp_ratio,
                     cond_hidden=cfg.cond_dim, dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.hidden, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.hidden, 2 * cfg.hidden))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.final_proj = nn.Linear(cfg.hidden, patch_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def _patchify(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, C, H, W] -> [N, n_patch, patch_dim]
        ps = self.cfg.patch_size
        N, C, H, W = z.shape
        z = z.reshape(N, C, H // ps, ps, W // ps, ps).permute(0, 2, 4, 1, 3, 5)
        return z.reshape(N, (H // ps) * (W // ps), ps * ps * C)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, n_patch, patch_dim] -> [N, C, H, W]
        ps = self.cfg.patch_size
        C = self.cfg.latent_channels
        H = W = self.cfg.latent_size
        N = x.shape[0]
        x = x.reshape(N, H // ps, W // ps, C, ps, ps).permute(0, 3, 1, 4, 2, 5)
        return x.reshape(N, C, H, W)

    def forward(self, z: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z:    [B, k, 4, 32, 32] noisy latents
        t:    [B, k]  integer timesteps (will be broadcast/flat)
        cond: [B, k, 64, cond_dim] (v3 pred_tokens for each future frame)
        returns predicted_noise same shape as z
        """
        B, k = z.shape[:2]
        N = B * k
        z = z.reshape(N, z.shape[2], z.shape[3], z.shape[4])
        cond = cond.reshape(N, cond.shape[2], cond.shape[3])
        t = t.reshape(N)

        # Patchify
        x = self.patch_embed(self._patchify(z)) + self.pos_emb

        # Timestep emb
        t_sin = sinusoidal_timestep_embedding(t, self.cfg.timestep_dim)
        t_emb = self.t_proj(t_sin)                # [N, hidden]

        for blk in self.blocks:
            x = blk(x, t_emb, cond)

        s_f, sh_f = self.final_ada(t_emb).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), sh_f, s_f)
        x = self.final_proj(x)
        out = self._unpatchify(x)                 # [N, 4, 32, 32]
        return out.reshape(B, k, *out.shape[1:])

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
