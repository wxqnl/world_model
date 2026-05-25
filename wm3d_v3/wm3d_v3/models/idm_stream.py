"""Stage-B IDM stream: attends jointly over s_in[t..t+T-1] and pred_tokens[t+T..t+T+k-1].

This is the architectural change Stage A could not effect: ActionStream only
saw s_in, so it couldn't get an inverse-dynamics gradient. IDMStream consumes
both endpoints — pre-action observations and the model's own predicted
future tokens — and runs a transformer over the combined sequence. Role
embeddings tag observed vs predicted frames so the model can distinguish
them.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class IDMStreamConfig:
    T: int = 16
    k: int = 8
    P: int = 64
    D: int = 2048
    hidden: int = 896
    n_layers: int = 10
    n_heads: int = 14
    z_dim: int = 192
    mlp_mult: int = 4
    dropout: float = 0.0
    cond_dim: int = 2048


class IDMStream(nn.Module):
    def __init__(self, cfg: IDMStreamConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.D, cfg.hidden)
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.hidden)
        self.frame_pos = nn.Parameter(torch.zeros(1, cfg.T + cfg.k, 1, cfg.hidden))
        self.patch_pos = nn.Parameter(torch.zeros(1, 1, cfg.P, cfg.hidden))
        self.role_emb = nn.Parameter(torch.zeros(2, cfg.hidden))  # [observed, predicted]
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
        nn.init.normal_(self.role_emb, std=0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden, nhead=cfg.n_heads,
                dim_feedforward=cfg.hidden * cfg.mlp_mult,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ) for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hidden)
        self.dec_q = nn.Parameter(torch.zeros(1, cfg.k, cfg.hidden))
        nn.init.normal_(self.dec_q, std=0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden, nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden * cfg.mlp_mult,
            dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.z_head = nn.Linear(cfg.hidden, cfg.z_dim)

    def forward(self, s_in: torch.Tensor, pred_tokens: torch.Tensor,
                c: torch.Tensor) -> dict:
        B = s_in.size(0)
        T, k, P, H = self.cfg.T, self.cfg.k, self.cfg.P, self.cfg.hidden
        xi = self.in_proj(s_in)                          # [B, T, P, H]
        xp = self.in_proj(pred_tokens)                   # [B, k, P, H]
        xi = xi + self.role_emb[0].view(1, 1, 1, H)
        xp = xp + self.role_emb[1].view(1, 1, 1, H)
        x = torch.cat([xi, xp], dim=1) + self.frame_pos + self.patch_pos  # [B, T+k, P, H]
        x = x.view(B, (T + k) * P, H)
        cond_tok = self.cond_proj(c).unsqueeze(1)
        x = torch.cat([cond_tok, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        q = self.dec_q.expand(B, -1, -1)
        z = self.z_head(self.decoder(q, x))
        return {"z_a": z}
