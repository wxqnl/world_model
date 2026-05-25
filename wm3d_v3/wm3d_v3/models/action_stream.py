"""Action stream (IDM) for v3."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ActionConfig:
    T: int = 16
    P: int = 64
    D: int = 2048
    hidden: int = 896
    n_layers: int = 10
    n_heads: int = 14
    k: int = 8
    z_dim: int = 192
    mlp_mult: int = 4
    dropout: float = 0.0
    cond_dim: int = 2048


class ActionStream(nn.Module):
    def __init__(self, cfg: ActionConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.D, cfg.hidden)
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.hidden)
        self.frame_pos = nn.Parameter(torch.zeros(1, cfg.T, 1, cfg.hidden))
        self.patch_pos = nn.Parameter(torch.zeros(1, 1, cfg.P, cfg.hidden))
        nn.init.normal_(self.frame_pos, std=0.02)
        nn.init.normal_(self.patch_pos, std=0.02)
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

    def encode(self, s, c):
        B = s.size(0)
        x = self.in_proj(s) + self.frame_pos + self.patch_pos
        x = x.view(B, self.cfg.T * self.cfg.P, self.cfg.hidden)
        cond_tok = self.cond_proj(c).unsqueeze(1)
        return torch.cat([cond_tok, x], dim=1)

    def apply_blocks(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def decode(self, h):
        B = h.size(0)
        q = self.dec_q.expand(B, -1, -1)
        return self.z_head(self.decoder(q, h))

    def forward(self, s, c):
        h = self.encode(s, c)
        h = self.apply_blocks(h)
        return h, self.decode(h)
