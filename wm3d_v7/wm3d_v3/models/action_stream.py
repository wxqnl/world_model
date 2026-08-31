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
    action_cond_dim: int = 0


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
        if cfg.action_cond_dim > 0:
            self.action_cond_proj = nn.Sequential(
                nn.Linear(cfg.action_cond_dim, cfg.hidden),
                nn.GELU(),
                nn.Linear(cfg.hidden, cfg.hidden),
            )
            self.action_cond_pos = nn.Parameter(torch.zeros(1, cfg.k, cfg.hidden))
            nn.init.normal_(self.action_cond_pos, std=0.02)
        else:
            self.action_cond_proj = None
            self.register_parameter("action_cond_pos", None)

    def _action_tokens(self, action_cond, B: int, device: torch.device):
        if self.action_cond_proj is None:
            return None
        if action_cond is None:
            action_cond = torch.zeros(B, self.cfg.k, self.cfg.action_cond_dim, device=device)
        if action_cond.shape != (B, self.cfg.k, self.cfg.action_cond_dim):
            raise ValueError(
                f"action_cond must have shape {(B, self.cfg.k, self.cfg.action_cond_dim)}, "
                f"got {tuple(action_cond.shape)}"
            )
        action_cond = action_cond.to(device=device, dtype=self.action_cond_pos.dtype)
        return self.action_cond_proj(action_cond) + self.action_cond_pos

    def encode(self, s, c, action_cond=None):
        B = s.size(0)
        x = self.in_proj(s) + self.frame_pos + self.patch_pos
        x = x.view(B, self.cfg.T * self.cfg.P, self.cfg.hidden)
        cond_tok = self.cond_proj(c).unsqueeze(1)
        tokens = [cond_tok, x]
        action_tok = self._action_tokens(action_cond, B, s.device)
        if action_tok is not None:
            tokens.append(action_tok)
        return torch.cat(tokens, dim=1)

    def apply_blocks(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def decode(self, h, action_cond=None):
        B = h.size(0)
        q = self.dec_q.expand(B, -1, -1)
        action_tok = self._action_tokens(action_cond, B, h.device)
        if action_tok is not None:
            q = q + action_tok
        return self.z_head(self.decoder(q, h))

    def forward(self, s, c, action_cond=None):
        h = self.encode(s, c, action_cond=action_cond)
        h = self.apply_blocks(h)
        return h, self.decode(h, action_cond=action_cond)
