"""Auxiliary inverse-dynamics head used only during VLA training.

Takes the last input frame's pooled VGGT tokens and the last predicted future
frame's tokens, concatenates them along the token dim, and predicts the action
chunk in standardized pose space + gripper logits.

The point is to give the dual stream a gradient that says "given where we
ended up, what action did we take", which the existing ActionStream cannot
get from self-attn over input frames alone.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class AuxIDMConfig:
    token_dim: int = 2048
    n_tokens: int = 64
    hidden: int = 1024
    n_layers: int = 3
    n_heads: int = 8
    k: int = 8
    mlp_mult: int = 4
    dropout: float = 0.0


class AuxIDM(nn.Module):
    def __init__(self, cfg: AuxIDMConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.token_dim * 2, cfg.hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.hidden))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.hidden, nhead=cfg.n_heads,
                dim_feedforward=cfg.hidden * cfg.mlp_mult,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=cfg.n_layers,
        )
        self.dec_q = nn.Parameter(torch.zeros(1, cfg.k, cfg.hidden))
        nn.init.normal_(self.dec_q, std=0.02)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.hidden, nhead=cfg.n_heads,
                dim_feedforward=cfg.hidden * cfg.mlp_mult,
                dropout=cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            ),
            num_layers=1,
        )
        self.norm = nn.LayerNorm(cfg.hidden)
        self.pose_head = nn.Linear(cfg.hidden, 6)
        self.grip_head = nn.Linear(cfg.hidden, 1)

    def forward(self, s_last: torch.Tensor, s_fut_last: torch.Tensor) -> dict:
        # s_last, s_fut_last: [B, n_tokens, token_dim]
        x = torch.cat([s_last, s_fut_last], dim=-1)         # [B, P, 2D]
        x = self.in_proj(x) + self.pos_emb                   # [B, P, H]
        h = self.enc(x)                                      # [B, P, H]
        B = h.size(0)
        q = self.dec_q.expand(B, -1, -1)
        z = self.norm(self.dec(q, h))                        # [B, k, H]
        return {
            "aux_pose_norm": self.pose_head(z),
            "aux_grip": self.grip_head(z).squeeze(-1),
        }
