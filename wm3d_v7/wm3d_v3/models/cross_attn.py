from __future__ import annotations
import torch
import torch.nn as nn


class BidirectionalCrossAttn(nn.Module):
    def __init__(self, d_state, d_action, n_heads=16, dropout=0.0):
        super().__init__()
        self.s_norm = nn.LayerNorm(d_state)
        self.a_norm = nn.LayerNorm(d_action)
        self.s_to_a = nn.MultiheadAttention(d_state, n_heads, dropout=dropout,
                                            kdim=d_action, vdim=d_action,
                                            batch_first=True)
        self.a_to_s = nn.MultiheadAttention(d_action, n_heads, dropout=dropout,
                                            kdim=d_state, vdim=d_state,
                                            batch_first=True)
        self.s_proj = nn.Linear(d_state, d_state)
        self.a_proj = nn.Linear(d_action, d_action)

    def forward(self, s, a):
        sn, an = self.s_norm(s), self.a_norm(a)
        s_upd, _ = self.s_to_a(sn, an, an, need_weights=False)
        a_upd, _ = self.a_to_s(an, sn, sn, need_weights=False)
        return s + self.s_proj(s_upd), a + self.a_proj(a_upd)
