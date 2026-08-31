"""Small CNN: 224x224 depth -> 8x8 grid of feature tokens.

Used by Stage C to feed VGGT depth into IDMStream as auxiliary per-patch features
alongside the pooled VGGT tokens. Outputs match the 8x8 patch grid that IDMStream
already expects (P=64 patches).
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DepthEncoderConfig:
    in_size: int = 224
    out_grid: int = 8
    hidden_d: int = 256
    base_ch: int = 16


class DepthEncoder(nn.Module):
    def __init__(self, cfg: DepthEncoderConfig):
        super().__init__()
        self.cfg = cfg
        c1, c2, c3, c4 = cfg.base_ch, cfg.base_ch * 2, cfg.base_ch * 4, cfg.base_ch * 8
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, cout), cout),
                nn.GELU(),
            )
        self.conv = nn.Sequential(
            block(1, c1),       # 224 -> 112
            block(c1, c2),      # 112 -> 56
            block(c2, c3),      # 56 -> 28
            block(c3, c4),      # 28 -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(cfg.out_grid)   # 14 -> 8
        self.head = nn.Linear(c4, cfg.hidden_d)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: [B, F, H, W] -> [B, F, P, hidden_d] with P = out_grid**2."""
        B, F_, H, W = depth.shape
        x = depth.reshape(B * F_, 1, H, W)
        # Normalize depth per-frame to be scale-invariant (median to 1)
        med = x.reshape(B * F_, -1).median(dim=-1, keepdim=True).values.clamp_min(1e-6)
        x = x / med.view(B * F_, 1, 1, 1)
        x = self.conv(x)                             # [B*F, c4, 14, 14]
        x = self.pool(x)                             # [B*F, c4, G, G]
        c4 = x.size(1); G = x.size(2)
        x = x.permute(0, 2, 3, 1).reshape(B * F_, G * G, c4)
        x = self.head(x)                             # [B*F, P, hidden_d]
        return x.reshape(B, F_, G * G, self.cfg.hidden_d)
