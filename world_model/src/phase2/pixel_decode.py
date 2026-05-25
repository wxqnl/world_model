"""Tiny learned depth decoder for Phase 2 pixel-space feasibility eval.

Problem
-------
Phase 1/2 operates on *pooled* VGGT tokens (8x8 grid, D=2048). VGGT's own
heads expect the full multi-layer aggregator output (1374 tokens per frame,
all layers), which we do **not** cache. We therefore cannot run VGGT's
depth_head on generated tokens directly.

For the feasibility check we only need to answer: "do generated pooled tokens
look like real pooled tokens when projected to a geometric observable?" A
small decoder learned on real (pooled token -> cached depth) pairs is
sufficient:

    TokenDepthDecoder : [T, 64, 2048] -> [T, 64, 64]

Trained on the Phase 1 cache train split (24 clips x 128 frames), applied to:
    1. real val tokens         (sanity; sets the ceiling)
    2. flow_only generations   (uncoupled baseline)
    3. flow_coupled generations (coupling-aware baseline)

This is feasibility-grade — not a claim about VGGT's decoded depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderConfig:
    token_dim: int = 2048
    token_grid: int = 8        # pooled grid side
    depth_hw: int = 64         # cached depth is 64x64
    hidden: int = 128


class TokenDepthDecoder(nn.Module):
    """Per-frame decoder: 8x8x2048 pooled tokens -> 64x64 depth map.

    Architecture (deliberately small, ~500k params):
        (B, 2048, 8, 8)
          -> 1x1 conv to hidden
          -> ConvTranspose 2x (hidden -> hidden)         -> 16x16
          -> ConvTranspose 2x (hidden -> hidden // 2)    -> 32x32
          -> ConvTranspose 2x (hidden // 2 -> hidden // 4) -> 64x64
          -> 1x1 conv to 1 channel, softplus (depth >= 0)
    """

    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        cfg = cfg or DecoderConfig()
        self.cfg = cfg
        H = cfg.hidden
        self.stem = nn.Conv2d(cfg.token_dim, H, kernel_size=1)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(H, H, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(H, H, 3, padding=1), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(H, H // 2, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(H // 2, H // 2, 3, padding=1), nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(H // 2, H // 4, kernel_size=4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(H // 4, H // 4, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Conv2d(H // 4, 1, kernel_size=1)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        """tok: [..., P, D] where P = G*G. Returns [..., depth_hw, depth_hw]."""
        G = self.cfg.token_grid
        lead = tok.shape[:-2]
        x = tok.reshape(-1, G * G, self.cfg.token_dim)
        x = x.transpose(1, 2).reshape(-1, self.cfg.token_dim, G, G)
        x = self.stem(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.head(x)                        # [N, 1, 64, 64]
        x = F.softplus(x).squeeze(1)            # depth >= 0
        return x.reshape(*lead, self.cfg.depth_hw, self.cfg.depth_hw)


def normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    """Rescale a depth map to unit median. Output is a shape-only representation;
    used as the training target so we ignore affine-depth scale ambiguity.
    depth: [..., H, W]."""
    med = depth.flatten(-2).median(dim=-1).values.clamp_min(1e-6)
    return depth / med.view(*med.shape, 1, 1)


def decoder_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between decoder output and unit-median-normalized target depth.
    Decoder is trained to output normalized depth directly; no alignment
    gymnastics at train time (avoids gradient-degenerate global-scale axis)."""
    return (pred - normalize_depth(target)).abs().mean()
