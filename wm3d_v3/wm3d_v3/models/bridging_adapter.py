"""Bridging adapter: depth (224x224) -> Cosmos depth input format.

Cosmos-Transfer1 expects depth as a 3-channel video at 704x480 normalized to [0,1] or [-1,1].
This adapter learns the per-domain normalization/edge enhancement to make our model's
predicted depth feed cleanly into Cosmos's depth ControlNet.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BridgingAdapter(nn.Module):
    """[B, T, 224, 224] depth (median-normalized) -> [B, T, 3, 480, 704] cosmos-style."""

    def __init__(self, hidden: int = 96):
        super().__init__()
        H = hidden
        self.refine = nn.Sequential(
            nn.Conv2d(1, H, 3, padding=1), nn.GroupNorm(min(8, H), H), nn.GELU(),
            nn.Conv2d(H, H, 3, padding=1), nn.GroupNorm(min(8, H), H), nn.GELU(),
            nn.Conv2d(H, H, 3, padding=1), nn.GroupNorm(min(8, H), H), nn.GELU(),
            nn.Conv2d(H, 1, 1),
        )

    def forward(self, depth: torch.Tensor, out_hw: tuple[int, int] = (480, 704)) -> torch.Tensor:
        B, T, H_in, W_in = depth.shape
        x = depth.view(B * T, 1, H_in, W_in)
        # Normalize per-frame to unit median (matches what Cosmos expects)
        med = x.flatten(1).median(dim=1).values.clamp_min(1e-6).view(-1, 1, 1, 1)
        x = x / med
        # Learn a residual refinement
        x = x + 0.1 * self.refine(x)
        x = x.clamp(0, 3.0) / 3.0   # squash to [0, 1]
        # Resize to cosmos input
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False, antialias=True)
        x = x.repeat(1, 3, 1, 1)  # 1 -> 3 channels
        return x.view(B, T, 3, *out_hw)
