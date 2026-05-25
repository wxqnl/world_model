"""Geometry decoder: pooled VGGT tokens -> depth (224x224) + point + pose."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeomDecoder(nn.Module):
    """[B, T, 64, 2048] -> depth [B, T, 224, 224], point [B, T, 224, 224, 3], pose [B, T, 9]"""

    def __init__(self, token_dim=2048, token_grid=8, hidden=384):
        super().__init__()
        self.token_grid = token_grid
        H = hidden
        # 8 -> 16 -> 32 -> 64 -> 128 -> 256 (then crop to 224)
        self.stem = nn.Sequential(nn.Conv2d(token_dim, H, 1), nn.GroupNorm(8, H), nn.GELU())
        def up(c_in, c_out):
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.GELU(),
            )
        self.up1 = up(H, H)
        self.up2 = up(H, H // 2)
        self.up3 = up(H // 2, H // 4)
        self.up4 = up(H // 4, H // 8)
        self.up5 = up(H // 8, H // 8)
        self.depth_head = nn.Conv2d(H // 8, 1, 1)
        self.point_head = nn.Conv2d(H // 8, 3, 1)
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, 9),
        )

    def forward(self, tok: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, P, D = tok.shape
        G = self.token_grid
        x = tok.reshape(B * T, G * G, D).transpose(1, 2).reshape(B * T, D, G, G)
        x = self.stem(x)
        pose = self.pose_head(x).view(B, T, 9)
        x = self.up1(x); x = self.up2(x); x = self.up3(x); x = self.up4(x); x = self.up5(x)
        x = x[:, :, 16:240, 16:240]  # crop 256 -> 224
        depth = F.softplus(self.depth_head(x)).squeeze(1).view(B, T, 224, 224)
        point = self.point_head(x).permute(0, 2, 3, 1).contiguous().view(B, T, 224, 224, 3)
        return {"depth": depth, "point": point, "pose": pose}
