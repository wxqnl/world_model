"""Big PixelDecoder for v3 — token sequence -> RGB video, trained end-to-end."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PixelDecoderConfig:
    token_dim: int = 2048
    token_grid: int = 8
    out_hw: int = 256
    hidden: int = 768
    n_res: int = 2


class _ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, c), c); self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, c), c); self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(F.gelu(self.n2(h)))
        return x + h


class PixelDecoder(nn.Module):
    def __init__(self, cfg: PixelDecoderConfig | None = None):
        super().__init__()
        cfg = cfg or PixelDecoderConfig()
        self.cfg = cfg
        # n_ups = log2(out_hw / token_grid). 8→256: 5, 16→256: 4.
        n_ups = 0
        g = cfg.token_grid
        while g < cfg.out_hw:
            g *= 2; n_ups += 1
        assert g == cfg.out_hw, f"out_hw {cfg.out_hw} not power-of-2 multiple of token_grid {cfg.token_grid}"
        H = cfg.hidden
        # Channel schedule: stem=H, then halve every 2 ups, floor at H//8.
        # n_ups=5: [H, H, H//2, H//4, H//4, H//8] (legacy)
        # n_ups=4: [H, H, H//2, H//4, H//8]
        if n_ups == 5:
            chans = [H, H, H // 2, H // 4, H // 4, H // 8]
        elif n_ups == 4:
            chans = [H, H, H // 2, H // 4, H // 8]
        else:
            chans = [H] + [max(H >> i, H // 8) for i in range(n_ups)]
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.token_dim, H, 1),
            nn.GroupNorm(min(8, H), H), nn.GELU(),
        )
        ups = []
        for i in range(n_ups):
            in_c, out_c = chans[i], chans[i + 1]
            blk = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                    nn.GroupNorm(min(8, out_c), out_c), nn.GELU()]
            for _ in range(cfg.n_res):
                blk.append(_ResBlock(out_c))
            ups.append(nn.Sequential(*blk))
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(chans[-1], 3, 1)

    def forward(self, tok):
        B, T, P, D = tok.shape
        G = self.cfg.token_grid
        x = tok.reshape(B * T, G * G, D).transpose(1, 2).reshape(B * T, D, G, G)
        x = self.stem(x)
        for up in self.ups:
            x = up(x)
        x = torch.sigmoid(self.head(x))
        return x.view(B, T, 3, self.cfg.out_hw, self.cfg.out_hw)
