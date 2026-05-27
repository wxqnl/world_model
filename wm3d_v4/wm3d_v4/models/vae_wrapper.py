"""Frozen SD-1.5 VAE wrapper for 256x256 RGB ↔ 32x32x4 latent."""
from __future__ import annotations
import torch
import torch.nn as nn
from diffusers import AutoencoderKL


SD_VAE_SCALE = 0.18215  # canonical scaling


class VAEWrapper(nn.Module):
    def __init__(self, pretrained: str = "stabilityai/sd-vae-ft-mse"):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained)
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 3, 256, 256] in [0, 1]. Returns [N, 4, 32, 32] scaled."""
        x = x * 2.0 - 1.0
        z = self.vae.encode(x).latent_dist.sample()
        return z * SD_VAE_SCALE

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N, 4, 32, 32] scaled. Returns RGB [N, 3, 256, 256] in [0, 1]."""
        z = z / SD_VAE_SCALE
        x = self.vae.decode(z).sample
        return ((x + 1.0) * 0.5).clamp(0.0, 1.0)
