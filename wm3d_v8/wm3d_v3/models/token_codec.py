"""Episode-shared 2048->384 token codec used only if its proof gate passes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokenCodecConfig:
    token_dim: int = 2048
    latent_dim: int = 384


class PCATokenCodec(nn.Module):
    """Fixed PCA projection plus symmetric int8 storage quantization."""

    def __init__(self, cfg: TokenCodecConfig | None = None):
        super().__init__()
        self.cfg = cfg or TokenCodecConfig()
        self.register_buffer("mean", torch.zeros(self.cfg.token_dim))
        self.register_buffer("components", torch.zeros(self.cfg.latent_dim, self.cfg.token_dim))
        self.register_buffer("fitted", torch.tensor(False))

    def set_basis(self, mean: torch.Tensor, components: torch.Tensor) -> None:
        if mean.shape != (self.cfg.token_dim,):
            raise ValueError(f"mean must be {(self.cfg.token_dim,)}")
        if components.shape != (self.cfg.latent_dim, self.cfg.token_dim):
            raise ValueError(f"components must be {(self.cfg.latent_dim, self.cfg.token_dim)}")
        self.mean.copy_(mean.float())
        self.components.copy_(components.float())
        self.fitted.fill_(True)

    def _require_fitted(self) -> None:
        if not bool(self.fitted):
            raise RuntimeError("token codec basis has not been fitted/loaded")

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        if tokens.shape[-1] != self.cfg.token_dim:
            raise ValueError(f"expected token dim {self.cfg.token_dim}")
        return F.linear(tokens.float() - self.mean, self.components)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        if latent.shape[-1] != self.cfg.latent_dim:
            raise ValueError(f"expected latent dim {self.cfg.latent_dim}")
        decoded = F.linear(latent.float(), self.components.t(), self.mean)
        return decoded.to(dtype=latent.dtype) if latent.is_floating_point() else decoded

    @staticmethod
    def quantize(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize with one scale per leading frame; preserves patch/channel shape."""
        if latent.ndim < 2:
            raise ValueError("latent must include frame and feature dimensions")
        reduce_dims = tuple(range(1, latent.ndim))
        scale = latent.float().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / 127.0
        codes = torch.round(latent.float() / scale).clamp(-127, 127).to(torch.int8)
        return codes, scale.to(torch.float16)

    @staticmethod
    def dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return codes.float() * scale.float()

    def quantized_reconstruction(self, tokens: torch.Tensor) -> torch.Tensor:
        latent = self.encode(tokens)
        codes, scale = self.quantize(latent)
        return self.decode(self.dequantize(codes, scale))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)
