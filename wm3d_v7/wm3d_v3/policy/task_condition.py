"""Task embedding provider for online policy adapters."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed


class TaskConditionProvider:
    """Encode task text/image into the 2048-D Qwen task vector used by WM3D.

    The provider is intentionally small and cache-friendly. Benchmark adapters
    can call it at episode reset and keep the returned vector fixed for the
    episode.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
        allow_zero_fallback: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_name = model_name
        self.device = device
        self.allow_zero_fallback = allow_zero_fallback
        self._encoder: QwenVLEmbed | None = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def cache_key(text: str) -> str:
        normalized = (text or "robot manipulation").strip().encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()[:24]

    def _load_encoder(self) -> QwenVLEmbed:
        if self._encoder is None:
            self._encoder = QwenVLEmbed(model_name=self.model_name, device=self.device)
        return self._encoder

    def embed(self, text: str, image=None) -> torch.Tensor:
        key = self.cache_key(text)
        cache_path = self.cache_dir / f"{key}.npy" if self.cache_dir is not None and image is None else None
        if cache_path is not None and cache_path.exists():
            return torch.from_numpy(np.load(cache_path)).float()

        try:
            emb = self._load_encoder().embed(text, image=image).float()
        except Exception:
            if not self.allow_zero_fallback:
                raise
            emb = torch.zeros(2048, dtype=torch.float32)
        if emb.numel() != 2048:
            raise ValueError(f"expected 2048-D task embedding, got {tuple(emb.shape)}")
        emb = emb.reshape(2048).cpu()
        if cache_path is not None:
            np.save(cache_path, emb.numpy().astype(np.float16))
        return emb
