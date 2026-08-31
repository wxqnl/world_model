"""Qwen3-VL embedding wrapper for per-episode task embeddings."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


class QwenVLEmbed:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_path, local_only = self._resolve_model_path(model_name)
        self.model = SentenceTransformer(
            model_path,
            device=self.device,
            cache_folder=os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE"),
            trust_remote_code=True,
            local_files_only=local_only,
            model_kwargs={"torch_dtype": dtype if self.device.startswith("cuda") else torch.float32},
        )
        self.model.eval()

    @staticmethod
    def _resolve_model_path(model_name: str) -> tuple[str, bool]:
        if Path(model_name).exists():
            return model_name, True

        explicit = os.environ.get("QWEN3_VL_EMBEDDING_PATH")
        if explicit and Path(explicit).exists():
            return explicit, True

        cache_roots: list[Path] = []
        for key in ("HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
            value = os.environ.get(key)
            if value:
                cache_roots.append(Path(value))
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_roots.append(Path(hf_home) / "hub")
        cache_roots.append(Path("/data/Minko/_network_cache_and_tls/hf-cache/hub"))
        cache_roots.append(Path("/root/.cache/huggingface/hub"))

        if model_name == "Qwen/Qwen3-VL-Embedding-2B":
            for cache_root in cache_roots:
                snapshots = cache_root / "models--Qwen--Qwen3-VL-Embedding-2B" / "snapshots"
                if not snapshots.exists():
                    continue
                candidates = [p for p in snapshots.iterdir() if p.is_dir() and (p / "modules.json").exists()]
                if candidates:
                    newest = max(candidates, key=lambda p: p.stat().st_mtime)
                    return str(newest), True

        offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        return model_name, offline

    @torch.inference_mode()
    def embed(self, text: str, image=None) -> torch.Tensor:
        payload = {"text": text or "robot manipulation"}
        if image is not None:
            payload["image"] = image
        emb = self.model.encode([payload], convert_to_tensor=True)
        if isinstance(emb, np.ndarray):
            arr = torch.from_numpy(emb)
        else:
            arr = torch.as_tensor(emb)
        arr = arr.detach().float().cpu()
        if arr.ndim == 2:
            arr = arr[0]
        if arr.numel() != 2048:
            raise ValueError(f"expected 2048-D Qwen embedding, got {tuple(arr.shape)}")
        return arr
