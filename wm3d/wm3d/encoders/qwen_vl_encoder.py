"""Qwen3-VL embedding wrapper for per-episode task embeddings."""
from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import torch


class QwenVLEmbed:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        model_revision: str | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_revision is None or re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
            raise RuntimeError(
                "Qwen embedding model_revision must be a pinned 40-hex commit SHA"
            )
        model_path = self._resolve_model_path(model_name, model_revision)
        resolved = Path(model_path).resolve(strict=True)
        if resolved.name != model_revision:
            raise RuntimeError(
                f"Qwen embedding snapshot revision mismatch: "
                f"{resolved.name} != {model_revision}"
            )
        self.model = SentenceTransformer(
            str(resolved),
            device=self.device,
            cache_folder=os.environ.get("HF_HUB_CACHE") or os.environ.get("TRANSFORMERS_CACHE"),
            trust_remote_code=True,
            # Resolution happens once above with the exact immutable revision;
            # SentenceTransformer must never perform a second floating lookup.
            local_files_only=True,
            model_kwargs={"torch_dtype": dtype if self.device.startswith("cuda") else torch.float32},
        )
        self.model.eval()
        self.model_name = str(model_name)
        self.model_revision = str(model_revision)
        self.model_snapshot_path = str(resolved)

    @staticmethod
    def _resolve_model_path(model_name: str, model_revision: str) -> str:
        from huggingface_hub import snapshot_download

        candidate = Path(model_name)
        if candidate.exists():
            return str(candidate.resolve(strict=True))

        explicit = os.environ.get("QWEN3_VL_EMBEDDING_PATH")
        if explicit and Path(explicit).exists():
            return str(Path(explicit).resolve(strict=True))

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
                exact = (
                    cache_root
                    / "models--Qwen--Qwen3-VL-Embedding-2B"
                    / "snapshots"
                    / model_revision
                )
                if exact.is_dir() and (exact / "modules.json").is_file():
                    return str(exact.resolve(strict=True))

        offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        return str(
            Path(
                snapshot_download(
                    repo_id=model_name,
                    revision=model_revision,
                    local_files_only=offline,
                )
            ).resolve(strict=True)
        )

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
