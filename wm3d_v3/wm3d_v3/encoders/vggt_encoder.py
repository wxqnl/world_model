"""VGGT encoder wrapper used by the OXE cache builders.

The training code expects VGGT tokens pooled to either 8x8 (64 tokens) or
16x16 (256 tokens), plus optional full-resolution 224x224 depth maps.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _ensure_local_vggt_on_path() -> None:
    root = Path("/data/world_model_workspace/world_model/vggt")
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


class VGGTEncoder(torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "facebook/VGGT-1B",
        token_grid: int = 8,
        return_depth: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _ensure_local_vggt_on_path()
        from vggt.models.vggt import VGGT

        self.device = torch.device(device)
        self.token_grid = int(token_grid)
        self.return_depth = bool(return_depth)
        if dtype is None:
            major = torch.cuda.get_device_capability(self.device)[0] if self.device.type == "cuda" else 0
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.dtype = dtype
        self.model = VGGT.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode images.

        Args:
            images: [B, T, 3, H, W] or [T, 3, H, W], float in [0, 1].

        Returns:
            pooled: [B, T, token_grid*token_grid, 2048] fp16
            depth: [B, T, H, W] fp16, only when return_depth=True
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(self.device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=self.device.type == "cuda"):
            aggregated_tokens, patch_start_idx = self.model.aggregator(images)

        tokens = aggregated_tokens[-1]
        patch_start = int(patch_start_idx)
        patch_tokens = tokens[:, :, patch_start:, :]
        pooled = self._pool_patch_tokens(patch_tokens).to(torch.float16)
        out = {"pooled": pooled}

        if self.return_depth:
            with torch.amp.autocast("cuda", enabled=False):
                depth, _ = self.model.depth_head(aggregated_tokens, images=images, patch_start_idx=patch_start_idx)
            out["depth"] = depth.squeeze(-1).to(torch.float16)
        return out

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b, t, n, d = patch_tokens.shape
        grid = int(math.isqrt(n))
        if grid * grid > n or grid <= 0:
            raise ValueError(f"cannot infer square token grid from {n} patch tokens")
        x = patch_tokens[:, :, : grid * grid, :].reshape(b * t, grid, grid, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.adaptive_avg_pool2d(x.float(), (self.token_grid, self.token_grid))
        x = x.permute(0, 2, 3, 1).reshape(b, t, self.token_grid * self.token_grid, d)
        return x
