"""VGGT encoder wrapper used by the OXE cache builders.

The training code expects VGGT tokens pooled to either 8x8 (64 tokens) or
16x16 (256 tokens), plus optional full-resolution 224x224 depth maps.
"""
from __future__ import annotations

import inspect
import math
import os
import sys
from pathlib import Path

import torch
from typing import Any
import torch.nn.functional as F


def _ensure_local_vggt_on_path() -> Path:
    root = Path(
        os.environ.get(
            "WM3D_VGGT_SOURCE_ROOT",
            "/data/world_model_workspace/world_model/vggt",
        )
    ).resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name != "vggt" and not name.startswith("vggt."):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path is None:
            continue
        try:
            Path(module_path).resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"preloaded VGGT module is outside the registered source tree: "
                f"{name}={module_path}"
            ) from exc
    sys.path[:] = [entry for entry in sys.path if entry != str(root)]
    sys.path.insert(0, str(root))
    return root


class VGGTEncoder(torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "facebook/VGGT-1B",
        token_grid: int = 8,
        return_depth: bool = False,
        return_depth_conf: bool = False,
        return_geom_extra: bool = False,
        dtype: torch.dtype | None = None,
        model_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        source_root = _ensure_local_vggt_on_path()
        from huggingface_hub import snapshot_download
        from vggt.models.vggt import VGGT

        source_file = Path(inspect.getsourcefile(VGGT) or "").resolve(strict=True)
        try:
            source_file.relative_to(source_root)
        except ValueError as exc:
            raise RuntimeError(
                f"VGGT class resolved outside registered source tree: {source_file}"
            ) from exc
        if model_revision is None:
            raise ValueError("model_revision is required for VGGTEncoder")
        bundled_snapshot = os.environ.get("WM3D_VGGT_MODEL_SNAPSHOT")
        if bundled_snapshot:
            snapshot_path = Path(bundled_snapshot).resolve(strict=True)
        else:
            snapshot_path = Path(
                snapshot_download(
                    repo_id=model_name,
                    revision=model_revision,
                    local_files_only=local_files_only,
                )
            ).resolve(strict=True)
        if snapshot_path.name != str(model_revision):
            raise RuntimeError(
                f"VGGT snapshot revision mismatch: {snapshot_path.name} != {model_revision}"
            )

        self.device = torch.device(device)
        self.token_grid = int(token_grid)
        self.return_depth = bool(return_depth)
        self.return_depth_conf = bool(return_depth_conf)
        self.return_geom_extra = bool(return_geom_extra)
        self.model_name = str(model_name)
        self.model_revision = str(model_revision)
        self.model_resolved_revision = snapshot_path.name
        self.model_snapshot_path = str(snapshot_path)
        self.vggt_source_root = str(source_root)
        self.vggt_source_file = str(source_file)
        self.local_files_only = bool(local_files_only)
        if dtype is None:
            major = torch.cuda.get_device_capability(self.device)[0] if self.device.type == "cuda" else 0
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.dtype = dtype
        self.model = VGGT.from_pretrained(
            str(snapshot_path), local_files_only=True
        ).to(self.device).eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        """Encode images.

        Args:
            images: [B, T, 3, H, W] or [T, 3, H, W], float in [0, 1].

        Returns:
            pooled: [B, T, token_grid*token_grid, 2048] fp16
            depth: [B, T, H, W] fp16, only when return_depth=True and a depth head exists
            depth_conf/world_points/world_points_conf/pose_enc: optional VGGT geometry extras
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
        out: dict[str, Any] = {"pooled": pooled}
        missing: list[str] = []

        need_depth_head = self.return_depth or self.return_depth_conf or self.return_geom_extra
        if need_depth_head:
            depth_head = getattr(self.model, "depth_head", None)
            if depth_head is None:
                missing.append("depth_head")
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    depth, depth_conf = depth_head(aggregated_tokens, images=images, patch_start_idx=patch_start_idx)
                if self.return_depth:
                    out["depth"] = depth.squeeze(-1).to(torch.float16)
                if self.return_depth_conf or self.return_geom_extra:
                    out["depth_conf"] = depth_conf.to(torch.float16)

        if self.return_geom_extra:
            camera_head = getattr(self.model, "camera_head", None)
            if camera_head is None:
                missing.append("camera_head")
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    out["pose_enc"] = camera_head(aggregated_tokens)[-1].to(torch.float16)
            point_head = getattr(self.model, "point_head", None)
            if point_head is None:
                missing.append("point_head")
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    world_points, world_points_conf = point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                out["world_points"] = world_points.to(torch.float16)
                out["world_points_conf"] = world_points_conf.to(torch.float16)
        if missing:
            out["geom_extra_missing"] = missing
        return out

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b, t, n, d = patch_tokens.shape
        grid = int(math.isqrt(n))
        if grid <= 0 or grid * grid != n:
            raise ValueError(f"cannot infer square token grid from {n} patch tokens")
        x = patch_tokens[:, :, : grid * grid, :].reshape(b * t, grid, grid, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.adaptive_avg_pool2d(x.float(), (self.token_grid, self.token_grid))
        x = x.permute(0, 2, 3, 1).reshape(b, t, self.token_grid * self.token_grid, d)
        return x
