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


def _patch_tokens_from_cached_layer(
    aggregated_tokens: list[torch.Tensor | None],
    *,
    patch_start_idx: int,
    layer: int,
    role: str,
) -> torch.Tensor:
    resolved_layer = len(aggregated_tokens) - 1 if int(layer) == -1 else int(layer)
    if not 0 <= resolved_layer < len(aggregated_tokens):
        raise RuntimeError(
            f"VGGT {role} feature layer {resolved_layer} is outside "
            f"[0, {len(aggregated_tokens) - 1}]"
        )
    tokens = aggregated_tokens[resolved_layer]
    if tokens is None:
        raise RuntimeError(
            f"VGGT {role} feature layer {resolved_layer} is not cached by the backbone"
        )
    if tokens.ndim != 4 or not 0 <= int(patch_start_idx) < tokens.shape[-2]:
        raise RuntimeError(
            f"VGGT {role} feature layout is invalid: {tuple(tokens.shape)}"
        )
    return tokens[:, :, int(patch_start_idx) :, :]


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
    # The source tree is part of the sealed asset bundle.  Importing it must
    # never add __pycache__ files and invalidate the exact file-set receipt.
    sys.dont_write_bytecode = True
    sys.path[:] = [entry for entry in sys.path if entry != str(root)]
    sys.path.insert(0, str(root))
    return root


class VGGTEncoder(torch.nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "facebook/VGGT-1B",
        token_grid: int = 8,
        appearance_token_grid: int | None = None,
        appearance_feature_layer: int | None = None,
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
        self.appearance_token_grid = (
            0 if appearance_token_grid is None else int(appearance_token_grid)
        )
        self.appearance_feature_layer = (
            -1 if appearance_feature_layer is None else int(appearance_feature_layer)
        )
        if self.appearance_token_grid < 0:
            raise ValueError("appearance token grid cannot be negative")
        if self.appearance_feature_layer < -1:
            raise ValueError("appearance feature layer must be -1 or non-negative")
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
            major = (
                torch.cuda.get_device_capability(self.device)[0]
                if self.device.type == "cuda"
                else 0
            )
            dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.dtype = dtype
        self.model = (
            VGGT.from_pretrained(str(snapshot_path), local_files_only=True)
            .to(self.device)
            .eval()
        )

    @staticmethod
    def _selected_rows(
        mask: torch.Tensor | None,
        *,
        batch_rows: int,
        device: torch.device,
        role: str,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        if tuple(mask.shape) != (batch_rows,):
            raise ValueError(
                f"VGGT {role} row mask must be [{batch_rows}], got {tuple(mask.shape)}"
            )
        indices = torch.nonzero(
            mask.to(device=device, dtype=torch.bool), as_tuple=False
        ).flatten()
        return indices

    @staticmethod
    def _scatter_selected_rows(
        value: torch.Tensor,
        indices: torch.Tensor | None,
        batch_rows: int,
    ) -> torch.Tensor:
        if indices is None:
            return value
        result = value.new_zeros((batch_rows, *value.shape[1:]))
        result.index_copy_(0, indices, value)
        return result

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        return self.forward_selective(images)

    @torch.inference_mode()
    def forward_selective(
        self,
        images: torch.Tensor,
        *,
        geometry_batch_mask: torch.Tensor | None = None,
        appearance_batch_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
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
        batch_rows = int(images.shape[0])
        geometry_rows = self._selected_rows(
            geometry_batch_mask,
            batch_rows=batch_rows,
            device=images.device,
            role="geometry",
        )
        appearance_rows = self._selected_rows(
            appearance_batch_mask,
            batch_rows=batch_rows,
            device=images.device,
            role="appearance",
        )
        geometry_is_empty = (
            geometry_rows is not None and not int(geometry_rows.numel())
        )
        appearance_is_empty = appearance_rows is not None and not int(appearance_rows.numel())
        with torch.amp.autocast(
            "cuda", dtype=self.dtype, enabled=self.device.type == "cuda"
        ):
            aggregated_tokens, patch_start_idx = self.model.aggregator(images)

        patch_start = int(patch_start_idx)
        geometry_head_tokens = aggregated_tokens
        geometry_head_images = images
        if geometry_rows is not None:
            geometry_head_tokens = [
                None if value is None else value.index_select(0, geometry_rows)
                for value in aggregated_tokens
            ]
            geometry_head_images = images.index_select(0, geometry_rows)
        geometry_patch_tokens = _patch_tokens_from_cached_layer(
            aggregated_tokens,
            patch_start_idx=patch_start,
            layer=-1,
            role="geometry",
        )
        pooled = self._pool_patch_tokens(
            geometry_patch_tokens, self.token_grid
        ).to(torch.float16)
        out: dict[str, Any] = {"pooled": pooled}
        if self.appearance_token_grid:
            appearance_patch_tokens = _patch_tokens_from_cached_layer(
                aggregated_tokens,
                patch_start_idx=patch_start,
                layer=self.appearance_feature_layer,
                role="appearance",
            )
            if appearance_patch_tokens.shape[-1] != geometry_patch_tokens.shape[-1]:
                raise RuntimeError(
                    "VGGT geometry/appearance feature dimensions differ: "
                    f"{geometry_patch_tokens.shape[-1]} != "
                    f"{appearance_patch_tokens.shape[-1]}"
                )
            if appearance_is_empty:
                out["appearance_pooled"] = appearance_patch_tokens.new_zeros(
                    (
                        batch_rows,
                        appearance_patch_tokens.shape[1],
                        self.appearance_token_grid**2,
                        appearance_patch_tokens.shape[-1],
                    ),
                    dtype=torch.float16,
                )
            else:
                selected_appearance = appearance_patch_tokens
                if appearance_rows is not None:
                    selected_appearance = appearance_patch_tokens.index_select(
                        0, appearance_rows
                    )
                appearance_pooled = self._pool_patch_tokens(
                    selected_appearance, self.appearance_token_grid
                ).to(torch.float16)
                out["appearance_pooled"] = self._scatter_selected_rows(
                    appearance_pooled, appearance_rows, batch_rows
                )
        missing: list[str] = []

        need_depth_head = (
            self.return_depth or self.return_depth_conf or self.return_geom_extra
        )
        if need_depth_head:
            depth_head = getattr(self.model, "depth_head", None)
            if depth_head is None:
                missing.append("depth_head")
            elif geometry_is_empty:
                geometry_shape = (
                    batch_rows,
                    images.shape[1],
                    images.shape[-2],
                    images.shape[-1],
                )
                if self.return_depth:
                    out["depth"] = images.new_zeros(
                        geometry_shape, dtype=torch.float16
                    )
                if self.return_depth_conf or self.return_geom_extra:
                    out["depth_conf"] = images.new_zeros(
                        geometry_shape, dtype=torch.float16
                    )
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    depth, depth_conf = depth_head(
                        geometry_head_tokens,
                        images=geometry_head_images,
                        patch_start_idx=patch_start_idx,
                    )
                if self.return_depth:
                    selected_depth = depth.squeeze(-1).to(torch.float16)
                    out["depth"] = self._scatter_selected_rows(
                        selected_depth, geometry_rows, batch_rows
                    )
                if self.return_depth_conf or self.return_geom_extra:
                    selected_depth_conf = depth_conf.to(torch.float16)
                    out["depth_conf"] = self._scatter_selected_rows(
                        selected_depth_conf, geometry_rows, batch_rows
                    )

        if self.return_geom_extra:
            camera_head = getattr(self.model, "camera_head", None)
            if camera_head is None:
                missing.append("camera_head")
            elif geometry_is_empty:
                out["pose_enc"] = images.new_zeros(
                    (batch_rows, images.shape[1], 9),
                    dtype=torch.float16,
                )
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    selected_pose = camera_head(geometry_head_tokens)[-1].to(
                        torch.float16
                    )
                out["pose_enc"] = self._scatter_selected_rows(
                    selected_pose, geometry_rows, batch_rows
                )
            point_head = getattr(self.model, "point_head", None)
            if point_head is None:
                missing.append("point_head")
            elif geometry_is_empty:
                point_shape = (
                    batch_rows,
                    images.shape[1],
                    images.shape[-2],
                    images.shape[-1],
                )
                out["world_points"] = images.new_zeros(
                    (*point_shape, 3), dtype=torch.float16
                )
                out["world_points_conf"] = images.new_zeros(
                    point_shape, dtype=torch.float16
                )
            else:
                with torch.amp.autocast("cuda", enabled=False):
                    world_points, world_points_conf = point_head(
                        geometry_head_tokens,
                        images=geometry_head_images,
                        patch_start_idx=patch_start_idx,
                    )
                selected_points = world_points.to(torch.float16)
                selected_point_conf = world_points_conf.to(torch.float16)
                out["world_points"] = self._scatter_selected_rows(
                    selected_points, geometry_rows, batch_rows
                )
                out["world_points_conf"] = self._scatter_selected_rows(
                    selected_point_conf, geometry_rows, batch_rows
                )
        if missing:
            out["geom_extra_missing"] = missing
        return out

    @staticmethod
    def _pool_patch_tokens(
        patch_tokens: torch.Tensor, target_grid: int
    ) -> torch.Tensor:
        b, t, n, d = patch_tokens.shape
        grid = int(math.isqrt(n))
        if grid <= 0 or grid * grid != n:
            raise ValueError(f"cannot infer square token grid from {n} patch tokens")
        x = patch_tokens[:, :, : grid * grid, :].reshape(b * t, grid, grid, d)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.adaptive_avg_pool2d(x.float(), (target_grid, target_grid))
        x = x.permute(0, 2, 3, 1).reshape(b, t, target_grid * target_grid, d)
        return x
