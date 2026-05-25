"""Thin wrapper around facebookresearch/vggt for Phase 0 diagnostics.

Three public methods:
  encode(images)           -> {aggregated_tokens, ps_idx}
  decode_geometry(...)     -> {depth, depth_conf, point_map, point_conf,
                               camera_extrinsic, camera_intrinsic}
  full_inference(images)   -> union of the two above

All inference runs under torch.cuda.amp.autocast(bfloat16) on Ampere+.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VGGTConfig:
    name: str = "facebook/VGGT-1B"
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class VGGTWrapper:
    def __init__(self, cfg: VGGTConfig | None = None):
        self.cfg = cfg or VGGTConfig()
        from vggt.models.vggt import VGGT  # local import so module loads without vggt installed

        dev_cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        if dev_cap < 8 and self.cfg.dtype == torch.bfloat16:
            self.cfg.dtype = torch.float16
        self.model = VGGT.from_pretrained(self.cfg.name).to(self.cfg.device).eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, Any]:
        """images: [N, 3, H, W] float in [0,1]. Adds batch dim internally."""
        images = self._prep(images)
        with torch.cuda.amp.autocast(dtype=self.cfg.dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(images)
        return {"aggregated_tokens": aggregated_tokens_list, "ps_idx": ps_idx, "images": images}

    @torch.no_grad()
    def decode_geometry(
        self,
        aggregated_tokens: list[torch.Tensor],
        ps_idx: torch.Tensor,
        images: torch.Tensor,
    ) -> dict[str, Any]:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        with torch.cuda.amp.autocast(dtype=self.cfg.dtype):
            pose_enc = self.model.camera_head(aggregated_tokens)[-1]
            depth, depth_conf = self.model.depth_head(aggregated_tokens, images, ps_idx)
            point, point_conf = self.model.point_head(aggregated_tokens, images, ps_idx)
        extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "point_map": point,
            "point_conf": point_conf,
            "camera_extrinsic": extri,
            "camera_intrinsic": intri,
            "pose_enc": pose_enc,
        }

    @torch.no_grad()
    def full_inference(self, images: torch.Tensor) -> dict[str, Any]:
        enc = self.encode(images)
        geo = self.decode_geometry(enc["aggregated_tokens"], enc["ps_idx"], enc["images"])
        return {**enc, **geo}

    def _prep(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        return images.to(self.cfg.device, non_blocking=True)
