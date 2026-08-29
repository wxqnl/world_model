"""Frozen training-only pixel correspondence targets for the RGB renderer."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FrozenBidirectionalRAFTConfig:
    source_root: str
    checkpoint: str
    input_size: int = 128
    iterations: int = 12
    output_grid: int = 32
    batch_chunk: int = 8
    flow_max_pixels: float = 128.0
    consistency_relative: float = 0.01
    consistency_absolute: float = 0.5


def _warp_flow_field(
    source: torch.Tensor,
    flow_pixels: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-warp a coarse field with flow in original RGB pixels."""

    if source.ndim != 4 or flow_pixels.ndim != 4:
        raise ValueError("flow warp tensors must be rank four")
    if source.shape[0] != flow_pixels.shape[0] or flow_pixels.shape[1] != 2:
        raise ValueError("flow warp batch/channel dimensions differ")
    if source.shape[-2:] != flow_pixels.shape[-2:]:
        flow_pixels = F.interpolate(
            flow_pixels.float(),
            size=source.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    height, width = source.shape[-2:]
    y = torch.linspace(-1.0, 1.0, height, device=source.device)
    x = torch.linspace(-1.0, 1.0, width, device=source.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)[None]
    displacement = torch.stack(
        (
            2.0 * flow_pixels[:, 0].float() / float(max(1, image_width - 1)),
            2.0 * flow_pixels[:, 1].float() / float(max(1, image_height - 1)),
        ),
        dim=-1,
    )
    sampling_grid = base_grid + displacement
    valid = (
        (sampling_grid[..., 0] >= -1.0)
        & (sampling_grid[..., 0] <= 1.0)
        & (sampling_grid[..., 1] >= -1.0)
        & (sampling_grid[..., 1] <= 1.0)
    )[:, None]
    warped = F.grid_sample(
        source.float(),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.to(dtype=source.dtype), valid


class FrozenBidirectionalRAFTRuntime:
    """Create target-to-context flow and visibility labels outside model state."""

    def __init__(self, cfg: FrozenBidirectionalRAFTConfig, device: torch.device):
        if cfg.input_size <= 0 or cfg.input_size % 8:
            raise ValueError("RAFT input size must be positive and divisible by 8")
        if cfg.output_grid <= 0 or cfg.iterations <= 0 or cfg.batch_chunk <= 0:
            raise ValueError("RAFT grid, iterations and chunk must be positive")
        if cfg.flow_max_pixels <= 0.0:
            raise ValueError("RAFT maximum flow must be positive")
        if cfg.consistency_relative < 0.0 or cfg.consistency_absolute < 0.0:
            raise ValueError("RAFT consistency thresholds cannot be negative")
        root = Path(cfg.source_root).resolve(strict=True)
        checkpoint = Path(cfg.checkpoint).resolve(strict=True)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        raft_module = importlib.import_module("core.raft")
        model = getattr(raft_module, "RAFT")(
            Namespace(
                small=False,
                mixed_precision=False,
                alternate_corr=False,
                dropout=0.0,
            )
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {
            str(name).removeprefix("module."): value for name, value in state.items()
        }
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.cfg = cfg
        self.device = device

    @staticmethod
    def _rgb(value: torch.Tensor) -> torch.Tensor:
        result = value.float()
        if float(result.detach().amax()) > 1.5:
            result = result / 255.0
        return result.clamp(0.0, 1.0)

    @torch.no_grad()
    def targets(
        self,
        context_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return [B,K,V,...] backward flow and disocclusion targets."""

        if context_rgb.ndim != 5 or context_rgb.shape[2] != 3:
            raise ValueError("RAFT context RGB must be [B,V,3,H,W]")
        if future_rgb.ndim != 6 or future_rgb.shape[3] != 3:
            raise ValueError("RAFT future RGB must be [B,K,V,3,H,W]")
        batch, horizon, views = future_rgb.shape[:3]
        if tuple(context_rgb.shape[:2]) != (batch, views):
            raise ValueError("RAFT context/future batch-view axes differ")
        height, width = future_rgb.shape[-2:]
        count = batch * horizon * views
        context = (
            self._rgb(context_rgb)[:, None]
            .expand(-1, horizon, -1, -1, -1, -1)
            .reshape(count, 3, height, width)
        )
        target = self._rgb(future_rgb).reshape(count, 3, height, width)
        size = int(self.cfg.input_size)
        context_small = F.interpolate(
            context, size=(size, size), mode="bilinear", align_corners=False
        )
        target_small = F.interpolate(
            target, size=(size, size), mode="bilinear", align_corners=False
        )
        first = torch.cat((target_small, context_small), dim=0).mul(255.0)
        second = torch.cat((context_small, target_small), dim=0).mul(255.0)
        flows: list[torch.Tensor] = []
        for start in range(0, int(first.shape[0]), self.cfg.batch_chunk):
            _, flow = self.model(
                first[start : start + self.cfg.batch_chunk],
                second[start : start + self.cfg.batch_chunk],
                iters=int(self.cfg.iterations),
                test_mode=True,
            )
            flows.append(flow.float())
        paired = torch.cat(flows, dim=0)
        paired[:, 0].mul_(float(width) / float(size))
        paired[:, 1].mul_(float(height) / float(size))
        grid = int(self.cfg.output_grid)
        paired = F.interpolate(
            paired,
            size=(grid, grid),
            mode="bilinear",
            align_corners=True,
        ).clamp(-self.cfg.flow_max_pixels, self.cfg.flow_max_pixels)
        backward, forward = paired[:count], paired[count:]
        forward_at_target, valid = _warp_flow_field(
            forward,
            backward,
            image_height=height,
            image_width=width,
        )
        residual_sq = (backward + forward_at_target).square().sum(
            dim=1, keepdim=True
        )
        magnitude_sq = backward.square().sum(dim=1, keepdim=True) + (
            forward_at_target.square().sum(dim=1, keepdim=True)
        )
        consistent = residual_sq <= (
            self.cfg.consistency_relative * magnitude_sq
            + self.cfg.consistency_absolute
        )
        disocclusion = (~(consistent & valid)).float()
        if not bool(torch.isfinite(backward).all()):
            raise FloatingPointError("RAFT flow target is non-finite")
        return {
            "rgb_flow_target_pixels": backward.reshape(
                batch, horizon, views, 2, grid, grid
            ).detach(),
            "rgb_disocclusion_target": disocclusion.reshape(
                batch, horizon, views, 1, grid, grid
            ).detach(),
        }


def raft_config_from_mapping(
    value: Mapping[str, Any],
) -> FrozenBidirectionalRAFTConfig:
    fields = set(FrozenBidirectionalRAFTConfig.__dataclass_fields__)
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown RAFT RGB teacher fields: {unknown}")
    return FrozenBidirectionalRAFTConfig(**dict(value))


__all__ = [
    "FrozenBidirectionalRAFTConfig",
    "FrozenBidirectionalRAFTRuntime",
    "raft_config_from_mapping",
]
