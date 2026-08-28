"""Frozen target-side runtimes for latent RGB renderer supervision."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from wm3d.models.latent_motion_renderer import warp_with_pixel_flow


@dataclass(frozen=True)
class FrozenCosmosRGBTokenizerConfig:
    source_root: str
    checkpoint: str
    encode_chunk: int = 4
    decode_chunk: int = 2
    latent_channels: int = 16
    spatial_compression: int = 8


class FrozenCosmosRGBTokenizerRuntime:
    """Unregistered tokenizer used for target encoding and RGB audits only."""

    def __init__(
        self,
        cfg: FrozenCosmosRGBTokenizerConfig,
        device: torch.device,
    ) -> None:
        if cfg.encode_chunk <= 0 or cfg.decode_chunk <= 0:
            raise ValueError("Cosmos tokenizer chunks must be positive")
        root = Path(cfg.source_root).resolve(strict=True)
        checkpoint = Path(cfg.checkpoint).resolve(strict=True)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = importlib.import_module("cosmos_predict2.tokenizers.tokenizer")
        interface = getattr(module, "TokenizerInterface")(
            chunk_duration=81,
            load_mean_std=False,
            vae_pth=str(checkpoint),
        )
        interface.to(device=device)
        interface.model.model.eval().requires_grad_(False)
        if int(interface.model.count_param()) <= 0:
            raise RuntimeError("Cosmos tokenizer has no parameters")
        self.interface = interface
        self.cfg = cfg
        self.device = device

    @property
    def parameter_count(self) -> int:
        return int(self.interface.model.count_param())

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [N,3,H,W] RGB in [0,1] to [N,C,H/8,W/8]."""

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Cosmos RGB encoder expects [N,3,H,W]")
        if images.shape[-2] % self.cfg.spatial_compression or (
            images.shape[-1] % self.cfg.spatial_compression
        ):
            raise ValueError("RGB size is not divisible by tokenizer compression")
        values: list[torch.Tensor] = []
        for start in range(0, int(images.shape[0]), self.cfg.encode_chunk):
            image = images[start : start + self.cfg.encode_chunk]
            value = image.to(self.device, dtype=torch.bfloat16)
            value = value.clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(2)
            latent = self.interface.encode(value)
            if latent.shape[2] != 1:
                raise RuntimeError("per-frame Cosmos encoding produced temporal pooling")
            values.append(latent[:, :, 0].float())
        result = torch.cat(values, dim=0)
        expected = (
            images.shape[0],
            self.cfg.latent_channels,
            images.shape[-2] // self.cfg.spatial_compression,
            images.shape[-1] // self.cfg.spatial_compression,
        )
        if tuple(result.shape) != expected or not bool(torch.isfinite(result).all()):
            raise RuntimeError("Cosmos RGB latent contract failed")
        return result

    @torch.inference_mode()
    def decode_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode [N,C,h,w] normalized latents to RGB in [0,1]."""

        if latents.ndim != 4 or latents.shape[1] != self.cfg.latent_channels:
            raise ValueError("Cosmos RGB decoder expects [N,C,h,w]")
        values: list[torch.Tensor] = []
        for start in range(0, int(latents.shape[0]), self.cfg.decode_chunk):
            latent = latents[start : start + self.cfg.decode_chunk]
            decoded = self.interface.decode(
                latent.to(self.device, dtype=torch.bfloat16).unsqueeze(2)
            )
            values.append(
                decoded[:, :, 0].float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
            )
        result = torch.cat(values, dim=0)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Cosmos RGB decode produced non-finite values")
        return result

    def materialize_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Attach detached per-frame context/target latents to a CUDA batch."""

        if "context_rgb" not in batch or "target_rgb" not in batch:
            raise ValueError("latent RGB training requires context and target RGB")
        context = batch["context_rgb"]
        target = batch["target_rgb"]
        if context.ndim != 5 or target.ndim != 6:
            raise ValueError("RGB batch ranks are incompatible with latent encoding")
        batch_size, views = context.shape[:2]
        horizon = target.shape[1]
        if target.shape[0] != batch_size or target.shape[2] != views:
            raise ValueError("context/target RGB batch-view axes differ")
        context_latent = self.encode_images(
            context.reshape(-1, *context.shape[-3:])
        ).reshape(batch_size, views, self.cfg.latent_channels, -1, context.shape[-1] // self.cfg.spatial_compression)
        target_latent = self.encode_images(
            target.reshape(-1, *target.shape[-3:])
        ).reshape(
            batch_size,
            horizon,
            views,
            self.cfg.latent_channels,
            target.shape[-2] // self.cfg.spatial_compression,
            target.shape[-1] // self.cfg.spatial_compression,
        )
        result = dict(batch)
        result["context_rgb_latent"] = context_latent
        result["target_rgb_latent"] = target_latent
        return result


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


class FrozenBidirectionalRAFTRuntime:
    """Return flow and forward/backward-consistency disocclusion targets."""

    def __init__(self, cfg: FrozenBidirectionalRAFTConfig, device: torch.device):
        if cfg.input_size <= 0 or cfg.input_size % 8:
            raise ValueError("RAFT input size must be positive and divisible by 8")
        if cfg.output_grid <= 0 or cfg.iterations <= 0 or cfg.batch_chunk <= 0:
            raise ValueError("RAFT grid/iterations/chunk must be positive")
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

    @torch.inference_mode()
    def targets(
        self,
        context_rgb: torch.Tensor,
        future_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return [B,K,V,...] target-to-context flow and disocclusion."""

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
        paired = F.interpolate(
            paired,
            size=(self.cfg.output_grid, self.cfg.output_grid),
            mode="bilinear",
            align_corners=True,
        ).clamp(-self.cfg.flow_max_pixels, self.cfg.flow_max_pixels)
        backward, forward = paired[:count], paired[count:]
        forward_at_target, valid = warp_with_pixel_flow(
            forward,
            backward,
            image_height=height,
            image_width=width,
        )
        residual_sq = (backward + forward_at_target).square().sum(dim=1, keepdim=True)
        magnitude_sq = backward.square().sum(dim=1, keepdim=True) + (
            forward_at_target.square().sum(dim=1, keepdim=True)
        )
        consistent = residual_sq <= (
            self.cfg.consistency_relative * magnitude_sq
            + self.cfg.consistency_absolute
        )
        disocclusion = (~(consistent & valid)).float()
        grid = int(self.cfg.output_grid)
        return {
            "rgb_flow_target_pixels": backward.reshape(
                batch, horizon, views, 2, grid, grid
            ).detach(),
            "rgb_disocclusion_target": disocclusion.reshape(
                batch, horizon, views, 1, grid, grid
            ).detach(),
            "rgb_flow_forward_pixels": forward.reshape(
                batch, horizon, views, 2, grid, grid
            ).detach(),
        }


def cosmos_config_from_mapping(
    value: Mapping[str, Any],
) -> FrozenCosmosRGBTokenizerConfig:
    fields = set(FrozenCosmosRGBTokenizerConfig.__dataclass_fields__)
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown Cosmos RGB tokenizer fields: {unknown}")
    return FrozenCosmosRGBTokenizerConfig(**dict(value))


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
    "FrozenCosmosRGBTokenizerConfig",
    "FrozenCosmosRGBTokenizerRuntime",
    "cosmos_config_from_mapping",
    "raft_config_from_mapping",
]
