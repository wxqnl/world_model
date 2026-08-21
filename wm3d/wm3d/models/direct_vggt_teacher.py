"""Frozen direct VGGT adapter with train-only geometry teacher targets.

The adapter is replicated per rank and is intentionally outside FSDP.  It
consumes a fixed-size T+K uint8 RGB window, runs bounded VGGT chunks, and
returns the ordinary WM3D input/target ABI.  No latent is written to disk and
no VGGT parameter participates in optimization or checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import torch

from wm3d.encoders.native_vggt import NativeVGGTConfig, NativeVGGTEncoder


@dataclass(frozen=True)
class DirectVGGTTeacherConfig:
    encoder: NativeVGGTConfig
    context_frames: int
    future_frames: int
    appearance_context_frames: int
    rgb_decode_indices: tuple[int, ...]
    encode_chunk_rows: int = 32
    minimum_chunk_rows: int = 4

    def validate(self) -> None:
        self.encoder.validate()
        if self.context_frames <= 0 or self.future_frames <= 0:
            raise ValueError("direct VGGT temporal sizes must be positive")
        if not 0 < self.appearance_context_frames <= self.context_frames:
            raise ValueError("direct VGGT appearance context is invalid")
        if not self.encoder.appearance_token_grid:
            raise ValueError("direct VGGT training requires appearance tokens")
        if (
            not self.rgb_decode_indices
            or min(self.rgb_decode_indices) < 0
            or max(self.rgb_decode_indices) >= self.future_frames
        ):
            raise ValueError("direct VGGT RGB decode indices are invalid")
        if (
            self.encode_chunk_rows <= 0
            or self.minimum_chunk_rows <= 0
            or self.minimum_chunk_rows > self.encode_chunk_rows
        ):
            raise ValueError("direct VGGT chunk configuration is invalid")


class DirectVGGTTeacherAdapter(torch.nn.Module):
    """Produce native WM3D tensors from raw RGB with bounded GPU work."""

    def __init__(
        self,
        config: DirectVGGTTeacherConfig,
        *,
        device: torch.device | str,
        encoder: NativeVGGTEncoder | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.device = torch.device(device)
        self.encoder = encoder or NativeVGGTEncoder(
            config.encoder,
            device=str(self.device),
            local_files_only=True,
        )
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.encode_calls = 0
        self.encoded_rows = 0
        self.encode_seconds = 0.0
        self.oom_backoffs = 0
        self.effective_chunk_rows = int(config.encode_chunk_rows)

    def train(self, mode: bool = True) -> "DirectVGGTTeacherAdapter":
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode(
        self,
        images_u8: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images_u8.ndim != 6:
            raise ValueError("direct RGB must be [B,L,V,3,H,W]")
        batch, length, views, channels, height, width = images_u8.shape
        expected_length = (
            self.config.context_frames + self.config.future_frames
        )
        if (
            length != expected_length
            or channels != 3
            or views != self.config.encoder.max_views
            or (height, width)
            != (
                self.config.encoder.input_rgb_size,
                self.config.encoder.input_rgb_size,
            )
        ):
            raise ValueError("direct RGB shape differs from sealed adapter contract")
        if tuple(view_mask.shape) != (batch, length, views):
            raise ValueError("direct RGB view mask shape is invalid")
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every direct RGB timestamp requires a real view")
        flat_images = images_u8.reshape(
            batch * length, views, channels, height, width
        )
        flat_mask = view_mask.reshape(batch * length, views).bool()
        outputs: dict[str, torch.Tensor] = {}
        output_names: tuple[str, ...] | None = None
        start = 0
        effective = min(self.effective_chunk_rows, len(flat_images))
        started = time.perf_counter()
        while start < len(flat_images):
            stop = min(len(flat_images), start + effective)
            chunk_images = (
                flat_images[start:stop]
                .to(self.device, non_blocking=True)
                .float()
                .div_(255.0)
                .unsqueeze(0)
            )
            chunk_mask = (
                flat_mask[start:stop]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            try:
                encoded = self.encoder(chunk_images, chunk_mask)
            except torch.OutOfMemoryError:
                del chunk_images, chunk_mask
                if effective <= self.config.minimum_chunk_rows:
                    raise
                effective = max(
                    self.config.minimum_chunk_rows, effective // 2
                )
                self.oom_backoffs += 1
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            names = tuple(encoded)
            if output_names is None:
                output_names = names
            elif names != output_names:
                raise RuntimeError("direct VGGT output schema changed between chunks")
            for name, value in encoded.items():
                chunk = value[0]
                if chunk.shape[0] != stop - start:
                    raise RuntimeError(
                        f"direct VGGT {name} row count changed between chunks"
                    )
                target = outputs.get(name)
                if target is None:
                    # Allocate outside NativeVGGTEncoder.forward's inference-mode
                    # context. The normal tensor can be saved by the trainable
                    # world-model autograd graph; slice copies avoid the old
                    # list-of-clones plus final torch.cat double allocation.
                    target = torch.empty(
                        (len(flat_images), *chunk.shape[1:]),
                        dtype=chunk.dtype,
                        device=chunk.device,
                    )
                    outputs[name] = target
                elif target.shape[1:] != chunk.shape[1:]:
                    raise RuntimeError(
                        f"direct VGGT {name} shape changed between chunks"
                    )
                target[start:stop].copy_(chunk)
            del encoded, chunk_images, chunk_mask
            start = stop
            self.encode_calls += 1
        self.effective_chunk_rows = min(
            self.effective_chunk_rows, effective
        )
        self.encoded_rows += batch * length
        self.encode_seconds += time.perf_counter() - started
        return {
            name: value.reshape(batch, length, *value.shape[1:])
            for name, value in outputs.items()
        }

    @staticmethod
    def _fuse_future_tokens(
        tokens: torch.Tensor,
        confidence: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            tokens.ndim != 5
            or confidence.shape != tokens.shape[:-1]
            or view_mask.shape != tokens.shape[:3]
        ):
            raise ValueError("direct future geometry shapes are inconsistent")
        weight = (
            confidence.float().clamp_min(0.0)
            * view_mask[..., None].float()
        )
        denominator = weight.sum(dim=2)
        target = (
            (tokens.float() * weight[..., None]).sum(dim=2)
            / denominator.clamp_min(1.0e-12)[..., None]
        )
        return target.to(torch.bfloat16), denominator > 0

    def materialize(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        images = batch["direct_rgb_uint8"]
        view_mask = batch["direct_view_mask"].bool()
        encoded = self._encode(images, view_mask)
        context = slice(0, self.config.context_frames)
        future = slice(self.config.context_frames, None)
        future_tokens = encoded["view_tokens"][:, future]
        future_confidence = encoded["geometry_confidence"][:, future]
        future_view_mask = encoded["view_mask"][:, future].bool()
        target_tokens, target_token_mask = self._fuse_future_tokens(
            future_tokens,
            future_confidence,
            future_view_mask,
        )
        real_patch = future_view_mask[..., None]
        geometry_valid = future_confidence > 0
        target_depth = encoded["depth"][:, future]
        target_point = encoded["point"][:, future]
        target_camera = encoded["camera_pose"][:, future]
        depth_mask = (
            real_patch
            & geometry_valid
            & torch.isfinite(target_depth)
            & (target_depth > 0)
        )
        point_mask = (
            real_patch
            & geometry_valid
            & torch.isfinite(target_point).all(dim=-1)
        )
        camera_mask = (
            future_view_mask
            & torch.isfinite(target_camera).all(dim=-1)
        )
        appearance = encoded["appearance_tokens"]
        appearance_start = (
            self.config.context_frames
            - self.config.appearance_context_frames
        )
        rgb_indices = torch.as_tensor(
            self.config.rgb_decode_indices,
            dtype=torch.long,
            device=encoded["rgb"].device,
        )
        future_rgb = encoded["rgb"][:, future].index_select(
            1, rgb_indices
        )
        future_rgb_mask = future_view_mask.index_select(1, rgb_indices)
        result = dict(batch)
        result.pop("direct_rgb_uint8", None)
        result.pop("direct_view_mask", None)
        result.update(
            {
                "world_tokens": encoded["view_tokens"][:, context],
                "view_mask": encoded["view_mask"][:, context].bool(),
                "appearance_context_tokens": appearance[
                    :, appearance_start : self.config.context_frames
                ],
                "appearance_context_mask": encoded["view_mask"][
                    :, appearance_start : self.config.context_frames, :, None
                ].expand(
                    -1,
                    -1,
                    -1,
                    self.config.encoder.appearance_token_grid**2,
                ),
                "target_appearance_tokens": appearance[:, future],
                "target_appearance_mask": future_view_mask[..., None].expand(
                    -1,
                    -1,
                    -1,
                    self.config.encoder.appearance_token_grid**2,
                ),
                "target_tokens": target_tokens,
                "target_token_mask": target_token_mask,
                "target_depth": target_depth,
                "target_depth_mask": depth_mask,
                "target_point": target_point,
                "target_point_mask": point_mask,
                "target_camera_pose": target_camera,
                "target_camera_pose_mask": camera_mask,
                "target_rgb": future_rgb.float().div_(255.0),
                "target_rgb_mask": future_rgb_mask[
                    ..., None, None, None
                ],
            }
        )
        return result

    @property
    def metrics(self) -> Mapping[str, float | int]:
        return {
            "encode_calls": self.encode_calls,
            "encoded_rows": self.encoded_rows,
            "encode_seconds": self.encode_seconds,
            "oom_backoffs": self.oom_backoffs,
            "effective_chunk_rows": self.effective_chunk_rows,
        }


def rgb_indices_tuple(value: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(item) for item in value)
