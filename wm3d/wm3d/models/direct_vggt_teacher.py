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
    appearance_enabled: bool
    rgb_decode_indices: tuple[int, ...]
    encode_chunk_rows: int = 32
    minimum_chunk_rows: int = 4

    def validate(self) -> None:
        self.encoder.validate()
        if self.context_frames <= 0 or self.future_frames <= 0:
            raise ValueError("direct VGGT temporal sizes must be positive")
        if self.appearance_enabled:
            if not 0 < self.appearance_context_frames <= self.context_frames:
                raise ValueError("direct VGGT appearance context is invalid")
            if not self.encoder.appearance_token_grid:
                raise ValueError("direct VGGT appearance training requires tokens")
        elif self.appearance_context_frames != 0:
            raise ValueError("disabled direct VGGT appearance context must be zero")
        elif self.encoder.appearance_token_grid:
            raise ValueError("disabled direct VGGT appearance cannot pool tokens")
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
        self.input_rows = 0
        self.encoded_rows = 0
        self.deduplicated_rows = 0
        self.encode_seconds = 0.0
        self.geometry_head_rows = 0
        self.appearance_pool_rows = 0
        self.rgb_resize_rows = 0
        self.oom_backoffs = 0
        self.effective_chunk_rows = int(config.encode_chunk_rows)
        self.temporal_teacher_calls = 0
        self.temporal_teacher_sequences = 0
        self.temporal_teacher_seconds = 0.0
        self.temporal_teacher_oom_backoffs = 0
        self.effective_temporal_teacher_samples = max(
            1,
            int(config.encode_chunk_rows) * int(config.encoder.max_views)
            // (int(config.context_frames) + int(config.future_frames)),
        )

    def train(self, mode: bool = True) -> "DirectVGGTTeacherAdapter":
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode(
        self,
        images_u8: torch.Tensor,
        view_mask: torch.Tensor,
        frame_keys: torch.Tensor | None = None,
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
        if frame_keys is not None and tuple(frame_keys.shape) != (batch, length):
            raise ValueError("direct frame keys must be [B,L]")
        if not bool(view_mask.any(dim=-1).all()):
            raise ValueError("every direct RGB timestamp requires a real view")
        flat_images = images_u8.reshape(
            batch * length, views, channels, height, width
        )
        flat_mask = view_mask.reshape(batch * length, views).bool()
        role_shape = (batch, length)
        geometry_row_mask = torch.zeros(
            role_shape, dtype=torch.bool, device=view_mask.device
        )
        geometry_row_mask[:, self.config.context_frames :] = True
        appearance_row_mask = torch.zeros_like(geometry_row_mask)
        if self.config.appearance_enabled:
            appearance_start = (
                self.config.context_frames
                - self.config.appearance_context_frames
            )
            appearance_row_mask[:, appearance_start:] = True
        rgb_row_mask = torch.zeros_like(geometry_row_mask)
        rgb_row_mask[:, : self.config.context_frames] = True
        for future_index in self.config.rgb_decode_indices:
            rgb_row_mask[:, self.config.context_frames + future_index] = True
        flat_geometry_rows = geometry_row_mask.reshape(batch * length)
        flat_appearance_rows = appearance_row_mask.reshape(batch * length)
        flat_rgb_rows = rgb_row_mask.reshape(batch * length)
        total_rows = batch * length
        inverse: torch.Tensor | None = None
        if frame_keys is not None:
            flat_keys = frame_keys.reshape(total_rows).to(
                device=flat_images.device,
                dtype=torch.int64,
            )
            _unique_keys, candidate_inverse = torch.unique(
                flat_keys,
                sorted=True,
                return_inverse=True,
            )
            unique_rows = int(_unique_keys.numel())
            if unique_rows < total_rows:
                positions = torch.arange(
                    total_rows,
                    dtype=torch.long,
                    device=candidate_inverse.device,
                )
                representatives = torch.full(
                    (unique_rows,),
                    total_rows,
                    dtype=torch.long,
                    device=candidate_inverse.device,
                )
                representatives.scatter_reduce_(
                    0,
                    candidate_inverse,
                    positions,
                    reduce="amin",
                    include_self=True,
                )
                original_mask = flat_mask
                flat_images = flat_images.index_select(0, representatives)
                flat_mask = flat_mask.index_select(0, representatives)
                if not torch.equal(
                    flat_mask.index_select(0, candidate_inverse),
                    original_mask,
                ):
                    raise ValueError(
                        "duplicate direct frame keys changed view availability"
                    )

                def reduce_role(value: torch.Tensor) -> torch.Tensor:
                    result = torch.zeros(
                        unique_rows,
                        dtype=torch.uint8,
                        device=value.device,
                    )
                    result.scatter_reduce_(
                        0,
                        candidate_inverse,
                        value.to(torch.uint8),
                        reduce="amax",
                        include_self=True,
                    )
                    return result.bool()

                flat_geometry_rows = reduce_role(flat_geometry_rows)
                flat_appearance_rows = reduce_role(flat_appearance_rows)
                flat_rgb_rows = reduce_role(flat_rgb_rows)
                inverse = candidate_inverse
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
            chunk_geometry_rows = (
                flat_geometry_rows[start:stop]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            chunk_appearance_rows = (
                flat_appearance_rows[start:stop]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            chunk_rgb_rows = (
                flat_rgb_rows[start:stop]
                .to(self.device, non_blocking=True)
                .unsqueeze(0)
            )
            try:
                encoded = self.encoder(
                    chunk_images,
                    chunk_mask,
                    geometry_row_mask=chunk_geometry_rows,
                    appearance_row_mask=chunk_appearance_rows,
                    rgb_row_mask=chunk_rgb_rows,
                )
            except torch.OutOfMemoryError:
                del chunk_images, chunk_mask
                del chunk_geometry_rows, chunk_appearance_rows, chunk_rgb_rows
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
            del chunk_geometry_rows, chunk_appearance_rows, chunk_rgb_rows
            start = stop
            self.encode_calls += 1
        self.effective_chunk_rows = min(
            self.effective_chunk_rows, effective
        )
        encoded_rows = len(flat_images)
        self.input_rows += total_rows
        self.encoded_rows += encoded_rows
        self.deduplicated_rows += total_rows - encoded_rows
        self.geometry_head_rows += int(flat_geometry_rows.sum().item())
        self.appearance_pool_rows += int(flat_appearance_rows.sum().item())
        self.rgb_resize_rows += int(flat_rgb_rows.sum().item())
        self.encode_seconds += time.perf_counter() - started
        if inverse is not None:
            outputs = {
                name: value.index_select(
                    0,
                    inverse.to(device=value.device, non_blocking=True),
                )
                for name, value in outputs.items()
            }
        return {
            name: value.reshape(batch, length, *value.shape[1:])
            for name, value in outputs.items()
        }

    def _encode_temporal_future_teacher(
        self,
        images_u8: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build V7-style temporal P64 labels without contaminating input."""

        batch, length, views, channels, _height, _width = images_u8.shape
        expected_length = self.config.context_frames + self.config.future_frames
        if (
            length != expected_length
            or channels != 3
            or views != self.config.encoder.max_views
            or tuple(view_mask.shape) != (batch, length, views)
        ):
            raise ValueError("temporal teacher input differs from direct RGB ABI")
        context_end = int(self.config.context_frames)
        # Match the original V7 cache's T+K temporal conditioning/order: the
        # anchor camera's complete observed+future window is one VGGT sequence.
        # Only future positions are returned as labels below. The causal model
        # input continues to come from the independent per-timestamp encode.
        sequence_images = images_u8[:, :, 0]
        sequence_valid = view_mask[:, :, 0].bool()
        if not bool(sequence_valid.all()):
            raise ValueError(
                "V7 temporal target requires the anchor camera at every "
                "observed and future timestamp"
            )

        chunks: list[torch.Tensor] = []
        start = 0
        effective = min(
            self.effective_temporal_teacher_samples, int(batch)
        )
        started = time.perf_counter()
        while start < batch:
            stop = min(batch, start + effective)
            images = (
                sequence_images[start:stop]
                .to(self.device, non_blocking=True)
                .float()
                .div_(255.0)
            )
            try:
                tokens = self.encoder.encode_temporal_teacher_tokens(images)
            except torch.OutOfMemoryError:
                del images
                if effective <= 1:
                    raise
                effective = max(1, effective // 2)
                self.temporal_teacher_oom_backoffs += 1
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            chunks.append(tokens)
            del images, tokens
            start = stop
            self.temporal_teacher_calls += 1
        self.effective_temporal_teacher_samples = min(
            self.effective_temporal_teacher_samples, effective
        )
        self.temporal_teacher_sequences += int(batch)
        self.temporal_teacher_seconds += time.perf_counter() - started
        temporal = torch.cat(chunks, dim=0)
        expected = (
            batch,
            expected_length,
            self.config.encoder.token_grid**2,
            self.config.encoder.token_dim,
        )
        if tuple(temporal.shape) != expected:
            raise RuntimeError(
                "temporal future teacher shape changed to "
                f"{tuple(temporal.shape)}; expected {expected}"
            )
        return temporal[:, context_end:]

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
        # P64 is an image-coordinate state. Patch ``p`` from two cameras does
        # not describe the same ray, so averaging equal patch indices across
        # views destroys the spatial coordinate needed by the V7 renderer.
        # Preserve the sealed ``head`` slot (view 0) as the anchor target;
        # auxiliary views remain available to the observed-state fuser.
        anchor_confidence = confidence[:, :, 0].float().clamp_min(0.0)
        anchor_valid = view_mask[:, :, 0, None] & (anchor_confidence > 0)
        return tokens[:, :, 0].to(torch.bfloat16), anchor_valid

    def materialize(
        self, batch: Mapping[str, Any]
    ) -> dict[str, Any]:
        images = batch["direct_rgb_uint8"]
        view_mask = batch["direct_view_mask"].bool()
        frame_keys = batch.get("direct_frame_keys")
        encoded = self._encode(images, view_mask, frame_keys)
        temporal_target_tokens = self._encode_temporal_future_teacher(
            images, view_mask
        )
        context = slice(0, self.config.context_frames)
        future = slice(self.config.context_frames, None)
        future_tokens = encoded["view_tokens"][:, future]
        future_confidence = encoded["geometry_confidence"][:, future]
        future_view_mask = encoded["view_mask"][:, future].bool()
        context_view_mask = encoded["view_mask"][:, context].bool()
        context_positions = torch.arange(
            self.config.context_frames,
            dtype=torch.long,
            device=context_view_mask.device,
        )[None, :, None]
        latest_context_index = torch.where(
            context_view_mask,
            context_positions,
            -1,
        ).amax(dim=1)
        context_rgb_mask = latest_context_index >= 0
        context_rgb_source = encoded["rgb"][:, context]
        context_rgb_gather = latest_context_index.clamp_min(0)[
            :, None, :, None, None, None
        ].expand(
            -1,
            1,
            -1,
            3,
            context_rgb_source.shape[-2],
            context_rgb_source.shape[-1],
        )
        context_rgb = context_rgb_source.gather(1, context_rgb_gather).squeeze(1)
        context_rgb = context_rgb.float().div_(255.0)
        context_rgb = context_rgb * context_rgb_mask[..., None, None, None]
        # Original V7 predicts RGB only in the anchor camera coordinate. Keep
        # auxiliary images for world-state encoding, but never ask one anchor
        # P64 state to render several incompatible camera frames.
        anchor_context_rgb_mask = torch.zeros_like(context_rgb_mask)
        anchor_context_rgb_mask[:, 0] = context_rgb_mask[:, 0]
        context_rgb_mask = anchor_context_rgb_mask
        target_tokens, target_token_mask = self._fuse_future_tokens(
            future_tokens,
            future_confidence,
            future_view_mask,
        )
        if tuple(temporal_target_tokens.shape) != tuple(target_tokens.shape):
            raise RuntimeError(
                "temporal P64 target does not match the anchor target ABI"
            )
        target_tokens = temporal_target_tokens
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
        rgb_indices = torch.as_tensor(
            self.config.rgb_decode_indices,
            dtype=torch.long,
            device=encoded["rgb"].device,
        )
        future_rgb = encoded["rgb"][:, future].index_select(
            1, rgb_indices
        )
        selected_future_view_mask = future_view_mask.index_select(1, rgb_indices)
        future_rgb_mask = torch.zeros_like(selected_future_view_mask)
        future_rgb_mask[:, :, 0] = (
            selected_future_view_mask[:, :, 0]
            & context_rgb_mask[:, None, 0]
        )
        result = dict(batch)
        result.pop("direct_rgb_uint8", None)
        result.pop("direct_view_mask", None)
        result.pop("direct_frame_keys", None)
        result.update(
            {
                "world_tokens": encoded["view_tokens"][:, context],
                "view_mask": encoded["view_mask"][:, context].bool(),
                "target_tokens": target_tokens,
                "target_token_mask": target_token_mask,
                "target_depth": target_depth,
                "target_depth_mask": depth_mask,
                "target_point": target_point,
                "target_point_mask": point_mask,
                "target_camera_pose": target_camera,
                "target_camera_pose_mask": camera_mask,
                "context_rgb": context_rgb,
                "context_rgb_mask": context_rgb_mask,
                "target_rgb": future_rgb.float().div_(255.0),
                "target_rgb_mask": future_rgb_mask[
                    ..., None, None, None
                ],
            }
        )
        if self.config.appearance_enabled:
            appearance = encoded["appearance_tokens"]
            appearance_start = (
                self.config.context_frames
                - self.config.appearance_context_frames
            )
            appearance_patches = self.config.encoder.appearance_token_grid**2
            result.update(
                {
                    "appearance_context_tokens": appearance[
                        :, appearance_start : self.config.context_frames
                    ],
                    "appearance_context_mask": encoded["view_mask"][
                        :,
                        appearance_start : self.config.context_frames,
                        :,
                        None,
                    ].expand(-1, -1, -1, appearance_patches),
                    "target_appearance_tokens": appearance[:, future],
                    "target_appearance_mask": future_view_mask[
                        ..., None
                    ].expand(-1, -1, -1, appearance_patches),
                }
            )
        return result

    @property
    def metrics(self) -> Mapping[str, float | int]:
        return {
            "encode_calls": self.encode_calls,
            "input_rows": self.input_rows,
            "encoded_rows": self.encoded_rows,
            "deduplicated_rows": self.deduplicated_rows,
            "deduplication_ratio": (
                self.deduplicated_rows / max(self.input_rows, 1)
            ),
            "encode_seconds": self.encode_seconds,
            "geometry_head_rows": self.geometry_head_rows,
            "appearance_pool_rows": self.appearance_pool_rows,
            "rgb_resize_rows": self.rgb_resize_rows,
            "oom_backoffs": self.oom_backoffs,
            "effective_chunk_rows": self.effective_chunk_rows,
            "temporal_teacher_calls": self.temporal_teacher_calls,
            "temporal_teacher_sequences": self.temporal_teacher_sequences,
            "temporal_teacher_seconds": self.temporal_teacher_seconds,
            "temporal_teacher_oom_backoffs": (
                self.temporal_teacher_oom_backoffs
            ),
            "effective_temporal_teacher_samples": (
                self.effective_temporal_teacher_samples
            ),
        }


def rgb_indices_tuple(value: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(item) for item in value)
