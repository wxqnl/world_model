"""Train/serve-identical inputs for the WSA-backed WM3D foundation model."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json
import os
from math import isqrt
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from wm3d.data.foundation_bridge import (
    FoundationBridgeConfig,
    FoundationBridgeError,
    build_foundation_sample,
)
from wm3d.encoders.native_vggt import NativeVGGTEncoder


FOUNDATION_CONTEXT_STORE_SCHEMA = "wm3d_foundation_context_store_v1"


class FoundationInputError(ValueError):
    pass


def _resize_token_grid(tokens: torch.Tensor, output_tokens: int) -> torch.Tensor:
    if tokens.ndim != 4:
        raise FoundationInputError("VGGT tokens must be [B,V,P,D]")
    batch, views, patches, width = tokens.shape
    source_grid = isqrt(patches)
    target_grid = isqrt(int(output_tokens))
    if source_grid**2 != patches or target_grid**2 != int(output_tokens):
        raise FoundationInputError("VGGT token grids must be square")
    if source_grid == target_grid:
        return tokens
    value = F.interpolate(
        tokens.permute(0, 1, 3, 2)
        .reshape(batch * views, width, source_grid, source_grid)
        .float(),
        size=(target_grid, target_grid),
        mode="bilinear",
        align_corners=False,
    )
    return value.reshape(batch, views, width, int(output_tokens)).permute(
        0, 1, 3, 2
    ).to(dtype=tokens.dtype)


class FoundationContextStore:
    """Immutable Wan text contexts keyed by the exact task string."""

    def __init__(
        self,
        tasks: Sequence[str],
        context: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        normalized = tuple(str(item).strip() for item in tasks)
        if not normalized or any(not item for item in normalized):
            raise FoundationInputError("foundation context tasks must be non-empty")
        if len(set(normalized)) != len(normalized):
            raise FoundationInputError("foundation context tasks must be unique")
        if context.ndim != 3 or mask.ndim != 2:
            raise FoundationInputError("Wan context store tensors have invalid ranks")
        if context.shape[:2] != mask.shape or context.shape[0] != len(normalized):
            raise FoundationInputError("Wan context store tensors disagree with task index")
        if mask.dtype != torch.bool:
            raise FoundationInputError("Wan context mask must be boolean")
        if not bool(torch.isfinite(context).all()) or not bool(mask.any(dim=1).all()):
            raise FoundationInputError("Wan context store contains invalid entries")
        self.tasks = normalized
        self.context = context.cpu().contiguous()
        self.mask = mask.cpu().contiguous()
        self._row = {task: index for index, task in enumerate(normalized)}

    @classmethod
    def from_directory(cls, root: Path | str) -> "FoundationContextStore":
        from safetensors.torch import load_file

        directory = Path(root).expanduser().resolve(strict=True)
        index_path = directory / "index.json"
        tensor_path = directory / "context.safetensors"
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("schema") == "wm3d_admitted_foundation_context_union_v1":
            from .admitted_context import load_context_union
            return load_context_union(directory, payload, cls)
        if payload.get("schema") != FOUNDATION_CONTEXT_STORE_SCHEMA:
            raise FoundationInputError("foundation context store schema mismatch")
        if set(payload) != {"schema", "tasks"}:
            raise FoundationInputError("foundation context store index fields changed")
        tensors = load_file(str(tensor_path), device="cpu")
        if set(tensors) != {"context", "mask"}:
            raise FoundationInputError("foundation context tensor fields changed")
        return cls(payload["tasks"], tensors["context"], tensors["mask"].bool())

    def lookup(self, tasks: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[int] = []
        for raw in tasks:
            task = str(raw).strip()
            if task not in self._row:
                raise FoundationInputError(
                    f"task is absent from the sealed Wan context store: {task!r}"
                )
            rows.append(self._row[task])
        index = torch.tensor(rows, dtype=torch.long)
        return self.context.index_select(0, index), self.mask.index_select(0, index)


class FoundationInputAdapter(torch.nn.Module):
    """Materialize WSA video, Wan task context and frozen VGGT targets."""

    def __init__(
        self,
        bridge: FoundationBridgeConfig,
        *,
        context_store: FoundationContextStore,
        vggt_encoder: NativeVGGTEncoder,
        device: torch.device | str,
        native_3d_context_tokens: int = 144,
        native_3d_rollout_tokens: int = 64,
        native_3d_target_tokens: int = 144,
        vggt_cache_bytes: int = 0,
        native_3d_mode: str = "legacy_readout",
    ) -> None:
        super().__init__()
        bridge.validate()
        self.bridge = bridge
        self.context_store = context_store
        self.vggt_encoder = vggt_encoder
        self.vggt_encoder.requires_grad_(False)
        self.vggt_encoder.eval()
        self.device = torch.device(device)
        self.native_3d_mode = native_3d_mode
        if native_3d_mode not in {"legacy_readout", "trajectory"}:
            raise FoundationInputError("unsupported native-3D target mode")
        self.native_3d_context_tokens = int(native_3d_context_tokens)
        self.native_3d_rollout_tokens = int(native_3d_rollout_tokens)
        self.native_3d_target_tokens = int(native_3d_target_tokens)
        if self.native_3d_context_tokens != self.native_3d_target_tokens:
            raise FoundationInputError(
                "current and final VGGT grids must share the P144 encoder pass"
            )
        self.calls = 0
        self.samples = 0
        self.vggt_frames = 0
        self.vggt_seconds = 0.0
        self.vggt_cache_limit = int(vggt_cache_bytes)
        if self.vggt_cache_limit < 0:
            raise FoundationInputError("VGGT cache budget must be non-negative")
        # Instance-local: one frozen encoder, preprocessing contract and data
        # namespace. A frame key includes its episode; the mask preserves the
        # simultaneous-camera context. No temporal teacher features are cached.
        self._vggt_cache: OrderedDict[tuple[int, tuple[bool, ...]], torch.Tensor] = OrderedDict()
        self.vggt_cache_bytes = 0
        self.vggt_cache_hits = 0
        self.vggt_cache_misses = 0
        self.vggt_cache_evictions = 0
        self.vggt_coalesced_rows = 0
        self.vggt_encoded_rows = 0

    def train(self, mode: bool = True) -> "FoundationInputAdapter":
        super().train(mode)
        self.vggt_encoder.eval()
        return self

    def _encode_native_rows(
        self, images: torch.Tensor, view_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.native_3d_mode == "trajectory" and images.shape[0] * images.shape[1] > 2:
            # VGGT is frozen: bound teacher memory without altering targets or
            # accumulating nine future frames' activations on the GPU.
            b, t = images.shape[:2]
            flat, mask = images.flatten(0, 1), view_mask.flatten(0, 1)
            encoded = [self._encode_native_rows(flat[i:i + 2, None], mask[i:i + 2, None])[:, 0]
                       for i in range(0, len(flat), 2)]
            return torch.cat(encoded).reshape(b, t, *encoded[0].shape[1:])
        selected = images.to(self.device, non_blocking=True).float().div_(255.0)
        selected_mask = view_mask.to(self.device, non_blocking=True)
        with torch.inference_mode():
            output = self.vggt_encoder(
                selected,
                selected_mask,
                geometry_row_mask=torch.zeros(selected.shape[:2], dtype=torch.bool, device=self.device),
                appearance_row_mask=torch.zeros(selected.shape[:2], dtype=torch.bool, device=self.device),
                rgb_row_mask=torch.zeros(selected.shape[:2], dtype=torch.bool, device=self.device),
            )
        self.vggt_encoded_rows += int(images.shape[0] * images.shape[1])
        # Differentiable consumers must receive ordinary, not inference tensors.
        return output["view_tokens"].clone()

    def _cached_native_rows(
        self, images: torch.Tensor, view_mask: torch.Tensor, frame_keys: torch.Tensor
    ) -> torch.Tensor:
        flat_images = images.flatten(0, 1)
        flat_mask = view_mask.flatten(0, 1)
        keys = [
            (int(key), tuple(bool(v) for v in mask))
            for key, mask in zip(frame_keys.flatten().cpu().tolist(), flat_mask.cpu().tolist(), strict=True)
        ]
        rows: dict[tuple[int, tuple[bool, ...]], torch.Tensor] = {}
        missing: dict[tuple[int, tuple[bool, ...]], int] = {}
        for index, key in enumerate(keys):
            if key in rows or key in missing:
                self.vggt_coalesced_rows += 1
            elif key in self._vggt_cache:
                rows[key] = self._vggt_cache[key]
                self._vggt_cache.move_to_end(key)
                self.vggt_cache_hits += 1
            else:
                missing[key] = index
                self.vggt_cache_misses += 1
        if missing:
            indices = torch.tensor(list(missing.values()), device=images.device)
            encoded = self._encode_native_rows(
                flat_images.index_select(0, indices).unsqueeze(1),
                flat_mask.index_select(0, indices.to(flat_mask.device)).unsqueeze(1),
            )[:, 0].detach().cpu()
            for (key, _index), value in zip(missing.items(), encoded, strict=True):
                # Clone each row so retaining one entry never retains a batch's storage.
                value = value.clone()
                rows[key] = value
                size = value.numel() * value.element_size()
                if size > self.vggt_cache_limit:
                    continue
                while self.vggt_cache_bytes + size > self.vggt_cache_limit:
                    _old_key, old = self._vggt_cache.popitem(last=False)
                    self.vggt_cache_bytes -= old.numel() * old.element_size()
                    self.vggt_cache_evictions += 1
                self._vggt_cache[key] = value
                self.vggt_cache_bytes += size
        return torch.stack([rows[key] for key in keys]).reshape(
            *images.shape[:3], self.native_3d_context_tokens, 2048
        ).to(self.device, non_blocking=True)

    def _native_3d_targets(
        self,
        images: torch.Tensor,
        view_mask: torch.Tensor,
        sample_index: torch.Tensor,
        frame_keys: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        if self.native_3d_mode == "trajectory":
            first = self.bridge.context_frames - 1
            selected = images[:, first:first + self.bridge.future_frames + 1]
            masks = view_mask[:, first:first + self.bridge.future_frames + 1].bool()
            if selected.shape[1] != self.bridge.future_frames + 1:
                raise FoundationInputError("native trajectory is missing future frames")
            if self.vggt_cache_limit and frame_keys is not None:
                if frame_keys.shape != images.shape[:2] or frame_keys.dtype != torch.long:
                    raise FoundationInputError("native trajectory frame identities changed")
                tokens = self._cached_native_rows(selected, masks, frame_keys[:, first:first + selected.shape[1]])
            else:
                tokens = self._encode_native_rows(selected, masks)
            expected = (batch, self.bridge.future_frames + 1, self.bridge.num_views,
                        self.native_3d_context_tokens, 2048)
            if tokens.shape != expected or not bool(torch.isfinite(tokens).all()):
                raise FoundationInputError("native trajectory VGGT shape/value changed")
            masks = masks.to(tokens.device)
            # View-major ordering is shared with the WSA attention mask.
            target = tokens[:, 1:].permute(0, 2, 1, 3, 4).contiguous()
            valid = masks[:, 1:].permute(0, 2, 1).contiguous()
            return {
                "vggt_current_tokens": tokens[:, 0],
                "vggt_current_view_mask": masks[:, 0],
                "vggt_future_tokens": tokens[:, -1],
                "vggt_rollout_target_tokens": tokens[:, -1],
                "vggt_rollout_target_horizon": torch.full((batch,), self.bridge.future_frames - 1, device=tokens.device, dtype=torch.long),
                "vggt_rollout_target_view_mask": masks[:, -1],
                "native_3d_trajectory_target": target,
                "native_3d_trajectory_mask": valid,
            }
        if sample_index.shape != (batch,) or sample_index.dtype != torch.long:
            raise FoundationInputError("sample_index must be int64 [B]")
        current_index = self.bridge.context_frames - 1
        horizon = sample_index.remainder(self.bridge.future_frames)
        rows = torch.arange(batch, device=images.device)
        sampled_index = horizon.to(device=images.device) + self.bridge.context_frames
        final_index = self.bridge.context_frames + self.bridge.future_frames - 1
        selected = torch.stack(
            (
                images[:, current_index],
                images[rows, sampled_index],
                images[:, final_index],
            ),
            dim=1,
        )
        selected_mask = torch.stack(
            (
                view_mask[:, current_index],
                view_mask[rows, sampled_index],
                view_mask[:, final_index],
            ),
            dim=1,
        ).to(self.device, non_blocking=True)
        if self.vggt_cache_limit and frame_keys is not None:
            if frame_keys.shape != images.shape[:2] or frame_keys.dtype != torch.long:
                raise FoundationInputError("direct frame keys must be int64 [B,T+K]")
            key_rows = torch.arange(batch, device=frame_keys.device)
            selected_keys = torch.stack((
                frame_keys[:, current_index],
                frame_keys[key_rows, sampled_index.to(frame_keys.device)],
                frame_keys[:, final_index],
            ), dim=1)
            tokens = self._cached_native_rows(selected, selected_mask, selected_keys)
        else:
            # Serving/legacy callers without sealed physical-frame identities
            # take the original path; sample_index alone is not a cache key.
            tokens = self._encode_native_rows(selected, selected_mask)
        expected = (
            images.shape[0],
            3,
            self.bridge.num_views,
            self.native_3d_context_tokens,
            2048,
        )
        if tuple(tokens.shape) != expected or not bool(torch.isfinite(tokens).all()):
            raise FoundationInputError(
                f"VGGT foundation sequence changed to {tuple(tokens.shape)}"
            )
        return {
            "vggt_current_tokens": tokens[:, 0],
            "vggt_current_view_mask": selected_mask[:, 0],
            "vggt_rollout_target_tokens": _resize_token_grid(
                tokens[:, 1], self.native_3d_rollout_tokens
            ),
            "vggt_rollout_target_horizon": horizon.to(self.device),
            "vggt_rollout_target_view_mask": selected_mask[:, 1],
            "vggt_future_tokens": tokens[:, 2],
        }

    def materialize(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        images = batch.get("direct_rgb_uint8")
        view_mask = batch.get("direct_view_mask")
        if not isinstance(images, torch.Tensor) or not isinstance(view_mask, torch.Tensor):
            raise FoundationBridgeError("foundation input requires direct raw RGB")
        sample = build_foundation_sample(batch, self.bridge)
        tasks = sample["task_text"]
        wan_context, wan_mask = self.context_store.lookup(tasks)
        started = time.perf_counter()
        sample_index = batch.get("sample_index")
        if not isinstance(sample_index, torch.Tensor):
            raise FoundationInputError(
                "foundation data must preserve sample_index for deterministic K8 supervision"
            )
        native_3d = self._native_3d_targets(
            images, view_mask.bool(), sample_index.long(), batch.get("direct_frame_keys")
        )
        self.vggt_seconds += time.perf_counter() - started
        sample["wan_context"] = wan_context
        sample["wan_context_mask"] = wan_mask
        sample.update(native_3d)
        self.calls += 1
        self.samples += int(images.shape[0])
        self.vggt_frames += int(images.shape[0]) * (self.bridge.future_frames + 1 if self.native_3d_mode == "trajectory" else 3)
        return sample

    @property
    def metrics(self) -> Mapping[str, float | int]:
        return {
            "calls": self.calls,
            "samples": self.samples,
            "vggt_frames": self.vggt_frames,
            "vggt_seconds": self.vggt_seconds,
            "vggt_encoded_rows": self.vggt_encoded_rows,
            "vggt_cache_bytes": self.vggt_cache_bytes,
            "vggt_cache_hits": self.vggt_cache_hits,
            "vggt_cache_misses": self.vggt_cache_misses,
            "vggt_cache_evictions": self.vggt_cache_evictions,
            "vggt_coalesced_rows": self.vggt_coalesced_rows,
        }


def build_foundation_input_adapter(
    runtime: Mapping[str, Any], *, device: torch.device | str
) -> FoundationInputAdapter:
    from scripts.data.run_cache_worker import _strict_encoder
    from wm3d.models.foundation_world_model import foundation_config_from_mapping

    model = foundation_config_from_mapping(runtime["model_profile"]["model"])
    closure = runtime["data_closure"]
    encoder_contract = _strict_encoder(Path(str(closure["encoder_contract_path"])))
    encoder_config = replace(
        encoder_contract,
        token_grid=isqrt(model.native_3d_context_tokens),
        appearance_token_grid=0,
        appearance_feature_layer=-1,
        target_rgb_size=model.video_height,
    )
    return FoundationInputAdapter(
        FoundationBridgeConfig(
            context_frames=model.T,
            future_frames=model.K,
            num_views=model.num_views,
            video_height=model.video_height,
            video_width=model.video_width,
            max_action_groups=model.max_action_groups,
            max_action_dim=model.max_action_dim,
            max_state_dim=model.max_state_dim,
        ),
        context_store=FoundationContextStore.from_directory(
            model.wan_context_store_path
        ),
        vggt_encoder=NativeVGGTEncoder(
            encoder_config,
            device=str(device),
            local_files_only=True,
        ),
        device=device,
        native_3d_context_tokens=model.native_3d_context_tokens,
        native_3d_rollout_tokens=model.native_3d_rollout_tokens,
        native_3d_target_tokens=model.native_3d_target_tokens,
        native_3d_mode=model.native_3d_mode,
        vggt_cache_bytes=int(os.environ.get("WM3D_FOUNDATION_VGGT_CACHE_BYTES_PER_RANK", "0")),
    )
