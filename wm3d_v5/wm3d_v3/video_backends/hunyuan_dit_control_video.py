"""Opt-in HunyuanVideo backend with DiT image-token control injection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from wm3d_v3.models.hunyuan_dit_control_adapter import (
    HunyuanDiTControlAdapter,
    HunyuanDiTControlConfig,
    HunyuanDiTControlInjector,
    load_hunyuan_dit_control_checkpoint,
)

from .base import VideoBackendOutput, VideoConditionBundle
from .hunyuan_video import HunyuanVideoBackend, HunyuanVideoBackendConfig, align_hunyuan_video_length, summarize_bundle_for_prompt


@dataclass
class HunyuanDiTControlVideoBackendConfig(HunyuanVideoBackendConfig):
    control_ckpt: str | None = None
    control_scale: float = 1.0
    allow_untrained_control: bool = False


class HunyuanDiTControlVideoBackend(HunyuanVideoBackend):
    """HunyuanVideo backend that injects a trained wm3d DiT control adapter.

    This backend is deliberately separate from ``HunyuanVideoBackend`` so the
    default T2V path and stage0/stage1/stage2 world-model training never import
    or instantiate the Hunyuan DiT control adapter unless explicitly selected.
    """

    name = "hunyuan_video_dit_control"

    def __init__(self, cfg: HunyuanDiTControlVideoBackendConfig | None = None, *, device: str | torch.device | None = None):
        super().__init__(cfg or HunyuanDiTControlVideoBackendConfig(), device=device)
        self.cfg: HunyuanDiTControlVideoBackendConfig
        self._control_adapter: HunyuanDiTControlAdapter | None = None
        self._control_payload: dict[str, Any] | None = None

    @staticmethod
    def _looks_like_transformer(obj: Any) -> bool:
        return isinstance(obj, nn.Module) and (hasattr(obj, "double_blocks") or hasattr(obj, "single_blocks"))

    @classmethod
    def resolve_transformer(cls, sampler: Any) -> nn.Module:
        """Find the Hunyuan DiT module on common sampler/pipeline layouts."""
        queue: list[Any] = [sampler]
        seen: set[int] = set()
        attr_names = (
            "transformer",
            "model",
            "dit",
            "denoising_model",
            "diffusion_model",
            "pipeline",
            "pipe",
            "predictor",
        )
        while queue:
            obj = queue.pop(0)
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if cls._looks_like_transformer(obj):
                return obj
            module = getattr(obj, "module", None)
            if module is not None and id(module) not in seen:
                queue.append(module)
            for name in attr_names:
                child = getattr(obj, name, None)
                if child is not None and id(child) not in seen:
                    queue.append(child)
        raise RuntimeError(
            "Could not locate Hunyuan transformer with double_blocks/single_blocks on sampler; "
            "the DiT-control backend requires access to the loaded Hunyuan DiT module."
        )


    @staticmethod
    def _iter_transformer_blocks(transformer: nn.Module | None, name: str) -> list[nn.Module]:
        if transformer is None:
            return []
        blocks = getattr(transformer, name, None)
        if isinstance(blocks, nn.ModuleList):
            return list(blocks)
        if isinstance(blocks, nn.Sequential):
            return list(blocks)
        if isinstance(blocks, (list, tuple)):
            return [b for b in blocks if isinstance(b, nn.Module)]
        return []

    @staticmethod
    def _align_to(value: int, multiple: int) -> int:
        return ((int(value) + multiple - 1) // multiple) * multiple

    def _latent_shape(self, *, batch: int, frames: int, height: int, width: int, adapter: HunyuanDiTControlAdapter) -> tuple[int, int, int, int, int]:
        height = self._align_to(height, 16)
        width = self._align_to(width, 16)
        latent_t = (int(frames) - 1) // 4 + 1
        return (int(batch), int(adapter.cfg.latent_channels), int(latent_t), height // 8, width // 8)

    def _adapter_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_control_adapter(self, transformer: nn.Module | None = None) -> HunyuanDiTControlAdapter:
        if self._control_adapter is not None:
            return self._control_adapter
        device = self._adapter_device()
        if self.cfg.control_ckpt:
            adapter, payload = load_hunyuan_dit_control_checkpoint(self.cfg.control_ckpt, device=device)
            self._control_adapter = adapter.eval()
            self._control_payload = payload
            return self._control_adapter
        if not self.cfg.allow_untrained_control:
            raise RuntimeError(
                "Hunyuan DiT control generation requires control_ckpt. "
                "Set allow_untrained_control=True only for smoke tests; untrained zero-init control is a no-op."
            )
        dit_hidden = int(getattr(transformer, "hidden_size", 3072)) if transformer is not None else 3072
        double_blocks = len(self._iter_transformer_blocks(transformer, "double_blocks")) if transformer is not None else 20
        single_blocks = len(self._iter_transformer_blocks(transformer, "single_blocks")) if transformer is not None else 40
        self._control_adapter = HunyuanDiTControlAdapter(
            HunyuanDiTControlConfig(
                dit_hidden=dit_hidden,
                double_blocks=double_blocks,
                single_blocks=single_blocks,
            )
        ).to(device).eval()
        self._control_payload = {
            "kind": "untrained_zero_init",
            "cfg": self._control_adapter.cfg.__dict__.copy(),
        }
        return self._control_adapter

    @staticmethod
    def _controls_from_bundle(bundle: VideoConditionBundle) -> dict[str, Any]:
        if bundle.pred_tokens is None:
            raise RuntimeError("Hunyuan DiT control backend requires bundle.pred_tokens")
        if bundle.depth is None:
            raise RuntimeError("Hunyuan DiT control backend requires bundle.depth")
        if int(bundle.pred_tokens.shape[0]) != 1:
            raise RuntimeError(
                "Hunyuan DiT-control backend v1 supports exactly one clip per generate() call; "
                "call it once per validation sample so sampler.predict(batch_size=1) stays aligned with controls."
            )
        return {
            "pred_tokens": bundle.pred_tokens,
            "depth": bundle.depth,
            "motion_hint": bundle.motion_hint,
            "contact_hint": bundle.contact_hint,
            "rough_rgb": bundle.rough_rgb,
            "context_rgb": bundle.context_rgb,
            "action_cond": bundle.action_cond,
            "task_emb": bundle.task_emb,
        }

    @torch.no_grad()
    def generate(
        self,
        bundle: VideoConditionBundle,
        *,
        num_frames: int,
        height: int,
        width: int,
        seed: int | None = None,
        **kwargs: Any,
    ) -> VideoBackendOutput:
        sampler = self.load()
        transformer = self.resolve_transformer(sampler)
        adapter = self.load_control_adapter(transformer)
        controls = self._controls_from_bundle(bundle)
        aligned_frames = align_hunyuan_video_length(num_frames)
        prompt = kwargs.pop("prompt", None) or summarize_bundle_for_prompt(bundle)

        injector = HunyuanDiTControlInjector(transformer, adapter)
        latent_shape = self._latent_shape(batch=int(controls["pred_tokens"].shape[0]), frames=aligned_frames, height=height, width=width, adapter=adapter)
        with injector.use_controls(**controls, scale=float(self.cfg.control_scale), latent_shape=latent_shape):
            outputs = sampler.predict(
                prompt=prompt,
                height=height,
                width=width,
                video_length=aligned_frames,
                seed=seed,
                infer_steps=int(kwargs.pop("infer_steps", self.cfg.infer_steps)),
                guidance_scale=float(kwargs.pop("guidance_scale", self.cfg.cfg_scale)),
                flow_shift=float(kwargs.pop("flow_shift", self.cfg.flow_shift)),
                embedded_guidance_scale=float(kwargs.pop("embedded_guidance_scale", self.cfg.embedded_cfg_scale)),
                batch_size=1,
                num_videos_per_prompt=1,
                **kwargs,
            )
        rgb_bcthw = outputs["samples"]
        rgb_btchw = rgb_bcthw.permute(0, 2, 1, 3, 4).contiguous()
        return VideoBackendOutput(
            rgb=rgb_btchw,
            metadata={
                "prompt": prompt,
                "requested_num_frames": num_frames,
                "aligned_num_frames": aligned_frames,
                "seed": outputs.get("seeds", [seed])[0],
                "backend": self.name,
                "control_active": True,
                "control_ckpt": self.cfg.control_ckpt,
                "control_scale": float(self.cfg.control_scale),
            },
        )


__all__ = [
    "HunyuanDiTControlVideoBackend",
    "HunyuanDiTControlVideoBackendConfig",
]
