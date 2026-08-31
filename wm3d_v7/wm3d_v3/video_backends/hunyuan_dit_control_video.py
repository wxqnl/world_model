"""Opt-in HunyuanVideo backend with DiT image-token control injection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from wm3d_v3.models.hunyuan_dit_control_adapter import (
    HunyuanDiTControlAdapter,
    HunyuanDiTControlConfig,
    HunyuanDiTControlInjector,
    load_hunyuan_dit_control_checkpoint,
)
from wm3d_v3.models.hunyuan_lora import load_hunyuan_trainable_checkpoint

from .base import VideoBackendOutput, VideoConditionBundle
from .hunyuan_video import HunyuanVideoBackend, HunyuanVideoBackendConfig, align_hunyuan_video_length, summarize_bundle_for_prompt


@dataclass
class HunyuanDiTControlVideoBackendConfig(HunyuanVideoBackendConfig):
    control_ckpt: str | None = None
    hunyuan_trainable_ckpt: str | None = None
    control_scale: float = 1.0
    double_pre_control_scale: float = 0.0
    single_pre_control_scale: float = 0.0
    rough_init_strength: float = 0.0
    rough_init_noise: float = 0.0
    context_init_strength: float = 0.0
    context_init_noise: float = 0.0
    context_clamp_strength: float = 0.0
    context_clamp_until: float = 1.0
    context_clamp_decay: str = "linear"
    context_clamp_noise: float = 0.0
    context_clamp_mask_source: str = "none"
    context_clamp_motion_threshold: float = 0.15
    context_clamp_dilate: int = 0
    context_clamp_dynamic_floor: float = 0.0
    final_velocity_residual_scale: float = 0.0
    final_velocity_residual_mask_source: str = "none"
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
        self._hunyuan_trainable_loaded_path: str | None = None
        self._hunyuan_trainable_report: dict[str, Any] | None = None

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

    @staticmethod
    def _resize_btchw(video: torch.Tensor, *, frames: int, height: int, width: int) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"expected video [B,T,C,H,W], got {tuple(video.shape)}")
        b, t, c, h, w = video.shape
        if c != 3:
            raise ValueError(f"rough init video must have 3 channels, got {c}")
        if t < frames:
            pad = video[:, -1:].expand(b, frames - t, c, h, w)
            video = torch.cat([video, pad], dim=1)
        elif t > frames:
            video = video[:, :frames]
        if tuple(video.shape[-2:]) != (height, width):
            x = video.permute(0, 2, 1, 3, 4).contiguous()
            x = F.interpolate(x, size=(frames, height, width), mode="trilinear", align_corners=False)
            video = x.permute(0, 2, 1, 3, 4).contiguous()
        return video.clamp(0.0, 1.0)

    @torch.no_grad()
    def _rough_init_latents(
        self,
        sampler: Any,
        bundle: VideoConditionBundle,
        *,
        frames: int,
        height: int,
        width: int,
        seed: int | None,
    ) -> torch.Tensor | None:
        strength = float(getattr(self.cfg, "rough_init_strength", 0.0) or 0.0)
        if strength <= 0.0 or bundle.rough_rgb is None or bundle.context_rgb is None:
            return None
        strength = max(0.0, min(1.0, strength))
        device = self._adapter_device()
        context = bundle.context_rgb.detach().to(device=device, dtype=torch.float32)
        rough = bundle.rough_rgb.detach().to(device=device, dtype=torch.float32)
        if context.ndim != 4:
            raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context.shape)}")
        if rough.ndim != 5:
            raise ValueError(f"rough_rgb must be [B,T,3,H,W], got {tuple(rough.shape)}")
        video_btchw = torch.cat([context[:, None], rough], dim=1)
        video_btchw = self._resize_btchw(video_btchw, frames=int(frames), height=int(height), width=int(width))
        video_bcthw = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        vae = sampler.pipeline.vae
        x = video_bcthw.mul(2.0).sub(1.0).to(device=next(vae.parameters()).device, dtype=vae.dtype)
        posterior = vae.encode(x).latent_dist
        latents = posterior.mode() * float(vae.config.scaling_factor)
        noise_scale = float(getattr(self.cfg, "rough_init_noise", 0.0) or 0.0)
        if strength < 1.0 or noise_scale > 0.0:
            gen = torch.Generator(device=latents.device)
            if seed is not None:
                gen.manual_seed(int(seed))
            noise = torch.randn(latents.shape, generator=gen, device=latents.device, dtype=latents.dtype)
            latents = strength * latents + (1.0 - strength + noise_scale) * noise
        return latents


    @torch.no_grad()
    def _context_init_latents(
        self,
        sampler: Any,
        bundle: VideoConditionBundle,
        *,
        frames: int,
        height: int,
        width: int,
        seed: int | None,
    ) -> torch.Tensor | None:
        strength = float(getattr(self.cfg, "context_init_strength", 0.0) or 0.0)
        if strength <= 0.0 or bundle.context_rgb is None:
            return None
        strength = max(0.0, min(1.0, strength))
        device = self._adapter_device()
        context = bundle.context_rgb.detach().to(device=device, dtype=torch.float32)
        if context.ndim != 4:
            raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context.shape)}")
        video_btchw = context[:, None].expand(-1, int(frames), -1, -1, -1).contiguous()
        video_btchw = self._resize_btchw(video_btchw, frames=int(frames), height=int(height), width=int(width))
        video_bcthw = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        vae = sampler.pipeline.vae
        x = video_bcthw.mul(2.0).sub(1.0).to(device=next(vae.parameters()).device, dtype=vae.dtype)
        posterior = vae.encode(x).latent_dist
        latents = posterior.mode() * float(vae.config.scaling_factor)
        noise_scale = float(getattr(self.cfg, "context_init_noise", 0.0) or 0.0)
        if strength < 1.0 or noise_scale > 0.0:
            gen = torch.Generator(device=latents.device)
            if seed is not None:
                gen.manual_seed(int(seed))
            noise = torch.randn(latents.shape, generator=gen, device=latents.device, dtype=latents.dtype)
            latents = strength * latents + (1.0 - strength + noise_scale) * noise
        return latents

    @torch.no_grad()
    def _context_reference_latents(
        self,
        sampler: Any,
        bundle: VideoConditionBundle,
        *,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        if bundle.context_rgb is None:
            return None
        device = self._adapter_device()
        context = bundle.context_rgb.detach().to(device=device, dtype=torch.float32)
        if context.ndim != 4:
            raise ValueError(f"context_rgb must be [B,3,H,W], got {tuple(context.shape)}")
        video_btchw = context[:, None].expand(-1, int(frames), -1, -1, -1).contiguous()
        video_btchw = self._resize_btchw(video_btchw, frames=int(frames), height=int(height), width=int(width))
        video_bcthw = video_btchw.permute(0, 2, 1, 3, 4).contiguous()
        vae = sampler.pipeline.vae
        x = video_bcthw.mul(2.0).sub(1.0).to(device=next(vae.parameters()).device, dtype=vae.dtype)
        posterior = vae.encode(x).latent_dist
        return posterior.mode() * float(vae.config.scaling_factor)

    @staticmethod
    def _as_b1t_hw(x: torch.Tensor, *, latent_t: int) -> torch.Tensor:
        if x.ndim == 5:
            if x.shape[1] == 1:
                out = x
            elif x.shape[2] == latent_t:
                out = x.mean(dim=1, keepdim=True)
            else:
                out = x.mean(dim=2, keepdim=False)[:, None]
        elif x.ndim == 4:
            out = x[:, None]
        elif x.ndim == 3:
            out = x[:, None, None]
        else:
            raise ValueError(f"expected 3D/4D/5D hint tensor, got {tuple(x.shape)}")
        if out.shape[2] != latent_t:
            if out.shape[2] == 1:
                out = out.expand(-1, -1, latent_t, -1, -1)
            else:
                out = F.interpolate(out, size=(latent_t, out.shape[-2], out.shape[-1]), mode="trilinear", align_corners=False)
        return out

    @torch.no_grad()
    def _context_clamp_mask(self, controls: dict[str, Any], *, latent_shape: tuple[int, int, int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        source = str(getattr(self.cfg, "context_clamp_mask_source", "none") or "none").strip().lower()
        if source in {"", "none", "off"}:
            return None
        b, _c, latent_t, latent_h, latent_w = latent_shape
        if source in {"all", "full"}:
            return torch.ones((b, 1, latent_t, latent_h, latent_w), device=device, dtype=dtype)

        hints: list[torch.Tensor] = []
        if source in {"motion", "motion_depth", "depth_motion"} and controls.get("motion_hint") is not None:
            hints.append(self._as_b1t_hw(controls["motion_hint"].detach().abs(), latent_t=latent_t))
        if source in {"depth", "motion_depth", "depth_motion"} and controls.get("depth") is not None:
            depth = self._as_b1t_hw(controls["depth"].detach(), latent_t=latent_t)
            from_first = (depth - depth[:, :, :1]).abs()
            step = torch.zeros_like(depth)
            if int(depth.shape[2]) > 1:
                step[:, :, 1:] = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
            hints.append(0.5 * from_first + 0.5 * step)
        if not hints:
            return None

        hint = torch.stack([h.to(device=device, dtype=dtype) for h in hints], dim=0).amax(dim=0)
        hint = F.interpolate(hint, size=(latent_t, latent_h, latent_w), mode="trilinear", align_corners=False)
        denom = hint.flatten(2).amax(dim=2).view(hint.shape[0], 1, 1, 1, 1).clamp_min(1e-6)
        threshold = float(getattr(self.cfg, "context_clamp_motion_threshold", 0.15) or 0.15)
        dynamic = ((hint / denom) - threshold).clamp_min(0.0) / max(1e-6, 1.0 - threshold)
        dynamic = dynamic.clamp(0.0, 1.0)
        dilate = max(0, int(getattr(self.cfg, "context_clamp_dilate", 0) or 0))
        if dilate > 0:
            k = 2 * dilate + 1
            dynamic = F.max_pool3d(dynamic, kernel_size=k, stride=1, padding=dilate)
        static = 1.0 - dynamic
        dynamic_floor = min(1.0, max(0.0, float(getattr(self.cfg, "context_clamp_dynamic_floor", 0.0) or 0.0)))
        if dynamic_floor > 0.0:
            static = static + dynamic_floor * (1.0 - static)
        return static.clamp(0.0, 1.0)

    @staticmethod
    def _context_clamp_weight(step: int, num_steps: int, *, strength: float, until: float, decay: str) -> float:
        strength = min(1.0, max(0.0, float(strength)))
        if strength <= 0.0:
            return 0.0
        until = min(1.0, max(1e-6, float(until)))
        denom = max(1, int(num_steps) - 1)
        frac = min(1.0, max(0.0, float(step) / float(denom)))
        if frac > until:
            return 0.0
        rel = frac / until
        mode = str(decay or "linear").strip().lower()
        if mode == "constant":
            scale = 1.0
        elif mode == "cosine":
            scale = 0.5 * (1.0 + math.cos(math.pi * rel))
        elif mode == "linear":
            scale = 1.0 - rel
        else:
            raise ValueError(f"unsupported context_clamp_decay={decay!r}")
        return strength * max(0.0, min(1.0, scale))

    def _make_context_clamp_callback(
        self,
        context_latents: torch.Tensor | None,
        controls: dict[str, Any],
        *,
        latent_shape: tuple[int, int, int, int, int],
        num_steps: int,
        seed: int | None,
    ):
        strength = float(getattr(self.cfg, "context_clamp_strength", 0.0) or 0.0)
        if strength <= 0.0 or context_latents is None:
            return None
        noise_strength = min(1.0, max(0.0, float(getattr(self.cfg, "context_clamp_noise", 0.0) or 0.0)))
        anchor_noise = None
        if noise_strength > 0.0:
            gen = torch.Generator(device=context_latents.device)
            if seed is not None:
                gen.manual_seed(int(seed) + 17017)
            anchor_noise = torch.randn(
                context_latents.shape,
                generator=gen,
                device=context_latents.device,
                dtype=context_latents.dtype,
            )
        mask = self._context_clamp_mask(
            controls,
            latent_shape=latent_shape,
            device=context_latents.device,
            dtype=context_latents.dtype,
        )

        def _callback(_pipeline, step: int, _timestep, callback_kwargs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            latents = callback_kwargs["latents"]
            ref = context_latents.to(device=latents.device, dtype=latents.dtype)
            if anchor_noise is not None:
                sigmas = getattr(getattr(_pipeline, "scheduler", None), "sigmas", None)
                step_index = getattr(getattr(_pipeline, "scheduler", None), "step_index", None)
                if sigmas is not None and len(sigmas) > 0:
                    idx = int(step_index) if step_index is not None else int(step) + 1
                    idx = max(0, min(int(idx), int(len(sigmas) - 1)))
                    sigma = sigmas[idx].to(device=latents.device, dtype=latents.dtype).view(1, 1, 1, 1, 1)
                else:
                    frac = 1.0 - min(1.0, max(0.0, float(step) / float(max(1, int(num_steps) - 1))))
                    sigma = latents.new_tensor(frac).view(1, 1, 1, 1, 1)
                ref_noise = anchor_noise.to(device=latents.device, dtype=latents.dtype)
                noise_mix = sigma.clamp(0.0, 1.0) * noise_strength
                ref = ref * (1.0 - noise_mix) + ref_noise * noise_mix
            weight = self._context_clamp_weight(
                int(step),
                int(num_steps),
                strength=strength,
                until=float(getattr(self.cfg, "context_clamp_until", 1.0) or 1.0),
                decay=str(getattr(self.cfg, "context_clamp_decay", "linear") or "linear"),
            )
            if weight <= 0.0:
                return {"latents": latents}
            if mask is None:
                blend = latents.new_tensor(weight)
            else:
                blend = mask.to(device=latents.device, dtype=latents.dtype) * weight
            return {"latents": latents * (1.0 - blend) + ref * blend}

        return _callback

    @torch.no_grad()
    def _final_velocity_residual_mask(
        self,
        controls: dict[str, Any],
        *,
        latent_shape: tuple[int, int, int, int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        source = str(getattr(self.cfg, "final_velocity_residual_mask_source", "none") or "none").strip().lower()
        if source in {"", "none", "off"}:
            return None
        if source != "context_dynamic":
            raise ValueError(f"unsupported final_velocity_residual_mask_source={source!r}")
        static_mask = self._context_clamp_mask(
            controls,
            latent_shape=latent_shape,
            device=device,
            dtype=dtype,
        )
        if static_mask is None:
            return None
        return (1.0 - static_mask).clamp(0.0, 1.0)


    def load_hunyuan_trainable_state(self, transformer: nn.Module) -> dict[str, Any] | None:
        ckpt = getattr(self.cfg, "hunyuan_trainable_ckpt", None)
        if not ckpt:
            return None
        ckpt_path = str(ckpt)
        if self._hunyuan_trainable_loaded_path == ckpt_path:
            return self._hunyuan_trainable_report
        report = load_hunyuan_trainable_checkpoint(transformer, ckpt_path, map_location=self._adapter_device())
        self._hunyuan_trainable_loaded_path = ckpt_path
        self._hunyuan_trainable_report = report
        return report

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
        extra = bundle.extra or {}
        return {
            "pred_tokens": bundle.pred_tokens,
            "depth": bundle.depth,
            "motion_hint": bundle.motion_hint,
            "contact_hint": bundle.contact_hint,
            "rough_rgb": bundle.rough_rgb,
            "context_rgb": bundle.context_rgb,
            "action_cond": bundle.action_cond,
            "task_emb": bundle.task_emb,
            "point": extra.get("point"),
            "pose_geom": extra.get("pose_geom"),
            "rgb_motion_features": extra.get("rgb_motion_features"),
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
        hunyuan_trainable_report = self.load_hunyuan_trainable_state(transformer)
        adapter = self.load_control_adapter(transformer)
        controls = self._controls_from_bundle(bundle)
        aligned_frames = align_hunyuan_video_length(num_frames)
        prompt = kwargs.pop("prompt", None) or summarize_bundle_for_prompt(bundle)

        injector = HunyuanDiTControlInjector(transformer, adapter)
        latent_shape = self._latent_shape(batch=int(controls["pred_tokens"].shape[0]), frames=aligned_frames, height=height, width=width, adapter=adapter)
        init_latents = self._rough_init_latents(
            sampler,
            bundle,
            frames=aligned_frames,
            height=height,
            width=width,
            seed=seed,
        )
        if init_latents is None:
            init_latents = self._context_init_latents(
                sampler,
                bundle,
                frames=aligned_frames,
                height=height,
                width=width,
                seed=seed,
            )
        source_latents = init_latents
        if source_latents is None and bool(getattr(adapter.cfg, "use_source_latents", False)):
            source_latents = self._context_reference_latents(
                sampler,
                bundle,
                frames=aligned_frames,
                height=height,
                width=width,
            )
        final_residual_mask = self._final_velocity_residual_mask(
            controls,
            latent_shape=latent_shape,
            device=(init_latents.device if init_latents is not None else source_latents.device if source_latents is not None else bundle.context_rgb.device),
            dtype=(init_latents.dtype if init_latents is not None else source_latents.dtype if source_latents is not None else bundle.context_rgb.dtype),
        )
        with injector.use_controls(
            **controls,
            source_latents=source_latents,
            scale=float(self.cfg.control_scale),
            latent_shape=latent_shape,
            double_pre_control_scale=float(getattr(self.cfg, "double_pre_control_scale", 0.0) or 0.0),
            single_pre_control_scale=float(getattr(self.cfg, "single_pre_control_scale", 0.0) or 0.0),
            final_velocity_residual_scale=float(getattr(self.cfg, "final_velocity_residual_scale", 0.0) or 0.0),
            final_velocity_residual_mask=final_residual_mask,
        ):
            infer_steps = int(kwargs.pop("infer_steps", self.cfg.infer_steps))
            context_ref = self._context_reference_latents(
                sampler,
                bundle,
                frames=aligned_frames,
                height=height,
                width=width,
            )
            clamp_callback = self._make_context_clamp_callback(
                context_ref,
                controls,
                latent_shape=latent_shape,
                num_steps=infer_steps,
                seed=seed,
            )
            outputs = sampler.predict(
                prompt=prompt,
                height=height,
                width=width,
                video_length=aligned_frames,
                seed=seed,
                latents=init_latents,
                infer_steps=infer_steps,
                guidance_scale=float(kwargs.pop("guidance_scale", self.cfg.cfg_scale)),
                flow_shift=float(kwargs.pop("flow_shift", self.cfg.flow_shift)),
                embedded_guidance_scale=float(kwargs.pop("embedded_guidance_scale", self.cfg.embedded_cfg_scale)),
                callback_on_step_end=clamp_callback,
                callback_on_step_end_tensor_inputs=["latents"],
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
                "hunyuan_trainable_ckpt": self.cfg.hunyuan_trainable_ckpt,
                "hunyuan_trainable_report": hunyuan_trainable_report,
                "control_scale": float(self.cfg.control_scale),
                "double_pre_control_scale": float(getattr(self.cfg, "double_pre_control_scale", 0.0) or 0.0),
                "single_pre_control_scale": float(getattr(self.cfg, "single_pre_control_scale", 0.0) or 0.0),
                "rough_init_strength": float(getattr(self.cfg, "rough_init_strength", 0.0) or 0.0),
                "context_init_strength": float(getattr(self.cfg, "context_init_strength", 0.0) or 0.0),
                "context_clamp_strength": float(getattr(self.cfg, "context_clamp_strength", 0.0) or 0.0),
                "context_clamp_noise": float(getattr(self.cfg, "context_clamp_noise", 0.0) or 0.0),
                "context_clamp_mask_source": str(getattr(self.cfg, "context_clamp_mask_source", "none") or "none"),
                "final_velocity_residual_scale": float(getattr(self.cfg, "final_velocity_residual_scale", 0.0) or 0.0),
                "final_velocity_residual_mask_source": str(getattr(self.cfg, "final_velocity_residual_mask_source", "none") or "none"),
                "init_latents_active": init_latents is not None,
            },
        )


__all__ = [
    "HunyuanDiTControlVideoBackend",
    "HunyuanDiTControlVideoBackendConfig",
]
