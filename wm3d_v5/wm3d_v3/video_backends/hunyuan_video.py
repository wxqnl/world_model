"""HunyuanVideo backend plumbing for wm3d_v3.

This is the first integration layer: it loads the local HunyuanVideo T2V
sampler and exposes it through the project video-backend interface. Structured
wm3d controls are summarized into the prompt for now; trainable control adapter
injection is the next step and should keep this public interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import torch

from .base import VideoBackendOutput, VideoConditionBundle


def align_hunyuan_video_length(num_frames: int) -> int:
    """Hunyuan 884 VAE requires video_length == 4n + 1."""
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames == 1 or (num_frames - 1) % 4 == 0:
        return num_frames
    return ((num_frames - 1 + 3) // 4) * 4 + 1


def summarize_bundle_for_prompt(bundle: VideoConditionBundle) -> str:
    """Create a small text hint from structured controls for T2V smoke tests."""
    prompt = bundle.task_text[0].strip() if bundle.task_text else ""
    if not prompt:
        prompt = "close-up robot manipulation scene on a tabletop"
    details: list[str] = []
    if bundle.motion_hint is not None:
        motion = float(bundle.motion_hint.detach().float().mean().cpu())
        if motion > 0.18:
            details.append("visible object and gripper motion")
        elif motion > 0.06:
            details.append("moderate robot motion")
        else:
            details.append("subtle robot motion")
    if bundle.contact_hint is not None:
        contact = float(bundle.contact_hint.detach().float().mean().cpu())
        if contact > 0.08:
            details.append("robot gripper contacting an object")
    if details:
        prompt = f"{prompt}, {', '.join(details)}"
    return prompt


@dataclass
class HunyuanVideoBackendConfig:
    external_repo: str = "/data/Minko/external/HunyuanVideo"
    model_base: str = "/data/Minko/models/hunyuan_video"
    dit_weight: str = "/data/Minko/models/hunyuan_video/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
    model_resolution: str = "540p"
    precision: str = "bf16"
    vae_precision: str = "fp16"
    text_encoder_precision: str = "fp16"
    text_encoder_precision_2: str = "fp16"
    use_fp8: bool = True
    use_cpu_offload: bool = False
    infer_steps: int = 8
    cfg_scale: float = 1.0
    embedded_cfg_scale: float = 6.0
    flow_shift: float = 7.0


class HunyuanVideoBackend:
    """Lazy-loading HunyuanVideo T2V backend.

    The local HunyuanVideo checkpoint is a text-to-video model. Until a
    trainable control adapter is added, structured wm3d controls are not injected
    into the DiT; they are retained in metadata and lightly summarized into text
    so the backend can be smoke-tested end to end.
    """

    name = "hunyuan_video_t2v"

    def __init__(self, cfg: HunyuanVideoBackendConfig | None = None, *, device: str | torch.device | None = None):
        self.cfg = cfg or HunyuanVideoBackendConfig()
        self.device = device
        self._sampler = None

    def _argv(self) -> list[str]:
        cfg = self.cfg
        argv = [
            "--model-base", cfg.model_base,
            "--dit-weight", cfg.dit_weight,
            "--model-resolution", cfg.model_resolution,
            "--precision", cfg.precision,
            "--vae-precision", cfg.vae_precision,
            "--text-encoder-precision", cfg.text_encoder_precision,
            "--text-encoder-precision-2", cfg.text_encoder_precision_2,
            "--infer-steps", str(cfg.infer_steps),
            "--cfg-scale", str(cfg.cfg_scale),
            "--embedded-cfg-scale", str(cfg.embedded_cfg_scale),
            "--flow-shift", str(cfg.flow_shift),
        ]
        if cfg.use_fp8:
            argv.append("--use-fp8")
        if cfg.use_cpu_offload:
            argv.append("--use-cpu-offload")
        return argv

    @staticmethod
    def _patch_text_encoder_loader() -> None:
        """Handle transformers versions where AutoModel returns full LlavaModel."""
        import hyvideo.text_encoder as text_encoder_mod  # type: ignore
        from hyvideo.constants import PRECISION_TO_TYPE, TEXT_ENCODER_PATH  # type: ignore
        from transformers import AutoModel, CLIPTextModel  # type: ignore

        if getattr(text_encoder_mod.load_text_encoder, "_wm3d_patched", False):
            return

        def load_text_encoder(
            text_encoder_type,
            text_encoder_precision=None,
            text_encoder_path=None,
            logger=None,
            device=None,
        ):
            if text_encoder_path is None:
                text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
            if logger is not None:
                logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")

            if text_encoder_type == "clipL":
                text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
                if hasattr(text_encoder, "text_model") and hasattr(text_encoder.text_model, "final_layer_norm"):
                    text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
                elif not hasattr(text_encoder, "final_layer_norm"):
                    raise AttributeError(f"CLIP text encoder has no final_layer_norm: {type(text_encoder).__name__}")
            elif text_encoder_type == "llm":
                try:
                    text_encoder = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True)
                    if hasattr(text_encoder, "language_model"):
                        text_encoder = text_encoder.language_model
                except ValueError:
                    from transformers import LlavaForConditionalGeneration  # type: ignore

                    full_model = LlavaForConditionalGeneration.from_pretrained(
                        text_encoder_path,
                        low_cpu_mem_usage=True,
                    )
                    text_encoder = full_model.language_model
                if hasattr(text_encoder, "norm"):
                    text_encoder.final_layer_norm = text_encoder.norm
                elif hasattr(text_encoder, "model") and hasattr(text_encoder.model, "norm"):
                    text_encoder.final_layer_norm = text_encoder.model.norm
                else:
                    raise AttributeError(f"text encoder has no norm attribute: {type(text_encoder).__name__}")
            else:
                raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

            if text_encoder_precision is not None:
                text_encoder = text_encoder.to(dtype=PRECISION_TO_TYPE[text_encoder_precision])
            text_encoder.requires_grad_(False)
            if logger is not None:
                logger.info(f"Text encoder to dtype: {text_encoder.dtype}")
            if device is not None:
                text_encoder = text_encoder.to(device)
            return text_encoder, text_encoder_path

        load_text_encoder._wm3d_patched = True  # type: ignore[attr-defined]
        text_encoder_mod.load_text_encoder = load_text_encoder

    def load(self):
        if self._sampler is not None:
            return self._sampler
        cfg = self.cfg
        repo = Path(cfg.external_repo)
        model_base = Path(cfg.model_base)
        if not repo.exists():
            raise FileNotFoundError(f"HunyuanVideo repo not found: {repo}")
        if not model_base.exists():
            raise FileNotFoundError(f"HunyuanVideo model_base not found: {model_base}")
        if not Path(cfg.dit_weight).exists():
            raise FileNotFoundError(f"HunyuanVideo dit_weight not found: {cfg.dit_weight}")

        os.environ.setdefault("MODEL_BASE", str(model_base))
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        from hyvideo.config import parse_args  # type: ignore
        from hyvideo.inference import HunyuanVideoSampler  # type: ignore
        self._patch_text_encoder_loader()

        old_argv = sys.argv
        try:
            sys.argv = ["wm3d_hunyuan_backend", *self._argv()]
            args = parse_args()
        finally:
            sys.argv = old_argv
        device = self.device
        self._sampler = HunyuanVideoSampler.from_pretrained(model_base, args=args, device=device)
        return self._sampler

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
        aligned_frames = align_hunyuan_video_length(num_frames)
        prompt = kwargs.pop("prompt", None) or summarize_bundle_for_prompt(bundle)
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
            },
        )
