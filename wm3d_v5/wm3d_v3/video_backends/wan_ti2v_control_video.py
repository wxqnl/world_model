"""Opt-in Wan2.2 TI2V backend with WM3D token control injection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

from wm3d_v3.models.wan_lora import (
    WanLoRAConfig,
    apply_lora_to_linear_modules,
    load_partial_state_dict,
    set_trainable_by_patterns,
)
from wm3d_v3.models.wan_ti2v_control_adapter import (
    WanTI2VControlAdapter,
    WanTI2VControlConfig,
    WanTI2VControlInjector,
    load_wan_ti2v_control_checkpoint,
)

try:
    from .base import VideoBackendOutput, VideoConditionBundle
except Exception:  # pragma: no cover - keeps the draft importable outside full WM3D tree
    VideoBackendOutput = Any
    VideoConditionBundle = Any


@dataclass
class WanTI2VControlVideoBackendConfig:
    repo: str = "/data/Minko/external/Wan2.2"
    checkpoint_dir: str = "/0604-10T-test/models/Wan2.2-TI2V-5B"
    task: str = "ti2v-5B"
    size: str = "1280*704"
    frame_num: int = 17
    sample_steps: int = 20
    sample_shift: float = 5.0
    sample_guide_scale: float = 5.0
    sample_solver: str = "unipc"
    offload_model: bool = False
    t5_cpu: bool = True
    t5_fsdp: bool = False
    dit_fsdp: bool = False
    use_sp: bool = False
    init_on_cpu: bool = False
    convert_model_dtype: bool = False
    control_ckpt: str | None = None
    wan_trainable_ckpt: str | None = None
    control_scale: float = 1.0
    allow_untrained_control: bool = True


class WanTI2VControlVideoBackend:
    """Thin loader/generator wrapper for the official Wan2.2 TI2V pipeline."""

    name = "wan_ti2v_control"

    def __init__(self, cfg: WanTI2VControlVideoBackendConfig | None = None, *, device: str | torch.device | None = None):
        self.cfg = cfg or WanTI2VControlVideoBackendConfig()
        self.device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._pipeline: Any | None = None
        self._control_adapter: WanTI2VControlAdapter | None = None
        self._wan_trainable_loaded_path: str | None = None
        self._wan_trainable_report: dict[str, Any] | None = None

    def _ensure_repo(self) -> None:
        repo = Path(self.cfg.repo)
        if not repo.exists():
            raise RuntimeError(f"Wan repo not found at {repo}; set train.wan_repo or --wan_repo before smoke")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

    def load(self):
        if self._pipeline is not None:
            return self._pipeline
        self._ensure_repo()
        import wan  # type: ignore
        from wan.configs import WAN_CONFIGS  # type: ignore

        cfg = WAN_CONFIGS[self.cfg.task]
        pipeline = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=self.cfg.checkpoint_dir,
            device_id=int(self.device.index or 0),
            rank=int(torch.distributed.get_rank()) if torch.distributed.is_available() and torch.distributed.is_initialized() else 0,
            t5_fsdp=bool(self.cfg.t5_fsdp),
            dit_fsdp=bool(self.cfg.dit_fsdp),
            use_sp=bool(self.cfg.use_sp),
            t5_cpu=bool(self.cfg.t5_cpu),
            init_on_cpu=bool(self.cfg.init_on_cpu),
            convert_model_dtype=bool(self.cfg.convert_model_dtype),
        )
        self._pipeline = pipeline
        return pipeline

    @staticmethod
    def _looks_like_transformer(obj: Any) -> bool:
        return isinstance(obj, nn.Module) and hasattr(obj, "blocks")

    @classmethod
    def resolve_transformer(cls, pipeline: Any) -> nn.Module:
        queue: list[Any] = [pipeline]
        seen: set[int] = set()
        attr_names = ("model", "transformer", "dit", "diffusion_model", "pipeline", "pipe")
        while queue:
            obj = queue.pop(0)
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if cls._looks_like_transformer(obj):
                return obj
            module = getattr(obj, "module", None)
            if module is not None:
                queue.append(module)
            for name in attr_names:
                child = getattr(obj, name, None)
                if child is not None:
                    queue.append(child)
        raise RuntimeError("Could not locate Wan transformer; expected a module with .blocks")

    @staticmethod
    def iter_transformer_blocks(transformer: nn.Module | None) -> list[nn.Module]:
        if transformer is None:
            return []
        blocks = getattr(transformer, "blocks", None)
        if isinstance(blocks, nn.ModuleList):
            return list(blocks)
        if isinstance(blocks, nn.Sequential):
            return list(blocks)
        if isinstance(blocks, (list, tuple)):
            return [b for b in blocks if isinstance(b, nn.Module)]
        return []

    @staticmethod
    def resolve_vae(pipeline: Any) -> Any:
        vae = getattr(pipeline, "vae", None)
        if vae is None:
            raise RuntimeError("Wan pipeline has no .vae")
        return vae

    @staticmethod
    def resolve_text_encoder(pipeline: Any) -> Any:
        text = getattr(pipeline, "text_encoder", None)
        if text is None:
            raise RuntimeError("Wan pipeline has no .text_encoder")
        return text

    def load_wan_trainable_state(self, transformer: nn.Module) -> dict[str, Any] | None:
        path = self.cfg.wan_trainable_ckpt
        if not path:
            return None
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            state = payload.get("state_dict", payload)
            if not isinstance(state, dict):
                raise RuntimeError(f"Wan trainable checkpoint {path} did not contain a state dict")
            has_lora_state = any("lora_A" in name or "lora_B" in name for name in state)
            lora_cfg = payload.get("lora_config")
            if (
                has_lora_state
                and isinstance(lora_cfg, dict)
                and not any("lora_A" in name or "lora_B" in name for name, _ in transformer.named_parameters())
            ):
                apply_lora_to_linear_modules(
                    transformer,
                    WanLoRAConfig(
                        rank=int(lora_cfg.get("rank", 8)),
                        alpha=float(lora_cfg.get("alpha", 16.0)),
                        dropout=float(lora_cfg.get("dropout", 0.0)),
                        include=tuple(lora_cfg.get("include", ("blocks",))),
                        exclude=tuple(lora_cfg.get("exclude", ())),
                        dtype=str(lora_cfg.get("dtype", "bf16")),
                        checkpoint=bool(lora_cfg.get("checkpoint", False)),
                        checkpoint_use_reentrant=bool(lora_cfg.get("checkpoint_use_reentrant", False)),
                    ),
                )
            patterns = payload.get("partial_unfreeze") or ()
            if patterns:
                set_trainable_by_patterns(transformer, patterns, ())
        else:
            state = payload
        if not isinstance(state, dict):
            raise RuntimeError(f"Wan trainable checkpoint {path} did not contain a state dict")
        report = load_partial_state_dict(transformer, state)
        self._wan_trainable_loaded_path = str(path)
        self._wan_trainable_report = report
        return {"path": str(path), **report}

    def load_control_adapter(self, transformer: nn.Module | None = None) -> WanTI2VControlAdapter:
        if self._control_adapter is not None:
            return self._control_adapter
        if self.cfg.control_ckpt:
            adapter, report = load_wan_ti2v_control_checkpoint(self.cfg.control_ckpt, device=self.device)
            self._control_adapter = adapter
            return adapter
        if not self.cfg.allow_untrained_control:
            raise RuntimeError("Wan control_ckpt is required unless allow_untrained_control=true")
        blocks = self.iter_transformer_blocks(transformer) if transformer is not None else []
        hidden = int(getattr(transformer, "dim", getattr(getattr(transformer, "config", None), "dim", 3072))) if transformer is not None else 3072
        adapter = WanTI2VControlAdapter(
            WanTI2VControlConfig(
                dit_hidden=hidden,
                num_layers=max(1, len(blocks) or 30),
            )
        ).to(self.device)
        self._control_adapter = adapter
        return adapter

    @staticmethod
    def _controls_from_bundle(bundle: VideoConditionBundle) -> dict[str, Any]:
        pred_tokens = getattr(bundle, "tokens", None)
        if pred_tokens is None:
            pred_tokens = getattr(bundle, "pred_tokens", None)
        return {
            "pred_tokens": pred_tokens,
            "depth": getattr(bundle, "depth", None),
            "motion_hint": getattr(bundle, "motion_hint", None),
            "contact_hint": getattr(bundle, "contact_hint", None),
            "context_rgb": getattr(bundle, "context_rgb", None),
            "action_cond": getattr(bundle, "action_cond", None),
            "task_emb": getattr(bundle, "task_emb", None),
            "point": getattr(bundle, "point", None),
            "pose_geom": getattr(bundle, "pose_geom", None),
        }

    @torch.no_grad()
    def generate(
        self,
        bundle: VideoConditionBundle,
        *,
        prompt: str = "robot manipulation scene, close-up tabletop robot arm manipulating an object",
        seed: int = 0,
        size: tuple[int, int] | None = None,
        frame_num: int | None = None,
        sample_steps: int | None = None,
    ) -> VideoBackendOutput:
        """Generate with control hooks installed around the Wan sampling loop.

        This method is for future integration smoke only. Training uses direct
        transformer forwards from ``scripts/train_stage0_wan_ti2v_joint_body.py``
        to avoid invoking the full sampler.
        """

        pipeline = self.load()
        transformer = self.resolve_transformer(pipeline)
        self.load_wan_trainable_state(transformer)
        adapter = self.load_control_adapter(transformer)
        injector = WanTI2VControlInjector(transformer, adapter)
        controls = self._controls_from_bundle(bundle)
        if controls["pred_tokens"] is None:
            raise ValueError("Wan control generation requires bundle.tokens or bundle.pred_tokens")
        injector.install()
        try:
            adapter.prepare_controls(**controls, scale=float(self.cfg.control_scale))
            video = pipeline.generate(
                prompt,
                img=None,
                size=size or (1280, 704),
                max_area=(size[0] * size[1] if size is not None else 1280 * 704),
                frame_num=int(frame_num or self.cfg.frame_num),
                shift=float(self.cfg.sample_shift),
                sample_solver=str(self.cfg.sample_solver),
                sampling_steps=int(sample_steps or self.cfg.sample_steps),
                guide_scale=float(self.cfg.sample_guide_scale),
                seed=int(seed),
                offload_model=bool(self.cfg.offload_model),
            )
        finally:
            adapter.clear_control_state()
            injector.remove()
        try:
            return VideoBackendOutput(video=video, metadata={"backend": self.name})
        except TypeError:
            return {"video": video, "metadata": {"backend": self.name}}
