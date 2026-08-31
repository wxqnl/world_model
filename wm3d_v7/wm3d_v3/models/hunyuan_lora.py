
"""Small LoRA/partial-train helpers for Hunyuan DiT integration.

The helpers are intentionally dependency-free so training and inference can both
apply the same adapter structure before loading trainable Hunyuan state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


HUNYUAN_TRAINABLE_CHECKPOINT_KIND = "hunyuan_dit_trainable_state_v1"


@dataclass
class HunyuanLoRAConfig:
    rank: int = 0
    alpha: float = 16.0
    dropout: float = 0.0
    include: tuple[str, ...] = ("double_blocks", "single_blocks")
    exclude: tuple[str, ...] = ()
    dtype: str = "fp32"
    checkpoint: bool = False
    checkpoint_use_reentrant: bool = False

    @classmethod
    def from_any(cls, value: Any) -> "HunyuanLoRAConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls(rank=0)
        if hasattr(value, "__dict__") and not isinstance(value, dict):
            value = vars(value)
        if not isinstance(value, dict):
            raise TypeError(f"LoRA config must be dict-like, got {type(value).__name__}")
        data = dict(value)
        data["include"] = _tuple_csv(data.get("include", ("double_blocks", "single_blocks")))
        data["exclude"] = _tuple_csv(data.get("exclude", ()))
        return cls(**data)


def _tuple_csv(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part).strip() for part in value if str(part).strip())


def _dtype_from_name(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in {"fp32", "float32"}:
        return torch.float32
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported LoRA dtype {name!r}")


def _matches(name: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    inc = tuple(include)
    exc = tuple(exclude)
    if inc and not any(p in name for p in inc):
        return False
    if exc and any(p in name for p in exc):
        return False
    return True


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        checkpoint: bool = False,
        checkpoint_use_reentrant: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRALinear rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.use_checkpoint = bool(checkpoint)
        self.checkpoint_use_reentrant = bool(checkpoint_use_reentrant)
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=base.weight.device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=base.weight.device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)

        def lora_delta(inp: torch.Tensor) -> torch.Tensor:
            x_lora = self.dropout(inp).to(dtype=self.lora_A.dtype)
            return F.linear(F.linear(x_lora, self.lora_A), self.lora_B) * self.scaling

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            delta = activation_checkpoint(lora_delta, x, use_reentrant=self.checkpoint_use_reentrant)
        else:
            delta = lora_delta(x)
        return out + delta.to(device=out.device, dtype=out.dtype)


def apply_lora_to_linear_modules(root: nn.Module, cfg: HunyuanLoRAConfig | dict[str, Any] | None) -> dict[str, Any]:
    cfg = HunyuanLoRAConfig.from_any(cfg)
    if int(cfg.rank) <= 0:
        return {"enabled": False, "modules": 0, "params": 0, "config": asdict(cfg)}
    dtype = _dtype_from_name(cfg.dtype)
    replaced: list[str] = []

    def visit(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and _matches(full, cfg.include, cfg.exclude):
                wrapped = LoRALinear(
                    child,
                    rank=int(cfg.rank),
                    alpha=float(cfg.alpha),
                    dropout=float(cfg.dropout),
                    dtype=dtype,
                    checkpoint=bool(cfg.checkpoint),
                    checkpoint_use_reentrant=bool(cfg.checkpoint_use_reentrant),
                )
                setattr(module, child_name, wrapped)
                replaced.append(full)
            else:
                visit(child, full)

    visit(root, "")
    params = sum(p.numel() for name, p in root.named_parameters() if ".lora_" in name)
    return {"enabled": True, "modules": len(replaced), "params": int(params), "config": asdict(cfg), "preview": replaced[:12]}


def set_trainable_by_patterns(module: nn.Module, include: Iterable[str] | str | None, exclude: Iterable[str] | str | None = None) -> dict[str, Any]:
    inc = _tuple_csv(include)
    exc = _tuple_csv(exclude)
    trainable = 0
    tensors = 0
    preview: list[str] = []
    for name, param in module.named_parameters():
        enable = _matches(name, inc, exc) if inc else False
        if enable:
            param.requires_grad_(True)
            trainable += param.numel()
            tensors += 1
            if len(preview) < 12:
                preview.append(name)
    return {"patterns": inc, "exclude": exc, "params": int(trainable), "tensors": int(tensors), "preview": preview}


def collect_trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in module.named_parameters() if param.requires_grad}


def load_partial_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    current = module.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value.to(dtype=current[key].dtype)
        else:
            skipped.append(key)
    result = module.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "skipped": len(skipped),
        "missing": len(getattr(result, "missing_keys", [])),
        "unexpected": len(getattr(result, "unexpected_keys", [])),
        "skipped_preview": skipped[:12],
    }


def save_hunyuan_trainable_checkpoint(
    path: str | Path,
    transformer: nn.Module,
    *,
    lora_config: HunyuanLoRAConfig | dict[str, Any] | None,
    partial_unfreeze: Iterable[str] | str | None = None,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = HunyuanLoRAConfig.from_any(lora_config)
    state = collect_trainable_state_dict(transformer)
    payload: dict[str, Any] = {
        "kind": HUNYUAN_TRAINABLE_CHECKPOINT_KIND,
        "state": state,
        "lora_config": asdict(cfg),
        "partial_unfreeze": _tuple_csv(partial_unfreeze),
        "step": step,
        "metrics": metrics or {},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path))
    return payload


def load_hunyuan_trainable_checkpoint(transformer: nn.Module, path: str | Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location=map_location or "cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != HUNYUAN_TRAINABLE_CHECKPOINT_KIND:
        raise RuntimeError(f"{ckpt_path} is not {HUNYUAN_TRAINABLE_CHECKPOINT_KIND}")
    lora_report = apply_lora_to_linear_modules(transformer, payload.get("lora_config"))
    load_report = load_partial_state_dict(transformer, payload.get("state", {}))
    return {"path": str(ckpt_path), "lora": lora_report, "load": load_report, "step": payload.get("step"), "metrics": payload.get("metrics", {})}


__all__ = [
    "HUNYUAN_TRAINABLE_CHECKPOINT_KIND",
    "HunyuanLoRAConfig",
    "LoRALinear",
    "apply_lora_to_linear_modules",
    "collect_trainable_state_dict",
    "load_hunyuan_trainable_checkpoint",
    "save_hunyuan_trainable_checkpoint",
    "set_trainable_by_patterns",
]
