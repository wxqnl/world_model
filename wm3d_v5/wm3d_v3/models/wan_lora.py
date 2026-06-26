"""Small LoRA helpers for trainable Wan DiT adapters.

The implementation is intentionally local to the Wan stage128 draft so it can
be reviewed independently from the existing Hunyuan LoRA utilities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


WAN_TRAINABLE_CHECKPOINT_KIND = "wm3d_wan_ti2v_trainable_v1"


@dataclass
class WanLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    include: tuple[str, ...] = ("blocks",)
    exclude: tuple[str, ...] = ()
    dtype: str = "bf16"
    checkpoint: bool = False
    checkpoint_use_reentrant: bool = False


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported LoRA dtype={name!r}")


def _matches(name: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    include = tuple(str(x) for x in include)
    exclude = tuple(str(x) for x in exclude)
    return (not include or any(part in name for part in include)) and not any(part in name for part in exclude)


class LoRALinear(nn.Module):
    """Wrap ``nn.Linear`` with a low-rank residual branch."""

    def __init__(self, base: nn.Linear, cfg: WanLoRAConfig):
        super().__init__()
        rank = max(1, int(cfg.rank))
        self.base = base
        self.rank = rank
        self.alpha = float(cfg.alpha)
        self.scale = self.alpha / float(rank)
        self.dropout = nn.Dropout(float(cfg.dropout)) if float(cfg.dropout) > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=_dtype(cfg.dtype), device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=_dtype(cfg.dtype), device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for param in self.base.parameters():
            param.requires_grad_(False)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        x_lora = self.dropout(x).to(dtype=self.lora_A.dtype)
        delta = F.linear(F.linear(x_lora, self.lora_A), self.lora_B).to(dtype=base.dtype)
        return base + delta * self.scale


def _set_child(root: nn.Module, dotted_name: str, child: nn.Module) -> None:
    parent = root
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


def apply_lora_to_linear_modules(model: nn.Module, cfg: WanLoRAConfig) -> dict[str, Any]:
    """Install LoRA wrappers on matching linear modules and freeze base weights."""

    for param in model.parameters():
        param.requires_grad_(False)

    selected: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _matches(name, cfg.include, cfg.exclude):
            continue
        _set_child(model, name, LoRALinear(module, cfg))
        selected.append(name)

    setattr(model, "_wm3d_wan_lora_config", asdict(cfg))
    setattr(model, "_wm3d_lora_checkpoint", bool(cfg.checkpoint))
    setattr(model, "_wm3d_lora_checkpoint_use_reentrant", bool(cfg.checkpoint_use_reentrant))
    return {
        "enabled": bool(selected),
        "modules": len(selected),
        "params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "preview": selected[:20],
    }


def set_trainable_by_patterns(model: nn.Module, patterns: Iterable[str], exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Mark non-LoRA parameters trainable by substring patterns."""

    patterns = tuple(str(p) for p in patterns)
    exclude = tuple(str(p) for p in exclude)
    selected: list[str] = []
    tensors = 0
    numel = 0
    for name, param in model.named_parameters():
        if not patterns or not any(part in name for part in patterns):
            continue
        if any(part in name for part in exclude):
            continue
        param.requires_grad_(True)
        selected.append(name)
        tensors += 1
        numel += int(param.numel())
    return {"params": int(numel), "tensors": int(tensors), "preview": selected[:40]}


def collect_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_partial_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    own = dict(model.named_parameters())
    loaded = []
    missing = []
    mismatched = []
    with torch.no_grad():
        for name, value in state.items():
            param = own.get(name)
            if param is None:
                missing.append(name)
                continue
            if tuple(param.shape) != tuple(value.shape):
                mismatched.append((name, tuple(value.shape), tuple(param.shape)))
                continue
            param.copy_(value.to(device=param.device, dtype=param.dtype))
            loaded.append(name)
    return {
        "loaded": len(loaded),
        "missing": len(missing),
        "mismatched": len(mismatched),
        "missing_preview": missing[:20],
        "mismatched_preview": mismatched[:10],
    }


def save_wan_trainable_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    lora_config: WanLoRAConfig | None,
    partial_unfreeze: Iterable[str] = (),
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    payload = {
        "kind": WAN_TRAINABLE_CHECKPOINT_KIND,
        "step": step,
        "lora_config": asdict(lora_config) if lora_config is not None else None,
        "partial_unfreeze": tuple(partial_unfreeze),
        "state_dict": collect_trainable_state_dict(model),
        "metrics": metrics or {},
    }
    tmp = path.with_name("." + path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
