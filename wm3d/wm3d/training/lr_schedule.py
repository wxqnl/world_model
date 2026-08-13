"""Learning-rate schedule helpers for WM3D training.

The training configs historically used ``train.lr`` + ``train.warmup_steps``
with cosine decay to 10% of the peak LR. Large pretraining runs can now use a
top-level ``lr_schedule`` block while old YAML files keep the same behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import torch


@dataclass(frozen=True)
class OptimizerSettings:
    type: str
    peak_lr: float
    min_lr: float
    betas: tuple[float, float]
    weight_decay: float
    grad_clip: float | None


def _section(cfg: Mapping[str, Any], key: str) -> dict[str, Any]:
    section = cfg.get(key) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return dict(section)


def _schedule_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    train_cfg = _section(cfg, "train")
    schedule = cfg.get("lr_schedule", train_cfg.get("lr_schedule", {})) or {}
    if not isinstance(schedule, Mapping):
        raise ValueError("lr_schedule must be a mapping")
    return dict(schedule)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def resolve_optimizer_settings(cfg: Mapping[str, Any]) -> OptimizerSettings:
    """Resolve optimizer/LR settings with backward-compatible defaults.

    New large-run configs can specify top-level ``optimizer`` and
    ``lr_schedule`` sections. Existing configs without those sections still use
    ``train.lr``, ``train.weight_decay``, and AdamW betas (0.9, 0.95).
    """

    train_cfg = _section(cfg, "train")
    opt_cfg = _section(cfg, "optimizer")
    schedule = _schedule_cfg(cfg)

    opt_type = str(opt_cfg.get("type", train_cfg.get("optimizer", "adamw"))).lower()
    if opt_type != "adamw":
        raise ValueError(f"unsupported optimizer.type={opt_type!r}; only adamw is implemented")

    lr_value = schedule.get("peak_lr", opt_cfg.get("peak_lr", train_cfg.get("lr")))
    if lr_value is None:
        raise ValueError("missing LR: set lr_schedule.peak_lr or train.lr")
    peak_lr = float(lr_value)
    if peak_lr <= 0:
        raise ValueError(f"peak_lr must be > 0, got {peak_lr}")

    min_lr = _float_or_none(schedule.get("min_lr"))
    if min_lr is None:
        min_factor = float(schedule.get("min_factor", 0.1))
        min_lr = peak_lr * min_factor
    if min_lr < 0:
        raise ValueError(f"min_lr must be >= 0, got {min_lr}")
    if min_lr > peak_lr:
        raise ValueError(f"min_lr must be <= peak_lr, got min_lr={min_lr} peak_lr={peak_lr}")

    betas_raw = opt_cfg.get("betas", train_cfg.get("betas", (0.9, 0.95)))
    if len(betas_raw) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    betas = (float(betas_raw[0]), float(betas_raw[1]))

    weight_decay = float(opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0)))
    grad_clip = _float_or_none(opt_cfg.get("grad_clip", train_cfg.get("grad_clip")))
    return OptimizerSettings(
        type=opt_type,
        peak_lr=peak_lr,
        min_lr=min_lr,
        betas=betas,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )


def _warmup_factor(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _cosine_decay_factor(progress: float, min_factor: float) -> float:
    progress = min(1.0, max(0.0, float(progress)))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_lambda(cfg: Mapping[str, Any], total_steps: int) -> Callable[[int], float]:
    """Build the LambdaLR multiplier for a config.

    Supported schedules:
    - default/cosine: old behavior, warmup then cosine decay to min_lr/peak_lr.
    - wsd: warmup, stable plateau, final cosine decay to min_lr/peak_lr.
    """

    train_cfg = _section(cfg, "train")
    schedule = _schedule_cfg(cfg)
    settings = resolve_optimizer_settings(cfg)
    total_steps = max(1, int(total_steps))
    schedule_type = str(schedule.get("type", train_cfg.get("lr_schedule_type", "cosine"))).lower()
    warmup_steps = int(schedule.get("warmup_steps", train_cfg.get("warmup_steps", 0)))
    min_factor = settings.min_lr / settings.peak_lr

    if schedule_type in ("cosine", "legacy_cosine"):
        def cosine_lambda(step: int) -> float:
            step = int(step)
            if step < warmup_steps:
                return _warmup_factor(step, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return _cosine_decay_factor(progress, min_factor)

        return cosine_lambda

    if schedule_type != "wsd":
        raise ValueError(f"unsupported lr_schedule.type={schedule_type!r}")

    if warmup_steps >= total_steps:
        def warmup_only_lambda(step: int) -> float:
            step = int(step)
            if step < total_steps:
                return _warmup_factor(step, warmup_steps)
            return min_factor

        return warmup_only_lambda

    post_warmup_steps = total_steps - warmup_steps
    decay_frac = schedule.get("decay_frac")
    if decay_frac is None:
        stable_frac = float(schedule.get("stable_frac", 0.85))
        decay_frac = max(0.0, 1.0 - stable_frac)
    decay_frac = float(decay_frac)
    if not (0.0 < decay_frac <= 1.0):
        raise ValueError(f"lr_schedule.decay_frac must be in (0, 1], got {decay_frac}")
    decay_steps = max(1, min(post_warmup_steps, int(round(post_warmup_steps * decay_frac))))
    stable_steps = max(0, post_warmup_steps - decay_steps)
    decay_start = warmup_steps + stable_steps
    decay_end = total_steps - 1

    def wsd_lambda(step: int) -> float:
        step = int(step)
        if step < warmup_steps:
            return _warmup_factor(step, warmup_steps)
        if step < decay_start:
            return 1.0
        if step >= decay_end:
            return min_factor
        progress = (step - decay_start) / max(1, decay_end - decay_start)
        return _cosine_decay_factor(progress, min_factor)

    return wsd_lambda


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Mapping[str, Any],
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(cfg, total_steps),
    )
