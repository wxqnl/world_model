"""Stage0 from-scratch wm3d pretraining with trainable Hunyuan DiT LoRA/final RGB generator.

This is not the adapter-only bridge used by stage113. The wm3d model is built
from cfg/random init, the small RGB decoder is disabled by cfg, native3D losses
stay active, and RGB supervision is a Hunyuan DiT velocity/flow loss.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        StateDictType,
    )
except Exception:  # pragma: no cover - depends on torch build
    BackwardPrefetch = None
    FullStateDictConfig = None
    FSDP = None
    ShardingStrategy = None
    StateDictType = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_hunyuan_dit_control_adapter import (
    HunyuanDiTControlInjector,
    adapter_target,
    build_hunyuan_backend_args,
    encode_hunyuan_latents,
    encode_hunyuan_prompts,
    latent_motion_mask_from_target,
    make_wrong_action,
    prompts_from_batch,
    rotary_freqs,
    target_video_from_batch,
    weighted_velocity_losses,
    context_video_from_batch,
)
from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.hunyuan_dit_control_adapter import (
    HunyuanDiTControlAdapter,
    HunyuanDiTControlConfig,
    save_hunyuan_dit_control_checkpoint,
)
from wm3d_v3.models.hunyuan_lora import (
    HunyuanLoRAConfig,
    apply_lora_to_linear_modules,
    collect_trainable_state_dict,
    load_partial_state_dict,
    save_hunyuan_trainable_checkpoint,
    set_trainable_by_patterns,
)
from wm3d_v3.training.train import (
    _all_reduce_gradients,
    _distributed_finite_count,
    _forward_joint_model,
    _sampler_scope,
    apply_condition_dropout,
    batch_to_device,
    build_datasets,
    build_model,
    load_action_stats_if_available,
    prior_clean_tokens_from_targets,
    WeightedDistributedSampler,
)
from wm3d_v3.training.lr_schedule import build_lr_scheduler
from wm3d_v3.video_backends.hunyuan_dit_control_video import HunyuanDiTControlVideoBackend


def setup_distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("WM3D_DDP_BACKEND", "nccl"))
    return rank, world, local


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_module(module: torch.nn.Module | None, world: int) -> None:
    if _is_fsdp(module):
        return
    if module is not None and world > 1:
        _all_reduce_gradients(module, world)


def _is_fsdp(module: torch.nn.Module | None) -> bool:
    return FSDP is not None and module is not None and isinstance(module, FSDP)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    while hasattr(current, "module"):
        current = current.module
    return current


def _fsdp_enabled(train_cfg: dict) -> bool:
    return bool(train_cfg.get("fsdp_enabled", False))


def _parse_fsdp_strategy(name: str):
    if ShardingStrategy is None:
        raise RuntimeError("torch.distributed.fsdp is not available in this environment")
    value = str(name or "full_shard").strip().lower()
    if value in {"full", "full_shard", "zero3"}:
        return ShardingStrategy.FULL_SHARD
    if value in {"grad", "shard_grad_op", "zero2"}:
        return ShardingStrategy.SHARD_GRAD_OP
    if value in {"hybrid", "hybrid_shard"}:
        return ShardingStrategy.HYBRID_SHARD
    if value in {"none", "no_shard"}:
        return ShardingStrategy.NO_SHARD
    raise ValueError(f"unsupported fsdp_sharding_strategy={name!r}")


def _parse_backward_prefetch(name: str | None):
    if BackwardPrefetch is None:
        return None
    value = str(name or "backward_pre").strip().lower()
    if value in {"none", "false", "0"}:
        return None
    if value in {"pre", "backward_pre"}:
        return BackwardPrefetch.BACKWARD_PRE
    if value in {"post", "backward_post"}:
        return BackwardPrefetch.BACKWARD_POST
    raise ValueError(f"unsupported fsdp_backward_prefetch={name!r}")


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return str(dtype).startswith("torch.float8")


def _fsdp_ignored_params(
    module: torch.nn.Module,
    *,
    ignore_frozen: bool,
    ignore_float8: bool,
) -> tuple[list[torch.nn.Parameter], dict[str, Any]]:
    ignored: list[torch.nn.Parameter] = []
    stats: dict[str, Any] = {
        "ignore_frozen": ignore_frozen,
        "ignore_float8": ignore_float8,
        "ignored_params": 0,
        "ignored_numel": 0,
        "ignored_trainable_numel": 0,
        "kept_params": 0,
        "kept_numel": 0,
        "kept_dtypes": {},
        "ignored_dtypes": {},
    }
    seen: set[int] = set()
    for param in module.parameters():
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        dtype_name = str(param.dtype)
        is_float8 = _is_float8_dtype(param.dtype)
        should_ignore = (ignore_float8 and is_float8) or (ignore_frozen and not param.requires_grad)
        if should_ignore:
            if is_float8 and param.requires_grad:
                raise RuntimeError("Refusing to ignore trainable float8 Hunyuan parameter under FSDP")
            ignored.append(param)
            stats["ignored_params"] += 1
            stats["ignored_numel"] += int(param.numel())
            if param.requires_grad:
                stats["ignored_trainable_numel"] += int(param.numel())
            stats["ignored_dtypes"][dtype_name] = int(stats["ignored_dtypes"].get(dtype_name, 0)) + int(param.numel())
        else:
            stats["kept_params"] += 1
            stats["kept_numel"] += int(param.numel())
            stats["kept_dtypes"][dtype_name] = int(stats["kept_dtypes"].get(dtype_name, 0)) + int(param.numel())
    return ignored, stats


def maybe_wrap_fsdp(
    wm_model: torch.nn.Module,
    transformer: torch.nn.Module,
    train_cfg: dict,
    *,
    device: torch.device,
    rank: int,
    world: int,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    if not _fsdp_enabled(train_cfg):
        return wm_model, transformer, {"enabled": False}
    if world <= 1:
        raise RuntimeError("FSDP requested but WORLD_SIZE <= 1")
    if FSDP is None:
        raise RuntimeError("FSDP requested but torch.distributed.fsdp is not available")

    modules = {str(x).strip().lower() for x in train_cfg.get("fsdp_modules", ("wm", "hunyuan_transformer"))}
    strategy = _parse_fsdp_strategy(str(train_cfg.get("fsdp_sharding_strategy", "full_shard")))
    use_orig_params = bool(train_cfg.get("fsdp_use_orig_params", True))
    limit_all_gathers = bool(train_cfg.get("fsdp_limit_all_gathers", True))
    forward_prefetch = bool(train_cfg.get("fsdp_forward_prefetch", False))
    backward_prefetch = _parse_backward_prefetch(train_cfg.get("fsdp_backward_prefetch", "backward_pre"))
    ignore_hunyuan_frozen = bool(train_cfg.get("fsdp_ignore_hunyuan_frozen_params", True))
    ignore_hunyuan_float8 = bool(train_cfg.get("fsdp_ignore_hunyuan_float8_params", True))
    report: dict[str, Any] = {
        "enabled": True,
        "modules": sorted(modules),
        "strategy": str(strategy),
        "use_orig_params": use_orig_params,
        "limit_all_gathers": limit_all_gathers,
        "forward_prefetch": forward_prefetch,
        "backward_prefetch": str(backward_prefetch),
        "ignore_hunyuan_frozen": ignore_hunyuan_frozen,
        "ignore_hunyuan_float8": ignore_hunyuan_float8,
    }

    kwargs = {
        "sharding_strategy": strategy,
        "use_orig_params": use_orig_params,
        "device_id": device if device.type == "cuda" else None,
        "limit_all_gathers": limit_all_gathers,
        "forward_prefetch": forward_prefetch,
        "backward_prefetch": backward_prefetch,
    }
    if "wm" in modules or "wm_model" in modules:
        wm_model = FSDP(wm_model, **kwargs)
    if "hunyuan" in modules or "hunyuan_transformer" in modules or "transformer" in modules:
        hunyuan_kwargs = dict(kwargs)
        ignored, ignored_stats = _fsdp_ignored_params(
            transformer,
            ignore_frozen=ignore_hunyuan_frozen,
            ignore_float8=ignore_hunyuan_float8,
        )
        if ignored:
            hunyuan_kwargs["ignored_states"] = ignored
        report["hunyuan_ignored"] = ignored_stats
        transformer = FSDP(transformer, **hunyuan_kwargs)
    if rank == 0:
        print(f"[rank0] fsdp={json.dumps(report, default=str)}", flush=True)
    return wm_model, transformer, report


def broadcast_module_state(module: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buf in module.buffers():
        dist.broadcast(buf.data, src=0)


def broadcast_trainable_parameters(module: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    for param in module.parameters():
        if param.requires_grad:
            dist.broadcast(param.data, src=0)


def _count_trainable(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _make_sampler(dataset, cfg: dict, *, world: int, rank: int, local: int, train: bool):
    sampler_world, sampler_rank, _scope = _sampler_scope(cfg, world, rank, local)
    seed = int(cfg["data"].get("seed", 0)) + (0 if train else 100000)
    if world > 1 and bool((cfg.get("data") or {}).get("node_sharded_window_cache", False)):
        local_len = torch.tensor([len(dataset)], device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.long)
        min_len = local_len.clone()
        dist.all_reduce(min_len, op=dist.ReduceOp.MIN)
        num_samples = max(1, int(min_len.item()) // max(1, sampler_world))
        return WeightedDistributedSampler(
            torch.ones(len(dataset), dtype=torch.double),
            num_replicas=sampler_world,
            rank=sampler_rank,
            replacement=False,
            num_samples=num_samples,
            seed=seed,
        )
    return DistributedSampler(dataset, num_replicas=sampler_world, rank=sampler_rank, shuffle=bool(train), drop_last=bool(train), seed=seed)


def _dtype(name: str) -> torch.dtype:
    value = str(name).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype={name!r}")


def _cast_trainable_params(module: torch.nn.Module, dtype: torch.dtype) -> dict[str, Any]:
    report: dict[str, Any] = {"target_dtype": str(dtype), "converted_tensors": 0, "converted_numel": 0, "source_dtypes": {}}
    with torch.no_grad():
        for param in module.parameters():
            if not param.requires_grad or param.dtype == dtype:
                continue
            if _is_float8_dtype(param.dtype):
                raise RuntimeError("Refusing to cast trainable float8 parameter")
            source = str(param.dtype)
            report["source_dtypes"][source] = int(report["source_dtypes"].get(source, 0)) + int(param.numel())
            param.data = param.data.to(dtype=dtype)
            report["converted_tensors"] += 1
            report["converted_numel"] += int(param.numel())
            if param.grad is not None:
                param.grad = param.grad.to(dtype=dtype)
    return report


def build_optimizer(
    wm_model: torch.nn.Module,
    hunyuan_control_adapter: HunyuanDiTControlAdapter,
    transformer: torch.nn.Module,
    train_cfg: dict,
    rank: int,
) -> torch.optim.Optimizer:
    groups = []
    wd = float(train_cfg.get("weight_decay", 0.02))
    wm_params = [p for p in wm_model.parameters() if p.requires_grad]
    control_params = [p for p in hunyuan_control_adapter.parameters() if p.requires_grad]
    hunyuan_lora_params = []
    hunyuan_base_params = []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if ".lora_" in name or name.endswith("lora_A") or name.endswith("lora_B"):
            hunyuan_lora_params.append(param)
        else:
            hunyuan_base_params.append(param)
    if wm_params:
        groups.append({"name": "wm3d", "params": wm_params, "lr": float(train_cfg.get("wm_lr", train_cfg["lr"])), "weight_decay": wd})
    if control_params:
        groups.append({"name": "hunyuan_control_adapter", "params": control_params, "lr": float(train_cfg.get("hunyuan_control_lr", 2e-5)), "weight_decay": wd})
    if hunyuan_lora_params:
        groups.append({"name": "hunyuan_lora", "params": hunyuan_lora_params, "lr": float(train_cfg.get("hunyuan_lora_lr", 5e-6)), "weight_decay": float(train_cfg.get("hunyuan_weight_decay", 0.0))})
    if hunyuan_base_params:
        groups.append({"name": "hunyuan_base_trainable", "params": hunyuan_base_params, "lr": float(train_cfg.get("hunyuan_base_lr", 2e-7)), "weight_decay": float(train_cfg.get("hunyuan_base_weight_decay", train_cfg.get("hunyuan_weight_decay", 0.0)))})
    if not groups:
        raise RuntimeError("no trainable parameters for joint stage0 Hunyuan DiT PT")
    if rank == 0:
        for group in groups:
            params = sum(p.numel() for p in group["params"])
            print(f"[rank0] opt_group={group['name']} params={params/1e6:.2f}M lr={group['lr']:.3g} wd={group['weight_decay']:.3g}", flush=True)
    return torch.optim.AdamW(groups, betas=tuple(train_cfg.get("betas", (0.9, 0.95))))


def build_hunyuan_modules(args: argparse.Namespace, cfg: dict, device: torch.device, rank: int, world: int):
    train_cfg = cfg["train"]
    backend = HunyuanDiTControlVideoBackend(build_hunyuan_backend_args(args), device=device)
    sampler = backend.load()
    transformer = backend.resolve_transformer(sampler)
    for module in (getattr(sampler.pipeline, "vae", None), getattr(sampler.pipeline, "text_encoder", None), getattr(sampler.pipeline, "text_encoder_2", None)):
        if module is not None:
            module.requires_grad_(False)
            module.eval()
    transformer.requires_grad_(False)
    transformer.eval()

    lora_report = {"enabled": False, "params": 0}
    if bool(train_cfg.get("hunyuan_dit_train_lora", True)):
        lora_cfg = HunyuanLoRAConfig(
            rank=int(train_cfg.get("hunyuan_lora_rank", 8)),
            alpha=float(train_cfg.get("hunyuan_lora_alpha", 16.0)),
            dropout=float(train_cfg.get("hunyuan_lora_dropout", 0.0)),
            include=tuple(train_cfg.get("hunyuan_lora_include", ("double_blocks", "single_blocks"))),
            exclude=tuple(train_cfg.get("hunyuan_lora_exclude", ())),
            dtype=str(train_cfg.get("hunyuan_lora_dtype", "fp32")),
            checkpoint=bool(train_cfg.get("hunyuan_lora_checkpoint", False)),
            checkpoint_use_reentrant=bool(train_cfg.get("hunyuan_lora_checkpoint_use_reentrant", False)),
        )
        lora_report = apply_lora_to_linear_modules(transformer, lora_cfg)
    patterns = train_cfg.get("hunyuan_dit_trainable_patterns") or ()
    if patterns:
        pattern_report = set_trainable_by_patterns(transformer, patterns, train_cfg.get("hunyuan_dit_trainable_exclude", ()))
    else:
        pattern_report = {"params": 0, "tensors": 0, "preview": []}
    trainable_dtype = _dtype(str(train_cfg.get("hunyuan_trainable_dtype", train_cfg.get("precision", "bf16"))))
    trainable_dtype_report = _cast_trainable_params(transformer, trainable_dtype)
    setattr(transformer, "_wm3d_activation_checkpoint", bool(train_cfg.get("hunyuan_activation_checkpoint", False)))
    setattr(
        transformer,
        "_wm3d_activation_checkpoint_use_reentrant",
        bool(train_cfg.get("hunyuan_activation_checkpoint_use_reentrant", False)),
    )

    adapter = HunyuanDiTControlAdapter(
        HunyuanDiTControlConfig(
            hidden=int(train_cfg.get("hunyuan_control_hidden", 192)),
            dit_hidden=int(getattr(transformer, "hidden_size", 3072)),
            double_blocks=len(backend._iter_transformer_blocks(transformer, "double_blocks")),
            single_blocks=len(backend._iter_transformer_blocks(transformer, "single_blocks")),
            use_rough=False,
            use_rgb_features=False,
            action_residual_scale=float(train_cfg.get("hunyuan_action_residual_scale", 1.0)),
            action_token_scale=float(train_cfg.get("hunyuan_action_token_scale", 1.0)),
            action_direct_scale=float(train_cfg.get("hunyuan_action_direct_scale", 0.0)),
            action_latent_scale=float(train_cfg.get("hunyuan_action_latent_scale", 0.0)),
            use_action_cross_attn=bool(train_cfg.get("hunyuan_use_action_cross_attn", False)),
            action_cross_attn_scale=float(train_cfg.get("hunyuan_action_cross_attn_scale", 0.0)),
            action_cross_attn_hidden=int(train_cfg.get("hunyuan_action_cross_attn_hidden", train_cfg.get("hunyuan_control_hidden", 192))),
            action_cross_attn_heads=int(train_cfg.get("hunyuan_action_cross_attn_heads", 4)),
            action_cross_attn_time_scale=float(train_cfg.get("hunyuan_action_cross_attn_time_scale", 1.0)),
            use_temporal_action_summary=bool(train_cfg.get("hunyuan_use_temporal_action_summary", False)),
            temporal_action_summary_scale=float(train_cfg.get("hunyuan_temporal_action_summary_scale", 1.0)),
            use_parallel_action_dit=bool(train_cfg.get("hunyuan_use_parallel_action_dit", False)),
            parallel_action_dit_scale=float(train_cfg.get("hunyuan_parallel_action_dit_scale", 0.0)),
            parallel_action_dit_hidden=int(train_cfg.get("hunyuan_parallel_action_dit_hidden", 256)),
            parallel_action_dit_heads=int(train_cfg.get("hunyuan_parallel_action_dit_heads", 4)),
            parallel_action_dit_mlp_mult=float(train_cfg.get("hunyuan_parallel_action_dit_mlp_mult", 2.0)),
            native_parallel_action_forward=bool(train_cfg.get("hunyuan_native_parallel_action_forward", False)),
            use_block_action_film=bool(train_cfg.get("hunyuan_use_block_action_film", True)),
            block_action_film_scale=float(train_cfg.get("hunyuan_block_action_film_scale", 1.0)),
            block_action_film_hidden=int(train_cfg.get("hunyuan_block_action_film_hidden", 192)),
            double_control_gain_start=float(train_cfg.get("hunyuan_double_control_gain_start", 1.0)),
            double_control_gain_end=float(train_cfg.get("hunyuan_double_control_gain_end", 1.0)),
            double_control_gain_power=float(train_cfg.get("hunyuan_double_control_gain_power", 1.0)),
            single_control_gain_start=float(train_cfg.get("hunyuan_single_control_gain_start", 1.0)),
            single_control_gain_end=float(train_cfg.get("hunyuan_single_control_gain_end", 1.0)),
            single_control_gain_power=float(train_cfg.get("hunyuan_single_control_gain_power", 1.0)),
            use_noisy_latents=bool(train_cfg.get("hunyuan_control_use_noisy_latents", False)),
            use_source_latents=bool(train_cfg.get("hunyuan_control_use_source_latents", False)),
            use_sigma_embed=bool(train_cfg.get("hunyuan_control_use_sigma_embed", False)),
        )
    ).to(device)
    broadcast_module_state(adapter, world)
    if rank == 0:
        print(f"[rank0] hunyuan_lora={json.dumps(lora_report, default=str)}", flush=True)
        print(f"[rank0] hunyuan_partial_train={json.dumps(pattern_report, default=str)}", flush=True)
        print(f"[rank0] hunyuan_trainable_dtype={json.dumps(trainable_dtype_report, default=str)}", flush=True)
        print(f"[rank0] HunyuanDiTControlAdapter params={_count_trainable(adapter)/1e6:.2f}M", flush=True)
        print(f"[rank0] Hunyuan trainable params={_count_trainable(transformer)/1e6:.2f}M", flush=True)
        print(
            "[rank0] hunyuan_activation_checkpoint="
            f"{bool(getattr(transformer, '_wm3d_activation_checkpoint', False))} "
            f"use_reentrant={bool(getattr(transformer, '_wm3d_activation_checkpoint_use_reentrant', False))}",
            flush=True,
        )
    if any(p.requires_grad for p in transformer.parameters()):
        transformer.train()
    return sampler, transformer, adapter, lora_report, pattern_report


def load_stage0_hunyuan_init(
    path: str | Path,
    *,
    wm_model: torch.nn.Module,
    adapter: HunyuanDiTControlAdapter,
    transformer: torch.nn.Module,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    ckpt_path = Path(path)
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ckpt_path} must contain a dict payload")
    if "model" not in payload or "hunyuan_control_adapter" not in payload or "hunyuan_trainable" not in payload:
        raise RuntimeError(
            f"{ckpt_path} missing required stage0 Hunyuan keys; "
            "expected model, hunyuan_control_adapter, hunyuan_trainable"
        )
    wm_report = wm_model.load_state_dict(payload["model"], strict=False)
    adapter_report = adapter.load_state_dict(payload["hunyuan_control_adapter"], strict=False)
    trainable_report = load_partial_state_dict(transformer, payload["hunyuan_trainable"])
    report = {
        "path": str(ckpt_path),
        "step": payload.get("step"),
        "wm_missing": len(getattr(wm_report, "missing_keys", [])),
        "wm_unexpected": len(getattr(wm_report, "unexpected_keys", [])),
        "adapter_missing": len(getattr(adapter_report, "missing_keys", [])),
        "adapter_unexpected": len(getattr(adapter_report, "unexpected_keys", [])),
        "hunyuan_trainable": trainable_report,
    }
    if rank == 0:
        print(f"[rank0] init_from_stage0_hunyuan={json.dumps(report, default=str)}", flush=True)
    return report


def _expand_latent_motion_mask(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    mask_f = mask.to(device=ref.device, dtype=ref.dtype).clamp(0.0, 1.0)
    while mask_f.ndim < ref.ndim:
        mask_f = mask_f.unsqueeze(1)
    return mask_f.expand_as(ref)


def _weighted_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _latent_source_weights(
    mask: torch.Tensor | None,
    ref: torch.Tensor,
    *,
    dynamic_weight: float,
    static_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = _expand_latent_motion_mask(mask, ref)
    if mask_f is None:
        mask_f = torch.ones_like(ref)
        weights = torch.ones_like(ref)
    else:
        weights = float(static_weight) * (1.0 - mask_f) + float(dynamic_weight) * mask_f
        weights = weights.clamp_min(1e-6)
    return weights, mask_f


def weighted_latent_source_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    dynamic_weight: float,
    static_weight: float,
    mse_weight: float,
    l1_weight: float,
    temporal_weight: float,
    from_first_weight: float,
    motion_floor_weight: float = 0.0,
    motion_floor_ratio: float = 0.0,
    from_first_floor_weight: float = 0.0,
    from_first_floor_ratio: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.float()
    weights, motion_focus = _latent_source_weights(
        mask,
        pred_f,
        dynamic_weight=dynamic_weight,
        static_weight=static_weight,
    )
    diff = pred_f - target_f
    mse = _weighted_mean(diff.square(), weights)
    l1 = _weighted_mean(diff.abs(), weights)
    if pred_f.shape[2] > 1:
        pred_step = pred_f[:, :, 1:] - pred_f[:, :, :-1]
        target_step = target_f[:, :, 1:] - target_f[:, :, :-1]
        step_focus = torch.maximum(motion_focus[:, :, 1:], motion_focus[:, :, :-1]).clamp_min(1e-6)
        temporal_l1 = _weighted_mean((pred_step - target_step).abs(), step_focus)
        pred_step_motion = _weighted_mean(pred_step.abs(), step_focus)
        target_step_motion = _weighted_mean(target_step.abs(), step_focus).detach()
    else:
        temporal_l1 = pred_f.new_zeros(())
        pred_step_motion = pred_f.new_zeros(())
        target_step_motion = pred_f.new_zeros(())
    pred_from = pred_f - pred_f[:, :, :1]
    target_from = target_f - target_f[:, :, :1]
    from_focus = motion_focus.clamp_min(1e-6)
    from_first_l1 = _weighted_mean((pred_from - target_from).abs(), from_focus)
    pred_motion = _weighted_mean(pred_from.abs(), from_focus)
    target_motion = _weighted_mean(target_from.abs(), from_focus).detach().clamp_min(1e-8)
    motion_ratio = pred_motion / target_motion
    motion_floor = F.relu(target_step_motion * float(motion_floor_ratio) - pred_step_motion)
    from_first_floor = F.relu(target_motion * float(from_first_floor_ratio) - pred_motion)
    loss = (
        float(mse_weight) * mse
        + float(l1_weight) * l1
        + float(temporal_weight) * temporal_l1
        + float(from_first_weight) * from_first_l1
        + float(motion_floor_weight) * motion_floor
        + float(from_first_floor_weight) * from_first_floor
    )
    return loss, {
        "wm_source_mse": float(mse.detach().cpu()),
        "wm_source_l1": float(l1.detach().cpu()),
        "wm_source_temporal_l1": float(temporal_l1.detach().cpu()),
        "wm_source_from_first_l1": float(from_first_l1.detach().cpu()),
        "wm_source_motion_ratio": float(motion_ratio.detach().cpu()),
        "wm_source_pred_step_motion": float(pred_step_motion.detach().cpu()),
        "wm_source_target_step_motion": float(target_step_motion.detach().cpu()),
        "wm_source_motion_floor": float(motion_floor.detach().cpu()),
        "wm_source_pred_from_first_motion": float(pred_motion.detach().cpu()),
        "wm_source_target_from_first_motion": float(target_motion.detach().cpu()),
        "wm_source_from_first_floor": float(from_first_floor.detach().cpu()),
    }


def _decode_hunyuan_latents_for_loss(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decode Hunyuan latents with gradients to the latents, while VAE params stay frozen."""
    param = next(vae.parameters())
    z = latents / float(vae.config.scaling_factor)
    decoded = vae.decode(z.to(device=param.device, dtype=param.dtype), return_dict=False)[0]
    return decoded.float().div(2.0).add(0.5)


def _rgb_motion_mask_from_target(
    context_rgb: torch.Tensor,
    target_video: torch.Tensor,
    *,
    threshold: float,
    dilate: int,
    power: float,
) -> torch.Tensor:
    frames = int(target_video.shape[2])
    context_video = context_rgb.float()[:, :, None].expand(-1, -1, frames, -1, -1)
    from_context = (target_video.float() - context_video).abs().mean(dim=1, keepdim=True)
    if frames > 1:
        step = torch.zeros_like(from_context)
        step[:, :, 1:] = (target_video.float()[:, :, 1:] - target_video.float()[:, :, :-1]).abs().mean(dim=1, keepdim=True)
        motion = torch.maximum(from_context, step)
    else:
        motion = from_context
    denom = motion.flatten(2).amax(dim=2).view(motion.shape[0], 1, 1, 1, 1).clamp_min(1e-6)
    threshold = float(threshold)
    mask = (motion / denom - threshold).clamp_min(0.0)
    mask = (mask / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)
    dilate = max(0, int(dilate))
    if dilate > 0:
        k = 2 * dilate + 1
        mask = F.max_pool3d(mask, kernel_size=(1, k, k), stride=1, padding=(0, dilate, dilate))
    power = max(1e-6, float(power))
    if abs(power - 1.0) > 1e-6:
        mask = mask.pow(power)
    return mask.clamp(0.0, 1.0)


def decoded_rgb_losses(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    train_cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred_video.float()
    target_f = target_video.to(device=pred_f.device).float()
    context_f = context_rgb.to(device=pred_f.device).float()
    size = int(train_cfg.get("hunyuan_decoded_rgb_size", 0) or 0)
    if size > 0:
        shape = (int(pred_f.shape[2]), size, size)
        pred_f = F.interpolate(pred_f, size=shape, mode="trilinear", align_corners=False)
        target_f = F.interpolate(target_f, size=shape, mode="trilinear", align_corners=False)
        context_f = F.interpolate(context_f, size=(size, size), mode="bilinear", align_corners=False)

    mask = _rgb_motion_mask_from_target(
        context_f,
        target_f,
        threshold=float(train_cfg.get("hunyuan_decoded_rgb_motion_threshold", 0.03)),
        dilate=int(train_cfg.get("hunyuan_decoded_rgb_motion_dilate", 4)),
        power=float(train_cfg.get("hunyuan_decoded_rgb_motion_power", 0.5)),
    )

    if bool(train_cfg.get("hunyuan_decoded_rgb_skip_context", True)) and pred_f.shape[2] > 1:
        pred_f = pred_f[:, :, 1:]
        target_f = target_f[:, :, 1:]
        mask = mask[:, :, 1:]

    mask_c = mask.expand(-1, int(pred_f.shape[1]), -1, -1, -1)
    weights = (
        float(train_cfg.get("hunyuan_decoded_rgb_static_weight", 0.25)) * (1.0 - mask_c)
        + float(train_cfg.get("hunyuan_decoded_rgb_dynamic_weight", 8.0)) * mask_c
    ).clamp_min(1e-6)
    diff = pred_f - target_f
    l1 = _weighted_mean(diff.abs(), weights)
    mse = _weighted_mean(diff.square(), weights)
    dyn_den = mask_c.sum().clamp_min(1.0)
    sta = 1.0 - mask_c
    sta_den = sta.sum().clamp_min(1.0)
    dyn_l1 = (diff.abs() * mask_c).sum() / dyn_den
    static_l1 = (diff.abs() * sta).sum() / sta_den

    if pred_f.shape[2] > 1:
        pred_step = pred_f[:, :, 1:] - pred_f[:, :, :-1]
        target_step = target_f[:, :, 1:] - target_f[:, :, :-1]
        step_focus = torch.maximum(mask_c[:, :, 1:], mask_c[:, :, :-1]).clamp_min(1e-6)
        temporal_l1 = _weighted_mean((pred_step - target_step).abs(), step_focus)
        pred_step_motion = _weighted_mean(pred_step.abs(), step_focus)
        target_step_motion = _weighted_mean(target_step.abs(), step_focus).detach().clamp_min(1e-8)
    else:
        temporal_l1 = pred_f.new_zeros(())
        pred_step_motion = pred_f.new_zeros(())
        target_step_motion = pred_f.new_zeros(()).clamp_min(1e-8)

    pred_from = pred_f - pred_f[:, :, :1]
    target_from = target_f - target_f[:, :, :1]
    from_focus = mask_c.clamp_min(1e-6)
    from_first_l1 = _weighted_mean((pred_from - target_from).abs(), from_focus)
    pred_from_motion = _weighted_mean(pred_from.abs(), from_focus)
    target_from_motion = _weighted_mean(target_from.abs(), from_focus).detach().clamp_min(1e-8)
    step_floor = F.relu(
        target_step_motion * float(train_cfg.get("hunyuan_decoded_rgb_motion_floor_ratio", 0.85))
        - pred_step_motion
    )
    from_floor = F.relu(
        target_from_motion * float(train_cfg.get("hunyuan_decoded_rgb_from_first_floor_ratio", 0.80))
        - pred_from_motion
    )
    loss = (
        float(train_cfg.get("hunyuan_decoded_rgb_l1_weight", 0.0)) * l1
        + float(train_cfg.get("hunyuan_decoded_rgb_mse_weight", 0.0)) * mse
        + float(train_cfg.get("hunyuan_decoded_rgb_temporal_weight", 0.0)) * temporal_l1
        + float(train_cfg.get("hunyuan_decoded_rgb_from_first_weight", 0.0)) * from_first_l1
        + float(train_cfg.get("hunyuan_decoded_rgb_motion_floor_weight", 0.0)) * step_floor
        + float(train_cfg.get("hunyuan_decoded_rgb_from_first_floor_weight", 0.0)) * from_floor
        + float(train_cfg.get("hunyuan_decoded_rgb_static_l1_weight", 0.0)) * static_l1
    )
    return loss, {
        "decoded_rgb_l1": float(l1.detach().cpu()),
        "decoded_rgb_mse": float(mse.detach().cpu()),
        "decoded_rgb_dynamic_l1": float(dyn_l1.detach().cpu()),
        "decoded_rgb_static_l1": float(static_l1.detach().cpu()),
        "decoded_rgb_temporal_l1": float(temporal_l1.detach().cpu()),
        "decoded_rgb_from_first_l1": float(from_first_l1.detach().cpu()),
        "decoded_rgb_motion_floor": float(step_floor.detach().cpu()),
        "decoded_rgb_from_first_floor": float(from_floor.detach().cpu()),
        "decoded_rgb_motion_ratio": float((pred_step_motion / target_step_motion).detach().cpu()),
        "decoded_rgb_from_first_ratio": float((pred_from_motion / target_from_motion).detach().cpu()),
        "decoded_rgb_mask_mean": float(mask_c.mean().detach().cpu()),
    }


def _parse_action_cf_modes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()]


def _velocity_action_cf_enabled(train_cfg: dict) -> bool:
    modes = _parse_action_cf_modes(train_cfg.get("hunyuan_velocity_action_cf_modes", ()))
    rank_weight = float(train_cfg.get("hunyuan_velocity_action_cf_rank_weight", 0.0))
    sep_weight = float(train_cfg.get("hunyuan_velocity_action_cf_sep_weight", 0.0))
    return bool(modes) and (rank_weight > 0.0 or sep_weight > 0.0)


def _velocity_action_cf_active_this_step(train_cfg: dict, step: int) -> bool:
    if not _velocity_action_cf_enabled(train_cfg):
        return False
    interval = max(1, int(train_cfg.get("hunyuan_velocity_action_cf_every", 1)))
    return int(step) % interval == 0


def _prepare_hunyuan_controls(
    target: HunyuanDiTControlAdapter,
    *,
    wm_out: dict[str, torch.Tensor],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
    noisy_latents: torch.Tensor | None = None,
    source_latents: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
) -> None:
    target.prepare_controls(
        pred_tokens=wm_out["pred_tokens"],
        depth=wm_out["depth"],
        motion_hint=wm_out.get("motion_hint"),
        contact_hint=wm_out.get("contact_hint"),
        rough_rgb=wm_out.get("rgb"),
        context_rgb=context_rgb,
        action_cond=action_cond,
        task_emb=task_emb,
        point=wm_out.get("point"),
        pose_geom=wm_out.get("pose_geom"),
        rgb_motion_features=wm_out.get("rgb_motion_features"),
        noisy_latents=noisy_latents,
        source_latents=source_latents,
        sigma=sigma,
        scale=float(train_cfg.get("hunyuan_control_scale", 1.0)),
    )


def action_counterfactual_source_loss(
    target: HunyuanDiTControlAdapter,
    *,
    source_latents: torch.Tensor,
    true_source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    motion_mask: torch.Tensor | None,
    wm_out: dict[str, torch.Tensor],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
    step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    modes = _parse_action_cf_modes(train_cfg.get("hunyuan_source_action_cf_modes", ()))
    interval = max(1, int(train_cfg.get("hunyuan_source_action_cf_every", 1)))
    rank_weight = float(train_cfg.get("hunyuan_source_action_cf_rank_weight", 0.0))
    sep_weight = float(train_cfg.get("hunyuan_source_action_cf_sep_weight", 0.0))
    if not modes or (rank_weight <= 0.0 and sep_weight <= 0.0) or step % interval != 0:
        return target_latents.new_zeros(()), {}

    true_f = true_source_latents.float()
    target_f = target_latents.float()
    weights, _ = _latent_source_weights(
        motion_mask,
        true_f,
        dynamic_weight=float(train_cfg.get("hunyuan_source_action_cf_dynamic_weight", train_cfg.get("hunyuan_wm_source_dynamic_weight", 24.0))),
        static_weight=float(train_cfg.get("hunyuan_source_action_cf_static_weight", train_cfg.get("hunyuan_wm_source_static_weight", 0.25))),
    )
    true_mse = _weighted_mean((true_f - target_f).square(), weights)
    total = true_source_latents.new_zeros(())
    rank_terms: list[torch.Tensor] = []
    sep_terms: list[torch.Tensor] = []
    wrong_mses: list[torch.Tensor] = []
    wrong_distances: list[torch.Tensor] = []
    source_scale = float(train_cfg.get("hunyuan_wm_source_scale", 1.0))
    rank_margin = float(train_cfg.get("hunyuan_source_action_cf_rank_margin", 0.01))
    sep_margin = float(train_cfg.get("hunyuan_source_action_cf_sep_margin", 0.03))

    for mode in modes:
        wrong_action = make_wrong_action(action_cond, mode).to(device=action_cond.device, dtype=action_cond.dtype)
        _prepare_hunyuan_controls(
            target,
            wm_out=wm_out,
            context_rgb=context_rgb,
            action_cond=wrong_action,
            task_emb=task_emb,
            train_cfg=train_cfg,
            noisy_latents=source_latents,
            source_latents=source_latents,
            sigma=source_latents.new_ones(source_latents.shape[0]),
        )
        wrong_source = source_latents + target.source_latent_delta(source_latents) * source_scale
        wrong_f = wrong_source.float()
        wrong_mse = _weighted_mean((wrong_f - target_f).square(), weights)
        wrong_distance = _weighted_mean((wrong_f - true_f).abs(), weights)
        rank = F.relu(true_mse + rank_margin - wrong_mse)
        sep = F.relu(sep_margin - wrong_distance)
        total = total + rank_weight * rank + sep_weight * sep
        rank_terms.append(rank)
        sep_terms.append(sep)
        wrong_mses.append(wrong_mse)
        wrong_distances.append(wrong_distance)

    denom = max(1, len(modes))
    total = total / float(denom)
    return total, {
        "source_action_cf_loss": float(total.detach().cpu()),
        "source_action_cf_true_mse": float(true_mse.detach().cpu()),
        "source_action_cf_wrong_mse": float(torch.stack(wrong_mses).mean().detach().cpu()) if wrong_mses else 0.0,
        "source_action_cf_rank": float(torch.stack(rank_terms).mean().detach().cpu()) if rank_terms else 0.0,
        "source_action_cf_sep": float(torch.stack(sep_terms).mean().detach().cpu()) if sep_terms else 0.0,
        "source_action_cf_distance": float(torch.stack(wrong_distances).mean().detach().cpu()) if wrong_distances else 0.0,
    }


def velocity_motion_floor_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    ratio: float,
    weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if weight <= 0.0 or ratio <= 0.0:
        return pred_velocity.new_zeros(()), {}
    pred_f = pred_velocity.float()
    target_f = target_velocity.float()
    focus = _expand_latent_motion_mask(mask, pred_f)
    if focus is None:
        focus = torch.ones_like(pred_f)
    else:
        focus = focus.clamp_min(1e-6)
    pred_mag = _weighted_mean(pred_f.abs(), focus)
    target_mag = _weighted_mean(target_f.abs(), focus).detach().clamp_min(1e-8)
    floor = F.relu(target_mag * float(ratio) - pred_mag)
    return float(weight) * floor, {
        "velocity_motion_floor": float(floor.detach().cpu()),
        "velocity_pred_motion_mag": float(pred_mag.detach().cpu()),
        "velocity_target_motion_mag": float(target_mag.detach().cpu()),
        "velocity_motion_ratio": float((pred_mag / target_mag).detach().cpu()),
    }


def _latent_velocity_residual(
    target: HunyuanDiTControlAdapter,
    noisy: torch.Tensor,
    motion_mask: torch.Tensor | None,
    ref: torch.Tensor,
    train_cfg: dict,
) -> torch.Tensor:
    residual = target.latent_residual(noisy).to(device=ref.device, dtype=ref.dtype)
    if bool(train_cfg.get("hunyuan_latent_control_motion_mask", False)):
        focus = _expand_latent_motion_mask(motion_mask, residual)
        if focus is not None:
            residual = residual * focus.to(device=residual.device, dtype=residual.dtype)
    return residual


def _forward_hunyuan_velocity(
    transformer: torch.nn.Module,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    text: dict[str, torch.Tensor],
    freqs: tuple[torch.Tensor, torch.Tensor],
    guidance: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out = transformer(
        noisy.to(dtype=dtype if device.type == "cuda" else noisy.dtype),
        timestep,
        text_states=text["text_states"],
        text_mask=text["text_mask"],
        text_states_2=text["text_states_2"],
        freqs_cos=freqs[0],
        freqs_sin=freqs[1],
        guidance=guidance,
        return_dict=True,
    )
    return out["x"] if isinstance(out, dict) else out


def action_counterfactual_velocity_loss(
    target: HunyuanDiTControlAdapter,
    *,
    transformer: torch.nn.Module,
    noisy: torch.Tensor,
    timestep: torch.Tensor,
    text: dict[str, torch.Tensor],
    freqs: tuple[torch.Tensor, torch.Tensor],
    guidance: torch.Tensor | None,
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    source_latents: torch.Tensor | None,
    motion_mask: torch.Tensor | None,
    wm_out: dict[str, torch.Tensor],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    modes = _parse_action_cf_modes(train_cfg.get("hunyuan_velocity_action_cf_modes", ()))
    rank_weight = float(train_cfg.get("hunyuan_velocity_action_cf_rank_weight", 0.0))
    sep_weight = float(train_cfg.get("hunyuan_velocity_action_cf_sep_weight", 0.0))
    if not _velocity_action_cf_active_this_step(train_cfg, step):
        return pred_velocity.new_zeros(()), {}
    interval = max(1, int(train_cfg.get("hunyuan_velocity_action_cf_every", 1)))
    if bool(train_cfg.get("hunyuan_velocity_action_cf_cycle_modes", True)) and len(modes) > 1:
        modes = [modes[(step // interval) % len(modes)]]

    pred_f = pred_velocity.float()
    target_f = target_velocity.float()
    weights, _ = _latent_source_weights(
        motion_mask,
        pred_f,
        dynamic_weight=float(train_cfg.get("hunyuan_velocity_action_cf_dynamic_weight", train_cfg.get("hunyuan_velocity_dynamic_weight", 8.0))),
        static_weight=float(train_cfg.get("hunyuan_velocity_action_cf_static_weight", train_cfg.get("hunyuan_velocity_static_weight", 1.0))),
    )
    true_mse = _weighted_mean((pred_f - target_f).square(), weights).detach()
    true_velocity = pred_velocity.detach()
    total = pred_velocity.new_zeros(())
    rank_terms: list[torch.Tensor] = []
    sep_terms: list[torch.Tensor] = []
    wrong_mses: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    rank_margin = float(train_cfg.get("hunyuan_velocity_action_cf_rank_margin", 0.004))
    sep_margin = float(train_cfg.get("hunyuan_velocity_action_cf_sep_margin", 0.012))
    latent_control_scale = float(train_cfg.get("hunyuan_latent_control_scale", 0.0))

    for mode in modes:
        wrong_action = make_wrong_action(action_cond, mode).to(device=action_cond.device, dtype=action_cond.dtype)
        _prepare_hunyuan_controls(
            target,
            wm_out=wm_out,
            context_rgb=context_rgb,
            action_cond=wrong_action,
            task_emb=task_emb,
            train_cfg=train_cfg,
            noisy_latents=noisy,
            source_latents=source_latents,
            sigma=timestep.float() / 1000.0,
        )
        wrong_velocity = _forward_hunyuan_velocity(transformer, noisy, timestep, text, freqs, guidance, dtype, device)
        if latent_control_scale != 0.0:
            wrong_velocity = wrong_velocity + _latent_velocity_residual(
                target,
                noisy,
                motion_mask,
                wrong_velocity,
                train_cfg,
            ) * latent_control_scale
        wrong_f = wrong_velocity.float()
        wrong_mse = _weighted_mean((wrong_f - target_f).square(), weights)
        distance = _weighted_mean((wrong_f - true_velocity.float()).abs(), weights)
        rank = F.relu(true_mse + rank_margin - wrong_mse)
        sep = F.relu(sep_margin - distance)
        total = total + rank_weight * rank + sep_weight * sep
        rank_terms.append(rank)
        sep_terms.append(sep)
        wrong_mses.append(wrong_mse)
        distances.append(distance)

    total = total / float(max(1, len(modes)))
    return total, {
        "velocity_action_cf_loss": float(total.detach().cpu()),
        "velocity_action_cf_true_mse": float(true_mse.detach().cpu()),
        "velocity_action_cf_wrong_mse": float(torch.stack(wrong_mses).mean().detach().cpu()) if wrong_mses else 0.0,
        "velocity_action_cf_rank": float(torch.stack(rank_terms).mean().detach().cpu()) if rank_terms else 0.0,
        "velocity_action_cf_sep": float(torch.stack(sep_terms).mean().detach().cpu()) if sep_terms else 0.0,
        "velocity_action_cf_distance": float(torch.stack(distances).mean().detach().cpu()) if distances else 0.0,
    }


def compute_hunyuan_dit_velocity_loss(
    *,
    adapter: HunyuanDiTControlAdapter,
    sampler,
    transformer: torch.nn.Module,
    wm_out: dict[str, torch.Tensor],
    batch: dict[str, Any],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    tgt: dict[str, torch.Tensor],
    device: torch.device,
    train_cfg: dict,
    precision: str,
    step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
    cleanup = lambda backward_ok=False: None
    if context_rgb is None or "rgb_tgt_p" not in tgt:
        ref = wm_out["pred_tokens"]
        zero = ref.new_zeros(())
        return zero, {"L_hunyuan_dit_velocity": zero}, cleanup

    with torch.no_grad():
        target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
        target_latents = encode_hunyuan_latents(sampler.pipeline.vae, target_video)
        frames = int(target_video.shape[2])
        height = int(target_video.shape[-2])
        width = int(target_video.shape[-1])
        prompts = prompts_from_batch(batch, int(target_latents.shape[0]))
        text = encode_hunyuan_prompts(sampler.pipeline, prompts, device)
        freqs = rotary_freqs(sampler, frames=frames, height=height, width=width, device=device)
        sigma_min = float(train_cfg.get("hunyuan_sigma_min", 1e-4))
        sigma_max = float(train_cfg.get("hunyuan_sigma_max", 1.0))
        sigma = torch.empty(target_latents.shape[0], device=device).uniform_(sigma_min, sigma_max).clamp(1e-4, 1.0)
        path_type = str(train_cfg.get("hunyuan_path_type", "context")).strip().lower()
        context_latents = encode_hunyuan_latents(sampler.pipeline.vae, context_video_from_batch(context_rgb, frames))
        if path_type == "noise":
            base_source_latents = torch.randn_like(target_latents)
        elif path_type in {"context", "wm_latent", "wm_source"}:
            base_source_latents = context_latents
        else:
            raise ValueError(
                f"stage0 Hunyuan DiT PT supports path_type noise/context/wm_latent, got {path_type!r}"
            )
        motion_mask = None
        if str(train_cfg.get("hunyuan_latent_motion_mask_source", "gt_rgb")).lower() == "gt_rgb":
            motion_mask = latent_motion_mask_from_target(
                context_rgb,
                target_video,
                latent_shape=tuple(target_latents.shape),
                threshold=float(train_cfg.get("hunyuan_latent_motion_threshold", 0.02)),
                dilate=int(train_cfg.get("hunyuan_latent_motion_dilate", 1)),
            )

    cf_step = _velocity_action_cf_active_this_step(train_cfg, step)
    separate_velocity_cf = cf_step and bool(train_cfg.get("hunyuan_velocity_action_cf_separate_backward", False))
    disable_ckpt_for_cf_step = cf_step and not separate_velocity_cf and bool(
        train_cfg.get("hunyuan_velocity_action_cf_disable_activation_checkpoint_on_cf_steps", True)
    )
    checkpoint_restore: bool | None = None
    checkpoint_disabled = False

    autocast_dtype = _dtype(precision)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        target = adapter_target(adapter)
        transformer_target = _unwrap_module(transformer)
        injector = HunyuanDiTControlInjector(transformer_target, target)
        guidance = None
        if bool(getattr(transformer_target, "guidance_embed", False)):
            guidance = torch.full(
                (target_latents.shape[0],),
                float(train_cfg.get("hunyuan_embedded_cfg_scale", 6.0)) * 1000.0,
                device=device,
                dtype=autocast_dtype if device.type == "cuda" else target_latents.dtype,
            )
        timestep = (sigma * 1000.0).to(device=device, dtype=autocast_dtype if device.type == "cuda" else target_latents.dtype)
        injector.install()
        injector._latent_shape = tuple(target_latents.shape)
        injector._double_pre_control_scale = float(train_cfg.get("hunyuan_double_pre_control_scale", 0.0))
        injector._single_pre_control_scale = float(train_cfg.get("hunyuan_single_pre_control_scale", 0.0))
        if disable_ckpt_for_cf_step:
            checkpoint_restore = bool(getattr(transformer_target, "_wm3d_activation_checkpoint", False))
            if checkpoint_restore:
                setattr(transformer_target, "_wm3d_activation_checkpoint", False)
                checkpoint_disabled = True

        def _clear_controls() -> None:
            if checkpoint_restore is not None:
                setattr(transformer_target, "_wm3d_activation_checkpoint", checkpoint_restore)
            target.clear_control_state()
            injector.remove()

        cleanup = lambda backward_ok=False: _clear_controls()
        try:
            _prepare_hunyuan_controls(
                target,
                wm_out=wm_out,
                context_rgb=context_rgb,
                action_cond=action_cond,
                task_emb=task_emb,
                train_cfg=train_cfg,
            )
            source_latents = base_source_latents.to(device=target_latents.device, dtype=autocast_dtype if device.type == "cuda" else target_latents.dtype)
            source_loss = target_latents.new_zeros(())
            source_parts: dict[str, float] = {}
            if path_type in {"wm_latent", "wm_source"}:
                _prepare_hunyuan_controls(
                    target,
                    wm_out=wm_out,
                    context_rgb=context_rgb,
                    action_cond=action_cond,
                    task_emb=task_emb,
                    train_cfg=train_cfg,
                    noisy_latents=source_latents,
                    source_latents=source_latents,
                    sigma=source_latents.new_ones(source_latents.shape[0]),
                )
                source_delta = target.source_latent_delta(source_latents) * float(train_cfg.get("hunyuan_wm_source_scale", 1.0))
                predicted_source_latents = source_latents + source_delta
                source_loss, source_parts = weighted_latent_source_losses(
                    predicted_source_latents,
                    target_latents,
                    motion_mask,
                    dynamic_weight=float(train_cfg.get("hunyuan_wm_source_dynamic_weight", 24.0)),
                    static_weight=float(train_cfg.get("hunyuan_wm_source_static_weight", 0.25)),
                    mse_weight=float(train_cfg.get("hunyuan_wm_source_mse_weight", 0.2)),
                    l1_weight=float(train_cfg.get("hunyuan_wm_source_l1_weight", 1.0)),
                    temporal_weight=float(train_cfg.get("hunyuan_wm_source_temporal_weight", 8.0)),
                    from_first_weight=float(train_cfg.get("hunyuan_wm_source_from_first_weight", 12.0)),
                    motion_floor_weight=float(train_cfg.get("hunyuan_wm_source_motion_floor_weight", 0.0)),
                    motion_floor_ratio=float(train_cfg.get("hunyuan_wm_source_motion_floor_ratio", 0.0)),
                    from_first_floor_weight=float(train_cfg.get("hunyuan_wm_source_from_first_floor_weight", 0.0)),
                    from_first_floor_ratio=float(train_cfg.get("hunyuan_wm_source_from_first_floor_ratio", 0.0)),
                )
                cf_loss, cf_parts = action_counterfactual_source_loss(
                    target,
                    source_latents=source_latents,
                    true_source_latents=predicted_source_latents,
                    target_latents=target_latents,
                    motion_mask=motion_mask,
                    wm_out=wm_out,
                    context_rgb=context_rgb,
                    action_cond=action_cond,
                    task_emb=task_emb,
                    train_cfg=train_cfg,
                    step=step,
                )
                source_loss = source_loss + cf_loss
                source_parts.update(cf_parts)
                _prepare_hunyuan_controls(
                    target,
                    wm_out=wm_out,
                    context_rgb=context_rgb,
                    action_cond=action_cond,
                    task_emb=task_emb,
                    train_cfg=train_cfg,
                )
                if bool(train_cfg.get("hunyuan_wm_source_detach_for_dit", True)):
                    source_latents = predicted_source_latents.detach()
                else:
                    source_latents = predicted_source_latents

            noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
            _prepare_hunyuan_controls(
                target,
                wm_out=wm_out,
                context_rgb=context_rgb,
                action_cond=action_cond,
                task_emb=task_emb,
                train_cfg=train_cfg,
                noisy_latents=noisy,
                source_latents=source_latents,
                sigma=sigma,
            )
            velocity_source = source_latents.detach() if bool(train_cfg.get("hunyuan_velocity_detach_source_target", True)) else source_latents
            target_velocity = float(train_cfg.get("hunyuan_velocity_target_sign", -1.0)) * (velocity_source - target_latents)
            pred_velocity = _forward_hunyuan_velocity(
                transformer,
                noisy,
                timestep,
                text,
                freqs,
                guidance,
                autocast_dtype,
                device,
            )
            latent_control_scale = float(train_cfg.get("hunyuan_latent_control_scale", 1.0))
            if latent_control_scale != 0.0:
                pred_velocity = pred_velocity + _latent_velocity_residual(
                    target,
                    noisy,
                    motion_mask,
                    pred_velocity,
                    train_cfg,
                ) * latent_control_scale

            mse, l1, parts = weighted_velocity_losses(
                pred_velocity,
                target_velocity,
                motion_mask,
                dynamic_weight=float(train_cfg.get("hunyuan_velocity_dynamic_weight", 8.0)),
                static_weight=float(train_cfg.get("hunyuan_velocity_static_weight", 1.0)),
            )
            velocity_floor, velocity_floor_parts = velocity_motion_floor_loss(
                pred_velocity,
                target_velocity,
                motion_mask,
                ratio=float(train_cfg.get("hunyuan_velocity_motion_floor_ratio", 0.0)),
                weight=float(train_cfg.get("hunyuan_velocity_motion_floor_weight", 0.0)),
            )
            parts.update(velocity_floor_parts)
            velocity_loss = (
                float(train_cfg.get("hunyuan_velocity_mse_weight", 1.0)) * mse
                + float(train_cfg.get("hunyuan_velocity_l1_weight", 0.05)) * l1
                + velocity_floor
            )
            decoded_rgb_loss = pred_velocity.new_zeros(())
            decoded_rgb_parts: dict[str, float] = {}
            decoded_rgb_enabled = bool(train_cfg.get("hunyuan_decoded_rgb_loss", False))
            decoded_rgb_every = max(1, int(train_cfg.get("hunyuan_decoded_rgb_every", 1) or 1))
            if decoded_rgb_enabled and step % decoded_rgb_every == 0:
                decoded_samples = int(train_cfg.get("hunyuan_decoded_rgb_samples", 1) or 1)
                decoded_samples = max(1, min(decoded_samples, int(pred_velocity.shape[0])))
                pred_target_latents = noisy[:decoded_samples].float() - (
                    sigma[:decoded_samples, None, None, None, None].float() * pred_velocity[:decoded_samples].float()
                )
                pred_video = _decode_hunyuan_latents_for_loss(sampler.pipeline.vae, pred_target_latents)
                decoded_rgb_loss, decoded_rgb_parts = decoded_rgb_losses(
                    pred_video,
                    target_video[:decoded_samples],
                    context_rgb[:decoded_samples],
                    train_cfg=train_cfg,
                )
            decoded_rgb_parts["decoded_rgb_loss"] = float(decoded_rgb_loss.detach().cpu())
            decoded_rgb_parts["decoded_rgb_active"] = float(decoded_rgb_enabled and step % decoded_rgb_every == 0)
            velocity_cf = pred_velocity.new_zeros(())
            velocity_cf_parts: dict[str, float] = {}
            if separate_velocity_cf:
                cf_wm_out = {
                    key: value.detach() if torch.is_tensor(value) else value
                    for key, value in wm_out.items()
                }
                cf_text = {key: value.detach() for key, value in text.items()}
                cf_freqs = (freqs[0].detach(), freqs[1].detach())
                cf_guidance = guidance.detach() if guidance is not None else None
                cf_context = {
                    "noisy": noisy.detach(),
                    "timestep": timestep.detach(),
                    "sigma": sigma.detach(),
                    "text": cf_text,
                    "freqs": cf_freqs,
                    "guidance": cf_guidance,
                    "pred_velocity": pred_velocity.detach(),
                    "target_velocity": target_velocity.detach(),
                    "source_latents": source_latents.detach(),
                    "motion_mask": motion_mask.detach() if motion_mask is not None else None,
                    "wm_out": cf_wm_out,
                    "context_rgb": context_rgb.detach(),
                    "action_cond": action_cond.detach(),
                    "task_emb": task_emb.detach(),
                }

                def _post_backward_cleanup(backward_ok: bool = False) -> None:
                    try:
                        if backward_ok:
                            target.clear_control_state()
                            with torch.enable_grad(), torch.autocast(
                                device_type="cuda",
                                dtype=autocast_dtype,
                                enabled=device.type == "cuda",
                            ):
                                cf_loss, cf_parts = action_counterfactual_velocity_loss(
                                    target,
                                    transformer=transformer,
                                    noisy=cf_context["noisy"],
                                    timestep=cf_context["timestep"],
                                    text=cf_context["text"],
                                    freqs=cf_context["freqs"],
                                    guidance=cf_context["guidance"],
                                    pred_velocity=cf_context["pred_velocity"],
                                    target_velocity=cf_context["target_velocity"],
                                    source_latents=cf_context["source_latents"],
                                    motion_mask=cf_context["motion_mask"],
                                    wm_out=cf_context["wm_out"],
                                    context_rgb=cf_context["context_rgb"],
                                    action_cond=cf_context["action_cond"],
                                    task_emb=cf_context["task_emb"],
                                    train_cfg=train_cfg,
                                    step=step,
                                    dtype=autocast_dtype,
                                    device=device,
                                )
                            finite_count = _distributed_finite_count(
                                torch.isfinite(cf_loss),
                                device,
                                dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1,
                            )
                            if finite_count == (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1):
                                cf_loss.backward()
                            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                                print(
                                    "[rank0] post_backward_velocity_cf "
                                    f"step={step} loss={float(cf_loss.detach().cpu()):.5f} "
                                    f"rank={float(cf_parts.get('velocity_action_cf_rank', 0.0)):.5f} "
                                    f"sep={float(cf_parts.get('velocity_action_cf_sep', 0.0)):.5f} "
                                    f"dist={float(cf_parts.get('velocity_action_cf_distance', 0.0)):.5f}",
                                    flush=True,
                                )
                    finally:
                        _clear_controls()

                cleanup = _post_backward_cleanup
                velocity_cf_parts["velocity_action_cf_separate_backward"] = 1.0
            else:
                velocity_cf, velocity_cf_parts = action_counterfactual_velocity_loss(
                    target,
                    transformer=transformer,
                    noisy=noisy,
                    timestep=timestep,
                    text=text,
                    freqs=freqs,
                    guidance=guidance,
                    pred_velocity=pred_velocity,
                    target_velocity=target_velocity,
                    source_latents=source_latents,
                    motion_mask=motion_mask,
                    wm_out=wm_out,
                    context_rgb=context_rgb,
                    action_cond=action_cond,
                    task_emb=task_emb,
                    train_cfg=train_cfg,
                    step=step,
                    dtype=autocast_dtype,
                    device=device,
                )
                if velocity_cf_parts:
                    _prepare_hunyuan_controls(
                        target,
                        wm_out=wm_out,
                        context_rgb=context_rgb,
                        action_cond=action_cond,
                        task_emb=task_emb,
                        train_cfg=train_cfg,
                    )
            velocity_loss = velocity_loss + velocity_cf + decoded_rgb_loss
            parts.update(velocity_cf_parts)
            parts.update(decoded_rgb_parts)
            if cf_step:
                parts["velocity_action_cf_checkpoint_disabled"] = float(checkpoint_disabled)
            loss = velocity_loss + source_loss
        except Exception:
            cleanup(False)
            cleanup = lambda backward_ok=False: None
            raise
    out = {
        "L_hunyuan_dit_velocity": velocity_loss,
        "L_hunyuan_wm_source": source_loss.detach(),
        "hunyuan_dit_velocity_mse": pred_velocity.new_tensor(parts.get("velocity_mse", 0.0)),
        "hunyuan_dit_velocity_l1": pred_velocity.new_tensor(parts.get("velocity_l1", 0.0)),
        "hunyuan_dit_dynamic_mse": pred_velocity.new_tensor(parts.get("velocity_dynamic_mse", 0.0)),
        "hunyuan_dit_static_mse": pred_velocity.new_tensor(parts.get("velocity_static_mse", 0.0)),
        "hunyuan_dit_velocity_motion_floor": pred_velocity.new_tensor(parts.get("velocity_motion_floor", 0.0)),
        "hunyuan_dit_velocity_motion_ratio": pred_velocity.new_tensor(parts.get("velocity_motion_ratio", 0.0)),
        "hunyuan_dit_velocity_action_cf": pred_velocity.new_tensor(parts.get("velocity_action_cf_loss", 0.0)),
        "hunyuan_dit_velocity_action_cf_rank": pred_velocity.new_tensor(parts.get("velocity_action_cf_rank", 0.0)),
        "hunyuan_dit_velocity_action_cf_sep": pred_velocity.new_tensor(parts.get("velocity_action_cf_sep", 0.0)),
        "hunyuan_dit_velocity_action_cf_distance": pred_velocity.new_tensor(parts.get("velocity_action_cf_distance", 0.0)),
        "hunyuan_dit_velocity_action_cf_wrong_mse": pred_velocity.new_tensor(parts.get("velocity_action_cf_wrong_mse", 0.0)),
        "hunyuan_dit_velocity_action_cf_ckpt_disabled": pred_velocity.new_tensor(parts.get("velocity_action_cf_checkpoint_disabled", 0.0)),
        "hunyuan_dit_velocity_action_cf_separate_backward": pred_velocity.new_tensor(parts.get("velocity_action_cf_separate_backward", 0.0)),
        "hunyuan_decoded_rgb_loss": pred_velocity.new_tensor(parts.get("decoded_rgb_loss", 0.0)),
        "hunyuan_decoded_rgb_active": pred_velocity.new_tensor(parts.get("decoded_rgb_active", 0.0)),
        "hunyuan_decoded_rgb_l1": pred_velocity.new_tensor(parts.get("decoded_rgb_l1", 0.0)),
        "hunyuan_decoded_rgb_mse": pred_velocity.new_tensor(parts.get("decoded_rgb_mse", 0.0)),
        "hunyuan_decoded_rgb_dynamic_l1": pred_velocity.new_tensor(parts.get("decoded_rgb_dynamic_l1", 0.0)),
        "hunyuan_decoded_rgb_static_l1": pred_velocity.new_tensor(parts.get("decoded_rgb_static_l1", 0.0)),
        "hunyuan_decoded_rgb_temporal_l1": pred_velocity.new_tensor(parts.get("decoded_rgb_temporal_l1", 0.0)),
        "hunyuan_decoded_rgb_from_first_l1": pred_velocity.new_tensor(parts.get("decoded_rgb_from_first_l1", 0.0)),
        "hunyuan_decoded_rgb_motion_floor": pred_velocity.new_tensor(parts.get("decoded_rgb_motion_floor", 0.0)),
        "hunyuan_decoded_rgb_from_first_floor": pred_velocity.new_tensor(parts.get("decoded_rgb_from_first_floor", 0.0)),
        "hunyuan_decoded_rgb_motion_ratio": pred_velocity.new_tensor(parts.get("decoded_rgb_motion_ratio", 0.0)),
        "hunyuan_decoded_rgb_from_first_ratio": pred_velocity.new_tensor(parts.get("decoded_rgb_from_first_ratio", 0.0)),
        "hunyuan_decoded_rgb_mask_mean": pred_velocity.new_tensor(parts.get("decoded_rgb_mask_mean", 0.0)),
        "hunyuan_dit_motion_mask": pred_velocity.new_tensor(parts.get("latent_motion_mask_mean", 0.0)),
        "hunyuan_dit_sigma": sigma.detach().float().mean(),
        "hunyuan_wm_source_mse": pred_velocity.new_tensor(source_parts.get("wm_source_mse", 0.0)),
        "hunyuan_wm_source_l1": pred_velocity.new_tensor(source_parts.get("wm_source_l1", 0.0)),
        "hunyuan_wm_source_temporal_l1": pred_velocity.new_tensor(source_parts.get("wm_source_temporal_l1", 0.0)),
        "hunyuan_wm_source_from_first_l1": pred_velocity.new_tensor(source_parts.get("wm_source_from_first_l1", 0.0)),
        "hunyuan_wm_source_motion_ratio": pred_velocity.new_tensor(source_parts.get("wm_source_motion_ratio", 0.0)),
        "hunyuan_wm_source_motion_floor": pred_velocity.new_tensor(source_parts.get("wm_source_motion_floor", 0.0)),
        "hunyuan_wm_source_from_first_floor": pred_velocity.new_tensor(source_parts.get("wm_source_from_first_floor", 0.0)),
        "hunyuan_source_action_cf_loss": pred_velocity.new_tensor(source_parts.get("source_action_cf_loss", 0.0)),
        "hunyuan_source_action_cf_rank": pred_velocity.new_tensor(source_parts.get("source_action_cf_rank", 0.0)),
        "hunyuan_source_action_cf_sep": pred_velocity.new_tensor(source_parts.get("source_action_cf_sep", 0.0)),
        "hunyuan_source_action_cf_distance": pred_velocity.new_tensor(source_parts.get("source_action_cf_distance", 0.0)),
        "hunyuan_source_action_cf_wrong_mse": pred_velocity.new_tensor(source_parts.get("source_action_cf_wrong_mse", 0.0)),
    }
    return loss, out, cleanup


def save_checkpoint(
    *,
    path: Path,
    wm_model: torch.nn.Module,
    adapter: HunyuanDiTControlAdapter,
    transformer: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched,
    cfg: dict,
    step: int,
    epoch: int,
    metrics: dict[str, float] | None,
    rank: int = 0,
    fsdp_enabled: bool = False,
) -> None:
    model_state = _state_dict_for_checkpoint(wm_model, rank=rank)
    hunyuan_state = _trainable_hunyuan_state_for_checkpoint(transformer, rank=rank)
    if rank != 0:
        return
    opt_state = {} if fsdp_enabled else opt.state_dict()
    payload = {
        "kind": "wm3d_stage0_hunyuan_dit_jointpt_v1",
        "model": model_state,
        "hunyuan_control_adapter": adapter_target(adapter).state_dict(),
        "hunyuan_control_adapter_cfg": adapter_target(adapter).cfg.__dict__,
        "hunyuan_trainable": hunyuan_state,
        "opt": opt_state,
        "sched": sched.state_dict(),
        "cfg": cfg,
        "step": int(step),
        "epoch": int(epoch),
        "metrics": metrics or {},
    }
    if fsdp_enabled:
        payload["fsdp_optimizer_state_omitted"] = True
    tmp = path.with_name("." + path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _state_dict_for_checkpoint(module: torch.nn.Module, *, rank: int) -> dict[str, Any]:
    if _is_fsdp(module):
        assert FSDP is not None and FullStateDictConfig is not None and StateDictType is not None
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
            state = module.state_dict()
        return state if rank == 0 else {}
    return module.state_dict() if rank == 0 else {}


def _trainable_hunyuan_state_for_checkpoint(transformer: torch.nn.Module, *, rank: int) -> dict[str, torch.Tensor]:
    if _is_fsdp(transformer):
        assert FSDP is not None
        with FSDP.summon_full_params(transformer, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True):
            if rank == 0:
                return collect_trainable_state_dict(_unwrap_module(transformer))
        return {}
    return collect_trainable_state_dict(transformer) if rank == 0 else {}


def save_hunyuan_trainable_checkpoint_maybe_fsdp(
    path: Path,
    transformer: torch.nn.Module,
    *,
    train_cfg: dict,
    step: int,
    rank: int,
) -> None:
    kwargs = {
        "lora_config": HunyuanLoRAConfig(
            rank=int(train_cfg.get("hunyuan_lora_rank", 8)),
            alpha=float(train_cfg.get("hunyuan_lora_alpha", 16.0)),
            dropout=float(train_cfg.get("hunyuan_lora_dropout", 0.0)),
            include=tuple(train_cfg.get("hunyuan_lora_include", ("double_blocks", "single_blocks"))),
            exclude=tuple(train_cfg.get("hunyuan_lora_exclude", ())),
            dtype=str(train_cfg.get("hunyuan_lora_dtype", "fp32")),
            checkpoint=bool(train_cfg.get("hunyuan_lora_checkpoint", False)),
            checkpoint_use_reentrant=bool(train_cfg.get("hunyuan_lora_checkpoint_use_reentrant", False)),
        ),
        "partial_unfreeze": train_cfg.get("hunyuan_dit_trainable_patterns", ()),
        "step": step,
    }
    if _is_fsdp(transformer):
        assert FSDP is not None
        with FSDP.summon_full_params(transformer, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True):
            if rank == 0:
                save_hunyuan_trainable_checkpoint(path, _unwrap_module(transformer), **kwargs)
    elif rank == 0:
        save_hunyuan_trainable_checkpoint(path, transformer, **kwargs)


def _step_from_path(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def prune_step_checkpoints(ckpt_dir: Path, train_cfg: dict) -> None:
    keep_last = max(0, int(train_cfg.get("keep_last_checkpoints", 2)))
    milestone_every = max(0, int(train_cfg.get("milestone_every_steps", 50000)))
    paths = sorted(ckpt_dir.glob("step_*.pt"), key=_step_from_path)
    keep = set(paths[-keep_last:]) if keep_last > 0 else set()
    if milestone_every > 0:
        keep.update(path for path in paths if _step_from_path(path) > 0 and _step_from_path(path) % milestone_every == 0)
    for path in paths:
        if path not in keep:
            path.unlink(missing_ok=True)


def link_latest(ckpt_dir: Path, path: Path) -> None:
    latest = ckpt_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="From-scratch wm3d stage0 PT with Hunyuan DiT RGB loss.")
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--hunyuan_repo", type=Path, default=Path("/data/Minko/external/HunyuanVideo"))
    ap.add_argument("--hunyuan_model_base", type=Path, default=Path("/data/Minko/models/hunyuan_video"))
    ap.add_argument("--hunyuan_dit_weight", type=Path, default=Path("/data/Minko/models/hunyuan_video/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"))
    ap.add_argument("--hunyuan_model_resolution", default="540p")
    ap.add_argument("--hunyuan_precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--vae_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--text_encoder_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--text_encoder_precision_2", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--hunyuan_use_fp8", action="store_true", default=True)
    ap.add_argument("--no_hunyuan_use_fp8", action="store_false", dest="hunyuan_use_fp8")
    ap.add_argument("--flow_shift", type=float, default=7.0)
    ap.add_argument("--embedded_cfg_scale", type=float, default=6.0)
    ap.add_argument("--max_train_windows", type=int, default=0)
    ap.add_argument("--max_val_windows", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--ckpt_every_steps", type=int, default=None)
    ap.add_argument("--eval_every_steps", type=int, default=None)
    ap.add_argument("--no_epoch_checkpoint", action="store_true")
    ap.add_argument("--disable_hunyuan_lora", action="store_true")
    args = ap.parse_args()

    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(args.cfg.read_text())
    cfg.setdefault("data", {})["load_task_text"] = True
    cfg["model"]["enable_pixel"] = False
    cfg["model"]["enable_context_pixel"] = False
    train_cfg = cfg["train"]
    velocity_cf_enabled = _velocity_action_cf_enabled(train_cfg)
    velocity_cf_uses_checkpoint = velocity_cf_enabled and bool(train_cfg.get("hunyuan_activation_checkpoint", False))
    velocity_cf_separate_backward = velocity_cf_enabled and bool(train_cfg.get("hunyuan_velocity_action_cf_separate_backward", False))
    if velocity_cf_uses_checkpoint and not velocity_cf_separate_backward and not bool(
        train_cfg.get("hunyuan_velocity_action_cf_disable_activation_checkpoint_on_cf_steps", True)
    ):
        raise RuntimeError(
            "hunyuan_velocity_action_cf cannot be used with hunyuan_activation_checkpoint=true: "
            "checkpoint recomputation would reuse mutable Hunyuan control hook state across true/wrong action forwards. "
            "Set hunyuan_velocity_action_cf_disable_activation_checkpoint_on_cf_steps=true, "
            "disable activation checkpointing, or use source-level action CF only."
        )
    if rank == 0 and velocity_cf_uses_checkpoint:
        if velocity_cf_separate_backward:
            print(
                "[rank0] velocity_action_cf uses separate post-backward forward/backward with Hunyuan activation checkpoint kept on",
                flush=True,
            )
        else:
            print(
                "[rank0] velocity_action_cf will temporarily disable Hunyuan activation checkpoint on CF steps",
                flush=True,
            )
    if rank == 0:
        print(
            "[rank0] Hunyuan control injection "
            f"output_scale={float(train_cfg.get('hunyuan_control_scale', 1.0)):.3f} "
            f"double_pre={float(train_cfg.get('hunyuan_double_pre_control_scale', 0.0)):.3f} "
            f"single_pre={float(train_cfg.get('hunyuan_single_pre_control_scale', 0.0)):.3f} "
            f"latent={float(train_cfg.get('hunyuan_latent_control_scale', 0.0)):.3f} "
            f"action_direct={float(train_cfg.get('hunyuan_action_direct_scale', 0.0)):.3f} "
            f"action_latent={float(train_cfg.get('hunyuan_action_latent_scale', 0.0)):.3f} "
            f"action_xattn={bool(train_cfg.get('hunyuan_use_action_cross_attn', False))} "
            f"action_xattn_scale={float(train_cfg.get('hunyuan_action_cross_attn_scale', 0.0)):.3f} "
            f"action_xattn_hidden={int(train_cfg.get('hunyuan_action_cross_attn_hidden', train_cfg.get('hunyuan_control_hidden', 192)))} "
            f"action_xattn_heads={int(train_cfg.get('hunyuan_action_cross_attn_heads', 4))} "
            f"temporal_action_summary={bool(train_cfg.get('hunyuan_use_temporal_action_summary', False))} "
            f"temporal_action_summary_scale={float(train_cfg.get('hunyuan_temporal_action_summary_scale', 1.0)):.3f} "
            f"parallel_action_dit={bool(train_cfg.get('hunyuan_use_parallel_action_dit', False))} "
            f"native_parallel_action_forward={bool(train_cfg.get('hunyuan_native_parallel_action_forward', False))} "
            f"parallel_action_dit_scale={float(train_cfg.get('hunyuan_parallel_action_dit_scale', 0.0)):.3f} "
            f"parallel_action_dit_hidden={int(train_cfg.get('hunyuan_parallel_action_dit_hidden', 256))} "
            f"parallel_action_dit_heads={int(train_cfg.get('hunyuan_parallel_action_dit_heads', 4))} "
            f"noisy_latents={bool(train_cfg.get('hunyuan_control_use_noisy_latents', False))} "
            f"source_latents={bool(train_cfg.get('hunyuan_control_use_source_latents', False))} "
            f"sigma_embed={bool(train_cfg.get('hunyuan_control_use_sigma_embed', False))}",
            flush=True,
        )
    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.disable_hunyuan_lora:
        train_cfg["hunyuan_dit_train_lora"] = False
    out_dir = args.out_dir or Path(cfg["out"]["root"])
    ckpt_dir = out_dir / cfg["out"].get("ckpt_dir", "ckpt")
    precision = str(train_cfg.get("precision", "bf16"))
    init_seed = int(cfg["data"].get("seed", 909))
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    train_ds, val_ds = build_datasets(cfg)
    if args.max_train_windows and args.max_train_windows < len(train_ds):
        train_ds = torch.utils.data.Subset(train_ds, list(range(int(args.max_train_windows))))
    if args.max_val_windows and args.max_val_windows < len(val_ds):
        val_ds = torch.utils.data.Subset(val_ds, list(range(int(args.max_val_windows))))
    train_sampler = _make_sampler(train_ds, cfg, world=world, rank=rank, local=local, train=True)
    val_sampler = _make_sampler(val_ds, cfg, world=world, rank=rank, local=local, train=False)
    bs = int(train_cfg.get("batch_size_per_gpu", 1))
    nw = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_sampler, num_workers=nw, pin_memory=True)

    wm_model = build_model(cfg).to(device)
    load_action_stats_if_available(wm_model, cfg, rank, device)
    sampler, transformer, hunyuan_control_adapter, lora_report, pattern_report = build_hunyuan_modules(args, cfg, device, rank, world)
    init_ckpt = train_cfg.get("init_from_stage0_hunyuan_ckpt")
    if init_ckpt:
        if rank == 0:
            load_stage0_hunyuan_init(
                init_ckpt,
                wm_model=wm_model,
                adapter=hunyuan_control_adapter,
                transformer=transformer,
                device=device,
                rank=rank,
            )
        if world > 1:
            dist.barrier()
    broadcast_module_state(wm_model, world)
    broadcast_trainable_parameters(transformer, world)
    broadcast_module_state(hunyuan_control_adapter, world)
    if world > 1:
        dist.barrier()
    wm_model, transformer, fsdp_report = maybe_wrap_fsdp(
        wm_model,
        transformer,
        train_cfg,
        device=device,
        rank=rank,
        world=world,
    )
    runtime_seed = init_seed + rank
    torch.manual_seed(runtime_seed)
    np.random.seed(runtime_seed)
    wm_model.train()
    hunyuan_control_adapter.train()

    weights = LossWeights(**cfg["loss"])
    opt = build_optimizer(wm_model, hunyuan_control_adapter, transformer, train_cfg, rank)
    max_steps = int(train_cfg.get("max_steps", 0) or 0)
    total_steps = max(1, len(train_loader) * int(train_cfg.get("epochs", 1)))
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    sched = build_lr_scheduler(opt, cfg, total_steps)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ckpt_every = int(args.ckpt_every_steps or train_cfg.get("ckpt_every_steps", 2500) or 0)
    eval_every = int(args.eval_every_steps or train_cfg.get("eval_every_steps", 0) or 0)
    empty_cache_every = int(train_cfg.get("empty_cache_every_steps", 0) or 0)

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_dir / cfg["out"].get("tb_dir", "tb"))
        metadata = {
            "kind": "wm3d_stage0_hunyuan_dit_jointpt_v1",
            "cfg": str(args.cfg),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "world_size": world,
            "total_steps": total_steps,
            "rgb_decoder": "disabled",
            "hunyuan_lora": lora_report,
            "hunyuan_partial_train": pattern_report,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[rank0] stage0_hunyuan_dit train_windows={len(train_ds)} val_windows={len(val_ds)} world={world} total_steps={total_steps}", flush=True)
    else:
        tb = None
    if world > 1:
        dist.barrier()

    step = 0
    best_val = float("inf")
    for epoch in range(int(train_cfg.get("epochs", 1))):
        if max_steps > 0 and step >= max_steps:
            break
        train_sampler.set_epoch(epoch)
        wm_model.train()
        hunyuan_control_adapter.train()
        transformer.train()
        for batch in train_loader:
            if max_steps > 0 and step >= max_steps:
                break
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, int(cfg["data"]["k"]))
            with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=_dtype(precision), enabled=device.type == "cuda"):
                s_cond, c_cond, action_cond_model, context_rgb_cond, dropout_losses = apply_condition_dropout(
                    s, c, action_cond, context_rgb, train_cfg, training=True
                )
                out = _forward_joint_model(
                    wm_model,
                    s_cond,
                    c_cond,
                    action_cond=action_cond_model,
                    context_rgb=context_rgb_cond,
                    prior_clean_tokens=prior_clean_tokens_from_targets(tgt),
                    pixel=False,
                    bridging=False,
                )
                losses = compute_losses(out, tgt, weights, None)
                losses.update({k: v.detach() for k, v in dropout_losses.items()})
            dit_cleanup = lambda backward_ok=False: None
            with torch.enable_grad():
                dit_loss, dit_parts, dit_cleanup = compute_hunyuan_dit_velocity_loss(
                    adapter=hunyuan_control_adapter,
                    sampler=sampler,
                    transformer=transformer,
                    wm_out=out,
                    batch=batch,
                    context_rgb=context_rgb_cond,
                    action_cond=action_cond_model,
                    task_emb=c_cond,
                    tgt=tgt,
                    device=device,
                    train_cfg=train_cfg,
                    precision=precision,
                    step=step,
                )
                losses.update({k: v.detach() for k, v in dit_parts.items()})
                loss = losses["L_total"] + float(train_cfg.get("hunyuan_dit_weight", 1.0)) * dit_loss
            finite_count = _distributed_finite_count(torch.isfinite(loss), device, world)
            opt.zero_grad(set_to_none=True)
            if finite_count != world:
                dit_cleanup(False)
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite loss step={step} finite={finite_count}/{world}", flush=True)
                sched.step()
                step += 1
                continue
            backward_ok = False
            try:
                loss.backward()
                backward_ok = True
            finally:
                dit_cleanup(backward_ok)
            sync_module(wm_model, world)
            sync_module(hunyuan_control_adapter, world)
            sync_module(transformer, world)
            params = [p for g in opt.param_groups for p in g["params"] if p.grad is not None]
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip) if params else loss.new_zeros(())
            grad_finite = _distributed_finite_count(torch.isfinite(gn), device, world)
            if grad_finite != world:
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite grad step={step} finite={grad_finite}/{world}", flush=True)
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                continue
            opt.step()
            sched.step()

            if rank == 0 and args.print_every and step % int(args.print_every) == 0:
                depth = float(losses.get("L_depth", loss.new_zeros(())).detach().float())
                point = float(losses.get("L_point", loss.new_zeros(())).detach().float())
                pose = float(losses.get("L_pose", loss.new_zeros(())).detach().float())
                action = float(losses.get("L_action", loss.new_zeros(())).detach().float())
                hv = float(losses.get("L_hunyuan_dit_velocity", loss.new_zeros(())).detach().float())
                hmse = float(losses.get("hunyuan_dit_velocity_mse", loss.new_zeros(())).detach().float())
                hdyn = float(losses.get("hunyuan_dit_dynamic_mse", loss.new_zeros(())).detach().float())
                hsta = float(losses.get("hunyuan_dit_static_mse", loss.new_zeros(())).detach().float())
                hvel_floor = float(losses.get("hunyuan_dit_velocity_motion_floor", loss.new_zeros(())).detach().float())
                hvel_ratio = float(losses.get("hunyuan_dit_velocity_motion_ratio", loss.new_zeros(())).detach().float())
                sig = float(losses.get("hunyuan_dit_sigma", loss.new_zeros(())).detach().float())
                hsrc = float(losses.get("L_hunyuan_wm_source", loss.new_zeros(())).detach().float())
                hsrc_ff = float(losses.get("hunyuan_wm_source_from_first_l1", loss.new_zeros(())).detach().float())
                hsrc_ratio = float(losses.get("hunyuan_wm_source_motion_ratio", loss.new_zeros(())).detach().float())
                hsrc_floor = float(losses.get("hunyuan_wm_source_motion_floor", loss.new_zeros(())).detach().float())
                hsrc_ff_floor = float(losses.get("hunyuan_wm_source_from_first_floor", loss.new_zeros(())).detach().float())
                hcf = float(losses.get("hunyuan_source_action_cf_loss", loss.new_zeros(())).detach().float())
                hcf_rank = float(losses.get("hunyuan_source_action_cf_rank", loss.new_zeros(())).detach().float())
                hcf_dist = float(losses.get("hunyuan_source_action_cf_distance", loss.new_zeros(())).detach().float())
                hvcf = float(losses.get("hunyuan_dit_velocity_action_cf", loss.new_zeros(())).detach().float())
                hvcf_rank = float(losses.get("hunyuan_dit_velocity_action_cf_rank", loss.new_zeros(())).detach().float())
                hvcf_sep = float(losses.get("hunyuan_dit_velocity_action_cf_sep", loss.new_zeros(())).detach().float())
                hvcf_dist = float(losses.get("hunyuan_dit_velocity_action_cf_distance", loss.new_zeros(())).detach().float())
                hvcf_ckpt = float(losses.get("hunyuan_dit_velocity_action_cf_ckpt_disabled", loss.new_zeros(())).detach().float())
                hvcf_sep_bw = float(losses.get("hunyuan_dit_velocity_action_cf_separate_backward", loss.new_zeros(())).detach().float())
                hrgb = float(losses.get("hunyuan_decoded_rgb_loss", loss.new_zeros(())).detach().float())
                hrgb_active = float(losses.get("hunyuan_decoded_rgb_active", loss.new_zeros(())).detach().float())
                hrgb_l1 = float(losses.get("hunyuan_decoded_rgb_l1", loss.new_zeros(())).detach().float())
                hrgb_dyn = float(losses.get("hunyuan_decoded_rgb_dynamic_l1", loss.new_zeros(())).detach().float())
                hrgb_static = float(losses.get("hunyuan_decoded_rgb_static_l1", loss.new_zeros(())).detach().float())
                hrgb_temp = float(losses.get("hunyuan_decoded_rgb_temporal_l1", loss.new_zeros(())).detach().float())
                hrgb_ff = float(losses.get("hunyuan_decoded_rgb_from_first_l1", loss.new_zeros(())).detach().float())
                hrgb_mratio = float(losses.get("hunyuan_decoded_rgb_motion_ratio", loss.new_zeros(())).detach().float())
                hrgb_ffratio = float(losses.get("hunyuan_decoded_rgb_from_first_ratio", loss.new_zeros(())).detach().float())
                hrgb_mask = float(losses.get("hunyuan_decoded_rgb_mask_mean", loss.new_zeros(())).detach().float())
                print(
                    f"[rank0] step {step} ep={epoch} L_total={float(loss.detach().float()):.4f} "
                    f"native={float(losses['L_total'].detach().float()):.4f} "
                    f"depth={depth:.4f} point={point:.4f} pose={pose:.4f} action={action:.4f} "
                    f"hunyuan_dit_velocity={hv:.4f} h_src={hsrc:.4f} "
                    f"h_mse={hmse:.5f} h_dyn={hdyn:.5f} h_static={hsta:.5f} "
                    f"h_vel_floor={hvel_floor:.5f} h_vel_ratio={hvel_ratio:.3f} "
                    f"h_src_ff={hsrc_ff:.5f} h_src_motion_ratio={hsrc_ratio:.3f} "
                    f"h_src_floor={hsrc_floor:.5f} h_src_ff_floor={hsrc_ff_floor:.5f} "
                    f"h_cf={hcf:.5f} h_cf_rank={hcf_rank:.5f} h_cf_dist={hcf_dist:.5f} "
                    f"h_vel_cf={hvcf:.5f} h_vel_cf_rank={hvcf_rank:.5f} h_vel_cf_sep={hvcf_sep:.5f} "
                    f"h_vel_cf_dist={hvcf_dist:.5f} h_vel_cf_ckpt_off={hvcf_ckpt:.0f} "
                    f"h_rgb={hrgb:.5f} h_rgb_active={hrgb_active:.0f} "
                    f"h_rgb_l1={hrgb_l1:.5f} h_rgb_dyn={hrgb_dyn:.5f} h_rgb_static={hrgb_static:.5f} "
                    f"h_rgb_temp={hrgb_temp:.5f} h_rgb_ff={hrgb_ff:.5f} "
                    f"h_rgb_mratio={hrgb_mratio:.3f} h_rgb_ffratio={hrgb_ffratio:.3f} h_rgb_mask={hrgb_mask:.3f} "
                    f"h_vel_cf_sep_bw={hvcf_sep_bw:.0f} sigma={sig:.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if rank == 0 and tb is not None and step % int(train_cfg.get("log_every", 50)) == 0:
                tb.add_scalar("train/L_total_with_hunyuan", float(loss.detach().float()), step)
                for key, value in losses.items():
                    tb.add_scalar(f"train/{key}", float(value.detach().float()), step)
                tb.add_scalar("lr/wm", sched.get_last_lr()[0], step)
            step += 1

            if ckpt_every > 0 and step % ckpt_every == 0:
                step_path = ckpt_dir / f"step_{step:08d}.pt"
                save_checkpoint(
                    path=step_path,
                    wm_model=wm_model,
                    adapter=hunyuan_control_adapter,
                    transformer=transformer,
                    opt=opt,
                    sched=sched,
                    cfg=cfg,
                    step=step,
                    epoch=epoch,
                    metrics=None,
                    rank=rank,
                    fsdp_enabled=bool(fsdp_report.get("enabled", False)),
                )
                save_hunyuan_trainable_checkpoint_maybe_fsdp(
                    ckpt_dir / f"hunyuan_trainable_step_{step:08d}.pt",
                    transformer,
                    train_cfg=train_cfg,
                    step=step,
                    rank=rank,
                )
                if rank == 0:
                    link_latest(ckpt_dir, step_path)
                    prune_step_checkpoints(ckpt_dir, train_cfg)
                    control_path = ckpt_dir / f"hunyuan_control_step_{step:08d}.pt"
                    save_hunyuan_dit_control_checkpoint(control_path, hunyuan_control_adapter, wm_ckpt=step_path, step=step)
                    control_latest = ckpt_dir / "hunyuan_control_latest.pt"
                    if control_latest.exists() or control_latest.is_symlink():
                        control_latest.unlink()
                    control_latest.symlink_to(control_path.name)
            if world > 1 and ckpt_every > 0 and step % ckpt_every == 0:
                dist.barrier()
            del loss, dit_loss, dit_parts, out, losses
            del s, c, action_cond, context_rgb, tgt
            del s_cond, c_cond, action_cond_model, context_rgb_cond
            if device.type == "cuda" and empty_cache_every > 0 and step % empty_cache_every == 0:
                torch.cuda.empty_cache()

        # Lightweight validation at epoch end only; step eval is intentionally
        # disabled by default because full Hunyuan DiT PT is memory-bound.
        wm_model.eval()
        agg: dict[str, float] = {}
        nb = 0
        max_val_batches = int(train_cfg.get("max_val_batches", 16))
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if max_val_batches > 0 and bi >= max_val_batches:
                    break
                s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, int(cfg["data"]["k"]))
                with torch.autocast(device_type="cuda", dtype=_dtype(precision), enabled=device.type == "cuda"):
                    out = _forward_joint_model(
                        wm_model,
                        s,
                        c,
                        action_cond=action_cond,
                        context_rgb=context_rgb,
                        prior_clean_tokens=prior_clean_tokens_from_targets(tgt),
                        pixel=False,
                        bridging=False,
                    )
                    losses = compute_losses(out, tgt, weights, None)
                for key, value in losses.items():
                    agg[key] = agg.get(key, 0.0) + float(value.detach().float())
                nb += 1
        if world > 1:
            keys = sorted(agg)
            vals = torch.tensor([agg[k] for k in keys] + [float(nb)], device=device)
            dist.all_reduce(vals)
            total_nb = max(1.0, float(vals[-1].item()))
            agg = {k: float(vals[i].item()) / total_nb for i, k in enumerate(keys)}
            nb = 1
        val_total = agg.get("L_total", float("inf")) / max(1, nb) if rank == 0 else float("inf")
        is_best = False
        if rank == 0:
            print(f"[rank0] epoch {epoch}: val_native_total={val_total:.4f} best={best_val:.4f}", flush=True)
            if tb is not None:
                for key, value in agg.items():
                    tb.add_scalar(f"val/{key}", value / max(1, nb), step)
            is_best = bool(val_total < best_val)
            if is_best:
                best_val = val_total
        if world > 1:
            flag = torch.tensor([1 if is_best else 0], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            is_best = bool(flag.item())
        if not args.no_epoch_checkpoint:
            save_checkpoint(
                path=ckpt_dir / f"epoch_{epoch:03d}.pt",
                wm_model=wm_model,
                adapter=hunyuan_control_adapter,
                transformer=transformer,
                opt=opt,
                sched=sched,
                cfg=cfg,
                step=step,
                epoch=epoch,
                metrics=agg,
                rank=rank,
                fsdp_enabled=bool(fsdp_report.get("enabled", False)),
            )
            if is_best:
                best_path = ckpt_dir / "best.pt"
                save_checkpoint(
                    path=best_path,
                    wm_model=wm_model,
                    adapter=hunyuan_control_adapter,
                    transformer=transformer,
                    opt=opt,
                    sched=sched,
                    cfg=cfg,
                    step=step,
                    epoch=epoch,
                    metrics=agg,
                    rank=rank,
                    fsdp_enabled=bool(fsdp_report.get("enabled", False)),
                )
                if rank == 0:
                    save_hunyuan_dit_control_checkpoint(ckpt_dir / "hunyuan_control_best.pt", hunyuan_control_adapter, metrics=agg, wm_ckpt=best_path, step=step)
        if world > 1:
            dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
