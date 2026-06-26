"""Stage0 WM3D native3D pretraining with trainable Wan2.2 TI2V RGB/video generator.

This is the Wan TI2V counterpart of the Hunyuan stage125/stage127
joint script. The small WM RGB decoder remains disabled; RGB supervision is
through Wan latent/source and velocity losses while depth/point/pose/action
native WM losses stay active.
"""
from __future__ import annotations

import argparse
import json
import math
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

from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.wan_lora import (
    WanLoRAConfig,
    apply_lora_to_linear_modules,
    collect_trainable_state_dict,
    load_partial_state_dict,
    save_wan_trainable_checkpoint,
    set_trainable_by_patterns,
)
from wm3d_v3.models.wan_ti2v_control_adapter import (
    WanTI2VControlAdapter,
    WanTI2VControlConfig,
    WanTI2VControlInjector,
    save_wan_ti2v_control_checkpoint,
)
from wm3d_v3.training.lr_schedule import build_lr_scheduler
from wm3d_v3.training.train import (
    WeightedDistributedSampler,
    _all_reduce_gradients,
    _distributed_finite_count,
    _forward_joint_model,
    _sampler_scope,
    apply_condition_dropout,
    action_policy_kwargs_from_targets,
    batch_to_device,
    build_datasets,
    build_model,
    load_action_stats_if_available,
    prior_clean_tokens_from_targets,
    target_video_from_batch,
)
from wm3d_v3.video_backends.wan_ti2v_control_video import (
    WanTI2VControlVideoBackend,
    WanTI2VControlVideoBackendConfig,
)


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


def _is_fsdp(module: torch.nn.Module | None) -> bool:
    return FSDP is not None and module is not None and isinstance(module, FSDP)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    current = module
    while hasattr(current, "module"):
        current = current.module
    return current


def adapter_target(module: torch.nn.Module) -> WanTI2VControlAdapter:
    return _unwrap_module(module)  # type: ignore[return-value]


def sync_module(module: torch.nn.Module | None, world: int) -> None:
    if module is None or _is_fsdp(module):
        return
    if world > 1:
        _all_reduce_gradients(module, world)


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


def _fsdp_ignored_params(module: torch.nn.Module, *, ignore_frozen: bool, ignore_float8: bool) -> tuple[list[torch.nn.Parameter], dict[str, Any]]:
    ignored: list[torch.nn.Parameter] = []
    stats: dict[str, Any] = {
        "ignore_frozen": ignore_frozen,
        "ignore_float8": ignore_float8,
        "ignored_params": 0,
        "ignored_numel": 0,
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
        should_ignore = (ignore_float8 and _is_float8_dtype(param.dtype)) or (ignore_frozen and not param.requires_grad)
        if should_ignore:
            if _is_float8_dtype(param.dtype) and param.requires_grad:
                raise RuntimeError("Refusing to ignore trainable float8 Wan parameter under FSDP")
            ignored.append(param)
            stats["ignored_params"] += 1
            stats["ignored_numel"] += int(param.numel())
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

    modules = {str(x).strip().lower() for x in train_cfg.get("fsdp_modules", ("wm", "wan_transformer"))}
    strategy = _parse_fsdp_strategy(str(train_cfg.get("fsdp_sharding_strategy", "full_shard")))
    use_orig_params = bool(train_cfg.get("fsdp_use_orig_params", True))
    kwargs = {
        "sharding_strategy": strategy,
        "use_orig_params": use_orig_params,
        "device_id": device if device.type == "cuda" else None,
        "limit_all_gathers": bool(train_cfg.get("fsdp_limit_all_gathers", True)),
        "forward_prefetch": bool(train_cfg.get("fsdp_forward_prefetch", False)),
        "backward_prefetch": _parse_backward_prefetch(train_cfg.get("fsdp_backward_prefetch", "backward_pre")),
    }
    report: dict[str, Any] = {"enabled": True, "modules": sorted(modules), "strategy": str(strategy), "use_orig_params": use_orig_params}
    if "wm" in modules or "wm_model" in modules:
        wm_model = FSDP(wm_model, **kwargs)
    if "wan" in modules or "wan_transformer" in modules or "transformer" in modules:
        wan_kwargs = dict(kwargs)
        ignored, ignored_stats = _fsdp_ignored_params(
            transformer,
            ignore_frozen=bool(train_cfg.get("fsdp_ignore_wan_frozen_params", True)),
            ignore_float8=bool(train_cfg.get("fsdp_ignore_wan_float8_params", True)),
        )
        if ignored:
            wan_kwargs["ignored_states"] = ignored
        report["wan_ignored"] = ignored_stats
        transformer = FSDP(transformer, **wan_kwargs)
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
    wan_control_adapter: WanTI2VControlAdapter,
    transformer: torch.nn.Module,
    train_cfg: dict,
    rank: int,
) -> torch.optim.Optimizer:
    groups = []
    wd = float(train_cfg.get("weight_decay", 0.02))
    wm_params = [p for p in wm_model.parameters() if p.requires_grad]
    control_params = [p for p in wan_control_adapter.parameters() if p.requires_grad]
    wan_lora_params = []
    wan_base_params = []
    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if ".lora_" in name or name.endswith("lora_A") or name.endswith("lora_B"):
            wan_lora_params.append(param)
        else:
            wan_base_params.append(param)
    if wm_params:
        groups.append({"name": "wm3d", "params": wm_params, "lr": float(train_cfg.get("wm_lr", train_cfg["lr"])), "weight_decay": wd})
    if control_params:
        groups.append({"name": "wan_control_adapter", "params": control_params, "lr": float(train_cfg.get("wan_control_lr", 2e-5)), "weight_decay": wd})
    if wan_lora_params:
        groups.append({"name": "wan_lora", "params": wan_lora_params, "lr": float(train_cfg.get("wan_lora_lr", 5e-6)), "weight_decay": float(train_cfg.get("wan_weight_decay", 0.0))})
    if wan_base_params:
        groups.append({"name": "wan_base_trainable", "params": wan_base_params, "lr": float(train_cfg.get("wan_base_lr", 2e-7)), "weight_decay": float(train_cfg.get("wan_base_weight_decay", train_cfg.get("wan_weight_decay", 0.0)))})
    if not groups:
        raise RuntimeError("no trainable parameters for joint Wan PT")
    if rank == 0:
        for group in groups:
            params = sum(p.numel() for p in group["params"])
            print(f"[rank0] opt_group={group['name']} params={params/1e6:.2f}M lr={group['lr']:.3g} wd={group['weight_decay']:.3g}", flush=True)
    return torch.optim.AdamW(groups, betas=tuple(train_cfg.get("betas", (0.9, 0.95))))


def _filter_state_by_prefix_and_shape(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    prefixes: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    kept: dict[str, torch.Tensor] = {}
    skipped_prefix = 0
    skipped_shape = 0
    for key, value in source.items():
        if not any(key.startswith(prefix) for prefix in prefixes):
            skipped_prefix += 1
            continue
        if key not in target or tuple(target[key].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        kept[key] = value
    return kept, {"kept": len(kept), "skipped_prefix": skipped_prefix, "skipped_shape": skipped_shape}


def load_action_policy_init(path: str | Path, *, wm_model: torch.nn.Module, rank: int, require: bool = True) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        state = payload.get("model", payload.get("state_dict", payload))
    else:
        state = payload
    if not isinstance(state, dict):
        raise RuntimeError(f"action policy checkpoint {path} did not contain a state dict")
    target = _unwrap_module(wm_model)
    filtered, counts = _filter_state_by_prefix_and_shape(state, target.state_dict(), ("action_policy.",))
    if require and not filtered:
        raise RuntimeError(f"no action_policy.* tensors loaded from {path}; refusing to start Wan-VAM training")
    report = target.load_state_dict(filtered, strict=False)
    out = {
        "path": str(path),
        **counts,
        "missing_action_policy": len([key for key in getattr(report, "missing_keys", []) if key.startswith("action_policy.")]),
        "unexpected_action_policy": len([key for key in getattr(report, "unexpected_keys", []) if key.startswith("action_policy.")]),
    }
    if rank == 0:
        print(f"[rank0] init_action_policy={json.dumps(out, default=str)}", flush=True)
    return out


def set_prefix_trainable(module: torch.nn.Module, prefixes: tuple[str, ...], trainable: bool) -> int:
    changed = 0
    target = _unwrap_module(module)
    for name, param in target.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            param.requires_grad_(bool(trainable))
            changed += int(param.numel())
    return changed


def set_action_policy_eval(module: torch.nn.Module) -> None:
    policy = getattr(_unwrap_module(module), "action_policy", None)
    if policy is not None:
        policy.eval()


def build_wan_backend_cfg(args: argparse.Namespace, train_cfg: dict) -> WanTI2VControlVideoBackendConfig:
    return WanTI2VControlVideoBackendConfig(
        repo=str(args.wan_repo or train_cfg.get("wan_repo", "/data/Minko/external/Wan2.2")),
        checkpoint_dir=str(args.wan_model_base or train_cfg.get("wan_model_base", "/0604-10T-test/models/Wan2.2-TI2V-5B")),
        task=str(train_cfg.get("wan_task", "ti2v-5B")),
        size=str(train_cfg.get("wan_size", "1280*704")),
        frame_num=int(train_cfg.get("wan_frame_num", 17)),
        sample_steps=int(train_cfg.get("wan_sample_steps", 20)),
        sample_shift=float(train_cfg.get("wan_sample_shift", 5.0)),
        sample_guide_scale=float(train_cfg.get("wan_sample_guide_scale", 5.0)),
        sample_solver=str(train_cfg.get("wan_sample_solver", "unipc")),
        offload_model=bool(train_cfg.get("wan_offload_model", False)),
        t5_cpu=bool(train_cfg.get("wan_t5_cpu", True)),
        t5_fsdp=bool(train_cfg.get("wan_t5_fsdp", False)),
        dit_fsdp=bool(train_cfg.get("wan_dit_fsdp", False)),
        use_sp=bool(train_cfg.get("wan_use_sp", False)),
        init_on_cpu=bool(train_cfg.get("wan_init_on_cpu", False)),
        convert_model_dtype=bool(train_cfg.get("wan_convert_model_dtype", False)),
        control_scale=float(train_cfg.get("wan_control_scale", 1.0)),
        allow_untrained_control=True,
    )


def build_wan_modules(args: argparse.Namespace, cfg: dict, device: torch.device, rank: int, world: int):
    train_cfg = cfg["train"]
    backend = WanTI2VControlVideoBackend(build_wan_backend_cfg(args, train_cfg), device=device)
    pipeline = backend.load()
    transformer = backend.resolve_transformer(pipeline)
    for obj in (getattr(pipeline, "vae", None), getattr(pipeline, "text_encoder", None)):
        module = getattr(obj, "model", obj)
        if isinstance(module, torch.nn.Module):
            module.requires_grad_(False)
            module.eval()
    transformer.requires_grad_(False)
    transformer.eval()

    lora_report = {"enabled": False, "params": 0}
    if bool(train_cfg.get("wan_dit_train_lora", True)):
        lora_cfg = WanLoRAConfig(
            rank=int(train_cfg.get("wan_lora_rank", 8)),
            alpha=float(train_cfg.get("wan_lora_alpha", 16.0)),
            dropout=float(train_cfg.get("wan_lora_dropout", 0.0)),
            include=tuple(train_cfg.get("wan_lora_include", ("blocks",))),
            exclude=tuple(train_cfg.get("wan_lora_exclude", ())),
            dtype=str(train_cfg.get("wan_lora_dtype", "bf16")),
            checkpoint=bool(train_cfg.get("wan_lora_checkpoint", False)),
            checkpoint_use_reentrant=bool(train_cfg.get("wan_lora_checkpoint_use_reentrant", False)),
        )
        lora_report = apply_lora_to_linear_modules(transformer, lora_cfg)
    patterns = train_cfg.get("wan_dit_trainable_patterns") or ()
    if patterns:
        pattern_report = set_trainable_by_patterns(transformer, patterns, train_cfg.get("wan_dit_trainable_exclude", ()))
    else:
        pattern_report = {"params": 0, "tensors": 0, "preview": []}
    trainable_dtype = _dtype(str(train_cfg.get("wan_trainable_dtype", train_cfg.get("precision", "bf16"))))
    trainable_dtype_report = _cast_trainable_params(transformer, trainable_dtype)
    setattr(transformer, "_wm3d_activation_checkpoint", bool(train_cfg.get("wan_activation_checkpoint", False)))
    setattr(transformer, "_wm3d_activation_checkpoint_use_reentrant", bool(train_cfg.get("wan_activation_checkpoint_use_reentrant", False)))

    wan_hidden = int(getattr(transformer, "dim", getattr(getattr(transformer, "config", None), "dim", 3072)))
    blocks = backend.iter_transformer_blocks(transformer)
    vae = backend.resolve_vae(pipeline)
    latent_channels = int(getattr(getattr(vae, "model", None), "z_dim", train_cfg.get("wan_latent_channels", 48)))
    patch_size = tuple(getattr(pipeline, "patch_size", train_cfg.get("wan_patch_size", (1, 2, 2))))
    vae_stride = tuple(getattr(pipeline, "vae_stride", train_cfg.get("wan_vae_stride", (4, 16, 16))))
    adapter = WanTI2VControlAdapter(
        WanTI2VControlConfig(
            hidden=int(train_cfg.get("wan_control_hidden", 192)),
            dit_hidden=wan_hidden,
            latent_channels=latent_channels,
            num_layers=max(1, len(blocks) or int(train_cfg.get("wan_num_layers", 30))),
            patch_size=patch_size,  # type: ignore[arg-type]
            vae_stride=vae_stride,  # type: ignore[arg-type]
            action_residual_scale=float(train_cfg.get("wan_action_residual_scale", 0.75)),
            action_direct_scale=float(train_cfg.get("wan_action_direct_scale", 0.25)),
            action_latent_scale=float(train_cfg.get("wan_action_latent_scale", 0.20)),
            use_action_token_block=bool(train_cfg.get("wan_use_action_token_block", True)),
            action_token_block_scale=float(train_cfg.get("wan_action_token_block_scale", 0.75)),
            action_token_hidden=int(train_cfg.get("wan_action_token_hidden", 256)),
            action_token_heads=int(train_cfg.get("wan_action_token_heads", 4)),
            use_parallel_action_video_blocks=bool(train_cfg.get("wan_use_parallel_action_video_blocks", False)),
            parallel_action_video_scale=float(train_cfg.get("wan_parallel_action_video_scale", 1.0)),
            parallel_action_video_hidden=int(train_cfg.get("wan_parallel_action_video_hidden", train_cfg.get("wan_action_token_hidden", 256))),
            parallel_action_video_heads=int(train_cfg.get("wan_parallel_action_video_heads", train_cfg.get("wan_action_token_heads", 4))),
            parallel_action_video_mlp_mult=float(train_cfg.get("wan_parallel_action_video_mlp_mult", train_cfg.get("wan_action_token_mlp_mult", 2.0))),
            parallel_action_video_gate_source=str(train_cfg.get("wan_parallel_action_video_gate_source", "none")),
            parallel_action_video_gate_min=float(train_cfg.get("wan_parallel_action_video_gate_min", 0.0)),
            parallel_action_video_gate_threshold=float(train_cfg.get("wan_parallel_action_video_gate_threshold", 0.05)),
            parallel_action_video_gate_dilate=int(train_cfg.get("wan_parallel_action_video_gate_dilate", 0)),
            parallel_action_video_gate_power=float(train_cfg.get("wan_parallel_action_video_gate_power", 1.0)),
            parallel_action_video_gate_detach=bool(train_cfg.get("wan_parallel_action_video_gate_detach", True)),
            use_action_context_tokens=bool(train_cfg.get("wan_use_action_context_tokens", False)),
            action_context_dim=int(train_cfg.get("wan_action_context_dim", 4096)),
            action_context_hidden=int(train_cfg.get("wan_action_context_hidden", 512)),
            action_context_pos_scale=float(train_cfg.get("wan_action_context_pos_scale", 0.05)),
            use_vam_action_expert=bool(train_cfg.get("wan_use_vam_action_expert", False)),
            vam_action_freq_dim=int(train_cfg.get("wan_vam_action_freq_dim", 256)),
            vam_action_video_delta_scale=float(train_cfg.get("wan_vam_action_video_delta_scale", 1.0)),
            vam_action_policy_cond_scale=float(train_cfg.get("wan_vam_action_policy_cond_scale", 0.0)),
            vam_video_use_clean_action=bool(train_cfg.get("wan_vam_video_use_clean_action", False)),
            vam_video_action_source=str(train_cfg.get("wan_vam_video_action_source", "clean")),
            vam_video_policy_blend=float(train_cfg.get("wan_vam_video_policy_blend", 0.0)),
            vam_update_noisy_action_stream=bool(train_cfg.get("wan_vam_update_noisy_action_stream", True)),
            layer_gain_start=float(train_cfg.get("wan_layer_gain_start", 0.85)),
            layer_gain_end=float(train_cfg.get("wan_layer_gain_end", 1.35)),
        )
    ).to(device)
    broadcast_module_state(adapter, world)
    if any(p.requires_grad for p in transformer.parameters()):
        transformer.train()
    if rank == 0:
        print(f"[rank0] wan_lora={json.dumps(lora_report, default=str)}", flush=True)
        print(f"[rank0] wan_partial_train={json.dumps(pattern_report, default=str)}", flush=True)
        print(f"[rank0] wan_trainable_dtype={json.dumps(trainable_dtype_report, default=str)}", flush=True)
        print(f"[rank0] WanTI2VControlAdapter params={_count_trainable(adapter)/1e6:.2f}M", flush=True)
        print(f"[rank0] Wan trainable params={_count_trainable(transformer)/1e6:.2f}M", flush=True)
    return pipeline, transformer, adapter, lora_report, pattern_report


def prompts_from_batch(batch: dict[str, Any], batch_size: int) -> list[str]:
    for key in ("task_text", "language_instruction", "instruction"):
        value = batch.get(key)
        if isinstance(value, (list, tuple)):
            prompts = [str(v).strip() or "robot manipulation scene" for v in value]
            if len(prompts) == batch_size:
                return prompts
        if isinstance(value, str) and value.strip():
            return [value.strip()] * batch_size
    return ["robot manipulation scene, close-up tabletop robot arm manipulating an object"] * batch_size


def context_video_from_batch(context_rgb: torch.Tensor, frames: int) -> torch.Tensor:
    return context_rgb[:, :, None].expand(-1, -1, int(frames), -1, -1).contiguous()


def first_frame_noise_source(target_latents: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    """Wan/TI2V-style source latents: keep the first latent frame fixed and denoise future frames from noise."""

    source = torch.randn_like(target_latents)
    keep = int(train_cfg.get("wan_condition_latent_frames", 1))
    keep = max(0, min(keep, int(target_latents.shape[2])))
    if keep > 0:
        source[:, :, :keep] = target_latents[:, :, :keep]
    return source


def preserve_condition_latent_frames(source_latents: torch.Tensor, condition_latents: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    """Hard-preserve TI2V condition latent frames in a predicted source tensor."""

    keep = int(train_cfg.get("wan_condition_latent_frames", 1))
    keep = max(0, min(keep, int(source_latents.shape[2]), int(condition_latents.shape[2])))
    if keep <= 0:
        return source_latents
    out = source_latents.clone()
    out[:, :, :keep] = condition_latents[:, :, :keep].to(device=out.device, dtype=out.dtype)
    return out


def _parse_hw(value: Any) -> tuple[int, int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        h, w = int(value[0]), int(value[1])
        return (h, w) if h > 0 and w > 0 else None
    if isinstance(value, str):
        text = value.lower().replace("x", "*").replace(",", "*")
        parts = [part.strip() for part in text.split("*") if part.strip()]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            h, w = int(parts[0]), int(parts[1])
            return (h, w) if h > 0 and w > 0 else None
    return None


def target_video_for_wan(context_rgb: torch.Tensor, rgb_tgt_p: torch.Tensor, train_cfg: dict) -> torch.Tensor:
    """Build the Wan training clip.

    Wan TI2V is pretrained on 4n+1 frame clips. The WM3D OXE stage0 cache uses
    k=8 future frames, so context+future is already 9 frames. If a later cache
    changes k, pad only the tail to the next valid 4n+1 length instead of
    silently training Wan on an off-distribution frame count.
    """

    video = target_video_from_batch(context_rgb, rgb_tgt_p)
    resize_hw = _parse_hw(train_cfg.get("wan_train_rgb_size", None))
    if resize_hw is not None and tuple(int(v) for v in video.shape[-2:]) != tuple(resize_hw):
        b, c, t, h, w = video.shape
        video = F.interpolate(
            video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float(),
            size=resize_hw,
            mode="bilinear",
            align_corners=False,
        ).view(b, t, c, resize_hw[0], resize_hw[1]).permute(0, 2, 1, 3, 4).contiguous()
    frames = int(video.shape[2])
    if (frames - 1) % 4 == 0:
        return video
    if not bool(train_cfg.get("wan_pad_to_4n1", True)):
        raise ValueError(f"Wan TI2V training clip has {frames} frames; expected 4n+1")
    target_frames = 1 + 4 * int(math.ceil(max(1, frames - 1) / 4.0))
    pad = video[:, :, -1:].expand(-1, -1, target_frames - frames, -1, -1)
    return torch.cat([video, pad], dim=2).contiguous()


def _wan_vae_device(vae: Any) -> torch.device:
    module = getattr(vae, "model", vae)
    if isinstance(module, torch.nn.Module):
        return next(module.parameters()).device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_wan_latents(vae: Any, video_bcthw: torch.Tensor) -> torch.Tensor:
    """Encode [B,C,T,H,W] RGB in [0,1] with Wan2.2 VAE into [B,C,T,H,W] latents."""

    device = _wan_vae_device(vae)
    x = video_bcthw.mul(2.0).sub(1.0).to(device=device, dtype=torch.float32)
    latents = vae.encode([x[i] for i in range(int(x.shape[0]))])
    if isinstance(latents, torch.Tensor):
        return latents
    return torch.stack([z.to(device=device, dtype=torch.float32) for z in latents], dim=0)


@torch.no_grad()
def encode_wan_prompts(
    pipeline: Any,
    prompts: list[str],
    device: torch.device,
    *,
    use_cache: bool = True,
    max_cache: int = 4096,
) -> list[torch.Tensor]:
    cache: dict[str, torch.Tensor] | None = None
    if use_cache:
        cache = getattr(pipeline, "_wm3d_prompt_cache", None)
        if cache is None:
            cache = {}
            setattr(pipeline, "_wm3d_prompt_cache", cache)
        hits = [cache.get(prompt) for prompt in prompts]
        if all(item is not None for item in hits):
            return [item.to(device=device, non_blocking=True) for item in hits if item is not None]

    text_encoder = pipeline.text_encoder
    missing_prompts = prompts
    if cache is not None:
        missing_prompts = [prompt for prompt in prompts if prompt not in cache]
    if not getattr(pipeline, "t5_cpu", False):
        text_encoder.model.to(device)
        encoded_missing = text_encoder(missing_prompts, device)
    else:
        encoded_missing = text_encoder(missing_prompts, torch.device("cpu"))
    if cache is None:
        return [x.to(device=device) for x in encoded_missing]
    for prompt, emb in zip(missing_prompts, encoded_missing):
        if len(cache) >= int(max_cache):
            cache.pop(next(iter(cache)))
        cache[prompt] = emb.detach().cpu()
    return [cache[prompt].to(device=device, non_blocking=True) for prompt in prompts]


def wan_seq_len_from_latents(pipeline: Any, latents: torch.Tensor) -> int:
    patch_size = tuple(getattr(pipeline, "patch_size", (1, 2, 2)))
    sp_size = int(getattr(pipeline, "sp_size", 1))
    _, _, t, h, w = latents.shape
    seq = math.ceil((int(h) * int(w)) / (int(patch_size[1]) * int(patch_size[2])) * int(t) / max(1, sp_size)) * max(1, sp_size)
    return int(seq)


def wan_timestep_tokens(sigma: torch.Tensor, seq_len: int, *, num_train_timesteps: int) -> torch.Tensor:
    return (sigma * float(num_train_timesteps)).reshape(-1, 1).expand(-1, int(seq_len)).contiguous()


def wan_timestep_tokens_for_latents(
    sigma: torch.Tensor,
    latents: torch.Tensor,
    pipeline: Any,
    seq_len: int,
    train_cfg: dict,
    *,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Build Wan timestep tokens with clean-condition latent frames at t=0.

    Wan TI2V sampling masks observed image/context latent tokens and gives
    those tokens timestep 0 while future/noisy slots use the current diffusion
    timestep.  Training must mirror that mask; otherwise the model sees clean
    first-frame latents while being told they are noisy.
    """

    base = wan_timestep_tokens(sigma, seq_len, num_train_timesteps=num_train_timesteps)
    path_type = str(train_cfg.get("wan_path_type", "wm_latent")).strip().lower()
    if path_type in {"noise"}:
        return base
    keep = int(train_cfg.get("wan_condition_latent_frames", 1))
    if keep <= 0:
        return base
    patch_size = tuple(getattr(pipeline, "patch_size", train_cfg.get("wan_patch_size", (1, 2, 2))))
    _, _, t, h, w = latents.shape
    pt, ph, pw = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
    grid_t = max(1, int(math.ceil(int(t) / max(1, pt))))
    grid_h = max(1, int(math.ceil(int(h) / max(1, ph))))
    grid_w = max(1, int(math.ceil(int(w) / max(1, pw))))
    cond_t = max(0, min(grid_t, int(math.ceil(float(keep) / max(1, pt)))))
    if cond_t <= 0:
        return base
    mask = torch.ones((int(latents.shape[0]), grid_t, grid_h, grid_w), device=base.device, dtype=base.dtype)
    mask[:, :cond_t] = 0.0
    flat = mask.flatten(1)
    if int(flat.shape[1]) < int(seq_len):
        flat = torch.cat([flat, flat.new_ones(flat.shape[0], int(seq_len) - int(flat.shape[1]))], dim=1)
    flat = flat[:, : int(seq_len)]
    return base * flat


def _forward_wan_velocity(transformer: torch.nn.Module, noisy: torch.Tensor, timestep: torch.Tensor, context: list[torch.Tensor], seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    latents = [noisy[i].to(device=device, dtype=dtype if device.type == "cuda" else noisy.dtype) for i in range(int(noisy.shape[0]))]
    out = transformer(latents, t=timestep.to(device=device), context=context, seq_len=int(seq_len))
    if isinstance(out, dict) and "video" in out:
        out = out["video"]
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        return torch.stack([x.to(device=device) for x in out], dim=0)
    raise RuntimeError(f"Wan transformer returned unsupported type {type(out)!r}")


def append_wan_action_context(
    text: list[torch.Tensor],
    adapter: WanTI2VControlAdapter,
    action_cond: torch.Tensor | None,
    train_cfg: dict,
    policy_action_cond: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    scale = float(train_cfg.get("wan_action_context_scale", 1.0))
    source = str(train_cfg.get("wan_action_context_source", "action")).strip().lower()
    context_action = action_cond
    if source in {"policy", "teacher"} and policy_action_cond is not None:
        context_action = policy_action_cond
    elif source in {"policy_blend", "teacher_blend"} and action_cond is not None and policy_action_cond is not None:
        action = action_cond
        policy = policy_action_cond.to(device=action.device, dtype=action.dtype)
        if policy.ndim == 2:
            policy = policy[:, None]
        if action.ndim == 2:
            action = action[:, None]
        if int(policy.shape[1]) != int(action.shape[1]):
            if int(policy.shape[1]) > int(action.shape[1]):
                policy = policy[:, : int(action.shape[1])]
            else:
                pad = policy[:, -1:].expand(-1, int(action.shape[1]) - int(policy.shape[1]), -1)
                policy = torch.cat([policy, pad], dim=1)
        blend = max(0.0, min(1.0, float(train_cfg.get("wan_action_context_policy_blend", train_cfg.get("wan_vam_video_policy_blend", 0.5)))))
        context_action = action * (1.0 - blend) + policy * blend
    tokens = adapter.action_context_tokens(context_action)
    if tokens is None or scale == 0.0:
        return text
    out: list[torch.Tensor] = []
    max_extra = max(1, int(train_cfg.get("wan_action_context_max_tokens", tokens.shape[1])))
    text_len = max(1, int(train_cfg.get("wan_text_len", 512)))
    for idx, emb in enumerate(text):
        extra_len = min(max_extra, max(1, text_len - 1))
        extra = tokens[idx, :extra_len].to(device=emb.device, dtype=emb.dtype) * scale
        keep = max(1, text_len - int(extra.shape[0]))
        out.append(torch.cat([emb[:keep], extra], dim=0))
    return out


def select_wan_wrong_action(action_cond: torch.Tensor, train_cfg: dict, step: int) -> tuple[torch.Tensor, str]:
    modes_raw = train_cfg.get("wan_action_cf_modes", ("zero", "reverse", "negreverse"))
    if isinstance(modes_raw, str):
        modes = [item.strip() for item in modes_raw.split(",") if item.strip()]
    else:
        modes = [str(item).strip() for item in modes_raw if str(item).strip()]
    if not modes:
        modes = ["zero"]
    mode = modes[int(step) % len(modes)]
    if mode == "zero":
        return torch.zeros_like(action_cond), mode
    if mode == "reverse":
        return torch.flip(action_cond, dims=(1,)), mode
    if mode == "negreverse":
        scale = float(train_cfg.get("wan_action_cf_negreverse_scale", 1.0))
        return -scale * torch.flip(action_cond, dims=(1,)), mode
    if mode == "negative":
        return -action_cond, mode
    raise ValueError(f"unsupported wan_action_cf mode {mode!r}")


def match_action_horizon(value: torch.Tensor, horizon: int) -> torch.Tensor:
    if int(value.shape[1]) == int(horizon):
        return value
    if int(value.shape[1]) > int(horizon):
        return value[:, : int(horizon)]
    pad = value[:, -1:].expand(-1, int(horizon) - int(value.shape[1]), *value.shape[2:])
    return torch.cat([value, pad], dim=1)


def policy_action_cond_from_wm_out(wm_out: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "policy_pose_norm" not in wm_out or "policy_gripper_logit" not in wm_out:
        return None
    if "pose_norm" not in wm_out or "gripper_logit" not in wm_out:
        return wm_out.get("policy_action_cond")
    policy_horizon = int(wm_out["policy_pose_norm"].shape[1])
    trunk_pose = match_action_horizon(wm_out["pose_norm"], policy_horizon)
    trunk_grip = match_action_horizon(wm_out["gripper_logit"], policy_horizon)
    pose = trunk_pose + wm_out["policy_pose_norm"]
    grip_logit = trunk_grip + wm_out["policy_gripper_logit"]
    return torch.cat([pose, torch.sigmoid(grip_logit)[..., None]], dim=-1)


def action_flow_source(
    target_action: torch.Tensor,
    train_cfg: dict,
    *,
    policy_action_cond: torch.Tensor | None,
) -> tuple[torch.Tensor, str]:
    mode = str(train_cfg.get("wan_vam_action_source", "noise")).strip().lower()
    target = target_action
    policy = None
    if policy_action_cond is not None:
        policy = match_action_horizon(policy_action_cond.to(device=target.device, dtype=target.dtype), int(target.shape[1]))
    if mode in {"policy", "teacher"} and policy is not None:
        return policy.detach(), "policy"
    if mode in {"policy_noise", "teacher_noise"} and policy is not None:
        std = float(train_cfg.get("wan_vam_action_policy_noise_std", 0.35))
        source = policy.detach() + torch.randn_like(target) * std
        source[..., 6:7] = torch.rand_like(source[..., 6:7])
        return source, "policy_noise"
    source = torch.randn_like(target)
    source[..., 6:7] = torch.rand_like(source[..., 6:7])
    return source, "noise"


def action_velocity_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    train_cfg: dict,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.to(device=pred_f.device).float()
    if int(pred_f.shape[1]) != int(target_f.shape[1]):
        pred_f = match_action_horizon(pred_f, int(target_f.shape[1]))
    valid = torch.isfinite(target_f).all(dim=-1, keepdim=True)
    if mask is not None:
        valid = valid & (mask.to(device=pred_f.device).bool().unsqueeze(-1) if mask.ndim == 2 else mask.to(device=pred_f.device).bool())
    valid_f = valid.to(dtype=pred_f.dtype).clamp(0.0, 1.0)
    valid_den = valid_f.sum().clamp_min(1.0)
    diff = pred_f - target_f
    pose_l1 = (diff[..., :6].abs() * valid_f).sum() / (valid_den * 6.0)
    pose_mse = (diff[..., :6].square() * valid_f).sum() / (valid_den * 6.0)
    grip_l1 = (diff[..., 6:7].abs() * valid_f).sum() / valid_den
    if int(pred_f.shape[1]) > 1:
        step_diff = (pred_f[:, 1:, :6] - pred_f[:, :-1, :6]) - (target_f[:, 1:, :6] - target_f[:, :-1, :6])
        step_valid = (valid_f[:, 1:] * valid_f[:, :-1]).clamp(0.0, 1.0)
        step_l1 = (step_diff.abs() * step_valid).sum() / (step_valid.sum().clamp_min(1.0) * 6.0)
    else:
        step_l1 = pred_f.new_zeros(())
    loss = (
        float(train_cfg.get("wan_vam_action_pose_weight", 1.0)) * pose_l1
        + float(train_cfg.get("wan_vam_action_mse_weight", 0.10)) * pose_mse
        + float(train_cfg.get("wan_vam_action_grip_weight", 0.25)) * grip_l1
        + float(train_cfg.get("wan_vam_action_delta_weight", 0.20)) * step_l1
    )
    return loss, {
        "vam_action_pose_l1": float(pose_l1.detach().cpu()),
        "vam_action_pose_mse": float(pose_mse.detach().cpu()),
        "vam_action_grip_l1": float(grip_l1.detach().cpu()),
        "vam_action_delta_l1": float(step_l1.detach().cpu()),
        "vam_action_valid": float(valid_f.mean().detach().cpu()),
    }


def latent_motion_mask_from_target(context_rgb: torch.Tensor, target_video: torch.Tensor, *, latent_shape: tuple[int, int, int, int, int], threshold: float, dilate: int) -> torch.Tensor:
    context = context_rgb
    if tuple(int(v) for v in context.shape[-2:]) != tuple(int(v) for v in target_video.shape[-2:]):
        context = F.interpolate(context.float(), size=target_video.shape[-2:], mode="bilinear", align_corners=False)
    context_video = context[:, :, None].expand(-1, -1, int(target_video.shape[2]), -1, -1)
    motion = (target_video.float() - context_video.float()).abs().mean(dim=1, keepdim=True)
    motion = F.interpolate(motion, size=(int(latent_shape[2]), int(latent_shape[3]), int(latent_shape[4])), mode="trilinear", align_corners=False)
    threshold = max(1e-6, float(threshold))
    mask = ((motion - threshold) / threshold).clamp(0.0, 1.0)
    dilate = max(0, int(dilate))
    if dilate > 0:
        k = 2 * dilate + 1
        mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=dilate)
    return mask.clamp(0.0, 1.0)


def _expand_latent_motion_mask(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    mask_f = mask.to(device=ref.device, dtype=ref.dtype).clamp(0.0, 1.0)
    while mask_f.ndim < ref.ndim:
        mask_f = mask_f.unsqueeze(1)
    return mask_f.expand_as(ref)


def _weighted_mean(value: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_velocity_losses(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None, *, dynamic_weight: float, static_weight: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.float()
    mask_f = _expand_latent_motion_mask(mask, pred_f)
    if mask_f is None:
        mse = F.mse_loss(pred_f, target_f)
        l1 = F.l1_loss(pred_f, target_f)
        return mse, l1, {"velocity_mse": float(mse.detach().cpu()), "velocity_l1": float(l1.detach().cpu())}
    weights = (float(static_weight) * (1.0 - mask_f) + float(dynamic_weight) * mask_f).clamp_min(1e-6)
    diff = pred_f - target_f
    mse = _weighted_mean(diff.square(), weights)
    l1 = _weighted_mean(diff.abs(), weights)
    dyn_den = mask_f.sum().clamp_min(1.0)
    sta = 1.0 - mask_f
    sta_den = sta.sum().clamp_min(1.0)
    return mse, l1, {
        "velocity_mse": float(mse.detach().cpu()),
        "velocity_l1": float(l1.detach().cpu()),
        "velocity_dynamic_mse": float((diff.square() * mask_f).sum().div(dyn_den).detach().cpu()),
        "velocity_static_mse": float((diff.square() * sta).sum().div(sta_den).detach().cpu()),
        "latent_motion_mask_mean": float(mask_f.mean().detach().cpu()),
    }


def velocity_region_mses(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    pred_f = pred.float()
    target_f = target.float()
    diff2 = (pred_f - target_f).square()
    mask_f = _expand_latent_motion_mask(mask, pred_f)
    if mask_f is None:
        mse = diff2.mean()
        return mse, mse
    dyn_den = mask_f.sum().clamp_min(1.0)
    static = 1.0 - mask_f
    sta_den = static.sum().clamp_min(1.0)
    dyn_mse = (diff2 * mask_f).sum().div(dyn_den)
    sta_mse = (diff2 * static).sum().div(sta_den)
    return dyn_mse, sta_mse


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
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.float()
    mask_f = _expand_latent_motion_mask(mask, pred_f)
    if mask_f is None:
        mask_f = torch.ones_like(pred_f)
    weights = (float(static_weight) * (1.0 - mask_f) + float(dynamic_weight) * mask_f).clamp_min(1e-6)
    diff = pred_f - target_f
    mse = _weighted_mean(diff.square(), weights)
    l1 = _weighted_mean(diff.abs(), weights)
    if pred_f.shape[2] > 1:
        temporal = _weighted_mean((pred_f[:, :, 1:] - pred_f[:, :, :-1] - target_f[:, :, 1:] + target_f[:, :, :-1]).abs(), mask_f[:, :, 1:].clamp_min(1e-6))
    else:
        temporal = pred_f.new_zeros(())
    from_first = _weighted_mean(((pred_f - pred_f[:, :, :1]) - (target_f - target_f[:, :, :1])).abs(), mask_f.clamp_min(1e-6))
    loss = float(mse_weight) * mse + float(l1_weight) * l1 + float(temporal_weight) * temporal + float(from_first_weight) * from_first
    return loss, {
        "wm_source_mse": float(mse.detach().cpu()),
        "wm_source_l1": float(l1.detach().cpu()),
        "wm_source_temporal_l1": float(temporal.detach().cpu()),
        "wm_source_from_first_l1": float(from_first.detach().cpu()),
        "wm_source_motion_mask": float(mask_f.mean().detach().cpu()),
    }


def velocity_motion_floor_loss(pred_velocity: torch.Tensor, target_velocity: torch.Tensor, mask: torch.Tensor | None, *, ratio: float, weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    if weight <= 0.0 or ratio <= 0.0:
        return pred_velocity.new_zeros(()), {}
    focus = _expand_latent_motion_mask(mask, pred_velocity.float())
    if focus is None:
        focus = torch.ones_like(pred_velocity.float())
    pred_mag = _weighted_mean(pred_velocity.float().abs(), focus.clamp_min(1e-6))
    target_mag = _weighted_mean(target_velocity.float().abs(), focus.clamp_min(1e-6)).detach().clamp_min(1e-8)
    floor = F.relu(target_mag * float(ratio) - pred_mag)
    return float(weight) * floor, {"velocity_motion_floor": float(floor.detach().cpu()), "velocity_motion_ratio": float((pred_mag / target_mag).detach().cpu())}


def _rgb_motion_mask_from_target(
    context_rgb: torch.Tensor,
    target_video: torch.Tensor,
    *,
    threshold: float,
    dilate: int,
    power: float,
) -> torch.Tensor:
    context_video = context_rgb.float()[:, :, None].expand(-1, -1, int(target_video.shape[2]), -1, -1)
    from_context = (target_video.float() - context_video).abs().mean(dim=1, keepdim=True)
    if target_video.shape[2] > 1:
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


def _decode_wan_latents_for_loss(vae: Any, latents: torch.Tensor) -> torch.Tensor:
    videos = []
    for sample in latents.float():
        decoded = vae.decode([sample])[0].float().clamp(-1, 1).div(2.0).add(0.5)
        videos.append(decoded)
    return torch.stack(videos, dim=0).contiguous()


def wan_decoded_rgb_losses(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    train_cfg: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_f = pred_video.float()
    target_f = target_video.to(device=pred_f.device).float()
    context_f = context_rgb.to(device=pred_f.device).float()
    size = int(train_cfg.get("wan_decoded_rgb_size", 0) or 0)
    if size > 0:
        shape = (int(pred_f.shape[2]), size, size)
        pred_f = F.interpolate(pred_f, size=shape, mode="trilinear", align_corners=False)
        target_f = F.interpolate(target_f, size=shape, mode="trilinear", align_corners=False)
        context_f = F.interpolate(context_f, size=(size, size), mode="bilinear", align_corners=False)

    mask = _rgb_motion_mask_from_target(
        context_f,
        target_f,
        threshold=float(train_cfg.get("wan_decoded_rgb_motion_threshold", 0.03)),
        dilate=int(train_cfg.get("wan_decoded_rgb_motion_dilate", 4)),
        power=float(train_cfg.get("wan_decoded_rgb_motion_power", 0.5)),
    )
    if bool(train_cfg.get("wan_decoded_rgb_skip_context", True)) and pred_f.shape[2] > 1:
        pred_f = pred_f[:, :, 1:]
        target_f = target_f[:, :, 1:]
        mask = mask[:, :, 1:]

    mask_c = mask.expand(-1, int(pred_f.shape[1]), -1, -1, -1)
    weights = (
        float(train_cfg.get("wan_decoded_rgb_static_weight", 1.0)) * (1.0 - mask_c)
        + float(train_cfg.get("wan_decoded_rgb_dynamic_weight", 2.0)) * mask_c
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
        target_step_motion * float(train_cfg.get("wan_decoded_rgb_motion_floor_ratio", 0.0))
        - pred_step_motion
    )
    from_floor = F.relu(
        target_from_motion * float(train_cfg.get("wan_decoded_rgb_from_first_floor_ratio", 0.0))
        - pred_from_motion
    )
    loss = (
        float(train_cfg.get("wan_decoded_rgb_l1_weight", 0.0)) * l1
        + float(train_cfg.get("wan_decoded_rgb_mse_weight", 0.0)) * mse
        + float(train_cfg.get("wan_decoded_rgb_temporal_weight", 0.0)) * temporal_l1
        + float(train_cfg.get("wan_decoded_rgb_from_first_weight", 0.0)) * from_first_l1
        + float(train_cfg.get("wan_decoded_rgb_motion_floor_weight", 0.0)) * step_floor
        + float(train_cfg.get("wan_decoded_rgb_from_first_floor_weight", 0.0)) * from_floor
        + float(train_cfg.get("wan_decoded_rgb_static_l1_weight", 0.0)) * static_l1
    )
    return loss, {
        "wan_decoded_rgb_l1": float(l1.detach().cpu()),
        "wan_decoded_rgb_mse": float(mse.detach().cpu()),
        "wan_decoded_rgb_dynamic_l1": float(dyn_l1.detach().cpu()),
        "wan_decoded_rgb_static_l1": float(static_l1.detach().cpu()),
        "wan_decoded_rgb_temporal_l1": float(temporal_l1.detach().cpu()),
        "wan_decoded_rgb_from_first_l1": float(from_first_l1.detach().cpu()),
        "wan_decoded_rgb_motion_floor": float(step_floor.detach().cpu()),
        "wan_decoded_rgb_from_first_floor": float(from_floor.detach().cpu()),
        "wan_decoded_rgb_motion_ratio": float((pred_step_motion / target_step_motion).detach().cpu()),
        "wan_decoded_rgb_from_first_ratio": float((pred_from_motion / target_from_motion).detach().cpu()),
        "wan_decoded_rgb_mask": float(mask_c.mean().detach().cpu()),
    }


def _prepare_wan_controls(
    target: WanTI2VControlAdapter,
    *,
    wm_out: dict[str, torch.Tensor],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
    noisy_latents: torch.Tensor | None = None,
    source_latents: torch.Tensor | None = None,
    sigma: torch.Tensor | None = None,
    action_noisy: torch.Tensor | None = None,
    action_sigma: torch.Tensor | None = None,
    policy_action_cond: torch.Tensor | None = None,
    latent_shape: tuple[int, int, int, int, int] | None = None,
) -> None:
    target.prepare_controls(
        pred_tokens=wm_out["pred_tokens"],
        depth=wm_out["depth"],
        motion_hint=wm_out.get("motion_hint"),
        contact_hint=wm_out.get("contact_hint"),
        context_rgb=context_rgb,
        action_cond=action_cond,
        task_emb=task_emb,
        point=wm_out.get("point"),
        pose_geom=wm_out.get("pose_geom"),
        noisy_latents=noisy_latents,
        source_latents=source_latents,
        sigma=sigma,
        action_noisy=action_noisy,
        action_sigma=action_sigma,
        policy_action_cond=policy_action_cond,
        latent_shape=latent_shape,
        scale=float(train_cfg.get("wan_control_scale", 1.0)),
    )


def compute_wan_ti2v_velocity_loss(
    *,
    adapter: WanTI2VControlAdapter,
    pipeline: Any,
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
    policy_action_cond: torch.Tensor | None = None,
    wan_wrong_action_cond: torch.Tensor | None = None,
    wan_wrong_action_mode: str | None = None,
    wan_wrong_wm_out: dict[str, torch.Tensor] | None = None,
    wan_wrong_policy_action_cond: torch.Tensor | None = None,
    wan_zero_wm_out: dict[str, torch.Tensor] | None = None,
    wan_zero_policy_action_cond: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any]:
    cleanup = lambda backward_ok=False: None
    if context_rgb is None or "rgb_tgt_p" not in tgt:
        zero = wm_out["pred_tokens"].new_zeros(())
        return zero, {"L_wan_ti2v_velocity": zero}, cleanup

    with torch.no_grad():
        target_video = target_video_for_wan(context_rgb, tgt["rgb_tgt_p"], train_cfg)
        target_latents = encode_wan_latents(pipeline.vae, target_video).to(device)
        frames = int(target_video.shape[2])
        prompts = prompts_from_batch(batch, int(target_latents.shape[0]))
        text_base = encode_wan_prompts(
            pipeline,
            prompts,
            device,
            use_cache=bool(train_cfg.get("wan_prompt_cache", True)),
            max_cache=int(train_cfg.get("wan_prompt_cache_max", 4096)),
        )
        sigma_min = float(train_cfg.get("wan_sigma_min", 0.02))
        sigma_max = float(train_cfg.get("wan_sigma_max", 1.0))
        sigma = torch.empty(target_latents.shape[0], device=device).uniform_(sigma_min, sigma_max).clamp(1e-4, 1.0)
        path_type = str(train_cfg.get("wan_path_type", "wm_latent")).strip().lower()
        if path_type == "noise":
            base_source_latents = torch.randn_like(target_latents)
        elif path_type in {"first_frame_noise", "ti2v_noise", "firstframe_noise"}:
            base_source_latents = first_frame_noise_source(target_latents, train_cfg)
        elif path_type in {"context", "wm_latent", "wm_source"}:
            context_video = target_video[:, :, :1].expand(-1, -1, int(frames), -1, -1).contiguous()
            context_latents = encode_wan_latents(pipeline.vae, context_video).to(device)
            base_source_latents = context_latents
        else:
            raise ValueError(f"Wan TI2V supports path_type noise/first_frame_noise/context/wm_latent, got {path_type!r}")
        motion_mask = None
        if str(train_cfg.get("wan_latent_motion_mask_source", "gt_rgb")).lower() == "gt_rgb":
            motion_mask = latent_motion_mask_from_target(
                context_rgb,
                target_video,
                latent_shape=tuple(target_latents.shape),
                threshold=float(train_cfg.get("wan_latent_motion_threshold", 0.015)),
                dilate=int(train_cfg.get("wan_latent_motion_dilate", 2)),
            )
        target_action = action_cond.to(device=device, dtype=target_latents.dtype)
        if target_action.ndim == 2:
            target_action = target_action[:, None]
        action_source, action_source_mode = action_flow_source(
            target_action,
            train_cfg,
            policy_action_cond=policy_action_cond,
        )
        action_sigma_min = float(train_cfg.get("wan_vam_action_sigma_min", sigma_min))
        action_sigma_max = float(train_cfg.get("wan_vam_action_sigma_max", sigma_max))
        action_sigma = torch.empty(target_action.shape[0], device=device).uniform_(action_sigma_min, action_sigma_max).clamp(1e-4, 1.0)
        action_noisy = action_sigma[:, None, None] * action_source + (1.0 - action_sigma)[:, None, None] * target_action
        target_action_velocity = action_source - target_action
        action_valid_mask = torch.isfinite(target_action).all(dim=-1)
        zero_hold_weight_cfg = float(train_cfg.get("wan_zero_action_hold_weight", 0.0) or 0.0)
        zero_hold_start = int(train_cfg.get("wan_zero_action_hold_start_step", train_cfg.get("wan_action_cf_start_step", 0)) or 0)
        zero_hold_every = max(1, int(train_cfg.get("wan_zero_action_hold_every", train_cfg.get("wan_action_cf_every", 1)) or 1))
        zero_hold_latents = None
        if zero_hold_weight_cfg > 0.0 and int(step) >= zero_hold_start and int(step) % zero_hold_every == 0:
            hold_video = target_video[:, :, :1].expand(-1, -1, int(target_video.shape[2]), -1, -1).contiguous()
            zero_hold_latents = encode_wan_latents(pipeline.vae, hold_video).to(device=device, dtype=target_latents.dtype)

    autocast_dtype = _dtype(precision)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        target = adapter_target(adapter)
        transformer_target = _unwrap_module(transformer)
        injector = WanTI2VControlInjector(transformer_target, target)
        injector.install()
        injector._latent_shape = tuple(target_latents.shape)
        post_backward_callbacks: list[Any] = []

        def _clear_controls() -> None:
            target.clear_control_state()
            injector.remove()

        def cleanup(backward_ok: bool = False) -> None:
            try:
                if backward_ok:
                    for callback in post_backward_callbacks:
                        callback()
            finally:
                _clear_controls()

        def _detach_tensor_tree(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.detach()
            if isinstance(obj, dict):
                return {k: _detach_tensor_tree(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_detach_tensor_tree(v) for v in obj)
            return obj

        def _zero_tensor_tree(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return torch.zeros_like(obj).detach()
            if isinstance(obj, dict):
                return {k: _zero_tensor_tree(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return type(obj)(_zero_tensor_tree(v) for v in obj)
            return obj

        try:
            source_latents = base_source_latents.to(device=target_latents.device, dtype=autocast_dtype if device.type == "cuda" else target_latents.dtype)
            source_loss = target_latents.new_zeros(())
            source_parts: dict[str, float] = {}
            source_cf_loss = target_latents.new_zeros(())
            source_cf_parts: dict[str, float] = {}
            if path_type in {"wm_latent", "wm_source"}:
                _prepare_wan_controls(
                    target,
                    wm_out=wm_out,
                    context_rgb=context_rgb,
                    action_cond=action_cond,
                    task_emb=task_emb,
                    train_cfg=train_cfg,
                    noisy_latents=source_latents,
                    source_latents=source_latents,
                    sigma=source_latents.new_ones(source_latents.shape[0]),
                    action_noisy=action_noisy,
                    action_sigma=action_sigma,
                    policy_action_cond=policy_action_cond,
                    latent_shape=tuple(target_latents.shape),
                )
                source_delta = target.source_latent_delta(source_latents) * float(train_cfg.get("wan_wm_source_scale", 1.0))
                predicted_source_latents = preserve_condition_latent_frames(source_latents + source_delta, target_latents, train_cfg)
                source_loss, source_parts = weighted_latent_source_losses(
                    predicted_source_latents,
                    target_latents,
                    motion_mask,
                    dynamic_weight=float(train_cfg.get("wan_wm_source_dynamic_weight", 48.0)),
                    static_weight=float(train_cfg.get("wan_wm_source_static_weight", 0.08)),
                    mse_weight=float(train_cfg.get("wan_wm_source_mse_weight", 0.15)),
                    l1_weight=float(train_cfg.get("wan_wm_source_l1_weight", 1.0)),
                    temporal_weight=float(train_cfg.get("wan_wm_source_temporal_weight", 18.0)),
                    from_first_weight=float(train_cfg.get("wan_wm_source_from_first_weight", 32.0)),
                )
                source_cf_weight = float(train_cfg.get("wan_source_action_cf_weight", 0.0))
                source_cf_every = max(1, int(train_cfg.get("wan_source_action_cf_every", train_cfg.get("wan_action_cf_every", 1))))
                source_cf_start = int(train_cfg.get("wan_source_action_cf_start_step", train_cfg.get("wan_action_cf_start_step", 0)) or 0)
                if source_cf_weight > 0.0 and int(step) >= source_cf_start and int(step) % source_cf_every == 0:
                    wrong_action = wan_wrong_action_cond
                    wrong_mode = "provided"
                    if wrong_action is None:
                        wrong_action, wrong_mode = select_wan_wrong_action(action_cond, train_cfg, step)
                    wrong_wm = wan_wrong_wm_out if wan_wrong_wm_out is not None else wm_out
                    wrong_policy_action_cond = (
                        wan_wrong_policy_action_cond if wan_wrong_policy_action_cond is not None else policy_action_cond
                    )
                    wrong_action_noisy = wrong_action.detach()
                    wrong_action_sigma = action_sigma.detach().new_zeros(action_sigma.shape)
                    target.clear_control_state()
                    _prepare_wan_controls(
                        target,
                        wm_out=wrong_wm,
                        context_rgb=context_rgb,
                        action_cond=wrong_action,
                        task_emb=task_emb,
                        train_cfg=train_cfg,
                        noisy_latents=source_latents,
                        source_latents=source_latents,
                        sigma=source_latents.new_ones(source_latents.shape[0]),
                        action_noisy=wrong_action_noisy,
                        action_sigma=wrong_action_sigma,
                        policy_action_cond=wrong_policy_action_cond,
                        latent_shape=tuple(target_latents.shape),
                    )
                    wrong_delta = target.source_latent_delta(source_latents) * float(train_cfg.get("wan_wm_source_scale", 1.0))
                    wrong_source_latents = preserve_condition_latent_frames(source_latents + wrong_delta, target_latents, train_cfg)
                    wrong_source_loss, wrong_source_parts = weighted_latent_source_losses(
                        wrong_source_latents,
                        target_latents,
                        motion_mask,
                        dynamic_weight=float(train_cfg.get("wan_source_action_cf_dynamic_weight", train_cfg.get("wan_wm_source_dynamic_weight", 36.0))),
                        static_weight=float(train_cfg.get("wan_source_action_cf_static_weight", train_cfg.get("wan_wm_source_static_weight", 0.08))),
                        mse_weight=float(train_cfg.get("wan_wm_source_mse_weight", 0.15)),
                        l1_weight=float(train_cfg.get("wan_wm_source_l1_weight", 1.0)),
                        temporal_weight=float(train_cfg.get("wan_wm_source_temporal_weight", 18.0)),
                        from_first_weight=float(train_cfg.get("wan_wm_source_from_first_weight", 32.0)),
                    )
                    source_margin = float(train_cfg.get("wan_source_action_cf_margin", train_cfg.get("wan_action_cf_margin", 0.02)))
                    source_cf_loss = torch.relu(source_loss.detach() + source_margin - wrong_source_loss)
                    source_cf_parts = {
                        "source_action_cf_loss": float(source_cf_loss.detach().cpu()),
                        "source_action_cf_true": float(source_loss.detach().cpu()),
                        "source_action_cf_wrong": float(wrong_source_loss.detach().cpu()),
                        "source_action_cf_gap": float((wrong_source_loss - source_loss.detach()).detach().cpu()),
                        "source_action_cf_mode_id": float({"zero": 1, "reverse": 2, "negreverse": 3, "negative": 4, "provided": 9}.get(wrong_mode, 0)),
                        "source_action_cf_wrong_l1": float(wrong_source_parts.get("wm_source_l1", 0.0)),
                    }
                    target.clear_control_state()
                    _prepare_wan_controls(
                        target,
                        wm_out=wm_out,
                        context_rgb=context_rgb,
                        action_cond=action_cond,
                        task_emb=task_emb,
                        train_cfg=train_cfg,
                        noisy_latents=source_latents,
                        source_latents=source_latents,
                        sigma=source_latents.new_ones(source_latents.shape[0]),
                        action_noisy=action_noisy,
                        action_sigma=action_sigma,
                        policy_action_cond=policy_action_cond,
                        latent_shape=tuple(target_latents.shape),
                    )
                if bool(train_cfg.get("wan_wm_source_detach_for_dit", False)):
                    source_latents = predicted_source_latents.detach()
                else:
                    source_latents = predicted_source_latents

            noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
            _prepare_wan_controls(
                target,
                wm_out=wm_out,
                context_rgb=context_rgb,
                action_cond=action_cond,
                task_emb=task_emb,
                train_cfg=train_cfg,
                noisy_latents=noisy,
                source_latents=source_latents,
                sigma=sigma,
                action_noisy=action_noisy,
                action_sigma=action_sigma,
                policy_action_cond=policy_action_cond,
                latent_shape=tuple(target_latents.shape),
            )
            velocity_source = source_latents.detach() if bool(train_cfg.get("wan_velocity_detach_source_target", True)) else source_latents
            target_velocity = float(train_cfg.get("wan_velocity_target_sign", 1.0)) * (velocity_source - target_latents)
            seq_len = wan_seq_len_from_latents(pipeline, noisy)
            timestep = wan_timestep_tokens_for_latents(
                sigma,
                noisy,
                pipeline,
                seq_len,
                train_cfg,
                num_train_timesteps=int(getattr(pipeline, "num_train_timesteps", 1000)),
            )
            text = append_wan_action_context(text_base, target, action_cond, train_cfg, policy_action_cond=policy_action_cond)
            pred_velocity = _forward_wan_velocity(transformer, noisy, timestep, text, seq_len, autocast_dtype, device)
            latent_scale = float(train_cfg.get("wan_latent_control_scale", 0.20))
            if latent_scale != 0.0:
                residual = target.latent_residual(noisy).to(device=pred_velocity.device, dtype=pred_velocity.dtype)
                pred_velocity = pred_velocity + residual * latent_scale
            vam_action_loss = target_latents.new_zeros(())
            vam_action_parts: dict[str, float] = {}
            if bool(train_cfg.get("wan_use_vam_action_expert", False)) and float(train_cfg.get("wan_vam_action_weight", 0.0)) > 0.0:
                pred_action_velocity = target.action_velocity_prediction()
                if pred_action_velocity is None:
                    raise RuntimeError("wan_use_vam_action_expert=true but adapter produced no action velocity prediction")
                vam_action_loss, vam_action_parts = action_velocity_losses(
                    pred_action_velocity,
                    target_action_velocity,
                    train_cfg,
                    mask=action_valid_mask,
                )
                vam_action_parts["vam_action_loss"] = float(vam_action_loss.detach().cpu())
                vam_action_parts["vam_action_source_mode_id"] = float({"noise": 1, "policy": 2, "policy_noise": 3}.get(action_source_mode, 0))
                vam_action_parts["vam_action_sigma"] = float(action_sigma.detach().float().mean().cpu())

            mse, l1, parts = weighted_velocity_losses(
                pred_velocity,
                target_velocity,
                motion_mask,
                dynamic_weight=float(train_cfg.get("wan_velocity_dynamic_weight", 36.0)),
                static_weight=float(train_cfg.get("wan_velocity_static_weight", 0.25)),
            )
            floor, floor_parts = velocity_motion_floor_loss(
                pred_velocity,
                target_velocity,
                motion_mask,
                ratio=float(train_cfg.get("wan_velocity_motion_floor_ratio", 0.92)),
                weight=float(train_cfg.get("wan_velocity_motion_floor_weight", 4.0)),
            )
            parts.update(floor_parts)
            velocity_loss = (
                float(train_cfg.get("wan_velocity_mse_weight", 1.0)) * mse
                + float(train_cfg.get("wan_velocity_l1_weight", 0.05)) * l1
                + floor
            )
            decoded_rgb_loss = pred_velocity.new_zeros(())
            decoded_rgb_parts: dict[str, float] = {}
            decoded_rgb_enabled = bool(train_cfg.get("wan_decoded_rgb_loss", False))
            decoded_rgb_every = max(1, int(train_cfg.get("wan_decoded_rgb_every", 1) or 1))
            decoded_rgb_start = int(train_cfg.get("wan_decoded_rgb_start_step", 0) or 0)
            if decoded_rgb_enabled and int(step) >= decoded_rgb_start and int(step) % decoded_rgb_every == 0:
                decoded_samples = int(train_cfg.get("wan_decoded_rgb_samples", 1) or 1)
                decoded_samples = max(1, min(decoded_samples, int(pred_velocity.shape[0])))
                pred_target_latents = noisy[:decoded_samples].float() - (
                    sigma[:decoded_samples, None, None, None, None].float() * pred_velocity[:decoded_samples].float()
                )
                pred_video = _decode_wan_latents_for_loss(pipeline.vae, pred_target_latents)
                decoded_rgb_loss, decoded_rgb_parts = wan_decoded_rgb_losses(
                    pred_video,
                    target_video[:decoded_samples],
                    context_rgb[:decoded_samples],
                    train_cfg=train_cfg,
                )
            decoded_rgb_parts["wan_decoded_rgb_loss"] = float(decoded_rgb_loss.detach().cpu())
            decoded_rgb_parts["wan_decoded_rgb_active"] = float(decoded_rgb_enabled and int(step) >= decoded_rgb_start and int(step) % decoded_rgb_every == 0)
            cf_loss = target_latents.new_zeros(())
            cf_parts: dict[str, float] = {}
            cf_weight = float(train_cfg.get("wan_action_cf_weight", 0.0))
            cf_every = max(1, int(train_cfg.get("wan_action_cf_every", 1)))
            cf_start = int(train_cfg.get("wan_action_cf_start_step", 0) or 0)
            if cf_weight > 0.0 and int(step) >= cf_start and int(step) % cf_every == 0:
                wrong_action = wan_wrong_action_cond
                wrong_mode = str(wan_wrong_action_mode or "provided")
                if wrong_action is None:
                    wrong_action, wrong_mode = select_wan_wrong_action(action_cond, train_cfg, step)
                wrong_wm = wan_wrong_wm_out if wan_wrong_wm_out is not None else wm_out
                wrong_policy_action_cond = (
                    wan_wrong_policy_action_cond if wan_wrong_policy_action_cond is not None else policy_action_cond
                )
                margin = float(train_cfg.get("wan_action_cf_margin", 0.02))
                separate_cf = bool(train_cfg.get("wan_action_cf_separate_backward", False))
                if separate_cf:
                    wrong_wm_cb = _detach_tensor_tree(wrong_wm)
                    wrong_action_cb = wrong_action.detach()
                    context_rgb_cb = context_rgb.detach()
                    task_emb_cb = task_emb.detach()
                    noisy_cb = noisy.detach()
                    timestep_cb = timestep.detach()
                    source_latents_cb = source_latents.detach()
                    sigma_cb = sigma.detach()
                    action_noisy_cb = wrong_action.detach()
                    action_sigma_cb = action_sigma.detach().new_zeros(action_sigma.shape)
                    policy_action_cond_cb = wrong_policy_action_cond.detach() if wrong_policy_action_cond is not None else None
                    target_velocity_cb = target_velocity.detach()
                    pred_velocity_cb = pred_velocity.detach()
                    motion_mask_cb = motion_mask.detach() if isinstance(motion_mask, torch.Tensor) else motion_mask
                    true_mse_cb = mse.detach()
                    text_base_cb = [item.detach() for item in text_base]
                    latent_shape_cb = tuple(target_latents.shape)
                    dynamic_weight_cb = float(train_cfg.get("wan_action_cf_dynamic_weight", train_cfg.get("wan_velocity_dynamic_weight", 36.0)))
                    static_weight_cb = float(train_cfg.get("wan_action_cf_static_weight", train_cfg.get("wan_velocity_static_weight", 0.25)))
                    cf_compare_region_cb = str(train_cfg.get("wan_action_cf_compare_region", "weighted")).strip().lower()
                    static_preserve_weight_cb = float(train_cfg.get("wan_action_cf_static_preserve_weight", 0.0))
                    dynamic_cap_weight_cb = float(train_cfg.get("wan_action_cf_dynamic_cap_weight", 0.0))
                    dynamic_cap_ratio_cb = float(train_cfg.get("wan_action_cf_dynamic_cap_ratio", 2.0))
                    dynamic_cap_margin_cb = float(train_cfg.get("wan_action_cf_dynamic_cap_margin", 0.0))
                    latent_scale_cb = float(latent_scale)
                    mode_id = float({"zero": 1, "reverse": 2, "negreverse": 3, "negative": 4, "provided": 9}.get(wrong_mode, 0))

                    def _run_action_cf_backward() -> None:
                        target.clear_control_state()
                        with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                            _prepare_wan_controls(
                                target,
                                wm_out=wrong_wm_cb,
                                context_rgb=context_rgb_cb,
                                action_cond=wrong_action_cb,
                                task_emb=task_emb_cb,
                                train_cfg=train_cfg,
                                noisy_latents=noisy_cb,
                                source_latents=source_latents_cb,
                                sigma=sigma_cb,
                                action_noisy=action_noisy_cb,
                                action_sigma=action_sigma_cb,
                                policy_action_cond=policy_action_cond_cb,
                                latent_shape=latent_shape_cb,
                            )
                            wrong_text_cb = append_wan_action_context(
                                text_base_cb,
                                target,
                                wrong_action_cb,
                                train_cfg,
                                policy_action_cond=policy_action_cond_cb,
                            )
                            wrong_velocity_cb = _forward_wan_velocity(transformer, noisy_cb, timestep_cb, wrong_text_cb, seq_len, autocast_dtype, device)
                            if latent_scale_cb != 0.0:
                                wrong_velocity_cb = wrong_velocity_cb + target.latent_residual(noisy_cb).to(device=wrong_velocity_cb.device, dtype=wrong_velocity_cb.dtype) * latent_scale_cb
                            wrong_mse_cb, wrong_l1_cb, wrong_parts_cb = weighted_velocity_losses(
                                wrong_velocity_cb,
                                target_velocity_cb,
                                motion_mask_cb,
                                dynamic_weight=dynamic_weight_cb,
                                static_weight=static_weight_cb,
                            )
                            true_dyn_mse_cb, true_static_mse_cb = velocity_region_mses(
                                pred_velocity_cb,
                                target_velocity_cb,
                                motion_mask_cb,
                            )
                            wrong_dyn_mse_cb, wrong_static_mse_cb = velocity_region_mses(
                                wrong_velocity_cb,
                                target_velocity_cb,
                                motion_mask_cb,
                            )
                            if cf_compare_region_cb == "dynamic":
                                true_compare_cb = true_dyn_mse_cb.detach()
                                wrong_compare_cb = wrong_dyn_mse_cb
                            else:
                                true_compare_cb = true_mse_cb
                                wrong_compare_cb = wrong_mse_cb
                            rank_loss_cb = torch.relu(true_compare_cb + margin - wrong_compare_cb)
                            static_preserve_cb = static_preserve_weight_cb * wrong_static_mse_cb
                            dynamic_cap_cb = dynamic_cap_weight_cb * torch.relu(
                                wrong_dyn_mse_cb - true_dyn_mse_cb.detach() * dynamic_cap_ratio_cb - dynamic_cap_margin_cb
                            ).square()
                            cf_loss_cb = rank_loss_cb + static_preserve_cb + dynamic_cap_cb
                            (cf_weight * cf_loss_cb).backward()
                        if int(os.environ.get("RANK", "0")) == 0:
                            print(
                                f"[rank0] post_backward_wan_action_cf step={step} "
                                f"loss={float(cf_loss_cb.detach().float()):.4f} "
                                f"rank={float(rank_loss_cb.detach().float()):.4f} "
                                f"static_guard={float(static_preserve_cb.detach().float()):.4f} "
                                f"dyn_cap={float(dynamic_cap_cb.detach().float()):.4f} "
                                f"gap={float((wrong_mse_cb - true_mse_cb).detach().float()):.5f} "
                                f"dyn_gap={float((wrong_dyn_mse_cb - true_dyn_mse_cb).detach().float()):.5f} "
                                f"wrong_mse={float(wrong_mse_cb.detach().float()):.5f} "
                                f"wrong_l1={float(wrong_l1_cb.detach().float()):.5f} "
                                f"mode_id={mode_id:.0f} "
                                f"wrong_dyn={float(wrong_dyn_mse_cb.detach().float()):.5f} "
                                f"wrong_static={float(wrong_static_mse_cb.detach().float()):.5f}",
                                flush=True,
                            )

                    post_backward_callbacks.append(_run_action_cf_backward)
                    cf_parts = {
                        "action_cf_loss": 0.0,
                        "action_cf_true_mse": float(mse.detach().cpu()),
                        "action_cf_wrong_mse": 0.0,
                        "action_cf_wrong_l1": 0.0,
                        "action_cf_gap": 0.0,
                        "action_cf_mode_id": mode_id,
                        "action_cf_wrong_dynamic_mse": 0.0,
                        "action_cf_wrong_static_mse": 0.0,
                    }
                else:
                    target.clear_control_state()
                    _prepare_wan_controls(
                        target,
                        wm_out=wrong_wm,
                        context_rgb=context_rgb,
                        action_cond=wrong_action,
                        task_emb=task_emb,
                        train_cfg=train_cfg,
                        noisy_latents=noisy,
                        source_latents=source_latents,
                        sigma=sigma,
                        action_noisy=wrong_action.detach(),
                        action_sigma=action_sigma.detach().new_zeros(action_sigma.shape),
                        policy_action_cond=wrong_policy_action_cond,
                        latent_shape=tuple(target_latents.shape),
                    )
                    wrong_text = append_wan_action_context(
                        text_base,
                        target,
                        wrong_action,
                        train_cfg,
                        policy_action_cond=wrong_policy_action_cond,
                    )
                    wrong_velocity = _forward_wan_velocity(transformer, noisy, timestep, wrong_text, seq_len, autocast_dtype, device)
                    if latent_scale != 0.0:
                        wrong_velocity = wrong_velocity + target.latent_residual(noisy).to(device=wrong_velocity.device, dtype=wrong_velocity.dtype) * latent_scale
                    wrong_mse, wrong_l1, wrong_parts = weighted_velocity_losses(
                        wrong_velocity,
                        target_velocity,
                        motion_mask,
                        dynamic_weight=float(train_cfg.get("wan_action_cf_dynamic_weight", train_cfg.get("wan_velocity_dynamic_weight", 36.0))),
                        static_weight=float(train_cfg.get("wan_action_cf_static_weight", train_cfg.get("wan_velocity_static_weight", 0.25))),
                    )
                    true_dyn_mse, true_static_mse = velocity_region_mses(pred_velocity.detach(), target_velocity.detach(), motion_mask)
                    wrong_dyn_mse, wrong_static_mse = velocity_region_mses(wrong_velocity, target_velocity, motion_mask)
                    if str(train_cfg.get("wan_action_cf_compare_region", "weighted")).strip().lower() == "dynamic":
                        true_compare = true_dyn_mse.detach()
                        wrong_compare = wrong_dyn_mse
                    else:
                        true_compare = mse.detach()
                        wrong_compare = wrong_mse
                    rank_loss = torch.relu(true_compare + margin - wrong_compare)
                    static_preserve = float(train_cfg.get("wan_action_cf_static_preserve_weight", 0.0)) * wrong_static_mse
                    dynamic_cap = float(train_cfg.get("wan_action_cf_dynamic_cap_weight", 0.0)) * torch.relu(
                        wrong_dyn_mse
                        - true_dyn_mse.detach() * float(train_cfg.get("wan_action_cf_dynamic_cap_ratio", 2.0))
                        - float(train_cfg.get("wan_action_cf_dynamic_cap_margin", 0.0))
                    ).square()
                    cf_loss = rank_loss + static_preserve + dynamic_cap
                    cf_parts = {
                        "action_cf_loss": float(cf_loss.detach().cpu()),
                        "action_cf_true_mse": float(mse.detach().cpu()),
                        "action_cf_wrong_mse": float(wrong_mse.detach().cpu()),
                        "action_cf_wrong_l1": float(wrong_l1.detach().cpu()),
                        "action_cf_gap": float((wrong_mse - mse.detach()).detach().cpu()),
                        "action_cf_mode_id": float({"zero": 1, "reverse": 2, "negreverse": 3, "negative": 4, "provided": 9}.get(wrong_mode, 0)),
                        "action_cf_wrong_dynamic_mse": float(wrong_dyn_mse.detach().cpu()),
                        "action_cf_wrong_static_mse": float(wrong_static_mse.detach().cpu()),
                    }
                    target.clear_control_state()
                    _prepare_wan_controls(
                        target,
                        wm_out=wm_out,
                        context_rgb=context_rgb,
                        action_cond=action_cond,
                        task_emb=task_emb,
                        train_cfg=train_cfg,
                        noisy_latents=noisy,
                        source_latents=source_latents,
                        sigma=sigma,
                        action_noisy=action_noisy,
                        action_sigma=action_sigma,
                        policy_action_cond=policy_action_cond,
                        latent_shape=tuple(target_latents.shape),
                    )
            zero_hold_parts: dict[str, float] = {}
            zero_hold_weight = float(train_cfg.get("wan_zero_action_hold_weight", 0.0) or 0.0)
            if zero_hold_weight > 0.0 and zero_hold_latents is not None:
                zero_action_cb = torch.zeros_like(action_cond).detach()
                zero_action_noisy_cb = zero_action_cb.detach()
                zero_action_sigma_cb = action_sigma.detach().new_zeros(action_sigma.shape)
                zero_wm_mode = "recomputed" if wan_zero_wm_out is not None else "zero" if bool(train_cfg.get("wan_zero_action_hold_zero_wm", True)) else "detach"
                if wan_zero_policy_action_cond is not None:
                    zero_policy_action_cb = wan_zero_policy_action_cond.detach()
                elif policy_action_cond is not None and wan_zero_wm_out is None and bool(train_cfg.get("wan_zero_action_hold_zero_wm", True)):
                    zero_policy_action_cb = torch.zeros_like(policy_action_cond).detach()
                elif policy_action_cond is not None:
                    zero_policy_action_cb = policy_action_cond.detach()
                else:
                    zero_policy_action_cb = None
                if wan_zero_wm_out is not None:
                    zero_wm_cb = _detach_tensor_tree(wan_zero_wm_out)
                elif bool(train_cfg.get("wan_zero_action_hold_zero_wm", True)):
                    zero_wm_cb = _zero_tensor_tree(wm_out)
                else:
                    zero_wm_cb = _detach_tensor_tree(wm_out)
                context_rgb_zero_cb = context_rgb.detach()
                task_emb_zero_cb = task_emb.detach()
                timestep_zero_cb = timestep.detach()
                source_latents_zero_cb = source_latents.detach()
                sigma_zero_cb = sigma.detach()
                hold_latents_zero_cb = zero_hold_latents.detach()
                text_base_zero_cb = [item.detach() for item in text_base]
                latent_shape_zero_cb = tuple(target_latents.shape)
                latent_scale_zero_cb = float(latent_scale)
                zero_mse_weight_cb = float(train_cfg.get("wan_zero_action_hold_mse_weight", 1.0))
                zero_l1_weight_cb = float(train_cfg.get("wan_zero_action_hold_l1_weight", 0.05))

                def _run_zero_action_hold_backward() -> None:
                    target.clear_control_state()
                    with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                        zero_target_latents_cb = hold_latents_zero_cb.to(device=source_latents_zero_cb.device, dtype=source_latents_zero_cb.dtype)
                        zero_sigma_view_cb = sigma_zero_cb[:, None, None, None, None].to(device=source_latents_zero_cb.device, dtype=source_latents_zero_cb.dtype)
                        noisy_zero_cb = zero_sigma_view_cb * source_latents_zero_cb + (1.0 - zero_sigma_view_cb) * zero_target_latents_cb
                        _prepare_wan_controls(
                            target,
                            wm_out=zero_wm_cb,
                            context_rgb=context_rgb_zero_cb,
                            action_cond=zero_action_cb,
                            task_emb=task_emb_zero_cb,
                            train_cfg=train_cfg,
                            noisy_latents=noisy_zero_cb,
                            source_latents=source_latents_zero_cb,
                            sigma=sigma_zero_cb,
                            action_noisy=zero_action_noisy_cb,
                            action_sigma=zero_action_sigma_cb,
                            policy_action_cond=zero_policy_action_cb,
                            latent_shape=latent_shape_zero_cb,
                        )
                        zero_text_cb = append_wan_action_context(
                            text_base_zero_cb,
                            target,
                            zero_action_cb,
                            train_cfg,
                            policy_action_cond=zero_policy_action_cb,
                        )
                        zero_velocity_cb = _forward_wan_velocity(transformer, noisy_zero_cb, timestep_zero_cb, zero_text_cb, seq_len, autocast_dtype, device)
                        if latent_scale_zero_cb != 0.0:
                            zero_velocity_cb = zero_velocity_cb + target.latent_residual(noisy_zero_cb).to(device=zero_velocity_cb.device, dtype=zero_velocity_cb.dtype) * latent_scale_zero_cb
                        zero_target_velocity_cb = source_latents_zero_cb.float() - zero_target_latents_cb.float()
                        zero_mse_cb, zero_l1_cb, _zero_parts_cb = weighted_velocity_losses(
                            zero_velocity_cb,
                            zero_target_velocity_cb,
                            None,
                            dynamic_weight=1.0,
                            static_weight=1.0,
                        )
                        zero_loss_cb = zero_mse_weight_cb * zero_mse_cb + zero_l1_weight_cb * zero_l1_cb
                        (zero_hold_weight * zero_loss_cb).backward()
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(
                            f"[rank0] post_backward_wan_zero_hold step={step} "
                            f"loss={float(zero_loss_cb.detach().float()):.4f} "
                            f"mse={float(zero_mse_cb.detach().float()):.5f} "
                            f"l1={float(zero_l1_cb.detach().float()):.5f} "
                            f"wm_mode={zero_wm_mode}",
                            flush=True,
                        )

                post_backward_callbacks.append(_run_zero_action_hold_backward)
                zero_hold_parts = {
                    "zero_hold_active": 1.0,
                    "zero_hold_weight": zero_hold_weight,
                }

            total = (
                velocity_loss
                + source_loss
                + float(train_cfg.get("wan_vam_action_weight", 0.0)) * vam_action_loss
                + float(train_cfg.get("wan_source_action_cf_weight", 0.0)) * source_cf_loss
                + cf_weight * cf_loss
                + decoded_rgb_loss
            )
        except Exception:
            _clear_controls()
            raise

    return total, {
        "L_wan_ti2v_velocity": velocity_loss.detach(),
        "L_wan_wm_source": source_loss.detach(),
        "wan_velocity_mse": target_latents.new_tensor(parts.get("velocity_mse", 0.0)),
        "wan_velocity_l1": target_latents.new_tensor(parts.get("velocity_l1", 0.0)),
        "wan_velocity_dynamic_mse": target_latents.new_tensor(parts.get("velocity_dynamic_mse", 0.0)),
        "wan_velocity_static_mse": target_latents.new_tensor(parts.get("velocity_static_mse", 0.0)),
        "wan_velocity_motion_floor": target_latents.new_tensor(parts.get("velocity_motion_floor", 0.0)),
        "wan_velocity_motion_ratio": target_latents.new_tensor(parts.get("velocity_motion_ratio", 0.0)),
        "wan_motion_mask": target_latents.new_tensor(parts.get("latent_motion_mask_mean", 0.0)),
        "wan_action_cf": target_latents.new_tensor(cf_parts.get("action_cf_loss", 0.0)),
        "wan_action_cf_true_mse": target_latents.new_tensor(cf_parts.get("action_cf_true_mse", 0.0)),
        "wan_action_cf_wrong_mse": target_latents.new_tensor(cf_parts.get("action_cf_wrong_mse", 0.0)),
        "wan_action_cf_wrong_l1": target_latents.new_tensor(cf_parts.get("action_cf_wrong_l1", 0.0)),
        "wan_action_cf_gap": target_latents.new_tensor(cf_parts.get("action_cf_gap", 0.0)),
        "wan_action_cf_mode_id": target_latents.new_tensor(cf_parts.get("action_cf_mode_id", 0.0)),
        "wan_action_cf_wrong_dynamic_mse": target_latents.new_tensor(cf_parts.get("action_cf_wrong_dynamic_mse", 0.0)),
        "wan_action_cf_wrong_static_mse": target_latents.new_tensor(cf_parts.get("action_cf_wrong_static_mse", 0.0)),
        "wan_wm_source_mse": target_latents.new_tensor(source_parts.get("wm_source_mse", 0.0)),
        "wan_wm_source_l1": target_latents.new_tensor(source_parts.get("wm_source_l1", 0.0)),
        "wan_wm_source_temporal_l1": target_latents.new_tensor(source_parts.get("wm_source_temporal_l1", 0.0)),
        "wan_wm_source_from_first_l1": target_latents.new_tensor(source_parts.get("wm_source_from_first_l1", 0.0)),
        "wan_source_action_cf": target_latents.new_tensor(source_cf_parts.get("source_action_cf_loss", 0.0)),
        "wan_source_action_cf_gap": target_latents.new_tensor(source_cf_parts.get("source_action_cf_gap", 0.0)),
        "wan_zero_hold_active": target_latents.new_tensor(zero_hold_parts.get("zero_hold_active", 0.0)),
        "wan_zero_hold_weight": target_latents.new_tensor(zero_hold_parts.get("zero_hold_weight", 0.0)),
        "wan_decoded_rgb_loss": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_loss", 0.0)),
        "wan_decoded_rgb_active": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_active", 0.0)),
        "wan_decoded_rgb_l1": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_l1", 0.0)),
        "wan_decoded_rgb_static_l1": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_static_l1", 0.0)),
        "wan_decoded_rgb_dynamic_l1": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_dynamic_l1", 0.0)),
        "wan_decoded_rgb_temporal_l1": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_temporal_l1", 0.0)),
        "wan_decoded_rgb_from_first_l1": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_from_first_l1", 0.0)),
        "wan_decoded_rgb_motion_ratio": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_motion_ratio", 0.0)),
        "wan_decoded_rgb_from_first_ratio": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_from_first_ratio", 0.0)),
        "wan_decoded_rgb_mask": target_latents.new_tensor(decoded_rgb_parts.get("wan_decoded_rgb_mask", 0.0)),
        "wan_vam_action_loss": target_latents.new_tensor(vam_action_parts.get("vam_action_loss", 0.0)),
        "wan_vam_action_pose_l1": target_latents.new_tensor(vam_action_parts.get("vam_action_pose_l1", 0.0)),
        "wan_vam_action_pose_mse": target_latents.new_tensor(vam_action_parts.get("vam_action_pose_mse", 0.0)),
        "wan_vam_action_grip_l1": target_latents.new_tensor(vam_action_parts.get("vam_action_grip_l1", 0.0)),
        "wan_vam_action_delta_l1": target_latents.new_tensor(vam_action_parts.get("vam_action_delta_l1", 0.0)),
        "wan_vam_action_valid": target_latents.new_tensor(vam_action_parts.get("vam_action_valid", 0.0)),
        "wan_vam_action_source_mode": target_latents.new_tensor(vam_action_parts.get("vam_action_source_mode_id", 0.0)),
        "wan_vam_action_sigma": target_latents.new_tensor(vam_action_parts.get("vam_action_sigma", 0.0)),
        "wan_sigma": sigma.detach().float().mean(),
    }, cleanup


def load_stage0_wan_init(path: str | Path, *, wm_model: torch.nn.Module, adapter: WanTI2VControlAdapter, transformer: torch.nn.Module, rank: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a dict payload")
    wm_report = wm_model.load_state_dict(payload.get("model", {}), strict=False)
    adapter_report = adapter.load_state_dict(payload.get("wan_control_adapter", {}), strict=False)
    trainable_report = load_partial_state_dict(transformer, payload.get("wan_trainable", {}))
    report = {
        "path": str(path),
        "step": payload.get("step"),
        "wm_missing": len(getattr(wm_report, "missing_keys", [])),
        "wm_unexpected": len(getattr(wm_report, "unexpected_keys", [])),
        "adapter_missing": len(getattr(adapter_report, "missing_keys", [])),
        "adapter_unexpected": len(getattr(adapter_report, "unexpected_keys", [])),
        "wan_trainable": trainable_report,
    }
    if rank == 0:
        print(f"[rank0] init_from_stage0_wan={json.dumps(report, default=str)}", flush=True)
    return report


def _state_dict_for_checkpoint(module: torch.nn.Module, *, rank: int) -> dict[str, Any]:
    if _is_fsdp(module):
        assert FSDP is not None and FullStateDictConfig is not None and StateDictType is not None
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
            state = module.state_dict()
        return state if rank == 0 else {}
    return module.state_dict() if rank == 0 else {}


def _trainable_wan_state_for_checkpoint(transformer: torch.nn.Module, *, rank: int) -> dict[str, torch.Tensor]:
    if _is_fsdp(transformer):
        assert FSDP is not None
        with FSDP.summon_full_params(transformer, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True):
            if rank == 0:
                return collect_trainable_state_dict(_unwrap_module(transformer))
        return {}
    return collect_trainable_state_dict(transformer) if rank == 0 else {}


def save_checkpoint(
    *,
    path: Path,
    wm_model: torch.nn.Module,
    adapter: WanTI2VControlAdapter,
    transformer: torch.nn.Module,
    opt: torch.optim.Optimizer,
    sched,
    cfg: dict,
    step: int,
    epoch: int,
    metrics: dict[str, float] | None,
    rank: int,
    fsdp_enabled: bool,
) -> None:
    model_state = _state_dict_for_checkpoint(wm_model, rank=rank)
    wan_state = _trainable_wan_state_for_checkpoint(transformer, rank=rank)
    if rank != 0:
        return
    payload = {
        "kind": "wm3d_stage0_wan_ti2v_jointpt_v1",
        "model": model_state,
        "wan_control_adapter": adapter_target(adapter).state_dict(),
        "wan_control_adapter_cfg": adapter_target(adapter).cfg.__dict__,
        "wan_trainable": wan_state,
        "opt": {} if fsdp_enabled else opt.state_dict(),
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


def save_wan_trainable_checkpoint_maybe_fsdp(path: Path, transformer: torch.nn.Module, *, train_cfg: dict, step: int, rank: int) -> None:
    kwargs = {
        "lora_config": (
            WanLoRAConfig(
                rank=int(train_cfg.get("wan_lora_rank", 8)),
                alpha=float(train_cfg.get("wan_lora_alpha", 16.0)),
                dropout=float(train_cfg.get("wan_lora_dropout", 0.0)),
                include=tuple(train_cfg.get("wan_lora_include", ("blocks",))),
                exclude=tuple(train_cfg.get("wan_lora_exclude", ())),
                dtype=str(train_cfg.get("wan_lora_dtype", "bf16")),
                checkpoint=bool(train_cfg.get("wan_lora_checkpoint", False)),
                checkpoint_use_reentrant=bool(train_cfg.get("wan_lora_checkpoint_use_reentrant", False)),
            )
            if bool(train_cfg.get("wan_dit_train_lora", True))
            else None
        ),
        "partial_unfreeze": train_cfg.get("wan_dit_trainable_patterns", ()),
        "step": step,
    }
    if _is_fsdp(transformer):
        assert FSDP is not None
        with FSDP.summon_full_params(transformer, recurse=True, writeback=False, rank0_only=True, offload_to_cpu=True):
            if rank == 0:
                save_wan_trainable_checkpoint(path, _unwrap_module(transformer), **kwargs)
    elif rank == 0:
        save_wan_trainable_checkpoint(path, transformer, **kwargs)


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
    ap = argparse.ArgumentParser(description="From-scratch WM3D stage0 PT with Wan2.2 TI2V RGB/video loss.")
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--wan_repo", type=Path, default=None)
    ap.add_argument("--wan_model_base", type=Path, default=None)
    ap.add_argument("--max_train_windows", type=int, default=0)
    ap.add_argument("--max_val_windows", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--ckpt_every_steps", type=int, default=None)
    ap.add_argument("--no_epoch_checkpoint", action="store_true")
    ap.add_argument("--disable_wan_lora", action="store_true")
    args = ap.parse_args()

    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(args.cfg.read_text())
    cfg.setdefault("data", {})["load_task_text"] = True
    cfg["model"]["enable_pixel"] = False
    cfg["model"]["enable_context_pixel"] = False
    train_cfg = cfg["train"]
    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.disable_wan_lora:
        train_cfg["wan_dit_train_lora"] = False
    out_dir = args.out_dir or Path(cfg["out"]["root"])
    ckpt_dir = out_dir / cfg["out"].get("ckpt_dir", "ckpt")
    precision = str(train_cfg.get("precision", "bf16"))
    init_seed = int(cfg["data"].get("seed", 909))
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    if rank == 0:
        print(
            "[rank0] Wan control injection "
            f"scale={float(train_cfg.get('wan_control_scale', 1.0)):.3f} "
            f"latent={float(train_cfg.get('wan_latent_control_scale', 0.20)):.3f} "
            f"action_block={bool(train_cfg.get('wan_use_action_token_block', True))} "
            f"parallel_action_video={bool(train_cfg.get('wan_use_parallel_action_video_blocks', False))} "
            f"path={train_cfg.get('wan_path_type', 'wm_latent')} "
            f"small_rgb_decoder=disabled",
            flush=True,
        )

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
    action_policy_ckpt = train_cfg.get("init_action_policy_ckpt")
    if bool(cfg.get("model", {}).get("enable_action_policy", False)):
        if action_policy_ckpt:
            if rank == 0:
                load_action_policy_init(
                    action_policy_ckpt,
                    wm_model=wm_model,
                    rank=rank,
                    require=bool(train_cfg.get("require_action_policy_init", True)),
                )
            if world > 1:
                dist.barrier()
        elif bool(train_cfg.get("require_action_policy_init", False)):
            raise RuntimeError("model.enable_action_policy=true but train.init_action_policy_ckpt is not set")
        if bool(train_cfg.get("freeze_action_policy", True)):
            frozen_params = set_prefix_trainable(wm_model, ("action_policy.",), False)
            if rank == 0:
                print(f"[rank0] freeze_action_policy params={frozen_params/1e6:.2f}M", flush=True)
    pipeline, transformer, wan_control_adapter, lora_report, pattern_report = build_wan_modules(args, cfg, device, rank, world)
    init_ckpt = train_cfg.get("init_from_stage0_wan_ckpt")
    if init_ckpt:
        if rank == 0:
            load_stage0_wan_init(init_ckpt, wm_model=wm_model, adapter=wan_control_adapter, transformer=transformer, rank=rank)
        if world > 1:
            dist.barrier()
    broadcast_module_state(wm_model, world)
    broadcast_trainable_parameters(transformer, world)
    broadcast_module_state(wan_control_adapter, world)
    if world > 1:
        dist.barrier()
    wm_model, transformer, fsdp_report = maybe_wrap_fsdp(wm_model, transformer, train_cfg, device=device, rank=rank, world=world)

    runtime_seed = init_seed + rank
    torch.manual_seed(runtime_seed)
    np.random.seed(runtime_seed)
    wm_model.train()
    if bool(train_cfg.get("freeze_action_policy", True)):
        set_action_policy_eval(wm_model)
    wan_control_adapter.train()

    weights = LossWeights(**cfg["loss"])
    opt = build_optimizer(wm_model, wan_control_adapter, transformer, train_cfg, rank)
    max_steps = int(train_cfg.get("max_steps", 0) or 0)
    total_steps = max(1, len(train_loader) * int(train_cfg.get("epochs", 1)))
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    sched = build_lr_scheduler(opt, cfg, total_steps)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    ckpt_every = int(args.ckpt_every_steps or train_cfg.get("ckpt_every_steps", 2500) or 0)
    empty_cache_every = int(train_cfg.get("empty_cache_every_steps", 0) or 0)

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_dir / cfg["out"].get("tb_dir", "tb"))
        metadata = {
            "kind": "wm3d_stage0_wan_ti2v_jointpt_v1",
            "cfg": str(args.cfg),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "world_size": world,
            "total_steps": total_steps,
            "rgb_decoder": "disabled",
            "wan_lora": lora_report,
            "wan_partial_train": pattern_report,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        print(f"[rank0] wan_ti2v_joint train_windows={len(train_ds)} val_windows={len(val_ds)} world={world} total_steps={total_steps}", flush=True)
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
        if bool(train_cfg.get("freeze_action_policy", True)):
            set_action_policy_eval(wm_model)
        wan_control_adapter.train()
        transformer.train()
        for batch in train_loader:
            if max_steps > 0 and step >= max_steps:
                break
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, int(cfg["data"]["k"]))
            with torch.enable_grad(), torch.autocast(device_type="cuda", dtype=_dtype(precision), enabled=device.type == "cuda"):
                s_cond, c_cond, action_cond_model, context_rgb_cond, dropout_losses = apply_condition_dropout(s, c, action_cond, context_rgb, train_cfg, training=True)
                policy_kwargs = action_policy_kwargs_from_targets(tgt) if bool(cfg.get("model", {}).get("enable_action_policy", False)) else None
                out = _forward_joint_model(
                    wm_model,
                    s_cond,
                    c_cond,
                    action_cond=action_cond_model,
                    context_rgb=context_rgb_cond,
                    prior_clean_tokens=prior_clean_tokens_from_targets(tgt),
                    pixel=False,
                    bridging=False,
                    policy_kwargs=policy_kwargs,
                )
                losses = compute_losses(out, tgt, weights, None)
                losses.update({k: v.detach() for k, v in dropout_losses.items()})
                policy_action_cond = policy_action_cond_from_wm_out(out)
                wan_wrong_action_cond = None
                wan_wrong_action_mode = None
                wan_wrong_wm_out = None
                wan_wrong_policy_action_cond = None
                wan_zero_wm_out = None
                wan_zero_policy_action_cond = None
                prior_clean_tokens = prior_clean_tokens_from_targets(tgt)
                if (
                    (
                        float(train_cfg.get("wan_action_cf_weight", 0.0)) > 0.0
                        or float(train_cfg.get("wan_source_action_cf_weight", 0.0)) > 0.0
                    )
                    and bool(train_cfg.get("wan_action_cf_recompute_wm", False))
                    and step >= int(train_cfg.get("wan_action_cf_start_step", 0) or 0)
                    and step % max(1, int(train_cfg.get("wan_action_cf_every", 1))) == 0
                ):
                    wan_wrong_action_cond, wan_wrong_action_mode = select_wan_wrong_action(action_cond_model, train_cfg, step)
                    recompute_no_grad = bool(train_cfg.get("wan_action_cf_recompute_wm_no_grad", True))
                    if recompute_no_grad:
                        with torch.no_grad():
                            wan_wrong_wm_out = _forward_joint_model(
                                wm_model,
                                s_cond,
                                c_cond,
                                action_cond=wan_wrong_action_cond,
                                context_rgb=context_rgb_cond,
                                prior_clean_tokens=prior_clean_tokens,
                                pixel=False,
                                bridging=False,
                                policy_kwargs=policy_kwargs,
                            )
                    else:
                        wan_wrong_wm_out = _forward_joint_model(
                            wm_model,
                            s_cond,
                            c_cond,
                            action_cond=wan_wrong_action_cond,
                            context_rgb=context_rgb_cond,
                            prior_clean_tokens=prior_clean_tokens,
                            pixel=False,
                            bridging=False,
                            policy_kwargs=policy_kwargs,
                        )
                    wan_wrong_policy_action_cond = policy_action_cond_from_wm_out(wan_wrong_wm_out)
                zero_hold_active = (
                    float(train_cfg.get("wan_zero_action_hold_weight", 0.0) or 0.0) > 0.0
                    and step >= int(train_cfg.get("wan_zero_action_hold_start_step", train_cfg.get("wan_action_cf_start_step", 0)) or 0)
                    and step % max(1, int(train_cfg.get("wan_zero_action_hold_every", train_cfg.get("wan_action_cf_every", 1)) or 1)) == 0
                )
                if zero_hold_active and bool(train_cfg.get("wan_zero_action_hold_recompute_wm", False)):
                    wan_zero_action_cond = torch.zeros_like(action_cond_model)
                    zero_recompute_no_grad = bool(
                        train_cfg.get("wan_zero_action_hold_recompute_wm_no_grad", train_cfg.get("wan_action_cf_recompute_wm_no_grad", True))
                    )
                    if zero_recompute_no_grad:
                        with torch.no_grad():
                            wan_zero_wm_out = _forward_joint_model(
                                wm_model,
                                s_cond,
                                c_cond,
                                action_cond=wan_zero_action_cond,
                                context_rgb=context_rgb_cond,
                                prior_clean_tokens=prior_clean_tokens,
                                pixel=False,
                                bridging=False,
                                policy_kwargs=policy_kwargs,
                            )
                    else:
                        wan_zero_wm_out = _forward_joint_model(
                            wm_model,
                            s_cond,
                            c_cond,
                            action_cond=wan_zero_action_cond,
                            context_rgb=context_rgb_cond,
                            prior_clean_tokens=prior_clean_tokens,
                            pixel=False,
                            bridging=False,
                            policy_kwargs=policy_kwargs,
                        )
                    wan_zero_policy_action_cond = policy_action_cond_from_wm_out(wan_zero_wm_out)
            wan_cleanup = lambda backward_ok=False: None
            with torch.enable_grad():
                wan_loss, wan_parts, wan_cleanup = compute_wan_ti2v_velocity_loss(
                    adapter=wan_control_adapter,
                    pipeline=pipeline,
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
                    policy_action_cond=policy_action_cond.detach() if policy_action_cond is not None else None,
                    wan_wrong_action_cond=wan_wrong_action_cond,
                    wan_wrong_action_mode=wan_wrong_action_mode,
                    wan_wrong_wm_out=wan_wrong_wm_out,
                    wan_wrong_policy_action_cond=wan_wrong_policy_action_cond,
                    wan_zero_wm_out=wan_zero_wm_out,
                    wan_zero_policy_action_cond=wan_zero_policy_action_cond,
                )
                losses.update({k: v.detach() for k, v in wan_parts.items()})
                loss = losses["L_total"] + float(train_cfg.get("wan_ti2v_weight", 1.0)) * wan_loss
            finite_count = _distributed_finite_count(torch.isfinite(loss), device, world)
            opt.zero_grad(set_to_none=True)
            if finite_count != world:
                wan_cleanup(False)
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
                wan_cleanup(backward_ok)
            sync_module(wm_model, world)
            sync_module(wan_control_adapter, world)
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
                wv = float(losses.get("L_wan_ti2v_velocity", loss.new_zeros(())).detach().float())
                wsrc = float(losses.get("L_wan_wm_source", loss.new_zeros(())).detach().float())
                wmse = float(losses.get("wan_velocity_mse", loss.new_zeros(())).detach().float())
                wdyn = float(losses.get("wan_velocity_dynamic_mse", loss.new_zeros(())).detach().float())
                wsta = float(losses.get("wan_velocity_static_mse", loss.new_zeros(())).detach().float())
                wcf = float(losses.get("wan_action_cf", loss.new_zeros(())).detach().float())
                wcfgap = float(losses.get("wan_action_cf_gap", loss.new_zeros(())).detach().float())
                wsrc_cf = float(losses.get("wan_source_action_cf", loss.new_zeros(())).detach().float())
                wzhold = float(losses.get("wan_zero_hold_active", loss.new_zeros(())).detach().float())
                wrgb = float(losses.get("wan_decoded_rgb_loss", loss.new_zeros(())).detach().float())
                wrgb_sta = float(losses.get("wan_decoded_rgb_static_l1", loss.new_zeros(())).detach().float())
                wrgb_dyn = float(losses.get("wan_decoded_rgb_dynamic_l1", loss.new_zeros(())).detach().float())
                wrgb_mr = float(losses.get("wan_decoded_rgb_motion_ratio", loss.new_zeros(())).detach().float())
                wvam = float(losses.get("wan_vam_action_loss", loss.new_zeros(())).detach().float())
                wvam_pose = float(losses.get("wan_vam_action_pose_l1", loss.new_zeros(())).detach().float())
                wvam_grip = float(losses.get("wan_vam_action_grip_l1", loss.new_zeros(())).detach().float())
                wvam_delta = float(losses.get("wan_vam_action_delta_l1", loss.new_zeros(())).detach().float())
                wvam_src = float(losses.get("wan_vam_action_source_mode", loss.new_zeros(())).detach().float())
                sig = float(losses.get("wan_sigma", loss.new_zeros(())).detach().float())
                print(
                    f"[rank0] step {step} ep={epoch} L_total={float(loss.detach().float()):.4f} "
                    f"native={float(losses['L_total'].detach().float()):.4f} "
                    f"depth={depth:.4f} point={point:.4f} pose={pose:.4f} action={action:.4f} "
                    f"wan_vel={wv:.4f} wan_src={wsrc:.4f} wan_mse={wmse:.5f} "
                    f"wan_dyn={wdyn:.5f} wan_static={wsta:.5f} wan_cf={wcf:.4f} cf_gap={wcfgap:.5f} "
                    f"wan_src_cf={wsrc_cf:.4f} wan_zero_hold={wzhold:.0f} wan_rgb={wrgb:.4f} wan_rgb_static={wrgb_sta:.5f} "
                    f"wan_rgb_dyn={wrgb_dyn:.5f} wan_rgb_mratio={wrgb_mr:.3f} "
                    f"wan_vam_action={wvam:.4f} vam_pose={wvam_pose:.4f} vam_grip={wvam_grip:.4f} "
                    f"vam_delta={wvam_delta:.4f} vam_src={wvam_src:.0f} sigma={sig:.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if rank == 0 and tb is not None and step % int(train_cfg.get("log_every", 50)) == 0:
                tb.add_scalar("train/L_total_with_wan", float(loss.detach().float()), step)
                for key, value in losses.items():
                    tb.add_scalar(f"train/{key}", float(value.detach().float()), step)
                tb.add_scalar("lr/wm", sched.get_last_lr()[0], step)
            step += 1

            if ckpt_every > 0 and step % ckpt_every == 0:
                step_path = ckpt_dir / f"step_{step:08d}.pt"
                wan_trainable_path = ckpt_dir / f"wan_trainable_step_{step:08d}.pt"
                save_checkpoint(
                    path=step_path,
                    wm_model=wm_model,
                    adapter=wan_control_adapter,
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
                save_wan_trainable_checkpoint_maybe_fsdp(wan_trainable_path, transformer, train_cfg=train_cfg, step=step, rank=rank)
                if rank == 0:
                    link_latest(ckpt_dir, step_path)
                    wan_trainable_latest = ckpt_dir / "wan_trainable_latest.pt"
                    if wan_trainable_latest.exists() or wan_trainable_latest.is_symlink():
                        wan_trainable_latest.unlink()
                    wan_trainable_latest.symlink_to(wan_trainable_path.name)
                    prune_step_checkpoints(ckpt_dir, train_cfg)
                    control_path = ckpt_dir / f"wan_control_step_{step:08d}.pt"
                    save_wan_ti2v_control_checkpoint(control_path, wan_control_adapter, wm_ckpt=step_path, step=step)
                    control_latest = ckpt_dir / "wan_control_latest.pt"
                    if control_latest.exists() or control_latest.is_symlink():
                        control_latest.unlink()
                    control_latest.symlink_to(control_path.name)
            if world > 1 and ckpt_every > 0 and step % ckpt_every == 0:
                dist.barrier()
            del loss, wan_loss, wan_parts, out, losses
            del s, c, action_cond, context_rgb, tgt
            del s_cond, c_cond, action_cond_model, context_rgb_cond
            if device.type == "cuda" and empty_cache_every > 0 and step % empty_cache_every == 0:
                torch.cuda.empty_cache()

        wm_model.eval()
        wan_control_adapter.eval()
        transformer.eval()
        agg: dict[str, float] = {}
        wan_agg: dict[str, float] = {}
        nb = 0
        max_val_batches = int(train_cfg.get("max_val_batches", 16))
        wan_val_batches = int(train_cfg.get("wan_val_batches", min(max(1, max_val_batches), 4)))
        wan_val_sum = 0.0
        wan_val_nb = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if max_val_batches > 0 and bi >= max_val_batches:
                    break
                s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, int(cfg["data"]["k"]))
                with torch.autocast(device_type="cuda", dtype=_dtype(precision), enabled=device.type == "cuda"):
                    policy_kwargs = action_policy_kwargs_from_targets(tgt) if bool(cfg.get("model", {}).get("enable_action_policy", False)) else None
                    out = _forward_joint_model(
                        wm_model,
                        s,
                        c,
                        action_cond=action_cond,
                        context_rgb=context_rgb,
                        prior_clean_tokens=prior_clean_tokens_from_targets(tgt),
                        pixel=False,
                        bridging=False,
                        policy_kwargs=policy_kwargs,
                    )
                    losses = compute_losses(out, tgt, weights, None)
                    policy_action_cond = policy_action_cond_from_wm_out(out)
                if bool(train_cfg.get("enable_wan_ti2v_loss", False)) and (wan_val_batches < 0 or bi < wan_val_batches):
                    wan_loss_val, wan_parts_val, wan_cleanup_val = compute_wan_ti2v_velocity_loss(
                        adapter=wan_control_adapter,
                        pipeline=pipeline,
                        transformer=transformer,
                        wm_out=out,
                        batch=batch,
                        context_rgb=context_rgb,
                        action_cond=action_cond,
                        task_emb=c,
                        tgt=tgt,
                        device=device,
                        train_cfg=train_cfg,
                        precision=precision,
                        step=step + bi,
                        policy_action_cond=policy_action_cond.detach() if policy_action_cond is not None else None,
                    )
                    try:
                        wan_val_sum += float(wan_loss_val.detach().float())
                        for key, value in wan_parts_val.items():
                            wan_agg[key] = wan_agg.get(key, 0.0) + float(value.detach().float())
                        wan_val_nb += 1
                    finally:
                        wan_cleanup_val(False)
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
            wan_keys = sorted(wan_agg)
            wan_vals = torch.tensor(
                [wan_agg[k] for k in wan_keys] + [float(wan_val_sum), float(wan_val_nb)],
                device=device,
            )
            dist.all_reduce(wan_vals)
            wan_val_sum = float(wan_vals[-2].item())
            wan_val_nb = int(max(0.0, float(wan_vals[-1].item())))
            wan_den = max(1.0, float(wan_val_nb))
            wan_agg = {k: float(wan_vals[i].item()) / wan_den for i, k in enumerate(wan_keys)}
        if rank == 0 and (world <= 1):
            for key in list(agg):
                agg[key] = agg[key] / max(1, nb)
            for key in list(wan_agg):
                wan_agg[key] = wan_agg[key] / max(1, wan_val_nb)
        wan_val_avg = wan_val_sum / max(1, wan_val_nb)
        val_native_total = agg.get("L_total", float("inf")) if rank == 0 else float("inf")
        val_total = (val_native_total + float(train_cfg.get("wan_ti2v_weight", 1.0)) * wan_val_avg) if rank == 0 else float("inf")
        if rank == 0:
            for key, value in wan_agg.items():
                agg[f"wan_val/{key}"] = float(value)
            agg["L_wan_val_total"] = float(wan_val_avg)
            agg["L_total_with_wan"] = float(val_total)
        is_best = False
        if rank == 0:
            print(
                f"[rank0] epoch {epoch}: val_native_total={val_native_total:.4f} "
                f"val_wan={wan_val_avg:.4f} val_total_with_wan={val_total:.4f} best={best_val:.4f}",
                flush=True,
            )
            if tb is not None:
                for key, value in agg.items():
                    tb.add_scalar(f"val/{key}", value, step)
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
                adapter=wan_control_adapter,
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
                    adapter=wan_control_adapter,
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
                save_wan_trainable_checkpoint_maybe_fsdp(ckpt_dir / "wan_trainable_best.pt", transformer, train_cfg=train_cfg, step=step, rank=rank)
                if rank == 0:
                    save_wan_ti2v_control_checkpoint(ckpt_dir / "wan_control_best.pt", wan_control_adapter, metrics=agg, wm_ckpt=best_path, step=step)
        if world > 1:
            dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
