"""Stage0 from-scratch wm3d pretraining with Hunyuan DiT as the RGB generator.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_hunyuan_dit_control_adapter import (
    build_hunyuan_backend_args,
    controlled_dit_forward,
    encode_hunyuan_latents,
    encode_hunyuan_prompts,
    latent_motion_mask_from_target,
    prompts_from_batch,
    rotary_freqs,
    target_video_from_batch,
    weighted_velocity_losses,
    context_video_from_batch,
)
from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.hunyuan_dit_control_adapter import HunyuanDiTControlAdapter, HunyuanDiTControlConfig
from wm3d_v3.models.hunyuan_lora import (
    HunyuanLoRAConfig,
    apply_lora_to_linear_modules,
    collect_trainable_state_dict,
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
    if module is not None and world > 1:
        _all_reduce_gradients(module, world)


def broadcast_module_state(module: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buf in module.buffers():
        dist.broadcast(buf.data, src=0)


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
    return torch.bfloat16 if str(name).lower() == "bf16" else torch.float16


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
    hunyuan_trainable = [p for p in transformer.parameters() if p.requires_grad]
    if wm_params:
        groups.append({"name": "wm3d", "params": wm_params, "lr": float(train_cfg.get("wm_lr", train_cfg["lr"])), "weight_decay": wd})
    if control_params:
        groups.append({"name": "hunyuan_control_adapter", "params": control_params, "lr": float(train_cfg.get("hunyuan_control_lr", 2e-5)), "weight_decay": wd})
    if hunyuan_trainable:
        groups.append({"name": "hunyuan_trainable", "params": hunyuan_trainable, "lr": float(train_cfg.get("hunyuan_lora_lr", 5e-6)), "weight_decay": float(train_cfg.get("hunyuan_weight_decay", 0.0))})
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
        )
        lora_report = apply_lora_to_linear_modules(transformer, lora_cfg)
    patterns = train_cfg.get("hunyuan_dit_trainable_patterns") or ()
    if patterns:
        pattern_report = set_trainable_by_patterns(transformer, patterns, train_cfg.get("hunyuan_dit_trainable_exclude", ()))
    else:
        pattern_report = {"params": 0, "tensors": 0, "preview": []}

    adapter = HunyuanDiTControlAdapter(
        HunyuanDiTControlConfig(
            hidden=int(train_cfg.get("hunyuan_control_hidden", 192)),
            dit_hidden=int(getattr(transformer, "hidden_size", 3072)),
            double_blocks=len(backend._iter_transformer_blocks(transformer, "double_blocks")),
            single_blocks=len(backend._iter_transformer_blocks(transformer, "single_blocks")),
            use_rough=False,
            use_rgb_features=False,
            use_block_action_film=bool(train_cfg.get("hunyuan_use_block_action_film", True)),
            block_action_film_scale=float(train_cfg.get("hunyuan_block_action_film_scale", 1.0)),
            block_action_film_hidden=int(train_cfg.get("hunyuan_block_action_film_hidden", 192)),
        )
    ).to(device)
    broadcast_module_state(adapter, world)
    if rank == 0:
        print(f"[rank0] hunyuan_lora={json.dumps(lora_report, default=str)}", flush=True)
        print(f"[rank0] hunyuan_partial_train={json.dumps(pattern_report, default=str)}", flush=True)
        print(f"[rank0] HunyuanDiTControlAdapter params={_count_trainable(adapter)/1e6:.2f}M", flush=True)
        print(f"[rank0] Hunyuan trainable params={_count_trainable(transformer)/1e6:.2f}M", flush=True)
    return sampler, transformer, adapter, lora_report, pattern_report


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if context_rgb is None or "rgb_tgt_p" not in tgt:
        ref = wm_out["pred_tokens"]
        zero = ref.new_zeros(())
        return zero, {"L_hunyuan_dit_velocity": zero}

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
        path_type = str(train_cfg.get("hunyuan_path_type", "context")).lower()
        if path_type == "noise":
            source_latents = torch.randn_like(target_latents)
        elif path_type == "context":
            source_latents = encode_hunyuan_latents(sampler.pipeline.vae, context_video_from_batch(context_rgb, frames))
        else:
            raise ValueError(f"stage0 Hunyuan DiT PT supports path_type noise/context only, got {path_type!r}")
        noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
        target_velocity = float(train_cfg.get("hunyuan_velocity_target_sign", -1.0)) * (source_latents - target_latents)
        motion_mask = None
        if str(train_cfg.get("hunyuan_latent_motion_mask_source", "gt_rgb")).lower() == "gt_rgb":
            motion_mask = latent_motion_mask_from_target(
                context_rgb,
                target_video,
                latent_shape=tuple(target_latents.shape),
                threshold=float(train_cfg.get("hunyuan_latent_motion_threshold", 0.02)),
                dilate=int(train_cfg.get("hunyuan_latent_motion_dilate", 1)),
            )

    autocast_dtype = _dtype(precision)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        pred_velocity = controlled_dit_forward(
            adapter=adapter,
            transformer=transformer,
            noisy_latents=noisy.to(dtype=autocast_dtype if device.type == "cuda" else noisy.dtype),
            sigma=sigma,
            text=text,
            freqs=freqs,
            wm_out=wm_out,
            context_rgb=context_rgb,
            action_cond=action_cond,
            task_emb=task_emb,
            embedded_cfg_scale=float(train_cfg.get("hunyuan_embedded_cfg_scale", 6.0)),
            control_scale=float(train_cfg.get("hunyuan_control_scale", 1.0)),
            latent_control_scale=float(train_cfg.get("hunyuan_latent_control_scale", 1.0)),
        )
        mse, l1, parts = weighted_velocity_losses(
            pred_velocity,
            target_velocity,
            motion_mask,
            dynamic_weight=float(train_cfg.get("hunyuan_velocity_dynamic_weight", 8.0)),
            static_weight=float(train_cfg.get("hunyuan_velocity_static_weight", 1.0)),
        )
        loss = float(train_cfg.get("hunyuan_velocity_mse_weight", 1.0)) * mse + float(train_cfg.get("hunyuan_velocity_l1_weight", 0.05)) * l1
    out = {
        "L_hunyuan_dit_velocity": loss,
        "hunyuan_dit_velocity_mse": pred_velocity.new_tensor(parts.get("velocity_mse", 0.0)),
        "hunyuan_dit_velocity_l1": pred_velocity.new_tensor(parts.get("velocity_l1", 0.0)),
        "hunyuan_dit_dynamic_mse": pred_velocity.new_tensor(parts.get("velocity_dynamic_mse", 0.0)),
        "hunyuan_dit_static_mse": pred_velocity.new_tensor(parts.get("velocity_static_mse", 0.0)),
        "hunyuan_dit_motion_mask": pred_velocity.new_tensor(parts.get("latent_motion_mask_mean", 0.0)),
        "hunyuan_dit_sigma": sigma.detach().float().mean(),
    }
    return loss, out


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
) -> None:
    payload = {
        "kind": "wm3d_stage0_hunyuan_dit_jointpt_v1",
        "model": wm_model.state_dict(),
        "hunyuan_control_adapter": adapter.state_dict(),
        "hunyuan_control_adapter_cfg": adapter.cfg.__dict__,
        "hunyuan_trainable": collect_trainable_state_dict(transformer),
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "cfg": cfg,
        "step": int(step),
        "epoch": int(epoch),
        "metrics": metrics or {},
    }
    tmp = path.with_name("." + path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


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
    if args.max_steps is not None:
        train_cfg["max_steps"] = int(args.max_steps)
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.disable_hunyuan_lora:
        train_cfg["hunyuan_dit_train_lora"] = False
    out_dir = args.out_dir or Path(cfg["out"]["root"])
    ckpt_dir = out_dir / cfg["out"].get("ckpt_dir", "ckpt")
    precision = str(train_cfg.get("precision", "bf16"))
    seed = int(cfg["data"].get("seed", 909)) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

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
            with torch.enable_grad():
                dit_loss, dit_parts = compute_hunyuan_dit_velocity_loss(
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
                if rank == 0:
                    print(f"[rank0] WARN skip non-finite loss step={step} finite={finite_count}/{world}", flush=True)
                sched.step()
                step += 1
                continue
            loss.backward()
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
                sig = float(losses.get("hunyuan_dit_sigma", loss.new_zeros(())).detach().float())
                print(
                    f"[rank0] step {step} ep={epoch} L_total={float(loss.detach().float()):.4f} "
                    f"native={float(losses['L_total'].detach().float()):.4f} "
                    f"depth={depth:.4f} point={point:.4f} pose={pose:.4f} action={action:.4f} "
                    f"hunyuan_dit_velocity={hv:.4f} h_mse={hmse:.5f} h_dyn={hdyn:.5f} h_static={hsta:.5f} sigma={sig:.3f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if rank == 0 and tb is not None and step % int(train_cfg.get("log_every", 50)) == 0:
                tb.add_scalar("train/L_total_with_hunyuan", float(loss.detach().float()), step)
                for key, value in losses.items():
                    tb.add_scalar(f"train/{key}", float(value.detach().float()), step)
                tb.add_scalar("lr/wm", sched.get_last_lr()[0], step)
            step += 1

            if rank == 0 and ckpt_every > 0 and step % ckpt_every == 0:
                step_path = ckpt_dir / f"step_{step:08d}.pt"
                save_checkpoint(path=step_path, wm_model=wm_model, adapter=hunyuan_control_adapter, transformer=transformer, opt=opt, sched=sched, cfg=cfg, step=step, epoch=epoch, metrics=None)
                link_latest(ckpt_dir, step_path)
                prune_step_checkpoints(ckpt_dir, train_cfg)
                save_hunyuan_trainable_checkpoint(ckpt_dir / f"hunyuan_trainable_step_{step:08d}.pt", transformer, lora_config=HunyuanLoRAConfig.from_any({"rank": int(train_cfg.get("hunyuan_lora_rank", 8)), "alpha": float(train_cfg.get("hunyuan_lora_alpha", 16.0))}), step=step)
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
        if rank == 0:
            val_total = agg.get("L_total", float("inf")) / max(1, nb)
            print(f"[rank0] epoch {epoch}: val_native_total={val_total:.4f} best={best_val:.4f}", flush=True)
            if tb is not None:
                for key, value in agg.items():
                    tb.add_scalar(f"val/{key}", value / max(1, nb), step)
            if not args.no_epoch_checkpoint:
                save_checkpoint(path=ckpt_dir / f"epoch_{epoch:03d}.pt", wm_model=wm_model, adapter=hunyuan_control_adapter, transformer=transformer, opt=opt, sched=sched, cfg=cfg, step=step, epoch=epoch, metrics=agg)
                if val_total < best_val:
                    best_val = val_total
                    save_checkpoint(path=ckpt_dir / "best.pt", wm_model=wm_model, adapter=hunyuan_control_adapter, transformer=transformer, opt=opt, sched=sched, cfg=cfg, step=step, epoch=epoch, metrics=agg)
        if world > 1:
            dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
