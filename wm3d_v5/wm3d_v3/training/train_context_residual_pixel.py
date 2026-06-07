"""Train a context-conditioned RGB decoder from GT future P256 tokens."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import lpips
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.models.context_residual_pixel_decoder import (
    ContextResidualPixelDecoder,
    ContextResidualPixelDecoderConfig,
)


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def build_datasets(cfg: dict) -> tuple[Subset, Subset]:
    records = read_manifest(cfg["data"]["manifest"])
    wcfg = WindowConfig(
        T=cfg["data"]["T"],
        k=cfg["data"]["k"],
        stride=cfg["data"]["stride"],
        cache_root=Path(cfg["data"]["cache_root"]),
        tokens_subdir=cfg["data"].get("tokens_subdir", "vggt_p256"),
        action_stats=Path(cfg["data"]["action_stats"]) if cfg["data"].get("action_stats") else None,
    )
    ds = OXEWindowDataset(records, wcfg)
    if len(ds) == 0:
        raise RuntimeError("OXEWindowDataset empty; check manifest and cache_root")

    gen = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(len(ds), generator=gen).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    max_train = cfg["train"].get("max_train_windows")
    max_val = cfg["train"].get("max_val_windows")
    if max_train:
        train_idx = train_idx[: int(max_train)]
    if max_val:
        val_idx = val_idx[: int(max_val)]
    return Subset(ds, train_idx), Subset(ds, val_idx)


def make_loader(ds: Subset, cfg: dict, world: int, rank: int, train: bool):
    bs = cfg["train"]["batch_size_per_gpu"]
    nw = cfg["train"]["num_workers"]
    sampler = None
    if world > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=world,
            rank=rank,
            shuffle=train,
            drop_last=train,
        )
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=train,
        persistent_workers=nw > 0,
    ), sampler


def batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    tokens = batch["s_tgt"].to(device, non_blocking=True)
    context = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
    target = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous()
    action = batch["action_tgt"].to(device, non_blocking=True)
    action_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
    action_cond = make_action_condition(action, action_norm)
    task = batch["c"].to(device, non_blocking=True)
    return tokens, context, target, action_cond, task


def apply_train_ablations(
    tokens: torch.Tensor,
    context: torch.Tensor,
    action_cond: torch.Tensor,
    task: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optional controls for causal ablations."""
    train_cfg = cfg["train"]
    if train_cfg.get("zero_tokens", False):
        tokens = torch.zeros_like(tokens)
    if train_cfg.get("zero_context", False):
        context = torch.zeros_like(context)
    if train_cfg.get("zero_action", False):
        action_cond = torch.zeros_like(action_cond)
    if train_cfg.get("zero_task", False):
        task = torch.zeros_like(task)
    return tokens, context, action_cond, task


def gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def compute_pixel_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    lpips_fn,
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    l1 = F.l1_loss(pred, target)
    ref = context[:, None].expand_as(target)
    motion = (target - ref).abs().mean(dim=2, keepdim=True)
    moving = (motion > float(loss_cfg.get("motion_threshold", 0.03))).to(dtype=pred.dtype)
    weight = 1.0 + float(loss_cfg.get("motion_gain", 4.0)) * moving
    motion_l1 = ((pred - target).abs() * weight).mean()
    edge = gradient_l1(pred, target)

    bsz, horizon, channels, height, width = pred.shape
    pred_flat = pred.reshape(bsz * horizon, channels, height, width).float()
    tgt_flat = target.reshape(bsz * horizon, channels, height, width).float()
    lpips_val = lpips_fn(pred_flat * 2.0 - 1.0, tgt_flat * 2.0 - 1.0).mean()

    total = (
        float(loss_cfg["l1"]) * l1
        + float(loss_cfg["lpips"]) * lpips_val
        + float(loss_cfg["motion_l1"]) * motion_l1
        + float(loss_cfg["edge"]) * edge
    )
    return {
        "L_total": total,
        "L_l1": l1,
        "L_lpips": lpips_val,
        "L_motion_l1": motion_l1,
        "L_edge": edge,
        "motion_frac": moving.mean(),
    }


def reduce_metrics(metrics: dict[str, float], count: int, device: torch.device, world: int) -> dict[str, float]:
    if world <= 1:
        return {k: v / max(1, count) for k, v in metrics.items()}
    keys = sorted(metrics.keys())
    payload = torch.tensor([metrics[k] for k in keys] + [float(count)], device=device)
    dist.all_reduce(payload)
    total_count = max(1.0, float(payload[-1].item()))
    return {k: float(payload[i].item()) / total_count for i, k in enumerate(keys)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--dry_run_batches", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_datasets(cfg)
    train_loader, train_sampler = make_loader(train_ds, cfg, world, rank, train=True)
    val_loader, _ = make_loader(val_ds, cfg, world, rank, train=False)

    model_cfg = ContextResidualPixelDecoderConfig(**cfg["model"])
    model = ContextResidualPixelDecoder(model_cfg).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local], find_unused_parameters=False)

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, 0.95),
    )
    total_steps = max(1, len(train_loader) * int(cfg["train"]["epochs"]))
    warmup = int(cfg["train"].get("warmup_steps", 0))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"].get("ckpt_dir", "ckpt")
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (out_root / cfg["out"].get("tb_dir", "tb")).mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_root / cfg["out"].get("tb_dir", "tb"))
        target = model.module if isinstance(model, DDP) else model
        print(
            f"[rank0] ContextResidualPixelDecoder: {target.num_trainable_params()/1e6:.1f}M; "
            f"train_windows={len(train_ds)} val_windows={len(val_ds)} world={world}",
            flush=True,
        )

    start_epoch = 0
    step = 0
    best_val = float("inf")
    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(sd["model"], strict=True)
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        start_epoch = int(sd["epoch"]) + 1
        step = int(sd["step"])
        best_val = float(sd.get("best_val", best_val))
        if rank == 0:
            print(f"[rank0] resumed from {args.resume} at epoch {start_epoch}", flush=True)

    amp_enabled = device.type == "cuda" and cfg["train"].get("precision", "bf16") == "bf16"
    for epoch in range(start_epoch, int(cfg["train"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            tokens, context, target_rgb, action_cond, task = batch_to_device(batch, device)
            tokens, context, action_cond, task = apply_train_ablations(
                tokens, context, action_cond, task, cfg
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(tokens, context, action_cond=action_cond, task_emb=task)
                losses = compute_pixel_losses(pred, target_rgb, context, lpips_fn, cfg["loss"])
            loss = losses["L_total"]
            if not torch.isfinite(loss):
                if rank == 0:
                    print(f"[rank0] non-finite loss at step {step}; skipping", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
            if not torch.isfinite(grad_norm):
                if rank == 0:
                    print(f"[rank0] non-finite grad_norm at step {step}; skipping", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.step()
            sched.step()

            if rank == 0 and args.print_every and step % args.print_every == 0:
                print(
                    f"[rank0] step {step} ep={epoch} "
                    f"L={float(losses['L_total'].detach().float()):.4f} "
                    f"L1={float(losses['L_l1'].detach().float()):.4f} "
                    f"LPIPS={float(losses['L_lpips'].detach().float()):.4f} "
                    f"motion={float(losses['L_motion_l1'].detach().float()):.4f} "
                    f"edge={float(losses['L_edge'].detach().float()):.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            if rank == 0 and step % int(cfg["train"]["log_every"]) == 0:
                for name, value in losses.items():
                    tb.add_scalar(f"train/{name}", float(value.detach().float()), step)
                tb.add_scalar("train/grad_norm", float(grad_norm.detach().float()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
            step += 1
            if args.dry_run_batches and batch_idx + 1 >= args.dry_run_batches:
                break

        model.eval()
        agg: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                tokens, context, target_rgb, action_cond, task = batch_to_device(batch, device)
                tokens, context, action_cond, task = apply_train_ablations(
                    tokens, context, action_cond, task, cfg
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                    pred = model(tokens, context, action_cond=action_cond, task_emb=task)
                    losses = compute_pixel_losses(pred, target_rgb, context, lpips_fn, cfg["loss"])
                for name, value in losses.items():
                    agg[name] = agg.get(name, 0.0) + float(value.detach().float())
                count += 1
                if args.dry_run_batches and batch_idx + 1 >= args.dry_run_batches:
                    break
        reduced = reduce_metrics(agg, count, device, world)

        if rank == 0:
            for name, value in reduced.items():
                tb.add_scalar(f"val/{name}", value, step)
            val_total = reduced["L_total"]
            target_model = model.module if isinstance(model, DDP) else model
            ckpt = {
                "model": target_model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "step": step,
                "val_total": val_total,
                "best_val": best_val,
                "cfg": cfg,
            }
            torch.save(ckpt, ckpt_dir / "latest.pt")
            if (epoch + 1) % int(cfg["train"]["ckpt_every_epochs"]) == 0:
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_total < best_val:
                best_val = val_total
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"[rank0] epoch {epoch}: val_total {val_total:.4f} best {best_val:.4f}", flush=True)

        if args.dry_run_batches:
            break

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
