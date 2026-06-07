"""Train a Hunyuan-VAE flow denoiser conditioned on wm3d controls."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from wm3d_v3.models.hunyuan_flow_denoiser import (
    HunyuanFlowDenoiser,
    HunyuanFlowDenoiserConfig,
)
from wm3d_v3.training.train import (
    batch_to_device,
    build_datasets,
    build_model,
    load_action_stats_if_available,
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


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def load_compatible_state_dict(model: torch.nn.Module, state: dict) -> SimpleNamespace:
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    return SimpleNamespace(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        skipped_keys=skipped,
    )


def load_hunyuan_vae(args: argparse.Namespace, device: torch.device):
    os.environ.setdefault("MODEL_BASE", str(args.hunyuan_model_base))
    repo = Path(args.hunyuan_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hyvideo.vae import load_vae  # type: ignore

    vae, _, _, _ = load_vae("884-16c-hy", args.vae_precision, device=device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def maybe_subset(ds, max_windows: int, seed: int):
    if max_windows <= 0 or max_windows >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:max_windows].tolist()
    return Subset(ds, idx)


def target_video_from_batch(context_rgb: torch.Tensor, rgb_tgt_p: torch.Tensor) -> torch.Tensor:
    video = torch.cat([context_rgb[:, None], rgb_tgt_p], dim=1)
    return video.permute(0, 2, 1, 3, 4).contiguous()


def rough_video_from_wm_out(context_rgb: torch.Tensor, wm_out: dict) -> torch.Tensor | None:
    if "rgb" not in wm_out:
        return None
    return torch.cat([context_rgb[:, None], wm_out["rgb"].float()], dim=1).permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def encode_hunyuan_latents(vae, video_bcthw: torch.Tensor) -> torch.Tensor:
    x = video_bcthw.mul(2.0).sub(1.0)
    posterior = vae.encode(x.to(dtype=vae.dtype)).latent_dist
    latents = posterior.mode()
    return latents * float(vae.config.scaling_factor)


@torch.no_grad()
def decode_hunyuan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents / float(vae.config.scaling_factor)
    out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]
    return out.div(2.0).add(0.5).clamp(0.0, 1.0).float()


def motion_mask_from_rgb(rgb_tgt_p: torch.Tensor, context_rgb: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    motion = (rgb_tgt_p.float() - context_rgb.float()[:, None]).abs().mean(dim=2, keepdim=True)
    return (motion > threshold).float()


def denoiser_forward(
    denoiser,
    noisy_latents: torch.Tensor,
    sigma: torch.Tensor,
    wm_out: dict,
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    c: torch.Tensor,
    rough_latents: torch.Tensor | None,
) -> torch.Tensor:
    return denoiser(
        noisy_latents,
        sigma,
        wm_out["pred_tokens"],
        wm_out["depth"],
        context_rgb=context_rgb,
        motion_hint=wm_out.get("motion_hint"),
        rough_rgb=wm_out.get("rgb"),
        rough_latents=rough_latents,
        action_cond=action_cond,
        task_emb=c,
    )


@torch.no_grad()
def sample_flow(
    *,
    denoiser,
    wm_out: dict,
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    c: torch.Tensor,
    rough_latents: torch.Tensor | None,
    shape: tuple[int, ...],
    steps: int,
    seed: int,
    precision: str,
    source_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    device = context_rgb.device
    gen = torch.Generator(device=device).manual_seed(seed)
    if source_latents is None:
        x = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    else:
        x = source_latents.detach().float().clone()
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device)
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for i in range(steps):
        sigma = sigmas[i].expand(shape[0])
        sigma_next = sigmas[i + 1]
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
            velocity = denoiser_forward(
                denoiser,
                x.to(dtype=autocast_dtype),
                sigma,
                wm_out,
                context_rgb,
                action_cond,
                c,
                rough_latents,
            )
        x = x + velocity.float() * (sigma_next - sigmas[i])
    return x


def save_demo(path: Path, pred_video_bcthw: torch.Tensor, target_video_bcthw: torch.Tensor, rough_btchw: torch.Tensor | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred_video_bcthw[0].permute(1, 2, 3, 0).detach().cpu().clamp(0, 1)
    target = target_video_bcthw[0].permute(1, 2, 3, 0).detach().cpu().clamp(0, 1)
    frames = []
    for i in range(pred.shape[0]):
        row = [target[i], pred[i]]
        if rough_btchw is not None:
            rough = target[i] if i == 0 else rough_btchw[0, i - 1].permute(1, 2, 0).detach().cpu().clamp(0, 1)
            row.append(rough)
        frame = torch.cat(row, dim=1)
        frames.append((frame.numpy() * 255).round().astype("uint8"))
    imageio.mimsave(path, frames, fps=6)


def evaluate(
    *,
    denoiser,
    wm_model,
    vae,
    loader,
    device: torch.device,
    max_batches: int,
    precision: str,
    sample_steps: int,
    seed: int,
    path_type: str,
) -> dict[str, float]:
    denoiser.eval()
    totals = {
        "velocity_mse": 0.0,
        "velocity_l1": 0.0,
        "sample_decoded_l1": 0.0,
        "rough_l1": 0.0,
        "rough_vae_l1": 0.0,
        "vae_recon_l1": 0.0,
        "motion_sample_l1": 0.0,
        "motion_rough_l1": 0.0,
        "motion_rough_vae_l1": 0.0,
        "motion_vae_recon_l1": 0.0,
    }
    count = 0
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k=0)
            target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
            target_latents = encode_hunyuan_latents(vae, target_video)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                wm_out = wm_model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
            rough_video = rough_video_from_wm_out(context_rgb, wm_out)
            rough_latents = encode_hunyuan_latents(vae, rough_video) if rough_video is not None else None

            g = torch.Generator(device=device).manual_seed(seed + 1009 * bi)
            sigma = torch.rand(target_latents.shape[0], generator=g, device=device).clamp(1e-4, 1.0)
            if path_type == "noise":
                source_latents = torch.randn(target_latents.shape, generator=g, device=device, dtype=target_latents.dtype)
            elif path_type == "rough":
                if rough_latents is None:
                    raise RuntimeError("--path_type rough requires rough_latents")
                source_latents = rough_latents.to(dtype=target_latents.dtype)
            else:
                raise ValueError(f"unknown path_type: {path_type}")
            noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
            target_velocity = source_latents - target_latents
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                pred_velocity = denoiser_forward(
                    denoiser,
                    noisy,
                    sigma,
                    wm_out,
                    context_rgb,
                    action_cond,
                    c,
                    rough_latents,
                )

            sampled_latents = sample_flow(
                denoiser=denoiser,
                wm_out=wm_out,
                context_rgb=context_rgb,
                action_cond=action_cond,
                c=c,
                rough_latents=rough_latents,
                shape=tuple(target_latents.shape),
                steps=sample_steps,
                seed=seed + 7919 * bi,
                precision=precision,
                source_latents=rough_latents if path_type == "rough" else None,
            )
            decoded = decode_hunyuan_latents(vae, sampled_latents)
            vae_recon = decode_hunyuan_latents(vae, target_latents.float())
            rough_vae = decode_hunyuan_latents(vae, rough_latents.float()) if rough_latents is not None else None
            target_f = target_video.float()
            n = target_f.shape[0]

            totals["velocity_mse"] += float(F.mse_loss(pred_velocity.float(), target_velocity.float(), reduction="sum").cpu())
            totals["velocity_l1"] += float(F.l1_loss(pred_velocity.float(), target_velocity.float(), reduction="sum").cpu())
            totals["sample_decoded_l1"] += float((decoded - target_f).abs().sum().cpu())
            totals["vae_recon_l1"] += float((vae_recon - target_f).abs().sum().cpu())
            if rough_video is not None:
                totals["rough_l1"] += float((rough_video - target_f).abs().sum().cpu())
            if rough_vae is not None:
                totals["rough_vae_l1"] += float((rough_vae - target_f).abs().sum().cpu())

            motion_mask = motion_mask_from_rgb(tgt["rgb_tgt_p"], context_rgb).permute(0, 2, 1, 3, 4)
            motion_mask = torch.cat([torch.zeros_like(motion_mask[:, :, :1]), motion_mask], dim=2)
            denom = (motion_mask.sum() * target_f.shape[1]).clamp_min(1.0)
            totals["motion_sample_l1"] += float(((decoded - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            totals["motion_vae_recon_l1"] += float(((vae_recon - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            if rough_video is not None:
                totals["motion_rough_l1"] += float(((rough_video - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            if rough_vae is not None:
                totals["motion_rough_vae_l1"] += float(((rough_vae - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            count += n

    cfg = denoiser.module.cfg if isinstance(denoiser, DDP) else denoiser.cfg
    latent_numel = count * cfg.latent_channels * 3 * 32 * 32
    pixel_numel = count * 3 * 9 * 256 * 256
    metrics = {
        "velocity_mse": totals["velocity_mse"] / max(1, latent_numel),
        "velocity_l1": totals["velocity_l1"] / max(1, latent_numel),
        "sample_decoded_l1": totals["sample_decoded_l1"] / max(1, pixel_numel),
        "rough_l1": totals["rough_l1"] / max(1, pixel_numel),
        "rough_vae_l1": totals["rough_vae_l1"] / max(1, pixel_numel),
        "vae_recon_l1": totals["vae_recon_l1"] / max(1, pixel_numel),
        "motion_sample_l1": totals["motion_sample_l1"] / max(1, count),
        "motion_rough_l1": totals["motion_rough_l1"] / max(1, count),
        "motion_rough_vae_l1": totals["motion_rough_vae_l1"] / max(1, count),
        "motion_vae_recon_l1": totals["motion_vae_recon_l1"] / max(1, count),
        "count": float(count),
    }
    if dist.is_available() and dist.is_initialized():
        keys = sorted(metrics)
        tensor = torch.tensor([metrics[k] for k in keys], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        metrics = {k: float(v) for k, v in zip(keys, tensor.detach().cpu().tolist())}
    denoiser.train()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_cfg", type=Path, required=True)
    ap.add_argument("--wm_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--hunyuan_repo", type=Path, default=Path("/data/Minko/external/HunyuanVideo"))
    ap.add_argument("--hunyuan_model_base", type=Path, default=Path("/data/Minko/models/hunyuan_video"))
    ap.add_argument("--vae_precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size_per_gpu", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--velocity_mse_weight", type=float, default=1.0)
    ap.add_argument("--velocity_l1_weight", type=float, default=0.05)
    ap.add_argument("--max_train_windows", type=int, default=20000)
    ap.add_argument("--max_val_windows", type=int, default=1600)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--sample_steps", type=int, default=8)
    ap.add_argument("--path_type", choices=["noise", "rough"], default="noise")
    ap.add_argument("--print_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--n_blocks", type=int, default=4)
    args = ap.parse_args()

    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    wm_cfg = yaml.safe_load(args.wm_cfg.read_text())
    train_ds, val_ds = build_datasets(wm_cfg)
    train_ds = maybe_subset(train_ds, args.max_train_windows, args.seed)
    val_ds = maybe_subset(val_ds, args.max_val_windows, args.seed + 1)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    wm_model = build_model(wm_cfg).to(device).eval()
    wm_sd = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    load_res = load_compatible_state_dict(wm_model, wm_sd["model"])
    load_action_stats_if_available(wm_model, wm_cfg, rank, device)
    for p in wm_model.parameters():
        p.requires_grad_(False)

    vae = load_hunyuan_vae(args, device)
    cfg = HunyuanFlowDenoiserConfig(hidden=args.hidden, n_blocks=args.n_blocks)
    denoiser = HunyuanFlowDenoiser(cfg).to(device)
    if world > 1:
        denoiser = DDP(denoiser, device_ids=[local])

    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "ckpt"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "wm_cfg": str(args.wm_cfg),
            "wm_ckpt": str(args.wm_ckpt),
            "wm_ckpt_epoch": wm_sd.get("epoch"),
            "wm_ckpt_val_total": wm_sd.get("val_total"),
            "load_missing": len(load_res.missing_keys),
            "load_skipped": len(load_res.skipped_keys),
            "train_windows": len(train_ds),
            "val_windows": len(val_ds),
            "world": world,
            "args": vars(args),
        }
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        params = sum(p.numel() for p in (denoiser.module if isinstance(denoiser, DDP) else denoiser).parameters() if p.requires_grad)
        print(f"[rank0] HunyuanFlowDenoiser: {params/1e6:.2f}M train_windows={len(train_ds)} val_windows={len(val_ds)} total_steps={total_steps}", flush=True)

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    best = float("inf")
    step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        denoiser.train()
        for batch in train_loader:
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k=0)
            target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
            with torch.no_grad():
                target_latents = encode_hunyuan_latents(vae, target_video)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                    wm_out = wm_model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
                rough_video = rough_video_from_wm_out(context_rgb, wm_out)
                rough_latents = encode_hunyuan_latents(vae, rough_video) if rough_video is not None else None
                sigma = torch.rand(target_latents.shape[0], device=device).clamp(1e-4, 1.0)
                if args.path_type == "noise":
                    source_latents = torch.randn_like(target_latents)
                elif args.path_type == "rough":
                    if rough_latents is None:
                        raise RuntimeError("--path_type rough requires wm_out['rgb'] so rough_latents can be encoded")
                    source_latents = rough_latents.to(dtype=target_latents.dtype)
                else:
                    raise ValueError(f"unknown path_type: {args.path_type}")
                noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
                target_velocity = source_latents - target_latents
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                pred_velocity = denoiser_forward(
                    denoiser,
                    noisy,
                    sigma,
                    wm_out,
                    context_rgb,
                    action_cond,
                    c,
                    rough_latents,
                )
                velocity_mse = F.mse_loss(pred_velocity.float(), target_velocity.float())
                velocity_l1 = F.l1_loss(pred_velocity.float(), target_velocity.float())
                loss = args.velocity_mse_weight * velocity_mse + args.velocity_l1_weight * velocity_l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if rank == 0 and (step % args.print_every == 0):
                print(
                    f"[rank0] step {step} ep {epoch} loss={float(loss.detach().cpu()):.6f} "
                    f"velocity_mse={float(velocity_mse.detach().cpu()):.6f} "
                    f"velocity_l1={float(velocity_l1.detach().cpu()):.6f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            step += 1

        metrics = evaluate(
            denoiser=denoiser,
            wm_model=wm_model,
            vae=vae,
            loader=val_loader,
            device=device,
            max_batches=args.eval_batches,
            precision=args.precision,
            sample_steps=args.sample_steps,
            seed=args.seed + epoch * 10000,
            path_type=args.path_type,
        )
        score = metrics["sample_decoded_l1"]
        if rank == 0:
            print(f"[rank0] epoch {epoch}: {json.dumps(metrics, sort_keys=True)}", flush=True)
            target = denoiser.module if isinstance(denoiser, DDP) else denoiser
            ckpt = {
                "epoch": epoch,
                "step": step,
                "model": target.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "metrics": metrics,
                "cfg": cfg.__dict__,
            }
            torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if score < best:
                best = score
                torch.save(ckpt, ckpt_dir / "best.pt")
                try:
                    demo_batch = next(iter(val_loader))
                    s, c, action_cond, context_rgb, tgt = batch_to_device(demo_batch, device, k=0)
                    target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
                    target_latents = encode_hunyuan_latents(vae, target_video)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                        wm_out = wm_model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
                    rough_video = rough_video_from_wm_out(context_rgb, wm_out)
                    rough_latents = encode_hunyuan_latents(vae, rough_video) if rough_video is not None else None
                    sampled_latents = sample_flow(
                        denoiser=denoiser,
                        wm_out=wm_out,
                        context_rgb=context_rgb,
                        action_cond=action_cond,
                        c=c,
                        rough_latents=rough_latents,
                        shape=tuple(target_latents.shape),
                        steps=args.sample_steps,
                        seed=args.seed + 4242 + epoch,
                        precision=args.precision,
                        source_latents=rough_latents if args.path_type == "rough" else None,
                    )
                    pred_video = decode_hunyuan_latents(vae, sampled_latents)
                    save_demo(args.out_dir / "demo_best.gif", pred_video, target_video.float(), wm_out.get("rgb"))
                except Exception as exc:
                    print(f"[rank0] demo save failed: {exc}", flush=True)
    cleanup_ddp()


if __name__ == "__main__":
    main()
