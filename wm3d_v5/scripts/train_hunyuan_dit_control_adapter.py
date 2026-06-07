"""Train wm3d controls as an opt-in Hunyuan DiT block-control adapter.

This is the real Hunyuan generation bridge, separate from the VAE latent adapter
and from the core stage0/stage1/stage2 world-model training. It freezes the
wm3d world model, Hunyuan VAE, Hunyuan text encoders, and Hunyuan DiT; only the
zero-init wm3d -> DiT control adapter is trainable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.models.hunyuan_dit_control_adapter import (
    HunyuanDiTControlAdapter,
    HunyuanDiTControlConfig,
    HunyuanDiTControlInjector,
    load_hunyuan_dit_control_checkpoint,
    save_hunyuan_dit_control_checkpoint,
)
from wm3d_v3.training.train import (
    batch_to_device,
    build_datasets,
    build_model,
    load_action_stats_if_available,
)
from wm3d_v3.video_backends.hunyuan_dit_control_video import HunyuanDiTControlVideoBackend
from wm3d_v3.video_backends.hunyuan_video import HunyuanVideoBackendConfig


class _Args(SimpleNamespace):
    pass


def refuse_distributed() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        raise RuntimeError(
            "Hunyuan DiT-control adapter training is intentionally single-process in v1. "
            "Use NPROC_PER_NODE=1; add explicit sequence-parallel/DDP support only after hook semantics are reviewed."
        )


def load_compatible_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> SimpleNamespace:
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


def maybe_subset(ds, max_windows: int, seed: int):
    if max_windows <= 0 or max_windows >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:max_windows].tolist()
    return Subset(ds, idx)


def freeze_module(module: Any) -> None:
    if module is None:
        return
    if hasattr(module, "requires_grad_"):
        module.requires_grad_(False)
    if hasattr(module, "eval"):
        module.eval()


def target_video_from_batch(context_rgb: torch.Tensor, rgb_tgt_p: torch.Tensor) -> torch.Tensor:
    video = torch.cat([context_rgb[:, None], rgb_tgt_p], dim=1)
    return video.permute(0, 2, 1, 3, 4).contiguous()


def rough_video_from_wm_out(context_rgb: torch.Tensor, wm_out: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "rgb" not in wm_out:
        return None
    return torch.cat([context_rgb[:, None], wm_out["rgb"].float()], dim=1).permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def encode_hunyuan_latents(vae, video_bcthw: torch.Tensor) -> torch.Tensor:
    x = video_bcthw.mul(2.0).sub(1.0)
    posterior = vae.encode(x.to(device=next(vae.parameters()).device, dtype=vae.dtype)).latent_dist
    latents = posterior.mode()
    return latents * float(vae.config.scaling_factor)


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


@torch.no_grad()
def encode_hunyuan_prompts(pipeline, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor | None]:
    prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = pipeline.encode_prompt(
        prompts,
        device,
        num_videos_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
        data_type="video",
    )
    if pipeline.text_encoder_2 is not None:
        prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2 = pipeline.encode_prompt(
            prompts,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            text_encoder=pipeline.text_encoder_2,
            data_type="video",
        )
    else:
        prompt_embeds_2 = None
        prompt_mask_2 = None
    return {
        "text_states": prompt_embeds,
        "text_mask": prompt_mask,
        "text_states_2": prompt_embeds_2,
        "text_mask_2": prompt_mask_2,
    }


def rotary_freqs(sampler, *, frames: int, height: int, width: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cos, freqs_sin = sampler.get_rotary_pos_embed(frames, height, width)
    if not isinstance(freqs_cos, torch.Tensor):
        freqs_cos = torch.as_tensor(freqs_cos)
    if not isinstance(freqs_sin, torch.Tensor):
        freqs_sin = torch.as_tensor(freqs_sin)
    return freqs_cos.to(device), freqs_sin.to(device)


def build_hunyuan_backend_args(args: argparse.Namespace) -> HunyuanVideoBackendConfig:
    return HunyuanVideoBackendConfig(
        external_repo=str(args.hunyuan_repo),
        model_base=str(args.hunyuan_model_base),
        dit_weight=str(args.hunyuan_dit_weight),
        model_resolution=args.hunyuan_model_resolution,
        precision=args.hunyuan_precision,
        vae_precision=args.vae_precision,
        text_encoder_precision=args.text_encoder_precision,
        text_encoder_precision_2=args.text_encoder_precision_2,
        use_fp8=bool(args.hunyuan_use_fp8),
        use_cpu_offload=False,
        infer_steps=1,
        cfg_scale=1.0,
        embedded_cfg_scale=float(args.embedded_cfg_scale),
        flow_shift=float(args.flow_shift),
    )


def adapter_target(adapter):
    return adapter.module if hasattr(adapter, "module") else adapter


def controlled_dit_forward(
    *,
    adapter: HunyuanDiTControlAdapter,
    transformer: torch.nn.Module,
    noisy_latents: torch.Tensor,
    sigma: torch.Tensor,
    text: dict[str, torch.Tensor | None],
    freqs: tuple[torch.Tensor, torch.Tensor],
    wm_out: dict[str, torch.Tensor],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    embedded_cfg_scale: float,
    control_scale: float,
) -> torch.Tensor:
    target = adapter_target(adapter)
    injector = HunyuanDiTControlInjector(transformer, target)
    guidance = None
    if bool(getattr(transformer, "guidance_embed", False)):
        guidance = torch.full(
            (noisy_latents.shape[0],),
            float(embedded_cfg_scale) * 1000.0,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
    timestep = (sigma * 1000.0).to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    with injector.use_controls(
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
        scale=float(control_scale),
        latent_shape=tuple(noisy_latents.shape),
    ):
        out = transformer(
            noisy_latents,
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


def forward_batch(
    *,
    adapter: HunyuanDiTControlAdapter,
    wm_model: torch.nn.Module,
    sampler,
    transformer: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    precision: str,
    path_type: str,
    embedded_cfg_scale: float,
    control_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k=0)
    if context_rgb is None or "rgb_tgt_p" not in tgt:
        raise RuntimeError("Hunyuan DiT-control training requires data.load_rgb=true")
    target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
    target_latents = encode_hunyuan_latents(sampler.pipeline.vae, target_video)
    h = int(target_video.shape[-2])
    w = int(target_video.shape[-1])
    frames = int(target_video.shape[2])
    prompts = prompts_from_batch(batch, int(target_latents.shape[0]))
    text = encode_hunyuan_prompts(sampler.pipeline, prompts, device)
    freqs = rotary_freqs(sampler, frames=frames, height=h, width=w, device=device)

    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        wm_out = wm_model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
        rough_video = rough_video_from_wm_out(context_rgb, wm_out)
        rough_latents = encode_hunyuan_latents(sampler.pipeline.vae, rough_video) if rough_video is not None else None
        sigma = torch.rand(target_latents.shape[0], device=device).clamp(1e-4, 1.0)
        if path_type == "noise":
            source_latents = torch.randn_like(target_latents)
        elif path_type == "rough":
            if rough_latents is None:
                raise RuntimeError("--path_type rough requires wm_out['rgb']")
            source_latents = rough_latents.to(dtype=target_latents.dtype)
        else:
            raise ValueError(f"unknown path_type: {path_type}")
        noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
        target_velocity = source_latents - target_latents

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
            task_emb=c,
            embedded_cfg_scale=embedded_cfg_scale,
            control_scale=control_scale,
        )
    mse = F.mse_loss(pred_velocity.float(), target_velocity.float())
    l1 = F.l1_loss(pred_velocity.float(), target_velocity.float())
    return mse, l1, {"velocity_mse": float(mse.detach().cpu()), "velocity_l1": float(l1.detach().cpu())}


@torch.no_grad()
def evaluate(
    *,
    adapter: HunyuanDiTControlAdapter,
    wm_model: torch.nn.Module,
    sampler,
    transformer: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    path_type: str,
    embedded_cfg_scale: float,
    control_scale: float,
    max_batches: int,
) -> dict[str, float]:
    adapter.eval()
    total_mse = 0.0
    total_l1 = 0.0
    count = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        mse, _l1, parts = forward_batch(
            adapter=adapter,
            wm_model=wm_model,
            sampler=sampler,
            transformer=transformer,
            batch=batch,
            device=device,
            precision=precision,
            path_type=path_type,
            embedded_cfg_scale=embedded_cfg_scale,
            control_scale=control_scale,
        )
        total_mse += parts["velocity_mse"]
        total_l1 += parts["velocity_l1"]
        count += 1
    adapter.train()
    return {
        "velocity_mse": total_mse / max(1, count),
        "velocity_l1": total_l1 / max(1, count),
        "batches": float(count),
    }


def main() -> None:
    refuse_distributed()
    ap = argparse.ArgumentParser(description="Train opt-in wm3d -> Hunyuan DiT control adapter.")
    ap.add_argument("--wm_cfg", type=Path, required=True)
    ap.add_argument("--wm_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
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
    ap.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size_per_gpu", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--velocity_mse_weight", type=float, default=1.0)
    ap.add_argument("--velocity_l1_weight", type=float, default=0.05)
    ap.add_argument("--max_train_windows", type=int, default=2000)
    ap.add_argument("--max_val_windows", type=int, default=128)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--control_scale", type=float, default=1.0)
    ap.add_argument("--flow_shift", type=float, default=7.0)
    ap.add_argument("--embedded_cfg_scale", type=float, default=6.0)
    ap.add_argument("--path_type", choices=["noise", "rough"], default="noise")
    ap.add_argument("--control_ckpt", type=Path, default=None, help="Optional DiT-control adapter checkpoint to resume weights from.")
    ap.add_argument("--load_task_text", dest="load_task_text", action="store_true", default=True, help="Feed per-sample task_text to Hunyuan text encoder (default on for text->video).")
    ap.add_argument("--no_load_task_text", dest="load_task_text", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wm_cfg = yaml.safe_load(args.wm_cfg.read_text())
    wm_cfg.setdefault("data", {})["load_task_text"] = bool(args.load_task_text)
    train_ds, val_ds = build_datasets(wm_cfg)
    train_ds = maybe_subset(train_ds, args.max_train_windows, args.seed)
    val_ds = maybe_subset(val_ds, args.max_val_windows, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    wm_model = build_model(wm_cfg).to(device).eval()
    wm_sd = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    load_res = load_compatible_state_dict(wm_model, wm_sd["model"])
    load_action_stats_if_available(wm_model, wm_cfg, 0, device)
    freeze_module(wm_model)

    backend = HunyuanDiTControlVideoBackend(build_hunyuan_backend_args(args), device=device)
    sampler = backend.load()
    transformer = backend.resolve_transformer(sampler)
    freeze_module(getattr(sampler.pipeline, "vae", None))
    freeze_module(getattr(sampler.pipeline, "text_encoder", None))
    freeze_module(getattr(sampler.pipeline, "text_encoder_2", None))
    freeze_module(transformer)

    if args.control_ckpt is not None:
        adapter, payload = load_hunyuan_dit_control_checkpoint(args.control_ckpt, device=device)
        print(f"loaded_control_ckpt path={args.control_ckpt} step={payload.get('step')}", flush=True)
    else:
        adapter = HunyuanDiTControlAdapter(
            HunyuanDiTControlConfig(
                hidden=args.hidden,
                dit_hidden=int(getattr(transformer, "hidden_size", 3072)),
                double_blocks=len(backend._iter_transformer_blocks(transformer, "double_blocks")),
                single_blocks=len(backend._iter_transformer_blocks(transformer, "single_blocks")),
            )
        ).to(device)
    adapter.train()

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "kind": "hunyuan_dit_control_adapter_training_v1",
        "wm_cfg": str(args.wm_cfg),
        "wm_ckpt": str(args.wm_ckpt),
        "wm_ckpt_epoch": wm_sd.get("epoch"),
        "wm_ckpt_val_total": wm_sd.get("val_total"),
        "load_missing": len(load_res.missing_keys),
        "load_skipped": len(load_res.skipped_keys),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "args": vars(args),
        "control_cfg": adapter.cfg.__dict__.copy(),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[dit-control] params={params/1e6:.2f}M train_windows={len(train_ds)} val_windows={len(val_ds)} total_steps={total_steps}", flush=True)

    best = float("inf")
    step = 0
    for epoch in range(args.epochs):
        adapter.train()
        for batch in train_loader:
            # Hunyuan sampler load globally calls torch.set_grad_enabled(False) and never restores it;
            # force grad on for the trainable adapter forward (eval stays under @torch.no_grad()).
            with torch.enable_grad():
                mse, l1, parts = forward_batch(
                    adapter=adapter,
                    wm_model=wm_model,
                    sampler=sampler,
                    transformer=transformer,
                    batch=batch,
                    device=device,
                    precision=args.precision,
                    path_type=args.path_type,
                    embedded_cfg_scale=args.embedded_cfg_scale,
                    control_scale=args.control_scale,
                )
                # build loss INSIDE enable_grad: Hunyuan globally disabled grad, so
                # mul/add outside would detach even though mse/l1 carry grad.
                loss = args.velocity_mse_weight * mse + args.velocity_l1_weight * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if step % args.print_every == 0:
                print(
                    f"[dit-control] step={step} epoch={epoch} loss={float(loss.detach().cpu()):.6f} "
                    f"velocity_mse={parts['velocity_mse']:.6f} velocity_l1={parts['velocity_l1']:.6f} lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            step += 1

        metrics = evaluate(
            adapter=adapter,
            wm_model=wm_model,
            sampler=sampler,
            transformer=transformer,
            loader=val_loader,
            device=device,
            precision=args.precision,
            path_type=args.path_type,
            embedded_cfg_scale=args.embedded_cfg_scale,
            control_scale=args.control_scale,
            max_batches=args.eval_batches,
        )
        score = metrics["velocity_mse"]
        print(f"[dit-control] epoch={epoch} metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
        save_hunyuan_dit_control_checkpoint(
            ckpt_dir / f"epoch_{epoch:03d}.pt",
            adapter,
            metrics=metrics,
            wm_ckpt=args.wm_ckpt,
            step=step,
            extra={"epoch": epoch, "training_args": vars(args)},
        )
        save_hunyuan_dit_control_checkpoint(
            ckpt_dir / "latest.pt",
            adapter,
            metrics=metrics,
            wm_ckpt=args.wm_ckpt,
            step=step,
            extra={"epoch": epoch, "training_args": vars(args)},
        )
        if score < best:
            best = score
            save_hunyuan_dit_control_checkpoint(
                ckpt_dir / "best.pt",
                adapter,
                metrics=metrics,
                wm_ckpt=args.wm_ckpt,
                step=step,
                extra={"epoch": epoch, "training_args": vars(args)},
            )


if __name__ == "__main__":
    main()
