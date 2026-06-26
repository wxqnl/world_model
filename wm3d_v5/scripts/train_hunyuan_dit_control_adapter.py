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
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

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


def setup_distributed() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world, local


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_gradients(module: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    for param in module.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(float(world))


def broadcast_module_state(module: torch.nn.Module, world: int) -> None:
    if world <= 1:
        return
    for param in module.parameters():
        dist.broadcast(param.data, src=0)
    for buf in module.buffers():
        dist.broadcast(buf.data, src=0)


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


def context_video_from_batch(context_rgb: torch.Tensor, frames: int) -> torch.Tensor:
    video = context_rgb[:, None].expand(-1, int(frames), -1, -1, -1)
    return video.permute(0, 2, 1, 3, 4).contiguous()


def rough_video_from_wm_out(context_rgb: torch.Tensor, wm_out: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if "rgb" not in wm_out:
        return None
    return torch.cat([context_rgb[:, None], wm_out["rgb"].float()], dim=1).permute(0, 2, 1, 3, 4).contiguous()


def latent_motion_mask_from_target(
    context_rgb: torch.Tensor,
    target_video: torch.Tensor,
    *,
    latent_shape: tuple[int, int, int, int, int],
    threshold: float,
    dilate: int,
) -> torch.Tensor:
    context_video = context_rgb[:, :, None].expand(-1, -1, int(target_video.shape[2]), -1, -1)
    motion = (target_video.float() - context_video.float()).abs().mean(dim=1, keepdim=True)
    motion = F.interpolate(
        motion,
        size=(int(latent_shape[2]), int(latent_shape[3]), int(latent_shape[4])),
        mode="trilinear",
        align_corners=False,
    )
    threshold = max(1e-6, float(threshold))
    mask = ((motion - threshold) / max(threshold, 1e-6)).clamp(0.0, 1.0)
    dilate = max(0, int(dilate))
    if dilate > 0:
        k = 2 * dilate + 1
        mask = F.max_pool3d(mask, kernel_size=k, stride=1, padding=dilate)
    return mask.clamp(0.0, 1.0)


def weighted_velocity_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    dynamic_weight: float,
    static_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    pred_f = pred.float()
    target_f = target.float()
    if mask is None:
        mse = F.mse_loss(pred_f, target_f)
        l1 = F.l1_loss(pred_f, target_f)
        return mse, l1, {
            "velocity_mse": float(mse.detach().cpu()),
            "velocity_l1": float(l1.detach().cpu()),
        }
    mask_f = mask.to(device=pred_f.device, dtype=pred_f.dtype).clamp(0.0, 1.0)
    while mask_f.ndim < pred_f.ndim:
        mask_f = mask_f.unsqueeze(1)
    mask_f = mask_f.expand_as(pred_f)
    weights = float(static_weight) * (1.0 - mask_f) + float(dynamic_weight) * mask_f
    weights = weights.clamp_min(1e-6)
    sq = (pred_f - target_f).square()
    ab = (pred_f - target_f).abs()
    mse = (sq * weights).sum() / weights.sum().clamp_min(1.0)
    l1 = (ab * weights).sum() / weights.sum().clamp_min(1.0)
    dyn_den = mask_f.sum().clamp_min(1.0)
    sta = (1.0 - mask_f)
    sta_den = sta.sum().clamp_min(1.0)
    dyn_mse = (sq * mask_f).sum() / dyn_den
    sta_mse = (sq * sta).sum() / sta_den
    dyn_l1 = (ab * mask_f).sum() / dyn_den
    sta_l1 = (ab * sta).sum() / sta_den
    return mse, l1, {
        "velocity_mse": float(mse.detach().cpu()),
        "velocity_l1": float(l1.detach().cpu()),
        "velocity_dynamic_mse": float(dyn_mse.detach().cpu()),
        "velocity_static_mse": float(sta_mse.detach().cpu()),
        "velocity_dynamic_l1": float(dyn_l1.detach().cpu()),
        "velocity_static_l1": float(sta_l1.detach().cpu()),
        "latent_motion_mask_mean": float(mask_f.mean().detach().cpu()),
    }


def make_wrong_action(action_cond: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode or "negreverse").strip().lower()
    if mode == "zero":
        return torch.zeros_like(action_cond)
    if mode == "neg":
        return -action_cond
    if mode == "reverse":
        return torch.flip(action_cond, dims=[1]) if action_cond.ndim >= 3 else action_cond
    if mode == "negreverse":
        return -torch.flip(action_cond, dims=[1]) if action_cond.ndim >= 3 else -action_cond
    raise ValueError(f"unsupported wrong_action_mode={mode!r}")


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
    source_latents: torch.Tensor | None = None,
    text: dict[str, torch.Tensor | None] | None = None,
    freqs: tuple[torch.Tensor, torch.Tensor],
    wm_out: dict[str, torch.Tensor],
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    embedded_cfg_scale: float,
    control_scale: float,
    double_pre_control_scale: float = 0.0,
    single_pre_control_scale: float = 0.0,
    latent_control_scale: float = 1.0,
    latent_control_scale_end: float | None = None,
    latent_control_schedule: str = "constant",
    latent_control_motion_gate: float = 0.0,
    latent_control_motion_gate_threshold: float = 0.15,
    latent_control_motion_gate_power: float = 1.0,
) -> torch.Tensor:
    if text is None:
        raise ValueError("text must be provided")
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
        rgb_motion_features=wm_out.get("rgb_motion_features"),
        noisy_latents=noisy_latents,
        source_latents=source_latents,
        sigma=sigma,
        scale=float(control_scale),
        latent_shape=tuple(noisy_latents.shape),
        double_pre_control_scale=float(double_pre_control_scale),
        single_pre_control_scale=float(single_pre_control_scale),
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
        pred_velocity = out["x"] if isinstance(out, dict) else out
        if float(latent_control_scale) != 0.0 or (
            latent_control_scale_end is not None and float(latent_control_scale_end) != 0.0
        ):
            residual = target.latent_residual(noisy_latents)
            start = float(latent_control_scale)
            end = start if latent_control_scale_end is None else float(latent_control_scale_end)
            schedule = str(latent_control_schedule or "constant").strip().lower()
            if schedule == "constant" or residual.shape[2] <= 1 or abs(end - start) < 1e-12:
                scale = residual.new_tensor(start)
            else:
                pos = torch.linspace(0.0, 1.0, int(residual.shape[2]), device=residual.device, dtype=residual.dtype)
                if schedule == "linear":
                    weight = pos
                elif schedule == "quadratic":
                    weight = pos.square()
                elif schedule == "sqrt":
                    weight = pos.clamp_min(0.0).sqrt()
                else:
                    raise ValueError(f"unsupported latent_control_schedule={latent_control_schedule!r}")
                scale = (start + (end - start) * weight).view(1, 1, -1, 1, 1)
            gate_strength = float(latent_control_motion_gate)
            if gate_strength > 0.0 and not torch.is_tensor(scale):
                scale = residual.new_tensor(float(scale))
            gate_source = wm_out.get("motion_hint")
            if gate_source is None and wm_out.get("depth") is not None:
                depth_hint = wm_out["depth"].to(device=residual.device, dtype=residual.dtype)
                if depth_hint.ndim == 4:
                    depth_hint = depth_hint[:, None]
                elif depth_hint.ndim == 5:
                    if depth_hint.shape[1] != 1 and depth_hint.shape[2] == residual.shape[2]:
                        depth_hint = depth_hint.mean(dim=1, keepdim=True)
                    elif depth_hint.shape[2] != residual.shape[2]:
                        depth_hint = depth_hint.mean(dim=2, keepdim=False)[:, None]
                    elif depth_hint.shape[1] != 1:
                        depth_hint = depth_hint.mean(dim=1, keepdim=True)
                else:
                    raise ValueError(f"depth must be 4D or 5D for latent motion gating, got {tuple(depth_hint.shape)}")
                from_first = (depth_hint - depth_hint[:, :, :1]).abs()
                if depth_hint.shape[2] > 1:
                    step = torch.zeros_like(depth_hint)
                    step[:, :, 1:] = (depth_hint[:, :, 1:] - depth_hint[:, :, :-1]).abs()
                    gate_source = 0.5 * from_first + 0.5 * step
                else:
                    gate_source = from_first
            if gate_strength > 0.0 and gate_source is not None:
                hint = gate_source.to(device=residual.device, dtype=residual.dtype).abs()
                if hint.ndim == 4:
                    hint = hint[:, None]
                elif hint.ndim == 5:
                    if hint.shape[1] != 1 and hint.shape[2] == residual.shape[2]:
                        hint = hint.mean(dim=1, keepdim=True)
                    elif hint.shape[2] != residual.shape[2]:
                        hint = hint.mean(dim=2, keepdim=False)[:, None]
                    elif hint.shape[1] != 1:
                        hint = hint.mean(dim=1, keepdim=True)
                else:
                    raise ValueError(f"motion_hint must be 4D or 5D for latent motion gating, got {tuple(hint.shape)}")
                hint = torch.nn.functional.interpolate(
                    hint,
                    size=(int(residual.shape[2]), int(residual.shape[3]), int(residual.shape[4])),
                    mode="trilinear",
                    align_corners=False,
                )
                denom = hint.flatten(2).amax(dim=2).view(hint.shape[0], 1, 1, 1, 1).clamp_min(1e-6)
                mask = (hint / denom - float(latent_control_motion_gate_threshold)).clamp_min(0.0)
                mask = (mask / max(1e-6, 1.0 - float(latent_control_motion_gate_threshold))).clamp(0.0, 1.0)
                power = max(1e-6, float(latent_control_motion_gate_power))
                if abs(power - 1.0) > 1e-6:
                    mask = mask.pow(power)
                mask = mask * min(1.0, max(0.0, gate_strength))
                scale = residual.new_tensor(start) + (scale - residual.new_tensor(start)) * mask
            pred_velocity = pred_velocity + residual * scale
    return pred_velocity


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
    adapter_use_rough: bool,
    double_pre_control_scale: float = 0.0,
    single_pre_control_scale: float = 0.0,
    use_rgb_features: bool = False,
    latent_motion_mask_source: str = "none",
    latent_motion_threshold: float = 0.02,
    latent_motion_dilate: int = 0,
    velocity_dynamic_weight: float = 1.0,
    velocity_static_weight: float = 1.0,
    velocity_target_sign: float = 1.0,
    action_rank_weight: float = 0.0,
    action_rank_margin: float = 0.01,
    action_rank_every: int = 1,
    wrong_action_mode: str = "negreverse",
    step: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
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
        need_wm_pixel = bool(adapter_use_rough) or path_type == "rough" or bool(use_rgb_features)
        wm_out = wm_model(
            s,
            c,
            action_cond=action_cond,
            context_rgb=context_rgb,
            pixel=need_wm_pixel,
            bridging=False,
            return_rgb_features=bool(use_rgb_features),
        )
        rough_latents = None
        if path_type == "rough":
            rough_video = rough_video_from_wm_out(context_rgb, wm_out)
            rough_latents = encode_hunyuan_latents(sampler.pipeline.vae, rough_video) if rough_video is not None else None
        sigma = torch.rand(target_latents.shape[0], device=device).clamp(1e-4, 1.0)
        if path_type == "noise":
            source_latents = torch.randn_like(target_latents)
        elif path_type == "context":
            source_latents = encode_hunyuan_latents(sampler.pipeline.vae, context_video_from_batch(context_rgb, frames))
        elif path_type == "rough":
            if rough_latents is None:
                raise RuntimeError("--path_type rough requires wm_out['rgb']")
            source_latents = rough_latents.to(dtype=target_latents.dtype)
        else:
            raise ValueError(f"unknown path_type: {path_type}")
        noisy = sigma[:, None, None, None, None] * source_latents + (1.0 - sigma)[:, None, None, None, None] * target_latents
        target_velocity = float(velocity_target_sign) * (source_latents - target_latents)
        motion_mask = None
        mask_source = str(latent_motion_mask_source or "none").strip().lower()
        if mask_source == "gt_rgb":
            motion_mask = latent_motion_mask_from_target(
                context_rgb,
                target_video,
                latent_shape=tuple(target_latents.shape),
                threshold=float(latent_motion_threshold),
                dilate=int(latent_motion_dilate),
            )
        elif mask_source not in {"none", "off", ""}:
            raise ValueError(f"unsupported latent_motion_mask_source={latent_motion_mask_source!r}")

    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
        pred_velocity = controlled_dit_forward(
            adapter=adapter,
            transformer=transformer,
            noisy_latents=noisy.to(dtype=autocast_dtype if device.type == "cuda" else noisy.dtype),
            sigma=sigma,
            source_latents=source_latents,
            text=text,
            freqs=freqs,
            wm_out=wm_out,
            context_rgb=context_rgb,
            action_cond=action_cond,
            task_emb=c,
            embedded_cfg_scale=embedded_cfg_scale,
            control_scale=control_scale,
            double_pre_control_scale=double_pre_control_scale,
            single_pre_control_scale=single_pre_control_scale,
        )
        mse, l1, parts = weighted_velocity_losses(
            pred_velocity,
            target_velocity,
            motion_mask,
            dynamic_weight=float(velocity_dynamic_weight),
            static_weight=float(velocity_static_weight),
        )
        rank_loss = pred_velocity.new_zeros(())
        should_rank = (
            float(action_rank_weight) > 0.0
            and int(action_rank_every) > 0
            and int(step) % int(action_rank_every) == 0
        )
        if should_rank:
            wrong_action = make_wrong_action(action_cond, wrong_action_mode)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                need_wm_pixel = bool(adapter_use_rough) or path_type == "rough"
                need_wm_pixel = bool(adapter_use_rough) or path_type == "rough" or bool(use_rgb_features)
                wrong_wm_out = wm_model(
                    s,
                    c,
                    action_cond=wrong_action,
                    context_rgb=context_rgb,
                    pixel=need_wm_pixel,
                    bridging=False,
                    return_rgb_features=bool(use_rgb_features),
                )
            wrong_pred = controlled_dit_forward(
                adapter=adapter,
                transformer=transformer,
                noisy_latents=noisy.to(dtype=autocast_dtype if device.type == "cuda" else noisy.dtype),
                sigma=sigma,
                source_latents=source_latents,
                text=text,
                freqs=freqs,
                wm_out=wrong_wm_out,
                context_rgb=context_rgb,
                action_cond=wrong_action,
                task_emb=c,
                embedded_cfg_scale=embedded_cfg_scale,
                control_scale=control_scale,
                double_pre_control_scale=double_pre_control_scale,
                single_pre_control_scale=single_pre_control_scale,
            )
            true_err, _true_l1, _ = weighted_velocity_losses(
                pred_velocity,
                target_velocity,
                motion_mask,
                dynamic_weight=1.0,
                static_weight=0.05,
            )
            wrong_err, _wrong_l1, _ = weighted_velocity_losses(
                wrong_pred,
                target_velocity,
                motion_mask,
                dynamic_weight=1.0,
                static_weight=0.05,
            )
            rank_loss = F.relu(float(action_rank_margin) + true_err - wrong_err)
            parts.update(
                {
                    "action_rank": float(rank_loss.detach().cpu()),
                    "wrong_velocity_mse": float(wrong_err.detach().cpu()),
                    "true_rank_velocity_mse": float(true_err.detach().cpu()),
                }
            )
        else:
            parts.update({"action_rank": 0.0, "wrong_velocity_mse": 0.0, "true_rank_velocity_mse": 0.0})
        parts["velocity_target_sign"] = float(velocity_target_sign)
    return mse, l1, rank_loss, parts


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
    adapter_use_rough: bool,
    double_pre_control_scale: float = 0.0,
    single_pre_control_scale: float = 0.0,
    use_rgb_features: bool = False,
    latent_motion_mask_source: str = "none",
    latent_motion_threshold: float = 0.02,
    latent_motion_dilate: int = 0,
    velocity_dynamic_weight: float = 1.0,
    velocity_static_weight: float = 1.0,
    velocity_target_sign: float = 1.0,
    action_rank_weight: float = 0.0,
    action_rank_margin: float = 0.01,
    action_rank_every: int = 1,
    wrong_action_mode: str = "negreverse",
) -> dict[str, float]:
    adapter.eval()
    totals: dict[str, float] = {}
    count = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        _mse, _l1, _rank, parts = forward_batch(
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
            adapter_use_rough=adapter_use_rough,
            double_pre_control_scale=double_pre_control_scale,
            single_pre_control_scale=single_pre_control_scale,
            use_rgb_features=use_rgb_features,
            latent_motion_mask_source=latent_motion_mask_source,
            latent_motion_threshold=latent_motion_threshold,
            latent_motion_dilate=latent_motion_dilate,
            velocity_dynamic_weight=velocity_dynamic_weight,
            velocity_static_weight=velocity_static_weight,
            velocity_target_sign=velocity_target_sign,
            action_rank_weight=action_rank_weight,
            action_rank_margin=action_rank_margin,
            action_rank_every=action_rank_every,
            wrong_action_mode=wrong_action_mode,
            step=bi,
        )
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    adapter.train()
    out = {key: value / max(1, count) for key, value in totals.items()}
    out["batches"] = float(count)
    return out


def main() -> None:
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
    ap.add_argument("--max_steps", type=int, default=0, help="Stop after this many optimizer steps; 0 means run all epochs.")
    ap.add_argument("--batch_size_per_gpu", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--velocity_mse_weight", type=float, default=1.0)
    ap.add_argument("--velocity_l1_weight", type=float, default=0.05)
    ap.add_argument("--latent_motion_mask_source", choices=["none", "gt_rgb"], default="none")
    ap.add_argument("--latent_motion_threshold", type=float, default=0.02)
    ap.add_argument("--latent_motion_dilate", type=int, default=0)
    ap.add_argument("--velocity_dynamic_weight", type=float, default=1.0)
    ap.add_argument("--velocity_static_weight", type=float, default=1.0)
    ap.add_argument(
        "--velocity_target_sign",
        type=float,
        default=1.0,
        help="Multiplier on (source_latents - target_latents). Use -1 when matching Hunyuan sampler direction empirically requires target-source velocity.",
    )
    ap.add_argument("--action_rank_weight", type=float, default=0.0)
    ap.add_argument("--action_rank_margin", type=float, default=0.01)
    ap.add_argument("--action_rank_every", type=int, default=1)
    ap.add_argument("--wrong_action_mode", choices=["zero", "neg", "reverse", "negreverse"], default="negreverse")
    ap.add_argument("--max_train_windows", type=int, default=2000)
    ap.add_argument("--max_val_windows", type=int, default=128)
    ap.add_argument("--eval_batches", type=int, default=8)
    ap.add_argument("--eval_every_steps", type=int, default=0, help="Run rank0 validation every N optimizer steps; 0 disables step eval.")
    ap.add_argument("--ckpt_every_steps", type=int, default=0, help="Save rank0 checkpoints every N optimizer steps; 0 disables step checkpoints.")
    ap.add_argument("--keep_last_checkpoints", type=int, default=3)
    ap.add_argument("--milestone_every_steps", type=int, default=1000)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--use_block_action_film", action="store_true")
    ap.add_argument("--block_action_film_scale", type=float, default=1.0)
    ap.add_argument("--block_action_film_hidden", type=int, default=192)
    ap.add_argument("--use_rgb_features", action="store_true", help="Run WM pixel branch only to pass rgb_motion_features into the Hunyuan DiT adapter; rough RGB remains disabled unless --adapter_use_rough is set.")
    ap.add_argument("--rgb_feature_dim", type=int, default=0)
    ap.add_argument("--rgb_feature_gain", type=float, default=1.0)
    ap.add_argument("--control_scale", type=float, default=1.0)
    ap.add_argument("--double_pre_control_scale", type=float, default=0.0)
    ap.add_argument("--single_pre_control_scale", type=float, default=0.0)
    ap.add_argument(
        "--adapter_use_rough",
        action="store_true",
        help="Opt into feeding WM small RGB decoder output to the DiT-control adapter. Default is no rough RGB.",
    )
    ap.add_argument("--flow_shift", type=float, default=7.0)
    ap.add_argument("--embedded_cfg_scale", type=float, default=6.0)
    ap.add_argument("--path_type", choices=["noise", "context", "rough"], default="noise")
    ap.add_argument("--control_ckpt", type=Path, default=None, help="Optional DiT-control adapter checkpoint to resume weights from.")
    ap.add_argument("--load_task_text", dest="load_task_text", action="store_true", default=True, help="Feed per-sample task_text to Hunyuan text encoder (default on for text->video).")
    ap.add_argument("--no_load_task_text", dest="load_task_text", action="store_false")
    args = ap.parse_args()

    rank, world, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wm_cfg = yaml.safe_load(args.wm_cfg.read_text())
    wm_cfg.setdefault("data", {})["load_task_text"] = bool(args.load_task_text)
    train_ds, val_ds = build_datasets(wm_cfg)
    train_ds = maybe_subset(train_ds, args.max_train_windows, args.seed)
    val_ds = maybe_subset(val_ds, args.max_val_windows, args.seed + 1)
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        if world > 1
        else None
    )
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
        num_workers=args.num_workers,
        pin_memory=True,
    )

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
        adapter.cfg.use_block_action_film = bool(args.use_block_action_film)
        adapter.cfg.block_action_film_scale = float(args.block_action_film_scale)
        adapter.cfg.block_action_film_hidden = int(args.block_action_film_hidden)
        adapter.cfg.use_rough = bool(args.adapter_use_rough)
        if bool(args.use_rgb_features):
            adapter.enable_rgb_features(dim=int(args.rgb_feature_dim), gain=float(args.rgb_feature_gain))
        if rank == 0:
            print(f"loaded_control_ckpt path={args.control_ckpt} step={payload.get('step')}", flush=True)
    else:
        adapter = HunyuanDiTControlAdapter(
            HunyuanDiTControlConfig(
                hidden=args.hidden,
                dit_hidden=int(getattr(transformer, "hidden_size", 3072)),
                double_blocks=len(backend._iter_transformer_blocks(transformer, "double_blocks")),
                single_blocks=len(backend._iter_transformer_blocks(transformer, "single_blocks")),
                use_block_action_film=bool(args.use_block_action_film),
                block_action_film_scale=float(args.block_action_film_scale),
                block_action_film_hidden=int(args.block_action_film_hidden),
                use_rgb_features=bool(args.use_rgb_features),
                rgb_feature_dim=int(args.rgb_feature_dim),
                rgb_feature_gain=float(args.rgb_feature_gain),
                use_rough=bool(args.adapter_use_rough),
            )
        ).to(device)
    broadcast_module_state(adapter, world)
    adapter.train()

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    total_available_steps = max(1, len(train_loader) * args.epochs)
    total_steps = min(total_available_steps, int(args.max_steps)) if int(args.max_steps) > 0 else total_available_steps

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ckpt_dir = args.out_dir / "ckpt"
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
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
            "world_size": world,
            "args": vars(args),
            "control_cfg": adapter.cfg.__dict__.copy(),
        }
        (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    if world > 1:
        dist.barrier()
    params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    if rank == 0:
        print(
            f"[dit-control] params={params/1e6:.2f}M train_windows={len(train_ds)} "
            f"val_windows={len(val_ds)} world={world} total_steps={total_steps}",
            flush=True,
        )

    def link_latest(path: Path) -> None:
        latest = ckpt_dir / "latest.pt"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)

    def step_from_ckpt(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    def prune_step_checkpoints() -> None:
        keep_last = max(0, int(args.keep_last_checkpoints))
        milestone_every = max(0, int(args.milestone_every_steps))
        paths = sorted(ckpt_dir.glob("step_*.pt"), key=step_from_ckpt)
        keep = set(paths[-keep_last:]) if keep_last > 0 else set()
        if milestone_every > 0:
            keep.update(path for path in paths if step_from_ckpt(path) > 0 and step_from_ckpt(path) % milestone_every == 0)
        for path in paths:
            if path not in keep:
                path.unlink(missing_ok=True)

    best = float("inf")
    last_saved_step = -1

    def save_step(step_value: int, epoch_value: int, metrics: dict[str, float] | None) -> None:
        nonlocal best, last_saved_step
        path = ckpt_dir / f"step_{step_value:08d}.pt"
        save_hunyuan_dit_control_checkpoint(
            path,
            adapter,
            metrics=metrics,
            wm_ckpt=args.wm_ckpt,
            step=step_value,
            extra={"epoch": epoch_value, "training_args": vars(args)},
        )
        link_latest(path)
        if metrics is not None:
            score = metrics["velocity_mse"]
            if score < best:
                best = score
                save_hunyuan_dit_control_checkpoint(
                    ckpt_dir / "best.pt",
                    adapter,
                    metrics=metrics,
                    wm_ckpt=args.wm_ckpt,
                    step=step_value,
                    extra={"epoch": epoch_value, "training_args": vars(args)},
                )
        prune_step_checkpoints()
        last_saved_step = step_value

    step = 0
    for epoch in range(args.epochs):
        if int(args.max_steps) > 0 and step >= int(args.max_steps):
            break
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        adapter.train()
        for batch in train_loader:
            if int(args.max_steps) > 0 and step >= int(args.max_steps):
                break
            # Hunyuan sampler load globally calls torch.set_grad_enabled(False) and never restores it;
            # force grad on for the trainable adapter forward (eval stays under @torch.no_grad()).
            with torch.enable_grad():
                mse, l1, rank_loss, parts = forward_batch(
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
                    adapter_use_rough=bool(args.adapter_use_rough),
                    double_pre_control_scale=args.double_pre_control_scale,
                    single_pre_control_scale=args.single_pre_control_scale,
                    use_rgb_features=bool(args.use_rgb_features),
                    latent_motion_mask_source=args.latent_motion_mask_source,
                    latent_motion_threshold=args.latent_motion_threshold,
                    latent_motion_dilate=args.latent_motion_dilate,
                    velocity_dynamic_weight=args.velocity_dynamic_weight,
                    velocity_static_weight=args.velocity_static_weight,
                    velocity_target_sign=args.velocity_target_sign,
                    action_rank_weight=args.action_rank_weight,
                    action_rank_margin=args.action_rank_margin,
                    action_rank_every=args.action_rank_every,
                    wrong_action_mode=args.wrong_action_mode,
                    step=step,
                )
                # build loss INSIDE enable_grad: Hunyuan globally disabled grad, so
                # mul/add outside would detach even though mse/l1 carry grad.
                loss = args.velocity_mse_weight * mse + args.velocity_l1_weight * l1 + args.action_rank_weight * rank_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sync_gradients(adapter, world)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            if rank == 0 and step % args.print_every == 0:
                print(
                    f"[dit-control] step={step} epoch={epoch} loss={float(loss.detach().cpu()):.6f} "
                    f"velocity_mse={parts['velocity_mse']:.6f} velocity_l1={parts['velocity_l1']:.6f} "
                    f"dynamic_mse={parts.get('velocity_dynamic_mse', 0.0):.6f} static_mse={parts.get('velocity_static_mse', 0.0):.6f} "
                    f"target_sign={parts.get('velocity_target_sign', 1.0):.1f} "
                    f"action_rank={parts.get('action_rank', 0.0):.6f} wrong_velocity_mse={parts.get('wrong_velocity_mse', 0.0):.6f} "
                    f"mask_mean={parts.get('latent_motion_mask_mean', 0.0):.4f} lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            step += 1

            should_eval = int(args.eval_every_steps) > 0 and step % int(args.eval_every_steps) == 0
            should_ckpt = int(args.ckpt_every_steps) > 0 and step % int(args.ckpt_every_steps) == 0
            if should_eval or should_ckpt:
                if world > 1:
                    dist.barrier()
                metrics = None
                if rank == 0 and should_eval:
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
                        adapter_use_rough=bool(args.adapter_use_rough),
                        double_pre_control_scale=args.double_pre_control_scale,
                        single_pre_control_scale=args.single_pre_control_scale,
                        use_rgb_features=bool(args.use_rgb_features),
                        latent_motion_mask_source=args.latent_motion_mask_source,
                        latent_motion_threshold=args.latent_motion_threshold,
                        latent_motion_dilate=args.latent_motion_dilate,
                        velocity_dynamic_weight=args.velocity_dynamic_weight,
                        velocity_static_weight=args.velocity_static_weight,
                        velocity_target_sign=args.velocity_target_sign,
                        action_rank_weight=args.action_rank_weight,
                        action_rank_margin=args.action_rank_margin,
                        action_rank_every=args.action_rank_every,
                        wrong_action_mode=args.wrong_action_mode,
                    )
                    print(f"[dit-control] step={step} metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
                if rank == 0 and should_ckpt:
                    save_step(step, epoch, metrics)
                if world > 1:
                    dist.barrier()

        if rank == 0:
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
                adapter_use_rough=bool(args.adapter_use_rough),
                double_pre_control_scale=args.double_pre_control_scale,
                single_pre_control_scale=args.single_pre_control_scale,
                use_rgb_features=bool(args.use_rgb_features),
                latent_motion_mask_source=args.latent_motion_mask_source,
                latent_motion_threshold=args.latent_motion_threshold,
                latent_motion_dilate=args.latent_motion_dilate,
                velocity_dynamic_weight=args.velocity_dynamic_weight,
                velocity_static_weight=args.velocity_static_weight,
                velocity_target_sign=args.velocity_target_sign,
                action_rank_weight=args.action_rank_weight,
                action_rank_margin=args.action_rank_margin,
                action_rank_every=args.action_rank_every,
                wrong_action_mode=args.wrong_action_mode,
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
            save_step(step, epoch, metrics)
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
        if world > 1:
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
