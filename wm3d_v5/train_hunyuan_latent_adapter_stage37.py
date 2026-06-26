"""Train wm3d controls -> Hunyuan VAE latent adapter."""
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

from wm3d_v3.losses import _normalize_depth
from wm3d_v3.models.hunyuan_latent_adapter import (
    HunyuanLatentAdapter,
    HunyuanLatentAdapterConfig,
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

    vae, _, _, _ = load_vae(
        "884-16c-hy",
        args.vae_precision,
        device=device,
    )
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


def context_video_from_batch(context_rgb: torch.Tensor, frames: int) -> torch.Tensor:
    video = context_rgb[:, None].expand(-1, int(frames), -1, -1, -1)
    return video.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def encode_hunyuan_latents(vae, video_btchw: torch.Tensor) -> torch.Tensor:
    x = video_btchw.mul(2.0).sub(1.0)
    posterior = vae.encode(x.to(dtype=vae.dtype)).latent_dist
    latents = posterior.mode()
    return latents * float(vae.config.scaling_factor)


@torch.no_grad()
def decode_hunyuan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents / float(vae.config.scaling_factor)
    out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]
    return out.div(2.0).add(0.5).clamp(0.0, 1.0).float()


def decode_hunyuan_latents_grad(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents / float(vae.config.scaling_factor)
    out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]
    return out.div(2.0).add(0.5)


def motion_mask_from_rgb(rgb_tgt_p: torch.Tensor, context_rgb: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    motion = (rgb_tgt_p.float() - context_rgb.float()[:, None]).abs().mean(dim=2, keepdim=True)
    return (motion > threshold).float()


def static_mask_from_rgb(rgb_tgt_p: torch.Tensor, context_rgb: torch.Tensor, threshold: float = 0.025) -> torch.Tensor:
    motion = (rgb_tgt_p.float() - context_rgb.float()[:, None]).abs().mean(dim=2, keepdim=True)
    return (motion < threshold).float()


def _topk_spatial_mask(mask: torch.Tensor, frac: float) -> torch.Tensor:
    frac = float(frac)
    if not (0.0 < frac < 1.0):
        return mask
    flat = mask.flatten(-2)
    k = max(1, int(math.ceil(frac * float(flat.shape[-1]))))
    idx = torch.topk(flat, k=k, dim=-1).indices
    hard = torch.zeros_like(flat).scatter_(-1, idx, 1.0)
    return (flat * hard).reshape_as(mask)


def rgb_scaffold_mask_from_wm_out(
    context_rgb: torch.Tensor,
    wm_out: dict,
    *,
    source: str,
    threshold: float,
    softness: float,
    topk: float,
) -> torch.Tensor:
    rough = wm_out.get("rgb")
    if rough is None:
        raise RuntimeError("RGB scaffold requires wm_out['rgb']; enable --wm_pixel")
    rough_mag = (rough.float() - context_rgb.float()[:, None]).abs().mean(dim=2, keepdim=True)
    rough_prior = _normalize_depth(rough_mag[:, :, 0]).unsqueeze(2).clamp(0.0, 1.0)
    geom_prior = motion_hint_from_wm_out(wm_out)
    if geom_prior is None:
        geom_prior = rough_prior
    else:
        geom_prior = geom_prior.to(device=rough_prior.device, dtype=rough_prior.dtype).clamp(0.0, 1.0)
        if geom_prior.shape[1] != rough_prior.shape[1] or geom_prior.shape[-2:] != rough_prior.shape[-2:]:
            geom_prior = F.interpolate(
                geom_prior.permute(0, 2, 1, 3, 4),
                size=(rough_prior.shape[1], rough_prior.shape[-2], rough_prior.shape[-1]),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 2, 1, 3, 4)
    if source == "rough":
        prior = rough_prior
    elif source == "geometry":
        prior = geom_prior
    elif source == "hybrid":
        prior = torch.sqrt((rough_prior * geom_prior).clamp_min(0.0))
    elif source == "min":
        prior = torch.minimum(rough_prior, geom_prior)
    elif source == "max":
        prior = torch.maximum(rough_prior, geom_prior)
    else:
        raise ValueError(f"unknown rgb_scaffold_mask_source={source!r}")
    if softness > 0:
        mask = torch.sigmoid((prior - float(threshold)) / float(softness))
    else:
        mask = (prior > float(threshold)).to(dtype=prior.dtype)
    return _topk_spatial_mask(mask, topk).clamp(0.0, 1.0)


def apply_rgb_scaffold(
    decoded_bcthw: torch.Tensor,
    context_rgb: torch.Tensor,
    wm_out: dict,
    *,
    scale: float,
    mask_source: str,
    mask_threshold: float,
    mask_softness: float,
    mask_topk: float,
    residual_clip: float,
    mask_override: torch.Tensor | None = None,
    clamp_output: bool,
) -> torch.Tensor:
    if abs(float(scale)) <= 0:
        return decoded_bcthw
    rough = wm_out.get("rgb")
    if rough is None:
        raise RuntimeError("RGB scaffold requires wm_out['rgb']; enable --wm_pixel")
    future = min(int(rough.shape[1]), max(0, int(decoded_bcthw.shape[2]) - 1))
    if future <= 0:
        return decoded_bcthw
    if mask_override is not None:
        mask = mask_override[:, :future].to(device=decoded_bcthw.device, dtype=decoded_bcthw.dtype)
        if mask.shape[-2:] != rough.shape[-2:]:
            mask = F.interpolate(
                mask.permute(0, 2, 1, 3, 4),
                size=(future, rough.shape[-2], rough.shape[-1]),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 2, 1, 3, 4)
        mask = _topk_spatial_mask(mask.clamp(0.0, 1.0), mask_topk)
    else:
        with torch.no_grad():
            mask = rgb_scaffold_mask_from_wm_out(
                context_rgb,
                wm_out,
                source=mask_source,
                threshold=mask_threshold,
                softness=mask_softness,
                topk=mask_topk,
            )[:, :future]
    with torch.no_grad():
        residual = rough[:, :future].float() - context_rgb.float()[:, None]
        clip = float(residual_clip)
        if clip > 0:
            residual = residual.clamp(-clip, clip)
    add = float(scale) * mask.float() * residual.to(device=mask.device)
    out = decoded_bcthw.clone()
    add_bcthw = add.permute(0, 2, 1, 3, 4).to(device=out.device, dtype=out.dtype)
    out[:, :, 1 : 1 + future] = out[:, :, 1 : 1 + future] + add_bcthw
    return out.clamp(0.0, 1.0) if clamp_output else out


def _dilate_video_mask(mask: torch.Tensor, spatial_radius: int, temporal_radius: int) -> torch.Tensor:
    spatial_radius = max(0, int(spatial_radius))
    temporal_radius = max(0, int(temporal_radius))
    if spatial_radius <= 0 and temporal_radius <= 0:
        return mask
    x = mask.permute(0, 2, 1, 3, 4).contiguous()
    x = F.max_pool3d(
        x,
        kernel_size=(2 * temporal_radius + 1, 2 * spatial_radius + 1, 2 * spatial_radius + 1),
        stride=1,
        padding=(temporal_radius, spatial_radius, spatial_radius),
    )
    return x.permute(0, 2, 1, 3, 4).contiguous()


def apply_static_context_composite(
    decoded_bcthw: torch.Tensor,
    context_rgb: torch.Tensor,
    wm_out: dict,
    *,
    scale: float,
    mask_source: str,
    mask_threshold: float,
    mask_softness: float,
    mask_topk: float,
    mask_spatial_dilate: int,
    mask_temporal_dilate: int,
    mask_floor: float,
    mask_override: torch.Tensor | None = None,
    clamp_output: bool,
) -> torch.Tensor:
    if abs(float(scale)) <= 0:
        return decoded_bcthw
    future = max(0, int(decoded_bcthw.shape[2]) - 1)
    if future <= 0:
        return decoded_bcthw
    if mask_override is not None:
        mask = mask_override.to(device=decoded_bcthw.device, dtype=decoded_bcthw.dtype)
        if mask.ndim != 5:
            raise ValueError(f"static composite mask_override must be 5D, got {tuple(mask.shape)}")
        if mask.shape[1] != future or mask.shape[-2:] != decoded_bcthw.shape[-2:]:
            mask = F.interpolate(
                mask.permute(0, 2, 1, 3, 4),
                size=(future, decoded_bcthw.shape[-2], decoded_bcthw.shape[-1]),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 2, 1, 3, 4)
        mask = _topk_spatial_mask(mask.clamp(0.0, 1.0), mask_topk)
    else:
        with torch.no_grad():
            mask = rgb_scaffold_mask_from_wm_out(
                context_rgb,
                wm_out,
                source=mask_source,
                threshold=mask_threshold,
                softness=mask_softness,
                topk=mask_topk,
            )[:, :future]
        if mask.shape[-2:] != decoded_bcthw.shape[-2:]:
            mask = F.interpolate(
                mask.permute(0, 2, 1, 3, 4),
                size=(future, decoded_bcthw.shape[-2], decoded_bcthw.shape[-1]),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 2, 1, 3, 4)
    mask = _dilate_video_mask(mask.clamp(0.0, 1.0), mask_spatial_dilate, mask_temporal_dilate)
    floor = min(max(float(mask_floor), 0.0), 1.0)
    if floor > 0:
        mask = floor + (1.0 - floor) * mask
    ctx = context_rgb.float()
    if ctx.shape[-2:] != decoded_bcthw.shape[-2:]:
        ctx = F.interpolate(ctx, size=decoded_bcthw.shape[-2:], mode="bilinear", align_corners=False)
    pred_future = decoded_bcthw[:, :, 1 : 1 + future].permute(0, 2, 1, 3, 4).float()
    mixed = mask.float() * pred_future + (1.0 - mask.float()) * ctx[:, None]
    scale_f = float(scale)
    if abs(scale_f - 1.0) > 1e-6:
        mixed = scale_f * mixed + (1.0 - scale_f) * pred_future
    out = decoded_bcthw.clone()
    out[:, :, 1 : 1 + future] = mixed.permute(0, 2, 1, 3, 4).to(device=out.device, dtype=out.dtype)
    return out.clamp(0.0, 1.0) if clamp_output else out


def latent_motion_mask_from_video(
    target_video: torch.Tensor,
    context_video: torch.Tensor,
    latent_shape: tuple[int, int, int],
    *,
    threshold: float,
    softness: float,
) -> torch.Tensor:
    motion = (target_video.float() - context_video.float()).abs().mean(dim=1, keepdim=True)
    motion = F.interpolate(motion, size=latent_shape, mode="trilinear", align_corners=False)
    if softness <= 0:
        return (motion > threshold).float()
    return torch.sigmoid((motion - float(threshold)) / float(softness))


def adapter_cfg_from(adapter) -> HunyuanLatentAdapterConfig:
    return adapter.module.cfg if isinstance(adapter, DDP) else adapter.cfg


def needs_context_anchor(adapter) -> bool:
    return getattr(adapter_cfg_from(adapter), "output_mode", "direct") in {
        "context_residual_mask",
        "context_residual_mask_velocity",
        "direct_context_blend",
        "direct_motion_region_blend",
    }


def needs_rough_latent_delta(adapter) -> bool:
    cfg = adapter_cfg_from(adapter)
    return (
        abs(float(getattr(cfg, "rough_latent_delta_scale", 0.0))) > 0.0
        or bool(getattr(cfg, "use_rough_latent_delta_condition", False))
    )


def rough_video_from_wm_out(context_rgb: torch.Tensor, wm_out: dict) -> torch.Tensor | None:
    if "rgb" not in wm_out:
        return None
    return torch.cat([context_rgb[:, None], wm_out["rgb"].float()], dim=1).permute(0, 2, 1, 3, 4).contiguous()


def motion_hint_from_wm_out(wm_out: dict) -> torch.Tensor | None:
    hint = wm_out.get("motion_hint")
    if hint is not None:
        hint = hint.float()
        if hint.ndim == 4:
            hint = hint[:, :, None]
        elif hint.ndim == 5 and hint.shape[-1] == 1:
            hint = hint.permute(0, 1, 4, 2, 3).contiguous()
        elif hint.ndim == 5 and hint.shape[2] != 1:
            hint = hint.mean(dim=2, keepdim=True)
        if hint.ndim == 5 and hint.shape[2] == 1:
            return _normalize_depth(hint[:, :, 0]).unsqueeze(2)
    point = wm_out.get("point")
    if point is not None:
        if point.ndim == 5 and point.shape[-1] == 3:
            point_ch = point.permute(0, 1, 4, 2, 3).contiguous().float()
        elif point.ndim == 5 and point.shape[2] == 3:
            point_ch = point.float()
        else:
            point_ch = None
        if point_ch is not None and point_ch.shape[1] > 0:
            diffs = point_ch.new_zeros(point_ch.shape[0], point_ch.shape[1], 1, point_ch.shape[-2], point_ch.shape[-1])
            if point_ch.shape[1] > 1:
                diffs[:, 1:] = (point_ch[:, 1:] - point_ch[:, :-1]).norm(dim=2, keepdim=True)
                diffs[:, :1] = diffs[:, 1:2]
            return _normalize_depth(diffs[:, :, 0]).unsqueeze(2)
    depth = wm_out.get("depth")
    if depth is None:
        return None
    depth = depth.float()
    diffs = depth.new_zeros(depth.shape)
    if depth.shape[1] > 1:
        diffs[:, 1:] = (depth[:, 1:] - depth[:, :-1]).abs()
        diffs[:, :1] = diffs[:, 1:2]
    return _normalize_depth(diffs).unsqueeze(2)


@torch.no_grad()
def base_latents_from_source(
    vae,
    context_rgb: torch.Tensor,
    target_video: torch.Tensor,
    wm_out: dict,
    *,
    source: str,
    rough_prior_weight: float,
    rough_prior_power: float,
    rough_prior_floor: float,
) -> torch.Tensor:
    context_video = context_video_from_batch(context_rgb, target_video.shape[2])
    context_latents = encode_hunyuan_latents(vae, context_video)
    if source == "context":
        return context_latents
    rough_video = rough_video_from_wm_out(context_rgb, wm_out)
    if rough_video is None:
        raise RuntimeError(f"base_latents_source={source!r} requires wm_out['rgb']; enable --wm_pixel")
    rough_latents = encode_hunyuan_latents(vae, rough_video)
    if source == "rough":
        return rough_latents
    if source != "rough_dynamic":
        raise ValueError(f"unknown base_latents_source={source!r}")
    motion_hint = motion_hint_from_wm_out(wm_out)
    if motion_hint is None:
        future_prior = (rough_video[:, :, 1:] - context_video[:, :, 1:]).abs().mean(dim=1, keepdim=True)
    else:
        future_prior = motion_hint.permute(0, 2, 1, 3, 4).contiguous().float()
    zero = future_prior.new_zeros(future_prior.shape[0], 1, 1, future_prior.shape[-2], future_prior.shape[-1])
    prior_video = torch.cat([zero, future_prior], dim=2)
    prior = F.interpolate(prior_video, size=rough_latents.shape[2:], mode="trilinear", align_corners=False)
    prior = prior.clamp(0.0, 1.0)
    power = max(float(rough_prior_power), 1e-6)
    if abs(power - 1.0) > 1e-6:
        prior = prior.pow(power)
    prior = (float(rough_prior_weight) * prior).clamp(0.0, 1.0)
    floor = min(max(float(rough_prior_floor), 0.0), 1.0)
    if floor > 0:
        prior = floor + (1.0 - floor) * prior
    return context_latents + prior.to(dtype=context_latents.dtype) * (rough_latents - context_latents)


@torch.no_grad()
def rough_latent_delta_from_wm(
    vae,
    context_rgb: torch.Tensor,
    target_video: torch.Tensor,
    wm_out: dict,
    *,
    context_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context_video = context_video_from_batch(context_rgb, target_video.shape[2])
    if context_latents is None:
        context_latents = encode_hunyuan_latents(vae, context_video)
    rough_video = rough_video_from_wm_out(context_rgb, wm_out)
    if rough_video is None:
        raise RuntimeError("rough_latent_delta requires wm_out['rgb']; enable --wm_pixel")
    rough_latents = encode_hunyuan_latents(vae, rough_video)
    motion_hint = motion_hint_from_wm_out(wm_out)
    if motion_hint is None:
        future_prior = (rough_video[:, :, 1:] - context_video[:, :, 1:]).abs().mean(dim=1, keepdim=True)
    else:
        future_prior = motion_hint.permute(0, 2, 1, 3, 4).contiguous().float()
    zero = future_prior.new_zeros(future_prior.shape[0], 1, 1, future_prior.shape[-2], future_prior.shape[-1])
    prior_video = torch.cat([zero, future_prior], dim=2)
    prior = F.interpolate(prior_video, size=rough_latents.shape[2:], mode="trilinear", align_corners=False)
    return rough_latents - context_latents.to(dtype=rough_latents.dtype, device=rough_latents.device), prior.clamp(0.0, 1.0)


def adapter_forward(
    adapter,
    wm_out: dict,
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    c: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    use_rough: bool,
    base_latents: torch.Tensor | None = None,
    rough_latent_delta: torch.Tensor | None = None,
    rough_delta_mask: torch.Tensor | None = None,
    return_components: bool = False,
):
    adapter_cfg = adapter_cfg_from(adapter)
    rough_rgb = wm_out.get("rgb") if (use_rough or bool(getattr(adapter_cfg, "use_rgb_scaffold_mask", False))) else None
    motion_hint = motion_hint_from_wm_out(wm_out) if bool(adapter_cfg.use_motion) else None
    rgb_motion_features = wm_out.get("rgb_motion_features") if bool(getattr(adapter_cfg, "use_rgb_features", False)) else None
    return adapter(
        wm_out["pred_tokens"],
        wm_out["depth"],
        context_rgb=context_rgb,
        motion_hint=motion_hint,
        rough_rgb=rough_rgb,
        rgb_motion_features=rgb_motion_features,
        rough_latent_delta=rough_latent_delta,
        rough_delta_mask=rough_delta_mask,
        action_cond=action_cond,
        task_emb=c,
        target_latents=target_latents,
        base_latents=base_latents,
        return_components=return_components,
    )


def split_adapter_output(output) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(output, dict):
        return output["latents"], output
    return output, {"latents": output}


def combine_with_residual_base(
    delta_latents: torch.Tensor,
    rough_latents: torch.Tensor | None,
    *,
    residual_from_rough: bool,
    residual_scale: float,
) -> torch.Tensor:
    if not residual_from_rough:
        return delta_latents
    if rough_latents is None:
        raise RuntimeError("--residual_from_rough requires wm_out['rgb'] so rough latents can be encoded")
    return rough_latents.to(dtype=delta_latents.dtype) + float(residual_scale) * delta_latents


def save_demo(path: Path, pred_video_bcthw: torch.Tensor, target_video_bcthw: torch.Tensor, rough_btchw: torch.Tensor | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = pred_video_bcthw[0].permute(1, 2, 3, 0).detach().cpu().clamp(0, 1)
    target = target_video_bcthw[0].permute(1, 2, 3, 0).detach().cpu().clamp(0, 1)
    frames = []
    for i in range(pred.shape[0]):
        row = [target[i], pred[i]]
        if rough_btchw is not None:
            if i == 0:
                rough = target[i]
            else:
                rough = rough_btchw[0, i - 1].permute(1, 2, 0).detach().cpu().clamp(0, 1)
            row.append(rough)
        frame = torch.cat(row, dim=1)
        frames.append((frame.numpy() * 255).round().astype("uint8"))
    imageio.mimsave(path, frames, fps=6)


def update_latest_symlink(link: Path, target: Path) -> None:
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.name)
    except OSError:
        pass


def _parse_step_ckpt(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (IndexError, ValueError):
        return -1


def prune_step_checkpoints(ckpt_dir: Path, *, keep_last: int, milestone_every: int) -> None:
    if keep_last <= 0 and milestone_every <= 0:
        return
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=_parse_step_ckpt)
    keep: set[Path] = set(ckpts[-keep_last:]) if keep_last > 0 else set()
    if milestone_every > 0:
        keep.update(path for path in ckpts if _parse_step_ckpt(path) % milestone_every == 0)
    for path in ckpts:
        if path not in keep:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def evaluate(
    *,
    adapter,
    wm_model,
    vae,
    loader,
    device: torch.device,
    max_batches: int,
    precision: str,
    residual_from_rough: bool,
    residual_scale: float,
    wm_pixel: bool,
    adapter_use_rough: bool,
    mask_motion_threshold: float,
    mask_motion_softness: float,
    base_latents_source: str,
    base_rough_prior_weight: float,
    base_rough_prior_power: float,
    base_rough_prior_floor: float,
    rgb_scaffold_scale: float,
    rgb_scaffold_mask_source: str,
    rgb_scaffold_mask_threshold: float,
    rgb_scaffold_mask_softness: float,
    rgb_scaffold_mask_topk: float,
    rgb_scaffold_residual_clip: float,
    static_context_composite_scale: float,
    static_context_mask_source: str,
    static_context_mask_threshold: float,
    static_context_mask_softness: float,
    static_context_mask_topk: float,
    static_context_mask_spatial_dilate: int,
    static_context_mask_temporal_dilate: int,
    static_context_mask_floor: float,
) -> dict[str, float]:
    adapter.eval()
    totals = {
        "latent_mse": 0.0,
        "latent_l1": 0.0,
        "vae_recon_l1": 0.0,
        "decoded_l1": 0.0,
        "rough_l1": 0.0,
        "rough_vae_l1": 0.0,
        "motion_vae_recon_l1": 0.0,
        "motion_decoded_l1": 0.0,
        "motion_rough_l1": 0.0,
        "motion_rough_vae_l1": 0.0,
        "decoded_temporal_l1": 0.0,
        "decoded_from_first_l1": 0.0,
        "latent_static_l1": 0.0,
        "latent_dynamic_l1": 0.0,
        "mask_l1": 0.0,
        "mask_area": 0.0,
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
            context_video = context_video_from_batch(context_rgb, target_video.shape[2])
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                wm_out = wm_model(
                    s,
                    c,
                    action_cond=action_cond,
                    context_rgb=context_rgb,
                    pixel=bool(wm_pixel),
                    bridging=False,
                    return_rgb_features=bool(getattr(adapter_cfg_from(adapter), "use_rgb_features", False)),
                )
                base_latents = (
                    base_latents_from_source(
                        vae,
                        context_rgb,
                        target_video,
                        wm_out,
                        source=base_latents_source,
                        rough_prior_weight=base_rough_prior_weight,
                        rough_prior_power=base_rough_prior_power,
                        rough_prior_floor=base_rough_prior_floor,
                    )
                    if needs_context_anchor(adapter)
                    else None
                )
                rough_latent_delta = None
                rough_delta_mask = None
                if needs_rough_latent_delta(adapter):
                    rough_latent_delta, rough_delta_mask = rough_latent_delta_from_wm(
                        vae,
                        context_rgb,
                        target_video,
                        wm_out,
                        context_latents=base_latents if base_latents_source == "context" else None,
                    )
                adapter_out = adapter_forward(
                    adapter,
                    wm_out,
                    context_rgb,
                    action_cond,
                    c,
                    target_latents,
                    use_rough=bool(adapter_use_rough),
                    base_latents=base_latents,
                    rough_latent_delta=rough_latent_delta,
                    rough_delta_mask=rough_delta_mask,
                    return_components=True,
                )
                delta_latents, components = split_adapter_output(adapter_out)
            rough_video = rough_video_from_wm_out(context_rgb, wm_out)
            rough_latents = encode_hunyuan_latents(vae, rough_video) if residual_from_rough and rough_video is not None else None
            pred_latents = combine_with_residual_base(
                delta_latents,
                rough_latents,
                residual_from_rough=residual_from_rough,
                residual_scale=residual_scale,
            )
            decoded = decode_hunyuan_latents(vae, pred_latents.float())
            decoded = apply_rgb_scaffold(
                decoded,
                context_rgb,
                wm_out,
                scale=rgb_scaffold_scale,
                mask_source=rgb_scaffold_mask_source,
                mask_threshold=rgb_scaffold_mask_threshold,
                mask_softness=rgb_scaffold_mask_softness,
                mask_topk=rgb_scaffold_mask_topk,
                residual_clip=rgb_scaffold_residual_clip,
                mask_override=components.get("rgb_scaffold_mask"),
                clamp_output=True,
            )
            decoded = apply_static_context_composite(
                decoded,
                context_rgb,
                wm_out,
                scale=static_context_composite_scale,
                mask_source="geometry" if static_context_mask_source == "learned" else static_context_mask_source,
                mask_threshold=static_context_mask_threshold,
                mask_softness=static_context_mask_softness,
                mask_topk=static_context_mask_topk,
                mask_spatial_dilate=static_context_mask_spatial_dilate,
                mask_temporal_dilate=static_context_mask_temporal_dilate,
                mask_floor=static_context_mask_floor,
                mask_override=components.get("rgb_scaffold_mask")
                if static_context_mask_source == "learned"
                else components.get("motion_region_prior"),
                clamp_output=True,
            )
            vae_recon = decode_hunyuan_latents(vae, target_latents.float())
            rough_vae = decode_hunyuan_latents(vae, rough_latents.float()) if rough_latents is not None else None
            target_f = target_video.float()
            n = target_f.shape[0]
            totals["latent_mse"] += float(F.mse_loss(pred_latents.float(), target_latents.float(), reduction="sum").cpu())
            totals["latent_l1"] += float(F.l1_loss(pred_latents.float(), target_latents.float(), reduction="sum").cpu())
            latent_mask = latent_motion_mask_from_video(
                target_video,
                context_video,
                tuple(target_latents.shape[2:]),
                threshold=mask_motion_threshold,
                softness=mask_motion_softness,
            ).to(pred_latents.device)
            latent_static = 1.0 - latent_mask
            latent_denom = (latent_mask.sum() * target_latents.shape[1]).clamp_min(1.0)
            static_denom = (latent_static.sum() * target_latents.shape[1]).clamp_min(1.0)
            totals["latent_dynamic_l1"] += float(((pred_latents.float() - target_latents.float()).abs() * latent_mask).sum().cpu() / latent_denom.cpu())
            if base_latents is not None:
                totals["latent_static_l1"] += float(((pred_latents.float() - base_latents.float()).abs() * latent_static).sum().cpu() / static_denom.cpu())
            if "mask" in components:
                pred_mask = components["mask"].float()
                totals["mask_l1"] += float(F.l1_loss(pred_mask, latent_mask.float(), reduction="mean").cpu()) * n
                totals["mask_area"] += float(pred_mask.mean().detach().cpu()) * n
            totals["vae_recon_l1"] += float((vae_recon - target_f).abs().sum().cpu())
            totals["decoded_l1"] += float((decoded - target_f).abs().sum().cpu())
            totals["decoded_temporal_l1"] += float(
                ((decoded[:, :, 1:] - decoded[:, :, :-1]) - (target_f[:, :, 1:] - target_f[:, :, :-1])).abs().sum().cpu()
            )
            totals["decoded_from_first_l1"] += float(
                ((decoded[:, :, 1:] - decoded[:, :, :1]) - (target_f[:, :, 1:] - target_f[:, :, :1])).abs().sum().cpu()
            )
            if rough_video is not None:
                totals["rough_l1"] += float((rough_video - target_f).abs().sum().cpu())
            if rough_vae is not None:
                totals["rough_vae_l1"] += float((rough_vae - target_f).abs().sum().cpu())
            motion_mask = motion_mask_from_rgb(tgt["rgb_tgt_p"], context_rgb).permute(0, 2, 1, 3, 4)
            motion_mask = torch.cat([torch.zeros_like(motion_mask[:, :, :1]), motion_mask], dim=2)
            denom = (motion_mask.sum() * target_f.shape[1]).clamp_min(1.0)
            totals["motion_vae_recon_l1"] += float(((vae_recon - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            totals["motion_decoded_l1"] += float(((decoded - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            if rough_video is not None:
                totals["motion_rough_l1"] += float(((rough_video - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            if rough_vae is not None:
                totals["motion_rough_vae_l1"] += float(((rough_vae - target_f).abs() * motion_mask).sum().cpu() / denom.cpu())
            count += n

    latent_numel = count * adapter.module.cfg.latent_channels * 3 * 32 * 32 if isinstance(adapter, DDP) else count * adapter.cfg.latent_channels * 3 * 32 * 32
    pixel_numel = count * 3 * 9 * 256 * 256
    temporal_numel = count * 3 * 8 * 256 * 256
    metrics = {
        "latent_mse": totals["latent_mse"] / max(1, latent_numel),
        "latent_l1": totals["latent_l1"] / max(1, latent_numel),
        "vae_recon_l1": totals["vae_recon_l1"] / max(1, pixel_numel),
        "decoded_l1": totals["decoded_l1"] / max(1, pixel_numel),
        "rough_l1": totals["rough_l1"] / max(1, pixel_numel),
        "rough_vae_l1": totals["rough_vae_l1"] / max(1, pixel_numel),
        "motion_vae_recon_l1": totals["motion_vae_recon_l1"] / max(1, count),
        "motion_decoded_l1": totals["motion_decoded_l1"] / max(1, count),
        "motion_rough_l1": totals["motion_rough_l1"] / max(1, count),
        "motion_rough_vae_l1": totals["motion_rough_vae_l1"] / max(1, count),
        "decoded_temporal_l1": totals["decoded_temporal_l1"] / max(1, temporal_numel),
        "decoded_from_first_l1": totals["decoded_from_first_l1"] / max(1, temporal_numel),
        "latent_static_l1": totals["latent_static_l1"] / max(1, count),
        "latent_dynamic_l1": totals["latent_dynamic_l1"] / max(1, count),
        "mask_l1": totals["mask_l1"] / max(1, count),
        "mask_area": totals["mask_area"] / max(1, count),
        "count": float(count),
    }
    if dist.is_available() and dist.is_initialized():
        keys = sorted(metrics)
        tensor = torch.tensor([metrics[k] for k in keys], device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
        metrics = {k: float(v) for k, v in zip(keys, tensor.detach().cpu().tolist())}
    adapter.train()
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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--batch_size_per_gpu", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--latent_mse_weight", type=float, default=1.0)
    ap.add_argument("--latent_l1_weight", type=float, default=0.05)
    ap.add_argument("--latent_dynamic_l1_weight", type=float, default=0.0)
    ap.add_argument("--latent_static_l1_weight", type=float, default=0.0)
    ap.add_argument("--mask_l1_weight", type=float, default=0.0)
    ap.add_argument("--mask_bce_weight", type=float, default=0.0)
    ap.add_argument("--mask_area_weight", type=float, default=0.0)
    ap.add_argument("--mask_tv_weight", type=float, default=0.0)
    ap.add_argument("--residual_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_motion_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_temporal_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_motion_mag_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_motion_mag_threshold", type=float, default=0.005)
    ap.add_argument("--decoded_from_first_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_static_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_static_threshold", type=float, default=0.025)
    ap.add_argument("--decoded_raw_motion_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_raw_temporal_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_raw_motion_mag_l1_weight", type=float, default=0.0)
    ap.add_argument("--decoded_raw_from_first_l1_weight", type=float, default=0.0)
    ap.add_argument("--rgb_scaffold_scale", type=float, default=0.0)
    ap.add_argument("--rgb_scaffold_mask_source", choices=["rough", "geometry", "hybrid", "min", "max"], default="hybrid")
    ap.add_argument("--rgb_scaffold_mask_threshold", type=float, default=0.35)
    ap.add_argument("--rgb_scaffold_mask_softness", type=float, default=0.10)
    ap.add_argument("--rgb_scaffold_mask_topk", type=float, default=0.0)
    ap.add_argument("--rgb_scaffold_residual_clip", type=float, default=0.35)
    ap.add_argument("--rgb_scaffold_mask_mode", choices=["heuristic", "learned"], default="heuristic")
    ap.add_argument("--rgb_scaffold_mask_hidden", type=int, default=64)
    ap.add_argument("--rgb_scaffold_mask_bias_init", type=float, default=-4.0)
    ap.add_argument("--rgb_scaffold_mask_l1_weight", type=float, default=0.0)
    ap.add_argument("--rgb_scaffold_mask_bce_weight", type=float, default=0.0)
    ap.add_argument("--rgb_scaffold_mask_area_weight", type=float, default=0.0)
    ap.add_argument("--static_context_composite_scale", type=float, default=0.0)
    ap.add_argument("--static_context_mask_source", choices=["rough", "geometry", "hybrid", "min", "max", "learned"], default="geometry")
    ap.add_argument("--static_context_mask_threshold", type=float, default=0.20)
    ap.add_argument("--static_context_mask_softness", type=float, default=0.08)
    ap.add_argument("--static_context_mask_topk", type=float, default=0.0)
    ap.add_argument("--static_context_mask_spatial_dilate", type=int, default=3)
    ap.add_argument("--static_context_mask_temporal_dilate", type=int, default=1)
    ap.add_argument("--static_context_mask_floor", type=float, default=0.02)
    ap.add_argument("--max_train_windows", type=int, default=20000)
    ap.add_argument("--max_val_windows", type=int, default=800)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--eval_every_steps", type=int, default=0)
    ap.add_argument("--ckpt_every_steps", type=int, default=0)
    ap.add_argument("--keep_last_checkpoints", type=int, default=2)
    ap.add_argument("--milestone_every_steps", type=int, default=4000)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ddp_find_unused_parameters", action="store_true")
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--n_blocks", type=int, default=4)
    ap.add_argument(
        "--output_mode",
        choices=[
            "direct",
            "context_residual_mask",
            "context_residual_mask_velocity",
            "direct_context_blend",
            "direct_motion_region_blend",
        ],
        default="direct",
    )
    ap.add_argument("--mask_motion_threshold", type=float, default=0.025)
    ap.add_argument("--mask_motion_softness", type=float, default=0.012)
    ap.add_argument("--mask_bias_init", type=float, default=-2.0)
    ap.add_argument("--mask_temperature", type=float, default=1.0)
    ap.add_argument("--mask_min", type=float, default=0.0)
    ap.add_argument("--mask_max", type=float, default=1.0)
    ap.add_argument("--motion_gain_init", type=float, default=1.0)
    ap.add_argument("--motion_mask_prior_weight", type=float, default=0.0)
    ap.add_argument("--motion_residual_boost", type=float, default=0.0)
    ap.add_argument("--velocity_scale", type=float, default=1.0)
    ap.add_argument("--velocity_blocks", type=int, default=0)
    ap.add_argument("--velocity_motion_prior_weight", type=float, default=0.0)
    ap.add_argument("--velocity_motion_prior_power", type=float, default=1.0)
    ap.add_argument("--velocity_mask_floor", type=float, default=0.0)
    ap.add_argument("--temporal_resampler", choices=["interp", "learned"], default="interp")
    ap.add_argument("--temporal_resampler_horizon", type=int, default=8)
    ap.add_argument("--temporal_resampler_sigma", type=float, default=1.15)
    ap.add_argument("--temporal_resampler_temperature", type=float, default=1.0)
    ap.add_argument("--train_velocity_only", action="store_true")
    ap.add_argument("--residual_from_rough", action="store_true")
    ap.add_argument("--residual_scale", type=float, default=1.0)
    ap.add_argument("--base_latents_source", choices=["context", "rough", "rough_dynamic"], default="context")
    ap.add_argument("--base_rough_prior_weight", type=float, default=1.0)
    ap.add_argument("--base_rough_prior_power", type=float, default=0.5)
    ap.add_argument("--base_rough_prior_floor", type=float, default=0.0)
    ap.add_argument("--rough_latent_delta_scale", type=float, default=0.0)
    ap.add_argument("--rough_latent_delta_mask_source", choices=["none", "prior", "mask", "max", "min"], default="prior")
    ap.add_argument("--rough_latent_delta_mask_power", type=float, default=1.0)
    ap.add_argument("--rough_latent_delta_mask_floor", type=float, default=0.0)
    ap.add_argument("--rough_latent_delta_mask_topk", type=float, default=0.0)
    ap.add_argument("--rough_latent_delta_condition", action="store_true")
    ap.add_argument("--adapter_init_ckpt", type=Path, default=None)
    ap.add_argument("--wm_pixel", dest="wm_pixel", action="store_true")
    ap.add_argument("--no_wm_pixel", dest="wm_pixel", action="store_false")
    ap.set_defaults(wm_pixel=False)
    ap.add_argument("--adapter_use_rough", dest="adapter_use_rough", action="store_true")
    ap.add_argument("--no_adapter_use_rough", dest="adapter_use_rough", action="store_false")
    ap.set_defaults(adapter_use_rough=False)
    ap.add_argument("--adapter_use_motion", dest="adapter_use_motion", action="store_true")
    ap.add_argument("--no_adapter_use_motion", dest="adapter_use_motion", action="store_false")
    ap.set_defaults(adapter_use_motion=False)
    ap.add_argument("--adapter_use_rgb_features", dest="adapter_use_rgb_features", action="store_true")
    ap.add_argument("--no_adapter_use_rgb_features", dest="adapter_use_rgb_features", action="store_false")
    ap.set_defaults(adapter_use_rgb_features=False)
    ap.add_argument("--adapter_rgb_feature_dim", type=int, default=0)
    ap.add_argument("--adapter_rgb_feature_gain", type=float, default=1.0)
    ap.add_argument("--adapter_use_temporal_memory", dest="adapter_use_temporal_memory", action="store_true")
    ap.add_argument("--no_adapter_use_temporal_memory", dest="adapter_use_temporal_memory", action="store_false")
    ap.set_defaults(adapter_use_temporal_memory=False)
    ap.add_argument("--adapter_temporal_memory_heads", type=int, default=4)
    ap.add_argument("--adapter_temporal_memory_layers", type=int, default=1)
    ap.add_argument("--adapter_temporal_memory_mlp_mult", type=float, default=2.0)
    ap.add_argument("--adapter_temporal_memory_gate_init", type=float, default=0.35)
    ap.add_argument("--motion_region_threshold", type=float, default=0.20)
    ap.add_argument("--motion_region_softness", type=float, default=0.08)
    ap.add_argument("--motion_region_power", type=float, default=1.0)
    ap.add_argument("--motion_region_dilate", type=int, default=1)
    ap.add_argument("--motion_region_temporal_dilate", type=int, default=1)
    ap.add_argument("--motion_region_topk", type=float, default=0.0)
    ap.add_argument("--motion_region_floor", type=float, default=0.02)
    ap.add_argument("--motion_region_prior_weight", type=float, default=0.75)
    ap.add_argument("--motion_region_bg_ceiling", type=float, default=0.06)
    args = ap.parse_args()
    if args.adapter_use_rough and not args.wm_pixel:
        raise ValueError("--adapter_use_rough requires --wm_pixel so wm_out['rgb'] exists")
    if args.residual_from_rough and (not args.wm_pixel or not args.adapter_use_rough):
        raise ValueError("--residual_from_rough requires --wm_pixel and --adapter_use_rough")
    if args.base_latents_source != "context" and not args.wm_pixel:
        raise ValueError("--base_latents_source rough/rough_dynamic requires --wm_pixel")
    if (abs(float(args.rough_latent_delta_scale)) > 0 or args.rough_latent_delta_condition) and not args.wm_pixel:
        raise ValueError("rough latent delta scaffold requires --wm_pixel")
    if abs(float(args.rgb_scaffold_scale)) > 0 and not args.wm_pixel:
        raise ValueError("--rgb_scaffold_scale requires --wm_pixel")
    if abs(float(args.static_context_composite_scale)) > 0 and not args.wm_pixel:
        raise ValueError("--static_context_composite_scale requires --wm_pixel")
    if args.adapter_use_rgb_features and not args.wm_pixel:
        raise ValueError("--adapter_use_rgb_features requires --wm_pixel")
    if args.adapter_use_rgb_features and int(args.adapter_rgb_feature_dim) <= 0:
        raise ValueError("--adapter_use_rgb_features requires --adapter_rgb_feature_dim > 0")

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
    adapter_cfg = HunyuanLatentAdapterConfig(
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        use_motion=bool(args.adapter_use_motion),
        use_rough_rgb=bool(args.adapter_use_rough),
        motion_gain_init=float(args.motion_gain_init),
        output_mode=str(args.output_mode),
        residual_scale=float(args.residual_scale),
        mask_bias_init=float(args.mask_bias_init),
        mask_temperature=float(args.mask_temperature),
        mask_min=float(args.mask_min),
        mask_max=float(args.mask_max),
        motion_mask_prior_weight=float(args.motion_mask_prior_weight),
        motion_residual_boost=float(args.motion_residual_boost),
        velocity_scale=float(args.velocity_scale),
        velocity_blocks=int(args.velocity_blocks),
        velocity_motion_prior_weight=float(args.velocity_motion_prior_weight),
        velocity_motion_prior_power=float(args.velocity_motion_prior_power),
        velocity_mask_floor=float(args.velocity_mask_floor),
        temporal_resampler=str(args.temporal_resampler),
        temporal_resampler_horizon=int(args.temporal_resampler_horizon),
        temporal_resampler_sigma=float(args.temporal_resampler_sigma),
        temporal_resampler_temperature=float(args.temporal_resampler_temperature),
        rough_latent_delta_scale=float(args.rough_latent_delta_scale),
        rough_latent_delta_mask_source=str(args.rough_latent_delta_mask_source),
        rough_latent_delta_mask_power=float(args.rough_latent_delta_mask_power),
        rough_latent_delta_mask_floor=float(args.rough_latent_delta_mask_floor),
        rough_latent_delta_mask_topk=float(args.rough_latent_delta_mask_topk),
        use_rough_latent_delta_condition=bool(args.rough_latent_delta_condition),
        use_rgb_scaffold_mask=(str(args.rgb_scaffold_mask_mode) == "learned" or str(args.static_context_mask_source) == "learned"),
        rgb_scaffold_mask_hidden=int(args.rgb_scaffold_mask_hidden),
        rgb_scaffold_mask_bias_init=float(args.rgb_scaffold_mask_bias_init),
        use_rgb_features=bool(args.adapter_use_rgb_features),
        rgb_feature_dim=int(args.adapter_rgb_feature_dim),
        rgb_feature_gain=float(args.adapter_rgb_feature_gain),
        use_temporal_memory=bool(args.adapter_use_temporal_memory),
        temporal_memory_heads=int(args.adapter_temporal_memory_heads),
        temporal_memory_layers=int(args.adapter_temporal_memory_layers),
        temporal_memory_mlp_mult=float(args.adapter_temporal_memory_mlp_mult),
        temporal_memory_gate_init=float(args.adapter_temporal_memory_gate_init),
        motion_region_threshold=float(args.motion_region_threshold),
        motion_region_softness=float(args.motion_region_softness),
        motion_region_power=float(args.motion_region_power),
        motion_region_dilate=int(args.motion_region_dilate),
        motion_region_temporal_dilate=int(args.motion_region_temporal_dilate),
        motion_region_topk=float(args.motion_region_topk),
        motion_region_floor=float(args.motion_region_floor),
        motion_region_prior_weight=float(args.motion_region_prior_weight),
        motion_region_bg_ceiling=float(args.motion_region_bg_ceiling),
    )
    adapter = HunyuanLatentAdapter(adapter_cfg).to(device)
    if not adapter_cfg.use_rough_rgb:
        for p in adapter.rough_proj.parameters():
            p.requires_grad_(False)
    if not adapter_cfg.use_rough_latent_delta_condition:
        for p in adapter.rough_latent_proj.parameters():
            p.requires_grad_(False)
    if not adapter_cfg.use_rgb_scaffold_mask:
        for p in adapter.rgb_scaffold_mask_head.parameters():
            p.requires_grad_(False)
    if not adapter_cfg.use_motion:
        for p in adapter.motion_proj.parameters():
            p.requires_grad_(False)
        adapter.motion_gain.requires_grad_(False)
    if not adapter_cfg.use_context:
        for p in adapter.context_proj.parameters():
            p.requires_grad_(False)
    if not adapter_cfg.use_action:
        for p in adapter.action_proj.parameters():
            p.requires_grad_(False)
    if not adapter_cfg.use_task:
        for p in adapter.task_proj.parameters():
            p.requires_grad_(False)
    if adapter_cfg.output_mode == "direct":
        for p in adapter.mask_out.parameters():
            p.requires_grad_(False)
    if adapter_cfg.output_mode != "context_residual_mask_velocity":
        for p in adapter.velocity_out.parameters():
            p.requires_grad_(False)
        for p in adapter.velocity_blocks.parameters():
            p.requires_grad_(False)
    if adapter_cfg.temporal_resampler != "learned":
        adapter.temporal_logits.requires_grad_(False)
    if not adapter_cfg.use_temporal_memory:
        adapter.temporal_memory_query.requires_grad_(False)
        adapter.temporal_memory_gate_logit.requires_grad_(False)
        for p in adapter.temporal_memory_layers.parameters():
            p.requires_grad_(False)
        for p in adapter.temporal_memory_refine.parameters():
            p.requires_grad_(False)
    if args.train_velocity_only:
        if adapter_cfg.output_mode != "context_residual_mask_velocity":
            raise ValueError("--train_velocity_only requires --output_mode context_residual_mask_velocity")
        for name, p in adapter.named_parameters():
            p.requires_grad_(name.startswith("velocity_out.") or name.startswith("velocity_blocks."))
    if args.residual_from_rough:
        adapter.zero_init_output()
    if args.adapter_init_ckpt is not None:
        adapter_payload = torch.load(args.adapter_init_ckpt, map_location="cpu", weights_only=False)
        adapter_state = adapter_payload["model"] if isinstance(adapter_payload, dict) and "model" in adapter_payload else adapter_payload
        adapter_load = adapter.load_state_dict(adapter_state, strict=False)
        if rank == 0:
            print(
                f"[rank0] loaded adapter_init_ckpt={args.adapter_init_ckpt} "
                f"missing={len(adapter_load.missing_keys)} unexpected={len(adapter_load.unexpected_keys)}",
                flush=True,
            )
    if world > 1:
        adapter = DDP(adapter, device_ids=[local], find_unused_parameters=bool(args.ddp_find_unused_parameters))

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    target_steps = int(args.max_steps) if args.max_steps and args.max_steps > 0 else len(train_loader) * args.epochs
    total_steps = max(1, target_steps)

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
        params = sum(p.numel() for p in (adapter.module if isinstance(adapter, DDP) else adapter).parameters() if p.requires_grad)
        print(f"[rank0] HunyuanLatentAdapter: {params/1e6:.2f}M train_windows={len(train_ds)} val_windows={len(val_ds)} total_steps={total_steps}", flush=True)

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    best = float("inf")
    step = 0

    def save_checkpoint(tag: str, *, epoch: int, step_value: int, metrics: dict[str, float] | None = None) -> Path | None:
        if rank != 0:
            return None
        target = adapter.module if isinstance(adapter, DDP) else adapter
        scaffold_cfg = {
            "rgb_scaffold_scale": float(args.rgb_scaffold_scale),
            "rgb_scaffold_mask_source": str(args.rgb_scaffold_mask_source),
            "rgb_scaffold_mask_threshold": float(args.rgb_scaffold_mask_threshold),
            "rgb_scaffold_mask_softness": float(args.rgb_scaffold_mask_softness),
            "rgb_scaffold_mask_topk": float(args.rgb_scaffold_mask_topk),
            "rgb_scaffold_residual_clip": float(args.rgb_scaffold_residual_clip),
            "rgb_scaffold_mask_mode": str(args.rgb_scaffold_mask_mode),
            "static_context_composite_scale": float(args.static_context_composite_scale),
            "static_context_mask_source": str(args.static_context_mask_source),
            "static_context_mask_threshold": float(args.static_context_mask_threshold),
            "static_context_mask_softness": float(args.static_context_mask_softness),
            "static_context_mask_topk": float(args.static_context_mask_topk),
            "static_context_mask_spatial_dilate": int(args.static_context_mask_spatial_dilate),
            "static_context_mask_temporal_dilate": int(args.static_context_mask_temporal_dilate),
            "static_context_mask_floor": float(args.static_context_mask_floor),
        }
        ckpt = {
            "epoch": epoch,
            "step": step_value,
            "model": target.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "metrics": metrics,
            "cfg": adapter_cfg.__dict__,
            "scaffold_cfg": scaffold_cfg,
        }
        path = ckpt_dir / f"{tag}.pt"
        torch.save(ckpt, path)
        if tag.startswith("step_"):
            update_latest_symlink(ckpt_dir / "latest.pt", path)
            prune_step_checkpoints(
                ckpt_dir,
                keep_last=args.keep_last_checkpoints,
                milestone_every=args.milestone_every_steps,
            )
        return path

    def run_eval(epoch: int, step_value: int) -> dict[str, float]:
        nonlocal best
        metrics = evaluate(
            adapter=adapter,
            wm_model=wm_model,
            vae=vae,
            loader=val_loader,
            device=device,
            max_batches=args.eval_batches,
            precision=args.precision,
            residual_from_rough=args.residual_from_rough,
            residual_scale=args.residual_scale,
            wm_pixel=args.wm_pixel,
            adapter_use_rough=args.adapter_use_rough,
            mask_motion_threshold=args.mask_motion_threshold,
            mask_motion_softness=args.mask_motion_softness,
            base_latents_source=args.base_latents_source,
            base_rough_prior_weight=args.base_rough_prior_weight,
            base_rough_prior_power=args.base_rough_prior_power,
            base_rough_prior_floor=args.base_rough_prior_floor,
            rgb_scaffold_scale=args.rgb_scaffold_scale,
            rgb_scaffold_mask_source=args.rgb_scaffold_mask_source,
            rgb_scaffold_mask_threshold=args.rgb_scaffold_mask_threshold,
            rgb_scaffold_mask_softness=args.rgb_scaffold_mask_softness,
            rgb_scaffold_mask_topk=args.rgb_scaffold_mask_topk,
            rgb_scaffold_residual_clip=args.rgb_scaffold_residual_clip,
            static_context_composite_scale=args.static_context_composite_scale,
            static_context_mask_source=args.static_context_mask_source,
            static_context_mask_threshold=args.static_context_mask_threshold,
            static_context_mask_softness=args.static_context_mask_softness,
            static_context_mask_topk=args.static_context_mask_topk,
            static_context_mask_spatial_dilate=args.static_context_mask_spatial_dilate,
            static_context_mask_temporal_dilate=args.static_context_mask_temporal_dilate,
            static_context_mask_floor=args.static_context_mask_floor,
        )
        score = metrics["decoded_l1"]
        if rank == 0:
            print(f"[rank0] eval step {step_value} ep {epoch}: {json.dumps(metrics, sort_keys=True)}", flush=True)
            save_checkpoint(f"step_{step_value:08d}", epoch=epoch, step_value=step_value, metrics=metrics)
            if score < best:
                best = score
                save_checkpoint("best", epoch=epoch, step_value=step_value, metrics=metrics)
                try:
                    demo_batch = next(iter(val_loader))
                    s, c, action_cond, context_rgb, tgt = batch_to_device(demo_batch, device, k=0)
                    target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
                    target_latents = encode_hunyuan_latents(vae, target_video)
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                        wm_out = wm_model(
                            s,
                            c,
                            action_cond=action_cond,
                            context_rgb=context_rgb,
                            pixel=bool(args.wm_pixel),
                            bridging=False,
                            return_rgb_features=bool(getattr(adapter_cfg_from(adapter), "use_rgb_features", False)),
                        )
                    base_latents = (
                        base_latents_from_source(
                            vae,
                            context_rgb,
                            target_video,
                            wm_out,
                            source=args.base_latents_source,
                            rough_prior_weight=args.base_rough_prior_weight,
                            rough_prior_power=args.base_rough_prior_power,
                            rough_prior_floor=args.base_rough_prior_floor,
                        )
	                        if needs_context_anchor(adapter)
	                        else None
	                    )
                    rough_latent_delta = None
                    rough_delta_mask = None
                    if needs_rough_latent_delta(adapter):
                        rough_latent_delta, rough_delta_mask = rough_latent_delta_from_wm(
                            vae,
                            context_rgb,
                            target_video,
                            wm_out,
                            context_latents=base_latents if args.base_latents_source == "context" else None,
                        )
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                        adapter_out = adapter_forward(
                            adapter,
                            wm_out,
                            context_rgb,
                            action_cond,
                            c,
                            target_latents,
                            use_rough=bool(args.adapter_use_rough),
                            base_latents=base_latents,
                            rough_latent_delta=rough_latent_delta,
                            rough_delta_mask=rough_delta_mask,
                            return_components=True,
                        )
                        delta_latents, demo_components = split_adapter_output(adapter_out)
                    rough_video = rough_video_from_wm_out(context_rgb, wm_out) if args.residual_from_rough else None
                    rough_latents = encode_hunyuan_latents(vae, rough_video) if rough_video is not None else None
                    pred_latents = combine_with_residual_base(
                        delta_latents,
                        rough_latents,
                        residual_from_rough=args.residual_from_rough,
                        residual_scale=args.residual_scale,
                    )
                    pred_video = decode_hunyuan_latents(vae, pred_latents.float())
                    pred_video = apply_rgb_scaffold(
                        pred_video,
                        context_rgb,
                        wm_out,
                        scale=args.rgb_scaffold_scale,
                        mask_source=args.rgb_scaffold_mask_source,
                        mask_threshold=args.rgb_scaffold_mask_threshold,
                        mask_softness=args.rgb_scaffold_mask_softness,
                        mask_topk=args.rgb_scaffold_mask_topk,
                        residual_clip=args.rgb_scaffold_residual_clip,
                        mask_override=demo_components.get("rgb_scaffold_mask"),
                        clamp_output=True,
                    )
                    pred_video = apply_static_context_composite(
                        pred_video,
                        context_rgb,
                        wm_out,
                        scale=args.static_context_composite_scale,
                        mask_source="geometry" if args.static_context_mask_source == "learned" else args.static_context_mask_source,
                        mask_threshold=args.static_context_mask_threshold,
                        mask_softness=args.static_context_mask_softness,
                        mask_topk=args.static_context_mask_topk,
                        mask_spatial_dilate=args.static_context_mask_spatial_dilate,
                        mask_temporal_dilate=args.static_context_mask_temporal_dilate,
                        mask_floor=args.static_context_mask_floor,
                        mask_override=demo_components.get("rgb_scaffold_mask")
                        if args.static_context_mask_source == "learned"
                        else demo_components.get("motion_region_prior"),
                        clamp_output=True,
                    )
                    save_demo(args.out_dir / "demo_best.gif", pred_video, target_video.float(), wm_out.get("rgb"))
                except Exception as exc:
                    print(f"[rank0] demo save failed: {exc}", flush=True)
        return metrics

    max_epochs = args.epochs
    if args.max_steps and args.max_steps > 0:
        max_epochs = max(args.epochs, math.ceil(total_steps / max(1, len(train_loader))) + 1)
    stop_training = False
    for epoch in range(max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        adapter.train()
        for batch in train_loader:
            if step >= total_steps:
                stop_training = True
                break
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k=0)
            target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
            with torch.no_grad():
                target_latents = encode_hunyuan_latents(vae, target_video)
                context_video = context_video_from_batch(context_rgb, target_video.shape[2])
                latent_motion_mask = latent_motion_mask_from_video(
                    target_video,
                    context_video,
                    tuple(target_latents.shape[2:]),
                    threshold=float(args.mask_motion_threshold),
                    softness=float(args.mask_motion_softness),
                )
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                    wm_out = wm_model(
                        s,
                        c,
                        action_cond=action_cond,
                        context_rgb=context_rgb,
                        pixel=bool(args.wm_pixel),
                        bridging=False,
                        return_rgb_features=bool(getattr(adapter_cfg_from(adapter), "use_rgb_features", False)),
                    )
                rough_video = rough_video_from_wm_out(context_rgb, wm_out) if args.residual_from_rough else None
                rough_latents = encode_hunyuan_latents(vae, rough_video) if rough_video is not None else None
                base_latents = (
                    base_latents_from_source(
                        vae,
                        context_rgb,
                        target_video,
                        wm_out,
                        source=args.base_latents_source,
                        rough_prior_weight=args.base_rough_prior_weight,
                        rough_prior_power=args.base_rough_prior_power,
                        rough_prior_floor=args.base_rough_prior_floor,
                    )
                    if needs_context_anchor(adapter)
                    else None
                )
                rough_latent_delta = None
                rough_delta_mask = None
                if needs_rough_latent_delta(adapter):
                    rough_latent_delta, rough_delta_mask = rough_latent_delta_from_wm(
                        vae,
                        context_rgb,
                        target_video,
                        wm_out,
                        context_latents=base_latents if args.base_latents_source == "context" else None,
                    )
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.type == "cuda"):
                delta_latents = adapter_forward(
                    adapter,
                    wm_out,
                    context_rgb,
                    action_cond,
                    c,
                    target_latents,
                    use_rough=bool(args.adapter_use_rough),
                    base_latents=base_latents,
                    rough_latent_delta=rough_latent_delta,
                    rough_delta_mask=rough_delta_mask,
                    return_components=True,
                )
                delta_latents, components = split_adapter_output(delta_latents)
                pred_latents = combine_with_residual_base(
                    delta_latents,
                    rough_latents,
                    residual_from_rough=args.residual_from_rough,
                    residual_scale=args.residual_scale,
                )
                latent_mse = F.mse_loss(pred_latents.float(), target_latents.float())
                latent_l1 = F.l1_loss(pred_latents.float(), target_latents.float())
                loss = args.latent_mse_weight * latent_mse + args.latent_l1_weight * latent_l1
                decoded_l1 = pred_latents.new_zeros(())
                decoded_motion_l1 = pred_latents.new_zeros(())
                decoded_temporal_l1 = pred_latents.new_zeros(())
                decoded_motion_mag_l1 = pred_latents.new_zeros(())
                decoded_from_first_l1 = pred_latents.new_zeros(())
                decoded_static_l1 = pred_latents.new_zeros(())
                decoded_raw_motion_l1 = pred_latents.new_zeros(())
                decoded_raw_temporal_l1 = pred_latents.new_zeros(())
                decoded_raw_motion_mag_l1 = pred_latents.new_zeros(())
                decoded_raw_from_first_l1 = pred_latents.new_zeros(())
                latent_dynamic_l1 = pred_latents.new_zeros(())
                latent_static_l1 = pred_latents.new_zeros(())
                mask_l1 = pred_latents.new_zeros(())
                mask_bce = pred_latents.new_zeros(())
                mask_area = pred_latents.new_zeros(())
                mask_tv = pred_latents.new_zeros(())
                residual_l1 = pred_latents.new_zeros(())
                rgb_scaffold_mask_l1 = pred_latents.new_zeros(())
                rgb_scaffold_mask_bce = pred_latents.new_zeros(())
                rgb_scaffold_mask_area = pred_latents.new_zeros(())
                latent_motion_mask_f = latent_motion_mask.to(device=pred_latents.device, dtype=pred_latents.dtype)
                latent_static_mask = 1.0 - latent_motion_mask_f
                dyn_denom = (latent_motion_mask_f.sum() * pred_latents.shape[1]).clamp_min(1.0)
                sta_denom = (latent_static_mask.sum() * pred_latents.shape[1]).clamp_min(1.0)
                if args.latent_dynamic_l1_weight > 0:
                    latent_dynamic_l1 = ((pred_latents.float() - target_latents.float()).abs() * latent_motion_mask_f.float()).sum() / dyn_denom
                if args.latent_static_l1_weight > 0 and base_latents is not None:
                    latent_static_l1 = ((pred_latents.float() - base_latents.float()).abs() * latent_static_mask.float()).sum() / sta_denom
                if "mask" in components:
                    pred_mask = components["mask"].float()
                    mask_target = latent_motion_mask_f.float().clamp(0.0, 1.0)
                    if args.mask_l1_weight > 0:
                        mask_l1 = F.l1_loss(pred_mask, mask_target)
                    if args.mask_bce_weight > 0:
                        logits = components.get("mask_logits")
                        if logits is None:
                            eps = 1e-5
                            logits = torch.logit(pred_mask.clamp(eps, 1.0 - eps))
                        mask_bce = F.binary_cross_entropy_with_logits(logits.float(), mask_target)
                    if args.mask_area_weight > 0:
                        mask_area = pred_mask.mean()
                    if args.mask_tv_weight > 0:
                        tv_t = (pred_mask[:, :, 1:] - pred_mask[:, :, :-1]).abs().mean() if pred_mask.shape[2] > 1 else pred_mask.new_zeros(())
                        tv_h = (pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]).abs().mean()
                        tv_w = (pred_mask[:, :, :, :, 1:] - pred_mask[:, :, :, :, :-1]).abs().mean()
                        mask_tv = tv_t + tv_h + tv_w
                if args.residual_l1_weight > 0 and "residual" in components:
                    residual_l1 = components["residual"].float().abs().mean()
                if "rgb_scaffold_mask" in components:
                    rgb_mask = components["rgb_scaffold_mask"].float()
                    rgb_mask_target = motion_mask_from_rgb(
                        tgt["rgb_tgt_p"],
                        context_rgb,
                        threshold=float(args.decoded_static_threshold),
                    ).to(device=rgb_mask.device, dtype=rgb_mask.dtype)
                    if rgb_mask.shape[-2:] != rgb_mask_target.shape[-2:]:
                        rgb_mask_target = F.interpolate(
                            rgb_mask_target.permute(0, 2, 1, 3, 4),
                            size=(rgb_mask.shape[1], rgb_mask.shape[-2], rgb_mask.shape[-1]),
                            mode="trilinear",
                            align_corners=False,
                        ).permute(0, 2, 1, 3, 4)
                    if args.rgb_scaffold_mask_l1_weight > 0:
                        rgb_scaffold_mask_l1 = F.l1_loss(rgb_mask, rgb_mask_target)
                    if args.rgb_scaffold_mask_bce_weight > 0:
                        logits = components.get("rgb_scaffold_mask_logits")
                        if logits is None:
                            eps = 1e-5
                            logits = torch.logit(rgb_mask.clamp(eps, 1.0 - eps))
                        rgb_scaffold_mask_bce = F.binary_cross_entropy_with_logits(logits.float(), rgb_mask_target.float())
                    if args.rgb_scaffold_mask_area_weight > 0:
                        rgb_scaffold_mask_area = rgb_mask.mean()
                if (
                    args.decoded_l1_weight > 0
                    or args.decoded_motion_l1_weight > 0
                    or args.decoded_temporal_l1_weight > 0
                    or args.decoded_motion_mag_l1_weight > 0
                    or args.decoded_from_first_l1_weight > 0
                    or args.decoded_static_l1_weight > 0
                    or args.decoded_raw_motion_l1_weight > 0
                    or args.decoded_raw_temporal_l1_weight > 0
                    or args.decoded_raw_motion_mag_l1_weight > 0
                    or args.decoded_raw_from_first_l1_weight > 0
                ):
                    decoded_raw = decode_hunyuan_latents_grad(vae, pred_latents)
                    decoded = decoded_raw
                    decoded = apply_rgb_scaffold(
                        decoded,
                        context_rgb,
                        wm_out,
                        scale=args.rgb_scaffold_scale,
                        mask_source=args.rgb_scaffold_mask_source,
                        mask_threshold=args.rgb_scaffold_mask_threshold,
                        mask_softness=args.rgb_scaffold_mask_softness,
                        mask_topk=args.rgb_scaffold_mask_topk,
                        residual_clip=args.rgb_scaffold_residual_clip,
                        mask_override=components.get("rgb_scaffold_mask"),
                        clamp_output=False,
                    )
                    decoded = apply_static_context_composite(
                        decoded,
                        context_rgb,
                        wm_out,
                        scale=args.static_context_composite_scale,
                        mask_source="geometry" if args.static_context_mask_source == "learned" else args.static_context_mask_source,
                        mask_threshold=args.static_context_mask_threshold,
                        mask_softness=args.static_context_mask_softness,
                        mask_topk=args.static_context_mask_topk,
                        mask_spatial_dilate=args.static_context_mask_spatial_dilate,
                        mask_temporal_dilate=args.static_context_mask_temporal_dilate,
                        mask_floor=args.static_context_mask_floor,
                        mask_override=components.get("rgb_scaffold_mask")
                        if args.static_context_mask_source == "learned"
                        else components.get("motion_region_prior"),
                        clamp_output=False,
                    )
                    target_float = target_video.float()
                    decoded_l1 = F.l1_loss(decoded.float(), target_float)
                    if args.decoded_raw_motion_l1_weight > 0:
                        motion_mask = motion_mask_from_rgb(tgt["rgb_tgt_p"], context_rgb).permute(0, 2, 1, 3, 4)
                        motion_mask = torch.cat([torch.zeros_like(motion_mask[:, :, :1]), motion_mask], dim=2)
                        denom = (motion_mask.sum() * target_float.shape[1]).clamp_min(1.0)
                        decoded_raw_motion_l1 = ((decoded_raw.float() - target_float).abs() * motion_mask).sum() / denom
                    if args.decoded_raw_temporal_l1_weight > 0:
                        decoded_raw_temporal_l1 = F.l1_loss(
                            decoded_raw.float()[:, :, 1:] - decoded_raw.float()[:, :, :-1],
                            target_float[:, :, 1:] - target_float[:, :, :-1],
                        )
                    if args.decoded_raw_motion_mag_l1_weight > 0:
                        pred_step_mag = (decoded_raw.float()[:, :, 1:] - decoded_raw.float()[:, :, :-1]).abs().mean(dim=1, keepdim=True)
                        target_step_mag = (target_float[:, :, 1:] - target_float[:, :, :-1]).abs().mean(dim=1, keepdim=True)
                        step_mask = (target_step_mag > float(args.decoded_motion_mag_threshold)).float()
                        denom = step_mask.sum().clamp_min(1.0)
                        decoded_raw_motion_mag_l1 = ((pred_step_mag - target_step_mag).abs() * step_mask).sum() / denom
                    if args.decoded_raw_from_first_l1_weight > 0:
                        decoded_raw_from_first_l1 = F.l1_loss(
                            decoded_raw.float()[:, :, 1:] - decoded_raw.float()[:, :, :1],
                            target_float[:, :, 1:] - target_float[:, :, :1],
                        )
                    if args.decoded_motion_l1_weight > 0:
                        motion_mask = motion_mask_from_rgb(tgt["rgb_tgt_p"], context_rgb).permute(0, 2, 1, 3, 4)
                        motion_mask = torch.cat([torch.zeros_like(motion_mask[:, :, :1]), motion_mask], dim=2)
                        denom = (motion_mask.sum() * target_float.shape[1]).clamp_min(1.0)
                        decoded_motion_l1 = ((decoded.float() - target_float).abs() * motion_mask).sum() / denom
                    if args.decoded_temporal_l1_weight > 0:
                        decoded_temporal_l1 = F.l1_loss(
                            decoded.float()[:, :, 1:] - decoded.float()[:, :, :-1],
                            target_float[:, :, 1:] - target_float[:, :, :-1],
                        )
                    if args.decoded_motion_mag_l1_weight > 0:
                        pred_step_mag = (decoded.float()[:, :, 1:] - decoded.float()[:, :, :-1]).abs().mean(dim=1, keepdim=True)
                        target_step_mag = (target_float[:, :, 1:] - target_float[:, :, :-1]).abs().mean(dim=1, keepdim=True)
                        step_mask = (target_step_mag > float(args.decoded_motion_mag_threshold)).float()
                        denom = step_mask.sum().clamp_min(1.0)
                        decoded_motion_mag_l1 = ((pred_step_mag - target_step_mag).abs() * step_mask).sum() / denom
                    if args.decoded_from_first_l1_weight > 0:
                        decoded_from_first_l1 = F.l1_loss(
                            decoded.float()[:, :, 1:] - decoded.float()[:, :, :1],
                            target_float[:, :, 1:] - target_float[:, :, :1],
                        )
                    if args.decoded_static_l1_weight > 0:
                        static_mask = static_mask_from_rgb(
                            tgt["rgb_tgt_p"],
                            context_rgb,
                            threshold=float(args.decoded_static_threshold),
                        ).permute(0, 2, 1, 3, 4)
                        static_target = context_rgb.float()[:, :, None].expand(-1, -1, static_mask.shape[2], -1, -1)
                        denom = (static_mask.sum() * target_float.shape[1]).clamp_min(1.0)
                        decoded_static_l1 = (
                            (decoded.float()[:, :, 1 : 1 + static_mask.shape[2]] - static_target).abs() * static_mask
                        ).sum() / denom
                    loss = (
                        loss
                        + args.decoded_l1_weight * decoded_l1
                        + args.decoded_motion_l1_weight * decoded_motion_l1
                        + args.decoded_temporal_l1_weight * decoded_temporal_l1
                        + args.decoded_motion_mag_l1_weight * decoded_motion_mag_l1
                        + args.decoded_from_first_l1_weight * decoded_from_first_l1
                        + args.decoded_static_l1_weight * decoded_static_l1
                        + args.decoded_raw_motion_l1_weight * decoded_raw_motion_l1
                        + args.decoded_raw_temporal_l1_weight * decoded_raw_temporal_l1
                        + args.decoded_raw_motion_mag_l1_weight * decoded_raw_motion_mag_l1
                        + args.decoded_raw_from_first_l1_weight * decoded_raw_from_first_l1
                    )
                loss = (
                    loss
                    + args.latent_dynamic_l1_weight * latent_dynamic_l1
                    + args.latent_static_l1_weight * latent_static_l1
                    + args.mask_l1_weight * mask_l1
                    + args.mask_bce_weight * mask_bce
                    + args.mask_area_weight * mask_area
                    + args.mask_tv_weight * mask_tv
                    + args.residual_l1_weight * residual_l1
                    + args.rgb_scaffold_mask_l1_weight * rgb_scaffold_mask_l1
                    + args.rgb_scaffold_mask_bce_weight * rgb_scaffold_mask_bce
                    + args.rgb_scaffold_mask_area_weight * rgb_scaffold_mask_area
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            if rank == 0 and (step % args.print_every == 0 or step == 1):
                target_adapter = adapter.module if isinstance(adapter, DDP) else adapter
                print(
	                    f"[rank0] step {step} ep {epoch} loss={float(loss.detach().cpu()):.6f} "
	                    f"latent_mse={float(latent_mse.detach().cpu()):.6f} "
	                    f"latent_l1={float(latent_l1.detach().cpu()):.6f} "
	                    f"decoded_l1={float(decoded_l1.detach().cpu()):.6f} "
	                    f"decoded_motion_l1={float(decoded_motion_l1.detach().cpu()):.6f} "
		                    f"decoded_temporal_l1={float(decoded_temporal_l1.detach().cpu()):.6f} "
		                    f"decoded_motion_mag_l1={float(decoded_motion_mag_l1.detach().cpu()):.6f} "
		                    f"decoded_from_first_l1={float(decoded_from_first_l1.detach().cpu()):.6f} "
		                    f"decoded_static_l1={float(decoded_static_l1.detach().cpu()):.6f} "
		                    f"decoded_raw_motion_l1={float(decoded_raw_motion_l1.detach().cpu()):.6f} "
		                    f"decoded_raw_temporal_l1={float(decoded_raw_temporal_l1.detach().cpu()):.6f} "
		                    f"decoded_raw_motion_mag_l1={float(decoded_raw_motion_mag_l1.detach().cpu()):.6f} "
		                    f"decoded_raw_from_first_l1={float(decoded_raw_from_first_l1.detach().cpu()):.6f} "
		                    f"latent_dynamic_l1={float(latent_dynamic_l1.detach().cpu()):.6f} "
		                    f"latent_static_l1={float(latent_static_l1.detach().cpu()):.6f} "
		                    f"mask_l1={float(mask_l1.detach().cpu()):.6f} "
		                    f"mask_bce={float(mask_bce.detach().cpu()):.6f} "
		                    f"mask_area={float(mask_area.detach().cpu()):.6f} "
		                    f"rgb_mask_l1={float(rgb_scaffold_mask_l1.detach().cpu()):.6f} "
		                    f"rgb_mask_bce={float(rgb_scaffold_mask_bce.detach().cpu()):.6f} "
		                    f"rgb_mask_area={float(rgb_scaffold_mask_area.detach().cpu()):.6f} "
		                    f"motion_gain={float(target_adapter.motion_gain.detach().cpu()):.4f} "
		                    f"lr={sched.get_last_lr()[0]:.2e}",
	                    flush=True,
	                )
            if args.ckpt_every_steps > 0 and step % args.ckpt_every_steps == 0:
                save_checkpoint(f"step_{step:08d}", epoch=epoch, step_value=step, metrics=None)
            if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                run_eval(epoch, step)
            if step >= total_steps:
                stop_training = True
                break

        if args.eval_every_steps <= 0:
            metrics = run_eval(epoch, step)
            if rank == 0:
                save_checkpoint(f"epoch_{epoch:03d}", epoch=epoch, step_value=step, metrics=metrics)
        if stop_training:
            break

    if args.eval_every_steps > 0 and step > 0 and step % args.eval_every_steps != 0:
        run_eval(epoch, step)
    cleanup_ddp()


if __name__ == "__main__":
    main()
