from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from wm3d_v3.models.wan_ti2v_control_adapter import WanTI2VControlInjector


class WanRolloutError(ValueError):
    pass


@dataclass(frozen=True)
class WanRolloutProfile:
    steps: int
    shift: float
    guide_scale: float
    height: int
    width: int
    frame_num: int = 9


@dataclass(frozen=True)
class WanRolloutResult:
    video: Tensor
    dit_forward_count: int


TRAINING_ROLLOUT_PROFILE = WanRolloutProfile(
    steps=4,
    shift=5.0,
    guide_scale=1.0,
    height=192,
    width=256,
    frame_num=9,
)


def rollout_profile_for_context(
    context_rgb: Tensor,
    profile: WanRolloutProfile = TRAINING_ROLLOUT_PROFILE,
) -> WanRolloutProfile:
    if context_rgb.ndim != 4 or int(context_rgb.shape[1]) != 3:
        raise WanRolloutError(
            f"context_rgb must be BCHW RGB, got {tuple(context_rgb.shape)}"
        )
    input_height, input_width = (int(value) for value in context_rgb.shape[-2:])
    input_portrait = input_height > input_width
    profile_portrait = int(profile.height) > int(profile.width)
    if (
        input_height == input_width
        or int(profile.height) == int(profile.width)
        or input_portrait == profile_portrait
    ):
        return profile
    return WanRolloutProfile(
        steps=profile.steps,
        shift=profile.shift,
        guide_scale=profile.guide_scale,
        height=profile.width,
        width=profile.height,
        frame_num=profile.frame_num,
    )


def build_target_independent_initial_latents(
    context_latents: Tensor,
    *,
    generator: torch.Generator,
    condition_latent_frames: int = 1,
) -> Tensor:
    if context_latents.ndim != 5:
        raise WanRolloutError(
            f"context_latents must be BCTHW, got {tuple(context_latents.shape)}"
        )
    keep = int(condition_latent_frames)
    if keep <= 0 or keep >= int(context_latents.shape[2]):
        raise WanRolloutError(
            "condition_latent_frames must preserve at least one but not all latent "
            f"frames, got {keep} for T={context_latents.shape[2]}"
        )
    initial = torch.randn(
        context_latents.shape,
        generator=generator,
        device=context_latents.device,
        dtype=context_latents.dtype,
    )
    initial[:, :, :keep] = context_latents[:, :, :keep]
    return initial


def _encode_context_latents(
    pipeline: Any,
    context_rgb: Tensor,
    profile: WanRolloutProfile,
) -> Tensor:
    if context_rgb.ndim != 4 or int(context_rgb.shape[1]) != 3:
        raise WanRolloutError(
            f"context_rgb must be BCHW RGB, got {tuple(context_rgb.shape)}"
        )
    resized = F.interpolate(
        context_rgb.float().clamp(0.0, 1.0),
        size=(profile.height, profile.width),
        mode="bilinear",
        align_corners=False,
    )
    video = resized[:, :, None].expand(
        -1, -1, profile.frame_num, -1, -1
    ).contiguous()
    encoded = pipeline.vae.encode(
        [sample.mul(2.0).sub(1.0) for sample in video]
    )
    if torch.is_tensor(encoded):
        return encoded.to(device=context_rgb.device, dtype=torch.float32)
    return torch.stack(
        [
            latent.to(device=context_rgb.device, dtype=torch.float32)
            for latent in encoded
        ],
        dim=0,
    )


def _seq_len(pipeline: Any, latents: Tensor) -> int:
    patch_size = tuple(getattr(pipeline, "patch_size", (1, 2, 2)))
    sp_size = max(1, int(getattr(pipeline, "sp_size", 1)))
    _, _, time, height, width = latents.shape
    tokens = math.ceil(
        int(time)
        * int(height)
        * int(width)
        / (int(patch_size[1]) * int(patch_size[2]))
        / sp_size
    )
    return int(tokens * sp_size)


def _timestep_tokens(
    pipeline: Any,
    latents: Tensor,
    timestep_scalar: Tensor,
    seq_len: int,
    condition_latent_frames: int,
) -> Tensor:
    batch = int(latents.shape[0])
    value = timestep_scalar.float().reshape(1, 1).expand(batch, seq_len)
    patch_size = tuple(getattr(pipeline, "patch_size", (1, 2, 2)))
    _, _, time, height, width = latents.shape
    pt, ph, pw = (int(value) for value in patch_size)
    grid_t = max(1, math.ceil(int(time) / max(1, pt)))
    grid_h = max(1, math.ceil(int(height) / max(1, ph)))
    grid_w = max(1, math.ceil(int(width) / max(1, pw)))
    cond_t = min(grid_t, math.ceil(condition_latent_frames / max(1, pt)))
    mask = torch.ones(
        (batch, grid_t, grid_h, grid_w),
        device=latents.device,
        dtype=value.dtype,
    )
    mask[:, :cond_t] = 0.0
    flat = mask.flatten(1)
    if int(flat.shape[1]) < seq_len:
        flat = torch.cat(
            [flat, flat.new_ones(batch, seq_len - int(flat.shape[1]))],
            dim=1,
        )
    return value * flat[:, :seq_len]


def _append_action_context(
    text_context: Sequence[Tensor],
    adapter: Any,
    action_cond: Tensor,
    *,
    scale: float,
    max_tokens: int,
    text_len: int,
) -> list[Tensor]:
    tokens = adapter.action_context_tokens(action_cond)
    if tokens is None or scale == 0.0:
        return [embedding for embedding in text_context]
    if len(text_context) != int(tokens.shape[0]):
        raise WanRolloutError(
            "text/action batch mismatch: "
            f"text={len(text_context)} action={tokens.shape[0]}"
        )
    output: list[Tensor] = []
    for index, embedding in enumerate(text_context):
        extra_len = min(max_tokens, max(1, text_len - 1))
        extra = tokens[index, :extra_len].to(
            device=embedding.device,
            dtype=embedding.dtype,
        )
        extra = extra * scale
        keep = max(1, text_len - int(extra.shape[0]))
        output.append(torch.cat([embedding[:keep], extra], dim=0))
    return output


def _forward_velocity(
    transformer: torch.nn.Module,
    latents: Tensor,
    timestep: Tensor,
    context: Sequence[Tensor],
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    inputs = [
        sample.to(device=device, dtype=dtype if device.type == "cuda" else sample.dtype)
        for sample in latents
    ]
    output = transformer(
        inputs,
        t=timestep.to(device=device),
        context=list(context),
        seq_len=seq_len,
    )
    if isinstance(output, dict) and "video" in output:
        output = output["video"]
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        return torch.stack([sample.to(device=device) for sample in output])
    raise WanRolloutError(
        f"Wan transformer returned unsupported type {type(output)!r}"
    )


@torch.no_grad()
def roll_chunk(
    *,
    pipeline: Any,
    transformer: torch.nn.Module,
    adapter: Any,
    wm_out: dict[str, Tensor],
    context_rgb: Tensor,
    action_cond: Tensor,
    task_emb: Tensor,
    text_context: Sequence[Tensor],
    device: torch.device,
    generator: torch.Generator,
    precision: str,
    control_scale: float,
    action_context_scale: float,
    action_context_max_tokens: int,
    text_len: int,
    condition_latent_frames: int = 1,
    profile: WanRolloutProfile = TRAINING_ROLLOUT_PROFILE,
    wan_repo: str = "/data/Minko/external/Wan2.2",
) -> WanRolloutResult:
    profile = rollout_profile_for_context(context_rgb, profile)
    if profile.guide_scale != 1.0:
        raise WanRolloutError(
            "source-only training rollout requires guide_scale=1.0"
        )
    if profile.steps <= 0 or profile.frame_num != 9:
        raise WanRolloutError(f"invalid training rollout profile: {profile}")
    if str(wan_repo) not in sys.path:
        sys.path.insert(0, str(wan_repo))
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    context_latents = _encode_context_latents(
        pipeline,
        context_rgb.to(device),
        profile,
    )
    latents = build_target_independent_initial_latents(
        context_latents,
        generator=generator,
        condition_latent_frames=condition_latent_frames,
    )
    source_latents = context_latents.detach()
    seq_len = _seq_len(pipeline, latents)
    context = _append_action_context(
        text_context,
        adapter,
        action_cond,
        scale=float(action_context_scale),
        max_tokens=max(1, int(action_context_max_tokens)),
        text_len=max(2, int(text_len)),
    )
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=int(pipeline.num_train_timesteps),
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(
        int(profile.steps),
        device=device,
        shift=float(profile.shift),
    )
    dtype_name = str(precision).lower()
    dtype = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise WanRolloutError(f"unsupported rollout precision: {precision!r}")

    injector = WanTI2VControlInjector(transformer, adapter)
    injector.install()
    forward_count = 0
    try:
        with torch.amp.autocast(
            "cuda",
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            for timestep_value in scheduler.timesteps:
                timestep_scalar = torch.as_tensor(
                    timestep_value,
                    device=device,
                )
                sigma = (
                    timestep_scalar.float()
                    / float(pipeline.num_train_timesteps)
                ).reshape(1).expand(int(latents.shape[0]))
                timestep = _timestep_tokens(
                    pipeline,
                    latents,
                    timestep_scalar,
                    seq_len,
                    condition_latent_frames,
                )
                adapter.prepare_controls(
                    pred_tokens=wm_out["pred_tokens"],
                    depth=wm_out["depth"],
                    motion_hint=wm_out.get("motion_hint"),
                    contact_hint=wm_out.get("contact_hint"),
                    context_rgb=context_rgb,
                    action_cond=action_cond,
                    task_emb=task_emb,
                    point=wm_out.get("point"),
                    pose_geom=wm_out.get("pose_geom"),
                    noisy_latents=latents,
                    source_latents=source_latents,
                    sigma=sigma,
                    action_noisy=action_cond,
                    action_sigma=torch.zeros_like(sigma),
                    policy_action_cond=None,
                    latent_shape=tuple(latents.shape),
                    scale=float(control_scale),
                )
                injector._latent_shape = tuple(latents.shape)
                velocity = _forward_velocity(
                    transformer,
                    latents,
                    timestep,
                    context,
                    seq_len,
                    dtype,
                    device,
                )
                forward_count += 1
                latents = scheduler.step(
                    velocity,
                    timestep_value,
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0]
                latents[:, :, :condition_latent_frames] = context_latents[
                    :, :, :condition_latent_frames
                ]
    finally:
        adapter.clear_control_state()
        injector.remove()
    if forward_count != profile.steps:
        raise WanRolloutError(
            f"unexpected DiT forward count: {forward_count} != {profile.steps}"
        )

    decoded = pipeline.vae.decode(
        [sample.float() for sample in latents]
    )
    if torch.is_tensor(decoded):
        video = decoded.float()
    else:
        video = torch.stack(
            [sample.to(device=device, dtype=torch.float32) for sample in decoded]
        )
    video = video.clamp(-1.0, 1.0).div(2.0).add(0.5)
    return WanRolloutResult(
        video=video.contiguous(),
        dit_forward_count=forward_count,
    )
