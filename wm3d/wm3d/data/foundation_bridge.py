"""Lossless bridge from WM3D grouped windows to a joint video/action model.

The bridge deliberately does not resample actions. A sealed WM3D window has
``K`` video transitions and a source-native number of fine commands per
transition. Those commands are flattened in time while robot group and
dimension axes are retained exactly. The same mapping is used at training
and serving time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


class FoundationBridgeError(ValueError):
    """A WM3D sample cannot be represented without guessing or resampling."""


@dataclass(frozen=True)
class FoundationBridgeConfig:
    context_frames: int = 16
    future_frames: int = 8
    num_views: int = 3
    video_height: int = 384
    video_width: int = 320
    max_action_groups: int = 3
    max_action_dim: int = 14
    max_state_dim: int = 10

    def validate(self) -> None:
        for name in (
            "context_frames",
            "future_frames",
            "num_views",
            "video_height",
            "video_width",
            "max_action_groups",
            "max_action_dim",
            "max_state_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise FoundationBridgeError(f"{name} must be positive")
        if self.future_frames % 4:
            raise FoundationBridgeError(
                "future_frames must be divisible by four for the Wan video VAE"
            )
        if self.num_views != 3:
            raise FoundationBridgeError(
                "the current WM3D foundation layout requires exactly three views"
            )
        if self.video_height % 3 or self.video_width % 2:
            raise FoundationBridgeError(
                "three-view layout requires height divisible by 3 and width by 2"
            )

    @property
    def action_width(self) -> int:
        return self.max_action_groups * self.max_action_dim

    @property
    def state_width(self) -> int:
        return self.max_action_groups * self.max_state_dim


def _require_tensor(
    batch: Mapping[str, Any], name: str, *, ndim: int | None = None
) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise FoundationBridgeError(f"batch field {name!r} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise FoundationBridgeError(
            f"batch field {name!r} must have rank {ndim}, got {value.ndim}"
        )
    return value


def _resize_views(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, frames, channels, height, width = value.shape
    resized = F.interpolate(
        value.reshape(batch * frames, channels, height, width),
        size=size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(batch, frames, channels, *size)


def compose_three_view_video(
    images_u8: torch.Tensor,
    view_mask: torch.Tensor,
    config: FoundationBridgeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build WSA's three-view canvas without changing temporal sampling."""

    config.validate()
    if images_u8.ndim != 6:
        raise FoundationBridgeError("direct RGB must be [B,L,V,3,H,W]")
    batch, length, views, channels, _height, _width = images_u8.shape
    expected = config.context_frames + config.future_frames
    if length != expected or views != config.num_views or channels != 3:
        raise FoundationBridgeError(
            "direct RGB shape differs from the foundation bridge contract"
        )
    if tuple(view_mask.shape) != (batch, length, views):
        raise FoundationBridgeError("direct view mask must be [B,L,V]")
    if images_u8.dtype != torch.uint8 or view_mask.dtype != torch.bool:
        raise FoundationBridgeError("direct RGB/mask dtypes must be uint8/bool")
    selected = torch.cat(
        (
            images_u8[:, config.context_frames - 1 : config.context_frames],
            images_u8[:, config.context_frames :],
        ),
        dim=1,
    ).float().div_(255.0)
    selected_mask = torch.cat(
        (
            view_mask[:, config.context_frames - 1 : config.context_frames],
            view_mask[:, config.context_frames :],
        ),
        dim=1,
    )
    selected = selected * selected_mask[..., None, None, None]
    top_height = config.video_height - config.video_height // 3
    bottom_height = config.video_height // 3
    half_width = config.video_width // 2
    top = _resize_views(selected[:, :, 0], (top_height, config.video_width))
    left = _resize_views(selected[:, :, 1], (bottom_height, half_width))
    right = _resize_views(selected[:, :, 2], (bottom_height, half_width))
    canvas = torch.cat((top, torch.cat((left, right), dim=-1)), dim=-2)
    video = canvas.mul(2.0).sub_(1.0).permute(0, 2, 1, 3, 4).contiguous()
    image_is_pad = ~selected_mask.any(dim=-1)
    return video, image_is_pad


def pack_grouped_future_action(
    batch: Mapping[str, Any], config: FoundationBridgeConfig
) -> dict[str, torch.Tensor]:
    """Flatten ``[K,G,S,A]`` into source-native ``[K*S,G*A]`` commands."""

    values = _require_tensor(batch, "future_factual_fine_action_values", ndim=5)
    mask = _require_tensor(batch, "future_factual_fine_action_mask", ndim=5)
    dt = _require_tensor(batch, "future_factual_fine_action_dt", ndim=4)
    sample_mask = _require_tensor(batch, "future_factual_fine_sample_mask", ndim=4)
    boundaries = _require_tensor(batch, "future_world_boundaries_dt", ndim=2)
    if mask.dtype != torch.bool or sample_mask.dtype != torch.bool:
        raise FoundationBridgeError("future action masks must be boolean")
    if mask.shape != values.shape or sample_mask.shape != values.shape[:-1]:
        raise FoundationBridgeError("future action values and masks disagree")
    if dt.shape != sample_mask.shape:
        raise FoundationBridgeError("future action dt and sample mask disagree")
    batch_size, future, groups, substeps, action_dim = values.shape
    if (
        future != config.future_frames
        or groups != config.max_action_groups
        or action_dim != config.max_action_dim
    ):
        raise FoundationBridgeError(
            "future grouped action shape differs from the foundation contract"
        )
    if tuple(boundaries.shape) != (batch_size, future + 1):
        raise FoundationBridgeError("future world boundaries must be [B,K+1]")
    if not bool(torch.isfinite(boundaries).all()) or not bool(
        torch.diff(boundaries, dim=1).gt(0).all()
    ):
        raise FoundationBridgeError("future world boundaries must strictly increase")
    if not bool(boundaries[:, 0].abs().le(1.0e-6).all()):
        raise FoundationBridgeError(
            "future world boundaries must start at policy time zero"
        )
    element_mask = mask & sample_mask[..., None]
    action = values.permute(0, 1, 3, 2, 4).reshape(
        batch_size, future * substeps, groups * action_dim
    )
    element_mask = element_mask.permute(0, 1, 3, 2, 4).reshape_as(action)
    action = action.masked_fill(~element_mask, 0.0)
    time_valid = element_mask.any(dim=-1)
    dim_valid = element_mask.any(dim=1)
    factorized_mask = time_valid[:, :, None] & dim_valid[:, None, :]
    if not torch.equal(element_mask, factorized_mask):
        raise FoundationBridgeError(
            "future action validity cannot be represented by WSA time/dimension masks"
        )
    if not bool(time_valid.any(dim=1).all()):
        raise FoundationBridgeError("foundation sample has no future action")
    # Keep each command in its original world interval. Compacting all active
    # rows globally can turn K*4 slots into 31 commands, or silently regroup
    # commands when the remaining count happens to still be divisible by K.
    # Only trim a common unused suffix of the per-interval storage capacity.
    slot_used = time_valid.reshape(batch_size, future, substeps).any(dim=(0, 1))
    kept_substeps = int(torch.nonzero(slot_used, as_tuple=False)[-1, 0]) + 1
    action_time_by_dim = (
        boundaries[:, :-1, None, None] + dt
    )[..., None].expand_as(values).permute(
        0, 1, 3, 2, 4
    ).reshape_as(action)
    width = groups * action_dim
    action = action.reshape(batch_size, future, substeps, width)[
        :, :, :kept_substeps
    ].reshape(batch_size, future * kept_substeps, width)
    element_mask = element_mask.reshape(batch_size, future, substeps, width)[
        :, :, :kept_substeps
    ].reshape_as(action)
    action_time_by_dim = action_time_by_dim.reshape(
        batch_size, future, substeps, width
    )[:, :, :kept_substeps].reshape_as(action)
    time_valid = element_mask.any(dim=-1)
    dim_valid = element_mask.any(dim=1)
    valid_time = action_time_by_dim[element_mask]
    if not bool(torch.isfinite(valid_time).all()) or not bool((valid_time >= 0).all()):
        raise FoundationBridgeError(
            "future action times must be finite and non-negative"
        )
    return {
        "action": action,
        "action_element_mask": element_mask,
        "action_is_pad": ~time_valid,
        "action_dim_is_pad": ~dim_valid,
        "action_time_s": action_time_by_dim.masked_fill(~element_mask, 0.0),
    }


def pack_grouped_history_action(
    batch: Mapping[str, Any], config: FoundationBridgeConfig
) -> dict[str, torch.Tensor]:
    """Keep the complete observed command history as conditioning evidence."""

    values = _require_tensor(batch, "history_fine_action_values", ndim=5)
    mask = _require_tensor(batch, "history_fine_action_mask", ndim=5)
    dt = _require_tensor(batch, "history_fine_action_dt", ndim=4)
    sample_mask = _require_tensor(batch, "history_fine_sample_mask", ndim=4)
    boundaries = _require_tensor(batch, "history_world_boundaries_dt", ndim=2)
    if mask.dtype != torch.bool or sample_mask.dtype != torch.bool:
        raise FoundationBridgeError("history action masks must be boolean")
    if mask.shape != values.shape or sample_mask.shape != values.shape[:-1]:
        raise FoundationBridgeError("history action values and masks disagree")
    if dt.shape != sample_mask.shape:
        raise FoundationBridgeError("history action dt and sample mask disagree")
    batch_size, context, groups, substeps, action_dim = values.shape
    if (
        context != config.context_frames
        or groups != config.max_action_groups
        or action_dim != config.max_action_dim
    ):
        raise FoundationBridgeError(
            "history grouped action shape differs from the foundation contract"
        )
    if tuple(boundaries.shape) != (batch_size, context + 1):
        raise FoundationBridgeError("history world boundaries must be [B,T+1]")
    if not bool(torch.isfinite(boundaries).all()) or not bool(
        torch.diff(boundaries, dim=1).gt(0).all()
    ):
        raise FoundationBridgeError("history world boundaries must strictly increase")
    if not bool(boundaries[:, -1].abs().le(1.0e-6).all()):
        raise FoundationBridgeError(
            "history world boundaries must end at policy time zero"
        )
    element_mask = (mask & sample_mask[..., None]).permute(
        0, 1, 3, 2, 4
    ).reshape(batch_size, context * substeps, groups * action_dim)
    history = values.permute(0, 1, 3, 2, 4).reshape_as(element_mask)
    history = history.masked_fill(~element_mask, 0.0)
    history_time = (
        boundaries[:, :-1, None, None] + dt
    )[..., None].expand_as(values).permute(
        0, 1, 3, 2, 4
    ).reshape_as(element_mask)
    valid_time = history_time[element_mask]
    if valid_time.numel() == 0:
        raise FoundationBridgeError("foundation sample has no action history")
    if not bool(torch.isfinite(valid_time).all()) or not bool((valid_time <= 0).all()):
        raise FoundationBridgeError(
            "history action times must be finite and non-positive"
        )
    return {
        "history_action": history,
        "history_action_element_mask": element_mask,
        "history_action_time_s": history_time.masked_fill(~element_mask, 0.0),
    }


def _supervision_kind(batch: Mapping[str, Any], batch_size: int) -> str:
    value = batch.get("foundation_action_supervision", "fine_command")
    labels = [value] * batch_size if isinstance(value, str) else list(value)
    if len(labels) != batch_size or len(set(labels)) != 1:
        raise FoundationBridgeError("one source batch must use one action supervision kind")
    kind = labels[0]
    if kind not in {"fine_command", "coarse_effect"}:
        raise FoundationBridgeError("unsupported foundation action supervision")
    return kind


def pack_grouped_effect_sequence(
    batch: Mapping[str, Any], config: FoundationBridgeConfig, *, history: bool = False,
) -> dict[str, torch.Tensor]:
    """Pack admitted world-interval outcomes, never high-rate commands.

    The existing grouped composer has already combined native observation
    intervals with each physical field's operator (absolute next states use
    'last'). One target remains at each observed interval END. No interpolation
    or repetition invents an unrecorded command between observations.
    """
    prefix = "history" if history else "future_factual"
    values = _require_tensor(batch, prefix + "_coarse_action_values", ndim=4)
    mask = _require_tensor(batch, prefix + "_coarse_action_mask", ndim=4)
    fine_mask = _require_tensor(batch, prefix + "_fine_action_mask", ndim=5)
    if bool(fine_mask.any()):
        raise FoundationBridgeError("coarse effects must not replace any fine commands")
    boundaries = _require_tensor(
        batch, ("history" if history else "future") + "_world_boundaries_dt", ndim=2
    )
    b, length, groups, width = values.shape
    expected_length = config.context_frames if history else config.future_frames
    if (length, groups, width) != (
        expected_length, config.max_action_groups, config.max_action_dim
    ):
        raise FoundationBridgeError("coarse effect shape differs from the foundation contract")
    if mask.dtype != torch.bool or mask.shape != values.shape:
        raise FoundationBridgeError("coarse effect masks must be boolean and match values")
    if boundaries.shape != (b, length + 1) or not bool(torch.isfinite(boundaries).all()):
        raise FoundationBridgeError("coarse effect boundary shape/values are invalid")
    if not bool(torch.diff(boundaries, dim=1).gt(0).all()):
        raise FoundationBridgeError("coarse effect boundaries must strictly increase")
    anchor = boundaries[:, -1] if history else boundaries[:, 0]
    if not bool(anchor.abs().le(1.0e-6).all()):
        raise FoundationBridgeError("coarse effects must align with policy time zero")
    values = values.reshape(b, length, groups * width)
    element_mask = mask.reshape_as(values)
    if not bool(torch.isfinite(values[element_mask]).all()):
        raise FoundationBridgeError("coarse effect contains a non-finite target")
    time_valid = element_mask.any(dim=-1)
    dim_valid = element_mask.any(dim=1)
    if not bool(time_valid.any(dim=1).all()) or not torch.equal(
        element_mask, time_valid[:, :, None] & dim_valid[:, None, :]
    ):
        raise FoundationBridgeError("coarse effect cannot use WSA time/dimension masks")
    times = boundaries[:, 1:, None].expand_as(values)
    if not bool((times[element_mask] <= 0 if history else times[element_mask] > 0).all()):
        raise FoundationBridgeError("coarse effect endpoint crosses the observation boundary")
    if history:
        return {
            "history_action": values.masked_fill(~element_mask, 0),
            "history_action_element_mask": element_mask,
            "history_action_time_s": times.masked_fill(~element_mask, 0),
        }
    return {
        "action": values.masked_fill(~element_mask, 0),
        "action_element_mask": element_mask,
        "action_is_pad": ~time_valid,
        "action_dim_is_pad": ~dim_valid,
        "action_time_s": times.masked_fill(~element_mask, 0),
    }


def pack_grouped_current_state(
    batch: Mapping[str, Any], config: FoundationBridgeConfig
) -> dict[str, torch.Tensor]:
    values = _require_tensor(batch, "current_state_values", ndim=3)
    mask = _require_tensor(batch, "current_state_mask", ndim=3)
    if mask.dtype != torch.bool or mask.shape != values.shape:
        raise FoundationBridgeError("current state values and masks disagree")
    batch_size, groups, state_dim = values.shape
    if groups != config.max_action_groups or state_dim != config.max_state_dim:
        raise FoundationBridgeError(
            "current grouped state shape differs from the foundation contract"
        )
    state = values.reshape(batch_size, 1, groups * state_dim)
    state_mask = mask.reshape(batch_size, groups * state_dim)
    return {
        "proprio": state.masked_fill(~state_mask[:, None], 0.0),
        "proprio_dim_is_pad": ~state_mask,
        "proprio_is_pad": ~state_mask.any(dim=-1, keepdim=True),
    }


def _task_texts(value: Any, batch_size: int) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        result = [str(item) for item in value]
    else:
        raise FoundationBridgeError("task_text must be a string sequence")
    if len(result) != batch_size or any(not item.strip() for item in result):
        raise FoundationBridgeError("every foundation sample requires task text")
    return [item.strip() for item in result]


def build_foundation_sample(
    batch: Mapping[str, Any], config: FoundationBridgeConfig
) -> dict[str, Any]:
    """Convert one source-homogeneous WM3D micro-batch without resampling."""

    config.validate()
    images = _require_tensor(batch, "direct_rgb_uint8", ndim=6)
    view_mask = _require_tensor(batch, "direct_view_mask", ndim=3).bool()
    source_id = _require_tensor(batch, "source_id", ndim=1)
    if int(torch.unique(source_id).numel()) != 1:
        raise FoundationBridgeError("one foundation micro-batch must use one source")
    video, image_is_pad = compose_three_view_video(images, view_mask, config)
    supervision = _supervision_kind(batch, int(images.shape[0]))
    if supervision == "coarse_effect":
        action = pack_grouped_effect_sequence(batch, config)
        history_action = pack_grouped_effect_sequence(batch, config, history=True)
    else:
        for prefix in ("history", "future_factual"):
            coarse = batch.get(prefix + "_coarse_action_mask")
            if isinstance(coarse, torch.Tensor) and bool(coarse.any()):
                raise FoundationBridgeError("coarse supervision requires explicit effect admission")
        action = pack_grouped_future_action(batch, config)
        history_action = pack_grouped_history_action(batch, config)
    state = pack_grouped_current_state(batch, config)
    future_index = config.context_frames + config.future_frames - 1
    result: dict[str, Any] = {
        "video": video,
        "image_is_pad": image_is_pad,
        "future_3d_images": images[:, future_index].float().div(255.0),
        "future_3d_img_masks": view_mask[:, future_index],
        "task_text": _task_texts(batch.get("task_text"), int(images.shape[0])),
        "source_id": source_id,
    }
    result.update(action)
    result.update(history_action)
    result["action_supervision_kind"] = supervision
    result.update(state)
    for name in (
        "action_group_ids",
        "action_group_mask",
        "action_semantic_ids",
        "state_semantic_ids",
        "embodiment_ids",
        "action_normalization_offset",
        "action_normalization_scale",
        "state_normalization_offset",
        "state_normalization_scale",
    ):
        result[name] = _require_tensor(batch, name)
    return result
