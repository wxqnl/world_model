"""Unified native-world and grouped-policy objectives for WM3D."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from wm3d.data.grouped_robot import COMPOSITION_OPERATOR_IDS


class NativeObjectiveError(ValueError):
    pass


@dataclass(frozen=True)
class NativeObjectiveConfig:
    token_mse: float = 1.0
    token_cosine: float = 0.1
    appearance_l1: float = 0.0
    appearance_teacher_l1: float = 0.0
    appearance_autoregressive_l1: float = 0.0
    appearance_motion_l1: float = 0.0
    appearance_mse: float = 0.0
    appearance_cosine: float = 0.0
    appearance_motion_mse: float = 0.0
    appearance_delta_cosine: float = 0.0
    rgb_l1: float = 0.0
    rgb_charbonnier: float = 2.0
    rgb_gradient: float = 0.5
    rgb_perceptual: float = 0.0
    rgb_motion_l1: float = 0.0
    rgb_motion_bce: float = 0.0
    rgb_motion_dice: float = 0.0
    rgb_motion_pos_weight: float = 1.0
    rgb_motion_threshold: float = 0.03
    rgb_motion_gain: float = 3.0
    rgb_flow_teacher: float = 0.0
    rgb_disocclusion_bce: float = 0.0
    rgb_disocclusion_dice: float = 0.0
    rgb_disocclusion_pos_weight: float = 1.0
    depth_log: float = 1.5
    depth_temporal: float = 0.0
    point: float = 0.5
    point_temporal: float = 0.0
    camera_pose: float = 0.1
    action_fine: float = 2.0
    action_coarse: float = 1.0
    action_velocity: float = 0.0
    action_counterfactual_token_advantage: float = 0.0
    action_counterfactual_token_margin: float = 0.0
    action_counterfactual_rgb_advantage: float = 0.0
    action_counterfactual_rgb_margin: float = 0.0
    # Original V7 actionbridge RGB curriculum.  This is deliberately separate
    # from the zero-action telemetry above: the historical 60K model learned
    # from a differentiable shuffled-action forward with both ranking and
    # output-separation terms.
    context_pixel_action_rank_weight: float = 0.0
    context_pixel_action_separation_weight: float = 0.0
    context_pixel_action_rank_start_step: int = 0
    context_pixel_action_rank_ramp_steps: int = 0
    context_pixel_action_rank_every: int = 1
    context_pixel_action_rank_batch_size: int = 0
    context_pixel_action_rank_margin: float = 0.003
    context_pixel_action_separation_margin: float = 0.006
    context_pixel_action_motion_threshold: float = 0.03
    context_pixel_action_motion_gain: float = 4.0
    context_pixel_action_negative_min_distance: float = 0.05
    epsilon: float = 1.0e-6
    rgb_charbonnier_epsilon: float = 1.0e-3
    huber_delta: float = 0.05

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name in {"epsilon", "rgb_charbonnier_epsilon", "huber_delta"}:
                if value <= 0:
                    raise NativeObjectiveError(f"{name} must be positive")
            elif value < 0:
                raise NativeObjectiveError(f"{name} cannot be negative")
        if self.rgb_motion_pos_weight <= 0.0:
            raise NativeObjectiveError("rgb_motion_pos_weight must be positive")
        if self.rgb_motion_threshold <= 0.0:
            raise NativeObjectiveError("rgb_motion_threshold must be positive")
        if self.rgb_disocclusion_pos_weight <= 0.0:
            raise NativeObjectiveError(
                "rgb_disocclusion_pos_weight must be positive"
            )
        if self.context_pixel_action_rank_every <= 0:
            raise NativeObjectiveError(
                "context_pixel_action_rank_every must be positive"
            )
        if self.context_pixel_action_rank_batch_size < 0:
            raise NativeObjectiveError(
                "context_pixel_action_rank_batch_size cannot be negative"
            )
        if (
            max(
                self.context_pixel_action_rank_weight,
                self.context_pixel_action_separation_weight,
            )
            > 0.0
            and self.context_pixel_action_rank_batch_size != 1
        ):
            raise NativeObjectiveError(
                "the distributed V7 RGB action curriculum requires "
                "context_pixel_action_rank_batch_size=1"
            )
        if self.context_pixel_action_motion_threshold <= 0.0:
            raise NativeObjectiveError(
                "context_pixel_action_motion_threshold must be positive"
            )
        if self.action_velocity != 0.0:
            raise NativeObjectiveError(
                "action_velocity must remain 0: mixed delta/absolute semantics and "
                "source-native cadences have no shared physical smoothness invariant"
            )
        for weight_name, margin_name in (
            (
                "action_counterfactual_token_advantage",
                "action_counterfactual_token_margin",
            ),
            (
                "action_counterfactual_rgb_advantage",
                "action_counterfactual_rgb_margin",
            ),
        ):
            if getattr(self, weight_name) > 0.0 and getattr(self, margin_name) <= 0.0:
                raise NativeObjectiveError(
                    f"{margin_name} must be positive when {weight_name} is enabled"
                )


def objective_config_from_mapping(mapping: Mapping[str, object]) -> NativeObjectiveConfig:
    config = NativeObjectiveConfig(**dict(mapping))
    config.validate()
    return config


def build_rgb_perceptual_model(
    config: NativeObjectiveConfig,
    *,
    device: torch.device,
) -> nn.Module | None:
    """Build one frozen VGG LPIPS network per training rank."""

    if config.rgb_perceptual <= 0:
        return None
    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - release environment gate
        raise NativeObjectiveError(
            "rgb_perceptual requires the sealed lpips dependency"
        ) from exc
    model = lpips.LPIPS(net="vgg", verbose=False).to(device=device).eval()
    model.requires_grad_(False)
    return model


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    if mask.shape != value.shape:
        mask = torch.broadcast_to(mask, value.shape)
    weight = mask.to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(epsilon)


def _normalise_depth_per_frame(
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Remove the independent scale of every [B,K,V] depth frame."""

    depth_float = depth.float()
    valid = mask.bool() & torch.isfinite(depth_float)
    masked = depth_float.masked_fill(~valid, torch.nan)
    median = torch.nanmedian(masked, dim=-1).values
    median = torch.where(
        valid.any(dim=-1) & torch.isfinite(median),
        median.clamp_min(epsilon),
        torch.ones_like(median),
    )
    normalised = depth_float / median[..., None]
    return torch.where(valid, normalised, torch.zeros_like(normalised))


def _normalise_points_per_window(
    points: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Remove each [B,K,V,P,3] window's translation and global scale."""

    points_float = points.float()
    valid = mask.bool() & torch.isfinite(points_float).all(dim=-1)
    weight = valid[..., None].to(dtype=points_float.dtype)
    safe_points = torch.where(valid[..., None], points_float, 0.0)
    denominator = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
    center = safe_points.sum(dim=(1, 2, 3), keepdim=True) / denominator
    centered = points_float - center
    safe_centered = torch.where(valid[..., None], centered, 0.0)
    squared_radius = safe_centered.square().sum(dim=-1, keepdim=True)
    scale = (
        (squared_radius * weight).sum(dim=(1, 2, 3), keepdim=True)
        / denominator
    ).sqrt().clamp_min(max(epsilon, 1.0e-3))
    normalised = centered / scale
    return torch.where(valid[..., None], normalised, torch.zeros_like(normalised))


def _warp_token_grid_with_pixel_flow(
    source: torch.Tensor,
    flow_pixels: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Backward-warp [B,K,V,P,D] source features with RGB-pixel flow."""

    if source.ndim != 5 or flow_pixels.ndim != 6:
        raise NativeObjectiveError(
            "flow-aligned appearance target requires [B,K,V,P,D] and flow"
        )
    batch, horizon, views, patches, dim = source.shape
    grid = isqrt(patches)
    if grid * grid != patches or tuple(flow_pixels.shape[:3]) != (
        batch,
        horizon,
        views,
    ) or flow_pixels.shape[3] != 2:
        raise NativeObjectiveError("appearance token grid and RGB flow differ")
    count = batch * horizon * views
    feature = source.reshape(count, grid, grid, dim).permute(0, 3, 1, 2)
    flow = flow_pixels.reshape(count, 2, *flow_pixels.shape[-2:]).float()
    if flow.shape[-2:] != (grid, grid):
        flow = F.interpolate(
            flow,
            size=(grid, grid),
            mode="bilinear",
            align_corners=True,
        )
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, grid, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    sample_grid = torch.stack((x, y), dim=-1)[None].expand(count, -1, -1, -1)
    sample_grid = sample_grid + torch.stack(
        (
            2.0 * flow[:, 0] / max(float(image_width - 1), 1.0),
            2.0 * flow[:, 1] / max(float(image_height - 1), 1.0),
        ),
        dim=-1,
    )
    warped = F.grid_sample(
        feature.float(),
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped.permute(0, 2, 3, 1).reshape(
        batch, horizon, views, patches, dim
    ).to(dtype=source.dtype)


def _masked_per_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    if mask.shape != value.shape:
        mask = torch.broadcast_to(mask, value.shape)
    weight = mask.to(dtype=value.dtype)
    numerator = (value * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1).clamp_min(epsilon)
    return numerator / denominator


def _factual_zero_advantage(
    *,
    factual: torch.Tensor,
    zero_action: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    epsilon: float,
    absolute_error: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if factual.shape != target.shape:
        raise NativeObjectiveError("factual prediction and target shapes differ")
    if zero_action.shape != target.shape:
        raise NativeObjectiveError("zero-action prediction and target shapes differ")
    factual_delta = factual - target
    zero_delta = zero_action - target
    factual_error = factual_delta.abs() if absolute_error else factual_delta.square()
    zero_error = zero_delta.abs() if absolute_error else zero_delta.square()
    factual_per_sample = _masked_per_sample_mean(
        factual_error,
        mask,
        epsilon=epsilon,
    )
    zero_per_sample = _masked_per_sample_mean(
        zero_error,
        mask,
        epsilon=epsilon,
    )
    sample_valid = torch.broadcast_to(mask, target.shape).flatten(1).any(dim=1)
    zero_error_mean = _masked_mean(
        zero_per_sample,
        sample_valid,
        epsilon=epsilon,
    )
    gain = _masked_mean(
        zero_per_sample.detach() - factual_per_sample.detach(),
        sample_valid,
        epsilon=epsilon,
    )
    advantage = _masked_mean(
        torch.relu(margin + factual_per_sample - zero_per_sample),
        sample_valid,
        epsilon=epsilon,
    )
    response_rms = _masked_mean(
        (factual - zero_action).square(),
        mask,
        epsilon=epsilon,
    ).sqrt()
    return zero_error_mean, gain, advantage, response_rms


def _charbonnier(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt(value.square() + epsilon * epsilon)


def _image_gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., 1:, :] - value[..., :-1, :], value[..., :, 1:] - value[..., :, :-1]


def _masked_rgb_perceptual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    model: nn.Module,
    *,
    chunk_size: int = 4,
) -> torch.Tensor:
    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise NativeObjectiveError("RGB perceptual chunk size must be a positive integer")
    if prediction.ndim != 6 or prediction.shape[-3] != 3:
        raise NativeObjectiveError("RGB perceptual tensors must be [B,F,V,3,H,W]")
    if prediction.shape != target.shape:
        raise NativeObjectiveError("RGB prediction/target shapes differ")
    if mask.shape == prediction.shape[:3] + (1, 1, 1):
        image_all = mask.reshape(-1).bool()
        image_any = image_all
    else:
        expanded_mask = torch.broadcast_to(mask, prediction.shape)
        flat_mask = expanded_mask.reshape(-1, *prediction.shape[-3:])
        image_all = flat_mask.all(dim=(1, 2, 3))
        image_any = flat_mask.any(dim=(1, 2, 3))
    if not bool((image_all == image_any).all()):
        raise NativeObjectiveError(
            "RGB perceptual supervision requires whole-image masks"
        )
    valid = torch.nonzero(image_all, as_tuple=False).flatten()
    if valid.numel() == 0:
        return prediction.new_zeros(())
    pred_images = prediction.reshape(-1, *prediction.shape[-3:]).index_select(0, valid)
    target_images = target.reshape(-1, *target.shape[-3:]).index_select(0, valid)
    total = prediction.new_zeros((), dtype=torch.float32)
    frozen_model = not any(parameter.requires_grad for parameter in model.parameters())
    for start in range(0, int(valid.numel()), chunk_size):
        pred_source = pred_images[start : start + chunk_size].float()
        target_source = target_images[start : start + chunk_size].float()
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            if (
                torch.is_grad_enabled()
                and pred_source.requires_grad
                and not target_source.requires_grad
                and frozen_model
            ):
                # LPIPS is a frozen first-order objective. Generic activation
                # checkpointing replays both VGG branches during the outer
                # backward even though the target branch has no gradient.
                # Compute the exact image gradient while this small chunk is
                # resident, then reconnect it to the world-model RGB tensor
                # through a zero-valued surrogate. The scalar and first-order
                # gradient are unchanged; higher-order derivatives, which the
                # training contract never requests, are not retained.
                leaf = pred_source.detach().requires_grad_(True)
                with torch.enable_grad():
                    distance = model(
                        leaf.mul(2.0).sub(1.0),
                        target_source.mul(2.0).sub(1.0),
                    )
                    chunk_value = distance.float().sum()
                    image_gradient = torch.autograd.grad(
                        chunk_value,
                        leaf,
                        create_graph=False,
                        retain_graph=False,
                    )[0]
                proxy = (pred_source * image_gradient.detach()).sum()
                total = total + chunk_value.detach() + proxy - proxy.detach()
            else:
                pred_chunk = pred_source.mul(2.0).sub(1.0)
                target_chunk = target_source.mul(2.0).sub(1.0)
                if torch.is_grad_enabled() and pred_chunk.requires_grad:
                    distance = checkpoint(
                        model,
                        pred_chunk,
                        target_chunk,
                        use_reentrant=False,
                    )
                else:
                    distance = model(pred_chunk, target_chunk)
                total = total + distance.float().sum()
    return total / valid.numel()


def _rotvec_to_quaternion(rotvec: torch.Tensor, epsilon: float) -> torch.Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(angle/2)/angle has a finite limit of 1/2.
    scale = torch.where(
        angle > epsilon,
        torch.sin(half) / angle.clamp_min(epsilon),
        0.5 - angle.square() / 48.0,
    )
    vector = rotvec * scale
    return torch.cat((torch.cos(half), vector), dim=-1)


def _quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternion_to_rotvec(quaternion: torch.Tensor, epsilon: float) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(epsilon)
    # q and -q encode the same rotation; use positive scalar for the shortest
    # principal rotation vector and a deterministic representation.
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    scalar = quaternion[..., :1].clamp(-1.0, 1.0)
    vector = quaternion[..., 1:]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, scalar)
    scale = torch.where(
        norm > epsilon,
        angle / norm.clamp_min(epsilon),
        2.0 + norm.square() / 3.0,
    )
    return vector * scale


def compose_axis_angle_sequence(
    rotvec: torch.Tensor,
    valid: torch.Tensor,
    *,
    left_multiply: bool = False,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequentially compose a masked [...,S,3] rotation-vector sequence.

    ``left_multiply=False`` is a body-frame/right-increment convention
    (``R <- R @ dR``); ``True`` is base-frame/left-increment
    (``R <- dR @ R``).  The choice is physical source metadata and must not be
    inferred from a robot or dataset name.
    """

    if rotvec.shape[-1] != 3 or valid.shape != rotvec.shape[:-1]:
        raise NativeObjectiveError("axis-angle sequence shapes are inconsistent")
    quaternion = torch.zeros(*rotvec.shape[:-2], 4, dtype=rotvec.dtype, device=rotvec.device)
    quaternion[..., 0] = 1.0
    for step in range(rotvec.shape[-2]):
        increment = _rotvec_to_quaternion(rotvec[..., step, :], epsilon)
        identity = torch.zeros_like(increment)
        identity[..., 0] = 1.0
        increment = torch.where(valid[..., step, None], increment, identity)
        quaternion = (
            _quaternion_multiply(increment, quaternion)
            if left_multiply
            else _quaternion_multiply(quaternion, increment)
        )
    return _quaternion_to_rotvec(quaternion, epsilon), valid.any(dim=-1)


def compose_policy_to_world_intervals(
    *,
    policy_action: torch.Tensor,
    policy_action_mask: torch.Tensor,
    policy_query_dt: torch.Tensor,
    future_world_boundaries_dt: torch.Tensor,
    composition_operator_ids: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose source/serving-rate policy predictions into world intervals.

    Args:
        policy_action: ``[B,G,C,A]`` decoded action values.
        policy_action_mask: identical shape, real query/dimension mask.
        policy_query_dt: ``[B,G,C]`` seconds from policy chunk start.
        future_world_boundaries_dt: ``[B,K+1]`` seconds from chunk start.
        composition_operator_ids: ``[B,G,A]`` static physical operators.
    """

    if policy_action.shape != policy_action_mask.shape:
        raise NativeObjectiveError("policy action and mask must have identical shape")
    batch, groups, queries, action_dim = policy_action.shape
    if tuple(policy_query_dt.shape) != (batch, groups, queries):
        raise NativeObjectiveError("policy_query_dt must be [B,G,C]")
    if tuple(composition_operator_ids.shape) != (batch, groups, action_dim):
        raise NativeObjectiveError("composition_operator_ids must be [B,G,A]")
    if future_world_boundaries_dt.ndim != 2 or future_world_boundaries_dt.shape[0] != batch:
        raise NativeObjectiveError("future world boundaries must be [B,K+1]")
    if not bool(torch.diff(future_world_boundaries_dt, dim=1).gt(0).all()):
        raise NativeObjectiveError("future world boundaries must be strictly increasing")
    valid_query = policy_action_mask.any(dim=-1)
    valid_pairs = valid_query[:, :, 1:] & valid_query[:, :, :-1]
    if bool((torch.diff(policy_query_dt, dim=-1)[valid_pairs] <= 0).any()):
        raise NativeObjectiveError(
            "valid policy query times must be strictly increasing per action group"
        )
    horizon = future_world_boundaries_dt.shape[1] - 1
    result = policy_action.new_zeros(batch, horizon, groups, action_dim)
    result_mask = torch.zeros_like(result, dtype=torch.bool)

    operator_none = COMPOSITION_OPERATOR_IDS["none"]
    operator_sum = COMPOSITION_OPERATOR_IDS["sum"]
    operator_so3_left = COMPOSITION_OPERATOR_IDS["so3_axis_angle_base_left"]
    operator_so3_right = COMPOSITION_OPERATOR_IDS["so3_axis_angle_body_right"]
    operator_last = COMPOSITION_OPERATOR_IDS["last"]
    operator_mean = COMPOSITION_OPERATOR_IDS["time_weighted_mean"]
    operator_logical_last = COMPOSITION_OPERATOR_IDS["logical_last"]

    operators = composition_operator_ids
    supported = (
        operators.eq(operator_none)
        | operators.eq(operator_sum)
        | operators.eq(operator_so3_left)
        | operators.eq(operator_so3_right)
        | operators.eq(operator_last)
        | operators.eq(operator_mean)
        | operators.eq(operator_logical_last)
    )
    if not bool(supported.all()):
        raise NativeObjectiveError("unsupported composition operator id")

    starts = future_world_boundaries_dt[:, :-1, None, None]
    stops = future_world_boundaries_dt[:, 1:, None, None]
    query_times = policy_query_dt[:, None]
    in_interval = (query_times >= starts) & (query_times < stops)
    valid = in_interval[..., None] & policy_action_mask[:, None]
    values = policy_action[:, None].expand(-1, horizon, -1, -1, -1)
    valid_any = valid.any(dim=3)

    summed = (values * valid.to(dtype=values.dtype)).sum(dim=3)
    query_indices = torch.arange(queries, device=policy_action.device).view(
        1, 1, 1, queries, 1
    )
    last_indices = torch.where(valid, query_indices, -1).amax(dim=3)
    last = values.gather(
        3, last_indices.clamp_min(0).unsqueeze(3)
    ).squeeze(3)
    last = torch.where(last_indices >= 0, last, torch.zeros_like(last))

    # Commands are zero-order held until the next valid command or interval
    # end.  The only Python loop is over the real query clock; it launches
    # batched tensor work and never reads individual CUDA scalars.
    next_seen = stops.expand(batch, horizon, groups, action_dim)
    durations: list[torch.Tensor] = [torch.empty(0)] * queries
    interval_starts = starts
    interval_stops = stops
    for query_index in range(queries - 1, -1, -1):
        current_valid = valid[:, :, :, query_index]
        current_time = query_times[:, :, :, query_index, None].expand(
            batch, horizon, groups, action_dim
        )
        duration = (
            torch.minimum(next_seen, interval_stops)
            - torch.maximum(current_time, interval_starts)
        ).clamp_min(0.0)
        durations[query_index] = duration * current_valid.to(duration.dtype)
        next_seen = torch.where(current_valid, current_time, next_seen)
    duration = torch.stack(durations, dim=3)
    duration_sum = duration.sum(dim=3)
    time_weighted = (values * duration).sum(dim=3) / duration_sum.clamp_min(
        epsilon
    )

    expanded_operators = operators[:, None]
    scalar_operator = (
        expanded_operators.eq(operator_sum)
        | expanded_operators.eq(operator_last)
        | expanded_operators.eq(operator_logical_last)
        | expanded_operators.eq(operator_mean)
    )
    result = torch.where(
        expanded_operators.eq(operator_sum), summed, torch.zeros_like(summed)
    )
    result = torch.where(
        expanded_operators.eq(operator_last)
        | expanded_operators.eq(operator_logical_last),
        last,
        result,
    )
    result = torch.where(
        expanded_operators.eq(operator_mean), time_weighted, result
    )
    result_mask = valid_any & scalar_operator

    # SO(3) runs are uncommon but may differ per sample/group.  Walk the
    # bounded action-dimension metadata once, while composing every batch,
    # interval and group in parallel.  Adjacent runs are consumed in triples.
    consumed = torch.zeros_like(operators, dtype=torch.bool)
    invalid_so3 = torch.zeros(batch, groups, dtype=torch.bool, device=operators.device)
    so3_result = torch.zeros_like(result)
    so3_mask = torch.zeros_like(result_mask)
    is_so3 = operators.eq(operator_so3_left) | operators.eq(operator_so3_right)
    candidate_dimensions = torch.nonzero(
        is_so3.any(dim=(0, 1)), as_tuple=False
    ).flatten().tolist()
    for dim in candidate_dimensions:
        active_left = operators[..., dim].eq(operator_so3_left) & ~consumed[..., dim]
        active_right = operators[..., dim].eq(operator_so3_right) & ~consumed[..., dim]
        active = active_left | active_right
        if not bool(active.any()):
            continue
        if dim + 3 > action_dim:
            invalid_so3 |= active
            continue
        segment = operators[..., dim : dim + 3]
        left_triplet = active_left & segment.eq(operator_so3_left).all(dim=-1)
        right_triplet = active_right & segment.eq(operator_so3_right).all(dim=-1)
        invalid_so3 |= active & ~(left_triplet | right_triplet)
        triplet = left_triplet | right_triplet
        if not bool(triplet.any()):
            continue

        rotation_values = values[..., dim : dim + 3]
        rotation_valid = valid[..., dim : dim + 3].all(dim=-1)
        left_value, left_valid = compose_axis_angle_sequence(
            rotation_values,
            rotation_valid,
            left_multiply=True,
            epsilon=epsilon,
        )
        right_value, right_valid = compose_axis_angle_sequence(
            rotation_values,
            rotation_valid,
            left_multiply=False,
            epsilon=epsilon,
        )
        composed = torch.where(
            left_triplet[:, None, :, None], left_value, right_value
        )
        composed_valid = torch.where(
            left_triplet[:, None, :], left_valid, right_valid
        )
        for offset in range(3):
            so3_result[..., dim + offset] = torch.where(
                triplet[:, None, :],
                composed[..., offset],
                so3_result[..., dim + offset],
            )
            so3_mask[..., dim + offset] = torch.where(
                triplet[:, None, :],
                composed_valid,
                so3_mask[..., dim + offset],
            )
        consumed[..., dim : dim + 3] |= triplet[..., None]

    invalid_so3 |= (is_so3 & ~consumed).any(dim=-1)
    if bool(invalid_so3.any()):
        raise NativeObjectiveError(
            "SO(3) composition must occupy one contiguous three-dimension run "
            "with one multiplication convention"
        )
    so3_dimensions = consumed[:, None]
    return (
        torch.where(so3_dimensions, so3_result, result),
        torch.where(so3_dimensions, so3_mask, result_mask),
    )


def compute_native_objective(
    *,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: NativeObjectiveConfig,
    perceptual_model: nn.Module | None = None,
    rgb_perceptual_chunk_size: int = 4,
) -> dict[str, torch.Tensor]:
    """Compute finite, mask-aware Stage0 world and policy losses."""

    config.validate()
    epsilon = config.epsilon
    target_tokens = batch["target_tokens"]
    token_mask = batch.get(
        "target_token_mask", torch.ones_like(target_tokens[..., 0], dtype=torch.bool)
    )
    token_error = output["pred_tokens"] - target_tokens
    token_mse = _masked_mean(token_error.square(), token_mask[..., None], epsilon=epsilon)
    cosine = 1.0 - F.cosine_similarity(
        output["pred_tokens"].float(), target_tokens.float(), dim=-1
    )
    token_cosine = _masked_mean(cosine, token_mask, epsilon=epsilon)

    zero = token_mse.new_zeros(())
    zero_action_token_mse = zero
    action_counterfactual_token_gain = zero
    action_counterfactual_token_advantage = zero
    action_counterfactual_token_response_rms = zero
    if config.action_counterfactual_token_advantage > 0.0:
        if "zero_action_pred_tokens" not in output:
            raise NativeObjectiveError(
                "token counterfactual requires zero_action_pred_tokens"
            )
        (
            zero_action_token_mse,
            action_counterfactual_token_gain,
            action_counterfactual_token_advantage,
            action_counterfactual_token_response_rms,
        ) = _factual_zero_advantage(
            factual=output["pred_tokens"],
            zero_action=output["zero_action_pred_tokens"],
            target=target_tokens,
            mask=token_mask[..., None],
            margin=config.action_counterfactual_token_margin,
            epsilon=epsilon,
            absolute_error=False,
        )
    appearance_l1 = zero
    appearance_teacher_l1 = zero
    appearance_autoregressive_l1 = zero
    appearance_motion_l1 = zero
    appearance_mse = zero
    appearance_cosine = zero
    appearance_motion_mse = zero
    appearance_delta_cosine = zero
    appearance_motion_fraction = zero
    appearance_supervised = zero
    has_appearance = (
        "appearance_pred_tokens" in output
        or "target_appearance_tokens" in batch
    )
    appearance_loss_enabled = any(
        weight > 0.0
        for weight in (
            config.appearance_l1,
            config.appearance_teacher_l1,
            config.appearance_autoregressive_l1,
            config.appearance_motion_l1,
            config.appearance_mse,
            config.appearance_cosine,
            config.appearance_motion_mse,
            config.appearance_delta_cosine,
        )
    )
    if appearance_loss_enabled and not has_appearance:
        raise NativeObjectiveError(
            "appearance loss is enabled but the model/data provide no appearance lane"
        )
    if has_appearance:
        if (
            "appearance_pred_tokens" not in output
            or "target_appearance_tokens" not in batch
            or "target_appearance_mask" not in batch
        ):
            raise NativeObjectiveError(
                "appearance prediction, target and mask must be provided together"
            )
        appearance_prediction = output["appearance_pred_tokens"]
        raw_appearance_target = batch["target_appearance_tokens"]
        normalized_appearance_target = F.layer_norm(
            raw_appearance_target.float(), (raw_appearance_target.shape[-1],)
        ).to(dtype=appearance_prediction.dtype)
        appearance_mask = batch["target_appearance_mask"].bool()
        if "appearance_pred_mask" in output:
            appearance_mask = appearance_mask & output["appearance_pred_mask"].bool()
        flow_aligned_detail = "appearance_flow_aligned_detail" in output
        state_detail = "appearance_state_detail" in output
        if state_detail:
            if (
                appearance_prediction.shape[:-1]
                != raw_appearance_target.shape[:-1]
                or raw_appearance_target.shape[-1]
                % appearance_prediction.shape[-1]
            ):
                raise NativeObjectiveError(
                    "state detail prediction is incompatible with P256 targets"
                )
            channel_group = (
                raw_appearance_target.shape[-1]
                // appearance_prediction.shape[-1]
            )
            compressed_target = normalized_appearance_target.reshape(
                *normalized_appearance_target.shape[:-1],
                appearance_prediction.shape[-1],
                channel_group,
            ).mean(dim=-1)
            detail_patches = int(compressed_target.shape[-2])
            detail_grid = isqrt(detail_patches)
            state_patches = int(output["pred_tokens"].shape[-2])
            state_grid = isqrt(state_patches)
            if (
                detail_grid * detail_grid != detail_patches
                or state_grid * state_grid != state_patches
            ):
                raise NativeObjectiveError(
                    "state/P256 detail targets require square spatial grids"
                )
            target_map = compressed_target.reshape(
                -1,
                detail_grid,
                detail_grid,
                appearance_prediction.shape[-1],
            ).permute(0, 3, 1, 2)
            low_frequency = F.interpolate(
                F.interpolate(
                    target_map,
                    size=(state_grid, state_grid),
                    mode="area",
                ),
                size=(detail_grid, detail_grid),
                mode="bilinear",
                align_corners=False,
            )
            appearance_target = (
                target_map - low_frequency
            ).permute(0, 2, 3, 1).reshape_as(appearance_prediction)
            appearance_target = appearance_target * appearance_mask[..., None].to(
                dtype=appearance_target.dtype
            )
        elif flow_aligned_detail:
            if appearance_prediction.shape != raw_appearance_target.shape:
                raise NativeObjectiveError(
                    "appearance prediction/target shapes differ"
                )
            required = (
                "appearance_context_tokens",
                "appearance_context_mask",
                "rgb_flow_target_pixels",
                "target_rgb",
            )
            missing = [name for name in required if name not in batch]
            if missing:
                raise NativeObjectiveError(
                    "flow-aligned appearance detail requires " + ", ".join(missing)
                )
            context_tokens = batch["appearance_context_tokens"]
            context_mask = batch["appearance_context_mask"].bool()
            if context_mask.shape != context_tokens.shape[:-1]:
                raise NativeObjectiveError(
                    "appearance context mask must align to context tokens"
                )
            normalized_context = F.layer_norm(
                context_tokens.float(), (context_tokens.shape[-1],)
            ).to(dtype=appearance_prediction.dtype)
            latest_context = torch.zeros_like(normalized_context[:, 0])
            for index in range(int(normalized_context.shape[1])):
                latest_context = torch.where(
                    context_mask[:, index, ..., None],
                    normalized_context[:, index],
                    latest_context,
                )
            latest_context = latest_context[:, None].expand_as(
                normalized_appearance_target
            )
            target_rgb = batch["target_rgb"]
            aligned_context = _warp_token_grid_with_pixel_flow(
                latest_context,
                batch["rgb_flow_target_pixels"],
                image_height=int(target_rgb.shape[-2]),
                image_width=int(target_rgb.shape[-1]),
            )
            appearance_target = normalized_appearance_target - aligned_context
            appearance_target = appearance_target * appearance_mask[..., None].to(
                dtype=appearance_target.dtype
            )
        else:
            if appearance_prediction.shape != raw_appearance_target.shape:
                raise NativeObjectiveError(
                    "appearance prediction/target shapes differ"
                )
            appearance_target = normalized_appearance_target
        appearance_error = appearance_prediction - appearance_target
        appearance_l1 = _masked_mean(
            appearance_error.abs(), appearance_mask[..., None], epsilon=epsilon
        )
        appearance_mse = _masked_mean(
            appearance_error.square(), appearance_mask[..., None], epsilon=epsilon
        )
        appearance_cosine = _masked_mean(
            1.0 - F.cosine_similarity(
                appearance_prediction.float(), appearance_target.float(), dim=-1
            ),
            appearance_mask,
            epsilon=epsilon,
        )
        appearance_supervised = torch.broadcast_to(
            appearance_mask[..., None], appearance_target.shape
        ).sum().to(dtype=token_mse.dtype)

        teacher_keys = (
            "appearance_teacher_pred_tokens",
            "appearance_teacher_pred_mask",
        )
        if any(name in output for name in teacher_keys):
            if not all(name in output for name in teacher_keys):
                raise NativeObjectiveError(
                    "teacher-forced appearance prediction and mask must be paired"
                )
            teacher_prediction = output[teacher_keys[0]]
            if teacher_prediction.shape != appearance_target.shape:
                raise NativeObjectiveError(
                    "teacher-forced appearance prediction/target shapes differ"
                )
            teacher_mask = appearance_mask & output[teacher_keys[1]].bool()
            appearance_teacher_l1 = _masked_mean(
                (teacher_prediction - appearance_target).abs(),
                teacher_mask[..., None],
                epsilon=epsilon,
            )
        elif config.appearance_teacher_l1 > 0.0:
            raise NativeObjectiveError(
                "teacher-forced appearance L1 requires its prediction path"
            )

        autoregressive_keys = (
            "appearance_autoregressive_pred_tokens",
            "appearance_autoregressive_pred_mask",
        )
        if any(name in output for name in autoregressive_keys):
            if not all(name in output for name in autoregressive_keys):
                raise NativeObjectiveError(
                    "autoregressive appearance prediction and mask must be paired"
                )
            autoregressive_prediction = output[autoregressive_keys[0]]
            rollout = int(autoregressive_prediction.shape[1])
            if (
                autoregressive_prediction.ndim != appearance_target.ndim
                or rollout <= 0
                or rollout > appearance_target.shape[1]
                or autoregressive_prediction.shape[:1]
                + autoregressive_prediction.shape[2:]
                != appearance_target.shape[:1] + appearance_target.shape[2:]
            ):
                raise NativeObjectiveError(
                    "autoregressive appearance prediction/target shapes differ"
                )
            autoregressive_mask = (
                appearance_mask[:, :rollout]
                & output[autoregressive_keys[1]].bool()
            )
            appearance_autoregressive_l1 = _masked_mean(
                (
                    autoregressive_prediction
                    - appearance_target[:, :rollout]
                ).abs(),
                autoregressive_mask[..., None],
                epsilon=epsilon,
            )
        elif config.appearance_autoregressive_l1 > 0.0:
            raise NativeObjectiveError(
                "autoregressive appearance L1 requires its prediction path"
            )
        if (
            config.appearance_motion_l1 > 0.0
            or config.appearance_motion_mse > 0.0
            or config.appearance_delta_cosine > 0.0
        ):
            required = (
                ("target_rgb", "context_rgb")
                if state_detail
                else (
                    "appearance_context_tokens",
                    "appearance_context_mask",
                    "target_rgb",
                    "context_rgb",
                )
            )
            missing = [name for name in required if name not in batch]
            if missing:
                raise NativeObjectiveError(
                    "motion-aware appearance loss requires " + ", ".join(missing)
                )
            if state_detail:
                context_token_valid = appearance_mask.any(dim=1)
                predicted_delta = appearance_prediction
                target_delta = appearance_target
            else:
                context_tokens = batch["appearance_context_tokens"]
                context_mask = batch["appearance_context_mask"].bool()
                if (
                    context_tokens.ndim != 5
                    or context_tokens.shape[0] != appearance_target.shape[0]
                ):
                    raise NativeObjectiveError(
                        "appearance context tokens must be [B,C,V,P,D]"
                    )
                if context_mask.shape != context_tokens.shape[:-1]:
                    raise NativeObjectiveError(
                        "appearance context mask must align to context tokens"
                    )
                if context_tokens.shape[2:-1] != appearance_target.shape[2:-1]:
                    raise NativeObjectiveError(
                        "appearance context and target view/patch shapes differ"
                    )
                normalized_context = F.layer_norm(
                    context_tokens.float(), (context_tokens.shape[-1],)
                ).to(dtype=appearance_prediction.dtype)
                latest_context = torch.zeros_like(normalized_context[:, 0])
                for index in range(int(normalized_context.shape[1])):
                    latest_context = torch.where(
                        context_mask[:, index, ..., None],
                        normalized_context[:, index],
                        latest_context,
                    )
                context_token_valid = context_mask.any(dim=1)
                if flow_aligned_detail:
                    predicted_delta = appearance_prediction
                    target_delta = appearance_target
                else:
                    predicted_delta = appearance_prediction - latest_context[:, None]
                    target_delta = appearance_target - latest_context[:, None]

            target_rgb = batch["target_rgb"].float()
            context_rgb = batch["context_rgb"].float()
            expected_rgb_prefix = appearance_target.shape[:3]
            if (
                target_rgb.ndim != 6
                or target_rgb.shape[:3] != expected_rgb_prefix
                or target_rgb.shape[3] != 3
            ):
                raise NativeObjectiveError(
                    "target RGB must align to appearance [B,K,V]"
                )
            if context_rgb.shape != (
                appearance_target.shape[0],
                appearance_target.shape[2],
                3,
                target_rgb.shape[-2],
                target_rgb.shape[-1],
            ):
                raise NativeObjectiveError(
                    "context RGB must align to appearance views and target resolution"
                )
            target_rgb_mask = batch.get(
                "target_rgb_mask",
                torch.ones_like(target_rgb[:, :, :, :1, :1, :1], dtype=torch.bool),
            ).bool()
            context_rgb_mask = batch.get(
                "context_rgb_mask",
                torch.ones(
                    appearance_target.shape[0],
                    appearance_target.shape[2],
                    dtype=torch.bool,
                    device=appearance_target.device,
                ),
            ).bool()
            if target_rgb_mask.shape != target_rgb.shape[:3] + (1, 1, 1):
                raise NativeObjectiveError("target RGB mask must be [B,K,V,1,1,1]")
            if context_rgb_mask.shape != (
                appearance_target.shape[:1] + appearance_target.shape[2:3]
            ):
                raise NativeObjectiveError("context RGB mask must be [B,V]")
            rgb_valid = (
                target_rgb_mask
                & context_rgb_mask[:, None, :, None, None, None]
            )
            pixel_motion = (
                (target_rgb - context_rgb[:, None]).abs().mean(dim=3, keepdim=True)
                > config.rgb_motion_threshold
            ) & rgb_valid
            appearance_patches = int(appearance_target.shape[3])
            appearance_grid = isqrt(appearance_patches)
            if appearance_grid * appearance_grid != appearance_patches:
                raise NativeObjectiveError(
                    "motion-aware appearance loss requires a square patch grid"
                )
            motion_weight = F.adaptive_avg_pool2d(
                pixel_motion.float().reshape(
                    -1, 1, target_rgb.shape[-2], target_rgb.shape[-1]
                ),
                output_size=(appearance_grid, appearance_grid),
            ).reshape(*appearance_target.shape[:3], appearance_patches)
            motion_valid = appearance_mask & context_token_valid[:, None]
            motion_weight = motion_weight * motion_valid.to(dtype=motion_weight.dtype)
            appearance_motion_fraction = _masked_mean(
                motion_weight,
                motion_valid,
                epsilon=epsilon,
            )
            appearance_motion_l1 = _masked_mean(
                appearance_error.abs(),
                motion_weight[..., None],
                epsilon=epsilon,
            )
            appearance_motion_mse = _masked_mean(
                appearance_error.square(),
                motion_weight[..., None],
                epsilon=epsilon,
            )
            delta_direction_weight = motion_weight * (
                target_delta.float().square().sum(dim=-1).sqrt() > epsilon
            ).to(dtype=motion_weight.dtype)
            appearance_delta_cosine = _masked_mean(
                1.0
                - F.cosine_similarity(
                    predicted_delta.float(), target_delta.float(), dim=-1
                ),
                delta_direction_weight,
                epsilon=epsilon,
            )
    rgb_l1 = zero
    rgb_charbonnier = zero
    rgb_gradient = zero
    rgb_perceptual = zero
    rgb_motion_l1 = zero
    rgb_motion_region_l1 = zero
    rgb_static_region_l1 = zero
    rgb_motion_bce = zero
    rgb_motion_dice = zero
    rgb_motion_fraction = zero
    rgb_flow_teacher = zero
    rgb_flow_epe = zero
    rgb_flow_moving_epe = zero
    rgb_flow_static_epe = zero
    rgb_flow_prediction_magnitude = zero
    rgb_flow_target_magnitude = zero
    rgb_flow_magnitude_ratio = zero
    rgb_disocclusion_bce = zero
    rgb_disocclusion_dice = zero
    rgb_disocclusion_fraction = zero
    rgb_disocclusion_predicted_fraction = zero
    zero_action_rgb_l1 = zero
    action_counterfactual_rgb_gain = zero
    action_counterfactual_rgb_advantage = zero
    action_counterfactual_rgb_response_rms = zero
    context_pixel_action_rank = zero
    context_pixel_action_separation = zero
    context_pixel_action_acc = zero
    context_pixel_action_gap = zero
    context_pixel_action_rgb_gap = zero
    context_pixel_action_response_rms = zero
    context_pixel_action_negative_distance = zero
    context_pixel_action_valid_fraction = output.get(
        "shuffled_action_valid_fraction", zero
    )
    context_pixel_action_rank_weight = output.get(
        "context_pixel_action_rank_weight", zero
    )
    context_pixel_action_separation_weight = output.get(
        "context_pixel_action_separation_weight", zero
    )
    if "target_rgb" in batch and output["rgb"].numel():
        target_rgb = batch["target_rgb"]
        rgb_mask = batch.get(
            "target_rgb_mask",
            torch.ones_like(target_rgb[:, :, :, :1, :1, :1], dtype=torch.bool),
        )
        rgb_error = output["rgb"] - target_rgb
        motion_objective_enabled = (
            config.rgb_motion_l1 > 0.0
            or config.rgb_motion_bce > 0.0
            or config.rgb_motion_dice > 0.0
        )
        transport_objective_enabled = (
            config.rgb_flow_teacher > 0.0
            or config.rgb_disocclusion_bce > 0.0
            or config.rgb_disocclusion_dice > 0.0
        )
        if "context_rgb" in batch:
            context_rgb = batch["context_rgb"]
            expected_context_shape = (
                target_rgb.shape[0],
                target_rgb.shape[2],
                *target_rgb.shape[3:],
            )
            if tuple(context_rgb.shape) != expected_context_shape:
                raise NativeObjectiveError(
                    "context_rgb must be [B,V,3,H,W] and align to target RGB"
                )
            context_rgb_mask = batch.get(
                "context_rgb_mask",
                torch.ones(
                    target_rgb.shape[0],
                    target_rgb.shape[2],
                    dtype=torch.bool,
                    device=target_rgb.device,
                ),
            ).bool()
            if (
                tuple(context_rgb_mask.shape)
                != target_rgb.shape[:1] + target_rgb.shape[2:3]
            ):
                raise NativeObjectiveError("context_rgb_mask must be [B,V]")
            valid_rgb_mask = (
                rgb_mask.bool()
                & context_rgb_mask[:, None, :, None, None, None]
            )
            expanded_context_rgb = context_rgb[:, None].expand(
                -1, target_rgb.shape[1], -1, -1, -1, -1
            )
            motion_value = (
                target_rgb.float() - expanded_context_rgb.float()
            ).abs().mean(dim=3, keepdim=True)
            motion_mask = (
                motion_value > config.rgb_motion_threshold
            ) & valid_rgb_mask
            static_mask = (~motion_mask) & valid_rgb_mask
            rgb_motion_fraction = _masked_mean(
                motion_mask.float(), valid_rgb_mask, epsilon=epsilon
            )
            rgb_motion_region_l1 = _masked_mean(
                rgb_error.abs(), motion_mask, epsilon=epsilon
            )
            rgb_static_region_l1 = _masked_mean(
                rgb_error.abs(), static_mask, epsilon=epsilon
            )
            motion_weight = 1.0 + config.rgb_motion_gain * motion_mask.to(
                dtype=rgb_error.dtype
            )
            rgb_motion_l1 = _masked_mean(
                rgb_error.abs() * motion_weight,
                valid_rgb_mask,
                epsilon=epsilon,
            )
            if config.rgb_motion_bce > 0.0 or config.rgb_motion_dice > 0.0:
                if "rgb_motion_logit" not in output:
                    raise NativeObjectiveError(
                        "motion-mask supervision requires rgb_motion_logit"
                    )
                motion_logit = output["rgb_motion_logit"].float()
                if motion_logit.shape != motion_mask.shape:
                    raise NativeObjectiveError(
                        "rgb_motion_logit must align to the RGB motion mask"
                    )
                motion_target = motion_mask.to(dtype=motion_logit.dtype)
                motion_valid = torch.broadcast_to(
                    valid_rgb_mask, motion_logit.shape
                )
                if config.rgb_motion_bce > 0.0:
                    motion_bce = F.binary_cross_entropy_with_logits(
                        motion_logit,
                        motion_target,
                        pos_weight=torch.as_tensor(
                            config.rgb_motion_pos_weight,
                            dtype=motion_logit.dtype,
                            device=motion_logit.device,
                        ),
                        reduction="none",
                    )
                    rgb_motion_bce = _masked_mean(
                        motion_bce, motion_valid, epsilon=epsilon
                    )
                if config.rgb_motion_dice > 0.0:
                    motion_weight_valid = motion_valid.to(
                        dtype=motion_logit.dtype
                    )
                    motion_probability = torch.sigmoid(motion_logit)
                    intersection = (
                        motion_probability
                        * motion_target
                        * motion_weight_valid
                    ).flatten(1).sum(dim=1)
                    denominator = (
                        (motion_probability + motion_target)
                        * motion_weight_valid
                    ).flatten(1).sum(dim=1)
                    valid_sample = motion_valid.flatten(1).any(dim=1)
                    dice_per_sample = 1.0 - (
                        (2.0 * intersection + epsilon)
                        / (denominator + epsilon)
                    )
                    rgb_motion_dice = _masked_mean(
                        dice_per_sample, valid_sample, epsilon=epsilon
                    )
            if transport_objective_enabled:
                required_output = (
                    "rgb_flow_pixels",
                    "rgb_disocclusion_logit",
                )
                required_batch = (
                    "rgb_flow_target_pixels",
                    "rgb_disocclusion_target",
                )
                missing = [
                    *(name for name in required_output if name not in output),
                    *(name for name in required_batch if name not in batch),
                ]
                if missing:
                    raise NativeObjectiveError(
                        "RGB transport supervision requires " + ", ".join(missing)
                    )
                flow_prediction = output["rgb_flow_pixels"].float()
                disocclusion_prediction = output[
                    "rgb_disocclusion_logit"
                ].float()
                flow_target = batch["rgb_flow_target_pixels"].float()
                disocclusion_target = batch["rgb_disocclusion_target"].float()
                expected_flow_prefix = target_rgb.shape[:3] + (2,)
                expected_disocclusion_prefix = target_rgb.shape[:3] + (1,)
                if (
                    flow_prediction.ndim != 6
                    or flow_prediction.shape[:4] != expected_flow_prefix
                    or flow_target.ndim != 6
                    or flow_target.shape[:4] != expected_flow_prefix
                ):
                    raise NativeObjectiveError(
                        "RGB flow tensors must align as [B,F,V,2,H,W]"
                    )
                if (
                    disocclusion_prediction.ndim != 6
                    or disocclusion_prediction.shape[:4]
                    != expected_disocclusion_prefix
                    or disocclusion_target.ndim != 6
                    or disocclusion_target.shape[:4]
                    != expected_disocclusion_prefix
                    or disocclusion_target.shape[-2:] != flow_target.shape[-2:]
                ):
                    raise NativeObjectiveError(
                        "RGB disocclusion tensors must align as [B,F,V,1,H,W]"
                    )
                pair_count = int(
                    target_rgb.shape[0]
                    * target_rgb.shape[1]
                    * target_rgb.shape[2]
                )
                teacher_grid = tuple(flow_target.shape[-2:])
                flow_at_teacher_grid = F.interpolate(
                    flow_prediction.reshape(
                        pair_count, 2, *flow_prediction.shape[-2:]
                    ),
                    size=teacher_grid,
                    mode="bilinear",
                    align_corners=True,
                ).reshape_as(flow_target)
                disocclusion_at_teacher_grid = F.interpolate(
                    disocclusion_prediction.reshape(
                        pair_count, 1, *disocclusion_prediction.shape[-2:]
                    ),
                    size=teacher_grid,
                    mode="bilinear",
                    align_corners=True,
                ).reshape_as(disocclusion_target)
                slot_valid = valid_rgb_mask[..., 0, 0, 0]
                transport_valid = torch.broadcast_to(
                    slot_valid[..., None, None, None], disocclusion_target.shape
                )
                finite_flow = torch.isfinite(flow_target).all(
                    dim=3, keepdim=True
                )
                finite_disocclusion = torch.isfinite(disocclusion_target)
                transport_valid = (
                    transport_valid & finite_flow & finite_disocclusion
                )
                visible = transport_valid & (disocclusion_target < 0.5)
                flow_delta = flow_at_teacher_grid - flow_target
                flow_epe_map = torch.sqrt(
                    flow_delta.square().sum(dim=3, keepdim=True)
                    + config.rgb_charbonnier_epsilon**2
                )
                target_flow_magnitude = flow_target.square().sum(
                    dim=3, keepdim=True
                ).sqrt()
                prediction_flow_magnitude = flow_at_teacher_grid.square().sum(
                    dim=3, keepdim=True
                ).sqrt()
                moving = target_flow_magnitude >= 1.0
                # Pair normalized head displacement with a dimensionless
                # teacher loss so its raw-logit gradient is resolution stable.
                # Moving pixels are sparse, so static and moving visible
                # regions receive equal per-sample mass instead of allowing
                # zero-flow background to dominate a global mean.
                flow_range_pixels = 0.5 * float(max(target_rgb.shape[-2:]))
                normalized_flow_epe = flow_epe_map / flow_range_pixels
                moving_visible = visible & moving
                static_visible = visible & ~moving
                rgb_flow_moving_epe = _masked_mean(
                    flow_epe_map, moving_visible, epsilon=epsilon
                )
                rgb_flow_static_epe = _masked_mean(
                    flow_epe_map, static_visible, epsilon=epsilon
                )
                moving_loss = _masked_per_sample_mean(
                    normalized_flow_epe, moving_visible, epsilon=epsilon
                )
                static_loss = _masked_per_sample_mean(
                    normalized_flow_epe, static_visible, epsilon=epsilon
                )
                has_moving = moving_visible.flatten(1).any(dim=1).to(
                    dtype=flow_epe_map.dtype
                )
                has_static = static_visible.flatten(1).any(dim=1).to(
                    dtype=flow_epe_map.dtype
                )
                per_sample_flow_teacher = (
                    moving_loss * has_moving + static_loss * has_static
                ) / (has_moving + has_static).clamp_min(1.0)
                has_visible = (has_moving + has_static) > 0
                rgb_flow_teacher = (
                    per_sample_flow_teacher
                    * has_visible.to(dtype=per_sample_flow_teacher.dtype)
                ).sum() / has_visible.sum().clamp_min(1).to(
                    dtype=per_sample_flow_teacher.dtype
                )
                rgb_flow_epe = _masked_mean(
                    flow_epe_map, visible, epsilon=epsilon
                )
                rgb_flow_prediction_magnitude = _masked_mean(
                    prediction_flow_magnitude, visible, epsilon=epsilon
                )
                rgb_flow_target_magnitude = _masked_mean(
                    target_flow_magnitude, visible, epsilon=epsilon
                )
                rgb_flow_magnitude_ratio = rgb_flow_prediction_magnitude / (
                    rgb_flow_target_magnitude + epsilon
                )
                disocclusion_loss = F.binary_cross_entropy_with_logits(
                    disocclusion_at_teacher_grid,
                    disocclusion_target,
                    pos_weight=torch.as_tensor(
                        config.rgb_disocclusion_pos_weight,
                        dtype=disocclusion_at_teacher_grid.dtype,
                        device=disocclusion_at_teacher_grid.device,
                    ),
                    reduction="none",
                )
                rgb_disocclusion_bce = _masked_mean(
                    disocclusion_loss, transport_valid, epsilon=epsilon
                )
                disocclusion_probability = torch.sigmoid(
                    disocclusion_at_teacher_grid
                )
                valid_float = transport_valid.to(
                    dtype=disocclusion_probability.dtype
                )
                intersection = (
                    disocclusion_probability
                    * disocclusion_target
                    * valid_float
                ).flatten(1).sum(dim=1)
                denominator = (
                    (disocclusion_probability + disocclusion_target)
                    * valid_float
                ).flatten(1).sum(dim=1)
                valid_sample = transport_valid.flatten(1).any(dim=1)
                rgb_disocclusion_dice = _masked_mean(
                    1.0
                    - (2.0 * intersection + epsilon)
                    / (denominator + epsilon),
                    valid_sample,
                    epsilon=epsilon,
                )
                rgb_disocclusion_fraction = _masked_mean(
                    disocclusion_target, transport_valid, epsilon=epsilon
                )
                rgb_disocclusion_predicted_fraction = _masked_mean(
                    disocclusion_probability, transport_valid, epsilon=epsilon
                )
        elif motion_objective_enabled or transport_objective_enabled:
            raise NativeObjectiveError(
                "RGB motion supervision requires context_rgb"
            )
        rgb_l1 = _masked_mean(
            rgb_error.abs(),
            rgb_mask,
            epsilon=epsilon,
        )
        if "shuffled_action_rgb" in output:
            if "context_rgb" not in batch:
                raise NativeObjectiveError(
                    "RGB action ranking requires context_rgb"
                )
            indices = output.get("shuffled_action_indices")
            if indices is None or indices.ndim != 1:
                raise NativeObjectiveError(
                    "shuffled_action_indices must be a one-dimensional tensor"
                )
            indices = indices.to(device=target_rgb.device, dtype=torch.long)
            selected_valid = output.get("shuffled_action_valid")
            if selected_valid is None or tuple(selected_valid.shape) != tuple(
                indices.shape
            ):
                raise NativeObjectiveError(
                    "shuffled_action_valid must align to shuffled_action_indices"
                )
            selected_valid = selected_valid.to(
                device=target_rgb.device, dtype=torch.bool
            )
            factual_rgb = output["rgb"].index_select(0, indices).float()
            wrong_rgb = output["shuffled_action_rgb"].float()
            selected_target = target_rgb.index_select(0, indices).float()
            selected_rgb_mask = rgb_mask.index_select(0, indices).bool()
            selected_context = batch["context_rgb"].index_select(
                0, indices
            ).float()
            selected_context_mask = batch.get(
                "context_rgb_mask",
                torch.ones(
                    target_rgb.shape[0],
                    target_rgb.shape[2],
                    dtype=torch.bool,
                    device=target_rgb.device,
                ),
            ).index_select(0, indices).bool()
            if wrong_rgb.shape != factual_rgb.shape:
                raise NativeObjectiveError(
                    "shuffled-action and factual RGB shapes differ"
                )
            valid = (
                selected_rgb_mask
                & selected_context_mask[:, None, :, None, None, None]
            )
            motion = (
                selected_target - selected_context[:, None]
            ).abs().mean(dim=3, keepdim=True)
            motion_weight = 1.0 + config.context_pixel_action_motion_gain * (
                motion > config.context_pixel_action_motion_threshold
            ).to(dtype=factual_rgb.dtype)

            def action_weighted_l1(
                prediction: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                error = (prediction.float() - target.float()).abs()
                weight = (
                    motion_weight.to(device=error.device, dtype=error.dtype)
                    * valid.to(device=error.device, dtype=error.dtype)
                ).expand_as(error)
                return (error * weight).flatten(1).sum(dim=1) / weight.flatten(
                    1
                ).sum(dim=1).clamp_min(epsilon)

            factual_error = action_weighted_l1(factual_rgb, selected_target)
            wrong_error = action_weighted_l1(wrong_rgb, selected_target)
            gap = wrong_error - factual_error
            separation = action_weighted_l1(wrong_rgb, factual_rgb.detach())
            selected_valid_float = selected_valid.to(dtype=gap.dtype)

            def valid_sample_mean(value: torch.Tensor) -> torch.Tensor:
                return (
                    value * selected_valid_float
                ).sum() / selected_valid_float.sum().clamp_min(1.0)

            context_pixel_action_rank = valid_sample_mean(
                torch.relu(config.context_pixel_action_rank_margin - gap)
            )
            context_pixel_action_separation = valid_sample_mean(
                torch.relu(
                    config.context_pixel_action_separation_margin - separation
                )
            )
            context_pixel_action_acc = valid_sample_mean((gap > 0).float())
            context_pixel_action_gap = valid_sample_mean(gap)
            context_pixel_action_rgb_gap = valid_sample_mean(separation)
            response_per_sample = (
                (wrong_rgb - factual_rgb.detach()).square().flatten(1).mean(dim=1).sqrt()
            )
            context_pixel_action_response_rms = valid_sample_mean(
                response_per_sample
            )
            context_pixel_action_negative_distance = valid_sample_mean(
                output.get("shuffled_action_distance", zero).float()
            )
        if config.action_counterfactual_rgb_advantage > 0.0:
            if "zero_action_rgb" not in output:
                raise NativeObjectiveError(
                    "RGB counterfactual requires zero_action_rgb"
                )
            (
                zero_action_rgb_l1,
                action_counterfactual_rgb_gain,
                action_counterfactual_rgb_advantage,
                action_counterfactual_rgb_response_rms,
            ) = _factual_zero_advantage(
                factual=output["rgb"],
                zero_action=output["zero_action_rgb"],
                target=target_rgb,
                mask=rgb_mask,
                margin=config.action_counterfactual_rgb_margin,
                epsilon=epsilon,
                absolute_error=True,
            )
        rgb_charbonnier = _masked_mean(
            _charbonnier(rgb_error, config.rgb_charbonnier_epsilon),
            rgb_mask,
            epsilon=epsilon,
        )
        pred_dy, pred_dx = _image_gradient(output["rgb"])
        target_dy, target_dx = _image_gradient(target_rgb)
        rgb_gradient = 0.5 * (
            _masked_mean(
                _charbonnier(
                    pred_dy - target_dy, config.rgb_charbonnier_epsilon
                ),
                rgb_mask,
                epsilon=epsilon,
            )
            + _masked_mean(
                _charbonnier(
                    pred_dx - target_dx, config.rgb_charbonnier_epsilon
                ),
                rgb_mask,
                epsilon=epsilon,
            )
        )
        if config.rgb_perceptual > 0:
            if perceptual_model is None:
                raise NativeObjectiveError(
                    "rgb_perceptual is enabled but no perceptual model was provided"
                )
            rgb_perceptual = _masked_rgb_perceptual(
                output["rgb"],
                target_rgb,
                rgb_mask,
                perceptual_model,
                chunk_size=rgb_perceptual_chunk_size,
            )
    elif (
        config.rgb_motion_l1 > 0.0
        or config.rgb_motion_bce > 0.0
        or config.rgb_motion_dice > 0.0
        or config.rgb_flow_teacher > 0.0
        or config.rgb_disocclusion_bce > 0.0
        or config.rgb_disocclusion_dice > 0.0
    ):
        raise NativeObjectiveError(
            "RGB motion objective is enabled but RGB supervision is absent"
        )
    elif config.action_counterfactual_rgb_advantage > 0.0:
        raise NativeObjectiveError(
            "RGB counterfactual is enabled but RGB supervision is absent"
        )

    depth_log = zero
    depth_temporal = zero
    if "target_depth" in batch:
        depth_mask = batch.get(
            "target_depth_mask", torch.isfinite(batch["target_depth"]) & (batch["target_depth"] > 0)
        )
        depth_log = _masked_mean(
            (
                torch.log(output["depth"].clamp_min(epsilon))
                - torch.log(batch["target_depth"].clamp_min(epsilon))
            ).abs(),
            depth_mask,
            epsilon=epsilon,
        )
        if config.depth_temporal > 0.0 and output["depth"].shape[1] >= 2:
            depth_valid_mask = depth_mask.bool()
            predicted_depth = _normalise_depth_per_frame(
                output["depth"], depth_valid_mask, epsilon=epsilon
            )
            target_depth = _normalise_depth_per_frame(
                batch["target_depth"], depth_valid_mask, epsilon=epsilon
            )
            depth_pair_mask = depth_valid_mask[:, 1:] & depth_valid_mask[:, :-1]
            predicted_depth_delta = predicted_depth[:, 1:] - predicted_depth[:, :-1]
            target_depth_delta = target_depth[:, 1:] - target_depth[:, :-1]
            depth_temporal = _masked_mean(
                (predicted_depth_delta - target_depth_delta).abs(),
                depth_pair_mask,
                epsilon=epsilon,
            )
    elif config.depth_temporal > 0.0:
        raise NativeObjectiveError(
            "depth_temporal requires target_depth supervision"
        )

    point = zero
    point_temporal = zero
    if "target_point" in batch:
        point_mask = batch.get(
            "target_point_mask", torch.isfinite(batch["target_point"]).all(dim=-1)
        )
        point = _masked_mean(
            F.smooth_l1_loss(
                output["point"], batch["target_point"], reduction="none", beta=config.huber_delta
            ),
            point_mask[..., None],
            epsilon=epsilon,
        )
        if config.point_temporal > 0.0 and output["point"].shape[1] >= 2:
            point_valid_mask = point_mask.bool()
            predicted_point = _normalise_points_per_window(
                output["point"], point_valid_mask, epsilon=epsilon
            )
            target_point = _normalise_points_per_window(
                batch["target_point"], point_valid_mask, epsilon=epsilon
            )
            point_pair_mask = point_valid_mask[:, 1:] & point_valid_mask[:, :-1]
            predicted_point_delta = predicted_point[:, 1:] - predicted_point[:, :-1]
            target_point_delta = target_point[:, 1:] - target_point[:, :-1]
            point_temporal = _masked_mean(
                F.smooth_l1_loss(
                    predicted_point_delta,
                    target_point_delta,
                    reduction="none",
                    beta=config.huber_delta,
                ),
                point_pair_mask[..., None],
                epsilon=epsilon,
            )
    elif config.point_temporal > 0.0:
        raise NativeObjectiveError(
            "point_temporal requires target_point supervision"
        )

    camera_pose = zero
    if "target_camera_pose" in batch:
        camera_mask = batch.get(
            "target_camera_pose_mask",
            torch.isfinite(batch["target_camera_pose"]).all(dim=-1),
        )
        camera_pose = _masked_mean(
            F.smooth_l1_loss(
                output["camera_pose"],
                batch["target_camera_pose"],
                reduction="none",
                beta=config.huber_delta,
            ),
            camera_mask[..., None],
            epsilon=epsilon,
        )

    fine_target = batch["target_fine_action"]
    fine_mask = batch["target_fine_action_mask"] & output["policy_action_mask"]
    binary_mask = output.get("policy_binary_mask", output["policy_gripper_mask"])
    continuous_mask = fine_mask & ~binary_mask
    gripper_mask = fine_mask & binary_mask
    fine_continuous = _masked_mean(
        F.smooth_l1_loss(
            output["policy_action_normalized"],
            fine_target,
            reduction="none",
            beta=config.huber_delta,
        ),
        continuous_mask,
        epsilon=epsilon,
    )
    fine_gripper = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["policy_action_raw"], fine_target.clamp(0, 1), reduction="none"
        ),
        gripper_mask,
        epsilon=epsilon,
    )
    action_fine = fine_continuous + fine_gripper

    composed, composed_mask = compose_policy_to_world_intervals(
        policy_action=output["policy_action"],
        policy_action_mask=output["policy_action_mask"],
        policy_query_dt=output["policy_query_dt"],
        future_world_boundaries_dt=batch["future_world_boundaries_dt"],
        composition_operator_ids=batch["composition_operator_ids"],
        epsilon=epsilon,
    )
    coarse_mask = composed_mask & batch["target_coarse_action_mask"]
    normalization_offset = batch["action_normalization_offset"]
    normalization_scale = batch["action_normalization_scale"]
    if (
        normalization_offset.shape != normalization_scale.shape
        or tuple(normalization_offset.shape)
        != (composed.shape[0], composed.shape[2], composed.shape[3])
        or not bool(torch.isfinite(normalization_offset).all())
        or not bool(torch.isfinite(normalization_scale).all())
        or bool((normalization_scale <= 0).any())
    ):
        raise NativeObjectiveError("action normalization tensors are invalid")
    coarse_normalized = (
        composed - normalization_offset[:, None]
    ) / normalization_scale[:, None]
    action_coarse = _masked_mean(
        F.smooth_l1_loss(
            coarse_normalized,
            batch["target_coarse_action_normalized"],
            reduction="none",
            beta=config.huber_delta,
        ),
        coarse_mask,
        epsilon=epsilon,
    )

    action_velocity = zero

    losses = {
        "token_mse": token_mse,
        "token_cosine": token_cosine,
        "zero_action_token_mse": zero_action_token_mse,
        "action_counterfactual_token_gain": action_counterfactual_token_gain,
        "action_counterfactual_token_advantage": (
            action_counterfactual_token_advantage
        ),
        "action_counterfactual_token_response_rms": (
            action_counterfactual_token_response_rms
        ),
        "appearance_l1": appearance_l1,
        "appearance_teacher_l1": appearance_teacher_l1,
        "appearance_autoregressive_l1": appearance_autoregressive_l1,
        "appearance_motion_l1": appearance_motion_l1,
        "appearance_mse": appearance_mse,
        "appearance_cosine": appearance_cosine,
        "appearance_motion_mse": appearance_motion_mse,
        "appearance_delta_cosine": appearance_delta_cosine,
        "appearance_motion_fraction": appearance_motion_fraction,
        "appearance_teacher_ratio": output.get("appearance_teacher_ratio", zero),
        "rgb_l1": rgb_l1,
        "rgb_charbonnier": rgb_charbonnier,
        "rgb_gradient": rgb_gradient,
        "rgb_perceptual": rgb_perceptual,
        "rgb_motion_l1": rgb_motion_l1,
        "rgb_motion_region_l1": rgb_motion_region_l1,
        "rgb_static_region_l1": rgb_static_region_l1,
        "rgb_motion_bce": rgb_motion_bce,
        "rgb_motion_dice": rgb_motion_dice,
        "rgb_motion_fraction": rgb_motion_fraction,
        "rgb_flow_teacher": rgb_flow_teacher,
        "rgb_flow_epe": rgb_flow_epe,
        "rgb_flow_moving_epe": rgb_flow_moving_epe,
        "rgb_flow_static_epe": rgb_flow_static_epe,
        "rgb_flow_prediction_magnitude": rgb_flow_prediction_magnitude,
        "rgb_flow_target_magnitude": rgb_flow_target_magnitude,
        "rgb_flow_magnitude_ratio": rgb_flow_magnitude_ratio,
        "rgb_disocclusion_bce": rgb_disocclusion_bce,
        "rgb_disocclusion_dice": rgb_disocclusion_dice,
        "rgb_disocclusion_fraction": rgb_disocclusion_fraction,
        "rgb_disocclusion_predicted_fraction": (
            rgb_disocclusion_predicted_fraction
        ),
        "zero_action_rgb_l1": zero_action_rgb_l1,
        "action_counterfactual_rgb_gain": action_counterfactual_rgb_gain,
        "action_counterfactual_rgb_advantage": (
            action_counterfactual_rgb_advantage
        ),
        "action_counterfactual_rgb_response_rms": (
            action_counterfactual_rgb_response_rms
        ),
        "context_pixel_action_rank": context_pixel_action_rank,
        "context_pixel_action_separation": context_pixel_action_separation,
        "context_pixel_action_acc": context_pixel_action_acc,
        "context_pixel_action_gap": context_pixel_action_gap,
        "context_pixel_action_rgb_gap": context_pixel_action_rgb_gap,
        "context_pixel_action_response_rms": context_pixel_action_response_rms,
        "context_pixel_action_negative_distance": (
            context_pixel_action_negative_distance
        ),
        "context_pixel_action_valid_fraction": (
            context_pixel_action_valid_fraction
        ),
        "context_pixel_action_rank_weight": context_pixel_action_rank_weight,
        "context_pixel_action_separation_weight": (
            context_pixel_action_separation_weight
        ),
        "depth_log": depth_log,
        "depth_temporal": depth_temporal,
        "point": point,
        "point_temporal": point_temporal,
        "camera_pose": camera_pose,
        "action_fine": action_fine,
        "action_fine_continuous": fine_continuous,
        "action_fine_gripper": fine_gripper,
        "action_coarse": action_coarse,
        "action_velocity": action_velocity,
        "fine_supervised_dimensions": fine_mask.sum().to(dtype=token_mse.dtype),
        "fine_continuous_supervised_dimensions": continuous_mask.sum().to(
            dtype=token_mse.dtype
        ),
        "fine_binary_supervised_dimensions": gripper_mask.sum().to(
            dtype=token_mse.dtype
        ),
        "coarse_supervised_dimensions": coarse_mask.sum().to(dtype=token_mse.dtype),
        "current_state_supervised_dimensions": (
            batch["current_state_mask"].sum().to(dtype=token_mse.dtype)
            if "current_state_mask" in batch
            else zero
        ),
        "native_token_supervised_elements": torch.broadcast_to(
            token_mask[..., None], target_tokens.shape
        ).sum().to(dtype=token_mse.dtype),
        "appearance_supervised_elements": appearance_supervised,
        "rgb_supervised_elements": (
            torch.broadcast_to(rgb_mask, target_rgb.shape).sum().to(dtype=token_mse.dtype)
            if "target_rgb" in batch and output["rgb"].numel()
            else zero
        ),
        "depth_supervised_elements": (
            depth_mask.sum().to(dtype=token_mse.dtype)
            if "target_depth" in batch
            else zero
        ),
        "point_supervised_elements": (
            torch.broadcast_to(point_mask[..., None], batch["target_point"].shape)
            .sum()
            .to(dtype=token_mse.dtype)
            if "target_point" in batch
            else zero
        ),
        "camera_pose_supervised_elements": (
            torch.broadcast_to(
                camera_mask[..., None], batch["target_camera_pose"].shape
            )
            .sum()
            .to(dtype=token_mse.dtype)
            if "target_camera_pose" in batch
            else zero
        ),
    }
    total = (
        config.token_mse * token_mse
        + config.token_cosine * token_cosine
        + config.appearance_l1 * appearance_l1
        + config.appearance_teacher_l1 * appearance_teacher_l1
        + config.appearance_autoregressive_l1 * appearance_autoregressive_l1
        + config.appearance_motion_l1 * appearance_motion_l1
        + config.appearance_mse * appearance_mse
        + config.appearance_cosine * appearance_cosine
        + config.appearance_motion_mse * appearance_motion_mse
        + config.appearance_delta_cosine * appearance_delta_cosine
        + config.rgb_l1 * rgb_l1
        + config.rgb_charbonnier * rgb_charbonnier
        + config.rgb_gradient * rgb_gradient
        + config.rgb_perceptual * rgb_perceptual
        + config.rgb_motion_l1 * rgb_motion_l1
        + config.rgb_motion_bce * rgb_motion_bce
        + config.rgb_motion_dice * rgb_motion_dice
        + config.rgb_flow_teacher * rgb_flow_teacher
        + config.rgb_disocclusion_bce * rgb_disocclusion_bce
        + config.rgb_disocclusion_dice * rgb_disocclusion_dice
        + config.depth_log * depth_log
        + config.depth_temporal * depth_temporal
        + config.point * point
        + config.point_temporal * point_temporal
        + config.camera_pose * camera_pose
        + config.action_fine * action_fine
        + config.action_coarse * action_coarse
        + config.action_velocity * action_velocity
        + config.action_counterfactual_token_advantage
        * action_counterfactual_token_advantage
        + config.action_counterfactual_rgb_advantage
        * action_counterfactual_rgb_advantage
        + context_pixel_action_rank_weight * context_pixel_action_rank
        + context_pixel_action_separation_weight
        * context_pixel_action_separation
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("WM3D native objective is non-finite")
    losses["total"] = total
    return losses
