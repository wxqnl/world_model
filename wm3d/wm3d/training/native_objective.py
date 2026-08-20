"""Unified native-world and grouped-policy objectives for WM3D."""

from __future__ import annotations

from dataclasses import dataclass
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
    appearance_mse: float = 0.0
    appearance_cosine: float = 0.0
    rgb_l1: float = 0.0
    rgb_charbonnier: float = 2.0
    rgb_gradient: float = 0.5
    rgb_perceptual: float = 0.0
    depth_log: float = 1.5
    point: float = 0.5
    camera_pose: float = 0.1
    action_fine: float = 2.0
    action_coarse: float = 1.0
    action_velocity: float = 0.0
    epsilon: float = 1.0e-6
    huber_delta: float = 0.05

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if name in {"epsilon", "huber_delta"}:
                if value <= 0:
                    raise NativeObjectiveError(f"{name} must be positive")
            elif value < 0:
                raise NativeObjectiveError(f"{name} cannot be negative")
        if self.action_velocity != 0.0:
            raise NativeObjectiveError(
                "action_velocity must remain 0: mixed delta/absolute semantics and "
                "source-native cadences have no shared physical smoothness invariant"
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
    for start in range(0, int(valid.numel()), chunk_size):
        pred_chunk = pred_images[start : start + chunk_size].float().mul(2.0).sub(1.0)
        target_chunk = target_images[start : start + chunk_size].float().mul(2.0).sub(1.0)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
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
    appearance_mse = zero
    appearance_cosine = zero
    appearance_supervised = zero
    has_appearance = (
        "appearance_pred_tokens" in output
        or "target_appearance_tokens" in batch
    )
    if (
        config.appearance_mse > 0.0 or config.appearance_cosine > 0.0
    ) and not has_appearance:
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
        appearance_target = batch["target_appearance_tokens"]
        if appearance_prediction.shape != appearance_target.shape:
            raise NativeObjectiveError("appearance prediction/target shapes differ")
        appearance_mask = batch["target_appearance_mask"].bool()
        if "appearance_pred_mask" in output:
            appearance_mask = appearance_mask & output["appearance_pred_mask"].bool()
        appearance_error = appearance_prediction - appearance_target
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
    rgb_l1 = zero
    rgb_charbonnier = zero
    rgb_gradient = zero
    rgb_perceptual = zero
    if "target_rgb" in batch and output["rgb"].numel():
        target_rgb = batch["target_rgb"]
        rgb_mask = batch.get(
            "target_rgb_mask",
            torch.ones_like(target_rgb[:, :, :, :1, :1, :1], dtype=torch.bool),
        )
        rgb_error = output["rgb"] - target_rgb
        rgb_l1 = _masked_mean(
            rgb_error.abs(),
            rgb_mask,
            epsilon=epsilon,
        )
        rgb_charbonnier = _masked_mean(
            _charbonnier(rgb_error, epsilon),
            rgb_mask,
            epsilon=epsilon,
        )
        pred_dy, pred_dx = _image_gradient(output["rgb"])
        target_dy, target_dx = _image_gradient(target_rgb)
        rgb_gradient = 0.5 * (
            _masked_mean(
                _charbonnier(pred_dy - target_dy, epsilon),
                rgb_mask,
                epsilon=epsilon,
            )
            + _masked_mean(
                _charbonnier(pred_dx - target_dx, epsilon),
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

    depth_log = zero
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

    point = zero
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
        "appearance_mse": appearance_mse,
        "appearance_cosine": appearance_cosine,
        "appearance_teacher_ratio": output.get("appearance_teacher_ratio", zero),
        "rgb_l1": rgb_l1,
        "rgb_charbonnier": rgb_charbonnier,
        "rgb_gradient": rgb_gradient,
        "rgb_perceptual": rgb_perceptual,
        "depth_log": depth_log,
        "point": point,
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
        + config.appearance_mse * appearance_mse
        + config.appearance_cosine * appearance_cosine
        + config.rgb_l1 * rgb_l1
        + config.rgb_charbonnier * rgb_charbonnier
        + config.rgb_gradient * rgb_gradient
        + config.rgb_perceptual * rgb_perceptual
        + config.depth_log * depth_log
        + config.point * point
        + config.camera_pose * camera_pose
        + config.action_fine * action_fine
        + config.action_coarse * action_coarse
        + config.action_velocity * action_velocity
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("WM3D native objective is non-finite")
    losses["total"] = total
    return losses
