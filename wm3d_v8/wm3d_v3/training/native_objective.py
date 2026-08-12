"""Unified native-world and grouped-policy objectives for WM3D V8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn.functional as F

from wm3d_v3.data.grouped_robot import COMPOSITION_OPERATOR_IDS


class NativeObjectiveError(ValueError):
    pass


@dataclass(frozen=True)
class NativeObjectiveConfig:
    token_mse: float = 1.0
    token_cosine: float = 0.1
    rgb_charbonnier: float = 2.0
    rgb_gradient: float = 0.5
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

    # Mixed-embodiment batches may use different physical operators in the
    # same padded group/dimension slot.  The loop is over small metadata axes;
    # all action arithmetic remains differentiable torch operations.
    for sample in range(batch):
        for interval in range(horizon):
            start = future_world_boundaries_dt[sample, interval]
            stop = future_world_boundaries_dt[sample, interval + 1]
            in_interval = (policy_query_dt[sample] >= start) & (
                policy_query_dt[sample] < stop
            )
            for group in range(groups):
                group_interval = in_interval[group]
                dim = 0
                while dim < action_dim:
                    operator = int(
                        composition_operator_ids[sample, group, dim].item()
                    )
                    if operator == operator_none:
                        dim += 1
                        continue
                    if operator in {operator_so3_left, operator_so3_right}:
                        if dim + 3 > action_dim or not bool(
                            composition_operator_ids[
                                sample, group, dim : dim + 3
                            ]
                            .eq(operator)
                            .all()
                        ):
                            raise NativeObjectiveError(
                                "SO(3) composition must occupy one contiguous "
                                "three-dimension run with one multiplication convention"
                            )
                        value = policy_action[
                            sample, group, :, dim : dim + 3
                        ].unsqueeze(0)
                        valid = (
                            group_interval
                            & policy_action_mask[
                                sample, group, :, dim : dim + 3
                            ].all(dim=-1)
                        ).unsqueeze(0)
                        composed, composed_valid = compose_axis_angle_sequence(
                            value,
                            valid,
                            left_multiply=(operator == operator_so3_left),
                            epsilon=epsilon,
                        )
                        result[
                            sample, interval, group, dim : dim + 3
                        ] = composed[0]
                        result_mask[
                            sample, interval, group, dim : dim + 3
                        ] = composed_valid[0]
                        dim += 3
                        continue

                    value = policy_action[sample, group, :, dim]
                    valid = (
                        group_interval
                        & policy_action_mask[sample, group, :, dim]
                    )
                    if operator == operator_sum:
                        composed = (value * valid.to(value.dtype)).sum()
                    elif operator in {operator_last, operator_logical_last}:
                        indices = torch.arange(queries, device=value.device)
                        last_index = torch.where(valid, indices, -1).max()
                        composed = torch.where(
                            last_index >= 0,
                            value[last_index.clamp_min(0)],
                            value.new_zeros(()),
                        )
                    elif operator == operator_mean:
                        # Commands are zero-order held until the next command or
                        # interval end.  Padding/invalid query slots are not real
                        # clock events and therefore cannot truncate the hold.
                        # This uses actual query times, not nominal Hz.
                        times = policy_query_dt[sample, group]
                        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
                        next_times = torch.full_like(times, stop)
                        if valid_indices.numel() > 1:
                            next_times[valid_indices[:-1]] = times[valid_indices[1:]]
                        effective_stop = torch.minimum(next_times, stop)
                        effective_start = torch.maximum(times, start)
                        duration = (effective_stop - effective_start).clamp_min(0.0)
                        duration = duration * valid.to(duration.dtype)
                        composed = (value * duration).sum() / duration.sum().clamp_min(
                            epsilon
                        )
                    else:
                        raise NativeObjectiveError(
                            f"unsupported composition operator id {operator}"
                        )
                    result[sample, interval, group, dim] = composed
                    result_mask[sample, interval, group, dim] = valid.any()
                    dim += 1
    return result, result_mask


def compute_native_objective(
    *,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: NativeObjectiveConfig,
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
    rgb_charbonnier = zero
    rgb_gradient = zero
    if "target_rgb" in batch and output["rgb"].numel():
        target_rgb = batch["target_rgb"]
        rgb_mask = batch.get(
            "target_rgb_mask",
            torch.ones_like(target_rgb[:, :, :, :1, :1, :1], dtype=torch.bool),
        )
        rgb_charbonnier = _masked_mean(
            _charbonnier(output["rgb"] - target_rgb, epsilon),
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
        "rgb_charbonnier": rgb_charbonnier,
        "rgb_gradient": rgb_gradient,
        "depth_log": depth_log,
        "point": point,
        "camera_pose": camera_pose,
        "action_fine": action_fine,
        "action_fine_continuous": fine_continuous,
        "action_fine_gripper": fine_gripper,
        "action_coarse": action_coarse,
        "action_velocity": action_velocity,
        "fine_supervised_dimensions": fine_mask.sum().to(dtype=token_mse.dtype),
        "coarse_supervised_dimensions": coarse_mask.sum().to(dtype=token_mse.dtype),
    }
    total = (
        config.token_mse * token_mse
        + config.token_cosine * token_cosine
        + config.rgb_charbonnier * rgb_charbonnier
        + config.rgb_gradient * rgb_gradient
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
