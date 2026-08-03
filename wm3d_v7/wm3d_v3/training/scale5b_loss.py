"""Native world/action objectives for WM3D-V7 5B.

All terms supervise explicit native outputs.  There is no video-generator,
VLA, language-model or latent-3D ownership path in this objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Native5BLossConfig:
    token_mse: float = 1.0
    token_cosine: float = 0.1
    rgb_charbonnier: float = 1.0
    rgb_gradient: float = 0.25
    rgb_laplacian: float = 0.1
    depth_log: float = 1.0
    depth_gradient: float = 0.15
    point: float = 0.25
    geometry_confidence: float = 0.05
    camera_pose: float = 0.05
    action_nll: float = 1.0
    action_velocity: float = 0.15
    action_contact: float = 0.2
    action_log_scale_reg: float = 1.0e-4
    epsilon: float = 1.0e-6

    @classmethod
    def from_mapping(cls, value: Mapping[str, float]) -> "Native5BLossConfig":
        return cls(**{str(key): float(item) for key, item in value.items()})


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    if mask is None:
        return values.float().mean()
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = torch.broadcast_to(weights, values.shape)
    denominator = weights.float().sum().clamp_min(float(epsilon))
    return (values.float() * weights.float()).sum() / denominator


def _image_gradient(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return image[..., :, 1:] - image[..., :, :-1], image[..., 1:, :] - image[
        ..., :-1, :
    ]


def _image_laplacian(image: torch.Tensor) -> torch.Tensor:
    flat = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
    channels = flat.shape[1]
    kernel = image.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3)
    value = F.conv2d(flat, kernel, padding=1, groups=channels)
    return value.view_as(image)


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"non-finite native5b loss input: {name}")


def native5b_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    cfg: Native5BLossConfig,
) -> dict[str, torch.Tensor]:
    """Compute weighted native world/action losses and fail closed on NaN/Inf."""

    required_output = {
        "pred_tokens",
        "rgb",
        "rgb_frame_indices",
        "depth",
        "point",
        "geometry_confidence",
        "camera_pose",
        "action_mean",
        "action_log_scale",
        "contact_logit",
    }
    required_batch = {
        "target_tokens",
        "target_rgb",
        "target_view_mask",
        "target_depth",
        "target_point",
        "target_geometry_confidence",
        "target_camera_pose",
        "target_action_values",
        "target_action_dim_mask",
        "target_contact",
        "target_contact_mask",
        "action_group_mask",
    }
    missing_output = required_output.difference(output)
    missing_batch = required_batch.difference(batch)
    if missing_output or missing_batch:
        raise KeyError(
            f"native5b loss fields missing output={sorted(missing_output)} "
            f"batch={sorted(missing_batch)}"
        )
    for name in required_output:
        _validate_finite(name, output[name])

    epsilon = cfg.epsilon
    pred_tokens = output["pred_tokens"]
    target_tokens = batch["target_tokens"].to(dtype=pred_tokens.dtype)
    if pred_tokens.shape != target_tokens.shape:
        raise ValueError(
            f"token target shape {target_tokens.shape} != prediction {pred_tokens.shape}"
        )
    token_mse = F.mse_loss(pred_tokens.float(), target_tokens.float())
    token_cosine = (
        1.0 - F.cosine_similarity(pred_tokens.float(), target_tokens.float(), dim=-1)
    ).mean()

    rgb = output["rgb"]
    target_rgb = batch["target_rgb"].to(dtype=rgb.dtype)
    if rgb.shape != target_rgb.shape:
        raise ValueError(
            f"RGB target shape {target_rgb.shape} != prediction {rgb.shape}"
        )
    target_view_mask = batch["target_view_mask"].bool()
    rgb_indices = output["rgb_frame_indices"].to(
        device=target_view_mask.device,
        dtype=torch.long,
    )
    rgb_view_mask = target_view_mask.index_select(1, rgb_indices)
    if tuple(rgb_view_mask.shape) != tuple(rgb.shape[:3]):
        raise ValueError("RGB view mask does not match decoded supervision")
    rgb_difference = rgb.float() - target_rgb.float()
    rgb_charbonnier = _masked_mean(
        torch.sqrt(rgb_difference.square() + epsilon),
        rgb_view_mask,
        epsilon,
    )
    pred_dx, pred_dy = _image_gradient(rgb.float())
    target_dx, target_dy = _image_gradient(target_rgb.float())
    rgb_gradient = 0.5 * (
        _masked_mean((pred_dx - target_dx).abs(), rgb_view_mask, epsilon)
        + _masked_mean((pred_dy - target_dy).abs(), rgb_view_mask, epsilon)
    )
    rgb_laplacian = _masked_mean(
        (_image_laplacian(rgb.float()) - _image_laplacian(target_rgb.float())).abs(),
        rgb_view_mask,
        epsilon,
    )

    depth = output["depth"].float().clamp_min(epsilon)
    target_depth = batch["target_depth"].float().clamp_min(epsilon)
    geometry_view_mask = target_view_mask[..., None]
    confidence_mask = (
        batch["target_geometry_confidence"].float() > 0
    ) & geometry_view_mask
    if depth.shape != target_depth.shape:
        raise ValueError(
            f"depth target shape {target_depth.shape} != prediction {depth.shape}"
        )
    depth_log_error = depth.log() - target_depth.log()
    depth_log_mean = _masked_mean(depth_log_error, confidence_mask, epsilon)
    depth_log = (
        _masked_mean(depth_log_error.square(), confidence_mask, epsilon)
        - 0.5 * depth_log_mean.square()
    )
    depth_gradient = (depth_log_error[:, 1:] - depth_log_error[:, :-1]).abs()
    depth_gradient = _masked_mean(
        depth_gradient, confidence_mask[:, 1:] & confidence_mask[:, :-1], epsilon
    )

    point = F.smooth_l1_loss(
        output["point"].float(),
        batch["target_point"].float(),
        reduction="none",
        beta=0.01,
    )
    point_loss = _masked_mean(point, confidence_mask, epsilon)
    geometry_confidence = _masked_mean(
        F.binary_cross_entropy(
            output["geometry_confidence"].float().clamp(epsilon, 1.0 - epsilon),
            batch["target_geometry_confidence"].float().clamp(0.0, 1.0),
            reduction="none",
        ),
        geometry_view_mask,
        epsilon,
    )
    camera_pose = _masked_mean(
        F.smooth_l1_loss(
            output["camera_pose"].float(),
            batch["target_camera_pose"].float(),
            beta=0.01,
            reduction="none",
        ),
        target_view_mask,
        epsilon,
    )

    action_mean = output["action_mean"].float()
    action_log_scale = output["action_log_scale"].float()
    target_action = batch["target_action_values"].float()
    action_mask = batch["target_action_dim_mask"].bool()
    group_mask = batch["action_group_mask"].bool()[:, None, :, None, None]
    action_mask = action_mask & group_mask
    if (
        action_mean.shape != target_action.shape
        or action_mask.shape != target_action.shape
    ):
        raise ValueError("grouped action prediction/target/mask shapes do not match")
    inverse_variance = torch.exp(-2.0 * action_log_scale)
    action_nll_values = (
        0.5 * (target_action - action_mean).square() * inverse_variance
        + action_log_scale
    )
    action_nll = _masked_mean(action_nll_values, action_mask, epsilon)
    velocity_error = (
        (action_mean[:, 1:] - action_mean[:, :-1])
        - (target_action[:, 1:] - target_action[:, :-1])
    ).abs()
    velocity_mask = action_mask[:, 1:] & action_mask[:, :-1]
    action_velocity = _masked_mean(velocity_error, velocity_mask, epsilon)
    contact_mask = batch["target_contact_mask"].bool()
    contact_mask = contact_mask & batch["action_group_mask"].bool()[:, None, :, None]
    if contact_mask.shape != batch["target_contact"].shape:
        raise ValueError("contact target/mask shapes do not match")
    action_contact = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output["contact_logit"].float(),
            batch["target_contact"].float(),
            reduction="none",
        ),
        contact_mask,
        epsilon,
    )
    action_log_scale_reg = _masked_mean(action_log_scale.square(), action_mask, epsilon)

    raw = {
        "token_mse": token_mse,
        "token_cosine": token_cosine,
        "rgb_charbonnier": rgb_charbonnier,
        "rgb_gradient": rgb_gradient,
        "rgb_laplacian": rgb_laplacian,
        "depth_log": depth_log,
        "depth_gradient": depth_gradient,
        "point": point_loss,
        "geometry_confidence": geometry_confidence,
        "camera_pose": camera_pose,
        "action_nll": action_nll,
        "action_velocity": action_velocity,
        "action_contact": action_contact,
        "action_log_scale_reg": action_log_scale_reg,
    }
    weights = {
        "token_mse": cfg.token_mse,
        "token_cosine": cfg.token_cosine,
        "rgb_charbonnier": cfg.rgb_charbonnier,
        "rgb_gradient": cfg.rgb_gradient,
        "rgb_laplacian": cfg.rgb_laplacian,
        "depth_log": cfg.depth_log,
        "depth_gradient": cfg.depth_gradient,
        "point": cfg.point,
        "geometry_confidence": cfg.geometry_confidence,
        "camera_pose": cfg.camera_pose,
        "action_nll": cfg.action_nll,
        "action_velocity": cfg.action_velocity,
        "action_contact": cfg.action_contact,
        "action_log_scale_reg": cfg.action_log_scale_reg,
    }
    total = sum(raw[name] * float(weights[name]) for name in raw)
    _validate_finite("total", total)
    return {"total": total, **raw}
