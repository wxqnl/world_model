"""Native state/task-to-action objectives for WM3D-v7 Stage 1.

The transferable multi-embodiment contract is ``z_a + pose_norm``.  Physical
metrics are reconstructed with the per-sample source statistics supplied by
the dataset; the single convenience mean/std buffer in ``ActionProjHead`` is
never used for multi-source evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NativeActionLossConfig:
    huber_delta: float = 0.01
    grip_weight: float = 0.75
    grip_positive_weight: float = 1.0
    first_step_weight: float = 0.5
    trajectory_weight: float = 0.2
    translation_direction_weight: float = 0.10
    rotation_direction_weight: float = 0.15
    translation_magnitude_weight: float = 0.20
    rotation_magnitude_weight: float = 0.25
    grip_event_weight: float = 1.0
    first_grip_weight: float = 0.5
    translation_active_threshold_m: float = 1e-4
    rotation_active_threshold_rad: float = 1e-3
    eps: float = 1e-6


def _masked_direction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_norm = target.norm(dim=-1)
    active = target_norm > float(threshold)
    cosine = F.cosine_similarity(predicted, target, dim=-1, eps=1e-6)
    if bool(active.any()):
        loss = (1.0 - cosine[active]).mean()
        metric = cosine[active].mean()
        count = active.sum().to(dtype=predicted.dtype)
    else:
        loss = predicted.sum() * 0.0
        metric = predicted.new_zeros(())
        count = predicted.new_zeros(())
    return loss, metric, count


def _magnitude_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    *,
    threshold: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_norm = predicted.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    active = target_norm > float(threshold)
    vector_scale = scale.norm(dim=-1).clamp_min(float(eps))
    if bool(active.any()):
        pred_scaled = predicted_norm / vector_scale
        target_scaled = target_norm / vector_scale
        loss = F.smooth_l1_loss(
            torch.log1p(pred_scaled[active]),
            torch.log1p(target_scaled[active]),
            beta=0.1,
        )
        ratio = (
            predicted_norm[active] / target_norm[active].clamp_min(float(eps))
        ).mean()
        count = active.sum().to(dtype=predicted.dtype)
    else:
        loss = predicted.sum() * 0.0
        ratio = predicted.new_zeros(())
        count = predicted.new_zeros(())
    return loss, ratio, count


def native_action_loss(
    pose_norm: torch.Tensor,
    gripper_logit: torch.Tensor,
    target_pose_norm: torch.Tensor,
    target_pose_physical: torch.Tensor,
    grip_close01: torch.Tensor,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    *,
    previous_grip_close01: torch.Tensor | None = None,
    cfg: NativeActionLossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Compute action supervision and physical, gain-vs-zero diagnostics."""

    settings = cfg or NativeActionLossConfig()
    if pose_norm.ndim != 3 or pose_norm.shape[-1] != 6:
        raise ValueError("pose_norm must be [B,H,6]")
    if target_pose_norm.shape != pose_norm.shape:
        raise ValueError("target_pose_norm must match pose_norm")
    if target_pose_physical.shape != pose_norm.shape:
        raise ValueError("target_pose_physical must match pose_norm")
    if gripper_logit.shape != pose_norm.shape[:2]:
        raise ValueError("gripper_logit must be [B,H]")
    if grip_close01.shape != gripper_logit.shape:
        raise ValueError("grip_close01 must match gripper_logit")

    batch = pose_norm.shape[0]
    mean = pose_mean.to(device=pose_norm.device, dtype=pose_norm.dtype)
    std = pose_std.to(device=pose_norm.device, dtype=pose_norm.dtype)
    if mean.ndim == 1:
        mean = mean[None].expand(batch, -1)
    if std.ndim == 1:
        std = std[None].expand(batch, -1)
    if mean.shape != (batch, 6) or std.shape != (batch, 6):
        raise ValueError("pose_mean and pose_std must be [B,6] or [6]")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("source action statistics must be finite")
    if not bool((std > 0).all()):
        raise ValueError("source action standard deviations must be positive")

    predicted_physical = pose_norm * std[:, None] + mean[:, None]
    base_pose = F.smooth_l1_loss(
        pose_norm,
        target_pose_norm,
        beta=float(settings.huber_delta),
    )
    first_pose = F.smooth_l1_loss(
        pose_norm[:, 0],
        target_pose_norm[:, 0],
        beta=float(settings.huber_delta),
    )
    cumulative_pred = predicted_physical.cumsum(dim=1) / std[:, None]
    cumulative_target = target_pose_physical.cumsum(dim=1) / std[:, None]
    trajectory = F.smooth_l1_loss(
        cumulative_pred,
        cumulative_target,
        beta=float(settings.huber_delta),
    )

    trans_dir, trans_cos, trans_dir_count = _masked_direction_loss(
        predicted_physical[..., :3],
        target_pose_physical[..., :3],
        threshold=settings.translation_active_threshold_m,
    )
    rot_dir, rot_cos, rot_dir_count = _masked_direction_loss(
        predicted_physical[..., 3:6],
        target_pose_physical[..., 3:6],
        threshold=settings.rotation_active_threshold_rad,
    )
    trans_mag, trans_ratio, trans_mag_count = _magnitude_loss(
        predicted_physical[..., :3],
        target_pose_physical[..., :3],
        std[:, None, :3],
        threshold=settings.translation_active_threshold_m,
        eps=settings.eps,
    )
    rot_mag, rot_ratio, rot_mag_count = _magnitude_loss(
        predicted_physical[..., 3:6],
        target_pose_physical[..., 3:6],
        std[:, None, 3:6],
        threshold=settings.rotation_active_threshold_rad,
        eps=settings.eps,
    )

    target_grip = grip_close01.to(dtype=gripper_logit.dtype)
    grip_raw = F.binary_cross_entropy_with_logits(
        gripper_logit,
        target_grip,
        reduction="none",
        pos_weight=gripper_logit.new_tensor(float(settings.grip_positive_weight)),
    )
    grip = grip_raw.mean()
    first_grip = grip_raw[:, 0].mean()
    target_state = target_grip > 0.5
    event_valid = torch.ones_like(target_state, dtype=torch.bool)
    if previous_grip_close01 is None:
        event_valid[:, 0] = False
        previous_state = torch.cat((target_state[:, :1], target_state[:, :-1]), dim=1)
    else:
        previous = previous_grip_close01.to(
            device=target_state.device, dtype=target_grip.dtype
        ).reshape(batch, -1)[:, -1] > 0.5
        previous_state = torch.cat((previous[:, None], target_state[:, :-1]), dim=1)
    event_mask = (target_state != previous_state) & event_valid
    if bool(event_mask.any()):
        grip_event = grip_raw[event_mask].mean()
    else:
        grip_event = gripper_logit.sum() * 0.0

    total = (
        base_pose
        + float(settings.first_step_weight) * first_pose
        + float(settings.trajectory_weight) * trajectory
        + float(settings.translation_direction_weight) * trans_dir
        + float(settings.rotation_direction_weight) * rot_dir
        + float(settings.translation_magnitude_weight) * trans_mag
        + float(settings.rotation_magnitude_weight) * rot_mag
        + float(settings.grip_weight) * grip
        + float(settings.grip_event_weight) * grip_event
        + float(settings.first_grip_weight) * first_grip
    )

    with torch.no_grad():
        error = (predicted_physical - target_pose_physical).square()
        zero_error = target_pose_physical.square()
        trans_error = error[..., :3].mean()
        rot_error = error[..., 3:6].mean()
        trans_zero = zero_error[..., :3].mean()
        rot_zero = zero_error[..., 3:6].mean()
        trans_gain = 1.0 - trans_error / trans_zero.clamp_min(float(settings.eps))
        rot_gain = 1.0 - rot_error / rot_zero.clamp_min(float(settings.eps))
        predicted_state = gripper_logit > 0
        positive = target_state
        negative = ~target_state
        pos_recall = (
            (predicted_state[positive] == target_state[positive]).float().mean()
            if bool(positive.any())
            else gripper_logit.new_zeros(())
        )
        neg_recall = (
            (predicted_state[negative] == target_state[negative]).float().mean()
            if bool(negative.any())
            else gripper_logit.new_zeros(())
        )
        event_recall = (
            (predicted_state[event_mask] == target_state[event_mask]).float().mean()
            if bool(event_mask.any())
            else gripper_logit.new_zeros(())
        )

    return {
        "loss": total,
        "pose_huber": base_pose,
        "first_pose_huber": first_pose,
        "trajectory_huber": trajectory,
        "translation_direction_loss": trans_dir,
        "rotation_direction_loss": rot_dir,
        "translation_magnitude_loss": trans_mag,
        "rotation_magnitude_loss": rot_mag,
        "grip_bce": grip,
        "first_grip_bce": first_grip,
        "grip_event_bce": grip_event,
        "translation_cosine": trans_cos,
        "rotation_cosine": rot_cos,
        "translation_magnitude_ratio": trans_ratio,
        "rotation_magnitude_ratio": rot_ratio,
        "translation_gain_vs_zero": trans_gain,
        "rotation_gain_vs_zero": rot_gain,
        "grip_positive_recall": pos_recall,
        "grip_negative_recall": neg_recall,
        "grip_balanced_accuracy": 0.5 * (pos_recall + neg_recall),
        "grip_event_recall": event_recall,
        "translation_direction_count": trans_dir_count,
        "rotation_direction_count": rot_dir_count,
        "translation_magnitude_count": trans_mag_count,
        "rotation_magnitude_count": rot_mag_count,
        "grip_event_count": event_mask.sum().to(dtype=pose_norm.dtype),
    }
