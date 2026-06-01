"""Joint v3 losses: tokens + geometry + action + RGB (LPIPS+L1)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Mean Huber loss. delta=1 => quadratic when |err|<=1, linear (|err|-0.5*delta) when |err|>1."""
    err = pred.float() - target.float()
    abs_err = err.abs()
    quad = 0.5 * err.pow(2)
    lin = delta * (abs_err - 0.5 * delta)
    return torch.where(abs_err <= delta, quad, lin).mean()


def focal_bce(logits: torch.Tensor, targets: torch.Tensor,
              alpha: float = 0.25, gamma: float = 2.0,
              pos_weight: float | None = None) -> torch.Tensor:
    """Focal binary cross-entropy with logits.

    L = -alpha_t * (1-p_t)^gamma * log(p_t)
    where p_t = sigmoid(logit) if y=1 else 1-sigmoid(logit),
    and alpha_t = alpha if y=1 else (1-alpha).
    pos_weight (if given) multiplicatively reweights positives on top.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none")
    p = torch.sigmoid(logits.float())
    p_t = torch.where(targets > 0.5, p, 1.0 - p)
    alpha_t = torch.where(targets > 0.5,
                          torch.full_like(p, alpha),
                          torch.full_like(p, 1.0 - alpha))
    focal_weight = alpha_t * (1.0 - p_t).pow(gamma)
    loss = focal_weight * bce
    if pos_weight is not None:
        loss = torch.where(targets > 0.5, loss * pos_weight, loss)
    return loss.mean()


@dataclass
class LossWeights:
    cos: float = 0.1
    grip: float = 0.5
    geom_depth: float = 0.3
    geom_point: float = 0.05
    geom_pose: float = 0.02
    action: float = 1.0
    idm_reg: float = 0.01
    rgb_l1: float = 1.0
    rgb_lpips: float = 0.5
    rgb_motion_l1: float = 0.0
    rgb_edge: float = 0.0
    rgb_motion_bce: float = 0.0
    rgb_motion_dice: float = 0.0
    rgb_motion_pos_weight: float = 1.0
    rgb_motion_threshold: float = 0.03
    rgb_motion_gain: float = 4.0


def _normalize_depth(d: torch.Tensor) -> torch.Tensor:
    med = d.flatten(-2).median(dim=-1).values.clamp_min(1e-6)
    return d / med[..., None, None]


def _rgb_edge_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dy = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def _dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits.float())
    target = target.float()
    reduce_dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=reduce_dims)
    denom = prob.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def compute_losses(out: dict, tgt: dict, w: LossWeights,
                   lpips_fn: nn.Module | None = None) -> dict[str, torch.Tensor]:
    pred_s = out["pred_tokens"]
    s_tgt = tgt["s_tgt"].to(pred_s.dtype)
    L_state_mse = F.mse_loss(pred_s, s_tgt)
    cos = F.cosine_similarity(pred_s.flatten(-2), s_tgt.flatten(-2), dim=-1).mean()
    L_state = L_state_mse + w.cos * (1.0 - cos)

    depth_tgt = tgt["depth_tgt"].to(out["depth"].dtype)
    L_depth = F.l1_loss(_normalize_depth(out["depth"]), _normalize_depth(depth_tgt))
    # point/pose are not directly supervised since OXE doesn't ship them; use depth-derived proxy
    L_point = torch.zeros_like(L_depth)
    L_pose_g = torch.zeros_like(L_depth)
    L_geom = (w.geom_depth * L_depth + w.geom_point * L_point + w.geom_pose * L_pose_g)

    a_tgt = tgt["action_tgt"]
    L_pose_a = F.mse_loss(out["pose"].float(), a_tgt[..., :6])
    grip_tgt = (a_tgt[..., 6] > 0.5).float()
    L_grip = F.binary_cross_entropy_with_logits(out["gripper_logit"].float(), grip_tgt)
    L_action = L_pose_a + w.grip * L_grip

    L_idm = (out["z_a"].float() ** 2).mean()

    L_total = L_state + L_geom + w.action * L_action + w.idm_reg * L_idm
    losses = {
        "L_total": L_total,
        "L_state": L_state.detach(), "L_state_mse": L_state_mse.detach(),
        "L_cos": (1 - cos).detach(),
        "L_geom": L_geom.detach(), "L_depth": L_depth.detach(),
        "L_action": L_action.detach(), "L_pose_action": L_pose_a.detach(),
        "L_grip": L_grip.detach(), "L_idm": L_idm.detach(),
    }

    motion_mask = None
    if "rgb_tgt_p" in tgt and "rgb_ref_p" in tgt and (
        w.rgb_motion_l1 > 0 or w.rgb_motion_bce > 0 or w.rgb_motion_dice > 0
    ):
        rgb_tgt_for_motion = tgt["rgb_tgt_p"]
        rgb_ref = tgt["rgb_ref_p"][:, None].expand_as(rgb_tgt_for_motion)
        motion = (rgb_tgt_for_motion.float() - rgb_ref.float()).abs().mean(dim=2, keepdim=True)
        motion_mask = (motion > w.rgb_motion_threshold).float()

    L_rgb_l1 = L_total.new_zeros(())
    L_rgb_lpips = L_total.new_zeros(())
    L_rgb_motion_l1 = L_total.new_zeros(())
    L_rgb_edge = L_total.new_zeros(())
    if "rgb" in out and "rgb_tgt_p" in tgt:
        # rgb_tgt_p shape matches out["rgb"] = [B, k, 3, H, W]
        rgb_pred = out["rgb"]
        rgb_tgt = tgt["rgb_tgt_p"].to(rgb_pred.dtype)
        L_rgb_l1 = F.l1_loss(rgb_pred, rgb_tgt)
        if lpips_fn is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                rp = (rgb_pred.float().flatten(0, 1) * 2 - 1)   # [B*k, 3, H, W]
                rt = (rgb_tgt.float().flatten(0, 1) * 2 - 1)
                L_rgb_lpips = lpips_fn(rp, rt).mean()
        else:
            L_rgb_lpips = torch.zeros_like(L_rgb_l1)
        if motion_mask is not None and w.rgb_motion_l1 > 0:
            motion_weight = 1.0 + w.rgb_motion_gain * motion_mask.to(dtype=rgb_pred.dtype)
            L_rgb_motion_l1 = ((rgb_pred - rgb_tgt).abs() * motion_weight).mean()
        else:
            L_rgb_motion_l1 = torch.zeros_like(L_rgb_l1)
        L_rgb_edge = _rgb_edge_l1(rgb_pred, rgb_tgt) if w.rgb_edge > 0 else torch.zeros_like(L_rgb_l1)

    if "motion_logit" in out and motion_mask is not None and w.rgb_motion_bce > 0:
        motion_logit = out["motion_logit"].float()
        target = motion_mask.to(device=motion_logit.device, dtype=motion_logit.dtype)
        if motion_logit.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target.flatten(0, 1),
                size=motion_logit.shape[-2:],
                mode="nearest",
            ).reshape_as(motion_logit)
        pos_weight = torch.as_tensor(w.rgb_motion_pos_weight, device=motion_logit.device)
        L_rgb_motion_bce = F.binary_cross_entropy_with_logits(
            motion_logit,
            target,
            pos_weight=pos_weight,
        )
    else:
        L_rgb_motion_bce = L_total.new_zeros(())
    if "motion_logit" in out and motion_mask is not None and w.rgb_motion_dice > 0:
        motion_logit = out["motion_logit"].float()
        target = motion_mask.to(device=motion_logit.device, dtype=motion_logit.dtype)
        if motion_logit.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target.flatten(0, 1),
                size=motion_logit.shape[-2:],
                mode="nearest",
            ).reshape_as(motion_logit)
        L_rgb_motion_dice = _dice_loss_from_logits(motion_logit, target)
    else:
        L_rgb_motion_dice = L_total.new_zeros(())
    if "rgb" in out or "motion_logit" in out:
        L_rgb = (
            w.rgb_l1 * L_rgb_l1
            + w.rgb_lpips * L_rgb_lpips
            + w.rgb_motion_l1 * L_rgb_motion_l1
            + w.rgb_edge * L_rgb_edge
            + w.rgb_motion_bce * L_rgb_motion_bce
            + w.rgb_motion_dice * L_rgb_motion_dice
        )
        L_total = L_total + L_rgb
        losses["L_total"] = L_total
        losses["L_rgb"] = L_rgb.detach()
        losses["L_rgb_l1"] = L_rgb_l1.detach()
        losses["L_rgb_lpips"] = L_rgb_lpips.detach()
        losses["L_rgb_motion_l1"] = L_rgb_motion_l1.detach()
        losses["L_rgb_edge"] = L_rgb_edge.detach()
        losses["L_rgb_motion_bce"] = L_rgb_motion_bce.detach()
        losses["L_rgb_motion_dice"] = L_rgb_motion_dice.detach()
    return losses


@dataclass
class VLALossWeights:
    action_pose: float = 10.0
    action_grip: float = 2.0
    aux_idm: float = 5.0
    grip_pos_weight: float = 1.0
    grip_focal_alpha: float = 0.25
    grip_focal_gamma: float = 2.0
    huber_delta: float = 1.0


def compute_losses_vla(out: dict, tgt: dict, w: VLALossWeights) -> dict[str, torch.Tensor]:
    """Loss for stage-A VLA fine-tune.

    Required out keys: pose_norm, gripper_logit (+ aux_pose_norm, aux_grip if aux_idm head present)
    Required tgt keys: action_tgt_norm, action_tgt
    """
    pose_norm = out["pose_norm"].float()
    a_norm = tgt["action_tgt_norm"].to(pose_norm.dtype)
    L_pose = huber(pose_norm, a_norm, delta=w.huber_delta)

    grip_logit = out["gripper_logit"].float()
    grip_tgt = (tgt["action_tgt"][..., 6] > 0.5).to(grip_logit.dtype)
    L_grip = focal_bce(grip_logit, grip_tgt,
                       alpha=w.grip_focal_alpha, gamma=w.grip_focal_gamma,
                       pos_weight=w.grip_pos_weight)

    L_total = w.action_pose * L_pose + w.action_grip * L_grip
    losses = {"L_pose": L_pose.detach(), "L_grip": L_grip.detach()}
    if "aux_pose_norm" in out:
        aux_pn = out["aux_pose_norm"].float()
        L_aux_pose = huber(aux_pn, a_norm, delta=w.huber_delta)
        aux_grip = out["aux_grip"].float()
        L_aux_grip = focal_bce(aux_grip, grip_tgt,
                               alpha=w.grip_focal_alpha, gamma=w.grip_focal_gamma,
                               pos_weight=w.grip_pos_weight)
        L_total = L_total + w.aux_idm * (L_aux_pose + 0.5 * L_aux_grip)
        losses["L_aux_pose"] = L_aux_pose.detach()
        losses["L_aux_grip"] = L_aux_grip.detach()
    else:
        losses["L_aux_pose"] = torch.zeros((), device=pose_norm.device)
        losses["L_aux_grip"] = torch.zeros((), device=pose_norm.device)
    losses["L_total"] = L_total
    return losses
