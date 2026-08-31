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
    depth_change: float = 0.0
    depth_motion_l1: float = 0.0
    depth_motion_gain: float = 4.0
    depth_temporal: float = 0.0
    depth_temporal_static_only: bool = True
    depth_tv: float = 0.0
    geom_point: float = 0.0
    point_temporal: float = 0.0
    geom_pose: float = 0.0
    action: float = 1.0
    # Legacy world-model runs supervised de-normalized physical deltas.  V7 S1
    # opts into normalized space so small robot deltas retain useful gradients.
    action_pose_space: str = "physical"
    action_grip_positive_weight: float = 1.0
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
    progress: float = 0.0
    terminal_progress: float = 0.0
    plausibility: float = 0.0
    proposer_pose: float = 0.0
    proposer_grip: float = 0.0
    proposer_anchor: float = 0.0
    proposer_diversity: float = 0.0
    proposer_diversity_margin: float = 0.5
    world_prior: float = 0.0
    world_prior_init: float = 1.0
    world_prior_future: float = 1.0
    world_prior_flow: float = 0.0
    world_prior_depth: float = 0.0
    world_prior_rgb_l1: float = 0.0
    huber_delta: float = 1.0


def _normalize_depth(d: torch.Tensor) -> torch.Tensor:
    med = d.flatten(-2).median(dim=-1).values.clamp_min(1e-6)
    return d / med[..., None, None]


def _prepare_weight(mask: torch.Tensor | None, target: torch.Tensor) -> torch.Tensor | None:
    if mask is None:
        return None
    weight = mask.to(device=target.device, dtype=target.dtype)
    while weight.ndim < target.ndim:
        weight = weight.unsqueeze(-1)
    return weight.expand_as(target)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred.float(), target.float(), beta=beta, reduction="none")
    weight = _prepare_weight(mask, loss)
    if weight is None:
        return loss.mean()
    valid = torch.isfinite(loss) & torch.isfinite(weight) & (weight > 0)
    weight = torch.where(valid, weight, torch.zeros_like(weight))
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


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


def _resize_spatial_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5 or ref.ndim != 5:
        return x
    if x.shape[-3:-1] == ref.shape[-3:-1]:
        return x
    b, t, h, w, c = x.shape
    y = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    y = F.interpolate(y, size=ref.shape[-3:-1], mode="bilinear", align_corners=False)
    return y.reshape(b, t, c, ref.shape[-3], ref.shape[-2]).permute(0, 1, 3, 4, 2).contiguous()


def _resize_conf_like(conf: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 5:
        return conf
    if conf.ndim == 5 and conf.shape[-1] == 1:
        conf_hw = conf.squeeze(-1)
    elif conf.ndim == 4:
        conf_hw = conf
    else:
        return conf
    if conf_hw.shape[-2:] == pred.shape[-3:-1]:
        return conf_hw.unsqueeze(-1)
    b, t, h, w = conf_hw.shape
    y = F.interpolate(
        conf_hw.reshape(b * t, 1, h, w),
        size=pred.shape[-3:-1],
        mode="bilinear",
        align_corners=False,
    )
    return y.reshape(b, t, pred.shape[-3], pred.shape[-2], 1)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    target = target.to(device=pred.device, dtype=pred.dtype)
    target = _resize_spatial_like(target, pred)
    pred_f = pred.float()
    target_f = target.float()
    valid = torch.isfinite(pred_f) & torch.isfinite(target_f)
    err = (pred_f - target_f).abs()
    if conf is not None:
        conf_f = conf.to(device=pred.device, dtype=pred.dtype).float().clamp_min(0.0)
        conf_f = _resize_conf_like(conf_f, pred)
        if conf_f.ndim == pred.ndim - 1:
            conf_f = conf_f.unsqueeze(-1)
        while conf_f.ndim < pred.ndim:
            conf_f = conf_f.unsqueeze(-1)
        if conf_f.shape != pred.shape:
            conf_f = conf_f.expand_as(pred)
        valid = valid & torch.isfinite(conf_f) & (conf_f > 0)
        weight = torch.where(valid, conf_f, torch.zeros_like(conf_f))
    else:
        weight = valid.to(err.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (err * weight).sum() / denom


def _resize_depth_conf_like(conf: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    conf = conf.float()
    if conf.ndim == 5 and conf.shape[-1] == 1:
        conf = conf.squeeze(-1)
    if conf.ndim != 4 or ref.ndim != 4:
        return conf
    if conf.shape[-2:] == ref.shape[-2:]:
        return conf
    b, t, h, w = conf.shape
    return F.interpolate(
        conf.reshape(b * t, 1, h, w),
        size=ref.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).reshape(b, t, ref.shape[-2], ref.shape[-1])


def _masked_depth_l1(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    pred_f = pred.float()
    target_f = target.to(device=pred.device, dtype=pred.dtype).float()
    if pred_f.ndim == 4 and target_f.ndim == 4 and target_f.shape[-2:] != pred_f.shape[-2:]:
        target_f = F.interpolate(
            target_f.flatten(0, 1)[:, None],
            size=pred_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(pred_f.shape[0], pred_f.shape[1], pred_f.shape[2], pred_f.shape[3])
    if conf is None:
        return F.l1_loss(pred_f, target_f)
    conf_f = _resize_depth_conf_like(conf.to(device=pred.device, dtype=pred.dtype), pred_f).float().clamp_min(0.0)
    valid = torch.isfinite(pred_f) & torch.isfinite(target_f) & torch.isfinite(conf_f) & (conf_f > 0)
    weight = torch.where(valid, conf_f, torch.zeros_like(conf_f))
    denom = weight.sum().clamp_min(1.0)
    return ((pred_f - target_f).abs() * weight).sum() / denom


def _normalise_points_relative(
    points: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize point maps independently per sample/window.

    VGGT world coordinates are window-relative. Predicted and target point maps
    must each lose their own translation and scale before comparison; sharing
    the target center would leave absolute window-coordinate offsets in the
    loss.
    """
    pts = points.float()
    if mask is not None:
        weight = _prepare_weight(mask, pts).float()
        denom = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        center = (pts * weight).sum(dim=(1, 2, 3), keepdim=True) / denom
        centered = pts - center
        scale = ((centered.pow(2).sum(dim=-1, keepdim=True) * weight[..., :1]).sum(
            dim=(1, 2, 3), keepdim=True
        ) / denom[..., :1]).sqrt().clamp_min(1e-3)
    else:
        center = pts.mean(dim=(1, 2, 3), keepdim=True)
        centered = pts - center
        scale = centered.pow(2).sum(dim=-1, keepdim=True).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-3)
    return centered / scale


def _resize_point_pred(point_pred: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if point_pred.shape[2:4] == target_hw:
        return point_pred
    bsz, horizon, _, _, _ = point_pred.shape
    point_pred_chw = point_pred.permute(0, 1, 4, 2, 3).flatten(0, 1)
    point_pred_chw = F.interpolate(
        point_pred_chw,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )
    return point_pred_chw.reshape(bsz, horizon, 3, target_hw[0], target_hw[1]).permute(0, 1, 3, 4, 2)


def _point_mask_hw(conf: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if conf is None:
        return None
    conf_f = conf.to(device=ref.device, dtype=ref.dtype).float().clamp_min(0.0)
    if conf_f.ndim == 5 and conf_f.shape[-1] == 1:
        conf_f = conf_f.squeeze(-1)
    if conf_f.ndim != 4:
        return conf_f
    if conf_f.shape[-2:] == ref.shape[2:4]:
        return conf_f
    b, t, h, w = conf_f.shape
    return F.interpolate(
        conf_f.reshape(b * t, 1, h, w),
        size=ref.shape[2:4],
        mode="bilinear",
        align_corners=False,
    ).reshape(b, t, ref.shape[2], ref.shape[3])


def _pose_geom_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    beta: float,
) -> torch.Tensor:
    pred = pred.float()
    target = target.to(device=pred.device, dtype=pred.dtype).float()
    horizon = min(pred.shape[1], target.shape[1])
    pred = pred[:, :horizon]
    target = target[:, :horizon]
    trans_pred = pred[..., :3] - pred[..., :3][:, :1]
    trans_tgt = target[..., :3] - target[..., :3][:, :1]
    trans_scale = trans_tgt.pow(2).sum(dim=-1, keepdim=True).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-3)
    quat_pred = F.normalize(pred[..., 3:7], dim=-1)
    quat_tgt = F.normalize(target[..., 3:7], dim=-1)
    rot_loss = 1.0 - (quat_pred * quat_tgt).sum(dim=-1).abs()
    pose_weight = None
    if mask is not None:
        pose_weight = mask.to(device=pred.device, dtype=pred.dtype)[:, :horizon].reshape(pred.shape[0], horizon)
        denom = pose_weight.sum().clamp_min(1.0)
        L_rot = (rot_loss * pose_weight).sum() / denom
    else:
        L_rot = rot_loss.mean()
    L_trans = _masked_smooth_l1(trans_pred / trans_scale, trans_tgt / trans_scale, pose_weight, beta=beta)
    L_fov = _masked_smooth_l1(pred[..., 7:9], target[..., 7:9], pose_weight, beta=beta)
    return L_trans + L_rot + 0.1 * L_fov


def _pairwise_conf(conf: torch.Tensor | None, ref: torch.Tensor, *, point: bool = False) -> torch.Tensor | None:
    if conf is None:
        return None
    if point:
        conf_f = _point_mask_hw(conf, ref)
        if conf_f is None:
            return None
        return conf_f[:, 1:] * conf_f[:, :-1]
    conf_f = _resize_depth_conf_like(conf.to(device=ref.device, dtype=ref.dtype), ref).float().clamp_min(0.0)
    return conf_f[:, 1:] * conf_f[:, :-1]


def compute_native_action_losses(
    out: dict,
    tgt: dict,
    w: LossWeights,
) -> dict[str, torch.Tensor]:
    """Compute the native action-head objective without any world losses."""

    a_tgt = tgt["action_tgt"]
    action_pose_space = str(w.action_pose_space).strip().lower()
    if action_pose_space not in {"physical", "normalized"}:
        raise ValueError(
            "action_pose_space must be either 'physical' or 'normalized', "
            f"got {w.action_pose_space!r}"
        )
    action_h_candidates = [int(out["pose"].shape[1]), int(a_tgt.shape[1])]
    if action_pose_space == "normalized":
        if "pose_norm" not in out or "action_tgt_norm" not in tgt:
            raise ValueError(
                "action_pose_space='normalized' requires out['pose_norm'] "
                "and tgt['action_tgt_norm']"
            )
        action_h_candidates.extend(
            (
                int(out["pose_norm"].shape[1]),
                int(tgt["action_tgt_norm"].shape[1]),
            )
        )
    action_h = min(action_h_candidates)
    if action_h <= 0:
        raise ValueError("native action loss requires a non-empty action horizon")
    pose_pred = out["pose"][:, :action_h].float()
    grip_pred = out["gripper_logit"][:, :action_h].float()
    a_tgt_core = a_tgt[:, :action_h]
    L_pose_a_physical = F.mse_loss(pose_pred, a_tgt_core[..., :6])
    L_pose_a_normalized = pose_pred.new_zeros(())
    if action_pose_space == "normalized":
        pose_norm = out["pose_norm"][:, :action_h].float()
        action_tgt_norm = tgt["action_tgt_norm"][:, :action_h].to(
            device=pose_norm.device,
            dtype=pose_norm.dtype,
        )
        L_pose_a_normalized = huber(
            pose_norm,
            action_tgt_norm,
            delta=w.huber_delta,
        )
        L_pose_a = L_pose_a_normalized
    else:
        L_pose_a = L_pose_a_physical
    grip_tgt = (a_tgt_core[..., 6] > 0.5).float()
    if float(w.action_grip_positive_weight) <= 0:
        raise ValueError("action_grip_positive_weight must be positive")
    L_grip = F.binary_cross_entropy_with_logits(
        grip_pred,
        grip_tgt,
        pos_weight=grip_pred.new_tensor(float(w.action_grip_positive_weight)),
    )
    L_action = L_pose_a + w.grip * L_grip
    return {
        "L_action": L_action,
        "L_pose_action": L_pose_a,
        "L_pose_action_physical": L_pose_a_physical,
        "L_pose_action_normalized": L_pose_a_normalized,
        "L_grip": L_grip,
    }


def compute_losses(out: dict, tgt: dict, w: LossWeights,
                   lpips_fn: nn.Module | None = None) -> dict[str, torch.Tensor]:
    pred_s = out["pred_tokens"]
    zero = pred_s.new_zeros(())
    if "s_tgt" in tgt:
        s_tgt = tgt["s_tgt"].to(pred_s.dtype)
        L_state_mse = F.mse_loss(pred_s, s_tgt)
        cos = F.cosine_similarity(pred_s.flatten(-2), s_tgt.flatten(-2), dim=-1).mean()
        L_cos = 1.0 - cos
        L_state = L_state_mse + w.cos * L_cos
    else:
        L_state_mse = zero
        L_cos = zero
        L_state = zero

    L_depth_change = zero
    L_depth_motion = zero
    L_depth_temporal = zero
    L_depth_tv = zero
    if "depth_tgt" in tgt and "depth" in out:
        depth_tgt = tgt["depth_tgt"].to(device=out["depth"].device, dtype=out["depth"].dtype)
        if depth_tgt.ndim == 4 and out["depth"].ndim == 4 and depth_tgt.shape[-2:] != out["depth"].shape[-2:]:
            b, t = depth_tgt.shape[:2]
            depth_tgt = F.interpolate(
                depth_tgt.flatten(0, 1)[:, None],
                size=out["depth"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(b, t, out["depth"].shape[-2], out["depth"].shape[-1])
        depth_pred_norm = _normalize_depth(out["depth"])
        depth_tgt_norm = _normalize_depth(depth_tgt)
        L_depth = _masked_depth_l1(depth_pred_norm, depth_tgt_norm, tgt.get("depth_conf_tgt"))
        if w.depth_change > 0 and "depth_ref" in tgt:
            depth_ref = tgt["depth_ref"].to(device=out["depth"].device, dtype=out["depth"].dtype)
            if depth_ref.ndim == 3 and depth_ref.shape[-2:] != depth_tgt.shape[-2:]:
                depth_ref = F.interpolate(
                    depth_ref[:, None],
                    size=depth_tgt.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            depth_ref_norm = _normalize_depth(depth_ref[:, None].expand_as(depth_tgt))
            pred_change = depth_pred_norm - depth_ref_norm
            tgt_change = depth_tgt_norm - depth_ref_norm
            L_depth_change = _masked_depth_l1(pred_change.abs(), tgt_change.abs(), tgt.get("depth_conf_tgt"))
        if w.depth_motion_l1 > 0 and "rgb_tgt_p" in tgt and "rgb_ref_p" in tgt:
            motion = (tgt["rgb_tgt_p"].float() - tgt["rgb_ref_p"][:, None].float()).abs().mean(dim=2)
            if motion.shape[-2:] != depth_pred_norm.shape[-2:]:
                motion = F.interpolate(
                    motion.flatten(0, 1)[:, None],
                    size=depth_pred_norm.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape_as(depth_pred_norm)
            motion_mask = (motion > w.rgb_motion_threshold).to(depth_pred_norm.dtype)
            motion_weight = 1.0 + w.depth_motion_gain * motion_mask
            L_depth_motion = ((depth_pred_norm - depth_tgt_norm).abs() * motion_weight).mean()
        if w.depth_temporal > 0:
            if depth_pred_norm.shape[1] < 2:
                L_depth_temporal = zero
            else:
                pred_delta = depth_pred_norm[:, 1:] - depth_pred_norm[:, :-1]
                tgt_delta = depth_tgt_norm[:, 1:] - depth_tgt_norm[:, :-1]
                temporal_conf = _pairwise_conf(tgt.get("depth_conf_tgt"), depth_pred_norm, point=False)
                if w.depth_temporal_static_only:
                    if "rgb_tgt_p" not in tgt:
                        raise ValueError("depth_temporal_static_only=True requires rgb_tgt_p in targets")
                    rgb_tgt = tgt["rgb_tgt_p"].to(device=depth_pred_norm.device).float()
                    temporal_motion = (rgb_tgt[:, 1:] - rgb_tgt[:, :-1]).abs().mean(dim=2)
                    if temporal_motion.shape[-2:] != pred_delta.shape[-2:]:
                        temporal_motion = F.interpolate(
                            temporal_motion.flatten(0, 1)[:, None],
                            size=pred_delta.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ).reshape_as(pred_delta)
                    static_mask = (temporal_motion <= w.rgb_motion_threshold).to(pred_delta.dtype)
                    temporal_conf = static_mask if temporal_conf is None else temporal_conf * static_mask
                L_depth_temporal = _masked_depth_l1(pred_delta, tgt_delta, temporal_conf)
        if w.depth_tv > 0:
            dx = (depth_pred_norm[..., :, 1:] - depth_pred_norm[..., :, :-1]).abs().mean()
            dy = (depth_pred_norm[..., 1:, :] - depth_pred_norm[..., :-1, :]).abs().mean()
            L_depth_tv = dx + dy
    else:
        if w.depth_temporal > 0:
            raise ValueError("depth_temporal > 0 requires both out['depth'] and tgt['depth_tgt']")
        L_depth = zero
    L_point = zero
    L_point_temporal = zero
    L_pose_g = zero
    geom_point_active = zero
    geom_pose_active = zero
    geom_point_missing = pred_s.new_tensor(1.0)
    geom_pose_missing = pred_s.new_tensor(1.0)
    if w.geom_point > 0 or w.point_temporal > 0:
        if "point" not in out or "point_tgt" not in tgt:
            raise ValueError("geom_point/point_temporal > 0 requires out['point'] and tgt['point_tgt']")
    if "point" in out and "point_tgt" in tgt:
        point_pred = out["point"].float()
        point_tgt = tgt["point_tgt"].to(device=point_pred.device, dtype=point_pred.dtype)
        horizon = min(point_pred.shape[1], point_tgt.shape[1])
        point_pred = _resize_point_pred(point_pred[:, :horizon], point_tgt.shape[2:4])
        point_tgt = point_tgt[:, :horizon]
        point_mask = _point_mask_hw(tgt.get("point_conf_tgt"), point_pred)
        if point_mask is not None:
            point_mask = point_mask[:, :horizon]
        pred_point_norm = _normalise_points_relative(point_pred, point_mask)
        tgt_point_norm = _normalise_points_relative(point_tgt, point_mask)
        L_point = _masked_smooth_l1(pred_point_norm, tgt_point_norm, point_mask, beta=w.huber_delta)
        if w.point_temporal > 0:
            if pred_point_norm.shape[1] >= 2:
                point_pair_conf = None if point_mask is None else point_mask[:, 1:] * point_mask[:, :-1]
                L_point_temporal = _masked_smooth_l1(
                    pred_point_norm[:, 1:] - pred_point_norm[:, :-1],
                    tgt_point_norm[:, 1:] - tgt_point_norm[:, :-1],
                    point_pair_conf,
                    beta=w.huber_delta,
                )
        geom_point_active = pred_s.new_tensor(1.0)
        geom_point_missing = zero
    if w.geom_pose > 0 and ("pose_geom" not in out or "pose_geom_tgt" not in tgt):
        raise ValueError("geom_pose > 0 requires out['pose_geom'] and tgt['pose_geom_tgt']")
    if "pose_geom" in out and "pose_geom_tgt" in tgt:
        L_pose_g = _pose_geom_loss(
            out["pose_geom"],
            tgt["pose_geom_tgt"],
            tgt.get("pose_geom_conf_tgt"),
            w.huber_delta,
        )
        geom_pose_active = pred_s.new_tensor(1.0)
        geom_pose_missing = zero
    L_geom = (
        w.geom_depth * L_depth
        + w.depth_change * L_depth_change
        + w.depth_motion_l1 * L_depth_motion
        + w.depth_temporal * L_depth_temporal
        + w.depth_tv * L_depth_tv
        + w.geom_point * L_point
        + w.point_temporal * L_point_temporal
        + w.geom_pose * L_pose_g
    )

    action_losses = compute_native_action_losses(out, tgt, w)
    L_action = action_losses["L_action"]
    L_pose_a = action_losses["L_pose_action"]
    L_pose_a_physical = action_losses["L_pose_action_physical"]
    L_pose_a_normalized = action_losses["L_pose_action_normalized"]
    L_grip = action_losses["L_grip"]

    L_idm = (out["z_a"].float() ** 2).mean()

    L_total = L_state + L_geom + w.action * L_action + w.idm_reg * L_idm

    L_world_prior_init = L_total.new_zeros(())
    L_world_prior_future = L_total.new_zeros(())
    L_world_prior_flow = L_total.new_zeros(())
    L_world_prior_depth = L_total.new_zeros(())
    L_world_prior_rgb_l1 = L_total.new_zeros(())
    if "prior_initial_tokens" in out and "s_init_tgt" in tgt:
        L_world_prior_init = F.mse_loss(
            out["prior_initial_tokens"].float(),
            tgt["s_init_tgt"].to(device=out["prior_initial_tokens"].device).float(),
        )
    if "prior_future_tokens" in out and "s_tgt" in tgt:
        prior_future = out["prior_future_tokens"].float()
        s_tgt_prior = tgt["s_tgt"].to(device=prior_future.device).float()
        horizon = min(prior_future.shape[1], s_tgt_prior.shape[1])
        L_world_prior_future = F.mse_loss(prior_future[:, :horizon], s_tgt_prior[:, :horizon])
    if "prior_velocity" in out and "prior_velocity_target" in out:
        L_world_prior_flow = F.mse_loss(
            out["prior_velocity"].float(),
            out["prior_velocity_target"].detach().float(),
        )
    if "prior_depth" in out and "depth_tgt" in tgt:
        prior_depth = _normalize_depth(out["prior_depth"])
        prior_depth_tgt = _normalize_depth(tgt["depth_tgt"].to(device=prior_depth.device, dtype=prior_depth.dtype))
        horizon = min(prior_depth.shape[1], prior_depth_tgt.shape[1])
        depth_conf = tgt.get("depth_conf_tgt")
        if depth_conf is not None:
            depth_conf = depth_conf[:, :horizon]
        L_world_prior_depth = _masked_depth_l1(prior_depth[:, :horizon], prior_depth_tgt[:, :horizon], depth_conf)
    if "prior_rgb" in out and "rgb_tgt_p" in tgt:
        prior_rgb = out["prior_rgb"]
        prior_rgb_tgt = tgt["rgb_tgt_p"].to(device=prior_rgb.device, dtype=prior_rgb.dtype)
        horizon = min(prior_rgb.shape[1], prior_rgb_tgt.shape[1])
        L_world_prior_rgb_l1 = F.l1_loss(prior_rgb[:, :horizon], prior_rgb_tgt[:, :horizon])
    L_world_prior = (
        w.world_prior * (w.world_prior_init * L_world_prior_init + w.world_prior_future * L_world_prior_future)
        + w.world_prior_flow * L_world_prior_flow
        + w.world_prior_depth * L_world_prior_depth
        + w.world_prior_rgb_l1 * L_world_prior_rgb_l1
    )
    if (
        "prior_initial_tokens" in out
        or "prior_future_tokens" in out
        or "prior_velocity" in out
    ):
        L_total = L_total + L_world_prior

    losses = {
        "L_total": L_total,
        "L_state": L_state.detach(), "L_state_mse": L_state_mse.detach(),
        "L_cos": L_cos.detach(),
        "L_geom": L_geom.detach(), "L_depth": L_depth.detach(),
        "L_depth_change": L_depth_change.detach(),
        "L_depth_motion_l1": L_depth_motion.detach(),
        "L_depth_temporal": L_depth_temporal.detach(),
        "L_depth_tv": L_depth_tv.detach(),
        "L_point": L_point.detach(), "L_point_temporal": L_point_temporal.detach(),
        "L_pose_g": L_pose_g.detach(),
        "geom_point_active": geom_point_active.detach(),
        "geom_pose_active": geom_pose_active.detach(),
        "geom_point_missing": geom_point_missing.detach(),
        "geom_pose_missing": geom_pose_missing.detach(),
        "L_action": L_action.detach(), "L_pose_action": L_pose_a.detach(),
        "L_pose_action_physical": L_pose_a_physical.detach(),
        "L_pose_action_normalized": L_pose_a_normalized.detach(),
        "L_grip": L_grip.detach(), "L_idm": L_idm.detach(),
        "L_world_prior": L_world_prior.detach(),
        "L_world_prior_init": L_world_prior_init.detach(),
        "L_world_prior_future": L_world_prior_future.detach(),
        "L_world_prior_flow": L_world_prior_flow.detach(),
        "L_world_prior_depth": L_world_prior_depth.detach(),
        "L_world_prior_rgb_l1": L_world_prior_rgb_l1.detach(),
    }

    L_progress = L_total.new_zeros(())
    L_terminal_progress = L_total.new_zeros(())
    L_plausibility = L_total.new_zeros(())
    if "progress" in out and "progress_tgt" in tgt:
        progress_tgt = tgt["progress_tgt"].to(device=out["progress"].device, dtype=out["progress"].dtype)
        progress_prob = torch.sigmoid(out["progress"].float())
        progress_h = min(int(progress_prob.shape[1]), int(progress_tgt.shape[1]))
        progress_prob = progress_prob[:, :progress_h]
        progress_tgt = progress_tgt[:, :progress_h]
        L_progress = F.smooth_l1_loss(
            progress_prob,
            progress_tgt.float(),
            beta=w.huber_delta,
        )
        L_total = L_total + w.progress * L_progress
        losses["progress_mae"] = (progress_prob - progress_tgt.float()).abs().mean().detach()
        losses["progress_rmse"] = (progress_prob - progress_tgt.float()).pow(2).mean().sqrt().detach()
        losses["progress_pred_mean"] = progress_prob.mean().detach()
        losses["progress_tgt_mean"] = progress_tgt.float().mean().detach()
    if "terminal_success_logit" in out and "terminal_success_tgt" in tgt:
        terminal_tgt = tgt["terminal_success_tgt"].to(
            device=out["terminal_success_logit"].device,
            dtype=out["terminal_success_logit"].dtype,
        )
        terminal_prob = torch.sigmoid(out["terminal_success_logit"].float())
        L_terminal_progress = F.binary_cross_entropy_with_logits(
            out["terminal_success_logit"].float(),
            terminal_tgt.float(),
        )
        L_total = L_total + w.terminal_progress * L_terminal_progress
        losses["terminal_progress_prob_mean"] = terminal_prob.mean().detach()
        losses["terminal_progress_tgt_mean"] = terminal_tgt.float().mean().detach()
    if "plausibility_logit" in out and "plausibility_tgt" in tgt:
        plausibility_tgt = tgt["plausibility_tgt"].to(
            device=out["plausibility_logit"].device,
            dtype=out["plausibility_logit"].dtype,
        )
        L_plausibility = F.binary_cross_entropy_with_logits(
            out["plausibility_logit"].float(),
            plausibility_tgt.float(),
        )
        L_total = L_total + w.plausibility * L_plausibility
        losses["plausibility_prob_mean"] = torch.sigmoid(out["plausibility_logit"].float()).mean().detach()
    if "progress" in out:
        losses["L_total"] = L_total
        losses["L_progress"] = L_progress.detach()
        losses["L_terminal_progress"] = L_terminal_progress.detach()
        losses["L_plausibility"] = L_plausibility.detach()

    L_proposer_pose = L_total.new_zeros(())
    L_proposer_grip = L_total.new_zeros(())
    L_proposer_anchor = L_total.new_zeros(())
    L_proposer_diversity = L_total.new_zeros(())
    if "proposer_pose_norm" in out and "action_tgt_norm" in tgt:
        pose_pred = out["proposer_pose_norm"].float()
        pose_tgt = tgt["action_tgt_norm"].to(device=pose_pred.device, dtype=pose_pred.dtype)
        if pose_pred.ndim != 4:
            raise ValueError(f"proposer_pose_norm must be [B,N,k,6], got {tuple(pose_pred.shape)}")
        proposer_h = min(int(pose_pred.shape[2]), int(pose_tgt.shape[1]))
        pose_pred = pose_pred[:, :, :proposer_h]
        pose_tgt = pose_tgt[:, :proposer_h]
        if pose_tgt.shape != (pose_pred.shape[0], pose_pred.shape[2], pose_pred.shape[3]):
            raise ValueError(f"action_tgt_norm must be [B,k,6], got {tuple(pose_tgt.shape)}")
        pose_err = F.smooth_l1_loss(
            pose_pred,
            pose_tgt[:, None].expand_as(pose_pred),
            beta=w.huber_delta,
            reduction="none",
        ).mean(dim=(2, 3))
        best_idx = pose_err.argmin(dim=1)
        L_proposer_pose = pose_err.gather(1, best_idx[:, None]).mean()
        L_proposer_anchor = pose_err[:, 0].mean()
        if "proposer_gripper_logit" in out and "action_tgt" in tgt:
            grip_logits = out["proposer_gripper_logit"].float()[:, :, :proposer_h]
            grip_tgt = (tgt["action_tgt"].to(device=grip_logits.device)[:, :proposer_h, 6] > 0.5).float()
            grip_err = F.binary_cross_entropy_with_logits(
                grip_logits,
                grip_tgt[:, None].expand_as(grip_logits),
                reduction="none",
            ).mean(dim=2)
            L_proposer_grip = grip_err.gather(1, best_idx[:, None]).mean()
        if pose_pred.shape[1] > 1:
            flat = pose_pred.flatten(2)
            pair_l1 = (flat[:, :, None] - flat[:, None]).abs().mean(dim=-1)
            mask = ~torch.eye(pose_pred.shape[1], device=pose_pred.device, dtype=torch.bool)
            L_proposer_diversity = torch.exp(
                -pair_l1[:, mask] / max(1e-6, w.proposer_diversity_margin)
            ).mean()
        L_total = (
            L_total
            + w.proposer_pose * L_proposer_pose
            + w.proposer_grip * L_proposer_grip
            + w.proposer_anchor * L_proposer_anchor
            + w.proposer_diversity * L_proposer_diversity
        )
        losses["L_total"] = L_total
        losses["L_proposer_pose"] = L_proposer_pose.detach()
        losses["L_proposer_grip"] = L_proposer_grip.detach()
        losses["L_proposer_anchor"] = L_proposer_anchor.detach()
        losses["L_proposer_diversity"] = L_proposer_diversity.detach()

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
