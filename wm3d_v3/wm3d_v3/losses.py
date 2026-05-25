"""Joint v3 losses: tokens + geometry + action + RGB (LPIPS+L1)."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _normalize_depth(d: torch.Tensor) -> torch.Tensor:
    med = d.flatten(-2).median(dim=-1).values.clamp_min(1e-6)
    return d / med[..., None, None]


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
        L_rgb = w.rgb_l1 * L_rgb_l1 + w.rgb_lpips * L_rgb_lpips
        L_total = L_total + L_rgb
        losses["L_total"] = L_total
        losses["L_rgb"] = L_rgb.detach()
        losses["L_rgb_l1"] = L_rgb_l1.detach()
        losses["L_rgb_lpips"] = L_rgb_lpips.detach()
    return losses
