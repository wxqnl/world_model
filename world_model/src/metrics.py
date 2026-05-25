"""Diagnostic metrics for Phase 0. Each function takes plain tensors and returns scalars/dicts.

Keep these pure and side-effect-free so experiments remain easy to re-run.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-6


# ---------- Exp 1: temporal coherence ----------

def median_align_depth(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
    """Return d2 scaled so its median matches d1's median (VGGT per-window scale fix)."""
    m1 = torch.median(d1[d1 > 0])
    m2 = torch.median(d2[d2 > 0])
    if m2 < EPS:
        return d2
    return d2 * (m1 / m2)


def depth_rel_err(d1: torch.Tensor, d2: torch.Tensor) -> float:
    """Mean pixelwise 2|d1-d2|/(d1+d2). Expects same shape."""
    d2 = median_align_depth(d1, d2)
    num = (d1 - d2).abs()
    den = (d1 + d2).abs() * 0.5 + EPS
    mask = (d1 > 0) & (d2 > 0)
    return (num[mask] / den[mask]).mean().item()


def so3_geodesic_deg(R1: torch.Tensor, R2: torch.Tensor) -> float:
    """Rotation angle between two 3x3 rotation matrices in degrees."""
    R = R1 @ R2.transpose(-1, -2)
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) * 0.5).clamp(-1 + EPS, 1 - EPS)
    return torch.rad2deg(torch.acos(cos)).mean().item()


def translation_err(t1: torch.Tensor, t2: torch.Tensor) -> float:
    return (t1 - t2).norm(dim=-1).mean().item()


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor, max_pts: int = 8192) -> float:
    """Symmetric Chamfer on point clouds [N,3]."""
    def _sub(x, n):
        if x.shape[0] > n:
            idx = torch.randperm(x.shape[0], device=x.device)[:n]
            return x[idx]
        return x

    p1, p2 = _sub(p1, max_pts), _sub(p2, max_pts)
    d12 = torch.cdist(p1, p2).min(dim=1).values.mean()
    d21 = torch.cdist(p2, p1).min(dim=1).values.mean()
    return ((d12 + d21) * 0.5).item()


# ---------- Exp 2: reconstruction quality ----------

def abs_rel(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = (gt > 0) & (pred > 0)
    return ((pred[mask] - gt[mask]).abs() / gt[mask].clamp_min(EPS)).mean().item()


def delta_1_25(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        mask = (gt > 0) & (pred > 0)
    r = torch.maximum(pred[mask] / gt[mask].clamp_min(EPS),
                      gt[mask] / pred[mask].clamp_min(EPS))
    return (r < 1.25).float().mean().item()


def photometric_reprojection_l1(
    img_src: torch.Tensor,
    img_tgt: torch.Tensor,
    depth_src: torch.Tensor,
    extri_src: torch.Tensor,
    extri_tgt: torch.Tensor,
    intri: torch.Tensor,
) -> float:
    """Warp img_tgt into img_src's frame using depth_src and relative pose, return mean L1.

    Simple pinhole reprojection. Shapes:
      img_*: [3,H,W] in [0,1]; depth_src: [H,W]; extri: [3,4]; intri: [3,3].
    """
    device = img_src.device
    H, W = depth_src.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], dim=0).float()  # [3,H,W]
    K_inv = torch.linalg.inv(intri)
    rays = K_inv @ pix.reshape(3, -1)                  # [3, H*W]
    pts_src = rays * depth_src.reshape(-1)             # [3, H*W]
    pts_src_h = torch.cat([pts_src, torch.ones(1, pts_src.shape[1], device=device)], dim=0)
    # world_from_src = inv(extri_src); tgt_from_world = extri_tgt
    T_src = torch.eye(4, device=device); T_src[:3, :4] = extri_src
    T_tgt = torch.eye(4, device=device); T_tgt[:3, :4] = extri_tgt
    T_rel = T_tgt @ torch.linalg.inv(T_src)
    pts_tgt = (T_rel @ pts_src_h)[:3]
    uv = intri @ pts_tgt
    uv = uv[:2] / uv[2:3].clamp_min(EPS)
    u, v = uv[0], uv[1]
    u_n = (u / (W - 1)) * 2 - 1
    v_n = (v / (H - 1)) * 2 - 1
    grid = torch.stack([u_n, v_n], dim=-1).reshape(1, H, W, 2)
    warped = torch.nn.functional.grid_sample(img_tgt.unsqueeze(0), grid, align_corners=True, mode="bilinear")
    u_n = u_n.reshape(H, W)
    v_n = v_n.reshape(H, W)
    valid = (u_n.abs() < 1) & (v_n.abs() < 1) & (pts_tgt[2].reshape(H, W) > 0)
    err = (warped.squeeze(0) - img_src).abs().mean(0)
    return err[valid].mean().item() if valid.any() else float("nan")


# ---------- Exp 3: token dynamics ----------

def token_l2_consecutive(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: [N, D] -> [N-1] L2 distances between consecutive rows."""
    return (tokens[1:] - tokens[:-1]).norm(dim=-1)


def token_cosine_consecutive(tokens: torch.Tensor) -> torch.Tensor:
    a = tokens[:-1]
    b = tokens[1:]
    return 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


def pearson_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    from scipy.stats import pearsonr

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())
