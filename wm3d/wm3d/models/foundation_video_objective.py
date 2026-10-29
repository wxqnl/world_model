"""Loss-only view validity and equal valid-camera reduction for a WSA canvas."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def valid_view_video_loss(
    pred: torch.Tensor, target: torch.Tensor, view_mask: torch.Tensor,
    image_is_pad: torch.Tensor, *, temporal_factor: int,
    include_initial: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average time/pixels within each valid view, then valid views per sample.

    Input masks retain the actual anchor and every future frame. A temporal
    latent block is supervised only when that camera is present at all of its
    constituent frames. The canvas and VAE are unchanged; this masks loss
    regions, not the VAE's cross-boundary receptive field.
    """
    if pred.ndim != 5 or pred.shape != target.shape:
        raise ValueError("video flow prediction/target must match [B,C,T,H,W]")
    batch, _, steps, height, width = pred.shape
    if height % 3 or width % 2:
        raise ValueError("latent canvas must preserve the three-view layout")
    if (view_mask.ndim != 3 or view_mask.shape[0] != batch
            or view_mask.shape[2] != 3 or view_mask.dtype != torch.bool):
        raise ValueError("video view mask must be bool [B,1+K,3]")
    if (temporal_factor <= 0 or view_mask.shape[1] < 1
            or (view_mask.shape[1] - 1) % temporal_factor):
        raise ValueError("view timestamps do not align with VAE temporal blocks")
    if (image_is_pad is None or image_is_pad.dtype != torch.bool
            or image_is_pad.shape != view_mask.shape[:2]
            or not torch.equal(image_is_pad.to(view_mask.device), ~view_mask.any(-1))):
        raise ValueError("whole-frame padding disagrees with actual view validity")
    valid = view_mask[:, 1:].reshape(batch, -1, temporal_factor, 3).all(2)
    if include_initial:
        valid = torch.cat((view_mask[:, :1], valid), dim=1)
    if valid.shape[1] != steps:
        raise ValueError("video latent steps differ from the loss-only view mask")
    valid = valid.to(pred.device)
    if not bool(torch.isfinite(pred).all() and torch.isfinite(target).all()):
        raise FloatingPointError("video flow prediction/target is non-finite")
    error = F.mse_loss(pred.float(), target.float(), reduction="none")
    top = height - height // 3
    half = width // 2
    per_time_view = torch.stack((
        error[:, :, :, :top, :].mean((1, 3, 4)),
        error[:, :, :, top:, :half].mean((1, 3, 4)),
        error[:, :, :, top:, half:].mean((1, 3, 4)),
    ), dim=-1)
    count = valid.sum(1)
    active = count > 0
    if not bool(active.any(1).all()):
        raise ValueError("sample has no complete valid camera block for video loss")
    per_view = (per_time_view * valid).sum(1) / count.clamp_min(1)
    loss = (per_view * active).sum(1) / active.sum(1)
    logs = {}
    for view in range(3):
        n = active[:, view].sum()
        logs[f"video_valid_view_{view}_samples"] = n.detach()
        logs[f"video_flow_view_{view}_mse"] = (
            (per_view[:, view] * active[:, view]).sum() / n.clamp_min(1)
        ).detach()
    return loss, logs
