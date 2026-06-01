"""JointWorldModel — one forward, end-to-end.

Input: pooled VGGT tokens [B, T, 64, 2048] (pre-cached) + task embedding c [B, 2048]
Output dict (always):
    pred_tokens:   [B, k, 64, 2048]
    z_a:           [B, k, z_dim]
    pose_norm:     [B, k, 6]   (standardized 6-DoF Δpose)
    pose:          [B, k, 6]   (de-normalized = pose_norm * std + mean)
    gripper_logit: [B, k]
    depth:         [B, k, 224, 224]
    point:         [B, k, 224, 224, 3] if geometry extras are enabled
    pose_geom:     [B, k, 9] if geometry extras are enabled
Conditional:
    rgb:                if pixel=True and self.pixel is not None
    cosmos_depth_input: if bridging=True and self.bridging is not None
    aux_pose_norm:      if aux_idm=True and self.aux_idm is not None
    aux_grip:
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .dual_stream import DualStreamDynamics, DualConfig
from .action_proj import ActionProjHead
from .geom_decoder import GeomDecoder
from .pixel_decoder import PixelDecoder, PixelDecoderConfig
from .context_residual_pixel_decoder import (
    ContextResidualPixelDecoder,
    ContextResidualPixelDecoderConfig,
)
from .control_head import ControlHead, ControlHeadConfig
from .progress_head import ProgressHead, ProgressHeadConfig
from .bridging_adapter import BridgingAdapter
from .aux_idm import AuxIDM, AuxIDMConfig


@dataclass
class JointConfig:
    dual: DualConfig = field(default_factory=DualConfig)
    action_proj_hidden: int = 768
    action_proj_layers: int = 5
    geom_hidden: int = 384
    enable_geom_extra: bool = True
    pixel_hidden: int = 768
    pixel_n_res: int = 2
    enable_pixel: bool = True
    enable_context_pixel: bool = False
    context_pixel_hidden: int = 384
    context_pixel_action_dim: int = 7
    context_pixel_task_dim: int | None = None
    context_pixel_residual_scale: float = 0.75
    context_pixel_use_action: bool = True
    context_pixel_use_task: bool = True
    context_pixel_predict_motion: bool = False
    context_pixel_motion_blend_gain: float = 0.0
    enable_control_head: bool = False
    control_hidden: int = 128
    control_output_size: int = 256
    control_fuse_size: int = 64
    control_refine_channels: int = 16
    control_use_refine: bool = True
    control_action_dim: int = 7
    control_task_dim: int | None = None
    control_use_context: bool = True
    control_use_action: bool = True
    control_use_task: bool = True
    enable_progress_head: bool = False
    progress_hidden: int = 256
    progress_layers: int = 2
    progress_heads: int = 4
    progress_action_dim: int = 7
    progress_task_dim: int | None = None
    progress_max_horizon: int = 32
    progress_use_action: bool = True
    progress_use_task: bool = True
    enable_bridging: bool = True
    enable_aux_idm: bool = False
    aux_idm_hidden: int = 1024
    aux_idm_layers: int = 3


class JointWorldModel(nn.Module):
    def __init__(self, cfg: JointConfig):
        super().__init__()
        self.cfg = cfg
        self.dual = DualStreamDynamics(cfg.dual)
        self.action_proj = ActionProjHead(
            z_dim=cfg.dual.action.z_dim,
            hidden=cfg.action_proj_hidden,
            n_layers=cfg.action_proj_layers,
        )
        self.geom = GeomDecoder(
            token_dim=cfg.dual.state.D,
            token_grid=int(cfg.dual.state.P ** 0.5),
            hidden=cfg.geom_hidden,
            enable_extra=cfg.enable_geom_extra,
        )
        token_grid = int(cfg.dual.state.P ** 0.5)
        self.pixel = PixelDecoder(
            PixelDecoderConfig(
                token_dim=cfg.dual.state.D,
                token_grid=token_grid,
                hidden=cfg.pixel_hidden,
                n_res=cfg.pixel_n_res,
            )
        ) if cfg.enable_pixel and not cfg.enable_context_pixel else None
        self.context_pixel = ContextResidualPixelDecoder(
            ContextResidualPixelDecoderConfig(
                token_dim=cfg.dual.state.D,
                token_grid=token_grid,
                hidden=cfg.context_pixel_hidden,
                action_dim=cfg.context_pixel_action_dim,
                task_dim=cfg.context_pixel_task_dim or cfg.dual.state.cond_dim,
                residual_scale=cfg.context_pixel_residual_scale,
                use_action=cfg.context_pixel_use_action,
                use_task=cfg.context_pixel_use_task,
                predict_motion=cfg.context_pixel_predict_motion,
                motion_blend_gain=cfg.context_pixel_motion_blend_gain,
            )
        ) if cfg.enable_pixel and cfg.enable_context_pixel else None
        self.control_head = ControlHead(
            ControlHeadConfig(
                token_dim=cfg.dual.state.D,
                hidden=cfg.control_hidden,
                output_size=cfg.control_output_size,
                fuse_size=cfg.control_fuse_size,
                refine_channels=cfg.control_refine_channels,
                use_refine=cfg.control_use_refine,
                action_dim=cfg.control_action_dim,
                task_dim=cfg.control_task_dim or cfg.dual.state.cond_dim,
                use_context=cfg.control_use_context,
                use_action=cfg.control_use_action,
                use_task=cfg.control_use_task,
            )
        ) if cfg.enable_control_head else None
        self.progress_head = ProgressHead(
            ProgressHeadConfig(
                token_dim=cfg.dual.state.D,
                hidden=cfg.progress_hidden,
                n_layers=cfg.progress_layers,
                n_heads=cfg.progress_heads,
                action_dim=cfg.progress_action_dim,
                task_dim=cfg.progress_task_dim or cfg.dual.state.cond_dim,
                max_horizon=cfg.progress_max_horizon,
                use_action=cfg.progress_use_action,
                use_task=cfg.progress_use_task,
            )
        ) if cfg.enable_progress_head else None
        self.bridging = BridgingAdapter() if cfg.enable_bridging else None
        if cfg.enable_aux_idm:
            self.aux_idm = AuxIDM(AuxIDMConfig(
                token_dim=cfg.dual.state.D,
                n_tokens=cfg.dual.state.P,
                hidden=cfg.aux_idm_hidden,
                n_layers=cfg.aux_idm_layers,
                k=cfg.dual.state.k,
            ))
        else:
            self.aux_idm = None

    def load_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Copy pre-computed per-axis mean[6]/std[6] into ActionProjHead buffers."""
        self.action_proj.mean.copy_(mean.float().view(1, 1, 6))
        self.action_proj.std.copy_(std.float().view(1, 1, 6))

    def forward(self, s: torch.Tensor, c: torch.Tensor,
                action_cond: torch.Tensor | None = None,
                context_rgb: torch.Tensor | None = None,
                pixel: bool = True, bridging: bool = False,
                aux_idm: bool = False) -> dict:
        dual_out = self.dual(s, c, action_cond=action_cond)
        pred = dual_out["pred_tokens"]
        proj = self.action_proj(dual_out["z_a"])
        geom = self.geom(pred)
        out = {
            "pred_tokens": pred,
            "z_a": dual_out["z_a"],
            "pose_norm": proj["pose_norm"],
            "pose": proj["pose"],
            "gripper_logit": proj["gripper_logit"],
            "depth": geom["depth"],
        }
        if "point" in geom:
            out["point"] = geom["point"]
        if "pose" in geom:
            out["pose_geom"] = geom["pose"]
        if pixel and self.context_pixel is not None:
            if context_rgb is None:
                raise ValueError("context_rgb is required when context pixel renderer is enabled")
            render = self.context_pixel(
                pred,
                context_rgb,
                action_cond=action_cond,
                task_emb=c,
                return_aux=self.cfg.context_pixel_predict_motion,
            )
            if isinstance(render, dict):
                out.update(render)
            else:
                out["rgb"] = render
        elif pixel and self.pixel is not None:
            out["rgb"] = self.pixel(pred)
        if self.control_head is not None:
            out.update(self.control_head(
                pred,
                geom["depth"],
                context_rgb=context_rgb,
                action_cond=action_cond,
                task_emb=c,
            ))
        if self.progress_head is not None:
            out.update(self.progress_head(
                pred,
                action_cond=action_cond,
                task_emb=c,
            ))
        if bridging and self.bridging is not None:
            out["cosmos_depth_input"] = self.bridging(geom["depth"])
        if aux_idm and self.aux_idm is not None:
            s_last = s[:, -1]
            pred_last = pred[:, -1]
            aux = self.aux_idm(s_last, pred_last)
            out["aux_pose_norm"] = aux["aux_pose_norm"]
            out["aux_grip"] = aux["aux_grip"]
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
