"""JointWorldModel — one forward, end-to-end.

Input: pooled VGGT tokens [B, T, 64, 2048] (pre-cached) + task embedding c [B, 2048]
Output dict (always):
    pred_tokens:   [B, k, 64, 2048]
    z_a:           [B, k, z_dim]
    pose_norm:     [B, k, 6]   (standardized 6-DoF Δpose)
    pose:          [B, k, 6]   (de-normalized = pose_norm * std + mean)
    gripper_logit: [B, k]
    depth:         [B, k, 224, 224]
    point:         [B, k, 224, 224, 3]
    pose_geom:     [B, k, 9]
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
from .bridging_adapter import BridgingAdapter
from .aux_idm import AuxIDM, AuxIDMConfig


@dataclass
class JointConfig:
    dual: DualConfig = field(default_factory=DualConfig)
    action_proj_hidden: int = 768
    action_proj_layers: int = 5
    geom_hidden: int = 384
    pixel_hidden: int = 768
    pixel_n_res: int = 2
    enable_pixel: bool = True
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
        )
        self.pixel = PixelDecoder(
            PixelDecoderConfig(
                token_dim=cfg.dual.state.D,
                token_grid=int(cfg.dual.state.P ** 0.5),
                hidden=cfg.pixel_hidden,
                n_res=cfg.pixel_n_res,
            )
        ) if cfg.enable_pixel else None
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
                pixel: bool = True, bridging: bool = False,
                aux_idm: bool = False) -> dict:
        dual_out = self.dual(s, c)
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
            "point": geom["point"],
            "pose_geom": geom["pose"],
        }
        if pixel and self.pixel is not None:
            out["rgb"] = self.pixel(pred)
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
