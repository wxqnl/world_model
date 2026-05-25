"""JointWorldModel — one forward, end-to-end.

Input: pooled VGGT tokens [B, T, 64, 2048] (pre-cached) + task embedding c [B, 2048]
Output dict:
    pred_tokens: [B, k, 64, 2048]
    z_a:         [B, k, z_dim]
    pose:        [B, k, 6]   (action 6-DoF pose)
    gripper_logit: [B, k]
    depth:       [B, k, 224, 224]   (geometry head)
    point:       [B, k, 224, 224, 3]
    pose_geom:   [B, k, 9]
    rgb:         [B, k, 3, 256, 256]  (pixel head, end-to-end trained)
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


class JointWorldModel(nn.Module):
    def __init__(self, cfg: JointConfig):
        super().__init__()
        self.cfg = cfg
        self.dual = DualStreamDynamics(cfg.dual)
        self.action_proj = ActionProjHead(
            z_dim=cfg.dual.action.z_dim,
            hidden=cfg.action_proj_hidden,
            n_layers=cfg.action_proj_layers,
            action_dim=7,
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

    def forward(self, s: torch.Tensor, c: torch.Tensor,
                pixel: bool = True, bridging: bool = False) -> dict:
        dual_out = self.dual(s, c)
        pred = dual_out["pred_tokens"]
        proj = self.action_proj(dual_out["z_a"])
        geom = self.geom(pred)
        out = {
            "pred_tokens": pred,
            "z_a": dual_out["z_a"],
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
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
