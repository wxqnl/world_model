"""JointWorldModelB — replaces ActionStream with IDMStream that sees s_in + pred_tokens.

State stream produces pred_tokens; IDMStream then attends jointly over both
to produce z_a. ActionProjHead consumes z_a (per-axis standardized output).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .state_stream import StateStream, StateConfig
from .action_proj import ActionProjHead
from .geom_decoder import GeomDecoder
from .pixel_decoder import PixelDecoder, PixelDecoderConfig
from .idm_stream import IDMStream, IDMStreamConfig


@dataclass
class JointBConfig:
    state: StateConfig = field(default_factory=StateConfig)
    idm: IDMStreamConfig = field(default_factory=IDMStreamConfig)
    action_proj_hidden: int = 1024
    action_proj_layers: int = 5
    geom_hidden: int = 384
    pixel_hidden: int = 768
    pixel_n_res: int = 2
    enable_pixel: bool = False


class JointWorldModelB(nn.Module):
    def __init__(self, cfg: JointBConfig):
        super().__init__()
        self.cfg = cfg
        self.state = StateStream(cfg.state)
        self.idm = IDMStream(cfg.idm)
        self.action_proj = ActionProjHead(
            z_dim=cfg.idm.z_dim,
            hidden=cfg.action_proj_hidden,
            n_layers=cfg.action_proj_layers,
        )
        self.geom = GeomDecoder(
            token_dim=cfg.state.D,
            token_grid=int(cfg.state.P ** 0.5),
            hidden=cfg.geom_hidden,
        )
        self.pixel = PixelDecoder(
            PixelDecoderConfig(
                token_dim=cfg.state.D,
                token_grid=int(cfg.state.P ** 0.5),
                hidden=cfg.pixel_hidden,
                n_res=cfg.pixel_n_res,
            )
        ) if cfg.enable_pixel else None

    def load_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_proj.mean.copy_(mean.float().view(1, 1, 6))
        self.action_proj.std.copy_(std.float().view(1, 1, 6))

    def forward(self, s: torch.Tensor, c: torch.Tensor,
                pixel: bool = False) -> dict:
        h_s = self.state.encode(s, c)
        for layer in self.state.layers:
            h_s = layer(h_s)
        h_s = self.state.norm(h_s)
        pred = self.state.decode(h_s)
        z_a = self.idm(s, pred, c)["z_a"]
        proj = self.action_proj(z_a)
        geom = self.geom(pred)
        out = {
            "pred_tokens": pred,
            "z_a": z_a,
            "pose_norm": proj["pose_norm"],
            "pose": proj["pose"],
            "gripper_logit": proj["gripper_logit"],
            "depth": geom["depth"],
            "point": geom["point"],
            "pose_geom": geom["pose"],
        }
        if pixel and self.pixel is not None:
            out["rgb"] = self.pixel(pred)
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
