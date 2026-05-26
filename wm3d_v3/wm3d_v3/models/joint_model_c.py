"""JointWorldModelC — depth-augmented IDM that uses VGGT depth as native-3D motion signal.

Adds a small DepthEncoder reading the VGGT depth maps (both input frames and the
model's own predicted future depth via GeomDecoder), produces 8x8 patch tokens,
and concatenates them per-patch with the pooled VGGT tokens fed into IDMStream.
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
from .depth_encoder import DepthEncoder, DepthEncoderConfig


@dataclass
class JointCConfig:
    state: StateConfig = field(default_factory=StateConfig)
    idm: IDMStreamConfig = field(default_factory=IDMStreamConfig)
    depth_enc: DepthEncoderConfig = field(default_factory=DepthEncoderConfig)
    action_proj_hidden: int = 1024
    action_proj_layers: int = 5
    geom_hidden: int = 384
    pixel_hidden: int = 768
    pixel_n_res: int = 2
    enable_pixel: bool = False


class JointWorldModelC(nn.Module):
    def __init__(self, cfg: JointCConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.idm.extra_token_dim == cfg.depth_enc.hidden_d, \
            "IDM.extra_token_dim must equal DepthEncoder.hidden_d"
        self.state = StateStream(cfg.state)
        self.depth_enc = DepthEncoder(cfg.depth_enc)
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
                depth_in: torch.Tensor, pixel: bool = False) -> dict:
        """
        s        : [B, T, P, D]      pooled VGGT tokens (input frames)
        c        : [B, cond_dim]     task embedding
        depth_in : [B, T, 224, 224]  VGGT depth on input frames
        """
        h_s = self.state.encode(s, c)
        for layer in self.state.layers:
            h_s = layer(h_s)
        h_s = self.state.norm(h_s)
        pred = self.state.decode(h_s)                            # [B, k, P, D]
        geom = self.geom(pred)                                   # depth: [B, k, 224, 224]
        depth_pred = geom["depth"]
        # Depth -> patch tokens (8x8 = P=64)
        depth_in_tok = self.depth_enc(depth_in)                  # [B, T, P, dd]
        depth_pred_tok = self.depth_enc(depth_pred)              # [B, k, P, dd]
        z_a = self.idm(s, pred, c,
                        extra_in=depth_in_tok,
                        extra_pred=depth_pred_tok)["z_a"]
        proj = self.action_proj(z_a)
        out = {
            "pred_tokens": pred,
            "z_a": z_a,
            "pose_norm": proj["pose_norm"],
            "pose": proj["pose"],
            "gripper_logit": proj["gripper_logit"],
            "depth": depth_pred,
            "point": geom["point"],
            "pose_geom": geom["pose"],
        }
        if pixel and self.pixel is not None:
            out["rgb"] = self.pixel(pred)
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
