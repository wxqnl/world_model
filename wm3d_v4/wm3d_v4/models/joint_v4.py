"""v4 joint model: reuse v3 dual+geom+action (frozen), replace PixelDecoder with DiffusionHead.

Forward modes:
- training:  forward_train(s, c, rgb_tgt, vae, schedule) -> dict{eps_pred, eps_noise, ...}
- inference: forward_sample(s, c, vae, schedule, n_steps) -> dict{rgb, depth, pose, ...}
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn

# Import v3 components (in the parent newwm/wm3d_v3 package)
from wm3d_v3.models.dual_stream import DualStreamDynamics, DualConfig
from wm3d_v3.models.action_proj import ActionProjHead
from wm3d_v3.models.geom_decoder import GeomDecoder

from .diffusion_head import DiffusionHead, DiffusionHeadConfig
from .vae_wrapper import VAEWrapper


@dataclass
class JointV4Config:
    dual: DualConfig = field(default_factory=DualConfig)
    action_proj_hidden: int = 768
    action_proj_layers: int = 5
    geom_hidden: int = 384
    diff: DiffusionHeadConfig = field(default_factory=DiffusionHeadConfig)
    vae_pretrained: str = "stabilityai/sd-vae-ft-mse"
    freeze_v3: bool = True


class JointV4(nn.Module):
    def __init__(self, cfg: JointV4Config):
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
        self.diff = DiffusionHead(cfg.diff)
        # VAE is built outside the module to avoid DDP wrapping its frozen params
        if cfg.freeze_v3:
            for m in (self.dual, self.action_proj, self.geom):
                for p in m.parameters():
                    p.requires_grad = False

    def backbone(self, s: torch.Tensor, c: torch.Tensor) -> dict:
        """Run v3 backbone in inference / no-grad mode (frozen)."""
        with torch.no_grad():
            d = self.dual(s, c)
            pred = d["pred_tokens"]
            proj = self.action_proj(d["z_a"])
            geom = self.geom(pred)
        return {
            "pred_tokens": pred,
            "z_a": d["z_a"],
            "pose": proj["pose"],
            "gripper_logit": proj["gripper_logit"],
            "depth": geom["depth"],
            "point": geom["point"],
            "pose_geom": geom["pose"],
        }

    def forward_train(self, s: torch.Tensor, c: torch.Tensor,
                       rgb_tgt: torch.Tensor, vae: VAEWrapper,
                       schedule) -> dict:
        """rgb_tgt: [B, k, 3, 256, 256] in [0, 1]."""
        back = self.backbone(s, c)
        pred_tokens = back["pred_tokens"]                        # [B, k, 64, 2048]
        B, k = rgb_tgt.shape[:2]
        # VAE encode (frozen, no grad). Run in fp32 for stability then cast.
        with torch.autocast(device_type="cuda", enabled=False):
            rgb_flat = rgb_tgt.float().reshape(B * k, 3, 256, 256)
            z0 = vae.encode(rgb_flat).reshape(B, k, 4, 32, 32)
        noise = torch.randn_like(z0)
        t = schedule.sample_timesteps(B * k, z0.device).reshape(B, k)
        zt = schedule.add_noise(z0.reshape(B * k, 4, 32, 32),
                                 noise.reshape(B * k, 4, 32, 32),
                                 t.reshape(B * k)).reshape(B, k, 4, 32, 32)
        eps_pred = self.diff(zt, t, pred_tokens.to(zt.dtype))
        return {
            "eps_pred": eps_pred,
            "eps_target": noise,
            "z0": z0,
            "zt": zt,
            **back,
        }

    @torch.no_grad()
    def forward_sample(self, s: torch.Tensor, c: torch.Tensor,
                        vae: VAEWrapper, schedule, n_steps: int = 25,
                        shared_noise: bool = False) -> dict:
        back = self.backbone(s, c)
        pred_tokens = back["pred_tokens"]
        B, k = pred_tokens.shape[:2]
        z = schedule.ddim_sample(self.diff,
                                  shape=(B, k, 4, 32, 32),
                                  cond=pred_tokens,
                                  n_steps=n_steps,
                                  device=pred_tokens.device,
                                  dtype=torch.bfloat16,
                                  shared_noise=shared_noise)
        rgb_flat = vae.decode(z.reshape(B * k, 4, 32, 32))
        rgb = rgb_flat.reshape(B, k, 3, 256, 256)
        return {"rgb": rgb, **back}

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
