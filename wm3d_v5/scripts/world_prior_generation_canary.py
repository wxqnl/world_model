#!/usr/bin/env python
"""CPU canary for text-conditioned world-prior generation and prior-Hunyuan plumbing."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.hunyuan_latent_adapter import HunyuanLatentAdapter, HunyuanLatentAdapterConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.training.train import compute_hunyuan_latent_loss


class FakeLatentDist:
    def __init__(self, latents: torch.Tensor):
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class FakePosterior:
    def __init__(self, latents: torch.Tensor):
        self.latent_dist = FakeLatentDist(latents)


class FakeVAE:
    dtype = torch.float32

    class config:
        scaling_factor = 1.0

    def __init__(self, channels: int, latent_t: int, latent_hw: int):
        self.channels = channels
        self.latent_t = latent_t
        self.latent_hw = latent_hw

    def encode(self, video: torch.Tensor) -> FakePosterior:
        bsz = video.shape[0]
        base = video.mean(dim=(1, 2, 3, 4), keepdim=True)
        latents = base.expand(bsz, self.channels, self.latent_t, self.latent_hw, self.latent_hw).contiguous()
        return FakePosterior(latents)


def build_tiny_model(pixel: bool = False) -> JointWorldModel:
    state = StateConfig(T=2, P=4, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                        cond_dim=16, action_cond_dim=7)
    action = ActionConfig(T=2, P=4, D=16, hidden=32, n_layers=1, n_heads=4, k=2,
                          z_dim=8, cond_dim=16, action_cond_dim=7)
    cfg = JointConfig(
        dual=DualConfig(state=state, action=action, xattn_layers_state=(), xattn_n_heads=4),
        action_proj_hidden=32,
        action_proj_layers=2,
        geom_hidden=16,
        enable_geom_extra=False,
        enable_pixel=pixel,
        enable_context_pixel=pixel,
        context_pixel_hidden=16,
        context_pixel_action_dim=7,
        context_pixel_task_dim=16,
        context_pixel_use_action=True,
        context_pixel_use_task=True,
        enable_bridging=False,
        enable_world_prior=True,
        world_prior_hidden=32,
        world_prior_layers=1,
        world_prior_heads=4,
        world_prior_task_dim=16,
        world_prior_action_dim=7,
    )
    return JointWorldModel(cfg).eval()


def run_generation_modes() -> dict:
    model = build_tiny_model(pixel=False)
    task = torch.randn(2, 16)
    context = torch.randn(2, 2, 4, 16)
    action = torch.randn(2, 2, 7)
    modes = {
        "text_only": (None, None),
        "text_context": (context, None),
        "text_action": (None, action),
        "full": (context, action),
    }
    result = {}
    for name, (ctx, act) in modes.items():
        out = model.generate_world_prior(task, context_tokens=ctx, action_cond=act, steps=2, pixel=False)
        result[name] = {
            "future_tokens": list(out["prior_future_tokens"].shape),
            "depth": list(out["prior_depth"].shape),
            "token_abs_mean": float(out["prior_future_tokens"].abs().mean()),
            "depth_mean": float(out["prior_depth"].mean()),
        }
    rgb_model = build_tiny_model(pixel=True)
    rgb_out = rgb_model.generate_world_prior(torch.randn(1, 16), steps=1, pixel=True)
    result["text_only_pixel"] = {
        "rgb": list(rgb_out["prior_rgb"].shape),
        "rgb_mean": float(rgb_out["prior_rgb"].mean()),
    }
    return result


def run_config_shaped_prior_hunyuan(cfg: dict) -> dict:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    token_dim = int(model_cfg["state"]["D"])
    patches = int(model_cfg["state"]["P"])
    horizon = int(data_cfg["k"])
    grid = int(math.isqrt(patches))
    assert grid * grid == patches
    adapter = HunyuanLatentAdapter(HunyuanLatentAdapterConfig(
        token_dim=token_dim,
        token_grid=grid,
        hidden=8,
        latent_channels=4,
        latent_time=2,
        latent_hw=8,
        action_dim=7,
        task_dim=int(model_cfg["state"].get("cond_dim", 2048)),
        n_blocks=1,
        use_motion=False,
        use_rough_rgb=False,
    ))
    bsz = 1
    out = {
        "pred_tokens": torch.randn(bsz, horizon, patches, token_dim),
        "depth": torch.rand(bsz, horizon, 64, 64),
        "prior_hunyuan_tokens": torch.randn(bsz, horizon, patches, token_dim),
        "prior_hunyuan_depth": torch.rand(bsz, horizon, 64, 64),
    }
    losses = compute_hunyuan_latent_loss(
        adapter,
        FakeVAE(channels=4, latent_t=2, latent_hw=8),
        out,
        {"rgb_tgt_p": torch.rand(bsz, horizon, 3, 32, 32)},
        torch.rand(bsz, 3, 32, 32),
        torch.randn(bsz, horizon, 7),
        torch.randn(bsz, int(model_cfg["state"].get("cond_dim", 2048))),
        {
            "enable_prior_hunyuan_latent_loss": True,
            "prior_hunyuan_latent_weight": 1.0,
            "hunyuan_use_rough_rgb": False,
            "hunyuan_latent_mse_weight": 1.0,
            "hunyuan_latent_l1_weight": 0.0,
            "hunyuan_latent_temporal_weight": 0.0,
            "hunyuan_latent_motion_weight": 0.0,
        },
    )
    for key in ("L_hunyuan_latent", "L_prior_hunyuan_latent"):
        if not torch.isfinite(losses[key]):
            raise RuntimeError(f"non-finite {key}")
    return {
        "token_dim": token_dim,
        "patches": patches,
        "horizon": horizon,
        "L_hunyuan_latent": float(losses["L_hunyuan_latent"]),
        "L_prior_hunyuan_latent": float(losses["L_prior_hunyuan_latent"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg",
        type=Path,
        default=Path("configs/v3_p64_300m_run1_droid_smoke_fromscratch_2node_v1.yaml"),
    )
    args = ap.parse_args()
    torch.manual_seed(0)
    cfg = yaml.safe_load(args.cfg.read_text())
    assert cfg["model"].get("enable_world_prior", False)
    assert cfg["train"].get("condition_dropout", {}).get("text_only_p", 0.0) > 0
    result = {
        "config": str(args.cfg),
        "generation": run_generation_modes(),
        "config_shaped_prior_hunyuan": run_config_shaped_prior_hunyuan(cfg),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
