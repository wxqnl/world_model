"""v4 sampling demo: DDIM 25-step + side-by-side compare vs v3 PixelDecoder + GT.

For each clip:
  - Load v4 ckpt (replaces v3 PixelDecoder with DiffusionHead)
  - Load v3 PixelDecoder weights separately for comparison column
  - Sample k=8 future RGB via DDIM
  - Build a 3-column gif: v3 PixelDec / v4 Diffusion / GT
"""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import _safe
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel

from wm3d_v4.models.diffusion_head import DiffusionHeadConfig
from wm3d_v4.models.joint_v4 import JointV4, JointV4Config
from wm3d_v4.models.vae_wrapper import VAEWrapper
from wm3d_v4.schedulers import CosineSchedule


def build_v3(cfg):
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    jc = JointConfig(
        dual=dc,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        pixel_hidden=768,
        pixel_n_res=2,
        enable_pixel=True,
        enable_bridging=True,
    )
    return JointWorldModel(jc)


def build_v4(cfg):
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    diff_cfg = DiffusionHeadConfig(
        latent_channels=4, latent_size=32, patch_size=cfg["diff"].get("patch_size", 2),
        hidden=cfg["diff"]["hidden"], n_layers=cfg["diff"]["n_layers"],
        n_heads=cfg["diff"]["n_heads"], mlp_ratio=cfg["diff"].get("mlp_ratio", 4.0),
        cond_dim=cfg["model"]["state"]["D"], cond_seq_len=cfg["model"]["state"]["P"],
        timestep_dim=cfg["diff"].get("timestep_dim", 256),
        dropout=0.0,
    )
    jc = JointV4Config(
        dual=dc, diff=diff_cfg,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        freeze_v3=True,
    )
    return JointV4(jc)


def annotate(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 7 + 6, 18], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.array(im)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4_cfg", type=Path, required=True)
    ap.add_argument("--v3_cfg", type=Path, required=True)
    ap.add_argument("--v4_ckpt", type=Path, required=True)
    ap.add_argument("--v3_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_ids", nargs="+", required=True)
    ap.add_argument("--n_steps", type=int, default=25, help="DDIM steps")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cell", type=int, default=256)
    ap.add_argument("--shared_noise", action="store_true",
                    help="share init noise across k frames for temporal coherence")
    args = ap.parse_args()

    v4cfg = yaml.safe_load(args.v4_cfg.read_text())
    v3cfg = yaml.safe_load(args.v3_cfg.read_text())
    cache_root = Path(v4cfg["data"]["cache_root"])
    T = v4cfg["data"]["T"]; k = v4cfg["data"]["k"]
    device = torch.device("cuda:0"); torch.cuda.set_device(0)

    # Build v3
    v3 = build_v3(v3cfg).to(device).eval()
    sd3 = torch.load(args.v3_ckpt, map_location=device, weights_only=False)
    v3.load_state_dict(sd3["model"])
    print(f"v3 loaded, epoch={sd3.get('epoch')} val={sd3.get('val_total'):.4f}")

    # Build v4
    v4 = build_v4(v4cfg).to(device).eval()
    sd4 = torch.load(args.v4_ckpt, map_location=device, weights_only=False)
    v4.load_state_dict(sd4["model"])
    print(f"v4 loaded, epoch={sd4.get('epoch')} val={sd4.get('val'):.4f}")

    vae = VAEWrapper(pretrained=v4cfg["model"].get("vae_pretrained", "stabilityai/sd-vae-ft-mse")).to(device).eval()
    schedule = CosineSchedule(num_train_timesteps=v4cfg["diff"].get("num_train_timesteps", 1000), device=device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, cid in enumerate(args.clip_ids):
        safe = _safe(cid)
        pooled = np.array(np.load(cache_root / "vggt_pooled" / f"{safe}.npy"))
        rgb_full = np.array(np.load(cache_root / "rgb_256" / f"{safe}.npy"))
        qwen_p = cache_root / "qwen_taskemb" / f"{safe}.npy"
        qwen = np.load(qwen_p) if qwen_p.exists() else np.zeros(2048, dtype=np.float16)
        n_frames = pooled.shape[0]
        s0 = args.start
        if s0 + T + k > n_frames:
            s0 = max(0, n_frames - T - k)
        s = torch.from_numpy(pooled[s0:s0 + T]).float().unsqueeze(0).to(device)
        c = torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float().unsqueeze(0).to(device)

        # v3 forward
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v3_out = v3(s, c, pixel=True, bridging=False)
        v3_rgb = v3_out["rgb"][0].float().clamp(0, 1).cpu().numpy()  # [k,3,256,256]
        depth_pred = v3_out["depth"][0].float().cpu().numpy()         # [k,224,224]

        # v4 sample
        v4_out = v4.forward_sample(s, c, vae, schedule,
                                    n_steps=args.n_steps,
                                    shared_noise=args.shared_noise)
        v4_rgb = v4_out["rgb"][0].float().clamp(0, 1).cpu().numpy()

        rgb_gt = rgb_full[s0 + T:s0 + T + k]                          # [k,256,256,3] uint8
        # depth GT from cache
        depth_gt = np.array(np.load(cache_root / "vggt_geom" / f"{safe}.npz")["depth"])[s0 + T:s0 + T + k]

        import matplotlib
        cmap = matplotlib.colormaps["viridis"]
        def depth_to_rgb(d):
            n = (d - d.min()) / max(1e-6, d.max() - d.min())
            return (cmap(n)[..., :3] * 255).astype(np.uint8)

        cell = args.cell
        frames = []
        for j in range(k):
            v3f = (v3_rgb[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            v3f = np.array(Image.fromarray(v3f).resize((cell, cell)))
            v4f = (v4_rgb[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            v4f = np.array(Image.fromarray(v4f).resize((cell, cell)))
            gtf = np.array(Image.fromarray(rgb_gt[j]).resize((cell, cell)))
            dp = np.array(Image.fromarray(depth_to_rgb(depth_pred[j])).resize((cell, cell)))
            dg = np.array(Image.fromarray(depth_to_rgb(depth_gt[j])).resize((cell, cell)))
            blank = np.full_like(dp, 32)  # filler for v4 depth (same as v3 — frozen backbone)
            v3a = annotate(v3f, f"v3 PixelDec t={j}")
            v4a = annotate(v4f, f"v4 Diff{'+share' if args.shared_noise else ''} t={j}")
            gta = annotate(gtf, f"GT RGB t={j}")
            dpa = annotate(dp, f"depth pred t={j}")
            blnk = annotate(blank, "(same as v3 depth)")
            dga = annotate(dg, f"depth GT t={j}")
            top = np.concatenate([v3a, v4a, gta], axis=1)
            bot = np.concatenate([dpa, blnk, dga], axis=1)
            frames.append(np.concatenate([top, bot], axis=0))
        out_gif = args.out_dir / f"v4cmp_{ix:02d}_{safe}.gif"
        imageio.mimsave(out_gif, frames, duration=0.3, loop=0)
        print(f"  -> {out_gif}")


if __name__ == "__main__":
    main()
