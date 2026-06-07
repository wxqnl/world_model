"""Demo: side-by-side compare v3 vs v3.5 end-to-end vs GT.

Layout per frame:
  top:  v3 RGB | v3.5_e2e RGB | GT RGB
  bot:  v3 depth | v3.5_e2e depth | GT depth
"""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
import matplotlib
from PIL import Image, ImageDraw

from wm3d_v3.data.window_dataset import _safe
from wm3d_v3.models.state_stream import StateConfig
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel


def build_model(cfg):
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
        pixel_hidden=cfg["model"].get("pixel_hidden", 768),
        pixel_n_res=cfg["model"].get("pixel_n_res", 2),
        enable_pixel=True,
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def annotate(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 7 + 6, 18], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.array(im)


CMAP = matplotlib.colormaps["viridis"]


def depth_to_rgb(d):
    n = (d - d.min()) / max(1e-6, d.max() - d.min())
    return (CMAP(n)[..., :3] * 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3_cfg", type=Path, required=True)
    ap.add_argument("--e2e_cfg", type=Path, required=True)
    ap.add_argument("--v3_ckpt", type=Path, required=True)
    ap.add_argument("--e2e_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_ids", nargs="+", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cell", type=int, default=256)
    args = ap.parse_args()

    cfg3 = yaml.safe_load(args.v3_cfg.read_text())
    cfg_e2e = yaml.safe_load(args.e2e_cfg.read_text())
    cache_root = Path(cfg3["data"]["cache_root"])
    T = cfg3["data"]["T"]; k = cfg3["data"]["k"]
    device = torch.device("cuda:0"); torch.cuda.set_device(0)

    v3 = build_model(cfg3).to(device).eval()
    sd3 = torch.load(args.v3_ckpt, map_location=device, weights_only=False)
    v3.load_state_dict(sd3["model"], strict=False)
    print(f"v3 loaded epoch={sd3.get('epoch')} val={sd3.get('val_total'):.4f}")

    e2e = build_model(cfg_e2e).to(device).eval()
    sde2e = torch.load(args.e2e_ckpt, map_location=device, weights_only=False)
    e2e.load_state_dict(sde2e["model"], strict=False)
    print(f"v3.5_e2e loaded epoch={sde2e.get('epoch')} val={sde2e.get('val_total'):.4f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, cid in enumerate(args.clip_ids):
        safe = _safe(cid)
        pooled = np.array(np.load(cache_root / "vggt_pooled" / f"{safe}.npy"))
        rgb_full = np.array(np.load(cache_root / "rgb_256" / f"{safe}.npy"))
        depth_cache = np.array(np.load(cache_root / "vggt_geom" / f"{safe}.npz")["depth"])
        qwen_p = cache_root / "qwen_taskemb" / f"{safe}.npy"
        qwen = np.load(qwen_p) if qwen_p.exists() else np.zeros(2048, dtype=np.float16)
        n_frames = pooled.shape[0]
        s0 = args.start
        if s0 + T + k > n_frames:
            s0 = max(0, n_frames - T - k)
        s = torch.from_numpy(pooled[s0:s0 + T]).float().unsqueeze(0).to(device)
        c = torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float().unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v3_out = v3(s, c, pixel=True, bridging=False)
            e2e_out = e2e(s, c, pixel=True, bridging=False)
        v3_rgb = v3_out["rgb"][0].float().clamp(0, 1).cpu().numpy()
        e2e_rgb = e2e_out["rgb"][0].float().clamp(0, 1).cpu().numpy()
        v3_depth = v3_out["depth"][0].float().cpu().numpy()
        e2e_depth = e2e_out["depth"][0].float().cpu().numpy()
        rgb_gt = rgb_full[s0 + T:s0 + T + k]
        depth_gt = depth_cache[s0 + T:s0 + T + k]

        cell = args.cell
        frames = []
        for j in range(k):
            v3f = (v3_rgb[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            v3f = np.array(Image.fromarray(v3f).resize((cell, cell)))
            e2ef = (e2e_rgb[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            e2ef = np.array(Image.fromarray(e2ef).resize((cell, cell)))
            gtf = np.array(Image.fromarray(rgb_gt[j]).resize((cell, cell)))
            dp3 = np.array(Image.fromarray(depth_to_rgb(v3_depth[j])).resize((cell, cell)))
            dpe = np.array(Image.fromarray(depth_to_rgb(e2e_depth[j])).resize((cell, cell)))
            dg = np.array(Image.fromarray(depth_to_rgb(depth_gt[j])).resize((cell, cell)))
            top = np.concatenate([
                annotate(v3f, f"v3 t={j}"),
                annotate(e2ef, f"v3.5_e2e t={j}"),
                annotate(gtf, f"GT t={j}"),
            ], axis=1)
            bot = np.concatenate([
                annotate(dp3, f"v3 depth t={j}"),
                annotate(dpe, f"e2e depth t={j}"),
                annotate(dg, f"GT depth t={j}"),
            ], axis=1)
            frames.append(np.concatenate([top, bot], axis=0))
        out_gif = args.out_dir / f"v3_e2e_cmp_{ix:02d}_{safe}.gif"
        imageio.mimsave(out_gif, frames, duration=0.3, loop=0)
        print(f"  -> {out_gif}")


if __name__ == "__main__":
    main()
