"""v3 demo: side-by-side RGB pred / GT + depth pred / GT for sample clips."""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


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
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_bridging=cfg["model"].get("enable_bridging", True),
    )
    return JointWorldModel(jc)


def depth_to_rgb(d: np.ndarray) -> np.ndarray:
    """d: [H, W] float -> uint8 [H, W, 3] viridis."""
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    d = (d - d.min()) / max(1e-6, d.max() - d.min())
    rgb = (cmap(d)[..., :3] * 255).astype(np.uint8)
    return rgb


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_clips", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    wcfg = WindowConfig(T=cfg["data"]["T"], k=cfg["data"]["k"],
                        stride=cfg["data"]["stride"],
                        cache_root=Path(cfg["data"]["cache_root"]))
    ds = OXEWindowDataset(records, wcfg)
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(len(ds), generator=g).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))
    val_idx = perm[:n_val]
    print(f"val windows: {len(val_idx)}")

    seen_clips: dict[tuple[str, str], int] = {}
    picks: list[int] = []
    for vi in val_idx:
        smp = ds[vi]
        key = (smp["dataset"], smp["clip_id"])
        if key in seen_clips:
            continue
        seen_clips[key] = vi
        picks.append(vi)
        if len(picks) >= args.n_clips:
            break
    print(f"picked {len(picks)} unique clips")

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={sd.get('val_total'):.4f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, vi in enumerate(picks):
        smp = ds[vi]
        s = smp["s_in"].unsqueeze(0).to(device)
        c = smp["c"].unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(s, c, pixel=True, bridging=False)
        rgb_pred = out["rgb"][0].float().clamp(0, 1).cpu().numpy()              # [k, 3, 256, 256]
        depth_pred = out["depth"][0].float().cpu().numpy()                       # [k, 224, 224]
        rgb_gt = smp["rgb_tgt"].numpy()                                          # [k, 256, 256, 3]
        depth_gt = smp["depth_tgt"].numpy()                                      # [k, 224, 224]
        rgb_in = smp["rgb_in"].numpy()                                           # [T, 256, 256, 3]
        k = rgb_pred.shape[0]

        from PIL import Image
        frames = []
        for j in range(k):
            rp = (rgb_pred[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            rg = (rgb_gt[j] * 255).astype(np.uint8)
            dp = np.array(Image.fromarray(depth_to_rgb(depth_pred[j])).resize((256, 256)))
            dg = np.array(Image.fromarray(depth_to_rgb(depth_gt[j])).resize((256, 256)))
            top = np.concatenate([rp, rg], axis=1)
            bot = np.concatenate([dp, dg], axis=1)
            grid = np.concatenate([top, bot], axis=0)
            frames.append(grid)
        # Save as GIF
        clip_id = smp["clip_id"].replace("/", "__")
        gif_path = args.out_dir / f"{ix:02d}_{smp['dataset']}_{clip_id}.gif"
        imageio.mimsave(gif_path, frames, duration=0.2, loop=0)
        print(f"  -> {gif_path}  pose0={out['pose'][0,0].float().cpu().tolist()}")
        # Also dump action prediction vs GT for the clip
        np.savez(
            args.out_dir / f"{ix:02d}_{smp['dataset']}_{clip_id}_action.npz",
            pose_pred=out["pose"][0].float().cpu().numpy(),
            grip_pred=torch.sigmoid(out["gripper_logit"][0]).float().cpu().numpy(),
            action_gt=smp["action_tgt"].numpy(),
        )


if __name__ == "__main__":
    main()
