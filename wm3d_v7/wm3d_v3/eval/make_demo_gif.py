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
from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.window_dataset import OXEWindowDataset
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig

# Re-export so demo modules (make_hunyuan_latent_demo, make_hunyuan_dit_control_demo)
# and tests can `from wm3d_v3.eval.make_demo_gif import window_config_from_cfg`.
# Canonical definition lives in run_eval; run_eval does not import this module, so
# this module-level import is cycle-free.
from wm3d_v3.eval.run_eval import window_config_from_cfg  # noqa: E402,F401


def build_model(cfg):
    from wm3d_v3.eval.run_eval import build_model as build_full_model
    return build_full_model(cfg)


def depth_to_rgb(d: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """d: [H, W] float -> uint8 [H, W, 3] viridis."""
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    lo = float(np.nanmin(d)) if vmin is None else float(vmin)
    hi = float(np.nanmax(d)) if vmax is None else float(vmax)
    d = (d - lo) / max(1e-6, hi - lo)
    d = np.clip(d, 0.0, 1.0)
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
    from wm3d_v3.eval.run_eval import window_config_from_cfg
    wcfg = window_config_from_cfg(cfg)
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

    from wm3d_v3.eval.run_eval import build_model as build_full_model
    model = build_full_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    val_total = sd.get("val_total")
    val_total_text = f"{val_total:.4f}" if isinstance(val_total, (float, int)) else str(val_total)
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={val_total_text}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, vi in enumerate(picks):
        smp = ds[vi]
        s = smp["s_in"].unsqueeze(0).to(device)
        c = smp["c"].unsqueeze(0).to(device)
        action_tgt = smp["action_tgt"].unsqueeze(0).to(device)
        action_tgt_norm = smp["action_tgt_norm"].unsqueeze(0).to(device)
        context_rgb = smp["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(s, c, action_cond=action_cond, context_rgb=context_rgb,
                        pixel=True, bridging=False)
        rgb_pred = out["rgb"][0].float().clamp(0, 1).cpu().numpy()              # [k, 3, 256, 256]
        depth_pred_t = out["depth"][0].float()
        depth_gt_t = smp["depth_tgt"].float()
        if depth_pred_t.shape[-2:] != depth_gt_t.shape[-2:]:
            depth_pred_t = torch.nn.functional.interpolate(
                depth_pred_t[:, None],
                size=depth_gt_t.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        depth_pred = depth_pred_t.cpu().numpy()
        rgb_gt = smp["rgb_tgt"].numpy()                                          # [k, 256, 256, 3]
        depth_gt = depth_gt_t.numpy()
        rgb_in = smp["rgb_in"].numpy()                                           # [T, 256, 256, 3]
        k = rgb_pred.shape[0]

        from PIL import Image
        frames = []
        for j in range(k):
            rp = (rgb_pred[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            rg = (rgb_gt[j] * 255).astype(np.uint8)
            both = np.concatenate([depth_pred[j].reshape(-1), depth_gt[j].reshape(-1)])
            vmin, vmax = np.nanpercentile(both, [2, 98])
            dp = np.array(Image.fromarray(depth_to_rgb(depth_pred[j], vmin=vmin, vmax=vmax)).resize((256, 256)))
            dg = np.array(Image.fromarray(depth_to_rgb(depth_gt[j], vmin=vmin, vmax=vmax)).resize((256, 256)))
            top = np.concatenate([rp, rg], axis=1)
            bot = np.concatenate([dp, dg], axis=1)
            grid = np.concatenate([top, bot], axis=0)
            frames.append(grid)
        # Save as GIF
        clip_id = smp["clip_id"].replace("/", "__")
        gif_path = args.out_dir / f"{ix:02d}_{smp['dataset']}_{clip_id}.gif"
        imageio.mimsave(gif_path, frames, duration=0.2, loop=0)
        print(f"  -> {gif_path}  pose0={out['pose'][0,0].float().cpu().tolist()}")
        if "motion_hint" in out:
            motion_pred = out["motion_hint"][0].float().clamp(0, 1).cpu().numpy()
            ref = rgb_in[-1]
            motion_gt = (np.abs(rgb_gt - ref[None]).mean(axis=-1, keepdims=True) > 0.03).astype(np.float32)
            motion_frames = []
            for j in range(k):
                mp = (motion_pred[j, 0] * 255).astype(np.uint8)
                mg = (motion_gt[j, :, :, 0] * 255).astype(np.uint8)
                if mp.shape != mg.shape:
                    mp = np.array(Image.fromarray(mp).resize((mg.shape[1], mg.shape[0])))
                mp = np.repeat(mp[..., None], 3, axis=-1)
                mg = np.repeat(mg[..., None], 3, axis=-1)
                motion_frames.append(np.concatenate([mp, mg], axis=1))
            motion_path = args.out_dir / f"{ix:02d}_{smp['dataset']}_{clip_id}_motion.gif"
            imageio.mimsave(motion_path, motion_frames, duration=0.2, loop=0)
        # Also dump action prediction vs GT for the clip
        np.savez(
            args.out_dir / f"{ix:02d}_{smp['dataset']}_{clip_id}_action.npz",
            pose_pred=out["pose"][0].float().cpu().numpy(),
            grip_pred=torch.sigmoid(out["gripper_logit"][0]).float().cpu().numpy(),
            action_gt=smp["action_tgt"].numpy(),
        )


if __name__ == "__main__":
    main()
