"""v3 long-horizon demo: autoregressive rollout in token space.

Take T=16 GT frames, predict k=8, slide the window using the model's own
predicted tokens, repeat N_STEPS times to get k*N_STEPS future frames.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.action_condition import make_action_condition_np
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig, _safe
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


def build_model(cfg):
    from wm3d_v3.eval.run_eval import build_model as build_full_model
    return build_full_model(cfg)

def tokens_subdir_from_cfg(cfg: dict) -> str:
    return cfg["data"].get("tokens_subdir", "vggt_pooled")


def depth_to_rgb(d: np.ndarray) -> np.ndarray:
    import matplotlib
    cmap = matplotlib.colormaps["viridis"]
    d = (d - d.min()) / max(1e-6, d.max() - d.min())
    return (cmap(d)[..., :3] * 255).astype(np.uint8)


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    """Draw small label on top-left."""
    from PIL import Image as PImage, ImageDraw
    im = PImage.fromarray(img)
    d = ImageDraw.Draw(im)
    # background box
    d.rectangle([0, 0, len(text) * 7 + 6, 18], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.array(im)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_ids", nargs="+", required=True,
                    help="manifest clip_ids to roll out, e.g. bridge/00035/sample_..")
    ap.add_argument("--n_steps", type=int, default=3,
                    help="number of k=8 rollout chunks (k*n_steps future frames)")
    ap.add_argument("--full", action="store_true",
                    help="rollout enough steps to cover the entire episode after T")
    ap.add_argument("--start", type=int, default=0,
                    help="start frame index in the episode")
    ap.add_argument("--include_context", action="store_true",
                    help="prepend the T=16 input frames to the GIF (labeled 'context')")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    T = cfg["data"]["T"]
    k = cfg["data"]["k"]
    cache_root = Path(cfg["data"]["cache_root"])
    tokens_subdir = tokens_subdir_from_cfg(cfg)
    stats_path = cfg["data"].get("action_stats")
    action_mean = action_std = None
    if stats_path and Path(stats_path).exists():
        stats = np.load(stats_path)
        action_mean, action_std = stats["mean"], stats["std"]

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    val_total = sd.get("val_total")
    val_total_text = f"{val_total:.4f}" if isinstance(val_total, (float, int)) else str(val_total)
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={val_total_text}")
    print(f"action-conditioned rollout: stats={'yes' if action_mean is not None else 'no'}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, cid in enumerate(args.clip_ids):
        safe = _safe(cid)
        pooled = np.array(np.load(cache_root / tokens_subdir / f"{safe}.npy"))
        rgb = np.array(np.load(cache_root / "rgb_256" / f"{safe}.npy"))
        depth = np.array(np.load(cache_root / "vggt_geom" / f"{safe}.npz")["depth"])
        qwen_p = cache_root / "qwen_taskemb" / f"{safe}.npy"
        action_p = cache_root / "actions" / f"{safe}.npy"
        if not action_p.exists():
            raise FileNotFoundError(f"missing action cache for rollout: {action_p}")
        actions = np.array(np.load(action_p), dtype=np.float32)
        qwen = np.load(qwen_p) if qwen_p.exists() else np.zeros(2048, dtype=np.float16)
        n = pooled.shape[0]
        s0 = args.start
        if args.full:
            n_steps = max(1, (n - s0 - T + k - 1) // k)
        else:
            n_steps = args.n_steps
        need = T + k * n_steps
        if s0 + T > n:
            s0 = max(0, n - T)
        print(f"\n[{cid}] n_frames={n}, start={s0}, n_steps={n_steps}, rolling out {k*n_steps} future frames")

        # Initialize token window with GT
        s_window = torch.from_numpy(pooled[s0:s0 + T]).float().unsqueeze(0).to(device)
        context_rgb = torch.from_numpy(rgb[s0 + T - 1]).float().div(255.0)
        context_rgb = context_rgb.permute(2, 0, 1).unsqueeze(0).to(device)
        c = torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float().unsqueeze(0).to(device)

        rgb_pred_all: list[np.ndarray] = []
        depth_pred_all: list[np.ndarray] = []
        pose_pred_all: list[np.ndarray] = []

        for step in range(n_steps):
            print(f"  step {step+1}/{n_steps}", flush=True)
            action_start = s0 + T + step * k
            action_chunk = actions[action_start: action_start + k]
            if action_chunk.shape[0] < k:
                pad = np.zeros((k - action_chunk.shape[0], actions.shape[-1]), dtype=np.float32)
                action_chunk = np.concatenate([action_chunk, pad], axis=0)
            action_cond_np = make_action_condition_np(
                action_chunk, mean=action_mean, std=action_std
            )
            action_cond = torch.from_numpy(action_cond_np).unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(s_window, c, action_cond=action_cond,
                            context_rgb=context_rgb, pixel=True, bridging=False)
            rgb_pred_all.append(out["rgb"][0].float().clamp(0, 1).cpu().numpy())     # [k,3,256,256]
            depth_pred_all.append(out["depth"][0].float().cpu().numpy())              # [k,224,224]
            pose_pred_all.append(out["pose"][0].float().cpu().numpy())                # [k,6]
            # Slide window: [s_window[k:], pred_tokens]
            pred_tok = out["pred_tokens"].float()
            s_window = torch.cat([s_window[:, k:], pred_tok], dim=1)
            context_rgb = out["rgb"][:, -1].float().clamp(0, 1)
            assert s_window.shape[1] == T

        rgb_pred_all_np = np.concatenate(rgb_pred_all, axis=0)      # [k*n, 3, 256, 256]
        depth_pred_all_np = np.concatenate(depth_pred_all, axis=0)
        pose_pred_all_np = np.concatenate(pose_pred_all, axis=0)    # [k*n, 6]

        # GT future, only available for as many frames as the episode allows
        future_start = s0 + T
        future_end = min(n, future_start + rgb_pred_all_np.shape[0])
        rgb_gt = rgb[future_start:future_end]                       # [<=k*n, 256, 256, 3]
        depth_gt = depth[future_start:future_end]

        frames = []
        # Optionally prepend the T=16 input context frames (RGB/depth GT)
        if args.include_context:
            for i in range(T):
                rg = rgb[s0 + i].astype(np.uint8)
                dg = np.array(Image.fromarray(depth_to_rgb(depth[s0 + i])).resize((256, 256)))
                rp_l = annotate(rg.copy(), f"CONTEXT GT t={i}")
                rg_l = annotate(rg.copy(), f"CONTEXT GT t={i}")
                dp_l = annotate(dg.copy(), f"context depth t={i}")
                dg_l = annotate(dg.copy(), f"context depth t={i}")
                top = np.concatenate([rp_l, rg_l], axis=1)
                bot = np.concatenate([dp_l, dg_l], axis=1)
                frames.append(np.concatenate([top, bot], axis=0))
        H_total = rgb_pred_all_np.shape[0]
        for j in range(H_total):
            rp = (rgb_pred_all_np[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            dp = np.array(Image.fromarray(depth_to_rgb(depth_pred_all_np[j])).resize((256, 256)))
            if j < len(rgb_gt):
                rg = rgb_gt[j].astype(np.uint8) if rgb_gt[j].dtype != np.uint8 else rgb_gt[j]
                dg = np.array(Image.fromarray(depth_to_rgb(depth_gt[j])).resize((256, 256)))
            else:
                rg = np.full_like(rp, 32)
                dg = np.full_like(dp, 32)
            chunk_idx = j // k
            rp_l = annotate(rp, f"RGB pred (step {chunk_idx+1}/{n_steps}, t={T+j})")
            rg_l = annotate(rg, f"RGB GT t={T+j}" if j < len(rgb_gt) else "no GT")
            dp_l = annotate(dp, f"depth pred t={T+j}")
            dg_l = annotate(dg, f"depth GT t={T+j}" if j < len(depth_gt) else "no GT")
            top = np.concatenate([rp_l, rg_l], axis=1)
            bot = np.concatenate([dp_l, dg_l], axis=1)
            grid = np.concatenate([top, bot], axis=0)
            frames.append(grid)

        safe = cid.replace("/", "__")
        out_gif = args.out_dir / f"long_{ix:02d}_{safe}_n{n_steps}.gif"
        imageio.mimsave(out_gif, frames, duration=0.18, loop=0)
        np.savez(
            args.out_dir / f"long_{ix:02d}_{safe}_n{n_steps}_action.npz",
            pose_pred=pose_pred_all_np,
        )
        print(f"  -> {out_gif}  ({H_total} frames, GT available for {len(rgb_gt)})")


if __name__ == "__main__":
    main()
