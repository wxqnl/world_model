"""v4 long-horizon autoregressive rollout with DDIM sampling.

For each clip:
  - Take T=16 GT frames as initial input
  - Predict next k=8 (token + RGB via DDIM)
  - Slide window: drop earliest k, append predicted tokens, repeat
  - Output a full-episode-length GIF with v4 RGB pred and GT side by side
"""
from __future__ import annotations
import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from wm3d_v3.data.window_dataset import _safe
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.state_stream import StateConfig

from wm3d_v4.models.diffusion_head import DiffusionHeadConfig
from wm3d_v4.models.joint_v4 import JointV4, JointV4Config
from wm3d_v4.models.vae_wrapper import VAEWrapper
from wm3d_v4.schedulers import CosineSchedule


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
    ap.add_argument("--v4_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_ids", nargs="+", required=True)
    ap.add_argument("--n_steps_diff", type=int, default=20, help="DDIM steps per chunk")
    ap.add_argument("--n_rollout", type=int, default=0,
                    help="number of k-chunk rollout steps (0=auto-cover full episode)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--include_context", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.v4_cfg.read_text())
    cache_root = Path(cfg["data"]["cache_root"])
    T = cfg["data"]["T"]; k = cfg["data"]["k"]
    device = torch.device("cuda:0"); torch.cuda.set_device(0)

    v4 = build_v4(cfg).to(device).eval()
    sd = torch.load(args.v4_ckpt, map_location=device, weights_only=False)
    v4.load_state_dict(sd["model"])
    print(f"v4 loaded, epoch={sd.get('epoch')} val={sd.get('val'):.4f}")
    vae = VAEWrapper(pretrained=cfg["model"].get("vae_pretrained", "stabilityai/sd-vae-ft-mse")).to(device).eval()
    schedule = CosineSchedule(num_train_timesteps=cfg["diff"].get("num_train_timesteps", 1000), device=device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, cid in enumerate(args.clip_ids):
        safe = _safe(cid)
        pooled = np.array(np.load(cache_root / "vggt_pooled" / f"{safe}.npy"))
        rgb_full = np.array(np.load(cache_root / "rgb_256" / f"{safe}.npy"))
        qwen_p = cache_root / "qwen_taskemb" / f"{safe}.npy"
        qwen = np.load(qwen_p) if qwen_p.exists() else np.zeros(2048, dtype=np.float16)
        n_frames = pooled.shape[0]
        s0 = args.start
        if args.n_rollout > 0:
            n_steps = args.n_rollout
        else:
            n_steps = max(1, (n_frames - s0 - T + k - 1) // k)
        if s0 + T > n_frames:
            s0 = max(0, n_frames - T)
        print(f"[{cid}] n_frames={n_frames}, s0={s0}, n_rollout={n_steps}, total_future={k * n_steps}")

        s_win = torch.from_numpy(pooled[s0:s0 + T]).float().unsqueeze(0).to(device)
        c = torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float().unsqueeze(0).to(device)

        rgb_pred_all: list[np.ndarray] = []
        for step in range(n_steps):
            print(f"  step {step+1}/{n_steps}", flush=True)
            # 1) run v3 backbone to get pred_tokens (also needed for chaining)
            with torch.no_grad():
                back = v4.backbone(s_win, c)
            pred_tokens = back["pred_tokens"]            # [1, k, 64, 2048]
            # 2) sample RGB via DDIM
            z = schedule.ddim_sample(v4.diff,
                                      shape=(1, k, 4, 32, 32),
                                      cond=pred_tokens,
                                      n_steps=args.n_steps_diff,
                                      device=device,
                                      dtype=torch.bfloat16)
            rgb = vae.decode(z.reshape(k, 4, 32, 32)).reshape(1, k, 3, 256, 256)
            rgb_pred_all.append(rgb[0].float().clamp(0, 1).cpu().numpy())
            # 3) slide window with predicted tokens
            s_win = torch.cat([s_win[:, k:], pred_tokens], dim=1)
            assert s_win.shape[1] == T

        rgb_pred_all_np = np.concatenate(rgb_pred_all, axis=0)            # [k*n,3,256,256]
        future_start = s0 + T
        future_end = min(n_frames, future_start + rgb_pred_all_np.shape[0])
        rgb_gt = rgb_full[future_start:future_end]                         # [<=k*n,256,256,3]

        frames = []
        if args.include_context:
            for i in range(T):
                gt = rgb_full[s0 + i].astype(np.uint8)
                f = annotate(gt, f"CONTEXT t={i}")
                frames.append(np.concatenate([f, f], axis=1))
        H_total = rgb_pred_all_np.shape[0]
        for j in range(H_total):
            rp = (rgb_pred_all_np[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            if j < len(rgb_gt):
                rg = rgb_gt[j].astype(np.uint8)
            else:
                rg = np.full_like(rp, 32)
            chunk_idx = j // k
            rp_l = annotate(rp, f"v4 Diff DDIM-{args.n_steps_diff} step{chunk_idx+1}/{n_steps} t={T+j}")
            rg_l = annotate(rg, f"GT t={T+j}" if j < len(rgb_gt) else "no GT")
            frames.append(np.concatenate([rp_l, rg_l], axis=1))
        out_gif = args.out_dir / f"long_v4_{ix:02d}_{safe}_n{n_steps}.gif"
        imageio.mimsave(out_gif, frames, duration=0.18, loop=0)
        print(f"  -> {out_gif} ({H_total} frames pred, GT available for {len(rgb_gt)})")


if __name__ == "__main__":
    main()
