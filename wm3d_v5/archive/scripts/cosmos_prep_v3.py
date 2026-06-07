"""v3 → Cosmos zero-shot prep:
- Pick OXE clips, run v3 to get predicted depth.
- Save a 24-frame control_depth.mp4 (16 GT past + 8 v3-predicted future) at Cosmos input size.
- Save a 24-frame input_video.mp4 (16 GT past + 8 GT future RGB) as seed.
- Write controlnet_specs.json with depth control + task_text as prompt.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig, _safe
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


def _depth_to_uint8_rgb(d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    n = np.clip((d.astype(np.float32) - vmin) / max(1e-6, vmax - vmin), 0, 1)
    g = (n * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _resize_uint8(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    t = F.interpolate(t.unsqueeze(0), size=(h, w), mode="bilinear",
                       align_corners=False, antialias=True)
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_ids", nargs="+", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--control_weight", type=float, default=1.5)
    ap.add_argument("--cosmos_hw", default="480,704",
                    help="Cosmos input resolution H,W")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    H, W = [int(x) for x in args.cosmos_hw.split(",")]
    T = cfg["data"]["T"]
    k = cfg["data"]["k"]
    cache_root = Path(cfg["data"]["cache_root"])

    records = {r.clip_id: r for r in read_manifest(cfg["data"]["manifest"])}
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={sd.get('val_total'):.4f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for ix, cid in enumerate(args.clip_ids):
        if cid not in records:
            print(f"SKIP {cid}: not in manifest")
            continue
        rec = records[cid]
        safe = _safe(cid)
        pooled = np.array(np.load(cache_root / "vggt_pooled" / f"{safe}.npy"))
        rgb_full = np.array(np.load(cache_root / "rgb_256" / f"{safe}.npy"))
        depth_full = np.array(np.load(cache_root / "vggt_geom" / f"{safe}.npz")["depth"])
        qwen_p = cache_root / "qwen_taskemb" / f"{safe}.npy"
        qwen = np.load(qwen_p) if qwen_p.exists() else np.zeros(2048, dtype=np.float16)
        n_frames = pooled.shape[0]
        s0 = args.start
        if s0 + T + k > n_frames:
            s0 = max(0, n_frames - T - k)

        s = torch.from_numpy(pooled[s0:s0 + T]).float().unsqueeze(0).to(device)
        c = torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float().unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(s, c, pixel=False, bridging=False)
        pred_depth = out["depth"][0].float().cpu().numpy()    # [k, 224, 224]

        # Build a unified depth range from past + predicted future for stable normalization
        past_depth = depth_full[s0:s0 + T]                    # [T, 224, 224]
        future_gt_depth = depth_full[s0 + T:s0 + T + k]       # [k, 224, 224]
        all_for_range = np.concatenate([past_depth, pred_depth], axis=0)
        vmin, vmax = float(np.percentile(all_for_range, 5)), float(np.percentile(all_for_range, 95))

        depth_seq, rgb_seq = [], []
        for i in range(T):
            d_u8 = _depth_to_uint8_rgb(past_depth[i], vmin, vmax)
            depth_seq.append(_resize_uint8(d_u8, H, W))
            rgb_seq.append(_resize_uint8(rgb_full[s0 + i], H, W))
        for j in range(k):
            d_u8 = _depth_to_uint8_rgb(pred_depth[j], vmin, vmax)
            depth_seq.append(_resize_uint8(d_u8, H, W))
            rgb_seq.append(_resize_uint8(rgb_full[s0 + T + j], H, W))

        win_dir = args.out_dir / f"win_{ix:02d}_{safe}_start{s0}"
        win_dir.mkdir(parents=True, exist_ok=True)
        depth_path = win_dir / "control_depth.mp4"
        rgb_path = win_dir / "input_video.mp4"
        imageio.mimsave(depth_path, depth_seq, fps=20, codec="libx264")
        imageio.mimsave(rgb_path, rgb_seq, fps=20, codec="libx264")
        prompt = (f"A {rec.robot} robot arm performing the task: {rec.task_text.rstrip('.')}. "
                  "Realistic indoor scene, the robot arm and gripper are clearly visible "
                  "and detailed. Natural lighting, photorealistic, high quality.")
        spec = {
            "prompt": prompt,
            "input_video_path": str(rgb_path),
            "depth": {"input_control": str(depth_path), "control_weight": args.control_weight},
        }
        (win_dir / "controlnet_specs.json").write_text(json.dumps(spec, indent=2))
        # also save the raw arrays so a comparison gif can be built later
        np.savez(win_dir / "raw.npz",
                 pred_depth=pred_depth, past_depth=past_depth,
                 future_gt_depth=future_gt_depth,
                 past_rgb=rgb_full[s0:s0 + T],
                 future_gt_rgb=rgb_full[s0 + T:s0 + T + k],
                 vmin=vmin, vmax=vmax)
        summary.append({"clip_id": cid, "dir": str(win_dir),
                         "task_text": rec.task_text, "start": s0,
                         "n_frames_total": n_frames})
        print(f"[{ix}] {cid}: depth+rgb mp4 + spec ready at {win_dir.name}")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary at {args.out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
