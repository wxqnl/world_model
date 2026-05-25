"""Build a side-by-side compare gif:
columns = [v3 PixelDecoder (256x256), Cosmos zero-shot (cropped to k future), GT (256x256)]
rows    = the 8 predicted future frames.
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
from wm3d_v3.data.window_dataset import _safe
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


def annotate(img, text):
    from PIL import ImageDraw
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, len(text) * 7 + 6, 18], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.array(im)


def read_mp4(path: Path) -> np.ndarray:
    r = imageio.mimread(path, memtest=False)
    return np.stack(r)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--cosmos_inputs", type=Path, required=True)
    ap.add_argument("--cosmos_outputs", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--clip_ids", nargs="+", required=True)
    ap.add_argument("--cell", type=int, default=256)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    cache_root = Path(cfg["data"]["cache_root"])
    T = cfg["data"]["T"]; k = cfg["data"]["k"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ix, cid in enumerate(args.clip_ids):
        safe = _safe(cid)
        win_name = next(p.name for p in args.cosmos_inputs.iterdir()
                         if p.is_dir() and safe in p.name)
        in_dir = args.cosmos_inputs / win_name
        out_dir = args.cosmos_outputs / win_name
        cosmos_out_mp4 = out_dir / "output.mp4"
        if not cosmos_out_mp4.exists():
            print(f"SKIP {cid}: no Cosmos output at {cosmos_out_mp4}")
            continue

        pooled = np.array(np.load(cache_root / "vggt_pooled" / f"{safe}.npy"))
        rgb_full = np.array(np.load(cache_root / "rgb_256" / f"{safe}.npy"))
        qwen = np.load(cache_root / "qwen_taskemb" / f"{safe}.npy")
        s_in = torch.from_numpy(pooled[:T]).float().unsqueeze(0).to(device)
        c = torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float().unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(s_in, c, pixel=True, bridging=False)
        v3_rgb = out["rgb"][0].float().clamp(0, 1).cpu().numpy()  # [k,3,256,256]

        cosmos_vid = read_mp4(cosmos_out_mp4)                     # [24, 704, 960, 3]
        future_cosmos = cosmos_vid[-k:]                            # last k frames
        future_gt = rgb_full[T:T + k]                              # GT future

        cell = args.cell
        frames = []
        for j in range(k):
            v3_f = (v3_rgb[j].transpose(1, 2, 0) * 255).astype(np.uint8)
            v3_f = np.array(Image.fromarray(v3_f).resize((cell, cell)))
            co_f = np.array(Image.fromarray(future_cosmos[j]).resize((cell, cell)))
            gt_f = np.array(Image.fromarray(future_gt[j]).resize((cell, cell)))
            v3_a = annotate(v3_f, f"v3 PixelDec t={j}")
            co_a = annotate(co_f, f"Cosmos zero-shot t={j}")
            gt_a = annotate(gt_f, f"GT t={j}")
            frames.append(np.concatenate([v3_a, co_a, gt_a], axis=1))
        out_gif = args.out_dir / f"compare_{ix:02d}_{safe}.gif"
        imageio.mimsave(out_gif, frames, duration=0.3, loop=0)
        print(f"  -> {out_gif}")


if __name__ == "__main__":
    main()
