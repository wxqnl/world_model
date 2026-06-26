"""Decode the trained Hunyuan latent adapter from a wm3d checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset
from wm3d_v3.eval.make_demo_gif import window_config_from_cfg
from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.losses import _normalize_depth
from wm3d_v3.models.hunyuan_latent_adapter import HunyuanLatentAdapter, HunyuanLatentAdapterConfig
from wm3d_v3.training.train import encode_hunyuan_latents, load_hunyuan_vae, target_video_from_batch


@torch.no_grad()
def decode_hunyuan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    z = latents / float(vae.config.scaling_factor)
    out = vae.decode(z.to(dtype=vae.dtype), return_dict=False)[0]
    return out.div(2.0).add(0.5).clamp(0.0, 1.0).float()


def resize_frame(frame: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
    arr = frame.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    if arr.shape[:2] != hw:
        arr = np.array(Image.fromarray(arr).resize((hw[1], hw[0]), Image.BILINEAR))
    return arr


def save_demo(path: Path, target: torch.Tensor, pred: torch.Tensor, rough: torch.Tensor | None) -> None:
    """Save target | Hunyuan-adapter decode | rough RGB rows as a GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target[0]
    pred = pred[0]
    rough0 = rough[0] if rough is not None else None
    t = min(target.shape[1], pred.shape[1])
    hw = tuple(target.shape[-2:])
    frames = []
    for i in range(t):
        row = [
            resize_frame(target[:, i], hw),
            resize_frame(pred[:, i], hw),
        ]
        if rough0 is not None:
            row.append(resize_frame(rough0[:, i], hw))
        frames.append(np.concatenate(row, axis=1))
    imageio.mimsave(path, frames, duration=0.18)


def context_video_from_batch(context_rgb: torch.Tensor, frames: int) -> torch.Tensor:
    return context_rgb[:, :, None].expand(-1, -1, frames, -1, -1).contiguous()


def motion_hint_from_wm_out(wm_out: dict) -> torch.Tensor | None:
    point = wm_out.get("point")
    if point is not None:
        if point.ndim == 5 and point.shape[-1] == 3:
            point_ch = point.permute(0, 1, 4, 2, 3).contiguous().float()
        elif point.ndim == 5 and point.shape[2] == 3:
            point_ch = point.float()
        else:
            point_ch = None
        if point_ch is not None and point_ch.shape[1] > 0:
            diffs = point_ch.new_zeros(point_ch.shape[0], point_ch.shape[1], 1, point_ch.shape[-2], point_ch.shape[-1])
            if point_ch.shape[1] > 1:
                diffs[:, 1:] = (point_ch[:, 1:] - point_ch[:, :-1]).norm(dim=2, keepdim=True)
                diffs[:, :1] = diffs[:, 1:2]
            return _normalize_depth(diffs[:, :, 0]).unsqueeze(2)
    depth = wm_out.get("depth")
    if depth is None:
        return None
    depth = depth.float()
    diffs = depth.new_zeros(depth.shape)
    if depth.shape[1] > 1:
        diffs[:, 1:] = (depth[:, 1:] - depth[:, :-1]).abs()
        diffs[:, :1] = diffs[:, 1:2]
    return _normalize_depth(diffs).unsqueeze(2)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_clips", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    ds = OXEWindowDataset(records, window_config_from_cfg(cfg))
    g = torch.Generator().manual_seed(cfg["data"]["seed"])
    perm = torch.randperm(len(ds), generator=g).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))

    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "hunyuan_adapter" not in sd:
        raise RuntimeError(f"checkpoint has no hunyuan_adapter: {args.ckpt}")

    model = build_model(cfg).to(device).eval()
    model.load_state_dict(sd["model"])
    adapter_cfg = HunyuanLatentAdapterConfig(**sd["hunyuan_adapter_cfg"])
    adapter = HunyuanLatentAdapter(adapter_cfg).to(device).eval()
    load = adapter.load_state_dict(sd["hunyuan_adapter"], strict=False)
    print(f"loaded adapter missing={len(load.missing_keys)} unexpected={len(load.unexpected_keys)}")
    vae = load_hunyuan_vae(cfg["train"], device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    made = 0
    for vi in perm[:n_val]:
        smp = ds[vi]
        key = (smp["dataset"], smp["clip_id"])
        if key in seen:
            continue
        seen.add(key)

        s = smp["s_in"].unsqueeze(0).to(device)
        c = smp["c"].unsqueeze(0).to(device)
        action_tgt = smp["action_tgt"].unsqueeze(0).to(device)
        action_tgt_norm = smp["action_tgt_norm"].unsqueeze(0).to(device)
        context_rgb = smp["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
        rgb_tgt_p = smp["rgb_tgt"].permute(0, 3, 1, 2).unsqueeze(0).to(device)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)
        target_video = target_video_from_batch(context_rgb, rgb_tgt_p)
        target_latents = encode_hunyuan_latents(vae, target_video.float())
        context_video = context_video_from_batch(context_rgb, target_video.shape[2])
        base_latents = (
            encode_hunyuan_latents(vae, context_video.float())
            if getattr(adapter_cfg, "output_mode", "direct") in {"context_residual_mask", "context_residual_mask_velocity"}
            else None
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
            pred_latents = adapter(
                out["pred_tokens"],
                out["depth"],
                context_rgb=context_rgb,
                motion_hint=motion_hint_from_wm_out(out) if adapter_cfg.use_motion else None,
                rough_rgb=out.get("rgb"),
                action_cond=action_cond,
                task_emb=c,
                point=out.get("point"),
                pose=out.get("pose_geom"),
                target_latents=target_latents,
                base_latents=base_latents,
            )
        rough_future = out.get("rgb")
        if rough_future is None:
            rough_future = context_rgb[:, None].expand(-1, rgb_tgt_p.shape[1], -1, -1, -1)
        rough_video = torch.cat([context_rgb[:, None], rough_future.float()], dim=1).permute(0, 2, 1, 3, 4)
        if bool(cfg["train"].get("hunyuan_residual_from_rough", False)):
            rough_latents = encode_hunyuan_latents(vae, rough_video.float())
            pred_latents = rough_latents.to(dtype=pred_latents.dtype) + float(
                cfg["train"].get("hunyuan_residual_scale", 1.0)
            ) * pred_latents
        pred_video = decode_hunyuan_latents(vae, pred_latents.float())

        clip_id = smp["clip_id"].replace("/", "__")
        out_path = args.out_dir / f"{made:02d}_{smp['dataset']}_{clip_id}_hunyuan_latent.gif"
        save_demo(out_path, target_video.float(), pred_video, rough_video.float())
        print(f"wrote {out_path} pred={tuple(pred_video.shape)} target={tuple(target_video.shape)}")
        made += 1
        if made >= args.n_clips:
            break


if __name__ == "__main__":
    main()
