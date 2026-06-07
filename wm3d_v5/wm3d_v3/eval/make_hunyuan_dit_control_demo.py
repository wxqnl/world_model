"""Run real Hunyuan DiT-control generation from frozen wm3d controls.

This script is not the latent adapter demo: it loads the HunyuanVideo sampler,
installs the DiT image-token control adapter hooks during sampling, and writes
real Hunyuan generated videos. Do not run it in lightweight tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset
from wm3d_v3.eval.make_demo_gif import window_config_from_cfg
from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.video_backends.base import VideoConditionBundle
from wm3d_v3.video_backends.hunyuan_dit_control_video import (
    HunyuanDiTControlVideoBackend,
    HunyuanDiTControlVideoBackendConfig,
)


def _sample_task_text(sample: dict[str, Any]) -> str:
    for key in ("task_text", "language_instruction", "instruction"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    dataset = str(sample.get("dataset", "robot"))
    clip_id = str(sample.get("clip_id", "clip"))
    return f"robot manipulation scene, {dataset} {clip_id}"


def _frame_to_uint8(frame: torch.Tensor) -> np.ndarray:
    arr = frame.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def save_video_pair(out_base: Path, rgb_btchw: torch.Tensor, fps: int = 6) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    video = rgb_btchw[0].float()
    frames = [_frame_to_uint8(video[i]) for i in range(video.shape[0])]
    imageio.mimsave(out_base.with_suffix(".gif"), frames, duration=1.0 / fps)
    imageio.mimsave(out_base.with_suffix(".mp4"), frames, fps=fps)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real Hunyuan DiT-control generation demo from wm3d controls; not a VAE latent adapter demo."
    )
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--wm_ckpt", type=Path, required=True)
    ap.add_argument("--control_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_clips", type=int, default=1)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control_scale", type=float, default=1.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    records = read_manifest(cfg["data"]["manifest"])
    ds = OXEWindowDataset(records, window_config_from_cfg(cfg))
    g = torch.Generator().manual_seed(int(cfg["data"].get("seed", args.seed)))
    perm = torch.randperm(len(ds), generator=g).tolist()
    n_val = max(1, int(len(ds) * float(cfg["data"].get("val_frac", 0.02))))

    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    backend = HunyuanDiTControlVideoBackend(
        HunyuanDiTControlVideoBackendConfig(
            control_ckpt=str(args.control_ckpt),
            control_scale=float(args.control_scale),
            infer_steps=int(args.steps),
        ),
        device=device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    made = 0
    for vi in perm[:n_val]:
        sample = ds[vi]
        key = (str(sample.get("dataset", "dataset")), str(sample.get("clip_id", vi)))
        if key in seen:
            continue
        seen.add(key)

        s = sample["s_in"].unsqueeze(0).to(device)
        c = sample["c"].unsqueeze(0).to(device)
        action_tgt = sample["action_tgt"].unsqueeze(0).to(device)
        action_tgt_norm = sample["action_tgt_norm"].unsqueeze(0).to(device)
        context_rgb = sample["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(s, c, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
        bundle = VideoConditionBundle(
            context_rgb=context_rgb.float(),
            action_cond=action_cond.float(),
            task_emb=c.float(),
            task_text=[_sample_task_text(sample)],
            pred_tokens=out["pred_tokens"].float(),
            depth=out["depth"].float(),
            motion_hint=out.get("motion_hint"),
            contact_hint=out.get("contact_hint"),
            rough_rgb=out.get("rgb"),
        )
        result = backend.generate(
            bundle,
            num_frames=int(args.frames),
            height=int(args.height),
            width=int(args.width),
            seed=int(args.seed) + made,
            infer_steps=int(args.steps),
        )
        dataset, clip_id = key
        safe_clip = clip_id.replace("/", "__")
        out_base = args.out_dir / f"{made:02d}_{dataset}_{safe_clip}_hunyuan_dit_control"
        save_video_pair(out_base, result.rgb)
        print(f"wrote {out_base.with_suffix('.mp4')} metadata={result.metadata}", flush=True)
        made += 1
        if made >= args.n_clips:
            break

    if made == 0:
        raise RuntimeError("No validation clips were available for Hunyuan DiT-control demo")


if __name__ == "__main__":
    main()
