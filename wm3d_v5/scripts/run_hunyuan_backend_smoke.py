"""Run a minimal HunyuanVideo backend smoke generation.

This verifies that the local HunyuanVideo repo, checkpoint, and wm3d_v3 backend
interface work together. It is text-to-video only until the trainable wm3d
control adapter is added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import torch

from wm3d_v3.video_backends import (
    HunyuanVideoBackend,
    HunyuanVideoBackendConfig,
    VideoConditionBundle,
)


def save_mp4(rgb: torch.Tensor, path: Path, fps: int) -> None:
    """Save `[B,T,3,H,W]` RGB in `[0,1]` to mp4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = rgb[0].detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
    frames = (frames * 255).round().astype("uint8")
    imageio.mimsave(path, list(frames), fps=fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="close-up robot arm picking up a small red block on a tabletop")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--model_base", type=str, default="/data/Minko/models/hunyuan_video")
    ap.add_argument("--external_repo", type=str, default="/data/Minko/external/HunyuanVideo")
    ap.add_argument(
        "--dit_weight",
        type=str,
        default="/data/Minko/models/hunyuan_video/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
    )
    ap.add_argument("--no_fp8", action="store_true")
    args = ap.parse_args()

    cfg = HunyuanVideoBackendConfig(
        external_repo=args.external_repo,
        model_base=args.model_base,
        dit_weight=args.dit_weight,
        use_fp8=not args.no_fp8,
        infer_steps=args.steps,
    )
    backend = HunyuanVideoBackend(cfg)
    bundle = VideoConditionBundle(
        context_rgb=torch.zeros(1, 3, args.height, args.width),
        task_text=[args.prompt],
    )
    out = backend.generate(
        bundle,
        num_frames=args.frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    save_mp4(out.rgb, args.out, fps=args.fps)
    print({"out": str(args.out), "shape": tuple(out.rgb.shape), "metadata": out.metadata})


if __name__ == "__main__":
    main()
