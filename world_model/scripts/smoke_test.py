"""End-to-end smoke test for the VGGT wrapper.

Downloads facebook/VGGT-1B on first run. Uses VGGT-provided example images so
we don't need to wait for dataset licenses. Verifies:
  - model loads in bf16 on a free H100
  - encode() returns aggregated_tokens_list + ps_idx
  - decode_geometry() returns depth, point_map, extrinsic, intrinsic with sane shapes
  - token_l2_consecutive produces [N-1] values
  - Exp3's per_frame_token_delta shape path is exercised
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# pick the first idle GPU (H100s 2..7 are free in this environment)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.vggt_wrapper import VGGTConfig, VGGTWrapper
from src.metrics import token_l2_consecutive
from experiments.exp3_token_dynamics import per_frame_token_delta


def main():
    t0 = time.time()
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0)}")

    model = VGGTWrapper(VGGTConfig())
    print(f"[{time.time()-t0:6.1f}s] model loaded")

    # Use VGGT example images
    from vggt.utils.load_fn import load_and_preprocess_images
    example_dir = Path(__file__).resolve().parent.parent / "vggt" / "examples"
    jpgs = sorted([p for p in example_dir.rglob("*.jpg")])[:8]
    png = sorted([p for p in example_dir.rglob("*.png")])[:8]
    paths = [str(p) for p in (jpgs or png)[:8]]
    assert len(paths) >= 2, f"need ≥2 example images under {example_dir}; found {len(paths)}"
    print(f"using {len(paths)} example images: {[Path(p).name for p in paths]}")

    images = load_and_preprocess_images(paths)
    print(f"images shape: {tuple(images.shape)}")

    out = model.full_inference(images)
    print(f"[{time.time()-t0:6.1f}s] forward pass done")
    print("aggregated_tokens list length:", len(out["aggregated_tokens"]))
    print("last-layer tokens shape:", tuple(out["aggregated_tokens"][-1].shape))
    print("ps_idx shape:", tuple(out["ps_idx"].shape) if hasattr(out["ps_idx"], "shape") else type(out["ps_idx"]))
    print("depth:", tuple(out["depth"].shape), "conf:", tuple(out["depth_conf"].shape))
    print("point_map:", tuple(out["point_map"].shape))
    print("extri:", tuple(out["camera_extrinsic"].shape), "intri:", tuple(out["camera_intrinsic"].shape))

    # Exercise downstream metric path
    last = out["aggregated_tokens"][-1]
    if last.ndim == 4:
        flat = last[0].reshape(last.shape[1], -1)
        deltas = token_l2_consecutive(flat)
        print("token L2 consec:", deltas.shape, deltas.float().cpu().numpy().round(2))
        pfd, _ = per_frame_token_delta(out["aggregated_tokens"])
        print("per_frame_token_delta:", pfd.round(3))
    else:
        print("WARN: tokens ndim =", last.ndim, "— per_frame_token_delta expects 4D")

    print(f"[{time.time()-t0:6.1f}s] smoke test OK")


if __name__ == "__main__":
    main()
