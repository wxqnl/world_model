"""DINOv2 feature caching — mirror of :mod:`src.phase1.cache` for the baseline.

Writes shards with the same schema (tokens, actions, states, frame_ids) into a
parallel directory so the training loop is indifferent to which backbone
produced the features. Depth / extrinsic / intrinsic are not produced (DINOv2
does not provide them); we set those arrays to empty placeholders so downstream
eval knows to skip the geometric metrics for the DINOv2 run.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
import yaml

from .cache import DroidActions, bf16_to_int16

log = logging.getLogger("phase1.cache_dinov2")


@torch.no_grad()
def _forward_features(model: torch.nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """Return per-image patch tokens [N, P, D] after mean-pooling to a fixed grid."""
    # timm DINOv2 returns CLS + patch tokens via forward_features.
    # Shape: [N, 1 + S*S, D]
    x = model.forward_features(imgs)
    if isinstance(x, dict):
        x = x["x"] if "x" in x else list(x.values())[0]
    patches = x[:, 1:]  # drop CLS
    return patches


def pool_patch_grid(patches: torch.Tensor, grid: int) -> torch.Tensor:
    """[N, P, D] -> [N, grid*grid, D] via 2-D adaptive pool where P = S*S."""
    N, P, D = patches.shape
    S = int(math.isqrt(P))
    if S * S != P:
        raise RuntimeError(f"expected square patch grid, got P={P}")
    x = patches.reshape(N, S, S, D).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x.float(), (grid, grid))
    return x.reshape(N, D, grid * grid).permute(0, 2, 1).to(patches.dtype).contiguous()


class DinoCachier:
    def __init__(self, cfg: dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        dcfg = cfg["dinov2"]
        self.grid = int(dcfg["token_grid"])
        self.image_size = int(dcfg["image_size"])
        log.info("loading DINOv2: %s", dcfg["name"])
        # timm model at the requested image size.
        self.model = timm.create_model(
            dcfg["name"], pretrained=True, num_classes=0, img_size=self.image_size
        ).to(device).eval()
        # bf16 for speed + parity with VGGT.
        self.model = self.model.to(torch.bfloat16)

    @torch.no_grad()
    def process_clip(
        self,
        clip: dict[str, Any],
        actions: DroidActions,
        out_dir: Path,
        max_frames: int,
    ) -> dict[str, Any]:
        from src.data_loader import Clip, load_frames

        frame_paths = clip["frames"][:max_frames]
        n = len(frame_paths)
        if n == 0:
            return {"skipped": True}
        clip_obj = Clip(
            clip_id=clip["clip_id"],
            set_name=clip.get("set", "A"),
            frames=frame_paths,
            fps=float(clip.get("fps", 15)),
            meta=clip.get("meta", {}),
        )

        imgs = load_frames(clip_obj)  # [N, 3, H, W]
        imgs = F.interpolate(
            imgs, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        imgs = imgs.to(self.device, dtype=torch.bfloat16)

        t0 = time.time()
        bs = 16
        feats = []
        for i in range(0, n, bs):
            part = imgs[i : i + bs]
            part_feats = _forward_features(self.model, part)
            part_feats = pool_patch_grid(part_feats, self.grid)  # [n, G*G, D]
            feats.append(part_feats)
        pooled = torch.cat(feats, dim=0)  # [N, G*G, D]

        episode_index = int(clip["meta"]["episode_index"])
        frame_ids = np.arange(n, dtype=np.int32)
        act, state = actions(episode_index, frame_ids)

        tokens_int16 = bf16_to_int16(pooled.contiguous())
        out_path = out_dir / f"{clip['clip_id']}.npz"
        meta = {
            "clip_id": clip["clip_id"],
            "episode_index": episode_index,
            "n_frames": n,
            "token_grid": self.grid,
            "token_dim": int(pooled.shape[-1]),
            "backbone": "dinov2_" + self.cfg["dinov2"]["name"],
        }
        empty_ext = np.zeros((0, 3, 4), dtype=np.float32)
        empty_int = np.zeros((0, 3, 3), dtype=np.float32)
        empty_d = np.zeros((0, 1, 1), dtype=np.float16)
        np.savez_compressed(
            out_path,
            tokens=tokens_int16,
            depth=empty_d,
            extrinsic=empty_ext,
            intrinsic=empty_int,
            conf=np.zeros(n, dtype=np.float32),
            actions=act,
            states=state,
            frame_ids=frame_ids,
        )
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        dt = time.time() - t0
        log.info("  cached %s: n=%d  elapsed=%.1fs", clip["clip_id"], n, dt)
        return {"skipped": False, "elapsed": dt, "n_frames": n}


def run(cfg_path: str, limit: int | None = None, force: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = yaml.safe_load(open(cfg_path))
    manifest = json.load(open(cfg["cache"]["manifest"]))

    out_dir = Path(cfg["cache"]["out_dir"]).parent / "cache_tokens_dinov2"
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = DroidActions(cfg["cache"]["droid_parquet"])
    cacher = DinoCachier(cfg)

    clips = manifest if limit is None else manifest[:limit]
    summary = {"n": len(clips), "backbone": cacher.cfg["dinov2"]["name"], "clips": []}
    for i, clip in enumerate(clips):
        out_path = out_dir / f"{clip['clip_id']}.npz"
        if out_path.exists() and not force:
            log.info("[%d/%d] %s already cached", i + 1, len(clips), clip["clip_id"])
            summary["clips"].append({"clip_id": clip["clip_id"], "cached": True, "skipped": True})
            continue
        log.info("[%d/%d] %s", i + 1, len(clips), clip["clip_id"])
        info = cacher.process_clip(clip, actions, out_dir, cfg["cache"]["max_frames_per_clip"])
        summary["clips"].append({"clip_id": clip["clip_id"], **info})
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/phase1/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.cfg, args.limit, args.force)
