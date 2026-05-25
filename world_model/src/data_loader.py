"""Uniform data interface for Set A/B/C.

Manifest format (JSON list), each entry:
  {
    "clip_id": "droid_0001",
    "set": "A",
    "frames": ["/abs/path/frame_0000.jpg", ...],
    "fps": 30,
    "meta": {"dataset": "droid", "task": "pour_cup"}
  }

For ScanNet/7-Scenes entries may include "gt_depth": [...] parallel to frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@dataclass
class Clip:
    clip_id: str
    set_name: str
    frames: list[str]
    fps: float
    meta: dict
    gt_depth: list[str] | None = None


def load_manifest(path: str | Path) -> list[Clip]:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return [
        Clip(
            clip_id=e["clip_id"],
            set_name=e.get("set", "?"),
            frames=e["frames"],
            fps=float(e.get("fps", 30)),
            meta=e.get("meta", {}),
            gt_depth=e.get("gt_depth"),
        )
        for e in data
    ]


def load_frames(clip: Clip, indices: list[int] | None = None, target_size: int = 518) -> torch.Tensor:
    """Load frames as [N, 3, H, W] float tensor in [0, 1], resized to target_size square via pad-center.

    Uses vggt.utils.load_fn.load_and_preprocess_images when available so the exact
    VGGT preprocessing is applied. Falls back to a plain PIL path for unit testing.
    """
    paths = clip.frames if indices is None else [clip.frames[i] for i in indices]

    try:
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images(paths)
    except Exception:
        imgs = []
        for p in paths:
            im = Image.open(p).convert("RGB").resize((target_size, target_size))
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        return torch.stack(imgs, dim=0)


def sliding_windows(n_frames: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Return (start, end) inclusive-exclusive index pairs covering the clip."""
    out = []
    start = 0
    while start + window <= n_frames:
        out.append((start, start + window))
        start += stride
    if not out and n_frames > 0:
        out.append((0, min(n_frames, window)))
    return out
