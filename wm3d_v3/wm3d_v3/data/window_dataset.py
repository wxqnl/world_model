"""Cache-backed window dataset for OXE training.

Each sample = (T_in input frames + k future frames) sliced from one episode.
Cache layout:
    cache/wm3d_v3/vggt_pooled/<safe_id>.npy        [n_frames, 64, 2048] fp16
    cache/wm3d_v3/vggt_geom/<safe_id>.npz          {"depth": [n,224,224] fp16}
    cache/wm3d_v3/rgb_256/<safe_id>.npy            [n_frames, 256, 256, 3] uint8
    cache/wm3d_v3/actions/<safe_id>.npy            [n_frames, 7] fp32
    cache/wm3d_v3/qwen_taskemb/<safe_id>.npy       [2048] fp16
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .manifest import OXEClipRecord


@dataclass
class WindowConfig:
    T: int = 16
    k: int = 8
    stride: int = 4
    cache_root: Path = Path("/home/user01/Minko/datasets/cache/wm3d_v3")


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


class OXEWindowDataset(Dataset):
    """One sample = T input frames + k target frames."""

    def __init__(self, records: list[OXEClipRecord], cfg: WindowConfig | None = None):
        self.cfg = cfg or WindowConfig()
        self.records = []
        for r in records:
            cid = _safe(r.clip_id)
            if not (self.cfg.cache_root / "vggt_pooled" / f"{cid}.npy").exists():
                continue
            if not (self.cfg.cache_root / "rgb_256" / f"{cid}.npy").exists():
                continue
            if not (self.cfg.cache_root / "actions" / f"{cid}.npy").exists():
                continue
            self.records.append(r)
        win = self.cfg.T + self.cfg.k
        self.index: list[tuple[int, int]] = []
        for i, r in enumerate(self.records):
            if r.n_frames < win:
                continue
            for start in range(0, r.n_frames - win + 1, self.cfg.stride):
                self.index.append((i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ri, start = self.index[i]
        rec = self.records[ri]
        cid = _safe(rec.clip_id)
        T, k = self.cfg.T, self.cfg.k
        pooled = np.load(self.cfg.cache_root / "vggt_pooled" / f"{cid}.npy", mmap_mode="r")
        geom = np.load(self.cfg.cache_root / "vggt_geom" / f"{cid}.npz")
        rgb = np.load(self.cfg.cache_root / "rgb_256" / f"{cid}.npy", mmap_mode="r")
        actions = np.load(self.cfg.cache_root / "actions" / f"{cid}.npy")
        qwen_p = self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy"
        if qwen_p.exists():
            qwen = np.load(qwen_p)
        else:
            qwen = np.zeros(2048, dtype=np.float16)
        depth = geom["depth"]
        end = min(start + T + k, pooled.shape[0], rgb.shape[0], actions.shape[0])
        if end - start < T + k:
            start = max(0, end - T - k)
        pooled_w = np.array(pooled[start : start + T + k])
        rgb_w = np.array(rgb[start : start + T + k])
        depth_w = np.array(depth[start : start + T + k])
        act_w = actions[start + T : start + T + k]
        return {
            "s_in":       torch.from_numpy(pooled_w[:T]).float(),
            "s_tgt":      torch.from_numpy(pooled_w[T:]).float(),
            "depth_in":   torch.from_numpy(depth_w[:T]).float(),
            "depth_tgt":  torch.from_numpy(depth_w[T:]).float(),
            "rgb_in":     torch.from_numpy(rgb_w[:T]).float() / 255.0,        # [T,H,W,3] in [0,1]
            "rgb_tgt":    torch.from_numpy(rgb_w[T:]).float() / 255.0,
            "action_tgt": torch.from_numpy(act_w).float(),
            "c":          torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float(),
            "clip_id":    rec.clip_id,
            "start":      start,
            "dataset":    rec.dataset,
        }
