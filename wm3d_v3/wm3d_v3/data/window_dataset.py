"""Cache-backed window dataset for OXE training.

Each sample = (T_in input frames + k future frames) sliced from one episode.
Cache layout:
    cache/wm3d_v3/vggt_pooled/<safe_id>.npy        [n_frames, 64, 2048] fp16
    cache/wm3d_v3/vggt_geom/<safe_id>.npz          {"depth": [n,224,224] fp16}
    cache/wm3d_v3/vggt_window_geom_p64/
        <safe_id>__start_<start>.npz               window-aligned pooled + point/pose targets
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
    tokens_subdir: str = "vggt_pooled"  # set to "vggt_p256" for 16×16 grid
    action_stats: Path | None = None  # set to action_stats.npz to enable normalization
    require_task_emb: bool = False  # fail fast instead of silently using zero Qwen embeddings
    load_rgb: bool = True
    load_geom: bool = True
    load_state_tgt: bool = True
    load_geom_extra: bool = False
    require_geom_extra: bool = False
    window_geom_subdir: str = "vggt_window_geom_p64"
    use_window_tokens: bool = False


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


def _window_geom_name(cid: str, start: int) -> str:
    return f"{cid}__start_{int(start):08d}.npz"


class OXEWindowDataset(Dataset):
    """One sample = T input frames + k target frames."""

    def __init__(self, records: list[OXEClipRecord], cfg: WindowConfig | None = None):
        self.cfg = cfg or WindowConfig()
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None
        if self.cfg.action_stats is not None and Path(self.cfg.action_stats).exists():
            d = np.load(self.cfg.action_stats)
            self.act_mean = d["mean"][:6].astype(np.float32)
            self.act_std = d["std"][:6].astype(np.float32)
        self.records = []
        missing_task_emb: list[str] = []
        for r in records:
            cid = _safe(r.clip_id)
            if not self.cfg.use_window_tokens and not (self.cfg.cache_root / self.cfg.tokens_subdir / f"{cid}.npy").exists():
                continue
            if self.cfg.load_rgb and not (self.cfg.cache_root / "rgb_256" / f"{cid}.npy").exists():
                continue
            if self.cfg.load_geom and not (self.cfg.cache_root / "vggt_geom" / f"{cid}.npz").exists():
                continue
            if not (self.cfg.cache_root / "actions" / f"{cid}.npy").exists():
                continue
            if self.cfg.require_task_emb and not (self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy").exists():
                missing_task_emb.append(r.clip_id)
                continue
            self.records.append(r)
        if missing_task_emb:
            preview = ", ".join(missing_task_emb[:5])
            more = "" if len(missing_task_emb) <= 5 else f", ... +{len(missing_task_emb) - 5} more"
            raise RuntimeError(
                "require_task_emb=True but Qwen task embeddings are missing for "
                f"{len(missing_task_emb)} cached clips: {preview}{more}"
            )
        win = self.cfg.T + self.cfg.k
        self.index: list[tuple[int, int]] = []
        for i, r in enumerate(self.records):
            if r.n_frames < win:
                continue
            cid = _safe(r.clip_id)
            for start in range(0, r.n_frames - win + 1, self.cfg.stride):
                if self.cfg.require_geom_extra or self.cfg.use_window_tokens:
                    geom_extra_path = (
                        self.cfg.cache_root
                        / self.cfg.window_geom_subdir
                        / _window_geom_name(cid, start)
                    )
                    if not geom_extra_path.exists():
                        continue
                self.index.append((i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ri, start = self.index[i]
        rec = self.records[ri]
        cid = _safe(rec.clip_id)
        T, k = self.cfg.T, self.cfg.k
        pooled = None
        if not self.cfg.use_window_tokens:
            pooled = np.load(self.cfg.cache_root / self.cfg.tokens_subdir / f"{cid}.npy", mmap_mode="r")
        geom = np.load(self.cfg.cache_root / "vggt_geom" / f"{cid}.npz") if self.cfg.load_geom else None
        rgb = np.load(self.cfg.cache_root / "rgb_256" / f"{cid}.npy", mmap_mode="r") if self.cfg.load_rgb else None
        actions = np.load(self.cfg.cache_root / "actions" / f"{cid}.npy", mmap_mode="r")
        qwen_p = self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy"
        if qwen_p.exists():
            qwen = np.load(qwen_p)
        else:
            if self.cfg.require_task_emb:
                raise FileNotFoundError(f"missing required Qwen task embedding: {qwen_p}")
            qwen = np.zeros(2048, dtype=np.float16)
        shapes = [actions.shape[0]]
        if pooled is not None:
            shapes.append(pooled.shape[0])
        if rgb is not None:
            shapes.append(rgb.shape[0])
        if geom is not None:
            shapes.append(geom["depth"].shape[0])
        end = min(start + T + k, *shapes)
        if end - start < T + k:
            start = max(0, end - T - k)
        geom_extra_path = self.cfg.cache_root / self.cfg.window_geom_subdir / _window_geom_name(cid, start)
        need_window_geom = self.cfg.load_geom_extra or self.cfg.use_window_tokens
        geom_extra = np.load(geom_extra_path) if need_window_geom and geom_extra_path.exists() else None
        if self.cfg.require_geom_extra and geom_extra is None:
            raise FileNotFoundError(f"missing required window geometry target: {geom_extra_path}")
        pooled_len = T + k if self.cfg.load_state_tgt else T
        if self.cfg.use_window_tokens:
            if geom_extra is None:
                raise FileNotFoundError(f"missing required window token cache: {geom_extra_path}")
            if "pooled" not in geom_extra:
                raise KeyError(f"missing required native3D keys in {geom_extra_path}: pooled")
            pooled_w = np.array(geom_extra["pooled"][:pooled_len])
        else:
            assert pooled is not None
            pooled_w = np.array(pooled[start : start + pooled_len])
        rgb_w = np.array(rgb[start : start + T + k]) if rgb is not None else None
        depth_w = np.array(geom["depth"][start : start + T + k]) if geom is not None else None
        act_w = actions[start + T : start + T + k]
        frame_ids = np.arange(start + T, start + T + k, dtype=np.float32)
        denom = np.float32(max(1, rec.n_frames - 1))
        progress_tgt = np.clip(frame_ids / denom, 0.0, 1.0).astype(np.float32)
        if self.act_mean is not None:
            action_tgt_norm = (act_w[:, :6] - self.act_mean) / self.act_std
        else:
            action_tgt_norm = act_w[:, :6].astype(np.float32)
        sample = {
            "s_in":            torch.from_numpy(pooled_w[:T]).float(),
            "action_tgt":      torch.from_numpy(act_w).float(),
            "action_tgt_norm": torch.from_numpy(action_tgt_norm).float(),
            "progress_tgt":    torch.from_numpy(progress_tgt).float(),
            "terminal_success_tgt": torch.tensor(float(progress_tgt[-1]), dtype=torch.float32),
            "plausibility_tgt": torch.tensor(1.0, dtype=torch.float32),
            "c":               torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float(),
            "clip_id":         rec.clip_id,
            "start":           start,
            "dataset":         rec.dataset,
        }
        if self.cfg.load_state_tgt:
            sample["s_tgt"] = torch.from_numpy(pooled_w[T:]).float()
        if depth_w is not None:
            sample["depth_in"] = torch.from_numpy(depth_w[:T]).float()
            sample["depth_tgt"] = torch.from_numpy(depth_w[T:]).float()
        if rgb_w is not None:
            sample["rgb_in"] = torch.from_numpy(rgb_w[:T]).float() / 255.0
            sample["rgb_tgt"] = torch.from_numpy(rgb_w[T:]).float() / 255.0
        if geom_extra is not None:
            if self.cfg.require_geom_extra:
                required = ("point", "point_conf", "pose", "pose_conf", "depth_conf")
                missing = [key for key in required if key not in geom_extra]
                if missing:
                    raise KeyError(f"missing required native3D keys in {geom_extra_path}: {', '.join(missing)}")
            if "point" in geom_extra:
                sample["point_tgt"] = torch.from_numpy(np.array(geom_extra["point"])).float()
            if "point_conf" in geom_extra:
                sample["point_conf_tgt"] = torch.from_numpy(np.array(geom_extra["point_conf"])).float()
            if "pose" in geom_extra:
                sample["pose_geom_tgt"] = torch.from_numpy(np.array(geom_extra["pose"])).float()
            if "pose_conf" in geom_extra:
                sample["pose_geom_conf_tgt"] = torch.from_numpy(np.array(geom_extra["pose_conf"])).float()
            if "depth_conf" in geom_extra:
                sample["depth_conf_tgt"] = torch.from_numpy(np.array(geom_extra["depth_conf"])).float()
        return sample
