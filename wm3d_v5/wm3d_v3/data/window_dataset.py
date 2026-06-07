"""Cache-backed window dataset for OXE training.

Each sample = (T_in input frames + k future frames) sliced from one episode.
Cache layout:
    cache/wm3d_v3/vggt_pooled/<safe_id>.npy        [n_frames, 64, 2048] fp16
    cache/wm3d_v3/vggt_geom/<safe_id>.npz          {"depth": [n,224,224] fp16,
                                                     optional extra VGGT geometry}
    cache/wm3d_v3/vggt_window_geom_p64/<safe_id>__start_<start>.npz
                                                     window-aligned VGGT pooled
                                                     tokens and native geometry
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
    tokens_subdir: str = "vggt_pooled"  # set to "vggt_p256" for 16x16 grid
    action_stats: Path | None = None  # set to action_stats.npz to enable normalization
    require_task_emb: bool = False  # fail fast instead of silently using zero Qwen embeddings
    load_task_text: bool = False  # emit raw task_text string in sample (Hunyuan text-cond generation)
    load_rgb: bool = True
    load_geom: bool = True
    load_state_tgt: bool = True
    load_geom_extra: bool = False
    require_geom_extra: bool = False
    window_geom_subdir: str = "vggt_window_geom_p64"
    use_window_tokens: bool = False
    max_windows_per_episode: int = 0
    trust_window_geom_cache: bool = False
    allow_pseudo_progress_targets: bool = False
    require_progress: bool = False
    load_policy_state: bool = False
    require_policy_state: bool = False
    policy_lowdim_dim: int = 0
    policy_object_state_dim: int = 0
    policy_plan_state_dim: int = 0
    policy_action_history_len: int = 0
    policy_action_history_dim: int = 7


def _safe(cid: str) -> str:
    return cid.replace("/", "__")


def _window_geom_path(cache_root: Path, subdir: str, cid: str, start: int) -> Path:
    return cache_root / subdir / f"{cid}__start_{int(start):06d}.npz"


def _npz_first(npz, names: tuple[str, ...]):
    if npz is None:
        return None
    for name in names:
        if name in npz.files:
            return npz[name]
    return None


def _slice_optional(arr, start: int, end: int):
    if arr is None:
        return None
    return np.array(arr[start:end])


def _future_slice_optional(arr, T: int, k: int):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.shape[0] >= T + k:
        return np.array(arr[T:T + k])
    if arr.shape[0] >= k:
        return np.array(arr[:k])
    return None


def _scalar_from_record(rec: OXEClipRecord, *names: str):
    for name in names:
        value = getattr(rec, name, None)
        if value is not None:
            return value
    return None


def _load_cache_array(cache_root: Path, subdir: str, cid: str):
    for suffix in (".npy", ".npz"):
        path = cache_root / subdir / f"{cid}{suffix}"
        if not path.exists():
            continue
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            try:
                key = subdir if subdir in arr.files else arr.files[0]
                return np.array(arr[key])
            finally:
                arr.close()
        return arr
    return None


def _select_frame_or_global(arr, frame_idx: int):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim >= 2 and arr.shape[0] > frame_idx:
        return arr[frame_idx]
    return arr


def _resize_vector(arr, dim: int):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if dim <= 0:
        return arr
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, arr.shape[0])
    out[:n] = arr[:n]
    return out


def _resize_action_history(arr, hist_len: int, action_dim: int):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, action_dim) if arr.size % max(1, action_dim) == 0 else arr.reshape(1, -1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    out = np.zeros((hist_len, action_dim), dtype=np.float32)
    if hist_len <= 0 or action_dim <= 0:
        return out
    arr = arr[-hist_len:, :action_dim]
    out[-arr.shape[0]:, :arr.shape[1]] = arr
    return out


def _action_history_from_actions(actions, start: int, T: int, hist_len: int, action_dim: int):
    end = start + T
    begin = max(0, end - hist_len)
    hist = np.asarray(actions[begin:end, :action_dim], dtype=np.float32)
    return _resize_action_history(hist, hist_len, action_dim)


def _cache_frame_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim < 1 or arr.shape[0] <= 0:
            return None
        return int(arr.shape[0])
    except Exception:
        return None


def _valid_len(arr, needed: int) -> bool:
    if arr is None:
        return False
    arr = np.asarray(arr)
    return arr.ndim >= 1 and int(arr.shape[0]) >= int(needed)


def _valid_world_points(arr, needed: int) -> bool:
    arr = np.asarray(arr) if arr is not None else None
    return arr is not None and arr.ndim == 4 and arr.shape[-1] == 3 and int(arr.shape[0]) >= int(needed)


def _valid_pose(arr, needed: int) -> bool:
    arr = np.asarray(arr) if arr is not None else None
    return arr is not None and arr.ndim >= 2 and int(arr.shape[0]) >= int(needed) and arr.shape[-1] > 0


def _valid_future_len(arr, T: int, k: int) -> bool:
    arr = np.asarray(arr) if arr is not None else None
    return arr is not None and arr.ndim >= 1 and int(arr.shape[0]) >= min(T + k, k)


def _valid_window_geom(path: Path, T: int, k: int, *, require_extra: bool, require_pooled: bool) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            if require_pooled:
                pooled = _npz_first(d, ("pooled", "vggt_pooled"))
                if not _valid_len(pooled, T + k):
                    return False
            if not require_extra:
                return True
            depth = _npz_first(d, ("depth", "depth_map"))
            point = _npz_first(d, ("point", "world_points", "points", "point_map"))
            point_conf = _npz_first(d, ("point_conf", "world_points_conf", "points_conf", "conf"))
            pose = _npz_first(d, ("pose", "pose_enc", "camera_pose"))
            depth_conf = _npz_first(d, ("depth_conf", "depth_confidence"))
            return (
                _valid_future_len(depth, T, k)
                and _valid_world_points(point, k)
                and _valid_future_len(point_conf, T, k)
                and _valid_pose(pose, k)
                and _valid_future_len(depth_conf, T, k)
            )
    except Exception:
        return False


def _progress_from_geom_or_record(geom, rec: OXEClipRecord):
    progress_arr = _npz_first(geom, ("progress", "progress_tgt"))
    if progress_arr is not None:
        return np.asarray(progress_arr, dtype=np.float32)
    rec_progress = getattr(rec, "progress", None)
    if rec_progress is not None:
        return np.asarray(rec_progress, dtype=np.float32)
    return None


def _valid_progress(arr, needed: int) -> bool:
    if arr is None:
        return False
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    return arr.shape[0] >= int(needed) and bool(np.isfinite(arr).all()) and float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0


def _has_policy_state(cache_root: Path, geom, cid: str, sample_key: str, geom_keys: tuple[str, ...], dim: int, needed: int) -> bool:
    if dim <= 0:
        return True
    arr = _npz_first(geom, geom_keys)
    if arr is None:
        arr = _load_cache_array(cache_root, sample_key, cid)
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return False
    if arr.ndim >= 2 and arr.shape[0] not in (1, dim) and arr.shape[0] < int(needed):
        return False
    return arr.size > 0


class OXEWindowDataset(Dataset):
    """One sample = T input frames + k target frames."""

    def __init__(self, records: list[OXEClipRecord], cfg: WindowConfig | None = None):
        self.cfg = cfg or WindowConfig()
        self.cfg.cache_root = Path(self.cfg.cache_root)
        if self.cfg.action_stats is not None:
            self.cfg.action_stats = Path(self.cfg.action_stats)
        self.act_mean: np.ndarray | None = None
        self.act_std: np.ndarray | None = None
        if self.cfg.action_stats is not None and Path(self.cfg.action_stats).exists():
            d = np.load(self.cfg.action_stats)
            self.act_mean = d["mean"][:6].astype(np.float32)
            self.act_std = d["std"][:6].astype(np.float32)
        self.records = []
        self._usable_frames: list[int] = []
        self._valid_starts: list[list[int] | None] = []
        missing_task_emb: list[str] = []
        init_errors: list[str] = []
        win = self.cfg.T + self.cfg.k
        for r in records:
            cid = _safe(r.clip_id)
            action_len = _cache_frame_count(self.cfg.cache_root / "actions" / f"{cid}.npy")
            if action_len is None:
                continue
            if self.cfg.use_window_tokens:
                token_len = int(getattr(r, "n_frames", action_len) or action_len)
            else:
                token_len = _cache_frame_count(self.cfg.cache_root / self.cfg.tokens_subdir / f"{cid}.npy")
                if token_len is None:
                    continue
            lengths = [token_len, action_len, int(getattr(r, "n_frames", token_len) or token_len)]
            if self.cfg.load_rgb:
                rgb_len = _cache_frame_count(self.cfg.cache_root / "rgb_256" / f"{cid}.npy")
                if rgb_len is None:
                    continue
                lengths.append(rgb_len)
            geom_path = self.cfg.cache_root / "vggt_geom" / f"{cid}.npz"
            geom = None
            try:
                skip_init_geom_decode = (
                    self.cfg.load_geom
                    and self.cfg.use_window_tokens
                    and self.cfg.trust_window_geom_cache
                    and not self.cfg.load_policy_state
                    and not self.cfg.require_progress
                )
                geom = None if skip_init_geom_decode else (
                    np.load(geom_path)
                    if (self.cfg.load_geom or self.cfg.load_policy_state or self.cfg.require_progress) and geom_path.exists()
                    else None
                )
                if self.cfg.load_geom:
                    if skip_init_geom_decode:
                        if not geom_path.exists():
                            continue
                    elif geom is None:
                        continue
                    else:
                        depth_arr = _npz_first(geom, ("depth", "depth_map"))
                        if not _valid_len(depth_arr, win):
                            if self.cfg.require_geom_extra:
                                init_errors.append(f"{r.clip_id}: depth missing/short in vggt_geom")
                            continue
                        lengths.append(int(np.asarray(depth_arr).shape[0]))
                        point_arr = _npz_first(geom, ("world_points", "point", "points", "point_map"))
                        point_conf_arr = _npz_first(geom, ("world_points_conf", "point_conf", "points_conf", "conf"))
                        pose_arr = _npz_first(geom, ("pose_enc", "pose", "camera_pose"))
                        depth_conf_arr = _npz_first(geom, ("depth_conf", "depth_confidence"))
                        if self.cfg.require_geom_extra and not (self.cfg.load_geom_extra or self.cfg.use_window_tokens):
                            missing = []
                            if not _valid_world_points(point_arr, win):
                                missing.append("world_points")
                            if not _valid_len(point_conf_arr, win):
                                missing.append("world_points_conf")
                            if not _valid_pose(pose_arr, win):
                                missing.append("pose_enc")
                            if not _valid_len(depth_conf_arr, win):
                                missing.append("depth_conf")
                            if missing:
                                init_errors.append("{}: require_geom_extra missing/short {}".format(r.clip_id, ", ".join(missing)))
                                continue
                        for extra_arr in (point_arr, point_conf_arr, pose_arr, depth_conf_arr):
                            if extra_arr is not None and np.asarray(extra_arr).ndim >= 1:
                                lengths.append(int(np.asarray(extra_arr).shape[0]))
                if self.cfg.load_policy_state and self.cfg.require_policy_state:
                    policy_specs = (
                        ("lowdim_state", ("lowdim_state", "lowdim", "robot_state"), self.cfg.policy_lowdim_dim),
                        ("object_state", ("object_state", "objects_state"), self.cfg.policy_object_state_dim),
                        ("plan_state", ("plan_state", "stage_state", "task_plan_state"), self.cfg.policy_plan_state_dim),
                    )
                    for sample_key, geom_keys, dim in policy_specs:
                        if not _has_policy_state(self.cfg.cache_root, geom, cid, sample_key, geom_keys, dim, win):
                            init_errors.append(f"{r.clip_id}: missing {sample_key}")
                if self.cfg.require_progress:
                    progress_arr = _progress_from_geom_or_record(geom, r)
                    if progress_arr is None and self.cfg.allow_pseudo_progress_targets:
                        pass
                    elif not _valid_progress(progress_arr, win):
                        init_errors.append(f"{r.clip_id}: require_progress missing, short, or outside [0, 1]")
            finally:
                if geom is not None:
                    geom.close()
            if self.cfg.require_task_emb and not (self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy").exists():
                missing_task_emb.append(r.clip_id)
                continue
            usable = min(lengths)
            if usable < win:
                continue
            valid_starts = None
            if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
                valid_starts = []
                starts = list(range(0, usable - win + 1, self.cfg.stride))
                if self.cfg.max_windows_per_episode and len(starts) > int(self.cfg.max_windows_per_episode):
                    starts = starts[: int(self.cfg.max_windows_per_episode)]
                for start in starts:
                    path = _window_geom_path(self.cfg.cache_root, self.cfg.window_geom_subdir, cid, start)
                    if self.cfg.trust_window_geom_cache:
                        if path.exists():
                            valid_starts.append(start)
                    elif _valid_window_geom(
                        path,
                        self.cfg.T,
                        self.cfg.k,
                        require_extra=self.cfg.require_geom_extra or self.cfg.load_geom_extra,
                        require_pooled=self.cfg.use_window_tokens,
                    ):
                        valid_starts.append(start)
                if not valid_starts:
                    if self.cfg.require_geom_extra or self.cfg.use_window_tokens:
                        init_errors.append(f"{r.clip_id}: no usable window geom cache in {self.cfg.window_geom_subdir}")
                    continue
            self.records.append(r)
            self._usable_frames.append(usable)
            self._valid_starts.append(valid_starts)
        if missing_task_emb:
            preview = ", ".join(missing_task_emb[:5])
            more = "" if len(missing_task_emb) <= 5 else f", ... +{len(missing_task_emb) - 5} more"
            raise RuntimeError(
                "require_task_emb=True but Qwen task embeddings are missing for "
                f"{len(missing_task_emb)} cached clips: {preview}{more}"
            )
        if init_errors:
            preview = "; ".join(init_errors[:5])
            more = "" if len(init_errors) <= 5 else f"; ... +{len(init_errors) - 5} more"
            raise RuntimeError(preview + more)
        self.index: list[tuple[int, int]] = []
        for i, usable in enumerate(self._usable_frames):
            starts = self._valid_starts[i]
            if starts is None:
                starts = list(range(0, usable - win + 1, self.cfg.stride))
            for start in starts:
                self.index.append((i, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ri, start = self.index[i]
        rec = self.records[ri]
        cid = _safe(rec.clip_id)
        T, k = self.cfg.T, self.cfg.k
        window_geom = None
        window_geom_path = _window_geom_path(self.cfg.cache_root, self.cfg.window_geom_subdir, cid, start)
        if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
            window_geom = np.load(window_geom_path) if window_geom_path.exists() else None
        pooled = None if self.cfg.use_window_tokens else np.load(self.cfg.cache_root / self.cfg.tokens_subdir / f"{cid}.npy", mmap_mode="r")
        need_geom_file = self.cfg.load_geom or self.cfg.load_policy_state or self.cfg.require_progress
        geom_path = self.cfg.cache_root / "vggt_geom" / f"{cid}.npz"
        geom = np.load(geom_path) if need_geom_file and geom_path.exists() else None
        rgb = np.load(self.cfg.cache_root / "rgb_256" / f"{cid}.npy", mmap_mode="r") if self.cfg.load_rgb else None
        actions = np.load(self.cfg.cache_root / "actions" / f"{cid}.npy", mmap_mode="r")
        qwen_p = self.cfg.cache_root / "qwen_taskemb" / f"{cid}.npy"
        if qwen_p.exists():
            qwen = np.load(qwen_p)
        else:
            if self.cfg.require_task_emb:
                raise FileNotFoundError(f"missing required Qwen task embedding: {qwen_p}")
            qwen = np.zeros(2048, dtype=np.float16)

        if self.cfg.load_geom:
            depth_arr = _npz_first(geom, ("depth", "depth_map"))
        else:
            depth_arr = None
        window_depth_arr = None
        if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
            window_depth_arr = _npz_first(window_geom, ("depth", "depth_map"))
            point_arr = _npz_first(window_geom, ("point", "world_points", "points", "point_map"))
            point_conf_arr = _npz_first(window_geom, ("point_conf", "world_points_conf", "points_conf", "conf"))
            pose_arr = _npz_first(window_geom, ("pose", "pose_enc", "camera_pose"))
            pose_conf_arr = _npz_first(window_geom, ("pose_conf", "pose_confidence"))
            depth_conf_arr = _npz_first(window_geom, ("depth_conf", "depth_confidence"))
        elif self.cfg.load_geom:
            point_arr = _npz_first(geom, ("world_points", "point", "points", "point_map"))
            point_conf_arr = _npz_first(geom, ("world_points_conf", "point_conf", "points_conf", "conf"))
            pose_arr = _npz_first(geom, ("pose_enc", "pose", "camera_pose"))
            pose_conf_arr = _npz_first(geom, ("pose_conf", "pose_confidence"))
            depth_conf_arr = _npz_first(geom, ("depth_conf", "depth_confidence"))
        else:
            point_arr = point_conf_arr = pose_arr = pose_conf_arr = depth_conf_arr = None
        if self.cfg.require_geom_extra and (point_arr is None or pose_arr is None):
            missing = []
            if point_arr is None:
                missing.append("world_points/point")
            if pose_arr is None:
                missing.append("pose_enc/pose")
            raise KeyError(f"require_geom_extra=True but vggt_geom missing {', '.join(missing)} for {rec.clip_id}")

        end = start + T + k
        required_lengths = [actions.shape[0]]
        if pooled is not None:
            required_lengths.append(pooled.shape[0])
        if rgb is not None:
            required_lengths.append(rgb.shape[0])
        if depth_arr is not None:
            required_lengths.append(depth_arr.shape[0])
        for extra_arr in (point_arr, point_conf_arr, pose_arr, pose_conf_arr, depth_conf_arr):
            if extra_arr is not None and np.asarray(extra_arr).shape[0] >= T + k:
                required_lengths.append(extra_arr.shape[0])
        if any(length < end for length in required_lengths):
            raise RuntimeError(f"cache window shorter than indexed range for {rec.clip_id}: end={end} lengths={required_lengths}")
        pooled_len = T + k if self.cfg.load_state_tgt else T
        if self.cfg.use_window_tokens:
            pooled_arr = _npz_first(window_geom, ("pooled", "vggt_pooled"))
            if pooled_arr is None or np.asarray(pooled_arr).shape[0] < pooled_len:
                raise KeyError(f"use_window_tokens=True but window geom missing pooled tokens for {rec.clip_id} start={start}")
            pooled_w = np.array(pooled_arr[:pooled_len])
        else:
            pooled_w = np.array(pooled[start : start + pooled_len])
        rgb_w = np.array(rgb[start : start + T + k]) if rgb is not None else None
        depth_w = _slice_optional(depth_arr, start, start + T + k)
        depth_tgt_w = depth_w[T:] if depth_w is not None else None
        if self.cfg.load_geom_extra or self.cfg.require_geom_extra or self.cfg.use_window_tokens:
            window_depth_tgt_w = _future_slice_optional(window_depth_arr, T, k)
            if window_depth_tgt_w is not None:
                depth_tgt_w = window_depth_tgt_w
            point_tgt_w = _future_slice_optional(point_arr, T, k)
            point_conf_tgt_w = _future_slice_optional(point_conf_arr, T, k)
            pose_tgt_w = _future_slice_optional(pose_arr, T, k)
            pose_conf_tgt_w = _future_slice_optional(pose_conf_arr, T, k)
            depth_conf_tgt_w = _future_slice_optional(depth_conf_arr, T, k)
        else:
            point_w = _slice_optional(point_arr, start, start + T + k)
            point_conf_w = _slice_optional(point_conf_arr, start, start + T + k)
            pose_w = _slice_optional(pose_arr, start, start + T + k)
            pose_conf_w = _slice_optional(pose_conf_arr, start, start + T + k)
            depth_conf_w = _slice_optional(depth_conf_arr, start, start + T + k)
            point_tgt_w = point_w[T:] if point_w is not None else None
            point_conf_tgt_w = point_conf_w[T:] if point_conf_w is not None else None
            pose_tgt_w = pose_w[T:] if pose_w is not None else None
            pose_conf_tgt_w = pose_conf_w[T:] if pose_conf_w is not None else None
            depth_conf_tgt_w = depth_conf_w[T:] if depth_conf_w is not None else None
        act_w = np.array(actions[start + T : start + T + k])
        frame_ids = np.arange(start + T, start + T + k, dtype=np.float32)
        denom = np.float32(max(1, rec.n_frames - 1))
        pseudo_progress_tgt = np.clip(frame_ids / denom, 0.0, 1.0).astype(np.float32)
        if self.act_mean is not None:
            action_tgt_norm = (act_w[:, :6] - self.act_mean) / self.act_std
        else:
            action_tgt_norm = act_w[:, :6].astype(np.float32)
        sample = {
            "s_in":            torch.from_numpy(pooled_w[:T]).float(),
            "action_tgt":      torch.from_numpy(act_w).float(),
            "action_tgt_norm": torch.from_numpy(action_tgt_norm).float(),
            "c":               torch.from_numpy(np.asarray(qwen, dtype=np.float16)).float(),
            "clip_id":         rec.clip_id,
            "start":           start,
            "dataset":         rec.dataset,
        }
        if self.cfg.load_state_tgt:
            sample["s_tgt"] = torch.from_numpy(pooled_w[T:]).float()
        if depth_w is not None:
            sample["depth_in"] = torch.from_numpy(depth_w[:T]).float()
        if depth_tgt_w is not None:
            sample["depth_tgt"] = torch.from_numpy(depth_tgt_w).float()
        if depth_conf_tgt_w is not None:
            sample["depth_conf_tgt"] = torch.from_numpy(depth_conf_tgt_w).float()
        if point_tgt_w is not None:
            sample["point_tgt"] = torch.from_numpy(point_tgt_w).float()
        if point_conf_tgt_w is not None:
            sample["point_conf_tgt"] = torch.from_numpy(point_conf_tgt_w).float()
        if pose_tgt_w is not None:
            sample["pose_geom_tgt"] = torch.from_numpy(pose_tgt_w).float()
        if pose_conf_tgt_w is not None:
            sample["pose_geom_conf_tgt"] = torch.from_numpy(pose_conf_tgt_w).float()
        if rgb_w is not None:
            sample["rgb_in"] = torch.from_numpy(rgb_w[:T]).float() / 255.0
            sample["rgb_tgt"] = torch.from_numpy(rgb_w[T:]).float() / 255.0

        if self.cfg.load_policy_state:
            frame_idx = start + T - 1
            policy_specs = (
                ("lowdim_state", ("lowdim_state", "lowdim", "robot_state"), self.cfg.policy_lowdim_dim),
                ("object_state", ("object_state", "objects_state"), self.cfg.policy_object_state_dim),
                ("plan_state", ("plan_state", "stage_state", "task_plan_state"), self.cfg.policy_plan_state_dim),
            )
            for sample_key, geom_keys, dim in policy_specs:
                arr = _npz_first(geom, geom_keys)
                if arr is None:
                    arr = _load_cache_array(self.cfg.cache_root, sample_key, cid)
                vec = _resize_vector(_select_frame_or_global(arr, frame_idx), dim)
                if vec is not None:
                    sample[sample_key] = torch.from_numpy(vec).float()
                elif self.cfg.require_policy_state and dim > 0:
                    raise KeyError(f"require_policy_state=True but missing {sample_key} for {rec.clip_id}")
            if self.cfg.policy_action_history_len > 0:
                hist_arr = _npz_first(geom, ("action_history", "action_hist"))
                if hist_arr is None:
                    hist_arr = _load_cache_array(self.cfg.cache_root, "action_history", cid)
                if hist_arr is not None:
                    hist_arr = _select_frame_or_global(hist_arr, frame_idx)
                    hist = _resize_action_history(
                        hist_arr,
                        self.cfg.policy_action_history_len,
                        self.cfg.policy_action_history_dim,
                    )
                else:
                    hist = _action_history_from_actions(
                        actions,
                        start,
                        T,
                        self.cfg.policy_action_history_len,
                        self.cfg.policy_action_history_dim,
                    )
                sample["action_history"] = torch.from_numpy(hist).float()

        progress_arr = _npz_first(geom, ("progress", "progress_tgt"))
        if progress_arr is not None:
            sample["progress_tgt"] = torch.from_numpy(np.array(progress_arr[start + T : start + T + k], dtype=np.float32)).float()
        elif getattr(rec, "progress", None) is not None:
            sample["progress_tgt"] = torch.tensor(rec.progress[start + T : start + T + k], dtype=torch.float32)
        elif self.cfg.allow_pseudo_progress_targets:
            sample["progress_tgt"] = torch.from_numpy(pseudo_progress_tgt).float()

        terminal_arr = _npz_first(geom, ("terminal_success", "terminal_success_tgt", "success"))
        terminal_value = None
        if terminal_arr is not None:
            terminal_value = terminal_arr[start + T + k - 1] if np.ndim(terminal_arr) > 0 else terminal_arr
        else:
            terminal_value = _scalar_from_record(rec, "terminal_success", "terminal_success_tgt", "success")
        if terminal_value is not None:
            sample["terminal_success_tgt"] = torch.tensor(float(terminal_value), dtype=torch.float32)
        elif self.cfg.allow_pseudo_progress_targets:
            sample["terminal_success_tgt"] = torch.tensor(float(pseudo_progress_tgt[-1]), dtype=torch.float32)

        plaus_arr = _npz_first(geom, ("plausibility", "plausibility_tgt"))
        plaus_value = None
        if plaus_arr is not None:
            plaus_value = plaus_arr[start + T + k - 1] if np.ndim(plaus_arr) > 0 else plaus_arr
        else:
            plaus_value = _scalar_from_record(rec, "plausibility", "plausibility_tgt")
        if plaus_value is not None:
            sample["plausibility_tgt"] = torch.tensor(float(plaus_value), dtype=torch.float32)
        elif self.cfg.allow_pseudo_progress_targets:
            sample["plausibility_tgt"] = torch.tensor(1.0, dtype=torch.float32)
        if self.cfg.load_task_text:
            sample["task_text"] = str(getattr(rec, "task_text", "") or "")
        return sample
