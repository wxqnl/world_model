"""Shared WM3D VGGT geometry cache validation and atomic writers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

GEOM_EXTRA_KEYS = ("world_points", "world_points_conf", "pose_enc", "depth_conf")


def _array_frame_count(arr: np.ndarray) -> int | None:
    if arr.ndim < 1:
        return None
    return int(arr.shape[0])


def _finite_array(arr: np.ndarray) -> bool:
    try:
        return bool(np.isfinite(arr).all())
    except TypeError:
        return False


def _spatial_hw(arr: np.ndarray) -> tuple[int, int] | None:
    if arr.ndim == 3:
        return int(arr.shape[1]), int(arr.shape[2])
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        return int(arr.shape[1]), int(arr.shape[2])
    return None


def _valid_depth_like(arr: np.ndarray, size: int | None = 224) -> bool:
    hw = _spatial_hw(arr)
    if hw is None or arr.shape[0] <= 0:
        return False
    if size is not None and hw != (int(size), int(size)):
        return False
    return _finite_array(arr)


def _valid_conf_like(arr: np.ndarray, size: int | None = 224) -> bool:
    hw = _spatial_hw(arr)
    if hw is None or arr.shape[0] <= 0:
        return False
    if size is not None and hw != (int(size), int(size)):
        return False
    return _finite_array(arr)


def _valid_world_points(arr: np.ndarray, size: int | None = 224) -> bool:
    if arr.ndim != 4 or arr.shape[0] <= 0 or arr.shape[-1] != 3:
        return False
    if size is not None and (arr.shape[1], arr.shape[2]) != (int(size), int(size)):
        return False
    return _finite_array(arr)


def _valid_pose_enc(arr: np.ndarray) -> bool:
    return arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[-1] > 0 and _finite_array(arr)


def validate_geom_npz(
    path: Path | str,
    expected_frames: int | None = None,
    require_geom_extra: bool = True,
    depth_size: int | None = 224,
) -> bool:
    """Return True only when a VGGT geom npz has usable depth and optional rich geometry.

    The check intentionally validates shapes and frame counts, not just key names,
    so partial/interrupted cache writes are not treated as complete.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            if "depth" not in d.files and "depth_map" not in d.files:
                return False
            depth_key = "depth" if "depth" in d.files else "depth_map"
            depth = np.asarray(d[depth_key])
            if not _valid_depth_like(depth, size=depth_size):
                return False
            frame_count = _array_frame_count(depth)
            if expected_frames is not None and frame_count != int(expected_frames):
                return False
            if not require_geom_extra:
                return True
            if not all(k in d.files for k in GEOM_EXTRA_KEYS):
                return False
            validators = {
                "world_points": lambda arr: _valid_world_points(arr, size=depth_size),
                "world_points_conf": lambda arr: _valid_conf_like(arr, size=depth_size),
                "pose_enc": _valid_pose_enc,
                "depth_conf": lambda arr: _valid_conf_like(arr, size=depth_size),
            }
            for key in GEOM_EXTRA_KEYS:
                arr = np.asarray(d[key])
                if not validators[key](arr):
                    return False
                if _array_frame_count(arr) != frame_count:
                    return False
            return True
    except Exception:
        return False


def frame_count_npy(path: Path | str) -> int | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim < 1 or arr.shape[0] <= 0:
            return None
        return int(arr.shape[0])
    except Exception:
        return None


def _load_npy_mmap(path: Path | str):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        return None


def validate_pooled_npy(
    path: Path | str,
    expected_frames: int | None = None,
    expected_tokens: int | None = 64,
    expected_dim: int | None = 2048,
) -> bool:
    arr = _load_npy_mmap(path)
    if arr is None or arr.ndim != 3:
        return False
    if arr.shape[0] <= 0:
        return False
    if expected_tokens is not None and int(arr.shape[1]) != int(expected_tokens):
        return False
    if expected_dim is not None and int(arr.shape[2]) != int(expected_dim):
        return False
    if expected_frames is not None and int(arr.shape[0]) != int(expected_frames):
        return False
    return True


def validate_actions_npy(path: Path | str, expected_frames: int | None = None, min_dim: int = 7) -> bool:
    arr = _load_npy_mmap(path)
    if arr is None or arr.ndim != 2:
        return False
    if arr.shape[0] <= 0 or arr.shape[1] < int(min_dim):
        return False
    if expected_frames is not None and int(arr.shape[0]) != int(expected_frames):
        return False
    return True


def validate_rgb_npy(path: Path | str, expected_frames: int | None = None, size: int = 256) -> bool:
    arr = _load_npy_mmap(path)
    if arr is None or arr.ndim != 4:
        return False
    if arr.shape[0] <= 0 or arr.shape[1] != int(size) or arr.shape[2] != int(size) or arr.shape[3] != 3:
        return False
    if expected_frames is not None and int(arr.shape[0]) != int(expected_frames):
        return False
    return True


def validate_qwen_npy(path: Path | str, dim: int = 2048) -> bool:
    arr = _load_npy_mmap(path)
    if arr is None or arr.ndim != 1 or arr.shape[0] != int(dim):
        return False
    try:
        return bool(np.isfinite(np.asarray(arr)).all())
    except Exception:
        return False


def atomic_savez_compressed(path: Path | str, **payload: np.ndarray) -> None:
    """Write a compressed npz by tmp file then os.replace in the same directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def existing_npz_payload(path: Path | str) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with np.load(path) as d:
            return {k: np.array(d[k]) for k in d.files}
    except Exception:
        return {}
