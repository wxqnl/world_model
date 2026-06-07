"""Validate window-aligned native3D cache coverage before v5 training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from scripts.cache_geom_utils import frame_count_npy


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def iter_window_starts(n_frames: int, T: int, k: int, stride: int, max_windows: int = 0) -> list[int]:
    win = int(T) + int(k)
    if n_frames < win:
        return []
    starts = list(range(0, n_frames - win + 1, int(stride)))
    if max_windows and len(starts) > int(max_windows):
        return starts[: int(max_windows)]
    return starts


def npz_has_valid_window(path: Path, *, T: int, k: int, P: int, D: int, hw: int) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        with np.load(path) as d:
            required = ("pooled", "depth", "depth_conf", "point", "point_conf", "pose", "pose_conf")
            missing = [key for key in required if key not in d.files]
            if missing:
                return False, "missing_keys:" + ",".join(missing)
            pooled = d["pooled"]
            depth = d["depth"]
            depth_conf = d["depth_conf"]
            point = d["point"]
            point_conf = d["point_conf"]
            pose = d["pose"]
            pose_conf = d["pose_conf"]
            checks = [
                (pooled.ndim == 3 and pooled.shape[0] >= T + k and pooled.shape[1] == P and pooled.shape[2] == D, f"bad_pooled:{pooled.shape}"),
                (depth.ndim == 3 and depth.shape[0] >= k and depth.shape[-2:] == (hw, hw), f"bad_depth:{depth.shape}"),
                (depth_conf.ndim >= 3 and depth_conf.shape[0] >= k and depth_conf.shape[-2:] == (hw, hw), f"bad_depth_conf:{depth_conf.shape}"),
                (point.ndim == 4 and point.shape[0] >= k and point.shape[1:3] == (hw, hw) and point.shape[-1] == 3, f"bad_point:{point.shape}"),
                (point_conf.ndim >= 3 and point_conf.shape[0] >= k and point_conf.shape[-2:] == (hw, hw), f"bad_point_conf:{point_conf.shape}"),
                (pose.ndim >= 2 and pose.shape[0] >= k and pose.shape[-1] >= 9, f"bad_pose:{pose.shape}"),
                (pose_conf.ndim >= 1 and pose_conf.shape[0] >= k, f"bad_pose_conf:{pose_conf.shape}"),
            ]
            for ok, msg in checks:
                if not ok:
                    return False, msg
            for key in required:
                if not np.isfinite(d[key]).all():
                    return False, f"nonfinite:{key}"
    except Exception as exc:
        return False, "exception:" + repr(exc)
    return True, "ok"


def base_cache_ready(cache_root: Path, cid: str, n: int, *, require_task_emb: bool) -> tuple[bool, str]:
    if frame_count_npy(cache_root / "vggt_pooled" / f"{cid}.npy") != n:
        return False, "base_pooled"
    if frame_count_npy(cache_root / "actions" / f"{cid}.npy") != n:
        return False, "base_actions"
    if frame_count_npy(cache_root / "rgb_256" / f"{cid}.npy") != n:
        return False, "base_rgb"
    if not (cache_root / "vggt_geom" / f"{cid}.npz").exists():
        return False, "base_geom"
    qwen_path = cache_root / "qwen_taskemb" / f"{cid}.npy"
    if require_task_emb and (not qwen_path.exists() or qwen_path.stat().st_size <= 0):
        return False, "base_qwen"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache_root", type=Path, required=True)
    ap.add_argument("--window_subdir", type=str, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max_windows_per_episode", type=int, default=0)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--D", type=int, default=2048)
    ap.add_argument("--hw", type=int, default=64)
    ap.add_argument("--require_task_emb", action="store_true")
    ap.add_argument("--sample_dataset", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    window_dir = args.cache_root / args.window_subdir
    summary = {
        "manifest": str(args.manifest),
        "cache_root": str(args.cache_root),
        "window_dir": str(window_dir),
        "records": len(records),
        "records_with_expected_windows": 0,
        "expected_windows": 0,
        "valid_windows": 0,
        "missing_windows": 0,
        "invalid_windows": 0,
        "base_missing_records": 0,
        "reasons": {},
        "sample_dataset_len": None,
        "sample_shapes": {},
    }

    for rec in records:
        cid = safe_id(str(rec.clip_id))
        n = frame_count_npy(args.cache_root / "rgb_256" / f"{cid}.npy")
        if n is None:
            n = int(getattr(rec, "n_frames", 0) or 0)
        base_ok, base_reason = base_cache_ready(args.cache_root, cid, n, require_task_emb=args.require_task_emb)
        if not base_ok:
            summary["base_missing_records"] += 1
            summary["reasons"][base_reason] = summary["reasons"].get(base_reason, 0) + 1
            continue
        starts = iter_window_starts(n, args.T, args.k, args.stride, args.max_windows_per_episode)
        if starts:
            summary["records_with_expected_windows"] += 1
        summary["expected_windows"] += len(starts)
        for start in starts:
            path = window_dir / f"{cid}__start_{start:06d}.npz"
            ok, reason = npz_has_valid_window(path, T=args.T, k=args.k, P=args.P, D=args.D, hw=args.hw)
            if ok:
                summary["valid_windows"] += 1
            else:
                if reason == "missing":
                    summary["missing_windows"] += 1
                else:
                    summary["invalid_windows"] += 1
                summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1

    if args.sample_dataset:
        ds = OXEWindowDataset(
            records,
            WindowConfig(
                T=args.T,
                k=args.k,
                stride=args.stride,
                cache_root=args.cache_root,
                action_stats=None,
                require_task_emb=args.require_task_emb,
                load_rgb=True,
                load_geom=True,
                load_state_tgt=True,
                load_geom_extra=True,
                require_geom_extra=True,
                window_geom_subdir=args.window_subdir,
                use_window_tokens=True,
                max_windows_per_episode=args.max_windows_per_episode,
            ),
        )
        summary["sample_dataset_len"] = len(ds)
        if len(ds):
            sample = ds[0]
            summary["sample_shapes"] = {key: list(value.shape) for key, value in sample.items() if hasattr(value, "shape")}

    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    if args.strict:
        if summary["expected_windows"] == 0:
            raise SystemExit("strict validation failed: no expected windows")
        if summary["valid_windows"] != summary["expected_windows"]:
            raise SystemExit("strict validation failed: invalid/missing windows")
        if args.sample_dataset and not summary["sample_dataset_len"]:
            raise SystemExit("strict validation failed: dataset empty")


if __name__ == "__main__":
    main()
