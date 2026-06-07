"""Cache OXE episodes: VGGT pooled tokens + optional VGGT geometry/RGB + actions + Qwen task emb.

One tar at a time (sharded across GPUs by tar+dataset hash). Each episode produces
files under cache/wm3d_v3/<kind>/<clip_id_safe>.{npy,npz}. vggt_geom remains
backward compatible with old depth-only caches and can optionally include VGGT
world_points, pose_enc, and confidence maps.
"""
from __future__ import annotations
import argparse
import pickle
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed

from wm3d_v3.data.manifest import read_manifest, OXEClipRecord
from wm3d_v3.data.oxe_loader import decode_episode

from scripts.cache_geom_utils import (
    GEOM_EXTRA_KEYS,
    atomic_savez_compressed,
    existing_npz_payload,
    frame_count_npy,
    validate_actions_npy,
    validate_geom_npz,
    validate_pooled_npy,
    validate_qwen_npy,
    validate_rgb_npy,
)


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def resize_image_batch(imgs: np.ndarray, size: int) -> torch.Tensor:
    """[T, H, W, 3] uint8 -> [T, 3, size, size] float [0,1]"""
    imgs = np.array(imgs, copy=True)
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                           align_corners=False, antialias=True)
    return t


def geom_extra_complete(path: Path, expected_frames: int | None = None) -> bool:
    return validate_geom_npz(path, expected_frames=expected_frames, require_geom_extra=True)


def cache_complete(
    cache_root: Path,
    cid: str,
    need_qwen: bool,
    need_rgb: bool = True,
    need_geom: bool = True,
    need_geom_extra: bool = False,
) -> bool:
    pool_path = cache_root / "vggt_pooled" / f"{cid}.npy"
    act_path = cache_root / "actions" / f"{cid}.npy"
    token_frames = frame_count_npy(pool_path)
    if token_frames is None or not validate_pooled_npy(pool_path, expected_frames=token_frames):
        return False
    expected_frames = token_frames
    if not validate_actions_npy(act_path, expected_frames=expected_frames):
        return False
    if need_rgb:
        rgb_path = cache_root / "rgb_256" / f"{cid}.npy"
        if not validate_rgb_npy(rgb_path, expected_frames=expected_frames):
            return False
    if need_geom:
        geom_path = cache_root / "vggt_geom" / f"{cid}.npz"
        if not validate_geom_npz(geom_path, expected_frames=expected_frames, require_geom_extra=need_geom_extra):
            return False
    if need_qwen:
        qwen_path = cache_root / "qwen_taskemb" / f"{cid}.npy"
        if not validate_qwen_npy(qwen_path):
            return False
    return True


def _existing_geom_payload(path: Path) -> dict[str, np.ndarray]:
    return existing_npz_payload(path)


def _append_chunk(chunks: dict[str, list[np.ndarray]], out: dict, key: str) -> None:
    if key not in out:
        return
    chunks[key].append(out[key][0].detach().cpu().numpy().astype(np.float16))


def window_geom_id(cid: str, start: int) -> str:
    return f"{cid}__start_{int(start):06d}"


def iter_window_starts(n_frames: int, T: int, k: int, stride: int, max_windows: int = 0) -> list[int]:
    win = int(T) + int(k)
    if n_frames < win:
        return []
    starts = list(range(0, n_frames - win + 1, int(stride)))
    if max_windows and len(starts) > int(max_windows):
        return starts[: int(max_windows)]
    return starts


def resize_map_batch(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize [T,H,W], [T,H,W,1], or [T,H,W,3] maps to a smaller native3D target grid."""
    arr = np.asarray(arr)
    squeeze = False
    if arr.ndim == 3:
        arr = arr[..., None]
        squeeze = True
    if arr.ndim != 4:
        return arr.astype(np.float16)
    if arr.shape[1] == int(size) and arr.shape[2] == int(size):
        out = arr
    else:
        t = torch.from_numpy(arr.astype(np.float32)).permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(int(size), int(size)), mode="bilinear", align_corners=False)
        out = t.permute(0, 2, 3, 1).cpu().numpy()
    if squeeze and out.shape[-1] == 1:
        out = out[..., 0]
    return out.astype(np.float16)


def window_geom_complete(path: Path, T: int, k: int, *, include_pooled: bool) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            if include_pooled:
                if "pooled" not in d.files or d["pooled"].ndim != 3 or d["pooled"].shape[0] < T + k:
                    return False
            required = ("depth", "point", "point_conf", "pose", "pose_conf", "depth_conf")
            if not all(key in d.files for key in required):
                return False
            depth = d["depth"]
            point = d["point"]
            pose = d["pose"]
            if depth.ndim != 3 or depth.shape[0] < k:
                return False
            if point.ndim != 4 or point.shape[0] < k or point.shape[-1] != 3:
                return False
            if pose.ndim < 2 or pose.shape[0] < k or pose.shape[-1] <= 0:
                return False
            for key in required:
                if not np.isfinite(d[key]).all():
                    return False
            return True
    except Exception:
        return False


def window_geom_cache_complete(
    out_dir: Path,
    cid: str,
    n_frames: int,
    T: int,
    k: int,
    stride: int,
    max_windows: int,
    *,
    include_pooled: bool,
) -> bool:
    starts = iter_window_starts(n_frames, T, k, stride, max_windows)
    if not starts:
        return False
    return all(
        window_geom_complete(out_dir / f"{window_geom_id(cid, s)}.npz", T, k, include_pooled=include_pooled)
        for s in starts
    )


def _tensor_out(out: dict, key: str) -> np.ndarray | None:
    if key not in out:
        return None
    return out[key][0].detach().cpu().numpy()


def cache_window_geom_extra(
    enc: VGGTEncoder,
    imgs_uint8: np.ndarray,
    out_dir: Path,
    cid: str,
    *,
    start: int,
    T: int,
    k: int,
    hw: int,
    include_pooled: bool,
    force: bool = False,
) -> bool:
    path = out_dir / f"{window_geom_id(cid, start)}.npz"
    if not force and window_geom_complete(path, T, k, include_pooled=include_pooled):
        return True
    window = np.asarray(imgs_uint8[start:start + T + k])
    if window.shape[0] < T + k:
        return False
    frames_224 = resize_image_batch(window, 224).unsqueeze(0).to("cuda")
    out = enc(frames_224)
    payload: dict[str, np.ndarray] = {
        "start": np.array(start, dtype=np.int64),
        "T": np.array(T, dtype=np.int64),
        "k": np.array(k, dtype=np.int64),
    }
    if include_pooled:
        payload["pooled"] = _tensor_out(out, "pooled").astype(np.float16)
    if "depth" in out:
        payload["depth"] = resize_map_batch(_tensor_out(out, "depth")[T:T + k], hw)
    if "depth_conf" in out:
        payload["depth_conf"] = resize_map_batch(_tensor_out(out, "depth_conf")[T:T + k], hw)
    if "world_points" in out:
        payload["point"] = resize_map_batch(_tensor_out(out, "world_points")[T:T + k], hw)
    if "world_points_conf" in out:
        payload["point_conf"] = resize_map_batch(_tensor_out(out, "world_points_conf")[T:T + k], hw)
    if "pose_enc" in out:
        pose = _tensor_out(out, "pose_enc")[T:T + k].astype(np.float16)
        payload["pose"] = pose
        payload["pose_conf"] = np.ones((pose.shape[0],), dtype=np.float16)
    if not all(key in payload for key in ("depth", "point", "point_conf", "pose", "pose_conf", "depth_conf")):
        missing = sorted({"depth", "point", "point_conf", "pose", "pose_conf", "depth_conf"} - set(payload))
        raise RuntimeError(f"VGGT window geom missing required keys for {cid} start={start}: {missing}")
    atomic_savez_compressed(path, **payload)
    return True


def _tensor_out_at(out: dict, key: str, batch_idx: int) -> np.ndarray | None:
    if key not in out:
        return None
    return out[key][batch_idx].detach().cpu().numpy()


def _save_window_geom_from_batch(
    out: dict,
    path: Path,
    *,
    batch_idx: int,
    start: int,
    T: int,
    k: int,
    hw: int,
    include_pooled: bool,
) -> None:
    payload: dict[str, np.ndarray] = {
        "start": np.array(start, dtype=np.int64),
        "T": np.array(T, dtype=np.int64),
        "k": np.array(k, dtype=np.int64),
    }
    if include_pooled:
        pooled = _tensor_out_at(out, "pooled", batch_idx)
        if pooled is not None:
            payload["pooled"] = pooled.astype(np.float16)
    depth = _tensor_out_at(out, "depth", batch_idx)
    if depth is not None:
        payload["depth"] = resize_map_batch(depth[T:T + k], hw)
    depth_conf = _tensor_out_at(out, "depth_conf", batch_idx)
    if depth_conf is not None:
        payload["depth_conf"] = resize_map_batch(depth_conf[T:T + k], hw)
    point = _tensor_out_at(out, "world_points", batch_idx)
    if point is not None:
        payload["point"] = resize_map_batch(point[T:T + k], hw)
    point_conf = _tensor_out_at(out, "world_points_conf", batch_idx)
    if point_conf is not None:
        payload["point_conf"] = resize_map_batch(point_conf[T:T + k], hw)
    pose = _tensor_out_at(out, "pose_enc", batch_idx)
    if pose is not None:
        pose = pose[T:T + k].astype(np.float16)
        payload["pose"] = pose
        payload["pose_conf"] = np.ones((pose.shape[0],), dtype=np.float16)
    if not all(key in payload for key in ("depth", "point", "point_conf", "pose", "pose_conf", "depth_conf")):
        missing = sorted({"depth", "point", "point_conf", "pose", "pose_conf", "depth_conf"} - set(payload))
        raise RuntimeError(f"VGGT window geom missing required keys for {path.name}: {missing}")
    atomic_savez_compressed(path, **payload)


def cache_window_geom_extra_many(
    enc: VGGTEncoder,
    imgs_uint8: np.ndarray,
    out_dir: Path,
    cid: str,
    *,
    starts: list[int],
    T: int,
    k: int,
    hw: int,
    include_pooled: bool,
    force: bool = False,
    batch_windows: int = 1,
) -> int:
    done = 0
    pending: list[int] = []
    batch_windows = max(1, int(batch_windows))

    def flush(batch_starts: list[int]) -> int:
        if not batch_starts:
            return 0
        windows: list[np.ndarray] = []
        paths: list[Path] = []
        valid_starts: list[int] = []
        for s in batch_starts:
            window = np.asarray(imgs_uint8[s:s + T + k])
            if window.shape[0] < T + k:
                continue
            windows.append(window)
            paths.append(out_dir / f"{window_geom_id(cid, s)}.npz")
            valid_starts.append(s)
        if not windows:
            return 0
        flat = np.concatenate(windows, axis=0)
        frames_224 = resize_image_batch(flat, 224).reshape(len(windows), T + k, 3, 224, 224).to("cuda")
        try:
            out = enc(frames_224)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(windows) == 1:
                raise
            return sum(flush([s]) for s in batch_starts)
        for b, (s, path) in enumerate(zip(valid_starts, paths)):
            _save_window_geom_from_batch(
                out,
                path,
                batch_idx=b,
                start=s,
                T=T,
                k=k,
                hw=hw,
                include_pooled=include_pooled,
            )
        return len(paths)

    for start in starts:
        path = out_dir / f"{window_geom_id(cid, start)}.npz"
        if not force and window_geom_complete(path, T, k, include_pooled=include_pooled):
            done += 1
            continue
        pending.append(start)
        if len(pending) >= batch_windows:
            done += flush(pending)
            pending = []
    done += flush(pending)
    return done


def cache_window_geom_extra_from_cached_rgb(
    cache_root: Path,
    enc: VGGTEncoder,
    out_dir: Path,
    cid: str,
    *,
    T: int,
    k: int,
    stride: int,
    max_windows: int,
    hw: int,
    include_pooled: bool,
    force: bool = False,
    batch_windows: int = 1,
) -> int:
    rgb_path = cache_root / "rgb_256" / f"{cid}.npy"
    if not validate_rgb_npy(rgb_path):
        return 0
    rgb = np.load(rgb_path, mmap_mode="r")
    starts = iter_window_starts(int(rgb.shape[0]), T, k, stride, max_windows)
    return cache_window_geom_extra_many(
        enc,
        rgb,
        out_dir,
        cid,
        starts=starts,
        T=T,
        k=k,
        hw=hw,
        include_pooled=include_pooled,
        force=force,
        batch_windows=batch_windows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("/data/Minko/world_model/wm3d_v5/manifests/oxe_droid20k_balanced_world_v2.jsonl"))
    ap.add_argument("--cache_root", type=Path,
                    default=Path("/0604-10T-test/wm3d_v5/cache"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch_frames", type=int, default=16,
                    help="VGGT chunk size per episode")
    ap.add_argument("--skip_qwen", action="store_true",
                    help="skip Qwen task embedding (saves loading the 2B model)")
    ap.add_argument("--only_qwen", action="store_true",
                    help="only backfill missing Qwen task embeddings; do not run VGGT")
    ap.add_argument("--no_rgb", action="store_true",
                    help="do not write rgb_256 cache")
    ap.add_argument("--no_geom", action="store_true",
                    help="do not write vggt_geom cache")
    ap.add_argument("--geom_extra", dest="geom_extra", action="store_true", default=True,
                    help="when writing vggt_geom, include VGGT point/pose/conf extras if heads are available")
    ap.add_argument("--no_geom_extra", "--no-geom_extra", dest="geom_extra", action="store_false",
                    help="write legacy depth-only vggt_geom cache")
    ap.add_argument("--geom_extra_subdir", type=str, default="",
                    help="optional window-aligned native3D cache subdir under cache_root")
    ap.add_argument("--geom_extra_hw", type=int, default=64,
                    help="spatial size for window point/conf targets")
    ap.add_argument("--no_geom_extra_pooled", action="store_true",
                    help="do not store same-forward pooled tokens in window geom npz")
    ap.add_argument("--geom_extra_T", type=int, default=16)
    ap.add_argument("--geom_extra_k", type=int, default=8)
    ap.add_argument("--geom_extra_stride", type=int, default=4)
    ap.add_argument("--geom_extra_max_windows_per_episode", type=int, default=0)
    ap.add_argument("--geom_extra_batch_windows", type=int, default=1,
                    help="number of T+k windows to encode in one VGGT forward for window-aligned native3D cache")
    ap.add_argument("--force_geom_extra", action="store_true",
                    help="rewrite existing window-aligned native3D npz files")
    args = ap.parse_args()
    if args.skip_qwen and args.only_qwen:
        raise SystemExit("--skip_qwen and --only_qwen are mutually exclusive")
    if args.only_qwen and (args.no_rgb or args.no_geom):
        raise SystemExit("--only_qwen does not combine with --no_rgb/--no_geom")
    if args.no_geom and args.geom_extra and not args.geom_extra_subdir:
        args.geom_extra = False

    records = read_manifest(args.manifest)
    by_tar: dict[str, list[OXEClipRecord]] = defaultdict(list)
    tarless_records: list[OXEClipRecord] = []
    for r in records:
        tar_path = str(getattr(r, "tar_path", "") or "")
        if tar_path and Path(tar_path).exists():
            by_tar[tar_path].append(r)
        else:
            tarless_records.append(r)
    tar_keys = sorted(by_tar.keys())[args.shard :: args.world]

    pool_dir = args.cache_root / "vggt_pooled"
    geom_dir = args.cache_root / "vggt_geom"
    rgb_dir = args.cache_root / "rgb_256"
    act_dir = args.cache_root / "actions"
    qwen_dir = args.cache_root / "qwen_taskemb"
    dirs = [pool_dir, act_dir, qwen_dir]
    window_geom_dir = args.cache_root / args.geom_extra_subdir if args.geom_extra_subdir else None
    if not args.no_geom:
        dirs.append(geom_dir)
    if not args.no_rgb:
        dirs.append(rgb_dir)
    if window_geom_dir is not None:
        dirs.append(window_geom_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    enc = None if args.only_qwen else VGGTEncoder(
        device="cuda",
        return_depth=(not args.no_geom) or bool(args.geom_extra_subdir),
        return_geom_extra=bool(args.geom_extra or args.geom_extra_subdir),
    )
    qwen = None if args.skip_qwen else QwenVLEmbed()
    need_qwen = qwen is not None
    warned_missing: set[str] = set()

    for tar_path in tar_keys:
        recs = by_tar[tar_path]
        tar = tarfile.open(tar_path, "r")
        wanted = {r.pickle_member: r for r in recs}
        done = 0
        for r in recs:
            cid = safe_id(r.clip_id)
            if args.only_qwen:
                if validate_qwen_npy(qwen_dir / f"{cid}.npy"):
                    done += 1
            else:
                base_ok = cache_complete(
                    args.cache_root,
                    cid,
                    need_qwen,
                    need_rgb=not args.no_rgb,
                    need_geom=not args.no_geom,
                    need_geom_extra=bool(args.geom_extra),
                )
                window_ok = True
                if window_geom_dir is not None:
                    n_frames = frame_count_npy(rgb_dir / f"{cid}.npy") or int(getattr(r, "n_frames", 0) or 0)
                    window_ok = window_geom_cache_complete(
                        window_geom_dir,
                        cid,
                        n_frames,
                        args.geom_extra_T,
                        args.geom_extra_k,
                        args.geom_extra_stride,
                        args.geom_extra_max_windows_per_episode,
                        include_pooled=not args.no_geom_extra_pooled,
                    )
                if base_ok and window_ok:
                    done += 1
        if done == len(recs):
            tar.close()
            continue

        pbar = tqdm(total=len(recs), desc=f"shard{args.shard} {Path(tar_path).name}")
        for member in tar.getmembers():
            if member.name not in wanted:
                continue
            r = wanted[member.name]
            cid = safe_id(r.clip_id)
            pool_path = pool_dir / f"{cid}.npy"
            geom_path = geom_dir / f"{cid}.npz"
            rgb_path = rgb_dir / f"{cid}.npy"
            act_path = act_dir / f"{cid}.npy"
            qwen_path = qwen_dir / f"{cid}.npy"

            if args.only_qwen:
                base_done = False
                if validate_qwen_npy(qwen_path):
                    pbar.update(1)
                    continue
            else:
                base_done = cache_complete(
                    args.cache_root,
                    cid,
                    need_qwen,
                    need_rgb=not args.no_rgb,
                    need_geom=not args.no_geom,
                    need_geom_extra=bool(args.geom_extra),
                )
                window_done = True
                if window_geom_dir is not None:
                    n_frames = frame_count_npy(rgb_path) or int(getattr(r, "n_frames", 0) or 0)
                    window_done = window_geom_cache_complete(
                        window_geom_dir,
                        cid,
                        n_frames,
                        args.geom_extra_T,
                        args.geom_extra_k,
                        args.geom_extra_stride,
                        args.geom_extra_max_windows_per_episode,
                        include_pooled=not args.no_geom_extra_pooled,
                    )
                if base_done and window_done:
                    pbar.update(1)
                    continue

            f = tar.extractfile(member)
            if f is None:
                pbar.update(1)
                continue
            try:
                raw = pickle.load(f)
                ep = decode_episode(raw, r.clip_id, r.dataset)
                if ep is None:
                    pbar.update(1)
                    continue
            except Exception:
                pbar.update(1)
                continue

            if not args.only_qwen and not base_done:
                T = int(ep.images.shape[0])
                need_actions = not validate_actions_npy(act_path, expected_frames=T)
                need_rgb = (not args.no_rgb) and not validate_rgb_npy(rgb_path, expected_frames=T)
                need_pooled = not validate_pooled_npy(pool_path, expected_frames=T)
                geom_needs_write = not args.no_geom and not validate_geom_npz(
                    geom_path,
                    expected_frames=T,
                    require_geom_extra=bool(args.geom_extra),
                )

                if need_actions:
                    np.save(act_path, ep.actions.astype(np.float32))
                if need_rgb:
                    rgb256 = resize_image_batch(ep.images, 256)
                    np.save(rgb_path, (rgb256.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy())

                if need_pooled or geom_needs_write:
                    frames_224 = resize_image_batch(ep.images, 224)
                    pooled_chunks: list[np.ndarray] = []
                    geom_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
                    for s in range(0, T, args.batch_frames):
                        chunk = frames_224[s : s + args.batch_frames].unsqueeze(0).to("cuda")
                        assert enc is not None
                        out = enc(chunk)
                        for missing in out.get("geom_extra_missing", []):
                            if missing not in warned_missing:
                                print(f"  [vggt geom warn] missing {missing}; continuing without that field")
                                warned_missing.add(missing)
                        if need_pooled:
                            pooled_chunks.append(out["pooled"][0].cpu().numpy().astype(np.float16))
                        if geom_needs_write:
                            for key in ("depth", *GEOM_EXTRA_KEYS):
                                _append_chunk(geom_chunks, out, key)
                    if need_pooled and pooled_chunks:
                        np.save(pool_path, np.concatenate(pooled_chunks, axis=0))
                    if geom_needs_write:
                        payload = _existing_geom_payload(geom_path)
                        for key, vals in geom_chunks.items():
                            if vals:
                                payload[key] = np.concatenate(vals, axis=0)
                        if payload:
                            atomic_savez_compressed(geom_path, **payload)

            if not args.only_qwen and window_geom_dir is not None:
                n_frames = int(ep.images.shape[0])
                starts = iter_window_starts(
                    n_frames,
                    args.geom_extra_T,
                    args.geom_extra_k,
                    args.geom_extra_stride,
                    args.geom_extra_max_windows_per_episode,
                )
                cache_window_geom_extra_many(
                    enc,
                    ep.images,
                    window_geom_dir,
                    cid,
                    starts=starts,
                    T=args.geom_extra_T,
                    k=args.geom_extra_k,
                    hw=args.geom_extra_hw,
                    include_pooled=not args.no_geom_extra_pooled,
                    force=args.force_geom_extra,
                    batch_windows=args.geom_extra_batch_windows,
                )

            if qwen is not None and not validate_qwen_npy(qwen_path):
                first_img = Image.fromarray(ep.images[0])
                try:
                    emb = qwen.embed(ep.task_text or "robot manipulation", first_img)
                    np.save(qwen_path, emb.numpy().astype(np.float16))
                except Exception as e:
                    print(f"  [qwen err] {cid}: {e}")
            pbar.update(1)
        pbar.close()
        tar.close()

    if tarless_records and window_geom_dir is not None and not args.only_qwen:
        unique: dict[str, OXEClipRecord] = {}
        for r in tarless_records:
            unique.setdefault(safe_id(r.clip_id), r)
        shard_records = list(unique.values())[args.shard :: args.world]
        pbar = tqdm(total=len(shard_records), desc=f"shard{args.shard} cached-rgb")
        for r in shard_records:
            cid = safe_id(r.clip_id)
            try:
                n_frames = frame_count_npy(rgb_dir / f"{cid}.npy") or int(getattr(r, "n_frames", 0) or 0)
                if window_geom_cache_complete(
                    window_geom_dir,
                    cid,
                    n_frames,
                    args.geom_extra_T,
                    args.geom_extra_k,
                    args.geom_extra_stride,
                    args.geom_extra_max_windows_per_episode,
                    include_pooled=not args.no_geom_extra_pooled,
                ):
                    pbar.update(1)
                    continue
                cache_window_geom_extra_from_cached_rgb(
                    args.cache_root,
                    enc,
                    window_geom_dir,
                    cid,
                    T=args.geom_extra_T,
                    k=args.geom_extra_k,
                    stride=args.geom_extra_stride,
                    max_windows=args.geom_extra_max_windows_per_episode,
                    hw=args.geom_extra_hw,
                    include_pooled=not args.no_geom_extra_pooled,
                    force=args.force_geom_extra,
                    batch_windows=args.geom_extra_batch_windows,
                )
            except Exception as e:
                print(f"  [window geom err] {cid}: {e}")
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    main()
