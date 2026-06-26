"""Cache OXE episodes: VGGT pooled tokens + optional VGGT depth/RGB + actions + Qwen task emb.

One tar at a time (sharded across GPUs by tar+dataset hash). Each episode produces
files under cache/wm3d_v3/<kind>/<clip_id_safe>.{npy,npz}.
"""
from __future__ import annotations
import argparse
import io
import pickle
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from wm3d_v3.encoders.vggt_encoder import VGGTEncoder
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed

from wm3d_v3.data.manifest import read_manifest, OXEClipRecord
from wm3d_v3.data.oxe_loader import decode_episode


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def resize_image_batch(imgs: np.ndarray, size: int) -> torch.Tensor:
    """[T, H, W, 3] uint8 -> [T, 3, size, size] float [0,1]"""
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                           align_corners=False, antialias=True)
    return t


def window_geom_id(cid: str, start: int) -> str:
    return f"{cid}__start_{int(start):08d}.npz"


def resize_map_chw(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode=mode, align_corners=False)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def cache_complete(
    cache_root: Path,
    cid: str,
    need_qwen: bool,
    need_rgb: bool = True,
    need_geom: bool = True,
    need_geom_extra: bool = False,
    geom_extra_subdir: str = "vggt_window_geom_p64",
) -> bool:
    base_ok = (
        (cache_root / "vggt_pooled" / f"{cid}.npy").exists()
        and (cache_root / "actions" / f"{cid}.npy").exists()
    )
    if need_geom:
        base_ok = base_ok and (cache_root / "vggt_geom" / f"{cid}.npz").exists()
    if need_rgb:
        base_ok = base_ok and (cache_root / "rgb_256" / f"{cid}.npy").exists()
    if need_geom_extra:
        base_ok = base_ok and (cache_root / geom_extra_subdir / f"{cid}.done").exists()
    qwen_ok = (not need_qwen) or (cache_root / "qwen_taskemb" / f"{cid}.npy").exists()
    return base_ok and qwen_ok


def cache_window_geom_extra(
    enc: VGGTEncoder,
    frames_224: torch.Tensor,
    out_dir: Path,
    cid: str,
    *,
    win_T: int,
    win_k: int,
    stride: int,
    geom_hw: int,
    max_windows: int,
    force: bool,
    store_pooled: bool,
) -> tuple[int, int]:
    """Cache native-3D targets per training window.

    VGGT world points are only in a consistent coordinate system inside one
    forward pass. Therefore this function runs VGGT on the exact T+k frames of
    each training window and stores only future-frame point/pose targets for
    that window. It intentionally does not concatenate fixed-size VGGT chunks.
    """
    win = int(win_T + win_k)
    n_frames = int(frames_224.shape[0])
    starts = list(range(0, max(0, n_frames - win + 1), int(stride)))
    if max_windows > 0:
        starts = starts[: int(max_windows)]
    done = 0
    for start in starts:
        out_path = out_dir / window_geom_id(cid, start)
        if out_path.exists() and not force:
            done += 1
            continue
        chunk = frames_224[start : start + win].unsqueeze(0).to("cuda")
        out = enc(chunk)
        if int(out.get("geom_extra_missing", torch.tensor(1)).item()) != 0:
            raise RuntimeError("VGGT geom extra heads are missing")
        points = out["world_points"][0, win_T:].float()  # [k,H,W,3]
        point_conf = out["world_points_conf"][0, win_T:].float()  # [k,H,W]
        pose = out["pose_enc"][0, win_T:].float()  # [k,9]
        depth_conf = out["depth_conf"][0, win_T:].float() if "depth_conf" in out else point_conf
        if geom_hw > 0 and (points.shape[1] != geom_hw or points.shape[2] != geom_hw):
            points_chw = points.permute(0, 3, 1, 2)
            points = resize_map_chw(points_chw, geom_hw).permute(0, 2, 3, 1)
            point_conf = resize_map_chw(point_conf[:, None], geom_hw).squeeze(1)
            depth_conf = resize_map_chw(depth_conf[:, None], geom_hw).squeeze(1)
        point_conf = point_conf / point_conf.flatten(1).median(dim=1).values[:, None, None].clamp_min(1e-6)
        depth_conf = depth_conf / depth_conf.flatten(1).median(dim=1).values[:, None, None].clamp_min(1e-6)
        arrays = {
            "point": points.cpu().numpy().astype(np.float16),
            "point_conf": point_conf.cpu().numpy().astype(np.float16),
            "pose": pose.cpu().numpy().astype(np.float16),
            "pose_conf": np.clip(point_conf.flatten(1).mean(dim=1).cpu().numpy(), 0.0, 1000.0).astype(np.float16),
            "depth_conf": depth_conf.cpu().numpy().astype(np.float16),
            "window_start": np.asarray(start, dtype=np.int64),
            "window_T": np.asarray(win_T, dtype=np.int64),
            "window_k": np.asarray(win_k, dtype=np.int64),
        }
        if store_pooled:
            arrays["pooled"] = out["pooled"][0].cpu().numpy().astype(np.float16)
        save_npz_atomic(out_path, **arrays)
        done += 1
    if starts and max_windows <= 0:
        (out_dir / f"{cid}.done").write_text(f"{done}/{len(starts)}\n")
    return done, len(starts)


def cache_window_geom_extra_from_cached_rgb(
    enc: VGGTEncoder,
    rgb_path: Path,
    out_dir: Path,
    cid: str,
    *,
    win_T: int,
    win_k: int,
    stride: int,
    geom_hw: int,
    max_windows: int,
    force: bool,
    store_pooled: bool,
) -> tuple[int, int]:
    """Generate window-native VGGT targets from cached rgb_256 frames.

    Some OXE records, notably DROID, are represented in the manifest only by
    existing caches and do not have a tar_path.  For native3D cache generation
    we can still run VGGT on cached RGB frames.  This path intentionally only
    handles window geometry/tokens; it does not pretend to rebuild missing base
    actions, depth, or task embeddings.
    """
    if not rgb_path.exists():
        raise FileNotFoundError(f"missing cached RGB for tar-less clip: {rgb_path}")
    rgb = np.load(rgb_path, mmap_mode="r")
    frames_224 = resize_image_batch(np.array(rgb), 224)
    return cache_window_geom_extra(
        enc,
        frames_224,
        out_dir,
        cid,
        win_T=win_T,
        win_k=win_k,
        stride=stride,
        geom_hw=geom_hw,
        max_windows=max_windows,
        force=force,
        store_pooled=store_pooled,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("/home/user01/Minko/newwm/wm3d_v3/manifests/oxe_train.jsonl"))
    ap.add_argument("--cache_root", type=Path,
                    default=Path("/home/user01/Minko/datasets/cache/wm3d_v3"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch_frames", type=int, default=16,
                    help="VGGT chunk size per episode")
    ap.add_argument("--geom_extra", action="store_true",
                    help="write window-aligned VGGT point/pose/conf native-3D targets")
    ap.add_argument("--geom_extra_subdir", default="vggt_window_geom_p64")
    ap.add_argument("--geom_extra_hw", type=int, default=64,
                    help="spatial size for cached point/conf targets; keep 64 for p64 training")
    ap.add_argument("--no_geom_extra_pooled", action="store_true",
                    help="do not store same-forward p64 tokens in window native3D cache")
    ap.add_argument("--geom_extra_T", type=int, default=16)
    ap.add_argument("--geom_extra_k", type=int, default=8)
    ap.add_argument("--geom_extra_stride", type=int, default=4)
    ap.add_argument("--geom_extra_max_windows_per_episode", type=int, default=0)
    ap.add_argument("--force_geom_extra", action="store_true")
    ap.add_argument("--skip_qwen", action="store_true",
                    help="skip Qwen task embedding (saves loading the 2B model)")
    ap.add_argument("--only_qwen", action="store_true",
                    help="only backfill missing Qwen task embeddings; do not run VGGT")
    ap.add_argument("--no_rgb", action="store_true",
                    help="do not write rgb_256 cache")
    ap.add_argument("--no_geom", action="store_true",
                    help="do not write vggt_geom depth cache")
    args = ap.parse_args()
    if args.skip_qwen and args.only_qwen:
        raise SystemExit("--skip_qwen and --only_qwen are mutually exclusive")
    if args.only_qwen and (args.no_rgb or args.no_geom):
        raise SystemExit("--only_qwen does not combine with --no_rgb/--no_geom")

    records = read_manifest(args.manifest)
    # Group by tar so we open each tar once.  Records without tar_path are
    # handled separately from cached rgb_256 and sharded by clip id; otherwise
    # all DROID cached-only data would collapse onto shard 0.
    by_tar: dict[str, list[OXEClipRecord]] = defaultdict(list)
    tarless_records: list[OXEClipRecord] = []
    for r in records:
        if not r.tar_path:
            tarless_records.append(r)
        else:
            by_tar[r.tar_path].append(r)
    tar_keys = sorted(by_tar.keys())
    # Shard across GPUs
    tar_keys = tar_keys[args.shard :: args.world]

    pool_dir = args.cache_root / "vggt_pooled"
    geom_dir = args.cache_root / "vggt_geom"
    rgb_dir = args.cache_root / "rgb_256"
    geom_extra_dir = args.cache_root / args.geom_extra_subdir
    act_dir = args.cache_root / "actions"
    qwen_dir = args.cache_root / "qwen_taskemb"
    dirs = [pool_dir, act_dir, qwen_dir]
    if not args.no_geom:
        dirs.append(geom_dir)
    if args.geom_extra:
        dirs.append(geom_extra_dir)
    if not args.no_rgb:
        dirs.append(rgb_dir)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    enc = None if args.only_qwen else VGGTEncoder(
        device="cuda",
        return_depth=not args.no_geom,
        return_geom_extra=bool(args.geom_extra),
    )
    qwen = None if args.skip_qwen else QwenVLEmbed()
    need_qwen = qwen is not None

    def process_cached_rgb_records(recs: list[OXEClipRecord], desc: str) -> None:
        pbar = tqdm(total=len(recs), desc=desc)
        seen_cids: set[str] = set()
        for r in recs:
            cid = safe_id(r.clip_id)
            if cid in seen_cids:
                pbar.update(1)
                continue
            seen_cids.add(cid)
            rgb_path = rgb_dir / f"{cid}.npy"
            qwen_path = qwen_dir / f"{cid}.npy"
            if args.geom_extra and not args.only_qwen:
                assert enc is not None
                try:
                    cache_window_geom_extra_from_cached_rgb(
                        enc,
                        rgb_path,
                        geom_extra_dir,
                        cid,
                        win_T=int(args.geom_extra_T),
                        win_k=int(args.geom_extra_k),
                        stride=int(args.geom_extra_stride),
                        geom_hw=int(args.geom_extra_hw),
                        max_windows=int(args.geom_extra_max_windows_per_episode),
                        force=bool(args.force_geom_extra),
                        store_pooled=not bool(args.no_geom_extra_pooled),
                    )
                except FileNotFoundError as e:
                    print(f"  [cached-rgb err] {cid}: {e}")
            if qwen is not None and not qwen_path.exists() and rgb_path.exists():
                rgb = np.load(rgb_path, mmap_mode="r")
                first_img = Image.fromarray(np.array(rgb[0]))
                try:
                    emb = qwen.embed(r.task_text or "robot manipulation", first_img)
                    np.save(qwen_path, emb.numpy().astype(np.float16))
                except Exception as e:
                    print(f"  [qwen err] {cid}: {e}")
            pbar.update(1)
        pbar.close()

    if tarless_records:
        unique_tarless: list[OXEClipRecord] = []
        seen_clip_ids: set[str] = set()
        for r in tarless_records:
            if r.clip_id in seen_clip_ids:
                continue
            seen_clip_ids.add(r.clip_id)
            unique_tarless.append(r)
        process_cached_rgb_records(
            unique_tarless[args.shard :: args.world],
            desc=f"shard{args.shard} cached_rgb",
        )

    for tar_path in tar_keys:
        recs = by_tar[tar_path]
        if not Path(tar_path).exists():
            process_cached_rgb_records(recs, desc=f"shard{args.shard} cached_rgb")
            continue
        tar = tarfile.open(tar_path, "r")
        # Build a map: pickle_member -> rec for quick lookup
        wanted = {r.pickle_member: r for r in recs}
        # Determine which members we need (skip if all already cached)
        done = 0
        for r in recs:
            cid = safe_id(r.clip_id)
            if args.only_qwen:
                if (qwen_dir / f"{cid}.npy").exists():
                    done += 1
            elif cache_complete(
                args.cache_root,
                cid,
                need_qwen,
                need_rgb=not args.no_rgb,
                need_geom=not args.no_geom,
                need_geom_extra=bool(args.geom_extra),
                geom_extra_subdir=args.geom_extra_subdir,
            ):
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

            base_done = (
                pool_path.exists()
                and act_path.exists()
                and (args.no_geom or geom_path.exists())
                and (args.no_rgb or rgb_path.exists())
                and (not args.geom_extra or (geom_extra_dir / f"{cid}.done").exists())
            )
            if base_done and ((not need_qwen) or qwen_path.exists()):
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
                frames_224 = None
                # Save actions first; RGB/depth are optional for action-policy-only runs.
                if not act_path.exists():
                    np.save(act_path, ep.actions.astype(np.float32))
                if not args.no_rgb and not rgb_path.exists():
                    rgb256 = resize_image_batch(ep.images, 256)  # [T, 3, 256, 256] float
                    np.save(rgb_path, (rgb256.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy())

                if not pool_path.exists() or (not args.no_geom and not geom_path.exists()):
                    T = ep.images.shape[0]
                    frames_224 = resize_image_batch(ep.images, 224)  # [T, 3, 224, 224]
                    pooled_chunks, depth_chunks = [], []
                    for s in range(0, T, args.batch_frames):
                        chunk = frames_224[s : s + args.batch_frames].unsqueeze(0).to("cuda")
                        assert enc is not None
                        out = enc(chunk)
                        if not pool_path.exists():
                            pooled_chunks.append(out["pooled"][0].cpu().numpy().astype(np.float16))
                        if not args.no_geom and not geom_path.exists():
                            depth_chunks.append(out["depth"][0].cpu().numpy().astype(np.float16))
                    if pooled_chunks:
                        np.save(pool_path, np.concatenate(pooled_chunks, axis=0))
                    if depth_chunks:
                        np.savez_compressed(geom_path, depth=np.concatenate(depth_chunks, axis=0))
                elif args.geom_extra:
                    frames_224 = resize_image_batch(ep.images, 224)

                if args.geom_extra:
                    assert enc is not None
                    if frames_224 is None:
                        frames_224 = resize_image_batch(ep.images, 224)
                    cache_window_geom_extra(
                        enc,
                        frames_224,
                        geom_extra_dir,
                        cid,
                        win_T=int(args.geom_extra_T),
                        win_k=int(args.geom_extra_k),
                        stride=int(args.geom_extra_stride),
                        geom_hw=int(args.geom_extra_hw),
                        max_windows=int(args.geom_extra_max_windows_per_episode),
                        force=bool(args.force_geom_extra),
                        store_pooled=not bool(args.no_geom_extra_pooled),
                    )

            # Qwen task embedding (per episode, one vector)
            if qwen is not None and not qwen_path.exists():
                first_img = Image.fromarray(ep.images[0])
                try:
                    emb = qwen.embed(ep.task_text or "robot manipulation", first_img)
                    np.save(qwen_path, emb.numpy().astype(np.float16))
                except Exception as e:
                    print(f"  [qwen err] {cid}: {e}")
            pbar.update(1)
        pbar.close()
        tar.close()


if __name__ == "__main__":
    main()
