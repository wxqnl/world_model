"""Cache OXE episodes: VGGT pooled tokens + VGGT 224×224 depth + RGB 256×256 + actions + Qwen task emb.

One tar at a time (sharded across GPUs by tar+dataset hash). Each episode produces
5 files under cache/wm3d_v3/<kind>/<clip_id_safe>.{npy,npz}.
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


def cache_complete(cache_root: Path, cid: str, need_qwen: bool) -> bool:
    base_ok = (
        (cache_root / "vggt_pooled" / f"{cid}.npy").exists()
        and (cache_root / "vggt_geom" / f"{cid}.npz").exists()
        and (cache_root / "rgb_256" / f"{cid}.npy").exists()
        and (cache_root / "actions" / f"{cid}.npy").exists()
    )
    qwen_ok = (not need_qwen) or (cache_root / "qwen_taskemb" / f"{cid}.npy").exists()
    return base_ok and qwen_ok


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
    ap.add_argument("--skip_qwen", action="store_true",
                    help="skip Qwen task embedding (saves loading the 2B model)")
    ap.add_argument("--only_qwen", action="store_true",
                    help="only backfill missing Qwen task embeddings; do not run VGGT")
    args = ap.parse_args()
    if args.skip_qwen and args.only_qwen:
        raise SystemExit("--skip_qwen and --only_qwen are mutually exclusive")

    records = read_manifest(args.manifest)
    # Group by tar so we open each tar once.
    by_tar: dict[str, list[OXEClipRecord]] = defaultdict(list)
    for r in records:
        by_tar[r.tar_path].append(r)
    tar_keys = sorted(by_tar.keys())
    # Shard across GPUs
    tar_keys = tar_keys[args.shard :: args.world]

    pool_dir = args.cache_root / "vggt_pooled"
    geom_dir = args.cache_root / "vggt_geom"
    rgb_dir = args.cache_root / "rgb_256"
    act_dir = args.cache_root / "actions"
    qwen_dir = args.cache_root / "qwen_taskemb"
    for d in (pool_dir, geom_dir, rgb_dir, act_dir, qwen_dir):
        d.mkdir(parents=True, exist_ok=True)

    enc = None if args.only_qwen else VGGTEncoder(device="cuda", return_depth=True)
    qwen = None if args.skip_qwen else QwenVLEmbed()
    need_qwen = qwen is not None

    for tar_path in tar_keys:
        recs = by_tar[tar_path]
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
            elif cache_complete(args.cache_root, cid, need_qwen):
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

            base_done = pool_path.exists() and geom_path.exists() and rgb_path.exists() and act_path.exists()
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
                # Save actions + RGB first (cheap)
                np.save(act_path, ep.actions.astype(np.float32))
                rgb256 = resize_image_batch(ep.images, 256)  # [T, 3, 256, 256] float
                np.save(rgb_path, (rgb256.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy())

                # VGGT cache (pooled + depth)
                T = ep.images.shape[0]
                frames_224 = resize_image_batch(ep.images, 224)  # [T, 3, 224, 224]
                pooled_chunks, depth_chunks = [], []
                for s in range(0, T, args.batch_frames):
                    chunk = frames_224[s : s + args.batch_frames].unsqueeze(0).to("cuda")
                    assert enc is not None
                    out = enc(chunk)
                    pooled_chunks.append(out["pooled"][0].cpu().numpy().astype(np.float16))
                    depth_chunks.append(out["depth"][0].cpu().numpy().astype(np.float16))
                np.save(pool_path, np.concatenate(pooled_chunks, axis=0))
                np.savez_compressed(geom_path, depth=np.concatenate(depth_chunks, axis=0))

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
