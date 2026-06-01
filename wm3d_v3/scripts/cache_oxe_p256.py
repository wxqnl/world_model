"""Cache OXE episodes with 16×16 = 256 VGGT patch tokens (unpooled).

Output: /mnt/data1/wm3d_v3_p256/vggt_p256/<safe_id>.npy   [n_frames, 256, 2048] fp16

Other caches (rgb_256, vggt_geom, qwen_taskemb, actions) are symlinked from the
existing /home/user01/Minko/datasets/cache/wm3d_v3/ — they don't change.

Sharded across GPUs by tar hash so 4 GPUs can run in parallel.
"""
from __future__ import annotations
import argparse
import os
import pickle
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from wm3d_v3.encoders.vggt_encoder import VGGTEncoder

from wm3d_v3.data.manifest import read_manifest, OXEClipRecord
from wm3d_v3.data.oxe_loader import decode_episode


def safe_id(clip_id: str) -> str:
    return clip_id.replace("/", "__")


def resize_image_batch(imgs: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t, size=(size, size), mode="bilinear",
                           align_corners=False, antialias=True)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path,
                    default=Path("/home/user01/Minko/newwm/wm3d_v3/manifests/oxe_train.jsonl"))
    ap.add_argument("--out_root", type=Path,
                    default=Path("/mnt/data1/wm3d_v3_p256"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch_frames", type=int, default=16)
    ap.add_argument("--token_grid", type=int, default=16)
    ap.add_argument("--limit_tars", type=int, default=0,
                    help="if >0, only process this many tars (smoke test)")
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    by_tar: dict[str, list[OXEClipRecord]] = defaultdict(list)
    for r in records:
        by_tar[r.tar_path].append(r)
    tar_keys = sorted(by_tar.keys())
    tar_keys = tar_keys[args.shard :: args.world]
    if args.limit_tars > 0:
        tar_keys = tar_keys[: args.limit_tars]

    pool_dir = args.out_root / "vggt_p256"
    pool_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shard {args.shard}/{args.world}] {len(tar_keys)} tars, token_grid={args.token_grid}")
    enc = VGGTEncoder(device="cuda", token_grid=args.token_grid)

    for tar_path in tar_keys:
        recs = by_tar[tar_path]
        tar = tarfile.open(tar_path, "r")
        wanted = {r.pickle_member: r for r in recs}
        # skip whole tar if all done
        done = sum(1 for r in recs if (pool_dir / f"{safe_id(r.clip_id)}.npy").exists())
        if done == len(recs):
            tar.close()
            continue

        pbar = tqdm(total=len(recs), desc=f"sh{args.shard} {Path(tar_path).name}")
        for member in tar.getmembers():
            if member.name not in wanted:
                continue
            r = wanted[member.name]
            cid = safe_id(r.clip_id)
            pool_path = pool_dir / f"{cid}.npy"
            if pool_path.exists():
                pbar.update(1)
                continue
            f = tar.extractfile(member)
            if f is None:
                pbar.update(1); continue
            try:
                raw = pickle.load(f)
                ep = decode_episode(raw, r.clip_id, r.dataset)
                if ep is None:
                    pbar.update(1); continue
            except Exception:
                pbar.update(1); continue

            T = ep.images.shape[0]
            frames_224 = resize_image_batch(ep.images, 224)
            pooled_chunks = []
            for s in range(0, T, args.batch_frames):
                chunk = frames_224[s : s + args.batch_frames].unsqueeze(0).to("cuda")
                out = enc(chunk)
                pooled_chunks.append(out["pooled"][0].cpu().numpy().astype(np.float16))
            arr = np.concatenate(pooled_chunks, axis=0)
            tmp = pool_path.with_name(pool_path.name + ".tmp")
            np.save(tmp, arr, allow_pickle=False)
            # np.save adds ".npy" if not already present — tmp had no .npy so it became "<name>.tmp.npy"
            actual_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npy")
            os.replace(actual_tmp, pool_path)
            pbar.update(1)
        pbar.close()
        tar.close()


if __name__ == "__main__":
    main()
