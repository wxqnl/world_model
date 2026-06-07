"""Cache wm3d_v3 world-model controls for video-backend training.

This script materializes the structured controls that a Hunyuan/Wan/tau0-style
renderer should consume. It intentionally treats the current RGB decoder as a
diagnostic rough renderer; the durable controls are tokens, depth, action, task,
and motion/contact hints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.splits import random_window_indices, split_mode_from_config, split_records
from wm3d_v3.data.window_dataset import OXEWindowDataset
from wm3d_v3.eval.run_eval import build_model, window_config_from_cfg


def split_indices(n: int, cfg: dict, split: str) -> list[int]:
    """Return dataset indices for train/val/all using the legacy random split."""

    if split == "all":
        return list(range(n))
    g = torch.Generator().manual_seed(cfg["data"].get("seed", 0))
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * cfg["data"].get("val_frac", 0.05)))
    if split == "val":
        return perm[:n_val]
    if split == "train":
        return perm[n_val:]
    raise ValueError(f"unknown split: {split}")


def dataset_for_split(records, cfg: dict, split: str):
    """Return `(dataset, source_indices)` for cache materialization.

    `source_indices` is present for all/random-window splits where the dataset is
    a subset of a full window dataset. Episode splits build a fresh dataset from
    selected records, so source indices are sequential in that split-local view.
    """
    wcfg = window_config_from_cfg(cfg)
    if split == "all":
        ds = OXEWindowDataset(records, wcfg)
        return ds, list(range(len(ds)))
    mode = split_mode_from_config(cfg["data"])
    if mode == "episode":
        train_records, val_records = split_records(records, cfg["data"])
        selected = train_records if split == "train" else val_records
        ds = OXEWindowDataset(selected, wcfg)
        if len(ds) == 0:
            raise RuntimeError(f"episode {split} split empty — caches missing?")
        return ds, list(range(len(ds)))
    if mode != "random_window":
        raise ValueError(f"unsupported data.split.mode: {mode}")
    ds = OXEWindowDataset(records, wcfg)
    train_idx, val_idx = random_window_indices(
        len(ds),
        val_frac=float(cfg["data"].get("val_frac", 0.05)),
        seed=int(cfg["data"].get("seed", 0)),
    )
    indices = train_idx if split == "train" else val_idx
    return Subset(ds, indices), indices


def tensor_to_numpy(x: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        x = x.float()
    arr = x.detach().cpu().numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def batch_metadata(batch: dict, subset_indices: list[int], offset: int, batch_size: int) -> dict:
    raw_indices = subset_indices[offset : offset + batch_size]
    return {
        "dataset": list(batch["dataset"]),
        "clip_id": list(batch["clip_id"]),
        "start": [int(v) for v in batch["start"]],
        "dataset_index": [int(v) for v in raw_indices],
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val", "all"], default="val")
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--shard_size", type=int, default=128)
    ap.add_argument("--no_rgb", action="store_true",
                    help="skip diagnostic rough RGB to reduce cache size")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    ds, indices = dataset_for_split(records, cfg, args.split)
    if args.max_windows > 0:
        indices = indices[: args.max_windows]
        if isinstance(ds, Subset):
            ds = Subset(ds.dataset, indices)
        else:
            ds = Subset(ds, indices)

    batch_size = args.batch_size or int(cfg["train"]["batch_size_per_gpu"])
    num_workers = int(cfg["train"]["num_workers"]) if args.num_workers < 0 else args.num_workers
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"], strict=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "ckpt_epoch": sd.get("epoch"),
        "ckpt_val_total": sd.get("val_total"),
        "split": args.split,
        "num_windows": len(indices),
        "shard_size": args.shard_size,
        "shards": [],
    }

    pending: dict[str, list[np.ndarray]] = {}
    pending_meta: list[dict] = []
    shard_id = 0
    seen = 0

    def append_field(name: str, value: torch.Tensor, dtype: np.dtype | None = None) -> None:
        pending.setdefault(name, []).append(tensor_to_numpy(value, dtype=dtype))

    def flush() -> None:
        nonlocal shard_id, pending, pending_meta
        if not pending_meta:
            return
        arrays = {k: np.concatenate(v, axis=0) for k, v in pending.items()}
        meta_json = np.asarray([json.dumps(m, ensure_ascii=True) for m in pending_meta])
        path = args.out_dir / f"bundle_{shard_id:05d}.npz"
        np.savez_compressed(path, metadata=meta_json, **arrays)
        manifest["shards"].append({
            "file": path.name,
            "count": len(pending_meta),
        })
        shard_id += 1
        pending = {}
        pending_meta = []

    for bi, batch in enumerate(loader):
        bsz = len(batch["clip_id"])
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        action_cond = make_action_condition(action_tgt, action_norm)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        target_rgb = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            kwargs = dict(action_cond=action_cond, context_rgb=context_rgb, pixel=not args.no_rgb, bridging=False)
            out = model(s, c, **kwargs)

        append_field("context_rgb", context_rgb, np.float16)
        append_field("target_rgb", target_rgb, np.float16)
        append_field("task_emb", c, np.float16)
        append_field("action_tgt", action_tgt, np.float32)
        append_field("action_cond", action_cond, np.float32)
        append_field("pred_tokens", out["pred_tokens"], np.float16)
        append_field("depth", out["depth"], np.float16)
        if "motion_hint" in out:
            append_field("motion_hint", out["motion_hint"], np.float16)
        if "contact_hint" in out:
            append_field("contact_hint", out["contact_hint"], np.float16)
        if "rgb" in out:
            append_field("rough_rgb", out["rgb"], np.float16)

        metas = batch_metadata(batch, indices, seen, bsz)
        for i in range(bsz):
            pending_meta.append({k: v[i] for k, v in metas.items()})
        seen += bsz

        if len(pending_meta) >= args.shard_size:
            flush()
        if (bi + 1) % 10 == 0:
            print(f"[cache] {seen}/{len(indices)} windows", flush=True)

    flush()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['shards'])} shards to {args.out_dir}")


if __name__ == "__main__":
    main()
