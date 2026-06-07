"""Generate RGB GIFs for the GT-token context residual decoder."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.models.context_residual_pixel_decoder import (
    ContextResidualPixelDecoder,
    ContextResidualPixelDecoderConfig,
)
from wm3d_v3.training.train_context_residual_pixel import apply_train_ablations


def window_config_from_cfg(cfg: dict) -> WindowConfig:
    data = cfg["data"]
    action_stats = data.get("action_stats")
    return WindowConfig(
        T=data["T"],
        k=data["k"],
        stride=data["stride"],
        cache_root=Path(data["cache_root"]),
        tokens_subdir=data.get("tokens_subdir", "vggt_p256"),
        action_stats=Path(action_stats) if action_stats else None,
    )


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_clips", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    ds = OXEWindowDataset(records, window_config_from_cfg(cfg))
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(ds), generator=gen).tolist()
    n_val = max(1, int(len(ds) * cfg["data"]["val_frac"]))
    val_idx = perm[:n_val]

    picks: list[int] = []
    seen: set[tuple[str, str]] = set()
    for idx in val_idx:
        sample = ds[idx]
        key = (sample["dataset"], sample["clip_id"])
        if key in seen:
            continue
        seen.add(key)
        picks.append(idx)
        if len(picks) >= args.n_clips:
            break

    model = ContextResidualPixelDecoder(ContextResidualPixelDecoderConfig(**cfg["model"])).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"loaded {args.ckpt} epoch={ckpt.get('epoch')} val_total={ckpt.get('val_total')}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, ds_idx in enumerate(picks):
        sample = ds[ds_idx]
        tokens = sample["s_tgt"].unsqueeze(0).to(device)
        context = sample["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
        action = sample["action_tgt"].unsqueeze(0).to(device)
        action_norm = sample["action_tgt_norm"].unsqueeze(0).to(device)
        action_cond = make_action_condition(action, action_norm)
        task = sample["c"].unsqueeze(0).to(device)
        tokens, context, action_cond, task = apply_train_ablations(
            tokens, context, action_cond, task, cfg
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(tokens, context, action_cond=action_cond, task_emb=task)

        pred_np = pred[0].float().cpu().numpy().transpose(0, 2, 3, 1)
        gt_np = sample["rgb_tgt"].numpy()
        ctx_np = sample["rgb_in"][-1].numpy()
        frames = []
        for t in range(pred_np.shape[0]):
            ctx = to_u8(ctx_np)
            pr = to_u8(pred_np[t])
            gt = to_u8(gt_np[t])
            diff = to_u8(np.abs(pred_np[t] - gt_np[t]) * 3.0)
            frames.append(np.concatenate([ctx, pr, gt, diff], axis=1))

        clip_id = sample["clip_id"].replace("/", "__")
        out_path = args.out_dir / f"{out_idx:02d}_{sample['dataset']}_{clip_id}_ctx_pred_gt_diff.gif"
        imageio.mimsave(out_path, frames, duration=0.2, loop=0)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
