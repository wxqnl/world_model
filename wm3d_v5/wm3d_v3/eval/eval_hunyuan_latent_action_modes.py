from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.models.hunyuan_latent_adapter import HunyuanLatentAdapter, HunyuanLatentAdapterConfig
from wm3d_v3.training.train import build_datasets, build_model


def _load_stage37(path: Path):
    spec = importlib.util.spec_from_file_location("stage37_train_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import stage37 helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resize_frame(frame: torch.Tensor, hw: tuple[int, int]) -> np.ndarray:
    arr = frame.permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255).round().astype(np.uint8)
    if arr.shape[:2] != hw:
        arr = np.array(Image.fromarray(arr).resize((hw[1], hw[0]), Image.BILINEAR))
    return arr


def _save_grid_gif(path: Path, labels: list[str], videos: list[torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    videos0 = [v[0].float().detach().cpu() for v in videos]
    t = min(int(v.shape[1]) for v in videos0)
    hw = tuple(videos0[0].shape[-2:])
    frames = []
    band_h = 20
    for ti in range(t):
        row = [_resize_frame(v[:, ti], hw) for v in videos0]
        canvas = np.concatenate(row, axis=1)
        # Add a top label band without relying on fonts.
        band = np.zeros((band_h, canvas.shape[1], 3), dtype=np.uint8)
        cell_w = canvas.shape[1] // max(1, len(labels))
        for i, label in enumerate(labels):
            x0 = i * cell_w
            # Simple deterministic hash stripe per label to visually separate panels.
            color = np.array([(sum(label.encode()) * 37) % 255, (len(label) * 53) % 255, 220], dtype=np.uint8)
            band[:, x0 : x0 + min(cell_w, 8)] = color
        frames.append(np.concatenate([band, canvas], axis=0))
    imageio.mimsave(path, frames, duration=0.18)


def _action_mode(action_cond: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "dataset":
        return action_cond
    if mode == "zero":
        return torch.zeros_like(action_cond)
    if mode in {"negreverse", "neg_reverse"}:
        return -torch.flip(action_cond, dims=[1])
    raise ValueError(f"unknown mode {mode!r}")


def _find_indices(ds, clip_ids: list[str], starts: list[int | None], n_clips: int) -> list[int]:
    if not clip_ids:
        out = []
        seen = set()
        for i, (ri, _start) in enumerate(ds.index):
            cid = ds.records[ri].clip_id
            if cid in seen:
                continue
            seen.add(cid)
            out.append(i)
            if len(out) >= n_clips:
                break
        return out
    want = list(zip(clip_ids, starts or [None] * len(clip_ids)))
    found: list[int] = []
    for cid, start_want in want:
        hit = None
        fallback = None
        for i, (ri, start) in enumerate(ds.index):
            if ds.records[ri].clip_id != cid:
                continue
            if fallback is None:
                fallback = i
            if start_want is None or int(start) == int(start_want):
                hit = i
                break
        if hit is None:
            hit = fallback
        if hit is not None:
            found.append(hit)
    return found[:n_clips]


def _motion_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # Inputs are [1,C,T,H,W], values in [0,1]. Ignore optional top label band elsewhere.
    pred_f = pred.float()
    tgt_f = target.float()
    if pred_f.shape[-2:] != tgt_f.shape[-2:]:
        pred_f = torch.nn.functional.interpolate(
            pred_f[0].permute(1, 0, 2, 3),
            size=tgt_f.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3).unsqueeze(0)
    pred_delta = (pred_f[:, :, 1:] - pred_f[:, :, :-1]).abs().mean().item()
    tgt_delta = (tgt_f[:, :, 1:] - tgt_f[:, :, :-1]).abs().mean().item()
    from_first = (pred_f[:, :, 1:] - pred_f[:, :, :1]).abs().mean().item()
    tgt_from_first = (tgt_f[:, :, 1:] - tgt_f[:, :, :1]).abs().mean().item()
    return {
        "motion_consecutive": pred_delta,
        "target_motion_consecutive": tgt_delta,
        "motion_ratio_consecutive": pred_delta / max(tgt_delta, 1e-8),
        "motion_from_first": from_first,
        "target_motion_from_first": tgt_from_first,
        "motion_ratio_from_first": from_first / max(tgt_from_first, 1e-8),
        "l1_to_target": (pred_f - tgt_f).abs().mean().item(),
    }


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--wm_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--stage37", type=Path, default=Path("/data/Minko/world_model/wm3d_v5/scripts/train_hunyuan_latent_adapter_stage37.py"))
    ap.add_argument("--hunyuan_repo", type=Path, default=Path("/data/Minko/external/HunyuanVideo"))
    ap.add_argument("--hunyuan_model_base", type=Path, default=Path("/data/Minko/models/hunyuan_video"))
    ap.add_argument("--vae_precision", default="fp32")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n_clips", type=int, default=4)
    ap.add_argument("--clip_ids", nargs="*", default=[])
    ap.add_argument("--starts", nargs="*", type=int, default=[])
    ap.add_argument("--modes", nargs="*", default=["dataset", "zero", "negreverse"])
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    cfg = yaml.safe_load(args.cfg.read_text())
    train_ds, val_ds = build_datasets(cfg)
    ds = train_ds if args.clip_ids else val_ds
    indices = _find_indices(ds, args.clip_ids, [int(x) for x in args.starts], args.n_clips)
    if not indices:
        raise RuntimeError("no clips found for requested selection")

    stage37 = _load_stage37(args.stage37)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    wm_sd = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    model = build_model(cfg).to(device).eval()
    stage37.load_compatible_state_dict(model, wm_sd["model"])
    stage37.load_action_stats_if_available(model, cfg, 0, device)
    adapter_cfg_dict = ckpt.get("hunyuan_adapter_cfg") or ckpt.get("cfg")
    if adapter_cfg_dict is None:
        raise RuntimeError(f"checkpoint has no adapter cfg: {args.ckpt}")
    adapter_cfg = HunyuanLatentAdapterConfig(**adapter_cfg_dict)
    adapter = HunyuanLatentAdapter(adapter_cfg).to(device).eval()
    adapter_state = ckpt.get("hunyuan_adapter") or ckpt["model"]
    adapter.load_state_dict(adapter_state, strict=True)
    vae = stage37.load_hunyuan_vae(
        SimpleNamespace(
            hunyuan_repo=args.hunyuan_repo,
            hunyuan_model_base=args.hunyuan_model_base,
            vae_precision=args.vae_precision,
        ),
        device,
    )
    if "vae_trainable" in ckpt:
        stage37.load_partial_module_state(vae, ckpt["vae_trainable"], label="vae_trainable")
    scaffold_cfg = ckpt.get("scaffold_cfg", {})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "ckpt": str(args.ckpt),
        "cfg": str(args.cfg),
        "modes": args.modes,
        "clips": [],
    }

    all_mode_videos: dict[str, list[torch.Tensor]] = {m: [] for m in args.modes}
    for local_i, ds_idx in enumerate(indices):
        smp = ds[ds_idx]
        s = smp["s_in"].unsqueeze(0).to(device)
        c = smp["c"].unsqueeze(0).to(device)
        action_tgt = smp["action_tgt"].unsqueeze(0).to(device)
        action_tgt_norm = smp["action_tgt_norm"].unsqueeze(0).to(device)
        context_rgb = smp["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
        rgb_tgt_p = smp["rgb_tgt"].permute(0, 3, 1, 2).unsqueeze(0).to(device)
        base_action = make_action_condition(action_tgt, action_tgt_norm)
        target_video = stage37.target_video_from_batch(context_rgb, rgb_tgt_p)
        target_latents = stage37.encode_hunyuan_latents(vae, target_video.float())

        mode_videos: dict[str, torch.Tensor] = {}
        mode_metrics: dict[str, dict[str, float]] = {}
        for mode in args.modes:
            action_cond = _action_mode(base_action, mode)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wm_out = model(
                    s,
                    c,
                    action_cond=action_cond,
                    context_rgb=context_rgb,
                    pixel=True,
                    return_rgb_features=bool(getattr(adapter_cfg, "use_rgb_features", False)),
                    bridging=False,
                )
                base_latents = stage37.base_latents_from_source(
                    vae,
                    context_rgb,
                    target_video.float(),
                    wm_out,
                    source=str(scaffold_cfg.get("base_latents_source", "context")),
                    rough_prior_weight=float(scaffold_cfg.get("rough_prior_weight", 1.0)),
                    rough_prior_power=float(scaffold_cfg.get("rough_prior_power", 1.0)),
                    rough_prior_floor=float(scaffold_cfg.get("rough_prior_floor", 0.0)),
                    motion_mask_source=str(scaffold_cfg.get("motion_mask_source", "point")),
                    motion_mask_threshold=float(scaffold_cfg.get("motion_mask_threshold", 0.03)),
                    motion_mask_softness=float(scaffold_cfg.get("motion_mask_softness", 0.03)),
                    motion_mask_topk=float(scaffold_cfg.get("motion_mask_topk", 0.0)),
                    motion_mask_spatial_dilate=int(scaffold_cfg.get("motion_mask_spatial_dilate", 0)),
                    motion_mask_temporal_dilate=int(scaffold_cfg.get("motion_mask_temporal_dilate", 0)),
                    motion_mask_floor=float(scaffold_cfg.get("motion_mask_floor", 0.0)),
                )
                pred_latents = stage37.adapter_forward(
                    adapter,
                    wm_out,
                    context_rgb,
                    action_cond,
                    c,
                    target_latents,
                    use_rough=False,
                    base_latents=base_latents,
                )
            pred_video = stage37.decode_hunyuan_latents(vae, pred_latents.float())
            mode_videos[mode] = pred_video.detach().cpu()
            all_mode_videos[mode].append(pred_video.detach().cpu())
            mode_metrics[mode] = _motion_metrics(pred_video.detach().cpu(), target_video.detach().cpu())

        videos = [target_video.detach().cpu()] + [mode_videos[m] for m in args.modes]
        labels = ["target"] + args.modes
        safe_clip = str(smp["clip_id"]).replace("/", "__")
        gif_path = args.out_dir / f"{local_i:02d}_{smp['dataset']}_{safe_clip}_start{smp['start']}_action_modes.gif"
        _save_grid_gif(gif_path, labels, videos)
        clip_record = {
            "dataset": smp["dataset"],
            "clip_id": smp["clip_id"],
            "start": int(smp["start"]),
            "gif": str(gif_path),
            "metrics": mode_metrics,
        }
        for mode in args.modes[1:]:
            clip_record[f"dataset_vs_{mode}_l1"] = (mode_videos["dataset"] - mode_videos[mode]).abs().mean().item()
        summary["clips"].append(clip_record)
        print(json.dumps(clip_record, sort_keys=True), flush=True)

    aggregate: dict[str, float] = {}
    for mode in args.modes:
        vals = [_motion_metrics(v, v.new_zeros(v.shape)) for v in all_mode_videos[mode]]
        # Use per-clip already-computed target-relative metrics for clearer aggregate.
        for key in ["motion_ratio_consecutive", "motion_ratio_from_first", "l1_to_target"]:
            per_clip = [clip["metrics"][mode][key] for clip in summary["clips"]]  # type: ignore[index]
            aggregate[f"{mode}_{key}"] = float(np.mean(per_clip))
    for mode in args.modes[1:]:
        per_clip_l1 = []
        for i in range(len(all_mode_videos["dataset"])):
            per_clip_l1.append((all_mode_videos["dataset"][i] - all_mode_videos[mode][i]).abs().mean().item())
        aggregate[f"dataset_vs_{mode}_l1"] = float(np.mean(per_clip_l1))
    summary["aggregate"] = aggregate
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"summary": str(out_json), "aggregate": aggregate}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
