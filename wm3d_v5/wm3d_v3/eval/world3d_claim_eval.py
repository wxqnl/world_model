"""Native-3D claim eval for WM3D checkpoints.

This evaluator is meant for reports and checkpoint canaries. It highlights the
claim that WM3D predicts action-conditioned 3D futures, not just rough RGB.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.action_sensitivity import (
    MetricAccumulator,
    compute_counterfactual_metrics,
    make_action_counterfactuals,
    parse_variants,
)
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model, policy_kwargs_from_batch, validate_eval_data_flags
from wm3d_v3.losses import _normalize_depth


def _sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(x.shape[0], -1).mean(dim=1)


def _sample_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        a.float().reshape(a.shape[0], -1),
        b.float().reshape(b.shape[0], -1),
        dim=1,
    )


def motion_mask_from_rgb(
    rgb_tgt: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Return motion mask [B, k, 1, H, W] from target RGB vs latest context RGB."""
    if rgb_tgt.ndim != 5 or context_rgb.ndim != 4:
        raise ValueError(f"expected rgb_tgt [B,k,3,H,W] and context [B,3,H,W], got {tuple(rgb_tgt.shape)} and {tuple(context_rgb.shape)}")
    motion = (rgb_tgt.float() - context_rgb.float().unsqueeze(1)).abs().mean(dim=2, keepdim=True)
    return (motion > float(threshold)).float()


def _resize_mask_to_depth(mask: torch.Tensor, depth_ref: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 5 or depth_ref.ndim != 4:
        raise ValueError(f"expected mask [B,k,1,H,W] and depth [B,k,H,W], got {tuple(mask.shape)} and {tuple(depth_ref.shape)}")
    if mask.shape[0] != depth_ref.shape[0] or mask.shape[1] != depth_ref.shape[1]:
        raise ValueError(f"mask/depth leading dimensions differ: {tuple(mask.shape)} vs {tuple(depth_ref.shape)}")
    if mask.shape[-2:] != depth_ref.shape[-2:]:
        bsz, horizon = mask.shape[:2]
        mask = F.interpolate(
            mask.flatten(0, 1),
            size=depth_ref.shape[-2:],
            mode="nearest",
        ).reshape(bsz, horizon, 1, depth_ref.shape[-2], depth_ref.shape[-1])
    return mask[:, :, 0].float()


def _resize_depth_sequence_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape == ref.shape:
        return x
    if x.ndim != 4 or ref.ndim != 4:
        raise ValueError(f"expected depth [B,T,H,W], got {tuple(x.shape)} and {tuple(ref.shape)}")
    if x.shape[:2] != ref.shape[:2]:
        raise ValueError(f"cannot align depth leading dims: {tuple(x.shape)} vs {tuple(ref.shape)}")
    bsz, horizon = x.shape[:2]
    return F.interpolate(
        x.float().flatten(0, 1)[:, None],
        size=ref.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).reshape(bsz, horizon, ref.shape[-2], ref.shape[-1])


def _resize_context_depth_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or ref.ndim != 4:
        raise ValueError(f"expected context [B,H,W] and depth [B,T,H,W], got {tuple(x.shape)} and {tuple(ref.shape)}")
    if x.shape[0] != ref.shape[0]:
        raise ValueError(f"cannot align context/depth batch dims: {tuple(x.shape)} vs {tuple(ref.shape)}")
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(
        x.float()[:, None],
        size=ref.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _masked_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1.0)
    return (values.float() * mask.float()).reshape(values.shape[0], -1).sum(dim=1) / denom


def _resize_point_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5 or ref.ndim != 5:
        return x
    if x.shape[-3:-1] == ref.shape[-3:-1]:
        return x
    bsz, horizon, h, w, channels = x.shape
    y = x.permute(0, 1, 4, 2, 3).reshape(bsz * horizon, channels, h, w)
    y = F.interpolate(y, size=ref.shape[-3:-1], mode="bilinear", align_corners=False)
    return y.reshape(bsz, horizon, channels, ref.shape[-3], ref.shape[-2]).permute(0, 1, 3, 4, 2).contiguous()


def _resize_conf_like(conf: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim != 5:
        return conf
    if conf.ndim == 5 and conf.shape[-1] == 1:
        conf_hw = conf.squeeze(-1)
    elif conf.ndim == 4:
        conf_hw = conf
    else:
        return conf
    if conf_hw.shape[-2:] == ref.shape[-3:-1]:
        return conf_hw.unsqueeze(-1)
    bsz, horizon, h, w = conf_hw.shape
    y = F.interpolate(
        conf_hw.reshape(bsz * horizon, 1, h, w),
        size=ref.shape[-3:-1],
        mode="bilinear",
        align_corners=False,
    )
    return y.reshape(bsz, horizon, ref.shape[-3], ref.shape[-2], 1)


def _masked_point_l1(pred: torch.Tensor, target: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    target = _resize_point_like(target.to(device=pred.device, dtype=pred.dtype), pred)
    err = (pred.float() - target.float()).abs()
    valid = torch.isfinite(err)
    if conf is not None:
        weight = _resize_conf_like(conf.to(device=pred.device, dtype=pred.dtype).float().clamp_min(0.0), pred)
        if weight.ndim == pred.ndim - 1:
            weight = weight.unsqueeze(-1)
        while weight.ndim < pred.ndim:
            weight = weight.unsqueeze(-1)
        if weight.shape != pred.shape:
            weight = weight.expand_as(pred)
        valid = valid & torch.isfinite(weight) & (weight > 0)
        weight = torch.where(valid, weight, torch.zeros_like(weight))
    else:
        weight = valid.to(err.dtype)
    denom = weight.reshape(weight.shape[0], -1).sum(dim=1).clamp_min(1.0)
    return (err * weight).reshape(err.shape[0], -1).sum(dim=1) / denom


def compute_3d_dynamics_metrics(
    pred_depth: torch.Tensor,
    depth_tgt: torch.Tensor,
    context_depth: torch.Tensor,
    rgb_tgt: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    motion_threshold: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Per-sample 3D dynamics metrics focused on moving regions."""
    pred_depth = _resize_depth_sequence_like(pred_depth.float(), depth_tgt.float())
    context_depth = _resize_context_depth_like(context_depth.float(), pred_depth)
    if pred_depth.shape != depth_tgt.shape:
        raise ValueError(f"pred_depth and depth_tgt must match, got {tuple(pred_depth.shape)} and {tuple(depth_tgt.shape)}")
    if context_depth.shape != pred_depth.shape[:1] + pred_depth.shape[2:]:
        raise ValueError(f"context_depth must be [B,H,W], got {tuple(context_depth.shape)} for pred {tuple(pred_depth.shape)}")

    pred_n = _normalize_depth(pred_depth.float())
    tgt_n = _normalize_depth(depth_tgt.float())
    context_n = _normalize_depth(context_depth.float()).unsqueeze(1)

    abs_err = (pred_n - tgt_n).abs()
    pred_change = pred_n - context_n
    tgt_change = tgt_n - context_n
    change_err = (pred_change - tgt_change).abs()
    change_mag_err = (pred_change.abs() - tgt_change.abs()).abs()
    depth_change_cos = _sample_cos(pred_change, tgt_change)

    motion = motion_mask_from_rgb(rgb_tgt, context_rgb, threshold=motion_threshold)
    motion_depth = _resize_mask_to_depth(motion, pred_depth)
    static_depth = 1.0 - motion_depth
    significant_depth_change = (tgt_change.abs() > float(motion_threshold)).float()
    depth_change_sign_match = (torch.sign(pred_change) == torch.sign(tgt_change)).float()

    if pred_depth.shape[1] > 1:
        temporal_err = (
            (pred_n[:, 1:] - pred_n[:, :-1]) - (tgt_n[:, 1:] - tgt_n[:, :-1])
        ).abs()
        depth_temporal_delta_l1 = _sample_mean(temporal_err)
        temporal_motion_depth = motion_depth[:, 1:]
        motion_region_depth_temporal_delta_l1 = _masked_sample_mean(temporal_err, temporal_motion_depth)
    else:
        depth_temporal_delta_l1 = pred_depth.new_zeros(pred_depth.shape[0], dtype=torch.float32)
        motion_region_depth_temporal_delta_l1 = depth_temporal_delta_l1

    return {
        # Future-depth accuracy.
        "global_depth_l1": _sample_mean(abs_err),
        "depth_future_l1": _sample_mean(abs_err),
        # Context-to-future 3D dynamics. These do not just restate RGB quality.
        "depth_change_vector_l1": _sample_mean(change_err),
        "depth_change_l1": _sample_mean(change_mag_err),
        "depth_change_cos": depth_change_cos,
        "depth_change_sign_acc": _masked_sample_mean(depth_change_sign_match, significant_depth_change),
        "significant_depth_change_frac": significant_depth_change.float().mean(dim=(1, 2, 3)),
        "depth_temporal_delta_l1": depth_temporal_delta_l1,
        "motion_region_depth_temporal_delta_l1": motion_region_depth_temporal_delta_l1,
        "motion_region_depth_l1": _masked_sample_mean(abs_err, motion_depth),
        "motion_region_depth_change_vector_l1": _masked_sample_mean(change_err, motion_depth),
        "motion_region_depth_change_l1": _masked_sample_mean(change_mag_err, motion_depth),
        "motion_region_depth_delta_l1": _masked_sample_mean(change_mag_err, motion_depth),
        "static_region_depth_l1": _masked_sample_mean(abs_err, static_depth),
        "static_region_depth_change_vector_l1": _masked_sample_mean(change_err, static_depth),
        "static_region_depth_change_l1": _masked_sample_mean(change_mag_err, static_depth),
        "motion_frac": motion.float().mean(dim=(1, 2, 3, 4)),
    }


def summarize_counterfactual_showcase(
    variant_metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Condense per-variant action sensitivity metrics into report-facing claims."""
    if not variant_metrics:
        return {
            "claim": "real_action_beats_counterfactual_action",
            "status": "missing_counterfactual_metrics",
        }

    def mean_present(key: str) -> float | None:
        vals = [float(m[key]) for m in variant_metrics.values() if key in m]
        if not vals:
            return None
        return sum(vals) / len(vals)

    depth_gap_items = [
        (name, float(metrics["depth_gt_l1_gap"]))
        for name, metrics in variant_metrics.items()
        if "depth_gt_l1_gap" in metrics
    ]
    strongest_depth_variant = max(depth_gap_items, key=lambda item: item[1])[0] if depth_gap_items else None

    return {
        "claim": "real_action_beats_counterfactual_action",
        "status": "computed",
        "mean_token_win_rate": mean_present("pred_tokens_gt_mse_acc"),
        "mean_depth_win_rate": mean_present("depth_gt_l1_acc"),
        "mean_depth_change_win_rate": mean_present("depth_change_gt_l1_acc"),
        "mean_motion_region_depth_win_rate": mean_present("motion_region_depth_gt_l1_acc"),
        "mean_motion_win_rate": mean_present("motion_hint_gt_l1_acc"),
        "mean_token_gt_mse_gap": mean_present("pred_tokens_gt_mse_gap"),
        "mean_depth_gt_l1_gap": mean_present("depth_gt_l1_gap"),
        "mean_depth_change_gt_l1_gap": mean_present("depth_change_gt_l1_gap"),
        "mean_motion_region_depth_gt_l1_gap": mean_present("motion_region_depth_gt_l1_gap"),
        "strongest_depth_variant": strongest_depth_variant,
    }


def _to_device(batch: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    return batch[key].to(device, non_blocking=True)


def _checkpoint_meta(sd: Mapping[str, Any]) -> dict[str, Any]:
    val_total = sd.get("val_total")
    return {
        "epoch": sd.get("epoch"),
        "step": sd.get("step"),
        "val_total": float(val_total) if isinstance(val_total, (float, int)) else None,
    }


def _dataset_name_for_index(dataset: Any, idx: int) -> str:
    if isinstance(dataset, Subset):
        return _dataset_name_for_index(dataset.dataset, int(dataset.indices[idx]))
    record_idx, _ = dataset.index[int(idx)]
    return str(dataset.records[record_idx].dataset)


def _clip_id_for_index(dataset: Any, idx: int) -> str:
    if isinstance(dataset, Subset):
        return _clip_id_for_index(dataset.dataset, int(dataset.indices[idx]))
    record_idx, _ = dataset.index[int(idx)]
    return str(dataset.records[record_idx].clip_id)


def _parse_dataset_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    out = {item.strip() for item in value.split(",") if item.strip()}
    return out or None


def _build_eval_subset(
    dataset: Any,
    *,
    balanced: bool,
    batch_size: int,
    max_batches_per_dataset: int,
    seed: int,
    dataset_filter: set[str] | None,
) -> tuple[Any, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        name = _dataset_name_for_index(dataset, idx)
        if dataset_filter is not None and name not in dataset_filter:
            continue
        groups[name].append(idx)

    info: dict[str, Any] = {
        "available_windows_by_dataset": {name: len(ixs) for name, ixs in sorted(groups.items())},
    }
    if not balanced and dataset_filter is None:
        return dataset, info

    selected: list[int] = []
    gen = torch.Generator().manual_seed(int(seed))
    take_per_dataset = None
    if balanced:
        take_per_dataset = max(1, int(max_batches_per_dataset)) * max(1, int(batch_size))
    for name in sorted(groups):
        ixs = groups[name]
        if balanced:
            perm = torch.randperm(len(ixs), generator=gen).tolist()
            chosen = [ixs[i] for i in perm[:take_per_dataset]]
        else:
            chosen = ixs
        selected.extend(chosen)
    info["selected_windows_by_dataset"] = {
        name: min(len(ixs), take_per_dataset if take_per_dataset is not None else len(ixs))
        for name, ixs in sorted(groups.items())
    }
    info["selected_total_windows"] = len(selected)
    return Subset(dataset, selected), info


def _world_core_metrics(out: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pred_tokens = out["pred_tokens"].float()
    s_tgt = batch["s_tgt"].to(pred_tokens.device, non_blocking=True).float()
    action_tgt = batch["action_tgt"].to(pred_tokens.device, non_blocking=True).float()
    pose_mse = (out["pose"].float() - action_tgt[..., :6]).pow(2).mean(dim=(1, 2))
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_acc = ((out["gripper_logit"].float().sigmoid() > 0.5).float() == grip_tgt).float().mean(dim=1)
    metrics = {
        "token_mse": _sample_mean((pred_tokens - s_tgt).pow(2)),
        "token_cos": _sample_cos(pred_tokens, s_tgt),
        "pose_mse": pose_mse,
        "grip_acc": grip_acc,
    }
    if "point" in out and "point_tgt" in batch:
        point_tgt = batch["point_tgt"].to(pred_tokens.device, non_blocking=True).float()
        point_conf = batch.get("point_conf_tgt")
        metrics["world_point_l1"] = _masked_point_l1(out["point"].float(), point_tgt, point_conf)
    if "pose_geom" in out and "pose_geom_tgt" in batch:
        pose_geom_tgt = batch["pose_geom_tgt"].to(pred_tokens.device, non_blocking=True).float()
        horizon = min(out["pose_geom"].shape[1], pose_geom_tgt.shape[1])
        metrics["camera_pose_enc_mse"] = (
            out["pose_geom"][:, :horizon].float() - pose_geom_tgt[:, :horizon]
        ).pow(2).mean(dim=(1, 2))
    return metrics


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def _as_uint8_rgb(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().float().cpu().numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.nan_to_num(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    return arr


def _depth_to_rgb(depth: torch.Tensor | np.ndarray) -> np.ndarray:
    import matplotlib

    arr = depth.detach().float().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
    arr = np.nan_to_num(arr.astype(np.float32))
    lo, hi = np.percentile(arr, [2.0, 98.0])
    arr = (arr - lo) / max(1e-6, hi - lo)
    cmap = matplotlib.colormaps["viridis"]
    return (cmap(np.clip(arr, 0.0, 1.0))[..., :3] * 255).round().astype(np.uint8)


def _heat_to_rgb(values: torch.Tensor | np.ndarray) -> np.ndarray:
    import matplotlib

    arr = values.detach().float().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)
    arr = np.nan_to_num(np.abs(arr).astype(np.float32))
    hi = np.percentile(arr, 98.0)
    arr = arr / max(1e-6, hi)
    cmap = matplotlib.colormaps["magma"]
    return (cmap(np.clip(arr, 0.0, 1.0))[..., :3] * 255).round().astype(np.uint8)


def _tile(arr: np.ndarray, label: str, *, size: int = 160) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(_as_uint8_rgb(arr)).resize((size, size))
    canvas = Image.new("RGB", (size, size + 20), "white")
    canvas.paste(img, (0, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label[:22], fill=(0, 0, 0))
    return np.asarray(canvas)


def _concat_rows(rows: list[list[np.ndarray]]) -> np.ndarray:
    row_imgs = [np.concatenate(row, axis=1) for row in rows if row]
    width = max(row.shape[1] for row in row_imgs)
    padded = []
    for row in row_imgs:
        if row.shape[1] < width:
            pad = np.full((row.shape[0], width - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
    return np.concatenate(padded, axis=0)


def _save_counterfactual_viz(
    *,
    out_dir: Path,
    batch: Mapping[str, Any],
    real_out: Mapping[str, torch.Tensor],
    variant_outs: Mapping[str, Mapping[str, torch.Tensor]],
    sample_idx: int,
    variants: list[str],
) -> Path:
    import imageio.v2 as imageio
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(batch["dataset"][sample_idx])
    clip_id = str(batch["clip_id"][sample_idx])
    start = int(batch["start"][sample_idx])
    safe = _safe_name(f"{dataset}_{clip_id}_s{start}")
    path = out_dir / f"{safe}_native3d_counterfactual.gif"

    context_rgb = batch["rgb_in"][sample_idx, -1].permute(2, 0, 1)
    rgb_tgt = batch["rgb_tgt"][sample_idx].permute(0, 3, 1, 2)
    depth_tgt = batch["depth_tgt"][sample_idx]
    real_depth = real_out["depth"][sample_idx].float()
    real_rgb = real_out.get("rgb")
    real_motion = real_out.get("motion_hint")
    motion_tgt = motion_mask_from_rgb(
        rgb_tgt.unsqueeze(0).to(real_depth.device),
        context_rgb.unsqueeze(0).to(real_depth.device),
    )[0]
    variant_depths = {
        name: out["depth"][sample_idx].float()
        for name, out in variant_outs.items()
        if "depth" in out
    }
    show_variants = [name for name in variants if name in variant_depths][:3]

    frames: list[np.ndarray] = []
    horizon = int(real_depth.shape[0])
    for t in range(horizon):
        row_rgb = [
            _tile(context_rgb, "context RGB"),
            _tile(rgb_tgt[t], f"GT RGB t{t}"),
        ]
        if real_rgb is not None:
            row_rgb.append(_tile(real_rgb[sample_idx, t].float(), "WM RGB real"))

        row_depth = [
            _tile(_depth_to_rgb(depth_tgt[t]), "GT depth"),
            _tile(_depth_to_rgb(real_depth[t]), "WM depth real"),
        ]
        for name in show_variants:
            row_depth.append(_tile(_depth_to_rgb(variant_depths[name][t]), f"depth {name}"))

        row_diff = [_tile(motion_tgt[t], "GT motion")]
        if real_motion is not None:
            row_diff.append(_tile(real_motion[sample_idx, t].float().clamp(0, 1), "WM motion"))
        for name in show_variants[:2]:
            diff = (real_depth[t] - variant_depths[name][t]).abs()
            row_diff.append(_tile(_heat_to_rgb(diff), f"|real-{name}| depth"))

        frame = _concat_rows([row_rgb, row_depth, row_diff])
        if frame.shape[0] > 0:
            frames.append(frame)

    if not frames:
        raise RuntimeError("no frames generated for native3d counterfactual visualization")
    imageio.mimsave(path, frames, duration=0.22, loop=0)
    meta = {
        "dataset": dataset,
        "clip_id": clip_id,
        "start": start,
        "variants": show_variants,
        "layout": [
            "context RGB | GT RGB | WM RGB real",
            "GT depth | WM depth real | WM depth under counterfactual actions",
            "GT motion | WM motion | real-vs-counterfactual depth difference heatmaps",
        ],
    }
    (out_dir / f"{safe}_native3d_counterfactual.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.cfg.read_text())
    validate_eval_data_flags(cfg, rgb_metrics=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    use_autocast = device.type == "cuda"
    variants = parse_variants(args.variants)

    records = read_manifest(cfg["data"]["manifest"])
    ds = build_dataset_for_split(records, cfg, split=args.split)
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["train"]["batch_size_per_gpu"])
    dataset_filter = _parse_dataset_filter(args.datasets)
    ds, sampling_info = _build_eval_subset(
        ds,
        balanced=bool(args.balanced_datasets),
        batch_size=batch_size,
        max_batches_per_dataset=int(args.max_batches_per_dataset or args.max_batches or 1),
        seed=int(args.seed),
        dataset_filter=dataset_filter,
    )
    if len(ds) == 0:
        raise RuntimeError(f"world3d eval dataset is empty after filter={dataset_filter}")
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    world3d_acc: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    cf_acc: dict[str, dict[str, MetricAccumulator]] = defaultdict(lambda: defaultdict(MetricAccumulator))
    counts: dict[str, int] = defaultdict(int)
    enable_context_pixel = cfg["model"].get("enable_context_pixel", False)
    variant_gen = torch.Generator(device=device).manual_seed(args.seed) if device.type == "cuda" else torch.Generator().manual_seed(args.seed)
    stop_after_batches = 0 if args.balanced_datasets else int(args.max_batches)
    viz_saved = 0
    viz_saved_by_dataset: dict[str, int] = defaultdict(int)

    for bi, batch in enumerate(loader):
        if stop_after_batches and bi >= stop_after_batches:
            break
        s = _to_device(batch, "s_in", device)
        c = _to_device(batch, "c", device)
        action_tgt = _to_device(batch, "action_tgt", device)
        action_tgt_norm = _to_device(batch, "action_tgt_norm", device)
        rgb_tgt = _to_device(batch, "rgb_tgt", device).permute(0, 1, 4, 2, 3).contiguous()
        context_rgb = _to_device(batch, "rgb_in", device)[:, -1].permute(0, 3, 1, 2).contiguous()
        context_depth = _to_device(batch, "depth_in", device)[:, -1].contiguous()
        depth_tgt = _to_device(batch, "depth_tgt", device)
        action_conds = make_action_counterfactuals(
            action_tgt,
            action_tgt_norm,
            variants=variants,
            generator=variant_gen,
        )
        kwargs: dict[str, Any] = {
            "pixel": not args.skip_pixel,
            "bridging": False,
        }
        kwargs.update(policy_kwargs_from_batch(batch, device))
        if enable_context_pixel:
            kwargs["context_rgb"] = context_rgb
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            real_out = model(s, c, action_cond=action_conds["real"], **kwargs)

        if bool(cfg["data"].get("require_geom_extra", False)):
            missing_targets = [key for key in ("point_tgt", "pose_geom_tgt") if key not in batch]
            missing_outputs = [key for key in ("point", "pose_geom") if key not in real_out]
            if missing_targets or missing_outputs:
                raise RuntimeError(
                    "require_geom_extra=True but native3d point/pose evidence is unavailable: "
                    f"missing_targets={missing_targets} missing_outputs={missing_outputs}"
                )
        core_metrics = _world_core_metrics(real_out, batch)
        depth_metrics = compute_3d_dynamics_metrics(
            real_out["depth"].float(),
            depth_tgt.float(),
            context_depth.float(),
            rgb_tgt.float(),
            context_rgb.float(),
            motion_threshold=args.motion_threshold,
        )
        combined_metrics = {**core_metrics, **depth_metrics}
        dataset_names = list(batch["dataset"])
        world3d_acc["ALL"].update(combined_metrics)
        for d in sorted(set(dataset_names)):
            mask = torch.tensor([name == d for name in dataset_names], device=device)
            world3d_acc[d].update(combined_metrics, mask=mask)
        counts["ALL"] += s.shape[0]
        for d in dataset_names:
            counts[d] += 1

        targets = {
            "s_tgt": _to_device(batch, "s_tgt", device),
            "depth_tgt": depth_tgt,
            "context_depth": context_depth,
            "motion_tgt": motion_mask_from_rgb(rgb_tgt, context_rgb, threshold=args.motion_threshold),
        }
        variant_outs: dict[str, Mapping[str, torch.Tensor]] = {}
        for variant in variants:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                variant_out = model(s, c, action_cond=action_conds[variant], **kwargs)
            if args.viz_dir and viz_saved < int(args.n_viz):
                variant_outs[variant] = variant_out
            metrics = compute_counterfactual_metrics(real_out, variant_out, targets)
            cf_acc["ALL"][variant].update(metrics)
            for d in sorted(set(dataset_names)):
                mask = torch.tensor([name == d for name in dataset_names], device=device)
                cf_acc[d][variant].update(metrics, mask=mask)

        if args.viz_dir and viz_saved < int(args.n_viz):
            batch_motion = combined_metrics["motion_frac"].detach().float().cpu()
            order = torch.argsort(batch_motion, descending=True).tolist()
            for sample_idx in order:
                if viz_saved >= int(args.n_viz):
                    break
                sample_dataset = str(batch["dataset"][sample_idx])
                if int(args.viz_per_dataset) > 0 and viz_saved_by_dataset[sample_dataset] >= int(args.viz_per_dataset):
                    continue
                if float(batch_motion[sample_idx]) < float(args.viz_min_motion_frac):
                    continue
                path = _save_counterfactual_viz(
                    out_dir=args.viz_dir,
                    batch=batch,
                    real_out=real_out,
                    variant_outs=variant_outs,
                    sample_idx=int(sample_idx),
                    variants=variants,
                )
                print(f"wrote_viz {path}")
                viz_saved += 1
                viz_saved_by_dataset[sample_dataset] += 1

        if (bi + 1) % max(1, args.log_every) == 0:
            m = world3d_acc["ALL"].means()
            print(
                f"[{bi + 1}/{len(loader)}] token_mse={m.get('token_mse', 0.0):.4f} "
                f"depth_change={m.get('depth_change_l1', 0.0):.4f} "
                f"motion_depth={m.get('motion_region_depth_l1', 0.0):.4f}"
            )

    metrics: dict[str, Any] = {}
    for dataset in sorted(world3d_acc):
        cf_metrics = {
            variant: acc.means()
            for variant, acc in sorted(cf_acc.get(dataset, {}).items())
        }
        metrics[dataset] = {
            "world3d": world3d_acc[dataset].means(),
            "counterfactual": cf_metrics,
            "counterfactual_showcase": summarize_counterfactual_showcase(cf_metrics),
        }

    all_showcase = metrics.get("ALL", {}).get("counterfactual_showcase", {})
    all_world3d = metrics.get("ALL", {}).get("world3d", {})
    report = {
        "mode": {
            "split": args.split,
            "max_batches": args.max_batches,
            "max_batches_per_dataset": args.max_batches_per_dataset,
            "balanced_datasets": bool(args.balanced_datasets),
            "datasets": sorted(dataset_filter) if dataset_filter else None,
            "batch_size": args.batch_size,
            "variants": variants,
            "pixel_active": not args.skip_pixel,
            "motion_threshold": args.motion_threshold,
            "viz_dir": str(args.viz_dir) if args.viz_dir else None,
            "n_viz": int(args.n_viz),
            "viz_per_dataset": int(args.viz_per_dataset),
        },
        "sampling": sampling_info,
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "checkpoint": _checkpoint_meta(sd),
        "counts": dict(counts),
        "core_contribution": {
            "name": "Action-conditioned native-3D future prediction",
            "stage": "Run1 world pretraining" if not cfg["model"].get("enable_action_proposer", False) else "world+action scaffold",
            "evidence_depth_only": {
                "motion_region_depth_l1": all_world3d.get("motion_region_depth_l1"),
                "motion_region_depth_change_l1": all_world3d.get("motion_region_depth_change_l1"),
                "motion_region_depth_temporal_delta_l1": all_world3d.get("motion_region_depth_temporal_delta_l1"),
                "depth_future_l1": all_world3d.get("depth_future_l1"),
                "depth_change_l1": all_world3d.get("depth_change_l1"),
                "depth_change_vector_l1_diagnostic": all_world3d.get("depth_change_vector_l1"),
                "depth_change_cos": all_world3d.get("depth_change_cos"),
                "depth_change_sign_acc": all_world3d.get("depth_change_sign_acc"),
                "depth_temporal_delta_l1": all_world3d.get("depth_temporal_delta_l1"),
                "mean_depth_win_rate_vs_counterfactual": all_showcase.get("mean_depth_win_rate"),
                "mean_depth_change_win_rate_vs_counterfactual": all_showcase.get("mean_depth_change_win_rate"),
                "mean_motion_region_depth_win_rate_vs_counterfactual": all_showcase.get("mean_motion_region_depth_win_rate"),
                "strongest_depth_counterfactual": all_showcase.get("strongest_depth_variant"),
            },
            "evidence_point_pose": {
                "available": all_world3d.get("world_point_l1") is not None and all_world3d.get("camera_pose_enc_mse") is not None,
                "world_point_l1": all_world3d.get("world_point_l1"),
                "camera_pose_enc_mse": all_world3d.get("camera_pose_enc_mse"),
            },
            "evidence": {
                "mean_token_win_rate_vs_counterfactual": all_showcase.get("mean_token_win_rate"),
                "mean_motion_win_rate_vs_counterfactual": all_showcase.get("mean_motion_win_rate"),
                "depth_only_evidence_available": True,
                "point_pose_evidence_available": all_world3d.get("world_point_l1") is not None and all_world3d.get("camera_pose_enc_mse") is not None,
            },
        },
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    print(json.dumps(report["core_contribution"], indent=2, sort_keys=True))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=("train", "val", "all"))
    ap.add_argument("--max_batches", type=int, default=32)
    ap.add_argument(
        "--balanced_datasets",
        action="store_true",
        help="sample up to --max_batches_per_dataset batches from each dataset instead of taking the first val batches",
    )
    ap.add_argument("--max_batches_per_dataset", type=int, default=0)
    ap.add_argument("--datasets", default="", help="optional comma-separated dataset filter, e.g. bridge,droid")
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--motion_threshold", type=float, default=0.03)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--skip_pixel", action="store_true")
    ap.add_argument("--viz_dir", type=Path, default=None)
    ap.add_argument("--n_viz", type=int, default=0)
    ap.add_argument("--viz_per_dataset", type=int, default=1)
    ap.add_argument("--viz_min_motion_frac", type=float, default=0.05)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["zero", "sign_flip", "grip_toggle"],
        help="action counterfactuals; comma-separated or space-separated",
    )
    run_eval(ap.parse_args())


if __name__ == "__main__":
    main()
