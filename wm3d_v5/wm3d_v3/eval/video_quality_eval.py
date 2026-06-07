"""Video-quality eval for WM3D future RGB generation.

The report-facing metrics here complement training losses with common world
model video metrics: PSNR, SSIM, LPIPS, motion-region RGB L1, and a practical
R3D-18/Kinetics Fréchet Video Distance proxy.
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
from torch import nn
from torch.utils.data import DataLoader

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.run_eval import (
    build_dataset_for_split,
    build_model,
    policy_kwargs_from_batch,
    validate_eval_data_flags,
)
from wm3d_v3.eval.world3d_claim_eval import (
    _build_eval_subset,
    _checkpoint_meta,
    _parse_dataset_filter,
)


def psnr_video(pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return per-sample PSNR for videos in [0, 1], shaped [B, T, C, H, W]."""
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError(f"expected matching [B,T,C,H,W] tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    mse = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3, 4))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(float(eps)))


def _gaussian_window(
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.reshape(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_video(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-sample local SSIM averaged over frames/channels/spatial dims."""
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError(f"expected matching [B,T,C,H,W] tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    bsz, horizon, channels = pred.shape[:3]
    x = pred.float().flatten(0, 1)
    y = target.float().flatten(0, 1)
    kernel = _gaussian_window(
        channels,
        device=x.device,
        dtype=x.dtype,
        window_size=window_size,
        sigma=sigma,
    )
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_xy
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return score.mean(dim=(1, 2, 3)).reshape(bsz, horizon).mean(dim=1).clamp(-1.0, 1.0)


def motion_region_rgb_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    context_rgb: torch.Tensor,
    *,
    threshold: float = 0.03,
) -> torch.Tensor:
    """Return per-sample RGB L1 on pixels that changed from latest context RGB."""
    if pred.shape != target.shape or pred.ndim != 5:
        raise ValueError(f"expected matching [B,T,C,H,W] tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    if context_rgb.ndim != 4 or context_rgb.shape[0] != target.shape[0] or context_rgb.shape[1:] != target.shape[2:]:
        raise ValueError(f"expected context_rgb [B,C,H,W], got {tuple(context_rgb.shape)} for target {tuple(target.shape)}")
    motion = (target.float() - context_rgb.float().unsqueeze(1)).abs().mean(dim=2, keepdim=True)
    motion_mask = (motion > float(threshold)).float()
    denom = (motion_mask.sum(dim=(1, 2, 3, 4)) * target.shape[2]).clamp_min(1.0)
    return ((pred.float() - target.float()).abs() * motion_mask).sum(dim=(1, 2, 3, 4)) / denom


def ensure_min_video_frames(videos: torch.Tensor, *, min_frames: int = 16) -> torch.Tensor:
    """Pad [B,T,C,H,W] videos by repeating the last frame until min_frames."""
    if videos.ndim != 5:
        raise ValueError(f"expected videos [B,T,C,H,W], got {tuple(videos.shape)}")
    if videos.shape[1] >= int(min_frames):
        return videos
    pad_count = int(min_frames) - int(videos.shape[1])
    tail = videos[:, -1:].expand(videos.shape[0], pad_count, *videos.shape[2:])
    return torch.cat([videos, tail], dim=1)


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray, *, eps: float = 1e-6) -> float:
    """Compute Fréchet distance between two feature distributions."""
    from scipy import linalg

    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"expected [N,D] and [M,D] features, got {a.shape} and {b.shape}")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("Fréchet distance needs at least two samples per distribution")

    mu_a = np.mean(a, axis=0)
    mu_b = np.mean(b, axis=0)
    sigma_a = np.cov(a, rowvar=False)
    sigma_b = np.cov(b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_a.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma_a + offset).dot(sigma_b + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    distance = float(diff.dot(diff) + np.trace(sigma_a + sigma_b - 2.0 * covmean))
    return 0.0 if abs(distance) < 1e-8 else max(distance, 0.0)


class R3D18FeatureExtractor(nn.Module):
    """Kinetics-pretrained R3D-18 feature extractor for an FVD-style metric."""

    def __init__(self, device: torch.device, *, image_size: int = 112) -> None:
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = nn.Identity()
        self.model = model.to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.image_size = int(image_size)
        self.register_buffer("mean", torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1))

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f"expected videos [B,T,C,H,W], got {tuple(videos.shape)}")
        x = videos.float().clamp(0.0, 1.0).permute(0, 2, 1, 3, 4).contiguous()
        bsz, channels, horizon, height, width = x.shape
        x_flat = x.permute(0, 2, 1, 3, 4).reshape(bsz * horizon, channels, height, width)
        x_flat = F.interpolate(
            x_flat,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x_flat.reshape(bsz, horizon, channels, self.image_size, self.image_size).permute(0, 2, 1, 3, 4)
        x = (x - self.mean.to(x.device, x.dtype)) / self.std.to(x.device, x.dtype)
        return self.model(x).float()


class I3DTorchscriptFeatureExtractor(nn.Module):
    """TorchScript I3D/Kinetics feature extractor for FVD-compatible reporting."""

    def __init__(self, device: torch.device, *, model_path: Path | None = None) -> None:
        super().__init__()
        if model_path is None:
            from huggingface_hub import hf_hub_download

            model_path = Path(
                hf_hub_download(
                    repo_id="flateon/FVD-I3D-torchscript",
                    filename="i3d_torchscript.pt",
                )
            )
        model = torch.jit.load(str(model_path), map_location=device)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.ndim != 5:
            raise ValueError(f"expected videos [B,T,C,H,W], got {tuple(videos.shape)}")
        videos = ensure_min_video_frames(videos, min_frames=32)
        x = videos.float().clamp(0.0, 1.0).permute(0, 2, 1, 3, 4).contiguous()
        try:
            features = self.model(x, rescale=True, resize=True, return_features=True)
        except TypeError:
            features = self.model(x, True, True, True)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.float()


def fvd_protocol_name(backend: str) -> str:
    if backend == "r3d18":
        return "r3d18_kinetics400_features_frechet_distance"
    if backend == "i3d_torchscript":
        return "i3d_torchscript_kinetics400_features_frechet_distance"
    raise ValueError(f"unknown fvd backend: {backend}")


def build_feature_extractor(
    backend: str,
    device: torch.device,
    *,
    image_size: int = 112,
    i3d_model_path: Path | None = None,
) -> nn.Module:
    if backend == "r3d18":
        return R3D18FeatureExtractor(device, image_size=image_size)
    if backend == "i3d_torchscript":
        return I3DTorchscriptFeatureExtractor(device, model_path=i3d_model_path)
    raise ValueError(f"unknown fvd backend: {backend}")


def batch_sample_rows(
    batch: Mapping[str, Any],
    *,
    batch_index: int,
    global_offset: int,
) -> list[dict[str, Any]]:
    datasets = list(batch["dataset"])
    clip_ids = list(batch["clip_id"])
    starts = batch["start"]
    rows: list[dict[str, Any]] = []
    for i, dataset in enumerate(datasets):
        start = starts[i]
        if isinstance(start, torch.Tensor):
            start = int(start.detach().cpu())
        rows.append(
            {
                "sample_index": int(global_offset + i),
                "batch_index": int(batch_index),
                "batch_sample_index": int(i),
                "dataset": str(dataset),
                "clip_id": str(clip_ids[i]),
                "start": int(start),
            }
        )
    return rows


def _append_sample_metrics(
    by_ds: dict[str, dict[str, float]],
    counts: dict[str, int],
    dataset_names: list[str],
    metrics: Mapping[str, torch.Tensor],
) -> None:
    for i, dataset_name in enumerate(dataset_names):
        keys = [dataset_name, "ALL"]
        for key in keys:
            for metric_name, values in metrics.items():
                by_ds[key][metric_name] += float(values[i].detach().cpu())
            counts[key] += 1


def _collect_features(
    groups: dict[str, dict[str, list[np.ndarray]]],
    dataset_names: list[str],
    *,
    target: torch.Tensor,
    model_pred: torch.Tensor,
    last_frame: torch.Tensor,
    extractor: nn.Module,
) -> None:
    target_features = extractor(target).detach().cpu().numpy()
    model_features = extractor(model_pred).detach().cpu().numpy()
    last_features = extractor(last_frame).detach().cpu().numpy()
    for i, dataset_name in enumerate(dataset_names):
        for key in (dataset_name, "ALL"):
            groups[key]["target"].append(target_features[i])
            groups[key]["model"].append(model_features[i])
            groups[key]["last_frame"].append(last_features[i])


def _summarize_features(groups: Mapping[str, Mapping[str, list[np.ndarray]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for dataset_name, features in groups.items():
        target = np.stack(features["target"], axis=0) if features["target"] else None
        model = np.stack(features["model"], axis=0) if features["model"] else None
        last_frame = np.stack(features["last_frame"], axis=0) if features["last_frame"] else None
        item: dict[str, Any] = {
            "num_videos": 0 if target is None else int(target.shape[0]),
            "model_fvd": None,
            "last_frame_fvd": None,
        }
        if target is not None and model is not None and last_frame is not None and target.shape[0] >= 2:
            item["model_fvd"] = frechet_distance(model, target)
            item["last_frame_fvd"] = frechet_distance(last_frame, target)
        out[dataset_name] = item
    return out


def _checkpoint_report(path: Path, sd: Mapping[str, Any]) -> dict[str, Any]:
    report = _checkpoint_meta(sd)
    report["path"] = str(path)
    return report


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.cfg.read_text())
    validate_eval_data_flags(cfg, rgb_metrics=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    records = read_manifest(cfg["data"]["manifest"])
    dataset = build_dataset_for_split(records, cfg, split=args.split)
    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else int(cfg["train"]["batch_size_per_gpu"])
    dataset_filter = _parse_dataset_filter(args.dataset_filter)
    dataset, subset_info = _build_eval_subset(
        dataset,
        balanced=bool(args.balanced_datasets),
        batch_size=batch_size,
        max_batches_per_dataset=int(args.max_batches_per_dataset),
        seed=int(args.sample_seed),
        dataset_filter=dataset_filter,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    lpips_model = None
    if args.include_lpips:
        import lpips

        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()
        for param in lpips_model.parameters():
            param.requires_grad = False

    fvd_extractor = (
        build_feature_extractor(
            args.fvd_backend,
            device,
            image_size=int(args.fvd_image_size),
            i3d_model_path=args.i3d_model_path,
        )
        if args.include_fvd
        else None
    )

    by_ds: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    fvd_groups: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    sample_rows: list[dict[str, Any]] = []

    for batch_idx, batch in enumerate(loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break

        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        action_cond = make_action_condition(action_tgt, action_tgt_norm)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        context_rgb_seq = None
        if args.fvd_context_frames > 0:
            context_rgb_seq = (
                batch["rgb_in"][:, -int(args.fvd_context_frames):]
                .to(device, non_blocking=True)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
                .float()
            )
        rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous().float()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            kwargs = dict(action_cond=action_cond, pixel=True, bridging=False)
            kwargs.update(policy_kwargs_from_batch(batch, device))
            if cfg["model"].get("enable_context_pixel", False):
                kwargs["context_rgb"] = context_rgb
            out = model(s, c, **kwargs)
        if "rgb" not in out:
            raise RuntimeError("model output has no 'rgb'; this evaluator requires enable_pixel=True")

        rgb_pred = out["rgb"].float().clamp(0.0, 1.0)
        last_frame = context_rgb.float().unsqueeze(1).expand_as(rgb_tgt).contiguous()
        dataset_names = [str(name) for name in batch["dataset"]]
        batch_rows = batch_sample_rows(batch, batch_index=batch_idx, global_offset=len(sample_rows))
        sample_rows.extend(batch_rows)

        metrics: dict[str, torch.Tensor] = {
            "model_psnr": psnr_video(rgb_pred, rgb_tgt),
            "last_frame_psnr": psnr_video(last_frame, rgb_tgt),
            "model_ssim": ssim_video(rgb_pred, rgb_tgt),
            "last_frame_ssim": ssim_video(last_frame, rgb_tgt),
            "model_rgb_l1": (rgb_pred - rgb_tgt).abs().mean(dim=(1, 2, 3, 4)),
            "last_frame_rgb_l1": (last_frame - rgb_tgt).abs().mean(dim=(1, 2, 3, 4)),
            "model_motion_rgb_l1": motion_region_rgb_l1(rgb_pred, rgb_tgt, context_rgb, threshold=float(args.motion_threshold)),
            "last_frame_motion_rgb_l1": motion_region_rgb_l1(last_frame, rgb_tgt, context_rgb, threshold=float(args.motion_threshold)),
        }

        if lpips_model is not None:
            with torch.autocast(device_type=device.type, enabled=False):
                pred_lp = rgb_pred.flatten(0, 1) * 2.0 - 1.0
                tgt_lp = rgb_tgt.flatten(0, 1) * 2.0 - 1.0
                last_lp = last_frame.flatten(0, 1) * 2.0 - 1.0
                model_lpips = lpips_model(pred_lp, tgt_lp).reshape(rgb_pred.shape[0], rgb_pred.shape[1]).mean(dim=1)
                last_lpips = lpips_model(last_lp, tgt_lp).reshape(rgb_pred.shape[0], rgb_pred.shape[1]).mean(dim=1)
            metrics["model_lpips"] = model_lpips
            metrics["last_frame_lpips"] = last_lpips

        _append_sample_metrics(by_ds, counts, dataset_names, metrics)

        if fvd_extractor is not None:
            fvd_target = rgb_tgt
            fvd_model = rgb_pred
            fvd_last = last_frame
            if context_rgb_seq is not None:
                fvd_target = torch.cat([context_rgb_seq, rgb_tgt], dim=1)
                fvd_model = torch.cat([context_rgb_seq, rgb_pred], dim=1)
                fvd_last = torch.cat([context_rgb_seq, last_frame], dim=1)
            _collect_features(
                fvd_groups,
                dataset_names,
                target=fvd_target,
                model_pred=fvd_model,
                last_frame=fvd_last,
                extractor=fvd_extractor,
            )

        if args.log_every and (batch_idx + 1) % int(args.log_every) == 0:
            all_count = max(1, counts["ALL"])
            print(
                f"[{batch_idx + 1}/{len(loader)}] "
                f"model_psnr={by_ds['ALL']['model_psnr'] / all_count:.3f} "
                f"model_ssim={by_ds['ALL']['model_ssim'] / all_count:.4f} "
                f"last_psnr={by_ds['ALL']['last_frame_psnr'] / all_count:.3f}"
            )

    report = {
        "mode": {
            "metric_protocol": "future_rgb_video_quality",
            "split": args.split,
            "balanced_datasets": bool(args.balanced_datasets),
            "max_batches": int(args.max_batches),
            "max_batches_per_dataset": int(args.max_batches_per_dataset),
            "batch_size": batch_size,
            "motion_threshold": float(args.motion_threshold),
            "include_lpips": bool(args.include_lpips),
            "include_fvd": bool(args.include_fvd),
            "fvd_backend": args.fvd_backend if args.include_fvd else None,
            "i3d_model_path": str(args.i3d_model_path) if args.include_fvd and args.i3d_model_path else None,
            "fvd_context_frames": int(args.fvd_context_frames) if args.include_fvd else 0,
            "fvd_min_frames": 32 if args.include_fvd and args.fvd_backend == "i3d_torchscript" else None,
            "fvd_protocol": fvd_protocol_name(args.fvd_backend) if args.include_fvd else None,
            "fvd_note": (
                "I3D TorchScript is the paper-facing FVD-style protocol. "
                "R3D-18 is kept only as a faster internal proxy."
                if args.include_fvd and args.fvd_backend == "i3d_torchscript"
                else "This is an R3D-18/Kinetics FVD-style proxy, not the original I3D-TF FVD implementation."
            ),
        },
        "checkpoint": _checkpoint_report(args.ckpt, sd),
        "subset": subset_info,
        "counts": dict(sorted(counts.items())),
        "metrics": {},
        "fvd": _summarize_features(fvd_groups) if fvd_extractor is not None else {},
        "samples": sample_rows,
    }
    for dataset_name, values in sorted(by_ds.items()):
        denom = max(1, counts[dataset_name])
        report["metrics"][dataset_name] = {key: value / denom for key, value in sorted(values.items())}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--balanced_datasets", action="store_true")
    parser.add_argument("--max_batches_per_dataset", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=20260607)
    parser.add_argument("--dataset_filter", default=None)
    parser.add_argument("--motion_threshold", type=float, default=0.03)
    parser.add_argument("--include_lpips", action="store_true")
    parser.add_argument("--lpips_net", default="vgg", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--include_fvd", action="store_true")
    parser.add_argument("--fvd_backend", choices=("r3d18", "i3d_torchscript"), default="r3d18")
    parser.add_argument("--fvd_image_size", type=int, default=112)
    parser.add_argument("--fvd_context_frames", type=int, default=0)
    parser.add_argument("--i3d_model_path", type=Path, default=None)
    parser.add_argument("--sample_manifest_out", type=Path, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if args.sample_manifest_out is not None:
        args.sample_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with args.sample_manifest_out.open("w") as f:
            for row in report.get("samples", []):
                f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    if "ALL" in report["metrics"]:
        print(json.dumps({"ALL": report["metrics"]["ALL"], "fvd": report["fvd"].get("ALL")}, indent=2))


if __name__ == "__main__":
    main()
