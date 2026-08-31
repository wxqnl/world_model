"""Future-RGB world-model eval on cached LIBERO windows.

This is the LIBERO-facing companion to ``video_quality_eval``. It uses the
already-tokenized WM3D LIBERO cache for model inputs and reads future RGB frames
from the original LIBERO HDF5 demos for PSNR/SSIM/LPIPS/FVD.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from wm3d_v3.benchmarks.online_tokenizer import resize_frames
from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.eval.run_eval import build_model, policy_kwargs_from_batch
from wm3d_v3.eval.video_quality_eval import (
    _append_sample_metrics,
    _checkpoint_report,
    _collect_features,
    _summarize_features,
    build_feature_extractor,
    fvd_protocol_name,
    motion_region_rgb_l1,
    psnr_video,
    ssim_video,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _select_balanced(
    rows: list[dict[str, Any]],
    *,
    max_windows_per_task: int,
    seed: int,
) -> list[dict[str, Any]]:
    if max_windows_per_task <= 0:
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_name") or row.get("instruction") or "unknown")].append(row)
    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    for task_name in sorted(groups):
        group = groups[task_name]
        order = rng.permutation(len(group)).tolist()
        selected.extend(group[i] for i in order[:max_windows_per_task])
    return selected


def _future_rgb_from_hdf5(row: Mapping[str, Any], *, camera_key: str, rgb_size: int) -> torch.Tensor:
    hdf5_path = Path(str(row["hdf5_path"]))
    demo_id = str(row["demo_id"])
    start = int(row["target_start"])
    k = int(row["k"])
    with h5py.File(hdf5_path, "r") as h5:
        frames = np.asarray(h5["data"][demo_id]["obs"][camera_key])
    if frames.ndim != 4:
        raise ValueError(f"expected LIBERO RGB frames [N,H,W,3] or [N,3,H,W], got {frames.shape}")
    if frames.shape[1] == 3 and frames.shape[-1] != 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.shape[-1] != 3:
        raise ValueError(f"expected RGB channels in last dim, got {frames.shape}")
    start = max(0, start)
    end = min(frames.shape[0], start + k)
    chunk = list(frames[start:end])
    if not chunk:
        chunk = [frames[-1]]
    while len(chunk) < k:
        chunk.append(chunk[-1])
    return resize_frames(chunk, int(rgb_size))


class LiberoCachedWorldVideoDataset(Dataset):
    """Read WM3D model inputs from LIBERO cache and future RGB from HDF5."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        camera_key: str = "agentview_rgb",
        rgb_size: int = 256,
        balanced_tasks: bool = False,
        max_windows_per_task: int = 0,
        seed: int = 20260607,
        dataset_name: str = "libero",
    ) -> None:
        rows = _read_jsonl(Path(manifest))
        if balanced_tasks:
            rows = _select_balanced(rows, max_windows_per_task=max_windows_per_task, seed=seed)
        self.rows = rows
        self.camera_key = str(camera_key)
        self.rgb_size = int(rgb_size)
        self.dataset_name = str(dataset_name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        cache = np.load(Path(str(row["cache_path"])))
        rgb_tgt = _future_rgb_from_hdf5(row, camera_key=self.camera_key, rgb_size=self.rgb_size)
        clip_id = f"{row.get('task_name', 'unknown')}/{row.get('demo_id', 'demo')}"
        return {
            "s_in": torch.from_numpy(np.asarray(cache["s_in"])).float(),
            "c": torch.from_numpy(np.asarray(cache["c"])).float(),
            "context_rgb": torch.from_numpy(np.asarray(cache["context_rgb"])).float(),
            "rgb_tgt": rgb_tgt.float(),
            "action_tgt": torch.from_numpy(np.asarray(cache["action_tgt"])).float(),
            "action_tgt_norm": torch.from_numpy(np.asarray(cache["action_tgt_norm"])).float(),
            "dataset": self.dataset_name,
            "task_name": str(row.get("task_name") or "unknown"),
            "clip_id": clip_id,
            "start": int(row.get("target_start") or 0),
            **{
                key: torch.from_numpy(np.asarray(cache[key])).float()
                for key in ("lowdim_state", "object_state", "plan_state", "action_history")
                if key in cache
            },
        }


def _sample_rows(batch: Mapping[str, Any], *, batch_index: int, global_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    starts = batch["start"]
    for i, dataset in enumerate(batch["dataset"]):
        start = starts[i]
        if isinstance(start, torch.Tensor):
            start = int(start.detach().cpu())
        rows.append(
            {
                "sample_index": int(global_offset + i),
                "batch_index": int(batch_index),
                "batch_sample_index": int(i),
                "dataset": str(dataset),
                "task_name": str(batch["task_name"][i]),
                "clip_id": str(batch["clip_id"][i]),
                "start": int(start),
            }
        )
    return rows


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = LiberoCachedWorldVideoDataset(
        args.cache_manifest,
        camera_key=args.camera_key,
        rgb_size=int(args.rgb_size),
        balanced_tasks=bool(args.balanced_tasks),
        max_windows_per_task=int(args.max_windows_per_task),
        seed=int(args.sample_seed),
        dataset_name=args.dataset_name,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"empty LIBERO cache manifest after filtering: {args.cache_manifest}")
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
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
        context_rgb = batch["context_rgb"].to(device, non_blocking=True).contiguous().float()
        if context_rgb.shape[-2:] != (int(args.rgb_size), int(args.rgb_size)):
            context_rgb = F.interpolate(context_rgb, size=(int(args.rgb_size), int(args.rgb_size)), mode="bilinear", align_corners=False)
        rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).contiguous().float()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            kwargs = dict(action_cond=action_cond, pixel=True, bridging=False)
            kwargs.update(policy_kwargs_from_batch(batch, device))
            if cfg["model"].get("enable_context_pixel", False):
                kwargs["context_rgb"] = context_rgb
            out = model(s, c, **kwargs)
        if "rgb" not in out:
            raise RuntimeError("model output has no 'rgb'; LIBERO video eval requires enable_pixel=True")

        rgb_pred = out["rgb"].float().clamp(0.0, 1.0)
        last_frame = context_rgb.float().unsqueeze(1).expand_as(rgb_tgt).contiguous()
        dataset_names = [str(name) for name in batch["dataset"]]
        sample_rows.extend(_sample_rows(batch, batch_index=batch_idx, global_offset=len(sample_rows)))

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
                metrics["model_lpips"] = lpips_model(pred_lp, tgt_lp).reshape(rgb_pred.shape[0], rgb_pred.shape[1]).mean(dim=1)
                metrics["last_frame_lpips"] = lpips_model(last_lp, tgt_lp).reshape(rgb_pred.shape[0], rgb_pred.shape[1]).mean(dim=1)

        _append_sample_metrics(by_ds, counts, dataset_names, metrics)

        if fvd_extractor is not None:
            _collect_features(
                fvd_groups,
                dataset_names,
                target=rgb_tgt,
                model_pred=rgb_pred,
                last_frame=last_frame,
                extractor=fvd_extractor,
            )

        if args.log_every and (batch_idx + 1) % int(args.log_every) == 0:
            all_count = max(1, counts["ALL"])
            print(
                f"[{batch_idx + 1}/{len(loader)}] "
                f"model_psnr={by_ds['ALL']['model_psnr'] / all_count:.3f} "
                f"model_ssim={by_ds['ALL']['model_ssim'] / all_count:.4f} "
                f"last_psnr={by_ds['ALL']['last_frame_psnr'] / all_count:.3f}",
                flush=True,
            )

    report = {
        "mode": {
            "metric_protocol": "libero_future_rgb_video_quality",
            "cache_manifest": str(args.cache_manifest),
            "camera_key": args.camera_key,
            "rgb_size": int(args.rgb_size),
            "balanced_tasks": bool(args.balanced_tasks),
            "max_windows_per_task": int(args.max_windows_per_task),
            "batch_size": int(args.batch_size),
            "motion_threshold": float(args.motion_threshold),
            "include_lpips": bool(args.include_lpips),
            "include_fvd": bool(args.include_fvd),
            "fvd_backend": args.fvd_backend if args.include_fvd else None,
            "fvd_protocol": fvd_protocol_name(args.fvd_backend) if args.include_fvd else None,
        },
        "checkpoint": _checkpoint_report(args.ckpt, sd),
        "subset": {"selected_total_windows": len(dataset)},
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
    parser.add_argument("--cache_manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--camera_key", default="agentview_rgb")
    parser.add_argument("--rgb_size", type=int, default=256)
    parser.add_argument("--dataset_name", default="libero")
    parser.add_argument("--balanced_tasks", action="store_true")
    parser.add_argument("--max_windows_per_task", type=int, default=16)
    parser.add_argument("--sample_seed", type=int, default=20260607)
    parser.add_argument("--motion_threshold", type=float, default=0.03)
    parser.add_argument("--include_lpips", action="store_true")
    parser.add_argument("--lpips_net", default="vgg", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--include_fvd", action="store_true")
    parser.add_argument("--fvd_backend", choices=("r3d18", "i3d_torchscript"), default="i3d_torchscript")
    parser.add_argument("--fvd_image_size", type=int, default=112)
    parser.add_argument("--i3d_model_path", type=Path, default=None)
    parser.add_argument("--sample_manifest_out", type=Path, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
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
