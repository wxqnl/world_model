"""Action counterfactual sensitivity benchmark."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model
from wm3d_v3.losses import _normalize_depth


DEFAULT_VARIANTS = ("zero", "shuffled", "sign_flip", "scaled", "grip_toggle")


def parse_variants(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_VARIANTS)
    raw = [value] if isinstance(value, str) else list(value)
    variants: list[str] = []
    for item in raw:
        variants.extend(v.strip() for v in item.split(",") if v.strip())
    invalid = sorted(set(variants) - set(DEFAULT_VARIANTS) - {"real"})
    if invalid:
        raise ValueError(f"unknown action variants: {invalid}")
    return [v for v in variants if v != "real"]


def make_action_counterfactuals(
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor | None = None,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    *,
    generator: torch.Generator | None = None,
    scaled_factor: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Build real and counterfactual action-conditioning tensors.

    The real condition is always present. Pose variants operate on the normalized
    pose channels produced by ``make_action_condition``; gripper variants operate
    on the canonical binary gripper channel from the raw action.
    """
    cond = make_action_condition(action_tgt, action_tgt_norm)
    out = {"real": cond}
    requested = parse_variants(variants)

    for variant in requested:
        cur = cond.clone()
        if variant == "zero":
            cur.zero_()
        elif variant == "shuffled":
            if cur.shape[0] > 1:
                perm = torch.randperm(cur.shape[0], generator=generator, device=cur.device)
                if torch.equal(perm, torch.arange(cur.shape[0], device=cur.device)):
                    perm = torch.roll(perm, shifts=1)
                cur = cur[perm]
            elif cur.shape[1] > 1:
                perm = torch.randperm(cur.shape[1], generator=generator, device=cur.device)
                if torch.equal(perm, torch.arange(cur.shape[1], device=cur.device)):
                    perm = torch.roll(perm, shifts=1)
                cur = cur[:, perm]
        elif variant == "sign_flip":
            cur[..., :6].neg_()
        elif variant == "scaled":
            cur[..., :6].mul_(scaled_factor)
        elif variant == "grip_toggle":
            cur[..., 6:7] = 1.0 - cur[..., 6:7]
        else:
            raise ValueError(f"unknown action variant: {variant}")
        out[variant] = cur
    return out


def _sample_mean(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(x.shape[0], -1).mean(dim=1)


def _sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _sample_mean((a.float() - b.float()).pow(2))


def _sample_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _sample_mean((a.float() - b.float()).abs())


def _resize_motion_like(motion: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if motion.shape == ref.shape:
        return motion
    if motion.ndim != 5 or ref.ndim != 5:
        raise ValueError(f"expected 5D motion tensors, got {tuple(motion.shape)} and {tuple(ref.shape)}")
    if motion.shape[:3] != ref.shape[:3]:
        raise ValueError(f"cannot align motion tensors with different leading dims: {tuple(motion.shape)} vs {tuple(ref.shape)}")
    b, t, c, _, _ = motion.shape
    return F.interpolate(
        motion.float().flatten(0, 1),
        size=ref.shape[-2:],
        mode="nearest",
    ).reshape(b, t, c, ref.shape[-2], ref.shape[-1])


def _sample_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1), dim=1)


def motion_target_from_rgb(rgb_tgt: torch.Tensor, context_rgb: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    """Return a binary motion target shaped like context-pixel motion hints."""
    motion = (rgb_tgt.float() - context_rgb.float().unsqueeze(1)).abs().mean(dim=2, keepdim=True)
    return (motion > threshold).float()


def compute_counterfactual_metrics(
    real_out: Mapping[str, torch.Tensor],
    variant_out: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-sample metrics comparing real-action and variant-action predictions."""
    metrics: dict[str, torch.Tensor] = {}

    real_tokens = real_out["pred_tokens"].float()
    variant_tokens = variant_out["pred_tokens"].float()
    s_tgt = targets["s_tgt"].float()

    metrics["pred_tokens_mse_gap"] = _sample_mse(real_tokens, variant_tokens)
    metrics["pred_tokens_cos_gap"] = 1.0 - _sample_cos(real_tokens, variant_tokens)

    real_token_mse = _sample_mse(real_tokens, s_tgt)
    variant_token_mse = _sample_mse(variant_tokens, s_tgt)
    metrics["pred_tokens_real_gt_mse"] = real_token_mse
    metrics["pred_tokens_variant_gt_mse"] = variant_token_mse
    metrics["pred_tokens_gt_mse_gap"] = variant_token_mse - real_token_mse
    metrics["pred_tokens_gt_mse_acc"] = (real_token_mse < variant_token_mse).float()

    real_token_cos = _sample_cos(real_tokens, s_tgt)
    variant_token_cos = _sample_cos(variant_tokens, s_tgt)
    metrics["pred_tokens_real_gt_cos"] = real_token_cos
    metrics["pred_tokens_variant_gt_cos"] = variant_token_cos
    metrics["pred_tokens_gt_cos_gap"] = real_token_cos - variant_token_cos
    metrics["pred_tokens_gt_cos_acc"] = (real_token_cos > variant_token_cos).float()

    real_depth = _normalize_depth(real_out["depth"].float())
    variant_depth = _normalize_depth(variant_out["depth"].float())
    depth_tgt = _normalize_depth(targets["depth_tgt"].float())
    metrics["depth_l1_gap"] = _sample_l1(real_depth, variant_depth)

    real_depth_l1 = _sample_l1(real_depth, depth_tgt)
    variant_depth_l1 = _sample_l1(variant_depth, depth_tgt)
    metrics["depth_real_gt_l1"] = real_depth_l1
    metrics["depth_variant_gt_l1"] = variant_depth_l1
    metrics["depth_gt_l1_gap"] = variant_depth_l1 - real_depth_l1
    metrics["depth_gt_l1_acc"] = (real_depth_l1 < variant_depth_l1).float()

    if "motion_hint" in real_out and "motion_hint" in variant_out:
        real_motion = real_out["motion_hint"].float()
        variant_motion = variant_out["motion_hint"].float()
        metrics["motion_hint_l1_gap"] = _sample_l1(real_motion, variant_motion)
        if "motion_tgt" in targets:
            motion_tgt = _resize_motion_like(targets["motion_tgt"].float(), real_motion)
            real_motion_l1 = _sample_l1(real_motion, motion_tgt)
            variant_motion_l1 = _sample_l1(variant_motion, motion_tgt)
            metrics["motion_hint_real_gt_l1"] = real_motion_l1
            metrics["motion_hint_variant_gt_l1"] = variant_motion_l1
            metrics["motion_hint_gt_l1_gap"] = variant_motion_l1 - real_motion_l1
            metrics["motion_hint_gt_l1_acc"] = (real_motion_l1 < variant_motion_l1).float()

    return metrics


class MetricAccumulator:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def update(self, metrics: Mapping[str, torch.Tensor | float], mask: torch.Tensor | None = None) -> None:
        for key, value in metrics.items():
            tensor = value.detach().float().reshape(-1) if isinstance(value, torch.Tensor) else torch.tensor([float(value)])
            if mask is not None:
                tensor = tensor[mask.reshape(-1).to(device=tensor.device)]
            if tensor.numel() == 0:
                continue
            self.totals[key] += float(tensor.sum().cpu())
            self.counts[key] += int(tensor.numel())

    def means(self) -> dict[str, float]:
        return {key: self.totals[key] / max(1, self.counts[key]) for key in sorted(self.totals)}


def aggregate_metric_batches(batches: Iterable[Mapping[str, torch.Tensor | float]]) -> dict[str, float]:
    acc = MetricAccumulator()
    for batch in batches:
        acc.update(batch)
    return acc.means()


def _to_device(batch: Mapping[str, Any], key: str, device: torch.device) -> torch.Tensor:
    return batch[key].to(device, non_blocking=True)


def _forward_model(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    action_cond: torch.Tensor,
    context_rgb: torch.Tensor | None,
    *,
    enable_context_pixel: bool,
    use_autocast: bool,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {"action_cond": action_cond, "pixel": True, "bridging": False}
    if enable_context_pixel:
        kwargs["context_rgb"] = context_rgb
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        return model(s, c, **kwargs)


@torch.no_grad()
def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    seed = args.seed if args.seed is not None else cfg["data"]["seed"]
    variants = parse_variants(args.variants)

    records = read_manifest(cfg["data"]["manifest"])
    val = build_dataset_for_split(records, cfg, split="val")
    print(f"val windows: {len(val)} seed={seed}")

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    val_total = sd.get("val_total")
    val_total_text = f"{val_total:.4f}" if isinstance(val_total, (float, int)) else str(val_total)
    print(f"loaded ckpt epoch={sd.get('epoch')} val_total={val_total_text}")

    loader = DataLoader(
        val,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    enable_context_pixel = cfg["model"].get("enable_context_pixel", False)
    use_autocast = device.type == "cuda"
    variant_gen = torch.Generator(device=device).manual_seed(seed + 1) if device.type == "cuda" else torch.Generator().manual_seed(seed + 1)
    by_dataset: dict[str, dict[str, MetricAccumulator]] = defaultdict(lambda: defaultdict(MetricAccumulator))
    counts: dict[str, int] = defaultdict(int)

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = _to_device(batch, "s_in", device)
        c = _to_device(batch, "c", device)
        s_tgt = _to_device(batch, "s_tgt", device)
        depth_tgt = _to_device(batch, "depth_tgt", device)
        action_tgt = _to_device(batch, "action_tgt", device)
        action_tgt_norm = _to_device(batch, "action_tgt_norm", device)
        rgb_tgt = _to_device(batch, "rgb_tgt", device).permute(0, 1, 4, 2, 3).contiguous()
        context_rgb = _to_device(batch, "rgb_in", device)[:, -1].permute(0, 3, 1, 2).contiguous()

        conds = make_action_counterfactuals(action_tgt, action_tgt_norm, variants, generator=variant_gen)
        real_out = _forward_model(
            model, s, c, conds["real"], context_rgb,
            enable_context_pixel=enable_context_pixel, use_autocast=use_autocast,
        )
        targets = {
            "s_tgt": s_tgt,
            "depth_tgt": depth_tgt,
            "motion_tgt": motion_target_from_rgb(rgb_tgt, context_rgb),
        }
        dataset_names = list(batch["dataset"])
        unique_datasets = sorted(set(dataset_names))

        for variant in variants:
            variant_out = _forward_model(
                model, s, c, conds[variant], context_rgb,
                enable_context_pixel=enable_context_pixel, use_autocast=use_autocast,
            )
            metrics = compute_counterfactual_metrics(real_out, variant_out, targets)
            by_dataset["ALL"][variant].update(metrics)
            for d in unique_datasets:
                mask = torch.tensor([name == d for name in dataset_names], device=device)
                by_dataset[d][variant].update(metrics, mask=mask)

        counts["ALL"] += s.shape[0]
        for d in dataset_names:
            counts[d] += 1

        if (bi + 1) % 10 == 0:
            first_variant = variants[0] if variants else "none"
            first_metrics = by_dataset["ALL"][first_variant].means() if variants else {}
            gap = first_metrics.get("pred_tokens_gt_mse_gap", 0.0)
            acc = first_metrics.get("pred_tokens_gt_mse_acc", 0.0)
            print(f"[{bi + 1}/{len(loader)}] {first_variant} token_gt_gap={gap:.6f} token_gt_acc={acc:.4f}")

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "seed": seed,
        "max_batches": args.max_batches,
        "variants": variants,
        "counts": dict(counts),
        "metrics": {
            dataset: {variant: acc.means() for variant, acc in sorted(vmap.items())}
            for dataset, vmap in sorted(by_dataset.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    print(json.dumps(report["metrics"].get("ALL", {}), indent=2, sort_keys=True))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_batches", type=int, default=0, help="cap batches for quicker eval (0=all)")
    ap.add_argument("--seed", type=int, default=None, help="validation split and shuffle seed (default: cfg data.seed)")
    ap.add_argument(
        "--variants",
        nargs="*",
        default=list(DEFAULT_VARIANTS),
        help="counterfactuals to run; comma-separated or space-separated",
    )
    run_benchmark(ap.parse_args())


if __name__ == "__main__":
    main()
