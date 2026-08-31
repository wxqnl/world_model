"""Probe whether a checkpoint can output actions through the WM3D policy API."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model
from wm3d_v3.policy import ScoreWeights, select_action_chunk, selected_first_action


class Accumulator:
    def __init__(self) -> None:
        self.total: dict[str, float] = defaultdict(float)
        self.count = 0

    def update(self, values: dict[str, torch.Tensor]) -> None:
        n = next(iter(values.values())).numel()
        self.count += n
        for key, value in values.items():
            self.total[key] += float(value.detach().float().sum().cpu())

    def means(self) -> dict[str, float]:
        return {key: value / max(1, self.count) for key, value in sorted(self.total.items())}


def paired_bootstrap_mean_ci(
    improvements: torch.Tensor,
    *,
    seed: int = 1729,
    samples: int = 10000,
) -> dict[str, float | int]:
    """Deterministic paired bootstrap interval for anchor-minus-selected error."""

    values = improvements.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        raise ValueError("paired bootstrap requires at least one sample")
    if not torch.isfinite(values).all():
        raise ValueError("paired bootstrap received non-finite improvements")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        values.numel(),
        (int(samples), values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    bounds = torch.quantile(means, torch.tensor([0.025, 0.975]))
    return {
        "samples": int(values.numel()),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "mean_improvement": float(values.mean()),
        "ci95_lower": float(bounds[0]),
        "ci95_upper": float(bounds[1]),
        "fraction_improved": float((values > 0).float().mean()),
    }


def clip_balanced_indices(dataset: Any, *, max_windows_per_clip: int, seed: int) -> list[int]:
    """Select a deterministic, shuffled cap of windows from every clip."""
    if max_windows_per_clip <= 0:
        return list(range(len(dataset)))
    by_clip: dict[str, list[int]] = defaultdict(list)
    for dataset_idx, (record_idx, _start) in enumerate(dataset.index):
        by_clip[str(dataset.records[record_idx].clip_id)].append(dataset_idx)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    clip_ids = sorted(by_clip)
    clip_order = torch.randperm(len(clip_ids), generator=rng).tolist()
    selected: list[int] = []
    for clip_pos in clip_order:
        indices = by_clip[clip_ids[clip_pos]]
        index_order = torch.randperm(len(indices), generator=rng).tolist()
        selected.extend(indices[pos] for pos in index_order[:max_windows_per_clip])
    return selected


def _action_errors(
    candidates: torch.Tensor,
    selected_idx: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    action_tgt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pose_abs = (candidates[..., :6].float() - action_tgt_norm.float()[:, None]).abs()
    pose_l1 = pose_abs.mean(dim=(2, 3))
    first_pose_l1 = pose_abs[:, :, 0].mean(dim=2)
    late_start = max(0, int(pose_abs.shape[2]) // 2)
    late_pose_l1 = pose_abs[:, :, late_start:].mean(dim=(2, 3))
    step_pose_l1 = pose_abs.mean(dim=3)
    worst_step_pose_l1 = step_pose_l1.max(dim=2).values
    cumulative_error = (
        candidates[..., :6].float().cumsum(dim=2)
        - action_tgt_norm.float()[:, None].cumsum(dim=2)
    ).abs()
    endpoint_pose_l1 = cumulative_error[:, :, -1].mean(dim=2)
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_bce = F.binary_cross_entropy(
        candidates[..., 6].float().clamp(1e-5, 1 - 1e-5),
        grip_tgt[:, None].expand_as(candidates[..., 6]).float(),
        reduction="none",
    ).mean(dim=2)
    gather = selected_idx[:, None]
    oracle_idx = pose_l1.argmin(dim=1)
    return {
        "selected_pose_l1": pose_l1.gather(1, gather).squeeze(1),
        "anchor_pose_l1": pose_l1[:, 0],
        "oracle_pose_l1": pose_l1.min(dim=1).values,
        "selected_first_pose_l1": first_pose_l1.gather(1, gather).squeeze(1),
        "anchor_first_pose_l1": first_pose_l1[:, 0],
        "oracle_first_pose_l1": first_pose_l1.min(dim=1).values,
        "selected_late_pose_l1": late_pose_l1.gather(1, gather).squeeze(1),
        "anchor_late_pose_l1": late_pose_l1[:, 0],
        "selected_worst_step_pose_l1": worst_step_pose_l1.gather(1, gather).squeeze(1),
        "anchor_worst_step_pose_l1": worst_step_pose_l1[:, 0],
        "selected_endpoint_pose_l1": endpoint_pose_l1.gather(1, gather).squeeze(1),
        "anchor_endpoint_pose_l1": endpoint_pose_l1[:, 0],
        "selected_grip_bce": grip_bce.gather(1, gather).squeeze(1),
        "anchor_grip_bce": grip_bce[:, 0],
        "oracle_grip_bce": grip_bce.min(dim=1).values,
        "selected_idx": selected_idx.float(),
        "oracle_idx": oracle_idx.float(),
        "selected_matches_action_oracle": (selected_idx == oracle_idx).float(),
    }


@torch.no_grad()
def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    records = read_manifest(cfg["data"]["manifest"])
    ds = build_dataset_for_split(records, cfg, split=args.split)
    eval_ds = ds
    if args.max_windows_per_clip > 0:
        eval_ds = Subset(
            ds,
            clip_balanced_indices(
                ds,
                max_windows_per_clip=args.max_windows_per_clip,
                seed=args.sampler_seed,
            ),
        )
    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size or cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])

    weights = ScoreWeights(
        progress=args.score_progress_weight,
        terminal=args.score_terminal_weight,
        plausibility=args.score_plausibility_weight,
    )
    acc = Accumulator()
    saved_examples: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        policy_kwargs = {}
        for key in ("lowdim_state", "object_state", "plan_state", "action_history"):
            if key in batch:
                policy_kwargs[key] = batch[key].to(device, non_blocking=True)
        if "progress_state" in batch:
            policy_kwargs["progress_state"] = batch["progress_state"].to(device, non_blocking=True)
        elif "progress_tgt" in batch:
            progress = batch["progress_tgt"].to(device, non_blocking=True)
            policy_kwargs["progress_state"] = progress[:, :1] if progress.ndim > 1 else progress
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            decision = select_action_chunk(
                model,
                s,
                c,
                context_rgb=context_rgb,
                pixel=False,
                score_weights=weights,
                selection_mode=args.selection_mode,
                **policy_kwargs,
            )
        candidates = decision["candidate_action_cond"].float()
        selected_idx = decision["selected_idx"]
        scores = decision["candidate_scores"].float()
        values = _action_errors(candidates, selected_idx, action_tgt_norm, action_tgt)
        values["score_margin"] = scores.max(dim=1).values - scores.mean(dim=1)
        values["score_std"] = scores.std(dim=1)
        acc.update(values)
        row_values = {
            key: value.detach().float().cpu()
            for key, value in values.items()
        }
        for row in range(selected_idx.shape[0]):
            selected_pose = float(row_values["selected_pose_l1"][row])
            anchor_pose = float(row_values["anchor_pose_l1"][row])
            oracle_pose = float(row_values["oracle_pose_l1"][row])
            paired_rows.append({
                "clip_id": str(batch["clip_id"][row]),
                "start": int(batch["start"][row]),
                "selected_idx": int(row_values["selected_idx"][row]),
                "oracle_idx": int(row_values["oracle_idx"][row]),
                "selected_matches_action_oracle": bool(
                    row_values["selected_matches_action_oracle"][row] > 0.5
                ),
                "selected_pose_l1": selected_pose,
                "anchor_pose_l1": anchor_pose,
                "oracle_pose_l1": oracle_pose,
                "selected_vs_anchor_improvement": anchor_pose - selected_pose,
                "proposer_oracle_headroom": anchor_pose - oracle_pose,
                "selected_grip_bce": float(row_values["selected_grip_bce"][row]),
                "anchor_grip_bce": float(row_values["anchor_grip_bce"][row]),
                "score_margin": float(row_values["score_margin"][row]),
            })


        if args.save_examples and len(saved_examples) < args.save_examples:
            first = selected_first_action(decision, raw=True).detach().float().cpu()
            first_norm = selected_first_action(decision, raw=False).detach().float().cpu()
            for row in range(min(first.shape[0], args.save_examples - len(saved_examples))):
                saved_examples.append({
                    "clip_id": str(batch["clip_id"][row]),
                    "start": int(batch["start"][row]),
                    "selected_idx": int(selected_idx[row].detach().cpu()),
                    "selected_score": float(decision["selected_score"][row].detach().float().cpu()),
                    "first_action_raw": first[row].tolist(),
                    "first_action_cond": first_norm[row].tolist(),
                })

        if (bi + 1) % 10 == 0:
            m = acc.means()
            print(
                f"[{bi + 1}/{len(loader)}] selected_pose={m['selected_pose_l1']:.4f} "
                f"anchor_pose={m['anchor_pose_l1']:.4f} oracle_pose={m['oracle_pose_l1']:.4f} "
                f"match={m['selected_matches_action_oracle']:.3f}"
            )

    mean, std = None, None
    action_proj = getattr(model, "action_proj", None)
    if action_proj is not None and hasattr(action_proj, "mean") and hasattr(action_proj, "std"):
        mean = action_proj.mean.detach().reshape(-1)[:6].float().cpu().tolist()
        std = action_proj.std.detach().reshape(-1)[:6].float().cpu().tolist()
    metrics = acc.means()
    improvements = torch.tensor([
        row["selected_vs_anchor_improvement"] for row in paired_rows
    ])
    clip_values: dict[str, list[float]] = defaultdict(list)
    for row in paired_rows:
        clip_values[row["clip_id"]].append(row["selected_vs_anchor_improvement"])
    clip_improvements = torch.tensor([
        sum(values) / len(values) for values in clip_values.values()
    ])
    paired = paired_bootstrap_mean_ci(
        clip_improvements,
        seed=args.bootstrap_seed,
        samples=args.bootstrap_samples,
    )
    paired["cluster_unit"] = "clip"
    paired["clusters"] = len(clip_values)
    paired["window_mean_improvement"] = float(improvements.mean())
    anchor_mean = metrics["anchor_pose_l1"]
    selected_mean = metrics["selected_pose_l1"]
    oracle_mean = metrics["oracle_pose_l1"]
    paired["selected_to_anchor_ratio"] = selected_mean / max(anchor_mean, 1.0e-12)
    paired["relative_improvement"] = (anchor_mean - selected_mean) / max(anchor_mean, 1.0e-12)
    paired["proposer_oracle_headroom_relative"] = (anchor_mean - oracle_mean) / max(anchor_mean, 1.0e-12)
    criteria = {
        "selected_to_anchor_ratio_max": float(args.max_selected_anchor_ratio),
        "paired_ci95_lower_min_exclusive": 0.0,
        "oracle_match_min": float(args.min_oracle_match),
        "proposer_oracle_headroom_relative_min": float(args.min_oracle_headroom_relative),
    }
    passed = (
        paired["selected_to_anchor_ratio"] <= criteria["selected_to_anchor_ratio_max"]
        and paired["ci95_lower"] > 0.0
        and metrics["selected_matches_action_oracle"] >= criteria["oracle_match_min"]
        and paired["proposer_oracle_headroom_relative"] >= criteria["proposer_oracle_headroom_relative_min"]
    )

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "split": args.split,
        "max_batches": args.max_batches,
        "can_output_actions": True,
        "action_space": "7D delta pose + gripper_closed",
        "action_units": "raw pose if action_stats/model buffers are loaded, otherwise normalized pose",
        "action_stats_mean": mean,
        "action_stats_std": std,
        "selection_mode": args.selection_mode,
        "score_weights": {
            "progress": weights.progress,
            "terminal": weights.terminal,
            "plausibility": weights.plausibility,
        },
        "metrics": metrics,
        "paired_selected_vs_anchor_pose_l1": paired,
        "evidence_gate": {"passed": bool(passed), "criteria": criteria},
        "rows": paired_rows,
        "examples": saved_examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "val", "all"), default="val")
    ap.add_argument("--max_batches", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument(
        "--selection_mode",
        choices=("ranked", "ranked_residual", "anchor"),
        default="ranked",
    )
    ap.add_argument("--score_progress_weight", type=float, default=1.0)
    ap.add_argument("--score_terminal_weight", type=float, default=1.0)
    ap.add_argument("--score_plausibility_weight", type=float, default=0.0)
    ap.add_argument("--save_examples", type=int, default=8)
    ap.add_argument("--bootstrap_seed", type=int, default=1729)
    ap.add_argument("--bootstrap_samples", type=int, default=10000)
    ap.add_argument("--max_windows_per_clip", type=int, default=0)
    ap.add_argument("--sampler_seed", type=int, default=1729)
    ap.add_argument("--max_selected_anchor_ratio", type=float, default=0.98)
    ap.add_argument("--min_oracle_match", type=float, default=0.35)
    ap.add_argument("--min_oracle_headroom_relative", type=float, default=0.10)
    run_probe(ap.parse_args())


if __name__ == "__main__":
    main()
