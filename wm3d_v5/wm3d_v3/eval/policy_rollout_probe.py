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
from torch.utils.data import DataLoader

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


def _action_errors(
    candidates: torch.Tensor,
    selected_idx: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    action_tgt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pose_l1 = (candidates[..., :6].float() - action_tgt_norm.float()[:, None]).abs().mean(dim=(2, 3))
    first_pose_l1 = (candidates[:, :, 0, :6].float() - action_tgt_norm[:, None, 0].float()).abs().mean(dim=2)
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
    loader = DataLoader(
        ds,
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

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            decision = select_action_chunk(
                model,
                s,
                c,
                context_rgb=context_rgb,
                pixel=False,
                score_weights=weights,
            )
        candidates = decision["candidate_action_cond"].float()
        selected_idx = decision["selected_idx"]
        scores = decision["candidate_scores"].float()
        values = _action_errors(candidates, selected_idx, action_tgt_norm, action_tgt)
        values["score_margin"] = scores.max(dim=1).values - scores.mean(dim=1)
        values["score_std"] = scores.std(dim=1)
        acc.update(values)

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
        "score_weights": {
            "progress": weights.progress,
            "terminal": weights.terminal,
            "plausibility": weights.plausibility,
        },
        "metrics": acc.means(),
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
    ap.add_argument("--score_progress_weight", type=float, default=1.0)
    ap.add_argument("--score_terminal_weight", type=float, default=1.0)
    ap.add_argument("--score_plausibility_weight", type=float, default=0.0)
    ap.add_argument("--save_examples", type=int, default=8)
    run_probe(ap.parse_args())


if __name__ == "__main__":
    main()
