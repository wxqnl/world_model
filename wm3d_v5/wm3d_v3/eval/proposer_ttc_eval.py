"""Offline proposer/TTC evaluation.

This is a lightweight tau0-style check:

1. predict K action chunks from context/task
2. simulate each candidate with the world core
3. rank candidates with progress/plausibility heads
4. compare ranked action error against candidate-0 and oracle-best
"""
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

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model
from wm3d_v3.losses import _normalize_depth


def _sample_action_errors(
    pose_pred: torch.Tensor,
    grip_logits: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    action_tgt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose_l1 = (pose_pred.float() - action_tgt_norm.float()[:, None]).abs().mean(dim=(2, 3))
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_bce = F.binary_cross_entropy_with_logits(
        grip_logits.float(),
        grip_tgt[:, None].expand_as(grip_logits).float(),
        reduction="none",
    ).mean(dim=2)
    return pose_l1, grip_bce


def _score_candidate(
    out: dict[str, torch.Tensor],
    *,
    progress_weight: float = 1.0,
    terminal_weight: float = 1.0,
    plausibility_weight: float = 0.0,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    if progress_weight != 0 and "progress" in out:
        terms.append(torch.sigmoid(out["progress"].float()).mean(dim=1) * progress_weight)
        weights.append(abs(progress_weight))
    if terminal_weight != 0 and "terminal_success_logit" in out:
        terms.append(torch.sigmoid(out["terminal_success_logit"].float()) * terminal_weight)
        weights.append(abs(terminal_weight))
    if plausibility_weight != 0 and "plausibility_logit" in out:
        terms.append(torch.sigmoid(out["plausibility_logit"].float()) * plausibility_weight)
        weights.append(abs(plausibility_weight))
    if not terms:
        return torch.zeros(out["pred_tokens"].shape[0], device=out["pred_tokens"].device)
    return torch.stack(terms, dim=0).sum(dim=0) / max(1e-6, sum(weights))


def _oracle_score(
    out: dict[str, torch.Tensor],
    s_tgt: torch.Tensor,
    depth_tgt: torch.Tensor,
    motion_tgt: torch.Tensor | None,
) -> torch.Tensor:
    token_mse = (out["pred_tokens"].float() - s_tgt.float()).pow(2).flatten(1).mean(dim=1)
    depth_l1 = (
        _normalize_depth(out["depth"].float()) - _normalize_depth(depth_tgt.float())
    ).abs().flatten(1).mean(dim=1)
    total = token_mse + 0.3 * depth_l1
    if motion_tgt is not None and "motion_hint" in out:
        motion_pred = out["motion_hint"].float()
        target = motion_tgt.float()
        if motion_pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target.flatten(0, 1),
                size=motion_pred.shape[-2:],
                mode="nearest",
            ).reshape_as(motion_pred)
        total = total + 0.1 * (motion_pred - target).abs().flatten(1).mean(dim=1)
    return -total


def _motion_target_from_rgb(rgb_tgt: torch.Tensor, context_rgb: torch.Tensor, threshold: float = 0.03) -> torch.Tensor:
    motion = (rgb_tgt.float() - context_rgb.float().unsqueeze(1)).abs().mean(dim=2, keepdim=True)
    return (motion > threshold).float()


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


@torch.no_grad()
def run_eval(args: argparse.Namespace) -> dict[str, Any]:
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
    print(f"loaded ckpt epoch={sd.get('epoch')} split={args.split} windows={len(ds)}")

    acc = Accumulator()
    enable_context_pixel = cfg["model"].get("enable_context_pixel", False)
    use_autocast = device.type == "cuda"

    for bi, batch in enumerate(loader):
        if args.max_batches and bi >= args.max_batches:
            break
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        s_tgt = batch["s_tgt"].to(device, non_blocking=True)
        depth_tgt = batch["depth_tgt"].to(device, non_blocking=True)
        rgb_tgt = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3).contiguous()
        real_cond = make_action_condition(action_tgt, action_tgt_norm)
        context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        motion_tgt = _motion_target_from_rgb(rgb_tgt, context_rgb)
        kwargs: dict[str, Any] = {"pixel": False, "bridging": False}
        if enable_context_pixel:
            kwargs["context_rgb"] = context_rgb
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            first_out = model(s, c, action_cond=real_cond, **kwargs)
        if "proposer_pose_norm" not in first_out:
            raise RuntimeError("checkpoint/config has no action proposer enabled")
        pose_pred = first_out["proposer_pose_norm"].float()
        grip_logits = first_out["proposer_gripper_logit"].float()
        pose_l1, grip_bce = _sample_action_errors(pose_pred, grip_logits, action_tgt_norm, action_tgt)

        learned_scores = []
        oracle_scores = []
        conds = [real_cond]
        n_candidates = pose_pred.shape[1]
        for ci in range(n_candidates):
            conds.append(first_out["proposer_action_cond"][:, ci].to(device=device, dtype=real_cond.dtype))
        for cand_cond in conds:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                cand_out = model(s, c, action_cond=cand_cond, **kwargs)
            learned_scores.append(_score_candidate(
                cand_out,
                progress_weight=args.score_progress_weight,
                terminal_weight=args.score_terminal_weight,
                plausibility_weight=args.score_plausibility_weight,
            ))
            oracle_scores.append(_oracle_score(cand_out, s_tgt, depth_tgt, motion_tgt))
        learned_score_t = torch.stack(learned_scores, dim=1)
        oracle_score_t = torch.stack(oracle_scores, dim=1)
        learned_prop_score_t = learned_score_t[:, 1:]
        oracle_prop_score_t = oracle_score_t[:, 1:]
        ranked_idx = learned_prop_score_t.argmax(dim=1)
        oracle_idx = oracle_prop_score_t.argmax(dim=1)
        gather_idx = ranked_idx[:, None]
        oracle_gather_idx = oracle_idx[:, None]

        ranked_pose_l1 = pose_l1.gather(1, gather_idx).squeeze(1)
        ranked_grip_bce = grip_bce.gather(1, gather_idx).squeeze(1)
        oracle_ranked_pose_l1 = pose_l1.gather(1, oracle_gather_idx).squeeze(1)
        oracle_ranked_grip_bce = grip_bce.gather(1, oracle_gather_idx).squeeze(1)
        oracle_pose_l1 = pose_l1.min(dim=1).values
        oracle_grip_bce = grip_bce.min(dim=1).values
        anchor_pose_l1 = pose_l1[:, 0]
        anchor_grip_bce = grip_bce[:, 0]
        real_oracle_rank = (oracle_score_t.argsort(dim=1, descending=True) == 0).nonzero()
        real_top1 = (oracle_score_t.argmax(dim=1) == 0).float()
        real_top3 = (oracle_score_t.argsort(dim=1, descending=True)[:, : min(3, oracle_score_t.shape[1])] == 0).any(dim=1).float()
        acc.update({
            "anchor_pose_l1": anchor_pose_l1,
            "oracle_pose_l1": oracle_pose_l1,
            "oracle_ranked_pose_l1": oracle_ranked_pose_l1,
            "ranked_pose_l1": ranked_pose_l1,
            "anchor_grip_bce": anchor_grip_bce,
            "oracle_grip_bce": oracle_grip_bce,
            "oracle_ranked_grip_bce": oracle_ranked_grip_bce,
            "ranked_grip_bce": ranked_grip_bce,
            "ranked_idx": ranked_idx.float(),
            "oracle_idx": oracle_idx.float(),
            "real_oracle_top1": real_top1,
            "real_oracle_top3": real_top3,
            "learned_oracle_idx_match": (ranked_idx == oracle_idx).float(),
            "learned_score_margin": learned_prop_score_t.max(dim=1).values - learned_prop_score_t.mean(dim=1),
            "oracle_score_margin": oracle_prop_score_t.max(dim=1).values - oracle_prop_score_t.mean(dim=1),
        })
        if (bi + 1) % 10 == 0:
            m = acc.means()
            print(
                f"[{bi + 1}/{len(loader)}] ranked_pose={m['ranked_pose_l1']:.4f} "
                f"oracle_pose={m['oracle_pose_l1']:.4f} anchor_pose={m['anchor_pose_l1']:.4f}"
            )

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "split": args.split,
        "max_batches": args.max_batches,
        "score_weights": {
            "progress": args.score_progress_weight,
            "terminal": args.score_terminal_weight,
            "plausibility": args.score_plausibility_weight,
        },
        "metrics": acc.means(),
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
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--score_progress_weight", type=float, default=1.0)
    ap.add_argument("--score_terminal_weight", type=float, default=1.0)
    ap.add_argument("--score_plausibility_weight", type=float, default=0.0)
    run_eval(ap.parse_args())


if __name__ == "__main__":
    main()
