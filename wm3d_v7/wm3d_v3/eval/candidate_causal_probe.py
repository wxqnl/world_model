"""Test whether action-conditioned WM futures identify the demonstrated action.

This is a pre-training diagnostic, not a serving metric. It asks whether the
demonstrated action predicts the observed future better than proposer
alternatives and whether that future error ranks proposer actions usefully.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.policy_rollout_probe import paired_bootstrap_mean_ci
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model
from wm3d_v3.losses import _normalize_depth


def _indices(args: argparse.Namespace, size: int) -> list[int]:
    if args.index_mode == "contiguous":
        return list(range(args.start, min(size, args.start + args.count)))
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if args.index_mode == "contiguous_sharded":
        total = max(0, min(int(args.count), size - int(args.start)))
        base, remainder = divmod(total, int(args.num_shards))
        shard_count = base + int(args.shard_index < remainder)
        offset = args.shard_index * base + min(args.shard_index, remainder)
        shard_start = int(args.start) + offset
        return list(range(shard_start, shard_start + shard_count))
    return list(range(args.shard_index, size, args.num_shards))[: args.count]


def _resize_depth(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    batch, horizon = pred.shape[:2]
    return F.interpolate(
        pred.reshape(batch * horizon, 1, *pred.shape[-2:]),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, horizon, *target.shape[-2:])


def _future_errors(
    out: dict[str, torch.Tensor],
    s_tgt: torch.Tensor,
    depth_tgt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred = out["pred_tokens"].float()
    target = s_tgt.float()
    token_mse = (pred - target).pow(2).flatten(1).mean(dim=1)
    token_cos = 1.0 - F.cosine_similarity(
        pred.flatten(-2), target.flatten(-2), dim=-1
    ).mean(dim=1)
    depth = _resize_depth(out["depth"].float(), depth_tgt)
    depth_l1 = (
        _normalize_depth(depth) - _normalize_depth(depth_tgt.float())
    ).abs().flatten(1).mean(dim=1)
    return {
        "token_mse": token_mse,
        "token_cos_error": token_cos,
        "depth_l1": depth_l1,
        "composite": token_mse + 0.3 * depth_l1,
    }


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.cfg.read_text())
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    dataset = build_dataset_for_split(
        read_manifest(cfg["data"]["manifest"]), cfg, split=args.split
    )
    indices = _indices(args, len(dataset))
    if not indices:
        raise RuntimeError("causal probe selected no rows")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device).eval()
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    chunks: dict[str, list[torch.Tensor]] = {
        "pose_l1": [],
        "grip_bce": [],
        "candidate_delta": [],
        "token_mse": [],
        "token_cos_error": [],
        "depth_l1": [],
        "composite": [],
    }
    row_ids: list[tuple[str, int]] = []
    seen = 0
    for batch in loader:
        s = batch["s_in"].to(device, non_blocking=True)
        task = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        s_tgt = batch["s_tgt"].to(device, non_blocking=True)
        depth_tgt = batch["depth_tgt"].to(device, non_blocking=True)
        factual = make_action_condition(action_tgt, action_tgt_norm).to(dtype=s.dtype)

        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            proposal_out = model(
                s, task, action_cond=factual, pixel=False, bridging=False,
                skip_action_policy=True,
            )
        candidates = proposal_out["proposer_action_cond"].detach().to(dtype=s.dtype)
        pose_l1 = (
            candidates[..., :6].float() - action_tgt_norm.float()[:, None]
        ).abs().mean(dim=(2, 3))
        grip_target = (action_tgt[..., 6] > 0.5).float()
        grip_prob = candidates[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
        grip_bce = F.binary_cross_entropy(
            grip_prob,
            grip_target[:, None].expand_as(grip_prob),
            reduction="none",
        ).mean(dim=2)
        chunks["pose_l1"].append(pose_l1.cpu())
        chunks["grip_bce"].append(grip_bce.cpu())
        chunks["candidate_delta"].append(
            (candidates[:, :, :, :6] - candidates[:, :1, :, :6])
            .abs().mean(dim=(2, 3)).float().cpu()
        )

        outputs = [proposal_out]
        for candidate_index in range(candidates.shape[1]):
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                outputs.append(model(
                    s,
                    task,
                    action_cond=candidates[:, candidate_index],
                    pixel=False,
                    bridging=False,
                    skip_action_proposer=True,
                    skip_action_policy=True,
                ))
        errors = [_future_errors(out, s_tgt, depth_tgt) for out in outputs]
        for key in ("token_mse", "token_cos_error", "depth_l1", "composite"):
            chunks[key].append(torch.stack([item[key] for item in errors], dim=1).cpu())

        batch_size = s.shape[0]
        row_ids.extend(
            (str(batch["clip_id"][row]), int(batch["start"][row]))
            for row in range(batch_size)
        )
        seen += batch_size
        if seen % 8 == 0 or seen == len(indices):
            print(f"extracted {seen}/{len(indices)}", flush=True)
        del proposal_out, outputs, errors

    payload: dict[str, Any] = {key: torch.cat(value, dim=0) for key, value in chunks.items()}
    payload.update({
        "ids": row_ids,
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "split": args.split,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(json.dumps({"out": str(args.out), "rows": len(row_ids)}, indent=2))


def _load_dirs(paths: list[Path]) -> dict[str, Any]:
    parts = []
    for path in paths:
        files = sorted(path.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"no causal probe shards in {path}")
        parts.extend(torch.load(file, map_location="cpu", weights_only=False) for file in files)
    tensor_keys = (
        "pose_l1", "grip_bce", "candidate_delta", "token_mse",
        "token_cos_error", "depth_l1", "composite",
    )
    merged = {key: torch.cat([part[key] for part in parts], dim=0) for key in tensor_keys}
    merged["ids"] = [tuple(row_id) for part in parts for row_id in part["ids"]]
    keep = []
    seen = set()
    for index, row_id in enumerate(merged["ids"]):
        if row_id not in seen:
            seen.add(row_id)
            keep.append(index)
    idx = torch.tensor(keep, dtype=torch.long)
    result = {key: value.index_select(0, idx) for key, value in merged.items() if key != "ids"}
    result["ids"] = [merged["ids"][index] for index in keep]
    return result


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = left.argsort(dim=1).argsort(dim=1).float()
    right_rank = right.argsort(dim=1).argsort(dim=1).float()
    left_rank = left_rank - left_rank.mean(dim=1, keepdim=True)
    right_rank = right_rank - right_rank.mean(dim=1, keepdim=True)
    return (left_rank * right_rank).sum(dim=1) / (
        left_rank.square().sum(dim=1).sqrt()
        * right_rank.square().sum(dim=1).sqrt()
    ).clamp_min(1.0e-12)


def summarize(args: argparse.Namespace) -> None:
    payload = _load_dirs(args.dirs)
    pose_l1 = payload["pose_l1"].float()
    anchor = pose_l1[:, 0]
    oracle, oracle_idx = pose_l1.min(dim=1)
    report: dict[str, Any] = {
        "rows": len(payload["ids"]),
        "candidate_count": int(pose_l1.shape[1]),
        "anchor_pose_l1": float(anchor.mean()),
        "oracle_pose_l1": float(oracle.mean()),
        "candidate0_oracle_fraction": float((oracle_idx == 0).float().mean()),
        "candidate_delta_mean": float(payload["candidate_delta"][:, 1:].mean()),
        "future_metrics": {},
    }
    for key in ("token_mse", "token_cos_error", "depth_l1", "composite"):
        future_error = payload[key].float()
        factual_rank = future_error.argsort(dim=1).argsort(dim=1)[:, 0]
        proposal_error = future_error[:, 1:]
        selected_idx = proposal_error.argmin(dim=1)
        selected = pose_l1.gather(1, selected_idx[:, None]).squeeze(1)
        paired = paired_bootstrap_mean_ci(anchor - selected, seed=args.seed, samples=10000)
        report["future_metrics"][key] = {
            "factual_top1": float((factual_rank == 0).float().mean()),
            "factual_top2": float((factual_rank < 2).float().mean()),
            "factual_mean_rank_zero_based": float(factual_rank.float().mean()),
            "factual_vs_best_margin": float(
                (future_error[:, 1:].min(dim=1).values - future_error[:, 0]).mean()
            ),
            "selected_pose_l1": float(selected.mean()),
            "relative_improvement": float((anchor.mean() - selected.mean()) / anchor.mean()),
            "ci95_lower": float(paired["ci95_lower"]),
            "ci95_upper": float(paired["ci95_upper"]),
            "fraction_improved": float(paired["fraction_improved"]),
            "oracle_match": float((selected_idx == oracle_idx).float().mean()),
            "selected_candidate0_fraction": float((selected_idx == 0).float().mean()),
            "candidate_spearman_mean": float(
                _rank_correlation(proposal_error, pose_l1).mean()
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--cfg", type=Path, required=True)
    extract_parser.add_argument("--ckpt", type=Path, required=True)
    extract_parser.add_argument("--out", type=Path, required=True)
    extract_parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    extract_parser.add_argument(
        "--index_mode",
        choices=("contiguous", "contiguous_sharded", "strided"),
        default="contiguous",
    )
    extract_parser.add_argument("--start", type=int, default=0)
    extract_parser.add_argument("--count", type=int, default=16)
    extract_parser.add_argument("--shard_index", type=int, default=0)
    extract_parser.add_argument("--num_shards", type=int, default=1)
    extract_parser.add_argument("--batch_size", type=int, default=1)
    extract_parser.add_argument("--num_workers", type=int, default=0)
    extract_parser.add_argument("--seed", type=int, default=1729)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--dirs", type=Path, nargs="+", required=True)
    summarize_parser.add_argument("--out", type=Path, required=True)
    summarize_parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
