"""Frozen-world feature probe for candidate action ranking.

This is deliberately a cheap pre-training diagnostic. It extracts deterministic
features from the actual propose -> rollout path, then fits permutation-invariant
pairwise ridge rankers without updating WM3D.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.eval.policy_rollout_probe import paired_bootstrap_mean_ci
from wm3d_v3.eval.run_eval import build_dataset_for_split, build_model
from wm3d_v3.policy import ScoreWeights, select_action_chunk


FEATURE_KEYS = ("action", "heads", "geometry", "tokens")


def _flat(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    return value.detach().float().reshape(batch_size, -1)


def candidate_world_features(
    rollout: dict[str, torch.Tensor],
    projection: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pool one candidate rollout into compact head, geometry, and token features."""

    pred = rollout["pred_tokens"].float()
    if pred.ndim != 4:
        raise ValueError(f"pred_tokens must be [B,T,P,D], got {tuple(pred.shape)}")
    batch_size = pred.shape[0]
    if projection.shape[0] != pred.shape[-1]:
        raise ValueError(
            f"projection input dim {projection.shape[0]} != token dim {pred.shape[-1]}"
        )

    progress = rollout["progress"].float()
    terminal = rollout["terminal_success_logit"].float().reshape(batch_size, -1)
    plausibility = rollout["plausibility_logit"].float().reshape(batch_size, -1)
    heads = torch.cat([_flat(progress, batch_size), terminal, plausibility], dim=1)

    depth = rollout["depth"].float()
    point = rollout["point"].float()
    z_a = rollout["z_a"].float()
    geometry_parts = [
        _flat(rollout["pose"], batch_size),
        _flat(rollout["pose_geom"], batch_size),
        _flat(depth.mean(dim=(-2, -1)), batch_size),
        _flat(depth.std(dim=(-2, -1), unbiased=False), batch_size),
        _flat(point.mean(dim=(-3, -2)), batch_size),
        _flat(point.std(dim=(-3, -2), unbiased=False), batch_size),
        _flat(z_a.mean(dim=-1), batch_size),
        _flat(z_a.std(dim=-1, unbiased=False), batch_size),
    ]
    geometry = torch.cat(geometry_parts, dim=1)

    token_mean = pred.mean(dim=2)
    tokens = _flat(token_mean @ projection.to(token_mean), batch_size)
    return {"heads": heads, "geometry": geometry, "tokens": tokens}


def action_features(candidates: torch.Tensor) -> torch.Tensor:
    """Candidate-local action control, including deviation from candidate zero."""

    if candidates.ndim != 4 or candidates.shape[-1] != 7:
        raise ValueError(f"candidates must be [B,K,T,7], got {tuple(candidates.shape)}")
    delta = candidates.float() - candidates[:, :1].float()
    pose_abs = candidates[..., :6].float().abs().mean(dim=(2, 3), keepdim=False)[..., None]
    delta_abs = delta[..., :6].abs().mean(dim=(2, 3), keepdim=False)[..., None]
    return torch.cat(
        [
            candidates.float().flatten(2),
            delta.flatten(2),
            pose_abs,
            delta_abs,
        ],
        dim=2,
    )


def _projection(input_dim: int, output_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    matrix.div_(math.sqrt(max(1, output_dim)))
    return matrix.to(device=device)


def _dataset_indices(args: argparse.Namespace, size: int) -> list[int]:
    if args.index_mode == "contiguous":
        stop = min(size, args.start + args.count)
        return list(range(args.start, stop))
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if args.index_mode == "contiguous_sharded":
        total = max(0, min(int(args.count), size - int(args.start)))
        base, remainder = divmod(total, int(args.num_shards))
        shard_count = base + int(args.shard_index < remainder)
        offset = args.shard_index * base + min(args.shard_index, remainder)
        shard_start = int(args.start) + offset
        return list(range(shard_start, shard_start + shard_count))
    return list(range(args.shard_index, size, args.num_shards))[: args.count]


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(args.cfg.read_text())
    torch.manual_seed(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    dataset = build_dataset_for_split(
        read_manifest(cfg["data"]["manifest"]),
        cfg,
        split=args.split,
    )
    indices = _dataset_indices(args, len(dataset))
    if not indices:
        raise RuntimeError("feature extraction selected no dataset rows")
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
        key: [] for key in (*FEATURE_KEYS, "pose_l1", "grip_bce", "oracle_error")
    }
    row_ids: list[tuple[str, int]] = []
    source_indices: list[int] = []
    projection: torch.Tensor | None = None
    seen = 0
    for batch in loader:
        s = batch["s_in"].to(device, non_blocking=True)
        task = batch["c"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        context_rgb = (
            batch["rgb_in"][:, -1]
            .to(device, non_blocking=True)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            decision = select_action_chunk(
                model,
                s,
                task,
                context_rgb=context_rgb,
                pixel=False,
                score_weights=ScoreWeights(progress=0.0, terminal=0.0, plausibility=1.0),
                return_rollouts=True,
            )
        candidates = decision["candidate_action_cond"].float()
        rollouts = decision["candidate_rollouts"]
        if projection is None:
            token_dim = int(rollouts[0]["pred_tokens"].shape[-1])
            projection = _projection(token_dim, args.projection_dim, args.projection_seed, device)

        per_candidate = [candidate_world_features(out, projection) for out in rollouts]
        for key in ("heads", "geometry", "tokens"):
            chunks[key].append(torch.stack([item[key] for item in per_candidate], dim=1).cpu())
        chunks["action"].append(action_features(candidates).cpu())

        pose_l1 = (
            candidates[..., :6] - action_tgt_norm.float()[:, None]
        ).abs().mean(dim=(2, 3))
        grip_target = (action_tgt[..., 6] > 0.5).float()
        grip_probability = candidates[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
        grip_bce = F.binary_cross_entropy(
            grip_probability,
            grip_target[:, None].expand_as(grip_probability),
            reduction="none",
        ).mean(dim=2)
        chunks["pose_l1"].append(pose_l1.cpu())
        chunks["grip_bce"].append(grip_bce.cpu())
        chunks["oracle_error"].append((pose_l1 + args.grip_weight * grip_bce).cpu())

        batch_size = candidates.shape[0]
        row_ids.extend(
            (str(batch["clip_id"][row]), int(batch["start"][row]))
            for row in range(batch_size)
        )
        source_indices.extend(indices[seen : seen + batch_size])
        seen += batch_size
        if seen % 32 == 0 or seen == len(indices):
            print(f"extracted {seen}/{len(indices)}", flush=True)
        del decision, rollouts, per_candidate

    payload: dict[str, Any] = {
        key: torch.cat(value, dim=0) for key, value in chunks.items()
    }
    payload.update(
        {
            "ids": row_ids,
            "source_indices": source_indices,
            "cfg": str(args.cfg),
            "ckpt": str(args.ckpt),
            "split": args.split,
            "projection_dim": int(args.projection_dim),
            "projection_seed": int(args.projection_seed),
            "index_mode": args.index_mode,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    dims = {key: list(payload[key].shape) for key in chunks}
    print(json.dumps({"out": str(args.out), "rows": len(row_ids), "shapes": dims}, indent=2))


def _load_feature_dir(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no feature shards in {path}")
    parts = [torch.load(file, map_location="cpu", weights_only=False) for file in files]
    keys = (*FEATURE_KEYS, "pose_l1", "grip_bce", "oracle_error")
    payload: dict[str, Any] = {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}
    payload["ids"] = [tuple(item) for part in parts for item in part["ids"]]
    return payload


def _deduplicate(payload: dict[str, Any]) -> dict[str, Any]:
    keep: list[int] = []
    seen: set[tuple[str, int]] = set()
    for index, row_id in enumerate(payload["ids"]):
        if row_id not in seen:
            seen.add(row_id)
            keep.append(index)
    index_tensor = torch.tensor(keep, dtype=torch.long)
    out = {key: payload[key].index_select(0, index_tensor) for key in (*FEATURE_KEYS, "pose_l1", "grip_bce", "oracle_error")}
    out["ids"] = [payload["ids"][index] for index in keep]
    return out


def _pairwise_ridge(features: torch.Tensor, target: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    flat = features.reshape(-1, features.shape[-1]).double()
    scale = flat.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    normalized = features.double() / scale
    diffs = []
    target_diffs = []
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            diffs.append(normalized[:, left] - normalized[:, right])
            target_diffs.append(target[:, left].double() - target[:, right].double())
    design = torch.cat(diffs, dim=0)
    response = torch.cat(target_diffs, dim=0)
    gram = design.T @ design
    gram.diagonal().add_(float(ridge))
    weight = torch.linalg.solve(gram, design.T @ response)
    return weight, scale


def _rank_metrics(features: torch.Tensor, pose_l1: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> dict[str, float | int]:
    predicted_error = (features.double() / scale) @ weight
    selected_idx = predicted_error.argmin(dim=1)
    selected = pose_l1.gather(1, selected_idx[:, None]).squeeze(1).float()
    anchor = pose_l1[:, 0].float()
    oracle, oracle_idx = pose_l1.min(dim=1)
    paired = paired_bootstrap_mean_ci(anchor - selected, seed=1729, samples=10000)
    return {
        "rows": int(pose_l1.shape[0]),
        "selected_pose_l1": float(selected.mean()),
        "anchor_pose_l1": float(anchor.mean()),
        "oracle_pose_l1": float(oracle.mean()),
        "relative_improvement": float((anchor.mean() - selected.mean()) / anchor.mean().clamp_min(1.0e-12)),
        "ci95_lower": float(paired["ci95_lower"]),
        "ci95_upper": float(paired["ci95_upper"]),
        "fraction_improved": float(paired["fraction_improved"]),
        "oracle_match": float((selected_idx == oracle_idx).float().mean()),
        "candidate0_oracle_fraction": float((oracle_idx == 0).float().mean()),
        "selected_candidate0_fraction": float((selected_idx == 0).float().mean()),
    }


def _feature_sets(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    world = torch.cat([payload["heads"], payload["geometry"], payload["tokens"]], dim=2)
    return {
        "action_only": payload["action"],
        "heads_only": payload["heads"],
        "geometry_only": payload["geometry"],
        "tokens_only": payload["tokens"],
        "world_only": world,
        "world_plus_action": torch.cat([world, payload["action"]], dim=2),
    }


def fit(args: argparse.Namespace) -> None:
    train_parts = [_load_feature_dir(path) for path in args.train_dirs]
    train = {
        key: torch.cat([part[key] for part in train_parts], dim=0)
        for key in (*FEATURE_KEYS, "pose_l1", "grip_bce", "oracle_error")
    }
    train["ids"] = [row_id for part in train_parts for row_id in part["ids"]]
    train = _deduplicate(train)
    train_sets = _feature_sets(train)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    permutation = torch.randperm(len(train["ids"]), generator=generator)
    tune_count = max(1, int(len(permutation) * args.tune_fraction))
    tune_index = permutation[:tune_count]
    fit_index = permutation[tune_count:]
    if fit_index.numel() == 0:
        raise RuntimeError("not enough train rows for fit/tune split")

    val_groups: dict[str, dict[str, Any]] = {}
    for spec in args.val:
        name, raw_path = spec.split("=", 1)
        val_groups[name] = _deduplicate(_load_feature_dir(Path(raw_path)))

    report: dict[str, Any] = {
        "train_rows": len(train["ids"]),
        "tune_rows": int(tune_index.numel()),
        "feature_sets": {},
    }
    for name, features in train_sets.items():
        best_ridge = None
        best_tune = None
        tuning = []
        for ridge in args.ridge:
            weight, scale = _pairwise_ridge(features[fit_index], train["pose_l1"][fit_index], ridge)
            metrics = _rank_metrics(features[tune_index], train["pose_l1"][tune_index], weight, scale)
            tuning.append({"ridge": float(ridge), "metrics": metrics})
            if best_tune is None or metrics["selected_pose_l1"] < best_tune:
                best_tune = float(metrics["selected_pose_l1"])
                best_ridge = float(ridge)
        assert best_ridge is not None
        weight, scale = _pairwise_ridge(features, train["pose_l1"], best_ridge)
        evaluations = {}
        for group_name, payload in val_groups.items():
            val_features = _feature_sets(payload)[name]
            evaluations[group_name] = _rank_metrics(val_features, payload["pose_l1"], weight, scale)
        report["feature_sets"][name] = {
            "feature_dim": int(features.shape[-1]),
            "selected_ridge": best_ridge,
            "tuning": tuning,
            "validation": evaluations,
        }
        print(name, json.dumps(evaluations, sort_keys=True), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--cfg", type=Path, required=True)
    extract_parser.add_argument("--ckpt", type=Path, required=True)
    extract_parser.add_argument("--out", type=Path, required=True)
    extract_parser.add_argument("--split", choices=("train", "val"), required=True)
    extract_parser.add_argument(
        "--index_mode",
        choices=("contiguous", "contiguous_sharded", "strided"),
        default="contiguous",
    )
    extract_parser.add_argument("--start", type=int, default=0)
    extract_parser.add_argument("--count", type=int, required=True)
    extract_parser.add_argument("--shard_index", type=int, default=0)
    extract_parser.add_argument("--num_shards", type=int, default=1)
    extract_parser.add_argument("--batch_size", type=int, default=1)
    extract_parser.add_argument("--num_workers", type=int, default=0)
    extract_parser.add_argument("--projection_dim", type=int, default=32)
    extract_parser.add_argument("--projection_seed", type=int, default=1729)
    extract_parser.add_argument("--grip_weight", type=float, default=0.1)
    extract_parser.add_argument("--seed", type=int, default=1729)

    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--train_dirs", type=Path, nargs="+", required=True)
    fit_parser.add_argument("--val", action="append", required=True, help="NAME=FEATURE_DIR")
    fit_parser.add_argument("--out", type=Path, required=True)
    fit_parser.add_argument("--ridge", type=float, nargs="+", default=(1.0e-4, 1.0e-2, 1.0, 100.0))
    fit_parser.add_argument("--tune_fraction", type=float, default=0.2)
    fit_parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        extract(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
