#!/usr/bin/env python3
"""Offline native-dynamics and planner evaluation for V7 Stage1-P."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from wm3d_v3.stage1_planner.dataset import Stage1BranchDataset, Stage1BranchDatasetConfig  # noqa: E402
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig  # noqa: E402
from wm3d_v3.stage1_planner.train import _load_stage1_state, sha256_file  # noqa: E402
from wm3d_v3.training.train import build_model, config_sha256, load_train_config  # noqa: E402


SCHEMA = "wm3d_v7_stage1_planner_offline_eval_v1"


def _move(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    return value


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = ranks[labels].sum()
    return float((positive_rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def _bootstrap_ci(values: Iterable[float], *, samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        estimates[index] = rng.choice(array, size=len(array), replace=True).mean()
    return {
        "mean": float(array.mean()),
        "lower95": float(np.quantile(estimates, 0.025)),
        "upper95": float(np.quantile(estimates, 0.975)),
        "roots": int(len(array)),
    }


def _selection_metrics(
    scores: np.ndarray,
    success: np.ndarray,
    *,
    direct_index: int = 1,
) -> dict[str, float]:
    # Candidate zero is factual teacher evidence and is never selectable.
    candidate_scores = scores[:, 1:]
    selected = 1 + candidate_scores.argmax(axis=1)
    row = np.arange(len(scores))
    selected_success = success[row, selected].astype(np.float64)
    direct_success = success[:, direct_index].astype(np.float64)
    oracle_success = success[:, 1:].max(axis=1).astype(np.float64)
    mixed = success[:, 1:].any(axis=1) & ~success[:, 1:].all(axis=1)
    result = {
        "success_at1": float(selected_success.mean()),
        "direct_success": float(direct_success.mean()),
        "success_at1_uplift": float((selected_success - direct_success).mean()),
        "oracle_success": float(oracle_success.mean()),
        "candidate_oracle_uplift": float((oracle_success - direct_success).mean()),
        "mixed_roots": int(mixed.sum()),
    }
    if mixed.any():
        result.update(
            {
                "mixed_success_at1": float(selected_success[mixed].mean()),
                "mixed_direct_success": float(direct_success[mixed].mean()),
                "mixed_success_at1_uplift": float(
                    (selected_success[mixed] - direct_success[mixed]).mean()
                ),
                "mixed_candidate_oracle_uplift": float(
                    (oracle_success[mixed] - direct_success[mixed]).mean()
                ),
            }
        )
    else:
        result.update(
            {
                "mixed_success_at1": None,
                "mixed_direct_success": None,
                "mixed_success_at1_uplift": None,
                "mixed_candidate_oracle_uplift": None,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--overlay-sha256", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.overlay) != args.overlay_sha256:
        raise SystemExit("Stage1 overlay SHA256 mismatch")
    cfg = load_train_config(args.cfg)
    phase_cfg = dict(cfg["planner_stage"])
    data_cfg = dict(cfg["planner_data"])
    source_path = Path(phase_cfg["source_checkpoint"])
    source_sha = sha256_file(source_path)
    if source_sha != phase_cfg["source_checkpoint_sha256"]:
        raise SystemExit("Stage0 checkpoint SHA256 mismatch")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    world = build_model(cfg)
    loaded = world.load_state_dict(source["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Stage0 strict load was not clean")
    system = NativePlanningSystem(
        world,
        Stage1SystemConfig(
            candidate_microbatch=int(phase_cfg.get("candidate_microbatch", 1)),
            activation_checkpointing=False,
        ),
    )
    overlay = torch.load(args.overlay, map_location="cpu", weights_only=False)
    if overlay.get("source_checkpoint_sha256") != source_sha:
        raise RuntimeError("overlay is bound to another Stage0 checkpoint")
    _load_stage1_state(system, overlay["stage1_state"])
    device = torch.device(args.device)
    system.to(device).eval()
    dataset = Stage1BranchDataset(
        Stage1BranchDatasetConfig(
            index_path=Path(data_cfg["index"]),
            split=args.split,
            action_stats=Path(data_cfg["action_stats"]),
            context_frames=16,
            future_frames=32,
            action_history_len=4,
        )
    )
    if args.max_roots > 0:
        dataset.records = dataset.records[: args.max_roots]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    effect_by_horizon: dict[int, list[float]] = {8: [], 16: [], 32: []}
    true_scores, imagined_scores, terminal_success = [], [], []
    true_success_logits, imagined_success_logits, flat_labels = [], [], []
    depth_errors, point_errors, pose_errors = [], [], []
    with torch.inference_mode():
        for batch in loader:
            moved = {key: _move(value, device) for key, value in batch.items()}
            result = system(
                moved["s_in"],
                moved["c"],
                moved["candidate_actions"],
                wrist=moved["s_wrist"],
                view_mask=moved["view_mask"],
                score_planner=True,
                true_future_codec=moved["branch_s_tgt_codec"],
                true_depth=moved["branch_depth_tgt"],
                true_point=moved["branch_point_tgt"],
                true_pose=moved["branch_pose_geom_tgt"],
            )
            truth = system.world.decode_input_tokens(
                moved["branch_s_tgt_codec"].flatten(0, 1)
            ).unflatten(0, moved["branch_s_tgt_codec"].shape[:2])
            predicted = result["rollout"].tokens.float()
            pred_effect = predicted - predicted[:, :1]
            true_effect = truth.float() - truth[:, :1]
            for horizon in effect_by_horizon:
                pred_h = pred_effect[:, 1:, :horizon]
                true_h = true_effect[:, 1:, :horizon]
                error = (pred_h - true_h).square().mean(dim=(-1, -2, -3))
                energy = true_h.square().mean(dim=(-1, -2, -3)).clamp_min(1.0e-6)
                effect_by_horizon[horizon].append(float((1.0 - error / energy).mean().cpu()))
            rollout = result["rollout"]
            bsz, candidates, horizon, height, width = rollout.depth.shape
            depth_pred = F.adaptive_avg_pool2d(
                rollout.depth.reshape(bsz * candidates * horizon, 1, height, width),
                moved["branch_depth_tgt"].shape[-2:],
            ).reshape_as(moved["branch_depth_tgt"])
            points = rollout.point
            point_pred = F.adaptive_avg_pool2d(
                points.permute(0, 1, 2, 5, 3, 4).reshape(
                    bsz * candidates * horizon, 3, points.shape[-3], points.shape[-2]
                ),
                moved["branch_point_tgt"].shape[-3:-1],
            ).reshape(
                bsz, candidates, horizon, 3, *moved["branch_point_tgt"].shape[-3:-1]
            ).permute(0, 1, 2, 4, 5, 3)
            depth_errors.append(float((torch.log1p(depth_pred.clamp_min(0)) - torch.log1p(moved["branch_depth_tgt"].clamp_min(0))).abs().mean().cpu()))
            point_errors.append(float((point_pred - moved["branch_point_tgt"]).abs().mean().cpu()))
            pose_errors.append(float((rollout.pose - moved["branch_pose_geom_tgt"]).abs().mean().cpu()))
            success = moved["branch_success"].any(dim=-1).cpu().numpy().astype(np.bool_)
            terminal_success.append(success[0])
            true_scores.append(result["true_planner"]["score"][0].float().cpu().numpy())
            imagined_scores.append(result["planner"]["score"][0].float().cpu().numpy())
            flat_labels.append(success[0, 1:])
            true_success_logits.append(result["true_planner"]["success_logit"][0, 1:].float().cpu().numpy())
            imagined_success_logits.append(result["planner"]["success_logit"][0, 1:].float().cpu().numpy())

    terminal = np.stack(terminal_success)
    true_score = np.stack(true_scores)
    imagined_score = np.stack(imagined_scores)
    labels = np.concatenate(flat_labels)
    true_selection = _selection_metrics(true_score, terminal)
    imagined_selection = _selection_metrics(imagined_score, terminal)
    true_uplift = true_selection["mixed_success_at1_uplift"]
    imagined_uplift = imagined_selection["mixed_success_at1_uplift"]
    retention = (
        float(imagined_uplift / true_uplift)
        if true_uplift is not None and imagined_uplift is not None and true_uplift > 0
        else None
    )
    report = {
        "schema": SCHEMA,
        "passed_execution": True,
        "config": str(args.cfg.resolve()),
        "config_sha256": config_sha256(cfg),
        "source_checkpoint_sha256": source_sha,
        "overlay": str(args.overlay.resolve()),
        "overlay_sha256": args.overlay_sha256,
        "overlay_phase": overlay.get("phase"),
        "overlay_step": overlay.get("step"),
        "split": args.split,
        "roots": len(dataset),
        "native_dynamics": {
            f"effect_gain_h{horizon}": _bootstrap_ci(
                values, samples=args.bootstrap_samples, seed=27081 + horizon
            )
            for horizon, values in effect_by_horizon.items()
        },
        "native_geometry_mae": {
            "depth_log": float(np.mean(depth_errors)),
            "point": float(np.mean(point_errors)),
            "pose": float(np.mean(pose_errors)),
        },
        "planner_true_future": {
            "success_auc": _auc(labels, np.concatenate(true_success_logits)),
            **true_selection,
        },
        "planner_imagined_future": {
            "success_auc": _auc(labels, np.concatenate(imagined_success_logits)),
            "uplift_retention_vs_true": retention,
            **imagined_selection,
        },
        "candidate_zero_excluded_from_selection": True,
        "planner_action_inputs": False,
        "future_observation_inputs": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    report["report_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
