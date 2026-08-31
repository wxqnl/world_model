"""Train the native ProgressHead to rank safe residual action candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm3d_v3.models.progress_head import ProgressHead, ProgressHeadConfig


def _load_dataset(features: Path, candidates: Path, stats_path: Path) -> dict[str, Any]:
    token_parts, action_parts, task_parts, expert_parts, rows = [], [], [], [], []
    for candidate_path in sorted(candidates.glob("shard_*.npz")):
        feature_path = features / candidate_path.name
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        candidate = np.load(candidate_path)
        feature = np.load(feature_path)
        if not np.array_equal(candidate["rows_json"], feature["rows_json"]):
            raise ValueError(f"row mismatch: {candidate_path} vs {feature_path}")
        if not np.allclose(candidate["candidate_cond"], feature["candidate_cond"]):
            raise ValueError(f"candidate mismatch: {candidate_path} vs {feature_path}")
        token_parts.append(np.asarray(feature["world_token_mean"], dtype=np.float32))
        action_parts.append(np.asarray(candidate["candidate_cond"], dtype=np.float32))
        task_parts.append(np.asarray(feature["task_emb"], dtype=np.float32))
        expert_parts.append(np.asarray(candidate["expert_action"], dtype=np.float32))
        rows.extend(json.loads(str(item)) for item in candidate["rows_json"])
    if not token_parts:
        raise RuntimeError(f"no candidate shards in {candidates}")
    stats = np.load(stats_path)
    mean = np.asarray(stats["mean"][:6], dtype=np.float32)
    std = np.maximum(np.asarray(stats["std"][:6], dtype=np.float32), 1e-6)
    actions = np.concatenate(action_parts)
    expert = np.concatenate(expert_parts)
    expert_norm_pose = (expert[..., :6] - mean) / std
    pose_error = np.abs(actions[..., :6] - expert_norm_pose[:, None]).mean(axis=(2, 3))
    grip_target = (expert[..., 6] > 0.5).astype(np.float32)
    grip_prob = np.clip(actions[..., 6], 1e-5, 1.0 - 1e-5)
    grip_bce = -(
        grip_target[:, None] * np.log(grip_prob)
        + (1.0 - grip_target[:, None]) * np.log(1.0 - grip_prob)
    ).mean(axis=2)
    grip_l1 = np.abs(np.clip(actions[..., 6], 0.0, 1.0) - grip_target[:, None])
    action_l1 = np.concatenate(
        [np.abs(actions[..., :6] - expert_norm_pose[:, None]), grip_l1[..., None]],
        axis=-1,
    ).mean(axis=(2, 3))
    return {
        "tokens": torch.from_numpy(np.concatenate(token_parts)),
        "actions": torch.from_numpy(actions),
        "tasks": torch.from_numpy(np.concatenate(task_parts)),
        "quality": torch.from_numpy(pose_error + 0.1 * grip_bce),
        "pose_l1": torch.from_numpy(pose_error),
        "action_l1": torch.from_numpy(action_l1),
        "expert": torch.from_numpy(expert),
        "mean": mean,
        "std": std,
        "rows": rows,
    }


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task_name"]), str(row["demo_id"])


def _split(rows: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    groups = sorted({_group_key(row) for row in rows})
    ordered = sorted(
        groups,
        key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest(),
    )
    calibration_groups = set(ordered[: max(1, round(0.2 * len(ordered)))])
    fit = [i for i, row in enumerate(rows) if _group_key(row) not in calibration_groups]
    calibration = [i for i, row in enumerate(rows) if _group_key(row) in calibration_groups]
    return fit, calibration


def _build_head(base_cfg: dict, checkpoint: Path, device: torch.device) -> ProgressHead:
    cfg = base_cfg["model"]
    head = ProgressHead(
        ProgressHeadConfig(
            token_dim=2048,
            hidden=int(cfg.get("progress_hidden", 256)),
            n_layers=int(cfg.get("progress_layers", 2)),
            n_heads=int(cfg.get("progress_heads", 4)),
            action_dim=int(cfg.get("progress_action_dim", 7)),
            task_dim=int(cfg.get("progress_task_dim") or 2048),
            max_horizon=int(cfg.get("progress_max_horizon", 32)),
            use_action=bool(cfg.get("progress_use_action", True)),
            use_task=bool(cfg.get("progress_use_task", True)),
        )
    )
    state = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )["model"]
    prefix = "progress_head."
    head.load_state_dict(
        {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)},
        strict=True,
    )
    return head.to(device)


def _set_trainable(head: ProgressHead, mode: str) -> int:
    for parameter in head.parameters():
        parameter.requires_grad = mode == "all"
    if mode in {"output", "output_norm"}:
        for module in (head.progress_head, head.terminal_success_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
    if mode == "output_norm":
        for parameter in head.norm.parameters():
            parameter.requires_grad = True
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def _scores(head: ProgressHead, tokens, actions, tasks) -> torch.Tensor:
    batch, candidates, horizon, dim = tokens.shape
    out = head(
        tokens.reshape(batch * candidates, horizon, dim).unsqueeze(2),
        action_cond=actions.reshape(batch * candidates, horizon, 7),
        task_emb=tasks[:, None]
        .expand(-1, candidates, -1)
        .reshape(batch * candidates, -1),
    )
    terminal = out["terminal_success_logit"].reshape(batch, candidates)
    progress = out["progress"].mean(dim=1).reshape(batch, candidates)
    return terminal + progress


def _loss(head, data, indices, device, temperature: float, margin: float) -> torch.Tensor:
    tokens = data["tokens"][indices].to(device)
    actions = data["actions"][indices].to(device)
    tasks = data["tasks"][indices].to(device)
    quality = data["quality"][indices].to(device)
    scores = _scores(head, tokens, actions, tasks)
    oracle = quality.argmin(dim=1)
    quality_gap = quality[:, None, :] - quality[:, :, None]
    ordered = quality_gap > 1e-4
    score_gap = scores[:, :, None] - scores[:, None, :]
    pairwise = (
        torch.relu(margin - score_gap[ordered]).mean()
        if bool(ordered.any().item())
        else scores.new_zeros(())
    )
    return pairwise + F.cross_entropy(scores / max(temperature, 1e-4), oracle)


@torch.no_grad()
def _all_scores(head, data, indices, device) -> torch.Tensor:
    head.eval()
    parts = []
    for start in range(0, len(indices), 64):
        batch = indices[start : start + 64]
        parts.append(
            _scores(
                head,
                data["tokens"][batch].to(device),
                data["actions"][batch].to(device),
                data["tasks"][batch].to(device),
            ).cpu()
        )
    return torch.cat(parts)


def _cluster_ci(
    values: np.ndarray,
    rows: list[dict[str, Any]],
    seed: int,
    draws: int = 2000,
) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for value, row in zip(values.tolist(), rows, strict=True):
        grouped.setdefault(_group_key(row), []).append(float(value))
    groups = [np.asarray(item, dtype=np.float64) for item in grouped.values()]
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        boot[index] = np.concatenate([groups[i] for i in sampled]).mean()
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def _select(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    best = scores.argmax(dim=1)
    gap = scores.gather(1, best[:, None]).squeeze(1) - scores[:, 0]
    return torch.where(
        (best != 0) & (gap > threshold),
        best,
        torch.zeros_like(best),
    )


def _metrics(data, indices, scores, threshold: float, seed: int) -> dict[str, Any]:
    selected = _select(scores, threshold)
    action = data["action_l1"][indices]
    pose = data["pose_l1"][indices]
    quality = data["quality"][indices]
    chosen_action = action.gather(1, selected[:, None]).squeeze(1)
    chosen_pose = pose.gather(1, selected[:, None]).squeeze(1)
    chosen_quality = quality.gather(1, selected[:, None]).squeeze(1)
    delta = (action[:, 0] - chosen_action).numpy()
    rows = [data["rows"][i] for i in indices]
    counts = torch.bincount(selected, minlength=action.shape[1]).tolist()
    anchor_mean = float(action[:, 0].mean())
    return {
        "rows": len(indices),
        "groups": len({_group_key(row) for row in rows}),
        "threshold": float(threshold),
        "selection_counts": counts,
        "nonanchor_rate": float((selected != 0).float().mean()),
        "anchor_action_l1": anchor_mean,
        "selected_action_l1": float(chosen_action.mean()),
        "oracle_action_l1": float(action.min(dim=1).values.mean()),
        "action_improvement": float(np.mean(delta)),
        "action_relative_improvement": float(np.mean(delta) / max(anchor_mean, 1e-12)),
        "action_improvement_ci95": _cluster_ci(delta, rows, seed),
        "anchor_pose_l1": float(pose[:, 0].mean()),
        "selected_pose_l1": float(chosen_pose.mean()),
        "oracle_pose_l1": float(pose.min(dim=1).values.mean()),
        "anchor_quality": float(quality[:, 0].mean()),
        "selected_quality": float(chosen_quality.mean()),
        "oracle_quality": float(quality.min(dim=1).values.mean()),
    }


def _calibrate(data, indices, scores, seed: int) -> tuple[float, dict[str, Any]]:
    best = scores.argmax(dim=1)
    gaps = (
        scores.gather(1, best[:, None]).squeeze(1) - scores[:, 0]
    ).numpy()
    positive = gaps[gaps > 0]
    thresholds = [0.0]
    if positive.size:
        thresholds.extend(
            float(value)
            for value in np.unique(np.quantile(positive, np.linspace(0.05, 0.95, 19)))
        )
    reports = [
        _metrics(data, indices, scores, threshold, seed)
        for threshold in thresholds
    ]
    eligible = [
        report
        for report in reports
        if report["action_improvement_ci95"][0] > 0.0
    ]
    chosen = max(
        eligible or reports,
        key=lambda report: (report["action_improvement"], -report["threshold"]),
    )
    return float(chosen["threshold"]), {"chosen": chosen, "candidates": reports}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--head_ckpt", type=Path)
    ap.add_argument("--train_features", type=Path, required=True)
    ap.add_argument("--train_candidates", type=Path, required=True)
    ap.add_argument("--eval_features", type=Path)
    ap.add_argument("--eval_candidates", type=Path)
    ap.add_argument("--action_stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--pairwise_margin", type=float, default=0.05)
    ap.add_argument("--trainable", choices=("output", "output_norm", "all"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split_seed", type=int, default=1729)
    ap.add_argument("--train_all", action="store_true")
    ap.add_argument("--selection_threshold", type=float)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    wrapper = yaml.safe_load(args.cfg.read_text())
    base_cfg = yaml.safe_load(Path(wrapper["base_cfg"]).read_text())
    data = _load_dataset(
        args.train_features,
        args.train_candidates,
        args.action_stats,
    )
    if args.train_all:
        fit_indices = list(range(len(data["rows"])))
        calibration_indices = fit_indices
    else:
        fit_indices, calibration_indices = _split(data["rows"], args.split_seed)
    head = _build_head(base_cfg, args.ckpt, device)
    if args.head_ckpt is not None:
        saved_head = torch.load(
            args.head_ckpt,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        head.load_state_dict(saved_head["model"], strict=True)
    trainable_parameters = _set_trainable(head, args.trainable)
    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    for _step in range(args.steps):
        head.train()
        sampled = torch.randint(
            len(fit_indices),
            (args.batch_size,),
            generator=generator,
        )
        batch = [fit_indices[index] for index in sampled.tolist()]
        loss = _loss(
            head,
            data,
            batch,
            device,
            args.temperature,
            args.pairwise_margin,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
    calibration_scores = _all_scores(
        head,
        data,
        calibration_indices,
        device,
    )
    if args.selection_threshold is None:
        threshold, calibration = _calibrate(
            data,
            calibration_indices,
            calibration_scores,
            args.seed,
        )
    else:
        threshold = args.selection_threshold
        calibration = {
            "chosen": _metrics(
                data,
                calibration_indices,
                calibration_scores,
                threshold,
                args.seed,
            )
        }
    report: dict[str, Any] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "trainable_parameters": trainable_parameters,
        "fit_rows": len(fit_indices),
        "calibration": calibration,
        "selection_threshold": threshold,
    }
    if args.eval_features and args.eval_candidates:
        evaluation = _load_dataset(
            args.eval_features,
            args.eval_candidates,
            args.action_stats,
        )
        evaluation_indices = list(range(len(evaluation["rows"])))
        evaluation_scores = _all_scores(
            head,
            evaluation,
            evaluation_indices,
            device,
        )
        report["heldout"] = _metrics(
            evaluation,
            evaluation_indices,
            evaluation_scores,
            threshold,
            args.seed + 1,
        )
        adjusted_scores = evaluation_scores.numpy().astype(np.float32)
        adjusted_scores[:, 0] += float(threshold)
        candidate_cond = evaluation["actions"].numpy().astype(np.float32)
        candidate_raw = np.concatenate(
            [
                candidate_cond[..., :6] * evaluation["std"] + evaluation["mean"],
                candidate_cond[..., 6:7],
            ],
            axis=-1,
        )
        np.savez_compressed(
            args.out / "heldout_simulator_payload.npz",
            candidate_cond=candidate_cond,
            candidate_raw=candidate_raw.astype(np.float32),
            candidate_score=adjusted_scores,
            expert_action=evaluation["expert"].numpy().astype(np.float32),
            rows_json=np.asarray(
                [json.dumps(row, sort_keys=True) for row in evaluation["rows"]],
                dtype=np.str_,
            ),
        )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": head.state_dict(), "report": report},
        args.out / "judge.pt",
    )
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
