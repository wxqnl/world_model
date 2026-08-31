"""Evaluate the serving plausibility judge on fixed real simulator branches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wm3d_v3.models.progress_head import ProgressHead, ProgressHeadConfig


def _row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["task_name"]), str(row["demo_id"]), int(row["target_start"])


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["task_name"]), str(row["demo_id"])


def _load_labels(branch_dir: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    labels: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(branch_dir.glob("shard*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            factual = next(branch for branch in item["branches"] if branch["candidate_index"] == -1)
            candidates = sorted(
                (branch for branch in item["branches"] if branch["candidate_index"] >= 0),
                key=lambda branch: int(branch["candidate_index"]),
            )
            labels[_row_key(item["row"])] = {
                "factual_success": bool(factual["success"]),
                "success": np.asarray([float(branch["success"]) for branch in candidates], dtype=np.float32),
                "post": np.asarray([float(branch["post_state_l1"]) for branch in candidates], dtype=np.float32),
            }
    if not labels:
        raise RuntimeError(f"no real branch labels found in {branch_dir}")
    return labels


def _load_split(feature_dir: Path, labels: dict[tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    token_parts, action_parts, task_parts = [], [], []
    success_parts, post_parts, valid_parts, rows = [], [], [], []
    for path in sorted(feature_dir.glob("shard*.npz")):
        payload = np.load(path)
        shard_rows = [json.loads(str(item)) for item in payload["rows_json"]]
        token_parts.append(np.asarray(payload["world_token_mean"], dtype=np.float32))
        action_parts.append(np.asarray(payload["candidate_cond"], dtype=np.float32))
        task_parts.append(np.asarray(payload["task_emb"], dtype=np.float32))
        for row in shard_rows:
            label = labels[_row_key(row)]
            success_parts.append(label["success"])
            post_parts.append(label["post"])
            valid_parts.append(label["factual_success"])
            rows.append(row)
    if not token_parts:
        raise RuntimeError(f"no feature shards found in {feature_dir}")
    data = {
        "tokens": torch.from_numpy(np.concatenate(token_parts)),
        "actions": torch.from_numpy(np.concatenate(action_parts)),
        "tasks": torch.from_numpy(np.concatenate(task_parts)),
        "success": torch.from_numpy(np.asarray(success_parts, dtype=np.float32)),
        "post": torch.from_numpy(np.asarray(post_parts, dtype=np.float32)),
        "valid": torch.from_numpy(np.asarray(valid_parts, dtype=np.bool_)),
        "rows": rows,
    }
    if data["tokens"].shape[1] != 9:
        raise ValueError(f"expected exact K9 features, got {tuple(data['tokens'].shape)}")
    return data


def _resolve_model_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if "model" in cfg:
        return cfg["model"]
    base_path = Path(cfg["base_cfg"])
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return yaml.safe_load(base_path.read_text())["model"]


def _build_head(model_cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[ProgressHead, dict[str, Any]]:
    head = ProgressHead(
        ProgressHeadConfig(
            token_dim=2048,
            hidden=int(model_cfg.get("progress_hidden", 256)),
            n_layers=int(model_cfg.get("progress_layers", 2)),
            n_heads=int(model_cfg.get("progress_heads", 4)),
            action_dim=int(model_cfg.get("progress_action_dim", 7)),
            task_dim=int(model_cfg.get("progress_task_dim") or 2048),
            max_horizon=int(model_cfg.get("progress_max_horizon", 32)),
            use_action=bool(model_cfg.get("progress_use_action", True)),
            use_task=bool(model_cfg.get("progress_use_task", True)),
        )
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = payload["model"]
    prefix = "progress_head."
    judge_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    head.load_state_dict(judge_state, strict=True)
    digest = hashlib.sha256()
    for key, value in sorted(judge_state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_step": payload.get("global_step", payload.get("step")),
        "judge_state_sha256": digest.hexdigest(),
        "use_action": head.cfg.use_action,
        "use_task": head.cfg.use_task,
    }
    del payload, state, judge_state
    return head.to(device).eval(), meta


@torch.no_grad()
def _score(head: ProgressHead, data: dict[str, Any], device: torch.device, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(data["rows"]), batch_size):
        end = min(start + batch_size, len(data["rows"]))
        tokens = data["tokens"][start:end].to(device)
        actions = data["actions"][start:end].to(device)
        tasks = data["tasks"][start:end].to(device)
        batch, candidates, horizon, dim = tokens.shape
        out = head(
            tokens.reshape(batch * candidates, horizon, dim).unsqueeze(2),
            action_cond=actions.reshape(batch * candidates, horizon, 7),
            task_emb=tasks[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1),
        )
        # This is the exact formal serving score when plausibility weight is 1
        # and progress/terminal weights are 0. Argmax(logit) == argmax(sigmoid(logit)).
        parts.append(out["plausibility_logit"].reshape(batch, candidates).float().cpu())
    return torch.cat(parts)


def _row_bootstrap(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        means[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _cluster_bootstrap(values: np.ndarray, rows: list[dict[str, Any]], seed: int, draws: int) -> list[float]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for value, row in zip(values.tolist(), rows, strict=True):
        grouped.setdefault(_group_key(row), []).append(float(value))
    groups = [np.asarray(group, dtype=np.float64) for group in grouped.values()]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        means[index] = np.concatenate([groups[item] for item in sampled]).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    for group, count in enumerate(counts):
        if count > 1:
            indices = np.flatnonzero(inverse == group)
            ranks[indices] = ranks[indices].mean()
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = _rank(x), _rank(y)
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _metrics(data: dict[str, Any], logits: torch.Tensor, seed: int, draws: int) -> dict[str, Any]:
    selected = logits.argmax(dim=1)
    post = data["post"]
    success = data["success"]
    valid = data["valid"]
    chosen_post = post.gather(1, selected[:, None]).squeeze(1)
    chosen_success = success.gather(1, selected[:, None]).squeeze(1)
    post_gain = (post[:, 0] - chosen_post).numpy().astype(np.float64)
    rows = data["rows"]
    flat_score = logits.numpy().reshape(-1)
    flat_quality = -post.numpy().reshape(-1)
    report: dict[str, Any] = {
        "rows": len(rows),
        "demo_groups": len({_group_key(row) for row in rows}),
        "valid_factual_success_rows": int(valid.sum()),
        "selection_counts": torch.bincount(selected, minlength=post.shape[1]).tolist(),
        "nonanchor_rate": float((selected != 0).float().mean()),
        "anchor_post_l1": float(post[:, 0].mean()),
        "selected_post_l1": float(chosen_post.mean()),
        "oracle_post_l1": float(post.min(dim=1).values.mean()),
        "post_gain_vs_anchor": float(post_gain.mean()),
        "post_gain_row_ci95": _row_bootstrap(post_gain, seed, draws),
        "post_gain_demo_cluster_ci95": _cluster_bootstrap(post_gain, rows, seed, draws),
        "candidate_score_vs_negative_post_spearman": _spearman(flat_score, flat_quality),
        "oracle_post_top1_rate": float((selected == post.argmin(dim=1)).float().mean()),
    }
    if bool(valid.any()):
        valid_rows = [row for row, keep in zip(rows, valid.tolist(), strict=True) if keep]
        success_gain = (chosen_success[valid] - success[valid, 0]).numpy().astype(np.float64)
        report.update(
            {
                "anchor_success_rate": float(success[valid, 0].mean()),
                "selected_success_rate": float(chosen_success[valid].mean()),
                "oracle_success_rate": float(success[valid].max(dim=1).values.mean()),
                "success_gain_pp": float(100.0 * success_gain.mean()),
                "success_gain_pp_row_ci95": [100.0 * value for value in _row_bootstrap(success_gain, seed, draws)],
                "success_gain_pp_demo_cluster_ci95": [
                    100.0 * value for value in _cluster_bootstrap(success_gain, valid_rows, seed, draws)
                ],
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    device = torch.device(args.device)
    labels = _load_labels(args.branches)
    splits = {
        name: _load_split(args.split_root / name / "features", labels)
        for name in ("dev", "test")
    }
    model_cfg = _resolve_model_config(args.cfg)
    reports = []
    for checkpoint in args.checkpoint:
        head, meta = _build_head(model_cfg, checkpoint, device)
        reports.append(
            {
                **meta,
                "splits": {
                    name: _metrics(data, _score(head, data, device, args.batch_size), args.seed, args.bootstrap_draws)
                    for name, data in splits.items()
                },
            }
        )
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "contract": {
            "candidate_count": 9,
            "selection": "argmax(sigmoid(plausibility_logit))",
            "real_outcome": "LIBERO simulator branch success and post_state_l1",
            "split": "demo-disjoint fixed dev/test",
            "seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
        },
        "checkpoints": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
