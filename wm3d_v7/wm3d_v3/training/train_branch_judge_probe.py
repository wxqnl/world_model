"""Train a small consequence judge from real simulator branch outcomes."""
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


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["task_name"], row["demo_id"], int(row["target_start"])


def _load_branches(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    for shard in sorted(path.glob("shard*.jsonl")):
        for line in shard.read_text().splitlines():
            item = json.loads(line)
            branches = item["branches"]
            factual = next(branch for branch in branches if branch["candidate_index"] == -1)
            candidates = sorted(
                (branch for branch in branches if branch["candidate_index"] >= 0),
                key=lambda branch: branch["candidate_index"],
            )
            result[_key(item["row"])] = {
                "factual_success": bool(factual["success"]),
                "success": [float(branch["success"]) for branch in candidates],
                "post": [float(branch["post_state_l1"]) for branch in candidates],
            }
    return result


def _load_dataset(features: Path, branches: Path) -> dict[str, Any]:
    labels = _load_branches(branches)
    tokens, actions, tasks, success, post, valid, rows = [], [], [], [], [], [], []
    for shard in sorted(features.glob("shard*.npz")):
        payload = np.load(shard)
        shard_rows = [json.loads(str(item)) for item in payload["rows_json"]]
        for index, row in enumerate(shard_rows):
            label = labels[_key(row)]
            tokens.append(payload["world_token_mean"][index])
            actions.append(payload["candidate_cond"][index])
            tasks.append(payload["task_emb"][index])
            success.append(label["success"])
            post.append(label["post"])
            valid.append(label["factual_success"])
            rows.append(row)
    return {
        "tokens": torch.from_numpy(np.asarray(tokens, dtype=np.float32)),
        "actions": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
        "tasks": torch.from_numpy(np.asarray(tasks, dtype=np.float32)),
        "success": torch.from_numpy(np.asarray(success, dtype=np.float32)),
        "post": torch.from_numpy(np.asarray(post, dtype=np.float32)),
        "valid": torch.from_numpy(np.asarray(valid, dtype=np.bool_)),
        "rows": rows,
    }


def _split(rows: list[dict[str, Any]], seed: int, train_all: bool) -> tuple[list[int], list[int]]:
    if train_all:
        all_indices = list(range(len(rows)))
        return all_indices, all_indices
    by_task: dict[str, set[str]] = {}
    for row in rows:
        by_task.setdefault(row["task_name"], set()).add(row["demo_id"])
    val_demos = set()
    for task, demos in by_task.items():
        ordered = sorted(
            demos,
            key=lambda demo: hashlib.sha256(f"{seed}:{task}:{demo}".encode()).hexdigest(),
        )
        val_demos.update((task, demo) for demo in ordered[: max(1, round(len(ordered) * 0.2))])
    val = [i for i, row in enumerate(rows) if (row["task_name"], row["demo_id"]) in val_demos]
    train = [i for i in range(len(rows)) if i not in set(val)]
    return train, val


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
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["model"]
    prefix = "progress_head."
    head.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}, strict=True)
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
    if mode not in {"output", "output_norm", "all"}:
        raise ValueError(f"unknown trainable mode: {mode}")
    return sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)


def _scores(head: ProgressHead, tokens, actions, tasks) -> tuple[torch.Tensor, torch.Tensor]:
    batch, candidates, horizon, dim = tokens.shape
    flat_tokens = tokens.reshape(batch * candidates, horizon, dim).unsqueeze(2)
    flat_actions = actions.reshape(batch * candidates, horizon, 7)
    flat_tasks = tasks[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
    out = head(flat_tokens, action_cond=flat_actions, task_emb=flat_tasks)
    terminal = out["terminal_success_logit"].reshape(batch, candidates)
    progress = out["progress"].mean(dim=1).reshape(batch, candidates)
    return terminal + progress, terminal


def _loss(head, data, indices, device, temperature: float) -> torch.Tensor:
    tokens = data["tokens"][indices].to(device)
    actions = data["actions"][indices].to(device)
    tasks = data["tasks"][indices].to(device)
    success = data["success"][indices].to(device)
    post = data["post"][indices].to(device)
    valid = data["valid"][indices].to(device)
    scores, terminal = _scores(head, tokens, actions, tasks)
    dense_target = post.argmin(dim=1)
    dense_ce = F.cross_entropy(scores / temperature, dense_target)
    if valid.any():
        valid_success = success[valid]
        valid_post = post[valid]
        has_success = valid_success.max(dim=1).values > 0.5
        success_post = valid_post.masked_fill(valid_success < 0.5, float("inf"))
        success_target = torch.where(has_success, success_post.argmin(dim=1), valid_post.argmin(dim=1))
        success_ce = F.cross_entropy(scores[valid] / temperature, success_target)
        success_bce = F.binary_cross_entropy_with_logits(terminal[valid], valid_success)
    else:
        success_ce = dense_ce.new_zeros(())
        success_bce = dense_ce.new_zeros(())
    return dense_ce + success_ce + 0.25 * success_bce


def _bootstrap_ci(values: torch.Tensor, seed: int = 1729) -> list[float]:
    array = values.detach().float().cpu().numpy().astype(np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(10000, len(array)), replace=True).mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


@torch.no_grad()
def _evaluate(head, data, indices, device) -> dict[str, Any]:
    head.eval()
    selected_parts = []
    for start in range(0, len(indices), 32):
        batch = indices[start : start + 32]
        scores, _terminal = _scores(
            head,
            data["tokens"][batch].to(device),
            data["actions"][batch].to(device),
            data["tasks"][batch].to(device),
        )
        selected_parts.append(scores.argmax(dim=1).cpu())
    selected = torch.cat(selected_parts)
    success = data["success"][indices]
    post = data["post"][indices]
    valid = data["valid"][indices]
    chosen_success = success.gather(1, selected[:, None]).squeeze(1)
    chosen_post = post.gather(1, selected[:, None]).squeeze(1)
    metrics = {
        "rows": float(len(indices)),
        "valid_success_rows": float(valid.sum()),
        "anchor_post_l1": float(post[:, 0].mean()),
        "selected_post_l1": float(chosen_post.mean()),
        "oracle_post_l1": float(post.min(dim=1).values.mean()),
        "post_gain_vs_anchor": float((post[:, 0] - chosen_post).mean()),
        "post_gain_vs_anchor_ci95": _bootstrap_ci(post[:, 0] - chosen_post),
        "nonanchor_rate": float((selected != 0).float().mean()),
    }
    if valid.any():
        metrics.update({
            "anchor_success_rate": float(success[valid, 0].mean()),
            "selected_success_rate": float(chosen_success[valid].mean()),
            "oracle_success_rate": float(success[valid].max(dim=1).values.mean()),
            "success_gain_pp": float(100.0 * (chosen_success[valid] - success[valid, 0]).mean()),
            "success_gain_pp_ci95": [
                100.0 * value for value in _bootstrap_ci(
                    chosen_success[valid] - success[valid, 0]
                )
            ],
        })
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--branches", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--trainable", choices=("output", "output_norm", "all"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split_seed", type=int, default=1729)
    ap.add_argument("--train_all", action="store_true")
    ap.add_argument("--eval_features", type=Path)
    ap.add_argument("--eval_branches", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    wrapper = yaml.safe_load(args.cfg.read_text())
    base_cfg = yaml.safe_load(Path(wrapper["base_cfg"]).read_text())
    data = _load_dataset(args.features, args.branches)
    train_indices, val_indices = _split(data["rows"], args.split_seed, args.train_all)
    head = _build_head(base_cfg, args.ckpt, device)
    trainable = _set_trainable(head, args.trainable)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in head.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        head.train()
        sampled = torch.randint(len(train_indices), (args.batch_size,), generator=generator)
        batch = [train_indices[index] for index in sampled.tolist()]
        loss = _loss(head, data, batch, device, args.temperature)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
    report = {
        "config": vars(args),
        "trainable_parameters": trainable,
        "train": _evaluate(head, data, train_indices, device),
        "validation": _evaluate(head, data, val_indices, device),
    }
    if args.eval_features and args.eval_branches:
        evaluation = _load_dataset(args.eval_features, args.eval_branches)
        report["heldout"] = _evaluate(head, evaluation, list(range(len(evaluation["rows"]))), device)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": head.state_dict(), "report": report}, args.out / "judge.pt")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
