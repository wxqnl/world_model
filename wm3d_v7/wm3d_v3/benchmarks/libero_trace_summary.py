"""Summarize LIBERO remote rollout traces and export training-oriented JSONL.

The remote runner proves the real loop:

    LIBERO observation -> WM3D HTTP policy -> 7D action -> env.step(action)

This module makes those rollouts useful beyond a smoke test. It reports whether
the trace contains benchmark feedback, action labels, policy candidate scores,
and optional frame paths that can later be re-tokenized for evaluator/proposer
training.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _load_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(item) for item in matches)
        else:
            paths.append(Path(pattern))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _episode_rows(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep_idx, ep in enumerate(payload.get("results") or []):
        trace = ep.get("step_trace") or []
        base = {
            "source": str(source),
            "suite": ep.get("suite", payload.get("suite")),
            "task_id": ep.get("task_id"),
            "task_name": ep.get("task_name"),
            "instruction": ep.get("instruction"),
            "init_state_id": ep.get("init_state_id"),
            "episode_index": ep_idx,
            "episode_success": bool(ep.get("success", False)),
            "episode_steps": int(ep.get("steps") or 0),
        }
        if trace:
            for item in trace:
                row = dict(base)
                row.update({
                    "row_type": "step",
                    "step": int(item.get("step") or 0),
                    "action": item.get("action"),
                    "action_norm": _float(item.get("action_norm")),
                    "reward": _float(item.get("reward")),
                    "done": bool(item.get("done", False)),
                    "step_success": bool(item.get("success", False)),
                    "selected_idx": item.get("selected_idx"),
                    "selected_score": item.get("selected_score"),
                    "candidate_scores": item.get("candidate_scores"),
                    "action_chunk_raw": item.get("action_chunk_raw"),
                    "frame_path": item.get("frame_path"),
                })
                rows.append(row)
        else:
            row = dict(base)
            row.update({
                "row_type": "episode",
                "step": None,
                "action": None,
                "action_norm": None,
                "reward": _float((ep.get("last_info") or {}).get("reward")),
                "done": bool((ep.get("last_info") or {}).get("done", False)),
                "step_success": bool(ep.get("success", False)),
                "selected_idx": None,
                "selected_score": None,
                "candidate_scores": None,
                "action_chunk_raw": None,
                "frame_path": None,
            })
            rows.append(row)
    return rows


def summarize(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text())
        rows = _episode_rows(payload, path)
        all_rows.extend(rows)
        inputs.append({
            "path": str(path),
            "success_rate": payload.get("success_rate"),
            "episodes": len(payload.get("results") or []),
            "trace_schema_version": payload.get("trace_schema_version", 1),
        })

    episode_keys: set[tuple[Any, Any, Any, str]] = set()
    success_episodes: set[tuple[Any, Any, Any, str]] = set()
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "trace_steps": 0,
        "mean_action_norm": 0.0,
        "_norms": [],
    })
    action_norms: list[float] = []
    rewards: list[float] = []
    traced_steps = 0
    frame_steps = 0
    candidate_steps = 0
    chunk_steps = 0

    for row in all_rows:
        ep_key = (row.get("source"), row.get("task_id"), row.get("init_state_id"), str(row.get("episode_index")))
        episode_keys.add(ep_key)
        if row.get("episode_success"):
            success_episodes.add(ep_key)
        task_name = str(row.get("task_name") or row.get("task_id"))
        task = by_task[task_name]
        if row.get("row_type") == "step":
            traced_steps += 1
            task["trace_steps"] += 1
            if row.get("frame_path"):
                frame_steps += 1
            if row.get("candidate_scores") is not None:
                candidate_steps += 1
            if row.get("action_chunk_raw") is not None:
                chunk_steps += 1
            norm = row.get("action_norm")
            if norm is not None:
                action_norms.append(float(norm))
                task["_norms"].append(float(norm))
            rewards.append(float(row.get("reward") or 0.0))

    for row in all_rows:
        task_name = str(row.get("task_name") or row.get("task_id"))
        ep_key = (row.get("source"), row.get("task_id"), row.get("init_state_id"), str(row.get("episode_index")))
        if row.get("row_type") in ("step", "episode") and row.get("step") in (1, None):
            by_task[task_name]["episodes"] += 1
            if ep_key in success_episodes:
                by_task[task_name]["successes"] += 1

    task_summaries: list[dict[str, Any]] = []
    for task_name, item in sorted(by_task.items()):
        norms = item.pop("_norms")
        item["success_rate"] = item["successes"] / max(1, item["episodes"])
        item["mean_action_norm"] = mean(norms) if norms else None
        item["task_name"] = task_name
        task_summaries.append(item)

    episodes = len(episode_keys)
    successes = len(success_episodes)
    failure_count = max(0, episodes - successes)
    summary = {
        "inputs": inputs,
        "episodes": episodes,
        "successes": successes,
        "failures": failure_count,
        "success_rate": successes / max(1, episodes),
        "trace_rows": len(all_rows),
        "trace_steps": traced_steps,
        "frame_steps": frame_steps,
        "candidate_score_steps": candidate_steps,
        "action_chunk_steps": chunk_steps,
        "mean_action_norm": mean(action_norms) if action_norms else None,
        "max_action_norm": max(action_norms) if action_norms else None,
        "min_action_norm": min(action_norms) if action_norms else None,
        "mean_reward": mean(rewards) if rewards else None,
        "nonzero_reward_steps": sum(1 for value in rewards if value != 0.0),
        "by_task": task_summaries,
        "training_signal": {
            "benchmark_feedback": episodes > 0,
            "action_trace": traced_steps > 0,
            "policy_candidate_scores": candidate_steps > 0,
            "action_chunks": chunk_steps > 0,
            "frame_retokenization": frame_steps > 0,
            "binary_success_supervision": successes > 0 and failure_count > 0,
            "failure_only_supervision": successes == 0 and failure_count > 0,
            "notes": (
                "Use failure_only_supervision only for diagnostics or negative mining. "
                "Evaluator/proposer training needs mixed success/failure or expert success traces."
            ),
        },
    }
    return summary, all_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="Rollout JSON paths or glob patterns.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jsonl_out", type=Path, default=None)
    args = ap.parse_args()

    paths = _load_paths(args.input)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing rollout files: {missing}")

    summary, rows = summarize(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with args.jsonl_out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "jsonl_out": str(args.jsonl_out) if args.jsonl_out else None,
        "episodes": summary["episodes"],
        "success_rate": summary["success_rate"],
        "trace_steps": summary["trace_steps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
