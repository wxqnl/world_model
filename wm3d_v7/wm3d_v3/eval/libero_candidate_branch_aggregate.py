"""Aggregate simulator-branched LIBERO candidate outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return None
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else None


def _bootstrap_mean_ci(values: np.ndarray, seed: int = 1729) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    _unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            ranks[inverse == index] = ranks[inverse == index].mean()
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    x_rank = _ranks(x[keep])
    y_rank = _ranks(y[keep])
    if x_rank.std() == 0.0 or y_rank.std() == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _parse(paths: list[Path]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    errors = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            branches = item.get("branches", [])
            if not branches or any("error" in branch for branch in branches):
                errors += 1
                continue
            candidates = sorted(
                (branch for branch in branches if branch["candidate_index"] >= 0),
                key=lambda branch: branch["candidate_index"],
            )
            factual = next(
                (branch for branch in branches if branch["candidate_index"] == -1), None
            )
            if factual is None or not candidates:
                errors += 1
                continue
            item["factual"] = factual
            item["candidates"] = candidates
            rows.append(item)
    return rows, errors


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["factual"]["success"]]
    selected_success_delta: list[float] = []
    selected_post_delta: list[float] = []
    selected_relative_post_gain: list[float] = []
    score_values: list[float] = []
    negative_post_values: list[float] = []
    anchor_success: list[float] = []
    selected_success: list[float] = []
    oracle_success: list[float] = []
    anchor_post: list[float] = []
    selected_post: list[float] = []
    oracle_post: list[float] = []
    selected_nonanchor: list[float] = []
    candidate_count = len(valid[0]["candidates"]) if valid else 0
    if any(len(row["candidates"]) != candidate_count for row in valid):
        raise RuntimeError("inconsistent candidate counts")
    candidate_success = [[] for _ in range(candidate_count)]
    discriminative_selected_success: list[float] = []

    for row in valid:
        candidates = row["candidates"]
        scores = np.asarray([branch["model_score"] for branch in candidates])
        success = np.asarray([branch["success"] for branch in candidates], dtype=np.float64)
        post = np.asarray([branch["post_state_l1"] for branch in candidates])
        selected_index = int(np.argmax(scores))
        anchor_success.append(float(success[0]))
        selected_success.append(float(success[selected_index]))
        oracle_success.append(float(success.max()))
        selected_success_delta.append(float(success[selected_index] - success[0]))
        anchor_post.append(float(post[0]))
        selected_post.append(float(post[selected_index]))
        oracle_post.append(float(post.min()))
        selected_post_delta.append(float(post[0] - post[selected_index]))
        selected_relative_post_gain.append(
            float((post[0] - post[selected_index]) / max(post[0], 1e-12))
        )
        selected_nonanchor.append(float(selected_index != 0))
        if success.min() != success.max():
            discriminative_selected_success.append(float(success[selected_index]))
        for index in range(candidate_count):
            candidate_success[index].append(float(success[index]))
            score_values.append(float(scores[index]))
            negative_post_values.append(float(-post[index]))

    success_delta_mean = _mean(selected_success_delta)
    success_delta_ci = _bootstrap_mean_ci(np.asarray(selected_success_delta))
    factual_post = [row["factual"]["post_state_l1"] for row in rows]
    factual_final = [row["factual"]["final_state_l1"] for row in rows]
    return {
        "rows": len(rows),
        "factual_success_rows": len(valid),
        "factual_success_rate": len(valid) / len(rows) if rows else None,
        "factual_post_state_l1_mean": _mean(factual_post),
        "factual_final_state_l1_mean": _mean(factual_final),
        "candidate_success_rate": [_mean(values) for values in candidate_success],
        "anchor_success_rate": _mean(anchor_success),
        "judge_selected_success_rate": _mean(selected_success),
        "oracle_success_rate": _mean(oracle_success),
        "judge_minus_anchor_success_pp": (
            100.0 * success_delta_mean if success_delta_mean is not None else None
        ),
        "judge_minus_anchor_success_pp_ci95": (
            [100.0 * value for value in success_delta_ci]
            if success_delta_ci is not None
            else None
        ),
        "judge_discriminative_top1_success_rate": _mean(discriminative_selected_success),
        "discriminative_rows": len(discriminative_selected_success),
        "judge_nonanchor_selection_rate": _mean(selected_nonanchor),
        "anchor_post_state_l1": _mean(anchor_post),
        "judge_selected_post_state_l1": _mean(selected_post),
        "oracle_post_state_l1": _mean(oracle_post),
        "judge_post_l1_gain_vs_anchor": _mean(selected_post_delta),
        "judge_relative_post_l1_gain_vs_anchor": _mean(selected_relative_post_gain),
        "judge_post_l1_gain_vs_anchor_ci95": _bootstrap_mean_ci(
            np.asarray(selected_post_delta)
        ),
        "score_vs_negative_post_l1_spearman": _spearman(
            score_values, negative_post_values
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected_rows", type=int, default=160)
    args = ap.parse_args()

    paths = sorted(args.input_dir.glob("shard*.jsonl"))
    if not paths:
        raise RuntimeError(f"no shard JSONL files found in {args.input_dir}")
    rows, errors = _parse(paths)
    if len(rows) + errors != args.expected_rows:
        raise RuntimeError(
            f"expected {args.expected_rows} rows, found {len(rows)} valid and {errors} errors"
        )

    by_task = {
        task: _summary([row for row in rows if row["row"]["task_name"] == task])
        for task in sorted({row["row"]["task_name"] for row in rows})
    }
    progress_bins = {
        "early_0_33": (0.0, 0.33),
        "middle_33_67": (0.33, 0.67),
        "late_67_101": (0.67, 1.01),
    }
    has_progress = all("progress_fraction" in row["row"] for row in rows)
    by_progress = (
        {
            name: _summary(
                [
                    row
                    for row in rows
                    if low <= float(row["row"]["progress_fraction"]) < high
                ]
            )
            for name, (low, high) in progress_bins.items()
        }
        if has_progress
        else {}
    )
    output = {
        "contract": {
            "anchor_candidate_index": 0,
            "judge_selection": "argmax(model_score)",
            "binary_success_scope": (
                "only rows where factual expert continuation succeeds"
            ),
            "dense_metric": (
                "post-state L1 to the expert state after the same K8 horizon"
            ),
        },
        "input_shards": [str(path) for path in paths],
        "parse_errors": errors,
        "overall": _summary(rows),
        "by_task": by_task,
        "by_progress": by_progress,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
