#!/usr/bin/env python3
"""Build a domain-balanced OXE+DROID manifest for WM3D pretraining.

The source manifest is intentionally imbalanced because it reflects available
clips. This script preserves all domains but caps large domains and upsamples
small ones by repeating clip records. Repeated records are acceptable for the
cache-backed window dataset: they simply resample the same episode windows.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_GROUPS = {
    "fractal": ["fractal20220817_data"],
    "bridge": ["bridge"],
    "droid": ["droid"],
    "small_robot": ["taco_play", "jaco_play", "kuka"],
}


def _parse_weights(text: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in text.split(","):
        name, value = item.split("=", 1)
        weights[name.strip()] = float(value)
    missing = set(DEFAULT_GROUPS) - set(weights)
    if missing:
        raise ValueError(f"missing group weights: {sorted(missing)}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("group weights must sum to a positive value")
    return {k: v / total for k, v in weights.items()}


def _group_for_dataset(dataset: str) -> str:
    for group, datasets in DEFAULT_GROUPS.items():
        if dataset in datasets:
            return group
    raise ValueError(f"dataset {dataset!r} is not in DEFAULT_GROUPS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--target_records", type=int, default=160_000)
    ap.add_argument(
        "--weights",
        default="fractal=0.25,bridge=0.25,droid=0.25,small_robot=0.25",
        help="Comma-separated group weights. Groups: fractal, bridge, droid, small_robot.",
    )
    ap.add_argument("--seed", type=int, default=606)
    args = ap.parse_args()

    weights = _parse_weights(args.weights)
    rng = random.Random(args.seed)
    by_group: dict[str, list[dict]] = defaultdict(list)
    source_counts: Counter[str] = Counter()

    with args.input.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            dataset = record["dataset"]
            source_counts[dataset] += 1
            by_group[_group_for_dataset(dataset)].append(record)

    output: list[dict] = []
    for group, group_weight in weights.items():
        records = list(by_group[group])
        if not records:
            raise RuntimeError(f"group {group!r} has no records")
        rng.shuffle(records)
        target_n = max(1, int(round(args.target_records * group_weight)))
        if len(records) >= target_n:
            output.extend(records[:target_n])
            continue
        repeats, remainder = divmod(target_n, len(records))
        group_out = records * repeats
        group_out.extend(rng.sample(records, remainder))
        output.extend(group_out)

    rng.shuffle(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    final_counts = Counter(record["dataset"] for record in output)
    group_counts = Counter(_group_for_dataset(record["dataset"]) for record in output)
    print(f"wrote {len(output)} records -> {args.output}")
    print("source_counts:")
    for dataset, count in source_counts.most_common():
        print(f"  {dataset}: {count}")
    print("balanced_dataset_counts:")
    for dataset, count in final_counts.most_common():
        print(f"  {dataset}: {count}")
    print("balanced_group_counts:")
    for group, count in sorted(group_counts.items()):
        print(f"  {group}: {count}")


if __name__ == "__main__":
    main()
