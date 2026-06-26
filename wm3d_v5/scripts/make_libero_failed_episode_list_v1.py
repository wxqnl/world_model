#!/usr/bin/env python3
"""Write a task/init episode list from failed LIBERO episode JSON files."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


NAME_RE = re.compile(r"libero_spatial_task(?P<task>\d+)_init(?P<init>\d+)\.json$")


def _success(path: Path) -> tuple[bool, int]:
    data = json.loads(path.read_text())
    results = data.get("results") or []
    ok = bool(data.get("success_rate", 0.0) or any(item.get("success") for item in results))
    steps = int(results[0].get("steps", 0)) if results else 0
    return ok, steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--task_ids", default="", help="Comma-separated task ids to include. Empty means all.")
    ap.add_argument("--init_start", type=int, default=0)
    ap.add_argument("--init_end", type=int, default=49)
    ap.add_argument("--max_per_task", type=int, default=0, help="0 keeps all failures.")
    ap.add_argument("--include_successes", action="store_true")
    args = ap.parse_args()

    task_filter = {int(x.strip()) for x in args.task_ids.split(",") if x.strip()}
    rows: dict[int, list[tuple[int, bool, int, Path]]] = defaultdict(list)
    for path in sorted(args.episodes_dir.glob("libero_spatial_task*_init*.json")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        task_id = int(match.group("task"))
        init_id = int(match.group("init"))
        if task_filter and task_id not in task_filter:
            continue
        if init_id < args.init_start or init_id > args.init_end:
            continue
        ok, steps = _success(path)
        if ok and not args.include_successes:
            continue
        rows[task_id].append((init_id, ok, steps, path))

    selected: list[tuple[int, int, bool, int, Path]] = []
    for task_id in sorted(rows):
        items = sorted(rows[task_id], key=lambda item: (-item[2], item[0]))
        if args.max_per_task > 0:
            items = items[: args.max_per_task]
        for init_id, ok, steps, path in items:
            selected.append((task_id, init_id, ok, steps, path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for task_id, init_id, ok, steps, path in selected:
            status = "success" if ok else "failure"
            fh.write(f"{task_id} {init_id} # {status} steps={steps} source={path}\n")

    summary = {
        "episodes_dir": str(args.episodes_dir),
        "out": str(args.out),
        "task_ids": sorted(task_filter),
        "init_start": args.init_start,
        "init_end": args.init_end,
        "max_per_task": args.max_per_task,
        "include_successes": bool(args.include_successes),
        "selected": len(selected),
        "per_task": {str(task_id): len(rows[task_id]) for task_id in sorted(rows)},
    }
    (args.out.with_suffix(args.out.suffix + ".summary.json")).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
