"""Probe LIBERO source, task metadata, and simulator availability."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wm3d_v3.benchmarks.libero_adapter import (
    _load_benchmark_api,
    _load_env_api,
    _resolve_libero_root,
)


@dataclass
class ProbeResult:
    root: str
    source_exists: bool
    task_api_available: bool
    env_api_available: bool
    suite: str
    num_tasks: int | None = None
    first_task: dict[str, Any] | None = None
    task_api_error: str | None = None
    env_api_error: str | None = None


def probe_libero(root: str | Path | None = None, suite: str = "libero_10") -> ProbeResult:
    resolved = _resolve_libero_root(root)
    result = ProbeResult(
        root=str(resolved),
        source_exists=resolved.exists(),
        task_api_available=False,
        env_api_available=False,
        suite=suite,
    )
    if not resolved.exists():
        result.task_api_error = f"LIBERO root does not exist: {resolved}"
        result.env_api_error = result.task_api_error
        return result

    try:
        get_benchmark = _load_benchmark_api(resolved)
        task_suite = get_benchmark(suite)()
        result.task_api_available = True
        result.num_tasks = int(task_suite.get_num_tasks())
        if result.num_tasks:
            task = task_suite.get_task(0)
            result.first_task = {
                "name": task.name,
                "language": task.language,
                "problem_folder": task.problem_folder,
                "bddl_file": task.bddl_file,
                "init_states_file": task.init_states_file,
            }
    except Exception as exc:
        result.task_api_error = repr(exc)

    try:
        _load_env_api(resolved)
        result.env_api_available = True
    except Exception as exc:
        result.env_api_error = repr(exc)

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = asdict(probe_libero(args.root, suite=args.suite))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
