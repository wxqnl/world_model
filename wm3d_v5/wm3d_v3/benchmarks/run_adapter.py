"""Run a WM3D benchmark adapter.

Mock and offline replay adapters are available now. LIBERO can run through this
same path once its external simulator dependencies are installed.
"""
from __future__ import annotations

import argparse
import json
import yaml
from dataclasses import asdict
from pathlib import Path

from wm3d_v3.policy import ScoreWeights, WM3DTokenPolicy


def _load_adapter(name: str, args: argparse.Namespace):
    if name == "mock":
        from wm3d_v3.benchmarks.mock_adapter import MockTokenAdapter

        return MockTokenAdapter()
    if name == "offline_replay":
        from wm3d_v3.benchmarks.offline_replay_adapter import OfflineReplayAdapter

        return OfflineReplayAdapter(
            args.cfg,
            split=args.split,
            success_pose_l1_threshold=args.success_pose_l1_threshold,
        )
    if name == "libero":
        from wm3d_v3.benchmarks.libero_adapter import LiberoAdapter

        cfg = yaml.safe_load(args.cfg.read_text())
        init_state_ids = [int(x) for x in args.libero_init_state_ids.split(",") if x.strip()]
        return LiberoAdapter(
            cfg=cfg,
            libero_root=args.libero_root,
            suite=args.libero_suite,
            task_order_index=args.libero_task_order_index,
            init_state_ids=init_state_ids,
            seed=args.libero_seed,
            camera_key=args.libero_camera_key,
            camera_size=args.libero_camera_size,
            warmup_steps=args.libero_warmup_steps,
            device=args.device,
            qwen_device=args.qwen_device,
            task_cache_dir=args.task_cache_dir,
            allow_zero_task_fallback=args.allow_zero_task_fallback,
        )
    raise RuntimeError(
        f"benchmark adapter '{name}' is not implemented yet. "
        "Install/link the benchmark package and add a concrete BenchmarkAdapter."
    )


def _mean_metrics(results) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / max(1, counts[key]) for key in sorted(totals)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="mock", choices=("mock", "offline_replay", "libero"))
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_tasks", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=1)
    ap.add_argument("--split", choices=("train", "val", "all"), default="val")
    ap.add_argument("--success_pose_l1_threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_device", default=None)
    ap.add_argument("--task_cache_dir", type=Path, default=None)
    ap.add_argument("--allow_zero_task_fallback", action="store_true")
    ap.add_argument("--libero_root", type=Path, default=None)
    ap.add_argument("--libero_suite", default="libero_10")
    ap.add_argument("--libero_task_order_index", type=int, default=0)
    ap.add_argument("--libero_init_state_ids", default="0")
    ap.add_argument("--libero_seed", type=int, default=0)
    ap.add_argument("--libero_camera_key", default="agentview_image")
    ap.add_argument("--libero_camera_size", type=int, default=224)
    ap.add_argument("--libero_warmup_steps", type=int, default=5)
    ap.add_argument("--score_progress_weight", type=float, default=1.0)
    ap.add_argument("--score_terminal_weight", type=float, default=1.0)
    ap.add_argument("--score_plausibility_weight", type=float, default=0.0)
    args = ap.parse_args()

    adapter = _load_adapter(args.adapter, args)
    policy = WM3DTokenPolicy.from_checkpoint(
        args.cfg,
        args.ckpt,
        device=args.device,
        score_weights=ScoreWeights(
            progress=args.score_progress_weight,
            terminal=args.score_terminal_weight,
            plausibility=args.score_plausibility_weight,
        ),
    )
    tasks = adapter.iter_tasks(limit=args.max_tasks)
    results = [adapter.rollout_episode(policy, task, max_steps=args.max_steps) for task in tasks]
    success_rate = sum(float(r.success) for r in results) / max(1, len(results))
    mean_metrics = _mean_metrics(results)
    report = {
        "adapter": adapter.name,
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "max_tasks": args.max_tasks,
        "max_steps": args.max_steps,
        "split": args.split,
        "success_rate": success_rate,
        "mean_metrics": mean_metrics,
        "results": [asdict(r) for r in results],
        "note": (
            "mock validates only the runner/policy API; offline_replay validates the same API "
            "on cached OXE windows but is still not an external simulator success benchmark."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out), "success_rate": success_rate, "mean_metrics": mean_metrics}, indent=2))


if __name__ == "__main__":
    main()
