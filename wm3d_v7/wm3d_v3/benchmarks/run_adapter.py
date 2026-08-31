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


def _flow_sample_from_arg(value: str) -> bool | None:
    value = value.strip().lower()
    if value == "auto":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"unsupported flow sample mode {value!r}")


def _selection_mode_from_cfg(cfg: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    train_cfg = cfg.get("train", {})
    if bool(train_cfg.get("candidate_planner_joint_action", False)):
        return "ranked_residual"
    model_cfg = cfg.get("model", {})
    if bool(model_cfg.get("enable_action_policy", False)) and bool(model_cfg.get("policy_enable_flow_head", False)):
        return "direct"
    return "ranked"


def _score_weights_from_cfg(
    cfg: dict,
    *,
    progress: float | None,
    terminal: float | None,
    plausibility: float | None,
) -> ScoreWeights:
    train_cfg = cfg.get("train", {})
    joint_candidate = bool(train_cfg.get("candidate_planner_joint_action", False))
    defaults = (
        (
            float(train_cfg.get("evaluator_score_progress_weight", 0.0)),
            float(train_cfg.get("evaluator_score_terminal_weight", 0.0)),
            float(train_cfg.get("evaluator_score_plausibility_weight", 1.0)),
        )
        if joint_candidate
        else (1.0, 1.0, 0.0)
    )
    return ScoreWeights(
        progress=defaults[0] if progress is None else float(progress),
        terminal=defaults[1] if terminal is None else float(terminal),
        plausibility=defaults[2] if plausibility is None else float(plausibility),
    )


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
    ap.add_argument("--score_progress_weight", type=float, default=None)
    ap.add_argument("--score_terminal_weight", type=float, default=None)
    ap.add_argument("--score_plausibility_weight", type=float, default=None)
    ap.add_argument(
        "--selection_mode",
        default="auto",
        choices=(
            "auto",
            "ranked",
            "ranked_residual",
            "anchor",
            "first",
            "candidate0",
            "direct",
            "policy",
            "bc",
            "action_policy",
            "direct_prior",
            "prior_policy",
            "direct_stage3_place",
            "direct_terminal_nn",
            "direct_terminal_linear",
            "direct_trace_linear",
            "plan_waypoint",
            "waypoint_servo",
        ),
    )
    ap.add_argument("--flow_sample", default="auto", choices=("auto", "true", "false"))
    ap.add_argument("--flow_sample_steps", "--flow_steps", dest="flow_sample_steps", type=int, default=None)
    ap.add_argument("--flow_noise_scale", type=float, default=None)
    ap.add_argument("--flow_seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    selection_mode = _selection_mode_from_cfg(cfg, args.selection_mode)
    adapter = _load_adapter(args.adapter, args)
    policy = WM3DTokenPolicy.from_checkpoint(
        args.cfg,
        args.ckpt,
        device=args.device,
        score_weights=_score_weights_from_cfg(
            cfg,
            progress=args.score_progress_weight,
            terminal=args.score_terminal_weight,
            plausibility=args.score_plausibility_weight,
        ),
        selection_mode=selection_mode,
        flow_sample=_flow_sample_from_arg(args.flow_sample),
        flow_sample_steps=args.flow_sample_steps,
        flow_noise_scale=args.flow_noise_scale,
        flow_seed=args.flow_seed,
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
        "selection_mode": selection_mode,
        "flow_sample": args.flow_sample,
        "flow_sample_steps": args.flow_sample_steps,
        "flow_noise_scale": args.flow_noise_scale,
        "flow_seed": args.flow_seed,
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
