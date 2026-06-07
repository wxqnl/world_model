"""Run the wm3d system-level scaffold checks for one checkpoint.

This is not a replacement for professional benchmarks. It is the mandatory
checkpoint-to-report harness that proves the current world-model loop is wired:

1. world-core quantitative eval
2. action counterfactual sensitivity
3. proposer -> simulate -> rank TTC sanity check
4. policy action-output probe
5. optional demo GIF generation
6. offline replay adapter smoke on cached validation windows
7. LIBERO source/task/env probe
8. external benchmark availability probe
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wm3d_v3.benchmarks import probe_benchmarks


def _run(name: str, cmd: list[str], log_path: Path, cwd: Path | None = None) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "seconds": time.time() - started,
        "log": str(log_path),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _nested_get(obj: dict[str, Any] | None, keys: list[str]) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _metric_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    all_metrics = _nested_get(report, ["metrics", "ALL"])
    return all_metrics if isinstance(all_metrics, dict) else {}


def _action_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    all_metrics = _nested_get(report, ["metrics", "ALL"])
    if not isinstance(all_metrics, dict):
        return {}
    summary: dict[str, Any] = {}
    for variant, metrics in all_metrics.items():
        if not isinstance(metrics, dict):
            continue
        summary[variant] = {
            key: metrics.get(key)
            for key in (
                "pred_tokens_gt_mse_acc",
                "pred_tokens_gt_mse_gap",
                "depth_gt_l1_acc",
                "motion_hint_gt_l1_acc",
                "progress_gt_l1_acc",
            )
            if key in metrics
        }
    return summary


def _ttc_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics.get(key)
        for key in (
            "anchor_pose_l1",
            "ranked_pose_l1",
            "oracle_ranked_pose_l1",
            "oracle_pose_l1",
            "learned_oracle_idx_match",
            "real_oracle_top1",
            "real_oracle_top3",
        )
        if key in metrics
    }


def _policy_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics.get(key)
        for key in (
            "selected_pose_l1",
            "anchor_pose_l1",
            "oracle_pose_l1",
            "selected_first_pose_l1",
            "anchor_first_pose_l1",
            "oracle_first_pose_l1",
            "selected_matches_action_oracle",
            "score_margin",
            "score_std",
        )
        if key in metrics
    }


def _adapter_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metrics = report.get("mean_metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics.get(key)
        for key in (
            "pose_l1_norm",
            "pose_l1_raw",
            "grip_match",
            "success",
        )
        if key in metrics
    }


def _libero_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        key: report.get(key)
        for key in (
            "source_exists",
            "task_api_available",
            "env_api_available",
            "suite",
            "num_tasks",
            "first_task",
            "env_api_error",
        )
        if key in report
    }


def _trace_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        key: report.get(key)
        for key in (
            "episodes",
            "successes",
            "failures",
            "success_rate",
            "trace_steps",
            "frame_steps",
            "candidate_score_steps",
            "action_chunk_steps",
            "mean_action_norm",
            "nonzero_reward_steps",
            "training_signal",
        )
        if key in report
    }


def _world3d_claim_subset(report: dict[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    core = report.get("core_contribution")
    return core if isinstance(core, dict) else {}


def _compare_eval(cur: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    if baseline is None:
        return {"status": "missing_baseline"}
    out: dict[str, Any] = {"status": "compared", "delta_current_minus_baseline": {}}
    cur_all = _metric_subset(cur)
    base_all = _metric_subset(baseline)
    for key in (
        "L_state_mse",
        "L_depth_rel_L1",
        "L_pose_mse",
        "grip_acc",
        "L_rgb_L1",
        "L_rgb_lpips",
        "L_rgb_motion_L1",
        "progress_abs_err",
        "proposer_best_pose_L1",
        "proposer_anchor_pose_L1",
    ):
        if key in cur_all and key in base_all:
            out["delta_current_minus_baseline"][key] = cur_all[key] - base_all[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--baseline_eval", type=Path, default=None)
    ap.add_argument("--max_eval_batches", type=int, default=80)
    ap.add_argument("--max_world3d_batches", type=int, default=32)
    ap.add_argument("--max_world_prior_batches", type=int, default=16)
    ap.add_argument("--world_prior_steps", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--max_action_batches", type=int, default=80)
    ap.add_argument("--max_ttc_batches", type=int, default=40)
    ap.add_argument("--max_policy_batches", type=int, default=80)
    ap.add_argument("--max_offline_replay_tasks", type=int, default=80)
    ap.add_argument("--offline_replay_success_pose_l1_threshold", type=float, default=0.5)
    ap.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    ap.add_argument("--libero_suite", default="libero_10")
    ap.add_argument(
        "--libero_python",
        type=Path,
        default=None,
        help="Optional Python executable for LIBERO/robosuite probes when the simulator lives in a separate env.",
    )
    ap.add_argument(
        "--libero_trace_input",
        action="append",
        default=[],
        help="Optional LIBERO remote rollout JSON path/glob to summarize into this system report.",
    )
    ap.add_argument("--demo_clips", type=int, default=2)
    ap.add_argument("--score_progress_weight", type=float, default=1.0)
    ap.add_argument("--score_terminal_weight", type=float, default=1.0)
    ap.add_argument("--score_plausibility_weight", type=float, default=0.0)
    ap.add_argument(
        "--no_video",
        action="store_true",
        help="do not activate RGB/video eval or demo generation; validate latent prediction/action loop only",
    )
    ap.add_argument("--skip_demo", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--skip_world3d_claim", action="store_true")
    ap.add_argument("--skip_world_prior_eval", action="store_true")
    ap.add_argument(
        "--run_world_prior_eval",
        action="store_true",
        help="explicitly run world_prior_eval even when --no_video is set",
    )
    ap.add_argument("--skip_action_sensitivity", action="store_true")
    ap.add_argument("--skip_ttc", action="store_true")
    ap.add_argument("--skip_policy_probe", action="store_true")
    ap.add_argument("--skip_offline_replay", action="store_true")
    ap.add_argument("--skip_libero_probe", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / "logs"
    py = sys.executable
    libero_py = str(args.libero_python) if args.libero_python is not None else py
    if args.no_video:
        args.skip_demo = True
    run_world_prior_eval = (not args.skip_world_prior_eval) and (not args.no_video or args.run_world_prior_eval)

    artifacts = {
        "eval": args.out_dir / "eval_best.json",
        "world3d_claim": args.out_dir / "world3d_claim_best.json",
        "world_prior_eval": args.out_dir / "world_prior_eval_best.json",
        "action_sensitivity": args.out_dir / "action_sensitivity_best.json",
        "proposer_ttc": args.out_dir / "proposer_ttc_best.json",
        "policy_probe": args.out_dir / "policy_probe_best.json",
        "offline_replay": args.out_dir / "offline_replay_adapter.json",
        "libero_probe": args.out_dir / "libero_probe.json",
        "libero_trace_summary": args.out_dir / "libero_trace_summary.json",
        "benchmark_probe": args.out_dir / "benchmark_probe.json",
        "demo_dir": args.out_dir / "demo_best",
        "system_report": args.out_dir / "system_report.json",
    }

    commands: list[dict[str, Any]] = []
    if not args.skip_eval:
        eval_cmd = [
            py, "-m", "wm3d_v3.eval.run_eval",
            "--cfg", str(args.cfg),
            "--ckpt", str(args.ckpt),
            "--out", str(artifacts["eval"]),
            "--max_batches", str(args.max_eval_batches),
            "--batch_size", str(args.eval_batch_size),
        ]
        if args.no_video:
            eval_cmd.append("--skip_rgb_metrics")
        commands.append(_run(
            "world_core_eval",
            eval_cmd,
            logs / "world_core_eval.log",
            cwd=root,
        ))
    if not args.skip_world3d_claim:
        claim_cmd = [
            py, "-m", "wm3d_v3.eval.world3d_claim_eval",
            "--cfg", str(args.cfg),
            "--ckpt", str(args.ckpt),
            "--out", str(artifacts["world3d_claim"]),
            "--max_batches", str(args.max_world3d_batches),
            "--batch_size", str(args.eval_batch_size),
        ]
        if args.no_video:
            claim_cmd.append("--skip_pixel")
        commands.append(_run(
            "world3d_claim_eval",
            claim_cmd,
            logs / "world3d_claim_eval.log",
            cwd=root,
        ))
    if run_world_prior_eval:
        prior_cmd = [
            py, "-m", "wm3d_v3.eval.world_prior_eval",
            "--cfg", str(args.cfg),
            "--ckpt", str(args.ckpt),
            "--out", str(artifacts["world_prior_eval"]),
            "--max_batches", str(args.max_world_prior_batches),
            "--batch_size", str(args.eval_batch_size),
            "--steps", str(args.world_prior_steps),
        ]
        if not args.no_video:
            prior_cmd.append("--pixel")
        commands.append(_run(
            "world_prior_eval",
            prior_cmd,
            logs / "world_prior_eval.log",
            cwd=root,
        ))
    if not args.skip_action_sensitivity:
        commands.append(_run(
            "action_sensitivity",
            [
                py, "-m", "wm3d_v3.eval.action_sensitivity",
                "--cfg", str(args.cfg),
                "--ckpt", str(args.ckpt),
                "--out", str(artifacts["action_sensitivity"]),
                "--max_batches", str(args.max_action_batches),
                "--batch_size", str(args.eval_batch_size),
            ],
            logs / "action_sensitivity.log",
            cwd=root,
        ))
    if not args.skip_ttc:
        commands.append(_run(
            "proposer_ttc",
            [
                py, "-m", "wm3d_v3.eval.proposer_ttc_eval",
                "--cfg", str(args.cfg),
                "--ckpt", str(args.ckpt),
                "--out", str(artifacts["proposer_ttc"]),
                "--max_batches", str(args.max_ttc_batches),
                "--batch_size", str(args.eval_batch_size),
                "--score_progress_weight", str(args.score_progress_weight),
                "--score_terminal_weight", str(args.score_terminal_weight),
                "--score_plausibility_weight", str(args.score_plausibility_weight),
            ],
            logs / "proposer_ttc.log",
            cwd=root,
        ))
    if not args.skip_policy_probe:
        commands.append(_run(
            "policy_probe",
            [
                py, "-m", "wm3d_v3.eval.policy_rollout_probe",
                "--cfg", str(args.cfg),
                "--ckpt", str(args.ckpt),
                "--out", str(artifacts["policy_probe"]),
                "--max_batches", str(args.max_policy_batches),
                "--batch_size", str(args.eval_batch_size),
                "--score_progress_weight", str(args.score_progress_weight),
                "--score_terminal_weight", str(args.score_terminal_weight),
                "--score_plausibility_weight", str(args.score_plausibility_weight),
            ],
            logs / "policy_probe.log",
            cwd=root,
        ))
    if not args.skip_offline_replay:
        commands.append(_run(
            "offline_replay_adapter",
            [
                py, "-m", "wm3d_v3.benchmarks.run_adapter",
                "--adapter", "offline_replay",
                "--cfg", str(args.cfg),
                "--ckpt", str(args.ckpt),
                "--out", str(artifacts["offline_replay"]),
                "--max_tasks", str(args.max_offline_replay_tasks),
                "--max_steps", "1",
                "--split", "val",
                "--success_pose_l1_threshold", str(args.offline_replay_success_pose_l1_threshold),
                "--score_progress_weight", str(args.score_progress_weight),
                "--score_terminal_weight", str(args.score_terminal_weight),
                "--score_plausibility_weight", str(args.score_plausibility_weight),
            ],
            logs / "offline_replay_adapter.log",
            cwd=root,
        ))
    if not args.skip_libero_probe:
        commands.append(_run(
            "libero_probe",
            [
                libero_py, "-m", "wm3d_v3.benchmarks.libero_probe",
                "--root", str(args.libero_root),
                "--suite", str(args.libero_suite),
                "--out", str(artifacts["libero_probe"]),
            ],
            logs / "libero_probe.log",
            cwd=root,
        ))
    if not args.skip_demo:
        commands.append(_run(
            "demo_gif",
            [
                py, "-m", "wm3d_v3.eval.make_demo_gif",
                "--cfg", str(args.cfg),
                "--ckpt", str(args.ckpt),
                "--out_dir", str(artifacts["demo_dir"]),
                "--n_clips", str(args.demo_clips),
            ],
            logs / "demo_gif.log",
            cwd=root,
        ))
    if args.libero_trace_input:
        commands.append(_run(
            "libero_trace_summary",
            [
                py, "-m", "wm3d_v3.benchmarks.libero_trace_summary",
                "--input", *args.libero_trace_input,
                "--out", str(artifacts["libero_trace_summary"]),
            ],
            logs / "libero_trace_summary.log",
            cwd=root,
        ))

    benchmark_report = {name: asdict(probe) for name, probe in probe_benchmarks().items()}
    artifacts["benchmark_probe"].write_text(json.dumps(benchmark_report, indent=2, sort_keys=True))

    eval_report = _read_json(artifacts["eval"])
    world3d_claim_report = _read_json(artifacts["world3d_claim"])
    world_prior_report = _read_json(artifacts["world_prior_eval"])
    action_report = _read_json(artifacts["action_sensitivity"])
    ttc_report = _read_json(artifacts["proposer_ttc"])
    policy_report = _read_json(artifacts["policy_probe"])
    offline_replay_report = _read_json(artifacts["offline_replay"])
    libero_probe_report = _read_json(artifacts["libero_probe"])
    libero_trace_report = _read_json(artifacts["libero_trace_summary"])
    baseline_report = _read_json(args.baseline_eval) if args.baseline_eval else None

    command_ok = {item["name"]: item["ok"] for item in commands}
    gates = {
        "world_core_eval": bool(command_ok.get("world_core_eval") and eval_report),
        "world3d_claim_eval": bool(command_ok.get("world3d_claim_eval") and world3d_claim_report),
        "world_prior_generation": bool(command_ok.get("world_prior_eval") and world_prior_report),
        "action_counterfactual": bool(command_ok.get("action_sensitivity") and action_report),
        "offline_ttc": bool(command_ok.get("proposer_ttc") and ttc_report),
        "policy_action_output": bool(command_ok.get("policy_probe") and policy_report and policy_report.get("can_output_actions")),
        "offline_replay_adapter": bool(command_ok.get("offline_replay_adapter") and offline_replay_report),
        "libero_task_api": bool(command_ok.get("libero_probe") and libero_probe_report and libero_probe_report.get("task_api_available")),
        "libero_env_api": bool(command_ok.get("libero_probe") and libero_probe_report and libero_probe_report.get("env_api_available")),
        "libero_remote_trace": bool(command_ok.get("libero_trace_summary") and libero_trace_report and libero_trace_report.get("trace_steps", 0) > 0),
        "libero_remote_trace_has_policy_candidates": bool(libero_trace_report and libero_trace_report.get("candidate_score_steps", 0) > 0),
        "libero_remote_trace_has_frames": bool(libero_trace_report and libero_trace_report.get("frame_steps", 0) > 0),
        "libero_remote_trace_has_binary_success": bool(
            (libero_trace_report or {}).get("training_signal", {}).get("binary_success_supervision", False)
        ),
        "demo_artifacts": bool(args.no_video or (command_ok.get("demo_gif") and artifacts["demo_dir"].exists())),
        "external_benchmark_available": any(v["status"] == "available" for v in benchmark_report.values()),
    }
    gates["system_scaffold_complete"] = all(
        gates[key] for key in (
            "world_core_eval",
            "action_counterfactual",
            "offline_ttc",
            "policy_action_output",
            "offline_replay_adapter",
        )
    )

    report = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "out_dir": str(args.out_dir),
        "mode": {
            "video_generation_active": False,
            "hunyuan_generation_active": False,
            "rgb_metrics_active": not args.no_video,
            "rough_demo_gif_active": not args.skip_demo,
            "world_prior_eval_active": run_world_prior_eval,
            "video_generation_optional": True,
            "wm3d_python": py,
            "libero_python": libero_py,
        },
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "commands": commands,
        "gates": gates,
        "metrics": {
            "world_core_eval_ALL": _metric_subset(eval_report),
            "world3d_claim": _world3d_claim_subset(world3d_claim_report),
            "world_prior_generation_ALL": (world_prior_report or {}).get("metrics", {}).get("ALL", {}) if isinstance(world_prior_report, dict) else {},
            "action_sensitivity_ALL": _action_subset(action_report),
            "proposer_ttc": _ttc_subset(ttc_report),
            "policy_probe": _policy_subset(policy_report),
            "offline_replay_adapter": _adapter_subset(offline_replay_report),
            "libero_probe": _libero_subset(libero_probe_report),
            "libero_trace_summary": _trace_subset(libero_trace_report),
            "proposer_ttc_score_weights": {
                "progress": args.score_progress_weight,
                "terminal": args.score_terminal_weight,
                "plausibility": args.score_plausibility_weight,
            },
            "eval_vs_baseline": _compare_eval(eval_report or {}, baseline_report),
        },
        "benchmarks": benchmark_report,
        "interpretation": {
            "can_output_actions": "yes, via wm3d_v3.policy.select_action_chunk; benchmark validation still required",
            "is_complete_vla": "no, this report validates offline world-model/proposer scaffolding plus cached-data adapter replay",
            "can_run_professional_benchmarks_now": gates["external_benchmark_available"],
            "next_missing_loop": "mixed LIBERO success/failure traces for evaluator/proposer training",
            "ttc_score_note": "Default learned TTC score excludes plausibility because current configs do not train plausibility negatives.",
            "video_note": "RGB/video generation is optional; --no_video validates latent prediction/action outputs without activating the renderer.",
            "world3d_claim_note": "world3d_claim_eval is the report-facing native-3D evidence: motion-region depth, depth-delta prediction, and real-vs-counterfactual action win rates.",
            "world_prior_note": "world_prior_eval is the generative 3D-native evidence: text-only/text+context/text+action/full prior token generation with depth/RGB/Hunyuan-compatible outputs.",
        },
    }
    artifacts["system_report"].write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "system_report": str(artifacts["system_report"]),
        "gates": gates,
        "commands": command_ok,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
