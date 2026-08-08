#!/usr/bin/env python3
"""Falsifiable promotion gate for each V7 Stage1-P phase."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SCHEMA = "wm3d_v7_stage1_planner_gate_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dynamics", "planner", "joint"), required=True)
    parser.add_argument("--offline-report", type=Path, required=True)
    parser.add_argument("--closed-loop-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    offline = json.loads(args.offline_report.read_text())
    if offline.get("schema") != "wm3d_v7_stage1_planner_offline_eval_v1":
        raise SystemExit("unexpected offline report schema")
    checks: dict[str, dict] = {}

    def add(name: str, value, threshold: float, operator: str = ">=") -> None:
        passed = value is not None and (value >= threshold if operator == ">=" else value > threshold)
        checks[name] = {"value": value, "operator": operator, "threshold": threshold, "passed": passed}

    native = offline["native_dynamics"]
    add("effect_gain_h8_lower95", native["effect_gain_h8"]["lower95"], 0.10, ">")
    add("effect_gain_h32_lower95", native["effect_gain_h32"]["lower95"], 0.0, ">")
    if args.phase in {"planner", "joint"}:
        true = offline["planner_true_future"]
        imagined = offline["planner_imagined_future"]
        add("true_future_success_auc", true["success_auc"], 0.80)
        add("imagined_uplift_retention", imagined["uplift_retention_vs_true"], 0.70)
        add("imagined_mixed_success_at1_uplift", imagined["mixed_success_at1_uplift"], 0.15)
        add("candidate_oracle_uplift", imagined["mixed_candidate_oracle_uplift"], 0.10)
    if args.phase == "joint":
        if args.closed_loop_report is None:
            add("closed_loop_success_uplift", None, 0.05)
        else:
            closed = json.loads(args.closed_loop_report.read_text())
            if closed.get("schema") != "wm3d_v7_stage1_planner_closed_loop_eval_v1":
                raise SystemExit("unexpected closed-loop report schema")
            add("closed_loop_success_uplift", closed.get("success_uplift_vs_stage0"), 0.05)
    report = {
        "schema": SCHEMA,
        "phase": args.phase,
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "offline_report": str(args.offline_report.resolve()),
        "closed_loop_report": str(args.closed_loop_report.resolve()) if args.closed_loop_report else None,
        "automatic_launch_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
