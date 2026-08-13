#!/usr/bin/env python3
"""Read-only operator summary for WM3D 5B data, training, and evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

# Keep direct execution equivalent to `run_wm3d.sh`, whose entrypoint exports the checkout.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wm3d.data.manifest_contract import load_data_profile, sha256_file  # noqa: E402
from wm3d.training.runtime_contract import load_materialized_runtime  # noqa: E402


SCHEMA = "wm3d_5b_operator_report_v1"


class ReportError(RuntimeError):
    pass


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"not a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportError(f"JSON root must be an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReportError(f"not a regular JSONL file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ReportError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    return True


def _jsonl_summary(path: Path) -> tuple[int, dict[str, int]]:
    """Count a potentially multi-million-row manifest without retaining it."""

    if path.is_symlink() or not path.is_file():
        raise ReportError(f"not a regular JSONL file: {path}")
    count = 0
    splits: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ReportError(f"JSONL row {line_number} is not an object: {path}")
            count += 1
            splits[f"{value.get('source', '?')}:{value.get('split', '?')}"] += 1
    return count, dict(sorted(splits.items()))


def _optional(
    path: Path | None,
    label: str,
    failures: list[str],
    pending: list[str],
    *,
    allow_incomplete: bool,
) -> Path | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file():
        destination = pending if allow_incomplete else failures
        destination.append(f"missing {label}: {path}")
        return None
    return path


def _checkpoint_summary(path: Path, failures: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "passed": False}
    if path.is_symlink() or not path.is_dir():
        failures.append(f"checkpoint is missing or not a real directory: {path}")
        return summary
    try:
        metadata = _json(path / "metadata.json")
        manifest = _json(path / "MANIFEST.json")
        committed = _json(path / "COMMITTED.json")
        if metadata.get("schema") != "wm3d_v8_distributed_checkpoint_v2":
            raise ReportError("checkpoint metadata schema mismatch")
        if manifest.get("schema") != "wm3d_v8_distributed_checkpoint_v2":
            raise ReportError("checkpoint manifest schema mismatch")
        if committed.get("schema") != "wm3d_v8_distributed_checkpoint_commit_v2":
            raise ReportError("checkpoint commit schema mismatch")
        step = int(metadata["step"])
        if path.name != f"step_{step:08d}":
            raise ReportError("checkpoint directory and metadata step differ")
        if {int(manifest["step"]), int(committed["step"])} != {step}:
            raise ReportError("checkpoint control files disagree on step")
        if sha256_file(path / "metadata.json") != committed["metadata_sha256"]:
            raise ReportError("checkpoint metadata SHA mismatch")
        if sha256_file(path / "MANIFEST.json") != committed["manifest_sha256"]:
            raise ReportError("checkpoint manifest SHA mismatch")
        if _canonical_sha256(manifest) != committed["manifest_content_sha256"]:
            raise ReportError("checkpoint canonical manifest SHA mismatch")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ReportError("checkpoint manifest has no payload")
        actual = {
            item.relative_to(path).as_posix()
            for item in path.rglob("*")
            if item.is_file() and not item.is_symlink()
        }
        expected = set(files) | {"MANIFEST.json", "COMMITTED.json"}
        if actual != expected:
            raise ReportError(
                f"checkpoint file set mismatch: missing={sorted(expected-actual)} "
                f"extra={sorted(actual-expected)}"
            )
        for relative, evidence in files.items():
            payload = path / relative
            if payload.stat().st_size != int(evidence["size"]):
                raise ReportError(f"checkpoint payload size mismatch: {relative}")
            if sha256_file(payload) != evidence["sha256"]:
                raise ReportError(f"checkpoint payload SHA mismatch: {relative}")
        summary.update(
            {
                "passed": True,
                "step": step,
                "world_size": int(metadata["world_size"]),
                "committed_sha256": sha256_file(path / "COMMITTED.json"),
                "payload_file_count": len(files),
                "payload_bytes": sum(int(item["size"]) for item in files.values()),
            }
        )
    except Exception as error:
        failures.append(f"checkpoint validation failed: {error}")
    return summary


def _training_summary(
    run_root: Path,
    runtime: Mapping[str, Any] | None,
    failures: list[str],
    pending: list[str],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    metrics_path = run_root / "train_metrics.jsonl"
    summary: dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "records": 0,
        "finite": False,
    }
    if metrics_path.is_symlink() or not metrics_path.is_file():
        destination = pending if allow_incomplete else failures
        destination.append(f"training metrics missing: {metrics_path}")
        return summary
    rows = _jsonl(metrics_path)
    train_rows = [row for row in rows if "step" in row and "validation" not in row]
    validation_rows = [row for row in rows if isinstance(row.get("validation"), dict)]
    if not train_rows:
        failures.append("training metrics contain no optimizer-step records")
        return summary
    steps = [int(row["step"]) for row in train_rows]
    if steps != sorted(set(steps)):
        failures.append("training optimizer steps are duplicated or non-monotonic")
    if not _finite_numbers(rows):
        failures.append("training metrics contain NaN or Inf")
    last = train_rows[-1]
    loss_keys = sorted(
        key
        for key, value in last.items()
        if key not in {"step", "lr", "source_id", "grad_norm", "seconds_per_log_interval", "gradient_ownership"}
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )
    losses = {key: float(last[key]) for key in loss_keys}
    first_total = next(
        (float(row["total"]) for row in train_rows if isinstance(row.get("total"), (int, float))),
        None,
    )
    last_total = float(last["total"]) if isinstance(last.get("total"), (int, float)) else None
    loss_change = (
        None
        if first_total is None or last_total is None or first_total == 0.0
        else (last_total - first_total) / abs(first_total)
    )
    throughput = None
    if runtime is not None and len(train_rows) >= 2:
        previous = train_rows[-2]
        seconds = float(last.get("seconds_per_log_interval", 0.0))
        delta = int(last["step"]) - int(previous["step"])
        if seconds > 0 and delta > 0:
            throughput = delta * int(runtime["train"]["global_batch_size"]) / seconds
    ownership_path = run_root / "gradient_ownership.json"
    ownership = None
    if ownership_path.is_file() and not ownership_path.is_symlink():
        ownership = _json(ownership_path)
        ownership_schema_ok = (
            ownership.get("schema") == "wm3d_v8_gradient_ownership_v2"
        )
        if not ownership_schema_ok:
            failures.append("gradient ownership schema mismatch")
        owners = ownership.get("owners")
        required = {
            name: value
            for name, value in (owners.items() if isinstance(owners, dict) else ())
            if isinstance(value, dict) and value.get("required") is True
        }
        ownership_passed = ownership_schema_ok and bool(ownership.get("passed")) and bool(required) and all(
            value.get("passed") is True
            and int(value.get("nonzero_elements", 0)) > 0
            and int(value.get("nonfinite_elements", -1)) == 0
            for value in required.values()
        )
        if not ownership_passed:
            failures.append("required gradient ownership did not pass")
        ownership = {
            "passed": ownership_passed,
            "required_owner_count": len(required),
            "required_owners": sorted(required),
        }
    else:
        destination = pending if allow_incomplete else failures
        destination.append(f"gradient ownership receipt missing: {ownership_path}")
    summary.update(
        {
            "records": len(rows),
            "optimizer_records": len(train_rows),
            "validation_records": len(validation_rows),
            "finite": _finite_numbers(rows),
            "latest_step": int(last["step"]),
            "learning_rate": float(last["lr"]),
            "gradient_norm": float(last["grad_norm"]),
            "losses": losses,
            "first_total_loss": first_total,
            "last_total_loss": last_total,
            "relative_total_loss_change": loss_change,
            "global_samples_per_second": throughput,
            "latest_validation": validation_rows[-1] if validation_rows else None,
            "gradient_ownership": ownership,
        }
    )
    return summary


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    pending: list[str] = []
    report: dict[str, Any] = {"schema": SCHEMA}

    data: dict[str, Any] = {}
    data_profile_path = _optional(
        args.data_profile,
        "data profile",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    if data_profile_path is not None:
        try:
            profile = load_data_profile(data_profile_path)
            data["profile"] = {
                "name": profile.name,
                "sha256": profile.profile_sha256,
                "sources": len(profile.sources),
                "source_names": [source.name for source in profile.sources],
            }
        except Exception as error:
            failures.append(f"data profile validation failed: {error}")
    task_path = _optional(
        args.task_manifest,
        "cache task manifest",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    if task_path is not None:
        count, splits = _jsonl_summary(task_path)
        data["cache_tasks"] = {
            "count": count,
            "by_source_split": splits,
            "manifest_sha256": sha256_file(task_path),
        }
        if not count:
            failures.append("cache task manifest is empty")
    episode_path = _optional(
        args.episode_index,
        "episode index",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    episode_seal_path = _optional(
        args.episode_seal,
        "episode seal",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    if episode_path is not None and episode_seal_path is not None:
        count, splits = _jsonl_summary(episode_path)
        seal = _json(episode_seal_path)
        passed = (
            seal.get("schema") == "wm3d_v8_episode_cache_seal_v4"
            and seal.get("episode_index_sha256") == sha256_file(episode_path)
            and int(seal.get("episode_count", -1)) == count
        )
        if task_path is not None:
            passed = passed and seal.get("task_manifest_sha256") == sha256_file(
                task_path
            )
        if not passed:
            failures.append("episode index/seal binding failed")
        data["episodes"] = {
            "passed": passed,
            "count": count,
            "by_source_split": splits,
            "index_sha256": sha256_file(episode_path),
        }
    window_path = _optional(
        args.window_index,
        "window index",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    window_seal_path = _optional(
        args.window_seal,
        "window seal",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    if window_path is not None and window_seal_path is not None:
        count, splits = _jsonl_summary(window_path)
        seal = _json(window_seal_path)
        passed = (
            seal.get("schema") == "wm3d_v8_window_index_seal_v3"
            and seal.get("window_index_sha256") == sha256_file(window_path)
            and int(seal.get("window_count", -1)) == count
        )
        if episode_seal_path is not None:
            passed = passed and seal.get("episode_seal_sha256") == sha256_file(
                episode_seal_path
            )
        if not passed:
            failures.append("window index/seal binding failed")
        data["windows"] = {
            "passed": passed,
            "count": count,
            "by_source_split": splits,
            "index_sha256": sha256_file(window_path),
        }
    report["data"] = data

    runtime: dict[str, Any] | None = None
    runtime_path = _optional(
        args.runtime,
        "sealed runtime",
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    if runtime_path is not None:
        try:
            runtime_config, runtime_sha = load_materialized_runtime(runtime_path)
            runtime = runtime_config["runtime_profile"]
            report["runtime"] = {
                "passed": True,
                "path": str(runtime_path.resolve()),
                "sha256": runtime_sha,
                "model": runtime_config["model_profile"]["name"],
                "expected_parameters": int(
                    runtime_config["model_profile"]["expected_parameter_count"]
                ),
                "world_size": int(runtime["expected_world_size"]),
                "global_batch_size": int(runtime["train"]["global_batch_size"]),
                "total_steps": int(runtime["train"]["total_steps"]),
            }
        except Exception as error:
            failures.append(f"runtime validation failed: {error}")
            report["runtime"] = {"passed": False, "path": str(runtime_path)}

    report["training"] = _training_summary(
        args.run_root,
        runtime,
        failures,
        pending,
        allow_incomplete=args.allow_incomplete,
    )
    if args.expected_step is not None:
        latest = int(report["training"].get("latest_step", -1))
        if latest < args.expected_step:
            failures.append(f"training step {latest} is below expected {args.expected_step}")

    if args.checkpoint is not None:
        report["checkpoint"] = _checkpoint_summary(args.checkpoint, failures)
        if args.expected_step is not None and report["checkpoint"].get("step") != args.expected_step:
            failures.append("checkpoint step differs from expected step")

    if args.eval is not None:
        eval_path = _optional(
            args.eval,
            "eval receipt",
            failures,
            pending,
            allow_incomplete=args.allow_incomplete,
        )
        if eval_path is not None:
            receipt = _json(eval_path)
            coverage = receipt.get("coverage")
            expected_coverage = receipt.get("expected_coverage_lanes")
            expected_coverage_valid = (
                isinstance(expected_coverage, list)
                and bool(expected_coverage)
                and len(expected_coverage) == len(set(expected_coverage))
                and all(
                    isinstance(name, str)
                    and name in coverage
                    and isinstance(coverage[name], (int, float))
                    and not isinstance(coverage[name], bool)
                    and math.isfinite(float(coverage[name]))
                    and float(coverage[name]) > 0
                    for name in expected_coverage
                )
                if isinstance(coverage, dict)
                else False
            )
            passed = (
                receipt.get("schema") == "wm3d_v8_unified_offline_eval_v2"
                and receipt.get("all_metrics_finite") is True
                and _finite_numbers(receipt.get("metrics"))
                and isinstance(coverage, dict)
                and bool(coverage)
                and expected_coverage_valid
            )
            if not passed:
                failures.append("offline evaluation is non-finite or has zero coverage")
            if report.get("checkpoint", {}).get("committed_sha256") is not None and (
                receipt.get("checkpoint_committed_sha256")
                != report["checkpoint"]["committed_sha256"]
            ):
                failures.append("eval receipt belongs to another checkpoint")
                passed = False
            if args.expected_step is not None and receipt.get("checkpoint_step") != args.expected_step:
                failures.append("eval receipt checkpoint step differs from expected step")
                passed = False
            report["evaluation"] = {
                "passed": passed,
                "path": str(eval_path),
                "checkpoint_step": receipt.get("checkpoint_step"),
                "metrics": receipt.get("metrics"),
                "coverage": coverage,
                "expected_coverage_lanes": expected_coverage,
            }

    report["failures"] = failures
    report["pending"] = pending
    required = ["profile", "cache_tasks", "episodes", "windows"]
    data_complete = all(name in data for name in required)
    training_complete = bool(report["training"].get("finite")) and bool(
        (report["training"].get("gradient_ownership") or {}).get("passed")
    )
    acceptance_complete = (
        data_complete
        and report.get("runtime", {}).get("passed") is True
        and training_complete
        and report.get("checkpoint", {}).get("passed") is True
        and report.get("evaluation", {}).get("passed") is True
        and not failures
        and not pending
    )
    report["status"] = "PASS" if acceptance_complete else ("FAIL" if failures else "INCOMPLETE")
    return report, failures


def _human(report: Mapping[str, Any]) -> str:
    lines = [f"WM3D 5B pipeline: {report['status']}"]
    data = report.get("data", {})
    profile = data.get("profile")
    if profile:
        lines.append(f"  data: {profile['name']} ({profile['sources']} sources)")
    if data.get("episodes"):
        lines.append(f"  cache: {data['episodes']['count']:,} episodes, seal={'PASS' if data['episodes']['passed'] else 'FAIL'}")
    if data.get("windows"):
        lines.append(f"  windows: {data['windows']['count']:,}, seal={'PASS' if data['windows']['passed'] else 'FAIL'}")
    runtime = report.get("runtime")
    if runtime:
        lines.append(
            f"  model: {runtime.get('model')} / {runtime.get('expected_parameters', 0):,} params / "
            f"world {runtime.get('world_size')}"
        )
    training = report.get("training", {})
    if training.get("latest_step") is not None:
        lines.append(
            f"  train: step {training['latest_step']}, total={training.get('last_total_loss')}, "
            f"grad_norm={training.get('gradient_norm')}"
        )
        if training.get("global_samples_per_second") is not None:
            lines.append(f"  throughput: {training['global_samples_per_second']:.2f} samples/s")
        ownership = training.get("gradient_ownership") or {}
        lines.append(
            f"  gradients: {'PASS' if ownership.get('passed') else 'FAIL'} "
            f"({ownership.get('required_owner_count', 0)} required owners)"
        )
    checkpoint = report.get("checkpoint")
    if checkpoint:
        lines.append(
            f"  checkpoint: {'PASS' if checkpoint.get('passed') else 'FAIL'} "
            f"step={checkpoint.get('step')} size={checkpoint.get('payload_bytes', 0) / 2**30:.1f} GiB"
        )
    evaluation = report.get("evaluation")
    if evaluation:
        lines.append(
            f"  eval: {'PASS' if evaluation.get('passed') else 'FAIL'} "
            f"coverage_lanes={len(evaluation.get('coverage') or {})}"
        )
    for failure in report.get("failures", []):
        lines.append(f"  ! {failure}")
    for item in report.get("pending", []):
        lines.append(f"  - pending: {item}")
    lines.append("  Note: PASS proves pipeline integrity and finite supervision, not model capability.")
    return "\n".join(lines)


def _write_output(path: Path, report: Mapping[str, Any]) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-profile", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--episode-index", type=Path)
    parser.add_argument("--episode-seal", type=Path)
    parser.add_argument("--window-index", type=Path)
    parser.add_argument("--window-seal", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.allow_incomplete == args.require_complete:
        parser.error("select exactly one of --allow-incomplete or --require-complete")
    if args.expected_step is not None and args.expected_step <= 0:
        parser.error("--expected-step must be positive")
    report, failures = build_report(args)
    print(_human(report))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if args.output is not None:
        _write_output(args.output, report)
    if args.require_complete and (failures or report["status"] != "PASS"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
