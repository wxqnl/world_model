#!/usr/bin/env python3
"""Prepare, score, and finalize the WorldArena context-pyramid val5 diagnostic."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from scripts.eval_worldarena_context_pyramid_val5 import (
    ALLOWED_EPISODES,
    PROJECT_ROOT,
    read_manifest,
    val_output_name,
)
from scripts.eval_worldarena_s1 import read_initial_frame, select_instruction
from scripts.worldarena_context_pyramid_val import (
    ProtocolError,
    aligned_video_psnr,
    locked_grid,
    select_candidate,
    variant_name,
)


STANDARD_METRICS = ("image_quality", "dynamic_degree", "motion_smoothness")
EXPECTED_VARIANTS = (
    "baseline",
    *(variant_name(config) for config in locked_grid()),
)
TEST_EPISODE_PATTERN = re.compile(r"episode4[0-9](?![0-9])")


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def _safe_symlink(source: Path, target: Path) -> None:
    source = Path(source).resolve(strict=True)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        try:
            if target.resolve(strict=True) == source:
                return
        except FileNotFoundError:
            pass
        raise ProtocolError(f"symlink collision: {target}")
    target.symlink_to(source)


def _instruction(row: Mapping[str, Any]) -> str:
    path = Path(str(row.get("instruction_file", "")))
    if not path.is_file():
        raise ProtocolError(f"missing instruction file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"instruction payload is not an object: {path}")
    return select_instruction(payload, str(row["task"]))


def _write_initial_frame(video: Path, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    frame = read_initial_frame(Path(video))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), frame):
        raise ProtocolError(f"cannot write initial frame: {target}")


def prepare_variant_summary(
    panel: Sequence[Mapping[str, Any]],
    prediction_dir: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Create a diagnostic-only summary without loosening the formal test preparer."""
    if len(panel) != 5 or len({str(row.get("id", "")) for row in panel}) != 5:
        raise ProtocolError("diagnostic panel coverage must equal five unique ids")
    if any(int(row.get("episode", -1)) not in ALLOWED_EPISODES for row in panel):
        raise ProtocolError("diagnostic rows must use episodes 36-39")
    prediction_dir = Path(prediction_dir)
    output_root = Path(output_root)
    summary: list[dict[str, Any]] = []
    for row in panel:
        name = val_output_name(row)
        prediction = prediction_dir / name
        if not prediction.is_file() or prediction.stat().st_size <= 0:
            raise ProtocolError(f"missing prediction: {prediction}")
        ground_truth = Path(str(row["video_file"])).resolve(strict=True)
        shaped_gt = (
            output_root
            / "structured_gt"
            / str(row["task"])
            / "source"
            / "video"
            / "frames"
            / ground_truth.name
        )
        _safe_symlink(ground_truth, shaped_gt)
        image = output_root / "initial" / f"{Path(name).stem}.png"
        _write_initial_frame(ground_truth, image)
        summary.append(
            {
                "gt_path": str(shaped_gt.absolute()),
                "image": str(image.resolve()),
                "prompt": [_instruction(row)],
                "generated_name": name,
                "id": str(row["id"]),
            }
        )
    return summary


def _tree_identities(root: Path, *, generated: bool) -> dict[tuple[str, str], list[Path]]:
    base = Path(root) / ("generated_dataset" if generated else "gt_dataset")
    identities: dict[tuple[str, str], list[Path]] = {}
    if not base.is_dir():
        return identities
    for task in sorted(path for path in base.iterdir() if path.is_dir()):
        for episode in sorted(path for path in task.iterdir() if path.is_dir()):
            video = episode / "1" / "video" if generated else episode / "video"
            frames = sorted(video.glob("frame_*.jpg")) if video.is_dir() else []
            identities[(task.name, episode.name)] = frames
    return identities


def validate_diagnostic_tree(root: Path, *, expected_count: int = 5) -> None:
    ground_truth = _tree_identities(root, generated=False)
    generated = _tree_identities(root, generated=True)
    if (
        len(ground_truth) != expected_count
        or len(generated) != expected_count
        or set(ground_truth) != set(generated)
    ):
        raise ProtocolError(
            "official frame-tree coverage mismatch: "
            f"gt={len(ground_truth)}, generated={len(generated)}, expected={expected_count}"
        )
    for identity in sorted(ground_truth):
        episode = identity[1]
        if not re.fullmatch(r"episode(?:36|37|38|39)", episode):
            raise ProtocolError(f"frame tree escaped episodes 36-39: {identity}")
        gt_frames = ground_truth[identity]
        generated_frames = generated[identity]
        if not gt_frames or len(gt_frames) != len(generated_frames):
            raise ProtocolError(
                f"frame coverage mismatch for {identity}: "
                f"{len(gt_frames)} != {len(generated_frames)}"
            )


def assert_no_test_episode_reference(value: Any) -> None:
    if isinstance(value, str):
        if TEST_EPISODE_PATTERN.search(value):
            raise ProtocolError(f"test episode reference found: {value}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_no_test_episode_reference(str(key))
            assert_no_test_episode_reference(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            assert_no_test_episode_reference(item)


def build_gpu_assignments(
    variants: Sequence[str], gpus: Sequence[int]
) -> dict[int, list[str]]:
    physical = tuple(int(gpu) for gpu in gpus)
    if len(physical) != 4 or set(physical) != {0, 1, 2, 3}:
        raise ProtocolError("metric workers require exactly node43 GPUs 0,1,2,3")
    if len(variants) != 7 or len(set(variants)) != 7:
        raise ProtocolError("metric workers require baseline plus six unique variants")
    if set(variants) != set(EXPECTED_VARIANTS):
        raise ProtocolError("metric variant set differs from the locked grid")
    assignments = {gpu: [] for gpu in physical}
    for index, variant in enumerate(variants):
        assignments[physical[index % len(physical)]].append(variant)
    return assignments


def _run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    provenance_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    provenance = {
        "command": list(command),
        "cwd": str(Path(cwd).resolve()),
        "started_unix": started,
        "finished_unix": time.time(),
        "returncode": int(result.returncode),
        "log": str(log_path.resolve()),
    }
    _atomic_json(provenance_path, provenance)
    if result.returncode != 0:
        raise ProtocolError(
            f"subprocess failed with code {result.returncode}; see {log_path}"
        )


def _load_panel(output_root: Path) -> list[dict[str, Any]]:
    path = Path(output_root) / "panel.jsonl"
    panel = read_manifest(path)
    if len(panel) != 5 or any(
        int(row.get("episode", -1)) not in ALLOWED_EPISODES for row in panel
    ):
        raise ProtocolError("runtime panel must be val-only coverage five")
    assert_no_test_episode_reference(panel)
    return panel


def _metric_base_config(
    config: Mapping[str, Any], variant: str, output_root: Path
) -> dict[str, Any]:
    base_path = _project_path(str(config["base_metric_config"]))
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ProtocolError("base WorldArena metric config is not a mapping")
    value = copy.deepcopy(dict(base))
    prepared = output_root / "prepared" / variant / "data"
    value.update(
        {
            "schema": "wm3d_v7_worldarena_context_pyramid_val5_metric_v1",
            "model_name": f"wm3d_context_pyramid_val5_{variant}",
            "checkpoint": str(_project_path(str(config["checkpoint"]))),
            "manifest": str(output_root / "panel.jsonl"),
            "generation_summary": str(output_root / "generation_summary.json"),
            "expected_coverage": 5,
            "metric_devices": [0, 1, 2, 3],
            "forbid_devices": [4, 5, 6, 7],
            "generated_videos": str(output_root / "rendered" / variant),
            "gt_videos": str(output_root / "flat_gt"),
            "data": {
                "gt_path": str(prepared / "gt_dataset"),
                "val_base": str(prepared / "generated_dataset"),
            },
            "save_path": str(output_root / "metrics" / variant / "standard"),
            "save_path_action_following": str(
                output_root / "metrics" / variant / "standard"
            ),
        }
    )
    return value


def prepare_all(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _project_path(str(config["output_root"]))
    panel = _load_panel(output_root)
    python = _project_path(str(config.get("python", sys.executable)))
    worldarena = _project_path(str(config["worldarena"]["repo"]))
    preprocessor = worldarena / "video_quality" / "preprocess_datasets.py"
    records: dict[str, Any] = {}

    flat_gt = output_root / "flat_gt"
    for row in panel:
        _safe_symlink(Path(str(row["video_file"])), flat_gt / val_output_name(row))

    for variant in EXPECTED_VARIANTS:
        prediction_dir = output_root / "rendered" / variant
        prepared = output_root / "prepared" / variant
        summary = prepare_variant_summary(panel, prediction_dir, prepared)
        summary_path = prepared / "summary.json"
        _atomic_json(summary_path, summary)
        data_root = prepared / "data"
        valid_existing = False
        try:
            validate_diagnostic_tree(data_root, expected_count=5)
            valid_existing = True
        except ProtocolError:
            valid_existing = False
        if not valid_existing:
            command = [
                str(python),
                str(preprocessor),
                "--summary_json",
                str(summary_path),
                "--gen_video_dir",
                str(prediction_dir),
                "--output_base",
                str(data_root),
            ]
            _run_logged(
                command,
                cwd=worldarena / "video_quality",
                env=os.environ.copy(),
                log_path=output_root / "logs" / f"prepare_{variant}.log",
                provenance_path=output_root
                / "provenance"
                / f"prepare_{variant}.json",
            )
        validate_diagnostic_tree(data_root, expected_count=5)
        metric_config_path = output_root / "configs" / f"{variant}.yaml"
        _atomic_yaml(
            metric_config_path, _metric_base_config(config, variant, output_root)
        )
        records[variant] = {
            "summary": str(summary_path.resolve()),
            "data": str(data_root.resolve()),
            "metric_config": str(metric_config_path.resolve()),
            "coverage": 5,
        }
    result = {
        "schema": "wm3d_v7_worldarena_context_pyramid_val5_prepare_v1",
        "coverage": 5,
        "variants": records,
    }
    _atomic_json(output_root / "prepare_summary.json", result)
    return result


def _validate_jedi(output_root: Path, *, expected_count: int = 5) -> float:
    result_path = Path(output_root) / "results.json"
    names_path = Path(output_root) / "intersection_names.json"
    if not result_path.is_file() or not names_path.is_file():
        raise ProtocolError(f"missing JEDi output under {output_root}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    names = json.loads(names_path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or len(names) != expected_count or len(set(names)) != expected_count:
        raise ProtocolError(
            f"JEDi coverage mismatch: {len(names) if isinstance(names, list) else -1}"
        )
    assert_no_test_episode_reference(names)
    score = float(result.get("score", float("nan")))
    if not math.isfinite(score):
        raise ProtocolError("JEDi score contains NaN/Inf")
    return score


def _run_variant_metrics(
    *,
    variant: str,
    gpu: int,
    config: Mapping[str, Any],
    output_root: Path,
) -> None:
    python = _project_path(str(config.get("python", sys.executable)))
    worldarena = _project_path(str(config["worldarena"]["repo"]))
    metric_config = output_root / "configs" / f"{variant}.yaml"
    if not metric_config.is_file():
        raise ProtocolError(f"missing prepared metric config: {metric_config}")
    standard_root = output_root / "metrics" / variant / "standard"
    standard_command = [
        str(python),
        str(PROJECT_ROOT / "scripts" / "run_worldarena_metric_queue.py"),
        "--config",
        str(metric_config),
        "--worldarena-repo",
        str(worldarena),
        "--run-root",
        str(standard_root),
        "--gpu",
        str(gpu),
        "--expected-count",
        "5",
        "--metrics",
        *STANDARD_METRICS,
    ]
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu/stubs",
            "PYTHONPATH": ":".join(
                (
                    "/data/Minko/models/worldarena_v1/python_overlays/base",
                    "/data/Minko/models/worldarena_v1/python_overlays/mamba",
                    str(worldarena),
                    str(PROJECT_ROOT),
                )
            ),
        }
    )
    _run_logged(
        standard_command,
        cwd=PROJECT_ROOT,
        env=env,
        log_path=output_root / "logs" / f"metrics_{variant}_gpu{gpu}.log",
        provenance_path=output_root
        / "provenance"
        / f"metrics_{variant}_gpu{gpu}.json",
    )

    jedi_root = output_root / "metrics" / variant / "jepa_similarity"
    jedi_root.mkdir(parents=True, exist_ok=True)
    jedi_dir = _project_path(str(config["weights"]["jedi_dir"]))
    jedi_command = [
        str(python),
        str(worldarena / "video_quality" / "JEDi" / "batch.py"),
        "--real_dir",
        str(output_root / "flat_gt"),
        "--gen_dir",
        str(output_root / "rendered" / variant),
        "--num_frames",
        "16",
        "--batch_size",
        "4",
        "--num_workers",
        "2",
        "--max_samples",
        "5",
        "--save_intersection",
        str(jedi_root / "intersection_names.json"),
        "--model_dir",
        str(jedi_dir),
        "--config_path",
        str(worldarena / "video_quality" / "JEDi" / "configs" / "vith16_ssv2_16x2x3.yaml"),
        "--output_root",
        str(jedi_root),
    ]
    jedi_env = env.copy()
    jedi_env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": ":".join(
                (
                    "/data/Minko/models/worldarena_v1/python_overlays/jedi",
                    "/data/Minko/models/worldarena_v1/python_overlays/base",
                    str(worldarena),
                )
            ),
        }
    )
    _run_logged(
        jedi_command,
        cwd=jedi_dir,
        env=jedi_env,
        log_path=output_root / "logs" / f"jepa_{variant}_gpu{gpu}.log",
        provenance_path=output_root
        / "provenance"
        / f"jepa_{variant}_gpu{gpu}.json",
    )
    _validate_jedi(jedi_root, expected_count=5)


def run_metrics(config: Mapping[str, Any], gpus: Sequence[int]) -> dict[str, Any]:
    output_root = _project_path(str(config["output_root"]))
    assignments = build_gpu_assignments(EXPECTED_VARIANTS, gpus)
    failures: list[str] = []

    def worker(gpu: int, variants: Sequence[str]) -> list[str]:
        completed: list[str] = []
        for variant in variants:
            _run_variant_metrics(
                variant=variant,
                gpu=gpu,
                config=config,
                output_root=output_root,
            )
            completed.append(variant)
        return completed

    completed: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(worker, gpu, variants): gpu
            for gpu, variants in assignments.items()
        }
        for future in as_completed(futures):
            gpu = futures[future]
            try:
                completed[gpu] = future.result()
            except Exception as exc:
                failures.append(f"GPU{gpu}: {type(exc).__name__}: {exc}")
    result = {
        "schema": "wm3d_v7_worldarena_context_pyramid_val5_metrics_v1",
        "assignments": assignments,
        "completed": completed,
        "failures": failures,
    }
    _atomic_json(output_root / "metric_run_summary.json", result)
    if failures:
        raise ProtocolError(f"metric workers failed: {failures}")
    return result


def _read_official_mean(root: Path, metric: str, *, expected_count: int = 5) -> float:
    path = Path(root) / "per_metric" / metric / "generated_results.json"
    if not path.is_file():
        raise ProtocolError(f"missing official metric result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get(metric)
    if not isinstance(value, list) or len(value) < 2 or not isinstance(value[1], list):
        raise ProtocolError(f"invalid result structure for {metric}")
    details = value[1]
    paths = [str(item.get("video_path", "")) for item in details if isinstance(item, Mapping)]
    if len(details) != expected_count or len(paths) != expected_count or len(set(paths)) != expected_count:
        raise ProtocolError(f"official {metric} coverage mismatch: {len(details)}")
    normalized = [float(item["video_results_normalized"]) for item in details]
    if not all(math.isfinite(item) for item in normalized):
        raise ProtocolError(f"official {metric} contains NaN/Inf")
    return float(np.mean(normalized))


def aggregate_variant(
    output_root: Path,
    variant: str,
    panel: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    standard_root = Path(output_root) / "metrics" / variant / "standard"
    values = {
        metric: _read_official_mean(standard_root, metric, expected_count=5)
        for metric in STANDARD_METRICS
    }
    values["jepa_similarity"] = _validate_jedi(
        Path(output_root) / "metrics" / variant / "jepa_similarity",
        expected_count=5,
    )
    details = []
    for row in panel:
        details.append(
            {
                "id": str(row["id"]),
                **aligned_video_psnr(
                    Path(output_root) / "rendered" / variant / val_output_name(row),
                    Path(str(row["video_file"])),
                ),
            }
        )
    values["psnr"] = float(np.mean([item["mean"] for item in details]))
    values["coverage"] = 5
    _atomic_json(
        Path(output_root) / "psnr" / f"{variant}.json",
        {"aggregate": values["psnr"], "details": details},
    )
    return values


def _read_video_bgr(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        if not capture.isOpened():
            raise ProtocolError(f"cannot open contact-sheet video: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ProtocolError(f"contact-sheet video has no frames: {path}")
    return np.stack(frames)


def _labeled_tile(frame: np.ndarray, label: str) -> np.ndarray:
    tile = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (240, 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def write_contact_sheets(
    output_root: Path,
    panel: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    decision: str,
) -> list[Path]:
    paths: list[Path] = []
    for row in panel:
        name = val_output_name(row)
        gt = _read_video_bgr(Path(str(row["video_file"])))
        baseline = _read_video_bgr(Path(output_root) / "rendered" / "baseline" / name)
        selected = _read_video_bgr(Path(output_root) / "rendered" / candidate / name)
        if len(gt) != len(baseline) or len(gt) != len(selected):
            raise ProtocolError(f"contact-sheet frame mismatch for {row['id']}")
        indices = np.linspace(0, len(gt) - 1, num=5).round().astype(int).tolist()
        initial = np.repeat(gt[:1], 5, axis=0)
        rows = []
        for label, video in (
            ("initial", initial),
            ("GT", gt),
            ("baseline", baseline),
            (f"{decision}:{candidate}", selected),
        ):
            rows.append(
                np.concatenate(
                    [_labeled_tile(video[index if label != "initial" else 0], f"{label} t={index}") for index in indices],
                    axis=1,
                )
            )
        sheet = np.concatenate(rows, axis=0)
        path = Path(output_root) / "contact_sheets" / f"{Path(name).stem}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), sheet) or path.stat().st_size <= 0:
            raise ProtocolError(f"cannot write contact sheet: {path}")
        paths.append(path)
    return paths


def finalize(config: Mapping[str, Any]) -> dict[str, Any]:
    output_root = _project_path(str(config["output_root"]))
    panel = _load_panel(output_root)
    aggregates = {
        variant: aggregate_variant(output_root, variant, panel)
        for variant in EXPECTED_VARIANTS
    }
    selection = select_candidate(
        aggregates["baseline"],
        {name: values for name, values in aggregates.items() if name != "baseline"},
    )
    candidate_values = {
        name: values for name, values in aggregates.items() if name != "baseline"
    }
    visualized = selection["selected"] or max(
        candidate_values, key=lambda name: float(candidate_values[name]["psnr"])
    )
    report = {
        "schema": "wm3d_v7_worldarena_context_pyramid_val5_result_v1",
        "decision": selection["decision"],
        "selected": selection["selected"],
        "visualized_candidate": visualized,
        "aggregates": aggregates,
        "gate_checks": selection["checks"],
        "gate": {
            "min_psnr_gain_db": 0.25,
            "min_image_quality_ratio": 1.0,
            "min_jepa_ratio": 0.97,
            "min_dynamic_ratio": 0.97,
            "min_smoothness_ratio": 0.97,
            "psnr_tie_db": 0.02,
        },
        "ids": [str(row["id"]) for row in panel],
        "episodes": [int(row["episode"]) for row in panel],
        "selection_scope": "five-record aggregate only",
        "test500_authorized": False,
    }
    assert_no_test_episode_reference(report)
    contact_sheets = write_contact_sheets(
        output_root,
        panel,
        visualized,
        decision=str(selection["decision"]),
    )
    report["contact_sheets"] = [str(path.resolve()) for path in contact_sheets]
    _atomic_json(output_root / "selection_report.json", report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (args.prepare or args.metrics or args.finalize):
        raise ProtocolError("select at least one of --prepare, --metrics, or --finalize")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ProtocolError("diagnostic config must be a mapping")
    if args.prepare:
        result = prepare_all(config)
        print(json.dumps({"status": "prepared", "variants": len(result["variants"])}))
    if args.metrics:
        result = run_metrics(config, args.gpus)
        print(json.dumps({"status": "metrics_complete", "workers": len(result["completed"])}))
    if args.finalize:
        result = finalize(config)
        print(
            json.dumps(
                {
                    "status": "finalized",
                    "decision": result["decision"],
                    "selected": result["selected"],
                }
            )
        )


if __name__ == "__main__":
    main()
