#!/usr/bin/env python3
"""Generate the fixed WorldArena val5 native cache and seven renderer variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import torch
import yaml

from scripts.eval_worldarena_s1 import (
    _load_instruction,
    _model_episode,
    _write_h264_atomic,
    canonical_bimanual_actions,
    load_checkpoint_for_eval,
    probe_video,
    read_initial_frame,
    validate_temporal_contract,
)
from scripts.worldarena_context_pyramid_val import (
    ProtocolError,
    locked_grid,
    render_baseline,
    render_context_pyramid,
    select_locked_panel,
    variant_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EPISODES = (36, 37, 38, 39)
OUTPUT_SIZE = (640, 480)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ProtocolError(f"manifest line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ProtocolError("manifest is empty")
    return rows


def val_output_name(row: Mapping[str, Any]) -> str:
    task = str(row.get("task", ""))
    episode = int(row.get("episode", -1))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ProtocolError(f"unsafe task name: {task!r}")
    if episode not in ALLOWED_EPISODES:
        raise ProtocolError("validation output accepts only episodes 36-39")
    return f"{task}_episode{episode}.mp4"


def build_panel_audit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel = select_locked_panel(rows)
    serialized = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in panel
    ]
    audit = {
        "schema": "wm3d_v7_worldarena_context_pyramid_val5_panel_v1",
        "allowed_episodes": list(ALLOWED_EPISODES),
        "forbidden_test_episode_range": [40, 49],
        "selection": {
            "task_sort": "lexicographic",
            "task_indices": [0, 12, 24, 36, 49],
            "episodes": [36, 37, 38, 39, 36],
        },
        "ids": [str(row["id"]) for row in panel],
        "manifest_row_sha256": [
            hashlib.sha256(value.encode("utf-8")).hexdigest() for value in serialized
        ],
        "future_gt_used_for_inference": False,
        "inference_inputs": ["initial_rgb", "instruction", "physical_actions"],
    }
    return panel, audit


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _uint8_rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint8:
        output = array.copy()
    else:
        array = array.astype(np.float32, copy=False)
        if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
            raise ProtocolError("RGB renderer input must be finite and in [0,1]")
        output = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    if output.ndim != 3 or output.shape[-1] != 3:
        raise ProtocolError("RGB renderer input must be H,W,3")
    return output


def _native64(native_rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(native_rgb)
    if value.ndim != 4:
        raise ProtocolError("native RGB must be rank four")
    if value.shape[1] == 3:
        value = np.moveaxis(value, 1, -1)
    if value.shape[-1] != 3:
        raise ProtocolError("native RGB must be T,H,W,3 or T,3,H,W")
    value = value.astype(np.float32, copy=False)
    if not np.isfinite(value).all() or value.min() < 0.0 or value.max() > 1.0:
        raise ProtocolError("native RGB must be finite and in [0,1]")
    return np.stack(
        [cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA) for frame in value]
    ).astype(np.float32)


def render_native_cache(
    initial_rgb: np.ndarray,
    native_rgb: np.ndarray,
    output_root: Path,
    name: str,
    *,
    fps: int,
    video_writer: Callable[[Sequence[np.ndarray], Path, int], None] = _write_h264_atomic,
) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+_episode(?:36|37|38|39)\.mp4", name):
        raise ProtocolError(f"unsafe validation output name: {name!r}")
    if not isinstance(fps, int) or fps <= 0:
        raise ProtocolError("fps must be a positive integer")
    initial_u8 = _uint8_rgb(initial_rgb)
    native = _native64(native_rgb)
    rendered: dict[str, np.ndarray] = {
        "baseline": render_baseline(initial_u8, native, output_size=OUTPUT_SIZE)
    }
    rendered.update(
        {
            variant_name(config): render_context_pyramid(
                initial_u8, native, config, output_size=OUTPUT_SIZE
            )
            for config in locked_grid()
        }
    )
    first_rgb = cv2.resize(initial_u8, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    first_bgr = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2BGR)
    written: dict[str, Path] = {}
    for renderer, predicted in rendered.items():
        frames_bgr = [first_bgr]
        frames_bgr.extend(
            cv2.cvtColor(_uint8_rgb(frame), cv2.COLOR_RGB2BGR) for frame in predicted
        )
        path = Path(output_root) / "rendered" / renderer / name
        path.parent.mkdir(parents=True, exist_ok=True)
        video_writer(frames_bgr, path, fps)
        if not path.is_file() or path.stat().st_size <= 0:
            raise ProtocolError(f"renderer did not produce a non-empty video: {path}")
        written[renderer] = path
    return written


def _validate_source_files(panel: Sequence[Mapping[str, Any]]) -> None:
    for row in panel:
        if int(row.get("episode", -1)) not in ALLOWED_EPISODES:
            raise ProtocolError("source validation escaped episodes 36-39")
        for key in ("video_file", "hdf5_file", "instruction_file"):
            path = Path(str(row.get(key, "")))
            if not path.is_file():
                raise ProtocolError(f"missing {key} for {row.get('id')}: {path}")


def _load_inference_inputs(
    row: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    video_path = Path(str(row["video_file"]))
    hdf5_path = Path(str(row["hdf5_file"]))
    probe = probe_video(video_path)
    stats = row.get("gripper_train_stats")
    if not isinstance(stats, Mapping):
        raise ProtocolError(f"missing gripper stats for {row.get('id')}")
    left_stats, right_stats = stats.get("left"), stats.get("right")
    if not isinstance(left_stats, Mapping) or not isinstance(right_stats, Mapping):
        raise ProtocolError(f"invalid gripper stats for {row.get('id')}")
    left, right, hdf5_frames = canonical_bimanual_actions(
        hdf5_path,
        float(left_stats["midpoint"]),
        float(right_stats["midpoint"]),
    )
    future_frames = validate_temporal_contract(hdf5_frames, int(probe["frames"]))
    initial_bgr = read_initial_frame(video_path)
    initial_rgb = cv2.cvtColor(initial_bgr, cv2.COLOR_BGR2RGB)
    return initial_rgb, left, right, future_frames


def _cache_paths(root: Path, row: Mapping[str, Any]) -> tuple[Path, Path]:
    stem = Path(val_output_name(row)).stem
    return root / "native" / f"{stem}.npz", root / "native" / f"{stem}.json"


def _load_valid_cache(
    cache_path: Path, audit_path: Path, expected_id: str
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file() or not audit_path.is_file():
        return None
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("id") != expected_id
            or audit.get("no_future_ground_truth") is not True
            or audit.get("native_resolution") != [64, 64]
        ):
            return None
        with np.load(cache_path) as payload:
            initial = np.asarray(payload["initial_rgb"])
            native = _native64(np.asarray(payload["native_rgb64"]))
        if len(native) <= 0:
            return None
        return initial, native
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ProtocolError):
        return None


def run_generation(
    config: Mapping[str, Any],
    *,
    physical_device: int,
    manifest_reader: Callable[[Path], list[dict[str, Any]]] = read_manifest,
    checkpoint_loader: Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]] = load_checkpoint_for_eval,
) -> dict[str, Any]:
    """Validate the panel first, then lazily load the model only for missing caches."""
    manifest_path = _project_path(str(config["manifest"]))
    rows = manifest_reader(manifest_path)
    panel, panel_audit = build_panel_audit(rows)
    output_root = _project_path(str(config["output_root"]))
    _atomic_json(output_root / "panel_audit.json", panel_audit)
    _atomic_jsonl(output_root / "panel.jsonl", panel)
    _validate_source_files(panel)

    cached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing: list[dict[str, Any]] = []
    for row in panel:
        cache_path, audit_path = _cache_paths(output_root, row)
        value = _load_valid_cache(cache_path, audit_path, str(row["id"]))
        if value is None:
            missing.append(row)
        else:
            cached[str(row["id"])] = value

    model = tokenizer = None
    action_mean = action_std = None
    load_audit: Mapping[str, Any] = {}
    device = torch.device("cuda:0")
    if missing:
        if physical_device not in (0, 1, 2, 3):
            raise ProtocolError("generation physical device must be one of node43 GPUs 0-3")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        elif visible != str(physical_device):
            raise ProtocolError(
                f"physical GPU mismatch: expected {physical_device}, CUDA_VISIBLE_DEVICES={visible!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("WorldArena val5 native generation requires CUDA")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        torch.cuda.set_device(0)
        checkpoint = _project_path(str(config["checkpoint"]))
        model, resolved_cfg, load_audit = checkpoint_loader(
            checkpoint,
            str(config.get("checkpoint_kind", "worldarena_bimanual_adapt")),
            device,
        )
        model.eval()
        stats_path = Path(str(resolved_cfg["data"]["action_stats"]))
        if not stats_path.is_file():
            raise ProtocolError(f"checkpoint action stats are missing: {stats_path}")
        with np.load(stats_path) as stats:
            action_mean = np.asarray(stats["mean"], dtype=np.float32)
            action_std = np.asarray(stats["std"], dtype=np.float32)
        from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer

        tokenizer = OnlineObservationTokenizer(
            T=int(config.get("native_t", 16)),
            token_grid=8,
            task_cache_dir=_project_path(
                str(config.get("task_cache_dir", output_root / "task_cache"))
            ),
            device="cuda:0",
            qwen_device="cuda:0",
            allow_zero_task_fallback=False,
        )

    records: list[dict[str, Any]] = []
    for row in panel:
        record_id = str(row["id"])
        name = val_output_name(row)
        cache_path, audit_path = _cache_paths(output_root, row)
        cache = cached.get(record_id)
        if cache is None:
            if model is None or tokenizer is None or action_mean is None or action_std is None:
                raise ProtocolError("model resources were not initialized for a missing cache")
            initial_rgb, left, right, future_frames = _load_inference_inputs(row)
            task_text = _load_instruction(row)
            tokenized = tokenizer.tokenize([initial_rgb], task_text)
            predicted, rollout_audit = _model_episode(
                model=model,
                task_embedding=tokenized.task_emb.to(device=device, dtype=torch.float32),
                initial_state=tokenized.context_tokens.to(device=device, dtype=torch.float32),
                initial_context_rgb=tokenized.context_rgb.to(device=device, dtype=torch.float32),
                left_physical=left,
                right_physical=right,
                action_mean=action_mean,
                action_std=action_std,
                device=device,
            )
            native = _native64(predicted)
            if len(native) != future_frames:
                raise ProtocolError(
                    f"native prediction length mismatch for {record_id}: {len(native)} != {future_frames}"
                )
            failures = [
                key
                for key, value in rollout_audit.get("gates", {}).items()
                if key != "chunk_boundary" and value is not True
            ]
            if failures:
                raise ProtocolError(f"native rollout gates failed for {record_id}: {failures}")
            _atomic_npz(
                cache_path,
                initial_rgb=initial_rgb,
                native_rgb64=native.astype(np.float16),
            )
            _atomic_json(
                audit_path,
                {
                    "schema": "wm3d_v7_worldarena_context_pyramid_native_cache_v1",
                    "id": record_id,
                    "task": row["task"],
                    "episode": int(row["episode"]),
                    "native_resolution": [64, 64],
                    "frames": int(len(native)),
                    "no_future_ground_truth": True,
                    "inference_inputs": [
                        "initial_rgb",
                        "instruction",
                        "physical_actions",
                    ],
                    "load": dict(load_audit),
                    "rollout": rollout_audit,
                },
            )
        else:
            initial_rgb, native = cache

        written = render_native_cache(
            initial_rgb,
            native,
            output_root,
            name,
            fps=int(config.get("fps", 10)),
        )
        records.append(
            {
                "id": record_id,
                "name": name,
                "episode": int(row["episode"]),
                "cache": str(cache_path.resolve()),
                "rendered": {key: str(value.resolve()) for key, value in written.items()},
            }
        )
        print(
            json.dumps(
                {"status": "episode_complete", "id": record_id, "variants": len(written)}
            ),
            flush=True,
        )

    summary = {
        "schema": "wm3d_v7_worldarena_context_pyramid_val5_generation_v1",
        "coverage": len(records),
        "variants": ["baseline", *(variant_name(config) for config in locked_grid())],
        "records": records,
        "future_gt_used_for_inference": False,
    }
    _atomic_json(output_root / "generation_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=int, choices=(0, 1, 2, 3), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ProtocolError("diagnostic config must be a mapping")
    summary = run_generation(config, physical_device=args.device)
    print(
        json.dumps(
            {
                "status": "complete",
                "coverage": summary["coverage"],
                "variants": len(summary["variants"]),
            }
        )
    )


if __name__ == "__main__":
    main()
