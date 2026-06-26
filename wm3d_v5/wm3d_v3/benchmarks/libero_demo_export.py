"""Export LIBERO expert demonstration windows into compact WM3D JSONL rows.

The exporter intentionally keeps image data in the source HDF5 files. Each JSONL
row points to one expert window and stores the 7D action chunk plus success
labels. A later training dataset can re-open the referenced HDF5 and tokenize
the RGB context on demand.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import h5py


_SCENE_PREFIX = re.compile(r"^[A-Z_]+_SCENE\d+_")


def _instruction_from_file(path: Path) -> str:
    name = path.stem
    if name.endswith("_demo"):
        name = name[:-5]
    name = _SCENE_PREFIX.sub("", name)
    return name.replace("_", " ")


def _task_name(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_demo") else name


def _iter_files(paths: list[Path], dataset_dir: Path | None) -> list[Path]:
    files: list[Path] = []
    if dataset_dir is not None:
        files.extend(sorted(dataset_dir.glob("*.hdf5")))
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.hdf5")))
        else:
            files.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _demo_ids(group: h5py.Group, max_demos: int) -> list[str]:
    demos = sorted(group.keys(), key=lambda name: int(name.split("_")[-1]) if "_" in name else 0)
    return demos[:max_demos] if max_demos > 0 else demos


def _is_noop(action: Any, prev_action: Any | None = None, threshold: float = 1e-4) -> bool:
    vals = list(action)
    if prev_action is None:
        return sum(float(x) * float(x) for x in vals[:-1]) ** 0.5 < threshold
    prev = list(prev_action)
    return sum(float(x) * float(x) for x in vals[:-1]) ** 0.5 < threshold and vals[-1] == prev[-1]


def _filtered_actions(actions: Any, *, drop_noops: bool) -> tuple[list[int], list[list[float]], int]:
    keep: list[int] = []
    out: list[list[float]] = []
    noops = 0
    for idx, action in enumerate(actions):
        prev = out[-1] if out else None
        if drop_noops and _is_noop(action, prev):
            noops += 1
            continue
        keep.append(idx)
        out.append([float(x) for x in action[:7]])
    return keep, out, noops


def export_windows(
    files: list[Path],
    *,
    out_jsonl: Path,
    summary_out: Path,
    T: int,
    k: int,
    stride: int,
    max_demos_per_file: int,
    camera_key: str,
    camera_keys: list[str] | None = None,
    drop_noops: bool = False,
    target_offset: int = 0,
    pad_episode_start: bool = False,
) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    demos_total = 0
    file_summaries: list[dict[str, Any]] = []
    action_lengths: list[int] = []
    with out_jsonl.open("w") as fh:
        for file_path in files:
            task = _task_name(file_path)
            instruction = _instruction_from_file(file_path)
            file_rows = 0
            file_demos = 0
            with h5py.File(file_path, "r") as h5:
                data = h5["data"]
                for demo_id in _demo_ids(data, max_demos_per_file):
                    demo = data[demo_id]
                    actions = demo["actions"]
                    obs = demo.get("obs")
                    obs_keys = sorted(obs.keys()) if obs is not None else []
                    row_camera_keys = camera_keys or [camera_key]
                    keep_indices, filtered_actions, noops = _filtered_actions(actions, drop_noops=drop_noops)
                    n = len(filtered_actions)
                    first_target = T + int(target_offset)
                    if first_target < 0:
                        raise ValueError(f"target_offset={target_offset} makes first target index negative for T={T}")
                    if n < first_target + k:
                        continue
                    file_demos += 1
                    demos_total += 1
                    action_lengths.append(n)
                    start_min = -first_target if pad_episode_start else 0
                    start_max = n - first_target - k
                    for start in range(start_min, start_max + 1, stride):
                        target_start = start + first_target
                        action_chunk = filtered_actions[target_start: target_start + k]
                        context_positions = range(start, start + T)
                        context_indices = [
                            keep_indices[min(max(pos, 0), n - 1)]
                            for pos in context_positions
                        ]
                        row = {
                            "row_type": "expert_window",
                            "source_format": "libero_hdf5",
                            "hdf5_path": str(file_path),
                            "task_name": task,
                            "instruction": instruction,
                            "demo_id": demo_id,
                            "episode_len": n,
                            "context_start": start,
                            "target_start": target_start,
                            "context_indices": context_indices,
                            "target_indices": keep_indices[target_start: target_start + k],
                            "T": T,
                            "k": k,
                            "stride": stride,
                            "target_offset": int(target_offset),
                            "pad_episode_start": bool(pad_episode_start),
                            "drop_noops": bool(drop_noops),
                            "num_noops": int(noops),
                            "camera_key": camera_key,
                            "camera_keys": row_camera_keys,
                            "camera_available": all(key in obs_keys for key in row_camera_keys),
                            "obs_keys": obs_keys,
                            "action_chunk": action_chunk,
                            "terminal_success_tgt": 1.0,
                            "plausibility_tgt": 1.0,
                            "benchmark_success": True,
                        }
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                        rows += 1
                        file_rows += 1
            file_summaries.append({
                "file": str(file_path),
                "task_name": task,
                "instruction": instruction,
                "demos": file_demos,
                "windows": file_rows,
            })
    summary = {
        "files": len(files),
        "demos": demos_total,
        "windows": rows,
        "T": T,
        "k": k,
        "stride": stride,
        "max_demos_per_file": max_demos_per_file,
        "camera_key": camera_key,
        "camera_keys": camera_keys or [camera_key],
        "drop_noops": bool(drop_noops),
        "target_offset": int(target_offset),
        "pad_episode_start": bool(pad_episode_start),
        "mean_episode_len": mean(action_lengths) if action_lengths else None,
        "positive_success_windows": rows,
        "training_signal": {
            "expert_action_chunks": rows > 0,
            "positive_success_supervision": rows > 0,
            "hdf5_frame_references": rows > 0,
            "notes": (
                "Rows reference source HDF5 frames instead of copying RGB. "
                "Use with LIBERO failure rollout JSONL to build mixed success/failure training."
            ),
        },
        "by_file": file_summaries,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=Path, default=None)
    ap.add_argument("--input", type=Path, nargs="*", default=[])
    ap.add_argument("--out_jsonl", type=Path, required=True)
    ap.add_argument("--summary_out", type=Path, required=True)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max_demos_per_file", type=int, default=0)
    ap.add_argument("--camera_key", default="agentview_rgb")
    ap.add_argument("--camera_keys", default=None, help="Comma-separated HDF5 obs camera keys to record per window.")
    ap.add_argument("--drop_noops", action="store_true")
    ap.add_argument("--target_offset", type=int, default=0, help="Offset from start+T for target action; -1 aligns first action to the last context frame.")
    ap.add_argument(
        "--pad_episode_start",
        action="store_true",
        help="Generate left-padded opening windows so target 0 sees a repeated first-frame context.",
    )
    args = ap.parse_args()
    camera_keys = [item.strip() for item in str(args.camera_keys).split(",") if item.strip()] if args.camera_keys else None

    files = _iter_files(args.input, args.dataset_dir)
    if not files:
        raise RuntimeError("no LIBERO hdf5 files found")
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing LIBERO hdf5 files: {missing}")
    summary = export_windows(
        files,
        out_jsonl=args.out_jsonl,
        summary_out=args.summary_out,
        T=args.T,
        k=args.k,
        stride=args.stride,
        max_demos_per_file=args.max_demos_per_file,
        camera_key=args.camera_key,
        camera_keys=camera_keys,
        drop_noops=bool(args.drop_noops),
        target_offset=int(args.target_offset),
        pad_episode_start=bool(args.pad_episode_start),
    )
    print(json.dumps({
        "out_jsonl": str(args.out_jsonl),
        "summary_out": str(args.summary_out),
        "files": summary["files"],
        "demos": summary["demos"],
        "windows": summary["windows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
