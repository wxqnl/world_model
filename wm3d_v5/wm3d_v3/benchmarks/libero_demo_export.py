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
                    n = int(actions.shape[0])
                    if n < T + k:
                        continue
                    file_demos += 1
                    demos_total += 1
                    action_lengths.append(n)
                    for start in range(0, n - T - k + 1, stride):
                        target_start = start + T
                        action_chunk = actions[target_start: target_start + k].astype("float32").tolist()
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
                            "T": T,
                            "k": k,
                            "stride": stride,
                            "camera_key": camera_key,
                            "camera_available": camera_key in obs_keys,
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
    args = ap.parse_args()

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
