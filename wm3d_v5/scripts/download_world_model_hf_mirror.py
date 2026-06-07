#!/usr/bin/env python3
"""Download public world-model pretraining datasets from Hugging Face mirror.

This script intentionally downloads raw snapshots only. It does not extract or
convert datasets, because expansion can multiply disk usage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_OUT = Path("/data/Minko/datasets/hf_world_model_pretrain")
DEFAULT_STATE = Path("/data/Minko/logs/download_world_model_hf_mirror_v1_state.jsonl")


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    priority: str
    note: str
    approx_gb: float


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        "lerobot/droid_1.0.1",
        "P0",
        "Real-world robot manipulation, DROID converted to LeRobot format.",
        384.0,
    ),
    DatasetSpec(
        "HuggingFaceVLA/community_dataset_v3",
        "P0",
        "Large cross-embodiment LeRobot community mixture.",
        351.0,
    ),
    DatasetSpec(
        "tonyfanggalaxies/RH20T",
        "P0",
        "RH20T HF mirror: contact-rich real robot tarballs. Keep compressed first.",
        310.0,
    ),
    DatasetSpec(
        "lerobot/aloha_mobile_chair",
        "P1",
        "Real Mobile ALOHA dual-arm manipulation with high and wrist cameras.",
        2.1,
    ),
    DatasetSpec(
        "lerobot/aloha_mobile_cabinet",
        "P1",
        "Real Mobile ALOHA dual-arm manipulation with high and wrist cameras.",
        1.7,
    ),
    DatasetSpec(
        "lerobot/aloha_mobile_shrimp",
        "P1",
        "Real Mobile ALOHA dual-arm manipulation with high and wrist cameras.",
        1.3,
    ),
    DatasetSpec(
        "lerobot/aloha_mobile_wipe_wine",
        "P1",
        "Real Mobile ALOHA dual-arm manipulation with high and wrist cameras.",
        1.3,
    ),
    DatasetSpec(
        "lerobot/aloha_mobile_wash_pan",
        "P1",
        "Real Mobile ALOHA dual-arm manipulation with high and wrist cameras.",
        1.2,
    ),
    DatasetSpec(
        "lerobot/aloha_mobile_elevator",
        "P1",
        "Real Mobile ALOHA dual-arm manipulation with high and wrist cameras.",
        0.6,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--repo", action="append", default=[], help="Download only these repo IDs.")
    parser.add_argument("--skip-repo", action="append", default=[], help="Skip these repo IDs.")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--reserve-gb", type=float, default=500.0)
    parser.add_argument("--repo-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def log_state(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def completed_marker(local_dir: Path) -> Path:
    return local_dir / ".wm3d_download_complete"


def main() -> int:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HOME", "/data/Minko/.cache/huggingface")
    os.environ.setdefault("HF_HUB_CACHE", "/data/Minko/.cache/huggingface/hub")

    selected = DATASETS
    if args.repo:
        wanted = set(args.repo)
        selected = [d for d in selected if d.repo_id in wanted]
        unknown = wanted - {d.repo_id for d in DATASETS}
        if unknown:
            print(f"Unknown repos requested: {sorted(unknown)}", file=sys.stderr)
            return 2
    if args.skip_repo:
        skipped = set(args.skip_repo)
        selected = [d for d in selected if d.repo_id not in skipped]

    args.out.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)

    print(f"endpoint={args.endpoint}")
    print(f"out={args.out}")
    print(f"state={args.state}")
    print(f"repos={len(selected)}")
    log_state(args.state, {"event": "start", "out": str(args.out), "endpoint": args.endpoint})

    for spec in selected:
        local_dir = args.out / safe_name(spec.repo_id)
        marker = completed_marker(local_dir)
        if marker.exists():
            print(f"[skip-complete] {spec.repo_id} -> {local_dir}", flush=True)
            log_state(args.state, {"event": "skip_complete", "repo_id": spec.repo_id})
            continue

        available = free_gb(args.out)
        if available < args.reserve_gb + min(spec.approx_gb, 100.0):
            msg = (
                f"free space too low before {spec.repo_id}: "
                f"free={available:.1f}GB reserve={args.reserve_gb:.1f}GB"
            )
            print(f"[stop] {msg}", flush=True)
            log_state(args.state, {"event": "stop_low_disk", "repo_id": spec.repo_id, "free_gb": available})
            return 3

        print(
            f"[start] {spec.priority} {spec.repo_id} approx={spec.approx_gb:.1f}GB "
            f"free={available:.1f}GB note={spec.note}",
            flush=True,
        )
        log_state(
            args.state,
            {
                "event": "repo_start",
                "repo_id": spec.repo_id,
                "priority": spec.priority,
                "approx_gb": spec.approx_gb,
                "free_gb": available,
            },
        )

        if args.dry_run:
            info = api.repo_info(repo_id=spec.repo_id, repo_type="dataset", files_metadata=True)
            total = sum((getattr(s, "size", None) or 0) for s in info.siblings) / 1024**3
            print(f"[dry-run] {spec.repo_id}: files={len(info.siblings)} size={total:.2f}GB")
            continue

        path = None
        for attempt in range(1, args.repo_retries + 2):
            try:
                path = snapshot_download(
                    repo_id=spec.repo_id,
                    repo_type="dataset",
                    endpoint=args.endpoint,
                    local_dir=local_dir,
                    max_workers=args.max_workers,
                    etag_timeout=30,
                )
                break
            except Exception as exc:
                print(
                    f"[error] {spec.repo_id} attempt={attempt}/{args.repo_retries + 1}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                log_state(
                    args.state,
                    {
                        "event": "repo_error",
                        "repo_id": spec.repo_id,
                        "attempt": attempt,
                        "max_attempts": args.repo_retries + 1,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if attempt > args.repo_retries:
                    return 4
                sleep_s = args.retry_sleep * attempt
                print(f"[retry] {spec.repo_id} sleeping {sleep_s:.0f}s", flush=True)
                time.sleep(sleep_s)
        if path is None:
            return 4

        marker.write_text(
            json.dumps(
                {
                    "repo_id": spec.repo_id,
                    "path": str(path),
                    "endpoint": args.endpoint,
                    "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        used = shutil.disk_usage(args.out)
        print(f"[done] {spec.repo_id} -> {path}; free={used.free / 1024**3:.1f}GB", flush=True)
        log_state(
            args.state,
            {
                "event": "repo_done",
                "repo_id": spec.repo_id,
                "path": str(path),
                "free_gb": used.free / 1024**3,
            },
        )

    log_state(args.state, {"event": "all_done"})
    print("[all-done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
