#!/usr/bin/env python3
"""Fail-closed node/GPU/storage guard for the public WM3D smoke."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-ip", default="172.27.0.5")
    parser.add_argument("--minimum-free-bytes", type=int, default=50_000_000_000)
    return parser.parse_args()


def _command(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _rows(command: tuple[str, ...]) -> list[list[str]]:
    text = _command(*command)
    if not text:
        return []
    return [[part.strip() for part in line.split(",")] for line in text.splitlines()]


def main() -> None:
    args = parse_args()
    devices = tuple(int(item) for item in args.devices.split(","))
    if devices != (0, 1):
        raise ValueError("smoke devices must be exactly 0,1")
    addresses = set(_command("hostname", "-I").split())
    if args.expected_ip and args.expected_ip not in addresses:
        raise RuntimeError(
            f"host addresses {sorted(addresses)} do not include {args.expected_ip}"
        )
    root = args.work_root
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"work root is not a real directory: {root}")
    root = root.resolve(strict=True)
    statvfs = os.statvfs(root)
    free = int(statvfs.f_bavail) * int(statvfs.f_frsize)
    if free < args.minimum_free_bytes:
        raise RuntimeError(f"work root free bytes {free} below {args.minimum_free_bytes}")

    gpu_rows = _rows(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total",
            "--format=csv,noheader,nounits",
        )
    )
    gpu_by_index = {int(row[0]): row for row in gpu_rows}
    if any(index not in gpu_by_index for index in devices):
        raise RuntimeError(f"selected GPU is missing: {devices}")
    uuid_to_index = {row[1]: int(row[0]) for row in gpu_rows}
    selected = []
    for index in devices:
        row = gpu_by_index[index]
        volatile = int(row[4])
        aggregate = int(row[5])
        if volatile != 0 or aggregate != 0:
            raise RuntimeError(
                f"GPU{index} uncorrected ECC is volatile={volatile}, aggregate={aggregate}"
            )
        selected.append(
            {
                "index": index,
                "uuid": row[1],
                "name": row[2],
                "memory_mib": int(row[3]),
                "uncorrected_ecc_volatile": volatile,
                "uncorrected_ecc_aggregate": aggregate,
            }
        )
    compute_rows = _rows(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
    )
    conflicts = [
        {
            "gpu": uuid_to_index.get(row[0]),
            "pid": int(row[1]),
            "process": row[2],
            "used_memory_mib": int(row[3]),
        }
        for row in compute_rows
        if uuid_to_index.get(row[0]) in devices
    ]
    if conflicts:
        raise RuntimeError(
            "selected smoke GPUs are not idle: " + json.dumps(conflicts, sort_keys=True)
        )
    print(
        json.dumps(
            {
                "pass": True,
                "expected_ip": args.expected_ip,
                "devices": list(devices),
                "gpus": selected,
                "work_root": str(root),
                "work_root_free_bytes": free,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
