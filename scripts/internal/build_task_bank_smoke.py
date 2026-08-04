#!/usr/bin/env python3
"""Build a deterministic 2048-D task bank for the public smoke only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from safetensors.torch import save_file
import torch

from wm3d.data.assets import verify_asset_bundle
from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
)


SCHEMA = "wm3d_v7_task_bank_v1"
MODEL = "wm3d/smoke-hash-2048"
REVISION = "52e6cc877548ebd0de720a7fe86177f8a5593a673f40162aa9006a3877fa97c1"
CONFIRMATION = "EXECUTE_V7_PUBLIC_SMOKE_HASH_TASK_BANK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--confirmation", required=True)
    return parser.parse_args()


def _read_tasks(path: Path) -> list[str]:
    tasks: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            text = str(json.loads(line).get("task_text", "")).strip()
            if not text:
                raise ValueError(f"{path}:{line_number} has no task_text")
            tasks.add(text)
    if not tasks:
        raise ValueError("episode plan has no task text")
    return sorted(tasks, key=lambda text: (hashlib.sha256(text.encode()).hexdigest(), text))


def deterministic_embedding(text: str) -> torch.Tensor:
    """Return one platform-independent normalized hash vector."""

    payload = bytearray()
    counter = 0
    while len(payload) < 2048:
        payload.extend(
            hashlib.sha256(
                b"wm3d_v7_smoke_hash_task_embedding_v1\x1f"
                + text.encode("utf-8")
                + counter.to_bytes(4, "big")
            ).digest()
        )
        counter += 1
    value = torch.tensor(list(payload[:2048]), dtype=torch.float32)
    value = (value - value.mean()) / value.std(unbiased=False).clamp_min(1.0e-6)
    return value.to(torch.bfloat16)


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    save_file(tensors, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.confirmation != CONFIRMATION:
        raise ValueError("smoke hash task bank requires the exact confirmation string")
    plan = resolve_regular_file(args.episode_plan.parent, args.episode_plan.name)
    tasks = _read_tasks(plan)
    asset_report = verify_asset_bundle(args.asset_root, deep=False)
    asset_receipt = asset_report["receipt"]
    task_asset = asset_receipt["assets"]["task_model"]
    if task_asset.get("repo_id") != MODEL or task_asset.get("revision") != REVISION:
        raise ValueError("smoke task asset identity mismatch")

    output = resolve_real_directory(args.output_root, "dataset output root")
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    bank_path = control / "task_embeddings.safetensors"
    index_path = control / "task_index.json"
    asset_path = control / "encoder_asset_receipt.json"
    embeddings = torch.stack([deterministic_embedding(text) for text in tasks])
    if embeddings.shape != (len(tasks), 2048) or not bool(torch.isfinite(embeddings).all()):
        raise RuntimeError("smoke task embedding shape/finiteness check failed")
    atomic_write_json(asset_path, asset_receipt, exclusive=True)
    _atomic_safetensors(bank_path, {"embeddings": embeddings.contiguous()})
    value = {
        "schema": SCHEMA,
        "model": MODEL,
        "revision": REVISION,
        "pooling": "deterministic_sha256_stream_layer_norm_smoke_only",
        "dimension": 2048,
        "smoke_only": True,
        "tasks": [
            {
                "task_id": index,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
            for index, text in enumerate(tasks)
        ],
        "episode_plan_sha256": sha256_file(plan),
        "encoder_asset_receipt_sha256": canonical_sha256(asset_receipt),
        "embeddings_sha256": sha256_file(bank_path),
    }
    atomic_write_json(index_path, value, exclusive=True)
    print(
        json.dumps(
            {
                "pass": True,
                "smoke_only": True,
                "tasks": len(tasks),
                "bank_sha256": value["embeddings_sha256"],
                "index_sha256": sha256_file(index_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
