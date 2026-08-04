#!/usr/bin/env python3
"""Build the frozen 2048-D task bank used by WM3D.

Formal preprocessing uses the pinned T5 encoder in the sealed asset bundle.
The deterministic hash backend is restricted to the public two-GPU smoke test.
Neither encoder is part of the online WM3D training graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

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
SMOKE_MODEL = "wm3d/smoke-hash-2048"
SMOKE_REVISION = "52e6cc877548ebd0de720a7fe86177f8a5593a673f40162aa9006a3877fa97c1"
SMOKE_CONFIRMATION = "EXECUTE_V7_PUBLIC_SMOKE_HASH_TASK_BANK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("flan-t5", "smoke-hash"), default="flan-t5"
    )
    parser.add_argument("--model", default="google/flan-t5-xl")
    parser.add_argument("--revision")
    parser.add_argument("--confirmation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
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
        raise ValueError("episode plan has no tasks")
    return sorted(
        tasks, key=lambda text: (hashlib.sha256(text.encode()).hexdigest(), text)
    )


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
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def deterministic_embedding(text: str) -> torch.Tensor:
    """Return the platform-independent vector used only by public smoke."""

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


def _flan_t5_embeddings(
    args: argparse.Namespace,
    tasks: list[str],
    asset_root: Path,
    task_asset: dict[str, Any],
) -> tuple[torch.Tensor, str, str, str, dict[str, Any]]:
    from transformers import AutoTokenizer, T5EncoderModel

    if not args.revision:
        raise ValueError("flan-t5 backend requires --revision")
    if task_asset["repo_id"] != args.model or task_asset["revision"] != args.revision:
        raise ValueError("task model/revision differs from the asset receipt")
    task_snapshot = (asset_root / str(task_asset["path"])).resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(task_snapshot), local_files_only=True, trust_remote_code=False
    )
    model = T5EncoderModel.from_pretrained(
        str(task_snapshot),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.requires_grad_(False).eval()
    if int(model.config.d_model) != 2048:
        raise ValueError(
            f"task encoder hidden size is {model.config.d_model}, not 2048"
        )

    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            batch = tasks[start : start + args.batch_size]
            tokens: dict[str, Any] = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(args.device) for name, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state.float()
            mask = tokens["attention_mask"].to(hidden.dtype)[..., None]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.layer_norm(pooled, (pooled.shape[-1],))
            chunks.append(pooled.to(torch.bfloat16).cpu())
    return (
        torch.cat(chunks, dim=0).contiguous(),
        str(args.model),
        str(args.revision),
        "masked_mean_then_layer_norm",
        {},
    )


def _smoke_embeddings(
    args: argparse.Namespace,
    tasks: list[str],
    task_asset: dict[str, Any],
) -> tuple[torch.Tensor, str, str, str, dict[str, Any]]:
    if args.confirmation != SMOKE_CONFIRMATION:
        raise ValueError("smoke hash backend requires the exact confirmation string")
    if (
        task_asset.get("repo_id") != SMOKE_MODEL
        or task_asset.get("revision") != SMOKE_REVISION
    ):
        raise ValueError("smoke task asset identity mismatch")
    return (
        torch.stack([deterministic_embedding(text) for text in tasks]),
        SMOKE_MODEL,
        SMOKE_REVISION,
        "deterministic_sha256_stream_layer_norm_smoke_only",
        {"smoke_only": True},
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch-size and max-length must be positive")
    plan = resolve_regular_file(args.episode_plan.parent, args.episode_plan.name)
    tasks = _read_tasks(plan)
    asset_report = verify_asset_bundle(args.asset_root, deep=False)
    asset_root = args.asset_root.resolve(strict=True)
    asset_receipt = asset_report["receipt"]
    task_asset = asset_receipt["assets"]["task_model"]

    if args.backend == "smoke-hash":
        embeddings, model_name, revision, pooling, extra = _smoke_embeddings(
            args, tasks, task_asset
        )
    else:
        embeddings, model_name, revision, pooling, extra = _flan_t5_embeddings(
            args, tasks, asset_root, task_asset
        )
    if embeddings.shape != (len(tasks), 2048) or not bool(
        torch.isfinite(embeddings).all()
    ):
        raise RuntimeError("task embedding bank shape/finiteness check failed")

    output = resolve_real_directory(args.output_root, "dataset output root")
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    bank_path = control / "task_embeddings.safetensors"
    index_path = control / "task_index.json"
    asset_path = control / "encoder_asset_receipt.json"
    atomic_write_json(asset_path, asset_receipt, exclusive=True)
    _atomic_safetensors(bank_path, {"embeddings": embeddings.contiguous()})
    value = {
        "schema": SCHEMA,
        "model": model_name,
        "revision": revision,
        "pooling": pooling,
        "dimension": 2048,
        **extra,
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
                **extra,
                "tasks": len(tasks),
                "bank_sha256": value["embeddings_sha256"],
                "index_sha256": sha256_file(index_path),
                "encoder_asset_receipt_sha256": canonical_sha256(asset_receipt),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
