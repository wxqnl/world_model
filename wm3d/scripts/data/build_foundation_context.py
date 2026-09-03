"""Build an immutable Wan text-context store for the foundation trainer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from safetensors.torch import save_file
import torch

from wm3d.data.foundation_input import FOUNDATION_CONTEXT_STORE_SCHEMA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--wan-model-root", type=Path, required=True)
    parser.add_argument("--upstream-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    return parser.parse_args()


def _tasks(metadata_root: Path) -> list[str]:
    root = metadata_root.expanduser().resolve(strict=True)
    manifest_paths = sorted((root / "tasks").glob("*/*/manifest.json"))
    if not manifest_paths:
        raise RuntimeError("metadata root contains no task manifests")
    tasks: set[str] = set()
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = str(payload.get("task", {}).get("task_text", "")).strip()
        if not text:
            raise RuntimeError(f"task manifest has no task text: {path}")
        tasks.add(text)
    return sorted(tasks)


def _load_text_stack(
    source_root: Path, model_root: Path, *, device: str, context_length: int
) -> tuple[Any, Any]:
    source = source_root.expanduser().resolve(strict=True)
    source_text = str(source / "src")
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    # Prepare the narrow upstream namespace before importing Wan helpers;
    # WSA's top-level policies package eagerly imports unrelated policies.
    from wm3d.models.foundation_world_model import _load_upstream_symbols

    _load_upstream_symbols(source)
    from lerobot.policies.WSA_Large.core.models.wan22.helpers.loader import (
        _load_registered_model,
    )
    from lerobot.policies.WSA_Large.core.models.wan22.wan_video_text_encoder import (
        HuggingfaceTokenizer,
    )

    root = model_root.expanduser().resolve(strict=True)
    text_encoder = _load_registered_model(
        str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        "wan_video_text_encoder",
        torch_dtype=torch.bfloat16,
        device=device,
    )
    text_encoder.eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(
        name=str(root / "google" / "umt5-xxl"),
        seq_len=context_length,
        clean="whitespace",
        local_files_only=True,
    )
    return text_encoder, tokenizer


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.context_length <= 0:
        raise ValueError("batch-size and context-length must be positive")
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"foundation context output already exists: {output}")
    tasks = _tasks(args.metadata_root)
    encoder, tokenizer = _load_text_stack(
        args.upstream_source_root,
        args.wan_model_root,
        device=args.device,
        context_length=args.context_length,
    )
    contexts: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tasks), args.batch_size):
            selected = tasks[start : start + args.batch_size]
            ids, mask = tokenizer(
                selected, return_mask=True, add_special_tokens=True
            )
            mask = mask.to(args.device, non_blocking=True).bool()
            hidden = encoder(
                ids.to(args.device, non_blocking=True), mask
            ).to(torch.bfloat16)
            hidden = hidden.masked_fill(~mask[..., None], 0)
            if hidden.shape[:2] != mask.shape or hidden.shape[-1] != 4096:
                raise RuntimeError("Wan text encoder output contract changed")
            if not bool(torch.isfinite(hidden).all()):
                raise FloatingPointError("Wan context contains non-finite values")
            contexts.append(hidden.cpu())
            masks.append(mask.cpu())
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.mkdir(mode=0o750)
    save_file(
        {
            "context": torch.cat(contexts).contiguous(),
            "mask": torch.cat(masks).contiguous(),
        },
        str(temporary / "context.safetensors"),
    )
    (temporary / "index.json").write_text(
        json.dumps(
            {"schema": FOUNDATION_CONTEXT_STORE_SCHEMA, "tasks": tasks},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.rename(temporary, output)
    print(json.dumps({"output": str(output), "tasks": len(tasks)}, sort_keys=True))


if __name__ == "__main__":
    main()
