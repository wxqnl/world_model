"""Build an immutable Wan text-context store for the foundation trainer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import timedelta
from pathlib import Path
import sys
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.distributed as dist

from wm3d.data.foundation_input import FOUNDATION_CONTEXT_STORE_SCHEMA_V2
from wm3d.data.foundation_prompt import RAW_TASK, PROMPT_FORMATS, format_wan_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata-root", type=Path)
    source.add_argument("--task-bank-index", type=Path,
                        help="Existing sealed task_bank/index.jsonl; no task text changes.")
    parser.add_argument("--wan-model-root", type=Path, required=True)
    parser.add_argument("--upstream-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default=RAW_TASK,
                        help="Explicit T5 input semantics; legacy callers remain raw-task.")
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


def _bank_tasks(index_path: Path) -> list[str]:
    """Reuse the sealed text inventory instead of scanning every episode."""
    tasks: set[str] = set()
    with index_path.expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = json.loads(line)
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"task-bank row {line_number} has no task text")
            text = text.strip()
            tasks.add(text)
    if not tasks:
        raise RuntimeError("task bank contains no tasks")
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
    tasks = (_bank_tasks(args.task_bank_index) if args.task_bank_index is not None
             else _tasks(args.metadata_root))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group("gloo", timeout=timedelta(hours=2))
        args.device = "cuda:" + os.environ["LOCAL_RANK"]
        torch.cuda.set_device(args.device)
    start_row = len(tasks) * rank // world_size
    stop_row = len(tasks) * (rank + 1) // world_size
    selected_tasks = tasks[start_row:stop_row]
    if not selected_tasks:
        raise RuntimeError("more workers than distinct tasks")
    shared_temporary = [str(output.with_name(f".{output.name}.tmp.{os.getpid()}")) if rank == 0 else None]
    if world_size > 1:
        dist.broadcast_object_list(shared_temporary, src=0)
    temporary = Path(shared_temporary[0])
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(mode=0o750)
        (temporary / "parts").mkdir()
    if world_size > 1:
        dist.barrier()
    encoder, tokenizer = _load_text_stack(
        args.upstream_source_root,
        args.wan_model_root,
        device=args.device,
        context_length=args.context_length,
    )
    contexts: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(selected_tasks), args.batch_size):
            selected = selected_tasks[start : start + args.batch_size]
            ids, mask = tokenizer(
                [format_wan_prompt(task, args.prompt_format) for task in selected],
                return_mask=True, add_special_tokens=True
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
            if start == 0 or (start // args.batch_size + 1) % 100 == 0:
                print(json.dumps({"rank": rank, "encoded": start + len(selected),
                                  "assigned": len(selected_tasks)}), flush=True)
    save_file(
        {"context": torch.cat(contexts).contiguous(),
         "mask": torch.cat(masks).contiguous()},
        str(temporary / "parts" / f"rank{rank}.safetensors"),
    )
    del contexts, masks, encoder
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        parts = [load_file(str(temporary / "parts" / f"rank{r}.safetensors"))
                 for r in range(world_size)]
        save_file(
            {key: torch.cat([part[key] for part in parts]).contiguous()
             for key in ("context", "mask")},
            str(temporary / "context.safetensors"),
        )
        (temporary / "index.json").write_text(
            json.dumps({"schema": FOUNDATION_CONTEXT_STORE_SCHEMA_V2,
                        "prompt_format": args.prompt_format, "tasks": tasks},
                       ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        # Release mmap-backed rank tensors before unlinking their NFS files.
        del parts
        shutil.rmtree(temporary / "parts")
        os.rename(temporary, output)
        print(json.dumps({"output": str(output), "tasks": len(tasks),
                          "workers": world_size}, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
