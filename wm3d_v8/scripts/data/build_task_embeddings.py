#!/usr/bin/env python3
"""Encode unique instruction text once, then reuse it across cache workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from safetensors.torch import save_file
import torch
import yaml

from wm3d_v3.data.manifest_contract import (
    iter_jsonl,
    load_data_profile,
    sha256_file,
)
from wm3d_v3.data.task_embedding_store import TASK_BANK_SCHEMA
from wm3d_v3.encoders.qwen_vl_encoder import QwenVLEmbed


def _text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical {path}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
    contract_path = args.encoder_contract.resolve(strict=True)
    contract_sha = sha256_file(contract_path)
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    required = {
        "schema", "model_name", "model_revision", "embedding_dim", "pooling",
        "image_conditioning", "dtype",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        raise RuntimeError("task encoder contract fields mismatch")
    if contract["schema"] != "wm3d_v8_task_encoder_v1" or int(contract["embedding_dim"]) != 2048:
        raise RuntimeError("unsupported task encoder contract")
    if contract["image_conditioning"] != "disabled_for_stage0_task_text":
        raise RuntimeError("Stage0 task bank must not consume episode images")
    unique: set[str] = set()
    source_manifest_sha256_by_name: dict[str, str] = {}
    for source in profile.sources:
        source_manifest_sha256_by_name[source.name] = source.manifest_sha256
        for _line, row in iter_jsonl(source.manifest_path):
            if str(row.get("source", "")) != source.name:
                raise RuntimeError("source manifest row belongs to another source")
            unique.add(str(row.get("task_text", "")))
    if any(not text.strip() for text in unique):
        raise RuntimeError("task text cannot be blank")
    encoder = QwenVLEmbed(
        model_name=str(contract["model_name"]),
        model_revision=str(contract["model_revision"]),
        device=args.device,
        dtype=torch.bfloat16,
    )
    root = args.output_root.absolute()
    entries: list[dict[str, object]] = []
    for text in sorted(unique):
        identity = _text_id(text)
        relative = f"embeddings/{identity[:2]}/{identity}.safetensors"
        destination = root / relative
        temporary = destination.with_name(
            f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        embedding = encoder.embed(text).to(torch.float32).contiguous()
        save_file({"embedding": embedding}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha256_file(temporary) != sha256_file(destination):
                raise FileExistsError(f"non-identical task embedding exists: {destination}")
        finally:
            temporary.unlink(missing_ok=True)
        entries.append(
            {
                "schema": TASK_BANK_SCHEMA,
                "text_id": identity,
                "text": text,
                "path": relative,
                "sha256": sha256_file(destination),
            }
        )
    index = root / "index.jsonl"
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in entries
    ).encode("utf-8")
    _publish(index, payload)
    receipt = {
        "schema": TASK_BANK_SCHEMA,
        "data_profile_sha256": profile.profile_sha256,
        "source_manifest_sha256_by_name": dict(
            sorted(source_manifest_sha256_by_name.items())
        ),
        "encoder_contract_sha256": contract_sha,
        "unique_text_count": len(entries),
        "index_path": str(index),
        "index_sha256": sha256_file(index),
    }
    _publish(
        root / "receipt.json",
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
