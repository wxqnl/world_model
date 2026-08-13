#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from wm3d.data.cache_tasks import plan_tasks
from wm3d.data.manifest_contract import (
    canonical_sha256,
    iter_jsonl,
    load_data_profile,
    sha256_file,
)
from wm3d.data.task_embedding_store import TaskEmbeddingStore


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise FileExistsError(f"non-identical task manifest exists: {path}")
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        action="append",
        default=[],
        help="可重复；省略时处理 data profile 中的全部 source manifest。",
    )
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--task-encoder-contract", type=Path, required=True)
    parser.add_argument("--task-bank-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
    source_by_path = {
        source.manifest_path.resolve(strict=True): source for source in profile.sources
    }
    requested = (
        [path.resolve(strict=True) for path in args.source_manifest]
        if args.source_manifest
        else list(source_by_path)
    )
    if len(requested) != len(set(requested)):
        raise RuntimeError("duplicate --source-manifest")
    encoder_sha = sha256_file(args.encoder_contract.resolve(strict=True))
    task_encoder_sha = sha256_file(args.task_encoder_contract.resolve(strict=True))
    task_bank_index_sha = sha256_file(args.task_bank_index.resolve(strict=True))
    representation_sha = canonical_sha256(profile.cache_representation)
    tasks = []
    for manifest_path in requested:
        source = source_by_path.get(manifest_path)
        if source is None:
            raise RuntimeError(
                f"source manifest is not bound by the data profile: {manifest_path}"
            )
        rows = [row for _line, row in iter_jsonl(manifest_path)]
        row_sources = {str(row.get("source", "")) for row in rows}
        if row_sources != {source.name}:
            raise RuntimeError(
                f"manifest rows do not belong exactly to source {source.name!r}"
            )
        tasks.extend(
            plan_tasks(
                rows,
                source_manifest_sha256=source.manifest_sha256,
                adapter_contract_sha256=source.adapter_contract_sha256,
                encoder_contract_sha256=encoder_sha,
                task_encoder_contract_sha256=task_encoder_sha,
                task_bank_index_sha256=task_bank_index_sha,
                representation_contract_sha256=representation_sha,
                canonical_view_slots=profile.cache_representation["view_slots"],
            )
        )
    tasks.sort(key=lambda item: item.task_id)
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("combined cache task plan is empty or duplicated")
    task_store = TaskEmbeddingStore(
        root=args.task_bank_index.resolve(strict=True).parent,
        index_sha256=task_bank_index_sha,
        expected_data_profile_sha256=profile.profile_sha256,
        expected_source_manifest_sha256_by_name={
            source.name: source.manifest_sha256 for source in profile.sources
        },
        expected_encoder_contract_sha256=task_encoder_sha,
    )
    expected_text_ids = {
        hashlib.sha256(task.task_text.encode("utf-8")).hexdigest() for task in tasks
    }
    if set(task_store.entries) != expected_text_ids:
        raise RuntimeError("task bank text set differs from the cache task plan")
    payload = "".join(
        json.dumps(task.as_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for task in tasks
    ).encode()
    _publish(args.output.absolute(), payload)
    print(json.dumps({"tasks": len(tasks), "output": str(args.output.absolute())}))


if __name__ == "__main__":
    main()
