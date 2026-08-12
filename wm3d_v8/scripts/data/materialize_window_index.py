#!/usr/bin/env python3
"""Build a small model-specific window index over the shared episode cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping
import uuid

import yaml

from wm3d_v3.data.cache_tasks import (
    CACHE_EPISODE_SEAL_SCHEMA,
    CACHE_WINDOW_SEAL_SCHEMA,
    cache_task_from_mapping,
)
from wm3d_v3.data.manifest_contract import (
    canonical_sha256,
    iter_jsonl,
    load_cache_episode_index,
    load_data_profile,
    sha256_file,
)
from wm3d_v3.data.window_index import plan_window_index
from wm3d_v3.models.model_factory import validate_model_data_compatibility


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
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--episode-seal", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-seal", type=Path, required=True)
    args = parser.parse_args()
    episode_index = args.episode_index.resolve(strict=True)
    episode_seal = args.episode_seal.resolve(strict=True)
    seal = json.loads(episode_seal.read_text(encoding="utf-8"))
    if (
        seal.get("schema") != CACHE_EPISODE_SEAL_SCHEMA
        or Path(str(seal.get("episode_index_path"))).resolve(strict=True) != episode_index
        or seal.get("episode_index_sha256") != sha256_file(episode_index)
    ):
        raise RuntimeError("episode seal does not bind the requested episode index")
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
    model = yaml.safe_load(args.model_profile.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise RuntimeError("model profile must be a mapping")
    validate_model_data_compatibility(model, profile)
    model_sha = canonical_sha256(model)
    task_manifest = Path(str(seal.get("task_manifest_path", ""))).resolve(strict=True)
    if seal.get("task_manifest_sha256") != sha256_file(task_manifest):
        raise RuntimeError("episode seal/task manifest SHA mismatch")
    tasks = [
        cache_task_from_mapping(dict(row)).as_dict()
        for _line, row in iter_jsonl(task_manifest)
    ]
    if len(tasks) != int(seal.get("task_count", -1)):
        raise RuntimeError("episode seal/task manifest count mismatch")
    source_by_name = {source.name: source for source in profile.sources}
    representation_sha = canonical_sha256(profile.cache_representation)
    encoder_digests: set[str] = set()
    task_encoder_digests: set[str] = set()
    task_bank_digests: set[str] = set()
    observed_sources: set[str] = set()
    for task in tasks:
        source_name = str(task.get("source", ""))
        source = source_by_name.get(source_name)
        if source is None:
            raise RuntimeError(
                f"episode cache task belongs to source outside data profile: {source_name!r}"
            )
        if str(task.get("embodiment", "")) != source.embodiment:
            raise RuntimeError(f"task/source embodiment mismatch for {source_name!r}")
        if task.get("source_manifest_sha256") != source.manifest_sha256:
            raise RuntimeError(f"task/source manifest SHA mismatch for {source_name!r}")
        if task.get("adapter_contract_sha256") != source.adapter_contract_sha256:
            raise RuntimeError(f"task/adapter SHA mismatch for {source_name!r}")
        if task.get("representation_contract_sha256") != representation_sha:
            raise RuntimeError(f"task/representation SHA mismatch for {source_name!r}")
        encoder_digest = str(task.get("encoder_contract_sha256", ""))
        if len(encoder_digest) != 64:
            raise RuntimeError(f"task/encoder SHA is invalid for {source_name!r}")
        encoder_digests.add(encoder_digest)
        task_encoder_digest = str(task.get("task_encoder_contract_sha256", ""))
        if len(task_encoder_digest) != 64:
            raise RuntimeError(f"task/task-encoder SHA is invalid for {source_name!r}")
        task_encoder_digests.add(task_encoder_digest)
        task_bank_digest = str(task.get("task_bank_index_sha256", ""))
        if len(task_bank_digest) != 64:
            raise RuntimeError(f"task/task-bank SHA is invalid for {source_name!r}")
        task_bank_digests.add(task_bank_digest)
        observed_sources.add(source_name)
    if observed_sources != set(source_by_name):
        raise RuntimeError(
            "episode cache source set differs from data profile: "
            f"cache={sorted(observed_sources)} profile={sorted(source_by_name)}"
        )
    if len(encoder_digests) != 1:
        raise RuntimeError("one episode cache seal cannot mix encoder contracts")
    if len(task_encoder_digests) != 1:
        raise RuntimeError("one episode cache seal cannot mix task encoder contracts")
    if len(task_bank_digests) != 1:
        raise RuntimeError("one episode cache seal cannot mix task embedding banks")
    if seal.get("source_manifest_sha256_by_name") != {
        name: source.manifest_sha256 for name, source in source_by_name.items()
    }:
        raise RuntimeError("episode seal/source manifest digest map mismatch")
    if seal.get("adapter_contract_sha256_by_name") != {
        name: source.adapter_contract_sha256 for name, source in source_by_name.items()
    }:
        raise RuntimeError("episode seal/adapter digest map mismatch")
    if seal.get("encoder_contract_sha256") != next(iter(encoder_digests)):
        raise RuntimeError("episode seal/encoder digest mismatch")
    if seal.get("task_encoder_contract_sha256") != next(iter(task_encoder_digests)):
        raise RuntimeError("episode seal/task encoder digest mismatch")
    if seal.get("task_bank_index_sha256") != next(iter(task_bank_digests)):
        raise RuntimeError("episode seal/task bank digest mismatch")
    if seal.get("representation_contract_sha256") != representation_sha:
        raise RuntimeError("episode seal/representation digest mismatch")

    source_rows: dict[str, dict[str, Mapping[str, object]]] = {}
    for source_name, source in source_by_name.items():
        by_digest: dict[str, Mapping[str, object]] = {}
        for _line, row in iter_jsonl(source.manifest_path):
            digest = canonical_sha256(row)
            if digest in by_digest:
                raise RuntimeError(
                    f"source manifest {source_name!r} has duplicate record identity"
                )
            by_digest[digest] = row
        source_rows[source_name] = by_digest
    task_source_fields = (
        "source",
        "episode_id",
        "payload",
        "payload_sha256",
        "payload_row_start",
        "payload_row_stop",
        "assets",
        "views",
        "task_text",
        "embodiment",
        "split",
        "observation_samples",
        "observation_clock",
        "robot_groups",
    )
    observed_record_digests: dict[str, set[str]] = {
        name: set() for name in source_by_name
    }
    for task in tasks:
        source_name = str(task["source"])
        digest = str(task["source_record_sha256"])
        row = source_rows[source_name].get(digest)
        if row is None:
            raise RuntimeError(
                f"cache task is not a row of source manifest {source_name!r}"
            )
        if any(task[field] != row[field] for field in task_source_fields):
            raise RuntimeError(
                f"cache task fields differ from source manifest row {source_name!r}/{digest}"
            )
        if digest in observed_record_digests[source_name]:
            raise RuntimeError("two cache tasks bind the same source manifest row")
        observed_record_digests[source_name].add(digest)
    for source_name, rows_by_digest in source_rows.items():
        if observed_record_digests[source_name] != set(rows_by_digest):
            raise RuntimeError(
                f"cache task closure is incomplete for source {source_name!r}"
            )
    episodes = load_cache_episode_index(
        episode_index, expected_sha256=seal["episode_index_sha256"]
    )
    rows = plan_window_index(
        episodes=episodes,
        cache_root=args.cache_root.resolve(strict=True),
        model_profile=model,
        model_profile_sha256=model_sha,
        data_profile=profile,
    )
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    _publish(args.output_index.absolute(), payload)
    window_seal = {
        "schema": CACHE_WINDOW_SEAL_SCHEMA,
        "episode_seal_path": str(episode_seal),
        "episode_seal_sha256": sha256_file(episode_seal),
        "episode_index_sha256": seal["episode_index_sha256"],
        "data_profile_sha256": profile.profile_sha256,
        "model_profile_path": str(args.model_profile.resolve(strict=True)),
        "model_profile_sha256": model_sha,
        "encoder_contract_sha256": seal["encoder_contract_sha256"],
        "task_encoder_contract_sha256": seal["task_encoder_contract_sha256"],
        "task_bank_index_sha256": seal["task_bank_index_sha256"],
        "representation_contract_sha256": seal["representation_contract_sha256"],
        "window_count": len(rows),
        "window_index_path": str(args.output_index.absolute()),
        "window_index_sha256": sha256_file(args.output_index.absolute()),
    }
    _publish(
        args.output_seal.absolute(),
        (json.dumps(window_seal, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(window_seal, sort_keys=True))


if __name__ == "__main__":
    main()
