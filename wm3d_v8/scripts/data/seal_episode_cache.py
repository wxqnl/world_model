#!/usr/bin/env python3
"""Seal the expensive model-independent episode cache without rehashing payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import uuid

from wm3d_v3.data.cache_tasks import (
    CACHE_EPISODE_SEAL_SCHEMA,
    CACHE_TASK_RECEIPT_SCHEMA,
    cache_task_from_mapping,
)
from wm3d_v3.data.cache_writer import CACHE_TASK_PAYLOAD_SCHEMA
from wm3d_v3.data.manifest_contract import (
    CACHE_EPISODE_INDEX_SCHEMA,
    canonical_sha256,
    iter_jsonl,
    sha256_file,
)


HEX64 = re.compile(r"[0-9a-f]{64}")


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
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--episode-index-fragment-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--output-seal", type=Path, required=True)
    args = parser.parse_args()
    tasks = [
        cache_task_from_mapping(dict(row)).as_dict()
        for _line, row in iter_jsonl(args.task_manifest)
    ]
    task_ids = [str(row["task_id"]) for row in tasks]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("task manifest is empty or has duplicate ids")
    rows: list[dict] = []
    receipt_digests: dict[str, str] = {}
    episode_ids: set[str] = set()
    cache_root: Path | None = None
    encoder_contracts: set[str] = set()
    task_encoder_contracts: set[str] = set()
    task_bank_indexes: set[str] = set()
    representation_contracts: set[str] = set()
    source_manifest_by_name: dict[str, str] = {}
    adapter_contract_by_name: dict[str, str] = {}
    for task, task_id in zip(tasks, task_ids, strict=True):
        encoder_contracts.add(str(task["encoder_contract_sha256"]))
        task_encoder_contracts.add(str(task["task_encoder_contract_sha256"]))
        task_bank_indexes.add(str(task["task_bank_index_sha256"]))
        representation_contracts.add(str(task["representation_contract_sha256"]))
        for destination, field in (
            (source_manifest_by_name, "source_manifest_sha256"),
            (adapter_contract_by_name, "adapter_contract_sha256"),
        ):
            previous = destination.setdefault(str(task["source"]), str(task[field]))
            if previous != task[field]:
                raise RuntimeError(
                    f"source {task['source']!r} mixes {field} values"
                )
        receipt_path = args.receipt_root / f"{task_id}.json"
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise RuntimeError(f"missing task receipt {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != CACHE_TASK_RECEIPT_SCHEMA or receipt.get("task") != task:
            raise RuntimeError(f"task receipt identity/schema mismatch {receipt_path}")
        outputs = receipt.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise RuntimeError(f"task receipt has no outputs {receipt_path}")
        # Generation already hashed each payload once.  Seal checks identity,
        # regular-file status and exact recorded size; full SHA is verified
        # lazily by the first training worker opening each shard.
        for path_value, evidence in outputs.items():
            path = Path(path_value)
            if (
                path.is_symlink()
                or not path.is_file()
                or not isinstance(evidence, dict)
                or set(evidence) != {"sha256", "size_bytes"}
                or HEX64.fullmatch(str(evidence["sha256"])) is None
                or path.stat().st_size != int(evidence["size_bytes"])
            ):
                raise RuntimeError(f"task output evidence is invalid: {path}")
        receipt_digests[task_id] = sha256_file(receipt_path)
        fragment = args.episode_index_fragment_root / f"{task_id}.jsonl"
        fragment_rows = [dict(row) for _line, row in iter_jsonl(fragment)]
        if len(fragment_rows) != 1:
            raise RuntimeError(f"episode fragment must have one row: {fragment}")
        row = fragment_rows[0]
        if row.get("schema") != CACHE_EPISODE_INDEX_SCHEMA:
            raise RuntimeError(f"episode fragment schema mismatch: {fragment}")
        identity = str(row["episode_id"])
        if identity != str(task["episode_id"]):
            raise RuntimeError(
                f"episode fragment identity {identity!r} != task episode "
                f"{task['episode_id']!r}"
            )
        for name in ("source", "split", "embodiment"):
            if row.get(name) != task.get(name):
                raise RuntimeError(
                    f"episode fragment {name} differs from its task: {fragment}"
                )
        if int(row.get("frame_count", 0)) < 2:
            raise RuntimeError(f"episode fragment has invalid frame_count: {fragment}")
        output_by_path = {
            Path(path_value).resolve(strict=True): evidence
            for path_value, evidence in outputs.items()
        }
        expected_payload = {
            "features.safetensors": (
                str(row["feature_shard"]), str(row["feature_sha256"])
            ),
            "robot.safetensors": (
                str(row["robot_shard"]), str(row["robot_sha256"])
            ),
            "rgb.jpgpack": (str(row["rgb_pack"]), str(row["rgb_pack_sha256"])),
        }
        inferred_roots: set[Path] = set()
        for basename, (relative, digest) in expected_payload.items():
            matches = [path for path in output_by_path if path.name == basename]
            if len(matches) != 1:
                raise RuntimeError(
                    f"task receipt must contain exactly one {basename}: {receipt_path}"
                )
            path = matches[0]
            evidence = output_by_path[path]
            if str(evidence["sha256"]) != digest:
                raise RuntimeError(
                    f"episode index {basename} SHA differs from receipt: {fragment}"
                )
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"unsafe episode index path {relative!r}")
            root = path
            for _part in relative_path.parts:
                root = root.parent
            if (root / relative_path).resolve(strict=True) != path:
                raise RuntimeError(
                    f"episode index {basename} path differs from receipt: {fragment}"
                )
            inferred_roots.add(root)
        if len(inferred_roots) != 1:
            raise RuntimeError(f"episode payloads do not share one cache root: {fragment}")
        root = inferred_roots.pop()
        if cache_root is None:
            cache_root = root
        elif cache_root != root:
            raise RuntimeError("task receipts span multiple cache roots")
        manifest_matches = [
            path for path in output_by_path if path.name == "manifest.json"
        ]
        commit_matches = [
            path for path in output_by_path if path.name == "COMMITTED.json"
        ]
        fragment_path = fragment.resolve(strict=True)
        if len(manifest_matches) != 1 or len(commit_matches) != 1:
            raise RuntimeError(f"task receipt misses manifest/commit: {receipt_path}")
        payload_parent = next(
            path for path in output_by_path if path.name == "features.safetensors"
        ).parent
        if (
            manifest_matches[0].parent != payload_parent
            or commit_matches[0].parent != payload_parent
        ):
            raise RuntimeError(
                f"task manifest/commit do not share the payload directory: {receipt_path}"
            )
        if fragment_path not in output_by_path:
            raise RuntimeError(f"task receipt does not bind its index fragment: {fragment}")
        if output_by_path[fragment_path]["sha256"] != sha256_file(fragment_path):
            raise RuntimeError(f"episode fragment SHA differs from receipt: {fragment}")
        payload_manifest = json.loads(
            manifest_matches[0].read_text(encoding="utf-8")
        )
        commit = json.loads(commit_matches[0].read_text(encoding="utf-8"))
        source_evidence = payload_manifest.get("source_evidence")
        if (
            not isinstance(source_evidence, dict)
            or source_evidence.get("task_bank_index_sha256")
            != task["task_bank_index_sha256"]
        ):
            raise RuntimeError(
                f"payload task-bank evidence differs from its task: {receipt_path}"
            )
        payload_files = payload_manifest.get("files")
        expected_file_evidence = {
            basename: {
                "size": int(output_by_path[path]["size_bytes"]),
                "sha256": str(output_by_path[path]["sha256"]),
            }
            for basename, path in (
                ("features.safetensors", next(item for item in output_by_path if item.name == "features.safetensors")),
                ("robot.safetensors", next(item for item in output_by_path if item.name == "robot.safetensors")),
                ("rgb.jpgpack", next(item for item in output_by_path if item.name == "rgb.jpgpack")),
            )
        }
        if any(int(item["size"]) <= 0 for item in expected_file_evidence.values()):
            raise RuntimeError(f"payload manifest records an empty payload: {receipt_path}")
        if (
            payload_manifest.get("schema") != CACHE_TASK_PAYLOAD_SCHEMA
            or payload_manifest.get("task") != task
            or int(payload_manifest.get("frame_count", -1))
            != int(row["frame_count"])
            or payload_files != expected_file_evidence
            or commit.get("schema") != CACHE_TASK_PAYLOAD_SCHEMA
            or commit.get("task_id") != task_id
            or commit.get("manifest_sha256")
            != output_by_path[manifest_matches[0]]["sha256"]
            or sha256_file(manifest_matches[0])
            != output_by_path[manifest_matches[0]]["sha256"]
            or sha256_file(commit_matches[0])
            != output_by_path[commit_matches[0]]["sha256"]
            or commit.get("manifest_content_sha256")
            != canonical_sha256(payload_manifest)
        ):
            raise RuntimeError(f"payload manifest/commit binding failed: {receipt_path}")
        if identity in episode_ids:
            raise RuntimeError(f"duplicate episode identity {identity}")
        episode_ids.add(identity)
        rows.append(row)
    if (
        len(encoder_contracts) != 1
        or len(task_encoder_contracts) != 1
        or len(task_bank_indexes) != 1
        or len(representation_contracts) != 1
    ):
        raise RuntimeError(
            "one episode cache seal cannot mix vision encoder, task encoder, "
            "or representation contracts"
        )
    rows.sort(key=lambda row: (str(row["source"]), str(row["episode_id"])))
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    ).encode()
    _publish(args.output_index.absolute(), payload)
    seal = {
        "schema": CACHE_EPISODE_SEAL_SCHEMA,
        "task_manifest_path": str(args.task_manifest.resolve(strict=True)),
        "task_manifest_sha256": sha256_file(args.task_manifest.resolve(strict=True)),
        "task_count": len(tasks),
        "episode_count": len(rows),
        "cache_root": str(cache_root),
        "source_manifest_sha256_by_name": dict(sorted(source_manifest_by_name.items())),
        "adapter_contract_sha256_by_name": dict(sorted(adapter_contract_by_name.items())),
        "encoder_contract_sha256": next(iter(encoder_contracts)),
        "task_encoder_contract_sha256": next(iter(task_encoder_contracts)),
        "task_bank_index_sha256": next(iter(task_bank_indexes)),
        "representation_contract_sha256": next(iter(representation_contracts)),
        "receipt_sha256_by_task": dict(sorted(receipt_digests.items())),
        "episode_index_path": str(args.output_index.absolute()),
        "episode_index_sha256": sha256_file(args.output_index.absolute()),
        "payload_verification": "generation_sha256_plus_seal_size_plus_lazy_open_sha256",
    }
    _publish(
        args.output_seal.absolute(),
        (json.dumps(seal, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps(seal, sort_keys=True))


if __name__ == "__main__":
    main()
