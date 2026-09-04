#!/usr/bin/env python3
"""Materialize the small metadata needed for raw-streaming WM3D training."""

from __future__ import annotations

import argparse
import filecmp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import groupby
import multiprocessing
import json
import os
from pathlib import Path
import shutil
import uuid

from safetensors.torch import save_file
import torch
import yaml

from scripts.data.run_cache_worker import _prepare_task, _strict_encoder
from wm3d.data.cache_tasks import cache_task_from_mapping
from wm3d.data.episode_io import VerifiedAssetStore
from wm3d.data.grouped_normalization import build_grouped_normalization_artifact
from wm3d.data.manifest_contract import (
    CACHE_EPISODE_INDEX_SCHEMA,
    canonical_sha256,
    iter_jsonl,
    load_cache_episode_index,
    load_data_profile,
    sha256_file,
)
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.data.streaming_raw import STREAMING_METADATA_SEAL_SCHEMA
from wm3d.data.task_embedding_store import TaskEmbeddingStore
from wm3d.data.window_index import iter_window_index
from wm3d.models.model_factory import validate_model_data_compatibility


class StreamingMetadataError(RuntimeError):
    pass


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        if path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _publish_jsonl(path: Path, rows) -> int:
    """Atomically publish the existing row encoding without a giant payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    count = 0
    try:
        with temporary.open("xb") as handle:
            for row in rows:
                handle.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
                count += 1
                if count % 100_000 == 0:
                    print(json.dumps({"event": "index_rows", "index": path.name, "rows": count}), flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not filecmp.cmp(temporary, path, shallow=False):
                raise FileExistsError(f"refusing to overwrite non-identical {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _write_episode(root: Path, prepared: object) -> dict[str, object]:
    task = prepared.task
    selected_rows = torch.tensor(
        prepared.source_evidence["selected_source_rows"], dtype=torch.int64
    )
    frame_times = prepared.robot_tensors["observation_times_s"].index_select(
        0, selected_rows
    ).to(torch.float64)
    final = root / "tasks" / task.task_id[:2] / task.task_id
    marker = b"WM3D streaming_raw: visual tensors are generated on demand.\n"
    if final.exists():
        if final.is_symlink() or not final.is_dir():
            raise StreamingMetadataError(f"invalid existing metadata episode: {final}")
    else:
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.parent / f".{task.task_id}.tmp.{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            save_file(
                {
                    "source_observation_row": selected_rows.contiguous(),
                    "frame_time_s": frame_times.contiguous(),
                },
                temporary / "boundaries.safetensors",
            )
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in prepared.robot_tensors.items()
                },
                temporary / "robot.safetensors",
            )
            (temporary / "visuals.streaming").write_bytes(marker)
            manifest = {
                "schema": STREAMING_METADATA_SEAL_SCHEMA,
                "task": task.as_dict(),
                "frame_count": int(selected_rows.numel()),
                "selected_source_rows": selected_rows.tolist(),
                "files": {
                    name: sha256_file(temporary / name)
                    for name in (
                        "boundaries.safetensors",
                        "robot.safetensors",
                        "visuals.streaming",
                    )
                },
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            for name in (
                "boundaries.safetensors",
                "robot.safetensors",
                "visuals.streaming",
                "manifest.json",
            ):
                _fsync(temporary / name)
            _fsync(temporary)
            os.rename(temporary, final)
            _fsync(final.parent)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    files = {
        name: sha256_file(final / name)
        for name in ("boundaries.safetensors", "robot.safetensors", "visuals.streaming")
    }
    relative = final.relative_to(root).as_posix()
    return {
        "schema": CACHE_EPISODE_INDEX_SCHEMA,
        "episode_id": task.episode_id,
        "source": task.source,
        "split": task.split,
        "embodiment": task.embodiment,
        "feature_shard": f"{relative}/boundaries.safetensors",
        "feature_sha256": files["boundaries.safetensors"],
        "robot_shard": f"{relative}/robot.safetensors",
        "robot_sha256": files["robot.safetensors"],
        "rgb_pack": f"{relative}/visuals.streaming",
        "rgb_pack_sha256": files["visuals.streaming"],
        "frame_count": int(selected_rows.numel()),
    }



def _completed_episode(root: Path, task: object) -> dict[str, object] | None:
    """Reuse an atomically published task, retaining the existing file contract."""
    final = root / "tasks" / task.task_id[:2] / task.task_id
    if not final.exists():
        return None
    if final.is_symlink() or not final.is_dir():
        raise StreamingMetadataError("existing metadata task is not a directory")
    manifest_path = final / "manifest.json"
    if manifest_path.is_symlink():
        raise StreamingMetadataError("existing metadata manifest is a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != STREAMING_METADATA_SEAL_SCHEMA
        or manifest.get("task") != task.as_dict()
        or int(manifest.get("frame_count", 0)) <= 0
        or len(manifest.get("selected_source_rows", [])) != manifest["frame_count"]
    ):
        raise StreamingMetadataError("existing metadata task contract differs")
    files = manifest["files"]
    for name in ("boundaries.safetensors", "robot.safetensors", "visuals.streaming"):
        path = final / name
        if path.is_symlink() or sha256_file(path) != files[name]:
            raise StreamingMetadataError("existing metadata file differs: " + name)
    relative = final.relative_to(root).as_posix()
    return {
        "schema": CACHE_EPISODE_INDEX_SCHEMA,
        "episode_id": task.episode_id, "source": task.source, "split": task.split,
        "embodiment": task.embodiment,
        "feature_shard": f"{relative}/boundaries.safetensors",
        "feature_sha256": files["boundaries.safetensors"],
        "robot_shard": f"{relative}/robot.safetensors",
        "robot_sha256": files["robot.safetensors"],
        "rgb_pack": f"{relative}/visuals.streaming",
        "rgb_pack_sha256": files["visuals.streaming"],
        "frame_count": int(manifest["frame_count"]),
    }


_METADATA_CONTEXT = None


def _metadata_worker_init(profile, adapters, store_config, root, encoder_input_size):
    global _METADATA_CONTEXT
    # CPU metadata preparation only; each process owns its reader and locks.
    torch.set_num_threads(1)
    _METADATA_CONTEXT = {
        "profile": profile, "adapters": adapters,
        "sources": {source.name: source for source in profile.sources},
        "store": TaskEmbeddingStore(**store_config),
        "verifier": VerifiedAssetStore(parquet_cache_bytes=1024**3),
        "root": root,
        "encoder_input_size": encoder_input_size,
        "task_bank_index_sha256": store_config["index_sha256"],
    }


def _metadata_worker_prepare(task):
    context = _METADATA_CONTEXT
    if context is None:
        raise StreamingMetadataError("metadata worker was not initialized")
    source = context["sources"].get(task.source)
    if source is None or source.embodiment != task.embodiment:
        raise StreamingMetadataError("task source/embodiment differs from profile")
    completed = _completed_episode(context["root"], task)
    if completed is not None:
        return completed
    prepared = _prepare_task(
        task=task, source=source, adapter=context["adapters"][task.source],
        profile=context["profile"], task_store=context["store"],
        asset_verifier=context["verifier"],
        encoder_input_size=context["encoder_input_size"],
        task_bank_index_sha256=context["task_bank_index_sha256"],
        decode_workers=1, decode_visuals=False,
    )
    return _write_episode(context["root"], prepared)


def _metadata_worker_group(tasks):
    # Keep one payload on one worker: verify/read its shard without multiplying
    # the same I/O across processes. Final episode/window ordering is unchanged.
    return [_metadata_worker_prepare(task) for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--task-bank-root", type=Path, required=True)
    parser.add_argument("--task-bank-index-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--window-index", type=Path, required=True)
    parser.add_argument("--grouped-normalization", type=Path, required=True)
    parser.add_argument("--output-seal", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--minimum-scale", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.workers <= 0 or args.processes <= 0:
        raise StreamingMetadataError("workers and processes must be positive")
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
    model_path = args.model_profile.resolve(strict=True)
    model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise StreamingMetadataError("model profile must be a mapping")
    encoder_path = args.encoder_contract.resolve(strict=True)
    encoder_config = _strict_encoder(encoder_path)
    representation = profile.cache_representation
    expected_encoder = {
        "token_grid": int(representation["token_grid"]),
        "appearance_token_grid": int(
            representation.get("appearance_token_grid", 0)
        ),
        "target_rgb_size": int(representation["rgb_size"]),
        "token_dim": int(representation["token_dim"]),
        "max_views": int(representation["num_views"]),
    }
    if expected_encoder["appearance_token_grid"] > 0:
        expected_encoder["appearance_feature_layer"] = int(
            representation.get("appearance_feature_layer", -1)
        )
    for name, expected in expected_encoder.items():
        if int(getattr(encoder_config, name)) != expected:
            raise StreamingMetadataError(
                f"encoder/data representation {name} mismatch: "
                f"{getattr(encoder_config, name)} != {expected}"
            )
    validate_model_data_compatibility(
        model,
        profile,
        appearance_cache_grid=expected_encoder["appearance_token_grid"],
    )
    model_sha = canonical_sha256(model)
    tasks = tuple(
        cache_task_from_mapping(dict(row))
        for _line, row in iter_jsonl(args.task_manifest)
    )
    if not tasks:
        raise StreamingMetadataError("task manifest is empty")
    sources = {source.name: source for source in profile.sources}
    adapters = {
        source.name: load_adapter_contract(
            source.adapter_config_path,
            expected_sha256=source.adapter_contract_sha256,
        )
        for source in profile.sources
    }
    task_encoder_digests = {task.task_encoder_contract_sha256 for task in tasks}
    if len(task_encoder_digests) != 1:
        raise StreamingMetadataError("task manifest mixes task encoders")
    store_config = dict(
        root=args.task_bank_root,
        index_sha256=args.task_bank_index_sha256,
        expected_data_profile_sha256=profile.profile_sha256,
        expected_source_manifest_sha256_by_name={
            source.name: source.manifest_sha256 for source in profile.sources
        },
        expected_encoder_contract_sha256=next(iter(task_encoder_digests)),
    )
    root = args.output_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise StreamingMetadataError("metadata output root cannot be a symlink")

    if not args.episode_index.exists():
        # Only work dispatch changes; published rows retain the original final sort.
        ordered_tasks = sorted(tasks, key=lambda task: (
            task.source, task.payload, task.payload_row_start, task.task_id))
        initargs = (profile, adapters, store_config, root, int(encoder_config.input_rgb_size))
        if args.processes > 1:
            pool = ProcessPoolExecutor(
                max_workers=args.processes, mp_context=multiprocessing.get_context("spawn"),
                initializer=_metadata_worker_init, initargs=initargs,
            )
        else:
            _metadata_worker_init(*initargs)
            pool = ThreadPoolExecutor(max_workers=args.workers)
        rows = []
        with pool:
            if args.processes > 1:
                groups = [list(group) for _key, group in groupby(
                    ordered_tasks, key=lambda task: (task.source, task.payload))]
                grouped_results = pool.map(_metadata_worker_group, groups, chunksize=1)
                results = (row for group in grouped_results for row in group)
            else:
                results = pool.map(_metadata_worker_prepare, ordered_tasks)
            for row in results:
                rows.append(row)
                if len(rows) % 5000 == 0:
                    print(json.dumps({"event": "metadata_episodes", "completed": len(rows),
                                      "total": len(tasks), "processes": args.processes}), flush=True)
        rows.sort(key=lambda row: (str(row["source"]), str(row["episode_id"])))
        _publish_jsonl(args.episode_index.absolute(), rows)
        del rows
    episodes = load_cache_episode_index(
        args.episode_index.absolute(),
        expected_sha256=sha256_file(args.episode_index.absolute()),
    )
    expected_episodes = {
        (task.source, task.episode_id): (
            task.split, task.embodiment,
            f"tasks/{task.task_id[:2]}/{task.task_id}/boundaries.safetensors",
        ) for task in tasks
    }
    if len(episodes) != len(tasks) or len(expected_episodes) != len(tasks):
        raise StreamingMetadataError("published episode index has different task coverage")
    for episode in episodes:
        if expected_episodes.get((episode.source, episode.episode_id)) != (
            episode.split, episode.embodiment, episode.feature_shard
        ):
            raise StreamingMetadataError("published episode index task binding differs")
    window_rows = iter_window_index(
        episodes=episodes,
        cache_root=root,
        model_profile=model,
        model_profile_sha256=model_sha,
        data_profile=profile,
        workers=args.processes,
    )
    window_count = _publish_jsonl(args.window_index.absolute(), window_rows)
    window_sha = sha256_file(args.window_index.absolute())
    normalization = build_grouped_normalization_artifact(
        data_profile=profile,
        model_profile=model,
        model_profile_sha256=model_sha,
        window_index_path=args.window_index.absolute(),
        window_index_sha256=window_sha,
        cache_root=root,
        minimum_scale=float(args.minimum_scale),
        workers=args.processes,
    )
    _publish(
        args.grouped_normalization.absolute(),
        (json.dumps(normalization, sort_keys=True, indent=2) + "\n").encode(),
    )
    seal = {
        "schema": STREAMING_METADATA_SEAL_SCHEMA,
        "data_profile_path": str(profile.path),
        "data_profile_sha256": profile.profile_sha256,
        "model_profile_path": str(model_path),
        "model_profile_sha256": model_sha,
        "task_manifest_path": str(args.task_manifest.resolve(strict=True)),
        "task_manifest_sha256": sha256_file(args.task_manifest.resolve(strict=True)),
        "task_count": len(tasks),
        "metadata_root": str(root),
        "episode_index_path": str(args.episode_index.absolute()),
        "episode_index_sha256": sha256_file(args.episode_index.absolute()),
        "episode_count": len(episodes),
        "window_index_path": str(args.window_index.absolute()),
        "window_index_sha256": window_sha,
        "window_count": window_count,
        "grouped_normalization_path": str(args.grouped_normalization.absolute()),
        "grouped_normalization_sha256": sha256_file(
            args.grouped_normalization.absolute()
        ),
        "encoder_contract_path": str(encoder_path),
        "encoder_contract_sha256": sha256_file(encoder_path),
        "task_bank_root": str(args.task_bank_root.resolve(strict=True)),
        "task_bank_index_sha256": args.task_bank_index_sha256,
        "task_encoder_contract_sha256": next(iter(task_encoder_digests)),
        "representation_contract_sha256": canonical_sha256(
            profile.cache_representation
        ),
        "source_manifest_sha256_by_name": {
            source.name: source.manifest_sha256 for source in profile.sources
        },
        "adapter_contract_sha256_by_name": {
            source.name: source.adapter_contract_sha256 for source in profile.sources
        },
    }
    _publish(
        args.output_seal.absolute(),
        (json.dumps(seal, sort_keys=True, indent=2) + "\n").encode(),
    )
    print(json.dumps({"event": "metadata_complete", "episodes": len(episodes),
                      "windows": window_count, "seal": str(args.output_seal)}), flush=True)


if __name__ == "__main__":
    main()
