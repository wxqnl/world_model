#!/usr/bin/env python3
"""Seal four orthogonal WM3D profiles into one immutable runtime YAML."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import yaml

from wm3d.data.manifest_contract import load_data_profile, sha256_file
from wm3d.data.direct_raw import DIRECT_RAW_DATA_CLOSURE_SCHEMA
from wm3d.data.streaming_raw import (
    STREAMING_DATA_CLOSURE_SCHEMA,
    load_streaming_metadata_seal,
)
from wm3d.training.runtime_contract import (
    DATA_CLOSURE_SCHEMA,
    RUNTIME_CONFIG_SCHEMA,
    canonical_sha256,
    load_yaml,
    validate_materialized_runtime,
    validate_runtime_profile,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _publish_no_clobber(path: Path, payload: bytes) -> None:
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
            raise FileExistsError(f"refusing to overwrite non-identical runtime: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _direct_ignored_action_dimensions(
    values: list[str],
    data_profile: Any,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], set[int]] = {}
    source_by_name = {source.name: source for source in data_profile.sources}
    for value in values:
        parts = str(value).split(":")
        if len(parts) != 3:
            raise RuntimeError(
                "direct ignored action dimension must be SOURCE:GROUP:DIMENSION"
            )
        source_name, group_name, raw_dimension = parts
        source = source_by_name.get(source_name)
        if source is None:
            raise RuntimeError(
                f"direct ignored action source is unknown: {source_name}"
            )
        embodiment = data_profile.embodiments[source.embodiment]
        group = next(
            (item for item in embodiment.groups if item.name == group_name),
            None,
        )
        try:
            dimension = int(raw_dimension)
        except ValueError as exc:
            raise RuntimeError(
                "direct ignored action dimension must be an integer"
            ) from exc
        if group is None or not 0 <= dimension < group.action_dim:
            raise RuntimeError(
                f"direct ignored action coordinate is invalid: {value}"
            )
        key = (source_name, group_name)
        dimensions = grouped.setdefault(key, set())
        if dimension in dimensions:
            raise RuntimeError(f"duplicate direct ignored action coordinate: {value}")
        dimensions.add(dimension)
    return [
        {
            "source": source,
            "group": group,
            "dimensions": sorted(dimensions),
        }
        for (source, group), dimensions in sorted(grouped.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument(
        "--data-mode",
        choices=("episode_cache", "streaming_raw", "direct_raw"),
        default="episode_cache",
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--episode-cache-index", type=Path)
    parser.add_argument("--episode-cache-seal", type=Path)
    parser.add_argument("--cache-index", type=Path)
    parser.add_argument("--cache-seal", type=Path)
    parser.add_argument("--grouped-normalization", type=Path)
    parser.add_argument("--streaming-metadata-seal", type=Path)
    parser.add_argument("--streaming-lru-root", type=Path)
    parser.add_argument("--streaming-lru-gib-per-rank", type=float, default=64.0)
    parser.add_argument("--streaming-encode-batch-frames", type=int, default=16)
    parser.add_argument("--streaming-decode-workers", type=int, default=4)
    parser.add_argument("--streaming-appearance-feature-layer", type=int)
    parser.add_argument("--direct-input-rgb-size", type=int, default=518)
    parser.add_argument("--direct-decode-workers", type=int, default=4)
    parser.add_argument("--direct-robot-cache-episodes", type=int, default=8)
    parser.add_argument("--direct-prefetch-windows", type=int, default=32)
    parser.add_argument("--direct-video-index-cache-assets", type=int, default=128)
    parser.add_argument("--direct-encode-chunk-rows", type=int, default=32)
    parser.add_argument("--direct-minimum-chunk-rows", type=int, default=4)
    parser.add_argument(
        "--direct-ignore-action-dimension",
        action="append",
        default=[],
        metavar="SOURCE:GROUP:DIMENSION",
    )
    parser.add_argument("--direct-appearance-feature-layer", type=int, default=4)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-lineage", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-dirty-smoke",
        action="store_true",
        help="Only for isolated smoke fixtures; formal materialization requires clean git.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    status = _git(repo, "status", "--porcelain")
    if status and not args.allow_dirty_smoke:
        raise RuntimeError("formal runtime materialization requires a clean git tree")
    if args.allow_dirty_smoke and "smoke" not in args.run_lineage.lower():
        raise RuntimeError("--allow-dirty-smoke requires 'smoke' in run lineage")
    commit = _git(repo, "rev-parse", "HEAD")

    model = load_yaml(args.model)
    runtime = load_yaml(args.runtime)
    validate_runtime_profile(runtime)
    objective = load_yaml(args.objective)
    data_profile = load_data_profile(args.data, verify_source_manifests=True)
    environment = args.environment_lock.resolve(strict=True)
    output_root = args.output_root.absolute()
    if output_root.is_symlink():
        raise RuntimeError("output root cannot be a symlink")
    source_manifest_sha256_by_name = {
        source.name: source.manifest_sha256 for source in data_profile.sources
    }
    adapter_contract_sha256_by_name = {
        source.name: source.adapter_contract_sha256 for source in data_profile.sources
    }
    if args.data_mode == "episode_cache":
        required = {
            "cache_root": args.cache_root,
            "episode_cache_index": args.episode_cache_index,
            "episode_cache_seal": args.episode_cache_seal,
            "cache_index": args.cache_index,
            "cache_seal": args.cache_seal,
            "grouped_normalization": args.grouped_normalization,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise RuntimeError(f"episode_cache runtime misses arguments: {missing}")
        cache_root = args.cache_root.resolve(strict=True)
        episode_cache_index = args.episode_cache_index.resolve(strict=True)
        episode_cache_seal = args.episode_cache_seal.resolve(strict=True)
        cache_index = args.cache_index.resolve(strict=True)
        cache_seal = args.cache_seal.resolve(strict=True)
        grouped_normalization = args.grouped_normalization.resolve(strict=True)
        closure = {
            "schema": DATA_CLOSURE_SCHEMA,
            "name": data_profile.name,
            "data_profile_path": str(data_profile.path),
            "data_profile_sha256": data_profile.profile_sha256,
            "cache_root": str(cache_root),
            "episode_cache_index_path": str(episode_cache_index),
            "episode_cache_index_sha256": sha256_file(episode_cache_index),
            "episode_cache_seal_path": str(episode_cache_seal),
            "episode_cache_seal_sha256": sha256_file(episode_cache_seal),
            "cache_index_path": str(cache_index),
            "cache_index_sha256": sha256_file(cache_index),
            "cache_seal_path": str(cache_seal),
            "cache_seal_sha256": sha256_file(cache_seal),
            "grouped_normalization_path": str(grouped_normalization),
            "grouped_normalization_sha256": sha256_file(grouped_normalization),
            "source_manifest_sha256_by_name": source_manifest_sha256_by_name,
            "adapter_contract_sha256_by_name": adapter_contract_sha256_by_name,
        }
    else:
        if args.streaming_metadata_seal is None:
            raise RuntimeError(
                f"{args.data_mode} requires --streaming-metadata-seal"
            )
        if args.data_mode == "streaming_raw":
            if args.streaming_lru_root is None:
                raise RuntimeError(
                    "streaming_raw requires --streaming-lru-root"
                )
            if (
                args.streaming_lru_gib_per_rank <= 0
                or args.streaming_encode_batch_frames <= 0
                or args.streaming_decode_workers <= 0
            ):
                raise RuntimeError(
                    "streaming_raw LRU/encode/decode values must be positive"
                )
        else:
            direct_positive = (
                args.direct_input_rgb_size,
                args.direct_decode_workers,
                args.direct_robot_cache_episodes,
                args.direct_prefetch_windows,
                args.direct_video_index_cache_assets,
                args.direct_encode_chunk_rows,
                args.direct_minimum_chunk_rows,
            )
            if any(value <= 0 for value in direct_positive):
                raise RuntimeError("direct_raw decode/queue/chunk values must be positive")
            if args.direct_input_rgb_size % 14:
                raise RuntimeError("direct_raw RGB size must be divisible by 14")
            if args.direct_minimum_chunk_rows > args.direct_encode_chunk_rows:
                raise RuntimeError("direct_raw minimum chunk exceeds initial chunk")
        metadata_seal_path = args.streaming_metadata_seal.resolve(strict=True)
        metadata_seal = load_streaming_metadata_seal(metadata_seal_path)
        if metadata_seal["data_profile_sha256"] != data_profile.profile_sha256:
            raise RuntimeError("streaming metadata belongs to another data profile")
        geometry_grid = int(data_profile.cache_representation["token_grid"])
        model_uses_appearance = bool(
            model["model"].get("appearance_enabled", False)
        )
        configured_appearance_grid = data_profile.cache_representation.get(
            "appearance_token_grid"
        )
        configured_appearance_layer = data_profile.cache_representation.get(
            "appearance_feature_layer"
        )
        requested_appearance_layer = (
            args.streaming_appearance_feature_layer
            if args.data_mode == "streaming_raw"
            else args.direct_appearance_feature_layer
        )
        if requested_appearance_layer is not None:
            requested_layer = int(requested_appearance_layer)
            if requested_layer not in (4, 11, 17, 23):
                raise RuntimeError(
                    "appearance layer must be a cached VGGT feature layer"
                )
            if (
                configured_appearance_layer is not None
                and int(configured_appearance_layer) != requested_layer
            ):
                raise RuntimeError(
                    "appearance layer differs from the data profile"
                )
            configured_appearance_layer = requested_layer
        if not model_uses_appearance:
            # The direct adapter previously extracted the sealed P256 feature
            # merely because the data profile advertised it, even when the
            # selected model could not consume appearance tokens.  Use the
            # geometry grid so NativeVGGTEncoder disables that extra feature
            # tap and its host/GPU transfer while retaining the same raw video.
            appearance_grid = geometry_grid
        elif configured_appearance_grid is not None:
            appearance_grid = int(configured_appearance_grid)
        elif model_uses_appearance:
            appearance_tokens = int(model["model"]["appearance_P"])
            appearance_grid = math.isqrt(appearance_tokens)
            if appearance_grid * appearance_grid != appearance_tokens:
                raise RuntimeError("appearance_P must be a square token grid")
        else:
            appearance_grid = geometry_grid
        if appearance_grid < geometry_grid:
            raise RuntimeError("appearance grid cannot be below the geometry grid")
        if (
            model_uses_appearance
            and configured_appearance_layer is not None
            and appearance_grid == geometry_grid
        ):
            raise RuntimeError(
                "streaming appearance layer requires a distinct appearance grid"
            )
        closure = {
            "name": data_profile.name,
            "data_profile_path": str(data_profile.path),
            "data_profile_sha256": data_profile.profile_sha256,
            "metadata_seal_path": str(metadata_seal_path),
            "metadata_seal_sha256": sha256_file(metadata_seal_path),
            "metadata_root": metadata_seal["metadata_root"],
            "episode_index_path": metadata_seal["episode_index_path"],
            "episode_index_sha256": metadata_seal["episode_index_sha256"],
            "cache_index_path": metadata_seal["window_index_path"],
            "cache_index_sha256": metadata_seal["window_index_sha256"],
            "grouped_normalization_path": metadata_seal[
                "grouped_normalization_path"
            ],
            "grouped_normalization_sha256": metadata_seal[
                "grouped_normalization_sha256"
            ],
            "task_manifest_path": metadata_seal["task_manifest_path"],
            "task_manifest_sha256": metadata_seal["task_manifest_sha256"],
            "encoder_contract_path": metadata_seal["encoder_contract_path"],
            "encoder_contract_sha256": metadata_seal["encoder_contract_sha256"],
            "task_bank_root": metadata_seal["task_bank_root"],
            "task_bank_index_sha256": metadata_seal["task_bank_index_sha256"],
            "source_manifest_sha256_by_name": source_manifest_sha256_by_name,
            "adapter_contract_sha256_by_name": adapter_contract_sha256_by_name,
            "appearance_token_grid": appearance_grid,
        }
        if configured_appearance_layer is not None:
            closure["appearance_feature_layer"] = int(
                configured_appearance_layer
            )
        if args.data_mode == "streaming_raw":
            lru_root = args.streaming_lru_root.absolute()
            if lru_root.is_symlink():
                raise RuntimeError("streaming LRU root cannot be a symlink")
            closure.update(
                {
                    "schema": STREAMING_DATA_CLOSURE_SCHEMA,
                    "lru_root": str(lru_root),
                    "lru_max_bytes_per_rank": int(
                        args.streaming_lru_gib_per_rank * 1024**3
                    ),
                    "encode_batch_frames": int(
                        args.streaming_encode_batch_frames
                    ),
                    "decode_workers": int(args.streaming_decode_workers),
                }
            )
        else:
            if configured_appearance_layer is None:
                raise RuntimeError("direct_raw requires an appearance feature layer")
            closure.update(
                {
                    "schema": DIRECT_RAW_DATA_CLOSURE_SCHEMA,
                    "direct_ignored_action_dimensions": (
                        _direct_ignored_action_dimensions(
                            args.direct_ignore_action_dimension,
                            data_profile,
                        )
                    ),
                    "direct_input_rgb_size": int(args.direct_input_rgb_size),
                    "direct_decode_workers": int(args.direct_decode_workers),
                    "direct_robot_cache_episodes": int(
                        args.direct_robot_cache_episodes
                    ),
                    "direct_prefetch_windows": int(args.direct_prefetch_windows),
                    "direct_video_index_cache_assets": int(
                        args.direct_video_index_cache_assets
                    ),
                    "direct_encode_chunk_rows": int(
                        args.direct_encode_chunk_rows
                    ),
                    "direct_minimum_chunk_rows": int(
                        args.direct_minimum_chunk_rows
                    ),
                }
            )
    value = {
        "schema": RUNTIME_CONFIG_SCHEMA,
        "run": {
            "name": args.run_name,
            "lineage": args.run_lineage,
            "output_root": str(output_root),
            "code_commit": commit,
            "environment_lock_path": str(environment),
            "environment_lock_sha256": sha256_file(environment),
        },
        "model_profile": model,
        "data_closure": closure,
        "runtime_profile": runtime,
        "objective_profile": objective,
        "bindings": {
            "model_profile_sha256": canonical_sha256(model),
            "data_closure_sha256": canonical_sha256(closure),
            "runtime_profile_sha256": canonical_sha256(runtime),
            "objective_profile_sha256": canonical_sha256(objective),
            "model_contract_sha256": canonical_sha256(
                {"architecture": model["architecture"], "model": model["model"]}
            ),
        },
    }
    validate_materialized_runtime(value)
    payload = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")
    _publish_no_clobber(args.output.absolute(), payload)
    print(
        json.dumps(
            {
                "runtime": str(args.output.absolute()),
                "runtime_sha256": sha256_file(args.output.absolute()),
                "data_closure_sha256": canonical_sha256(closure),
                "model_profile": model["name"],
                "runtime_profile": runtime["name"],
                "world_size": runtime["expected_world_size"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
