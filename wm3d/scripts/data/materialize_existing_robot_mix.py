#!/usr/bin/env python3
"""Materialize the already-downloaded GAM/OXE + RoboCasa WM3D canary mix.

This is deliberately a local-reuse materializer: it does not download data,
does not apply the legacy GAM canonical action transforms, and does not build a
full visual cache.  It emits strict source-native adapters, a WM3D data-profile
template, one train/val/test episode per source, and a receipt binding the
existing raw roots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

import pyarrow.parquet as pq
import yaml

from wm3d.data.manifest_contract import sha256_file
from wm3d.data.source_inventory import deterministic_split


@dataclass(frozen=True)
class SourcePlan:
    name: str
    root_kind: str
    relative_root: str
    weight: int
    views: tuple[str, ...]
    bgr_views: tuple[str, ...] = ()


OXE = (
    SourcePlan("oxe_bridge", "oxe", "BrunoM42/bridge_orig_lerobot", 224,
               ("observation.images.image_0", "observation.images.image_1")),
    SourcePlan("oxe_droid", "droid", ".", 303,
               ("observation.images.exterior_2_left", "observation.images.wrist_left",
                "observation.images.exterior_1_left")),
    SourcePlan("oxe_taco_play", "oxe", "lerobot/taco_play", 60,
               ("observation.images.rgb_static", "observation.images.rgb_gripper")),
    SourcePlan("oxe_utaustin_mutex", "oxe", "lerobot/utaustin_mutex", 39,
               ("observation.images.image", "observation.images.wrist_image"),
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_stanford_hydra", "oxe", "lerobot/stanford_hydra_dataset", 24,
               ("observation.images.image", "observation.images.wrist_image"),
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_berkeley_autolab_ur5", "oxe", "lerobot/berkeley_autolab_ur5", 32,
               ("observation.images.image", "observation.images.hand_image"),
               ("observation.images.hand_image",)),
    SourcePlan("oxe_austin_sailor", "oxe", "lerobot/austin_sailor_dataset", 15,
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_austin_sirius", "oxe", "lerobot/austin_sirius_dataset", 24,
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_berkeley_fanuc", "oxe", "lerobot/berkeley_fanuc_manipulation", 20,
               ("observation.images.image", "observation.images.wrist_image"),
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_jaco_play", "oxe", "lerobot/jaco_play", 33,
               ("observation.images.image", "observation.images.image_wrist")),
    SourcePlan("oxe_fmb", "oxe", "lerobot/fmb", 42,
               ("observation.images.image_side_1", "observation.images.image_wrist_1"),
               ("observation.images.image_side_1", "observation.images.image_side_2",
                "observation.images.image_wrist_1", "observation.images.image_wrist_2")),
    SourcePlan("oxe_kuka", "oxe", "lerobot/stanford_kuka_multimodal_dataset", 50,
               ("observation.images.image",)),
    SourcePlan("oxe_fractal", "oxe", "BrunoM42/fractal20220817_data_lerobot", 271,
               ("observation.images.image",)),
    SourcePlan("oxe_berkeley_cable", "oxe", "lerobot/berkeley_cable_routing", 8,
               ("observation.images.image", "observation.images.wrist45_image")),
    SourcePlan("oxe_roboturk", "oxe", "lerobot/roboturk", 20,
               ("observation.images.front_rgb",)),
    SourcePlan("oxe_dlr_edan", "oxe", "lerobot/dlr_edan_shared_control", 1,
               ("observation.images.image",)),
    SourcePlan("oxe_austin_buds", "oxe", "lerobot/austin_buds_dataset", 7,
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_nyu_franka", "oxe", "lerobot/nyu_franka_play_dataset", 10,
               ("observation.images.image", "observation.images.image_additional_view")),
    SourcePlan("oxe_nyu_door", "oxe", "lerobot/nyu_door_opening_surprising_effectiveness", 10,
               ("observation.images.image",)),
    SourcePlan("oxe_cmu_stretch", "oxe", "lerobot/cmu_stretch", 5,
               ("observation.images.image",)),
    SourcePlan("oxe_furniture_bench", "oxe", "tailong-wu/furniture_bench_dataset_lerobot_v30", 71,
               ("observation.images.image", "observation.images.wrist_image")),
    SourcePlan("oxe_bc_z", "oxe", "tailong-wu/bc_z_lerobot_v30", 208,
               ("observation.images.image",)),
    SourcePlan("oxe_language_table", "oxe", "tailong-wu/language_table_lerobot_v30", 100,
               ("observation.images.rgb",)),
)

ROBOCASA = (
    SourcePlan("robocasa_atomic", "robocasa", "robocasa365-pretrain-atomic", 7,
               ("observation.images.robot0_agentview_left",
                "observation.images.robot0_eye_in_hand")),
    SourcePlan("robocasa_composite", "robocasa", "robocasa365-pretrain-composite", 160,
               ("observation.images.robot0_agentview_left",
                "observation.images.robot0_eye_in_hand")),
    SourcePlan("robocasa_mg", "robocasa", "robocasa365-pretrain-mg", 52,
               ("observation.images.robot0_agentview_left",
                "observation.images.robot0_eye_in_hand")),
)

PLANS = OXE + ROBOCASA
VIEW_SLOTS = ("head", "left_wrist", "right_wrist")


def _real_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be an absolute real directory: {path}")
    return path.resolve(strict=True)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _root(plan: SourcePlan, *, oxe: Path, droid: Path, robocasa: Path) -> Path:
    base = {"oxe": oxe, "droid": droid, "robocasa": robocasa}[plan.root_kind]
    root = base if plan.relative_root == "." else base / plan.relative_root
    return _real_dir(root, plan.name)


def _width(features: dict, key: str, maximum: int) -> int:
    value = features.get(key)
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 1:
        raise RuntimeError(f"{key} must declare a one-dimensional shape")
    width = int(shape[0])
    if not 0 < width <= maximum:
        raise RuntimeError(f"{key} width {width} exceeds WM3D capacity {maximum}")
    return width


def _adapter(plan: SourcePlan, info: dict) -> tuple[dict, dict]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise RuntimeError(f"{plan.name}: missing features")
    action_dim = _width(features, "action", 16)
    state_dim = _width(features, "observation.state", 32)
    missing = set(plan.views) - set(features)
    if missing:
        raise RuntimeError(f"{plan.name}: selected RGB views are missing: {sorted(missing)}")
    adapter = {
        "schema": "wm3d_source_adapter_v4",
        "name": f"{plan.name}_source_native_controller",
        "raw_format": "lerobot_parquet_video",
        "observation_time_key": "timestamp",
        "views": [
            {
                "name": slot,
                "key": key,
                "color_order": "bgr" if key in plan.bgr_views else "rgb",
            }
            for slot, key in zip(VIEW_SLOTS, plan.views, strict=False)
        ],
        "groups": [
            {
                "group": "controller",
                "supervision": "fine_command",
                "action": [{
                    "key": "action",
                    "columns": list(range(action_dim)),
                    "scale": [1.0] * action_dim,
                    "offset": [0.0] * action_dim,
                }],
                "state": [{
                    "key": "observation.state",
                    "columns": list(range(state_dim)),
                    "scale": [1.0] * state_dim,
                    "offset": [0.0] * state_dim,
                }],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
            }
        ],
    }
    embodiment = {
        "name": f"{plan.name}_controller",
        "embodiment_id": 200 + PLANS.index(plan),
        "groups": [{
            "name": "controller",
            "group_id": 30,
            "action_semantics": ["controller_command"] * action_dim,
            "state_semantics": ["controller_state"] * state_dim,
            "action_frame": "source_controller_native",
            "state_frame": "source_controller_native",
            "composition_operators": ["last"] * action_dim,
        }],
    }
    return adapter, embodiment


def _episode_indices(root: Path):
    jsonl = root / "meta/episodes.jsonl"
    if jsonl.is_file() and not jsonl.is_symlink():
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield int(json.loads(line)["episode_index"])
        return
    paths = sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise RuntimeError(f"no episode metadata under {root}")
    for path in paths:
        safe = _regular(path, "episode metadata")
        parquet = pq.ParquetFile(safe)
        for batch in parquet.iter_batches(columns=["episode_index"], batch_size=4096):
            for index in batch.column(0).to_pylist():
                yield int(index)


def _selected_indices(source: str, root: Path) -> tuple[int, int, int]:
    selected: dict[str, int] = {}
    for index in _episode_indices(root):
        episode_id = f"{source}:{index:09d}"
        split = deterministic_split(
            source,
            episode_id,
            seed=3407,
            train_fraction=0.8,
            validation_fraction=0.1,
        )
        selected.setdefault(split, index)
        if set(selected) == {"train", "val", "test"}:
            return selected["train"], selected["val"], selected["test"]
    raise RuntimeError(f"{source}: could not select train/val/test canary episodes")


def _profiles(model_path: Path, encoder_path: Path) -> tuple[dict, dict]:
    model = yaml.safe_load(_regular(model_path, "model profile").read_text())
    encoder = yaml.safe_load(_regular(encoder_path, "encoder contract").read_text())
    model_body = model["model"]
    sampling = model["sampling"]
    grid = int(encoder["token_grid"])
    observed = (grid * grid, int(encoder["token_dim"]),
                int(encoder["max_views"]), int(encoder["target_rgb_size"]))
    expected = (int(model_body["P"]), int(model_body["token_dim"]),
                int(model_body["num_views"]), int(model_body["rgb_size"]))
    if observed != expected:
        raise RuntimeError(f"model/encoder representation mismatch: {observed} != {expected}")
    representation = {
        "schema": "wm3d_v8_episode_representation_v1",
        "token_grid": grid,
        "spatial_tokens": grid * grid,
        "token_dim": observed[1],
        "num_views": observed[2],
        "view_slots": list(VIEW_SLOTS),
        "rgb_size": observed[3],
        "time_binding": "episode_row_ordinal_with_pts_audit",
        "missing_view_policy": "mask_without_duplication",
        "state_frame_selection": {
            "mode": "observed_greedy_minimum_separation",
            "minimum_separation_seconds": float(
                sampling["minimum_anchor_separation_seconds"]
            ),
            "preserve_observed_timestamps": True,
            "interpolation": "forbidden",
        },
    }
    cache = {
        "schema": "wm3d_v8_unified_window_index_v3",
        "task_partition": "episode",
        "feature_workers_per_node": 8,
        "decode_workers_per_gpu": 4,
        "writer_threads_per_worker": 2,
        "task_claim": "atomic_no_clobber",
        "resume": "receipt_and_sha",
        "view_token_codec": "int8_per_vector",
        "depth_codec": "fp16",
        "point_codec": "fp16",
        "camera_pose_codec": "fp32",
        "rgb_codec": "jpeg_pack",
        "action_proprio_storage": "same_episode_artifact",
    }
    return representation, cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oxe-root", type=Path, required=True)
    parser.add_argument("--droid-root", type=Path, required=True)
    parser.add_argument("--robocasa-root", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--encoder-contract", type=Path, required=True)
    parser.add_argument("--old-spec", type=Path, required=True)
    parser.add_argument("--old-config", type=Path, required=True)
    parser.add_argument("--old-log", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    roots = {
        "oxe": _real_dir(args.oxe_root, "OXE root"),
        "droid": _real_dir(args.droid_root, "DROID root"),
        "robocasa": _real_dir(args.robocasa_root, "RoboCasa root"),
    }
    output = args.output_root.absolute()
    output.mkdir(parents=True, exist_ok=True)
    representation, cache = _profiles(args.model_profile, args.encoder_contract)
    sources = []
    embodiments = []
    receipt_rows = []
    adapter_root = output / "adapters"
    episode_root = output / "episode_indices"
    for plan in PLANS:
        root = _root(plan, **roots)
        info_path = _regular(root / "meta/info.json", f"{plan.name} info")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        adapter, embodiment = _adapter(plan, info)
        adapter_path = adapter_root / f"{plan.name}.yaml"
        _publish(
            adapter_path,
            yaml.safe_dump(adapter, sort_keys=False, allow_unicode=True).encode("utf-8"),
        )
        selected = _selected_indices(plan.name, root)
        episode_path = episode_root / f"{plan.name}.txt"
        _publish(episode_path, ("\n".join(str(item) for item in selected) + "\n").encode())
        sources.append({
            "name": plan.name,
            "adapter": "lerobot",
            "raw_root": str(root),
            "adapter_config": str(adapter_path.absolute()),
            "adapter_contract_sha256": sha256_file(adapter_path.absolute()),
            "manifest": f"__MATERIALIZE_REQUIRED__/{plan.name}.jsonl",
            "manifest_sha256": "__MATERIALIZE_REQUIRED__",
            "embodiment": embodiment["name"],
            "weight": plan.weight,
            "nominal_hours": (
                float(info["total_frames"]) / float(info["fps"]) / 3600.0
            ),
            "license_id": "operator_verified_upstream_license",
        })
        embodiments.append(embodiment)
        receipt_rows.append({
            "name": plan.name,
            "raw_root": str(root),
            "info_path": str(info_path),
            "info_sha256": sha256_file(info_path),
            "total_episodes": int(info["total_episodes"]),
            "total_frames": int(info["total_frames"]),
            "adapter_sha256": sha256_file(adapter_path.absolute()),
            "episode_index_sha256": sha256_file(episode_path.absolute()),
            "selected_episode_indices": list(selected),
        })
    template = {
        "schema": "wm3d_v8_data_profile_v4",
        "name": "wm3d_1b_existing_robot_raw_canary",
        "cache_representation": representation,
        "cache": cache,
        "sources": sources,
        "embodiments": embodiments,
        "notes": {
            "purpose": "no-PCA raw streaming canary over the existing GAM/OXE and RoboCasa data",
            "source_count": len(PLANS),
            "split_policy": "one deterministic train/val/test episode per source for canary",
            "action_state_policy": "source-native opaque controller vectors with recorded timestamps",
            "color_policy": "legacy GAM BGR declarations are explicitly restored before RGB supervision",
        },
    }
    template_path = output / "data_template.yaml"
    _publish(
        template_path,
        yaml.safe_dump(template, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )
    receipt = {
        "schema": "wm3d_existing_robot_mix_reuse_v1",
        "code_commit": args.code_commit,
        "model_profile_sha256": sha256_file(_regular(args.model_profile, "model profile")),
        "encoder_contract_sha256": sha256_file(
            _regular(args.encoder_contract, "encoder contract")
        ),
        "legacy_provenance": {
            "pretraining_spec_sha256": sha256_file(_regular(args.old_spec, "old spec")),
            "training_config_sha256": sha256_file(_regular(args.old_config, "old config")),
            "completed_training_log_sha256": sha256_file(_regular(args.old_log, "old log")),
        },
        "data_template_path": str(template_path.absolute()),
        "data_template_sha256": sha256_file(template_path.absolute()),
        "source_count": len(receipt_rows),
        "sources": receipt_rows,
    }
    receipt_path = output / "local_reuse_receipt.json"
    _publish(
        receipt_path,
        (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "source_count": len(PLANS),
        "data_template": str(template_path.absolute()),
        "data_template_sha256": sha256_file(template_path.absolute()),
        "local_reuse_receipt": str(receipt_path.absolute()),
        "local_reuse_receipt_sha256": sha256_file(receipt_path.absolute()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
