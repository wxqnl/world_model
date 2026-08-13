#!/usr/bin/env python3
"""Generate the full official OXE replacement for AgiBotWorld Beta."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import uuid

import yaml


COLLECTION_API = (
    "https://huggingface.co/api/collections/lerobot/"
    "open-x-embodiment-68de658d8b544a43be4c6687"
)
SOURCE_SCHEMA = "wm3d_v8_raw_source_lock_v1"
DATA_SCHEMA = "wm3d_v8_data_profile_v4"
MAX_ACTION_DIM = 16
MAX_STATE_DIM = 32
VIEW_SLOTS = ("head", "left_wrist", "right_wrist")


def _publish(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite non-identical OXE artifact: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_yaml(path: Path, *, schema: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular YAML file: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema}")
    return value


def _load_json_url(url: str) -> dict:
    last: Exception | None = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "wm3d-oxe/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError(f"remote JSON is not an object: {url}")
            return value
        except Exception as exc:  # pragma: no cover - exercised by real network retries
            last = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed to read {url}: {last}")


def _download_json(url: str, destination: Path) -> dict:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"JSON cache path is not a regular file: {destination}")
        value = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"JSON cache is not an object: {destination}")
        return value
    value = _load_json_url(url)
    _publish(
        destination,
        (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return value


def _repo_ids(path: Path | None, *, collection_url: str) -> tuple[str, ...]:
    if path is None:
        value = _load_json_url(collection_url)
    else:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"collection JSON must be a regular file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError("official OXE collection has no items")
    repos = tuple(str(item.get("id", "")) for item in items if isinstance(item, dict))
    if any(not repo.startswith("lerobot/") for repo in repos):
        raise ValueError("official OXE collection contains a non-LeRobot dataset")
    if len(repos) != len(set(repos)):
        raise ValueError("official OXE collection contains duplicate datasets")
    if "lerobot/droid_1.0.1" not in repos:
        raise ValueError("official OXE collection no longer contains the existing DROID source")
    return repos


def _slug(repo_id: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", repo_id.split("/", 1)[1].lower()).strip("_")
    if not value:
        raise ValueError(f"cannot derive source name from {repo_id!r}")
    return value


def _metadata(
    repo_id: str,
    *,
    metadata_root: Path | None,
) -> dict:
    slug = _slug(repo_id)
    if metadata_root is not None:
        path = metadata_root / f"{slug}.json"
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing metadata fixture for {repo_id}: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"metadata fixture is not an object: {path}")
        return value
    return _load_json_url(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json"
    )


def _width(features: dict, key: str, maximum: int) -> int:
    value = features.get(key)
    shape = value.get("shape") if isinstance(value, dict) else None
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"{key} must declare one-dimensional shape")
    width = int(shape[0])
    if not 0 < width <= maximum:
        raise ValueError(f"{key} width {width} is outside WM3D capacity {maximum}")
    return width


def _adapter_and_embodiment(
    *,
    repo_id: str,
    info: dict,
    embodiment_id: int,
    group_id: int,
) -> tuple[dict, dict]:
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{repo_id}: meta/info.json has no features mapping")
    action_dim = _width(features, "action", MAX_ACTION_DIM)
    state_dim = _width(features, "observation.state", MAX_STATE_DIM)
    view_keys = sorted(
        str(key)
        for key, value in features.items()
        if str(key).startswith("observation.images.")
        and isinstance(value, dict)
        and str(value.get("dtype", "")).lower() in {"video", "image"}
    )
    if not view_keys:
        raise ValueError(f"{repo_id}: dataset declares no RGB observation stream")
    view_keys = view_keys[: len(VIEW_SLOTS)]
    slug = _slug(repo_id)
    source_name = f"oxe_{slug}"
    embodiment_name = f"{source_name}_controller"
    adapter = {
        "schema": "wm3d_v8_source_adapter_v3",
        "name": f"{source_name}_opaque_controller",
        "raw_format": "lerobot_parquet_video",
        "observation_time_key": "timestamp",
        "views": [
            {"name": slot, "key": key}
            for slot, key in zip(VIEW_SLOTS, view_keys, strict=False)
        ],
        "groups": [
            {
                "group": "controller",
                "supervision": "fine_command",
                "action": [
                    {
                        "key": "action",
                        "columns": list(range(action_dim)),
                        "scale": [1.0] * action_dim,
                        "offset": [0.0] * action_dim,
                    }
                ],
                "state": [
                    {
                        "key": "observation.state",
                        "columns": list(range(state_dim)),
                        "scale": [1.0] * state_dim,
                        "offset": [0.0] * state_dim,
                    }
                ],
                "action_time_key": "timestamp",
                "state_time_key": "timestamp",
                "world_interval_index_key": None,
            }
        ],
    }
    embodiment = {
        "name": embodiment_name,
        "embodiment_id": embodiment_id,
        "groups": [
            {
                "name": "controller",
                "group_id": group_id,
                "action_semantics": ["controller_command"] * action_dim,
                "state_semantics": ["controller_state"] * state_dim,
                "action_frame": "source_controller_native",
                "state_frame": "source_controller_native",
                "composition_operators": ["last"] * action_dim,
            }
        ],
    }
    return adapter, embodiment


def build_templates(
    *,
    base_source: dict,
    base_data: dict,
    repo_ids: tuple[str, ...],
    metadata_by_repo: dict[str, dict],
) -> tuple[dict, dict, dict[str, dict]]:
    source_rows = base_source.get("sources")
    data_rows = base_data.get("sources")
    embodiments = base_data.get("embodiments")
    if not isinstance(source_rows, list) or not isinstance(data_rows, list):
        raise ValueError("base templates must contain source lists")
    if not isinstance(embodiments, list):
        raise ValueError("base data template must contain embodiments")

    source_output = copy.deepcopy(base_source)
    source_output["sources"] = [
        copy.deepcopy(row)
        for row in source_rows
        if row.get("name") not in {"agibot_beta", "agibot_alpha_converter"}
    ]
    data_output = copy.deepcopy(base_data)
    data_output["name"] = "public_robot_oxe_no_beta"
    data_output["sources"] = [
        copy.deepcopy(row) for row in data_rows if row.get("name") != "agibot_beta"
    ]
    data_output["embodiments"] = copy.deepcopy(embodiments)

    oxe = [repo for repo in repo_ids if repo != "lerobot/droid_1.0.1"]
    if not oxe:
        raise ValueError("OXE replacement would be empty after de-duplicating DROID")
    adapters: dict[str, dict] = {}
    for offset, repo_id in enumerate(oxe):
        slug = _slug(repo_id)
        name = f"oxe_{slug}"
        adapter, embodiment = _adapter_and_embodiment(
            repo_id=repo_id,
            info=metadata_by_repo[repo_id],
            embodiment_id=5 + offset,
            # Group identity may be shared between embodiments. Keeping it at
            # 30 avoids exceeding the model's max_group_id capacity.
            group_id=30,
        )
        adapters[name] = adapter
        source_output["sources"].append(
            {
                "name": name,
                "transport": "huggingface_dataset",
                "repo_id": repo_id,
                "revision": "__MATERIALIZE_REQUIRED_40_HEX_COMMIT__",
                "include": ["**"],
                "destination": f"oxe/{slug}",
                "gated": False,
            }
        )
        data_output["sources"].append(
            {
                "name": name,
                "adapter": "lerobot",
                "raw_root": f"__MATERIALIZE_REQUIRED__/raw/oxe/{slug}",
                "adapter_config": f"__MATERIALIZE_REQUIRED__/adapters/{name}.yaml",
                "adapter_contract_sha256": "__MATERIALIZE_REQUIRED__",
                "manifest": f"__MATERIALIZE_REQUIRED__/{name}.jsonl",
                "manifest_sha256": "__MATERIALIZE_REQUIRED__",
                "embodiment": embodiment["name"],
                "weight": 30,
                "nominal_hours": None,
                "license_id": "operator_verified_upstream_license",
            }
        )
        data_output["embodiments"].append(embodiment)

    # Original non-Beta sources sum to 70. Multiplying each unchanged weight
    # by N and assigning 30 to each of N OXE datasets preserves every main
    # ratio and gives the OXE pool exactly 30% of the deterministic schedule.
    for row in data_output["sources"]:
        if not str(row["name"]).startswith("oxe_"):
            row["weight"] = int(row["weight"]) * len(oxe)
    data_output["notes"] = {
        **dict(data_output.get("notes") or {}),
        "replacement_policy": (
            "AgiBotWorld Beta is omitted; every official LeRobot OXE dataset except "
            "the already-present DROID source is included"
        ),
        "oxe_collection_url": COLLECTION_API,
        "oxe_dataset_count_including_droid": len(repo_ids),
        "oxe_new_source_count": len(oxe),
        "oxe_sampling_share": "30_percent_equal_per_dataset",
    }
    return source_output, data_output, adapters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-source-template", type=Path, required=True)
    parser.add_argument("--base-data-template", type=Path, required=True)
    parser.add_argument("--collection-json", type=Path)
    parser.add_argument("--collection-url", default=COLLECTION_API)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--output-source-template", type=Path, required=True)
    parser.add_argument("--output-data-template", type=Path, required=True)
    parser.add_argument("--output-adapter-root", type=Path, required=True)
    args = parser.parse_args()
    source = _load_yaml(args.base_source_template, schema=SOURCE_SCHEMA)
    data = _load_yaml(args.base_data_template, schema=DATA_SCHEMA)
    collection_path = (
        args.collection_json
        if args.collection_json is not None
        else args.output_source_template.parent / "oxe_collection.json"
    )
    if args.collection_json is None:
        _download_json(args.collection_url, collection_path)
    repos = _repo_ids(collection_path, collection_url=args.collection_url)
    metadata_root = (
        args.metadata_root
        if args.metadata_root is not None
        else args.output_source_template.parent / "oxe_metadata"
    )
    if metadata_root.exists() or metadata_root.is_symlink():
        if metadata_root.is_symlink() or not metadata_root.is_dir():
            raise ValueError(f"metadata root must be a real directory: {metadata_root}")
    else:
        metadata_root.mkdir(parents=True, exist_ok=False)
    metadata = {
        repo: (
            _metadata(repo, metadata_root=metadata_root)
            if (metadata_root / f"{_slug(repo)}.json").exists()
            else _download_json(
                f"https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json",
                metadata_root / f"{_slug(repo)}.json",
            )
        )
        for repo in repos
        if repo != "lerobot/droid_1.0.1"
    }
    source_output, data_output, adapters = build_templates(
        base_source=source,
        base_data=data,
        repo_ids=repos,
        metadata_by_repo=metadata,
    )
    _publish(
        args.output_source_template,
        yaml.safe_dump(source_output, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )
    _publish(
        args.output_data_template,
        yaml.safe_dump(data_output, sort_keys=False, allow_unicode=True).encode("utf-8"),
    )
    for name, adapter in adapters.items():
        _publish(
            args.output_adapter_root / f"{name}.yaml",
            yaml.safe_dump(adapter, sort_keys=False, allow_unicode=True).encode("utf-8"),
        )
    print(
        json.dumps(
            {
                "official_oxe_dataset_count": len(repos),
                "new_oxe_source_count": len(repos) - 1,
                "source_template": str(args.output_source_template.absolute()),
                "data_template": str(args.output_data_template.absolute()),
                "adapter_root": str(args.output_adapter_root.absolute()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
