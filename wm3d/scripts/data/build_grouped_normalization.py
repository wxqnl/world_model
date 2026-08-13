#!/usr/bin/env python3
"""Build the immutable train-only grouped robot normalization artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

import yaml

from wm3d.data.grouped_normalization import build_grouped_normalization_artifact
from wm3d.data.manifest_contract import canonical_sha256, load_data_profile, sha256_file


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
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--window-index", type=Path, required=True)
    parser.add_argument("--window-index-sha256", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--minimum-scale", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = load_data_profile(args.data_profile, verify_source_manifests=True)
    model_path = args.model_profile.resolve(strict=True)
    model_profile = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    if not isinstance(model_profile, dict):
        raise ValueError("model profile root must be a mapping")
    artifact = build_grouped_normalization_artifact(
        data_profile=profile,
        model_profile=model_profile,
        model_profile_sha256=canonical_sha256(model_profile),
        window_index_path=args.window_index,
        window_index_sha256=args.window_index_sha256,
        cache_root=args.cache_root,
        minimum_scale=args.minimum_scale,
    )
    payload = (json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    output = args.output.absolute()
    _publish_no_clobber(output, payload)
    print(
        json.dumps(
            {
                "schema": artifact["schema"],
                "output": str(output),
                "sha256": sha256_file(output),
                "rows_sha256": artifact["rows_sha256"],
                "train_window_count_by_source": artifact[
                    "train_window_count_by_source"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
