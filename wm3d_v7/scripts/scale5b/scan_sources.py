#!/usr/bin/env python3
"""Create the immutable vendor-neutral episode plan for V7 native 5B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wm3d_v3.data.scale5b_contracts import (
    atomic_write_json,
    load_contract,
    resolve_real_directory,
    resolve_regular_file,
    sha256_file,
)
from wm3d_v3.data.scale5b_sources import (
    SourceLayout,
    publish_scan_receipt,
    scan_lerobot,
    scan_lerobot_collection,
    scan_normalized_manifest,
    validate_collection_receipt,
    validate_episode_inputs,
    write_episode_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--source-layouts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_input = args.dataset_contract
    contract_path = resolve_regular_file(
        contract_input.parent,
        contract_input.name,
    )
    layout_input = args.source_layouts
    layout_path = resolve_regular_file(
        layout_input.parent,
        layout_input.name,
    )
    contract = load_contract(contract_path)
    raw_layouts = json.loads(layout_path.read_text(encoding="utf-8"))
    if raw_layouts.get("schema") != "wm3d_v7_native5b_source_layouts_v1":
        raise ValueError("source-layout collection schema mismatch")
    layouts = tuple(
        SourceLayout.from_mapping(item) for item in raw_layouts.get("layouts", ())
    )
    by_source = {layout.source: layout for layout in layouts}
    if len(by_source) != len(layouts) or tuple(by_source) != contract.source_order:
        raise ValueError("source layouts must uniquely follow contract source_order")
    embodiments = {item.name: item for item in contract.embodiments}
    episodes = []
    collection_receipts = {}
    for source in contract.sources:
        layout = by_source[source.name]
        if layout.adapter != source.adapter:
            raise ValueError(f"{source.name}: layout/contract adapter mismatch")
        if layout.embodiment not in source.embodiment_names:
            raise ValueError(f"{source.name}: layout embodiment is not allowed")
        embodiment = embodiments[layout.embodiment]
        if tuple(layout.view_keys) != embodiment.views:
            raise ValueError(f"{source.name}: canonical view order mismatch")
        if tuple(item.group_name for item in layout.action_columns) != tuple(
            item.name for item in embodiment.action_groups
        ):
            raise ValueError(f"{source.name}: action group order mismatch")
        for mapping, group in zip(
            layout.action_columns,
            embodiment.action_groups,
            strict=True,
        ):
            if len(mapping.indices) != len(group.dimensions):
                raise ValueError(
                    f"{source.name}: action width for {group.name} differs "
                    "from embodiment contract"
                )
            expected_discrete = (
                "grip" in group.control_mode or "discrete" in group.control_mode
            )
            if bool(mapping.discrete) != expected_discrete:
                raise ValueError(
                    f"{source.name}: discrete semantics for {group.name} "
                    "differ from embodiment contract"
                )
        if tuple(item.modality_name for item in layout.auxiliary_columns) != tuple(
            item.name for item in embodiment.auxiliary_modalities
        ):
            raise ValueError(f"{source.name}: auxiliary modality order mismatch")
        for mapping, modality in zip(
            layout.auxiliary_columns,
            embodiment.auxiliary_modalities,
            strict=True,
        ):
            if len(mapping.indices) != len(modality.dimensions):
                raise ValueError(
                    f"{source.name}: auxiliary width for {modality.name} "
                    "differs from embodiment contract"
                )
        root = resolve_real_directory(
            Path(source.raw_root),
            f"{source.name} raw root",
        )
        if layout.adapter == "lerobot":
            source_episodes = scan_lerobot(
                root,
                layout,
                split_seed=source.split_seed,
                train_fraction=source.train_fraction,
            )
        elif layout.adapter == "lerobot_collection":
            source_episodes = scan_lerobot_collection(
                root,
                layout,
                split_seed=source.split_seed,
                train_fraction=source.train_fraction,
            )
            collection_receipt = validate_collection_receipt(root, layout)
            if collection_receipt is not None:
                collection_receipts[source.name] = {
                    "path": layout.collection_receipt_path,
                    "schema": layout.collection_receipt_schema,
                    "sha256": sha256_file(collection_receipt),
                }
        else:
            manifest = resolve_regular_file(
                root,
                str(layout.normalized_manifest_path),
            )
            source_episodes = scan_normalized_manifest(
                manifest,
                layout,
                split_seed=source.split_seed,
                train_fraction=source.train_fraction,
                raw_root=root,
            )
        minimum_seconds = (contract.T + contract.K) / contract.feature_fps
        source_episodes = [
            item for item in source_episodes if item.duration_seconds >= minimum_seconds
        ]
        split_names = {item.split for item in source_episodes}
        if not {"train", "val"}.issubset(split_names):
            raise ValueError(
                f"{source.name}: deterministic split must contain train and val"
            )
        episodes.extend(source_episodes)

    input_validation = validate_episode_inputs(episodes)
    output_input = args.output_root
    if output_input.exists() or output_input.is_symlink():
        output = resolve_real_directory(output_input, "dataset output root")
        if any(output.iterdir()):
            raise FileExistsError(f"dataset output root must be empty: {output}")
    else:
        parent = resolve_real_directory(
            output_input.parent,
            "dataset output parent",
        )
        output = parent / output_input.name
        output.mkdir(mode=0o750)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    control = output / "control"
    receipts = output / "receipts"
    control.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    contract_out = control / "dataset_contract.json"
    layouts_out = control / "source_layouts.json"
    plan_out = control / "episode_plan.jsonl"
    atomic_write_json(contract_out, contract.as_dict(), exclusive=True)
    atomic_write_json(layouts_out, raw_layouts, exclusive=True)
    summary = write_episode_plan(plan_out, episodes)
    summary["input_validation"] = input_validation
    summary["collection_receipts"] = collection_receipts
    receipt = publish_scan_receipt(
        receipts / "source_scan.json",
        layout_path=layouts_out,
        plan_path=plan_out,
        summary=summary,
    )
    print(
        json.dumps(
            {
                "pass": True,
                "contract_sha256": contract.sha256,
                "summary": summary,
                "scan_receipt": receipt,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
