#!/usr/bin/env python3
"""Compile an operator-edited YAML inventory into canonical strict JSON."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from wm3d_v3.data.scale5b_contracts import (
    DatasetContract,
    atomic_write_json,
    resolve_regular_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-non-100-cycle",
        action="store_true",
        help="Permit source weights whose integer cycle does not total 100.",
    )
    return parser.parse_args()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"unresolved environment variable {value!r}")
        return expanded
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    inventory = resolve_regular_file(
        args.inventory.parent,
        args.inventory.name,
    )
    raw = yaml.safe_load(inventory.read_text(encoding="utf-8"))
    raw = _expand(raw)
    contract = DatasetContract.from_mapping(raw)
    weight_total = sum(contract.source_weights.values())
    if weight_total != 100 and not args.allow_non_100_cycle:
        raise ValueError(f"formal source sampling weights total {weight_total}, not 100")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, contract.as_dict(), exclusive=True)
    print(
        json.dumps(
            {
                "pass": True,
                "output": str(output),
                "dataset_contract_sha256": contract.sha256,
                "source_weights": contract.source_weights,
                "nominal_hours": sum(source.nominal_hours for source in contract.sources),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
