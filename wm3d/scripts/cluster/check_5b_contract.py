#!/usr/bin/env python3
"""Check the deployed 5B recipe; cluster preflight still owns execution readiness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from wm3d.models.model_factory import build_world_model, validate_model_profile
from wm3d.training.runtime_contract import validate_runtime_profile

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def validate_contract(
    model: dict[str, Any],
    encoder: dict[str, Any],
    objective: dict[str, Any],
    runtime_profile: dict[str, Any],
) -> None:
    expected_model = load_yaml(ROOT / "configs/model/native_5b_v8_native_direct_rgb.yaml")
    expected_encoder = load_yaml(ROOT / "configs/encoder/vggt_native_p144.yaml")
    expected_objective = load_yaml(ROOT / "configs/objective/stage0_v8_native_direct_rgb.yaml")
    validate_model_profile(model)
    validate_runtime_profile(runtime_profile)
    if model != expected_model:
        raise ValueError("5B requires the current native-direct model profile; old transport/P256 profiles are not supported by this launcher")
    if encoder != expected_encoder:
        raise ValueError("5B requires the P144/384px encoder contract without absolute P256")
    if objective != expected_objective:
        raise ValueError("5B objective differs from the current shared native-direct objective")
    train = runtime_profile["train"]
    if any(train.get(k, 0.0) != 0.0 for k in (
        "appearance_teacher_start_ratio", "appearance_teacher_end_ratio"
    )):
        raise ValueError("native-direct RGB cannot enable appearance teacher forcing")
    if train.get("appearance_validation_three_way", False):
        raise ValueError("native-direct RGB cannot enable appearance three-way evaluation")


def validate_sealed_recipe(
    runtime: dict[str, Any],
    model: dict[str, Any],
    objective: dict[str, Any],
    runtime_profile: dict[str, Any],
) -> None:
    # The production loader/preflight additionally verifies data, code, environment
    # and checkpoint bindings. This guard rejects a stale recipe before torchrun.
    for key, expected in (
        ("model_profile", model),
        ("objective_profile", objective),
        ("runtime_profile", runtime_profile),
    ):
        if runtime.get(key) != expected:
            raise ValueError(f"sealed {key} differs from the selected 5B recipe; materialize a fresh runtime, do not edit or reuse an old run")


def validate_built_model(built: torch.nn.Module) -> None:
    factual_action = getattr(built, "factual_action", None)
    if factual_action is None or factual_action.direct_action_space != "physical":
        raise ValueError("5B factual direct action must use physical units")
    for name in ("direct_fine_value", "direct_fine_aggregate", "direct_coarse_value"):
        if type(getattr(factual_action, name, None)) is not torch.nn.Linear:
            raise ValueError(f"5B {name} must be a live Linear projection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    model, encoder, objective, profile = (
        load_yaml(path) for path in (
            args.model, args.encoder, args.objective, args.runtime_profile
        )
    )
    validate_contract(model, encoder, objective, profile)
    if args.runtime is not None:
        validate_sealed_recipe(load_yaml(args.runtime), model, objective, profile)
    # Construction is metadata-only. It is deliberately not labelled a GPU test.
    with torch.device("meta"):
        built = build_world_model(model)
    validate_built_model(built)
    print(json.dumps({
        "recipe": "native_direct_5b",
        "parameters": sum(p.numel() for p in built.parameters()),
        "sealed_recipe_matches": args.runtime is not None,
        "gpu_qualification": "NOT_TESTED_BY_THIS_COMMAND",
    }))


if __name__ == "__main__":
    main()
