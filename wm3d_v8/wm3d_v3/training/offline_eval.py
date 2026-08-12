"""Read-only offline evaluation for unified WM3D V8 DCP checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
import torch.distributed as dist

from wm3d_v3.data.source_adapters import load_adapter_contract
from wm3d_v3.models.model_factory import build_world_model
from wm3d_v3.models.native_world_model import NativeWorldModel
from wm3d_v3.training.distributed_checkpoint import (
    DistributedCheckpointManager,
    ResumeExpectations,
    sha256_file,
)
from wm3d_v3.training.distributed_runtime import (
    autocast_context,
    destroy_distributed,
    initialize_distributed,
    reduce_metrics,
    strategy_from_mapping,
    wrap_model,
)
from wm3d_v3.training.native_objective import (
    compute_native_objective,
    objective_config_from_mapping,
)
from wm3d_v3.training.pretrain import (
    _atomic_json_no_clobber,
    _batch_to_device,
    _build_mixed_dataset,
    _configure_reproducibility,
    _forward,
    _make_loader,
    _require_recent_resource_preflight,
    _publish_and_validate_launch,
    _run_contract,
    _topology_contract_sha256,
)
from wm3d_v3.training.launch_qualification import (
    LaunchQualificationError,
    verify_clean_runtime_checkout,
)
from wm3d_v3.training.runtime_contract import load_materialized_runtime


EVAL_RECEIPT_SCHEMA = "wm3d_v8_unified_offline_eval_v2"
_COVERAGE_WEIGHTS = {
    "native_token_supervised_elements": ("token_mse", "token_cosine"),
    "rgb_supervised_elements": ("rgb_charbonnier", "rgb_gradient"),
    "depth_supervised_elements": ("depth_log",),
    "point_supervised_elements": ("point",),
    "camera_pose_supervised_elements": ("camera_pose",),
    "fine_supervised_dimensions": ("action_fine",),
    "coarse_supervised_dimensions": ("action_coarse",),
}
_KNOWN_COVERAGE_LANES = frozenset(_COVERAGE_WEIGHTS) | {
    "current_state_supervised_dimensions",
    "fine_continuous_supervised_dimensions",
    "fine_binary_supervised_dimensions",
}


class OfflineEvalError(RuntimeError):
    pass


def declared_eval_coverage_lanes(
    profile: Any,
    objective: Any,
    *,
    active_source_names: frozenset[str],
) -> frozenset[str]:
    """Intersect objective lanes with supervision declared by the sealed profile."""

    declared_sources = {source.name for source in profile.sources}
    if not active_source_names or not active_source_names <= declared_sources:
        raise OfflineEvalError(
            "eval active-source closure is empty or differs from sealed profile"
        )
    cache = profile.cache
    lanes = {
        "native_token_supervised_elements",
        "current_state_supervised_dimensions",
    }
    for key, lane in (
        ("rgb_codec", "rgb_supervised_elements"),
        ("depth_codec", "depth_supervised_elements"),
        ("point_codec", "point_supervised_elements"),
        ("camera_pose_codec", "camera_pose_supervised_elements"),
    ):
        if cache.get(key) is not None:
            lanes.add(lane)
    fine_continuous = False
    fine_binary = False
    declared_action: set[str] = set()
    for source in profile.sources:
        if source.name not in active_source_names:
            continue
        adapter = load_adapter_contract(
            source.adapter_config_path,
            expected_sha256=source.adapter_contract_sha256,
        )
        embodiment = profile.embodiments[source.embodiment]
        groups = {group.name: group for group in embodiment.groups}
        for mapping in adapter.groups:
            declared_action.add(mapping.supervision)
            if mapping.supervision != "fine_command":
                continue
            semantics = groups[mapping.group].action_semantics
            for semantic in semantics:
                if semantic in {
                    "absolute_gripper_open01",
                    "absolute_gripper_close01",
                    "binary_contact",
                    "controller_mode",
                }:
                    fine_binary = True
                else:
                    fine_continuous = True
    if "fine_command" in declared_action:
        lanes.add("fine_supervised_dimensions")
        if fine_continuous:
            lanes.add("fine_continuous_supervised_dimensions")
        if fine_binary:
            lanes.add("fine_binary_supervised_dimensions")
    if "coarse_effect" in declared_action:
        lanes.add("coarse_supervised_dimensions")
    return frozenset(
        lane
        for lane in lanes
        if lane == "current_state_supervised_dimensions"
        or lane in {
            "fine_continuous_supervised_dimensions",
            "fine_binary_supervised_dimensions",
        }
        or any(
            float(getattr(objective, weight)) > 0.0
            for weight in _COVERAGE_WEIGHTS[lane]
        )
    )


def validate_eval_coverage(
    metrics: dict[str, float], *, expected_lanes: frozenset[str]
) -> dict[str, float]:
    unknown = set(expected_lanes) - _KNOWN_COVERAGE_LANES
    if unknown:
        raise OfflineEvalError(f"unknown expected coverage lanes: {sorted(unknown)}")
    coverage: dict[str, float] = {}
    for count_name in _COVERAGE_WEIGHTS:
        count = float(metrics.get(count_name, 0.0))
        if count_name in expected_lanes and (not math.isfinite(count) or count <= 0.0):
            raise OfflineEvalError(
                f"offline eval has zero coverage for declared lane {count_name}"
            )
        coverage[count_name] = count
    current_state = float(metrics.get("current_state_supervised_dimensions", 0.0))
    if "current_state_supervised_dimensions" in expected_lanes and (
        not math.isfinite(current_state) or current_state <= 0.0
    ):
        raise OfflineEvalError("offline eval has zero current-state coverage")
    coverage["current_state_supervised_dimensions"] = current_state
    for lane in (
        "fine_continuous_supervised_dimensions",
        "fine_binary_supervised_dimensions",
    ):
        value = float(metrics.get(lane, 0.0))
        if lane in expected_lanes and (not math.isfinite(value) or value <= 0.0):
            raise OfflineEvalError(f"offline eval has zero coverage for declared lane {lane}")
        coverage[lane] = value
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, runtime_sha = load_materialized_runtime(args.runtime)
    runtime = config["runtime_profile"]
    strategy = strategy_from_mapping(runtime["distributed"])
    context = initialize_distributed(strategy)
    try:
        if int(runtime["expected_world_size"]) != context.world_size:
            raise OfflineEvalError("evaluation world size differs from sealed runtime")
        resource_preflight = _require_recent_resource_preflight(
            config, runtime_sha, context
        )
        if re.fullmatch(r"step_[0-9]{8}", args.checkpoint.name) is None:
            raise OfflineEvalError("checkpoint must be an explicit step_XXXXXXXX directory")
        checkpoint_step = int(args.checkpoint.name.split("_")[1])
        validation_steps = int(runtime["train"]["validation_steps"])
        if validation_steps <= 0:
            raise OfflineEvalError("validation steps must be positive")
        repo = Path(__file__).resolve().parents[2]
        try:
            current_commit = verify_clean_runtime_checkout(
                repo, str(config["run"]["code_commit"])
            )
        except LaunchQualificationError as exc:
            raise OfflineEvalError("runtime code provenance failed") from exc

        seed = int(runtime["train"]["validation_seed"])
        _configure_reproducibility(seed)
        construction_device = (
            torch.device("meta")
            if strategy.initialization == "meta_sharded"
            else context.device
        )
        with torch.device(construction_device):
            raw_model = build_world_model(config["model_profile"])
        if not isinstance(raw_model, NativeWorldModel):
            raise OfflineEvalError("offline eval requires native_world_model")
        parameter_counts = raw_model.parameter_counts()
        wrapped = wrap_model(
            raw_model,
            context,
            strategy,
            initialization_seed=(
                int(runtime["train"]["seed"])
                if strategy.initialization == "meta_sharded"
                else None
            ),
        )
        model = wrapped.model
        manager = DistributedCheckpointManager(args.checkpoint.parent)
        expectations = ResumeExpectations(
            step=checkpoint_step,
            run_lineage=config["run"]["lineage"],
            runtime_config_sha256=runtime_sha,
            data_closure_sha256=config["bindings"]["data_closure_sha256"],
            model_contract_sha256=config["bindings"]["model_contract_sha256"],
            world_size=context.world_size,
            shard_degree=int(runtime["distributed"]["shard_degree"]),
            distributed_strategy=strategy.strategy,
            global_batch_size=int(runtime["train"]["global_batch_size"]),
            topology_contract_sha256=_topology_contract_sha256(config),
            allow_topology_reshard=False,
        )
        inspection: list[Any] = [None]
        if context.is_rank0:
            try:
                source = manager.inspect_committed(
                    path=args.checkpoint, expected=expectations
                )
                if source["resume_mode"] != "exact":
                    raise OfflineEvalError("offline eval requires exact topology")
                inspection[0] = {"ok": True, "source": source}
            except Exception as exc:
                inspection[0] = {
                    "ok": False,
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(inspection, src=0)
        if not inspection[0]["ok"]:
            raise OfflineEvalError(f"checkpoint inspection failed: {inspection[0]}")
        run_contract = _run_contract(config, parameter_counts, raw_model)
        run_contract_path = Path(config["run"]["output_root"]) / "run_contract.json"
        if not run_contract_path.is_file() or json.loads(
            run_contract_path.read_text(encoding="utf-8")
        ) != run_contract:
            raise OfflineEvalError("stable run contract is missing or differs")
        launch_qualification_path, launch_qualification_sha256 = (
            _publish_and_validate_launch(
                config=config,
                config_sha=runtime_sha,
                context=context,
                strategy=strategy,
                run_contract=run_contract,
                resource_preflight=resource_preflight,
                source_checkpoint=inspection[0]["source"],
                launch_kind="eval",
            )
        )
        metadata = manager.load_model_for_evaluation(
            path=args.checkpoint,
            model=model,
            expected=expectations,
        )
        dataset, profile = _build_mixed_dataset(config, split="val")
        loader = _make_loader(
            dataset,
            profile,
            runtime,
            rank=context.rank,
            world_size=context.world_size,
            start_step=0,
            num_steps=validation_steps,
            seed=seed,
            gradient_accumulation=1,
        )
        objective = objective_config_from_mapping(
            config["objective_profile"]["objective"]
        )
        totals: dict[str, torch.Tensor] = {}
        model.eval()
        with torch.no_grad():
            for cpu_batch in loader:
                batch = _batch_to_device(cpu_batch, context.device)
                with autocast_context(strategy):
                    losses = compute_native_objective(
                        output=_forward(model, batch), batch=batch, config=objective
                    )
                for name, value in losses.items():
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"non-finite eval metric {name}")
                    totals[name] = totals.get(name, torch.zeros_like(value)) + value
        metrics = reduce_metrics(
            {name: value / validation_steps for name, value in totals.items()}
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("offline eval receipt contains non-finite metrics")
        expected_coverage_lanes = declared_eval_coverage_lanes(
            profile,
            objective,
            active_source_names=frozenset(dataset.source_names),
        )
        coverage = validate_eval_coverage(
            metrics, expected_lanes=expected_coverage_lanes
        )
        publication: list[Any] = [None]
        if context.is_rank0:
            try:
                commit_path = args.checkpoint / "COMMITTED.json"
                manifest_path = args.checkpoint / "MANIFEST.json"
                commit = json.loads(commit_path.read_text(encoding="utf-8"))
                receipt = {
                    "schema": EVAL_RECEIPT_SCHEMA,
                    "runtime_path": str(args.runtime.resolve(strict=True)),
                    "runtime_sha256": runtime_sha,
                    "checkpoint_path": str(args.checkpoint.resolve(strict=True)),
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_committed_sha256": sha256_file(commit_path),
                    "checkpoint_manifest_sha256": sha256_file(manifest_path),
                    "checkpoint_manifest_content_sha256": commit[
                        "manifest_content_sha256"
                    ],
                    "data_closure_sha256": config["bindings"][
                        "data_closure_sha256"
                    ],
                    # The sealed index contains every split.  Validation rows
                    # are selected deterministically from it by split="val";
                    # do not mislabel the full-index digest as a val-only file.
                    "cache_window_index_sha256": config["data_closure"][
                        "cache_index_sha256"
                    ],
                    "evaluated_split": "val",
                    "validation_seed": seed,
                    "validation_steps": validation_steps,
                    "world_size": context.world_size,
                    "launch_qualification_path": launch_qualification_path,
                    "launch_qualification_sha256": launch_qualification_sha256,
                    "code_commit": current_commit,
                    "metrics": metrics,
                    "coverage": coverage,
                    "expected_coverage_lanes": sorted(expected_coverage_lanes),
                    "all_metrics_finite": True,
                    "checkpoint_metadata": {
                        "run_lineage": metadata["run_lineage"],
                        "runtime_config_sha256": metadata["runtime_config_sha256"],
                        "model_contract_sha256": metadata[
                            "model_contract_sha256"
                        ],
                    },
                }
                _atomic_json_no_clobber(args.output, receipt)
                publication[0] = {"ok": True, "receipt": receipt}
            except Exception as exc:
                publication[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(publication, src=0)
        if not publication[0]["ok"]:
            raise OfflineEvalError(f"eval receipt publication failed: {publication[0]}")
        if context.is_rank0:
            print(json.dumps(publication[0]["receipt"], sort_keys=True), flush=True)
        dist.barrier()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
