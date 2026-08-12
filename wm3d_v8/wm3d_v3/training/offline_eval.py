"""Read-only offline evaluation for unified WM3D V8 DCP checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

import torch
import torch.distributed as dist

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
    _topology_contract_sha256,
)
from wm3d_v3.training.runtime_contract import load_materialized_runtime


EVAL_RECEIPT_SCHEMA = "wm3d_v8_unified_offline_eval_v2"


class OfflineEvalError(RuntimeError):
    pass


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
        if re.fullmatch(r"step_[0-9]{8}", args.checkpoint.name) is None:
            raise OfflineEvalError("checkpoint must be an explicit step_XXXXXXXX directory")
        checkpoint_step = int(args.checkpoint.name.split("_")[1])
        validation_steps = int(runtime["train"]["validation_steps"])
        if validation_steps <= 0:
            raise OfflineEvalError("validation steps must be positive")
        repo = Path(__file__).resolve().parents[2]
        current_commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        if current_commit != config["run"]["code_commit"]:
            raise OfflineEvalError("runtime code commit does not match current checkout")

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
        metadata = manager.load_model_for_evaluation(
            path=args.checkpoint,
            model=model,
            expected=ResumeExpectations(
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
            ),
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
                    "code_commit": current_commit,
                    "metrics": metrics,
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
