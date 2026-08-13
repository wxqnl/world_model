"""Real CUDA proof that the published 5B profile materializes as FSDP2 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from wm3d.models.model_factory import build_world_model
from wm3d.training.distributed_runtime import (
    DistributedStrategyConfig,
    destroy_distributed,
    initialize_distributed,
    wrap_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strategy = DistributedStrategyConfig(
        strategy="fsdp2",
        initialization="meta_sharded",
        shard_degree=dist_world_size_from_env(),
        param_dtype="bf16",
        reduce_dtype="fp32",
        output_dtype="bf16",
    )
    context = initialize_distributed(strategy)
    try:
        profile = yaml.safe_load(args.model_profile.read_text(encoding="utf-8"))
        with torch.device("meta"):
            raw = build_world_model(profile)
        global_numel = sum(parameter.numel() for parameter in raw.parameters())
        if any(not parameter.is_meta for parameter in raw.parameters()):
            raise RuntimeError("5B construction allocated a non-meta parameter")
        torch.cuda.reset_peak_memory_stats(context.device)
        wrapped = wrap_model(raw, context, strategy, initialization_seed=3407)
        local_numel = sum(
            parameter.to_local().numel()
            if hasattr(parameter, "to_local")
            else parameter.numel()
            for parameter in wrapped.model.parameters()
        )
        evidence = {
            "rank": context.rank,
            "world_size": context.world_size,
            "global_parameter_numel": global_numel,
            "local_parameter_storage_numel": local_numel,
            "local_fraction": local_numel / global_numel,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(context.device),
            "all_parameters_materialized": all(
                not parameter.is_meta for parameter in wrapped.model.parameters()
            ),
        }
        gathered: list[object] = [None] * context.world_size
        dist.all_gather_object(gathered, evidence)
        if context.is_rank0:
            args.output.write_text(
                json.dumps(gathered, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(gathered, sort_keys=True), flush=True)
    finally:
        destroy_distributed()


def dist_world_size_from_env() -> int:
    import os

    return int(os.environ["WORLD_SIZE"])


if __name__ == "__main__":
    main()
