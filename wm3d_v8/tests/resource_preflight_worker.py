"""Real distributed resource-preflight probe used by release validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from wm3d_v3.training.resource_preflight import run_resource_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        context = SimpleNamespace(
            rank=dist.get_rank(),
            local_rank=local_rank,
            local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            world_size=dist.get_world_size(),
            device=torch.device("cuda", local_rank),
        )
        resources = {
            "gpu_name_substring": "H100",
            "minimum_gpu_memory_mib": 80000,
            "require_zero_uncorrected_ecc": True,
            "require_idle_gpu": True,
            "require_full_local_nvlink_clique": True,
            "minimum_ib_rate_gbps": 100.0,
            "forbid_nccl_ib_disable": True,
            "minimum_memlock_bytes": 1,
            "minimum_nofile": 1024,
            "minimum_shm_bytes": 1,
            "minimum_data_free_bytes": 1,
            "minimum_output_free_bytes": 1,
            "minimum_allreduce_gbps": 1.0,
            "maximum_preflight_age_seconds": 300,
        }
        receipt = run_resource_preflight(
            resources=resources,
            context=context,
            runtime_config_sha256="a" * 64,
            cache_root=Path("/data/Minko"),
            output_root=args.output,
        )
        if context.rank == 0:
            print(json.dumps(receipt, sort_keys=True), flush=True)
        if receipt["passed"] is not True:
            raise RuntimeError(f"real resource preflight failed: {receipt['errors']}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
