#!/usr/bin/env python3
"""Bounded 24-rank NCCL/IB validation for the repaired WM3D-V7 run."""

from __future__ import annotations

import datetime
import json
import os
import socket
import time

import torch
import torch.distributed as dist


EXPECTED_HCAS = {
    0: "mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10",
    1: "mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10",
    2: "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8",
}


def main() -> None:
    node_rank = int(os.environ["WM3D_NODE_RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("NCCL_IB_DISABLE") != "0":
        raise RuntimeError("NCCL_IB_DISABLE must be exactly 0")
    if os.environ.get("NCCL_NET") != "IB":
        raise RuntimeError("NCCL_NET must be exactly IB")
    if os.environ.get("NCCL_IB_HCA") != EXPECTED_HCAS[node_rank]:
        raise RuntimeError(
            f"unexpected HCA allowlist for node rank {node_rank}: "
            f"{os.environ.get('NCCL_IB_HCA')!r}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=240),
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 24:
        raise RuntimeError(f"expected world size 24, got {world}")

    expected = world * (world + 1) / 2
    started = time.monotonic()
    for numel in (1, 262_144, 4_194_304):
        for _ in range(4):
            value = torch.full(
                (numel,),
                float(rank + 1),
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(device)
            if float(value[0].item()) != expected or float(value[-1].item()) != expected:
                raise RuntimeError(
                    f"all-reduce mismatch rank={rank} numel={numel} "
                    f"first={float(value[0].item())} last={float(value[-1].item())} "
                    f"expected={expected}"
                )
    dist.barrier()
    payload = {
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "hca": os.environ["NCCL_IB_HCA"],
        "host": socket.gethostname(),
        "local_rank": local_rank,
        "net": os.environ["NCCL_NET"],
        "node_rank": node_rank,
        "rank": rank,
        "schema": "wm3d_v7_1b_ib_allreduce_smoke_v1",
        "world_size": world,
    }
    print("WM3D_V7_IB_ALLREDUCE_OK " + json.dumps(payload, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
