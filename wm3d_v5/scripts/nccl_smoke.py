from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    backend = os.environ.get("WM3D_DDP_BACKEND", "nccl")
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if backend == "nccl" else "cpu"
    x = torch.ones((), device=device) * (rank + 1)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"{backend}_smoke_ok world={world} sum={float(x.item()):.1f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
