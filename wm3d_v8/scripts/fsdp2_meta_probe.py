#!/usr/bin/env python3
"""Tiny real-CUDA proof for FSDP2 meta initialization; not a trainer."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import _random as dtensor_random


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    with torch.device("meta"):
        model = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.Sequential(nn.Linear(64, 64, bias=False)),
            nn.Linear(64, 8, bias=False),
        )
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("shard",))
    fully_shard(model[1], mesh=mesh)
    fully_shard(model, mesh=mesh)
    dtensor_random.manual_seed(3407, mesh)
    for module in model.modules():
        state = list(module.parameters(recurse=False)) + list(
            module.buffers(recurse=False)
        )
        if state and any(value.is_meta for value in state):
            module.to_empty(device=device, recurse=False)
            module.reset_parameters()
    assert all(not parameter.is_meta for parameter in model.parameters())
    full_weight = model[0].weight.full_tensor()
    if torch.equal(full_weight[:32], full_weight[32:]):
        raise RuntimeError("FSDP2 global initialization repeated a local shard")
    gathered_weight = [torch.empty_like(full_weight) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_weight, full_weight)
    if any(not torch.equal(full_weight, item) for item in gathered_weight):
        raise RuntimeError("FSDP2 ranks disagree on the reconstructed global tensor")
    value = model(torch.randn(4, 32, device=device)).square().mean()
    value.backward()
    print(
        rank,
        type(next(model.parameters())).__name__,
        tuple(next(model.parameters()).shape),
        float(value),
        int(torch.cuda.max_memory_allocated()),
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
