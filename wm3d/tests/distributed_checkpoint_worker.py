"""Real torchrun worker for FSDP2/DCP integration tests.

This file is executed explicitly and is not a mock: it uses CUDA, NCCL,
FSDP2 DTensors, AdamW state, and PyTorch Distributed Checkpoint across fresh
process boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from wm3d.training.distributed_checkpoint import (
    DistributedCheckpointManager,
    ResumeExpectations,
)
from wm3d.training.distributed_runtime import (
    DistributedStrategyConfig,
    destroy_distributed,
    initialize_adamw_state,
    initialize_distributed,
    wrap_model,
)


LINEAGE = "wm3d-dcp-real-integration"
RUNTIME_SHA = "1" * 64
DATA_SHA = "2" * 64
MODEL_SHA = "3" * 64
TOPOLOGY_SHA = "4" * 64
SEED = 9371


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(torch.nn.functional.gelu(self.linear(value)))


class RealFSDPModel(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [checkpoint_wrapper(Block(width)), checkpoint_wrapper(Block(width))]
        )
        self.output = nn.Linear(width, 8)

    def iter_fsdp_units(self):
        return iter(self.blocks)

    def iter_activation_checkpoint_units(self):
        return iter(self.blocks)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return self.output(value)


def _step(model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    value = torch.arange(256, device="cuda", dtype=torch.float32).reshape(4, 64)
    target = torch.arange(32, device="cuda", dtype=torch.float32).reshape(4, 8) / 32
    optimizer.zero_grad(set_to_none=True)
    output = model(value)
    loss = torch.nn.functional.mse_loss(output.float(), target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _draw() -> dict[str, float]:
    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "torch_cpu": float(torch.rand(())),
        "torch_cuda": float(torch.rand((), device="cuda")),
    }


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("save", "exact", "reshard", "eval"), required=True
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard-degree", type=int, required=True)
    parser.add_argument("--strategy", choices=("fsdp2", "ddp"), default="fsdp2")
    args = parser.parse_args()

    if args.mode == "reshard" and args.strategy != "fsdp2":
        raise ValueError("topology reshard is FSDP2-only")
    strategy = DistributedStrategyConfig(
        strategy=args.strategy,
        shard_degree=args.shard_degree if args.strategy == "fsdp2" else 1,
        initialization="meta_sharded" if args.strategy == "fsdp2" else "direct",
        param_dtype="bf16",
        reduce_dtype="fp32",
        output_dtype="bf16",
    )
    context = initialize_distributed(strategy)
    try:
        random.seed(SEED + context.rank)
        np.random.seed(SEED + context.rank)
        torch.manual_seed(SEED + context.rank)
        torch.cuda.manual_seed(SEED + context.rank)
        construction_device = (
            torch.device("meta")
            if args.strategy == "fsdp2"
            else context.device
        )
        with torch.device(construction_device):
            raw_model = RealFSDPModel()
        wrapped = wrap_model(
            raw_model,
            context,
            strategy,
            initialization_seed=SEED if args.strategy == "fsdp2" else None,
        )
        model = wrapped.model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        initialize_adamw_state(optimizer)
        manager = DistributedCheckpointManager(args.root / "checkpoints")
        checkpoint = args.root / "checkpoints" / "step_00000001"

        common_metadata = {
            "run_name": "real-dcp-integration",
            "run_lineage": LINEAGE,
            "runtime_config_sha256": RUNTIME_SHA,
            "data_closure_sha256": DATA_SHA,
            "model_contract_sha256": MODEL_SHA,
            "shard_degree": strategy.shard_degree,
            "distributed_strategy": args.strategy,
            "global_batch_size": 8,
            "topology_contract_sha256": TOPOLOGY_SHA,
            "sampler_progress": {"next_optimizer_step": 1},
            "initial_seed": SEED,
        }
        if args.mode == "save":
            _step(model, optimizer)
            manager.save(
                step=1,
                model=model,
                optimizer=optimizer,
                metadata=common_metadata,
                rank_state={"next_optimizer_step": 1},
            )
            value = torch.arange(
                256, device="cuda", dtype=torch.float32
            ).reshape(4, 64)
            checkpoint_output = model(value).detach().float().cpu().tolist()
            _write(
                args.root / "checkpoint_output" / f"rank_{context.rank:05d}.json",
                checkpoint_output,
            )
            golden = {"rng": _draw(), "loss": _step(model, optimizer)}
            _write(args.root / "golden" / f"rank_{context.rank:05d}.json", golden)
        elif args.mode == "eval":
            metadata = manager.load_model_for_evaluation(
                path=checkpoint,
                model=model,
                expected=ResumeExpectations(
                    step=1,
                    run_lineage=LINEAGE,
                    runtime_config_sha256=RUNTIME_SHA,
                    data_closure_sha256=DATA_SHA,
                    model_contract_sha256=MODEL_SHA,
                    world_size=context.world_size,
                    shard_degree=strategy.shard_degree,
                    distributed_strategy=args.strategy,
                    global_batch_size=8,
                    topology_contract_sha256=TOPOLOGY_SHA,
                    allow_topology_reshard=False,
                ),
            )
            assert int(metadata["step"]) == 1
            value = torch.arange(
                256, device="cuda", dtype=torch.float32
            ).reshape(4, 64)
            observed = model(value).detach().float().cpu().tolist()
            expected = json.loads(
                (
                    args.root
                    / "checkpoint_output"
                    / f"rank_{context.rank:05d}.json"
                ).read_text(encoding="utf-8")
            )
            assert observed == expected
            _write(args.root / "eval" / f"rank_{context.rank:05d}.json", observed)
        else:
            metadata, progress = manager.load(
                path=checkpoint,
                model=model,
                optimizer=optimizer,
                expected=ResumeExpectations(
                    step=1,
                    run_lineage=LINEAGE,
                    runtime_config_sha256=RUNTIME_SHA,
                    data_closure_sha256=DATA_SHA,
                    model_contract_sha256=MODEL_SHA,
                    world_size=context.world_size,
                    shard_degree=strategy.shard_degree,
                    distributed_strategy=args.strategy,
                    global_batch_size=8,
                    topology_contract_sha256=TOPOLOGY_SHA,
                    allow_topology_reshard=args.mode == "reshard",
                ),
            )
            assert int(metadata["step"]) == 1
            assert progress == {"next_optimizer_step": 1}
            observed = {"rng": _draw(), "loss": _step(model, optimizer)}
            if args.mode == "exact":
                expected = json.loads(
                    (args.root / "golden" / f"rank_{context.rank:05d}.json").read_text(
                        encoding="utf-8"
                    )
                )
                assert observed == expected, (observed, expected)
            _write(
                args.root / args.mode / f"rank_{context.rank:05d}.json", observed
            )
        dist.barrier()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
