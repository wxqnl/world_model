#!/usr/bin/env python3
"""Exercise the formal V7 data scan and first distributed audit on every rank.

This is a bounded systems test: it initializes the default process group,
eagerly proves the transport before any asymmetric data I/O, builds the exact
formal five-source datasets, and executes the same global mixed-source audit
used by training.  It never constructs a model, writes a checkpoint, or starts
an optimizer step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from wm3d_v3.training.train import (
    MixedSourceWindowDataset,
    audit_distributed_global_mixed_source_contract,
    build_datasets,
    eager_initialize_distributed_transport,
    load_train_config,
    seed_process_rng,
    setup_ddp,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_train_config(args.cfg)
    rank, world, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    eager_initialize_distributed_transport(rank=rank, world=world, device=device)

    train_cfg = cfg.get("train", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    seed_process_rng(int(train_cfg.get("seed", data_cfg.get("seed", 0)) or 0))
    train_dataset, val_dataset = build_datasets(cfg)
    if not isinstance(train_dataset, MixedSourceWindowDataset) or not isinstance(
        val_dataset, MixedSourceWindowDataset
    ):
        raise RuntimeError("formal distributed audit smoke requires mixed datasets")
    audit = audit_distributed_global_mixed_source_contract(
        train_dataset,
        val_dataset,
        world=world,
        rank=rank,
        device=device,
    )
    dist.barrier()
    if rank == 0:
        print(
            "WM3D_V7_FORMAL_DISTRIBUTED_AUDIT_SMOKE_OK "
            + json.dumps(audit, sort_keys=True),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
