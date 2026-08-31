"""Distributed multi-task LIBERO BC training for a benchmark suite."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from wm3d_v3.benchmarks.libero_bc_teacher import (
    _bootstrap_libero,
    _compose_cfg,
    _enable_trusted_legacy_torch_load,
    _init_benchmark_and_embs,
    _prepare_task_dataset,
    _resolve_paths,
)


def _sync_model(model: torch.nn.Module) -> None:
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, src=0)


def _average_gradients(model: torch.nn.Module, world_size: int) -> None:
    grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not grads:
        return
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for grad in grads:
        count = grad.numel()
        grad.copy_(flat[offset : offset + count].view_as(grad))
        offset += count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--libero_root",
        type=Path,
        default=Path("/data/Minko/benchmarks/LIBERO"),
    )
    ap.add_argument("--benchmark_name", default="LIBERO_SPATIAL")
    ap.add_argument("--policy", default="bc_rnn_policy")
    ap.add_argument("--task_embedding_format", default="onehot_no_bert")
    ap.add_argument("--seed", type=int, default=260614)
    ap.add_argument("--seq_len", type=int, default=10)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    _bootstrap_libero(args.libero_root)
    _enable_trusted_legacy_torch_load()
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import control_seed, safe_device, torch_save_model

    cfg_args = argparse.Namespace(
        seed=args.seed,
        benchmark_name=args.benchmark_name,
        policy=args.policy,
        algo="multitask",
        device=f"cuda:{local_rank}",
        task_embedding_format=args.task_embedding_format,
        task_order_index=0,
        seq_len=args.seq_len,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        no_augmentation=False,
        no_eval=True,
        eval_episodes=1,
        eval_num_procs=1,
        eval_max_steps=300,
        eval_every=args.save_every,
        libero_root=args.libero_root,
    )
    cfg = _compose_cfg(cfg_args)
    _resolve_paths(cfg)
    cfg.experiment_dir = str(args.out_dir)
    cfg.experiment_name = args.out_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    control_seed(args.seed)

    benchmark = _init_benchmark_and_embs(cfg)
    datasets = []
    shape_meta = None
    for task_id in range(benchmark.n_tasks):
        dataset, task_shape_meta = _prepare_task_dataset(cfg, benchmark, task_id)
        datasets.append(dataset)
        shape_meta = task_shape_meta
    cfg.shape_meta = shape_meta
    dataset = ConcatDataset(datasets)

    algo = safe_device(
        get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg),
        cfg.device,
    )
    algo.start_task(-1)
    _sync_model(algo.policy)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "benchmark": args.benchmark_name,
                    "policy": args.policy,
                    "tasks": benchmark.n_tasks,
                    "sequences": len(dataset),
                    "world_size": world_size,
                    "global_batch_size": args.batch_size * world_size,
                    "epochs": args.epochs,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        algo.policy.train()
        total_loss = torch.zeros((), device=cfg.device)
        batches = torch.zeros((), device=cfg.device)
        started = time.time()
        for data in loader:
            data = algo.map_tensor_to_device(data)
            algo.optimizer.zero_grad(set_to_none=True)
            loss = algo.policy.compute_loss(data)
            (algo.loss_scale * loss).backward()
            _average_gradients(algo.policy, world_size)
            if cfg.train.grad_clip is not None:
                clip_grad_norm_(algo.policy.parameters(), cfg.train.grad_clip)
            algo.optimizer.step()
            total_loss += loss.detach()
            batches += 1
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(batches, op=dist.ReduceOp.SUM)
        if algo.scheduler is not None:
            algo.scheduler.step()
        if rank == 0:
            mean_loss = float((total_loss / batches.clamp_min(1)).cpu())
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": mean_loss,
                        "elapsed_sec": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                path = args.out_dir / f"multitask_model_ep{epoch}.pth"
                torch_save_model(algo.policy, str(path), cfg=cfg)
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
