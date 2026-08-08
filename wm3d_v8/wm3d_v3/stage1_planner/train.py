"""DDP trainer for the isolated V7 Stage1-P planning phases.

The trainer always reconstructs the unchanged Stage0 model from a pinned full
checkpoint, applies a small cumulative Stage1 overlay, and fail-closes if the
serving direct policy or native inverse-action head changes.  Data sampling is
addressed by optimizer step, so an exact resume does not depend on iterator
history or worker prefetch timing.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from wm3d_v3.training.train import (
    build_model,
    config_sha256,
    load_train_config,
    module_state_sha256,
)

from .dataset import Stage1BranchDataset, Stage1BranchDatasetConfig
from .losses import (
    DynamicsLossConfig,
    PlannerLossConfig,
    imagined_uncertainty_target,
    native_dynamics_loss,
    planner_loss,
)
from .system import NativePlanningSystem, Stage1SystemConfig


DYNAMICS_PREFIXES = (
    "world.dual.state.action_cond_proj.",
    "world.dual.state.action_cond_pos",
    "world.dual.state.decoder.",
    "world.dual.state.out_proj.",
    "world.dual.action.action_cond_proj.",
    "world.dual.action.action_cond_pos",
    "world.dual.xattn_blocks.",
    "world.dual.action_up.",
    "world.dual.action_down.",
    "world.geom.",
)
PLANNER_PREFIXES = ("planner.",)
SERVING_GUARD_PREFIXES = ("world.action_policy.", "world.action_proj.")


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


class StepAddressedBatchSampler(Sampler[list[int]]):
    """Deterministic rank-local batches keyed only by (seed, step, micro)."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        start_step: int,
        stop_step: int,
        accumulation_steps: int,
        seed: int,
        rank: int,
        world_size: int,
    ):
        if min(dataset_size, batch_size, accumulation_steps, world_size) <= 0:
            raise ValueError("sampler dimensions must be positive")
        if not 0 <= start_step <= stop_step:
            raise ValueError("sampler step interval is invalid")
        if not 0 <= rank < world_size:
            raise ValueError("sampler rank is outside world size")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.stop_step = int(stop_step)
        self.accumulation_steps = int(accumulation_steps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[list[int]]:
        global_batch = self.batch_size * self.world_size
        lo = self.rank * self.batch_size
        hi = lo + self.batch_size
        for step in range(self.start_step, self.stop_step):
            for micro in range(self.accumulation_steps):
                rng = np.random.default_rng(
                    np.random.SeedSequence((self.seed, step, micro, self.world_size))
                )
                indices = rng.integers(
                    0, self.dataset_size, size=global_batch, endpoint=False
                )
                yield [int(value) for value in indices[lo:hi]]

    def __len__(self) -> int:
        return (self.stop_step - self.start_step) * self.accumulation_steps


def _setup_distributed() -> tuple[int, int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Stage1-P formal training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            timeout=torch.distributed.constants.default_pg_timeout * 4,
        )
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank) * 100_003
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _move(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": torch.cuda.get_rng_state().cpu(),
    }


def _restore_rng(payload: dict) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state(payload["torch_cuda"])


def _gather_rng(rank: int, world_size: int) -> list[dict] | None:
    local = _capture_rng()
    if world_size == 1:
        return [local]
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0)
    return gathered


def _load_stage1_state(system: NativePlanningSystem, state: dict[str, torch.Tensor]) -> None:
    current = system.state_dict()
    allowed_prefixes = (*DYNAMICS_PREFIXES, *PLANNER_PREFIXES)
    unknown = sorted(set(state) - set(current))
    if unknown:
        raise RuntimeError(f"overlay contains unknown keys: {unknown[:8]}")
    forbidden = sorted(name for name in state if not name.startswith(allowed_prefixes))
    if forbidden:
        raise RuntimeError(f"overlay contains out-of-scope keys: {forbidden[:8]}")
    missing_stage1 = sorted(
        name
        for name in current
        if name.startswith(allowed_prefixes) and name not in state
    )
    if missing_stage1:
        raise RuntimeError(f"overlay is incomplete; first missing={missing_stage1[:8]}")
    with torch.no_grad():
        for name, value in state.items():
            if current[name].shape != value.shape:
                raise RuntimeError(f"overlay tensor shape mismatch: {name}")
            current[name].copy_(value)


def _stage1_state(system: NativePlanningSystem) -> dict[str, torch.Tensor]:
    prefixes = (*DYNAMICS_PREFIXES, *PLANNER_PREFIXES)
    state = {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
        if name.startswith(prefixes)
    }
    if not state:
        raise RuntimeError("Stage1 overlay selection is empty")
    return state


def _set_trainable(system: NativePlanningSystem, phase: str) -> list[str]:
    phase_prefixes = {
        "dynamics": DYNAMICS_PREFIXES,
        "planner": PLANNER_PREFIXES,
        "joint": (*DYNAMICS_PREFIXES, *PLANNER_PREFIXES),
    }
    if phase not in phase_prefixes:
        raise ValueError(f"unsupported Stage1-P phase: {phase}")
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    matched: list[str] = []
    for name, parameter in system.named_parameters():
        if name.startswith(phase_prefixes[phase]):
            parameter.requires_grad_(True)
            matched.append(name)
    if not matched:
        raise RuntimeError("phase trainable allowlist matched no parameters")
    leaked = [
        name
        for name, parameter in system.named_parameters()
        if parameter.requires_grad and name.startswith(SERVING_GUARD_PREFIXES)
    ]
    if leaked:
        raise RuntimeError(f"serving action parameters became trainable: {leaked[:8]}")
    return matched


def _optimizer(system: NativePlanningSystem, phase_cfg: dict) -> torch.optim.Optimizer:
    world, planner = [], []
    for name, parameter in system.named_parameters():
        if not parameter.requires_grad:
            continue
        (planner if name.startswith("planner.") else world).append(parameter)
    groups = []
    if world:
        groups.append(
            {
                "params": world,
                "lr": float(phase_cfg["world_lr"]),
                "name": "native_world",
            }
        )
    if planner:
        groups.append(
            {
                "params": planner,
                "lr": float(phase_cfg["planner_lr"]),
                "name": "planner",
            }
        )
    return torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=float(phase_cfg.get("weight_decay", 0.01)),
    )


def _scheduler(optimizer, *, max_steps: int, warmup_steps: int):
    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _guard_hashes(system: NativePlanningSystem) -> dict[str, str | None]:
    return {
        "direct_policy": module_state_sha256(system.world.action_policy),
        "native_action_projection": module_state_sha256(system.world.action_proj),
    }


def _reduce_metrics(metrics: dict[str, torch.Tensor], world_size: int) -> dict[str, float]:
    result = {}
    for name, value in metrics.items():
        scalar = value.detach().float().mean()
        if world_size > 1:
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            scalar /= world_size
        result[name] = float(scalar.cpu())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = load_train_config(args.cfg)
    phase_cfg = dict(cfg["planner_stage"])
    data_cfg = dict(cfg["planner_data"])
    rank, world_size, local_rank, device = _setup_distributed()
    expected_world = int(phase_cfg["num_nodes"]) * int(phase_cfg["gpus_per_node"])
    if world_size != expected_world:
        raise RuntimeError(f"world size {world_size} != formal contract {expected_world}")
    _seed(int(phase_cfg["seed"]), rank)

    source_checkpoint = Path(phase_cfg["source_checkpoint"])
    source_sha = sha256_file(source_checkpoint)
    if source_sha != str(phase_cfg["source_checkpoint_sha256"]):
        raise RuntimeError("Stage0 source checkpoint SHA256 mismatch")
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    world = build_model(cfg)
    loaded = world.load_state_dict(source["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Stage0 strict model load was not clean")
    system_cfg = Stage1SystemConfig(
        candidate_microbatch=int(phase_cfg.get("candidate_microbatch", 1)),
        detach_between_chunks=False,
        activation_checkpointing=bool(phase_cfg.get("activation_checkpointing", True)),
    )
    system = NativePlanningSystem(world, system_cfg)

    predecessor = phase_cfg.get("predecessor_overlay")
    if predecessor:
        predecessor_path = Path(predecessor)
        predecessor_sha = sha256_file(predecessor_path)
        if predecessor_sha != str(phase_cfg["predecessor_overlay_sha256"]):
            raise RuntimeError("predecessor Stage1 overlay SHA256 mismatch")
        payload = torch.load(predecessor_path, map_location="cpu", weights_only=False)
        if payload.get("source_checkpoint_sha256") != source_sha:
            raise RuntimeError("predecessor overlay is bound to another Stage0 checkpoint")
        _load_stage1_state(system, payload["stage1_state"])

    phase = str(phase_cfg["phase"])
    trainable_names = _set_trainable(system, phase)
    initial_guards = _guard_hashes(system)
    expected_guards = phase_cfg.get("serving_action_hashes") or initial_guards
    if initial_guards != expected_guards:
        raise RuntimeError("serving action hashes differ from the phase contract")
    system.to(device)
    optimizer = _optimizer(system, phase_cfg)
    max_steps = int(phase_cfg["max_steps"])
    scheduler = _scheduler(
        optimizer,
        max_steps=max_steps,
        warmup_steps=int(phase_cfg.get("warmup_steps", 0)),
    )
    start_step = 0
    resume_payload = None
    if args.resume is not None:
        resume_sha = sha256_file(args.resume)
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload.get("run_lineage") != phase_cfg["run_lineage"]:
            raise RuntimeError("exact resume run lineage mismatch")
        if resume_payload.get("config_sha256") != config_sha256(cfg):
            raise RuntimeError("exact resume config SHA256 mismatch")
        if int(resume_payload.get("world_size", -1)) != world_size:
            raise RuntimeError("exact resume world size mismatch")
        if resume_payload.get("source_checkpoint_sha256") != source_sha:
            raise RuntimeError("exact resume Stage0 source mismatch")
        _load_stage1_state(system, resume_payload["stage1_state"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        start_step = int(resume_payload["step"])
        if rank == 0:
            print(json.dumps({"event": "exact_resume", "path": str(args.resume), "sha256": resume_sha, "step": start_step}), flush=True)

    dataset = Stage1BranchDataset(
        Stage1BranchDatasetConfig(
            index_path=Path(data_cfg["index"]),
            split="train",
            action_stats=Path(data_cfg["action_stats"]),
            context_frames=int(data_cfg["context_frames"]),
            future_frames=int(data_cfg["future_frames"]),
            action_history_len=int(data_cfg["action_history_len"]),
        )
    )
    index_sha = sha256_file(Path(data_cfg["index"]))
    if index_sha != str(data_cfg["index_sha256"]):
        raise RuntimeError("Stage1-P branch index SHA256 mismatch")
    accumulation = int(phase_cfg["gradient_accumulation_steps"])
    sampler = StepAddressedBatchSampler(
        dataset_size=len(dataset),
        batch_size=int(phase_cfg["batch_size_per_gpu"]),
        start_step=start_step,
        stop_step=max_steps,
        accumulation_steps=accumulation,
        seed=int(phase_cfg["sampler_seed"]),
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(phase_cfg.get("num_workers", 4)),
        pin_memory=True,
        persistent_workers=int(phase_cfg.get("num_workers", 4)) > 0,
    )
    model = DDP(
        system,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
    ) if world_size > 1 else system
    target = model.module if isinstance(model, DDP) else model
    cfg_sha = config_sha256(cfg)
    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / "ckpt"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "event": "startup_pass",
            "phase": phase,
            "world_size": world_size,
            "source_checkpoint_sha256": source_sha,
            "index_sha256": index_sha,
            "config_sha256": cfg_sha,
            "run_lineage": phase_cfg["run_lineage"],
            "trainable_parameters": sum(p.numel() for p in system.parameters() if p.requires_grad),
            "trainable_tensor_count": len(trainable_names),
            "serving_action_hashes": initial_guards,
        }, sort_keys=True), flush=True)

    dynamics_cfg = DynamicsLossConfig(**dict(phase_cfg.get("dynamics_loss") or {}))
    planner_cfg = PlannerLossConfig(**dict(phase_cfg.get("planner_loss") or {}))
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    # DataLoader worker seeding and DDP construction may consume process RNG.
    # Restore only after both are complete so model dropout resumes exactly.
    if resume_payload is not None:
        _restore_rng(resume_payload["rng_by_rank"][rank])
    step = start_step
    while step < max_steps:
        started = time.monotonic()
        local_sums: dict[str, torch.Tensor] = {}
        for micro in range(accumulation):
            batch = _move(next(iterator), device)
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and micro + 1 < accumulation
                else nullcontext()
            )
            with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                wants_planner = phase in {"planner", "joint"}
                result = model(
                    batch["s_in"],
                    batch["c"],
                    batch["candidate_actions"],
                    wrist=batch["s_wrist"],
                    view_mask=batch["view_mask"],
                    score_planner=wants_planner,
                    true_future_codec=(batch["branch_s_tgt_codec"] if wants_planner else None),
                    true_depth=(batch["branch_depth_tgt"] if wants_planner else None),
                    true_point=(batch["branch_point_tgt"] if wants_planner else None),
                    true_pose=(batch["branch_pose_geom_tgt"] if wants_planner else None),
                )
                decoded_target = target.world.decode_input_tokens(
                    batch["branch_s_tgt_codec"].flatten(0, 1)
                ).unflatten(0, batch["branch_s_tgt_codec"].shape[:2])
                metrics: dict[str, torch.Tensor] = {}
                total = decoded_target.sum() * 0.0
                if phase in {"dynamics", "joint"}:
                    dynamics = native_dynamics_loss(
                        result["rollout"],
                        decoded_target,
                        target_depth=batch["branch_depth_tgt"],
                        target_depth_conf=batch["branch_depth_conf_tgt"],
                        target_point=batch["branch_point_tgt"],
                        target_point_conf=batch["branch_point_conf_tgt"],
                        target_pose=batch["branch_pose_geom_tgt"],
                        branch_valid=batch["branch_valid"],
                        factual_index=0,
                        cfg=dynamics_cfg,
                    )
                    total = total + float(phase_cfg.get("dynamics_weight", 1.0)) * dynamics["loss"]
                    metrics.update({f"dynamics/{key}": value for key, value in dynamics.items()})
                if wants_planner:
                    imagined_uncertainty = imagined_uncertainty_target(
                        result["rollout"].tokens, decoded_target
                    )
                    imagined = planner_loss(
                        result["planner"],
                        branch_rewards=batch["branch_rewards"],
                        branch_dones=batch["branch_dones"],
                        branch_success=batch["branch_success"],
                        branch_valid=batch["planning_mask"],
                        uncertainty_target=imagined_uncertainty,
                        cfg=planner_cfg,
                    )
                    true = planner_loss(
                        result["true_planner"],
                        branch_rewards=batch["branch_rewards"],
                        branch_dones=batch["branch_dones"],
                        branch_success=batch["branch_success"],
                        branch_valid=batch["planning_mask"],
                        uncertainty_target=torch.zeros_like(imagined_uncertainty),
                        cfg=planner_cfg,
                    )
                    total = (
                        total
                        + float(phase_cfg.get("imagined_planner_weight", 0.5)) * imagined["loss"]
                        + float(phase_cfg.get("true_planner_weight", 1.0)) * true["loss"]
                    )
                    metrics.update({f"planner_imagined/{key}": value for key, value in imagined.items()})
                    metrics.update({f"planner_true/{key}": value for key, value in true.items()})
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError(f"non-finite Stage1-P loss at step {step}")
                (total / accumulation).backward()
                metrics["loss"] = total
            for name, value in metrics.items():
                detached = value.detach().float().mean()
                local_sums[name] = local_sums.get(name, detached.new_zeros(())) + detached
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in system.parameters() if parameter.requires_grad],
            float(phase_cfg.get("max_grad_norm", 1.0)),
        )
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        reduced_last = _reduce_metrics(
            {name: value / accumulation for name, value in local_sums.items()},
            world_size,
        )
        log_every = int(phase_cfg.get("log_every", 10))
        if rank == 0 and (step == 1 or step % log_every == 0):
            print(json.dumps({
                "event": "train",
                "phase": phase,
                "step": step,
                "seconds": time.monotonic() - started,
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": [float(group["lr"]) for group in optimizer.param_groups],
                **reduced_last,
            }, sort_keys=True), flush=True)

        checkpoint_due = step % int(phase_cfg["checkpoint_every_steps"]) == 0 or step == max_steps
        if checkpoint_due:
            rng_by_rank = _gather_rng(rank, world_size)
            if rank == 0:
                current_guards = _guard_hashes(target)
                if current_guards != initial_guards:
                    raise RuntimeError("frozen Stage0 serving action state changed")
                checkpoint = {
                    "schema": "wm3d_v7_stage1_planner_overlay_v1",
                    "phase": phase,
                    "step": step,
                    "stage1_state": _stage1_state(target),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_by_rank": rng_by_rank,
                    "world_size": world_size,
                    "config_sha256": cfg_sha,
                    "run_lineage": phase_cfg["run_lineage"],
                    "source_checkpoint": str(source_checkpoint.resolve()),
                    "source_checkpoint_sha256": source_sha,
                    "branch_index_sha256": index_sha,
                    "serving_action_hashes": current_guards,
                    "sampler": {
                        "schema": "wm3d_v7_stage1_step_addressed_sampler_v1",
                        "seed": int(phase_cfg["sampler_seed"]),
                        "next_step": step,
                        "world_size": world_size,
                        "batch_size_per_gpu": int(phase_cfg["batch_size_per_gpu"]),
                        "gradient_accumulation_steps": accumulation,
                    },
                }
                path = ckpt_dir / f"step_{step:08d}.pt"
                _atomic_checkpoint(path, checkpoint)
                print(json.dumps({"event": "checkpoint", "path": str(path), "step": step}), flush=True)
            if world_size > 1:
                dist.barrier()

    final_guards = _guard_hashes(target)
    if final_guards != initial_guards:
        raise RuntimeError("Stage0 action ownership changed during Stage1-P")
    if rank == 0:
        print(json.dumps({"event": "hard_stop", "phase": phase, "step": step}), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
