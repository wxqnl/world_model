"""Exact-resumable planner-only trainer for WM3D-V8 Stage1.

Stage0 is reconstructed by a strict full-state load and remains frozen.  Only
the action-blind planner reads real, explicit future token/depth/point/pose
evidence.  This is intentionally a planning stage, not another VLA policy.
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

from .candidates import deterministic_action_cost
from .dataset import Stage1BranchDataset, Stage1BranchDatasetConfig, sha256_file
from .losses import PlannerLossConfig, planner_loss
from .planner_head import NativePlannerConfig, planning_score
from .system import NativePlanningSystem, Stage1SystemConfig


OVERLAY_SCHEMA = "wm3d_v8_stage1_native_planner_overlay_v1"
SAMPLER_SCHEMA = "wm3d_v8_stage1_step_addressed_sampler_v1"
SOURCE_CONTRACT_SCHEMA = "wm3d_v8_stage0_action_policy_contract_v3"
PLANNER_PREFIX = "planner."


class StepAddressedBatchSampler(Sampler[list[int]]):
    """Deterministic rank-local batches keyed only by step and microstep."""

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
        raise RuntimeError("V8 Stage1 validation requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
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


def _broadcast_text(value: str | None, *, rank: int, world_size: int) -> str:
    values = [value]
    if world_size > 1:
        dist.broadcast_object_list(values, src=0)
    result = values[0]
    if not isinstance(result, str):
        raise RuntimeError("distributed text broadcast failed")
    return result


def _planner_state(system: NativePlanningSystem) -> dict[str, torch.Tensor]:
    state = {
        f"{PLANNER_PREFIX}{name}": value.detach().cpu().clone()
        for name, value in system.planner.state_dict().items()
    }
    if not state:
        raise RuntimeError("planner overlay selection is empty")
    return state


def _load_planner_state(
    system: NativePlanningSystem, state: dict[str, torch.Tensor]
) -> None:
    expected = {
        f"{PLANNER_PREFIX}{name}": value
        for name, value in system.planner.state_dict().items()
    }
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise RuntimeError(
            f"planner overlay key mismatch missing={missing[:4]} extra={extra[:4]}"
        )
    stripped = {}
    for name, value in state.items():
        if expected[name].shape != value.shape:
            raise RuntimeError(f"planner overlay tensor shape mismatch: {name}")
        stripped[name[len(PLANNER_PREFIX) :]] = value
    loaded = system.planner.load_state_dict(stripped, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("planner strict overlay load was not clean")


def _set_planner_only(system: NativePlanningSystem) -> list[str]:
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    for parameter in system.planner.parameters():
        parameter.requires_grad_(True)
    trainable = [name for name, value in system.named_parameters() if value.requires_grad]
    if not trainable or any(not name.startswith(PLANNER_PREFIX) for name in trainable):
        raise RuntimeError(f"Stage1 trainable scope escaped planner: {trainable[:8]}")
    return trainable


def _optimizer(system: NativePlanningSystem, cfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        system.planner.parameters(),
        lr=float(cfg["planner_lr"]),
        betas=(0.9, 0.95),
        eps=1.0e-8,
        weight_decay=float(cfg.get("weight_decay", 0.01)),
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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite numbered checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_checkpoint_directory(
    ckpt_dir: Path,
    *,
    require_empty: bool,
    resume: Path | None,
) -> None:
    existing = sorted(ckpt_dir.glob("step_*.pt")) if ckpt_dir.is_dir() else []
    if not require_empty:
        return
    if resume is None:
        if existing:
            raise FileExistsError(
                f"fresh Stage1 run requires an empty checkpoint directory: {ckpt_dir}"
            )
        return
    try:
        resume_step = int(resume.stem.removeprefix("step_"))
    except ValueError as exc:
        raise ValueError(f"resume checkpoint is not numbered: {resume}") from exc
    unexpected = [
        path
        for path in existing
        if int(path.stem.removeprefix("step_")) > resume_step
    ]
    if unexpected:
        raise FileExistsError(
            f"exact resume directory contains checkpoints beyond the source: "
            f"{unexpected[:4]}"
        )


def _reduce_metrics(
    metrics: dict[str, torch.Tensor], world_size: int
) -> dict[str, float]:
    result = {}
    for name, value in metrics.items():
        scalar = value.detach().float().mean()
        if world_size > 1:
            dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
            scalar /= world_size
        result[name] = float(scalar.cpu())
    return result


def _dataset_config(data_cfg: dict, split: str) -> Stage1BranchDatasetConfig:
    return Stage1BranchDatasetConfig(
        branch_index=Path(data_cfg["branch_index"]),
        branch_index_sha256=str(data_cfg["branch_index_sha256"]),
        branch_payload_sha256_manifest=Path(
            data_cfg["branch_payload_sha256_manifest"]
        ),
        branch_payload_sha256_manifest_sha256=str(
            data_cfg["branch_payload_sha256_manifest_sha256"]
        ),
        runtime_index=Path(data_cfg["runtime_index"]),
        runtime_index_sha256=str(data_cfg["runtime_index_sha256"]),
        action_stats=Path(data_cfg["action_stats"]),
        action_stats_sha256=str(data_cfg["action_stats_sha256"]),
        action_adapter_audit=Path(data_cfg["action_adapter_audit"]),
        action_adapter_audit_sha256=str(data_cfg["action_adapter_audit_sha256"]),
        split=split,
        context_frames=int(data_cfg.get("context_frames", 16)),
        future_frames=int(data_cfg.get("future_frames", 32)),
        verify_runtime_payload_sha256=bool(
            data_cfg.get("verify_runtime_payload_sha256", True)
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    cfg = load_train_config(args.cfg)
    stage_cfg = dict(cfg["planner_stage"])
    data_cfg = dict(cfg["planner_data"])
    rank, world_size, local_rank, device = _setup_distributed()
    expected_world = int(stage_cfg["num_nodes"]) * int(stage_cfg["gpus_per_node"])
    if world_size != expected_world:
        raise RuntimeError(f"world size {world_size} != contract {expected_world}")
    _seed(int(stage_cfg["seed"]), rank)

    source_path = Path(stage_cfg["source_checkpoint"])
    observed_source_sha = sha256_file(source_path) if rank == 0 else None
    source_sha = _broadcast_text(observed_source_sha, rank=rank, world_size=world_size)
    if source_sha != str(stage_cfg["source_checkpoint_sha256"]):
        raise RuntimeError("Stage0 source checkpoint SHA256 mismatch")
    source = torch.load(
        source_path, map_location="cpu", weights_only=False, mmap=True
    )
    source_contract = source.get("action_policy_contract")
    if not isinstance(source_contract, dict):
        raise RuntimeError("Stage0 source lacks action policy contract")
    if source_contract.get("schema") != SOURCE_CONTRACT_SCHEMA:
        raise RuntimeError("Stage1 source is not the promoted V8 v3 contract")
    if source_contract.get("stage0_native3d_owner") is not True:
        raise RuntimeError("Stage1 source does not preserve native 3D ownership")
    if source_contract.get("proprio", {}).get("required") is not True:
        raise RuntimeError("Stage1 source lacks required current-state proprio")
    if int(source.get("step", -1)) != int(stage_cfg["source_checkpoint_step"]):
        raise RuntimeError("Stage1 source checkpoint step mismatch")

    world = build_model(cfg)
    loaded = world.load_state_dict(source["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Stage0 strict full-state load was not clean")
    del source
    system = NativePlanningSystem(
        world,
        Stage1SystemConfig(
            planner=NativePlannerConfig(**dict(stage_cfg.get("planner_model") or {})),
            candidate_microbatch=int(stage_cfg.get("candidate_microbatch", 1)),
            detach_between_chunks=True,
            activation_checkpointing=False,
        ),
    )
    trainable_names = _set_planner_only(system)
    frozen_action_hash = {
        "action_policy": module_state_sha256(system.world.action_policy),
        "action_projection": module_state_sha256(system.world.action_proj),
    }
    initial_planner_hash = module_state_sha256(system.planner)
    system.to(device)
    system.world.eval()
    system.planner.train()

    optimizer = _optimizer(system, stage_cfg)
    max_steps = int(stage_cfg["max_steps"])
    stop_step = max_steps if args.stop_after_step is None else int(args.stop_after_step)
    if not 0 < stop_step <= max_steps:
        raise ValueError("--stop-after-step must lie in (0,max_steps]")
    scheduler = _scheduler(
        optimizer,
        max_steps=max_steps,
        warmup_steps=int(stage_cfg.get("warmup_steps", 0)),
    )
    cfg_sha = config_sha256(cfg)
    evidence_mode = str(stage_cfg.get("evidence_mode", "mixed"))
    if evidence_mode not in {"observed", "imagined", "mixed"}:
        raise ValueError(f"unsupported Stage1 evidence mode: {evidence_mode}")
    start_step = 0
    resume_payload = None
    if args.resume is not None:
        observed_resume_sha = sha256_file(args.resume) if rank == 0 else None
        resume_sha = _broadcast_text(
            observed_resume_sha, rank=rank, world_size=world_size
        )
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_payload.get("schema") != OVERLAY_SCHEMA:
            raise RuntimeError("exact resume overlay schema mismatch")
        if resume_payload.get("run_lineage") != stage_cfg["run_lineage"]:
            raise RuntimeError("exact resume run lineage mismatch")
        if resume_payload.get("config_sha256") != cfg_sha:
            raise RuntimeError("exact resume config SHA256 mismatch")
        if int(resume_payload.get("world_size", -1)) != world_size:
            raise RuntimeError("exact resume world size mismatch")
        if resume_payload.get("source_checkpoint_sha256") != source_sha:
            raise RuntimeError("exact resume Stage0 source mismatch")
        _load_planner_state(system, resume_payload["stage1_state"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        start_step = int(resume_payload["step"])
        if not 0 <= start_step < stop_step:
            raise RuntimeError("exact resume interval is empty or reversed")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "exact_resume",
                        "path": str(args.resume),
                        "sha256": resume_sha,
                        "step": start_step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    dataset = Stage1BranchDataset(_dataset_config(data_cfg, "train"))
    accumulation = int(stage_cfg["gradient_accumulation_steps"])
    sampler = StepAddressedBatchSampler(
        dataset_size=len(dataset),
        batch_size=int(stage_cfg["batch_size_per_gpu"]),
        start_step=start_step,
        stop_step=stop_step,
        accumulation_steps=accumulation,
        seed=int(stage_cfg["sampler_seed"]),
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(stage_cfg.get("num_workers", 0)),
        pin_memory=True,
    )
    planner_model = (
        DDP(
            system.planner,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world_size > 1
        else system.planner
    )
    planner_target = (
        planner_model.module if isinstance(planner_model, DDP) else planner_model
    )
    output_root = Path(cfg["out"]["root"])
    ckpt_dir = output_root / "ckpt"
    _validate_checkpoint_directory(
        ckpt_dir,
        require_empty=bool(cfg["out"].get("require_empty_checkpoint_dir", True)),
        resume=args.resume,
    )
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(
            json.dumps(
                {
                    "event": "startup_pass",
                    "schema": OVERLAY_SCHEMA,
                    "world_size": world_size,
                    "source_checkpoint": str(source_path),
                    "source_checkpoint_sha256": source_sha,
                    "source_checkpoint_step": int(stage_cfg["source_checkpoint_step"]),
                    "source_action_contract_sha256": source_contract["contract_sha256"],
                    "config_sha256": cfg_sha,
                    "run_lineage": stage_cfg["run_lineage"],
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in system.parameters()
                        if parameter.requires_grad
                    ),
                    "trainable_tensor_count": len(trainable_names),
                    "frozen_action_hash": frozen_action_hash,
                    "initial_planner_hash": initial_planner_hash,
                    "branch_index_sha256": data_cfg["branch_index_sha256"],
                    "branch_payload_sha256_manifest_sha256": data_cfg[
                        "branch_payload_sha256_manifest_sha256"
                    ],
                    "runtime_index_sha256": data_cfg["runtime_index_sha256"],
                    "stage0_frozen": True,
                    "planner_action_inputs": False,
                    "evidence_mode": evidence_mode,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    loss_cfg = PlannerLossConfig(**dict(stage_cfg.get("planner_loss") or {}))
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    if resume_payload is not None:
        _restore_rng(resume_payload["rng_by_rank"][rank])
    step = start_step
    last_metrics: dict[str, float] = {}
    while step < stop_step:
        started = time.monotonic()
        local_sums: dict[str, torch.Tensor] = {}
        for micro in range(accumulation):
            batch = _move(next(iterator), device)
            sync_context = (
                planner_model.no_sync()
                if isinstance(planner_model, DDP) and micro + 1 < accumulation
                else nullcontext()
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                codec = batch["branch_s_tgt_codec"]
                bsz, candidates, horizon, patches, codec_dim = codec.shape
                decoded = system.world.decode_input_tokens(
                    codec.reshape(bsz * candidates, horizon, patches, codec_dim)
                ).reshape(bsz, candidates, horizon, patches, -1)
                imagined_rollout = (
                    system.imagine(
                        batch["s_in"],
                        batch["c"],
                        batch["candidate_actions"],
                        wrist=batch["s_wrist"],
                        view_mask=batch["view_mask"],
                    )
                    if evidence_mode in {"imagined", "mixed"}
                    else None
                )
                physical_cost = deterministic_action_cost(
                    batch["branch_actions_physical"]
                )
            with sync_context, torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                total = decoded.sum() * 0.0
                metrics: dict[str, torch.Tensor] = {}
                evidence = []
                if evidence_mode in {"observed", "mixed"}:
                    evidence.append(
                        (
                            "observed",
                            decoded,
                            batch["branch_depth_tgt"],
                            batch["branch_point_tgt"],
                            batch["branch_pose_geom_tgt"],
                            torch.zeros(
                                decoded.shape[:2], device=decoded.device, dtype=decoded.dtype
                            ),
                            float(stage_cfg.get("observed_planner_weight", 1.0)),
                        )
                    )
                if imagined_rollout is not None:
                    token_error = (
                        imagined_rollout.tokens.float() - decoded.float()
                    ).square().mean(dim=(-1, -2, -3))
                    token_scale = decoded.float().square().mean(
                        dim=(-1, -2, -3)
                    ).clamp_min(1.0e-6)
                    evidence.append(
                        (
                            "imagined",
                            imagined_rollout.tokens,
                            imagined_rollout.depth,
                            imagined_rollout.point,
                            imagined_rollout.pose,
                            (token_error / token_scale).clamp(0.0, 1.0),
                            float(stage_cfg.get("imagined_planner_weight", 1.0)),
                        )
                    )
                for label, tokens, depth, point, pose, uncertainty, weight in evidence:
                    outputs = planner_model(
                        tokens,
                        batch["c"],
                        depth=depth,
                        point=point,
                        pose=pose,
                    )
                    outputs["score"] = planning_score(
                        outputs, physical_cost
                    )
                    current = planner_loss(
                        outputs,
                        branch_rewards=batch["branch_rewards"],
                        branch_dones=batch["branch_dones"],
                        branch_success=batch["branch_success"],
                        branch_valid=batch["planning_mask"],
                        uncertainty_target=uncertainty,
                        cfg=loss_cfg,
                    )
                    total = total + weight * current["loss"]
                    metrics.update(
                        {f"{label}/{name}": value for name, value in current.items()}
                    )
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError(f"non-finite Stage1 loss at step {step}")
                (total / accumulation).backward()
            for name, value in metrics.items():
                detached = value.detach().float().mean()
                local_sums[name] = local_sums.get(
                    name, detached.new_zeros(())
                ) + detached
        grad_norm = torch.nn.utils.clip_grad_norm_(
            planner_target.parameters(), float(stage_cfg.get("max_grad_norm", 1.0))
        )
        if not bool(torch.isfinite(grad_norm)) or float(grad_norm) <= 0.0:
            raise FloatingPointError(f"invalid planner gradient norm at step {step}")
        if any(parameter.grad is not None for parameter in system.world.parameters()):
            raise RuntimeError("frozen Stage0 received a gradient")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        last_metrics = _reduce_metrics(
            {name: value / accumulation for name, value in local_sums.items()},
            world_size,
        )
        log_every = int(stage_cfg.get("log_every", 5))
        if rank == 0 and (step == 1 or step % log_every == 0):
            print(
                json.dumps(
                    {
                        "event": "train",
                        "step": step,
                        "seconds": time.monotonic() - started,
                        "grad_norm": float(grad_norm.detach().cpu()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        **last_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        checkpoint_due = (
            step % int(stage_cfg["checkpoint_every_steps"]) == 0
            or step == stop_step
        )
        if checkpoint_due:
            rng_by_rank = _gather_rng(rank, world_size)
            if rank == 0:
                current_action_hash = {
                    "action_policy": module_state_sha256(system.world.action_policy),
                    "action_projection": module_state_sha256(system.world.action_proj),
                }
                if current_action_hash != frozen_action_hash:
                    raise RuntimeError("frozen V8 action owner changed during Stage1")
                checkpoint = {
                    "schema": OVERLAY_SCHEMA,
                    "phase": "native_3d_planner",
                    "step": step,
                    "stage1_state": _planner_state(system),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_by_rank": rng_by_rank,
                    "world_size": world_size,
                    "config_sha256": cfg_sha,
                    "run_lineage": stage_cfg["run_lineage"],
                    "source_checkpoint": str(source_path.resolve()),
                    "source_checkpoint_sha256": source_sha,
                    "source_checkpoint_step": int(stage_cfg["source_checkpoint_step"]),
                    "source_action_contract": source_contract,
                    "branch_index_sha256": data_cfg["branch_index_sha256"],
                    "branch_payload_sha256_manifest_sha256": data_cfg[
                        "branch_payload_sha256_manifest_sha256"
                    ],
                    "runtime_index_sha256": data_cfg["runtime_index_sha256"],
                    "action_stats_sha256": data_cfg["action_stats_sha256"],
                    "action_adapter_audit_sha256": data_cfg[
                        "action_adapter_audit_sha256"
                    ],
                    "frozen_action_hash": current_action_hash,
                    "planner_hash": module_state_sha256(system.planner),
                    "last_metrics": last_metrics,
                    "stage0_frozen": True,
                    "planner_action_inputs": False,
                    "evidence_mode": evidence_mode,
                    "sampler": {
                        "schema": SAMPLER_SCHEMA,
                        "seed": int(stage_cfg["sampler_seed"]),
                        "next_step": step,
                        "world_size": world_size,
                        "batch_size_per_gpu": int(stage_cfg["batch_size_per_gpu"]),
                        "gradient_accumulation_steps": accumulation,
                    },
                }
                path = ckpt_dir / f"step_{step:08d}.pt"
                _atomic_checkpoint(path, checkpoint)
                print(
                    json.dumps(
                        {"event": "checkpoint", "path": str(path), "step": step},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if world_size > 1:
                dist.barrier()

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "hard_stop",
                    "step": step,
                    "configured_max_steps": max_steps,
                    "bounded_stop": step < max_steps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
