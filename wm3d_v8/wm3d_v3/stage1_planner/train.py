"""DCP-exact trainer for the unified WM3D V8 Stage1 planner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from wm3d_v3.data.manifest_contract import load_data_profile, sha256_file
from wm3d_v3.data.grouped_normalization import GroupedRobotNormalizer
from wm3d_v3.data.step_sampler import StepAddressedBatchSampler
from wm3d_v3.data.unified_cache_dataset import UnifiedCacheDataset
from wm3d_v3.models.model_factory import build_world_model
from wm3d_v3.training.distributed_checkpoint import (
    DistributedCheckpointManager,
    ResumeExpectations,
    canonical_sha256,
    checkpoint_name,
)
from wm3d_v3.training.distributed_runtime import (
    autocast_context,
    destroy_distributed,
    initialize_adamw_state,
    initialize_distributed,
    no_sync_context,
    reduce_metrics,
    strategy_from_mapping,
    wrap_model,
)
from wm3d_v3.training.runtime_contract import load_materialized_runtime

from .dataset import Stage1BranchDataset, Stage1BranchDatasetConfig
from .losses import PlannerLossConfig, planner_loss
from .planner_head import NativePlannerConfig
from .system import NativePlanningSystem, Stage1SystemConfig


STAGE1_RUNTIME_SCHEMA = "wm3d_v8_unified_stage1_runtime_v1"
STAGE1_RECEIPT_SCHEMA = "wm3d_v8_unified_stage1_train_receipt_v2"
_BRANCH_FIELDS = {
    "index", "index_sha256", "seal", "seal_sha256",
    "stage0_runtime_sha256", "stage0_checkpoint_commit_sha256",
}
_PLANNER_FIELDS = {"horizon", "candidate_microbatch", "model", "loss", "score"}
_SCORE_FIELDS = {
    "progress_weight", "success_weight", "risk_weight",
    "uncertainty_weight", "action_cost_weight",
}
_RUN_FIELDS = {
    "lineage", "output_root", "seed", "total_steps", "checkpoint_interval",
    "micro_batch_size", "gradient_accumulation", "global_batch_size",
    "num_workers", "lr", "weight_decay", "gradient_clip",
}


def _load_stage1(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve(strict=True)
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"schema", "stage0_runtime", "stage0_checkpoint", "branch", "planner", "run"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != STAGE1_RUNTIME_SCHEMA:
        raise ValueError("Stage1 runtime fields/schema mismatch")
    if set(value["branch"]) != _BRANCH_FIELDS:
        raise ValueError("Stage1 branch runtime fields mismatch")
    if set(value["planner"]) != _PLANNER_FIELDS or set(value["run"]) != _RUN_FIELDS:
        raise ValueError("Stage1 planner/run fields mismatch")
    if any("PENDING" in str(item) for item in _walk(value)):
        raise ValueError("Stage1 runtime contains unresolved PENDING values")
    model_fields = set(NativePlannerConfig.__dataclass_fields__)
    loss_fields = set(PlannerLossConfig.__dataclass_fields__)
    if set(value["planner"]["model"]) != model_fields:
        raise ValueError("Stage1 planner model fields mismatch")
    if set(value["planner"]["loss"]) != loss_fields:
        raise ValueError("Stage1 planner loss fields mismatch")
    if set(value["planner"]["score"]) != _SCORE_FIELDS:
        raise ValueError("Stage1 planner score fields mismatch")
    for name in ("total_steps", "checkpoint_interval", "micro_batch_size", "gradient_accumulation", "global_batch_size"):
        if int(value["run"][name]) <= 0:
            raise ValueError(f"Stage1 run.{name} must be positive")
    return value, sha256_file(path)


def _walk(value: object):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def validate_stage1_bindings(stage1: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    model = runtime["model_profile"]["model"]
    planner = stage1["planner"]
    if not 0 < int(planner["horizon"]) <= int(model["K"]):
        raise ValueError("Stage1 horizon must satisfy 0 < H <= sealed Stage0 K")
    derived = {
        "token_dim": int(model["token_dim"]), "task_dim": int(model["task_dim"]),
        "patches": int(model["P"]), "num_views": int(model["num_views"]),
        "max_horizon": int(planner["horizon"]),
        "time_fourier_dim": int(model["time_fourier_dim"]),
        "time_min_period_s": float(model["time_min_period_s"]),
        "time_max_period_s": float(model["time_max_period_s"]),
    }
    for name, expected in derived.items():
        configured = planner["model"][name]
        if configured not in {0, expected}:
            raise ValueError(f"Stage1 planner {name} differs from sealed Stage0 profile")
    seal = json.loads(Path(stage1["branch"]["seal"]).read_text(encoding="utf-8"))
    if int(seal.get("horizon", -1)) != int(planner["horizon"]):
        raise ValueError("Stage1 planner horizon differs from sealed branch horizon")


def _checkpoint_commit_sha(path: Path) -> str:
    commit = path.resolve(strict=True) / "COMMITTED.json"
    if commit.is_symlink() or not commit.is_file():
        raise ValueError("Stage0 source must be a committed DCP checkpoint")
    return sha256_file(commit)


def _topology_sha(stage1: Mapping[str, Any], runtime: Mapping[str, Any]) -> str:
    train = stage1["run"]
    return canonical_sha256({
        "lineage": train["lineage"], "global_batch_size": int(train["global_batch_size"]),
        "strategy": "ddp", "shard_degree": 1,
        "planner_model_contract_sha256": _planner_contract_sha(stage1, runtime),
    })


def _planner_contract_sha(
    stage1: Mapping[str, Any], runtime: Mapping[str, Any]
) -> str:
    model = runtime["model_profile"]["model"]
    configured = dict(stage1["planner"]["model"])
    derived = {
        "token_dim": int(model["token_dim"]),
        "task_dim": int(model["task_dim"]),
        "patches": int(model["P"]),
        "num_views": int(model["num_views"]),
        "max_horizon": int(stage1["planner"]["horizon"]),
        "time_fourier_dim": int(model["time_fourier_dim"]),
        "time_min_period_s": float(model["time_min_period_s"]),
        "time_max_period_s": float(model["time_max_period_s"]),
    }
    configured.update(derived)
    return canonical_sha256({
        "schema": "wm3d_v8_unified_stage1_planner_contract_v1",
        "stage0_model_contract_sha256": runtime["bindings"]["model_contract_sha256"],
        "action_blind": True,
        "single_trained_horizon_only": True,
        "planner_model": configured,
        "planner_score": dict(stage1["planner"]["score"]),
    })


def _expectations(*, step: int, stage1: Mapping[str, Any], stage1_sha: str,
                  runtime: Mapping[str, Any], world_size: int) -> ResumeExpectations:
    run = stage1["run"]
    branch = stage1["branch"]
    return ResumeExpectations(
        step=step, run_lineage=str(run["lineage"]), runtime_config_sha256=stage1_sha,
        data_closure_sha256=canonical_sha256(branch),
        model_contract_sha256=_planner_contract_sha(stage1, runtime),
        world_size=world_size, shard_degree=1,
        distributed_strategy="ddp",
        global_batch_size=int(run["global_batch_size"]),
        topology_contract_sha256=_topology_sha(stage1, runtime), allow_topology_reshard=False,
    )


def _stage0_dataset(
    runtime: Mapping[str, Any], split: str
) -> UnifiedCacheDataset:
    closure = runtime["data_closure"]
    profile = load_data_profile(Path(closure["data_profile_path"]), verify_source_manifests=False)
    normalizer = GroupedRobotNormalizer.load(
        Path(closure["grouped_normalization_path"]), expected_sha256=closure["grouped_normalization_sha256"],
        expected_data_profile_sha256=closure["data_profile_sha256"],
        expected_model_profile_sha256=runtime["bindings"]["model_profile_sha256"],
        expected_window_index_sha256=closure["cache_index_sha256"], data_profile=profile,
    )
    return UnifiedCacheDataset(
        cache_root=Path(closure["cache_root"]), index_path=Path(closure["cache_index_path"]),
        index_sha256=closure["cache_index_sha256"], data_profile=profile,
        model_profile=runtime["model_profile"], split=split, grouped_normalizer=normalizer,
    )


def _dataset(runtime: Mapping[str, Any], stage1: Mapping[str, Any], split: str) -> Stage1BranchDataset:
    closure = runtime["data_closure"]
    stage0 = _stage0_dataset(runtime, split)
    branch = stage1["branch"]
    window_seal = json.loads(Path(closure["cache_seal_path"]).read_text())
    return Stage1BranchDataset(Stage1BranchDatasetConfig(
        branch_index=Path(branch["index"]), branch_index_sha256=branch["index_sha256"],
        branch_seal=Path(branch["seal"]), branch_seal_sha256=branch["seal_sha256"],
        runtime_config_sha256=branch["stage0_runtime_sha256"],
        data_profile_sha256=closure["data_profile_sha256"],
        model_profile_sha256=runtime["bindings"]["model_profile_sha256"],
        window_index_sha256=closure["cache_index_sha256"],
        grouped_normalization_sha256=closure["grouped_normalization_sha256"],
        task_bank_index_sha256=window_seal["task_bank_index_sha256"],
        encoder_contract_sha256=window_seal["encoder_contract_sha256"],
        task_encoder_contract_sha256=window_seal["task_encoder_contract_sha256"],
        representation_contract_sha256=window_seal["representation_contract_sha256"],
        stage0_checkpoint_commit_sha256=branch["stage0_checkpoint_commit_sha256"], split=split,
    ), stage0)


def _trim(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    batch = default_collate(samples)
    # Stage0 fields and candidate fields share the same storage capacities.
    masks = (batch["history_fine_sample_mask"], batch["candidate_fine_sample_mask"])
    nonempty = [mask for mask in masks if bool(mask.any())]
    # Coarse-only sources are part of the same unified ABI.  Keep one masked
    # fine slot so collation remains shape-stable without fabricating samples.
    substeps = (
        max(int(torch.nonzero(mask, as_tuple=False)[:, -1].max()) + 1 for mask in nonempty)
        if nonempty else 1
    )
    for name in (
        "history_fine_action_values", "history_fine_action_mask",
        "candidate_fine_action_values", "candidate_fine_action_mask",
    ):
        batch[name] = batch[name][..., :substeps, :]
    for name in (
        "history_fine_action_dt", "history_fine_sample_mask",
        "candidate_fine_action_dt", "candidate_fine_sample_mask",
    ):
        batch[name] = batch[name][..., :substeps]
    return batch


def _device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _atomic_receipt(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    stage1, stage1_sha = _load_stage1(args.runtime)
    stage0_runtime, stage0_sha = load_materialized_runtime(Path(stage1["stage0_runtime"]))
    repo = Path(__file__).resolve().parents[2]
    current_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != stage0_runtime["run"]["code_commit"]:
        raise ValueError("Stage1 runtime code commit does not match current checkout")
    validate_stage1_bindings(stage1, stage0_runtime)
    if stage0_sha != stage1["branch"]["stage0_runtime_sha256"]:
        raise ValueError("Stage1 branch belongs to another Stage0 runtime")
    source = Path(stage1["stage0_checkpoint"])
    source_commit_sha = _checkpoint_commit_sha(source)
    if source_commit_sha != stage1["branch"]["stage0_checkpoint_commit_sha256"]:
        raise ValueError("Stage1 branch belongs to another Stage0 checkpoint")
    strategy = strategy_from_mapping(stage0_runtime["runtime_profile"]["distributed"])
    context = initialize_distributed(strategy)
    try:
        seed = int(stage1["run"]["seed"])
        random.seed(seed + context.rank)
        np.random.seed((seed + context.rank) % 2**32)
        torch.manual_seed(seed + context.rank)
        torch.cuda.manual_seed(seed + context.rank)
        with torch.device("meta" if strategy.initialization == "meta_sharded" else context.device):
            world = build_world_model(stage0_runtime["model_profile"])
        wrapped_world = wrap_model(world, context, strategy, initialization_seed=seed if strategy.initialization == "meta_sharded" else None).model
        source_step = int(source.name.split("_")[1])
        DistributedCheckpointManager(source.parent).load_model_for_evaluation(
            path=source, model=wrapped_world,
            expected=ResumeExpectations(
                step=source_step, run_lineage=stage0_runtime["run"]["lineage"],
                runtime_config_sha256=stage0_sha,
                data_closure_sha256=stage0_runtime["bindings"]["data_closure_sha256"],
                model_contract_sha256=stage0_runtime["bindings"]["model_contract_sha256"],
                world_size=context.world_size,
                shard_degree=int(stage0_runtime["runtime_profile"]["distributed"]["shard_degree"]),
                distributed_strategy=strategy.strategy,
                global_batch_size=int(stage0_runtime["runtime_profile"]["train"]["global_batch_size"]),
                topology_contract_sha256=str(json.loads((Path(stage0_runtime["run"]["output_root"]) / "run_contract.json").read_text())["topology_contract_sha256"]),
            ),
        )
        for parameter in wrapped_world.parameters():
            parameter.requires_grad_(False)
        planner_cfg = NativePlannerConfig(**stage1["planner"]["model"])
        system = NativePlanningSystem(wrapped_world, Stage1SystemConfig(
            planner=planner_cfg, horizon=int(stage1["planner"]["horizon"]),
            candidate_microbatch=int(stage1["planner"]["candidate_microbatch"]),
            **stage1["planner"]["score"],
        ))
        planner = system.planner.to(context.device)
        if context.world_size > 1:
            planner = torch.nn.parallel.DistributedDataParallel(
                planner, device_ids=[context.local_rank], broadcast_buffers=False
            )
            system.planner = planner
        optimizer = torch.optim.AdamW(planner.parameters(), lr=float(stage1["run"]["lr"]), weight_decay=float(stage1["run"]["weight_decay"]), foreach=False)
        initialize_adamw_state(optimizer)
        output_root = Path(stage1["run"]["output_root"])
        manager = DistributedCheckpointManager(output_root / "checkpoints")
        start = 0
        if args.resume:
            start = int(args.resume.name.split("_")[1])
            _metadata, progress = manager.load(path=args.resume, model=planner, optimizer=optimizer,
                expected=_expectations(step=start, stage1=stage1, stage1_sha=stage1_sha, runtime=stage0_runtime, world_size=context.world_size))
            if int(progress.get("next_optimizer_step", -1)) != start:
                raise ValueError("Stage1 exact resume sampler progress mismatch")
        total = int(stage1["run"]["total_steps"])
        stop = total if args.stop_after_step is None else int(args.stop_after_step)
        if not start < stop <= total or stop % int(stage1["run"]["checkpoint_interval"]):
            raise ValueError("Stage1 stop must be a future sealed checkpoint step")
        dataset = _dataset(stage0_runtime, stage1, "train")
        micro = int(stage1["run"]["micro_batch_size"])
        accum = int(stage1["run"]["gradient_accumulation"])
        if context.world_size * micro * accum != int(stage1["run"]["global_batch_size"]):
            raise ValueError("Stage1 global batch contract mismatch")
        sampler = StepAddressedBatchSampler({"stage1": (0, len(dataset))}, ("stage1",), {"stage1": 1},
            world_size=context.world_size, rank=context.rank, micro_batch_size=micro,
            gradient_accumulation=accum, start_optimizer_step=start, num_optimizer_steps=stop-start, seed=seed)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=int(stage1["run"]["num_workers"]), collate_fn=_trim)
        iterator = iter(loader)
        loss_cfg = PlannerLossConfig(**stage1["planner"]["loss"])
        planner.train()
        wrapped_world.eval()
        last = {}
        for step in range(start, stop):
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for micro_step in range(accum):
                batch = _device(next(iterator), context.device)
                with no_sync_context(planner, enabled=micro_step + 1 < accum):
                    with autocast_context(strategy):
                        outputs = system.score_observed_batch(batch)
                        losses = planner_loss(outputs,
                            branch_rewards=batch["branch_rewards"][:, :, :system.cfg.horizon],
                            branch_dones=batch["branch_dones"][:, :, :system.cfg.horizon],
                            branch_success=batch["branch_success"][:, :, :system.cfg.horizon],
                            branch_valid=batch["branch_valid"], uncertainty_target=torch.zeros_like(batch["branch_valid"], dtype=torch.float32), cfg=loss_cfg)
                    (losses["loss"] / accum).backward()
                for name, value in losses.items():
                    accumulated[name] = (
                        accumulated.get(name, torch.zeros_like(value))
                        + value.detach() / accum
                    )
            if any(parameter.grad is not None for parameter in wrapped_world.parameters()):
                raise RuntimeError("frozen Stage0 received gradients")
            grad = torch.nn.utils.clip_grad_norm_(planner.parameters(), float(stage1["run"]["gradient_clip"]))
            if not bool(torch.isfinite(grad)) or float(grad) <= 0:
                raise FloatingPointError("planner gradient is not finite/nonzero")
            optimizer.step()
            completed = step + 1
            last = reduce_metrics(accumulated)
            if completed % int(stage1["run"]["checkpoint_interval"]) == 0:
                manager.save(step=completed, model=planner, optimizer=optimizer,
                    metadata={"run_lineage": stage1["run"]["lineage"], "runtime_config_sha256": stage1_sha,
                        "data_closure_sha256": canonical_sha256(stage1["branch"]),
                        "model_contract_sha256": _planner_contract_sha(stage1, stage0_runtime),
                        "shard_degree": 1,
                        "distributed_strategy": "ddp", "global_batch_size": int(stage1["run"]["global_batch_size"]),
                        "topology_contract_sha256": _topology_sha(stage1, stage0_runtime),
                        "stage0_checkpoint_commit_sha256": source_commit_sha, "branch_index_sha256": stage1["branch"]["index_sha256"],
                        "stage0_frozen": True, "planner_action_inputs": False, "imagined_rollout": "single_trained_K_only"},
                    rank_state={"next_optimizer_step": completed})
        if context.is_rank0:
            final_checkpoint = output_root / "checkpoints" / checkpoint_name(stop)
            _atomic_receipt(output_root / f"train_receipt_step_{stop:08d}.json", {
                "schema": STAGE1_RECEIPT_SCHEMA, "step": stop, "runtime_sha256": stage1_sha,
                "planner_model_contract_sha256": _planner_contract_sha(stage1, stage0_runtime),
                "code_commit": current_commit,
                "stage0_checkpoint_commit_sha256": source_commit_sha,
                "stage1_checkpoint_commit_sha256": sha256_file(final_checkpoint / "COMMITTED.json"),
                "branch_index_sha256": stage1["branch"]["index_sha256"], "metrics": last,
                "resumed_from_step": start,
                "stage0_frozen": True, "planner_action_inputs": False,
            })
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
