"""Formal native WM3D-V7 5B pretraining entrypoint.

Launch only through torchrun/Slurm wrappers in ``scripts/scale5b``.  The
trainer uses FSDP2/HSDP, a step-addressed exact-ratio sampler and transactional
Distributed Checkpoint shards.  It never resumes an implicit ``latest`` path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import yaml

from wm3d_v3.data.scale5b_contracts import (
    DATASET_SCHEMA,
    DatasetContract,
    DatasetSeal,
    atomic_write_json,
    canonical_sha256,
    load_contract,
    load_seal,
    resolve_regular_file,
    verify_dataset_seal,
)
from wm3d_v3.data.scale5b_dataset import (
    Native5BMixedDataset,
    Native5BSourceDataset,
    WindowLoaderConfig,
)
from wm3d_v3.data.scale5b_sampler import StepAddressedBatchSampler
from wm3d_v3.models.native5b import NativeWM3D5B, config_from_mapping
from wm3d_v3.training.scale5b_checkpoint import (
    Native5BCheckpointManager,
    ResumeExpectations,
)
from wm3d_v3.training.scale5b_config import (
    TRAIN_CONFIG_SCHEMA,
    training_contract_sha256,
    verify_code_receipt,
)
from wm3d_v3.training.scale5b_environment import (
    load_environment_contract,
    verify_environment_receipt,
)
from wm3d_v3.training.scale5b_loss import (
    Native5BLossConfig,
    native5b_loss,
)
from wm3d_v3.training.scale5b_runtime import (
    RuntimeContractError,
    all_reduce_mean,
    apply_fsdp2,
    assert_v7_native_dependency_boundary,
    destroy_distributed,
    initialize_adamw_state,
    initialize_distributed,
    reduce_metrics,
    set_gradient_sync,
    verify_parameter_budget,
    wsd_learning_rate,
)


RUN_CONTRACT_SCHEMA = "wm3d_v7_native5b_run_contract_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Explicit numbered committed checkpoint directory; never 'latest'.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate config/data/hardware without constructing the 5B model.",
    )
    return parser.parse_args()


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    value = _expand_environment(value)
    if not isinstance(value, dict) or value.get("schema") != TRAIN_CONFIG_SCHEMA:
        raise RuntimeContractError(
            f"config must use schema {TRAIN_CONFIG_SCHEMA}: {path}"
        )
    return value


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise RuntimeContractError(
                f"unresolved environment variable in config value {value!r}"
            )
        return expanded
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def _strict_checkpoint_steps(config: Mapping[str, Any]) -> set[int]:
    total = int(config["train"]["total_steps"])
    values = {int(value) for value in config["train"]["checkpoint_steps"]}
    interval = int(config["train"].get("checkpoint_interval", 0))
    if interval > 0:
        values.update(range(interval, total + 1, interval))
    values.add(total)
    if any(value <= 0 or value > total for value in values):
        raise RuntimeContractError("checkpoint steps must lie in (0,total_steps]")
    return values


def _configure_reproducibility(seed: int, rank: int) -> None:
    # Identical model initialization on every rank; data randomness is entirely
    # step-addressed and rank-aware in the sampler.
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    del rank


def _available_bytes(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail) * int(value.f_frsize)


def _validate_config(
    config: Mapping[str, Any],
    *,
    world_size: int,
    config_path: Path,
) -> dict[str, Any]:
    dist_cfg = config["distributed"]
    train_cfg = config["train"]
    data_cfg = config["data"]
    if int(dist_cfg["expected_world_size"]) != int(world_size):
        raise RuntimeContractError(
            f"WORLD_SIZE={world_size} != expected {dist_cfg['expected_world_size']}"
        )
    if int(dist_cfg["shard_degree"]) <= 1:
        raise RuntimeContractError("formal native5b requires FSDP2 sharding")
    if world_size % int(dist_cfg["shard_degree"]):
        raise RuntimeContractError("world size is not divisible by shard degree")
    global_batch = (
        world_size
        * int(train_cfg["micro_batch_size"])
        * int(train_cfg["gradient_accumulation"])
    )
    if global_batch != int(train_cfg["global_batch_size"]):
        raise RuntimeContractError(
            f"derived global batch {global_batch} != configured "
            f"{train_cfg['global_batch_size']}"
        )
    if int(train_cfg["total_steps"]) <= 0:
        raise RuntimeContractError("total_steps must be positive")
    _strict_checkpoint_steps(config)
    root = Path(data_cfg["root"]).resolve(strict=True)
    minimum = int(data_cfg["minimum_free_bytes"])
    free = _available_bytes(root)
    if free < minimum:
        raise RuntimeContractError(
            f"dataset filesystem free bytes {free} below {minimum}"
        )
    if config_path.name.lower() == "latest.yaml":
        raise RuntimeContractError("implicit latest config names are forbidden")
    return {"global_batch_size": global_batch, "data_free_bytes": free}


def _dataset_preflight_local(
    config: Mapping[str, Any],
) -> tuple[DatasetContract, DatasetSeal, dict[str, Any]]:
    data_cfg = config["data"]
    root = Path(data_cfg["root"]).resolve(strict=True)
    receipt_relative = str(data_cfg["seal_receipt"])
    report = verify_dataset_seal(root, receipt_relative)
    if not report["pass"]:
        raise RuntimeContractError(
            "dataset seal failed:\n" + "\n".join(report["errors"])
        )
    receipt = load_seal(resolve_regular_file(root, receipt_relative))
    expected_receipt = str(data_cfg["seal_receipt_sha256"])
    if receipt.sha256 != expected_receipt:
        raise RuntimeContractError(
            f"dataset receipt {receipt.sha256} != configured {expected_receipt}"
        )
    contract_relative = str(data_cfg["contract"])
    contract = load_contract(resolve_regular_file(root, contract_relative))
    if contract.schema != DATASET_SCHEMA:
        raise RuntimeContractError("dataset contract schema mismatch")
    expected_contract = str(data_cfg["contract_sha256"])
    if contract.sha256 != expected_contract:
        raise RuntimeContractError(
            f"dataset contract {contract.sha256} != configured {expected_contract}"
        )
    model = config_from_mapping(dict(config["model"]))
    representation = (
        contract.T,
        contract.P,
        contract.K,
        contract.token_dim,
        contract.task_dim,
        contract.num_views,
        contract.max_action_groups,
        contract.max_action_dim,
        contract.action_substeps,
        contract.max_aux_tokens,
        contract.aux_dim,
        contract.max_aux_type_id,
    )
    expected_representation = (
        model.T,
        model.P,
        model.K,
        model.token_dim,
        model.task_dim,
        model.num_views,
        model.max_action_groups,
        model.max_action_dim,
        model.action_substeps,
        model.max_aux_tokens,
        model.aux_dim,
        model.max_aux_type_id,
    )
    if representation != expected_representation:
        raise RuntimeContractError(
            f"model/data representation mismatch {expected_representation} != "
            f"{representation}"
        )
    if tuple(contract.source_order) != tuple(data_cfg["source_order"]):
        raise RuntimeContractError("configured source order differs from seal")
    weights = {str(k): int(v) for k, v in data_cfg["source_weights"].items()}
    if weights != contract.source_weights:
        raise RuntimeContractError("configured source weights differ from contract")
    return contract, receipt, report


def _dataset_preflight(
    config: Mapping[str, Any],
    context: Any,
) -> tuple[DatasetContract, DatasetSeal, dict[str, Any]]:
    payload: list[Any] = [None]
    if context.is_rank0:
        try:
            contract, receipt, report = _dataset_preflight_local(config)
            payload[0] = {
                "ok": True,
                "contract": contract.as_dict(),
                "receipt": receipt.as_dict(),
                "report": report,
            }
        except Exception as exc:
            payload[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(payload, src=0)
    if not payload[0]["ok"]:
        raise RuntimeContractError(f"rank0 dataset preflight failed: {payload[0]}")
    return (
        DatasetContract.from_mapping(payload[0]["contract"]),
        DatasetSeal.from_mapping(payload[0]["receipt"]),
        payload[0]["report"],
    )


def _code_preflight(
    config: Mapping[str, Any],
    context: Any,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_rank0:
        try:
            receipt = verify_code_receipt(
                Path(config["run"]["code_receipt_path"]),
                expected_sha256=str(config["run"]["code_receipt_sha256"]),
                repo_root=repo_root,
            )
            payload[0] = {"ok": True, "receipt": receipt}
        except Exception as exc:
            payload[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    dist.broadcast_object_list(payload, src=0)
    if not payload[0]["ok"]:
        raise RuntimeContractError(f"rank0 code preflight failed: {payload[0]}")
    return dict(payload[0]["receipt"])


def _environment_preflight(
    config: Mapping[str, Any],
    context: Any,
) -> dict[str, Any]:
    try:
        contract_path = Path(config["run"]["environment_contract_path"])
        contract = load_environment_contract(contract_path)
        contract_sha = canonical_sha256(contract)
        if contract_sha != str(
            config["run"]["environment_contract_sha256"]
        ):
            raise RuntimeContractError(
                "configured environment contract SHA does not match file"
            )
        receipt = verify_environment_receipt(
            Path(config["run"]["environment_receipt_path"]),
            expected_sha256=str(
                config["run"]["environment_receipt_sha256"]
            ),
            contract_path=contract_path,
            check_current=True,
        )
        local: dict[str, Any] = {
            "ok": True,
            "rank": context.rank,
            "receipt": receipt,
            "fingerprint_sha256": receipt["environment"][
                "fingerprint_sha256"
            ],
        }
    except Exception as exc:
        local = {
            "ok": False,
            "rank": context.rank,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, local)
    errors = [
        f"rank {item['rank']}: {item['error_type']}: {item['error']}"
        for item in gathered
        if not item["ok"]
    ]
    if errors:
        raise RuntimeContractError(
            "distributed environment preflight failed:\n" + "\n".join(errors)
        )
    fingerprints = {item["fingerprint_sha256"] for item in gathered}
    receipts = {canonical_sha256(item["receipt"]) for item in gathered}
    if len(fingerprints) != 1 or len(receipts) != 1:
        raise RuntimeContractError(
            "software environment or receipt differs across ranks"
        )
    return dict(gathered[0]["receipt"])


def _build_dataset(
    config: Mapping[str, Any],
    contract: Any,
    *,
    split: str,
) -> Native5BMixedDataset:
    data_cfg = config["data"]
    loader_cfg = WindowLoaderConfig(
        rgb_decode_indices=tuple(config["model"]["rgb_decode_indices"]),
        memory_slots=int(data_cfg["memory_slots"]),
        memory_stride_frames=int(data_cfg["memory_stride_frames"]),
        row_group_cache_size=int(data_cfg["row_group_cache_size"]),
        task_cache_size=int(data_cfg["task_cache_size"]),
        strict_shapes=True,
    )
    sources = [
        (
            source_name,
            Native5BSourceDataset(
                Path(data_cfg["root"]),
                contract,
                source_name=source_name,
                split=split,
                config=loader_cfg,
            ),
        )
        for source_name in contract.source_order
    ]
    return Native5BMixedDataset(sources)


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.to(device, non_blocking=True)
        else:
            result[name] = value
    return result


def _forward(
    model: NativeWM3D5B,
    batch: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    indices = batch["rgb_frame_indices"]
    if indices.ndim != 2 or not bool((indices == indices[0]).all()):
        raise RuntimeContractError("RGB supervision indices drifted within a batch")
    kwargs: dict[str, Any] = {
        "world_tokens": batch["world_tokens"],
        "view_mask": batch["view_mask"],
        "task_embedding": batch["task_embedding"],
        "context_action_values": batch["context_action_values"],
        "context_action_dim_mask": batch["context_action_dim_mask"],
        "future_factual_action_values": batch["future_factual_action_values"],
        "future_factual_action_dim_mask": batch[
            "future_factual_action_dim_mask"
        ],
        "action_group_ids": batch["action_group_ids"],
        "action_group_mask": batch["action_group_mask"],
        "embodiment_ids": batch["embodiment_ids"],
        "memory_tokens": batch["memory_tokens"],
        "memory_mask": batch["memory_mask"],
        "rgb_frame_indices": indices[0].tolist(),
    }
    if "aux_tokens" in batch:
        kwargs["aux_tokens"] = batch["aux_tokens"]
        kwargs["aux_mask"] = batch["aux_mask"]
    return model(**kwargs)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o640)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def _validate(
    model: NativeWM3D5B,
    dataset: Native5BMixedDataset,
    config: Mapping[str, Any],
    context: Any,
    loss_cfg: Native5BLossConfig,
) -> dict[str, float]:
    train_cfg = config["train"]
    count = int(train_cfg["validation_steps"])
    sampler = StepAddressedBatchSampler(
        dataset.source_spans,
        dataset.source_names,
        config["data"]["source_weights"],
        world_size=context.world_size,
        rank=context.rank,
        micro_batch_size=int(train_cfg["micro_batch_size"]),
        gradient_accumulation=1,
        start_optimizer_step=0,
        num_optimizer_steps=count,
        seed=int(train_cfg["validation_seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=True,
        persistent_workers=False,
    )
    totals: dict[str, torch.Tensor] = {}
    model.eval()
    for cpu_batch in loader:
        batch = _batch_to_device(cpu_batch, context.device)
        output = _forward(model, batch)
        losses = native5b_loss(output, batch, loss_cfg)
        for name, value in losses.items():
            totals[name] = totals.get(name, torch.zeros_like(value)) + value
    totals = {name: value / count for name, value in totals.items()}
    result = reduce_metrics(totals)
    model.train()
    return result


def main() -> None:
    args = parse_args()
    context = initialize_distributed()
    try:
        config_path = args.config.resolve(strict=True)
        config = _read_config(config_path)
        repo_root = Path(__file__).resolve().parents[2]
        boundary_paths = [
            config_path,
            Path(__file__).resolve(),
            repo_root / "wm3d_v3/models/native5b.py",
            repo_root / "wm3d_v3/data/scale5b_dataset.py",
            repo_root / "wm3d_v3/data/scale5b_sampler.py",
            repo_root / "wm3d_v3/training/scale5b_loss.py",
            repo_root / "wm3d_v3/training/scale5b_runtime.py",
            repo_root / "wm3d_v3/training/scale5b_checkpoint.py",
            repo_root / "wm3d_v3/training/scale5b_config.py",
            repo_root / "wm3d_v3/training/scale5b_environment.py",
            Path(config["run"]["environment_contract_path"]),
        ]
        assert_v7_native_dependency_boundary(boundary_paths)
        environment_receipt = _environment_preflight(config, context)
        code_receipt = _code_preflight(config, context, repo_root=repo_root)
        hardware_report = _validate_config(
            config, world_size=context.world_size, config_path=config_path
        )
        contract, receipt, data_report = _dataset_preflight(config, context)
        config_sha = training_contract_sha256(config)
        if config_sha != str(config["run"]["training_contract_sha256"]):
            raise RuntimeContractError(
                f"training contract SHA {config_sha} != configured "
                f"{config['run']['training_contract_sha256']}"
            )
        preflight = {
            "pass": True,
            "rank": context.rank,
            "world_size": context.world_size,
            "config_sha256": config_sha,
            "dataset_receipt_sha256": receipt.sha256,
            "dataset_report": data_report,
            "code_receipt_sha256": canonical_sha256(code_receipt),
            "environment_receipt_sha256": canonical_sha256(
                environment_receipt
            ),
            "environment_fingerprint_sha256": environment_receipt[
                "environment"
            ]["fingerprint_sha256"],
            **hardware_report,
        }
        if context.is_rank0:
            print(json.dumps({"preflight": preflight}, sort_keys=True), flush=True)
        if args.preflight_only:
            return

        train_cfg = config["train"]
        run_cfg = config["run"]
        seed = int(train_cfg["seed"])
        _configure_reproducibility(seed, context.rank)
        with torch.device(context.device):
            model = NativeWM3D5B(config_from_mapping(dict(config["model"])))
        parameter_counts = verify_parameter_budget(model)
        mesh = apply_fsdp2(
            model,
            context,
            shard_degree=int(config["distributed"]["shard_degree"]),
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            reshard_after_forward=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["optimizer"]["peak_lr"]),
            betas=tuple(float(v) for v in config["optimizer"]["betas"]),
            eps=float(config["optimizer"]["eps"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            foreach=False,
        )
        initialize_adamw_state(optimizer)
        output_root = Path(run_cfg["output_root"])
        run_contract_path = output_root / "run_contract.json"
        run_contract = {
            "schema": RUN_CONTRACT_SCHEMA,
            "run_name": run_cfg["name"],
            "run_lineage": run_cfg["run_lineage"],
            "training_contract_sha256": config_sha,
            "dataset_receipt_sha256": receipt.sha256,
            "dataset_contract_sha256": contract.sha256,
            "code_receipt_sha256": canonical_sha256(code_receipt),
            "environment_contract_sha256": run_cfg[
                "environment_contract_sha256"
            ],
            "environment_receipt_sha256": canonical_sha256(
                environment_receipt
            ),
            "environment_fingerprint_sha256": environment_receipt[
                "environment"
            ]["fingerprint_sha256"],
            "parameter_counts": parameter_counts,
            "world_size": context.world_size,
            "shard_degree": int(config["distributed"]["shard_degree"]),
            "mesh": str(mesh),
            "initial_seed": seed,
        }
        run_status: list[Any] = [None]
        if context.is_rank0:
            try:
                if output_root.exists() and output_root.is_symlink():
                    raise RuntimeContractError("output root may not be a symlink")
                output_root.mkdir(parents=True, exist_ok=True)
                if run_contract_path.exists():
                    existing = json.loads(
                        run_contract_path.read_text(encoding="utf-8")
                    )
                    if existing != run_contract:
                        raise RuntimeContractError(
                            "existing run contract does not match"
                        )
                else:
                    if any(output_root.iterdir()):
                        raise RuntimeContractError(
                            "non-empty output root has no run contract"
                        )
                    atomic_write_json(
                        run_contract_path, run_contract, exclusive=True
                    )
                run_status[0] = {"ok": True}
            except Exception as exc:
                run_status[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(run_status, src=0)
        if not run_status[0]["ok"]:
            raise RuntimeContractError(
                f"rank0 output contract precheck failed: {run_status[0]}"
            )

        checkpoint_root = output_root / "checkpoints"
        checkpoint_manager = Native5BCheckpointManager(checkpoint_root)
        start_step = 0
        if args.resume is not None:
            match = re.fullmatch(r"step_([0-9]{8})", args.resume.name)
            if match is None:
                raise RuntimeContractError(
                    "resume must be an explicit step_XXXXXXXX directory"
                )
            expected_step = int(match.group(1))
            metadata = checkpoint_manager.load(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                expected=ResumeExpectations(
                    step=expected_step,
                    run_lineage=str(run_cfg["run_lineage"]),
                    config_sha256=config_sha,
                    dataset_receipt_sha256=receipt.sha256,
                    world_size=context.world_size,
                    shard_degree=int(config["distributed"]["shard_degree"]),
                    allow_topology_reshard=False,
                ),
            )
            start_step = int(metadata["step"])
        total_steps = int(train_cfg["total_steps"])
        if start_step >= total_steps:
            raise RuntimeContractError("resume checkpoint is already at/after total_steps")

        train_dataset = _build_dataset(config, contract, split="train")
        val_dataset = _build_dataset(config, contract, split="val")
        sampler = StepAddressedBatchSampler(
            train_dataset.source_spans,
            train_dataset.source_names,
            config["data"]["source_weights"],
            world_size=context.world_size,
            rank=context.rank,
            micro_batch_size=int(train_cfg["micro_batch_size"]),
            gradient_accumulation=int(train_cfg["gradient_accumulation"]),
            start_optimizer_step=start_step,
            num_optimizer_steps=total_steps - start_step,
            seed=seed,
        )
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=True,
            persistent_workers=bool(config["data"]["persistent_workers"]),
            prefetch_factor=int(config["data"]["prefetch_factor"]),
        )
        iterator = iter(loader)
        loss_cfg = Native5BLossConfig.from_mapping(config["loss"])
        dist.barrier()

        log_path = output_root / "train_metrics.jsonl"
        checkpoint_steps = _strict_checkpoint_steps(config)
        grad_accumulation = int(train_cfg["gradient_accumulation"])
        log_every = int(train_cfg["log_every"])
        validate_every = int(train_cfg["validate_every"])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        last_time = time.monotonic()
        for step in range(start_step, total_steps):
            lr = wsd_learning_rate(
                step,
                total_steps=total_steps,
                warmup_steps=int(config["schedule"]["warmup_steps"]),
                stable_fraction=float(config["schedule"]["stable_fraction"]),
                peak_lr=float(config["optimizer"]["peak_lr"]),
                min_lr=float(config["optimizer"]["min_lr"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            accumulated: dict[str, torch.Tensor] = {}
            source_id: int | None = None
            for micro_step in range(grad_accumulation):
                set_gradient_sync(model, micro_step == grad_accumulation - 1)
                cpu_batch = next(iterator)
                batch = _batch_to_device(cpu_batch, context.device)
                unique_sources = torch.unique(batch["source_id"])
                if unique_sources.numel() != 1:
                    raise RuntimeContractError("micro-batch mixed multiple sources")
                current_source = int(unique_sources.item())
                if source_id is None:
                    source_id = current_source
                elif source_id != current_source:
                    raise RuntimeContractError(
                        "gradient-accumulation group mixed multiple sources"
                    )
                output = _forward(model, batch)
                losses = native5b_loss(output, batch, loss_cfg)
                (losses["total"] / grad_accumulation).backward()
                for name, value in losses.items():
                    accumulated[name] = accumulated.get(
                        name, torch.zeros_like(value)
                    ) + value.detach() / grad_accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg["gradient_clip"])
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_step = step + 1
            if completed_step % log_every == 0 or completed_step == total_steps:
                metrics = reduce_metrics(accumulated)
                elapsed = time.monotonic() - last_time
                last_time = time.monotonic()
                grad_norm_mean = float(all_reduce_mean(grad_norm).cpu())
                if context.is_rank0:
                    record = {
                        "step": completed_step,
                        "lr": lr,
                        "source_id": source_id,
                        "seconds_per_log_interval": elapsed,
                        "grad_norm": grad_norm_mean,
                        **metrics,
                    }
                    print(json.dumps(record, sort_keys=True), flush=True)
                    _append_jsonl(log_path, record)
            if validate_every > 0 and completed_step % validate_every == 0:
                validation = _validate(
                    model, val_dataset, config, context, loss_cfg
                )
                if context.is_rank0:
                    record = {"step": completed_step, "validation": validation}
                    print(json.dumps(record, sort_keys=True), flush=True)
                    _append_jsonl(log_path, record)
            if completed_step in checkpoint_steps:
                checkpoint_manager.save(
                    step=completed_step,
                    model=model,
                    optimizer=optimizer,
                    metadata={
                        "run_name": run_cfg["name"],
                        "run_lineage": run_cfg["run_lineage"],
                        "config_sha256": config_sha,
                        "dataset_receipt_sha256": receipt.sha256,
                        "dataset_contract_sha256": contract.sha256,
                        "environment_receipt_sha256": canonical_sha256(
                            environment_receipt
                        ),
                        "initial_seed": seed,
                        "shard_degree": int(
                            config["distributed"]["shard_degree"]
                        ),
                        "global_batch_size": int(
                            train_cfg["global_batch_size"]
                        ),
                        "parameter_count": parameter_counts["total"],
                    },
                )
        dist.barrier()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
