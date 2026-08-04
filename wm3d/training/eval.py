"""Distributed checkpoint evaluation for WM3D.

The report is a correctness/health gate, not a claim that one checkpoint is
better than another. It evaluates explicit RGB, depth, point and grouped-action
outputs and writes a small target/prediction contact sheet on rank zero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from wm3d.data.sampler import StepAddressedBatchSampler
from wm3d.models.wm3d import WM3D, config_from_mapping
from wm3d.training.checkpoint import (
    CheckpointManager,
    ResumeExpectations,
)
from wm3d.training.config import training_contract_sha256
from wm3d.training.loss import WM3DLossConfig, wm3d_loss
from wm3d.training.runtime import (
    apply_fsdp2,
    destroy_distributed,
    initialize_adamw_state,
    initialize_distributed,
    verify_parameter_budget,
)
from wm3d.training.train import (
    _batch_to_device,
    _build_dataset,
    _code_preflight,
    _configure_reproducibility,
    _dataset_preflight,
    _environment_preflight,
    _forward,
    _read_config,
    _validate_config,
)


EVAL_SCHEMA = "wm3d_v7_checkpoint_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    return parser.parse_args()


def _sum(
    values: dict[str, torch.Tensor], name: str, value: torch.Tensor | float
) -> None:
    device = next(iter(values.values())).device
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    values[name] = values.get(name, torch.zeros_like(tensor)) + tensor


def _masked_stats(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=torch.float64)
    while weights.ndim < prediction.ndim:
        weights = weights.unsqueeze(-1)
    weights = torch.broadcast_to(weights, prediction.shape)
    difference = prediction.double() - target.double()
    return (
        (difference.abs() * weights).sum(),
        (difference.square() * weights).sum(),
        (prediction.double() * weights).sum(),
        (prediction.double().square() * weights).sum(),
        weights.sum(),
    )


def _to_uint8(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _preview(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    destination: Path,
) -> None:
    prediction = output["rgb"][0, 0]
    target = batch["target_rgb"][0, 0]
    views = min(int(prediction.shape[0]), 3)
    images = [
        _to_uint8(value)
        for row in (target[:views], prediction[:views])
        for value in row
    ]
    width, height = images[0].size
    canvas = Image.new("RGB", (views * width, 2 * height), color=(0, 0, 0))
    for index, image in enumerate(images[:views]):
        canvas.paste(image, (index * width, 0))
    for index, image in enumerate(images[views:]):
        canvas.paste(image, (index * width, height))
    canvas.save(destination, format="PNG", compress_level=6)


def _atomic_directory(final: Path) -> Path:
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"refusing to overwrite eval output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent)
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("eval steps must be positive")
    context = initialize_distributed()
    temporary: Path | None = None
    try:
        config_path = args.config.resolve(strict=True)
        checkpoint = args.checkpoint.resolve(strict=True)
        match = re.fullmatch(r"step_([0-9]{8})", checkpoint.name)
        if match is None or checkpoint.is_symlink():
            raise ValueError("checkpoint must be an explicit step_XXXXXXXX directory")
        checkpoint_step = int(match.group(1))
        config = _read_config(config_path)
        repo_root = Path(__file__).resolve().parents[2]
        environment = _environment_preflight(config, context)
        code = _code_preflight(config, context, repo_root=repo_root)
        hardware = _validate_config(
            config, world_size=context.world_size, config_path=config_path
        )
        contract, seal, dataset_report = _dataset_preflight(config, context)
        config_sha = training_contract_sha256(config)
        if config_sha != str(config["run"]["training_contract_sha256"]):
            raise ValueError("training contract SHA drift")

        seed = int(config["train"]["validation_seed"])
        _configure_reproducibility(seed, context.rank)
        with torch.device(context.device):
            model = WM3D(config_from_mapping(dict(config["model"])))
        parameter_counts = verify_parameter_budget(model, config["model_budget"])
        apply_fsdp2(
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
            betas=tuple(float(value) for value in config["optimizer"]["betas"]),
            eps=float(config["optimizer"]["eps"]),
            weight_decay=float(config["optimizer"]["weight_decay"]),
            foreach=False,
        )
        initialize_adamw_state(optimizer)
        manager = CheckpointManager(checkpoint.parent)
        metadata = manager.load(
            path=checkpoint,
            model=model,
            optimizer=optimizer,
            expected=ResumeExpectations(
                step=checkpoint_step,
                run_lineage=str(config["run"]["run_lineage"]),
                config_sha256=config_sha,
                dataset_receipt_sha256=seal.sha256,
                world_size=context.world_size,
                shard_degree=int(config["distributed"]["shard_degree"]),
                allow_topology_reshard=False,
            ),
        )

        dataset = _build_dataset(config, contract, split="val")
        sampler = StepAddressedBatchSampler(
            dataset.source_spans,
            dataset.source_names,
            config["data"]["source_weights"],
            world_size=context.world_size,
            rank=context.rank,
            micro_batch_size=int(config["train"]["micro_batch_size"]),
            gradient_accumulation=1,
            start_optimizer_step=0,
            num_optimizer_steps=args.steps,
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=True,
            persistent_workers=False,
        )
        loss_cfg = WM3DLossConfig.from_mapping(config["loss"])
        sums: dict[str, torch.Tensor] = {
            "batches": torch.zeros((), dtype=torch.float64, device=context.device)
        }
        preview_payload: (
            tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]] | None
        ) = None
        model.eval()
        with torch.no_grad():
            for cpu_batch in loader:
                batch = _batch_to_device(cpu_batch, context.device)
                output = _forward(model, batch)
                losses = wm3d_loss(output, batch, loss_cfg)
                _sum(sums, "batches", 1.0)
                for name, value in losses.items():
                    _sum(sums, f"loss/{name}", value.detach().double())

                rgb_indices = output["rgb_frame_indices"].long()
                rgb_mask = batch["target_view_mask"].bool().index_select(1, rgb_indices)
                for prefix, prediction, target, mask in (
                    ("rgb", output["rgb"], batch["target_rgb"], rgb_mask),
                    (
                        "depth",
                        output["depth"],
                        batch["target_depth"],
                        (batch["target_geometry_confidence"] > 0)
                        & batch["target_view_mask"].bool().unsqueeze(-1),
                    ),
                    (
                        "point",
                        output["point"],
                        batch["target_point"],
                        (batch["target_geometry_confidence"] > 0)
                        & batch["target_view_mask"].bool().unsqueeze(-1),
                    ),
                    (
                        "action",
                        output["action_mean"],
                        batch["target_action_values"],
                        batch["target_action_dim_mask"].bool()
                        & batch["action_group_mask"].bool()[:, None, :, None, None],
                    ),
                ):
                    absolute, squared, predicted, predicted_squared, count = (
                        _masked_stats(
                            prediction, target.to(dtype=prediction.dtype), mask
                        )
                    )
                    _sum(sums, f"{prefix}/absolute", absolute)
                    _sum(sums, f"{prefix}/squared", squared)
                    _sum(sums, f"{prefix}/prediction", predicted)
                    _sum(sums, f"{prefix}/prediction_squared", predicted_squared)
                    _sum(sums, f"{prefix}/count", count)
                if context.is_rank0 and preview_payload is None:
                    preview_payload = (output, batch)

        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        report_status: list[Any] = [None]
        if context.is_rank0:
            try:
                batches = float(sums["batches"].item())
                metrics: dict[str, float] = {
                    name: float(value.item() / batches)
                    for name, value in sums.items()
                    if name.startswith("loss/")
                }
                for prefix in ("rgb", "depth", "point", "action"):
                    count = float(sums[f"{prefix}/count"].item())
                    if count <= 0:
                        raise ValueError(f"{prefix} supervision coverage is zero")
                    mae = float(sums[f"{prefix}/absolute"].item() / count)
                    mse = float(sums[f"{prefix}/squared"].item() / count)
                    mean = float(sums[f"{prefix}/prediction"].item() / count)
                    second = float(sums[f"{prefix}/prediction_squared"].item() / count)
                    metrics[f"{prefix}/mae"] = mae
                    metrics[f"{prefix}/mse"] = mse
                    metrics[f"{prefix}/prediction_std"] = math.sqrt(
                        max(0.0, second - mean * mean)
                    )
                    metrics[f"{prefix}/supervised_values"] = count
                metrics["rgb/psnr"] = -10.0 * math.log10(
                    max(metrics["rgb/mse"], 1.0e-12)
                )
                finite = all(math.isfinite(value) for value in metrics.values())
                checks = {
                    "all_metrics_finite": finite,
                    "rgb_supervision_present": metrics["rgb/supervised_values"] > 0,
                    "geometry_supervision_present": metrics["depth/supervised_values"]
                    > 0
                    and metrics["point/supervised_values"] > 0,
                    "action_supervision_present": metrics["action/supervised_values"]
                    > 0,
                    "rgb_prediction_not_constant": metrics["rgb/prediction_std"]
                    > 1.0e-4,
                    "action_prediction_not_constant": metrics["action/prediction_std"]
                    > 1.0e-6,
                }
                output_root = args.output_root.resolve()
                temporary = _atomic_directory(output_root)
                if preview_payload is None:
                    raise RuntimeError("rank0 did not observe an eval batch")
                preview_path = temporary / "rgb_target_top_prediction_bottom.png"
                _preview(*preview_payload, preview_path)
                report = {
                    "schema": EVAL_SCHEMA,
                    "pass": all(checks.values()),
                    "meaning": "correctness_gate_not_comparative_quality_claim",
                    "config": str(config_path),
                    "config_sha256": config_sha,
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_commit_sha256": sha256_file(
                        checkpoint / "COMMITTED.json"
                    ),
                    "world_size": context.world_size,
                    "eval_steps_per_rank": args.steps,
                    "metrics": metrics,
                    "checks": checks,
                    "preview": {
                        "path": preview_path.name,
                        "sha256": sha256_file(preview_path),
                    },
                    "bindings": {
                        "dataset_seal_sha256": seal.sha256,
                        "code_receipt_sha256": canonical_sha256(code),
                        "environment_receipt_sha256": canonical_sha256(environment),
                        "parameter_count": parameter_counts["total"],
                        "checkpoint_metadata": metadata,
                    },
                    "preflight": {
                        "hardware": hardware,
                        "dataset": dataset_report,
                    },
                }
                atomic_write_json(temporary / "report.json", report, exclusive=True)
                os.replace(temporary, output_root)
                temporary = None
                if not report["pass"]:
                    raise RuntimeError(f"eval gate failed: {checks}")
                report_status[0] = {
                    "ok": True,
                    "report": str(output_root / "report.json"),
                }
            except Exception as exc:
                report_status[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(report_status, src=0)
        if not report_status[0]["ok"]:
            raise RuntimeError(f"rank0 eval publication failed: {report_status[0]}")
        if context.is_rank0:
            print(json.dumps({"pass": True, **report_status[0]}, sort_keys=True))
    finally:
        if temporary is not None and context.is_rank0:
            print(f"保留失败 eval 临时目录：{temporary}", flush=True)
        destroy_distributed()


if __name__ == "__main__":
    main()
