#!/usr/bin/env python3
"""Read-only paired action audit for a committed Stage0 DCP checkpoint.

The audit holds observation, task, state history and RGB targets fixed while
comparing the factual K-step action with a physical no-op and a compatible,
distant action from another real validation window. It never trains a model
or changes the sealed runtime/data artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from wm3d.data.direct_raw import DIRECT_RAW_DATA_CLOSURE_SCHEMA
from wm3d.data.step_sampler import StepAddressedBatchSampler
from wm3d.models.direct_vggt_builder import build_direct_vggt_teacher
from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.distributed_checkpoint import DistributedCheckpointManager
from wm3d.training.pretrain import (
    _batch_index_select,
    _batch_to_device,
    _build_mixed_dataset,
    _configure_reproducibility,
    _context_pixel_action_derangement,
    _forward,
    _make_loader,
    _materialize_rgb_flow_targets,
    _resume_expectations,
    _shuffled_future_factual_action,
    _zero_future_factual_action,
)
from wm3d.training.rgb_flow_runtime import (
    FrozenBidirectionalRAFTRuntime,
    raft_config_from_mapping,
)
from wm3d.training.runtime_contract import load_materialized_runtime


SCHEMA = "wm3d_action_owned_transport_checkpoint_audit_v1"


class AuditError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-count", type=int, default=3)
    parser.add_argument("--pair-batch-size", type=int, default=2)
    parser.add_argument("--pairs-per-source", type=int, default=1)
    parser.add_argument("--max-loader-steps", type=int, default=4096)
    parser.add_argument("--minimum-distance", type=float)
    return parser.parse_args()


def load_full_model(model: torch.nn.Module, checkpoint: Path) -> None:
    """Reshard a distributed checkpoint into one unwrapped evaluation model."""

    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        strict=True,
    )
    state = get_model_state_dict(model, options=options)
    dcp.load({"model": state}, checkpoint_id=checkpoint / "distcp")
    incompatible = set_model_state_dict(model, state, options=options)
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise AuditError(
            f"model load incomplete: missing={missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )


def inspect_checkpoint(
    config: Mapping[str, Any],
    runtime_sha: str,
    checkpoint: Path,
) -> tuple[int, Mapping[str, Any]]:
    if re.fullmatch(r"step_[0-9]{8}", checkpoint.name) is None:
        raise AuditError("checkpoint must be an explicit step_XXXXXXXX directory")
    step = int(checkpoint.name.split("_")[1])
    expectations = _resume_expectations(
        config,
        runtime_sha,
        step=step,
        world_size=int(config["runtime_profile"]["expected_world_size"]),
    )
    inspection = DistributedCheckpointManager(checkpoint.parent).inspect_committed(
        path=checkpoint,
        expected=expectations,
    )
    if inspection["resume_mode"] != "exact":
        raise AuditError("paired audit requires the original exact-topology checkpoint")
    return step, inspection


def build_action_variants(
    batch: Mapping[str, torch.Tensor],
    *,
    step: int,
    minimum_distance: float,
) -> tuple[dict[str, Mapping[str, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the production physical-noop and compatible derangement primitives."""

    permutation, valid, distance = _context_pixel_action_derangement(
        batch,
        step=step,
        minimum_distance=minimum_distance,
    )
    return (
        {
            "normal": dict(batch),
            "physical_noop": _zero_future_factual_action(batch),
            "distant_mismatch": _shuffled_future_factual_action(batch, permutation),
        },
        permutation,
        valid,
        distance,
    )


def validate_action_k8_batch(batch: Mapping[str, torch.Tensor]) -> None:
    """Validate fields available before direct-raw VGGT materialization."""

    expected = 8
    for name in (
        "future_factual_fine_action_values",
        "future_factual_fine_action_mask",
        "future_factual_fine_action_dt",
        "future_factual_fine_sample_mask",
        "future_factual_coarse_action_values",
        "future_factual_coarse_action_mask",
    ):
        if name not in batch or batch[name].ndim < 2 or batch[name].shape[1] != expected:
            raise AuditError(f"{name} does not preserve the real K8 horizon")
    if "rgb_frame_indices" in batch:
        expected_indices = torch.arange(
            expected,
            dtype=batch["rgb_frame_indices"].dtype,
            device=batch["rgb_frame_indices"].device,
        )
        if not bool((batch["rgb_frame_indices"] == expected_indices).all()):
            raise AuditError("RGB frame indices are not the canonical K8 order")


def validate_materialized_k8_batch(batch: Mapping[str, torch.Tensor]) -> None:
    """Validate action and VGGT/RGB targets after materialization."""

    validate_action_k8_batch(batch)
    expected = 8
    for name in (
        "target_tokens",
        "target_token_mask",
        "target_rgb",
        "target_rgb_mask",
    ):
        if name not in batch or batch[name].ndim < 2 or batch[name].shape[1] != expected:
            raise AuditError(f"{name} does not preserve the real K8 horizon")


def plan_source_candidate_steps(
    sampler: StepAddressedBatchSampler,
    *,
    max_steps: int,
    source_count: int,
) -> dict[str, list[int]]:
    """Scan deterministic step addresses without reading dataset samples."""

    selected: dict[str, list[int]] = {}
    for optimizer_step in range(max_steps):
        source_name = str(sampler.describe_step(optimizer_step)["source_name"])
        selected.setdefault(source_name, []).append(optimizer_step)
    if len(selected) < source_count:
        raise AuditError(
            f"schedule exposes only {len(selected)} sources in {max_steps} steps"
        )
    return selected


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = torch.broadcast_to(mask, value.shape).to(dtype=value.dtype)
    if not bool(weight.bool().any()):
        raise AuditError("metric has no valid support")
    return (value * weight).sum() / weight.sum()


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(value.float().square(), mask).sqrt()


def _masked_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = torch.broadcast_to(mask, left.shape).to(dtype=left.dtype)
    if not bool(weight.bool().any()):
        raise AuditError("cosine metric has no valid support")
    left = left.float() * weight
    right = right.float() * weight
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return (left * right).sum() / denominator.clamp_min(1.0e-12)


def _rgb_delta(
    video: torch.Tensor,
    context: torch.Tensor,
    target_mask: torch.Tensor,
    context_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = torch.cat((context[:, None], video), dim=1).float()
    delta = sequence[:, 1:] - sequence[:, :-1]
    context_slot = context_mask[:, None, :, None, None, None].bool()
    sequence_mask = torch.cat((context_slot, target_mask.bool()), dim=1)
    valid = sequence_mask[:, 1:] & sequence_mask[:, :-1]
    return delta, valid


def _flow_at_teacher_grid(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.shape[-2:] == target.shape[-2:]:
        return prediction.float()
    count = int(prediction.shape[0] * prediction.shape[1] * prediction.shape[2])
    return F.interpolate(
        prediction.float().reshape(count, 2, *prediction.shape[-2:]),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=True,
    ).reshape_as(target)


def variant_metrics(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    motion_threshold: float,
) -> dict[str, float]:
    required_output = (
        "pred_tokens",
        "rgb",
        "policy_action_raw",
        "action_free_pred_tokens",
    )
    missing = [name for name in required_output if name not in output]
    if missing:
        raise AuditError("model output misses " + ", ".join(missing))

    prediction_tokens = output["pred_tokens"].float()
    target_tokens = batch["target_tokens"].float()
    token_valid = batch["target_token_mask"].bool()[..., None]
    if prediction_tokens.shape != target_tokens.shape:
        raise AuditError("predicted and target P64 tensors differ in shape")
    token_delta = prediction_tokens[:, 1:] - prediction_tokens[:, :-1]
    target_token_delta = target_tokens[:, 1:] - target_tokens[:, :-1]
    token_delta_valid = (
        batch["target_token_mask"][:, 1:]
        & batch["target_token_mask"][:, :-1]
    )[..., None]

    prediction_rgb = output["rgb"].float().clamp(0.0, 1.0)
    target_rgb = batch["target_rgb"].float().clamp(0.0, 1.0)
    context_rgb = batch["context_rgb"].float().clamp(0.0, 1.0)
    rgb_valid = (
        batch["target_rgb_mask"].bool()
        & batch["context_rgb_mask"].bool()[:, None, :, None, None, None]
    )
    if prediction_rgb.shape != target_rgb.shape:
        raise AuditError("predicted and target RGB tensors differ in shape")
    context = context_rgb[:, None].expand_as(target_rgb)
    target_motion = (
        (target_rgb - context).abs().mean(dim=3, keepdim=True) > motion_threshold
    ) & rgb_valid
    target_static = (~target_motion) & rgb_valid
    if not bool(target_motion.any()) or not bool(target_static.any()):
        raise AuditError("paired sample lacks RGB motion or static support")
    prediction_delta, delta_valid = _rgb_delta(
        prediction_rgb,
        context_rgb,
        rgb_valid,
        batch["context_rgb_mask"],
    )
    target_delta, _ = _rgb_delta(
        target_rgb,
        context_rgb,
        rgb_valid,
        batch["context_rgb_mask"],
    )

    values = {
        "p64_error_rms": _masked_rms(
            prediction_tokens - target_tokens, token_valid
        ),
        "p64_temporal_delta_rms": _masked_rms(token_delta, token_delta_valid),
        "p64_target_temporal_delta_rms": _masked_rms(
            target_token_delta, token_delta_valid
        ),
        "p64_temporal_direction_cosine": _masked_cosine(
            token_delta, target_token_delta, token_delta_valid
        ),
        "rgb_l1": _masked_mean((prediction_rgb - target_rgb).abs(), rgb_valid),
        "rgb_motion_l1": _masked_mean(
            (prediction_rgb - target_rgb).abs(), target_motion
        ),
        "rgb_static_l1": _masked_mean(
            (prediction_rgb - target_rgb).abs(), target_static
        ),
        "rgb_frame_delta_rms": _masked_rms(prediction_delta, delta_valid),
        "rgb_target_frame_delta_rms": _masked_rms(target_delta, delta_valid),
        "rgb_frame_delta_direction_cosine": _masked_cosine(
            prediction_delta, target_delta, delta_valid
        ),
        "rgb_motion_from_context_rms": _masked_rms(
            prediction_rgb - context, rgb_valid
        ),
        "rgb_copy_last_l1": _masked_mean((context - target_rgb).abs(), rgb_valid),
        "rgb_motion_fraction": _masked_mean(target_motion.float(), rgb_valid),
    }
    # Separate forecast-to-forecast motion from the context-to-first jump.
    # A constant but wrong forecast can score nonzero on the latter alone.
    future_pred = prediction_delta[:, 1:]
    future_target = target_delta[:, 1:]
    future_valid = delta_valid[:, 1:]
    values.update({
        "rgb_future_frame_delta_rms": _masked_rms(future_pred, future_valid),
        "rgb_target_future_frame_delta_rms": _masked_rms(future_target, future_valid),
        "rgb_future_frame_delta_error_rms": _masked_rms(
            future_pred - future_target, future_valid),
        "rgb_future_frame_delta_direction_cosine": _masked_cosine(
            future_pred, future_target, future_valid),
    })
    for horizon in range(prediction_rgb.shape[1]):
        valid_h = delta_valid[:, horizon]
        values[f"rgb_h{horizon + 1}_l1"] = _masked_mean(
            (prediction_rgb[:, horizon] - target_rgb[:, horizon]).abs(),
            rgb_valid[:, horizon])
        values[f"rgb_h{horizon + 1}_delta_rms"] = _masked_rms(
            prediction_delta[:, horizon], valid_h)
        values[f"rgb_h{horizon + 1}_target_delta_rms"] = _masked_rms(
            target_delta[:, horizon], valid_h)
        values[f"rgb_h{horizon + 1}_delta_direction_cosine"] = _masked_cosine(
            prediction_delta[:, horizon], target_delta[:, horizon], valid_h)
    if "rgb_flow_pixels" in output:
        flow_target = batch["rgb_flow_target_pixels"].float()
        flow_prediction = _flow_at_teacher_grid(
            output["rgb_flow_pixels"], flow_target
        )
        slot_valid = rgb_valid[..., 0, 0, 0]
        flow_valid = torch.broadcast_to(
            slot_valid[..., None, None, None],
            batch["rgb_disocclusion_target"].shape,
        )
        flow_valid = (
            flow_valid
            & torch.isfinite(flow_target).all(dim=3, keepdim=True)
            & torch.isfinite(batch["rgb_disocclusion_target"])
            & (batch["rgb_disocclusion_target"] < 0.5)
        )
        flow_vector_valid = torch.broadcast_to(flow_valid, flow_target.shape)
        flow_error = flow_prediction - flow_target
        values.update({
            "flow_epe_pixels": _masked_mean(
                flow_error.square().sum(dim=3, keepdim=True).sqrt(), flow_valid
            ),
            "flow_prediction_magnitude_pixels": _masked_mean(
                flow_prediction.square().sum(dim=3, keepdim=True).sqrt(), flow_valid
            ),
            "flow_target_magnitude_pixels": _masked_mean(
                flow_target.square().sum(dim=3, keepdim=True).sqrt(), flow_valid
            ),
            "flow_direction_cosine": _masked_cosine(
                flow_prediction, flow_target, flow_vector_valid
            ),
        })
    result = {name: float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise AuditError("paired audit produced a non-finite metric")
    return result


def _response_rms(
    factual: Mapping[str, torch.Tensor],
    control: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    rgb_mask = batch["target_rgb_mask"].bool()
    values = {
        "p64_response_rms": _masked_rms(
            factual["pred_tokens"].float() - control["pred_tokens"].float(),
            batch["target_token_mask"].bool()[..., None],
        ),
        "rgb_response_rms": _masked_rms(
            factual["rgb"].float() - control["rgb"].float(), rgb_mask
        ),
    }
    if "rgb_flow_pixels" in factual and "rgb_flow_pixels" in control:
        values.update({
            "flow_response_rms_pixels": _masked_rms(
                factual["rgb_flow_pixels"].float()
                - control["rgb_flow_pixels"].float(),
                rgb_mask,
            ),
        })
    return {name: float(value) for name, value in values.items()}


def _policy_invariants(
    factual: Mapping[str, torch.Tensor],
    control: Mapping[str, torch.Tensor],
) -> dict[str, bool]:
    return {
        "policy_action_raw_equal": torch.equal(
            factual["policy_action_raw"], control["policy_action_raw"]
        ),
        "action_free_tokens_equal": torch.equal(
            factual["action_free_pred_tokens"],
            control["action_free_pred_tokens"],
        ),
    }


def _mean_dict(values: list[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise AuditError("cannot aggregate an empty metric list")
    keys = tuple(values[0])
    if any(tuple(value) != keys for value in values):
        raise AuditError("metric dictionaries differ across paired samples")
    return {
        key: sum(float(value[key]) for value in values) / len(values)
        for key in keys
    }


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise AuditError("paired audit has no records")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["source_name"]), []).append(record)
    source_summaries: dict[str, dict[str, Any]] = {}
    for source_name, source_records in grouped.items():
        source_variants = {
            label: _mean_dict(
                [record["variants"][label] for record in source_records]
            )
            for label in ("normal", "physical_noop", "distant_mismatch")
        }
        source_responses = {
            label: _mean_dict(
                [record["responses"][label] for record in source_records]
            )
            for label in ("physical_noop", "distant_mismatch")
        }
        source_summaries[source_name] = {
            "pair_count": len(source_records),
            "variants": source_variants,
            "responses": source_responses,
            "gains": {
                "rgb_l1_normal_vs_physical_noop": (
                    source_variants["physical_noop"]["rgb_l1"]
                    - source_variants["normal"]["rgb_l1"]
                ),
                "rgb_l1_normal_vs_distant_mismatch": (
                    source_variants["distant_mismatch"]["rgb_l1"]
                    - source_variants["normal"]["rgb_l1"]
                ),
                "p64_error_normal_vs_physical_noop": (
                    source_variants["physical_noop"]["p64_error_rms"]
                    - source_variants["normal"]["p64_error_rms"]
                ),
                "p64_error_normal_vs_distant_mismatch": (
                    source_variants["distant_mismatch"]["p64_error_rms"]
                    - source_variants["normal"]["p64_error_rms"]
                ),
            },
        }
    variants = {
        label: _mean_dict(
            [summary["variants"][label] for summary in source_summaries.values()]
        )
        for label in ("normal", "physical_noop", "distant_mismatch")
    }
    responses = {
        label: _mean_dict(
            [summary["responses"][label] for summary in source_summaries.values()]
        )
        for label in ("physical_noop", "distant_mismatch")
    }
    invariants = {
        label: {
            key: all(record["invariants"][label][key] for record in records)
            for key in ("policy_action_raw_equal", "action_free_tokens_equal")
        }
        for label in ("physical_noop", "distant_mismatch")
    }
    gains = {
        "rgb_l1_normal_vs_physical_noop": (
            variants["physical_noop"]["rgb_l1"]
            - variants["normal"]["rgb_l1"]
        ),
        "rgb_l1_normal_vs_distant_mismatch": (
            variants["distant_mismatch"]["rgb_l1"]
            - variants["normal"]["rgb_l1"]
        ),
        "p64_error_normal_vs_physical_noop": (
            variants["physical_noop"]["p64_error_rms"]
            - variants["normal"]["p64_error_rms"]
        ),
        "p64_error_normal_vs_distant_mismatch": (
            variants["distant_mismatch"]["p64_error_rms"]
            - variants["normal"]["p64_error_rms"]
        ),
    }
    return {
        "pair_count": len(records),
        "source_count": len(source_summaries),
        "variants": variants,
        "responses": responses,
        "gains": gains,
        "positive_source_counts": {
            key: sum(
                summary["gains"][key] > 0.0
                for summary in source_summaries.values()
            )
            for key in gains
        },
        "per_source": source_summaries,
        "invariants": invariants,
        "all_policy_invariants_passed": all(
            value for branch in invariants.values() for value in branch.values()
        ),
    }


def _single(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    selected = torch.tensor([index], device=batch["source_id"].device)
    return _batch_index_select(
        batch,
        selected,
        batch_size=int(batch["source_id"].shape[0]),
    )


def _forward_eval(
    model: NativeWorldModel, batch: Mapping[str, Any]
) -> Mapping[str, torch.Tensor]:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return _forward(model, batch, appearance_teacher_ratio=0.0)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise AuditError("output receipt must be new")
    if args.source_count <= 0:
        raise AuditError("source-count must be positive; fewer than three is limited coverage, not a cross-source gate")
    if args.pair_batch_size < 2 or args.pairs_per_source <= 0:
        raise AuditError("paired audit requires a batch of at least two real windows")
    if args.max_loader_steps < args.source_count:
        raise AuditError("max loader steps cannot cover the requested sources")

    config, runtime_sha = load_materialized_runtime(args.runtime.resolve(strict=True))
    checkpoint = args.checkpoint.resolve(strict=True)
    checkpoint_step, inspection = inspect_checkpoint(
        config, runtime_sha, checkpoint
    )
    runtime = config["runtime_profile"]
    seed = int(runtime["train"]["validation_seed"]) if args.seed is None else args.seed
    minimum_distance = (
        float(config["objective_profile"]["objective"][
            "context_pixel_action_negative_min_distance"
        ])
        if args.minimum_distance is None
        else float(args.minimum_distance)
    )
    if minimum_distance < 0.0:
        raise AuditError("minimum action distance must be non-negative")

    _configure_reproducibility(seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise AuditError("the production checkpoint audit requires a CUDA device")
    torch.cuda.set_device(device)
    with torch.device(device):
        model = build_world_model(config["model_profile"])
    if not isinstance(model, NativeWorldModel):
        raise AuditError("paired checkpoint audit requires NativeWorldModel")
    if (
        model.cfg.K != 8
        or tuple(model.cfg.rgb_decode_indices) != tuple(range(8))
        or not model.cfg.rgb_action_owned_transport
        or model.cfg.rgb_original_v7_context
    ):
        raise AuditError("model is not the full-K8 action-owned transport contract")
    load_full_model(model, checkpoint)
    model.eval()

    dataset, profile = _build_mixed_dataset(
        config,
        split=args.split,
        device=device,
        rank=0,
    )
    address_sampler = StepAddressedBatchSampler(
        dataset.source_spans,
        dataset.source_names,
        {
            name: profile.source_weights[name]
            for name in dataset.source_names
        },
        rank=0,
        world_size=1,
        start_optimizer_step=0,
        num_optimizer_steps=args.max_loader_steps,
        seed=seed,
        gradient_accumulation=1,
        micro_batch_size=args.pair_batch_size,
        source_episode_spans=getattr(dataset, "source_episode_spans", None),
    )
    candidate_steps = plan_source_candidate_steps(
        address_sampler,
        max_steps=args.max_loader_steps,
        source_count=args.source_count,
    )
    input_adapter = None
    if config["data_closure"].get("schema") == DIRECT_RAW_DATA_CLOSURE_SCHEMA:
        input_adapter = build_direct_vggt_teacher(config, device=device)
        input_adapter.eval()
    flow_mapping = runtime["train"].get("rgb_flow_teacher")
    if flow_mapping is None:
        raise AuditError("action-owned transport audit requires sealed RGB flow teacher")
    flow_teacher = FrozenBidirectionalRAFTRuntime(
        raft_config_from_mapping(flow_mapping), device=device
    )

    records: list[dict[str, Any]] = []
    completed_sources: set[int] = set()
    attempted_sources: dict[int, int] = {}
    motion_threshold = float(
        config["objective_profile"]["objective"]["rgb_motion_threshold"]
    )
    for planned_source, optimizer_steps in candidate_steps.items():
        accepted = 0
        for loader_step in optimizer_steps:
            loader = _make_loader(
                dataset,
                profile,
                runtime,
                rank=0,
                world_size=1,
                start_step=loader_step,
                num_steps=1,
                seed=seed,
                gradient_accumulation=1,
                micro_batch_size=args.pair_batch_size,
            )
            cpu_batch = next(iter(loader))
            validate_action_k8_batch(cpu_batch)
            source_ids = torch.unique(cpu_batch["source_id"])
            if source_ids.numel() != 1:
                raise AuditError("step-addressed validation batch mixed sources")
            source_id = int(source_ids.item())
            source_name = str(profile.source_order[source_id])
            if source_name != planned_source:
                raise AuditError("candidate loader did not preserve its planned source")
            attempted_sources[source_id] = attempted_sources.get(source_id, 0) + 1
            _, _, preliminary_valid, _ = build_action_variants(
                cpu_batch,
                step=checkpoint_step + loader_step,
                minimum_distance=minimum_distance,
            )
            if not bool(preliminary_valid.any()):
                continue
            if input_adapter is not None:
                cpu_batch = input_adapter.materialize(cpu_batch)
            validate_materialized_k8_batch(cpu_batch)
            batch = _batch_to_device(cpu_batch, device)
            batch = _materialize_rgb_flow_targets(batch, flow_teacher)
            variants, permutation, valid, distance = build_action_variants(
                batch,
                step=checkpoint_step + loader_step,
                minimum_distance=minimum_distance,
            )
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
            for index in valid_indices:
                selected_batches = {
                    label: _single(value, int(index))
                    for label, value in variants.items()
                }
                target_batch = selected_batches["normal"]
                target_context = target_batch["context_rgb"][:, None]
                target_motion = (
                    target_batch["target_rgb"].float() - target_context.float()
                ).abs().mean(dim=3, keepdim=True) > motion_threshold
                target_motion &= target_batch["target_rgb_mask"].bool()
                if not bool(target_motion.any()):
                    continue
                outputs = {
                    label: _forward_eval(model, value)
                    for label, value in selected_batches.items()
                }
                invariant_noop = _policy_invariants(
                    outputs["normal"], outputs["physical_noop"]
                )
                invariant_wrong = _policy_invariants(
                    outputs["normal"], outputs["distant_mismatch"]
                )
                if not all((*invariant_noop.values(), *invariant_wrong.values())):
                    raise AuditError(
                        "future candidate action leaked into policy/action-free"
                    )
                paired_index = int(permutation[int(index)].item())
                records.append(
                    {
                        "source_id": source_id,
                        "source_name": source_name,
                        "sample_index": int(
                            batch["sample_index"][int(index)].item()
                        ),
                        "paired_sample_index": int(
                            batch["sample_index"][paired_index].item()
                        ),
                        "loader_step": loader_step,
                        "pair_distance": float(distance[int(index)].item()),
                        "variants": {
                            label: variant_metrics(
                                output,
                                target_batch,
                                motion_threshold=motion_threshold,
                            )
                            for label, output in outputs.items()
                        },
                        "responses": {
                            label: _response_rms(
                                outputs["normal"], outputs[label], target_batch
                            )
                            for label in ("physical_noop", "distant_mismatch")
                        },
                        "invariants": {
                            "physical_noop": invariant_noop,
                            "distant_mismatch": invariant_wrong,
                        },
                    }
                )
                accepted += 1
                if accepted >= args.pairs_per_source:
                    break
            if accepted >= args.pairs_per_source:
                break
        if accepted >= args.pairs_per_source:
            completed_sources.add(source_id)
            latest = records[-1]
            print(
                json.dumps(
                    {
                        "progress": "accepted_source",
                        "source": planned_source,
                        "sample_index": latest["sample_index"],
                        "paired_sample_index": latest["paired_sample_index"],
                        "pair_distance": latest["pair_distance"],
                        "pairs": accepted,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if len(completed_sources) >= args.source_count:
            break

    if len(completed_sources) < args.source_count:
        raise AuditError(
            f"only found {len(completed_sources)} compatible moving sources; "
            f"attempts={attempted_sources}"
        )
    summary = summarize(records)
    if summary["source_count"] < args.source_count:
        raise AuditError("paired audit source coverage is incomplete")
    receipt = {
        "schema": SCHEMA,
        "runtime_path": str(args.runtime.resolve(strict=True)),
        "checkpoint_path": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_resume_mode": inspection["resume_mode"],
        "validation_seed": seed,
        "data_split": args.split,
        "generalization_evaluation": args.split == "val",
        "multi_source_heldout_evaluation": args.split == "val" and summary["source_count"] >= 3,
        "coverage_limitation": (
            "Training-split diagnostic; not held-out generalization."
            if args.split != "val" else
            "Fewer than three held-out sources; cannot establish multi-source generalization."
            if summary["source_count"] < 3 else None
        ),
        "minimum_action_distance": minimum_distance,
        "requested_source_count": args.source_count,
        "pair_batch_size": args.pair_batch_size,
        "pairs_per_source": args.pairs_per_source,
        "records": records,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
