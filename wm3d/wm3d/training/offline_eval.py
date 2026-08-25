"""Read-only offline evaluation for unified WM3D DCP checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageDraw
import torch
import torch.distributed as dist
import torch.nn.functional as F

from wm3d.data.direct_raw import DIRECT_RAW_DATA_CLOSURE_SCHEMA
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.models.direct_vggt_builder import build_direct_vggt_teacher
from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.distributed_checkpoint import (
    DistributedCheckpointManager,
    ResumeExpectations,
    sha256_file,
)
from wm3d.training.distributed_runtime import (
    autocast_context,
    destroy_distributed,
    initialize_distributed,
    reduce_metrics,
    strategy_from_mapping,
    wrap_model,
)
from wm3d.training.native_objective import (
    build_rgb_perceptual_model,
    compute_native_objective,
    objective_config_from_mapping,
)
from wm3d.training.pretrain import (
    _atomic_json_no_clobber,
    _batch_to_device,
    _build_mixed_dataset,
    _configure_reproducibility,
    _forward_with_action_counterfactual,
    _make_loader,
    _validation_micro_batch_size,
    _require_recent_resource_preflight,
    _publish_and_validate_launch,
    _run_contract,
    _topology_contract_sha256,
)
from wm3d.training.launch_qualification import (
    LaunchQualificationError,
    verify_clean_runtime_checkout,
)
from wm3d.training.runtime_contract import load_materialized_runtime


EVAL_RECEIPT_SCHEMA = "wm3d_v8_unified_offline_eval_v2"
_COVERAGE_WEIGHTS = {
    "native_token_supervised_elements": ("token_mse", "token_cosine"),
    "appearance_supervised_elements": ("appearance_mse", "appearance_cosine"),
    "rgb_supervised_elements": (
        "rgb_l1",
        "rgb_charbonnier",
        "rgb_gradient",
        "rgb_perceptual",
        "rgb_motion_l1",
        "rgb_motion_bce",
        "rgb_motion_dice",
    ),
    "depth_supervised_elements": ("depth_log",),
    "point_supervised_elements": ("point",),
    "camera_pose_supervised_elements": ("camera_pose",),
    "fine_supervised_dimensions": ("action_fine",),
    "coarse_supervised_dimensions": ("action_coarse",),
}
_KNOWN_COVERAGE_LANES = frozenset(_COVERAGE_WEIGHTS) | {
    "current_state_supervised_dimensions",
    "fine_continuous_supervised_dimensions",
    "fine_binary_supervised_dimensions",
}


class OfflineEvalError(RuntimeError):
    pass


def declared_eval_coverage_lanes(
    profile: Any,
    objective: Any,
    *,
    active_source_names: frozenset[str],
) -> frozenset[str]:
    """Intersect objective lanes with supervision declared by the sealed profile."""

    formal_lanes = getattr(profile, "declared_eval_coverage_lanes", None)
    if formal_lanes is not None:
        declared_sources = {source.name for source in profile.sources}
        if not active_source_names or not active_source_names <= declared_sources:
            raise OfflineEvalError(
                "eval active-source closure is empty or differs from sealed profile"
            )
        lanes = frozenset(str(value) for value in formal_lanes)
        if not lanes or not lanes <= _KNOWN_COVERAGE_LANES:
            raise OfflineEvalError("formal cache declared unknown coverage lanes")
        return lanes

    declared_sources = {source.name for source in profile.sources}
    if not active_source_names or not active_source_names <= declared_sources:
        raise OfflineEvalError(
            "eval active-source closure is empty or differs from sealed profile"
        )
    cache = profile.cache
    lanes = {
        "native_token_supervised_elements",
        "current_state_supervised_dimensions",
    }
    if (
        getattr(objective, "appearance_mse", 0.0) > 0.0
        or getattr(objective, "appearance_cosine", 0.0) > 0.0
    ):
        lanes.add("appearance_supervised_elements")
    for key, lane in (
        ("rgb_codec", "rgb_supervised_elements"),
        ("depth_codec", "depth_supervised_elements"),
        ("point_codec", "point_supervised_elements"),
        ("camera_pose_codec", "camera_pose_supervised_elements"),
    ):
        if cache.get(key) is not None:
            lanes.add(lane)
    fine_continuous = False
    fine_binary = False
    declared_action: set[str] = set()
    for source in profile.sources:
        if source.name not in active_source_names:
            continue
        adapter = load_adapter_contract(
            source.adapter_config_path,
            expected_sha256=source.adapter_contract_sha256,
        )
        embodiment = profile.embodiments[source.embodiment]
        groups = {group.name: group for group in embodiment.groups}
        for mapping in adapter.groups:
            declared_action.add(mapping.supervision)
            if mapping.supervision != "fine_command":
                continue
            semantics = groups[mapping.group].action_semantics
            for semantic in semantics:
                if semantic in {
                    "absolute_gripper_open01",
                    "absolute_gripper_close01",
                    "binary_contact",
                    "controller_mode",
                }:
                    fine_binary = True
                else:
                    fine_continuous = True
    if "fine_command" in declared_action:
        lanes.add("fine_supervised_dimensions")
        if fine_continuous:
            lanes.add("fine_continuous_supervised_dimensions")
        if fine_binary:
            lanes.add("fine_binary_supervised_dimensions")
    if "coarse_effect" in declared_action:
        lanes.add("coarse_supervised_dimensions")
    return frozenset(
        lane
        for lane in lanes
        if lane == "current_state_supervised_dimensions"
        or lane
        in {
            "fine_continuous_supervised_dimensions",
            "fine_binary_supervised_dimensions",
        }
        or any(
            float(getattr(objective, weight)) > 0.0
            for weight in _COVERAGE_WEIGHTS[lane]
        )
    )


def validate_eval_coverage(
    metrics: dict[str, float], *, expected_lanes: frozenset[str]
) -> dict[str, float]:
    unknown = set(expected_lanes) - _KNOWN_COVERAGE_LANES
    if unknown:
        raise OfflineEvalError(f"unknown expected coverage lanes: {sorted(unknown)}")
    coverage: dict[str, float] = {}
    for count_name in _COVERAGE_WEIGHTS:
        if (
            count_name == "appearance_supervised_elements"
            and count_name not in expected_lanes
            and count_name not in metrics
        ):
            continue
        count = float(metrics.get(count_name, 0.0))
        if count_name in expected_lanes and (not math.isfinite(count) or count <= 0.0):
            raise OfflineEvalError(
                f"offline eval has zero coverage for declared lane {count_name}"
            )
        coverage[count_name] = count
    current_state = float(metrics.get("current_state_supervised_dimensions", 0.0))
    if "current_state_supervised_dimensions" in expected_lanes and (
        not math.isfinite(current_state) or current_state <= 0.0
    ):
        raise OfflineEvalError("offline eval has zero current-state coverage")
    coverage["current_state_supervised_dimensions"] = current_state
    for lane in (
        "fine_continuous_supervised_dimensions",
        "fine_binary_supervised_dimensions",
    ):
        value = float(metrics.get(lane, 0.0))
        if lane in expected_lanes and (not math.isfinite(value) or value <= 0.0):
            raise OfflineEvalError(
                f"offline eval has zero coverage for declared lane {lane}"
            )
        coverage[lane] = value
    return coverage


def rgb_quality_metrics(
    output: dict[str, torch.Tensor] | Any,
    batch: dict[str, torch.Tensor] | Any,
    *,
    motion_threshold: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Return image-weighted PSNR/SSIM for fully supervised RGB frames."""

    prediction = output["rgb"].float().clamp(0.0, 1.0)
    target = batch["target_rgb"].float().clamp(0.0, 1.0)
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise OfflineEvalError("RGB eval tensors must align as [B,F,V,3,H,W]")
    mask = batch.get(
        "target_rgb_mask",
        torch.ones_like(target[:, :, :, :1, :1, :1], dtype=torch.bool),
    )
    expanded = torch.broadcast_to(mask.bool(), target.shape)
    flat_mask = expanded.reshape(-1, *target.shape[-3:])
    image_all = flat_mask.all(dim=(1, 2, 3))
    image_any = flat_mask.any(dim=(1, 2, 3))
    if not bool((image_all == image_any).all()):
        raise OfflineEvalError("RGB quality metrics require whole-image masks")
    valid = torch.nonzero(image_all, as_tuple=False).flatten()
    if valid.numel() == 0:
        raise OfflineEvalError("RGB quality metrics have no supervised images")
    motion_mask: torch.Tensor | None = None
    if "context_rgb" in batch:
        context_rgb = batch["context_rgb"].float().clamp(0.0, 1.0)
        expected_context = (
            target.shape[0],
            target.shape[2],
            *target.shape[3:],
        )
        if tuple(context_rgb.shape) != expected_context:
            raise OfflineEvalError("context RGB does not align to RGB eval targets")
        context_rgb_mask = batch.get(
            "context_rgb_mask",
            torch.ones(
                target.shape[0],
                target.shape[2],
                dtype=torch.bool,
                device=target.device,
            ),
        ).bool()
        if tuple(context_rgb_mask.shape) != target.shape[:1] + target.shape[2:3]:
            raise OfflineEvalError("context RGB mask must be [B,V]")
        motion_mask = (target - context_rgb[:, None]).abs().mean(
            dim=3, keepdim=True
        ) > motion_threshold
        motion_mask = (
            motion_mask & mask.bool() & context_rgb_mask[:, None, :, None, None, None]
        )
        motion_mask = motion_mask.reshape(
            -1, 1, target.shape[-2], target.shape[-1]
        ).index_select(0, valid)
    prediction = prediction.reshape(-1, *prediction.shape[-3:]).index_select(0, valid)
    target = target.reshape(-1, *target.shape[-3:]).index_select(0, valid)
    per_image_mse = (prediction - target).square().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(per_image_mse.clamp_min(1.0e-10))

    kernel = min(11, int(prediction.shape[-2]), int(prediction.shape[-1]))
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 1:
        raise OfflineEvalError("RGB images have invalid spatial dimensions")
    mu_prediction = F.avg_pool2d(prediction, kernel, stride=1)
    mu_target = F.avg_pool2d(target, kernel, stride=1)
    prediction_variance = (
        F.avg_pool2d(prediction.square(), kernel, stride=1) - mu_prediction.square()
    ).clamp_min(0.0)
    target_variance = (
        F.avg_pool2d(target.square(), kernel, stride=1) - mu_target.square()
    ).clamp_min(0.0)
    covariance = (
        F.avg_pool2d(prediction * target, kernel, stride=1) - mu_prediction * mu_target
    )
    c1 = 0.01**2
    c2 = 0.03**2
    ssim = ((2.0 * mu_prediction * mu_target + c1) * (2.0 * covariance + c2)) / (
        (mu_prediction.square() + mu_target.square() + c1)
        * (prediction_variance + target_variance + c2)
    ).clamp_min(1.0e-12)
    metrics = {"rgb_psnr_db": psnr.mean(), "rgb_ssim": ssim.mean()}
    if motion_mask is None:
        return metrics

    def region_mean(value: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        weight = torch.broadcast_to(region, value.shape).to(dtype=value.dtype)
        numerator = (value * weight).flatten(1).sum(dim=1)
        denominator = weight.flatten(1).sum(dim=1)
        sample_valid = denominator > 0
        if not bool(sample_valid.any()):
            return value.new_zeros(())
        return (numerator[sample_valid] / denominator[sample_valid]).mean()

    static_mask = ~motion_mask
    absolute_error = (prediction - target).abs()
    squared_error = (prediction - target).square()
    motion_mse = region_mean(squared_error, motion_mask)
    static_mse = region_mean(squared_error, static_mask)
    motion_ssim_mask = F.avg_pool2d(motion_mask.float(), kernel, stride=1) > 0.0
    static_ssim_mask = F.avg_pool2d(motion_mask.float(), kernel, stride=1) == 0.0
    pred_dy, pred_dx = (
        prediction[..., 1:, :] - prediction[..., :-1, :],
        prediction[..., :, 1:] - prediction[..., :, :-1],
    )
    target_dy, target_dx = (
        target[..., 1:, :] - target[..., :-1, :],
        target[..., :, 1:] - target[..., :, :-1],
    )
    motion_dy = motion_mask[..., 1:, :] | motion_mask[..., :-1, :]
    motion_dx = motion_mask[..., :, 1:] | motion_mask[..., :, :-1]
    static_dy = (~motion_mask[..., 1:, :]) & (~motion_mask[..., :-1, :])
    static_dx = (~motion_mask[..., :, 1:]) & (~motion_mask[..., :, :-1])
    metrics.update(
        {
            "rgb_eval_motion_fraction": motion_mask.float().mean(),
            "rgb_eval_motion_l1": region_mean(absolute_error, motion_mask),
            "rgb_eval_static_l1": region_mean(absolute_error, static_mask),
            "rgb_eval_motion_psnr_db": -10.0
            * torch.log10(motion_mse.clamp_min(1.0e-10)),
            "rgb_eval_static_psnr_db": -10.0
            * torch.log10(static_mse.clamp_min(1.0e-10)),
            "rgb_eval_motion_ssim": region_mean(ssim, motion_ssim_mask),
            "rgb_eval_static_ssim": region_mean(ssim, static_ssim_mask),
            "rgb_eval_motion_gradient": 0.5
            * (
                region_mean((pred_dy - target_dy).abs(), motion_dy)
                + region_mean((pred_dx - target_dx).abs(), motion_dx)
            ),
            "rgb_eval_static_gradient": 0.5
            * (
                region_mean((pred_dy - target_dy).abs(), static_dy)
                + region_mean((pred_dx - target_dx).abs(), static_dx)
            ),
        }
    )
    return metrics


def _rgb_tile(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _depth_tile(value: torch.Tensor, *, low: float, high: float) -> Image.Image:
    side = math.isqrt(int(value.numel()))
    if side * side != value.numel():
        raise OfflineEvalError("depth demo expects a square native patch grid")
    normalized = (value.detach().float().reshape(side, side) - low) / max(
        high - low, 1.0e-6
    )
    array = normalized.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    image = Image.fromarray(array, mode="L")
    return image.resize((256, 256), Image.Resampling.NEAREST).convert("RGB")


def _save_demo_grid(
    path: Path,
    *,
    rows: list[tuple[str, Image.Image, Image.Image, Image.Image]],
) -> None:
    if path.exists():
        raise OfflineEvalError(f"refusing to overwrite demo image: {path}")
    label_height = 20
    width = 3 * rows[0][1].width
    row_height = label_height + rows[0][1].height
    canvas = Image.new("RGB", (width, row_height * len(rows)), color="white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, target, prediction, error) in enumerate(rows):
        top = row_index * row_height
        for column, (name, image) in enumerate(
            (("target", target), ("prediction", prediction), ("abs error", error))
        ):
            left = column * image.width
            draw.text((left + 4, top + 3), f"{label} {name}", fill="black")
            canvas.paste(image, (left, top + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")


def save_rgb_depth_demo(
    root: Path,
    *,
    output: dict[str, torch.Tensor] | Any,
    batch: dict[str, torch.Tensor] | Any,
    sample_index: int,
    file_index: int,
) -> list[str]:
    """Save target/prediction/error panels for one deterministic val sample."""

    rgb_prediction = output["rgb"][sample_index]
    rgb_target = batch["target_rgb"][sample_index]
    rgb_rows: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for frame in range(min(4, int(rgb_prediction.shape[0]))):
        for view in range(int(rgb_prediction.shape[1])):
            target_tile = _rgb_tile(rgb_target[frame, view])
            prediction_tile = _rgb_tile(rgb_prediction[frame, view])
            error_tile = _rgb_tile(
                (rgb_prediction[frame, view] - rgb_target[frame, view]).abs()
            )
            rgb_rows.append(
                (
                    f"future={frame} view={view}",
                    target_tile,
                    prediction_tile,
                    error_tile,
                )
            )
    rgb_path = root / f"sample_{file_index:03d}_rgb.png"
    _save_demo_grid(rgb_path, rows=rgb_rows)

    depth_prediction = output["depth"][sample_index]
    depth_target = batch["target_depth"][sample_index]
    depth_rows: list[tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for frame in range(min(4, int(depth_prediction.shape[0]))):
        for view in range(int(depth_prediction.shape[1])):
            target_value = depth_target[frame, view]
            prediction_value = depth_prediction[frame, view]
            finite = torch.cat((target_value.flatten(), prediction_value.flatten()))
            finite = finite[torch.isfinite(finite)]
            if finite.numel() == 0:
                continue
            low = float(finite.min())
            high = float(finite.max())
            error = (prediction_value - target_value).abs()
            depth_rows.append(
                (
                    f"future={frame} view={view}",
                    _depth_tile(target_value, low=low, high=high),
                    _depth_tile(prediction_value, low=low, high=high),
                    _depth_tile(
                        error,
                        low=0.0,
                        high=float(error.max().clamp_min(1.0e-6)),
                    ),
                )
            )
    paths = [str(rgb_path)]
    if depth_rows:
        depth_path = root / f"sample_{file_index:03d}_depth.png"
        _save_demo_grid(depth_path, rows=depth_rows)
        paths.append(str(depth_path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path)
    parser.add_argument("--demo-samples", type=int, default=0)
    parser.add_argument("--appearance-teacher-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo_samples < 0 or (args.demo_samples > 0) != (args.demo_root is not None):
        raise OfflineEvalError(
            "--demo-root and a positive --demo-samples must be supplied together"
        )
    if not 0.0 <= args.appearance_teacher_ratio <= 1.0:
        raise OfflineEvalError("--appearance-teacher-ratio must lie in [0,1]")
    config, runtime_sha = load_materialized_runtime(args.runtime)
    runtime = config["runtime_profile"]
    strategy = strategy_from_mapping(runtime["distributed"])
    context = initialize_distributed(strategy)
    try:
        if int(runtime["expected_world_size"]) != context.world_size:
            raise OfflineEvalError("evaluation world size differs from sealed runtime")
        resource_preflight = _require_recent_resource_preflight(
            config, runtime_sha, context
        )
        if re.fullmatch(r"step_[0-9]{8}", args.checkpoint.name) is None:
            raise OfflineEvalError(
                "checkpoint must be an explicit step_XXXXXXXX directory"
            )
        checkpoint_step = int(args.checkpoint.name.split("_")[1])
        validation_steps = int(runtime["train"]["validation_steps"])
        if validation_steps <= 0:
            raise OfflineEvalError("validation steps must be positive")
        repo = Path(__file__).resolve().parents[2]
        try:
            current_commit = verify_clean_runtime_checkout(
                repo, str(config["run"]["code_commit"])
            )
        except LaunchQualificationError as exc:
            raise OfflineEvalError("runtime code provenance failed") from exc

        seed = int(runtime["train"]["validation_seed"])
        _configure_reproducibility(seed)
        construction_device = (
            torch.device("meta")
            if strategy.initialization == "meta_sharded"
            else context.device
        )
        with torch.device(construction_device):
            raw_model = build_world_model(config["model_profile"])
        if not isinstance(raw_model, NativeWorldModel):
            raise OfflineEvalError("offline eval requires native_world_model")
        parameter_counts = raw_model.parameter_counts()
        wrapped = wrap_model(
            raw_model,
            context,
            strategy,
            initialization_seed=(
                int(runtime["train"]["seed"])
                if strategy.initialization == "meta_sharded"
                else None
            ),
        )
        model = wrapped.model
        manager = DistributedCheckpointManager(args.checkpoint.parent)
        expectations = ResumeExpectations(
            step=checkpoint_step,
            run_lineage=config["run"]["lineage"],
            runtime_config_sha256=runtime_sha,
            data_closure_sha256=config["bindings"]["data_closure_sha256"],
            model_contract_sha256=config["bindings"]["model_contract_sha256"],
            world_size=context.world_size,
            shard_degree=int(runtime["distributed"]["shard_degree"]),
            distributed_strategy=strategy.strategy,
            global_batch_size=int(runtime["train"]["global_batch_size"]),
            topology_contract_sha256=_topology_contract_sha256(config),
            allow_topology_reshard=False,
        )
        inspection: list[Any] = [None]
        if context.is_rank0:
            try:
                source = manager.inspect_committed(
                    path=args.checkpoint, expected=expectations
                )
                if source["resume_mode"] != "exact":
                    raise OfflineEvalError("offline eval requires exact topology")
                inspection[0] = {"ok": True, "source": source}
            except Exception as exc:
                inspection[0] = {
                    "ok": False,
                    "type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(inspection, src=0)
        if not inspection[0]["ok"]:
            raise OfflineEvalError(f"checkpoint inspection failed: {inspection[0]}")
        run_contract = _run_contract(config, parameter_counts, raw_model)
        run_contract_path = Path(config["run"]["output_root"]) / "run_contract.json"
        if (
            not run_contract_path.is_file()
            or json.loads(run_contract_path.read_text(encoding="utf-8")) != run_contract
        ):
            raise OfflineEvalError("stable run contract is missing or differs")
        launch_qualification_path, launch_qualification_sha256 = (
            _publish_and_validate_launch(
                config=config,
                config_sha=runtime_sha,
                context=context,
                strategy=strategy,
                run_contract=run_contract,
                resource_preflight=resource_preflight,
                source_checkpoint=inspection[0]["source"],
                launch_kind="eval",
            )
        )
        metadata = manager.load_model_for_evaluation(
            path=args.checkpoint,
            model=model,
            expected=expectations,
        )
        dataset, profile = _build_mixed_dataset(
            config, split="val", device=context.device, rank=context.rank
        )
        input_adapter = None
        if config["data_closure"].get("schema") == DIRECT_RAW_DATA_CLOSURE_SCHEMA:
            input_adapter = build_direct_vggt_teacher(config, device=context.device)
            input_adapter.eval()
        loader = _make_loader(
            dataset,
            profile,
            runtime,
            rank=context.rank,
            world_size=context.world_size,
            start_step=0,
            num_steps=validation_steps,
            seed=seed,
            gradient_accumulation=1,
            micro_batch_size=_validation_micro_batch_size(runtime),
        )
        objective = objective_config_from_mapping(
            config["objective_profile"]["objective"]
        )
        perceptual_model = build_rgb_perceptual_model(objective, device=context.device)
        totals: dict[str, torch.Tensor] = {}
        demo_paths: list[str] = []
        demo_count = 0
        model.eval()
        with torch.no_grad():
            for cpu_batch in loader:
                if input_adapter is not None:
                    cpu_batch = input_adapter.materialize(cpu_batch)
                batch = _batch_to_device(cpu_batch, context.device)
                with autocast_context(strategy):
                    model_output = _forward_with_action_counterfactual(
                        model,
                        batch,
                        appearance_teacher_ratio=args.appearance_teacher_ratio,
                        objective=objective,
                    )
                    losses = compute_native_objective(
                        output=model_output,
                        batch=batch,
                        config=objective,
                        perceptual_model=perceptual_model,
                        rgb_perceptual_chunk_size=int(
                            runtime["train"].get("rgb_perceptual_chunk_size", 4)
                        ),
                    )
                losses.update(
                    rgb_quality_metrics(
                        model_output,
                        batch,
                        motion_threshold=objective.rgb_motion_threshold,
                    )
                )
                for name, value in losses.items():
                    if not bool(torch.isfinite(value).all()):
                        raise FloatingPointError(f"non-finite eval metric {name}")
                    totals[name] = totals.get(name, torch.zeros_like(value)) + value
                if context.is_rank0 and demo_count < args.demo_samples:
                    for sample_index in range(int(batch["target_rgb"].shape[0])):
                        if demo_count >= args.demo_samples:
                            break
                        demo_paths.extend(
                            save_rgb_depth_demo(
                                args.demo_root,
                                output=model_output,
                                batch=batch,
                                sample_index=sample_index,
                                file_index=demo_count,
                            )
                        )
                        demo_count += 1
        metrics = reduce_metrics(
            {name: value / validation_steps for name, value in totals.items()}
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("offline eval receipt contains non-finite metrics")
        expected_coverage_lanes = declared_eval_coverage_lanes(
            profile,
            objective,
            active_source_names=frozenset(dataset.source_names),
        )
        coverage = validate_eval_coverage(
            metrics, expected_lanes=expected_coverage_lanes
        )
        publication: list[Any] = [None]
        if context.is_rank0:
            try:
                commit_path = args.checkpoint / "COMMITTED.json"
                manifest_path = args.checkpoint / "MANIFEST.json"
                commit = json.loads(commit_path.read_text(encoding="utf-8"))
                receipt = {
                    "schema": EVAL_RECEIPT_SCHEMA,
                    "runtime_path": str(args.runtime.resolve(strict=True)),
                    "runtime_sha256": runtime_sha,
                    "checkpoint_path": str(args.checkpoint.resolve(strict=True)),
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_committed_sha256": sha256_file(commit_path),
                    "checkpoint_manifest_sha256": sha256_file(manifest_path),
                    "checkpoint_manifest_content_sha256": commit[
                        "manifest_content_sha256"
                    ],
                    "data_closure_sha256": config["bindings"]["data_closure_sha256"],
                    # The sealed index contains every split.  Validation rows
                    # are selected deterministically from it by split="val";
                    # do not mislabel the full-index digest as a val-only file.
                    "cache_window_index_sha256": config["data_closure"][
                        "cache_index_sha256"
                    ],
                    "evaluated_split": "val",
                    "validation_seed": seed,
                    "validation_steps": validation_steps,
                    "appearance_teacher_ratio": args.appearance_teacher_ratio,
                    "demo_paths": demo_paths,
                    "world_size": context.world_size,
                    "launch_qualification_path": launch_qualification_path,
                    "launch_qualification_sha256": launch_qualification_sha256,
                    "code_commit": current_commit,
                    "metrics": metrics,
                    "coverage": coverage,
                    "expected_coverage_lanes": sorted(expected_coverage_lanes),
                    "all_metrics_finite": True,
                    "checkpoint_metadata": {
                        "run_lineage": metadata["run_lineage"],
                        "runtime_config_sha256": metadata["runtime_config_sha256"],
                        "model_contract_sha256": metadata["model_contract_sha256"],
                    },
                }
                _atomic_json_no_clobber(args.output, receipt)
                publication[0] = {"ok": True, "receipt": receipt}
            except Exception as exc:
                publication[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(publication, src=0)
        if not publication[0]["ok"]:
            raise OfflineEvalError(f"eval receipt publication failed: {publication[0]}")
        if context.is_rank0:
            print(json.dumps(publication[0]["receipt"], sort_keys=True), flush=True)
        dist.barrier()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
