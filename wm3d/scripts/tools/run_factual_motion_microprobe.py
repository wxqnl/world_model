"""Fast real-data learnability gate for the factual-action RGB path.

The probe deliberately keeps the production implementation and physical
action ABI, but reduces width, depth and RGB resolution. One real
K8 high-motion window is expanded into eight K1 examples that share exactly
the same observation, task and future timestamp. Their physical action is
therefore the only input that can explain the eight different future targets.

This is a qualification gate, not a claim about final 1B quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping

from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

from wm3d.data.grouped_normalization import GroupedRobotNormalizer
from wm3d.data.manifest_contract import load_data_profile
from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.native_objective import (
    NativeObjectiveConfig,
    build_rgb_perceptual_model,
    compute_native_objective,
)
from wm3d.training.pretrain import (
    _batch_to_device,
    _forward,
    _forward_with_action_counterfactual,
    _zero_future_factual_action,
)


DEFAULT_RUNTIME = Path(
    "/data/Minko/"
    "wm3d_v8_v7_base_factual_qualification_1b_2node16_step100_20260831/"
    "runtime.yaml"
)
DEFAULT_BATCH = Path(
    "/data/Minko/wm3d_cosmos_rectified_flow_20260829/"
    "validation_seed7340_materialized_batch.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--materialized-batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--state-normalization", type=Path)
    parser.add_argument(
        "--mode",
        choices=("structural", "learnability"),
        default="learnability",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--rgb-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7340)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-loss-reduction", type=float, default=0.20)
    parser.add_argument("--minimum-cosine-improvement", type=float, default=0.05)
    return parser.parse_args()


def _load_runtime(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("schema") != "wm3d_v8_materialized_runtime_v2":
        raise RuntimeError("micro-probe requires a materialized runtime")
    return value


def _add_state_normalization(
    batch: dict[str, object],
    runtime: Mapping[str, object],
    state_normalization: Path | None,
) -> None:
    if "state_normalization_offset" in batch:
        return
    if state_normalization is not None:
        value = torch.load(
            state_normalization.resolve(strict=True),
            map_location="cpu",
            weights_only=False,
        )
        batch["state_normalization_offset"] = value[
            "state_normalization_offset"
        ]
        batch["state_normalization_scale"] = value[
            "state_normalization_scale"
        ]
        return
    closure = runtime["data_closure"]
    profile = load_data_profile(Path(str(closure["data_profile_path"])))
    artifact = json.loads(
        Path(str(closure["grouped_normalization_path"])).read_text(encoding="utf-8")
    )
    normalizer = GroupedRobotNormalizer(artifact, data_profile=profile)
    source_id = int(batch["source_id"][0])
    values = normalizer.tensors_for(
        source=profile.source_order[source_id],
        embodiment_id=int(batch["embodiment_ids"][0]),
        group_ids=batch["action_group_ids"][0],
        action_semantic_ids=batch["action_semantic_ids"][0],
        state_semantic_ids=batch["state_semantic_ids"][0],
    )
    batch["state_normalization_offset"] = values.state_offset.unsqueeze(0)
    batch["state_normalization_scale"] = values.state_scale.unsqueeze(0)


def _repeat(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.ndim == 0 or value.shape[0] != 1:
        raise RuntimeError("fixed probe tensors must have batch size one")
    return value.repeat(count, *([1] * (value.ndim - 1)))


def _resize_rgb(value: torch.Tensor, size: int) -> torch.Tensor:
    shape = value.shape
    flat = value.reshape(-1, 3, shape[-2], shape[-1]).float()
    resized = F.interpolate(
        flat,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(*shape[:-2], size, size)


def _future_slices(value: torch.Tensor, horizons: int) -> torch.Tensor:
    if value.shape[0] != 1 or value.shape[1] < horizons:
        raise RuntimeError("future tensor does not contain the fixed K8 horizon")
    return value[0, :horizons].unsqueeze(1).clone()


def prepare_same_context_action_batch(
    raw: dict[str, object],
    runtime: Mapping[str, object],
    *,
    rgb_size: int,
    state_normalization: Path | None = None,
) -> dict[str, torch.Tensor]:
    if raw.pop("_schema", None) != "wm3d_fixed_materialized_batch_v1":
        raise RuntimeError("fixed real batch schema is invalid")
    if int(raw.pop("_fixed_validation_seed", -1)) != 7340:
        raise RuntimeError("fixed real batch seed is invalid")
    _add_state_normalization(raw, runtime, state_normalization)
    horizons = 8
    model_inputs = (
        "task_embedding",
        "history_fine_action_values",
        "history_fine_action_mask",
        "history_fine_action_dt",
        "history_fine_sample_mask",
        "history_coarse_action_values",
        "history_coarse_action_mask",
        "action_group_ids",
        "action_group_mask",
        "action_semantic_ids",
        "current_state_values",
        "current_state_mask",
        "state_semantic_ids",
        "embodiment_ids",
        "policy_query_dt",
        "policy_query_mask",
        "action_normalization_offset",
        "action_normalization_scale",
        "state_normalization_offset",
        "state_normalization_scale",
        "aux_values",
        "aux_mask",
        "aux_type_ids",
    )
    batch = {name: _repeat(raw[name], horizons) for name in model_inputs}
    batch["world_tokens"] = _repeat(raw["world_tokens"], horizons)
    batch["view_mask"] = _repeat(raw["view_mask"], horizons)
    context_times = raw["world_times_s"][:, :16]
    # Every expanded example sees the same future timestamp. Time and context
    # cannot identify the target; only the physical action can.
    shared_future_time = raw["world_times_s"][:, 16:17]
    batch["world_times_s"] = torch.cat(
        (
            _repeat(context_times, horizons),
            _repeat(shared_future_time, horizons),
        ),
        dim=1,
    )
    for name in (
        "future_factual_fine_action_values",
        "future_factual_fine_action_mask",
        "future_factual_fine_action_dt",
        "future_factual_fine_sample_mask",
        "future_factual_coarse_action_values",
        "future_factual_coarse_action_mask",
    ):
        batch[name] = _future_slices(raw[name], horizons)

    batch["context_rgb"] = _repeat(
        _resize_rgb(raw["context_rgb"], rgb_size), horizons
    )
    anchor_context_mask = torch.zeros_like(raw["context_rgb_mask"])
    anchor_context_mask[:, 0] = raw["context_rgb_mask"][:, 0]
    batch["context_rgb_mask"] = _repeat(anchor_context_mask, horizons)
    batch["target_tokens"] = _future_slices(raw["target_tokens"], horizons)
    batch["target_token_mask"] = _future_slices(
        raw["target_token_mask"], horizons
    )
    batch["target_fine_action"] = _repeat(
        raw["target_fine_action"], horizons
    )
    batch["target_fine_action_mask"] = _repeat(
        raw["target_fine_action_mask"], horizons
    )
    for name in (
        "target_coarse_action",
        "target_coarse_action_mask",
        "target_coarse_action_normalized",
    ):
        batch[name] = _future_slices(raw[name], horizons)
    batch["composition_operator_ids"] = _repeat(
        raw["composition_operator_ids"], horizons
    )
    boundaries = raw["future_world_boundaries_dt"]
    batch["future_world_boundaries_dt"] = torch.stack(
        (boundaries[0, 0].expand(horizons), boundaries[0, 1 : horizons + 1]),
        dim=1,
    )
    target_rgb = raw["target_rgb"][0, :horizons].unsqueeze(1)
    batch["target_rgb"] = _resize_rgb(target_rgb, rgb_size)
    target_rgb_mask = raw["target_rgb_mask"][0, :horizons].unsqueeze(1).clone()
    target_rgb_mask[:, :, 1:] = False
    batch["target_rgb_mask"] = target_rgb_mask
    batch["rgb_frame_indices"] = torch.zeros(horizons, 1, dtype=torch.long)
    return batch


def build_micro_model(
    runtime: Mapping[str, object], *, rgb_size: int, num_views: int
) -> NativeWorldModel:
    profile = dict(runtime["model_profile"])
    profile.pop("expected_parameter_count", None)
    profile["name"] = "native_micro_v7_factual_motion_gate"
    model = dict(profile["model"])
    model.update(
        {
            "K": 1,
            "num_views": num_views,
            "state_hidden": 128,
            "state_layers": 2,
            "state_heads": 8,
            "state_ff_mult": 2.0,
            "action_hidden": 128,
            "action_layers": 2,
            "action_heads": 8,
            "action_ff_mult": 2.0,
            "bridge_layers_state": [1],
            "bridge_heads": 8,
            "dynamics_layers": 2,
            "view_hidden": 64,
            "view_heads": 8,
            "view_ff_mult": 2.0,
            "max_action_substeps": 4,
            "max_policy_queries": 33,
            "time_fourier_dim": 32,
            "rgb_hidden": 128,
            "rgb_decode_chunk_size": 8,
            "rgb_size": rgb_size,
            "rgb_decode_indices": [0],
            "geom_hidden": 64,
            "appearance_enabled": False,
            "activation_checkpointing": False,
        }
    )
    profile["model"] = model
    result = build_world_model(profile)
    if not isinstance(result, NativeWorldModel):
        raise RuntimeError("micro-probe did not build the native model")
    return result


def probe_objective() -> NativeObjectiveConfig:
    # These are the production V7 token/RGB/motion terms. Geometry and policy
    # are excluded: this gate isolates factual motion learning without
    # inventing an auxiliary loss or rebalancing the formal objective.
    return NativeObjectiveConfig(
        token_mse=1.0,
        token_cosine=0.1,
        rgb_l1=1.2,
        rgb_charbonnier=0.0,
        rgb_gradient=0.08,
        rgb_perceptual=0.55,
        rgb_motion_l1=1.0,
        rgb_motion_bce=0.03,
        rgb_motion_dice=0.03,
        rgb_motion_pos_weight=2.0,
        rgb_motion_threshold=0.03,
        rgb_motion_gain=3.0,
        depth_log=0.0,
        point=0.0,
        camera_pose=0.0,
        action_fine=0.0,
        action_coarse=0.0,
        action_counterfactual_token_advantage=1.0,
        action_counterfactual_token_margin=0.005,
        action_counterfactual_rgb_advantage=0.0,
        action_counterfactual_rgb_margin=0.002,
    )


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float:
    weight = torch.broadcast_to(mask, value.shape).to(value.dtype)
    return float(((value.float().square() * weight).sum() / weight.sum()).sqrt())


def temporal_metrics(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    pred_delta = prediction[:, 1:].float() - prediction[:, :-1].float()
    target_delta = target[:, 1:].float() - target[:, :-1].float()
    pair_mask = mask[:, 1:].bool() & mask[:, :-1].bool()
    weight = torch.broadcast_to(pair_mask, pred_delta.shape).float()
    count = weight.sum().clamp_min(1.0)
    pred_rms = ((pred_delta.square() * weight).sum() / count).sqrt()
    target_rms = ((target_delta.square() * weight).sum() / count).sqrt()
    error_rms = (((pred_delta - target_delta).square() * weight).sum() / count).sqrt()
    dot = (pred_delta * target_delta * weight).sum()
    cosine = dot / (
        (pred_delta.square() * weight).sum().sqrt()
        * (target_delta.square() * weight).sum().sqrt()
    ).clamp_min(1.0e-12)
    return {
        "prediction_delta_rms": float(pred_rms),
        "target_delta_rms": float(target_rms),
        "delta_amplitude_ratio": float(pred_rms / target_rms.clamp_min(1.0e-12)),
        "delta_error_rms": float(error_rms),
        "delta_direction_cosine": float(cosine),
    }


def _variant(
    batch: Mapping[str, torch.Tensor], mode: str
) -> dict[str, torch.Tensor]:
    if mode == "zero":
        return _zero_future_factual_action(batch)
    if mode != "shuffle":
        raise ValueError(mode)
    result = dict(batch)
    for name in (
        "future_factual_fine_action_values",
        "future_factual_coarse_action_values",
    ):
        result[name] = batch[name].roll(1, dims=0)
    return result


def evaluate(
    model: NativeWorldModel, batch: Mapping[str, torch.Tensor]
) -> tuple[dict[str, object], torch.Tensor]:
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        factual = _forward(model, batch, appearance_teacher_ratio=0.0)
        zero = _forward(model, _variant(batch, "zero"), appearance_teacher_ratio=0.0)
        shuffle = _forward(
            model, _variant(batch, "shuffle"), appearance_teacher_ratio=0.0
        )
    target_tokens = batch["target_tokens"][:, 0].unsqueeze(0).float()
    token_mask = batch["target_token_mask"][:, 0].unsqueeze(0)[..., None]
    factual_tokens = factual["pred_tokens"][:, 0].unsqueeze(0).float()
    zero_tokens = zero["pred_tokens"][:, 0].unsqueeze(0).float()
    shuffle_tokens = shuffle["pred_tokens"][:, 0].unsqueeze(0).float()
    target_rgb = batch["target_rgb"][:, 0, 0].unsqueeze(0).float()
    rgb_mask = batch["target_rgb_mask"][:, 0, 0].unsqueeze(0)
    factual_rgb = factual["rgb"][:, 0, 0].unsqueeze(0).float()
    zero_rgb = zero["rgb"][:, 0, 0].unsqueeze(0).float()
    shuffle_rgb = shuffle["rgb"][:, 0, 0].unsqueeze(0).float()
    context = batch["context_rgb"][0, 0].view(1, 1, *target_rgb.shape[2:])
    copy_last = context.expand_as(target_rgb)

    def l1(value: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
        weight = torch.broadcast_to(mask, value.shape).float()
        return float(((value - truth).abs() * weight).sum() / weight.sum())

    factual_token_error = _masked_rms(factual_tokens - target_tokens, token_mask)
    zero_token_error = _masked_rms(zero_tokens - target_tokens, token_mask)
    shuffle_token_error = _masked_rms(shuffle_tokens - target_tokens, token_mask)
    factual_rgb_error = l1(factual_rgb, target_rgb, rgb_mask)
    zero_rgb_error = l1(zero_rgb, target_rgb, rgb_mask)
    shuffle_rgb_error = l1(shuffle_rgb, target_rgb, rgb_mask)
    copy_last_error = l1(copy_last, target_rgb, rgb_mask)
    metrics = {
        "token": {
            "factual_error_rms": factual_token_error,
            "zero_error_rms": zero_token_error,
            "shuffle_error_rms": shuffle_token_error,
            "factual_gain_vs_zero": zero_token_error - factual_token_error,
            "factual_gain_vs_shuffle": shuffle_token_error - factual_token_error,
            "factual_zero_response_rms": _masked_rms(
                factual_tokens - zero_tokens, token_mask
            ),
            "temporal": temporal_metrics(factual_tokens, target_tokens, token_mask),
        },
        "rgb": {
            "factual_l1": factual_rgb_error,
            "zero_l1": zero_rgb_error,
            "shuffle_l1": shuffle_rgb_error,
            "copy_last_l1": copy_last_error,
            "factual_gain_vs_zero": zero_rgb_error - factual_rgb_error,
            "factual_gain_vs_shuffle": shuffle_rgb_error - factual_rgb_error,
            "factual_zero_response_rms": _masked_rms(
                factual_rgb - zero_rgb, rgb_mask
            ),
            "temporal": temporal_metrics(factual_rgb, target_rgb, rgb_mask),
        },
        "invariants": {
            "policy_equal_under_zero": torch.equal(
                factual["policy_action"], zero["policy_action"]
            ),
            "policy_equal_under_shuffle": torch.equal(
                factual["policy_action"], shuffle["policy_action"]
            ),
            "action_free_equal_under_zero": torch.equal(
                factual["action_free_pred_tokens"],
                zero["action_free_pred_tokens"],
            ),
            "action_free_equal_under_shuffle": torch.equal(
                factual["action_free_pred_tokens"],
                shuffle["action_free_pred_tokens"],
            ),
        },
    }
    return metrics, factual_rgb[0].detach().cpu()


def _first_gradient_norm(module: torch.nn.Module) -> float:
    for parameter in module.parameters():
        if parameter.grad is not None:
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError("micro-probe encountered a non-finite gradient")
            value = float(parameter.grad.detach().float().norm())
            if value > 0.0:
                return value
    return 0.0


def save_gif(
    path: Path,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> None:
    target = batch["target_rgb"][:, 0, 0].detach().cpu()
    context = batch["context_rgb"][0, 0].detach().cpu()
    frames: list[Image.Image] = []
    for horizon in range(target.shape[0]):
        tiles = []
        for label, tensor in (
            ("copy-last", context),
            ("target", target[horizon]),
            ("prediction", prediction[horizon]),
        ):
            array = (
                tensor.float()
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .permute(1, 2, 0)
                .numpy()
            )
            image = Image.fromarray(array, "RGB")
            tile = Image.new("RGB", (image.width, image.height + 20), "white")
            ImageDraw.Draw(tile).text((4, 4), label, fill="black")
            tile.paste(image, (0, 20))
            tiles.append(tile)
        frame = Image.new(
            "RGB",
            (sum(tile.width for tile in tiles), tiles[0].height + 18),
            "white",
        )
        ImageDraw.Draw(frame).text((4, 2), f"horizon {horizon + 1}", fill="black")
        x = 0
        for tile in tiles:
            frame.paste(tile, (x, 18))
            x += tile.width
        frames.append(frame.resize((frame.width * 2, frame.height * 2)))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=350,
        loop=0,
    )


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise RuntimeError("--steps must be positive")
    if args.mode == "structural" and args.steps != 1:
        raise RuntimeError("structural mode requires exactly one optimizer step")
    if args.output.exists():
        raise RuntimeError("micro-probe output directory must be new")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    runtime = _load_runtime(args.runtime)
    raw = torch.load(
        args.materialized_batch.resolve(strict=True),
        map_location="cpu",
        weights_only=False,
    )
    cpu_batch = prepare_same_context_action_batch(
        raw,
        runtime,
        rgb_size=args.rgb_size,
        state_normalization=args.state_normalization,
    )
    batch = _batch_to_device(cpu_batch, device)
    model = build_micro_model(
        runtime,
        rgb_size=args.rgb_size,
        num_views=int(cpu_batch["world_tokens"].shape[2]),
    ).to(device)
    objective = probe_objective()
    perceptual = build_rgb_perceptual_model(objective, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.02,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.monotonic()
    before, before_rgb = evaluate(model, batch)
    save_gif(args.output / "before.gif", before_rgb, cpu_batch)
    initial_loss = None
    final_loss = None
    gradient_norms: dict[str, float] = {}
    trace = []
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = _forward_with_action_counterfactual(
                model,
                batch,
                appearance_teacher_ratio=0.0,
                objective=objective,
            )
            losses = compute_native_objective(
                output=output,
                batch=batch,
                config=objective,
                perceptual_model=perceptual,
                rgb_perceptual_chunk_size=8,
            )
        loss = losses["total"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("micro-probe loss became non-finite")
        loss.backward()
        if step == 1:
            factual_action_module = (
                model.factual_v7_query_action
                if model.factual_v7_query_action is not None
                else model.factual_action
            )
            gradient_norms = {
                "factual_action_encoder": _first_gradient_norm(
                    factual_action_module
                ),
                "early_factual_state_block": _first_gradient_norm(
                    model.state_blocks[0]
                ),
                "factual_decoder": _first_gradient_norm(model.dynamics_blocks[0]),
                "rgb_decoder": _first_gradient_norm(model.rgb_head.image_decoder),
            }
            if any(value <= 0.0 for value in gradient_norms.values()):
                raise RuntimeError(
                    f"required factual/RGB gradient is zero: {gradient_norms}"
                )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        if step == 1 or step % 10 == 0 or step == args.steps:
            row = {
                "step": step,
                "total": value,
                "token_gain": float(
                    losses["action_counterfactual_token_gain"].detach()
                ),
                "rgb_l1": float(losses["rgb_l1"].detach()),
                "rgb_motion_l1": float(losses["rgb_motion_l1"].detach()),
            }
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    after, after_rgb = evaluate(model, batch)
    save_gif(args.output / "after.gif", after_rgb, cpu_batch)
    elapsed = time.monotonic() - started
    assert initial_loss is not None and final_loss is not None
    loss_reduction = 1.0 - final_loss / max(initial_loss, 1.0e-12)
    token_before_cosine = before["token"]["temporal"]["delta_direction_cosine"]
    token_after_cosine = after["token"]["temporal"]["delta_direction_cosine"]
    rgb_before_cosine = before["rgb"]["temporal"]["delta_direction_cosine"]
    rgb_after_cosine = after["rgb"]["temporal"]["delta_direction_cosine"]
    common_checks = {
        "finite_and_fast": elapsed <= args.max_seconds,
        "required_gradients_nonzero": all(
            value > 0.0 for value in gradient_norms.values()
        ),
        "policy_and_action_free_invariant": all(
            bool(value) for value in after["invariants"].values()
        ),
    }
    if args.mode == "structural":
        checks = {
            **common_checks,
            "factual_token_responds_to_action": (
                after["token"]["factual_zero_response_rms"] > 1.0e-6
            ),
            "factual_rgb_responds_to_action": (
                after["rgb"]["factual_zero_response_rms"] > 1.0e-8
            ),
        }
    else:
        checks = {
            **common_checks,
            "focused_loss_reduced": loss_reduction >= args.minimum_loss_reduction,
            "token_factual_beats_zero": (
                after["token"]["factual_gain_vs_zero"] > 0.0
            ),
            "token_factual_beats_shuffle": (
                after["token"]["factual_gain_vs_shuffle"] > 0.0
            ),
            "rgb_factual_beats_zero": after["rgb"]["factual_gain_vs_zero"] > 0.0,
            "rgb_factual_beats_shuffle": (
                after["rgb"]["factual_gain_vs_shuffle"] > 0.0
            ),
            "token_motion_direction_improved": token_after_cosine >= max(
                0.0, token_before_cosine + args.minimum_cosine_improvement
            ),
            "rgb_motion_direction_improved": rgb_after_cosine >= max(
                0.0, rgb_before_cosine + args.minimum_cosine_improvement
            ),
        }
    receipt = {
        "schema": "wm3d_factual_motion_microprobe_v1",
        "mode": args.mode,
        "purpose": "fast structural and real-data learnability gate, not final quality",
        "real_fixed_seed": 7340,
        "same_context_examples": 8,
        "same_future_timestamp": True,
        "model_parameters": parameter_count,
        "rgb_size": args.rgb_size,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / args.steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "gradient_norms": gradient_norms,
        "before": before,
        "after": after,
        "trace": trace,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
