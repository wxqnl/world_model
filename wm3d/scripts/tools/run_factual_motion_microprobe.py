"""Fast real-data gate for the production K8 factual-action/RGB path.

The probe keeps real K8 timestamps, grouped controller tensors, normalization,
policy isolation, RAFT targets, and the production objective. Only model width,
depth, and pixel resolution are reduced. It must never reshape K8 into K1
examples: that old shortcut could look successful while the formal path failed.
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
    build_rgb_perceptual_model,
    compute_native_objective,
    objective_config_from_mapping,
)
from wm3d.training.pretrain import (
    _batch_to_device,
    _forward,
    _forward_with_action_counterfactual,
    _materialize_rgb_flow_targets,
    _zero_future_factual_action,
)
from wm3d.training.rgb_flow_runtime import (
    FrozenBidirectionalRAFTRuntime,
    raft_config_from_mapping,
)


DEFAULT_RUNTIME = Path(
    "/data/Minko/"
    "wm3d_v8_action_owned_transport_physical_qualification_1b_2node16_step100_20260902/"
    "runtime.yaml"
)
DEFAULT_BATCH = Path(
    "/data/Minko/wm3d_cosmos_rectified_flow_20260829/"
    "validation_seed7340_materialized_batch.pt"
)
DEFAULT_OBJECTIVE = Path("configs/objective/stage0_v8_action_owned_transport.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--objective", type=Path, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--materialized-batch", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--state-normalization", type=Path)
    parser.add_argument(
        "--mode", choices=("structural", "learnability"), default="learnability"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--rgb-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7340)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--minimum-loss-reduction", type=float, default=0.15)
    parser.add_argument("--minimum-cosine-improvement", type=float, default=0.03)
    return parser.parse_args()


def _load_runtime(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("schema") != "wm3d_v8_materialized_runtime_v2":
        raise RuntimeError("micro-probe requires a materialized runtime")
    return value


def _load_objective(path: Path):
    profile = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    if profile.get("schema") != "wm3d_v8_objective_profile_v1":
        raise RuntimeError("micro-probe objective profile is invalid")
    return objective_config_from_mapping(profile["objective"])


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
        batch["state_normalization_offset"] = value["state_normalization_offset"]
        batch["state_normalization_scale"] = value["state_normalization_scale"]
        return
    closure = runtime["data_closure"]
    profile = load_data_profile(Path(str(closure["data_profile_path"])))
    artifact = json.loads(
        Path(str(closure["grouped_normalization_path"])).read_text(
            encoding="utf-8"
        )
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


def prepare_production_k8_batch(
    raw: dict[str, object],
    runtime: Mapping[str, object],
    *,
    rgb_size: int,
    state_normalization: Path | None = None,
) -> dict[str, object]:
    if raw.get("_schema") != "wm3d_fixed_materialized_batch_v1":
        raise RuntimeError("fixed real batch schema is invalid")
    if int(raw.get("_fixed_validation_seed", -1)) != 7340:
        raise RuntimeError("fixed real batch seed is invalid")
    batch = dict(raw)
    batch.pop("_schema")
    batch.pop("_fixed_validation_seed")
    _add_state_normalization(batch, runtime, state_normalization)
    if tuple(batch["future_factual_fine_action_values"].shape[:2]) != (1, 8):
        raise RuntimeError("micro-probe requires the real B1 K8 future action")
    if tuple(batch["world_times_s"].shape) != (1, 24):
        raise RuntimeError("micro-probe requires the real T16+K8 timestamps")
    if tuple(batch["rgb_frame_indices"].shape) != (1, 8):
        raise RuntimeError("micro-probe requires all K8 RGB horizons")
    batch["context_rgb"] = _resize_rgb(batch["context_rgb"], rgb_size)
    batch["target_rgb"] = _resize_rgb(batch["target_rgb"], rgb_size)
    for name in (
        "appearance_context_tokens",
        "appearance_context_mask",
        "target_appearance_tokens",
        "target_appearance_mask",
    ):
        batch.pop(name, None)
    return batch


def build_micro_model(
    runtime: Mapping[str, object], *, rgb_size: int, num_views: int
) -> NativeWorldModel:
    profile = dict(runtime["model_profile"])
    profile.pop("expected_parameter_count", None)
    profile["name"] = "native_micro_causal_normalized_transport_gate"
    model = dict(profile["model"])
    if not bool(model.get("rgb_action_owned_transport")):
        raise RuntimeError("runtime does not select action-owned transport")
    model.update(
        {
            "K": 8,
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
            "factual_v7_bridge_layers_state": [],
            "bridge_heads": 8,
            "dynamics_layers": 1,
            "factual_dynamics_repeats": 1,
            "factual_v7_early_action_conditioning": False,
            "factual_v7_early_action_scale": 0.0,
            "view_hidden": 64,
            "view_heads": 8,
            "view_ff_mult": 2.0,
            "max_action_substeps": 4,
            "max_policy_queries": 33,
            "time_fourier_dim": 32,
            "rgb_hidden": 128,
            "rgb_decode_chunk_size": 24,
            "rgb_size": rgb_size,
            "rgb_decode_indices": list(range(8)),
            "geom_hidden": 64,
            "appearance_enabled": False,
            "activation_checkpointing": False,
        }
    )
    profile["model"] = model
    result = build_world_model(profile)
    if not isinstance(result, NativeWorldModel):
        raise RuntimeError("micro-probe did not build the production model class")
    if result.cfg.K != 8 or result.cfg.T != 16:
        raise RuntimeError("micro-probe silently changed the production trajectory")
    return result


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float:
    weight = torch.broadcast_to(mask, value.shape).to(value.dtype)
    return float(((value.float().square() * weight).sum() / weight.sum()).sqrt())


def _masked_l1(value: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> float:
    weight = torch.broadcast_to(mask, value.shape).float()
    return float(((value.float() - truth.float()).abs().mul(weight).sum() / weight.sum()))


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
    if mode != "horizon_shuffle":
        raise ValueError(mode)
    result = dict(batch)
    # A one-horizon roll is not a reliable negative for real robot data: on
    # several OXE sources it closely matches normal actuator/observation
    # latency.  Use a distant, layout-preserving derangement instead.
    for name in (
        "future_factual_fine_action_values",
        "future_factual_coarse_action_values",
    ):
        horizon = int(batch[name].shape[1])
        if horizon < 2 or horizon % 2:
            raise RuntimeError("horizon derangement requires an even K >= 2")
        result[name] = batch[name].roll(horizon // 2, dims=1)
    return result


def _flow_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    disocclusion: torch.Tensor,
    rgb_mask: torch.Tensor,
) -> dict[str, float]:
    batch, horizon, views = prediction.shape[:3]
    grid = int(target.shape[-1])
    resized = F.interpolate(
        prediction.float().reshape(batch * horizon * views, 2, *prediction.shape[-2:]),
        size=(grid, grid),
        mode="bilinear",
        align_corners=True,
    ).reshape_as(target)
    slot_valid = rgb_mask[..., 0, 0, 0, None, None, None]
    valid = (
        slot_valid
        & (disocclusion < 0.5)
        & torch.isfinite(target).all(dim=3, keepdim=True)
    )
    weight = torch.broadcast_to(valid, target.shape).float()
    count = weight.sum().clamp_min(1.0)
    pred_rms = ((resized.square() * weight).sum() / count).sqrt()
    target_rms = ((target.float().square() * weight).sum() / count).sqrt()
    error_rms = (((resized - target.float()).square() * weight).sum() / count).sqrt()
    dot = (resized * target.float() * weight).sum()
    cosine = dot / (
        (resized.square() * weight).sum().sqrt()
        * (target.float().square() * weight).sum().sqrt()
    ).clamp_min(1.0e-12)
    return {
        "prediction_rms_pixels": float(pred_rms),
        "target_rms_pixels": float(target_rms),
        "amplitude_ratio": float(pred_rms / target_rms.clamp_min(1.0e-12)),
        "error_rms_pixels": float(error_rms),
        "direction_cosine": float(cosine),
    }


def evaluate(
    model: NativeWorldModel, batch: Mapping[str, torch.Tensor]
) -> tuple[dict[str, object], torch.Tensor]:
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        factual = _forward(model, batch, appearance_teacher_ratio=0.0)
        zero = _forward(model, _variant(batch, "zero"), appearance_teacher_ratio=0.0)
        shuffled = _forward(
            model,
            _variant(batch, "horizon_shuffle"),
            appearance_teacher_ratio=0.0,
        )
    target_tokens = batch["target_tokens"].float()
    token_mask = batch["target_token_mask"][..., None]
    target_rgb = batch["target_rgb"].float()
    rgb_mask = batch["target_rgb_mask"].bool()
    copy_last = batch["context_rgb"][:, None].expand_as(target_rgb)

    def branch_metrics(output: Mapping[str, torch.Tensor]) -> dict[str, object]:
        return {
            "token_error_rms": _masked_rms(
                output["pred_tokens"].float() - target_tokens, token_mask
            ),
            "rgb_l1": _masked_l1(output["rgb"], target_rgb, rgb_mask),
            "token_temporal": temporal_metrics(
                output["pred_tokens"].float(), target_tokens, token_mask
            ),
            "rgb_temporal": temporal_metrics(
                output["rgb"].float(), target_rgb, rgb_mask
            ),
            "flow": _flow_metrics(
                output["rgb_flow_pixels"],
                batch["rgb_flow_target_pixels"],
                batch["rgb_disocclusion_target"],
                rgb_mask,
            ),
        }

    normal = branch_metrics(factual)
    zero_metrics = branch_metrics(zero)
    shuffle_metrics = branch_metrics(shuffled)
    metrics = {
        "normal": normal,
        "zero": zero_metrics,
        "horizon_shuffle": shuffle_metrics,
        "copy_last_rgb_l1": _masked_l1(copy_last, target_rgb, rgb_mask),
        "gains": {
            "token_vs_zero": zero_metrics["token_error_rms"] - normal["token_error_rms"],
            "token_vs_shuffle": shuffle_metrics["token_error_rms"]
            - normal["token_error_rms"],
            "token_temporal_error_vs_shuffle": shuffle_metrics["token_temporal"][
                "delta_error_rms"
            ]
            - normal["token_temporal"]["delta_error_rms"],
            "rgb_vs_zero": zero_metrics["rgb_l1"] - normal["rgb_l1"],
            "rgb_vs_shuffle": shuffle_metrics["rgb_l1"] - normal["rgb_l1"],
            "token_response_zero_rms": _masked_rms(
                factual["pred_tokens"] - zero["pred_tokens"], token_mask
            ),
            "rgb_response_zero_rms": _masked_rms(
                factual["rgb"] - zero["rgb"], rgb_mask
            ),
            "flow_response_zero_rms": _masked_rms(
                factual["rgb_flow_pixels"] - zero["rgb_flow_pixels"], rgb_mask
            ),
        },
        "invariants": {
            "policy_equal_under_zero": torch.equal(
                factual["policy_action_raw"], zero["policy_action_raw"]
            ),
            "policy_equal_under_shuffle": torch.equal(
                factual["policy_action_raw"], shuffled["policy_action_raw"]
            ),
            "action_free_equal_under_zero": torch.equal(
                factual["action_free_pred_tokens"], zero["action_free_pred_tokens"]
            ),
            "action_free_equal_under_shuffle": torch.equal(
                factual["action_free_pred_tokens"],
                shuffled["action_free_pred_tokens"],
            ),
        },
    }
    return metrics, factual["rgb"][0, :, 0].detach().cpu()


def _gradient_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("micro-probe encountered a non-finite gradient")
        total += float(parameter.grad.detach().float().square().sum())
    return total**0.5


def save_gif(
    path: Path,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> None:
    target = batch["target_rgb"][0, :, 0].detach().cpu()
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
            "RGB", (sum(tile.width for tile in tiles), tiles[0].height + 18), "white"
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
    objective = _load_objective(args.objective)
    raw = torch.load(
        args.materialized_batch.resolve(strict=True),
        map_location="cpu",
        weights_only=False,
    )
    cpu_batch = prepare_production_k8_batch(
        raw,
        runtime,
        rgb_size=args.rgb_size,
        state_normalization=args.state_normalization,
    )
    batch = _batch_to_device(cpu_batch, device)
    flow_mapping = runtime["runtime_profile"]["train"].get("rgb_flow_teacher")
    if not isinstance(flow_mapping, dict):
        raise RuntimeError("production runtime does not define the RAFT flow target")
    flow_teacher = FrozenBidirectionalRAFTRuntime(
        raft_config_from_mapping(flow_mapping), device
    )
    batch = _materialize_rgb_flow_targets(batch, flow_teacher)
    del flow_teacher

    model = build_micro_model(
        runtime,
        rgb_size=args.rgb_size,
        num_views=int(cpu_batch["world_tokens"].shape[2]),
    ).to(device)
    perceptual = build_rgb_perceptual_model(objective, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.02
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.monotonic()
    before, before_rgb = evaluate(model, batch)
    save_gif(args.output / "before.gif", before_rgb, cpu_batch)
    initial_loss: float | None = None
    final_loss: float | None = None
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
                step=step,
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
            decoder = model.rgb_head.image_decoder
            gradient_norms = {
                "factual_action_encoder": _gradient_norm(model.factual_action),
                "grouped_state_conditioner": _gradient_norm(
                    model.factual_state_action_cross
                ),
                "early_factual_state_block": _gradient_norm(model.state_blocks[0]),
                "factual_token_output": _gradient_norm(model.factual_token_output),
                # The action-owned renderer has no direct action shortcut:
                # factual P64 is its only motion input.  Audit that actual
                # P64-to-pixel projection instead of a removed bypass module.
                "renderer_token_projection": _gradient_norm(decoder.token_proj),
                "flow_head": _gradient_norm(decoder.flow_head),
            }
            if any(value <= 0.0 for value in gradient_norms.values()):
                raise RuntimeError(
                    f"required factual/transport gradient is zero: {gradient_norms}"
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
                "token_gain": float(losses["action_counterfactual_token_gain"].detach()),
                "rgb_l1": float(losses["rgb_l1"].detach()),
                "rgb_motion_l1": float(losses["rgb_motion_l1"].detach()),
                "flow_epe_pixels": float(losses["rgb_flow_epe"].detach()),
                "flow_amplitude_ratio": float(
                    losses["rgb_flow_magnitude_ratio"].detach()
                ),
            }
            trace.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    after, after_rgb = evaluate(model, batch)
    save_gif(args.output / "after.gif", after_rgb, cpu_batch)
    elapsed = time.monotonic() - started
    assert initial_loss is not None and final_loss is not None
    loss_reduction = 1.0 - final_loss / max(initial_loss, 1.0e-12)
    common_checks = {
        "finite_and_fast": elapsed <= args.max_seconds,
        "required_gradients_nonzero": all(
            value > 0.0 for value in gradient_norms.values()
        ),
        "policy_and_action_free_invariant": all(
            bool(value) for value in after["invariants"].values()
        ),
        "real_k8_timestamps_preserved": tuple(batch["world_times_s"].shape) == (1, 24),
    }
    if args.mode == "structural":
        checks = {
            **common_checks,
            "factual_token_responds_to_action": (
                after["gains"]["token_response_zero_rms"] > 1.0e-6
            ),
            "transport_responds_to_action": (
                after["gains"]["flow_response_zero_rms"] > 1.0e-8
            ),
        }
    else:
        before_token_cosine = before["normal"]["token_temporal"][
            "delta_direction_cosine"
        ]
        after_token_cosine = after["normal"]["token_temporal"][
            "delta_direction_cosine"
        ]
        before_flow_cosine = before["normal"]["flow"]["direction_cosine"]
        after_flow_cosine = after["normal"]["flow"]["direction_cosine"]
        checks = {
            **common_checks,
            "focused_loss_reduced": loss_reduction >= args.minimum_loss_reduction,
            "token_factual_beats_zero": after["gains"]["token_vs_zero"] > 0.0,
            # Absolute token RMS is dominated by static content in this
            # single-sample tiny-model probe.  Keep its exact gain in the
            # receipt, but gate the action-sensitive temporal delta instead.
            "token_temporal_factual_beats_shuffle": (
                after["gains"]["token_temporal_error_vs_shuffle"] > 0.0
            ),
            "rgb_factual_beats_zero": after["gains"]["rgb_vs_zero"] > 0.0,
            "rgb_factual_beats_shuffle": after["gains"]["rgb_vs_shuffle"] > 0.0,
            "token_motion_direction_improved": after_token_cosine
            >= max(0.0, before_token_cosine + args.minimum_cosine_improvement),
            "flow_direction_improved": after_flow_cosine
            >= max(0.0, before_flow_cosine + args.minimum_cosine_improvement),
            "flow_amplitude_nontrivial": after["normal"]["flow"]["amplitude_ratio"]
            > 0.05,
        }
    receipt = {
        "schema": "wm3d_factual_motion_microprobe_v2",
        "purpose": "production K8 causal transport gate, not final quality",
        "mode": args.mode,
        "real_fixed_seed": 7340,
        "trajectory_shape": "T16+K8",
        "same_future_timestamp": False,
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
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
