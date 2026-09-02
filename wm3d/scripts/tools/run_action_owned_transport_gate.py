#!/usr/bin/env python3
"""Train the production action-owned RGB renderer on one sealed real sequence.

This is a renderer capacity/causality gate, not a reduced model or a quality
benchmark. It uses the production 256x256 renderer, real target P64 tokens,
the real canonical physical action adapter, and the release RGB objective.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

from wm3d.models.model_factory import validate_model_profile
from wm3d.models.native_world_model import (
    NativeActionOwnedTransportRGBImageDecoder,
    OriginalV7RGBActionAdapter,
    native_config_from_mapping,
)
from wm3d.training.native_objective import (
    _image_gradient,
    _masked_rgb_perceptual,
    build_rgb_perceptual_model,
    objective_config_from_mapping,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--objective", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7340)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = torch.broadcast_to(mask, value.shape).to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0e-6)


def _rgb_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    motion_logit: torch.Tensor,
    objective,
    perceptual_model: torch.nn.Module | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Shapes are [K,3,H,W] and [K,1,H,W]. This is algebraically the same
    # release objective applied to the one valid camera in the sealed sample.
    error = prediction - target
    valid = torch.ones_like(target[:, :1], dtype=torch.bool)
    motion_value = (target.float() - context.float()).abs().mean(dim=1, keepdim=True)
    motion_mask = motion_value > float(objective.rgb_motion_threshold)
    static_mask = ~motion_mask
    l1 = _masked_mean(error.abs(), valid)
    motion_weight = 1.0 + float(objective.rgb_motion_gain) * motion_mask.to(error.dtype)
    motion_l1 = _masked_mean(error.abs() * motion_weight, valid)

    pred_dy, pred_dx = _image_gradient(prediction)
    target_dy, target_dx = _image_gradient(target)
    gradient = 0.5 * (
        (pred_dy - target_dy).abs().mean() + (pred_dx - target_dx).abs().mean()
    )

    motion_target = motion_mask.to(dtype=motion_logit.dtype)
    motion_bce = F.binary_cross_entropy_with_logits(
        motion_logit.float(),
        motion_target.float(),
        pos_weight=torch.as_tensor(
            float(objective.rgb_motion_pos_weight),
            device=prediction.device,
        ),
    )
    probability = torch.sigmoid(motion_logit.float())
    intersection = (probability * motion_target).flatten(1).sum(dim=1)
    denominator = (probability + motion_target).flatten(1).sum(dim=1)
    motion_dice = (1.0 - (2.0 * intersection + 1.0e-6) / (denominator + 1.0e-6)).mean()

    if perceptual_model is None:
        perceptual = prediction.new_zeros(())
    else:
        perceptual = _masked_rgb_perceptual(
            prediction[None, :, None],
            target[None, :, None],
            torch.ones(
                1,
                target.shape[0],
                1,
                1,
                1,
                1,
                dtype=torch.bool,
                device=prediction.device,
            ),
            perceptual_model,
            chunk_size=4,
        )

    total = (
        float(objective.rgb_l1) * l1
        + float(objective.rgb_perceptual) * perceptual
        + float(objective.rgb_gradient) * gradient
        + float(objective.rgb_motion_l1) * motion_l1
        + float(objective.rgb_motion_bce) * motion_bce
        + float(objective.rgb_motion_dice) * motion_dice
    )
    terms = {
        "total": total,
        "l1": l1,
        "perceptual": perceptual,
        "gradient": gradient,
        "motion_weighted_l1": motion_l1,
        "motion_bce": motion_bce,
        "motion_dice": motion_dice,
        "motion_region_l1": _masked_mean(error.abs(), motion_mask),
        "static_region_l1": _masked_mean(error.abs(), static_mask),
        "motion_fraction": motion_mask.float().mean(),
    }
    return total, terms


def _cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    weight = torch.broadcast_to(mask, left.shape).to(left.dtype)
    a = (left * weight).flatten()
    b = (right * weight).flatten()
    denom = a.norm() * b.norm()
    if float(denom) <= 1.0e-12:
        return 0.0
    return float((a @ b / denom).item())


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    motion_logit: torch.Tensor,
    flow: torch.Tensor,
    disocclusion_logit: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    pred = prediction.float()
    tgt = target.float()
    ctx = context.float()
    motion = (tgt - ctx).abs().mean(dim=1, keepdim=True) > threshold
    static = ~motion
    pred_delta = pred - ctx
    target_delta = tgt - ctx
    copy_l1 = (ctx - tgt).abs().mean()
    pred_l1 = (pred - tgt).abs().mean()
    pred_temporal = pred[1:] - pred[:-1]
    target_temporal = tgt[1:] - tgt[:-1]
    pred_temporal_rms = pred_temporal.square().mean().sqrt()
    target_temporal_rms = target_temporal.square().mean().sqrt()
    target_delta_rms = target_delta.square().mean().sqrt()
    pred_delta_rms = pred_delta.square().mean().sqrt()
    return {
        "l1": float(pred_l1.item()),
        "copy_last_l1": float(copy_l1.item()),
        "improvement_over_copy_l1": float((copy_l1 - pred_l1).item()),
        "motion_region_l1": float(_masked_mean((pred - tgt).abs(), motion).item()),
        "static_region_l1": float(_masked_mean((pred - tgt).abs(), static).item()),
        "context_delta_rms": float(pred_delta_rms.item()),
        "target_context_delta_rms": float(target_delta_rms.item()),
        "context_delta_amplitude_ratio": float(
            (pred_delta_rms / target_delta_rms.clamp_min(1.0e-12)).item()
        ),
        "context_delta_direction_cosine": _cosine(pred_delta, target_delta, motion),
        "temporal_delta_rms": float(pred_temporal_rms.item()),
        "target_temporal_delta_rms": float(target_temporal_rms.item()),
        "temporal_delta_amplitude_ratio": float(
            (pred_temporal_rms / target_temporal_rms.clamp_min(1.0e-12)).item()
        ),
        "temporal_delta_direction_cosine": _cosine(
            pred_temporal,
            target_temporal,
            motion[1:] | motion[:-1],
        ),
        "predicted_motion_fraction": float(torch.sigmoid(motion_logit.float()).mean().item()),
        "flow_rms_pixels": float(flow.float().square().mean().sqrt().item()),
        "flow_motion_region_rms_pixels": float(
            _masked_mean(flow.float().square(), motion).sqrt().item()
        ),
        "flow_static_region_rms_pixels": float(
            _masked_mean(flow.float().square(), static).sqrt().item()
        ),
        "warp_invalid_fraction": float((disocclusion_logit.float() > 0).float().mean().item()),
    }


def _to_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _save_panel_gif(
    output: Path,
    target: torch.Tensor,
    prediction: torch.Tensor,
    copy_last: torch.Tensor,
) -> None:
    frames: list[Image.Image] = []
    for index in range(target.shape[0]):
        tiles = [_to_image(target[index]), _to_image(prediction[index]), _to_image(copy_last[index])]
        canvas = Image.new("RGB", (tiles[0].width * 3, tiles[0].height + 24), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (label, tile) in enumerate(zip(("target", "prediction", "copy-last"), tiles)):
            canvas.paste(tile, (column * tile.width, 24))
            draw.text((column * tile.width + 4, 4), f"{label} h{index}", fill="black")
        frames.append(canvas)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=350, loop=0)


def main() -> None:
    args = _parse_args()
    if args.steps <= 0 or args.lr <= 0:
        raise ValueError("steps and learning rate must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    profile = _load_yaml(args.profile)
    validate_model_profile(profile)
    cfg = native_config_from_mapping(profile["model"])
    if not cfg.rgb_action_owned_transport or cfg.rgb_original_v7_context:
        raise ValueError("gate requires the action-owned transport profile")
    objective_profile = _load_yaml(args.objective)
    objective = objective_config_from_mapping(objective_profile["objective"])
    batch = torch.load(args.batch.resolve(strict=True), map_location="cpu", weights_only=False)
    if batch.get("_schema") != "wm3d_fixed_materialized_batch_v1":
        raise ValueError("gate requires a sealed fixed materialized batch")
    if int(batch["target_tokens"].shape[0]) != 1:
        raise ValueError("gate currently requires exactly one real sequence")

    valid_views = (
        batch["target_rgb_mask"][0, :, :, 0, 0, 0].all(dim=0)
        & batch["context_rgb_mask"][0]
    ).nonzero(as_tuple=False).flatten()
    if valid_views.numel() == 0:
        raise ValueError("sealed sample has no camera valid across K")
    view = int(valid_views[0].item())
    device = torch.device(args.device)

    action_adapter = OriginalV7RGBActionAdapter(cfg)
    with torch.no_grad():
        action = action_adapter(
            fine_values=batch["future_factual_fine_action_values"],
            fine_dim_mask=batch["future_factual_fine_action_mask"],
            fine_sample_mask=batch["future_factual_fine_sample_mask"],
            coarse_values=batch["future_factual_coarse_action_values"],
            coarse_dim_mask=batch["future_factual_coarse_action_mask"],
            action_semantic_ids=batch["action_semantic_ids"],
            group_mask=batch["action_group_mask"],
            normalization_offset=batch["action_normalization_offset"],
            normalization_scale=batch["action_normalization_scale"],
        )[0]

    decoder = NativeActionOwnedTransportRGBImageDecoder(cfg).to(device=device)
    view_embedding = torch.nn.Parameter(
        torch.empty(1, cfg.rgb_hidden, 1, 1, device=device)
    )
    torch.nn.init.normal_(view_embedding, std=0.02)
    parameters = list(decoder.parameters()) + [view_embedding]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    perceptual_model = build_rgb_perceptual_model(objective, device=device)

    tokens = batch["target_tokens"][0].to(device=device)
    target = batch["target_rgb"][0, :, view].to(device=device)
    context_bank = batch["context_rgb"][0, view : view + 1].to(device=device)
    context = context_bank.expand(cfg.K, -1, -1, -1)
    action = action.to(device=device)
    task = batch["task_embedding"][0:1].to(device=device).expand(cfg.K, -1)
    context_indices = torch.zeros(cfg.K, dtype=torch.long, device=device)

    def forward(use_tokens: torch.Tensor, use_action: torch.Tensor):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return decoder(
                use_tokens,
                view_embedding.expand(cfg.K, -1, -1, -1),
                None,
                None,
                use_action,
                task,
                context_bank,
                context_indices,
            )

    logs: list[dict[str, float]] = []
    decoder.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction, motion_logit, _, flow, _ = forward(tokens, action)
        loss, terms = _rgb_terms(
            prediction, target, context, motion_logit, objective, perceptual_model
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {"step": step, "grad_norm": float(grad_norm)}
            record.update({name: float(value.detach()) for name, value in terms.items()})
            print(json.dumps(record, sort_keys=True), flush=True)
            logs.append(record)

    decoder.eval()
    with torch.no_grad():
        normal = forward(tokens, action)
        reverse = torch.arange(cfg.K - 1, -1, -1, device=device)
        shuffled_tokens = forward(tokens.index_select(0, reverse), action)
        shuffled_action = forward(tokens, action.index_select(0, reverse))
        zero_action = forward(tokens, torch.zeros_like(action))

    normal_metrics = _metrics(
        normal[0], target, context, normal[1], normal[3], normal[4], objective.rgb_motion_threshold
    )
    variants = {
        "normal": normal_metrics,
        "reversed_p64": _metrics(
            shuffled_tokens[0], target, context, shuffled_tokens[1], shuffled_tokens[3], shuffled_tokens[4], objective.rgb_motion_threshold
        ),
        "reversed_action": _metrics(
            shuffled_action[0], target, context, shuffled_action[1], shuffled_action[3], shuffled_action[4], objective.rgb_motion_threshold
        ),
        "zero_action": _metrics(
            zero_action[0], target, context, zero_action[1], zero_action[3], zero_action[4], objective.rgb_motion_threshold
        ),
    }
    variants["normal"]["rgb_response_vs_zero_rms"] = float(
        (normal[0].float() - zero_action[0].float()).square().mean().sqrt().item()
    )
    variants["normal"]["rgb_response_vs_reversed_p64_rms"] = float(
        (normal[0].float() - shuffled_tokens[0].float()).square().mean().sqrt().item()
    )
    variants["normal"]["flow_response_vs_zero_rms_pixels"] = float(
        (normal[3].float() - zero_action[3].float()).square().mean().sqrt().item()
    )
    variants["normal"]["flow_response_vs_reversed_action_rms_pixels"] = float(
        (normal[3].float() - shuffled_action[3].float()).square().mean().sqrt().item()
    )

    # The gate checks capability and causal use, not convergence to a final
    # image-quality threshold. Normal must beat copy-last and broken alignment,
    # and its motion direction must be positive on the real sequence.
    passed = bool(
        variants["normal"]["improvement_over_copy_l1"] > 0.0
        and variants["normal"]["l1"] < variants["reversed_p64"]["l1"]
        and variants["normal"]["l1"] < variants["reversed_action"]["l1"]
        and variants["normal"]["l1"] < variants["zero_action"]["l1"]
        and variants["normal"]["context_delta_direction_cosine"] > 0.0
        and variants["normal"]["temporal_delta_direction_cosine"] > 0.0
        and variants["normal"]["temporal_delta_direction_cosine"]
        > variants["zero_action"]["temporal_delta_direction_cosine"]
        and variants["normal"]["temporal_delta_direction_cosine"]
        > variants["reversed_action"]["temporal_delta_direction_cosine"]
        and variants["normal"]["temporal_delta_direction_cosine"]
        > variants["reversed_p64"]["temporal_delta_direction_cosine"]
        and variants["normal"]["rgb_response_vs_zero_rms"] > 1.0e-5
        and variants["normal"]["rgb_response_vs_reversed_p64_rms"] > 1.0e-5
        and variants["normal"]["flow_response_vs_zero_rms_pixels"] > 1.0e-4
        and variants["normal"]["flow_response_vs_reversed_action_rms_pixels"]
        > 1.0e-4
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _save_panel_gif(
        args.output / "target_prediction_copy_last.gif",
        target,
        normal[0],
        context,
    )
    receipt = {
        "schema": "wm3d_action_owned_transport_gate_v2",
        "passed": passed,
        "sample_index": int(batch["sample_index"][0]),
        "source_id": int(batch["source_id"][0]),
        "view": view,
        "steps": args.steps,
        "learning_rate": args.lr,
        "train_log": logs,
        "variants": variants,
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
