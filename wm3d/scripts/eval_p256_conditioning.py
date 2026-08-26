"""Audit whether predicted per-view P256 latents preserve future motion."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


_PACKAGE = os.environ.get("WM3D_P256_AUDIT_PACKAGE", "wm3d")
offline_eval = importlib.import_module(f"{_PACKAGE}.training.offline_eval")


_ORIGINAL_FORWARD = offline_eval._forward_with_action_counterfactual
_ORIGINAL_OBJECTIVE = offline_eval.compute_native_objective
_ORIGINAL_SAVE_DEMO = offline_eval.save_rgb_depth_demo
_PREFIX = "_p256_conditioning_audit_"
_previous_prediction: torch.Tensor | None = None


def _appearance_dynamics(model: torch.nn.Module) -> torch.nn.Module:
    matches = [
        module
        for name, module in model.named_modules()
        if name.endswith("appearance_dynamics")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"P256 audit expected one appearance dynamics module, found {len(matches)}"
        )
    return matches[0]


def _last_context(
    context_tokens: torch.Tensor, context_mask: torch.Tensor, horizon: int
) -> torch.Tensor:
    if context_mask.shape != context_tokens.shape[:-1]:
        raise RuntimeError("P256 context mask does not align to tokens")
    latest = torch.zeros_like(context_tokens[:, 0])
    for index in range(int(context_tokens.shape[1])):
        latest = torch.where(
            context_mask[:, index, ..., None].bool(),
            context_tokens[:, index],
            latest,
        )
    return latest[:, None].expand(-1, horizon, -1, -1, -1)


def _forward_with_variant(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    variant: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[Mapping[str, torch.Tensor], torch.Tensor]:
    module = _appearance_dynamics(model)
    effective = next(iter(batch.values())).new_ones((), dtype=torch.float32)

    def replace(
        _module: torch.nn.Module,
        module_args: tuple[torch.Tensor, ...],
        module_output: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal effective
        predicted, predicted_mask = module_output
        context_tokens, context_mask = module_args[:2]
        if variant == "last_context":
            replacement = _last_context(
                context_tokens, context_mask, int(predicted.shape[1])
            )
        elif variant == "shuffle":
            if (
                _previous_prediction is None
                or _previous_prediction.shape != predicted.shape
            ):
                replacement = predicted
                effective = effective.new_zeros(())
            else:
                replacement = _previous_prediction.to(
                    device=predicted.device, dtype=predicted.dtype
                )
        elif variant == "teacher":
            replacement = batch["target_appearance_tokens"].to(
                device=predicted.device, dtype=predicted.dtype
            )
        else:
            raise RuntimeError(f"unknown P256 audit variant: {variant}")
        if replacement.shape != predicted.shape:
            raise RuntimeError("P256 audit replacement shape differs")
        replacement_mask = predicted_mask
        if variant == "teacher":
            replacement_mask = (
                replacement_mask
                & batch["target_appearance_mask"].to(device=predicted.device).bool()
            )
        replacement = replacement * replacement_mask[..., None].to(
            dtype=replacement.dtype
        )
        return replacement, replacement_mask

    handle = module.register_forward_hook(replace)
    try:
        output = _ORIGINAL_FORWARD(model, batch, *args, **kwargs)
    finally:
        handle.remove()
    return output, effective


def _masked_mean(
    value: torch.Tensor, mask: torch.Tensor, *, epsilon: float = 1.0e-6
) -> torch.Tensor:
    weight = torch.broadcast_to(mask, value.shape).to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(epsilon)


def _masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(_masked_mean(value.float().square(), mask).clamp_min(0.0))


def _audited_forward(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> Mapping[str, torch.Tensor]:
    global _previous_prediction

    baseline = dict(_ORIGINAL_FORWARD(model, batch, *args, **kwargs))
    prediction = baseline["appearance_pred_tokens"]
    prediction_mask = baseline["appearance_pred_mask"].bool()
    last_context = _last_context(
        batch["appearance_context_tokens"],
        batch["appearance_context_mask"],
        int(prediction.shape[1]),
    )
    target = batch["target_appearance_tokens"].to(dtype=prediction.dtype)
    mask = prediction_mask & batch["target_appearance_mask"].bool()
    target_delta = target - last_context
    prediction_delta = prediction - last_context
    target_delta_rms = _masked_rms(target_delta, mask[..., None])
    prediction_delta_rms = _masked_rms(prediction_delta, mask[..., None])
    baseline[f"{_PREFIX}target_delta_rms"] = target_delta_rms
    baseline[f"{_PREFIX}prediction_delta_rms"] = prediction_delta_rms
    baseline[f"{_PREFIX}prediction_delta_ratio"] = prediction_delta_rms / (
        target_delta_rms + 1.0e-6
    )
    baseline[f"{_PREFIX}prediction_delta_error_rms"] = _masked_rms(
        prediction_delta - target_delta, mask[..., None]
    )

    for variant in ("last_context", "shuffle", "teacher"):
        changed, effective = _forward_with_variant(
            model, batch, variant, *args, **kwargs
        )
        baseline[f"{_PREFIX}{variant}_rgb"] = changed["rgb"]
        baseline[f"{_PREFIX}{variant}_blend"] = changed["rgb_blend"]
        baseline[f"{_PREFIX}{variant}_motion_logit"] = changed["rgb_motion_logit"]
        baseline[f"{_PREFIX}{variant}_effective"] = effective
    _previous_prediction = prediction.detach().clone()
    return baseline


def _temporal_change_rms(
    rgb: torch.Tensor,
    context_rgb: torch.Tensor,
    rgb_mask: torch.Tensor,
    context_mask: torch.Tensor,
) -> torch.Tensor:
    video = torch.cat((context_rgb[:, None], rgb), dim=1)
    delta = video[:, 1:] - video[:, :-1]
    valid = rgb_mask.bool() & context_mask[:, None, :, None, None, None]
    return _masked_rms(delta, valid)


def _audited_objective(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    losses = dict(
        _ORIGINAL_OBJECTIVE(*args, output=output, batch=batch, config=config, **kwargs)
    )
    target = batch["target_rgb"].float()
    context = batch["context_rgb"].float()
    rgb_mask = batch.get(
        "target_rgb_mask",
        torch.ones_like(target[:, :, :, :1, :1, :1], dtype=torch.bool),
    ).bool()
    context_mask = batch.get(
        "context_rgb_mask",
        torch.ones(
            target.shape[0],
            target.shape[2],
            dtype=torch.bool,
            device=target.device,
        ),
    ).bool()
    valid = rgb_mask & context_mask[:, None, :, None, None, None]
    motion = (
        (target - context[:, None]).abs().mean(dim=3, keepdim=True)
        > float(config.rgb_motion_threshold)
    ) & valid
    static = (~motion) & valid
    target_temporal = _temporal_change_rms(target, context, rgb_mask, context_mask)
    losses["p256_target_rgb_temporal_change_rms"] = target_temporal

    variants = {"normal": output["rgb"]}
    variants.update(
        {
            name: output[f"{_PREFIX}{name}_rgb"]
            for name in ("last_context", "shuffle", "teacher")
        }
    )
    blends = {"normal": output["rgb_blend"]}
    motion_logits = {"normal": output["rgb_motion_logit"]}
    for name in ("last_context", "shuffle", "teacher"):
        blends[name] = output[f"{_PREFIX}{name}_blend"]
        motion_logits[name] = output[f"{_PREFIX}{name}_motion_logit"]
    normal = variants["normal"].float()
    for name, rgb in variants.items():
        value = rgb.float()
        error = (value - target).abs()
        temporal = _temporal_change_rms(value, context, rgb_mask, context_mask)
        losses[f"p256_{name}_rgb_l1"] = _masked_mean(error, valid)
        losses[f"p256_{name}_rgb_motion_l1"] = _masked_mean(error, motion)
        losses[f"p256_{name}_rgb_static_l1"] = _masked_mean(error, static)
        losses[f"p256_{name}_rgb_temporal_change_rms"] = temporal
        losses[f"p256_{name}_rgb_temporal_change_ratio"] = temporal / (
            target_temporal + 1.0e-6
        )
        losses[f"p256_{name}_rgb_response_vs_normal_rms"] = _masked_rms(
            value - normal, valid
        )
        quality = offline_eval.rgb_quality_metrics(
            {"rgb": value},
            batch,
            motion_threshold=float(config.rgb_motion_threshold),
        )
        for metric, metric_value in quality.items():
            losses[f"p256_{name}_{metric}"] = metric_value

        blend = blends[name].float()
        motion_probability = motion_logits[name].float().sigmoid()
        losses[f"p256_{name}_blend_mean"] = _masked_mean(blend, valid)
        losses[f"p256_{name}_blend_motion_mean"] = _masked_mean(blend, motion)
        losses[f"p256_{name}_blend_static_mean"] = _masked_mean(blend, static)
        losses[f"p256_{name}_motion_probability_mean"] = _masked_mean(
            motion_probability, valid
        )
        losses[f"p256_{name}_motion_probability_motion_mean"] = _masked_mean(
            motion_probability, motion
        )
        losses[f"p256_{name}_motion_probability_static_mean"] = _masked_mean(
            motion_probability, static
        )
        if name != "normal":
            losses[f"p256_{name}_effective"] = output[f"{_PREFIX}{name}_effective"]
    for name in (
        "target_delta_rms",
        "prediction_delta_rms",
        "prediction_delta_ratio",
        "prediction_delta_error_rms",
    ):
        losses[f"p256_{name}"] = output[f"{_PREFIX}{name}"]
    return losses


def _save_demos(
    root: Path,
    *,
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    sample_index: int,
    file_index: int,
) -> list[str]:
    paths: list[str] = []
    for name in ("normal", "last_context", "shuffle", "teacher"):
        variant_output = dict(output)
        if name != "normal":
            variant_output["rgb"] = output[f"{_PREFIX}{name}_rgb"]
        paths.extend(
            _ORIGINAL_SAVE_DEMO(
                root / name,
                output=variant_output,
                batch=batch,
                sample_index=sample_index,
                file_index=file_index,
            )
        )
    return paths


def main() -> None:
    diagnostic_skip_launch = "--diagnostic-skip-launch-qualification" in sys.argv
    if diagnostic_skip_launch:
        sys.argv.remove("--diagnostic-skip-launch-qualification")
        offline_eval._require_recent_resource_preflight = lambda *args, **kwargs: None
        offline_eval._publish_and_validate_launch = lambda *args, **kwargs: (None, None)
    if "--appearance-teacher-ratio" not in sys.argv:
        sys.argv.extend(("--appearance-teacher-ratio", "0"))
    offline_eval._forward_with_action_counterfactual = _audited_forward
    offline_eval.compute_native_objective = _audited_objective
    offline_eval.save_rgb_depth_demo = _save_demos
    offline_eval.main()


if __name__ == "__main__":
    main()
