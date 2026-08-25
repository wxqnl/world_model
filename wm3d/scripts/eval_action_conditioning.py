"""Measure whether a trained Stage0 policy uses real VLA conditioning inputs."""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


_PACKAGE = os.environ.get("WM3D_ACTION_AUDIT_PACKAGE", "wm3d")
offline_eval = importlib.import_module(f"{_PACKAGE}.training.offline_eval")


_ORIGINAL_FORWARD = offline_eval._forward
_ORIGINAL_OBJECTIVE = offline_eval.compute_native_objective
_PREFIX = "_policy_conditioning_audit_"
_previous_task: torch.Tensor | None = None
_previous_inputs: dict[str, torch.Tensor] = {}


def _changed_fraction(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if before.shape != after.shape or before.ndim == 0:
        raise RuntimeError("conditioning audit replacement shape differs")
    return before.ne(after).reshape(before.shape[0], -1).any(dim=1).float().mean()


def _roll(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] < 2:
        return value
    order = torch.arange(value.shape[0], device=value.device).roll(1)
    return value.index_select(0, order)


def _variant(
    batch: Mapping[str, torch.Tensor],
    keys: tuple[str, ...],
    *,
    first_replacement: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    result = dict(batch)
    effective: torch.Tensor | None = None
    for index, key in enumerate(keys):
        if key not in batch:
            continue
        value = batch[key]
        preferred = (
            first_replacement
            if index == 0 and first_replacement is not None
            else _previous_inputs.get(key)
        )
        replacement = (
            preferred
            if preferred is not None and preferred.shape == value.shape
            else _roll(value)
        )
        if replacement.shape != value.shape:
            continue
        result[key] = replacement
        fraction = _changed_fraction(value, replacement)
        effective = fraction if effective is None else torch.maximum(effective, fraction)
    if effective is None:
        tensor = next(value for value in batch.values() if isinstance(value, torch.Tensor))
        effective = tensor.new_zeros((), dtype=torch.float32)
    return result, effective


def _audited_forward(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> Mapping[str, torch.Tensor]:
    global _previous_task
    baseline = dict(_ORIGINAL_FORWARD(model, batch, *args, **kwargs))
    current_task = batch.get("task_embedding")
    previous = None
    if (
        current_task is not None
        and _previous_task is not None
        and _previous_task.shape == current_task.shape
    ):
        previous = _previous_task.to(device=current_task.device)

    variants = {
        "task": _variant(
            batch, ("task_embedding",), first_replacement=previous
        ),
        "observation": _variant(batch, ("world_tokens", "view_mask")),
        "current_state": _variant(
            batch, ("current_state_values", "current_state_mask")
        ),
        "history_action": _variant(
            batch,
            (
                "history_fine_action_values",
                "history_fine_action_mask",
                "history_fine_action_dt",
                "history_fine_sample_mask",
                "history_coarse_action_values",
                "history_coarse_action_mask",
            ),
        ),
    }
    fields = (
        "policy_action_raw",
        "policy_action_normalized",
        "policy_action_mask",
        "policy_gripper_mask",
        "policy_binary_mask",
    )
    for name, (changed_batch, effective) in variants.items():
        changed_output = _ORIGINAL_FORWARD(
            model, changed_batch, *args, **kwargs
        )
        baseline[f"{_PREFIX}{name}_effective_fraction"] = effective
        for field in fields:
            if field in changed_output:
                baseline[f"{_PREFIX}{name}_{field}"] = changed_output[field]
    if current_task is not None:
        _previous_task = current_task.detach().clone()
    for key in (
        "world_tokens",
        "view_mask",
        "current_state_values",
        "current_state_mask",
        "history_fine_action_values",
        "history_fine_action_mask",
        "history_fine_action_dt",
        "history_fine_sample_mask",
        "history_coarse_action_values",
        "history_coarse_action_mask",
    ):
        value = batch.get(key)
        if value is not None:
            _previous_inputs[key] = value.detach().clone()
    return baseline


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = torch.broadcast_to(mask, value.shape).to(dtype=value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _fine_loss(
    raw: torch.Tensor,
    normalized: torch.Tensor,
    output_mask: torch.Tensor,
    binary_mask: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    huber_delta: float,
) -> torch.Tensor:
    target = batch["target_fine_action"]
    mask = batch["target_fine_action_mask"] & output_mask
    binary = mask & binary_mask
    continuous = mask & ~binary_mask
    continuous_loss = _masked_mean(
        F.smooth_l1_loss(
            normalized, target, reduction="none", beta=huber_delta
        ),
        continuous,
    )
    binary_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            raw, target.clamp(0, 1), reduction="none"
        ),
        binary,
    )
    return continuous_loss + binary_loss


def _audited_objective(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    objective_output = dict(output)
    if (
        float(getattr(config, "action_counterfactual_token_advantage", 0.0)) > 0.0
        and "zero_action_pred_tokens" not in objective_output
    ):
        # Older checkpoints did not request the training-only zero-action
        # branch during offline evaluation.  The conditioning audit only
        # compares policy actions, so provide a neutral token control solely
        # to keep the historical objective compatible.
        objective_output["zero_action_pred_tokens"] = objective_output[
            "pred_tokens"
        ]
    losses = dict(
        _ORIGINAL_OBJECTIVE(
            *args,
            output=objective_output,
            batch=batch,
            config=config,
            **kwargs,
        )
    )
    baseline_normalized = output["policy_action_normalized"]
    baseline_mask = output["policy_action_mask"]
    baseline_binary = output.get(
        "policy_binary_mask", output["policy_gripper_mask"]
    )
    for name in ("task", "observation", "current_state", "history_action"):
        raw = output[f"{_PREFIX}{name}_policy_action_raw"]
        normalized = output[f"{_PREFIX}{name}_policy_action_normalized"]
        variant_mask = output[f"{_PREFIX}{name}_policy_action_mask"]
        binary = output.get(
            f"{_PREFIX}{name}_policy_binary_mask",
            output[f"{_PREFIX}{name}_policy_gripper_mask"],
        )
        losses[f"policy_audit_{name}_action_fine"] = _fine_loss(
            raw,
            normalized,
            variant_mask,
            binary,
            batch,
            float(config.huber_delta),
        )
        losses[f"policy_audit_{name}_output_rms"] = torch.sqrt(
            _masked_mean(
                (normalized - baseline_normalized).square(),
                baseline_mask & variant_mask,
            )
        )
        losses[f"policy_audit_{name}_effective_fraction"] = output[
            f"{_PREFIX}{name}_effective_fraction"
        ]

    neutral_raw = torch.zeros_like(output["policy_action_raw"])
    neutral_normalized = torch.where(
        baseline_binary, torch.full_like(neutral_raw, 0.5), neutral_raw
    )
    losses["policy_audit_neutral_action_fine"] = _fine_loss(
        neutral_raw,
        neutral_normalized,
        baseline_mask,
        baseline_binary,
        batch,
        float(config.huber_delta),
    )
    recomputed = _fine_loss(
        output["policy_action_raw"],
        baseline_normalized,
        baseline_mask,
        baseline_binary,
        batch,
        float(config.huber_delta),
    )
    losses["policy_audit_baseline_consistency"] = (
        losses["action_fine"] - recomputed
    ).abs()
    return losses


def main() -> None:
    compat_shim: Path | None = None
    if "--compat-shim" in sys.argv:
        index = sys.argv.index("--compat-shim")
        try:
            compat_shim = Path(sys.argv[index + 1]).resolve(strict=True)
        except (IndexError, OSError) as exc:
            raise SystemExit(
                "--compat-shim requires an existing Python file"
            ) from exc
        del sys.argv[index : index + 2]
    diagnostic_skip_launch = "--diagnostic-skip-launch-qualification" in sys.argv
    if diagnostic_skip_launch:
        sys.argv.remove("--diagnostic-skip-launch-qualification")
        offline_eval._require_recent_resource_preflight = lambda *args, **kwargs: None
        offline_eval._publish_and_validate_launch = (
            lambda *args, **kwargs: (None, None)
        )
    offline_eval._forward = _audited_forward
    offline_eval.compute_native_objective = _audited_objective
    if compat_shim is None:
        offline_eval.main()
    else:
        runpy.run_path(str(compat_shim), run_name="__main__")


if __name__ == "__main__":
    main()
