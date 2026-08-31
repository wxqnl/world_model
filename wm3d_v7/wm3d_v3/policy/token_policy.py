"""Checkpoint-backed WM3D policy for tokenized observations."""
from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.policy.world_model_policy import ScoreWeights, select_action_chunk, selected_first_action


@dataclass
class PolicyDecision:
    first_action_raw: torch.Tensor
    action_chunk_raw: torch.Tensor
    selected_idx: torch.Tensor
    selected_score: torch.Tensor
    candidate_scores: torch.Tensor
    raw: dict[str, Any]


def load_terminal_reference(path: str | Path) -> dict[str, torch.Tensor]:
    """Load stage3 terminal actions from a successful LIBERO trace."""
    data = json.loads(Path(path).read_text())
    features: list[list[float]] = []
    actions: list[list[float]] = []
    trace_features: list[list[float]] = []
    trace_actions: list[list[float]] = []
    for episode in data.get("results") or []:
        for item in episode.get("step_trace") or []:
            plan = item.get("plan_state")
            if not isinstance(plan, list) or len(plan) < 17:
                continue
            action = item.get("policy_action") or item.get("action")
            if not isinstance(action, list) or len(action) < 7:
                continue
            raw = [float(v) for v in action[:7]]
            raw[6] = 1.0 if raw[6] > 0.5 else 0.0
            trace_features.append([float(v) for v in plan[:17]])
            trace_actions.append(raw)
            stage = int(item.get("plan_stage", max(range(4), key=lambda idx: float(plan[idx]))))
            if stage != 3:
                continue
            features.append([float(v) for v in plan[8:17]])
            actions.append(raw)
    if not features:
        raise RuntimeError(f"no stage3 terminal reference rows found in {path}")
    feature_t = torch.tensor(features, dtype=torch.float32)
    action_t = torch.tensor(actions, dtype=torch.float32)
    design = torch.cat([torch.ones(feature_t.shape[0], 1), feature_t], dim=1)
    ridge = 1e-3 * torch.eye(design.shape[1], dtype=torch.float32)
    ridge[0, 0] = 0.0
    linear_weights = torch.linalg.solve(design.T @ design + ridge, design.T @ action_t)
    trace_feature_t = torch.tensor(trace_features, dtype=torch.float32)
    trace_action_t = torch.tensor(trace_actions, dtype=torch.float32)
    trace_design = torch.cat([torch.ones(trace_feature_t.shape[0], 1), trace_feature_t], dim=1)
    trace_ridge = 1e-3 * torch.eye(trace_design.shape[1], dtype=torch.float32)
    trace_ridge[0, 0] = 0.0
    trace_linear_weights = torch.linalg.solve(trace_design.T @ trace_design + trace_ridge, trace_design.T @ trace_action_t)
    return {
        "features": feature_t,
        "actions": action_t,
        "linear_weights": linear_weights,
        "trace_features": trace_feature_t,
        "trace_actions": trace_action_t,
        "trace_linear_weights": trace_linear_weights,
    }


def _expand_action_policy_horizon_tensor(key: str, value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor | None:
    if key not in {
        "action_policy.horizon_embed",
        "action_policy.prior_horizon_embed",
        "action_policy.local_residual_head.7.weight",
        "action_policy.local_residual_head.7.bias",
    }:
        return None
    if value.ndim == 3 and len(target_shape) == 3:
        if value.shape[0] != target_shape[0] or value.shape[2] != target_shape[2]:
            return None
        expanded = F.interpolate(
            value.permute(0, 2, 1).float(),
            size=int(target_shape[1]),
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)
        return expanded.to(dtype=value.dtype, device=value.device)
    action_dim = 7
    if value.ndim == 2 and len(target_shape) == 2:
        if value.shape[1] != target_shape[1]:
            return None
        old_h = int(value.shape[0]) // action_dim
        new_h = int(target_shape[0]) // action_dim
        if old_h <= 0 or new_h <= 0 or old_h * action_dim != value.shape[0] or new_h * action_dim != target_shape[0]:
            return None
        hidden = int(value.shape[1])
        x = value.reshape(old_h, action_dim, hidden).permute(1, 2, 0).reshape(1, action_dim * hidden, old_h)
        expanded = F.interpolate(x.float(), size=new_h, mode="linear", align_corners=True)
        expanded = expanded.reshape(action_dim, hidden, new_h).permute(2, 0, 1).reshape(new_h * action_dim, hidden)
        return expanded.to(dtype=value.dtype, device=value.device)
    if value.ndim == 1 and len(target_shape) == 1:
        old_h = int(value.shape[0]) // action_dim
        new_h = int(target_shape[0]) // action_dim
        if old_h <= 0 or new_h <= 0 or old_h * action_dim != value.shape[0] or new_h * action_dim != target_shape[0]:
            return None
        x = value.reshape(old_h, action_dim).permute(1, 0).unsqueeze(0)
        expanded = F.interpolate(x.float(), size=new_h, mode="linear", align_corners=True)
        expanded = expanded.squeeze(0).permute(1, 0).reshape(new_h * action_dim)
        return expanded.to(dtype=value.dtype, device=value.device)
    return None


def _load_compatible_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str], list[str], list[str]]:
    current = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    expanded: list[str] = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        elif key in current:
            expanded_value = _expand_action_policy_horizon_tensor(key, value, current[key].shape)
            if expanded_value is not None and expanded_value.shape == current[key].shape:
                compatible[key] = expanded_value
                expanded.append(key)
            else:
                skipped.append(key)
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys), skipped, expanded


def _load_overlay_checkpoint(
    model: torch.nn.Module,
    overlay_ckpt_path: str | Path,
) -> None:
    payload = torch.load(overlay_ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("overlay checkpoint must contain a model state_dict")
    overlay = payload["model"]
    if not overlay:
        raise RuntimeError("overlay checkpoint model state_dict must not be empty")

    target = model.state_dict()
    module_prefixes = {name for name, _module in model.named_children()}
    unprefixed: list[str] = []
    unknown: list[str] = []
    non_tensor: list[str] = []
    shape_mismatches: list[str] = []
    for key, value in overlay.items():
        if not isinstance(key, str):
            unknown.append(repr(key))
            continue
        prefix, separator, _suffix = key.partition(".")
        if not separator or prefix not in module_prefixes:
            unprefixed.append(key)
            continue
        if key not in target:
            unknown.append(key)
            continue
        if not isinstance(value, torch.Tensor):
            non_tensor.append(key)
            continue
        if value.shape != target[key].shape:
            shape_mismatches.append(
                f"{key}: overlay={tuple(value.shape)} target={tuple(target[key].shape)}"
            )

    errors = []
    if unprefixed:
        errors.append(f"keys missing a complete module prefix: {sorted(unprefixed)}")
    if unknown:
        errors.append(f"unknown keys: {sorted(unknown)}")
    if non_tensor:
        errors.append(f"non-tensor values: {sorted(non_tensor)}")
    if shape_mismatches:
        errors.append(f"shape mismatch: {sorted(shape_mismatches)}")
    if errors:
        raise RuntimeError("invalid overlay checkpoint: " + "; ".join(errors))
    model.load_state_dict(overlay, strict=False)


def _load_checkpoint_model(
    cfg_path: str | Path,
    ckpt_path: str | Path,
) -> torch.nn.Module:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    file_cfg = yaml.safe_load(Path(cfg_path).read_text())
    ckpt_cfg = payload.get("cfg")
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = payload.get("base_cfg")
    model_keys = payload["model"].keys()
    inferred_geom_mode = None
    if any(key.startswith("geom.ups.0.block.") for key in model_keys):
        inferred_geom_mode = "resize_conv"
    elif any(key.startswith("geom.ups.0.0.") for key in model_keys):
        inferred_geom_mode = "transpose"

    def with_geom_mode(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
        out = copy.deepcopy(cfg)
        out.setdefault("model", {})["geom_upsample_mode"] = mode
        return out

    cfg_candidates = []
    if ckpt_cfg is not None and inferred_geom_mode is not None:
        cfg_candidates.append(with_geom_mode(ckpt_cfg, inferred_geom_mode))
    if inferred_geom_mode is not None:
        cfg_candidates.append(with_geom_mode(file_cfg, inferred_geom_mode))
    if ckpt_cfg is not None:
        cfg_candidates.append(ckpt_cfg)
    if not cfg_candidates or ckpt_cfg != file_cfg:
        cfg_candidates.append(file_cfg)
    last_error: RuntimeError | None = None
    for cfg in cfg_candidates:
        try:
            candidate = build_model(cfg)
            candidate.load_state_dict(payload["model"], strict=True)
        except RuntimeError as exc:
            last_error = exc
            continue
        return candidate
    assert last_error is not None
    raise last_error


def _prune_anchor_model(model: torch.nn.Module) -> None:
    """Keep only the frozen OFT serving path required by act_policy."""
    for name in (
        "geom",
        "context_pixel",
        "pixel",
        "control_head",
        "progress_head",
        "action_proposer",
        "bridging",
        "aux_idm",
    ):
        if hasattr(model, name):
            setattr(model, name, None)


class WM3DTokenPolicy:
    """Policy adapter for already-tokenized observations.

    External benchmark runners still need an observation adapter that turns RGB
    frames into VGGT token windows. Once they have `[B,T,P,D]` tokens and one
    Qwen task embedding, this class owns the common WM3D action-selection path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        anchor_model: torch.nn.Module | None = None,
        device: str | torch.device = "cuda:0",
        score_weights: ScoreWeights | None = None,
        selection_mode: str = "ranked",
        terminal_reference: dict[str, torch.Tensor] | None = None,
        flow_sample: bool | None = None,
        flow_sample_steps: int | None = None,
        flow_noise_scale: float | None = None,
        flow_seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        if anchor_model is not None and hasattr(model, "action_policy"):
            model.action_policy = None
        self.model = model.to(self.device).eval()
        self.anchor_model = (
            anchor_model.to(self.device).eval() if anchor_model is not None else None
        )
        self.score_weights = score_weights or ScoreWeights()
        self.selection_mode = selection_mode
        self.terminal_reference = terminal_reference
        self.flow_sample = flow_sample
        self.flow_sample_steps = flow_sample_steps
        self.flow_noise_scale = flow_noise_scale
        self.flow_seed = flow_seed
        self._flow_call_index = 0

    @classmethod
    def from_checkpoint(
        cls,
        cfg_path: str | Path,
        ckpt_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        overlay_ckpt_path: str | Path | None = None,
        anchor_cfg_path: str | Path | None = None,
        anchor_ckpt_path: str | Path | None = None,
        score_weights: ScoreWeights | None = None,
        selection_mode: str = "ranked",
        terminal_reference_path: str | Path | None = None,
        flow_sample: bool | None = None,
        flow_sample_steps: int | None = None,
        flow_noise_scale: float | None = None,
        flow_seed: int | None = None,
    ) -> "WM3DTokenPolicy":
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        file_cfg = yaml.safe_load(Path(cfg_path).read_text())
        ckpt_cfg = sd.get("cfg")
        if not isinstance(ckpt_cfg, dict):
            ckpt_cfg = sd.get("base_cfg")
        model_keys = sd["model"].keys()
        inferred_geom_mode = None
        if any(key.startswith("geom.ups.0.block.") for key in model_keys):
            inferred_geom_mode = "resize_conv"
        elif any(key.startswith("geom.ups.0.0.") for key in model_keys):
            inferred_geom_mode = "transpose"

        def with_geom_mode(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
            out = copy.deepcopy(cfg)
            out.setdefault("model", {})["geom_upsample_mode"] = mode
            return out

        cfg_candidates = []
        if ckpt_cfg is not None and inferred_geom_mode is not None:
            cfg_candidates.append(with_geom_mode(ckpt_cfg, inferred_geom_mode))
        if inferred_geom_mode is not None:
            cfg_candidates.append(with_geom_mode(file_cfg, inferred_geom_mode))
        if ckpt_cfg is not None:
            cfg_candidates.append(ckpt_cfg)
        if not cfg_candidates or ckpt_cfg != file_cfg:
            cfg_candidates.append(file_cfg)
        last_error: RuntimeError | None = None
        model = None
        for cfg in cfg_candidates:
            try:
                candidate = build_model(cfg)
                candidate.load_state_dict(sd["model"], strict=True)
            except RuntimeError as exc:
                last_error = exc
                continue
            model = candidate
            break
        if model is None:
            assert last_error is not None
            raise last_error
        if overlay_ckpt_path is not None:
            _load_overlay_checkpoint(model, overlay_ckpt_path)
        anchor_model = None
        if anchor_ckpt_path is not None:
            anchor_model = _load_checkpoint_model(
                anchor_cfg_path or cfg_path,
                anchor_ckpt_path,
            )
            _prune_anchor_model(anchor_model)
        terminal_reference = load_terminal_reference(terminal_reference_path) if terminal_reference_path else None
        return cls(
            model,
            anchor_model=anchor_model,
            device=device,
            score_weights=score_weights,
            selection_mode=selection_mode,
            terminal_reference=terminal_reference,
            flow_sample=flow_sample,
            flow_sample_steps=flow_sample_steps,
            flow_noise_scale=flow_noise_scale,
            flow_seed=flow_seed,
        )

    @torch.no_grad()
    def act_from_tokens(
        self,
        context_tokens: torch.Tensor,
        task_emb: torch.Tensor,
        *,
        wrist_tokens: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        lowdim_state: torch.Tensor | None = None,
        object_state: torch.Tensor | None = None,
        plan_state: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        progress_state: torch.Tensor | None = None,
        flow_sample: bool | None = None,
        flow_sample_steps: int | None = None,
        flow_noise_scale: float | None = None,
        flow_seed: int | None = None,
        return_rollouts: bool = False,
    ) -> PolicyDecision:
        context_tokens = context_tokens.to(self.device, non_blocking=True)
        task_emb = task_emb.to(self.device, non_blocking=True)
        if wrist_tokens is not None:
            wrist_tokens = wrist_tokens.to(self.device, non_blocking=True)
        if view_mask is not None:
            view_mask = view_mask.to(self.device, non_blocking=True).bool()
        if context_rgb is not None:
            context_rgb = context_rgb.to(self.device, non_blocking=True)
        if lowdim_state is not None:
            lowdim_state = lowdim_state.to(self.device, non_blocking=True)
        if object_state is not None:
            object_state = object_state.to(self.device, non_blocking=True)
        if plan_state is not None:
            plan_state = plan_state.to(self.device, non_blocking=True)
        if action_history is not None:
            action_history = action_history.to(self.device, non_blocking=True)
        if progress_state is not None:
            progress_state = progress_state.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            effective_flow_sample = self.flow_sample if flow_sample is None else flow_sample
            effective_flow_steps = self.flow_sample_steps if flow_sample_steps is None else flow_sample_steps
            effective_flow_noise_scale = self.flow_noise_scale if flow_noise_scale is None else flow_noise_scale

            def _select() -> dict[str, Any]:
                return select_action_chunk(
                    self.model,
                    context_tokens,
                    task_emb,
                    anchor_model=self.anchor_model,
                    wrist_tokens=wrist_tokens,
                    view_mask=view_mask,
                    context_rgb=context_rgb,
                    lowdim_state=lowdim_state,
                    object_state=object_state,
                    plan_state=plan_state,
                    action_history=action_history,
                    progress_state=progress_state,
                    flow_sample=effective_flow_sample,
                    flow_sample_steps=effective_flow_steps,
                    flow_noise_scale=effective_flow_noise_scale,
                    pixel=False,
                    score_weights=self.score_weights,
                    selection_mode=self.selection_mode,
                    terminal_reference=self.terminal_reference,
                    return_rollouts=return_rollouts,
                )

            effective_flow_seed = self.flow_seed if flow_seed is None else flow_seed
            if effective_flow_seed is None:
                raw = _select()
            else:
                seed = int(effective_flow_seed)
                if flow_seed is None:
                    seed += int(self._flow_call_index)
                    self._flow_call_index += 1
                if self.device.type == "cuda":
                    device_index = self.device.index
                    devices = [device_index] if device_index is not None else None
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        raw = _select()
                else:
                    with torch.random.fork_rng():
                        torch.manual_seed(seed)
                        raw = _select()
        return PolicyDecision(
            first_action_raw=selected_first_action(raw, raw=True).detach().float().cpu(),
            action_chunk_raw=raw["selected_action_raw"].detach().float().cpu(),
            selected_idx=raw["selected_idx"].detach().cpu(),
            selected_score=raw["selected_score"].detach().float().cpu(),
            candidate_scores=raw["candidate_scores"].detach().float().cpu(),
            raw=raw,
        )

    @property
    def action_stats_model(self) -> torch.nn.Module:
        return self.anchor_model if self.anchor_model is not None else self.model
