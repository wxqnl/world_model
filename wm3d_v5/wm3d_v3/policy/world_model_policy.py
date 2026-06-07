"""World-model action selection helpers.

This module is the narrow policy-facing API for the current WM3D loop:

    context tokens + task embedding -> propose K action chunks
    -> simulate each chunk with the world core -> rank with evaluator heads
    -> return the selected action chunk.

It deliberately does not own online perception or simulator integration. External
benchmarks still need adapters that convert environment observations into VGGT
tokens and convert the selected 7-D action into the benchmark action space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ScoreWeights:
    progress: float = 1.0
    terminal: float = 1.0
    plausibility: float = 0.0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def score_world_output(out: dict[str, torch.Tensor], weights: ScoreWeights | None = None) -> torch.Tensor:
    """Return one learned scalar score per batch item from evaluator heads."""
    weights = weights or ScoreWeights()
    terms: list[torch.Tensor] = []
    norm = 0.0
    if weights.progress != 0 and "progress" in out:
        terms.append(torch.sigmoid(out["progress"].float()).mean(dim=1) * weights.progress)
        norm += abs(weights.progress)
    if weights.terminal != 0 and "terminal_success_logit" in out:
        terms.append(torch.sigmoid(out["terminal_success_logit"].float()) * weights.terminal)
        norm += abs(weights.terminal)
    if weights.plausibility != 0 and "plausibility_logit" in out:
        terms.append(torch.sigmoid(out["plausibility_logit"].float()) * weights.plausibility)
        norm += abs(weights.plausibility)
    if not terms:
        if "pred_tokens" not in out:
            raise ValueError("cannot infer batch size for zero score without pred_tokens")
        return out["pred_tokens"].new_zeros(out["pred_tokens"].shape[0], dtype=torch.float32)
    return torch.stack(terms, dim=0).sum(dim=0) / max(norm, 1e-6)


def _model_action_stats(
    model: torch.nn.Module,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    target = _unwrap_model(model)
    action_proj = getattr(target, "action_proj", None)
    if action_proj is None or not hasattr(action_proj, "mean") or not hasattr(action_proj, "std"):
        return None, None
    mean = action_proj.mean.detach().reshape(-1)[:6].float()
    std = action_proj.std.detach().reshape(-1)[:6].float().clamp_min(1e-6)
    return mean, std


def denormalize_action_cond(
    action_cond: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    binarize_gripper: bool = True,
) -> torch.Tensor:
    """Convert `[... , 7]` normalized action condition to raw 7-D actions.

    Pose axes are de-normalized with explicit `mean/std`, or with the model's
    `action_proj` buffers when available. If neither exists, pose axes are left
    in normalized units and callers can inspect `action_units` in probe reports.
    """
    if action_cond.shape[-1] != 7:
        raise ValueError(f"action_cond must end with 7 dims, got {tuple(action_cond.shape)}")
    if mean is None or std is None:
        if model is not None:
            mean, std = _model_action_stats(model)
    pose = action_cond[..., :6].float()
    if mean is not None and std is not None:
        mean = mean.to(device=pose.device, dtype=pose.dtype)
        std = std.to(device=pose.device, dtype=pose.dtype).clamp_min(1e-6)
        pose = pose * std + mean
    grip = action_cond[..., 6:7].float()
    if binarize_gripper:
        grip = (grip > 0.5).float()
    return torch.cat([pose, grip], dim=-1)


def _forward_world(
    model: torch.nn.Module,
    s: torch.Tensor,
    task_emb: torch.Tensor,
    action_cond: torch.Tensor | None,
    *,
    context_rgb: torch.Tensor | None,
    pixel: bool,
) -> dict[str, torch.Tensor]:
    kwargs: dict[str, Any] = {
        "action_cond": action_cond,
        "pixel": pixel,
        "bridging": False,
    }
    if context_rgb is not None:
        kwargs["context_rgb"] = context_rgb
    return model(s, task_emb, **kwargs)


def _plan_waypoint_action_chunk(plan_state: torch.Tensor, horizon: int) -> torch.Tensor:
    """Diagnostic object-centric waypoint servo from plan_state17 to raw actions.

    This is not the learned policy path. It lets the benchmark runner validate
    plan_state/object tracking and action-space conventions without activating
    video generation or the proposer.
    """
    if plan_state.ndim != 2 or plan_state.shape[-1] < 17:
        raise ValueError(f"plan_waypoint requires plan_state [B,>=17], got {tuple(plan_state.shape)}")
    plan = plan_state.float()
    bsz = plan.shape[0]
    stage = plan[:, :4].argmax(dim=1)
    target_to_eef = plan[:, 8:11]
    target_to_basket = plan[:, 11:14]
    basket_to_eef = plan[:, 14:17]
    is_pick = (stage == 0) | (stage == 2)
    goal = torch.where(is_pick[:, None], target_to_eef, basket_to_eef)
    gains = plan.new_tensor([4.0, 4.0, 4.0])
    pose_xyz = (goal * gains).clamp(-0.9, 0.9)
    pose = plan.new_zeros(bsz, 6)
    pose[:, :3] = pose_xyz
    place_close = target_to_basket[:, :2].norm(dim=1) < 0.06
    carry = (stage == 1) | (stage == 3)
    grip_closed = (carry & ~place_close).float()
    chunk = plan.new_zeros(bsz, horizon, 7)
    chunk[..., :6] = pose[:, None]
    chunk[..., 6] = grip_closed[:, None]
    return chunk


def _apply_stage3_place_overlay(action_raw: torch.Tensor, plan_state: torch.Tensor) -> torch.Tensor:
    """Diagnostic terminal place/release overlay for task1-style plan_state17.

    The learned policy remains in charge until the online plan tracker enters
    stage3. Stage3 is the current failure point: the policy contacts butter but
    releases or drifts before the object reaches the basket.
    """
    if plan_state.ndim != 2 or plan_state.shape[-1] < 17:
        raise ValueError(f"stage3 place overlay requires plan_state [B,>=17], got {tuple(plan_state.shape)}")
    if action_raw.ndim != 3 or action_raw.shape[-1] != 7:
        raise ValueError(f"action_raw must be [B,k,7], got {tuple(action_raw.shape)}")
    out = action_raw.clone()
    plan = plan_state.to(device=out.device, dtype=out.dtype)
    stage = plan[:, :4].argmax(dim=1)
    active = stage == 3
    if not bool(active.any().detach().cpu()):
        return out

    target_to_eef = plan[:, 8:11]
    target_to_basket = plan[:, 11:14]
    basket_to_eef = plan[:, 14:17]
    target_dist = target_to_eef.norm(dim=1)
    basket_xy = target_to_basket[:, :2].norm(dim=1)
    eef_basket_xy = basket_to_eef[:, :2].norm(dim=1)

    regrasp = target_dist > 0.10
    transport = basket_xy > 0.055
    release = ~transport

    servo = out.new_zeros(out.shape[0], 7)
    regrasp_xyz = (target_to_eef * out.new_tensor([4.5, 4.5, 4.5])).clamp(-0.9, 0.9)
    transport_xyz = out.new_zeros(out.shape[0], 3)
    transport_xyz[:, :2] = (target_to_basket[:, :2] * 4.2).clamp(-0.9, 0.9)
    lift = torch.where(basket_xy > 0.14, out.new_full((out.shape[0],), 0.85), out.new_full((out.shape[0],), -0.25))
    settle = torch.where(eef_basket_xy < 0.08, out.new_full((out.shape[0],), -0.35), lift)
    transport_xyz[:, 2] = settle
    servo[:, :3] = torch.where(regrasp[:, None], regrasp_xyz, transport_xyz)
    servo[:, 6] = torch.where(release, out.new_zeros(out.shape[0]), out.new_ones(out.shape[0]))
    servo[:, 6] = torch.where(regrasp & (target_dist > 0.13), out.new_zeros(out.shape[0]), servo[:, 6])

    out[active] = servo[active, None].expand(-1, out.shape[1], -1)
    return out


def _apply_terminal_reference_overlay(
    action_raw: torch.Tensor,
    plan_state: torch.Tensor,
    terminal_reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Nearest-neighbor terminal action overlay from a successful trace."""
    if plan_state.ndim != 2 or plan_state.shape[-1] < 17:
        raise ValueError(f"terminal reference overlay requires plan_state [B,>=17], got {tuple(plan_state.shape)}")
    features = terminal_reference.get("features")
    actions = terminal_reference.get("actions")
    if features is None or actions is None or features.ndim != 2 or actions.ndim != 2:
        raise ValueError("terminal_reference must contain features [N,F] and actions [N,7]")
    out = action_raw.clone()
    plan = plan_state.to(device=out.device, dtype=out.dtype)
    stage = plan[:, :4].argmax(dim=1)
    active = stage == 3
    if not bool(active.any().detach().cpu()):
        return out
    ref_f = features.to(device=out.device, dtype=out.dtype)
    ref_a = actions.to(device=out.device, dtype=out.dtype)
    query = plan[:, 8:17]
    dist = (query[:, None] - ref_f[None]).pow(2).mean(dim=2)
    nearest = ref_a[dist.argmin(dim=1)]
    out[active] = nearest[active, None].expand(-1, out.shape[1], -1)
    return out


def _apply_terminal_linear_overlay(
    action_raw: torch.Tensor,
    plan_state: torch.Tensor,
    terminal_reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Linear stage3 reference controller fit from a successful trace."""
    if plan_state.ndim != 2 or plan_state.shape[-1] < 17:
        raise ValueError(f"terminal linear overlay requires plan_state [B,>=17], got {tuple(plan_state.shape)}")
    weights = terminal_reference.get("linear_weights")
    if weights is None or weights.ndim != 2 or weights.shape[-1] != 7:
        raise ValueError("terminal_reference must contain linear_weights [F+1,7]")
    out = action_raw.clone()
    plan = plan_state.to(device=out.device, dtype=out.dtype)
    stage = plan[:, :4].argmax(dim=1)
    active = stage == 3
    if not bool(active.any().detach().cpu()):
        return out
    query = plan[:, 8:17]
    bias = torch.ones(query.shape[0], 1, device=query.device, dtype=query.dtype)
    design = torch.cat([bias, query], dim=1)
    pred = (design @ weights.to(device=out.device, dtype=out.dtype)).to(dtype=out.dtype)
    pred[..., :6] = pred[..., :6].clamp(-1.0, 1.0)
    pred[..., 6] = (pred[..., 6] > 0.5).to(dtype=pred.dtype)
    out[active] = pred[active, None].expand(-1, out.shape[1], -1)
    return out


def _apply_trace_linear_overlay(
    action_raw: torch.Tensor,
    plan_state: torch.Tensor,
    terminal_reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    """All-stage linear reference controller fit from a successful trace."""
    if plan_state.ndim != 2 or plan_state.shape[-1] < 17:
        raise ValueError(f"trace linear overlay requires plan_state [B,>=17], got {tuple(plan_state.shape)}")
    weights = terminal_reference.get("trace_linear_weights")
    if weights is None or weights.ndim != 2 or weights.shape[-1] != 7:
        raise ValueError("terminal_reference must contain trace_linear_weights [18,7]")
    out = action_raw.clone()
    plan = plan_state[:, :17].to(device=out.device, dtype=out.dtype)
    bias = torch.ones(plan.shape[0], 1, device=plan.device, dtype=plan.dtype)
    design = torch.cat([bias, plan], dim=1)
    pred = (design @ weights.to(device=out.device, dtype=out.dtype)).to(dtype=out.dtype)
    pred[..., :6] = pred[..., :6].clamp(-1.0, 1.0)
    pred[..., 6] = (pred[..., 6] > 0.5).to(dtype=pred.dtype)
    out = pred[:, None].expand(-1, out.shape[1], -1).clone()
    return out


@torch.no_grad()
def select_action_chunk(
    model: torch.nn.Module,
    s: torch.Tensor,
    task_emb: torch.Tensor,
    *,
    context_rgb: torch.Tensor | None = None,
    lowdim_state: torch.Tensor | None = None,
    object_state: torch.Tensor | None = None,
    plan_state: torch.Tensor | None = None,
    action_history: torch.Tensor | None = None,
    progress_state: torch.Tensor | None = None,
    pixel: bool = False,
    score_weights: ScoreWeights | None = None,
    selection_mode: str = "ranked",
    terminal_reference: dict[str, torch.Tensor] | None = None,
    return_rollouts: bool = False,
) -> dict[str, Any]:
    """Select one action chunk with propose -> simulate -> rank.

    Args:
        model: `JointWorldModel` or DDP-wrapped equivalent in eval mode.
        s: context VGGT tokens `[B,T,P,D]`.
        task_emb: Qwen/task embedding `[B,2048]`.
        context_rgb: optional last RGB frame for heads that need it.
        progress_state: optional normalized task phase `[B,1]` for direct policies.
        pixel: whether candidate rollouts should render RGB.
        score_weights: evaluator-head score weights.
        return_rollouts: include candidate rollout dicts; useful for debugging
            but expensive to keep around.
    """
    if selection_mode in ("plan_waypoint", "waypoint_servo"):
        if plan_state is None:
            raise RuntimeError("selection_mode=plan_waypoint requires plan_state")
        horizon = int(getattr(_unwrap_model(model).cfg.dual.state, "k", 1))
        selected_raw = _plan_waypoint_action_chunk(plan_state.to(device=s.device), horizon=horizon)
        score_t = selected_raw.new_zeros(selected_raw.shape[0], 1)
        selected_idx = torch.full((selected_raw.shape[0],), -3, dtype=torch.long, device=selected_raw.device)
        out: dict[str, Any] = {
            "candidate_action_cond": selected_raw[:, None],
            "candidate_scores": score_t,
            "selected_idx": selected_idx,
            "selected_score": score_t[:, 0],
            "selected_action_cond": selected_raw,
            "selected_action_raw": selected_raw,
            "selected_pose_norm": selected_raw[..., :6],
            "selected_gripper_prob": selected_raw[..., 6],
            "first_proposer_out": {"plan_waypoint_action": selected_raw},
        }
        if return_rollouts:
            out["candidate_rollouts"] = []
        return out

    if selection_mode in (
        "direct",
        "policy",
        "bc",
        "action_policy",
        "direct_prior",
        "prior_policy",
        "direct_stage3_place",
        "direct_terminal_nn",
        "direct_terminal_linear",
        "direct_trace_linear",
    ):
        target = _unwrap_model(model)
        action_policy = getattr(target, "action_policy", None)
        if action_policy is None:
            raise RuntimeError("selection_mode=direct requires enable_action_policy")
        policy_out = action_policy(
            s,
            task_emb=task_emb,
            lowdim_state=lowdim_state,
            object_state=object_state,
            plan_state=plan_state,
            action_history=action_history,
            progress_state=progress_state,
        )
        action_key = "prior_policy_action_cond" if selection_mode in ("direct_prior", "prior_policy") else "policy_action_cond"
        if action_key not in policy_out:
            raise RuntimeError(f"selection_mode={selection_mode} requires {action_key}")
        selected_cond = policy_out[action_key].float()
        if selected_cond.ndim != 3 or selected_cond.shape[-1] != 7:
            raise ValueError(f"expected policy_action_cond [B,k,7], got {tuple(selected_cond.shape)}")
        selected_raw = denormalize_action_cond(selected_cond, model=model)
        if selection_mode == "direct_stage3_place":
            if plan_state is None:
                raise RuntimeError("selection_mode=direct_stage3_place requires plan_state")
            selected_raw = _apply_stage3_place_overlay(selected_raw, plan_state.to(device=s.device))
        elif selection_mode == "direct_terminal_nn":
            if plan_state is None:
                raise RuntimeError("selection_mode=direct_terminal_nn requires plan_state")
            if terminal_reference is None:
                raise RuntimeError("selection_mode=direct_terminal_nn requires terminal_reference")
            selected_raw = _apply_terminal_reference_overlay(
                selected_raw,
                plan_state.to(device=s.device),
                terminal_reference,
            )
        elif selection_mode == "direct_terminal_linear":
            if plan_state is None:
                raise RuntimeError("selection_mode=direct_terminal_linear requires plan_state")
            if terminal_reference is None:
                raise RuntimeError("selection_mode=direct_terminal_linear requires terminal_reference")
            selected_raw = _apply_terminal_linear_overlay(
                selected_raw,
                plan_state.to(device=s.device),
                terminal_reference,
            )
        elif selection_mode == "direct_trace_linear":
            if plan_state is None:
                raise RuntimeError("selection_mode=direct_trace_linear requires plan_state")
            if terminal_reference is None:
                raise RuntimeError("selection_mode=direct_trace_linear requires terminal_reference")
            selected_raw = _apply_trace_linear_overlay(
                selected_raw,
                plan_state.to(device=s.device),
                terminal_reference,
            )
        score_t = selected_cond.new_zeros(selected_cond.shape[0], 1)
        selected_idx = torch.full((selected_cond.shape[0],), -1, dtype=torch.long, device=selected_cond.device)
        out: dict[str, Any] = {
            "candidate_action_cond": selected_cond[:, None],
            "candidate_scores": score_t,
            "selected_idx": selected_idx,
            "selected_score": score_t[:, 0],
            "selected_action_cond": selected_cond,
            "selected_action_raw": selected_raw,
            "selected_pose_norm": selected_cond[..., :6],
            "selected_gripper_prob": selected_cond[..., 6],
            "first_proposer_out": {"policy_action_cond": selected_cond},
        }
        if return_rollouts:
            out["candidate_rollouts"] = []
        return out

    first_out = _forward_world(
        model,
        s,
        task_emb,
        action_cond=None,
        context_rgb=context_rgb,
        pixel=False,
    )
    if "proposer_action_cond" not in first_out:
        raise RuntimeError("model output has no proposer_action_cond; enable_action_proposer is required")

    candidate_cond = first_out["proposer_action_cond"].float()
    if candidate_cond.ndim != 4 or candidate_cond.shape[-1] != 7:
        raise ValueError(f"expected proposer_action_cond [B,K,k,7], got {tuple(candidate_cond.shape)}")

    scores: list[torch.Tensor] = []
    rollouts: list[dict[str, torch.Tensor]] = []
    for ci in range(candidate_cond.shape[1]):
        cand_out = _forward_world(
            model,
            s,
            task_emb,
            action_cond=candidate_cond[:, ci].to(device=s.device, dtype=s.dtype),
            context_rgb=context_rgb,
            pixel=pixel,
        )
        scores.append(score_world_output(cand_out, score_weights))
        if return_rollouts:
            rollouts.append(cand_out)

    score_t = torch.stack(scores, dim=1)
    if selection_mode in ("ranked", "score", "evaluator"):
        selected_idx = score_t.argmax(dim=1)
    elif selection_mode in ("anchor", "first", "candidate0"):
        selected_idx = torch.zeros(score_t.shape[0], dtype=torch.long, device=score_t.device)
    else:
        raise ValueError(f"unknown selection_mode={selection_mode!r}; expected ranked, anchor, direct, or plan_waypoint")
    gather = selected_idx[:, None, None, None].expand(-1, 1, candidate_cond.shape[2], candidate_cond.shape[3])
    selected_cond = candidate_cond.gather(1, gather).squeeze(1)
    selected_raw = denormalize_action_cond(selected_cond, model=model)

    out: dict[str, Any] = {
        "candidate_action_cond": candidate_cond,
        "candidate_scores": score_t,
        "selected_idx": selected_idx,
        "selected_score": score_t.gather(1, selected_idx[:, None]).squeeze(1),
        "selected_action_cond": selected_cond,
        "selected_action_raw": selected_raw,
        "selected_pose_norm": selected_cond[..., :6],
        "selected_gripper_prob": selected_cond[..., 6],
        "first_proposer_out": first_out,
    }
    if return_rollouts:
        out["candidate_rollouts"] = rollouts
    return out


def selected_first_action(decision: dict[str, Any], *, raw: bool = True) -> torch.Tensor:
    key = "selected_action_raw" if raw else "selected_action_cond"
    action = decision[key]
    if action.ndim != 3:
        raise ValueError(f"{key} must be [B,k,7], got {tuple(action.shape)}")
    return action[:, 0]
