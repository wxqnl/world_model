"""Causal contracts for Stage2 policy-conditioned world-model rollouts."""
from __future__ import annotations

import copy
import socket
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist


class Stage2PolicyClosedLoopError(ValueError):
    """Raised when a Stage2 rollout would violate the serving contract."""


def _global_mean_with_ddp_grad(
    local_sum: torch.Tensor,
    global_sum: torch.Tensor,
    global_count: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Expose a global mean while compensating for later gradient averaging."""

    count = global_count.to(device=local_sum.device, dtype=local_sum.dtype)
    if float(count.detach().cpu()) <= 0.0:
        return local_sum * 0.0
    global_value = global_sum.to(device=local_sum.device, dtype=local_sum.dtype) / count
    grad_carrier = local_sum * (float(world_size) / count)
    return grad_carrier + (global_value - grad_carrier.detach())


def _distributed_active_mean(values: torch.Tensor) -> torch.Tensor:
    """Average positive hinge violations without dilution by satisfied entries."""

    active = (values.detach() > 0.0).to(device=values.device, dtype=values.dtype)
    local_sum = (values * active).sum()
    local_count = active.sum()
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return local_sum / local_count.clamp_min(1.0)

    world_size = dist.get_world_size()
    reduce_on_cpu = dist.get_backend() == "gloo" and values.is_cuda
    reduce_device = torch.device("cpu") if reduce_on_cpu else values.device
    reduced = torch.stack((local_sum.detach(), local_count.detach())).to(reduce_device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return _global_mean_with_ddp_grad(
        local_sum,
        reduced[0],
        reduced[1],
        world_size,
    )


_GATE_GRIP_PARTITIONS = (
    "pos",
    "neg",
    "transition_up",
    "transition_down",
    "boundary_up",
    "boundary_down",
)


def _distributed_pair_sum_count_means(
    local_student_sums: torch.Tensor,
    local_reference_sums: torch.Tensor,
    local_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return DDP-global paired means with gradients only through the student."""

    if not (
        local_student_sums.ndim == 1
        and local_student_sums.shape == local_reference_sums.shape
        and local_student_sums.shape == local_counts.shape
    ):
        raise Stage2PolicyClosedLoopError(
            "paired global means require equally shaped 1D sum/count tensors"
        )
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        safe_counts = local_counts.clamp_min(1.0)
        support = local_counts > 0.0
        student_means = torch.where(
            support,
            local_student_sums / safe_counts,
            local_student_sums * 0.0,
        )
        reference_means = torch.where(
            support,
            local_reference_sums / safe_counts,
            local_reference_sums * 0.0,
        )
        return student_means, reference_means, local_counts.detach()

    world_size = dist.get_world_size()
    reduce_on_cpu = dist.get_backend() == "gloo" and local_student_sums.is_cuda
    reduce_device = torch.device("cpu") if reduce_on_cpu else local_student_sums.device
    reduced = torch.stack(
        (
            local_student_sums.detach(),
            local_reference_sums.detach(),
            local_counts.detach(),
        ),
        dim=0,
    ).to(reduce_device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    global_student_sums = reduced[0].to(local_student_sums.device)
    global_reference_sums = reduced[1].to(local_student_sums.device)
    global_counts = reduced[2].to(local_student_sums.device)
    safe_counts = global_counts.clamp_min(1.0)
    support = global_counts > 0.0
    global_student_values = global_student_sums / safe_counts
    grad_carrier = local_student_sums * (float(world_size) / safe_counts)
    student_means = torch.where(
        support,
        grad_carrier + (global_student_values - grad_carrier.detach()),
        local_student_sums * 0.0,
    )
    reference_means = torch.where(
        support,
        global_reference_sums / safe_counts,
        local_reference_sums * 0.0,
    )
    return student_means, reference_means, global_counts.detach()


def _teacher_relative_ceiling(
    reference_mean: torch.Tensor,
    *,
    performance_ratio: float,
    tolerance: float,
) -> torch.Tensor:
    """Use the stricter of the fixed relative gate and absolute tolerance."""

    return torch.minimum(
        reference_mean * float(performance_ratio),
        reference_mean + float(tolerance),
    )


def _supported_active_mean(
    losses: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Macro-average globally supported, currently violated gate partitions."""

    active = ((counts > 0.0) & (losses.detach() > 0.0)).to(losses.dtype)
    return (losses * active).sum() / active.sum().clamp_min(1.0)


def _gate_grip_partitions(
    target_grip: torch.Tensor,
    action_prev_grip: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the exact six masks consumed by the fixed closed-loop gate."""

    if target_grip.ndim != 2 or int(target_grip.shape[1]) <= 0:
        raise Stage2PolicyClosedLoopError("gripper target must have shape [B,K]")
    previous_boundary = action_prev_grip.reshape(action_prev_grip.shape[0], -1)[:, -1]
    if int(previous_boundary.shape[0]) != int(target_grip.shape[0]):
        raise Stage2PolicyClosedLoopError(
            "action_prev_grip batch size does not match gripper target"
        )
    target = target_grip > 0.5
    previous = torch.cat(((previous_boundary > 0.5)[:, None], target[:, :-1]), dim=1)
    up = target & (~previous)
    down = (~target) & previous
    boundary = torch.zeros_like(target)
    boundary[:, 0] = True
    return {
        "pos": target,
        "neg": ~target,
        "transition_up": up & (~boundary),
        "transition_down": down & (~boundary),
        "boundary_up": up & boundary,
        "boundary_down": down & boundary,
    }


class FrozenReferenceServingPolicy(torch.nn.Module):
    """Exact frozen core_pred/no-action to OFT serving path.

    The reference core and OFT must come from the same checkpoint. Feeding a
    historical OFT with tokens from the current core is not a valid teacher:
    those latent spaces were trained jointly and are not interchangeable.
    """

    def __init__(
        self,
        *,
        dual: torch.nn.Module,
        action_proj: torch.nn.Module,
        action_policy: torch.nn.Module,
        policy_action_add_trunk: bool,
    ) -> None:
        super().__init__()
        self.dual = dual
        self.action_proj = action_proj
        self.action_policy = action_policy
        self.policy_action_add_trunk = bool(policy_action_add_trunk)

    @staticmethod
    def _match_horizon(value: torch.Tensor, horizon: int) -> torch.Tensor:
        if int(value.shape[1]) == int(horizon):
            return value
        if int(value.shape[1]) > int(horizon):
            return value[:, :horizon]
        pad = value[:, -1:].expand(
            -1,
            int(horizon) - int(value.shape[1]),
            *value.shape[2:],
        )
        return torch.cat((value, pad), dim=1)

    def forward(
        self,
        s: torch.Tensor,
        *,
        task_emb: torch.Tensor,
        context_rgb: torch.Tensor | None,
        **policy_kwargs,
    ) -> dict[str, torch.Tensor]:
        dual_out = self.dual(s, task_emb, action_cond=None)
        policy_out = self.action_policy(
            dual_out["pred_tokens"],
            task_emb=task_emb,
            context_rgb=context_rgb,
            **policy_kwargs,
        )
        if "policy_pose_norm" not in policy_out:
            return policy_out

        projected = self.action_proj(dual_out["z_a"])
        horizon = int(policy_out["policy_pose_norm"].shape[1])
        trunk_pose = self._match_horizon(projected["pose_norm"], horizon)
        trunk_grip = self._match_horizon(projected["gripper_logit"], horizon)
        if self.policy_action_add_trunk:
            pose_norm = trunk_pose + policy_out["policy_pose_norm"]
            gripper_logit = trunk_grip + policy_out["policy_gripper_logit"]
            action_cond = torch.cat(
                (pose_norm, torch.sigmoid(gripper_logit)[..., None]),
                dim=-1,
            )
        else:
            pose_norm = policy_out["policy_pose_norm"]
            gripper_logit = policy_out["policy_gripper_logit"]
            action_cond = policy_out.get("policy_action_cond")
            if action_cond is None:
                action_cond = torch.cat(
                    (pose_norm, torch.sigmoid(gripper_logit)[..., None]),
                    dim=-1,
                )

        out = dict(policy_out)
        out["policy_pose_norm"] = pose_norm
        out["policy_gripper_logit"] = gripper_logit
        out["policy_action_cond"] = action_cond
        out["trunk_pose_norm"] = trunk_pose
        out["trunk_gripper_logit"] = trunk_grip
        out["trunk_pose"] = projected["pose"]
        return out


def cadence_normalized_reference_weight(base_weight: float, every: int) -> float:
    """Keep a sparse closed-loop objective's average per-update weight stable."""

    base = float(base_weight)
    cadence = int(every)
    if not math.isfinite(base) or base <= 0.0:
        raise Stage2PolicyClosedLoopError(
            "reference grip base weight must be finite and positive"
        )
    if cadence <= 0:
        raise Stage2PolicyClosedLoopError(
            "reference grip cadence must be positive"
        )
    return base * cadence


def validate_frozen_reference_policy_contract(
    *,
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> dict[str, object]:
    """Require an exact OFT contract modulo detach-only context semantics.

    A frozen teacher cannot propagate gradients. ``core_pred`` and
    ``core_pred_detach`` therefore produce the same teacher inputs and forward
    values; every other contract field remains exact.
    """

    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        raise Stage2PolicyClosedLoopError(
            "frozen reference action-policy contracts must be mappings"
        )

    def _forward_contract(contract: Mapping[str, object]) -> dict[str, object]:
        normalized = copy.deepcopy(dict(contract))
        joint = normalized.get("joint_behavior")
        if isinstance(joint, Mapping):
            joint = dict(joint)
            source = joint.get("policy_context_source")
            if source in {"core_pred", "core_pred_detach"}:
                joint["policy_context_source"] = "core_pred_forward_value"
            normalized["joint_behavior"] = joint
        return normalized

    if _forward_contract(expected) != _forward_contract(actual):
        raise Stage2PolicyClosedLoopError(
            "frozen reference action-policy contract is not forward-equivalent"
        )

    expected_joint = expected.get("joint_behavior")
    actual_joint = actual.get("joint_behavior")
    expected_source = (
        expected_joint.get("policy_context_source")
        if isinstance(expected_joint, Mapping)
        else None
    )
    actual_source = (
        actual_joint.get("policy_context_source")
        if isinstance(actual_joint, Mapping)
        else None
    )
    return {
        "forward_equivalent": True,
        "detach_only_context_difference": expected_source != actual_source,
        "expected_context_source": expected_source,
        "actual_context_source": actual_source,
    }


def validate_policy_only_fsdp_wrap_report(
    report: Mapping[str, object],
) -> None:
    """Validate FSDP coverage when only the OFT/action policy is trainable."""

    roots = {"wm", "wan_transformer", "wan_control_adapter"}
    errors: list[str] = []
    if report.get("enabled") is not True:
        errors.append("FSDP must be enabled")
    if report.get("use_orig_params") is not True:
        errors.append("FSDP use_orig_params must be true")
    if set(report.get("modules") or ()) != roots:
        errors.append("FSDP modules are not the exact roots")

    coverage = report.get("root_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != roots:
        errors.append("FSDP root coverage keys are not exact")
    else:
        for root in sorted(roots):
            entry = coverage.get(root)
            if not isinstance(entry, Mapping):
                errors.append(f"FSDP root {root} coverage is missing")
                continue
            total = int(entry.get("trainable_tensors", 0) or 0)
            covered = int(entry.get("covered_trainable_tensors", 0) or 0)
            uncovered = int(entry.get("uncovered_trainable_tensors", 0) or 0)
            if covered != total or uncovered != 0:
                errors.append(f"FSDP root {root} has uncovered trainable tensors")
            if root == "wm" and total <= 0:
                errors.append("FSDP wm root must contain the trainable action policy")
            if root != "wm" and total != 0:
                errors.append(f"FSDP root {root} must be fully frozen")

    if errors:
        raise Stage2PolicyClosedLoopError(
            "policy-only Stage2 FSDP wrap report failed: " + "; ".join(errors)
        )


@dataclass(frozen=True)
class Stage2PolicyClosedLoopSources:
    """Auditable source ledger for the second policy decision in a pair."""

    token_prefix: str = "a_observed_overlap"
    token_suffix: str = "core_pred_from_policy_action"
    context_rgb: str = "wan_rollout_last_frame"
    task_context: str = "a_fixed_task_context"
    policy_state: str = "a_last_observed_state"
    action_history: str = "executed_policy_action_tail"
    target_action: str = "supervision_only"
    counterfactual_context: str = "real_core_and_wan_rollout_negative"

    def as_dict(self) -> dict[str, str]:
        return {
            "token_prefix": self.token_prefix,
            "token_suffix": self.token_suffix,
            "context_rgb": self.context_rgb,
            "task_context": self.task_context,
            "policy_state": self.policy_state,
            "action_history": self.action_history,
            "target_action": self.target_action,
            "counterfactual_context": self.counterfactual_context,
        }


def resolve_stage2_policy_pair_report(
    train_cfg: Mapping[str, object],
    *,
    hostname: str | None = None,
) -> str:
    """Resolve the immutable pair report for the current data host.

    Window-token shard indexes are node-local. A multi-node run therefore
    requires a separately built immutable report for every hostname instead
    of reusing one node's cache identity on another node.
    """

    reports = train_cfg.get("stage2_policy_closed_loop_pair_reports")
    legacy = train_cfg.get("stage2_policy_closed_loop_pair_report")
    if reports is not None:
        if legacy not in (None, ""):
            raise Stage2PolicyClosedLoopError(
                "closed-loop Stage2 pair report config is ambiguous"
            )
        if not isinstance(reports, Mapping) or not reports:
            raise Stage2PolicyClosedLoopError(
                "stage2_policy_closed_loop_pair_reports must be a non-empty mapping"
            )
        current_host = str(hostname or socket.gethostname()).strip()
        raw_path = reports.get(current_host)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise Stage2PolicyClosedLoopError(
                f"no immutable pair report is bound for hostname={current_host!r}"
            )
        path = Path(raw_path)
        if not path.is_absolute():
            raise Stage2PolicyClosedLoopError(
                f"pair report for hostname={current_host!r} must be absolute"
            )
        return str(path)
    if not isinstance(legacy, str) or not legacy.strip():
        raise Stage2PolicyClosedLoopError(
            "closed-loop Stage2 requires an immutable pair report"
        )
    path = Path(legacy)
    if not path.is_absolute():
        raise Stage2PolicyClosedLoopError(
            "stage2_policy_closed_loop_pair_report must be absolute"
        )
    return str(path)


def _validated_domain_masses(
    raw: object,
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(raw, Mapping) or not raw:
        raise Stage2PolicyClosedLoopError(f"{label} must be a non-empty mapping")
    masses: dict[str, float] = {}
    for domain, value in raw.items():
        name = str(domain).strip()
        mass = float(value)
        if not name or not math.isfinite(mass) or mass <= 0.0:
            raise Stage2PolicyClosedLoopError(f"invalid {label} entry: {domain!r}={value!r}")
        if name in masses:
            raise Stage2PolicyClosedLoopError(f"duplicate {label} domain: {name}")
        masses[name] = mass
    if not math.isclose(sum(masses.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise Stage2PolicyClosedLoopError(f"{label} must sum to one")
    return masses


def resolve_stage2_policy_pair_domain_masses(
    train_cfg: Mapping[str, object],
    *,
    default_masses: Mapping[str, float],
    hostname: str | None = None,
) -> dict[str, float]:
    """Return the local sampler mass whose equal-host mean is the global mass."""

    by_host = train_cfg.get("stage2_policy_closed_loop_pair_domain_masses_by_host")
    if by_host is None:
        raw = train_cfg.get(
            "stage2_policy_closed_loop_pair_domain_masses",
            default_masses,
        )
        return _validated_domain_masses(raw, label="closed-loop pair domain masses")
    if not isinstance(by_host, Mapping) or not by_host:
        raise Stage2PolicyClosedLoopError(
            "stage2_policy_closed_loop_pair_domain_masses_by_host must be non-empty"
        )
    current_host = str(hostname or socket.gethostname()).strip()
    if current_host not in by_host:
        raise Stage2PolicyClosedLoopError(
            f"no closed-loop pair domain masses are bound for hostname={current_host!r}"
        )
    return _validated_domain_masses(
        by_host[current_host],
        label=f"closed-loop pair domain masses for hostname={current_host!r}",
    )


def validate_stage2_policy_closed_loop_config(
    *,
    train_cfg: Mapping[str, object],
    model_cfg: Mapping[str, object],
    data_cfg: Mapping[str, object],
) -> None:
    """Fail before launch when training and serving use different inputs."""

    if not bool(train_cfg.get("stage2_policy_closed_loop", False)):
        return
    errors: list[str] = []
    horizon = int(model_cfg.get("policy_horizon", 0) or 0)
    data_horizon = int(data_cfg.get("k", 0) or 0)
    if horizon != 8 or data_horizon != 8:
        errors.append(f"closed-loop Stage2 requires K8, got policy={horizon} data={data_horizon}")
    if str(model_cfg.get("policy_head_type", "")).strip().lower() != "oft":
        errors.append("closed-loop Stage2 requires policy_head_type=oft")
    world_update = bool(train_cfg.get("stage2_policy_closed_loop_world_update", True))
    policy_source = str(model_cfg.get("policy_context_source", "")).strip().lower()
    expected_policy_source = "core_pred" if world_update else "core_pred_detach"
    if policy_source != expected_policy_source:
        errors.append(
            "closed-loop Stage2 requires "
            f"policy_context_source={expected_policy_source} when "
            f"stage2_policy_closed_loop_world_update={str(world_update).lower()}"
        )
    if str(model_cfg.get("policy_core_action_cond", "")).strip().lower() not in {
        "none",
        "no_action",
    }:
        errors.append("closed-loop Stage2 requires policy_core_action_cond=none")
    if not bool(model_cfg.get("policy_use_context_rgb", False)):
        errors.append("closed-loop Stage2 requires policy_use_context_rgb=true")
    if bool(model_cfg.get("policy_use_progress", False)):
        errors.append("closed-loop Stage2 forbids future-derived policy progress")
    if int(model_cfg.get("policy_action_history_len", 0) or 0) < 1:
        errors.append("closed-loop Stage2 requires at least one executed action-history token")
    if bool(train_cfg.get("wan_policy_action_cond_detach_for_renderer", True)):
        errors.append("closed-loop Stage2 requires differentiable Wan policy conditioning")
    if not bool(train_cfg.get("enable_wan_ti2v_loss", False)):
        errors.append("closed-loop Stage2 requires the real Wan TI2V path")
    if int(train_cfg.get("stage2_policy_closed_loop_every", 0) or 0) <= 0:
        errors.append("stage2_policy_closed_loop_every must be positive")
    reports = train_cfg.get("stage2_policy_closed_loop_pair_reports")
    legacy_report = train_cfg.get("stage2_policy_closed_loop_pair_report")
    if reports is not None and legacy_report not in (None, ""):
        errors.append("closed-loop Stage2 pair report config is ambiguous")
    elif reports is not None:
        if not isinstance(reports, Mapping) or not reports:
            errors.append("stage2_policy_closed_loop_pair_reports must be a non-empty mapping")
        else:
            for host, raw_path in reports.items():
                if not str(host).strip():
                    errors.append("closed-loop Stage2 pair report hostname must be non-empty")
                if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
                    errors.append(f"pair report for hostname={host!r} must be absolute")
    elif not isinstance(legacy_report, str) or not Path(legacy_report).is_absolute():
        errors.append("closed-loop Stage2 requires an absolute immutable pair report")
    masses_by_host = train_cfg.get("stage2_policy_closed_loop_pair_domain_masses_by_host")
    if reports is not None:
        if not bool(train_cfg.get("stage2_policy_closed_loop_equal_ranks_per_host", False)):
            errors.append("multi-host closed-loop Stage2 requires equal ranks per host")
        if not isinstance(masses_by_host, Mapping) or set(masses_by_host) != set(reports):
            errors.append("pair report and domain-mass hostname sets must match")
        else:
            try:
                global_masses = _validated_domain_masses(
                    train_cfg.get("stage2_policy_closed_loop_pair_domain_masses"),
                    label="global closed-loop pair domain masses",
                )
                local_masses = {
                    str(host): _validated_domain_masses(
                        masses,
                        label=f"closed-loop pair domain masses for hostname={host!r}",
                    )
                    for host, masses in masses_by_host.items()
                }
                domains = set(global_masses)
                domains.update(*(set(masses) for masses in local_masses.values()))
                for domain in sorted(domains):
                    actual = sum(masses.get(domain, 0.0) for masses in local_masses.values()) / len(local_masses)
                    expected = global_masses.get(domain, 0.0)
                    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-9):
                        errors.append(
                            f"equal-host pair mass mismatch for {domain}: {actual} != {expected}"
                        )
            except (Stage2PolicyClosedLoopError, TypeError, ValueError) as exc:
                errors.append(str(exc))
    policy_weight = float(train_cfg.get("stage2_policy_closed_loop_policy_weight", 0.0) or 0.0)
    if policy_weight <= 0.0:
        errors.append("stage2_policy_closed_loop_policy_weight must be positive")
    native_weight = float(train_cfg.get("stage2_policy_closed_loop_native_weight", 0.0) or 0.0)
    wan_weight = float(train_cfg.get("stage2_policy_closed_loop_wan_weight", 0.0) or 0.0)
    if world_update:
        if native_weight <= 0.0:
            errors.append("stage2_policy_closed_loop_native_weight must be positive")
        if wan_weight <= 0.0:
            errors.append("stage2_policy_closed_loop_wan_weight must be positive")
    else:
        if native_weight != 0.0 or wan_weight != 0.0:
            errors.append(
                "policy-only closed-loop Stage2 requires native_weight=0 and wan_weight=0"
            )
        prefixes = list(train_cfg.get("trainable_prefixes") or [])
        if prefixes != ["action_policy."]:
            errors.append(
                "policy-only closed-loop Stage2 requires trainable_prefixes exactly action_policy."
            )
        modes = tuple(train_cfg.get("stage2_policy_closed_loop_counterfactual_modes") or ())
        if not modes or any(mode not in {"reverse_pose", "toggle_grip"} for mode in modes):
            errors.append(
                "policy-only closed-loop Stage2 requires real reverse_pose/toggle_grip counterfactual modes"
            )
        if float(train_cfg.get("stage2_policy_closed_loop_counterfactual_weight", 0.0) or 0.0) <= 0.0:
            errors.append("stage2_policy_closed_loop_counterfactual_weight must be positive")
        counterfactual_objective = str(
            train_cfg.get("stage2_policy_closed_loop_counterfactual_objective", "ranking")
        ).strip().lower()
        if counterfactual_objective == "ranking":
            if float(train_cfg.get("stage2_policy_closed_loop_counterfactual_margin", 0.0) or 0.0) <= 0.0:
                errors.append("stage2_policy_closed_loop_counterfactual_margin must be positive")
        elif counterfactual_objective == "output_separation":
            margins = train_cfg.get(
                "stage2_policy_closed_loop_counterfactual_sensitivity_margins"
            )
            if not isinstance(margins, dict):
                errors.append(
                    "output_separation requires stage2_policy_closed_loop_counterfactual_sensitivity_margins"
                )
            else:
                missing_modes = sorted(set(modes).difference(margins))
                extra_modes = sorted(set(margins).difference(modes))
                if missing_modes or extra_modes:
                    errors.append(
                        "counterfactual sensitivity margins must exactly match configured modes; "
                        f"missing={missing_modes} extra={extra_modes}"
                    )
                for mode in modes:
                    try:
                        sensitivity_margin = float(margins[mode])
                    except (KeyError, TypeError, ValueError):
                        sensitivity_margin = float("nan")
                    if not math.isfinite(sensitivity_margin) or sensitivity_margin <= 0.0:
                        errors.append(
                            f"counterfactual sensitivity margin for {mode} must be finite and positive"
                        )
                    if mode == "toggle_grip" and sensitivity_margin > 1.0:
                        errors.append("toggle_grip sensitivity margin must be <=1")
        else:
            errors.append(
                "stage2_policy_closed_loop_counterfactual_objective must be ranking or output_separation"
            )
        reference_grip = bool(
            train_cfg.get("stage2_policy_closed_loop_reference_grip_enabled", False)
        )
        if reference_grip:
            reference_contract = str(
                train_cfg.get("stage2_policy_closed_loop_reference_grip_contract", "")
            ).strip()
            if reference_contract not in {
                "signed_margin_v1",
                "full_oft_v1",
                "full_serving_v1",
                "performance_floor",
            }:
                errors.append(
                    "reference retention requires contract=signed_margin_v1, "
                    "full_oft_v1, full_serving_v1, or performance_floor"
                )
            reference_checkpoint = str(
                train_cfg.get("stage2_policy_closed_loop_reference_checkpoint", "")
            ).strip()
            init_checkpoint = str(train_cfg.get("init_from_stage0_wan_ckpt", "")).strip()
            if not reference_checkpoint:
                errors.append(
                    "reference grip retention requires stage2_policy_closed_loop_reference_checkpoint"
                )
            if reference_checkpoint == init_checkpoint:
                errors.append(
                    "reference grip checkpoint must be a frozen teacher independent of the Stage2 init"
                )
            if not bool(
                train_cfg.get(
                    "stage2_policy_closed_loop_reference_grip_cadence_normalize",
                    False,
                )
            ):
                errors.append(
                    "reference grip retention requires cadence normalization"
                )
            reference_weight = float(
                train_cfg.get("stage2_policy_closed_loop_reference_grip_weight", 0.0)
                or 0.0
            )
            if not math.isfinite(reference_weight) or reference_weight <= 0.0:
                errors.append("reference grip weight must be finite and positive")
            if reference_contract in {
                "full_oft_v1",
                "full_serving_v1",
                "performance_floor",
            }:
                observed_weight = float(
                    train_cfg.get(
                        "stage2_policy_closed_loop_reference_observed_weight",
                        0.0,
                    )
                    or 0.0
                )
                if not math.isfinite(observed_weight) or observed_weight <= 0.0:
                    errors.append(
                        "full OFT reference retention requires a finite positive observed weight"
                    )
            if reference_contract in {"full_serving_v1", "performance_floor"}:
                context_source = str(
                    model_cfg.get("policy_context_source", "")
                ).strip().lower()
                if context_source not in {
                    "core_pred",
                    "core_pred_detach",
                    "serving",
                    "serving_detach",
                }:
                    errors.append(
                        "full serving reference requires a core_pred policy context"
                    )
                core_action_cond = str(
                    model_cfg.get("policy_core_action_cond", "")
                ).strip().lower()
                if core_action_cond not in {
                    "none",
                    "no_action",
                    "off",
                    "disabled",
                }:
                    errors.append(
                        "full serving reference requires no-action core prediction"
                    )
            reference_tolerance = float(
                train_cfg.get("stage2_policy_closed_loop_reference_grip_tolerance", -1.0)
            )
            if (
                not math.isfinite(reference_tolerance)
                or reference_tolerance < 0.0
                or reference_tolerance >= 0.5
            ):
                errors.append("reference grip tolerance must be finite and in [0,0.5)")
            reference_sensitivity_ratio = float(
                train_cfg.get(
                    "stage2_policy_closed_loop_reference_sensitivity_ratio",
                    0.0,
                )
                or 0.0
            )
            if (
                not math.isfinite(reference_sensitivity_ratio)
                or not 0.0 < reference_sensitivity_ratio <= 1.0
            ):
                errors.append("reference sensitivity ratio must be finite and in (0,1]")
            if reference_contract == "performance_floor":
                reference_performance_ratio = float(
                    train_cfg.get(
                        "stage2_policy_closed_loop_reference_performance_ratio",
                        0.0,
                    )
                    or 0.0
                )
                if (
                    not math.isfinite(reference_performance_ratio)
                    or not 1.0 <= reference_performance_ratio <= 1.5
                ):
                    errors.append(
                        "performance-floor reference ratio must be finite and in [1,1.5]"
                    )
            if bool(model_cfg.get("policy_action_add_trunk", False)):
                errors.append("reference grip retention requires policy_action_add_trunk=false")
    if float(train_cfg.get("direct_policy_weight", 0.0) or 0.0) <= 0.0:
        errors.append("closed-loop Stage2 requires direct_policy_weight>0")
    for key in (
        "source_feed_weight",
        "wan_source_feed_weight",
        "wan_source_loss_weight",
        "wan_source_l1_weight",
        "wan_source_cf_weight",
        "wan_source_action_cf_weight",
    ):
        if float(train_cfg.get(key, 0.0) or 0.0) != 0.0:
            errors.append(f"closed-loop Stage2 requires {key}=0")
    if errors:
        raise Stage2PolicyClosedLoopError("; ".join(errors))


def executed_policy_action(
    policy_action_cond: torch.Tensor | None,
    *,
    horizon: int,
    grip_threshold: float = 0.5,
) -> torch.Tensor:
    """Return the exact K8 action sent to the native core and Wan rollout.

    Pose remains continuous in the canonical normalized action space. Gripper is
    thresholded because deployment executes an absolute binary gripper state.
    The returned tensor is detached: rollout sampling is an environment step,
    not a differentiable surrogate.
    """

    if policy_action_cond is None:
        raise Stage2PolicyClosedLoopError("OFT did not emit policy_action_cond")
    if policy_action_cond.ndim != 3 or int(policy_action_cond.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError(
            f"policy_action_cond must have shape [B,K,7], got {tuple(policy_action_cond.shape)}"
        )
    if int(policy_action_cond.shape[1]) != int(horizon):
        raise Stage2PolicyClosedLoopError(
            f"policy action horizon must be exactly {horizon}, got {policy_action_cond.shape[1]}"
        )
    if not bool(torch.isfinite(policy_action_cond).all().detach().cpu()):
        raise Stage2PolicyClosedLoopError("policy_action_cond contains non-finite values")
    if not 0.0 < float(grip_threshold) < 1.0:
        raise Stage2PolicyClosedLoopError("grip_threshold must be in (0,1)")
    grip = policy_action_cond[..., 6:7]
    if bool(((grip < 0.0) | (grip > 1.0)).any().detach().cpu()):
        raise Stage2PolicyClosedLoopError("policy gripper channel must be a probability in [0,1]")
    executed = policy_action_cond.detach().clone()
    executed[..., 6:7] = (grip.detach() >= float(grip_threshold)).to(executed.dtype)
    return executed


def counterfactual_policy_action(
    executed_action: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Build an exact negative action for a second real core+Wan rollout."""

    if executed_action.ndim != 3 or int(executed_action.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError(
            f"executed action must have shape [B,K,7], got {tuple(executed_action.shape)}"
        )
    negative = executed_action.detach().clone()
    if mode == "reverse_pose":
        negative[..., :6] = -negative[..., :6]
    elif mode == "toggle_grip":
        negative[..., 6] = 1.0 - negative[..., 6]
    else:
        raise Stage2PolicyClosedLoopError(f"unsupported counterfactual mode: {mode!r}")
    return negative


def policy_counterfactual_ranking_loss(
    true_prediction: torch.Tensor,
    negative_prediction: torch.Tensor,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    *,
    mode: str,
    margin: float,
) -> dict[str, torch.Tensor]:
    """Require the true rollout policy to beat a real counterfactual rollout.

    The negative branch is never assigned a fabricated action target.  It is
    only ranked against the expert continuation attached to the true adjacent
    pair, which prevents the policy from ignoring action-conditioned changes
    in the native-core and Wan-generated observation.
    """

    if true_prediction.shape != negative_prediction.shape:
        raise Stage2PolicyClosedLoopError("true and counterfactual policy shapes must match")
    if true_prediction.ndim != 3 or int(true_prediction.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError("policy predictions must have shape [B,K,7]")
    horizon = int(true_prediction.shape[1])
    if action_tgt.shape[:2] != (true_prediction.shape[0], horizon):
        raise Stage2PolicyClosedLoopError("action_tgt does not match policy batch/horizon")
    if action_tgt_norm.shape[:2] != (true_prediction.shape[0], horizon):
        raise Stage2PolicyClosedLoopError("action_tgt_norm does not match policy batch/horizon")
    if not math.isfinite(float(margin)) or float(margin) <= 0.0:
        raise Stage2PolicyClosedLoopError("counterfactual ranking margin must be finite and positive")

    if mode == "reverse_pose":
        target = action_tgt_norm[..., :6].float()
        true_error = (true_prediction[..., :6].float() - target).abs().mean(dim=(1, 2))
        negative_error = (negative_prediction[..., :6].float() - target).abs().mean(dim=(1, 2))
    elif mode == "toggle_grip":
        target = (action_tgt[..., 6].float() > 0.5).float()
        true_prob = true_prediction[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
        negative_prob = negative_prediction[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
        true_error = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(true_prob), target, reduction="none"
        ).mean(dim=1)
        negative_error = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(negative_prob), target, reduction="none"
        ).mean(dim=1)
    else:
        raise Stage2PolicyClosedLoopError(f"unsupported counterfactual mode: {mode!r}")
    gap = negative_error - true_error
    return {
        "loss": torch.relu(true_error.new_tensor(float(margin)) - gap).mean(),
        "true_error": true_error.mean(),
        "negative_error": negative_error.mean(),
        "gap": gap.mean(),
    }


def policy_counterfactual_sensitivity_loss(
    true_prediction: torch.Tensor,
    negative_prediction: torch.Tensor,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    *,
    mode: str,
    margin: float,
) -> dict[str, torch.Tensor]:
    """Preserve action sensitivity without assigning a label to a counterfactual state.

    The true rollout remains supervised by the normal direct-policy loss.  This
    term only requires predictions from two real core+Wan rollouts to remain
    separated in the channels changed by the counterfactual action.  It never
    maximizes an error against the true branch's expert continuation, because
    that continuation is not a valid label for the counterfactual state.
    """

    if true_prediction.shape != negative_prediction.shape:
        raise Stage2PolicyClosedLoopError("true and counterfactual policy shapes must match")
    if true_prediction.ndim != 3 or int(true_prediction.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError("policy predictions must have shape [B,K,7]")
    horizon = int(true_prediction.shape[1])
    if action_tgt.shape[:2] != (true_prediction.shape[0], horizon):
        raise Stage2PolicyClosedLoopError("action_tgt does not match policy batch/horizon")
    if action_tgt_norm.shape[:2] != (true_prediction.shape[0], horizon):
        raise Stage2PolicyClosedLoopError("action_tgt_norm does not match policy batch/horizon")
    if not math.isfinite(float(margin)) or float(margin) <= 0.0:
        raise Stage2PolicyClosedLoopError("counterfactual sensitivity margin must be finite and positive")

    if mode == "reverse_pose":
        target = action_tgt_norm[..., :6].float()
        true_channels = true_prediction[..., :6].float()
        negative_channels = negative_prediction[..., :6].float()
        true_error = (true_channels - target).abs().mean(dim=(1, 2))
        negative_error = (negative_channels - target).abs().mean(dim=(1, 2))
        sensitivity = (true_channels - negative_channels).abs().mean(dim=(1, 2))
    elif mode == "toggle_grip":
        if float(margin) > 1.0:
            raise Stage2PolicyClosedLoopError("toggle_grip sensitivity margin must be <=1")
        target = (action_tgt[..., 6].float() > 0.5).float()
        true_channels = true_prediction[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
        negative_channels = negative_prediction[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
        true_error = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(true_channels), target, reduction="none"
        ).mean(dim=1)
        negative_error = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(negative_channels), target, reduction="none"
        ).mean(dim=1)
        sensitivity = (true_channels - negative_channels).abs().mean(dim=1)
    else:
        raise Stage2PolicyClosedLoopError(f"unsupported counterfactual mode: {mode!r}")

    gap = negative_error - true_error
    separation_loss = torch.relu(sensitivity.new_tensor(float(margin)) - sensitivity)
    return {
        "loss": separation_loss.mean(),
        "true_error": true_error.mean(),
        "negative_error": negative_error.mean(),
        "gap": gap.mean(),
        "sensitivity": sensitivity.mean(),
    }


def policy_reference_observed_retention_loss(
    student: torch.Tensor,
    reference: torch.Tensor,
    *,
    tolerance: float,
    decision_ratio: float,
) -> dict[str, torch.Tensor]:
    """Preserve the frozen OFT serving policy on real observed contexts.

    The direct supervised objective remains active and can improve inside the
    trust region. This term prevents closed-loop refinement from forgetting
    the teacher's observed-state pose and absolute-gripper policy, which is
    exactly the oracle path exercised by the held-out gate and deployment.
    """

    if student.shape != reference.shape:
        raise Stage2PolicyClosedLoopError(
            "student/reference observed policy shapes must match"
        )
    if student.ndim != 3 or int(student.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError("policy predictions must have shape [B,K,7]")
    if not math.isfinite(float(tolerance)) or not 0.0 <= float(tolerance) < 0.5:
        raise Stage2PolicyClosedLoopError("reference tolerance must be in [0,0.5)")
    if not math.isfinite(float(decision_ratio)) or not 0.0 < float(decision_ratio) <= 1.0:
        raise Stage2PolicyClosedLoopError("reference decision ratio must be in (0,1]")

    student_f = student.float()
    reference_f = reference.detach().float()
    pose_deviation = (student_f[..., :6] - reference_f[..., :6]).abs()
    pose_trust_region_loss = torch.relu(pose_deviation - float(tolerance)).mean()

    student_grip = student_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    reference_grip = reference_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    grip_deviation = (student_grip - reference_grip).abs()
    grip_trust_region_loss = torch.relu(
        grip_deviation - float(tolerance)
    ).mean()
    reference_positive = reference_grip >= 0.5
    reference_sign = reference_positive.to(student_grip.dtype).mul(2.0).sub(1.0)
    reference_margin = (reference_grip - 0.5).abs()
    grip_decision_loss = torch.relu(
        float(decision_ratio) * reference_margin
        - reference_sign * (student_grip - 0.5)
    ).mean()
    hard_agreement = ((student_grip >= 0.5) == reference_positive).float().mean()
    return {
        "loss": pose_trust_region_loss
        + grip_trust_region_loss
        + grip_decision_loss,
        "pose_trust_region_loss": pose_trust_region_loss,
        "grip_trust_region_loss": grip_trust_region_loss,
        "grip_decision_loss": grip_decision_loss,
        "pose_deviation": pose_deviation.mean(),
        "grip_deviation": grip_deviation.mean(),
        "hard_agreement": hard_agreement,
    }


def policy_reference_observed_performance_floor_loss(
    student: torch.Tensor,
    reference: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    action_tgt: torch.Tensor,
    action_prev_grip: torch.Tensor | None,
    *,
    tolerance: float,
    decision_ratio: float,
    performance_ratio: float = 1.02,
) -> dict[str, torch.Tensor]:
    """Enforce the fixed gate's global oracle and six gripper partition floors."""

    if student.shape != reference.shape:
        raise Stage2PolicyClosedLoopError(
            "student/reference observed policy shapes must match"
        )
    if student.ndim != 3 or int(student.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError("policy predictions must have shape [B,K,7]")
    if action_tgt_norm.shape[:2] != student.shape[:2] or int(action_tgt_norm.shape[-1]) < 6:
        raise Stage2PolicyClosedLoopError("normalized action target does not match policy output")
    if action_tgt.shape[:2] != student.shape[:2] or int(action_tgt.shape[-1]) < 7:
        raise Stage2PolicyClosedLoopError("raw action target does not match policy output")
    if action_prev_grip is None:
        raise Stage2PolicyClosedLoopError(
            "gate-aligned performance floor requires action_prev_grip"
        )
    if not math.isfinite(float(tolerance)) or not 0.0 <= float(tolerance) < 0.5:
        raise Stage2PolicyClosedLoopError("reference tolerance must be in [0,0.5)")
    if not math.isfinite(float(decision_ratio)) or not 0.0 < float(decision_ratio) <= 1.0:
        raise Stage2PolicyClosedLoopError("reference decision ratio must be in (0,1]")
    if not math.isfinite(float(performance_ratio)) or not 1.0 <= float(performance_ratio) <= 1.5:
        raise Stage2PolicyClosedLoopError("reference performance ratio must be in [1,1.5]")

    student_f = student.float()
    reference_f = reference.detach().float()
    target_pose = action_tgt_norm[..., :6].to(device=student.device).float()
    target_grip = (action_tgt[..., 6].to(device=student.device).float() > 0.5).float()
    student_grip = student_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    reference_grip = reference_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    student_pose_error = (student_f[..., :6] - target_pose).abs()
    reference_pose_error = (reference_f[..., :6] - target_pose).abs()
    student_grip_abs = (student_grip - target_grip).abs()
    reference_grip_abs = (reference_grip - target_grip).abs()
    student_action_error = torch.cat((student_pose_error, student_grip_abs[..., None]), dim=-1)
    reference_action_error = torch.cat((reference_pose_error, reference_grip_abs[..., None]), dim=-1)
    student_grip_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.logit(student_grip), target_grip, reduction="none"
    )
    reference_grip_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.logit(reference_grip), target_grip, reduction="none"
    )
    target_positive = target_grip >= 0.5
    target_sign = target_positive.to(student_grip.dtype).mul(2.0).sub(1.0)
    student_target_margin = target_sign * (student_grip - 0.5)
    reference_target_margin = target_sign * (reference_grip - 0.5)
    partitions = _gate_grip_partitions(
        target_grip,
        action_prev_grip.to(device=student.device),
    )

    action_regressions = torch.relu(
        student_action_error
        - _teacher_relative_ceiling(
            reference_action_error,
            performance_ratio=performance_ratio,
            tolerance=tolerance,
        )
    )
    pose_regressions = torch.relu(
        student_pose_error
        - _teacher_relative_ceiling(
            reference_pose_error,
            performance_ratio=performance_ratio,
            tolerance=tolerance,
        )
    )
    grip_abs_regressions = torch.relu(
        student_grip_abs
        - _teacher_relative_ceiling(
            reference_grip_abs,
            performance_ratio=performance_ratio,
            tolerance=tolerance,
        )
    )
    grip_bce_regressions = torch.relu(
        student_grip_bce
        - _teacher_relative_ceiling(
            reference_grip_bce,
            performance_ratio=performance_ratio,
            tolerance=tolerance,
        )
    )
    regression_values = [
        action_regressions,
        pose_regressions,
        grip_abs_regressions,
    ]
    regression_masks = [
        torch.ones_like(action_regressions, dtype=torch.bool),
        torch.ones_like(pose_regressions, dtype=torch.bool),
        torch.ones_like(grip_abs_regressions, dtype=torch.bool),
    ]
    for name in _GATE_GRIP_PARTITIONS:
        regression_values.append(grip_bce_regressions)
        regression_masks.append(partitions[name])
    regression_sums = []
    regression_counts = []
    for values, mask in zip(regression_values, regression_masks, strict=True):
        active = mask & (values.detach() > 0.0)
        active_f = active.to(values.dtype)
        regression_sums.append((values * active_f).sum())
        regression_counts.append(active_f.sum())
    for name in _GATE_GRIP_PARTITIONS:
        regression_sums.append(student_grip.new_zeros(()))
        regression_counts.append(partitions[name].to(student_grip.dtype).sum())
    (
        regression_losses,
        _unused_reference_regression_means,
        combined_regression_counts,
    ) = _distributed_pair_sum_count_means(
        torch.stack(regression_sums),
        torch.zeros_like(torch.stack(regression_sums)),
        torch.stack(regression_counts),
    )
    regression_violation_counts = combined_regression_counts[:9]
    global_counts = combined_regression_counts[9:15]
    action_performance_floor_loss = regression_losses[0]
    pose_performance_floor_loss = regression_losses[1]
    grip_abs_performance_floor_loss = regression_losses[2]
    partition_bce_losses = regression_losses[3:9]
    reference_correct = (reference_grip >= 0.5) == target_positive
    decision_sums = []
    decision_counts = []
    for name in _GATE_GRIP_PARTITIONS:
        teacher_correct_mask = partitions[name] & reference_correct
        decision_hinge = torch.relu(
            float(decision_ratio) * reference_target_margin - student_target_margin
        )
        active = teacher_correct_mask & (decision_hinge.detach() > 0.0)
        active_f = active.to(decision_hinge.dtype)
        decision_sums.append((decision_hinge * active_f).sum())
        decision_counts.append(active_f.sum())
    (
        partition_margin_losses,
        _unused_reference_decision_means,
        decision_violation_counts,
    ) = _distributed_pair_sum_count_means(
        torch.stack(decision_sums),
        torch.zeros_like(torch.stack(decision_sums)),
        torch.stack(decision_counts),
    )
    grip_performance_floor_loss = _supported_active_mean(
        partition_bce_losses,
        regression_violation_counts[3:9],
    )
    grip_decision_floor_loss = _supported_active_mean(
        partition_margin_losses,
        decision_violation_counts,
    )

    pose_deviation = (student_f[..., :6] - reference_f[..., :6]).abs().mean()
    grip_deviation = (student_grip - reference_grip).abs().mean()
    reference_positive = reference_grip >= 0.5
    reference_correct = reference_positive == target_positive
    reference_correct_f = reference_correct.to(student_grip.dtype)
    hard_agreement = ((student_grip >= 0.5) == reference_positive).float().mean()
    result = {
        "loss": action_performance_floor_loss
        + grip_performance_floor_loss
        + grip_decision_floor_loss,
        "action_performance_floor_loss": action_performance_floor_loss,
        "pose_performance_floor_loss": pose_performance_floor_loss,
        "grip_abs_performance_floor_loss": grip_abs_performance_floor_loss,
        "grip_performance_floor_loss": grip_performance_floor_loss,
        "grip_decision_floor_loss": grip_decision_floor_loss,
        "pose_deviation": pose_deviation,
        "grip_deviation": grip_deviation,
        "hard_agreement": hard_agreement,
        "reference_correct_fraction": reference_correct_f.mean(),
        "pose_violation_fraction": (student_pose_error.detach() > reference_pose_error).float().mean(),
        "grip_violation_fraction": (student_grip_bce.detach() > reference_grip_bce).float().mean(),
        "decision_violation_fraction": (
            student_target_margin.detach()
            < float(decision_ratio) * reference_target_margin
        ).float().mean(),
    }
    for index, name in enumerate(_GATE_GRIP_PARTITIONS):
        result[f"grip_partition_{name}_count"] = global_counts[index]
        result[f"grip_partition_{name}_bce_floor_loss"] = partition_bce_losses[index]
        result[f"grip_partition_{name}_bce_violation_count"] = (
            regression_violation_counts[3 + index]
        )
        result[f"grip_partition_{name}_decision_floor_loss"] = partition_margin_losses[index]
        result[f"grip_partition_{name}_decision_violation_count"] = (
            decision_violation_counts[index]
        )
    return result


def policy_reference_grip_retention_loss(
    student_true: torch.Tensor,
    student_negative: torch.Tensor,
    reference_true: torch.Tensor,
    reference_negative: torch.Tensor,
    *,
    tolerance: float,
    sensitivity_ratio: float,
    counterfactual_mode: str | None = None,
) -> dict[str, torch.Tensor]:
    """Keep generated-context grip behavior inside a frozen-policy trust region.

    All four tensors are policy outputs for the same real core+Wan contexts.
    The reference policy is the exact fresh-init checkpoint and is never
    trained.  This loss preserves its grip decisions and counterfactual
    sensitivity without assigning an expert label to the negative context.
    """

    expected = student_true.shape
    if any(
        value.shape != expected
        for value in (student_negative, reference_true, reference_negative)
    ):
        raise Stage2PolicyClosedLoopError(
            "student/reference true/counterfactual policy shapes must match"
        )
    if student_true.ndim != 3 or int(student_true.shape[-1]) != 7:
        raise Stage2PolicyClosedLoopError("policy predictions must have shape [B,K,7]")
    if not math.isfinite(float(tolerance)) or not 0.0 <= float(tolerance) < 0.5:
        raise Stage2PolicyClosedLoopError("reference grip tolerance must be in [0,0.5)")
    if (
        not math.isfinite(float(sensitivity_ratio))
        or not 0.0 < float(sensitivity_ratio) <= 1.0
    ):
        raise Stage2PolicyClosedLoopError("reference sensitivity ratio must be in (0,1]")

    student_true_grip = student_true[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
    student_negative_grip = student_negative[..., 6].float().clamp(1.0e-5, 1.0 - 1.0e-5)
    reference_true_grip = reference_true[..., 6].detach().float().clamp(1.0e-5, 1.0 - 1.0e-5)
    reference_negative_grip = (
        reference_negative[..., 6].detach().float().clamp(1.0e-5, 1.0 - 1.0e-5)
    )

    true_deviation = (student_true_grip - reference_true_grip).abs()
    negative_deviation = (student_negative_grip - reference_negative_grip).abs()
    trust_region = torch.cat((true_deviation, negative_deviation), dim=1)
    trust_region_loss = torch.relu(trust_region - float(tolerance)).mean()

    reference_true_positive = reference_true_grip >= 0.5
    reference_negative_positive = reference_negative_grip >= 0.5
    true_sign = reference_true_positive.to(student_true_grip.dtype).mul(2.0).sub(1.0)
    negative_sign = reference_negative_positive.to(student_negative_grip.dtype).mul(2.0).sub(1.0)
    true_reference_margin = (reference_true_grip - 0.5).abs()
    negative_reference_margin = (reference_negative_grip - 0.5).abs()
    decision_loss = torch.cat(
        (
            torch.relu(
                float(sensitivity_ratio) * true_reference_margin
                - true_sign * (student_true_grip - 0.5)
            ),
            torch.relu(
                float(sensitivity_ratio) * negative_reference_margin
                - negative_sign * (student_negative_grip - 0.5)
            ),
        ),
        dim=1,
    ).mean()

    student_delta = student_true_grip - student_negative_grip
    reference_delta = reference_true_grip - reference_negative_grip
    student_sensitivity = student_delta.abs()
    reference_sensitivity = reference_delta.abs()
    sensitivity_floor = float(sensitivity_ratio) * reference_sensitivity
    reference_direction = torch.sign(reference_delta)
    sensitivity_loss = torch.relu(
        sensitivity_floor - reference_direction * student_delta
    ).mean()
    directional_support = reference_sensitivity > 1.0e-6
    signed_direction_agreement = torch.where(
        directional_support,
        reference_direction * student_delta > 0.0,
        torch.ones_like(directional_support),
    ).float().mean()
    hard_agreement = torch.cat(
        (
            (student_true_grip >= 0.5) == reference_true_positive,
            (student_negative_grip >= 0.5) == reference_negative_positive,
        ),
        dim=1,
    ).float().mean()
    pose_sensitivity_loss = sensitivity_loss.new_zeros(())
    pose_signed_direction_agreement = sensitivity_loss.new_ones(())
    pose_student_sensitivity = sensitivity_loss.new_zeros(())
    pose_reference_sensitivity = sensitivity_loss.new_zeros(())
    if counterfactual_mode not in {None, "reverse_pose", "toggle_grip"}:
        raise Stage2PolicyClosedLoopError(
            f"unsupported reference counterfactual mode: {counterfactual_mode!r}"
        )
    if counterfactual_mode == "reverse_pose":
        student_pose_delta = (
            student_true[..., :6].float() - student_negative[..., :6].float()
        )
        reference_pose_delta = (
            reference_true[..., :6].detach().float()
            - reference_negative[..., :6].detach().float()
        )
        pose_student_sensitivity = student_pose_delta.abs().mean()
        pose_reference_sensitivity = reference_pose_delta.abs().mean()
        pose_reference_direction = torch.sign(reference_pose_delta)
        pose_sensitivity_loss = torch.relu(
            float(sensitivity_ratio) * reference_pose_delta.abs()
            - pose_reference_direction * student_pose_delta
        ).mean()
        pose_directional_support = reference_pose_delta.abs() > 1.0e-6
        pose_signed_direction_agreement = torch.where(
            pose_directional_support,
            pose_reference_direction * student_pose_delta > 0.0,
            torch.ones_like(pose_directional_support),
        ).float().mean()

    return {
        "loss": trust_region_loss
        + decision_loss
        + sensitivity_loss
        + pose_sensitivity_loss,
        "trust_region_loss": trust_region_loss,
        "decision_loss": decision_loss,
        "sensitivity_loss": sensitivity_loss,
        "deviation": trust_region.mean(),
        "hard_agreement": hard_agreement,
        "signed_direction_agreement": signed_direction_agreement,
        "student_sensitivity": student_sensitivity.mean(),
        "reference_sensitivity": reference_sensitivity.mean(),
        "pose_sensitivity_loss": pose_sensitivity_loss,
        "pose_signed_direction_agreement": pose_signed_direction_agreement,
        "pose_student_sensitivity": pose_student_sensitivity,
        "pose_reference_sensitivity": pose_reference_sensitivity,
    }


def policy_reference_closed_loop_performance_floor_loss(
    student_true: torch.Tensor,
    student_negative: torch.Tensor,
    reference_true: torch.Tensor,
    reference_negative: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    action_tgt: torch.Tensor,
    action_prev_grip: torch.Tensor | None,
    *,
    tolerance: float,
    sensitivity_ratio: float,
    counterfactual_mode: str,
    performance_ratio: float = 1.02,
) -> dict[str, torch.Tensor]:
    """Use the teacher as a task-error floor and sensitivity-direction floor."""

    observed = policy_reference_observed_performance_floor_loss(
        student_true,
        reference_true,
        action_tgt_norm,
        action_tgt,
        action_prev_grip,
        tolerance=tolerance,
        decision_ratio=sensitivity_ratio,
        performance_ratio=performance_ratio,
    )
    expected = student_true.shape
    if any(
        value.shape != expected
        for value in (student_negative, reference_true, reference_negative)
    ):
        raise Stage2PolicyClosedLoopError(
            "student/reference true/counterfactual policy shapes must match"
        )
    if counterfactual_mode not in {"reverse_pose", "toggle_grip"}:
        raise Stage2PolicyClosedLoopError(
            f"unsupported reference counterfactual mode: {counterfactual_mode!r}"
        )

    student_true_f = student_true.float()
    student_negative_f = student_negative.float()
    reference_true_f = reference_true.detach().float()
    reference_negative_f = reference_negative.detach().float()
    student_true_grip = student_true_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    student_negative_grip = student_negative_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    reference_true_grip = reference_true_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    reference_negative_grip = reference_negative_f[..., 6].clamp(1.0e-5, 1.0 - 1.0e-5)
    student_delta = student_true_grip - student_negative_grip
    reference_delta = reference_true_grip - reference_negative_grip
    reference_direction = torch.sign(reference_delta)
    directional_support = reference_delta.abs() > 1.0e-6
    grip_direction_agreement = torch.where(
        directional_support,
        reference_direction * student_delta > 0.0,
        torch.ones_like(directional_support),
    ).float()

    student_action_delta = student_true_f - student_negative_f
    reference_action_delta = reference_true_f - reference_negative_f
    action_direction = torch.sign(reference_action_delta)
    action_directional_support = reference_action_delta.abs() > 1.0e-6
    action_direction_agreement = torch.where(
        action_directional_support,
        action_direction * student_action_delta > 0.0,
        torch.ones_like(action_directional_support),
    ).float()
    toggle_mask = torch.full_like(
        student_delta,
        float(counterfactual_mode == "toggle_grip"),
    )
    reverse_mask = torch.full_like(
        student_action_delta,
        float(counterfactual_mode == "reverse_pose"),
    )
    toggle_violations = (
        torch.relu(
            float(sensitivity_ratio) * reference_delta.abs()
            - reference_direction * student_delta
        )
        * toggle_mask
    )
    reverse_violations = (
        torch.relu(
            float(sensitivity_ratio) * reference_action_delta.abs()
            - action_direction * student_action_delta
        )
        * reverse_mask
    )
    toggle_active = (toggle_violations.detach() > 0.0).to(toggle_violations.dtype)
    reverse_active = (reverse_violations.detach() > 0.0).to(reverse_violations.dtype)
    student_sums = torch.stack(
        (
            (reference_direction * student_delta * toggle_mask).sum(),
            (student_delta.abs() * toggle_mask).sum(),
            (action_direction * student_action_delta * reverse_mask).sum(),
            (student_action_delta.abs() * reverse_mask).sum(),
            (grip_direction_agreement * toggle_mask).sum(),
            (action_direction_agreement * reverse_mask).sum(),
            (toggle_violations * toggle_active).sum(),
            (reverse_violations * reverse_active).sum(),
        )
    )
    reference_sums = torch.stack(
        (
            (reference_delta.abs() * toggle_mask).sum(),
            (reference_delta.abs() * toggle_mask).sum(),
            (reference_action_delta.abs() * reverse_mask).sum(),
            (reference_action_delta.abs() * reverse_mask).sum(),
            student_delta.new_zeros(()),
            student_delta.new_zeros(()),
            student_delta.new_zeros(()),
            student_delta.new_zeros(()),
        )
    )
    counts = torch.stack(
        (
            toggle_mask.sum(),
            toggle_mask.sum(),
            reverse_mask.sum(),
            reverse_mask.sum(),
            toggle_mask.sum(),
            reverse_mask.sum(),
            toggle_active.sum(),
            reverse_active.sum(),
        )
    )
    sensitivity_student_means, sensitivity_reference_means, sensitivity_counts = (
        _distributed_pair_sum_count_means(student_sums, reference_sums, counts)
    )
    sensitivity_loss = sensitivity_student_means[6]
    pose_sensitivity_loss = sensitivity_student_means[7]
    signed_direction_agreement = sensitivity_student_means[4]
    pose_signed_direction_agreement = sensitivity_student_means[5]
    pose_student_sensitivity = sensitivity_student_means[3]
    pose_reference_sensitivity = sensitivity_reference_means[3]

    hard_agreement = torch.cat(
        (
            (student_true_grip >= 0.5) == (reference_true_grip >= 0.5),
            (student_negative_grip >= 0.5) == (reference_negative_grip >= 0.5),
        ),
        dim=1,
    ).float().mean()
    deviation = torch.cat(
        (
            (student_true_grip - reference_true_grip).abs(),
            (student_negative_grip - reference_negative_grip).abs(),
        ),
        dim=1,
    ).mean()
    return {
        "loss": observed["loss"] + sensitivity_loss + pose_sensitivity_loss,
        "performance_floor_loss": observed["loss"],
        "pose_performance_floor_loss": observed["pose_performance_floor_loss"],
        "grip_performance_floor_loss": observed["grip_performance_floor_loss"],
        "grip_decision_floor_loss": observed["grip_decision_floor_loss"],
        "sensitivity_loss": sensitivity_loss,
        "deviation": deviation,
        "hard_agreement": hard_agreement,
        "signed_direction_agreement": signed_direction_agreement,
        "student_sensitivity": sensitivity_student_means[1],
        "reference_sensitivity": sensitivity_reference_means[1],
        "pose_sensitivity_loss": pose_sensitivity_loss,
        "pose_signed_direction_agreement": pose_signed_direction_agreement,
        "pose_student_sensitivity": pose_student_sensitivity,
        "pose_reference_sensitivity": pose_reference_sensitivity,
        "sensitivity_violation_fraction": (
            sensitivity_loss.detach() > 0.0
        ).float(),
        "pose_sensitivity_violation_fraction": (
            pose_sensitivity_loss.detach() > 0.0
        ).float(),
    }


def build_policy_closed_loop_context(
    observed_tokens_a: torch.Tensor,
    predicted_tokens_a: torch.Tensor,
    *,
    overlap_frames: int = 8,
) -> torch.Tensor:
    """Build the next K8 context exclusively from A and A's prediction.

    Adjacent-pair caches can prove that ``B[:8] == A[8:16]``, but reading the
    B tensor at all weakens the rollout contract and makes future leakage hard
    to audit.  This helper therefore retains the last eight *observed* A
    tokens and appends the eight tokens predicted under the executed OFT
    action.  B is reserved for supervision only.
    """

    if overlap_frames <= 0:
        raise Stage2PolicyClosedLoopError("overlap_frames must be positive")
    if observed_tokens_a.ndim < 3 or predicted_tokens_a.ndim != observed_tokens_a.ndim:
        raise Stage2PolicyClosedLoopError(
            "closed-loop token tensors must have matching rank >=3; "
            f"got A={tuple(observed_tokens_a.shape)} pred={tuple(predicted_tokens_a.shape)}"
        )
    if int(observed_tokens_a.shape[1]) != 2 * int(overlap_frames):
        raise Stage2PolicyClosedLoopError(
            f"A must contain {2 * overlap_frames} observed frames, "
            f"got {observed_tokens_a.shape[1]}"
        )
    if int(predicted_tokens_a.shape[1]) != int(overlap_frames):
        raise Stage2PolicyClosedLoopError(
            f"prediction must contain {overlap_frames} frames, "
            f"got {predicted_tokens_a.shape[1]}"
        )
    if (
        observed_tokens_a.shape[0] != predicted_tokens_a.shape[0]
        or observed_tokens_a.shape[2:] != predicted_tokens_a.shape[2:]
    ):
        raise Stage2PolicyClosedLoopError(
            "A observation and prediction batch/token shapes must match; "
            f"got A={tuple(observed_tokens_a.shape)} pred={tuple(predicted_tokens_a.shape)}"
        )
    return torch.cat(
        (
            observed_tokens_a[:, -overlap_frames:].detach(),
            predicted_tokens_a.detach(),
        ),
        dim=1,
    )


def observed_policy_kwargs(
    observed_targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return policy inputs that are observable at the decision boundary."""

    kwargs: dict[str, torch.Tensor] = {}
    for key in ("lowdim_state", "object_state", "plan_state", "action_history"):
        if key in observed_targets:
            kwargs[key] = observed_targets[key]
    # An explicitly cached progress_state is allowed, but progress_tgt is a
    # future label and is intentionally never promoted to an input here.
    if "progress_state" in observed_targets:
        kwargs["progress_state"] = observed_targets["progress_state"]
    return kwargs


def closed_loop_policy_kwargs(
    observed_targets_a: Mapping[str, torch.Tensor],
    executed_action_a: torch.Tensor,
    *,
    history_len: int,
    history_dim: int = 7,
) -> dict[str, torch.Tensor]:
    """Build B-policy inputs without reading B's future-derived state.

    Robot/object/plan/progress inputs are carried from A's last observed frame.
    The action history is replaced by the action actually used for the rollout.
    B targets are deliberately not accepted by this function.
    """

    if history_len <= 0:
        raise Stage2PolicyClosedLoopError("history_len must be positive")
    if history_dim != 7:
        raise Stage2PolicyClosedLoopError(
            f"canonical Stage2 closed loop requires history_dim=7, got {history_dim}"
        )
    if executed_action_a.ndim != 3 or int(executed_action_a.shape[-1]) != history_dim:
        raise Stage2PolicyClosedLoopError(
            f"executed action must have shape [B,K,{history_dim}], got {tuple(executed_action_a.shape)}"
        )
    if int(executed_action_a.shape[1]) < history_len:
        raise Stage2PolicyClosedLoopError(
            f"executed action horizon {executed_action_a.shape[1]} is shorter than history {history_len}"
        )

    kwargs = observed_policy_kwargs(observed_targets_a)
    kwargs["action_history"] = executed_action_a[:, -history_len:].detach()
    return kwargs
