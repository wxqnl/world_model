"""Fail-closed gradient ownership audit for unified WM3D Stage0.

The old V7 run could report a finite total loss while the executable action
owner received no useful supervision.  A release run therefore proves, on a
real optimizer step, that every required world/action/proprio branch owns a
finite non-zero gradient.  The implementation operates on local FSDP2 DTensor
shards and reduces only scalar statistics per owner; it never gathers a full
parameter or gradient.  Replicated DDP/HSDP dimensions are explicitly
de-duplicated so receipts describe the logical model rather than the launch
topology.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import torch
import torch.distributed as dist
from torch import nn

from wm3d.models.native_world_model import NativeWorldModel


GRADIENT_OWNERSHIP_SCHEMA = "wm3d_v8_gradient_ownership_v2"
_OPTIONAL_OWNERS = frozenset({"auxiliary_inputs"})


class GradientOwnershipError(RuntimeError):
    pass


def _required_owner_modules(model: NativeWorldModel) -> Mapping[str, tuple[nn.Module, ...]]:
    """Return disjoint, capability-level module owners required by Stage0."""

    policy_modules: list[nn.Module] = [
        model.history_action,
        model.action_blocks,
        model.policy_spatial_cross,
    ]
    if model.policy_spatial_task_modulation is not None:
        policy_modules.append(model.policy_spatial_task_modulation)
    if model.policy_calibration is not None:
        policy_modules.append(model.policy_calibration)
    owners: dict[str, tuple[nn.Module, ...]] = {
        "native_state_trunk": (
            model.view_fuser,
            model.state_blocks,
            model.token_output,
        ),
        "factual_dynamics": (model.factual_action, model.dynamics_blocks),
        "policy_action_trunk": tuple(policy_modules),
        "state_action_bridges": (model.bridges,),
        "current_state_proprio": (model.current_state,),
        "unified_action_head": (model.action_head,),
        "rgb_decoder": (model.rgb_head,),
        "geometry_decoder": (model.geometry_head,),
    }
    if model.appearance_dynamics is not None:
        owners["appearance_dynamics"] = (model.appearance_dynamics,)
    return owners


def _required_owner_parameters(
    model: NativeWorldModel,
) -> Mapping[str, tuple[nn.Parameter, ...]]:
    """Assign root parameters that are not owned by a child module."""

    return {
        "native_state_inputs": (
            model.state_space,
            model.future_queries,
            *tuple(model.state_time.parameters()),
            *tuple(model.task_state.parameters()),
            *tuple(model.state_input_norm.parameters()),
            *tuple(model.state_norm.parameters()),
        ),
        "policy_action_inputs": (
            model.policy_query_seed,
            *tuple(model.action_time.parameters()),
            *tuple(model.task_action.parameters()),
            *tuple(model.policy_spatial_norm.parameters()),
            *tuple(model.action_norm.parameters()),
        ),
        "auxiliary_inputs": (
            *tuple(model.aux_value.parameters()),
            *tuple(model.aux_type.parameters()),
        ),
    }


def required_gradient_owner_names(model: NativeWorldModel) -> tuple[str, ...]:
    return tuple(
        name
        for name in (
            tuple(_required_owner_modules(model))
            + tuple(_required_owner_parameters(model))
        )
        if name not in _OPTIONAL_OWNERS
    )


def _owner_parameters(
    model: NativeWorldModel,
) -> Mapping[str, tuple[nn.Parameter, ...]]:
    owners = {
        name: tuple(_unique_parameters(modules))
        for name, modules in _required_owner_modules(model).items()
    }
    owners.update(
        {
            name: tuple(parameter for parameter in parameters if parameter.requires_grad)
            for name, parameters in _required_owner_parameters(model).items()
        }
    )
    by_parameter: dict[int, list[str]] = {}
    for owner, parameters in owners.items():
        if not parameters:
            raise GradientOwnershipError(f"gradient owner {owner} has no parameters")
        for parameter in parameters:
            by_parameter.setdefault(id(parameter), []).append(owner)
    duplicated = {
        name: by_parameter[id(parameter)]
        for name, parameter in model.named_parameters()
        if len(by_parameter.get(id(parameter), ())) > 1
    }
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in by_parameter
    ]
    if duplicated or missing:
        raise GradientOwnershipError(
            "gradient owner table must cover every trainable parameter exactly once: "
            f"missing={missing}, duplicated={duplicated}"
        )
    return owners


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _replication_factor(value: torch.Tensor) -> int:
    """Return how many process-group ranks own an identical local value.

    A regular tensor is the DDP replica in the only supported distributed use
    of this audit.  A DTensor carries enough placement metadata to distinguish
    FSDP2's shard axis from HSDP's replicate axis.  Weighting local statistics
    by the inverse of this factor before the world reduction prevents both
    parameter counts and L2 norms from growing when topology changes.
    """

    placements = getattr(value, "placements", None)
    mesh = getattr(value, "device_mesh", None)
    if placements is None or mesh is None:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1
    factor = 1
    for dimension, placement in enumerate(placements):
        is_replicate = getattr(placement, "is_replicate", None)
        if callable(is_replicate) and is_replicate():
            factor *= int(mesh.size(dimension))
    if factor <= 0:
        raise GradientOwnershipError("invalid DTensor replication factor")
    return factor


def _logical_numel(value: torch.Tensor) -> int:
    """Logical (unsharded and de-duplicated) parameter element count."""

    return int(value.numel())


def _unique_parameters(modules: Iterable[nn.Module]) -> Iterable[nn.Parameter]:
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            if parameter.requires_grad:
                yield parameter


def audit_gradient_ownership(model: NativeWorldModel) -> dict[str, object]:
    """Collect and validate one real backward pass without gathering shards."""

    owners: dict[str, dict[str, object]] = {}
    device = next(model.parameters()).device
    for name, parameters in _owner_parameters(model).items():
        # [gradient elements, nonzero, nonfinite, sumsq].  Parameter counts
        # come from logical DTensor shapes and therefore need no collective.
        statistics = torch.zeros(4, dtype=torch.float64, device=device)
        parameter_elements = sum(_logical_numel(parameter) for parameter in parameters)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            gradient = _local_tensor(parameter.grad.detach())
            if gradient.numel() == 0:
                continue
            weight = 1.0 / float(_replication_factor(parameter))
            finite = torch.isfinite(gradient)
            finite_gradient = torch.where(finite, gradient, torch.zeros_like(gradient))
            statistics[0] += gradient.numel() * weight
            statistics[1] += torch.count_nonzero(finite_gradient) * weight
            statistics[2] += (gradient.numel() - finite.sum()) * weight
            statistics[3] += finite_gradient.double().square().sum() * weight
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        elements, nonzero, nonfinite, squared_norm = statistics.cpu().tolist()
        # Counts were inverse-weighted across replicated ranks.  Round rather
        # than truncate so non-power-of-two replica meshes cannot turn a
        # floating representation such as 0.999999999999 into zero.
        elements = int(round(elements))
        nonzero = int(round(nonzero))
        nonfinite = int(round(nonfinite))
        norm = math.sqrt(float(squared_norm))
        required = name not in _OPTIONAL_OWNERS
        passed = parameter_elements > 0 and nonfinite == 0 and math.isfinite(norm)
        if required:
            passed = passed and elements > 0 and nonzero > 0 and norm > 0.0
        owners[name] = {
            "parameter_elements": parameter_elements,
            "gradient_elements": elements,
            "nonzero_elements": nonzero,
            "nonfinite_elements": nonfinite,
            "l2_norm": norm,
            "required": required,
            "passed": passed,
        }

    failed = sorted(name for name, value in owners.items() if not bool(value["passed"]))
    if failed:
        raise GradientOwnershipError(
            "required Stage0 gradient owners are missing, zero, or non-finite: "
            + ", ".join(failed)
        )
    return {
        "schema": GRADIENT_OWNERSHIP_SCHEMA,
        "passed": True,
        "owners": owners,
    }


def validate_gradient_ownership_receipt(
    receipt: Mapping[str, Any], model: NativeWorldModel
) -> None:
    """Validate checkpoint audit metadata without trusting a boolean flag."""

    if set(receipt) != {"schema", "passed", "owners"}:
        raise GradientOwnershipError("gradient ownership receipt fields mismatch")
    if receipt.get("schema") != GRADIENT_OWNERSHIP_SCHEMA or receipt.get("passed") is not True:
        raise GradientOwnershipError("gradient ownership receipt schema/pass mismatch")
    owners = receipt.get("owners")
    # Rebuild the ownership table while validating: this simultaneously proves
    # that the current model ABI is still covered exactly once.  The receipt
    # contains both required and explicitly optional owners.
    owner_parameters = _owner_parameters(model)
    expected = set(owner_parameters)
    actual = set(owners) if isinstance(owners, Mapping) else set()
    if not isinstance(owners, Mapping) or actual != expected:
        raise GradientOwnershipError(
            "gradient ownership receipt owner set mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    fields = {
        "parameter_elements",
        "gradient_elements",
        "nonzero_elements",
        "nonfinite_elements",
        "l2_norm",
        "required",
        "passed",
    }
    for name, value in owners.items():
        if not isinstance(value, Mapping) or set(value) != fields:
            raise GradientOwnershipError(f"gradient owner {name} fields mismatch")
        count_fields = (
            "parameter_elements",
            "gradient_elements",
            "nonzero_elements",
            "nonfinite_elements",
        )
        if any(type(value[field]) is not int for field in count_fields):
            raise GradientOwnershipError(
                f"gradient owner {name} count fields must be JSON integers"
            )
        if type(value["l2_norm"]) not in {int, float}:
            raise GradientOwnershipError(
                f"gradient owner {name} l2_norm must be a JSON number"
            )
        parameter_elements = int(value["parameter_elements"])
        elements = int(value["gradient_elements"])
        nonzero = int(value["nonzero_elements"])
        nonfinite = int(value["nonfinite_elements"])
        norm = float(value["l2_norm"])
        required = name not in _OPTIONAL_OWNERS
        expected_parameter_elements = sum(
            _logical_numel(parameter) for parameter in owner_parameters[name]
        )
        valid = (
            value["passed"] is True
            and value["required"] is required
            and parameter_elements == expected_parameter_elements
            and 0 <= nonzero <= elements <= parameter_elements
            and nonfinite == 0
            and math.isfinite(norm)
        )
        if required:
            valid = valid and elements > 0 and nonzero > 0 and norm > 0.0
        if not valid:
            raise GradientOwnershipError(f"gradient owner {name} did not pass")
