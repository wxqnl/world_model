"""Distributed runtime primitives for native WM3D-V7 5B."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
from typing import Iterable, Mapping

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
import yaml

from wm3d_v3.models.native5b import NativeWM3D5B


class RuntimeContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def initialize_distributed(timeout_minutes: int = 30) -> DistributedContext:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeContractError(
            f"torchrun environment misses {missing}; direct python launch is forbidden"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if not torch.cuda.is_available():
        raise RuntimeContractError("native5b formal training requires CUDA")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeContractError(
            f"LOCAL_RANK={local_rank} outside visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
            timeout=timedelta(minutes=int(timeout_minutes)),
        )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeContractError("torchrun environment disagrees with process group")
    return DistributedContext(rank, local_rank, world_size, device)


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_hsdp_mesh(
    context: DistributedContext,
    *,
    shard_degree: int,
) -> DeviceMesh:
    shard_degree = int(shard_degree)
    if shard_degree <= 1:
        raise RuntimeContractError("FSDP2 shard_degree must be greater than one")
    if context.world_size % shard_degree:
        raise RuntimeContractError(
            f"world_size {context.world_size} is not divisible by shard_degree "
            f"{shard_degree}"
        )
    replicate_degree = context.world_size // shard_degree
    if replicate_degree == 1:
        return init_device_mesh("cuda", (shard_degree,), mesh_dim_names=("shard",))
    return init_device_mesh(
        "cuda",
        (replicate_degree, shard_degree),
        mesh_dim_names=("replicate", "shard"),
    )


def apply_fsdp2(
    model: NativeWM3D5B,
    context: DistributedContext,
    *,
    shard_degree: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    reshard_after_forward: bool = True,
) -> DeviceMesh:
    """Bottom-up FSDP2/HSDP wrapping for the V7 native core."""

    mesh = build_hsdp_mesh(context, shard_degree=shard_degree)
    policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=param_dtype,
        cast_forward_inputs=True,
    )
    units = tuple(model.iter_transformer_units())
    if len(units) != (
        model.cfg.state_layers
        + model.cfg.action_layers
        + len(model.cfg.bridge_layers_state)
        + 1
    ):
        raise RuntimeContractError("native5b transformer-unit enumeration drifted")
    seen: set[int] = set()
    for unit in units:
        if id(unit) in seen:
            raise RuntimeContractError("FSDP2 unit appears more than once")
        seen.add(id(unit))
        fully_shard(
            unit,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=reshard_after_forward,
    )
    if not isinstance(model, FSDPModule):
        raise RuntimeContractError("native5b root did not become an FSDPModule")
    return mesh


def set_gradient_sync(model: NativeWM3D5B, enabled: bool) -> None:
    if not isinstance(model, FSDPModule):
        raise RuntimeContractError("gradient sync requested before FSDP2 wrapping")
    model.set_requires_gradient_sync(bool(enabled), recurse=True)
    model.set_reshard_after_backward(bool(enabled), recurse=True)


def initialize_adamw_state(optimizer: torch.optim.AdamW) -> None:
    """Eagerly create every AdamW state tensor before the first DCP save/load.

    PyTorch 2.7's distributed optimizer-state loader assumes every parameter
    listed in a param group has a state entry.  AdamW is normally lazy, so an
    optional modality absent from an early batch would otherwise make that
    checkpoint unloadable.  Eager zero state is equivalent to AdamW's own
    first-use initialization and removes this version-dependent failure mode.
    """

    for group in optimizer.param_groups:
        if group.get("amsgrad", False):
            raise RuntimeContractError("native5b AdamW does not support AMSGrad")
        if group.get("capturable", False) or group.get("fused", False):
            raise RuntimeContractError(
                "native5b eager AdamW state expects capturable=fused=False"
            )
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                required = {"step", "exp_avg", "exp_avg_sq"}
                if set(state) != required:
                    raise RuntimeContractError(
                        "partially initialized AdamW state is forbidden"
                    )
                continue
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.full_tensor().detach().float()
    result = value.detach().float().clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(dist.get_world_size())
    return result


def reduce_metrics(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    names = sorted(metrics)
    packed = torch.stack([metrics[name].detach().float() for name in names])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed.div_(dist.get_world_size())
    return {name: float(value) for name, value in zip(names, packed.cpu())}


def wsd_learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    stable_fraction: float,
    peak_lr: float,
    min_lr: float,
) -> float:
    """Stateless warmup-stable-decay schedule addressed by optimizer step."""

    step = int(step)
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    if not 0 <= step <= total_steps or not 0 <= warmup_steps < total_steps:
        raise RuntimeContractError("invalid WSD step bounds")
    if not 0.0 < stable_fraction < 1.0:
        raise RuntimeContractError("stable_fraction must be in (0,1)")
    if not 0.0 <= min_lr <= peak_lr:
        raise RuntimeContractError("invalid WSD learning rates")
    if step < warmup_steps:
        return peak_lr * float(step + 1) / float(max(1, warmup_steps))
    remaining = total_steps - warmup_steps
    stable_steps = int(round(remaining * stable_fraction))
    decay_start = warmup_steps + stable_steps
    if step < decay_start:
        return peak_lr
    progress = min(1.0, (step - decay_start) / max(1, total_steps - decay_start))
    cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
    return min_lr + (peak_lr - min_lr) * cosine


_FORBIDDEN_CONFIG_TOKEN = re.compile(
    r"(^|[/_.-])(wm3d_v8|qwen[a-z0-9_-]*|wan[a-z0-9_-]*|a2)"
    r"(?=$|[/_.-])"
)


def _forbidden_import(module: str) -> bool:
    parts = str(module).lower().split(".")
    return any(
        part == "wm3d_v8"
        or part == "a2"
        or part.startswith("qwen")
        or part.startswith("wan")
        for part in parts
    )


def _walk_config_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_config_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_config_strings(item)


def assert_v7_native_dependency_boundary(paths: Iterable[Path]) -> None:
    """Reject imports/config values from later V8/VLA/video-generator lines."""

    violations: list[str] = []
    for path in sorted(Path(value) for value in paths):
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules: list[str] = []
                if isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.append(node.module or "")
                for module in modules:
                    if _forbidden_import(module):
                        violations.append(f"{path}: forbidden import {module}")
        elif path.suffix in {".json", ".yaml", ".yml"}:
            text = path.read_text(encoding="utf-8")
            value = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
            for token in _walk_config_strings(value):
                match = _FORBIDDEN_CONFIG_TOKEN.search(token.lower())
                if match:
                    violations.append(
                        f"{path}: forbidden config token {match.group(2)}"
                    )
    if violations:
        raise RuntimeContractError(
            "V7 native dependency boundary failed:\n" + "\n".join(violations)
        )


def verify_parameter_budget(
    model: NativeWM3D5B,
    *,
    minimum: int = 4_800_000_000,
    maximum: int = 5_200_000_000,
) -> dict[str, int]:
    counts = model.parameter_counts()
    total = counts["total"]
    if not int(minimum) <= total <= int(maximum):
        raise RuntimeContractError(
            f"native5b parameter count {total} outside [{minimum}, {maximum}]"
        )
    return counts
