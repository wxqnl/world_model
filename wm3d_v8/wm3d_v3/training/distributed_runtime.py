"""Model-size independent DDP/FSDP2 runtime for WM3D V8."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
import os
from typing import ContextManager, Mapping, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.tensor import _random as dtensor_random
from torch.nn.parallel import DistributedDataParallel


class DistributedRuntimeError(RuntimeError):
    pass


_DTYPES: Mapping[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class DistributedStrategyConfig:
    strategy: str = "ddp"
    shard_degree: int = 1
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    output_dtype: str = "bf16"
    reshard_after_forward: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = False
    timeout_minutes: int = 30
    initialization: str = "direct"

    def validate(self, *, world_size: Optional[int] = None) -> None:
        if self.strategy not in {"ddp", "fsdp2"}:
            raise DistributedRuntimeError(
                f"distributed.strategy must be ddp or fsdp2, got {self.strategy!r}"
            )
        for name in ("param_dtype", "reduce_dtype", "output_dtype"):
            if getattr(self, name).lower() not in _DTYPES:
                raise DistributedRuntimeError(f"unsupported {name}={getattr(self, name)!r}")
        if self.timeout_minutes <= 0:
            raise DistributedRuntimeError("timeout_minutes must be positive")
        if self.initialization not in {"direct", "meta_sharded"}:
            raise DistributedRuntimeError(
                "distributed.initialization must be direct or meta_sharded"
            )
        if self.strategy == "ddp":
            if self.initialization != "direct":
                raise DistributedRuntimeError("DDP requires direct initialization")
            if self.shard_degree not in {0, 1}:
                raise DistributedRuntimeError("DDP does not accept shard_degree > 1")
        else:
            if self.shard_degree <= 1:
                raise DistributedRuntimeError("FSDP2 requires shard_degree > 1")
            if world_size is not None and world_size % self.shard_degree:
                raise DistributedRuntimeError(
                    f"world_size={world_size} is not divisible by shard_degree={self.shard_degree}"
                )
            if self.initialization != "meta_sharded":
                raise DistributedRuntimeError(
                    "FSDP2 requires meta_sharded initialization to avoid a full-model "
                    "replica on every rank"
                )


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    backend: str

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class WrappedModel:
    model: torch.nn.Module
    strategy: str
    mesh: Optional[DeviceMesh]


def strategy_from_mapping(mapping: Mapping[str, object]) -> DistributedStrategyConfig:
    values = dict(mapping)
    config = DistributedStrategyConfig(**values)
    config.validate()
    return config


def initialize_distributed(
    config: DistributedStrategyConfig,
    *,
    allow_single_process: bool = False,
) -> DistributedContext:
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        if not allow_single_process:
            raise DistributedRuntimeError(
                f"torchrun environment misses {missing}; direct launch is forbidden"
            )
        rank = local_rank = 0
        world_size = local_world_size = 1
    else:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    config.validate(world_size=world_size)
    if not torch.cuda.is_available():
        raise DistributedRuntimeError("WM3D pretraining requires CUDA")
    if not 0 <= local_rank < torch.cuda.device_count():
        raise DistributedRuntimeError(
            f"LOCAL_RANK={local_rank} is outside {torch.cuda.device_count()} visible devices"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    backend = "nccl"
    if not dist.is_initialized():
        if world_size == 1 and allow_single_process and missing:
            # Distributed Checkpoint and FSDP2 still require a process group.
            init_method = f"file:///tmp/wm3d-v8-single-{os.getpid()}.pg"
        else:
            init_method = "env://"
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            device_id=device,
            timeout=timedelta(minutes=config.timeout_minutes),
        )
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise DistributedRuntimeError("torchrun environment disagrees with process group")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        device=device,
        backend=backend,
    )


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_fsdp_mesh(
    context: DistributedContext,
    *,
    shard_degree: int,
) -> DeviceMesh:
    if context.world_size % shard_degree:
        raise DistributedRuntimeError(
            f"world_size={context.world_size} is not divisible by shard_degree={shard_degree}"
        )
    replicate_degree = context.world_size // shard_degree
    if replicate_degree == 1:
        return init_device_mesh("cuda", (shard_degree,), mesh_dim_names=("shard",))
    return init_device_mesh(
        "cuda",
        (replicate_degree, shard_degree),
        mesh_dim_names=("replicate", "shard"),
    )


def wrap_model(
    model: torch.nn.Module,
    context: DistributedContext,
    config: DistributedStrategyConfig,
    *,
    initialization_seed: Optional[int] = None,
) -> WrappedModel:
    """Move and wrap any WM3D model according to runtime config.

    FSDP2 requires the model to expose ``iter_fsdp_units``.  This is an
    architecture interface, not a model-size check.
    """

    config.validate(world_size=context.world_size)
    if config.strategy == "ddp":
        if initialization_seed is not None:
            raise DistributedRuntimeError(
                "initialization_seed is reserved for FSDP2 meta_sharded initialization"
            )
        if any(parameter.is_meta for parameter in model.parameters()):
            raise DistributedRuntimeError("DDP cannot materialize a meta model")
        model = model.to(context.device)
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                find_unused_parameters=config.find_unused_parameters,
                broadcast_buffers=config.broadcast_buffers,
            )
        return WrappedModel(model=model, strategy="ddp", mesh=None)

    is_meta = any(parameter.is_meta for parameter in model.parameters())
    if config.initialization == "meta_sharded" and not is_meta:
        raise DistributedRuntimeError(
            "FSDP2 meta_sharded runtime received an already materialized model"
        )
    if config.initialization == "meta_sharded" and initialization_seed is None:
        raise DistributedRuntimeError(
            "FSDP2 meta_sharded initialization requires an explicit global seed"
        )
    iterator = getattr(model, "iter_fsdp_units", None)
    if not callable(iterator):
        raise DistributedRuntimeError(
            f"{type(model).__name__} does not expose iter_fsdp_units()"
        )
    units = tuple(iterator())
    if not units:
        raise DistributedRuntimeError("iter_fsdp_units() returned no units")
    if len({id(unit) for unit in units}) != len(units):
        raise DistributedRuntimeError("one module appears more than once in FSDP units")
    checkpoint_iterator = getattr(model, "iter_activation_checkpoint_units", None)
    if callable(checkpoint_iterator):
        checkpoint_units = tuple(checkpoint_iterator())
        if len({id(unit) for unit in checkpoint_units}) != len(checkpoint_units):
            raise DistributedRuntimeError(
                "one module appears more than once in activation checkpoint units"
            )
        fsdp_unit_ids = {id(unit) for unit in units}
        missing = [
            type(unit).__name__
            for unit in checkpoint_units
            if id(unit) not in fsdp_unit_ids
        ]
        if missing:
            raise DistributedRuntimeError(
                "activation checkpoint units must be identical FSDP units: "
                + ", ".join(missing)
            )
        not_wrapped = [
            type(unit).__name__
            for unit in checkpoint_units
            if not isinstance(unit, CheckpointWrapper)
        ]
        if not_wrapped:
            raise DistributedRuntimeError(
                "activation checkpoint units must be checkpoint-wrapped before "
                "fully_shard so mixed-precision recomputation crosses one stable "
                "FSDP boundary: "
                + ", ".join(not_wrapped)
            )
    mesh = _build_fsdp_mesh(context, shard_degree=config.shard_degree)
    policy = MixedPrecisionPolicy(
        param_dtype=_DTYPES[config.param_dtype.lower()],
        reduce_dtype=_DTYPES[config.reduce_dtype.lower()],
        output_dtype=_DTYPES[config.output_dtype.lower()],
        cast_forward_inputs=True,
    )
    for unit in units:
        fully_shard(
            unit,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=config.reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=config.reshard_after_forward,
    )
    if not isinstance(model, FSDPModule):
        raise DistributedRuntimeError("root module did not become an FSDPModule")
    if is_meta:
        # Install the DTensor offset RNG before reset_parameters().  Plain
        # per-rank CUDA seeds would initialize every local shard with the same
        # numbers, silently repeating parameter blocks in the global tensor.
        dtensor_random.manual_seed(int(initialization_seed), mesh)
        _materialize_meta_shards(model, context.device)
    return WrappedModel(model=model, strategy="fsdp2", mesh=mesh)


def _materialize_meta_shards(model: torch.nn.Module, device: torch.device) -> None:
    """Materialize and initialize each FSDP2-owned shard without full replicas."""

    with torch.no_grad():
        for module in model.modules():
            state = list(module.parameters(recurse=False)) + list(
                module.buffers(recurse=False)
            )
            if not state or not any(value.is_meta for value in state):
                continue
            reset = getattr(module, "reset_parameters", None)
            if not callable(reset):
                raise DistributedRuntimeError(
                    f"meta-owned module {type(module).__name__} lacks reset_parameters()"
                )
            module.to_empty(device=device, recurse=False)
            reset()
    remaining = [name for name, value in model.state_dict().items() if value.is_meta]
    if remaining:
        raise DistributedRuntimeError(
            f"meta materialization left unresolved state: {remaining[:8]}"
        )


def set_gradient_sync(model: torch.nn.Module, enabled: bool) -> None:
    """Toggle synchronization for gradient accumulation on either strategy."""

    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(bool(enabled), recurse=True)
        model.set_reshard_after_backward(bool(enabled), recurse=True)
        return
    if isinstance(model, DistributedDataParallel):
        model.require_backward_grad_sync = bool(enabled)
        return
    if dist.get_world_size() != 1:
        raise DistributedRuntimeError("unwrapped multi-rank model cannot toggle sync")


def no_sync_context(model: torch.nn.Module, *, enabled: bool) -> ContextManager[object]:
    if not enabled:
        return nullcontext()
    if isinstance(model, DistributedDataParallel):
        return model.no_sync()
    if isinstance(model, FSDPModule):
        @contextmanager
        def _fsdp_no_sync():
            set_gradient_sync(model, False)
            try:
                yield
            finally:
                set_gradient_sync(model, True)

        return _fsdp_no_sync()
    return nullcontext()


def initialize_adamw_state(optimizer: torch.optim.AdamW) -> None:
    """Eagerly create every state tensor so early sparse batches are resumable."""

    for group in optimizer.param_groups:
        if group.get("amsgrad", False):
            raise DistributedRuntimeError("WM3D AdamW does not support AMSGrad")
        if group.get("capturable", False) or group.get("fused", False):
            raise DistributedRuntimeError(
                "eager AdamW state requires capturable=fused=False"
            )
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise DistributedRuntimeError("partially initialized AdamW state")
                continue
            state["step"] = torch.tensor(0.0, dtype=torch.float32)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)


def autocast_context(config: DistributedStrategyConfig) -> ContextManager[object]:
    dtype = _DTYPES[config.param_dtype.lower()]
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def reduce_metrics(metrics: Mapping[str, torch.Tensor]) -> dict[str, float]:
    names = sorted(metrics)
    packed = torch.stack([metrics[name].detach().float() for name in names])
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed.div_(dist.get_world_size())
    return {name: float(value) for name, value in zip(names, packed.cpu())}
