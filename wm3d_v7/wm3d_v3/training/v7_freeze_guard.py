"""Auditable freeze guards for Stage1/Stage2 world-model isolation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn as nn


def assert_module_frozen(module: nn.Module) -> None:
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"module contains trainable parameters: {trainable[:8]}")


def assert_no_grad(module: nn.Module) -> None:
    offenders = []
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and bool(torch.any(parameter.grad != 0)):
            offenders.append(name)
    if offenders:
        raise RuntimeError(f"frozen module received nonzero gradients: {offenders[:8]}")


def assert_optimizer_excludes(module: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    frozen_ids = {id(parameter) for parameter in module.parameters()}
    overlaps = sum(
        id(parameter) in frozen_ids
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    if overlaps:
        raise RuntimeError(f"optimizer contains {overlaps} frozen world parameters")


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    # ``view(dtype)`` rejects zero-dimensional tensors; flatten first.  Moving
    # after the byte view also keeps this valid for bfloat16, which NumPy does
    # not represent directly.
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).digest()


@dataclass(frozen=True)
class ModuleFingerprint:
    sha256: str
    excluded_prefixes: tuple[str, ...] = ()

    @classmethod
    def capture(
        cls,
        module: nn.Module,
        *,
        exclude_prefixes: tuple[str, ...] = (),
    ) -> "ModuleFingerprint":
        excluded = tuple(str(prefix) for prefix in exclude_prefixes)
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            if name.startswith(excluded):
                continue
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(_tensor_digest(tensor))
        return cls(digest.hexdigest(), excluded)

    def assert_unchanged(self, module: nn.Module) -> None:
        current = ModuleFingerprint.capture(
            module,
            exclude_prefixes=self.excluded_prefixes,
        )
        if current.sha256 != self.sha256:
            raise RuntimeError(f"frozen module changed: {self.sha256} != {current.sha256}")
