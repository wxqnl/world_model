"""Keep continuous training-flow time and interpolation coefficients in FP32.

The released scheduler samples in FP32 but casts its result to the activation
dtype. BF16 rounds an interval near t=1000 to the endpoint, whose official
training weight is zero. Retain the same random draws, shift and loss formula;
cast only the final noisy activation back to the expert's compute dtype.
Inference schedules and Euler updates are deliberately unchanged.
"""
from __future__ import annotations

from typing import Any

import torch


class FP32TrainingFlowScheduler:
    """Precision-only adapter around the existing upstream training scheduler."""

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    def __getattr__(self, name: str) -> Any:
        scheduler = object.__getattribute__(self, "_scheduler")
        return getattr(scheduler, name)

    def sample_training_t(
        self, batch_size: int, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        del dtype  # Activation precision must not quantize the continuous clock.
        return self._scheduler.sample_training_t(
            batch_size=batch_size, device=device, dtype=torch.float32,
        )

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        # Delegating the FP32 interpolation preserves the upstream formula and
        # broadcasting. Merely retaining FP32 t is insufficient: upstream
        # add_noise casts sigma to original_samples.dtype before mixing.
        with torch.autocast(device_type=original_samples.device.type, enabled=False):
            mixed = self._scheduler.add_noise(
                original_samples.float(), noise.float(), timestep.float(),
            )
        return mixed.to(dtype=original_samples.dtype)
