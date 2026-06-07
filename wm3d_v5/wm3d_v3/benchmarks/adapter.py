"""Benchmark adapter interfaces for WM3D policy evaluation.

The external benchmark packages are optional and currently not installed on the
training machine. These interfaces define the shape each concrete adapter must
provide once LIBERO/CALVIN/SimplerEnv/WorldArena are available.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass
class BenchmarkTask:
    name: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizedObservation:
    """Model-ready observation for one policy call."""

    context_tokens: torch.Tensor
    task_emb: torch.Tensor
    context_rgb: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkStep:
    env_action: Any
    raw_action: torch.Tensor
    selected_idx: int | None = None
    selected_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    task: BenchmarkTask
    success: bool
    steps: int
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenPolicy(Protocol):
    def act_from_tokens(
        self,
        context_tokens: torch.Tensor,
        task_emb: torch.Tensor,
        *,
        context_rgb: torch.Tensor | None = None,
        return_rollouts: bool = False,
    ) -> Any:
        ...


class BenchmarkAdapter(ABC):
    """Abstract external benchmark adapter.

    Concrete adapters own benchmark-specific details:
    observation encoding, coordinate frames, action scale, gripper convention,
    receding-horizon execution, and success extraction. The WM3D policy only
    sees tokenized observations and emits canonical 7-D delta-pose actions.
    """

    name: str

    @abstractmethod
    def iter_tasks(self, *, limit: int | None = None) -> list[BenchmarkTask]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task: BenchmarkTask) -> Any:
        raise NotImplementedError

    @abstractmethod
    def observe(self, env_state: Any, task: BenchmarkTask) -> TokenizedObservation:
        raise NotImplementedError

    @abstractmethod
    def to_env_action(self, raw_action: torch.Tensor, env_state: Any, task: BenchmarkTask) -> Any:
        raise NotImplementedError

    @abstractmethod
    def step(self, env_state: Any, env_action: Any) -> tuple[Any, bool, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def is_success(self, env_state: Any, info: dict[str, Any], task: BenchmarkTask) -> bool:
        raise NotImplementedError

    def close(self, env_state: Any) -> None:
        """Release benchmark resources for one episode."""
        return None

    def rollout_episode(
        self,
        policy: TokenPolicy,
        task: BenchmarkTask,
        *,
        max_steps: int,
    ) -> EpisodeResult:
        env_state = self.reset(task)
        success = False
        info: dict[str, Any] = {}
        try:
            for step_i in range(max_steps):
                obs = self.observe(env_state, task)
                decision = policy.act_from_tokens(
                    obs.context_tokens,
                    obs.task_emb,
                    context_rgb=obs.context_rgb,
                )
                raw_action = decision.first_action_raw[0] if decision.first_action_raw.ndim == 2 else decision.first_action_raw
                env_action = self.to_env_action(raw_action, env_state, task)
                env_state, done, info = self.step(env_state, env_action)
                success = self.is_success(env_state, info, task)
                if done or success:
                    metrics = {"success": float(success)}
                    metrics.update(_numeric_metrics(info))
                    return EpisodeResult(
                        task=task,
                        success=success,
                        steps=step_i + 1,
                        metrics=metrics,
                        metadata={"last_info": info},
                    )
            metrics = {"success": float(success)}
            metrics.update(_numeric_metrics(info))
            return EpisodeResult(
                task=task,
                success=success,
                steps=max_steps,
                metrics=metrics,
                metadata={"last_info": info},
            )
        finally:
            self.close(env_state)


def _numeric_metrics(info: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in info.items():
        if isinstance(value, bool):
            metrics[key] = float(value)
        elif isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics
