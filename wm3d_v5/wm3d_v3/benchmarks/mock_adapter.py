"""Mock adapter for benchmark-runner smoke tests."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch

from wm3d_v3.benchmarks.adapter import BenchmarkAdapter, BenchmarkTask, TokenizedObservation


@dataclass
class MockState:
    tokens: torch.Tensor
    task_emb: torch.Tensor
    step: int = 0


class MockTokenAdapter(BenchmarkAdapter):
    name = "mock"

    def iter_tasks(self, *, limit: int | None = None) -> list[BenchmarkTask]:
        n = limit or 1
        return [
            BenchmarkTask(name=f"mock_{i:03d}", instruction="mock robot manipulation")
            for i in range(n)
        ]

    def reset(self, task: BenchmarkTask) -> MockState:
        digest = hashlib.sha256(task.name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**31)
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randn(1, 16, 64, 2048, generator=generator) * 0.02
        task_emb = torch.randn(1, 2048, generator=generator) * 0.02
        return MockState(tokens=tokens, task_emb=task_emb)

    def observe(self, env_state: MockState, task: BenchmarkTask) -> TokenizedObservation:
        return TokenizedObservation(
            context_tokens=env_state.tokens,
            task_emb=env_state.task_emb,
            metadata={"mock": True},
        )

    def to_env_action(self, raw_action: torch.Tensor, env_state: MockState, task: BenchmarkTask) -> torch.Tensor:
        return raw_action.detach().float().cpu()

    def step(self, env_state: MockState, env_action: torch.Tensor) -> tuple[MockState, bool, dict]:
        env_state.step += 1
        return env_state, True, {"env_action_norm": float(env_action.float().norm())}

    def is_success(self, env_state: MockState, info: dict, task: BenchmarkTask) -> bool:
        return bool(info.get("env_action_norm", 0.0) >= 0.0)
