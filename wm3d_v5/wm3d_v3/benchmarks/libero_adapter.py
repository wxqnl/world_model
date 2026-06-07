"""LIBERO benchmark adapter for WM3D policies.

The adapter follows the official LIBERO entrypoints:

* `libero.libero.benchmark.get_benchmark`
* `libero.libero.envs.OffScreenRenderEnv`

It is safe to import without LIBERO/robosuite installed. Construction fails
with an actionable message if the external environment is not available.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from wm3d_v3.benchmarks.adapter import BenchmarkAdapter, BenchmarkTask, TokenizedObservation


DEFAULT_LIBERO_ROOT = Path("/data/Minko/benchmarks/LIBERO")


@dataclass
class LiberoState:
    env: Any
    obs: dict[str, Any]
    frame_window: Any
    task_text: str
    step: int = 0


def _resolve_libero_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).expanduser().resolve()
    env_root = os.environ.get("LIBERO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return DEFAULT_LIBERO_ROOT


def _bootstrap_libero(root: Path) -> None:
    if not root.exists():
        raise RuntimeError(
            f"LIBERO root does not exist: {root}. Clone https://github.com/Lifelong-Robot-Learning/LIBERO "
            "or set LIBERO_ROOT to an installed checkout."
        )
    package_parent = root
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    benchmark_root = root / "libero" / "libero"
    config_dir = root / ".wm3d_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        config = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(root / "datasets"),
            "assets": str(benchmark_root / "assets"),
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))


def _load_benchmark_api(root: Path):
    _bootstrap_libero(root)
    try:
        from libero.libero.benchmark import get_benchmark
    except Exception as exc:  # pragma: no cover - depends on external package
        raise RuntimeError(
            "Failed to import LIBERO benchmark API. Ensure the LIBERO checkout is installed or on PYTHONPATH."
        ) from exc
    return get_benchmark


def _load_env_api(root: Path):
    _bootstrap_libero(root)
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:  # pragma: no cover - depends on robosuite/mujoco
        raise RuntimeError(
            "Failed to import LIBERO OffScreenRenderEnv. Install LIBERO runtime dependencies in a separate "
            "environment: python=3.8.13, robosuite==1.4.0, robomimic==0.2.0, bddl==1.0.1, "
            "and torch/torchvision versions compatible with that environment."
        ) from exc
    return OffScreenRenderEnv


class LiberoAdapter(BenchmarkAdapter):
    name = "libero"

    def __init__(
        self,
        *,
        cfg: dict[str, Any],
        libero_root: str | Path | None = None,
        suite: str = "libero_10",
        task_order_index: int = 0,
        init_state_ids: list[int] | None = None,
        seed: int = 0,
        camera_key: str = "agentview_image",
        camera_size: int = 224,
        warmup_steps: int = 5,
        device: str = "cuda:0",
        qwen_device: str | None = None,
        task_cache_dir: str | Path | None = None,
        allow_zero_task_fallback: bool = False,
    ) -> None:
        self.cfg = cfg
        self.root = _resolve_libero_root(libero_root)
        get_benchmark = _load_benchmark_api(self.root)
        self.suite_name = suite
        self.suite = get_benchmark(suite)(task_order_index)
        self.init_state_ids = init_state_ids or [0]
        self.seed = int(seed)
        self.camera_key = camera_key
        self.camera_size = int(camera_size)
        self.warmup_steps = int(warmup_steps)
        data_cfg = cfg["data"]
        token_grid = 16 if data_cfg.get("tokens_subdir") == "vggt_p256" else 8
        from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer

        self.tokenizer = OnlineObservationTokenizer(
            T=int(data_cfg["T"]),
            token_grid=token_grid,
            task_cache_dir=task_cache_dir,
            device=device,
            qwen_device=qwen_device,
            allow_zero_task_fallback=allow_zero_task_fallback,
        )

    def iter_tasks(self, *, limit: int | None = None) -> list[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for task_id in range(self.suite.get_num_tasks()):
            task = self.suite.get_task(task_id)
            for init_state_id in self.init_state_ids:
                tasks.append(
                    BenchmarkTask(
                        name=f"{self.suite_name}/{task_id:03d}/init{init_state_id:03d}/{task.name}",
                        instruction=task.language,
                        metadata={
                            "suite": self.suite_name,
                            "task_id": task_id,
                            "task_name": task.name,
                            "init_state_id": int(init_state_id),
                            "problem_folder": task.problem_folder,
                            "bddl_file": task.bddl_file,
                        },
                    )
                )
                if limit is not None and len(tasks) >= limit:
                    return tasks
        return tasks

    def reset(self, task: BenchmarkTask) -> LiberoState:
        OffScreenRenderEnv = _load_env_api(self.root)
        task_id = int(task.metadata["task_id"])
        libero_task = self.suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(self.root / "libero" / "libero" / "bddl_files" / libero_task.problem_folder / libero_task.bddl_file),
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
        )
        env.seed(self.seed)
        env.reset()
        init_states = self.suite.get_task_init_states(task_id)
        init_state_id = int(task.metadata.get("init_state_id", 0)) % int(init_states.shape[0])
        obs = env.set_init_state(init_states[init_state_id])
        for _ in range(self.warmup_steps):
            obs, _reward, _done, _info = env.step(np.zeros(7, dtype=np.float32))
        from wm3d_v3.benchmarks.online_tokenizer import FrameWindow

        frame_window = FrameWindow(int(self.cfg["data"]["T"]))
        frame_window.reset(self._extract_frame(obs))
        return LiberoState(env=env, obs=obs, frame_window=frame_window, task_text=task.instruction)

    def observe(self, env_state: LiberoState, task: BenchmarkTask) -> TokenizedObservation:
        return self.tokenizer.tokenize(env_state.frame_window.list(), env_state.task_text)

    def to_env_action(self, raw_action: torch.Tensor, env_state: LiberoState, task: BenchmarkTask) -> np.ndarray:
        action = raw_action.detach().float().cpu().numpy().astype(np.float32)
        if action.shape != (7,):
            raise ValueError(f"expected one 7D action, got {action.shape}")
        return action

    def step(self, env_state: LiberoState, env_action: np.ndarray) -> tuple[LiberoState, bool, dict[str, Any]]:
        obs, reward, done, info = env_state.env.step(env_action)
        env_state.obs = obs
        env_state.step += 1
        env_state.frame_window.append(self._extract_frame(obs))
        success = bool(done) or bool(reward >= 1.0) or bool(getattr(env_state.env, "check_success", lambda: False)())
        info = dict(info or {})
        info.update({
            "reward": float(reward),
            "env_done": float(done),
            "success": float(success),
            "step": float(env_state.step),
        })
        return env_state, bool(done), info

    def is_success(self, env_state: LiberoState, info: dict[str, Any], task: BenchmarkTask) -> bool:
        return bool(info.get("success", 0.0))

    def close(self, env_state: LiberoState) -> None:
        try:
            env_state.env.close()
        except Exception:
            pass

    def _extract_frame(self, obs: dict[str, Any]) -> np.ndarray:
        if self.camera_key not in obs:
            raise KeyError(f"LIBERO observation missing camera key {self.camera_key!r}; available keys={sorted(obs.keys())}")
        frame = np.asarray(obs[self.camera_key])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"camera {self.camera_key!r} must be HWC RGB, got {frame.shape}")
        return frame
