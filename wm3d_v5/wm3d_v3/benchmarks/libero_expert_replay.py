"""Replay LIBERO expert HDF5 actions in the simulator for action-space sanity."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


def _bootstrap_libero(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
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


def _task_name_from_hdf5(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_demo") else name


def _find_task(suite: Any, task_name: str) -> tuple[int, Any]:
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        if task.name == task_name:
            return task_id, task
    raise RuntimeError(f"task {task_name!r} not found in suite")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_order_index", type=int, default=0)
    ap.add_argument("--hdf5", type=Path, required=True)
    ap.add_argument("--demo_id", default="demo_0")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--camera_size", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _bootstrap_libero(args.libero_root)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = get_benchmark(args.suite)(args.task_order_index)
    task_name = _task_name_from_hdf5(args.hdf5)
    task_id, task = _find_task(suite, task_name)
    bddl = args.libero_root / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file

    with h5py.File(args.hdf5, "r") as h5:
        demo = h5["data"][args.demo_id]
        init_state = np.asarray(demo.attrs["init_state"])
        actions = np.asarray(demo["actions"], dtype=np.float32)
        demo_rewards = np.asarray(demo["rewards"]) if "rewards" in demo else None
        demo_dones = np.asarray(demo["dones"]) if "dones" in demo else None

    if args.max_steps > 0:
        actions = actions[: args.max_steps]

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    env.seed(0)
    env.reset()
    env.set_init_state(init_state)

    step_trace: list[dict[str, Any]] = []
    success = False
    last_reward = 0.0
    last_done = False
    try:
        for step_i, action in enumerate(actions, start=1):
            _obs, reward, done, _info = env.step(action)
            success = bool(done) or bool(reward >= 1.0) or bool(env.check_success())
            last_reward = float(reward)
            last_done = bool(done)
            step_trace.append({
                "step": step_i,
                "reward": last_reward,
                "done": last_done,
                "success": bool(success),
                "action_norm": float(np.linalg.norm(action)),
            })
            if success:
                break
    finally:
        env.close()

    report = {
        "suite": args.suite,
        "task_id": task_id,
        "task_name": task_name,
        "hdf5": str(args.hdf5),
        "demo_id": args.demo_id,
        "num_actions": int(actions.shape[0]),
        "demo_reward_sum": int(demo_rewards.sum()) if demo_rewards is not None else None,
        "demo_done_sum": int(demo_dones.sum()) if demo_dones is not None else None,
        "success": bool(success),
        "steps": int(step_trace[-1]["step"]) if step_trace else 0,
        "last_reward": last_reward,
        "last_done": last_done,
        "step_trace_tail": step_trace[-5:],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out), "success": success, "steps": report["steps"]}, sort_keys=True))


if __name__ == "__main__":
    main()
