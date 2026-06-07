"""Export LIBERO expert lowdim/object-state references by replaying a demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wm3d_v3.benchmarks.libero_remote_runner import (
    _bootstrap_libero,
    _extract_lowdim,
    _extract_named_poses,
    _extract_object_state,
    _load_hdf5_init_state,
    _task_id_for_name,
    _task_name_from_hdf5,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_order_index", type=int, default=0)
    ap.add_argument("--expert_hdf5", type=Path, required=True)
    ap.add_argument("--demo_id", default="demo_0")
    ap.add_argument("--camera_size", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _bootstrap_libero(args.libero_root)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    task_name = _task_name_from_hdf5(args.expert_hdf5)
    suite = get_benchmark(args.suite)(args.task_order_index)
    task_id = _task_id_for_name(suite, task_name)
    task = suite.get_task(task_id)
    bddl = args.libero_root / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file

    with h5py.File(args.expert_hdf5, "r") as h5:
        actions = np.asarray(h5["data"][args.demo_id]["actions"], dtype=np.float32)
    init_state = _load_hdf5_init_state(args.expert_hdf5, args.demo_id)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
    )
    lowdim: list[np.ndarray] = []
    object_state: list[np.ndarray] = []
    named_poses: list[dict[str, dict[str, list[float]]]] = []
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(init_state)
        for action in actions:
            lowdim.append(_extract_lowdim(obs))
            object_state.append(_extract_object_state(obs))
            named_poses.append(_extract_named_poses(obs))
            obs, _reward, done, _info = env.step(action.astype(np.float32))
            if bool(done):
                break
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        lowdim=np.stack(lowdim).astype(np.float32),
        object_state=np.stack(object_state).astype(np.float32),
        actions=actions[: len(lowdim)].astype(np.float32),
        named_poses_json=np.asarray([json.dumps(named_poses, sort_keys=True)]),
        task_id=np.asarray([task_id], dtype=np.int32),
        task_name=np.asarray([task_name]),
        instruction=np.asarray([task.language]),
        expert_hdf5=np.asarray([str(args.expert_hdf5)]),
        demo_id=np.asarray([args.demo_id]),
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "task_id": int(task_id),
                "task_name": task_name,
                "steps": len(lowdim),
                "object_dim": int(object_state[0].shape[0]) if object_state else 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
