"""Label proposer candidates by branching the LIBERO simulator from expert states."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def _bootstrap_libero(root):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    benchmark_root = root / "libero" / "libero"
    config_dir = root / ".wm3d_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def _task_map(suite):
    return {suite.get_task(index).name: suite.get_task(index) for index in range(suite.get_num_tasks())}


def _candidate_env_action(action):
    out = np.asarray(action, dtype=np.float32).copy()
    if out.shape != (7,) or not np.isfinite(out).all():
        raise ValueError("candidate action must be finite [7]")
    out[6] = 1.0 if out[6] > 0.5 else -1.0
    return out


def _state_l1(state, reference):
    left = np.asarray(state, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        return float("nan")
    return float(np.mean(np.abs(left - right)))


def _run_branch(env, init_state, chunk, continuation, post_reference, final_reference, candidate):
    env.reset()
    env.set_init_state(init_state)
    success = False
    reward_sum = 0.0
    steps = 0
    post_state = None
    for action in chunk:
        env_action = _candidate_env_action(action) if candidate else np.asarray(action, dtype=np.float32)
        _obs, reward, done, _info = env.step(env_action)
        steps += 1
        reward_sum += float(reward)
        success = bool(done) or bool(reward >= 1.0) or bool(env.check_success())
        if success:
            break
    post_state = np.asarray(env.get_sim_state())
    if not success:
        for action in continuation:
            _obs, reward, done, _info = env.step(np.asarray(action, dtype=np.float32))
            steps += 1
            reward_sum += float(reward)
            success = bool(done) or bool(reward >= 1.0) or bool(env.check_success())
            if success:
                break
    final_state = np.asarray(env.get_sim_state())
    return {
        "success": bool(success),
        "reward_sum": float(reward_sum),
        "steps": int(steps),
        "post_state_l1": _state_l1(post_state, post_reference),
        "final_state_l1": _state_l1(final_state, final_reference),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_order_index", type=int, default=0)
    ap.add_argument("--camera_size", type=int, default=64)
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    _bootstrap_libero(args.libero_root)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs.env_wrapper import ControlEnv, OffScreenRenderEnv

    payload = np.load(args.input)
    rows = [json.loads(str(item)) for item in payload["rows_json"]]
    candidate_raw = np.asarray(payload["candidate_raw"], dtype=np.float32)
    candidate_score = np.asarray(payload["candidate_score"], dtype=np.float32)
    expert_action = np.asarray(payload["expert_action"], dtype=np.float32)
    if len(rows) != len(candidate_raw):
        raise ValueError("row and candidate counts differ")

    suite = get_benchmark(args.suite)(args.task_order_index)
    tasks = _task_map(suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp_out = args.out.with_suffix(args.out.suffix + ".tmp")
    completed = 0
    with temp_out.open("w") as fh:
        for row_index, row in enumerate(rows):
            task = tasks[row["task_name"]]
            bddl = (
                args.libero_root
                / "libero"
                / "libero"
                / "bddl_files"
                / task.problem_folder
                / task.bddl_file
            )
            with h5py.File(row["hdf5_path"], "r") as h5:
                demo = h5["data"][row["demo_id"]]
                states = np.asarray(demo["states"])
                actions = np.asarray(demo["actions"], dtype=np.float32)
            start = int(row["target_start"])
            horizon = int(candidate_raw.shape[2])
            post_index = min(start + horizon, len(states) - 1)
            continuation = actions[min(start + horizon, len(actions)) :]
            env_class = ControlEnv if args.no_render else OffScreenRenderEnv
            env_kwargs = dict(
                bddl_file_name=str(bddl),
                camera_heights=args.camera_size,
                camera_widths=args.camera_size,
            )
            if args.no_render:
                env_kwargs.update(
                    use_camera_obs=False,
                    has_offscreen_renderer=False,
                )
            env = env_class(**env_kwargs)
            env.seed(0)
            branches = []
            try:
                factual = _run_branch(
                    env,
                    states[start],
                    expert_action[row_index],
                    continuation,
                    states[post_index],
                    states[-1],
                    False,
                )
                factual.update({"candidate_index": -1, "model_score": None})
                branches.append(factual)
                for candidate_index, chunk in enumerate(candidate_raw[row_index]):
                    result = _run_branch(
                        env,
                        states[start],
                        chunk,
                        continuation,
                        states[post_index],
                        states[-1],
                        True,
                    )
                    result.update(
                        {
                            "candidate_index": int(candidate_index),
                            "model_score": float(candidate_score[row_index, candidate_index]),
                        }
                    )
                    branches.append(result)
            except Exception as exc:
                branches.append({"candidate_index": -99, "error": repr(exc)})
            finally:
                env.close()
            output = {
                "row": row,
                "source_shard": str(args.input),
                "source_row_index": int(row_index),
                "factual_chunk": expert_action[row_index].tolist(),
                "candidate_cond": payload["candidate_cond"][row_index].astype(np.float32).tolist(),
                "candidate_raw": candidate_raw[row_index].tolist(),
                "branches": branches,
            }
            fh.write(json.dumps(output, sort_keys=True) + "\n")
            fh.flush()
            completed += 1
            print(json.dumps({"completed": completed, "total": len(rows), "task": row["task_name"]}), flush=True)
    temp_out.replace(args.out)
    print(json.dumps({"out": str(args.out), "rows": completed}, sort_keys=True))


if __name__ == "__main__":
    main()
