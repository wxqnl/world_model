"""Run LIBERO in a separate environment against a WM3D HTTP policy server."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml
from PIL import Image


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


def _frame_to_b64(frame: np.ndarray) -> str:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _policy_act(
    server_url: str,
    task_text: str,
    frames: list[np.ndarray],
    timeout: float,
    *,
    lowdim_state: np.ndarray | None = None,
    object_state: np.ndarray | None = None,
    plan_state: np.ndarray | None = None,
    action_history: np.ndarray | None = None,
    progress_state: float | None = None,
) -> dict[str, Any]:
    payload_obj: dict[str, Any] = {
        "task_text": task_text,
        "frames": [_frame_to_b64(frame) for frame in frames],
    }
    if lowdim_state is not None:
        payload_obj["lowdim_state"] = np.asarray(lowdim_state, dtype=np.float32).reshape(-1).tolist()
    if object_state is not None:
        payload_obj["object_state"] = np.asarray(object_state, dtype=np.float32).reshape(-1).tolist()
    if plan_state is not None:
        payload_obj["plan_state"] = np.asarray(plan_state, dtype=np.float32).reshape(-1).tolist()
    if action_history is not None:
        payload_obj["action_history"] = np.asarray(action_history, dtype=np.float32).tolist()
    if progress_state is not None:
        payload_obj["progress_state"] = [float(progress_state)]
    payload = json.dumps(payload_obj).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/act",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def _extract_frame(obs: dict[str, Any], camera_key: str) -> np.ndarray:
    if camera_key not in obs:
        raise KeyError(f"missing camera key {camera_key!r}; available={sorted(obs.keys())}")
    frame = np.asarray(obs[camera_key])
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"camera frame must be HWC RGB, got {frame.shape}")
    return frame


def _extract_lowdim(obs: dict[str, Any]) -> np.ndarray:
    parts = [
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1),
    ]
    out = np.concatenate(parts).astype(np.float32)
    if out.shape != (12,):
        raise ValueError(f"expected 12D lowdim state, got {out.shape}")
    return out


def _extract_object_state(obs: dict[str, Any]) -> np.ndarray:
    if "object-state" not in obs:
        raise KeyError(f"missing object-state; available={sorted(obs.keys())}")
    out = np.asarray(obs["object-state"], dtype=np.float32).reshape(-1)
    if out.size == 0:
        raise ValueError("object-state is empty")
    return out


def _extract_named_poses(obs: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Extract object/eef pose keys from LIBERO observations for diagnostics.

    This is intentionally not used as policy input. It makes closed-loop
    failures explainable without reverse-engineering the flat object-state
    vector layout.
    """
    out: dict[str, dict[str, list[float]]] = {}
    suffixes = (
        "_to_robot0_eef_pos",
        "_to_robot0_eef_quat",
        "_pos",
        "_quat",
    )
    explicit = {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
    for key, value in obs.items():
        if key not in explicit and not key.endswith(suffixes):
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        if key.startswith("robot0_"):
            entity = "robot0"
            field = key[len("robot0_") :]
        else:
            field = None
            entity = key
            for suffix in suffixes:
                if key.endswith(suffix):
                    entity = key[: -len(suffix)]
                    field = suffix[1:]
                    break
            if field is None:
                continue
        out.setdefault(entity, {})[field] = arr.astype(float).tolist()
    return out


def _is_task1_put_cream_butter(task_text: str) -> bool:
    text = task_text.lower()
    return ("cream_cheese" in text or "cream cheese" in text) and "butter" in text and "basket" in text


def _plan_stage_target(stage: int) -> tuple[int, int, int]:
    stage = int(np.clip(stage, 0, 3))
    target = 0 if stage < 2 else 1
    subgoal = 0 if stage in (0, 2) else 1
    return stage, target, subgoal


def _plan_state_from_stage(
    stage: int,
    named_poses: dict[str, dict[str, list[float]]] | None = None,
    *,
    dim: int = 8,
) -> np.ndarray:
    if dim < 8:
        raise ValueError(f"plan_state_dim must be >= 8, got {dim}")
    stage, target, subgoal = _plan_stage_target(stage)
    out = np.zeros(int(dim), dtype=np.float32)
    out[stage] = 1.0
    out[4 + target] = 1.0
    out[6 + subgoal] = 1.0
    if dim >= 17 and named_poses:
        target_entity = "cream_cheese_1" if target == 0 else "butter_1"
        target_pos = _entity_pos(named_poses, target_entity)
        eef_pos = _entity_pos(named_poses, "robot0")
        basket_pos = _entity_pos(named_poses, "basket_1")
        if target_pos is not None and eef_pos is not None:
            out[8:11] = np.clip(target_pos[:3] - eef_pos[:3], -1.0, 1.0)
        if target_pos is not None and basket_pos is not None:
            out[11:14] = np.clip(basket_pos[:3] - target_pos[:3], -1.0, 1.0)
        if eef_pos is not None and basket_pos is not None:
            out[14:17] = np.clip(basket_pos[:3] - eef_pos[:3], -1.0, 1.0)
    return out


def _entity_pos(named_poses: dict[str, dict[str, list[float]]] | None, entity: str) -> np.ndarray | None:
    if not named_poses:
        return None
    fields = named_poses.get(entity, {})
    value = fields.get("pos")
    if value is None and entity == "robot0":
        value = fields.get("eef_pos")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr[:3] if arr.size >= 3 else None


def _object_eef_dist(named_poses: dict[str, dict[str, list[float]]] | None, entity: str) -> float | None:
    if not named_poses:
        return None
    rel = named_poses.get(entity, {}).get("to_robot0_eef_pos")
    if rel is not None:
        arr = np.asarray(rel, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return float(np.linalg.norm(arr[:3]))
    obj = _entity_pos(named_poses, entity)
    eef = _entity_pos(named_poses, "robot0")
    if obj is None or eef is None:
        return None
    return float(np.linalg.norm(obj[:3] - eef[:3]))


def _object_in_receptacle_xy(
    named_poses: dict[str, dict[str, list[float]]] | None,
    entity: str,
    receptacle: str = "basket_1",
    threshold: float = 0.14,
) -> bool:
    obj = _entity_pos(named_poses, entity)
    rec = _entity_pos(named_poses, receptacle)
    if obj is None or rec is None:
        return False
    return bool(float(np.linalg.norm(obj[:2] - rec[:2])) <= threshold)


def _update_task1_plan_stage(
    stage: int,
    named_poses: dict[str, dict[str, list[float]]] | None,
    *,
    contact_threshold: float = 0.08,
) -> int:
    cream_in = _object_in_receptacle_xy(named_poses, "cream_cheese_1")
    butter_in = _object_in_receptacle_xy(named_poses, "butter_1")
    cream_dist = _object_eef_dist(named_poses, "cream_cheese_1")
    butter_dist = _object_eef_dist(named_poses, "butter_1")
    cream_contact = cream_dist is not None and cream_dist <= contact_threshold
    butter_contact = butter_dist is not None and butter_dist <= contact_threshold

    out = int(np.clip(stage, 0, 3))
    if cream_in:
        out = max(out, 2)
    elif out < 1 and cream_contact:
        out = 1
    if out >= 2:
        if butter_in:
            out = max(out, 3)
        elif out < 3 and butter_contact:
            out = 3
    return out


def _save_frame(frame: np.ndarray, path: Path) -> str:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path)


def _as_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _adapt_gripper(action: np.ndarray, mode: str) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if mode == "identity":
        return out
    if mode == "closed01_to_libero":
        out[..., 6] = np.where(out[..., 6] > 0.5, 1.0, -1.0)
        return out
    raise ValueError(f"unknown gripper_mode={mode!r}")


def _task_name_from_hdf5(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_demo") else name


def _task_id_for_name(suite: Any, task_name: str) -> int:
    for task_id in range(suite.get_num_tasks()):
        if suite.get_task(task_id).name == task_name:
            return task_id
    raise RuntimeError(f"task {task_name!r} not found in suite")


def _load_hdf5_init_state(path: Path, demo_id: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["data"][demo_id].attrs["init_state"])


def _load_hdf5_action_len(path: Path, demo_id: str) -> int:
    with h5py.File(path, "r") as h5:
        return int(np.asarray(h5["data"][demo_id]["actions"]).shape[0])


def _load_hdf5_actions(path: Path, demo_id: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["data"][demo_id]["actions"], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    ap.add_argument("--server_url", default="http://127.0.0.1:8765")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--task_order_index", type=int, default=0)
    ap.add_argument("--max_tasks", type=int, default=1)
    ap.add_argument("--task_ids", default=None, help="Comma-separated explicit task ids. Overrides --max_tasks.")
    ap.add_argument("--init_states", default="0")
    ap.add_argument("--init_state_hdf5", type=Path, default=None)
    ap.add_argument("--init_state_demo_id", default="demo_0")
    ap.add_argument("--expert_action_hdf5", type=Path, default=None)
    ap.add_argument("--expert_action_demo_id", default="demo_0")
    ap.add_argument(
        "--teacher_bc_ckpt",
        type=Path,
        default=None,
        help="Optional official LIBERO BC checkpoint used to label policy-visited states for DAgger caches.",
    )
    ap.add_argument("--teacher_device", default="cuda:0")
    ap.add_argument("--teacher_low_eval_noise", action="store_true")
    ap.add_argument("--teacher_deterministic_action", action="store_true")
    ap.add_argument(
        "--teacher_action_override_from_step",
        type=int,
        default=0,
        help="Diagnostic/intervention: execute the BC teacher from this 1-based step onward.",
    )
    ap.add_argument(
        "--teacher_action_full_replay",
        action="store_true",
        help="Diagnostic/intervention: execute the BC teacher for the whole episode while still logging WM3D outputs.",
    )
    ap.add_argument(
        "--expert_action_prefix_steps",
        type=int,
        default=0,
        help="Use hdf5 expert actions for the first N steps, then hand off to the HTTP policy.",
    )
    ap.add_argument(
        "--expert_action_override_from_step",
        type=int,
        default=0,
        help="Use hdf5 expert actions from this 1-based rollout step onward; useful for DAgger-style recovery traces.",
    )
    ap.add_argument(
        "--expert_action_full_replay",
        action="store_true",
        help="Use hdf5 expert actions for all available steps; useful for runner action-space sanity.",
    )
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera_key", default="agentview_image")
    ap.add_argument("--camera_size", type=int, default=224)
    ap.add_argument("--context_T", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--gripper_mode", default="closed01_to_libero", choices=("identity", "closed01_to_libero"))
    ap.add_argument("--pose_scale", type=float, default=1.0)
    ap.add_argument("--max_pose_norm", type=float, default=0.0)
    ap.add_argument("--send_lowdim", action="store_true")
    ap.add_argument("--send_object_state", action="store_true")
    ap.add_argument("--send_plan_state", action="store_true")
    ap.add_argument("--plan_state_dim", type=int, default=8)
    ap.add_argument(
        "--force_plan_stage3_grip_closed",
        action="store_true",
        help="Diagnostic only: force gripper closed once the task1 plan tracker reaches stage3.",
    )
    ap.add_argument("--action_history_len", type=int, default=0)
    ap.add_argument("--send_progress", action="store_true")
    ap.add_argument("--trace_object_state", action="store_true")
    ap.add_argument(
        "--progress_denominator",
        type=float,
        default=0.0,
        help="denominator for progress_state=(step-1)/denom; <=0 uses hdf5 action length or max_steps",
    )
    ap.add_argument("--request_timeout", type=float, default=120.0)
    ap.add_argument("--save_frames_dir", type=Path, default=None)
    ap.add_argument("--save_frame_every", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    _bootstrap_libero(args.libero_root)
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = get_benchmark(args.suite)(args.task_order_index)
    init_state_ids = [int(x) for x in args.init_states.split(",") if x.strip()]
    hdf5_init_state = None
    hdf5_action_len = None
    expert_actions = None
    if args.init_state_hdf5 is not None:
        hdf5_task_name = _task_name_from_hdf5(args.init_state_hdf5)
        hdf5_init_state = _load_hdf5_init_state(args.init_state_hdf5, args.init_state_demo_id)
        hdf5_action_len = _load_hdf5_action_len(args.init_state_hdf5, args.init_state_demo_id)
        task_ids = [_task_id_for_name(suite, hdf5_task_name)] if not args.task_ids else [int(x) for x in args.task_ids.split(",") if x.strip()]
        init_state_ids = [0]
    elif args.task_ids:
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]
    else:
        task_ids = list(range(min(args.max_tasks, suite.get_num_tasks())))
    if args.expert_action_hdf5 is not None:
        expert_actions = _load_hdf5_actions(args.expert_action_hdf5, args.expert_action_demo_id)
    teacher_cfg = None
    teacher_benchmark = None
    teacher_algo = None
    teacher_raw_obs_to_tensor_obs = None
    teacher_action_fn = None
    if args.teacher_bc_ckpt is not None:
        from libero.lifelong.metric import raw_obs_to_tensor_obs
        import robomimic.utils.obs_utils as ObsUtils
        from wm3d_v3.benchmarks.libero_bc_teacher import (
            _load_rollout_algo as _load_bc_teacher_algo,
            _teacher_action as _bc_teacher_action,
        )

        teacher_args = argparse.Namespace(
            libero_root=args.libero_root,
            ckpt=args.teacher_bc_ckpt,
            device=args.teacher_device,
            low_eval_noise=bool(args.teacher_low_eval_noise),
        )
        teacher_cfg, teacher_benchmark, teacher_algo = _load_bc_teacher_algo(teacher_args)
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": teacher_cfg.data.obs.modality})
        teacher_raw_obs_to_tensor_obs = raw_obs_to_tensor_obs
        teacher_action_fn = _bc_teacher_action
    results: list[dict[str, Any]] = []
    started = time.time()

    for task_id in task_ids:
        if task_id < 0 or task_id >= suite.get_num_tasks():
            raise ValueError(f"task id {task_id} out of range for {args.suite} ({suite.get_num_tasks()} tasks)")
        task = suite.get_task(task_id)
        teacher_task_id = _task_id_for_name(teacher_benchmark, task.name) if teacher_benchmark is not None else None
        teacher_task_emb = teacher_benchmark.get_task_emb(teacher_task_id) if teacher_task_id is not None else None
        bddl = args.libero_root / "libero" / "libero" / "bddl_files" / task.problem_folder / task.bddl_file
        init_states = suite.get_task_init_states(task_id) if hdf5_init_state is None else None
        for init_state_id in init_state_ids:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=args.camera_size,
                camera_widths=args.camera_size,
            )
            env.seed(args.seed)
            env.reset()
            if hdf5_init_state is None:
                obs = env.set_init_state(init_states[init_state_id % int(init_states.shape[0])])
            else:
                obs = env.set_init_state(hdf5_init_state)
            for _ in range(args.warmup_steps):
                obs, _reward, _done, _info = env.step(np.zeros(7, dtype=np.float32))
            first_frame = _extract_frame(obs, args.camera_key)
            frames = deque([first_frame] * args.context_T, maxlen=args.context_T)
            action_history = deque(
                [np.zeros(7, dtype=np.float32) for _ in range(max(0, args.action_history_len))],
                maxlen=max(0, args.action_history_len),
            )
            success = False
            last_info: dict[str, Any] = {}
            step_trace: list[dict[str, Any]] = []
            steps = 0
            plan_stage = 0
            if teacher_algo is not None:
                teacher_algo.reset()
            try:
                for steps in range(1, args.max_steps + 1):
                    frame_path = None
                    if args.save_frames_dir is not None and (
                        args.save_frame_every <= 1 or (steps - 1) % args.save_frame_every == 0
                    ):
                        frame_path = _save_frame(
                            frames[-1],
                            args.save_frames_dir
                            / f"task{task_id:03d}_init{init_state_id:03d}_step{steps:04d}.png",
                    )
                    lowdim_state = _extract_lowdim(obs) if args.send_lowdim else None
                    object_state = _extract_object_state(obs) if (args.trace_object_state or args.send_object_state) else None
                    named_poses = _extract_named_poses(obs) if (args.trace_object_state or args.send_plan_state) else None
                    plan_state = None
                    if args.send_plan_state:
                        if _is_task1_put_cream_butter(task.language):
                            plan_stage = _update_task1_plan_stage(plan_stage, named_poses)
                            plan_state = _plan_state_from_stage(plan_stage, named_poses, dim=args.plan_state_dim)
                        else:
                            plan_state = np.zeros(int(args.plan_state_dim), dtype=np.float32)
                    hist_arr = np.stack(list(action_history), axis=0) if args.action_history_len > 0 else None
                    progress_denominator = float(args.progress_denominator)
                    if progress_denominator <= 0:
                        if hdf5_action_len is not None:
                            progress_denominator = max(1.0, float(hdf5_action_len - 1))
                        else:
                            progress_denominator = max(1.0, float(args.max_steps - 1))
                    progress_state = (
                        min(1.0, max(0.0, float(steps - 1) / progress_denominator))
                        if args.send_progress
                        else None
                    )
                    teacher_action = None
                    if teacher_algo is not None:
                        if teacher_raw_obs_to_tensor_obs is None or teacher_action_fn is None or teacher_task_emb is None:
                            raise RuntimeError("teacher BC components are not initialized")
                        teacher_data = teacher_raw_obs_to_tensor_obs([obs], teacher_task_emb, teacher_cfg)
                        teacher_action = teacher_action_fn(
                            teacher_algo.policy,
                            teacher_data,
                            deterministic=bool(args.teacher_deterministic_action),
                        ).reshape(-1, 7)[0].astype(np.float32)
                    use_expert_action = (
                        expert_actions is not None
                        and steps <= len(expert_actions)
                        and (
                            args.expert_action_full_replay
                            or steps <= int(args.expert_action_prefix_steps)
                            or (
                                int(args.expert_action_override_from_step) > 0
                                and steps >= int(args.expert_action_override_from_step)
                            )
                        )
                    )
                    use_teacher_action = (
                        teacher_action is not None
                        and (
                            args.teacher_action_full_replay
                            or (
                                int(args.teacher_action_override_from_step) > 0
                                and steps >= int(args.teacher_action_override_from_step)
                            )
                        )
                    )
                    if use_expert_action:
                        expert_chunk = expert_actions[steps - 1 : steps - 1 + 8].astype(np.float32)
                        if len(expert_chunk) == 0:
                            expert_chunk = expert_actions[-1:].astype(np.float32)
                        if len(expert_chunk) < 8:
                            expert_chunk = np.concatenate(
                                [expert_chunk, np.repeat(expert_chunk[-1:], 8 - len(expert_chunk), axis=0)],
                                axis=0,
                            )
                        policy_result = {
                            "first_action_raw": expert_chunk[0].astype(float).tolist(),
                            "action_chunk_raw": expert_chunk.astype(float).tolist(),
                            "selected_idx": -2,
                            "selected_score": 0.0,
                            "candidate_scores": [],
                        }
                        policy_action = expert_chunk[0].astype(np.float32)
                        if args.expert_action_full_replay:
                            action_source = "expert_full"
                        elif (
                            int(args.expert_action_override_from_step) > 0
                            and steps >= int(args.expert_action_override_from_step)
                        ):
                            action_source = "expert_override"
                        else:
                            action_source = "expert_prefix"
                    elif use_teacher_action:
                        policy_result = {
                            "first_action_raw": teacher_action.astype(float).tolist(),
                            "action_chunk_raw": np.repeat(teacher_action[None], 8, axis=0).astype(float).tolist(),
                            "selected_idx": -4,
                            "selected_score": 0.0,
                            "candidate_scores": [],
                        }
                        policy_action = teacher_action.astype(np.float32)
                        action_source = "teacher_bc"
                    else:
                        policy_result = _policy_act(
                            args.server_url,
                            task.language,
                            list(frames),
                            args.request_timeout,
                            lowdim_state=lowdim_state,
                            object_state=object_state if args.send_object_state else None,
                            plan_state=plan_state,
                            action_history=hist_arr,
                            progress_state=progress_state,
                        )
                        policy_action = np.asarray(policy_result["first_action_raw"], dtype=np.float32)
                        action_source = "policy"
                    forced_plan_grip = False
                    if args.force_plan_stage3_grip_closed and plan_state is not None and int(plan_stage) >= 3:
                        policy_action = np.asarray(policy_action, dtype=np.float32).copy()
                        policy_action[6] = 1.0
                        forced_plan_grip = True
                    action = _adapt_gripper(policy_action, args.gripper_mode)
                    if float(args.pose_scale) != 1.0:
                        action[:6] *= float(args.pose_scale)
                    if float(args.max_pose_norm) > 0:
                        pose_norm = float(np.linalg.norm(action[:6].astype(np.float32)))
                        if pose_norm > float(args.max_pose_norm):
                            action[:6] *= float(args.max_pose_norm) / max(pose_norm, 1e-6)
                    obs, reward, done, info = env.step(action)
                    if args.action_history_len > 0:
                        action_history.append(action.astype(np.float32))
                    frames.append(_extract_frame(obs, args.camera_key))
                    success = bool(done) or bool(reward >= 1.0) or bool(env.check_success())
                    last_info = dict(info or {})
                    last_info.update({"reward": float(reward), "done": float(done)})
                    trace_item = {
                        "step": steps,
                        "action": action.astype(float).tolist(),
                        "policy_action": policy_action.astype(float).tolist(),
                        "action_norm": float(np.linalg.norm(action.astype(np.float32))),
                        "policy_action_norm": float(np.linalg.norm(policy_action.astype(np.float32))),
                        "action_source": action_source,
                        "pose_scale": float(args.pose_scale),
                        "max_pose_norm": float(args.max_pose_norm),
                        "gripper_mode": args.gripper_mode,
                        "policy_gripper": float(policy_action[6]),
                        "env_gripper": float(action[6]),
                        "forced_plan_grip": bool(forced_plan_grip),
                        "reward": float(reward),
                        "done": bool(done),
                        "success": bool(success),
                    }
                    if lowdim_state is not None:
                        trace_item["lowdim_state"] = lowdim_state.astype(float).tolist()
                    if teacher_action is not None:
                        trace_item["teacher_action"] = teacher_action.astype(float).tolist()
                    if object_state is not None:
                        trace_item["object_state"] = object_state.astype(float).tolist()
                    if plan_state is not None:
                        trace_item["plan_state"] = plan_state.astype(float).tolist()
                        trace_item["plan_stage"] = int(plan_stage)
                    if named_poses is not None:
                        trace_item["named_poses"] = named_poses
                    if hist_arr is not None:
                        trace_item["action_history"] = hist_arr.astype(float).tolist()
                    if progress_state is not None:
                        trace_item["progress_state"] = float(progress_state)
                        trace_item["progress_denominator"] = float(progress_denominator)
                    if frame_path is not None:
                        trace_item["frame_path"] = frame_path
                    if "selected_idx" in policy_result:
                        trace_item["selected_idx"] = int(policy_result["selected_idx"])
                    if "selected_score" in policy_result:
                        trace_item["selected_score"] = float(policy_result["selected_score"])
                    candidate_scores = _as_float_list(policy_result.get("candidate_scores"))
                    if candidate_scores is not None:
                        trace_item["candidate_scores"] = candidate_scores
                    if isinstance(policy_result.get("action_chunk_raw"), list):
                        policy_chunk = np.asarray(policy_result["action_chunk_raw"], dtype=np.float32)
                        trace_item["policy_action_chunk_raw"] = policy_chunk.astype(float).tolist()
                        env_chunk = _adapt_gripper(policy_chunk, args.gripper_mode)
                        if float(args.pose_scale) != 1.0:
                            env_chunk[..., :6] *= float(args.pose_scale)
                        if float(args.max_pose_norm) > 0:
                            flat = env_chunk.reshape(-1, env_chunk.shape[-1])
                            norms = np.linalg.norm(flat[:, :6].astype(np.float32), axis=1)
                            over = norms > float(args.max_pose_norm)
                            flat[over, :6] *= (float(args.max_pose_norm) / np.maximum(norms[over], 1e-6))[:, None]
                        trace_item["action_chunk_raw"] = env_chunk.astype(float).tolist()
                    step_trace.append(trace_item)
                    if success or done:
                        break
            finally:
                env.close()
            results.append({
                "suite": args.suite,
                "task_id": task_id,
                "task_name": task.name,
                "instruction": task.language,
                "init_state_id": init_state_id,
                "init_state_source": "hdf5" if hdf5_init_state is not None else "suite",
                "init_state_hdf5": str(args.init_state_hdf5) if args.init_state_hdf5 is not None else None,
                "init_state_demo_id": args.init_state_demo_id if args.init_state_hdf5 is not None else None,
                "success": success,
                "steps": steps,
                "last_info": last_info,
                "step_trace": step_trace,
            })

    success_rate = sum(float(item["success"]) for item in results) / max(1, len(results))
    report = {
        "trace_schema_version": 2,
        "suite": args.suite,
        "max_tasks": args.max_tasks,
        "task_ids": task_ids,
        "init_states": init_state_ids,
        "init_state_source": "hdf5" if hdf5_init_state is not None else "suite",
        "init_state_hdf5": str(args.init_state_hdf5) if args.init_state_hdf5 is not None else None,
        "init_state_demo_id": args.init_state_demo_id if args.init_state_hdf5 is not None else None,
        "max_steps": args.max_steps,
        "camera_key": args.camera_key,
        "camera_size": args.camera_size,
        "context_T": args.context_T,
        "warmup_steps": args.warmup_steps,
        "gripper_mode": args.gripper_mode,
        "pose_scale": float(args.pose_scale),
        "max_pose_norm": float(args.max_pose_norm),
        "send_lowdim": bool(args.send_lowdim),
        "send_object_state": bool(args.send_object_state),
        "send_plan_state": bool(args.send_plan_state),
        "plan_state_dim": int(args.plan_state_dim),
        "force_plan_stage3_grip_closed": bool(args.force_plan_stage3_grip_closed),
        "action_history_len": int(args.action_history_len),
        "send_progress": bool(args.send_progress),
        "trace_object_state": bool(args.trace_object_state),
        "expert_action_hdf5": str(args.expert_action_hdf5) if args.expert_action_hdf5 is not None else None,
        "expert_action_demo_id": args.expert_action_demo_id if args.expert_action_hdf5 is not None else None,
        "expert_action_prefix_steps": int(args.expert_action_prefix_steps),
        "expert_action_override_from_step": int(args.expert_action_override_from_step),
        "expert_action_full_replay": bool(args.expert_action_full_replay),
        "teacher_bc_ckpt": str(args.teacher_bc_ckpt) if args.teacher_bc_ckpt is not None else None,
        "teacher_action_override_from_step": int(args.teacher_action_override_from_step),
        "teacher_action_full_replay": bool(args.teacher_action_full_replay),
        "progress_denominator": float(args.progress_denominator),
        "hdf5_action_len": int(hdf5_action_len) if hdf5_action_len is not None else None,
        "save_frames_dir": str(args.save_frames_dir) if args.save_frames_dir is not None else None,
        "server_url": args.server_url,
        "success_rate": success_rate,
        "seconds": time.time() - started,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out), "success_rate": success_rate}, sort_keys=True))


if __name__ == "__main__":
    main()
