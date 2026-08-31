"""Run LIBERO in a separate environment against a WM3D HTTP policy server."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
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


def _frame_payload(
    frames: list[np.ndarray] | list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    if not frames:
        raise ValueError("empty frame window")
    first = frames[0]
    if isinstance(first, dict):
        cameras = sorted(first.keys())
        return {
            "frames_by_camera": {
                camera: [_frame_to_b64(step_frames[camera]) for step_frames in frames]
                for camera in cameras
            }
        }
    return {"frames": [_frame_to_b64(frame) for frame in frames]}  # type: ignore[arg-type]


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _assert_canonical_actions(value: np.ndarray, name: str, *, ndim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim or arr.shape[-1] != 7:
        raise ValueError(f"{name} must have shape {'[7]' if ndim == 1 else '[N,7]'}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    grip = arr[..., 6]
    if np.any(grip < 0.0) or np.any(grip > 1.0):
        raise ValueError(f"{name} grip must use canonical close01 values in [0,1]")
    return arr


def _policy_request_payload(
    task_text: str,
    frames: list[np.ndarray] | list[dict[str, np.ndarray]],
    *,
    lowdim_state: np.ndarray | None = None,
    object_state: np.ndarray | None = None,
    plan_state: np.ndarray | None = None,
    action_history: np.ndarray | None = None,
    progress_state: float | None = None,
) -> dict[str, Any]:
    if not isinstance(task_text, str) or not task_text.strip():
        raise ValueError("task_text must be a non-empty string")
    payload: dict[str, Any] = {"task_text": task_text, **_frame_payload(frames)}
    for name, value in (
        ("lowdim_state", lowdim_state),
        ("object_state", object_state),
        ("plan_state", plan_state),
    ):
        if value is not None:
            payload[name] = _finite_vector(value, name).tolist()
    if action_history is not None:
        payload["action_history"] = _assert_canonical_actions(
            action_history,
            "action_history",
            ndim=2,
        ).tolist()
    if progress_state is not None:
        progress = float(progress_state)
        if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress_state must be finite and in [0,1], got {progress_state!r}")
        payload["progress_state"] = [progress]
    return payload


def _validate_policy_response(result: dict[str, Any]) -> dict[str, Any]:
    if "error" in result:
        raise RuntimeError(result["error"])
    if "first_action_raw" not in result:
        raise ValueError("policy response is missing first_action_raw")
    _assert_canonical_actions(result["first_action_raw"], "first_action_raw", ndim=1)
    if "first_action_continuous_raw" in result:
        _assert_canonical_actions(
            result["first_action_continuous_raw"],
            "first_action_continuous_raw",
            ndim=1,
        )
    for key in ("action_chunk_raw", "action_chunk_continuous_raw"):
        if key in result:
            chunk = _assert_canonical_actions(result[key], key, ndim=2)
            if chunk.shape[0] == 0:
                raise ValueError(f"{key} must contain at least one action")
    if "selected_gripper_prob" in result:
        grip = np.asarray(result["selected_gripper_prob"], dtype=np.float32)
        if grip.ndim != 1 or not np.isfinite(grip).all() or np.any(grip < 0.0) or np.any(grip > 1.0):
            raise ValueError("selected_gripper_prob must be a finite [K] array in [0,1]")
    return result


def _urlopen_json(request: urllib.request.Request, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        text = body.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {text}") from exc
    return json.loads(body.decode("utf-8"))


def _policy_act(
    server_url: str,
    task_text: str,
    frames: list[np.ndarray] | list[dict[str, np.ndarray]],
    timeout: float,
    *,
    lowdim_state: np.ndarray | None = None,
    object_state: np.ndarray | None = None,
    plan_state: np.ndarray | None = None,
    action_history: np.ndarray | None = None,
    progress_state: float | None = None,
) -> dict[str, Any]:
    payload_obj = _policy_request_payload(
        task_text,
        frames,
        lowdim_state=lowdim_state,
        object_state=object_state,
        plan_state=plan_state,
        action_history=action_history,
        progress_state=progress_state,
    )
    payload = json.dumps(payload_obj).encode("utf-8")
    request = urllib.request.Request(
        server_url.rstrip("/") + "/act",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result = _urlopen_json(request, timeout)
    return _validate_policy_response(result)


def _post_json(server_url: str, endpoint: str, payload_obj: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        server_url.rstrip("/") + endpoint,
        data=json.dumps(payload_obj).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    result = _urlopen_json(request, timeout)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def _teacher_server_reset(server_url: str, timeout: float) -> None:
    _post_json(server_url, "/reset", {}, timeout)


def _teacher_server_action(
    server_url: str,
    obs: dict[str, Any],
    task_name: str,
    timeout: float,
) -> np.ndarray:
    result = _post_json(
        server_url,
        "/act",
        {
            "task_name": task_name,
            "agentview_image": _frame_to_b64(_extract_frame(obs, "agentview_image")),
            "eye_in_hand_image": _frame_to_b64(
                _extract_frame(obs, "robot0_eye_in_hand_image")
            ),
            "gripper_qpos": np.asarray(
                obs["robot0_gripper_qpos"], dtype=np.float32
            ).reshape(-1).tolist(),
            "joint_pos": np.asarray(
                obs["robot0_joint_pos"], dtype=np.float32
            ).reshape(-1).tolist(),
        },
        timeout,
    )
    return np.asarray(result["action"], dtype=np.float32).reshape(7)


def _extract_frame(obs: dict[str, Any], camera_key: str, *, rotate_180: bool = False) -> np.ndarray:
    if camera_key not in obs:
        raise KeyError(f"missing camera key {camera_key!r}; available={sorted(obs.keys())}")
    frame = np.asarray(obs[camera_key])
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"camera frame must be HWC RGB, got {frame.shape}")
    if rotate_180:
        frame = frame[::-1, ::-1, :]
    return frame


def _extract_frame_window_item(
    obs: dict[str, Any],
    camera_keys: list[str],
    *,
    rotate_180: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    if len(camera_keys) == 1:
        return _extract_frame(obs, camera_keys[0], rotate_180=rotate_180)
    return {
        camera_key: _extract_frame(obs, camera_key, rotate_180=rotate_180)
        for camera_key in camera_keys
    }


def _debug_frame(frame_item: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(frame_item, dict):
        return np.concatenate([frame_item[key] for key in sorted(frame_item)], axis=1)
    return frame_item


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


def _canonical_executed_action(
    policy_action: np.ndarray,
    *,
    pose_scale: float,
    max_pose_norm: float,
    closed_threshold: float,
) -> np.ndarray:
    source = np.asarray(policy_action, dtype=np.float32)
    if source.shape != (7,):
        raise ValueError(f"policy action must have shape [7], got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("policy action contains non-finite values")
    canonical = source.copy()
    canonical[:6] *= float(pose_scale)
    if float(max_pose_norm) > 0:
        pose_norm = float(np.linalg.norm(canonical[:6]))
        if pose_norm > float(max_pose_norm):
            canonical[:6] *= float(max_pose_norm) / max(pose_norm, 1e-6)
    canonical[6] = float(source[6] > float(closed_threshold))
    return _assert_canonical_actions(canonical, "canonical executed action", ndim=1)


def _apply_gripper_height_gate(
    policy_action: np.ndarray,
    obs: dict[str, Any],
    *,
    min_close_z: float,
    closed_threshold: float,
    released: bool,
) -> tuple[np.ndarray, bool, bool, float | None]:
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"policy action must be a finite [7] vector, got {action.shape}")
    gated = action.copy()
    if float(min_close_z) <= 0.0 or bool(released):
        return gated, False, bool(released), None
    if "robot0_eef_pos" not in obs:
        raise KeyError("gripper_min_close_z requires obs['robot0_eef_pos']")
    eef_position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    if eef_position.size < 3 or not np.isfinite(eef_position[:3]).all():
        raise ValueError("obs['robot0_eef_pos'] must contain a finite xyz position")
    eef_z = float(eef_position[2])
    if eef_z <= float(min_close_z):
        return gated, False, True, eef_z
    if float(gated[6]) > float(closed_threshold):
        gated[6] = 0.0
        return gated, True, False, eef_z
    return gated, False, False, eef_z


def _env_step_action(canonical_action: np.ndarray, mode: str) -> np.ndarray:
    out = _assert_canonical_actions(canonical_action, "canonical env-step action", ndim=1).copy()
    if mode == "identity":
        return out
    if mode == "closed01_to_libero":
        out[6] = 1.0 if out[6] > 0.5 else -1.0
        return out
    raise ValueError(f"unknown gripper_mode={mode!r}")


def _trace_action_fields(
    policy_action: np.ndarray,
    executed_action_canonical: np.ndarray,
    env_action: np.ndarray,
) -> dict[str, Any]:
    policy = np.asarray(policy_action, dtype=np.float32)
    if policy.shape != (7,) or not np.isfinite(policy).all():
        raise ValueError(f"policy_action must be a finite [7] vector, got {policy.shape}")
    canonical = _assert_canonical_actions(
        executed_action_canonical,
        "executed_action_canonical",
        ndim=1,
    )
    env = np.asarray(env_action, dtype=np.float32)
    if env.shape != (7,) or not np.isfinite(env).all():
        raise ValueError(f"env_action must be a finite [7] vector, got {env.shape}")
    return {
        "action": canonical.astype(float).tolist(),
        "executed_action_canonical": canonical.astype(float).tolist(),
        "env_action": env.astype(float).tolist(),
        "policy_action": policy.astype(float).tolist(),
        "action_norm": float(np.linalg.norm(canonical)),
        "policy_action_norm": float(np.linalg.norm(policy)),
        "policy_gripper": float(policy[6]),
        "env_gripper": float(env[6]),
    }


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


def _build_arg_parser() -> argparse.ArgumentParser:
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
    ap.add_argument(
        "--teacher_server_url",
        default=None,
        help="Optional remote BC teacher server; keeps GPU inference outside the simulator environment.",
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
    ap.add_argument(
        "--camera_keys",
        default=None,
        help="Comma-separated camera keys. Overrides --camera_key and sends frames_by_camera to the policy server.",
    )
    ap.add_argument("--camera_size", type=int, default=224)
    ap.add_argument("--rotate_180", action="store_true", help="Rotate camera frames 180 degrees before tokenization.")
    ap.add_argument("--context_T", type=int, default=16)
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument(
        "--exec_horizon",
        type=int,
        default=1,
        help="Number of actions to execute from one policy chunk before querying again.",
    )
    ap.add_argument("--gripper_mode", default="closed01_to_libero", choices=("identity", "closed01_to_libero"))
    ap.add_argument("--gripper_closed_threshold", type=float, default=0.5)
    ap.add_argument(
        "--gripper_min_close_z",
        "--gripper-min-close-z",
        dest="gripper_min_close_z",
        type=float,
        default=0.0,
        help=(
            "Serve-side grasp gate: block gripper close until eef z <= this value; 0 disables. "
            "The gate releases permanently after reaching the configured height."
        ),
    )
    ap.add_argument(
        "--use_policy_gripper_prob",
        action="store_true",
        help="Use continuous policy gripper probability from the server instead of the binarized raw action.",
    )
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
    ap.add_argument("--action_history_len", type=int, default=1)
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
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()
    camera_keys = (
        [item.strip() for item in str(args.camera_keys).split(",") if item.strip()]
        if args.camera_keys
        else [str(args.camera_key)]
    )
    if not camera_keys:
        raise ValueError("at least one camera key is required")

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
    if args.teacher_bc_ckpt is not None and args.teacher_server_url is not None:
        raise ValueError("use only one of --teacher_bc_ckpt or --teacher_server_url")
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
            first_frame = _extract_frame_window_item(obs, camera_keys, rotate_180=bool(args.rotate_180))
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
            chunk_buf = None
            chunk_cursor = 0
            last_policy_result: dict[str, Any] | None = None
            grasp_gate_released = False
            if teacher_algo is not None:
                teacher_algo.reset()
            if args.teacher_server_url is not None:
                _teacher_server_reset(args.teacher_server_url, args.request_timeout)
            try:
                for steps in range(1, args.max_steps + 1):
                    frame_path = None
                    if args.save_frames_dir is not None and (
                        args.save_frame_every <= 1 or (steps - 1) % args.save_frame_every == 0
                    ):
                        frame_path = _save_frame(
                            _debug_frame(frames[-1]),
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
                    if args.teacher_server_url is not None:
                        teacher_action = _teacher_server_action(
                            args.teacher_server_url,
                            obs,
                            task.name,
                            args.request_timeout,
                        )
                    elif teacher_algo is not None:
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
                        if chunk_buf is not None and chunk_cursor < len(chunk_buf):
                            policy_result = dict(last_policy_result or {})
                            policy_action = np.asarray(chunk_buf[chunk_cursor], dtype=np.float32)
                            chunk_cursor += 1
                            action_source = "policy_chunk"
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
                            action_chunk = np.asarray(
                                (
                                    policy_result.get("action_chunk_continuous_raw")
                                    if args.use_policy_gripper_prob
                                    else policy_result.get("action_chunk_raw")
                                )
                                or [policy_result["first_action_raw"]],
                                dtype=np.float32,
                            ).reshape(-1, 7)
                            horizon = max(1, int(args.exec_horizon))
                            chunk_buf = action_chunk[:horizon]
                            if len(chunk_buf) == 0:
                                chunk_buf = np.asarray(policy_result["first_action_raw"], dtype=np.float32).reshape(1, 7)
                            policy_action = np.asarray(chunk_buf[0], dtype=np.float32)
                            chunk_cursor = 1
                            last_policy_result = dict(policy_result)
                            action_source = "policy"
                    forced_plan_grip = False
                    policy_action_raw = np.asarray(policy_action, dtype=np.float32).copy()
                    execution_policy_action = policy_action_raw.copy()
                    if args.force_plan_stage3_grip_closed and plan_state is not None and int(plan_stage) >= 3:
                        execution_policy_action[6] = 1.0
                        forced_plan_grip = True
                    (
                        execution_policy_action,
                        grasp_gate_active,
                        grasp_gate_released,
                        grasp_gate_eef_z,
                    ) = _apply_gripper_height_gate(
                        execution_policy_action,
                        obs,
                        min_close_z=float(args.gripper_min_close_z),
                        closed_threshold=float(args.gripper_closed_threshold),
                        released=grasp_gate_released,
                    )
                    canonical_action = _canonical_executed_action(
                        execution_policy_action,
                        pose_scale=float(args.pose_scale),
                        max_pose_norm=float(args.max_pose_norm),
                        closed_threshold=float(args.gripper_closed_threshold),
                    )
                    # LIBERO's -1/1 encoding exists only for the actual env call.
                    env_action = _env_step_action(canonical_action, args.gripper_mode)
                    obs, reward, done, info = env.step(env_action)
                    if args.action_history_len > 0:
                        action_history.append(canonical_action.copy())
                    frames.append(_extract_frame_window_item(obs, camera_keys, rotate_180=bool(args.rotate_180)))
                    success = bool(done) or bool(reward >= 1.0) or bool(env.check_success())
                    last_info = dict(info or {})
                    last_info.update({"reward": float(reward), "done": float(done)})
                    trace_item = {
                        "step": steps,
                        **_trace_action_fields(policy_action_raw, canonical_action, env_action),
                        "action_source": action_source,
                        "pose_scale": float(args.pose_scale),
                        "max_pose_norm": float(args.max_pose_norm),
                        "gripper_mode": args.gripper_mode,
                        "gripper_closed_threshold": float(args.gripper_closed_threshold),
                        "use_policy_gripper_prob": bool(args.use_policy_gripper_prob),
                        "forced_plan_grip": bool(forced_plan_grip),
                        "grasp_gate_active": bool(grasp_gate_active),
                        "grasp_gate_released": bool(grasp_gate_released),
                        "grasp_gate_eef_z": grasp_gate_eef_z,
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
                        if isinstance(policy_result.get("action_chunk_continuous_raw"), list):
                            trace_item["policy_action_chunk_continuous_raw"] = np.asarray(
                                policy_result["action_chunk_continuous_raw"],
                                dtype=np.float32,
                            ).astype(float).tolist()
                        if isinstance(policy_result.get("selected_gripper_prob"), list):
                            trace_item["selected_gripper_prob"] = [
                                float(x) for x in policy_result["selected_gripper_prob"]
                            ]
                        canonical_chunk = np.stack(
                            [
                                _canonical_executed_action(
                                    item,
                                    pose_scale=float(args.pose_scale),
                                    max_pose_norm=float(args.max_pose_norm),
                                    closed_threshold=float(args.gripper_closed_threshold),
                                )
                                for item in policy_chunk
                            ],
                            axis=0,
                        )
                        trace_item["action_chunk_raw"] = canonical_chunk.astype(float).tolist()
                    step_trace.append(trace_item)
                    if success or done:
                        break
            finally:
                if os.environ.get("LIBERO_SKIP_ENV_CLOSE", "0") != "1":
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
        "trace_schema_version": 3,
        "suite": args.suite,
        "max_tasks": args.max_tasks,
        "task_ids": task_ids,
        "init_states": init_state_ids,
        "init_state_source": "hdf5" if hdf5_init_state is not None else "suite",
        "init_state_hdf5": str(args.init_state_hdf5) if args.init_state_hdf5 is not None else None,
        "init_state_demo_id": args.init_state_demo_id if args.init_state_hdf5 is not None else None,
        "max_steps": args.max_steps,
        "camera_key": args.camera_key,
        "camera_keys": camera_keys,
        "camera_size": args.camera_size,
        "rotate_180": bool(args.rotate_180),
        "context_T": args.context_T,
        "warmup_steps": args.warmup_steps,
        "gripper_mode": args.gripper_mode,
        "gripper_min_close_z": float(args.gripper_min_close_z),
        "gripper_min_close_z_deprecated": False,
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
        "teacher_server_url": args.teacher_server_url,
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
