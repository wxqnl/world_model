#!/usr/bin/env python3
"""Generate exact same-root RoboCasa K=4 counterfactual rollouts.

This script is deliberately fail-closed.  Every branch starts from a full XML
and initial-state reset followed by the same factual action prefix.  Directly
loading ``states[t0]`` is forbidden because controller and environment hidden
state would not be restored by that shortcut.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROBOSUITE_VERSION = "1.5.2"
ROBOSUITE_COMMIT = "6c10ef24a4bb52f59199976125060ce793470e6e"
MUJOCO_VERSION = "3.3.1"
ROBOCASA_COMMIT = "8f3c96ec8d1bfcd8126cad2bca887da98d30e997"
ROBOCASA_DATASET_VERSION = "0.5.1"
SOURCE_REVISION = "bf736c0cc8f9ea8740c812901eec02bce09517f1"
CAMERAS = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)
BRANCH_NAMES = ("factual", "eef_x_plus", "eef_x_minus", "arm_hold")
ACTION_KEY_ORDERING_HDF5 = {
    "end_effector_position": (0, 3),
    "end_effector_rotation": (3, 6),
    "gripper_close": (6, 7),
    "base_motion": (7, 11),
    "control_mode": (11, 12),
}


class ReplayDivergence(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{array.shape}".encode("ascii")
    return sha256_bytes(header + array.tobytes())


def array_bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare dtype, shape, and payload bytes (including signed zero)."""
    lhs = np.ascontiguousarray(left)
    rhs = np.ascontiguousarray(right)
    return lhs.dtype == rhs.dtype and lhs.shape == rhs.shape and lhs.tobytes() == rhs.tobytes()


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def stable_split(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:16], 16) % 10_000
    return "train" if bucket < 8_000 else ("val" if bucket < 9_000 else "test")


def root_split_group(task_name: str, ep_meta: dict[str, Any], episode_id: int) -> str:
    """Keep all K branches together while splitting at independent roots.

    Task/layout/style alone is too coarse: a whole task-scene pair then lands
    in one split.  Episode identity is the independent simulator provenance
    unit, and all branches from that episode remain inseparable.
    """
    scene = f"layout={ep_meta.get('layout_id', 'na')}/style={ep_meta.get('style_id', 'na')}"
    return f"{task_name}/{scene}/episode={int(episode_id):06d}"


def select_arm_root_candidates(
    actions: np.ndarray,
    state_count: int,
    *,
    horizon: int,
    perturb_steps: int,
    minimum_prefix: int,
    motion_threshold: float,
    priority_steps: list[int] | tuple[int, ...] | None = None,
) -> list[int]:
    """Return deterministic high-motion roots that stay in arm control mode."""
    candidates: list[tuple[int, int, float, int]] = []
    priority_steps = sorted(int(step) for step in (priority_steps or ()))
    for t0 in range(minimum_prefix, state_count - horizon):
        prefix = actions[t0 : t0 + perturb_steps]
        if len(prefix) != perturb_steps or np.any(prefix[:, 11] > 0):
            continue
        motion = float(np.linalg.norm(prefix[:, :3], axis=1).mean())
        if motion > motion_threshold:
            in_horizon = [step for step in priority_steps if t0 <= step < t0 + horizon]
            priority_rank = 0 if in_horizon else 1
            distance = min((abs((t0 + horizon - 1) - step) for step in priority_steps), default=10**9)
            candidates.append((priority_rank, distance, -motion, t0))
    # Current-runtime success proximity is only a root-selection hint.  Every
    # accepted label is still recomputed from the true same-root branches.
    return [t0 for _priority, _distance, _neg_motion, t0 in sorted(candidates)]


def limit_diverse_candidates(
    candidates: list[int], *, max_candidates: int, minimum_gap: int
) -> list[int]:
    """Keep the priority order while avoiding near-duplicate simulator roots."""

    if max_candidates <= 0:
        return []
    minimum_gap = max(1, int(minimum_gap))
    selected: list[int] = []
    for candidate in candidates:
        if all(abs(candidate - previous) >= minimum_gap for previous in selected):
            selected.append(candidate)
            if len(selected) >= max_candidates:
                break
    return selected


def make_k4_action_chunks(
    factual: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    perturb_steps: int,
    delta_fraction: float,
) -> np.ndarray:
    """Create factual, +/-EEF-x, and arm-hold chunks in simulator ordering."""
    if factual.ndim != 2 or factual.shape[1] != 12:
        raise ValueError(f"expected simulator-order [H,12] actions, got {factual.shape}")
    chunks = np.repeat(factual[None], 4, axis=0)
    delta = float(delta_fraction) * float(high[0] - low[0])
    chunks[1, :perturb_steps, 0] += delta
    chunks[2, :perturb_steps, 0] -= delta
    chunks[3, :perturb_steps, :6] = 0.0
    return np.clip(chunks, low[None, None], high[None, None])


def make_grip_flip_action_chunk(
    factual: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    grip_steps: int,
) -> np.ndarray:
    """Flip only the gripper command for a bounded true-simulator branch.

    The branch deliberately preserves all six EEF dimensions, base motion, and
    control mode.  Each gripper command is sent to the opposite side of the
    actuator range, which guarantees a non-factual action even when the factual
    chunk itself contains an open/close transition.
    """

    if factual.ndim != 2 or factual.shape[1] != 12:
        raise ValueError(f"expected simulator-order [H,12] actions, got {factual.shape}")
    if low.shape != (12,) or high.shape != (12,):
        raise ValueError(f"expected action limits [12], got {low.shape} and {high.shape}")
    if grip_steps <= 0 or grip_steps > factual.shape[0]:
        raise ValueError(
            f"grip_steps must be in [1,{factual.shape[0]}], got {grip_steps}"
        )
    chunk = factual.copy()
    midpoint = 0.5 * float(low[6] + high[6])
    factual_grip = factual[:grip_steps, 6]
    chunk[:grip_steps, 6] = np.where(
        factual_grip > midpoint,
        float(low[6]),
        float(high[6]),
    )
    chunk = np.clip(chunk, low[None], high[None])
    if np.array_equal(chunk, factual):
        raise RuntimeError("grip flip produced a factual duplicate")
    return chunk


def _runtime_modules():
    import mujoco
    import robocasa
    import robosuite

    if robosuite.__version__ != ROBOSUITE_VERSION:
        raise RuntimeError(f"robosuite must be {ROBOSUITE_VERSION}, got {robosuite.__version__}")
    if mujoco.__version__ != MUJOCO_VERSION:
        raise RuntimeError(f"mujoco must be {MUJOCO_VERSION}, got {mujoco.__version__}")
    robosuite_source = str(Path(robosuite.__file__).resolve())
    robocasa_source = str(Path(robocasa.__file__).resolve())
    if ROBOSUITE_COMMIT not in robosuite_source:
        raise RuntimeError(
            f"robosuite must be installed from pinned commit {ROBOSUITE_COMMIT}, "
            f"got {robosuite_source}"
        )
    if ROBOCASA_COMMIT not in robocasa_source:
        raise RuntimeError(
            f"robocasa must be installed from pinned commit {ROBOCASA_COMMIT}, "
            f"got {robocasa_source}"
        )
    return mujoco, robocasa, robosuite


def _env_metadata(dataset: Path) -> dict[str, Any]:
    payload = json.loads((dataset / "extras" / "dataset_meta.json").read_text())
    expected = {
        "robocasa_version": ROBOCASA_DATASET_VERSION,
        "robosuite_version": ROBOSUITE_VERSION,
        "mujoco_version": MUJOCO_VERSION,
    }
    actual = {key: str(payload.get(key)) for key in expected}
    if actual != expected:
        raise RuntimeError(f"dataset simulator provenance mismatch: {actual} != {expected}")
    return dict(payload["env_args"])


def load_task_texts(dataset: Path) -> dict[int, str]:
    """Load the LeRobot task table from either supported metadata layout."""

    meta = dataset / "meta"
    parquet = meta / "tasks.parquet"
    jsonl = meta / "tasks.jsonl"
    if parquet.is_file():
        task_frame = pd.read_parquet(parquet, columns=["task_index", "task"])
        rows = task_frame.to_dict(orient="records")
    elif jsonl.is_file():
        rows = [
            json.loads(line)
            for line in jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise FileNotFoundError(f"missing task metadata: {parquet} or {jsonl}")
    task_texts = {int(row["task_index"]): str(row["task"]) for row in rows}
    if not task_texts or any(not text.strip() for text in task_texts.values()):
        raise ValueError("task metadata is empty or contains blank task text")
    return task_texts


def _reorder_lerobot_action(actions: np.ndarray, dataset: Path) -> np.ndarray:
    modality = json.loads((dataset / "meta" / "modality.json").read_text())
    action_info = modality["action"]
    result = np.zeros_like(actions)
    for key in sorted(action_info, key=lambda name: action_info[name]["start"]):
        source_start = int(action_info[key]["start"])
        source_stop = int(action_info[key]["end"])
        target_start, target_stop = ACTION_KEY_ORDERING_HDF5[key]
        result[:, target_start:target_stop] = actions[:, source_start:source_stop]
    return result


def _official_reset_to(env, episode: dict[str, Any]) -> None:
    """Exact RoboCasa v1.0 playback reset, kept local to avoid LeRobot deps."""
    ep_meta = episode["ep_meta"]
    if hasattr(env, "set_attrs_from_ep_meta"):
        env.set_attrs_from_ep_meta(ep_meta)
    elif hasattr(env, "set_ep_meta"):
        env.set_ep_meta(ep_meta)
    env.reset()
    xml = env.edit_model_xml(episode["model_xml"])
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(episode["states"][0])
    env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _make_env(dataset: Path, *, render_rgb: bool):
    _mujoco, _robocasa, robosuite = _runtime_modules()
    env_meta = _env_metadata(dataset)
    env_kwargs = dict(env_meta["env_kwargs"])
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = bool(render_rgb)
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs), env_meta


def _episode(dataset: Path, episode_id: int) -> dict[str, Any]:
    episode_root = dataset / "extras" / f"episode_{episode_id:06d}"
    with np.load(episode_root / "states.npz", allow_pickle=False) as archive:
        states = np.asarray(archive["states"])
    parquet = next(iter((dataset / "data").glob(f"*/episode_{episode_id:06d}.parquet")), None)
    if parquet is None:
        raise FileNotFoundError(f"episode parquet not found: {episode_id}")
    frame = pd.read_parquet(
        parquet, columns=["action", "next.reward", "next.done", "task_index"]
    )
    lerobot_actions = np.stack(frame["action"].to_list()).astype(np.float64)
    actions = _reorder_lerobot_action(lerobot_actions, dataset)
    recorded_rewards = frame["next.reward"].to_numpy(dtype=np.float32)
    recorded_dones = frame["next.done"].to_numpy(dtype=np.bool_)
    task_indices = frame["task_index"].to_numpy(dtype=np.int64)
    if len(np.unique(task_indices)) != 1:
        raise ValueError(f"episode {episode_id}: task_index changes within episode")
    with gzip.open(episode_root / "model.xml.gz", "rt", encoding="utf-8") as handle:
        model_xml = handle.read()
    ep_meta = json.loads((episode_root / "ep_meta.json").read_text())
    if (
        states.ndim != 2
        or actions.ndim != 2
        or len(states) != len(actions)
        or len(recorded_rewards) != len(actions)
        or len(recorded_dones) != len(actions)
    ):
        raise ValueError(
            f"episode {episode_id}: states/actions mismatch {states.shape} vs {actions.shape}"
        )
    if actions.shape[1] != 12 or not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError(f"episode {episode_id}: invalid state/action payload")
    return {
        "states": states,
        "actions": actions,
        "recorded_rewards": recorded_rewards,
        "recorded_dones": recorded_dones,
        "task_index": int(task_indices[0]),
        "model_xml": model_xml,
        "ep_meta": ep_meta,
    }


def _reset_episode(env, episode: dict[str, Any]) -> np.ndarray:
    _official_reset_to(env, episode)
    actual = np.asarray(env.sim.get_state().flatten()).copy()
    if not array_bytes_equal(actual, episode["states"][0]):
        raise ReplayDivergence("initial simulator state is not byte-identical to recorded state[0]")
    return actual


def _historical_outcome_matches(
    env, episode: dict[str, Any], step: int, reward: float, done: bool
) -> tuple[bool, bool, bool]:
    recorded_reward = np.float32(episode["recorded_rewards"][step])
    actual_reward = np.float32(reward)
    recorded_done = bool(episode["recorded_dones"][step])
    success = bool(env._check_success())
    return (
        actual_reward.tobytes() == recorded_reward.tobytes(),
        bool(done) == recorded_done,
        success == bool(recorded_reward > 0.0),
    )


def _historical_state_error(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    delta = np.asarray(actual) - np.asarray(expected)
    return float(np.linalg.norm(delta)), float(np.max(np.abs(delta)))


def _collection_warmup(env, episode: dict[str, Any]) -> None:
    """Reproduce the unrecorded all-zero step present in RoboCasa365 demos."""
    env.step(np.zeros_like(episode["actions"][0]))


def qualify_full_factual_episode(env, episode: dict[str, Any]) -> dict[str, Any]:
    """Audit the current pinned runtime against every historical transition."""
    _reset_episode(env, episode)
    _collection_warmup(env, episode)
    l2_errors: list[float] = []
    linf_errors: list[float] = []
    reward_mismatches = 0
    done_mismatches = 0
    success_mismatches = 0
    current_success_steps: list[int] = []
    for step, action in enumerate(episode["actions"]):
        _obs, reward, done, _info = env.step(action)
        if step + 1 < len(episode["states"]):
            actual = np.asarray(env.sim.get_state().flatten()).copy()
            expected = episode["states"][step + 1]
            l2, linf = _historical_state_error(actual, expected)
            l2_errors.append(l2)
            linf_errors.append(linf)
        reward_ok, done_ok, success_ok = _historical_outcome_matches(
            env, episode, step, reward, done
        )
        if bool(env._check_success()):
            current_success_steps.append(step)
        reward_mismatches += int(not reward_ok)
        done_mismatches += int(not done_ok)
        success_mismatches += int(not success_ok)
    return {
        "compared_state_steps": len(l2_errors),
        "state_l2_p50": float(np.quantile(l2_errors, 0.5)),
        "state_l2_p90": float(np.quantile(l2_errors, 0.9)),
        "state_l2_max": float(max(l2_errors)),
        "state_linf_max": float(max(linf_errors)),
        "reward_mismatches": reward_mismatches,
        "done_mismatches": done_mismatches,
        "success_mismatches": success_mismatches,
        "current_success_steps": current_success_steps,
        "historical_runtime_exact": bool(
            not any(l2_errors)
            and reward_mismatches == 0
            and done_mismatches == 0
            and success_mismatches == 0
        ),
    }


def replay_prefix_to_root(
    env, episode: dict[str, Any], t0: int
) -> tuple[np.ndarray, dict[str, float]]:
    _reset_episode(env, episode)
    _collection_warmup(env, episode)
    l2_errors: list[float] = []
    linf_errors: list[float] = []
    for step in range(t0):
        env.step(episode["actions"][step])
        actual = np.asarray(env.sim.get_state().flatten()).copy()
        expected = episode["states"][step + 1]
        l2, linf = _historical_state_error(actual, expected)
        l2_errors.append(l2)
        linf_errors.append(linf)
    return np.asarray(env.sim.get_state().flatten()).copy(), {
        "root_l2": l2_errors[-1] if l2_errors else 0.0,
        "prefix_l2_max": max(l2_errors, default=0.0),
        "prefix_linf_max": max(linf_errors, default=0.0),
    }


def _render_views(env, height: int, width: int) -> np.ndarray:
    frames = [
        env.sim.render(height=height, width=width, camera_name=camera)[::-1].copy()
        for camera in CAMERAS
    ]
    return np.stack(frames).astype(np.uint8, copy=False)


def _roll_branch(
    env,
    episode: dict[str, Any],
    t0: int,
    actions: np.ndarray,
    *,
    factual: bool,
    render_rgb: bool,
    rgb_stride: int,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    root_state, prefix_audit = replay_prefix_to_root(env, episode, t0)
    root_rgb = _render_views(env, height, width) if render_rgb else np.zeros((0,), dtype=np.uint8)
    states = [root_state]
    rgb = [root_rgb] if render_rgb else []
    rewards, dones, successes = [], [], []
    historical_l2 = []
    historical_linf = []
    for offset, action in enumerate(actions):
        _obs, reward, done, _info = env.step(action)
        state = np.asarray(env.sim.get_state().flatten()).copy()
        if factual:
            expected = episode["states"][t0 + offset + 1]
            l2, linf = _historical_state_error(state, expected)
            historical_l2.append(l2)
            historical_linf.append(linf)
        states.append(state)
        rewards.append(float(reward))
        dones.append(bool(done))
        successes.append(bool(env._check_success()))
        if render_rgb and (offset + 1) % rgb_stride == 0:
            rgb.append(_render_views(env, height, width))
    return {
        "root_state": root_state,
        "root_rgb": root_rgb,
        "states": np.stack(states),
        "rgb": np.stack(rgb) if rgb else np.zeros((0,), dtype=np.uint8),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "success": np.asarray(successes, dtype=np.bool_),
        "prefix_audit": prefix_audit,
        "historical_factual_l2": np.asarray(historical_l2, dtype=np.float64),
        "historical_factual_linf": np.asarray(historical_linf, dtype=np.float64),
    }


def _atomic_savez(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _root_bundle(
    env,
    episode: dict[str, Any],
    episode_id: int,
    t0: int,
    *,
    task_name: str,
    horizon: int,
    perturb_steps: int,
    delta_fraction: float,
    render_rgb: bool,
    rgb_stride: int,
    height: int,
    width: int,
    max_root_historical_l2: float,
    runtime_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # RoboCasa365's compatibility environment defers model/controller creation.
    # Perform the same full episode reset used by every branch before querying
    # controller-owned action limits.  This is not a direct mid-state shortcut.
    _reset_episode(env, episode)
    low, high = (np.asarray(x, dtype=np.float64) for x in env.action_spec)
    factual = episode["actions"][t0 : t0 + horizon]
    chunks = make_k4_action_chunks(
        factual,
        low,
        high,
        perturb_steps=perturb_steps,
        delta_fraction=delta_fraction,
    )
    if any(np.array_equal(chunks[index], chunks[0]) for index in range(1, 4)):
        raise RuntimeError("counterfactual transform produced a factual duplicate")

    branches = [
        _roll_branch(
            env,
            episode,
            t0,
            chunk,
            factual=index == 0,
            render_rgb=render_rgb,
            rgb_stride=rgb_stride,
            height=height,
            width=width,
        )
        for index, chunk in enumerate(chunks)
    ]
    root_state_hashes = [sha256_array(branch["root_state"]) for branch in branches]
    if len(set(root_state_hashes)) != 1:
        raise ReplayDivergence("K=4 root simulator states are not identical")
    root_rgb_hashes = [sha256_array(branch["root_rgb"]) for branch in branches]
    if render_rgb and len(set(root_rgb_hashes)) != 1:
        raise ReplayDivergence("K=4 root RGB observations are not identical")
    historical_root_l2 = float(branches[0]["prefix_audit"]["root_l2"])
    if historical_root_l2 > max_root_historical_l2:
        raise ReplayDivergence(
            f"historical root drift l2={historical_root_l2:.9g} exceeds "
            f"{max_root_historical_l2:.9g}"
        )

    state_divergence, rgb_divergence = [0.0], [0]
    factual_endpoint = branches[0]["states"][-1]
    factual_rgb = branches[0]["rgb"][-1] if render_rgb else None
    for branch in branches[1:]:
        state_delta = float(np.linalg.norm(branch["states"][-1] - factual_endpoint))
        rgb_delta = int(np.count_nonzero(branch["rgb"][-1] != factual_rgb)) if render_rgb else -1
        if state_delta == 0.0 or (render_rgb and rgb_delta == 0):
            raise RuntimeError("ineffective counterfactual branch")
        state_divergence.append(state_delta)
        rgb_divergence.append(rgb_delta)

    ep_meta_canonical = json.dumps(episode["ep_meta"], sort_keys=True, separators=(",", ":"))
    branch_actions_sha256 = sha256_array(chunks)
    # The training example is a same-root branch bundle, not merely a state.
    # Include the exact candidate actions so deliberately different perturbation
    # magnitudes at the same episode/t0 remain distinct, auditable examples.
    root_id = sha256_bytes(
        (
            f"robocasa365_same_root_v3:{task_name}:{episode_id}:{t0}:"
            f"{horizon}:{branch_actions_sha256}"
        ).encode("ascii")
    )
    metadata = {
        "schema": "wm3d_v7_robocasa_same_root_k4_v1",
        "root_id": root_id,
        "episode_id": episode_id,
        "t0": t0,
        "horizon": horizon,
        "k": 4,
        "branch_names": list(BRANCH_NAMES),
        "branch_actions_sha256": branch_actions_sha256,
        "root_state_sha256": root_state_hashes[0],
        "root_rgb_sha256": root_rgb_hashes[0] if render_rgb else None,
        "model_xml_sha256": sha256_bytes(episode["model_xml"].encode("utf-8")),
        "ep_meta_sha256": sha256_bytes(ep_meta_canonical.encode("utf-8")),
        "exact_factual_replay": False,
        "current_runtime_factual_rollout": True,
        "same_root_current_runtime_exact": True,
        "historical_runtime_reconstruction_exact": False,
        "historical_root_l2": historical_root_l2,
        "historical_prefix_l2_max": float(branches[0]["prefix_audit"]["prefix_l2_max"]),
        "historical_factual_horizon_l2_max": float(
            branches[0]["historical_factual_l2"].max()
        ),
        "historical_full_episode_audit": runtime_audit,
        "outcome_source": "current_pinned_robocasa_simulator",
        "state_endpoint_l2_vs_factual": state_divergence,
        "rgb_endpoint_changed_values_vs_factual": rgb_divergence,
        "branch_transform": {
            "delta_fraction_of_action_range": delta_fraction,
            "perturb_steps": perturb_steps,
            "hold_dimensions": [0, 1, 2, 3, 4, 5],
        },
    }
    arrays = {
        "actions": chunks.astype(np.float32),
        "states": np.stack([branch["states"] for branch in branches]),
        "rewards": np.stack([branch["rewards"] for branch in branches]),
        "dones": np.stack([branch["dones"] for branch in branches]),
        "success": np.stack([branch["success"] for branch in branches]),
        "root_rgb": branches[0]["root_rgb"],
        "branch_rgb": np.stack([branch["rgb"] for branch in branches]),
    }
    return metadata, arrays


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-tar", type=Path, required=True)
    parser.add_argument("--expected-source-tar-sha256", required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument(
        "--required-split",
        choices=("all", "train", "val", "test"),
        default="all",
    )
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--perturb-steps", type=int, default=4)
    parser.add_argument("--minimum-prefix", type=int, default=16)
    parser.add_argument("--motion-threshold", type=float, default=1e-3)
    parser.add_argument("--max-candidates-per-episode", type=int, default=12)
    parser.add_argument("--candidate-minimum-gap", type=int, default=4)
    parser.add_argument("--max-roots-per-episode", type=int, default=1)
    parser.add_argument(
        "--accepted-root-minimum-gap",
        type=int,
        default=0,
        help="minimum t0 distance between accepted roots; 0 uses half the horizon",
    )
    parser.add_argument("--delta-fraction", type=float, default=0.075)
    parser.add_argument("--max-root-historical-l2", type=float, default=0.05)
    parser.add_argument("--rgb-stride", type=int, default=4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument(
        "--require-mixed-outcomes",
        action="store_true",
        help="accept only roots with both successful and unsuccessful true K branches",
    )
    args = parser.parse_args()

    if args.horizon <= 0 or args.horizon % args.rgb_stride:
        raise SystemExit("horizon must be positive and divisible by rgb-stride")
    if args.max_roots_per_episode <= 0:
        raise SystemExit("max-roots-per-episode must be positive")
    accepted_root_minimum_gap = int(args.accepted_root_minimum_gap)
    if accepted_root_minimum_gap <= 0:
        accepted_root_minimum_gap = max(1, args.horizon // 2)
    actual_tar_sha = sha256_file(args.source_tar)
    if actual_tar_sha != str(args.expected_source_tar_sha256):
        raise SystemExit(f"source tar SHA256 mismatch: {actual_tar_sha}")
    task_name = str(args.task_name or args.dataset.parents[1].name)
    _runtime_modules()
    env, env_meta = _make_env(args.dataset, render_rgb=not args.no_rgb)
    task_texts = load_task_texts(args.dataset)
    episode_dirs = sorted((args.dataset / "extras").glob("episode_*"))
    episode_ids = [int(path.name.rsplit("_", 1)[1]) for path in episode_dirs]
    episode_ids = [ep for ep in episode_ids if ep >= args.start_episode]
    if args.max_episodes > 0:
        episode_ids = episode_ids[: args.max_episodes]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for episode_id in episode_ids:
            try:
                episode = _episode(args.dataset, episode_id)
                ep_meta = episode["ep_meta"]
                split_group = root_split_group(task_name, ep_meta, episode_id)
                episode_split = stable_split(split_group)
                if args.required_split != "all" and episode_split != args.required_split:
                    continue
                runtime_audit = qualify_full_factual_episode(env, episode)
                candidates = select_arm_root_candidates(
                    episode["actions"],
                    len(episode["states"]),
                    horizon=args.horizon,
                    perturb_steps=args.perturb_steps,
                    minimum_prefix=args.minimum_prefix,
                    motion_threshold=args.motion_threshold,
                    priority_steps=runtime_audit.get("current_success_steps", []),
                )
                if not candidates:
                    raise RuntimeError("no eligible arm-control root")
                eligible_candidate_count = len(candidates)
                candidates = limit_diverse_candidates(
                    candidates,
                    max_candidates=args.max_candidates_per_episode,
                    minimum_gap=args.candidate_minimum_gap,
                )
                print(
                    json.dumps(
                        {
                            "episode": episode_id,
                            "eligible_roots": eligible_candidate_count,
                            "tested_roots": len(candidates),
                            "status": "candidate_scan",
                        }
                    ),
                    flush=True,
                )
                accepted_bundles = []
                accepted_t0s: list[int] = []
                candidate_errors = []
                for t0 in candidates:
                    if any(
                        abs(t0 - previous_t0) < accepted_root_minimum_gap
                        for previous_t0 in accepted_t0s
                    ):
                        continue
                    try:
                        accepted = _root_bundle(
                            env,
                            episode,
                            episode_id,
                            t0,
                            task_name=task_name,
                            horizon=args.horizon,
                            perturb_steps=args.perturb_steps,
                            delta_fraction=args.delta_fraction,
                            render_rgb=not args.no_rgb,
                            rgb_stride=args.rgb_stride,
                            height=args.height,
                            width=args.width,
                            max_root_historical_l2=args.max_root_historical_l2,
                            runtime_audit=runtime_audit,
                        )
                        terminal_success = accepted[1]["success"].any(axis=1)
                        mixed_outcomes = bool(
                            terminal_success.any() and not terminal_success.all()
                        )
                        if args.require_mixed_outcomes and not mixed_outcomes:
                            raise RuntimeError(
                                "root lacks mixed true terminal outcomes across K branches"
                            )
                        accepted[0]["terminal_success_per_branch"] = terminal_success.tolist()
                        accepted[0]["mixed_terminal_outcomes"] = mixed_outcomes
                        accepted_bundles.append(accepted)
                        accepted_t0s.append(t0)
                        if len(accepted_bundles) >= args.max_roots_per_episode:
                            break
                    except (ReplayDivergence, RuntimeError) as exc:
                        candidate_errors.append(f"t0={t0}:{exc}")
                if not accepted_bundles:
                    raise RuntimeError("; ".join(candidate_errors[:8]))
                scene = f"layout={ep_meta.get('layout_id', 'na')}/style={ep_meta.get('style_id', 'na')}"
                for root_index, (metadata, arrays) in enumerate(accepted_bundles):
                    metadata["task_text"] = task_texts.get(
                        int(episode["task_index"]), task_name
                    )
                    metadata["task_index"] = int(episode["task_index"])
                    metadata["episode_root_index"] = int(root_index)
                    metadata["accepted_root_minimum_gap"] = accepted_root_minimum_gap
                    destination = (
                        args.output_root
                        / stable_split(split_group)
                        / f"{metadata['root_id']}.npz"
                    )
                    _atomic_savez(
                        destination,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                        **arrays,
                    )
                    rows.append(
                        {
                            **metadata,
                            "path": str(destination.resolve()),
                            "split": stable_split(split_group),
                            "split_group": split_group,
                            "task": task_name,
                            "scene": scene,
                            "true_simulator_rollout": True,
                            "same_root_current_runtime_exact": True,
                            "historical_runtime_reconstruction_exact": False,
                            "outcome_source": "current_pinned_robocasa_simulator",
                            "pseudo_outcomes": False,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "episode": episode_id,
                                "root_id": metadata["root_id"],
                                "root_index": root_index,
                                "accepted_roots_in_episode": len(accepted_bundles),
                                "status": "passed",
                            }
                        ),
                        flush=True,
                    )
            except Exception as exc:  # record every rejected episode, then continue
                failures.append({"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"})
                print(json.dumps({"episode": episode_id, "status": "rejected", "error": str(exc)}), flush=True)
    finally:
        env.close()

    _write_jsonl(args.manifest, rows)
    report = {
        "schema": "wm3d_v7_robocasa_same_root_audit_v1",
        "passed": bool(rows),
        "accepted_roots": len(rows),
        "accepted_episodes": len({int(row["episode_id"]) for row in rows}),
        "rejected_episodes": len(failures),
        "failures": failures,
        "dataset": str(args.dataset.resolve()),
        "source_repo": "nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos",
        "source_revision": SOURCE_REVISION,
        "source_tar_sha256": actual_tar_sha,
        "task_name": task_name,
        "robocasa_commit": ROBOCASA_COMMIT,
        "robocasa_dataset_version": ROBOCASA_DATASET_VERSION,
        "robosuite_version": ROBOSUITE_VERSION,
        "robosuite_commit": ROBOSUITE_COMMIT,
        "mujoco_version": MUJOCO_VERSION,
        "environment_metadata": env_meta,
        "camera_names": list(CAMERAS),
        "resolution": [args.height, args.width],
        "simulator_hz": 20,
        "rgb_stride": args.rgb_stride,
        "action_format": "simulator_hdf5_order_eef6_grip_base4_mode",
        "same_root_state_comparison": "dtype_shape_and_contiguous_payload_bytes",
        "historical_full_episode_audit": True,
        "historical_runtime_reconstruction_exact": False,
        "collection_warmup": "one_unrecorded_all_zero_action",
        "max_root_historical_l2": args.max_root_historical_l2,
        "horizon": args.horizon,
        "k": 4,
        "max_candidates_per_episode": args.max_candidates_per_episode,
        "start_episode": args.start_episode,
        "required_split": args.required_split,
        "candidate_minimum_gap": args.candidate_minimum_gap,
        "max_roots_per_episode": args.max_roots_per_episode,
        "accepted_root_minimum_gap": accepted_root_minimum_gap,
        "require_mixed_outcomes": bool(args.require_mixed_outcomes),
        "split_unit": "task_scene_episode_root",
        "mixed_terminal_roots": sum(bool(row.get("mixed_terminal_outcomes")) for row in rows),
        "positive_terminal_branches": sum(
            sum(bool(value) for value in row.get("terminal_success_per_branch", []))
            for row in rows
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted_roots": len(rows), "rejected_episodes": len(failures)}, sort_keys=True))
    if not rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
