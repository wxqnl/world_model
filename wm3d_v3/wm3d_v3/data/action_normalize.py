"""Unify per-dataset action formats to a canonical 7-D vector.

Canonical action: [dx, dy, dz, drx, dry, drz, gripper_close01]
   - dx/dy/dz: delta xyz in robot/world frame (meters, range ~[-0.05, 0.05])
   - drx/dry/drz: delta rotation (axis-angle radians, range ~[-0.3, 0.3])
   - gripper_close01: 0 = open, 1 = closed

Per dataset:
   bridge: action dict {world_vector (3,), rotation_delta (3,), open_gripper (bool),
                        terminate_episode (float)}
           gripper_close01 = 1 - open_gripper
   fractal20220817: same as bridge (RT-1 schema)
   taco_play: action vector (7,)  [dx,dy,dz,drx,dry,drz,grip] where grip in [-1,1]
   jaco_play: action vector (7,) similar
   kuka: RT-1-style world_vector + rotation_delta + gripper_closedness_action
"""
from __future__ import annotations
import numpy as np


def normalize_action(action, dataset: str) -> np.ndarray:
    if dataset in ("bridge", "fractal20220817_data", "fractal20220817"):
        return _normalize_bridge_rt1(action)
    if dataset == "taco_play":
        return _normalize_taco(action)
    if dataset == "jaco_play":
        return _normalize_jaco(action)
    if dataset == "kuka":
        return _normalize_kuka(action)
    raise ValueError(f"unknown dataset {dataset}")


def _normalize_taco(a) -> np.ndarray:
    """taco_play action: dict with rel_actions_world (7-d) -> use directly."""
    if isinstance(a, dict) and "rel_actions_world" in a:
        v = np.asarray(a["rel_actions_world"], dtype=np.float32).reshape(-1)[:7]
        if v.shape[0] == 7:
            grip01 = (v[6] + 1.0) * 0.5 if v[6] >= -1.0 and v[6] <= 1.0 else float(v[6] > 0)
            return np.array([*v[:6], grip01], dtype=np.float32)
    raise TypeError(f"taco action unsupported: {type(a)}")


def _normalize_jaco(a) -> np.ndarray:
    """jaco_play has only world_vector (3) + gripper -> pad rotation with zeros."""
    if isinstance(a, dict):
        wv = np.asarray(a["world_vector"], dtype=np.float32).reshape(3)
        grip = float(np.asarray(a["gripper_closedness_action"]).flat[0]) > 0
        return np.array([wv[0], wv[1], wv[2], 0., 0., 0., float(grip)], dtype=np.float32)
    raise TypeError(f"jaco action unsupported: {type(a)}")


def _normalize_kuka(a) -> np.ndarray:
    """Kuka has world delta, rotation delta, and gripper closedness."""
    if isinstance(a, dict):
        wv = np.asarray(a["world_vector"], dtype=np.float32).reshape(3)
        rd = np.asarray(a["rotation_delta"], dtype=np.float32).reshape(3)
        grip = float(np.asarray(a["gripper_closedness_action"]).flat[0]) > 0
        return np.array(
            [wv[0], wv[1], wv[2], rd[0], rd[1], rd[2], float(grip)], dtype=np.float32)
    raise TypeError(f"kuka action unsupported: {type(a)}")


def _normalize_bridge_rt1(a) -> np.ndarray:
    """RT-1-style action dict."""
    if isinstance(a, np.ndarray):
        # Already a 7-d vector?  Some shards may flatten.
        if a.shape == (7,):
            out = a.astype(np.float32).copy()
            return out
    if isinstance(a, dict):
        wv = np.asarray(a["world_vector"], dtype=np.float32).reshape(3)
        rd = np.asarray(a["rotation_delta"], dtype=np.float32).reshape(3)
        og = a.get("open_gripper", False)
        if isinstance(og, np.ndarray):
            og = bool(og.item())
        grip = 0.0 if bool(og) else 1.0
        return np.array([wv[0], wv[1], wv[2], rd[0], rd[1], rd[2], grip], dtype=np.float32)
    raise TypeError(f"unrecognized action {type(a)}")


def _normalize_dense7(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float32).reshape(-1)
    if arr.shape[0] == 7:
        # Map gripper [-1,1] → [0,1] closed
        grip01 = (arr[6] + 1.0) * 0.5
        return np.array([*arr[:6], grip01], dtype=np.float32)
    raise ValueError(f"unexpected dense7 action shape {arr.shape}")
