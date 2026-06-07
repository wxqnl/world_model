"""Object/contact diagnostics for LIBERO closed-loop rollout traces.

This evaluator is a diagnostic layer, not a replacement for the LIBERO reward.
It uses named object/eef poses recorded by `libero_remote_runner --trace_object_state`
to explain whether a failed rollout reached, contacted, or moved target objects
toward the receptacle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _load_expert_named_poses(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = np.load(path, allow_pickle=False)
    if "named_poses_json" not in data.files:
        return []
    raw = str(np.asarray(data["named_poses_json"]).reshape(-1)[0])
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, list) else []


def _vec(named: dict[str, Any], entity: str, field: str) -> np.ndarray | None:
    item = named.get(entity)
    if not isinstance(item, dict) or field not in item:
        return None
    arr = np.asarray(item[field], dtype=np.float32).reshape(-1)
    return arr if arr.size > 0 else None


def _norm(vec: np.ndarray | None) -> float | None:
    if vec is None:
        return None
    return float(np.linalg.norm(vec.astype(np.float32)))


def _first_at_or_below(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values):
        if value <= threshold:
            return idx + 1
    return None


def _finite(values: list[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and np.isfinite(float(v))]


def _named_trace(episode: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trace in episode.get("step_trace") or []:
        named = trace.get("named_poses")
        if isinstance(named, dict):
            out.append(named)
    return out


def _eef_distance(named: dict[str, Any], obj: str) -> float | None:
    rel = _vec(named, obj, "to_robot0_eef_pos")
    if rel is not None:
        return _norm(rel)
    obj_pos = _vec(named, obj, "pos")
    eef_pos = _vec(named, "robot0", "eef_pos")
    if obj_pos is None or eef_pos is None:
        return None
    return _norm(obj_pos[:3] - eef_pos[:3])


def _receptacle_distance(named: dict[str, Any], obj: str, receptacle: str) -> tuple[float | None, float | None]:
    obj_pos = _vec(named, obj, "pos")
    rec_pos = _vec(named, receptacle, "pos")
    if obj_pos is None or rec_pos is None:
        return None, None
    diff = obj_pos[:3] - rec_pos[:3]
    return _norm(diff), _norm(diff[:2])


def _object_metrics(
    trace: list[dict[str, Any]],
    expert_trace: list[dict[str, Any]],
    *,
    obj: str,
    receptacle: str,
    contact_threshold: float,
    receptacle_xy_threshold: float,
) -> dict[str, Any]:
    eef = _finite([_eef_distance(named, obj) for named in trace])
    rec_xyz_raw: list[float | None] = []
    rec_xy_raw: list[float | None] = []
    for named in trace:
        xyz, xy = _receptacle_distance(named, obj, receptacle)
        rec_xyz_raw.append(xyz)
        rec_xy_raw.append(xy)
    rec_xyz = _finite(rec_xyz_raw)
    rec_xy = _finite(rec_xy_raw)

    exp_eef = _finite([_eef_distance(named, obj) for named in expert_trace])
    exp_rec_xy_raw: list[float | None] = []
    for named in expert_trace:
        _xyz, xy = _receptacle_distance(named, obj, receptacle)
        exp_rec_xy_raw.append(xy)
    exp_rec_xy = _finite(exp_rec_xy_raw)

    min_eef = min(eef) if eef else None
    min_rec_xy = min(rec_xy) if rec_xy else None
    final_rec_xy = rec_xy[-1] if rec_xy else None
    exp_min_eef = min(exp_eef) if exp_eef else None
    exp_min_rec_xy = min(exp_rec_xy) if exp_rec_xy else None
    exp_final_rec_xy = exp_rec_xy[-1] if exp_rec_xy else None

    contact_step = _first_at_or_below(eef, contact_threshold) if eef else None
    receptacle_step = _first_at_or_below(rec_xy, receptacle_xy_threshold) if rec_xy else None

    return {
        "object": obj,
        "receptacle": receptacle,
        "trace_available": bool(eef or rec_xy),
        "contact_threshold": float(contact_threshold),
        "receptacle_xy_threshold": float(receptacle_xy_threshold),
        "min_eef_dist": min_eef,
        "contact_hit": bool(min_eef is not None and min_eef <= contact_threshold),
        "contact_step": contact_step,
        "min_receptacle_xyz_dist": min(rec_xyz) if rec_xyz else None,
        "min_receptacle_xy_dist": min_rec_xy,
        "final_receptacle_xy_dist": final_rec_xy,
        "receptacle_hit": bool(min_rec_xy is not None and min_rec_xy <= receptacle_xy_threshold),
        "receptacle_step": receptacle_step,
        "expert_min_eef_dist": exp_min_eef,
        "expert_min_receptacle_xy_dist": exp_min_rec_xy,
        "expert_final_receptacle_xy_dist": exp_final_rec_xy,
        "min_eef_dist_gap_vs_expert": None if min_eef is None or exp_min_eef is None else min_eef - exp_min_eef,
        "min_receptacle_xy_gap_vs_expert": None
        if min_rec_xy is None or exp_min_rec_xy is None
        else min_rec_xy - exp_min_rec_xy,
        "final_receptacle_xy_gap_vs_expert": None
        if final_rec_xy is None or exp_final_rec_xy is None
        else final_rec_xy - exp_final_rec_xy,
    }


def _episode_metrics(
    episode: dict[str, Any],
    *,
    expert_trace: list[dict[str, Any]],
    target_objects: list[str],
    receptacle: str,
    contact_threshold: float,
    receptacle_xy_threshold: float,
) -> dict[str, Any]:
    trace = _named_trace(episode)
    objects = {
        obj: _object_metrics(
            trace,
            expert_trace,
            obj=obj,
            receptacle=receptacle,
            contact_threshold=contact_threshold,
            receptacle_xy_threshold=receptacle_xy_threshold,
        )
        for obj in target_objects
    }
    n = max(1, len(target_objects))
    contact_count = sum(int(m["contact_hit"]) for m in objects.values())
    receptacle_count = sum(int(m["receptacle_hit"]) for m in objects.values())
    success = bool(episode.get("success"))
    stage_score = (contact_count + receptacle_count) / float(2 * n)
    if success:
        stage_score = 1.0
    if not trace:
        diagnosis = "missing_named_pose_trace"
    elif success:
        diagnosis = "libero_success"
    elif contact_count == 0:
        diagnosis = "no_target_contact"
    elif receptacle_count == 0:
        diagnosis = "contact_without_receptacle_progress"
    elif receptacle_count < n:
        diagnosis = "partial_receptacle_progress"
    else:
        diagnosis = "proxy_success_but_reward_missing"
    return {
        "task_id": episode.get("task_id"),
        "task_name": episode.get("task_name"),
        "instruction": episode.get("instruction"),
        "init_state_id": episode.get("init_state_id"),
        "success": success,
        "steps": int(episode.get("steps", len(trace))),
        "named_pose_steps": len(trace),
        "target_objects": target_objects,
        "receptacle": receptacle,
        "contact_objects_hit": contact_count,
        "receptacle_objects_hit": receptacle_count,
        "stage_score": stage_score,
        "diagnosis": diagnosis,
        "objects": objects,
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate(
    rollout: dict[str, Any],
    *,
    expert_trace: list[dict[str, Any]],
    target_objects: list[str],
    receptacle: str,
    contact_threshold: float,
    receptacle_xy_threshold: float,
) -> dict[str, Any]:
    episodes = [
        _episode_metrics(
            episode,
            expert_trace=expert_trace,
            target_objects=target_objects,
            receptacle=receptacle,
            contact_threshold=contact_threshold,
            receptacle_xy_threshold=receptacle_xy_threshold,
        )
        for episode in rollout.get("results") or []
    ]
    stage_scores = [float(ep["stage_score"]) for ep in episodes]
    successes = [float(ep["success"]) for ep in episodes]
    return {
        "trace_schema_version": rollout.get("trace_schema_version"),
        "rollout_success_rate": rollout.get("success_rate"),
        "episodes": len(episodes),
        "success_rate": _mean(successes),
        "stage_score_mean": _mean(stage_scores),
        "target_objects": target_objects,
        "receptacle": receptacle,
        "contact_threshold": float(contact_threshold),
        "receptacle_xy_threshold": float(receptacle_xy_threshold),
        "expert_named_pose_steps": len(expert_trace),
        "episode_metrics": episodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout_json", type=Path, required=True)
    ap.add_argument("--expert_ref_npz", type=Path, default=None)
    ap.add_argument("--target_objects", default="cream_cheese_1,butter_1")
    ap.add_argument("--receptacle", default="basket_1")
    ap.add_argument("--contact_threshold", type=float, default=0.08)
    ap.add_argument("--receptacle_xy_threshold", type=float, default=0.14)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rollout = _load_json(args.rollout_json)
    expert_trace = _load_expert_named_poses(args.expert_ref_npz)
    target_objects = [x.strip() for x in args.target_objects.split(",") if x.strip()]
    if not target_objects:
        raise ValueError("--target_objects cannot be empty")
    report = evaluate(
        rollout,
        expert_trace=expert_trace,
        target_objects=target_objects,
        receptacle=args.receptacle,
        contact_threshold=args.contact_threshold,
        receptacle_xy_threshold=args.receptacle_xy_threshold,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"out": str(args.out), "stage_score_mean": report["stage_score_mean"]}, sort_keys=True))


if __name__ == "__main__":
    main()
