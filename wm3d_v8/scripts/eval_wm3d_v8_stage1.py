#!/usr/bin/env python3
"""Evaluate V8 Stage1 on sealed real branches and optional native imagination."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from wm3d_v3.stage1_planner.candidates import deterministic_action_cost  # noqa: E402
from wm3d_v3.stage1_planner.dataset import Stage1BranchDataset  # noqa: E402
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig  # noqa: E402
from wm3d_v3.stage1_planner.train import (  # noqa: E402
    OVERLAY_SCHEMA,
    _dataset_config,
    _load_planner_state,
    sha256_file,
)
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig  # noqa: E402
from wm3d_v3.training.train import build_model, config_sha256, load_train_config  # noqa: E402
from wm3d_v3.training.train import module_state_sha256  # noqa: E402


SCHEMA = "wm3d_v8_stage1_native_planner_eval_v1"


def _move(value, device: torch.device):
    return value.to(device=device, non_blocking=True) if torch.is_tensor(value) else value


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    rank_sum = ranks[labels].sum()
    return float(
        (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    )


def _selection(scores: np.ndarray, success: np.ndarray) -> dict[str, float | int]:
    selectable = scores[:, 1:]
    selected = 1 + selectable.argmax(axis=1)
    rows = np.arange(len(scores))
    selected_success = success[rows, selected].astype(np.float64)
    direct_success = success[:, 1].astype(np.float64)
    oracle_success = success[:, 1:].max(axis=1).astype(np.float64)
    ranks = []
    reciprocal = []
    for score, labels in zip(selectable, success[:, 1:]):
        order = np.argsort(-score, kind="mergesort")
        successful = np.flatnonzero(labels[order])
        rank = int(successful[0]) + 1 if len(successful) else len(order) + 1
        ranks.append(rank)
        reciprocal.append(1.0 / rank if rank <= len(order) else 0.0)
    return {
        "success_at1": float(selected_success.mean()),
        "direct_success": float(direct_success.mean()),
        "success_at1_uplift": float((selected_success - direct_success).mean()),
        "oracle_success": float(oracle_success.mean()),
        "mean_first_success_rank": float(np.mean(ranks)),
        "mean_reciprocal_success_rank": float(np.mean(reciprocal)),
        "roots": int(len(scores)),
    }


def _planner_outputs(
    system: NativePlanningSystem,
    batch: dict,
    *,
    imagined: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float] | None]:
    physical = batch["branch_actions_physical"]
    cost = deterministic_action_cost(physical)
    if not imagined:
        return (
            system.score_true_futures(
                batch["branch_s_tgt_codec"],
                batch["c"],
                depth=batch["branch_depth_tgt"],
                point=batch["branch_point_tgt"],
                pose=batch["branch_pose_geom_tgt"],
                action_cost=cost,
            ),
            None,
        )
    rollout = system.imagine(
        batch["s_in"],
        batch["c"],
        batch["candidate_actions"],
        wrist=batch["s_wrist"],
        view_mask=batch["view_mask"],
    )
    outputs = system.score_rollout(rollout, batch["c"], cost)
    truth = system.world.decode_input_tokens(
        batch["branch_s_tgt_codec"].flatten(0, 1)
    ).unflatten(0, batch["branch_s_tgt_codec"].shape[:2])
    token_mse = float((rollout.tokens.float() - truth.float()).square().mean().cpu())
    token_cosine = float(
        F.cosine_similarity(
            rollout.tokens.float().flatten(start_dim=3),
            truth.float().flatten(start_dim=3),
            dim=-1,
        ).mean().cpu()
    )
    return outputs, {"token_mse": token_mse, "token_cosine": token_cosine}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--overlay-sha256")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--mode", choices=("true", "imagined", "both"), default="true")
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_train_config(args.cfg)
    stage_cfg = dict(cfg["planner_stage"])
    data_cfg = dict(cfg["planner_data"])
    seed = int(stage_cfg["seed"])
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    source_path = Path(stage_cfg["source_checkpoint"])
    source_sha = sha256_file(source_path)
    if source_sha != stage_cfg["source_checkpoint_sha256"]:
        raise SystemExit("Stage0 checkpoint SHA256 mismatch")
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    world = build_model(cfg)
    loaded = world.load_state_dict(source["model"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("Stage0 strict load was not clean")
    source_contract = source["action_policy_contract"]
    del source
    system = NativePlanningSystem(
        world,
        Stage1SystemConfig(
            planner=NativePlannerConfig(**dict(stage_cfg.get("planner_model") or {})),
            candidate_microbatch=int(stage_cfg.get("candidate_microbatch", 1)),
            activation_checkpointing=False,
            detach_between_chunks=True,
        ),
    )
    overlay_sha = None
    overlay_step = 0
    if args.overlay is not None:
        if not args.overlay_sha256:
            raise SystemExit("--overlay-sha256 is required with --overlay")
        overlay_sha = sha256_file(args.overlay)
        if overlay_sha != args.overlay_sha256:
            raise SystemExit("Stage1 overlay SHA256 mismatch")
        payload = torch.load(args.overlay, map_location="cpu", weights_only=False)
        if payload.get("schema") != OVERLAY_SCHEMA:
            raise RuntimeError("Stage1 overlay schema mismatch")
        if payload.get("source_checkpoint_sha256") != source_sha:
            raise RuntimeError("Stage1 overlay is bound to another Stage0 checkpoint")
        _load_planner_state(system, payload["stage1_state"])
        overlay_step = int(payload["step"])
    planner_hash = module_state_sha256(system.planner)

    device = torch.device(args.device)
    system.to(device).eval()
    dataset = Stage1BranchDataset(_dataset_config(data_cfg, args.split))
    if args.max_roots > 0:
        dataset.records = dataset.records[: args.max_roots]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    requested_modes = ("true", "imagined") if args.mode == "both" else (args.mode,)
    mode_scores = {mode: [] for mode in requested_modes}
    mode_logits = {mode: [] for mode in requested_modes}
    successes = []
    imagined_fidelity = []
    roots = []
    with torch.inference_mode():
        for raw in loader:
            batch = {key: _move(value, device) for key, value in raw.items()}
            success = batch["branch_success"].any(dim=-1)[0].cpu().numpy().astype(bool)
            successes.append(success)
            roots.append(str(raw["root_id"][0]))
            for mode in requested_modes:
                outputs, fidelity = _planner_outputs(
                    system, batch, imagined=mode == "imagined"
                )
                mode_scores[mode].append(outputs["score"][0].float().cpu().numpy())
                mode_logits[mode].append(
                    outputs["success_logit"][0].float().cpu().numpy()
                )
                if fidelity is not None:
                    imagined_fidelity.append(fidelity)

    success_array = np.stack(successes)
    report_modes = {}
    for mode in requested_modes:
        score = np.stack(mode_scores[mode])
        logits = np.stack(mode_logits[mode])
        report_modes[mode] = {
            "success_auc_selectable": _auc(
                success_array[:, 1:].reshape(-1), logits[:, 1:].reshape(-1)
            ),
            "serving_score_auc_selectable": _auc(
                success_array[:, 1:].reshape(-1), score[:, 1:].reshape(-1)
            ),
            **_selection(score, success_array),
        }
    report = {
        "schema": SCHEMA,
        "passed_execution": True,
        "config": str(args.cfg.resolve()),
        "config_sha256": config_sha256(cfg),
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_sha,
        "source_checkpoint_step": int(stage_cfg["source_checkpoint_step"]),
        "source_action_contract_sha256": source_contract["contract_sha256"],
        "overlay": str(args.overlay.resolve()) if args.overlay else None,
        "overlay_sha256": overlay_sha,
        "overlay_step": overlay_step,
        "planner_hash": planner_hash,
        "split": args.split,
        "roots": roots,
        "modes": report_modes,
        "imagined_fidelity": (
            {
                key: float(np.mean([item[key] for item in imagined_fidelity]))
                for key in ("token_mse", "token_cosine")
            }
            if imagined_fidelity
            else None
        ),
        "branch_index_sha256": data_cfg["branch_index_sha256"],
        "branch_payload_sha256_manifest_sha256": data_cfg[
            "branch_payload_sha256_manifest_sha256"
        ],
        "runtime_index_sha256": data_cfg["runtime_index_sha256"],
        "candidate_zero_excluded_from_selection": True,
        "planner_action_inputs": False,
        "stage0_frozen": True,
        "future_observation_leakage": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"report_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
