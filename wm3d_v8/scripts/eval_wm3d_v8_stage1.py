#!/usr/bin/env python3
"""Evaluate a committed unified Stage1 DCP and publish an immutable receipt."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from wm3d_v3.stage1_planner.candidates import deterministic_action_cost
from wm3d_v3.stage1_planner.train import (
    _checkpoint_commit_sha,
    _dataset,
    _device,
    _expectations,
    _load_stage1,
    validate_stage1_bindings,
)
from wm3d_v3.stage1_planner.losses import PlannerLossConfig, planner_loss
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig
from wm3d_v3.training.distributed_checkpoint import (
    DistributedCheckpointManager,
    ResumeExpectations,
    sha256_file,
)
from wm3d_v3.training.distributed_runtime import (
    destroy_distributed,
    initialize_distributed,
    strategy_from_mapping,
    wrap_model,
)
from wm3d_v3.training.runtime_contract import load_materialized_runtime
from wm3d_v3.models.model_factory import build_world_model


EVAL_RECEIPT_SCHEMA = "wm3d_v8_unified_stage1_eval_receipt_v1"


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(bool).reshape(-1); scores = scores.reshape(-1)
    positive = int(labels.sum()); negative = int((~labels).sum())
    if not positive or not negative: return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle: handle.write(payload); handle.flush()
    try: os.link(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--max-branches", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage1, stage1_sha = _load_stage1(args.runtime)
    stage0, stage0_sha = load_materialized_runtime(Path(stage1["stage0_runtime"]))
    repo = Path(__file__).resolve().parents[1]
    current_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != stage0["run"]["code_commit"]:
        raise ValueError("Stage1 evaluation code commit differs from sealed runtime")
    validate_stage1_bindings(stage1, stage0)
    if stage0_sha != stage1["branch"]["stage0_runtime_sha256"]:
        raise ValueError("Stage1 branch belongs to another Stage0 runtime")
    stage0_source = Path(stage1["stage0_checkpoint"])
    if _checkpoint_commit_sha(stage0_source) != stage1["branch"]["stage0_checkpoint_commit_sha256"]:
        raise ValueError("Stage1 branch belongs to another Stage0 checkpoint")
    strategy = strategy_from_mapping(stage0["runtime_profile"]["distributed"])
    context = initialize_distributed(strategy)
    try:
        seed = int(stage1["run"]["seed"]); torch.manual_seed(seed)
        with torch.device("meta" if strategy.initialization == "meta_sharded" else context.device):
            world = build_world_model(stage0["model_profile"])
        world = wrap_model(world, context, strategy, initialization_seed=seed if strategy.initialization == "meta_sharded" else None).model
        source_step = int(stage0_source.name.split("_")[1])
        run_contract = json.loads((Path(stage0["run"]["output_root"]) / "run_contract.json").read_text())
        DistributedCheckpointManager(stage0_source.parent).load_model_for_evaluation(
            path=stage0_source, model=world, expected=ResumeExpectations(
                step=source_step, run_lineage=stage0["run"]["lineage"], runtime_config_sha256=stage0_sha,
                data_closure_sha256=stage0["bindings"]["data_closure_sha256"],
                model_contract_sha256=stage0["bindings"]["model_contract_sha256"], world_size=context.world_size,
                shard_degree=int(stage0["runtime_profile"]["distributed"]["shard_degree"]), distributed_strategy=strategy.strategy,
                global_batch_size=int(stage0["runtime_profile"]["train"]["global_batch_size"]),
                topology_contract_sha256=run_contract["topology_contract_sha256"]),
        )
        for parameter in world.parameters(): parameter.requires_grad_(False)
        system = NativePlanningSystem(world, Stage1SystemConfig(
            planner=NativePlannerConfig(**stage1["planner"]["model"]), horizon=int(stage1["planner"]["horizon"]),
            candidate_microbatch=int(stage1["planner"]["candidate_microbatch"]),
            **stage1["planner"]["score"]))
        planner = system.planner.to(context.device)
        if context.world_size > 1:
            planner = DistributedDataParallel(
                planner, device_ids=[context.local_rank], broadcast_buffers=False
            )
            system.planner = planner
        step = int(args.checkpoint.name.split("_")[1])
        DistributedCheckpointManager(args.checkpoint.parent).load_model_for_evaluation(
            path=args.checkpoint, model=planner,
            expected=_expectations(step=step, stage1=stage1, stage1_sha=stage1_sha, runtime=stage0, world_size=context.world_size))
        system.eval(); dataset = _dataset(stage0, stage1, args.split)
        limit = len(dataset) if args.max_branches <= 0 else min(len(dataset), args.max_branches)
        if limit <= 0:
            raise ValueError("Stage1 evaluation selected no sealed branches")
        scores=[]; success=[]; imagined=[]; roots=[]
        gate_batch = _device(next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))), context.device)
        planner.eval()
        planner.zero_grad(set_to_none=True)
        gate_output = system.score_observed_batch(gate_batch)
        horizon = system.cfg.horizon
        loss_cfg = PlannerLossConfig(**stage1["planner"]["loss"])
        gate_loss = planner_loss(
            gate_output,
            branch_rewards=gate_batch["branch_rewards"][:, :, :horizon],
            branch_dones=gate_batch["branch_dones"][:, :, :horizon],
            branch_success=gate_batch["branch_success"][:, :, :horizon],
            branch_valid=gate_batch["branch_valid"],
            uncertainty_target=torch.zeros_like(gate_batch["branch_valid"], dtype=torch.float32),
            cfg=loss_cfg,
        )["loss"]
        # A one-slot cyclic permutation changes every non-constant candidate
        # target vector; the branch seal already rejects constant utility.
        permutation = torch.roll(
            torch.arange(gate_batch["branch_success"].shape[1], device=context.device),
            shifts=1,
        )
        shuffled_loss = planner_loss(
            gate_output,
            branch_rewards=gate_batch["branch_rewards"][:, permutation, :horizon],
            branch_dones=gate_batch["branch_dones"][:, permutation, :horizon],
            branch_success=gate_batch["branch_success"][:, permutation, :horizon],
            branch_valid=gate_batch["branch_valid"],
            uncertainty_target=torch.zeros_like(gate_batch["branch_valid"], dtype=torch.float32),
            cfg=loss_cfg,
        )["loss"]
        gate_loss.backward()
        planner_grads = [parameter.grad for parameter in planner.parameters() if parameter.grad is not None]
        planner_grad_finite_nonzero = bool(planner_grads) and all(bool(torch.isfinite(value).all()) for value in planner_grads) and sum(float(value.abs().sum()) for value in planner_grads) > 0
        stage0_grad_owned = any(parameter.grad is not None for parameter in world.parameters())
        label_shuffle_sensitive = not bool(torch.isclose(gate_loss.detach(), shuffled_loss.detach(), atol=1e-8, rtol=1e-6))
        # Shuffle every grouped action lane while holding evidence fixed.  The
        # learned fields must remain bitwise identical; only the deterministic
        # external action-cost component is allowed to change.
        shuffled_batch = dict(gate_batch)
        for name in (
            "candidate_fine_action_values", "candidate_fine_action_mask",
            "candidate_fine_action_dt", "candidate_fine_sample_mask",
            "candidate_coarse_action_values", "candidate_coarse_action_mask",
        ):
            shuffled_batch[name] = gate_batch[name][:, permutation]
        with torch.no_grad():
            action_after = system.score_observed_batch(shuffled_batch)
        learned_fields = ("progress_logit", "success_logit", "risk_logit", "uncertainty_logit")
        action_shuffle_invariant = all(
            torch.equal(gate_output[name].detach(), action_after[name].detach())
            for name in learned_fields
        )
        if context.world_size > 1:
            pass_gates = torch.tensor(
                [planner_grad_finite_nonzero, label_shuffle_sensitive, action_shuffle_invariant],
                dtype=torch.int64,
                device=context.device,
            )
            owned_gate = torch.tensor(
                [stage0_grad_owned], dtype=torch.int64, device=context.device
            )
            torch.distributed.all_reduce(pass_gates, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(owned_gate, op=torch.distributed.ReduceOp.MAX)
            planner_grad_finite_nonzero = bool(pass_gates[0].item())
            label_shuffle_sensitive = bool(pass_gates[1].item())
            action_shuffle_invariant = bool(pass_gates[2].item())
            stage0_grad_owned = bool(owned_gate[0].item())
        if not (planner_grad_finite_nonzero and not stage0_grad_owned and label_shuffle_sensitive and action_shuffle_invariant):
            raise RuntimeError("Stage1 ownership/invariance/sensitivity gate failed")
        planner.zero_grad(set_to_none=True); system.eval()
        with torch.inference_mode():
            for index, raw in enumerate(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)):
                if index >= limit: break
                batch = _device(raw, context.device)
                observed = system.score_observed_batch(batch)
                costs = deterministic_action_cost(
                    batch["candidate_fine_action_values"][:, :, :system.cfg.horizon],
                    batch["candidate_fine_action_mask"][:, :, :system.cfg.horizon],
                    batch["candidate_fine_sample_mask"][:, :, :system.cfg.horizon],
                    batch["candidate_coarse_action_values"][:, :, :system.cfg.horizon],
                    batch["candidate_coarse_action_mask"][:, :, :system.cfg.horizon],
                )
                rollout = system.imagine(batch)
                imagined_output = system.score_rollout(rollout, batch["task_embedding"], costs)
                scores.append(observed["score"][0].float().cpu().numpy())
                imagined.append(imagined_output["score"][0].float().cpu().numpy())
                success.append(batch["branch_success"][0, :, :system.cfg.horizon].any(dim=-1).cpu().numpy())
                roots.append(str(raw["branch_id"][0]))
        score=np.stack(scores); imagined_score=np.stack(imagined); labels=np.stack(success)
        receipt={"schema": EVAL_RECEIPT_SCHEMA, "passed": True, "runtime_sha256": stage1_sha,
            "code_commit": current_commit,
            "stage0_checkpoint_commit_sha256": _checkpoint_commit_sha(stage0_source),
            "stage1_checkpoint_commit_sha256": sha256_file(args.checkpoint / "COMMITTED.json"),
            "branch_index_sha256": stage1["branch"]["index_sha256"], "split": args.split,
            "branch_ids": roots, "success_auc": _auc(labels, score), "imagined_success_auc": _auc(labels, imagined_score),
            "selected_success": float(labels[np.arange(len(labels)), score.argmax(1)].mean()),
            "oracle_success": float(labels.max(1).mean()), "stage0_frozen": True,
            "planner_action_inputs": False, "imagined_rollout": "single_trained_K_only",
            "gates": {"action_shuffle_invariance": action_shuffle_invariant,
                "label_shuffle_sensitivity": label_shuffle_sensitive,
                "planner_gradient_finite_nonzero": planner_grad_finite_nonzero,
                "stage0_gradient_absent": not stage0_grad_owned}}
        if context.rank == 0: _publish(args.output, receipt); print(json.dumps(receipt, sort_keys=True))
    finally:
        destroy_distributed()


if __name__ == "__main__": main()
