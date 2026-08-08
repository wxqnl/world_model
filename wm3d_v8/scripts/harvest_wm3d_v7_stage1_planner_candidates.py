#!/usr/bin/env python3
"""Harvest H32 candidates from a pinned V7 Stage0 checkpoint.

The direct head remains branch ``direct``.  Four flow samples alter pose only;
their gripper trajectory is copied from the direct event owner.  Later hard
negatives are deterministic transforms of the direct plan.  Candidate
generation sees only the root observation, task and executed H4 history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm3d_v3.stage1_planner.candidates import DEFAULT_ROLES, build_candidate_set
from wm3d_v3.training.train import build_model, load_train_config


SCHEMA = "wm3d_v7_stage1_planner_candidates_v1"
ROOT_CONTEXT_SCHEMA = "wm3d_v7_stage1_planner_root_context_v1"


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def dequantize(archive, prefix: str) -> np.ndarray:
    return np.asarray(archive[f"{prefix}_codes"], dtype=np.int8).astype(
        np.float32
    ) * np.asarray(archive[f"{prefix}_scale"], dtype=np.float32)


def physical(action_cond: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    pose = action_cond[..., :6] * std + mean
    grip = action_cond[..., 6:7].clamp(0.0, 1.0) * 2.0 - 1.0
    return torch.cat((pose, grip), dim=-1)


@torch.inference_mode()
def propose_h32(
    model,
    context: torch.Tensor,
    wrist: torch.Tensor,
    task: torch.Tensor,
    history: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    seed: int,
) -> np.ndarray:
    device = context.device
    core_horizon = int(model.cfg.dual.state.k)
    if core_horizon != 8:
        raise RuntimeError("Stage1-P candidate harvesting requires the native K8 core")
    flow_action_dim = int(model.action_policy.cfg.flow_action_dim)
    if flow_action_dim != 6:
        raise RuntimeError("Stage1-P requires the audited pose-only 6D flow proposer")

    def one_path(flow_index: int | None) -> torch.Tensor:
        state = model.fuse_views(
            context,
            wrist,
            view_mask=torch.ones((1, context.shape[1], 2), dtype=torch.bool, device=device),
        )
        local_history = history.clone()
        chunks: list[torch.Tensor] = []
        generator = torch.Generator(device=device)
        stream_index = -1 if flow_index is None else int(flow_index)
        generator.manual_seed(int(seed) + 104_729 * (stream_index + 2))
        for _chunk in range(4):
            noise = None
            if flow_index is not None:
                noise = torch.randn(
                    (1, core_horizon, flow_action_dim),
                    generator=generator,
                    device=device,
                    dtype=state.dtype,
                )
            policy = model.act_policy(
                state,
                task,
                action_history=local_history,
                flow_sample=flow_index is not None,
                flow_noise=noise,
            )
            direct = policy["policy_action_cond"]
            if flow_index is None:
                chosen = direct
            else:
                chosen = torch.cat(
                    (policy["policy_flow_action_cond"][..., :6], direct[..., 6:7]),
                    dim=-1,
                )
            chunks.append(chosen)
            predicted = model.dual(state, task, action_cond=chosen)["pred_tokens"]
            state = torch.cat((state, predicted), dim=1)[:, -int(model.cfg.dual.state.T) :]
            history_physical = physical(chosen, mean, std)
            history_exec = history_physical.clone()
            history_exec[..., 6] = chosen[..., 6]
            local_history = torch.cat((local_history, history_exec), dim=1)[:, -4:]
        return torch.cat(chunks, dim=1)

    direct = one_path(None)
    flows = torch.stack([one_path(index)[..., :6] for index in range(4)], dim=1)
    candidates = build_candidate_set(direct, flows, roles=DEFAULT_ROLES)
    return physical(candidates.actions, mean, std).squeeze(0).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root-context-index", type=Path, required=True)
    parser.add_argument("--action-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-roots", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard contract")

    cfg = load_train_config(args.model_config)
    checkpoint_sha = sha256_file(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    result = model.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("Stage0 checkpoint strict load was not clean")
    model.eval().to(args.device)
    if model.action_policy is None or not model.action_policy.cfg.enable_flow_head:
        raise RuntimeError("Stage0 must contain direct and pose-flow proposal heads")
    if model.action_policy.cfg.flow_use_as_policy:
        raise RuntimeError("flow is auxiliary and must not own Stage0 serving")

    with np.load(args.action_stats, allow_pickle=False) as stats:
        if str(stats["split"].item()) != "train":
            raise RuntimeError("action statistics are not train-only")
        mean_np = np.asarray(stats["mean"], dtype=np.float32)
        std_np = np.asarray(stats["std"], dtype=np.float32)
    mean = torch.from_numpy(mean_np).to(args.device).view(1, 1, 6)
    std = torch.from_numpy(std_np).to(args.device).view(1, 1, 6)
    model_config_sha = sha256_file(args.model_config)
    action_stats_sha = sha256_file(args.action_stats)
    context_index_sha = sha256_file(args.root_context_index)
    context_rows = [
        json.loads(line)
        for line in args.root_context_index.read_text().splitlines()
        if line.strip()
    ]
    context_rows = context_rows[args.shard_index :: args.num_shards]
    if args.max_roots > 0:
        context_rows = context_rows[: args.max_roots]
    if not context_rows:
        raise RuntimeError("candidate shard is empty")

    rows: list[dict] = []
    for ordinal, row in enumerate(context_rows):
        if row.get("schema") != ROOT_CONTEXT_SCHEMA:
            raise RuntimeError(f"root-context schema mismatch: {row.get('root_id')}")
        if row.get("future_observation_leakage") is not False:
            raise RuntimeError(f"non-causal root context: {row.get('root_id')}")
        if int(row.get("max_observation_state_step", -1)) != int(row["t0"]):
            raise RuntimeError(f"root-context endpoint mismatch: {row.get('root_id')}")
        context_path = Path(row["path"])
        context_sha = sha256_file(context_path)
        with np.load(context_path, allow_pickle=False) as archive:
            if str(archive["schema"].item()) != ROOT_CONTEXT_SCHEMA:
                raise RuntimeError("root-context payload schema mismatch")
            if str(archive["root_id"].item()) != row["root_id"]:
                raise RuntimeError("root-context payload identity mismatch")
            context_np = dequantize(archive, "anchor")
            wrist_np = dequantize(archive, "wrist")
            task_np = np.asarray(archive["task_emb"], dtype=np.float32)
            history = np.asarray(archive["action_history_physical"], dtype=np.float32)
        if context_np.shape[:2] != (16, 64) or wrist_np.shape != context_np.shape:
            raise RuntimeError(f"causal context shape mismatch: {row['root_id']}")
        if task_np.shape != (2048,) or not np.isfinite(task_np).all():
            raise RuntimeError(f"task embedding mismatch: {row['root_id']}")
        if history.shape != (4, 7) or not np.isfinite(history).all():
            raise RuntimeError(f"action-history mismatch: {row['root_id']}")
        history_policy = history.copy()
        history_policy[:, 6] = np.clip((history_policy[:, 6] + 1.0) * 0.5, 0.0, 1.0)
        context = torch.from_numpy(context_np).to(args.device)[None]
        wrist = torch.from_numpy(wrist_np).to(args.device)[None]
        task = torch.from_numpy(task_np).to(args.device)[None]
        history_tensor = torch.from_numpy(history_policy).to(args.device)[None]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            actions = propose_h32(
                model,
                context,
                wrist,
                task,
                history_tensor,
                mean,
                std,
                seed=args.seed + ordinal,
            )
        destination = args.output_root / row["split"] / f"{row['root_id']}.npz"
        atomic_savez(
            destination,
            schema=np.asarray(SCHEMA),
            root_id=np.asarray(row["root_id"]),
            branch_roles=np.asarray(DEFAULT_ROLES),
            branch_actions_physical=actions,
            action_history_physical=history,
            root_context_sha256=np.asarray(context_sha),
            stage0_checkpoint_sha256=np.asarray(checkpoint_sha),
        )
        rows.append({
            "schema": SCHEMA,
            "root_id": row["root_id"],
            "candidate_path": str(destination.resolve()),
            "root_context_path": str(context_path.resolve()),
            "root_context_sha256": context_sha,
            "source_dataset": row["source_dataset"],
            "task": row["task"],
            "task_text": row["task_text"],
            "episode_id": int(row["episode_id"]),
            "episode_root_index": int(row["episode_root_index"]),
            "t0": int(row["t0"]),
            "split": row["split"],
            "split_group": row["split_group"],
            "branch_roles": list(DEFAULT_ROLES),
            "future_frames": 32,
            "future_observation_leakage": False,
            "stage0_checkpoint": str(args.checkpoint.resolve()),
            "stage0_checkpoint_sha256": checkpoint_sha,
            "model_config_sha256": model_config_sha,
            "action_stats_sha256": action_stats_sha,
            "root_context_index_sha256": context_index_sha,
        })
        atomic_jsonl(args.output_index.with_suffix(args.output_index.suffix + ".partial"), rows)
        print(json.dumps({"root_id": row["root_id"], "status": "harvested"}), flush=True)
    os.replace(args.output_index.with_suffix(args.output_index.suffix + ".partial"), args.output_index)


if __name__ == "__main__":
    main()
