"""Real 2-GPU FSDP2 BF16 activation-checkpoint proof for native_1b.

This worker constructs the sealed 1.319B profile on meta, materializes only
FSDP2 shards, and executes one complete forward/backward/AdamW step.  It is an
explicit integration worker, not part of the CPU pytest suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS, STATE_SEMANTIC_IDS
from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.distributed_runtime import (
    DistributedStrategyConfig,
    autocast_context,
    destroy_distributed,
    initialize_distributed,
    wrap_model,
)


def _batch(model: NativeWorldModel, device: torch.device) -> dict[str, torch.Tensor]:
    cfg = model.cfg
    batch = 1
    groups = cfg.max_action_groups
    action_dim = cfg.max_action_dim
    state_dim = cfg.max_state_dim
    substeps = 4
    queries = 8
    group_mask = torch.zeros(batch, groups, dtype=torch.bool, device=device)
    group_mask[:, 0] = True

    action_semantics = torch.zeros(
        batch, groups, action_dim, dtype=torch.long, device=device
    )
    action_semantics[:, 0, :3] = ACTION_SEMANTIC_IDS["delta_position_m"]
    action_semantics[:, 0, 3:6] = ACTION_SEMANTIC_IDS[
        "delta_rotation_axis_angle_rad"
    ]
    action_semantics[:, 0, 6] = ACTION_SEMANTIC_IDS["absolute_gripper_open01"]
    state_semantics = torch.zeros(
        batch, groups, state_dim, dtype=torch.long, device=device
    )
    state_semantics[:, 0, :3] = STATE_SEMANTIC_IDS["eef_position_m"]
    state_semantics[:, 0, 3:9] = STATE_SEMANTIC_IDS["eef_rotation_6d"]
    state_semantics[:, 0, 9] = STATE_SEMANTIC_IDS["gripper_close01"]

    history_values = torch.randn(
        batch, cfg.T, groups, substeps, action_dim, device=device
    )
    future_values = torch.randn(
        batch, cfg.K, groups, substeps, action_dim, device=device
    )
    history_dim_mask = torch.zeros_like(history_values, dtype=torch.bool)
    future_dim_mask = torch.zeros_like(future_values, dtype=torch.bool)
    history_dim_mask[:, :, 0, :, :7] = True
    future_dim_mask[:, :, 0, :, :7] = True
    history_sample_mask = history_dim_mask.any(dim=-1)
    future_sample_mask = future_dim_mask.any(dim=-1)
    substep_dt = torch.tensor(
        [0.0, 0.05, 0.1, 0.15], dtype=torch.float32, device=device
    )
    history_dt = substep_dt.view(1, 1, 1, -1).expand(
        batch, cfg.T, groups, -1
    ).clone()
    future_dt = substep_dt.view(1, 1, 1, -1).expand(
        batch, cfg.K, groups, -1
    ).clone()

    current_mask = torch.zeros(
        batch, groups, state_dim, dtype=torch.bool, device=device
    )
    current_mask[:, 0, :10] = True
    query_dt = torch.linspace(
        0.0, 0.35, queries, dtype=torch.float32, device=device
    ).view(1, 1, queries).expand(batch, groups, -1).clone()
    query_mask = group_mask[..., None].expand(-1, -1, queries).clone()
    world_times = torch.arange(
        cfg.T + cfg.K, dtype=torch.float32, device=device
    ).mul_(0.2).view(1, -1)

    return {
        "world_tokens": torch.randn(
            batch, cfg.T, cfg.num_views, cfg.P, cfg.token_dim, device=device
        ),
        "view_mask": torch.ones(
            batch, cfg.T, cfg.num_views, dtype=torch.bool, device=device
        ),
        "world_times_s": world_times,
        "task_embedding": torch.randn(batch, cfg.task_dim, device=device),
        "history_fine_action_values": history_values,
        "history_fine_action_mask": history_dim_mask,
        "history_fine_action_dt": history_dt,
        "history_fine_sample_mask": history_sample_mask,
        "history_coarse_action_values": torch.zeros(
            batch, cfg.T, groups, action_dim, device=device
        ),
        "history_coarse_action_mask": torch.zeros(
            batch, cfg.T, groups, action_dim, dtype=torch.bool, device=device
        ),
        "future_factual_fine_action_values": future_values,
        "future_factual_fine_action_mask": future_dim_mask,
        "future_factual_fine_action_dt": future_dt,
        "future_factual_fine_sample_mask": future_sample_mask,
        "future_factual_coarse_action_values": torch.zeros(
            batch, cfg.K, groups, action_dim, device=device
        ),
        "future_factual_coarse_action_mask": torch.zeros(
            batch, cfg.K, groups, action_dim, dtype=torch.bool, device=device
        ),
        "action_group_ids": torch.tensor(
            [[1] + [0] * (groups - 1)], dtype=torch.long, device=device
        ),
        "action_group_mask": group_mask,
        "action_semantic_ids": action_semantics,
        "current_state_values": torch.randn(
            batch, groups, state_dim, device=device
        ),
        "current_state_mask": current_mask,
        "state_semantic_ids": state_semantics,
        "embodiment_ids": torch.ones(batch, dtype=torch.long, device=device),
        "policy_query_dt": query_dt,
        "policy_query_mask": query_mask,
        "action_normalization_offset": torch.zeros(
            batch, groups, action_dim, device=device
        ),
        "action_normalization_scale": torch.ones(
            batch, groups, action_dim, device=device
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strategy = DistributedStrategyConfig(
        strategy="fsdp2",
        shard_degree=2,
        initialization="meta_sharded",
        param_dtype="bf16",
        reduce_dtype="fp32",
        output_dtype="bf16",
    )
    context = initialize_distributed(strategy)
    try:
        profile = yaml.safe_load(args.model_profile.read_text(encoding="utf-8"))
        with torch.device("meta"):
            raw = build_world_model(profile)
        assert isinstance(raw, NativeWorldModel)
        wrapped = wrap_model(raw, context, strategy, initialization_seed=3407)
        optimizer = torch.optim.AdamW(
            wrapped.model.parameters(), lr=1.0e-5, foreach=False
        )
        batch = _batch(raw, context.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(strategy):
            output = wrapped.model(**batch)
            components = {
                "native": output["pred_tokens"].square().mean(),
                "action": output["policy_action_raw"].square().mean(),
                "rgb": output["rgb"].square().mean(),
                "depth": output["depth"].square().mean(),
                "point": output["point"].square().mean(),
            }
            loss = sum(components.values())
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite formal-profile loss")
        loss.backward()
        local_nonfinite = torch.zeros((), dtype=torch.long, device=context.device)
        local_nonzero = torch.zeros((), dtype=torch.long, device=context.device)
        for parameter in wrapped.model.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
            local_nonfinite += (~torch.isfinite(gradient)).sum()
            local_nonzero += torch.count_nonzero(
                torch.where(torch.isfinite(gradient), gradient, torch.zeros_like(gradient))
            )
        dist.all_reduce(local_nonfinite)
        dist.all_reduce(local_nonzero)
        if int(local_nonfinite) or not int(local_nonzero):
            raise RuntimeError(
                f"invalid gradients: nonfinite={int(local_nonfinite)}, "
                f"nonzero={int(local_nonzero)}"
            )
        optimizer.step()
        torch.cuda.synchronize(context.device)
        evidence = {
            "rank": context.rank,
            "profile": str(profile["name"]),
            "global_parameter_count": sum(p.numel() for p in raw.parameters()),
            "loss": float(loss.detach()),
            "components": {
                name: float(value.detach()) for name, value in components.items()
            },
            "gradient_nonfinite": int(local_nonfinite),
            "gradient_nonzero_global": int(local_nonzero),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(context.device),
            "optimizer_step_completed": True,
        }
        gathered: list[object] = [None] * context.world_size
        dist.all_gather_object(gathered, evidence)
        if context.is_rank0:
            args.output.write_text(
                json.dumps(gathered, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(gathered, sort_keys=True), flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
