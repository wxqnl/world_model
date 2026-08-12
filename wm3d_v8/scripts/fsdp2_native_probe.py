#!/usr/bin/env python3
"""Real two-rank forward/backward proof for the unified native core."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from wm3d_v3.data.grouped_robot import ACTION_SEMANTIC_IDS, STATE_SEMANTIC_IDS
from wm3d_v3.models.native_world_model import NativeWorldModel, NativeWorldModelConfig
from wm3d_v3.training.distributed_runtime import (
    DistributedStrategyConfig,
    initialize_distributed,
    wrap_model,
)


def config() -> NativeWorldModelConfig:
    return NativeWorldModelConfig(
        T=2,
        P=4,
        K=2,
        token_dim=16,
        task_dim=12,
        num_views=2,
        state_hidden=64,
        state_layers=2,
        state_heads=4,
        state_ff_mult=2,
        action_hidden=48,
        action_layers=2,
        action_heads=4,
        action_ff_mult=2,
        bridge_layers_state=(1,),
        bridge_heads=4,
        dynamics_layers=1,
        view_hidden=32,
        view_heads=4,
        view_ff_mult=2,
        max_action_groups=2,
        max_action_dim=7,
        max_state_dim=10,
        max_action_substeps=3,
        max_policy_queries=3,
        max_group_id=8,
        max_embodiments=8,
        max_action_semantic_id=16,
        max_state_semantic_id=16,
        time_fourier_dim=8,
        max_aux_tokens=2,
        aux_dim=8,
        max_aux_type_id=8,
        rgb_hidden=16,
        rgb_size=16,
        rgb_decode_indices=(0, 1),
        geom_hidden=16,
        activation_checkpointing=True,
    )


def batch(cfg: NativeWorldModelConfig, device: torch.device) -> dict[str, torch.Tensor]:
    torch.manual_seed(91)
    batch_size, groups, substeps, action_dim = 1, 2, 3, 7
    state_dim, queries = 10, 3
    semantic = torch.tensor(
        [1, 1, 1, 3, 3, 3, ACTION_SEMANTIC_IDS["absolute_gripper_open01"]]
    ).view(1, 1, action_dim).expand(batch_size, groups, -1)
    state_semantic = torch.tensor(
        [1, 1, 1, 2, 2, 2, 2, 2, 2, STATE_SEMANTIC_IDS["gripper_close01"]]
    ).view(1, 1, state_dim).expand(batch_size, groups, -1)
    dt = torch.tensor([0.0, 0.037, 0.119]).view(1, 1, 1, substeps)
    query = torch.tensor([0.0, 0.071, 0.203]).view(1, 1, queries)
    fine_h = torch.randn(batch_size, cfg.T, groups, substeps, action_dim)
    fine_f = torch.randn(batch_size, cfg.K, groups, substeps, action_dim)
    values = {
        "world_tokens": torch.randn(batch_size, cfg.T, 2, 4, 16),
        "view_mask": torch.ones(batch_size, cfg.T, 2, dtype=torch.bool),
        "world_times_s": torch.tensor([[0.0, 0.17, 0.48, 0.91]]),
        "task_embedding": torch.randn(batch_size, 12),
        "history_fine_action_values": fine_h,
        "history_fine_action_mask": torch.ones_like(fine_h, dtype=torch.bool),
        "history_fine_action_dt": dt.expand(batch_size, cfg.T, groups, substeps),
        "history_fine_sample_mask": torch.ones(batch_size, cfg.T, groups, substeps, dtype=torch.bool),
        "history_coarse_action_values": torch.zeros(batch_size, cfg.T, groups, action_dim),
        "history_coarse_action_mask": torch.zeros(batch_size, cfg.T, groups, action_dim, dtype=torch.bool),
        "future_factual_fine_action_values": fine_f,
        "future_factual_fine_action_mask": torch.ones_like(fine_f, dtype=torch.bool),
        "future_factual_fine_action_dt": dt.expand(batch_size, cfg.K, groups, substeps),
        "future_factual_fine_sample_mask": torch.ones(batch_size, cfg.K, groups, substeps, dtype=torch.bool),
        "future_factual_coarse_action_values": torch.zeros(batch_size, cfg.K, groups, action_dim),
        "future_factual_coarse_action_mask": torch.zeros(batch_size, cfg.K, groups, action_dim, dtype=torch.bool),
        "action_group_ids": torch.tensor([[1, 2]]),
        "action_group_mask": torch.ones(batch_size, groups, dtype=torch.bool),
        "action_semantic_ids": semantic,
        "current_state_values": torch.randn(batch_size, groups, state_dim),
        "current_state_mask": torch.ones(batch_size, groups, state_dim, dtype=torch.bool),
        "state_semantic_ids": state_semantic,
        "embodiment_ids": torch.tensor([2]),
        "policy_query_dt": query.expand(batch_size, groups, queries),
        "policy_query_mask": torch.ones(batch_size, groups, queries, dtype=torch.bool),
    }
    return {name: value.to(device) for name, value in values.items()}


def main() -> None:
    strategy = DistributedStrategyConfig(
        strategy="fsdp2", shard_degree=2, initialization="meta_sharded"
    )
    context = initialize_distributed(strategy)
    torch.manual_seed(3407)
    with torch.device("meta"):
        model = NativeWorldModel(config())
    model = wrap_model(model, context, strategy, initialization_seed=3407).model
    output = model(**batch(config(), context.device))
    loss = (
        output["pred_tokens"].float().square().mean()
        + output["policy_action_raw"].float().square().mean()
        + output["depth"].float().mean()
        + output["point"].float().square().mean()
    )
    loss.backward()
    value = torch.tensor([float(loss), torch.cuda.max_memory_allocated()], device=context.device)
    gathered = [torch.zeros_like(value) for _ in range(context.world_size)]
    dist.all_gather(gathered, value)
    if context.is_rank0:
        losses = [float(item[0]) for item in gathered]
        if max(losses) - min(losses) > 1.0e-5:
            raise RuntimeError(f"rank initialization drifted: {losses}")
        print({"passed": True, "loss_by_rank": losses, "peak_bytes_by_rank": [int(item[1]) for item in gathered]})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
