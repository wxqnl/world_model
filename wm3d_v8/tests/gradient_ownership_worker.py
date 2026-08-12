"""Real NCCL FSDP2/DDP worker for the Stage0 gradient ownership audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from wm3d_v3.data.grouped_robot import COMPOSITION_OPERATOR_IDS
from tests.test_native_world_model import _batch, _tiny_config
from wm3d_v3.models.native_world_model import NativeWorldModel
from wm3d_v3.training.distributed_runtime import (
    DistributedStrategyConfig,
    autocast_context,
    destroy_distributed,
    initialize_distributed,
    wrap_model,
)
from wm3d_v3.training.gradient_ownership import (
    _local_tensor,
    _owner_parameters,
    _replication_factor,
    audit_gradient_ownership,
    validate_gradient_ownership_receipt,
)
from wm3d_v3.training.native_objective import (
    NativeObjectiveConfig,
    compute_native_objective,
)


def _supervised_batch(
    inputs: dict[str, torch.Tensor], output: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    batch, groups, queries, action_dim = output["policy_action_raw"].shape
    horizon = output["pred_tokens"].shape[1]
    target_fine = torch.randn_like(output["policy_action_raw"])
    target_fine[output["policy_gripper_mask"]] = torch.randint(
        0,
        2,
        target_fine[output["policy_gripper_mask"]].shape,
        device=target_fine.device,
        dtype=torch.int64,
    ).to(target_fine.dtype)
    operators = torch.full(
        (batch, groups, action_dim),
        COMPOSITION_OPERATOR_IDS["sum"],
        dtype=torch.long,
        device=target_fine.device,
    )
    operators[..., 3:6] = COMPOSITION_OPERATOR_IDS[
        "so3_axis_angle_body_right"
    ]
    operators[..., 6] = COMPOSITION_OPERATOR_IDS["logical_last"]
    supervised = dict(inputs)
    supervised.update(
        {
            "target_tokens": torch.randn_like(output["pred_tokens"]),
            "target_rgb": torch.randn_like(output["rgb"]),
            "target_rgb_mask": torch.ones_like(
                output["rgb"][:, :, :, :1, :1, :1], dtype=torch.bool
            ),
            "target_depth": torch.rand_like(output["depth"]) + 0.1,
            "target_depth_mask": torch.ones_like(output["depth"], dtype=torch.bool),
            "target_point": torch.randn_like(output["point"]),
            "target_point_mask": torch.ones_like(
                output["point"][..., 0], dtype=torch.bool
            ),
            "target_camera_pose": torch.randn_like(output["camera_pose"]),
            "target_camera_pose_mask": torch.ones_like(
                output["camera_pose"][..., 0], dtype=torch.bool
            ),
            "target_fine_action": target_fine,
            "target_fine_action_mask": output["policy_action_mask"].clone(),
            "future_world_boundaries_dt": torch.tensor(
                [0.0, 0.2, 0.6], device=target_fine.device
            ).expand(batch, horizon + 1),
            "composition_operator_ids": operators,
            "target_coarse_action": torch.randn(
                batch,
                horizon,
                groups,
                action_dim,
                device=target_fine.device,
                dtype=target_fine.dtype,
            ),
            "target_coarse_action_normalized": torch.randn(
                batch,
                horizon,
                groups,
                action_dim,
                device=target_fine.device,
                dtype=target_fine.dtype,
            ),
            "target_coarse_action_mask": torch.ones(
                batch,
                horizon,
                groups,
                action_dim,
                device=target_fine.device,
                dtype=torch.bool,
            ),
        }
    )
    return supervised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=("fsdp2", "ddp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strategy = DistributedStrategyConfig(
        strategy=args.strategy,
        shard_degree=2 if args.strategy == "fsdp2" else 1,
        initialization="meta_sharded" if args.strategy == "fsdp2" else "direct",
        param_dtype="bf16",
        reduce_dtype="fp32",
        output_dtype="bf16",
    )
    context = initialize_distributed(strategy)
    try:
        torch.manual_seed(1707)
        torch.cuda.manual_seed_all(1707)
        construction_device = (
            torch.device("meta") if args.strategy == "fsdp2" else context.device
        )
        with torch.device(construction_device):
            native_model = NativeWorldModel(_tiny_config())
        wrapped = wrap_model(
            native_model,
            context,
            strategy,
            initialization_seed=1707 if args.strategy == "fsdp2" else None,
        )
        if args.strategy == "fsdp2":
            assert wrapped.model is native_model
            assert any(hasattr(parameter, "to_local") for parameter in native_model.parameters())
        else:
            assert wrapped.model.module is native_model
        batch = {
            name: value.to(context.device) if isinstance(value, torch.Tensor) else value
            for name, value in _batch(native_model.cfg).items()
        }
        with autocast_context(strategy):
            output = wrapped.model(**batch)
            losses = compute_native_objective(
                output=output,
                batch=_supervised_batch(batch, output),
                config=NativeObjectiveConfig(),
            )
        losses["total"].backward()
        receipt = audit_gradient_ownership(native_model)
        validate_gradient_ownership_receipt(receipt, native_model)
        # Prove the audit counted local shards, not global DTensor numel on
        # every rank.  Summing local owner storage must recover global counts.
        local_storage = {}
        for owner, parameters in _owner_parameters(native_model).items():
            local_storage[owner] = sum(
                _local_tensor(parameter).numel() / _replication_factor(parameter)
                for parameter in parameters
            )
        for owner, local in local_storage.items():
            total = torch.tensor(local, dtype=torch.float64, device=context.device)
            dist.all_reduce(total)
            assert int(total.item()) == receipt["owners"][owner]["parameter_elements"]
        gathered: list[object] = [None] * context.world_size
        dist.all_gather_object(gathered, receipt)
        assert all(value == receipt for value in gathered)
        if context.is_rank0:
            args.output.write_text(
                json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
