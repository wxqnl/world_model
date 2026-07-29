"""Two-GPU FSDP2 + DCP smoke for native WM3D-V7 5B infrastructure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from wm3d_v3.models.native5b import Native5BConfig, NativeWM3D5B
from wm3d_v3.training.scale5b_checkpoint import (
    Native5BCheckpointManager,
    ResumeExpectations,
)
from wm3d_v3.training.scale5b_runtime import (
    apply_fsdp2,
    destroy_distributed,
    initialize_adamw_state,
    initialize_distributed,
)


def _config() -> Native5BConfig:
    return Native5BConfig(
        T=3,
        P=4,
        K=2,
        token_dim=16,
        task_dim=12,
        num_views=3,
        state_hidden=64,
        state_layers=4,
        state_heads=4,
        state_ff_mult=2,
        action_hidden=48,
        action_layers=3,
        action_heads=4,
        action_ff_mult=2,
        bridge_layers_state=(1, 3),
        bridge_heads=4,
        view_hidden=32,
        view_heads=4,
        view_ff_mult=2,
        max_action_groups=3,
        max_action_dim=4,
        action_substeps=2,
        max_group_id=8,
        max_embodiments=4,
        memory_dim=10,
        memory_every_state_layers=2,
        max_aux_tokens=3,
        aux_dim=6,
        max_aux_type_id=2,
        rgb_hidden=32,
        rgb_size=8,
        rgb_decode_indices=(0, 1),
        geom_hidden=24,
        activation_checkpointing=True,
    )


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    batch = 1
    return {
        "world_tokens": torch.randn(batch, 3, 3, 4, 16, device=device),
        "view_mask": torch.ones(batch, 3, 3, dtype=torch.bool, device=device),
        "task_embedding": torch.randn(batch, 12, device=device),
        "context_action_values": torch.randn(batch, 3, 3, 2, 4, device=device),
        "context_action_dim_mask": torch.ones(
            batch, 3, 3, 2, 4, dtype=torch.bool, device=device
        ),
        "future_factual_action_values": torch.randn(
            batch, 2, 3, 2, 4, device=device
        ),
        "future_factual_action_dim_mask": torch.ones(
            batch, 2, 3, 2, 4, dtype=torch.bool, device=device
        ),
        "action_group_ids": torch.tensor([[0, 1, 2]], device=device),
        "action_group_mask": torch.ones(
            batch, 3, dtype=torch.bool, device=device
        ),
        "embodiment_ids": torch.zeros(batch, dtype=torch.long, device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    context = initialize_distributed(timeout_minutes=5)
    try:
        if context.world_size != 2:
            raise RuntimeError("this smoke requires exactly two ranks")
        torch.manual_seed(91)
        torch.cuda.manual_seed(91)
        with torch.device(context.device):
            model = NativeWM3D5B(_config())
        apply_fsdp2(model, context, shard_degree=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, foreach=False)
        initialize_adamw_state(optimizer)
        inputs = _inputs(context.device)
        output = model(**inputs)
        loss = output["pred_tokens"].float().square().mean()
        loss = loss + output["action_mean"].float().square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        root_value: list[str | None] = [None]
        if context.is_rank0:
            if args.root is None:
                root_value[0] = tempfile.mkdtemp(prefix="wm3d_v7_fsdp2_smoke_")
            else:
                args.root.mkdir(parents=True, exist_ok=False)
                root_value[0] = str(args.root)
        dist.broadcast_object_list(root_value, src=0)
        root = Path(str(root_value[0]))
        manager = Native5BCheckpointManager(root)
        path = manager.save(
            step=1,
            model=model,
            optimizer=optimizer,
            metadata={
                "run_name": "native5b_fsdp2_smoke",
                "run_lineage": "smoke-lineage",
                "config_sha256": "1" * 64,
                "dataset_receipt_sha256": "2" * 64,
                "dataset_contract_sha256": "3" * 64,
                "initial_seed": 91,
                "shard_degree": 2,
                "global_batch_size": 2,
                "parameter_count": model.parameter_counts()["total"],
            },
        )
        before = model(**inputs)["pred_tokens"].detach().float()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.25)
        manager.load(
            path=path,
            model=model,
            optimizer=optimizer,
            expected=ResumeExpectations(
                step=1,
                run_lineage="smoke-lineage",
                config_sha256="1" * 64,
                dataset_receipt_sha256="2" * 64,
                world_size=2,
                shard_degree=2,
            ),
        )
        after = model(**inputs)["pred_tokens"].detach().float()
        difference = (before - after).abs().max()
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        if float(difference) != 0.0:
            raise AssertionError(f"checkpoint restore difference {float(difference)}")
        if context.is_rank0:
            metadata = manager.verify(path)
            print(
                json.dumps(
                    {
                        "pass": True,
                        "checkpoint": str(path),
                        "step": metadata["step"],
                        "max_restore_difference": float(difference),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
