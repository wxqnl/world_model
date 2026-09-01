"""Real two-rank FSDP2 regression for rank-asymmetric RGB-action negatives."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule

from wm3d.data.grouped_robot import COMPOSITION_OPERATOR_IDS
from wm3d.training import pretrain
from wm3d.training.distributed_runtime import (
    DistributedStrategyConfig,
    destroy_distributed,
    initialize_distributed,
    wrap_model,
)
from wm3d.training.native_objective import NativeObjectiveConfig, compute_native_objective


class _TinyActionRGBModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False)
        self.forward_batch_sizes: list[int] = []

    def iter_fsdp_units(self):
        return iter((self.projection,))

    def forward(self, fine: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_batch_sizes.append(int(fine.shape[0]))
        mask_float = mask.to(fine.dtype)
        signal = (fine * mask_float).flatten(1).sum(1) / mask_float.flatten(
            1
        ).sum(1).clamp_min(1)
        response = self.projection(signal[:, None].expand(-1, 8)).mean(-1)
        rgb = response[:, None, None, None, None, None].expand(-1, 1, 1, 3, 2, 2)
        policy = response[:, None, None, None] * 0.0
        return {
            "pred_tokens": response[:, None, None, None],
            "rgb": rgb,
            "policy_action_raw": policy,
            "policy_action_normalized": policy,
            "policy_action": policy,
            "policy_action_mask": torch.zeros_like(policy, dtype=torch.bool),
            "policy_gripper_mask": torch.zeros_like(policy, dtype=torch.bool),
            "policy_binary_mask": torch.zeros_like(policy, dtype=torch.bool),
            "policy_query_dt": torch.ones(
                fine.shape[0], 1, 1, device=fine.device, dtype=fine.dtype
            ),
        }


def _tiny_forward(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    appearance_teacher_ratio: float,
    compute_zero_action_control: bool = False,
) -> dict[str, torch.Tensor]:
    assert appearance_teacher_ratio == 0.0
    assert not compute_zero_action_control
    return model(
        batch["future_factual_fine_action_values"],
        batch["future_factual_fine_action_mask"],
    )


def _batch(rank: int, device: torch.device) -> dict[str, torch.Tensor]:
    b = 2
    action = (
        torch.tensor([0.0, 0.2], device=device)
        if rank == 0
        else torch.zeros(b, device=device)
    )
    fine = action.reshape(b, 1, 1, 1, 1)
    coarse = action.reshape(b, 1, 1, 1)
    policy = torch.zeros(b, 1, 1, 1, device=device)
    return {
        "future_factual_fine_action_values": fine,
        "future_factual_fine_action_mask": torch.ones_like(fine, dtype=torch.bool),
        "future_factual_fine_action_dt": torch.ones(b, 1, 1, 1, device=device),
        "future_factual_fine_sample_mask": torch.ones(
            b, 1, 1, 1, device=device, dtype=torch.bool
        ),
        "future_factual_coarse_action_values": coarse,
        "future_factual_coarse_action_mask": torch.ones_like(coarse, dtype=torch.bool),
        "action_group_ids": torch.ones(b, 1, device=device, dtype=torch.long),
        "action_group_mask": torch.ones(b, 1, device=device, dtype=torch.bool),
        "action_semantic_ids": torch.ones(b, 1, 1, device=device, dtype=torch.long),
        "embodiment_ids": torch.ones(b, device=device, dtype=torch.long),
        "action_normalization_offset": torch.zeros(b, 1, 1, device=device),
        "action_normalization_scale": torch.ones(b, 1, 1, device=device),
        "target_tokens": torch.zeros(b, 1, 1, 1, device=device),
        "target_rgb": torch.ones(b, 1, 1, 3, 2, 2, device=device),
        "target_rgb_mask": torch.ones(
            b, 1, 1, 1, 1, 1, device=device, dtype=torch.bool
        ),
        "context_rgb": torch.zeros(b, 1, 3, 2, 2, device=device),
        "context_rgb_mask": torch.ones(b, 1, device=device, dtype=torch.bool),
        "target_fine_action": policy,
        "target_fine_action_mask": torch.zeros_like(policy, dtype=torch.bool),
        "future_world_boundaries_dt": torch.tensor(
            [[0.0, 1.0]], device=device
        ).expand(b, -1),
        "composition_operator_ids": torch.full(
            (b, 1, 1),
            COMPOSITION_OPERATOR_IDS["last"],
            device=device,
            dtype=torch.long,
        ),
        "target_coarse_action_normalized": torch.zeros(b, 1, 1, 1, device=device),
        "target_coarse_action_mask": torch.zeros(
            b, 1, 1, 1, device=device, dtype=torch.bool
        ),
    }


def _objective() -> NativeObjectiveConfig:
    return NativeObjectiveConfig(
        token_mse=0.0,
        token_cosine=0.0,
        rgb_l1=0.0,
        rgb_charbonnier=0.0,
        rgb_gradient=0.0,
        depth_log=0.0,
        point=0.0,
        camera_pose=0.0,
        action_fine=0.0,
        action_coarse=0.0,
        context_pixel_action_rank_weight=2.0,
        context_pixel_action_separation_weight=0.5,
        context_pixel_action_rank_batch_size=1,
        context_pixel_action_rank_margin=10.0,
        context_pixel_action_separation_margin=10.0,
        context_pixel_action_negative_min_distance=0.05,
    )


def _worker(receipt: Path) -> None:
    strategy = DistributedStrategyConfig(
        strategy="fsdp2",
        shard_degree=2,
        initialization="meta_sharded",
        param_dtype="bf16",
        reduce_dtype="fp32",
        output_dtype="bf16",
        reshard_after_forward=True,
        timeout_minutes=2,
    )
    context = initialize_distributed(strategy)
    original_forward = pretrain._forward
    try:
        with torch.device("meta"):
            raw_model = _TinyActionRGBModel()
        model = wrap_model(
            raw_model, context, strategy, initialization_seed=9137
        ).model
        assert isinstance(model, FSDPModule)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.001)
        pretrain._forward = _tiny_forward
        batch = _batch(context.rank, context.device)
        objective = _objective()
        output = dict(
            pretrain._forward_with_action_counterfactual(
                model,
                batch,
                appearance_teacher_ratio=0.0,
                objective=objective,
                step=0,
                diagnostic_force_context_pixel_action=True,
            )
        )
        assert model.forward_batch_sizes == [2, 1]
        expected_valid = context.rank == 0
        assert bool(output["shuffled_action_valid"].item()) is expected_valid
        output["rgb"].retain_grad()
        output["shuffled_action_rgb"].retain_grad()
        losses = compute_native_objective(output=output, batch=batch, config=objective)
        rank_loss = losses["context_pixel_action_rank"]
        separation_loss = losses["context_pixel_action_separation"]
        if expected_valid:
            assert float(rank_loss) > 0.0 and float(separation_loss) > 0.0
        else:
            assert float(rank_loss) == 0.0
            assert float(separation_loss) == 0.0
            assert float(losses["total"]) == 0.0
        losses["total"].backward()
        primary_grad = output["rgb"].grad
        wrong_grad = output["shuffled_action_rgb"].grad
        assert primary_grad is not None and wrong_grad is not None
        primary_grad_max = float(primary_grad.abs().max())
        wrong_grad_max = float(wrong_grad.abs().max())
        if expected_valid:
            assert primary_grad_max > 0.0 and wrong_grad_max > 0.0
        else:
            assert primary_grad_max == 0.0 and wrong_grad_max == 0.0
        parameter_grads = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        assert parameter_grads
        assert all(
            bool(torch.isfinite(gradient).all())
            for gradient in parameter_grads
        )
        marker = torch.tensor(float(context.rank + 1), device=context.device)
        dist.all_reduce(marker)
        assert float(marker) == 3.0
        local = {
            "rank": context.rank,
            "forward_batch_sizes": model.forward_batch_sizes,
            "valid_negative": expected_valid,
            "rank_loss": float(rank_loss.detach()),
            "separation_loss": float(separation_loss.detach()),
            "primary_rgb_grad_max": primary_grad_max,
            "wrong_rgb_grad_max": wrong_grad_max,
            "all_reduce_sum": float(marker),
            "reshard_after_forward": strategy.reshard_after_forward,
        }
        gathered: list[dict[str, object] | None] = [None, None]
        dist.all_gather_object(gathered, local)
        if context.is_rank0:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(
                json.dumps(gathered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        pretrain._forward = original_forward
        destroy_distributed()


@pytest.mark.skipif(
    os.environ.get("WM3D_RUN_REAL_FSDP2_TESTS") != "1",
    reason="set WM3D_RUN_REAL_FSDP2_TESTS=1 for the real two-GPU test",
)
def test_rank_asymmetric_curriculum_keeps_fsdp2_collectives_aligned(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.update(
        {
            "NCCL_NVLS_ENABLE": "0",
            "NCCL_IB_DISABLE": "1",
            "PYTHONPATH": str(root),
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--worker-receipt",
            str(receipt),
        ],
        cwd=root,
        env=environment,
        check=True,
        timeout=180,
    )
    values = json.loads(receipt.read_text(encoding="utf-8"))
    assert [value["forward_batch_sizes"] for value in values] == [[2, 1], [2, 1]]
    assert values[0]["valid_negative"] is True
    assert values[0]["primary_rgb_grad_max"] > 0.0
    assert values[0]["wrong_rgb_grad_max"] > 0.0
    assert values[1]["valid_negative"] is False
    assert values[1]["rank_loss"] == 0.0
    assert values[1]["separation_loss"] == 0.0
    assert values[1]["primary_rgb_grad_max"] == 0.0
    assert values[1]["wrong_rgb_grad_max"] == 0.0
    assert all(value["all_reduce_sum"] == 3.0 for value in values)
    assert all(value["reshard_after_forward"] is True for value in values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-receipt", type=Path, required=True)
    _worker(parser.parse_args().worker_receipt)
