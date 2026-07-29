from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import torch

from wm3d_v3.data.scale5b_sampler import (
    ExactSourceSchedule,
    StepAddressedBatchSampler,
)
from wm3d_v3.models.native5b import Native5BConfig, NativeWM3D5B
from wm3d_v3.training.scale5b_loss import Native5BLossConfig, native5b_loss
from wm3d_v3.training.scale5b_runtime import (
    RuntimeContractError,
    assert_v7_native_dependency_boundary,
)


def _tiny_config(*, activation_checkpointing: bool = False) -> Native5BConfig:
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
        activation_checkpointing=activation_checkpointing,
    )


def _tiny_inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "world_tokens": torch.randn(batch, 3, 3, 4, 16),
        "view_mask": torch.ones(batch, 3, 3, dtype=torch.bool),
        "task_embedding": torch.randn(batch, 12),
        "context_action_values": torch.randn(batch, 3, 3, 2, 4),
        "context_action_dim_mask": torch.ones(
            batch, 3, 3, 2, 4, dtype=torch.bool
        ),
        "future_factual_action_values": torch.randn(batch, 2, 3, 2, 4),
        "future_factual_action_dim_mask": torch.ones(
            batch, 2, 3, 2, 4, dtype=torch.bool
        ),
        "action_group_ids": torch.tensor([[0, 1, 2]]).expand(batch, -1),
        "action_group_mask": torch.ones(batch, 3, dtype=torch.bool),
        "embodiment_ids": torch.zeros(batch, dtype=torch.long),
        "aux_tokens": torch.randn(batch, 3, 3, 6),
        "aux_mask": torch.ones(batch, 3, 3, dtype=torch.bool),
    }


def test_default_parameter_budget_is_frozen() -> None:
    with torch.device("meta"):
        model = NativeWM3D5B(Native5BConfig())
    counts = model.parameter_counts()
    assert counts["total"] == 4_956_589_929
    assert counts["state_trunk"] == 3_250_831_360
    assert counts["action_trunk"] == 1_195_474_944
    assert counts["bridges"] == 424_719_360


@pytest.mark.parametrize("checkpointing", [False, True])
def test_multiview_forward_backward_and_native_losses(checkpointing: bool) -> None:
    torch.manual_seed(4)
    model = NativeWM3D5B(
        _tiny_config(activation_checkpointing=checkpointing)
    )
    inputs = _tiny_inputs()
    output = model(**inputs)
    action_mask = inputs["future_factual_action_dim_mask"]
    batch = {
        "target_tokens": torch.randn_like(output["pred_tokens"]),
        "target_rgb": torch.rand_like(output["rgb"]),
        "target_view_mask": torch.ones(
            output["depth"].shape[:3],
            dtype=torch.bool,
        ),
        "target_depth": torch.rand_like(output["depth"]) + 0.1,
        "target_point": torch.randn_like(output["point"]),
        "target_geometry_confidence": torch.rand_like(
            output["geometry_confidence"]
        ),
        "target_camera_pose": torch.randn_like(output["camera_pose"]),
        "target_action_values": torch.randn_like(output["action_mean"]),
        "target_action_dim_mask": action_mask,
        "target_contact": torch.zeros_like(output["contact_logit"]),
        "target_contact_mask": torch.ones_like(
            output["contact_logit"], dtype=torch.bool
        ),
        "action_group_mask": inputs["action_group_mask"],
    }
    losses = native5b_loss(output, batch, Native5BLossConfig())
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert sum(parameter.grad is not None for parameter in model.parameters()) > 100


def test_future_factual_action_does_not_leak_to_policy() -> None:
    torch.manual_seed(7)
    model = NativeWM3D5B(_tiny_config()).eval()
    inputs = _tiny_inputs(batch=1)
    factual = inputs["future_factual_action_values"].requires_grad_(True)
    inputs["future_factual_action_values"] = factual
    first = model(**inputs)
    second_inputs = dict(inputs)
    second_inputs["future_factual_action_values"] = factual.detach() + 3.0
    second = model(**second_inputs)
    assert torch.equal(first["action_mean"], second["action_mean"])
    assert not torch.equal(first["pred_tokens"], second["pred_tokens"])
    gradient = torch.autograd.grad(first["action_mean"].sum(), factual)[0]
    assert torch.count_nonzero(gradient) == 0


def test_step_addressed_sampler_exact_mix_disjoint_and_resumable() -> None:
    names = ("droid", "bridge", "atomic", "composite", "mg")
    weights = {
        "droid": 35,
        "bridge": 15,
        "atomic": 10,
        "composite": 20,
        "mg": 20,
    }
    spans = {name: (index * 10_000, (index + 1) * 10_000) for index, name in enumerate(names)}
    schedule = ExactSourceSchedule(names, weights, seed=11)
    assert Counter(schedule.address(step).source_name for step in range(100)) == Counter(
        weights
    )
    uninterrupted = [
        list(
            StepAddressedBatchSampler(
                spans,
                names,
                weights,
                world_size=8,
                rank=rank,
                micro_batch_size=2,
                gradient_accumulation=4,
                start_optimizer_step=300,
                num_optimizer_steps=20,
                seed=123,
            )
        )
        for rank in range(8)
    ]
    resumed = [
        list(
            StepAddressedBatchSampler(
                spans,
                names,
                weights,
                world_size=8,
                rank=rank,
                micro_batch_size=2,
                gradient_accumulation=4,
                start_optimizer_step=317,
                num_optimizer_steps=3,
                seed=123,
            )
        )
        for rank in range(8)
    ]
    for rank in range(8):
        assert resumed[rank] == uninterrupted[rank][17 * 4 : 20 * 4]
    for step_offset in range(3):
        values = [
            sample
            for rank in range(8)
            for micro in range(4)
            for sample in resumed[rank][step_offset * 4 + micro]
        ]
        assert len(values) == 64
        assert len(set(values)) == 64


def test_dependency_guard_rejects_later_architecture(tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text(
        "schema: wm3d_v7_native5b\nsha256: 0123456789a2abcdef\n",
        encoding="utf-8",
    )
    assert_v7_native_dependency_boundary([good])
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_backend: qwen3-vl\n", encoding="utf-8")
    with pytest.raises(RuntimeContractError):
        assert_v7_native_dependency_boundary([bad])
