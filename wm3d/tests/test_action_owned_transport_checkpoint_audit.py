from __future__ import annotations

import sys

import pytest
import torch

from scripts.tools import audit_action_owned_transport_checkpoint as audit
from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS


def _action_batch() -> dict[str, torch.Tensor]:
    batch, horizons, groups, substeps, dimensions = 2, 3, 1, 1, 2
    fine = torch.zeros(batch, horizons, groups, substeps, dimensions)
    coarse = torch.zeros(batch, horizons, groups, dimensions)
    fine[1] = 4.0
    coarse[1] = 3.0
    fine_mask = torch.ones_like(fine, dtype=torch.bool)
    coarse_mask = torch.ones_like(coarse, dtype=torch.bool)
    offset = torch.tensor([[[0.25, -0.5]]]).expand(batch, -1, -1).clone()
    scale = torch.tensor([[[0.5, 2.0]]]).expand(batch, -1, -1).clone()
    return {
        "source_id": torch.tensor([2, 2]),
        "future_factual_fine_action_values": fine,
        "future_factual_fine_action_mask": fine_mask,
        "future_factual_fine_action_dt": torch.ones(
            batch, horizons, groups, substeps
        ),
        "future_factual_fine_sample_mask": torch.ones(
            batch, horizons, groups, substeps, dtype=torch.bool
        ),
        "future_factual_coarse_action_values": coarse,
        "future_factual_coarse_action_mask": coarse_mask,
        "action_group_ids": torch.ones(batch, groups, dtype=torch.long),
        "action_group_mask": torch.ones(batch, groups, dtype=torch.bool),
        "action_semantic_ids": torch.full(
            (batch, groups, dimensions),
            ACTION_SEMANTIC_IDS["delta_position_m"],
            dtype=torch.long,
        ),
        "embodiment_ids": torch.full((batch,), 9, dtype=torch.long),
        "action_normalization_offset": offset,
        "action_normalization_scale": scale,
    }


def test_cli_defaults_to_val_and_allows_explicit_train(monkeypatch) -> None:
    base = [
        "audit",
        "--runtime",
        "runtime.yaml",
        "--checkpoint",
        "step_00000100",
        "--output",
        "receipt.json",
    ]
    monkeypatch.setattr(sys, "argv", base)
    assert audit.parse_args().split == "val"
    monkeypatch.setattr(sys, "argv", [*base, "--split", "train"])
    assert audit.parse_args().split == "train"


def test_action_variants_use_physical_noop_and_compatible_distant_pair() -> None:
    batch = _action_batch()
    variants, permutation, valid, distance = audit.build_action_variants(
        batch, step=100, minimum_distance=0.05
    )

    assert valid.tolist() == [True, True]
    assert permutation.tolist() == [1, 0]
    assert bool((distance > 0.05).all())
    expected_noop = (
        -batch["action_normalization_offset"]
        / batch["action_normalization_scale"]
    )
    torch.testing.assert_close(
        variants["physical_noop"]["future_factual_fine_action_values"],
        expected_noop[:, None, :, None, :].expand_as(
            batch["future_factual_fine_action_values"]
        ),
    )
    torch.testing.assert_close(
        variants["distant_mismatch"]["future_factual_fine_action_values"][0],
        batch["future_factual_fine_action_values"][1],
    )
    assert variants["normal"]["future_factual_fine_action_values"] is batch[
        "future_factual_fine_action_values"
    ]


def _metric_batch() -> dict[str, torch.Tensor]:
    batch, horizons, patches, hidden, views, size = 1, 3, 2, 2, 1, 4
    target_tokens = torch.arange(
        batch * horizons * patches * hidden, dtype=torch.float32
    ).reshape(batch, horizons, patches, hidden) / 10.0
    context = torch.zeros(batch, views, 3, size, size)
    target_rgb = torch.stack(
        [
            torch.full((batch, views, 3, size, size), value)
            for value in (0.1, 0.2, 0.3)
        ],
        dim=1,
    )
    target_rgb[..., :2] = 0.0
    flow = torch.ones(batch, horizons, views, 2, 2, 2)
    return {
        "target_tokens": target_tokens,
        "target_token_mask": torch.ones(
            batch, horizons, patches, dtype=torch.bool
        ),
        "target_rgb": target_rgb,
        "target_rgb_mask": torch.ones(
            batch, horizons, views, 1, 1, 1, dtype=torch.bool
        ),
        "context_rgb": context,
        "context_rgb_mask": torch.ones(batch, views, dtype=torch.bool),
        "rgb_flow_target_pixels": flow,
        "rgb_disocclusion_target": torch.zeros(
            batch, horizons, views, 1, 2, 2
        ),
    }


def _output(batch: dict[str, torch.Tensor], *, scale: float) -> dict[str, torch.Tensor]:
    target_rgb = batch["target_rgb"]
    return {
        "pred_tokens": batch["target_tokens"] * scale,
        "rgb": target_rgb * scale,
        "rgb_flow_pixels": torch.nn.functional.interpolate(
            batch["rgb_flow_target_pixels"].reshape(-1, 2, 2, 2),
            size=target_rgb.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).reshape(1, 3, 1, 2, 4, 4)
        * scale,
        "policy_action_raw": torch.tensor([[1.0, 2.0]]),
        "action_free_pred_tokens": torch.tensor([[[1.0, 2.0]]]),
    }


def test_metrics_summary_reports_quality_response_and_strict_policy_isolation() -> None:
    batch = _metric_batch()
    normal = _output(batch, scale=1.0)
    noop = _output(batch, scale=0.0)
    wrong = _output(batch, scale=-1.0)
    variants = {
        "normal": audit.variant_metrics(normal, batch, motion_threshold=0.03),
        "physical_noop": audit.variant_metrics(
            noop, batch, motion_threshold=0.03
        ),
        "distant_mismatch": audit.variant_metrics(
            wrong, batch, motion_threshold=0.03
        ),
    }
    invariants = {
        "physical_noop": audit._policy_invariants(normal, noop),
        "distant_mismatch": audit._policy_invariants(normal, wrong),
    }
    record = {
        "source_name": "source_a",
        "variants": variants,
        "responses": {
            "physical_noop": audit._response_rms(normal, noop, batch),
            "distant_mismatch": audit._response_rms(normal, wrong, batch),
        },
        "invariants": invariants,
    }
    summary = audit.summarize([record])

    assert variants["normal"]["rgb_l1"] == 0.0
    assert variants["normal"]["p64_error_rms"] == 0.0
    assert variants["normal"]["flow_epe_pixels"] == 0.0
    assert variants["normal"]["rgb_motion_fraction"] == 0.5
    assert summary["gains"]["rgb_l1_normal_vs_physical_noop"] > 0.0
    assert summary["gains"]["rgb_l1_normal_vs_distant_mismatch"] > 0.0
    assert summary["responses"]["physical_noop"]["rgb_response_rms"] > 0.0
    assert summary["all_policy_invariants_passed"] is True


def test_metrics_exclude_views_without_valid_context_rgb() -> None:
    batch = _metric_batch()
    batch["context_rgb_mask"].zero_()

    with pytest.raises(audit.AuditError, match="lacks RGB motion or static support"):
        audit.variant_metrics(_output(batch, scale=1.0), batch, motion_threshold=0.03)


def test_policy_invariant_detects_future_action_leak() -> None:
    batch = _metric_batch()
    factual = _output(batch, scale=1.0)
    leaked = _output(batch, scale=0.5)
    leaked["policy_action_raw"] = factual["policy_action_raw"] + 1.0

    result = audit._policy_invariants(factual, leaked)

    assert result["policy_action_raw_equal"] is False
    assert result["action_free_tokens_equal"] is True


def test_summary_weights_sources_equally_when_pair_counts_differ() -> None:
    def record(source: str, normal: float, control: float) -> dict[str, object]:
        variants = {
            label: {"rgb_l1": value, "p64_error_rms": value}
            for label, value in (
                ("normal", normal),
                ("physical_noop", control),
                ("distant_mismatch", control),
            )
        }
        responses = {
            label: {"rgb_response_rms": control - normal}
            for label in ("physical_noop", "distant_mismatch")
        }
        invariants = {
            label: {
                "policy_action_raw_equal": True,
                "action_free_tokens_equal": True,
            }
            for label in ("physical_noop", "distant_mismatch")
        }
        return {
            "source_name": source,
            "variants": variants,
            "responses": responses,
            "invariants": invariants,
        }

    summary = audit.summarize(
        [
            record("many_pairs", 0.0, 1.0),
            record("many_pairs", 0.0, 1.0),
            record("many_pairs", 0.0, 1.0),
            record("one_pair", 10.0, 14.0),
        ]
    )

    assert summary["variants"]["normal"]["rgb_l1"] == 5.0
    assert summary["gains"]["rgb_l1_normal_vs_physical_noop"] == 2.5
    assert summary["positive_source_counts"][
        "rgb_l1_normal_vs_physical_noop"
    ] == 2
    assert summary["per_source"]["many_pairs"]["pair_count"] == 3


def test_pre_materialization_k8_validation_accepts_direct_raw_fields() -> None:
    batch = _action_batch()
    batch["rgb_frame_indices"] = torch.arange(8).unsqueeze(0).expand(2, -1)
    for name in tuple(batch):
        value = batch[name]
        if name.startswith("future_factual_") and value.ndim >= 2:
            batch[name] = value[:, :1].expand(-1, 8, *value.shape[2:]).clone()

    audit.validate_action_k8_batch(batch)


def test_materialized_k8_validation_requires_targets() -> None:
    batch = {
        name: torch.ones(1, 8)
        for name in (
            "target_tokens",
            "target_token_mask",
            "target_rgb",
            "target_rgb_mask",
            "future_factual_fine_action_values",
            "future_factual_fine_action_mask",
            "future_factual_fine_action_dt",
            "future_factual_fine_sample_mask",
            "future_factual_coarse_action_values",
            "future_factual_coarse_action_mask",
        )
    }
    batch["rgb_frame_indices"] = torch.arange(8).unsqueeze(0)
    audit.validate_materialized_k8_batch(batch)

    batch["future_factual_fine_action_values"] = torch.ones(1, 7)
    try:
        audit.validate_materialized_k8_batch(batch)
    except audit.AuditError as error:
        assert "K8" in str(error)
    else:
        raise AssertionError("short action horizon passed K8 validation")


def test_post_materialization_k8_validation_rejects_missing_targets() -> None:
    batch = _action_batch()
    for name in tuple(batch):
        value = batch[name]
        if name.startswith("future_factual_") and value.ndim >= 2:
            batch[name] = value[:, :1].expand(-1, 8, *value.shape[2:]).clone()

    try:
        audit.validate_materialized_k8_batch(batch)
    except audit.AuditError as error:
        assert "target_tokens" in str(error)
    else:
        raise AssertionError("missing materialized targets passed K8 validation")


def test_candidate_plan_scans_addresses_without_loading_batches() -> None:
    class Describer:
        def __init__(self) -> None:
            self.sources = ("a", "a", "b", "c", "d", "a", "b", "c")
            self.calls: list[int] = []

        def describe_step(self, optimizer_step: int) -> dict[str, object]:
            self.calls.append(optimizer_step)
            return {"source_name": self.sources[optimizer_step]}

    describer = Describer()
    result = audit.plan_source_candidate_steps(
        describer, max_steps=8, source_count=3
    )

    assert result == {
        "a": [0, 1, 5],
        "b": [2, 6],
        "c": [3, 7],
        "d": [4],
    }
    assert describer.calls == list(range(8))


def test_candidate_plan_keeps_fourth_source_if_an_earlier_source_fails() -> None:
    class Describer:
        sources = ("no_motion", "valid_b", "valid_c", "valid_d")

        def describe_step(self, optimizer_step: int) -> dict[str, object]:
            return {"source_name": self.sources[optimizer_step]}

    result = audit.plan_source_candidate_steps(
        Describer(), max_steps=4, source_count=3
    )
    successful = {
        source for source in result if source != "no_motion"
    }

    assert tuple(result) == ("no_motion", "valid_b", "valid_c", "valid_d")
    assert len(successful) == 3
