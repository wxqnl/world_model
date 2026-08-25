from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wm3d.data.grouped_normalization import (
    GROUPED_NORMALIZATION_ESTIMATOR,
    GROUPED_NORMALIZATION_SCHEMA,
    GROUPED_NORMALIZATION_SPLIT,
    GroupedNormalizationError,
    GroupedRobotNormalizer,
    normalize_grouped_masked,
    validate_grouped_lane_mask,
)
from wm3d.data.grouped_robot import (
    ACTION_SEMANTIC_IDS,
    STATE_SEMANTIC_IDS,
    bimanual_arm_spec,
)
from wm3d.data.manifest_contract import DataProfile, SourceSpec, canonical_sha256, sha256_file
from wm3d.models.native_world_model import NativeWorldModel, NativeWorldModelConfig
from wm3d.training.native_objective import NativeObjectiveError, objective_config_from_mapping


def _profile(tmp_path: Path) -> DataProfile:
    source = SourceSpec(
        name="source_a",
        adapter="fixture",
        raw_root=tmp_path,
        adapter_config_path=tmp_path / "adapter.yaml",
        adapter_contract_sha256="a" * 64,
        manifest_path=tmp_path / "manifest.jsonl",
        manifest_sha256="b" * 64,
        embodiment="bimanual_arm",
        weight=1,
        nominal_hours=None,
        license_id="fixture",
    )
    return DataProfile(
        path=tmp_path / "profile.yaml",
        profile_sha256="c" * 64,
        name="fixture",
        sources=(source,),
        embodiments={"bimanual_arm": bimanual_arm_spec()},
        cache_representation={},
        cache={},
    )


def _artifact(profile: DataProfile) -> dict:
    rows = []
    source = profile.sources[0]
    embodiment = profile.embodiments[source.embodiment]
    for group in embodiment.groups:
        for kind, semantics in (
            ("action", group.action_semantics),
            ("state", group.state_semantics),
        ):
            for dimension, semantic in enumerate(semantics):
                identity = "gripper" in semantic or semantic in {
                    "binary_contact",
                    "controller_mode",
                }
                rows.append(
                    {
                        "kind": kind,
                        "source": source.name,
                        "source_id": 0,
                        "embodiment": embodiment.name,
                        "embodiment_id": embodiment.embodiment_id,
                        "group": group.name,
                        "group_id": group.group_id,
                        "dimension": dimension,
                        "semantic": semantic,
                        "semantic_id": (
                            ACTION_SEMANTIC_IDS[semantic]
                            if kind == "action"
                            else STATE_SEMANTIC_IDS[semantic]
                        ),
                        "lane": "fine_command" if kind == "action" else "current_state",
                        "transform": "identity" if identity else "zscore",
                        "count": 4,
                        "observed_mean": 0.5 if identity else float(dimension + 1),
                        "observed_std": 0.5 if identity else 2.0,
                        "observed_min": 0.0 if identity else float(dimension - 3),
                        "observed_max": 1.0 if identity else float(dimension + 5),
                        "offset": 0.0 if identity else float(dimension + 1),
                        "scale": 1.0 if identity else 2.0,
                    }
                )
    return {
        "schema": GROUPED_NORMALIZATION_SCHEMA,
        "estimator": GROUPED_NORMALIZATION_ESTIMATOR,
        "split": GROUPED_NORMALIZATION_SPLIT,
        "minimum_scale": 1.0e-4,
        "data_profile_path": str(profile.path),
        "data_profile_sha256": profile.profile_sha256,
        "model_profile_sha256": "d" * 64,
        "window_index_path": str(profile.path.parent / "windows.jsonl"),
        "window_index_sha256": "e" * 64,
        "train_window_count_by_source": {"source_a": 2},
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }


def test_artifact_is_source_embodiment_group_dimension_bound_and_mask_aware(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    artifact = _artifact(profile)
    path = tmp_path / "normalization.json"
    path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
    normalizer = GroupedRobotNormalizer.load(
        path,
        expected_sha256=sha256_file(path),
        expected_data_profile_sha256=profile.profile_sha256,
        expected_model_profile_sha256="d" * 64,
        expected_window_index_sha256="e" * 64,
        data_profile=profile,
    )
    embodiment = profile.embodiments["bimanual_arm"]
    group_ids = torch.tensor([1, 2])
    action_ids = torch.tensor(
        [[ACTION_SEMANTIC_IDS[name] for name in group.action_semantics] for group in embodiment.groups]
    )
    state_ids = torch.tensor(
        [[STATE_SEMANTIC_IDS[name] for name in group.state_semantics] for group in embodiment.groups]
    )
    tensors = normalizer.tensors_for(
        source="source_a",
        embodiment_id=embodiment.embodiment_id,
        group_ids=group_ids,
        action_semantic_ids=action_ids,
        state_semantic_ids=state_ids,
    )
    assert tensors.fine_action_offset[:, -1].eq(0).all()
    assert tensors.fine_action_scale[:, -1].eq(1).all()
    assert tensors.fine_action_available.all()
    assert not tensors.coarse_action_available.any()
    values = torch.zeros(1, 1, 2, 7)
    mask = torch.zeros_like(values, dtype=torch.bool)
    values[0, 0, 0, 0] = 3.0
    values[0, 0, 0, 1] = 99.0
    values[0, 0, 0, 6] = 1.0
    mask[0, 0, 0, 0] = True
    mask[0, 0, 0, 6] = True
    normalized = normalize_grouped_masked(
        values,
        mask,
        offset=tensors.fine_action_offset,
        scale=tensors.fine_action_scale,
        group_axis=2,
    )
    assert normalized[0, 0, 0, 0].item() == 1.0
    assert normalized[0, 0, 0, 1].item() == 0.0
    assert normalized[0, 0, 0, 6].item() == 1.0


@pytest.mark.parametrize("field", ["source_id", "embodiment_id", "group_id", "semantic_id"])
def test_artifact_identity_drift_fails_closed(tmp_path: Path, field: str) -> None:
    profile = _profile(tmp_path)
    artifact = _artifact(profile)
    artifact["rows"][0][field] = int(artifact["rows"][0][field]) + 1
    artifact["rows_sha256"] = canonical_sha256(artifact["rows"])
    with pytest.raises(GroupedNormalizationError):
        GroupedRobotNormalizer(artifact, data_profile=profile)


def test_gripper_zscore_is_forbidden(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    artifact = _artifact(profile)
    row = next(row for row in artifact["rows"] if "gripper" in row["semantic"])
    row.update(transform="zscore", offset=0.5, scale=0.5)
    artifact["rows_sha256"] = canonical_sha256(artifact["rows"])
    with pytest.raises(GroupedNormalizationError, match="transform"):
        GroupedRobotNormalizer(artifact, data_profile=profile)


def test_continuous_scale_tamper_fails_closed(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    artifact = _artifact(profile)
    row = next(row for row in artifact["rows"] if row["transform"] == "zscore")
    row["scale"] = float(row["scale"]) + 0.25
    artifact["rows_sha256"] = canonical_sha256(artifact["rows"])
    with pytest.raises(GroupedNormalizationError, match="offset/scale"):
        GroupedRobotNormalizer(artifact, data_profile=profile)


def test_valid_mask_cannot_cross_action_lane() -> None:
    available = torch.tensor([[True, True], [False, False]])
    mask = torch.zeros(1, 2, 3, 2, dtype=torch.bool)
    mask[0, 1, 0, 0] = True
    with pytest.raises(GroupedNormalizationError, match="outside declared"):
        validate_grouped_lane_mask(
            mask,
            available=available,
            group_axis=1,
            lane="fine_command",
        )


def test_explicitly_unsupervised_coordinate_has_no_stats_or_availability(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    artifact = _artifact(profile)
    row = next(
        item
        for item in artifact["rows"]
        if item["kind"] == "action" and item["dimension"] == 0
    )
    row.update(
        count=0,
        observed_mean=0.0,
        observed_std=0.0,
        observed_min=0.0,
        observed_max=0.0,
        offset=0.0,
        scale=1.0,
    )
    artifact["rows_sha256"] = canonical_sha256(artifact["rows"])
    normalizer = GroupedRobotNormalizer(artifact, data_profile=profile)
    embodiment = profile.embodiments["bimanual_arm"]
    tensors = normalizer.tensors_for(
        source="source_a",
        embodiment_id=embodiment.embodiment_id,
        group_ids=torch.tensor([group.group_id for group in embodiment.groups]),
        action_semantic_ids=torch.tensor(
            [
                [ACTION_SEMANTIC_IDS[name] for name in group.action_semantics]
                for group in embodiment.groups
            ]
        ),
        state_semantic_ids=torch.tensor(
            [
                [STATE_SEMANTIC_IDS[name] for name in group.state_semantics]
                for group in embodiment.groups
            ]
        ),
    )
    assert not tensors.fine_action_available[0, 0]
    assert tensors.fine_action_available[0, 1:].all()
    mask = torch.zeros(1, 2, 1, 7, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    with pytest.raises(GroupedNormalizationError, match="outside declared"):
        validate_grouped_lane_mask(
            mask,
            available=tensors.fine_action_available,
            group_axis=1,
            lane="fine_command",
        )


def test_coarse_lane_has_distinct_statistics_and_no_fine_availability(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    artifact = _artifact(profile)
    for row in artifact["rows"]:
        if row["kind"] == "action":
            row["lane"] = "coarse_effect"
    artifact["rows_sha256"] = canonical_sha256(artifact["rows"])
    normalizer = GroupedRobotNormalizer(artifact, data_profile=profile)
    embodiment = profile.embodiments["bimanual_arm"]
    tensors = normalizer.tensors_for(
        source="source_a",
        embodiment_id=embodiment.embodiment_id,
        group_ids=torch.tensor([group.group_id for group in embodiment.groups]),
        action_semantic_ids=torch.tensor(
            [
                [ACTION_SEMANTIC_IDS[name] for name in group.action_semantics]
                for group in embodiment.groups
            ]
        ),
        state_semantic_ids=torch.tensor(
            [
                [STATE_SEMANTIC_IDS[name] for name in group.state_semantics]
                for group in embodiment.groups
            ]
        ),
    )
    assert not tensors.fine_action_available.any()
    assert tensors.coarse_action_available.all()
    assert tensors.coarse_action_offset[0, 0].item() == 1.0
    assert tensors.coarse_action_scale[0, 0].item() == 2.0


def test_action_head_trains_normalized_but_serves_physical_and_keeps_gripper() -> None:
    cfg = NativeWorldModelConfig(
        T=1,
        P=1,
        K=1,
        token_dim=8,
        task_dim=8,
        num_views=1,
        state_hidden=16,
        state_layers=1,
        state_heads=2,
        action_hidden=16,
        action_layers=1,
        action_heads=2,
        bridge_layers_state=(0,),
        bridge_heads=2,
        dynamics_layers=1,
        view_hidden=8,
        view_heads=2,
        max_action_groups=1,
        max_action_dim=7,
        max_state_dim=1,
        max_action_substeps=1,
        max_policy_queries=1,
        max_group_id=4,
        max_embodiments=4,
        max_action_semantic_id=32,
        max_state_semantic_id=8,
        time_fourier_dim=4,
        max_aux_tokens=1,
        aux_dim=2,
        max_aux_type_id=4,
        rgb_hidden=8,
        rgb_size=1,
        rgb_decode_indices=(0,),
        geom_hidden=8,
        activation_checkpointing=False,
    )
    model = NativeWorldModel(cfg)
    with torch.no_grad():
        model.action_head.output.weight.zero_()
        model.action_head.output.bias.zero_()
    query = torch.zeros(1, 1, 1, cfg.action_hidden)
    semantics = torch.tensor([[[1, 1, 1, 3, 3, 3, 4]]])
    result = model.action_head(
        query,
        semantics,
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.tensor([[[10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.0]]]),
        torch.tensor([[[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]]]),
    )
    torch.testing.assert_close(result["policy_action_normalized"][0, 0, 0, :6], torch.zeros(6))
    torch.testing.assert_close(
        result["policy_action"][0, 0, 0, :6],
        torch.tensor([10.0, 20.0, 30.0, 1.0, 2.0, 3.0]),
    )
    assert result["policy_action"][0, 0, 0, 6].item() == 0.5

    controller_semantics = semantics.clone()
    controller_semantics[..., 0] = ACTION_SEMANTIC_IDS["controller_mode"]
    controller_offset = torch.tensor(
        [[[0.0, 20.0, 30.0, 1.0, 2.0, 3.0, 0.0]]]
    )
    controller_scale = torch.tensor(
        [[[1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]]]
    )
    controller_result = model.action_head(
        query,
        controller_semantics,
        torch.ones(1, 1, 1, dtype=torch.bool),
        controller_offset,
        controller_scale,
    )
    assert controller_result["policy_binary_mask"][0, 0, 0, 0]
    assert controller_result["policy_action_normalized"][0, 0, 0, 0].item() == 0.5
    assert controller_result["policy_action"][0, 0, 0, 0].item() == 0.5

    invalid_offset = controller_offset.clone()
    invalid_offset[..., 0] = 1.0
    with pytest.raises(ValueError, match="identity"):
        model.action_head(
            query,
            controller_semantics,
            torch.ones(1, 1, 1, dtype=torch.bool),
            invalid_offset,
            controller_scale,
        )


def test_action_velocity_nonzero_fails_closed() -> None:
    with pytest.raises(NativeObjectiveError, match="action_velocity"):
        objective_config_from_mapping({"action_velocity": 0.1})
