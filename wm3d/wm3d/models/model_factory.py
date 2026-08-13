"""Single native-model construction path for training, evaluation and serving."""
from __future__ import annotations

import math
from typing import Any, Mapping

from wm3d.data.grouped_robot import ACTION_SEMANTIC_IDS, STATE_SEMANTIC_IDS

from .native_world_model import (
    NativeWorldModel,
    native_config_from_mapping,
)


def validate_model_profile(profile: Mapping[str, Any]) -> None:
    allowed = {
        "schema",
        "name",
        "architecture",
        "model",
        "sampling",
        "expected_parameter_count",
    }
    unknown = sorted(set(profile) - allowed)
    if unknown:
        raise ValueError(f"unknown model profile fields: {unknown}")
    if profile.get("schema") != "wm3d_v8_model_profile_v1":
        raise ValueError("model profile schema must be wm3d_v8_model_profile_v1")
    if not str(profile.get("name", "")):
        raise ValueError("model profile name must be non-empty")
    sampling = profile.get("sampling")
    required_sampling = {
        "mode",
        "history_action_leading_boundary",
        "context_horizon_seconds",
        "future_horizon_seconds",
        "minimum_horizon_coverage",
        "minimum_anchor_separation_seconds",
        "policy_target_horizon_seconds",
        "policy_training_times",
        "interpolation",
    }
    optional_sampling = {"future_offsets_seconds"}
    if (
        not isinstance(sampling, dict)
        or not required_sampling.issubset(sampling)
        or not set(sampling).issubset(required_sampling | optional_sampling)
    ):
        raise ValueError(
            "model sampling fields mismatch: "
            f"required={sorted(required_sampling)} "
            f"optional={sorted(optional_sampling)}"
        )
    if sampling["mode"] != "observed_monotonic_subsequence":
        raise ValueError("world sampling must use observed timestamps")
    if sampling["history_action_leading_boundary"] != "observed_previous_state":
        raise ValueError("history actions require a real leading state")
    if sampling["policy_training_times"] != "observed_action_timestamps":
        raise ValueError("policy targets must use recorded action timestamps")
    if sampling["interpolation"] != "forbidden":
        raise ValueError("world/action interpolation is forbidden")
    for name in (
        "context_horizon_seconds",
        "future_horizon_seconds",
        "minimum_anchor_separation_seconds",
        "policy_target_horizon_seconds",
    ):
        if float(sampling[name]) <= 0:
            raise ValueError(f"sampling.{name} must be positive")
    if not 0 < float(sampling["minimum_horizon_coverage"]) <= 1:
        raise ValueError("sampling.minimum_horizon_coverage must be in (0,1]")
    architecture = str(profile.get("architecture", ""))
    model_mapping = profile.get("model")
    if architecture != "native_world_model":
        raise ValueError(
            "WM3D release profiles require architecture=native_world_model"
        )
    if not isinstance(model_mapping, dict):
        raise ValueError("native_world_model profile requires a model mapping")
    config = native_config_from_mapping(model_mapping)
    if "future_offsets_seconds" in sampling:
        offsets = sampling["future_offsets_seconds"]
        if not isinstance(offsets, list) or len(offsets) != int(config.K):
            raise ValueError(
                "sampling.future_offsets_seconds must contain exactly model.K values"
            )
        numeric = [float(item) for item in offsets]
        if (
            any(not math.isfinite(item) for item in numeric)
            or any(item <= 0 for item in numeric)
            or any(right <= left for left, right in zip(numeric, numeric[1:]))
        ):
            raise ValueError(
                "sampling.future_offsets_seconds must be finite, positive and strictly increasing"
            )
        if not math.isclose(
            numeric[-1],
            float(sampling["future_horizon_seconds"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "last future offset must equal sampling.future_horizon_seconds"
            )


def validate_model_data_compatibility(
    profile: Mapping[str, Any], data_profile: Any
) -> None:
    """Reject a model/data pairing that would truncate a robot or cache.

    Capacity fields are padding ceilings, not robot-specific branches.  This
    check deliberately inspects every declared embodiment before window
    materialization so a bimanual/whole-body source cannot be silently
    skipped merely because a selected model profile is too small.
    """

    validate_model_profile(profile)
    if profile.get("architecture") != "native_world_model":
        raise ValueError("unified WM3D data requires architecture=native_world_model")
    model = profile["model"]
    cfg = native_config_from_mapping(model)
    representation = data_profile.cache_representation

    model_p = int(cfg.P)
    cache_p = int(representation["spatial_tokens"])
    model_grid = int(round(model_p**0.5))
    cache_grid = int(representation["token_grid"])
    if model_grid * model_grid != model_p or cache_grid * cache_grid != cache_p:
        raise ValueError("model/cache spatial token counts must be square grids")
    if model_grid > cache_grid:
        raise ValueError("model spatial grid exceeds the shared episode cache")
    for model_field, cache_field, label in (
        ("token_dim", "token_dim", "token dimension"),
        ("num_views", "num_views", "canonical view count"),
    ):
        if int(getattr(cfg, model_field)) != int(representation[cache_field]):
            raise ValueError(f"model/cache {label} mismatch")
    if int(cfg.rgb_size) > int(representation["rgb_size"]):
        raise ValueError("model RGB target exceeds the shared episode cache")

    max_groups = int(cfg.max_action_groups)
    max_action_dim = int(cfg.max_action_dim)
    max_state_dim = int(cfg.max_state_dim)
    max_group_id = int(cfg.max_group_id)
    max_embodiments = int(cfg.max_embodiments)
    max_action_semantic = int(cfg.max_action_semantic_id)
    max_state_semantic = int(cfg.max_state_semantic_id)
    for embodiment in data_profile.embodiments.values():
        if len(embodiment.groups) > max_groups:
            raise ValueError(
                f"embodiment {embodiment.name!r} has {len(embodiment.groups)} "
                f"groups, model capacity is {max_groups}"
            )
        if int(embodiment.embodiment_id) >= max_embodiments:
            raise ValueError(
                f"embodiment id {embodiment.embodiment_id} is outside embedding "
                f"capacity {max_embodiments}"
            )
        for group in embodiment.groups:
            if group.action_dim > max_action_dim or group.state_dim > max_state_dim:
                raise ValueError(
                    f"group {embodiment.name}/{group.name} dimensions "
                    f"action={group.action_dim}, state={group.state_dim} exceed "
                    f"model capacities action={max_action_dim}, state={max_state_dim}"
                )
            if int(group.group_id) >= max_group_id:
                raise ValueError(
                    f"group id {group.group_id} is outside embedding capacity "
                    f"{max_group_id}"
                )
            action_ids = [ACTION_SEMANTIC_IDS[name] for name in group.action_semantics]
            state_ids = [STATE_SEMANTIC_IDS[name] for name in group.state_semantics]
            if action_ids and max(action_ids) >= max_action_semantic:
                raise ValueError("action semantic id exceeds model embedding capacity")
            if state_ids and max(state_ids) >= max_state_semantic:
                raise ValueError("state semantic id exceeds model embedding capacity")


def build_world_model(profile: Mapping[str, object]) -> NativeWorldModel:
    """Build the shared 1B/5B native model from an architecture profile."""

    validate_model_profile(profile)
    model_mapping = profile.get("model")
    if not isinstance(model_mapping, dict):
        raise ValueError("native_world_model profile requires a model mapping")
    model = NativeWorldModel(native_config_from_mapping(model_mapping))
    expected = profile.get("expected_parameter_count")
    if expected is not None:
        observed = sum(parameter.numel() for parameter in model.parameters())
        if observed != int(expected):
            raise ValueError(
                f"model parameter count {observed} != sealed expectation {expected}"
            )
    return model
