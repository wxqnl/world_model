"""Single construction path shared by training, evaluation, and serving."""
from __future__ import annotations

from dataclasses import fields
from typing import Any

from .action_stream import ActionConfig
from .dual_stream import DualConfig
from .joint_model import JointConfig, JointWorldModel
from .state_stream import StateConfig


_TUPLE_FIELDS = {
    "policy_waypoint_active_stages",
    "policy_oft_grip_indices",
    "policy_oft_adapters",
}


def build_joint_config(model_cfg: dict[str, Any]) -> JointConfig:
    """Build `JointConfig` without duplicating policy fields across entrypoints."""
    state = StateConfig(**model_cfg["state"])
    action = ActionConfig(**model_cfg["action"])
    dual = DualConfig(
        state=state,
        action=action,
        xattn_layers_state=tuple(model_cfg["xattn_layers_state"]),
        xattn_n_heads=model_cfg["xattn_n_heads"],
    )
    valid_fields = {item.name for item in fields(JointConfig)} - {"dual"}
    kwargs: dict[str, Any] = {
        key: value for key, value in model_cfg.items() if key in valid_fields
    }
    for key in _TUPLE_FIELDS:
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    kwargs.setdefault("policy_flow_hidden", model_cfg.get("policy_hidden", 768))
    kwargs.setdefault("world_prior_hidden", model_cfg["state"].get("hidden", 768))
    kwargs.setdefault("world_prior_heads", model_cfg["state"].get("n_heads", 8))
    return JointConfig(dual=dual, **kwargs)


def build_joint_world_model(model_cfg: dict[str, Any]) -> JointWorldModel:
    return JointWorldModel(build_joint_config(model_cfg))
