from .native_world_model import (
    NATIVE_WORLD_MODEL_SCHEMA,
    NativeWorldModel,
    NativeWorldModelConfig,
    native_config_from_mapping,
)

# These additions do not hide legacy submodules; explicit imports such as
# ``wm3d_v3.models.joint_model`` remain supported for sealed V8 checkpoints.
__all__ = [
    "NATIVE_WORLD_MODEL_SCHEMA",
    "NativeWorldModel",
    "NativeWorldModelConfig",
    "native_config_from_mapping",
]
