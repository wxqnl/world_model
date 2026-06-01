"""Video backend interfaces for wm3d_v3 renderers."""

from .base import VideoBackend, VideoBackendOutput, VideoConditionBundle
from .hunyuan_video import (
    HunyuanVideoBackend,
    HunyuanVideoBackendConfig,
    align_hunyuan_video_length,
    summarize_bundle_for_prompt,
)

__all__ = [
    "VideoBackend",
    "VideoBackendOutput",
    "VideoConditionBundle",
    "HunyuanVideoBackend",
    "HunyuanVideoBackendConfig",
    "align_hunyuan_video_length",
    "summarize_bundle_for_prompt",
]
