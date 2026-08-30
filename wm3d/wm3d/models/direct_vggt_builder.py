"""Runtime construction for the frozen direct VGGT teacher."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch

from wm3d.models.direct_vggt_teacher import (
    DirectVGGTTeacherAdapter,
    DirectVGGTTeacherConfig,
)


def build_direct_vggt_teacher(
    runtime: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> DirectVGGTTeacherAdapter:
    closure = runtime["data_closure"]
    model = runtime["model_profile"]["model"]
    from scripts.data.run_cache_worker import _strict_encoder

    base = _strict_encoder(Path(str(closure["encoder_contract_path"])))
    if int(closure["direct_input_rgb_size"]) != int(base.input_rgb_size):
        raise ValueError("direct RGB size differs from the VGGT encoder contract")
    geometry_grid = int(round(int(model["P"]) ** 0.5))
    if geometry_grid * geometry_grid != int(model["P"]):
        raise ValueError("direct VGGT geometry P must be a square grid")
    appearance_enabled = bool(model.get("appearance_enabled", False))
    if appearance_enabled:
        if int(closure["appearance_token_grid"]) ** 2 != int(
            model["appearance_P"]
        ):
            raise ValueError(
                "direct appearance grid differs from the world-model contract"
            )
        appearance_grid = int(round(int(model["appearance_P"]) ** 0.5))
        if appearance_grid * appearance_grid != int(model["appearance_P"]):
            raise ValueError("direct VGGT appearance P must be a square grid")
        appearance_feature_layer = int(closure["appearance_feature_layer"])
        appearance_context_frames = int(model["appearance_context_frames"])
    else:
        if int(closure["appearance_token_grid"]) != geometry_grid:
            raise ValueError(
                "disabled direct appearance must use the geometry grid"
            )
        appearance_grid = 0
        appearance_feature_layer = -1
        appearance_context_frames = 0
    encoder = replace(
        base,
        token_grid=geometry_grid,
        appearance_token_grid=appearance_grid,
        appearance_feature_layer=appearance_feature_layer,
        target_rgb_size=int(model["rgb_size"]),
    )
    return DirectVGGTTeacherAdapter(
        DirectVGGTTeacherConfig(
            encoder=encoder,
            context_frames=int(model["T"]),
            future_frames=int(model["K"]),
            appearance_context_frames=appearance_context_frames,
            appearance_enabled=appearance_enabled,
            rgb_decode_indices=tuple(
                int(item) for item in model["rgb_decode_indices"]
            ),
            encode_chunk_rows=int(
                closure.get("direct_encode_chunk_rows", 32)
            ),
            minimum_chunk_rows=int(
                closure.get("direct_minimum_chunk_rows", 4)
            ),
        ),
        device=device,
    )
