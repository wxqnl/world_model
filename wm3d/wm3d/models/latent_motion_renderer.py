"""Action-conditioned RGB rendering in a reconstructable latent space.

The renderer has one strict appearance rule: observed appearance may reach a
future output only after it has been spatially aligned. It predicts backward
flow from future P256/geometry/action conditions, warps the observed Cosmos
latent and its feature pyramid, and routes disoccluded pixels to a synthesis
branch. There is no unwarped RGB or feature skip into the output compositor.
"""

from __future__ import annotations

from math import isqrt, pi
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = F.silu(self.norm1(value), inplace=True)
        residual = self.conv1(residual)
        residual = F.silu(self.norm2(residual), inplace=True)
        return value + self.conv2(residual)


class _ZeroFlowHead(nn.Conv2d):
    """A flow head whose meta-shard reset remains identity initialized."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class _DisocclusionHead(nn.Conv2d):
    """Start from visible transport while target masks retain authority."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, -2.0)


def _pixel_flow_grid(
    flow_pixels: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build target-to-source sampling coordinates from pixel-unit flow."""

    if flow_pixels.ndim != 4 or flow_pixels.shape[1] != 2:
        raise ValueError("pixel flow must be [N,2,H,W]")
    height, width = flow_pixels.shape[-2:]
    y = torch.linspace(
        -1.0, 1.0, height, dtype=flow_pixels.dtype, device=flow_pixels.device
    )
    x = torch.linspace(
        -1.0, 1.0, width, dtype=flow_pixels.dtype, device=flow_pixels.device
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base = torch.stack((grid_x, grid_y), dim=-1)[None]
    displacement = torch.stack(
        (
            2.0 * flow_pixels[:, 0] / float(max(1, image_width - 1)),
            2.0 * flow_pixels[:, 1] / float(max(1, image_height - 1)),
        ),
        dim=-1,
    )
    grid = base + displacement
    valid = (
        (grid[..., 0] >= -1.0)
        & (grid[..., 0] <= 1.0)
        & (grid[..., 1] >= -1.0)
        & (grid[..., 1] <= 1.0)
    )[:, None]
    return grid, valid


def warp_with_pixel_flow(
    source: torch.Tensor,
    flow_pixels: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-warp source using flow expressed in original RGB pixels."""

    if source.ndim != 4 or source.shape[0] != flow_pixels.shape[0]:
        raise ValueError("warp source and flow batch dimensions must align")
    if source.shape[-2:] != flow_pixels.shape[-2:]:
        flow_pixels = F.interpolate(
            flow_pixels.float(),
            size=source.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).to(dtype=source.dtype)
    grid, valid = _pixel_flow_grid(
        flow_pixels,
        image_height=image_height,
        image_width=image_width,
    )
    warped = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return warped, valid


class NativeLatentFlowRGBDecoder(nn.Module):
    """Warp visible Cosmos latents and synthesize only disoccluded content."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg
        self.appearance_grid = isqrt(int(cfg.appearance_P))
        self.geometry_grid = isqrt(int(cfg.P))
        self.latent_grid = int(cfg.rgb_latent_grid)
        self.hidden = int(cfg.rgb_latent_hidden)
        self.correlation_channels = int(cfg.rgb_flow_correlation_channels)
        self.correlation_radius = int(cfg.rgb_flow_correlation_radius)
        self.correlation_temperature = float(cfg.rgb_flow_correlation_temperature)
        self.correlation_bins = (2 * self.correlation_radius + 1) ** 2
        self.context_high_channels = self.hidden // 3
        self.context_low_channels = self.hidden // 2

        self.future_appearance = nn.Conv2d(cfg.token_dim, self.hidden, 1)
        self.context_appearance = nn.Conv2d(cfg.token_dim, self.hidden, 1)
        self.appearance_delta = nn.Conv2d(cfg.token_dim, self.hidden, 1)
        self.geometry = nn.Conv2d(cfg.token_dim, self.hidden, 1)
        self.action = nn.Sequential(
            nn.LayerNorm(cfg.state_hidden),
            nn.Linear(cfg.state_hidden, self.hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
        )
        self.task = nn.Sequential(
            nn.LayerNorm(cfg.task_dim),
            nn.Linear(cfg.task_dim, self.hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
        )
        self.time = nn.Sequential(
            nn.Linear(17, self.hidden),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
        )
        self.view_embedding = nn.Parameter(
            torch.empty(cfg.num_views, self.hidden, 1, 1)
        )
        self.condition_blocks = nn.Sequential(
            _ResidualBlock(self.hidden),
            _ResidualBlock(self.hidden),
        )
        # P256 is a spatial grid, but an additive future-minus-context feature
        # cannot identify where a moved patch came from.  Build the same kind
        # of explicit correspondence volume used by PWC-Net/RAFT before flow
        # regression.  A shared projection keeps context/future descriptors in
        # one matching space; the relative volume makes displacement channels
        # translation equivariant.
        self.correlation_projection = nn.Linear(
            cfg.token_dim, self.correlation_channels, bias=False
        )
        self.correlation_encoder = nn.Sequential(
            nn.Conv2d(
                self.correlation_bins,
                self.context_low_channels,
                1,
            ),
            nn.GroupNorm(
                _groups(self.context_low_channels), self.context_low_channels
            ),
            nn.SiLU(inplace=True),
            _ResidualBlock(self.context_low_channels),
        )
        self.flow_fusion = nn.Sequential(
            nn.Conv2d(
                self.hidden + self.context_low_channels,
                self.hidden,
                1,
            ),
            nn.GroupNorm(_groups(self.hidden), self.hidden),
            nn.SiLU(inplace=True),
            _ResidualBlock(self.hidden),
        )

        latent_channels = int(cfg.rgb_latent_channels)
        self.context_high = nn.Sequential(
            nn.Conv2d(latent_channels, self.context_high_channels, 3, padding=1),
            nn.GroupNorm(
                _groups(self.context_high_channels), self.context_high_channels
            ),
            nn.SiLU(inplace=True),
            _ResidualBlock(self.context_high_channels),
        )
        self.context_low = nn.Sequential(
            nn.Conv2d(
                self.context_high_channels,
                self.context_low_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(
                _groups(self.context_low_channels), self.context_low_channels
            ),
            nn.SiLU(inplace=True),
            _ResidualBlock(self.context_low_channels),
        )
        self.flow_tower = nn.Sequential(
            _ResidualBlock(self.hidden),
            _ResidualBlock(self.hidden),
        )
        self.flow_head = _ZeroFlowHead(self.hidden, 2, 3, padding=1)

        synthesis_inputs = (
            self.hidden
            + self.context_high_channels
            + self.context_low_channels
            + latent_channels
        )
        self.synthesis_stem = nn.Sequential(
            nn.Conv2d(synthesis_inputs, self.hidden, 3, padding=1),
            nn.GroupNorm(_groups(self.hidden), self.hidden),
            nn.SiLU(inplace=True),
            _ResidualBlock(self.hidden),
            _ResidualBlock(self.hidden),
        )
        self.synthesis_head = nn.Conv2d(self.hidden, latent_channels, 3, padding=1)
        self.disocclusion_head = _DisocclusionHead(self.hidden, 1, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.view_embedding, std=0.02)

    @staticmethod
    def _time_features(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 1:
            raise ValueError("future renderer time must be a vector")
        frequencies = torch.logspace(
            0.0,
            2.0,
            8,
            dtype=value.dtype,
            device=value.device,
        )
        phase = 2.0 * pi * value[:, None] * frequencies[None]
        return torch.cat((value[:, None], phase.sin(), phase.cos()), dim=-1)

    def _condition(
        self,
        *,
        future_appearance: torch.Tensor,
        context_appearance: torch.Tensor,
        geometry: torch.Tensor,
        action: torch.Tensor,
        task: torch.Tensor,
        future_time: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> torch.Tensor:
        expected_appearance = (
            int(self.cfg.appearance_P),
            int(self.cfg.token_dim),
        )
        if tuple(future_appearance.shape[1:]) != expected_appearance:
            raise ValueError("future appearance must end in [appearance_P,token_dim]")
        if context_appearance.shape != future_appearance.shape:
            raise ValueError("context and future appearance tensors must align")
        if tuple(geometry.shape[1:]) != (int(self.cfg.P), int(self.cfg.token_dim)):
            raise ValueError("renderer geometry must end in [P,token_dim]")
        if tuple(action.shape[1:]) != (int(self.cfg.state_hidden),):
            raise ValueError("renderer action must be [N,state_hidden]")
        if tuple(task.shape[1:]) != (int(self.cfg.task_dim),):
            raise ValueError("renderer task must be [N,task_dim]")

        def token_map(tokens: torch.Tensor, grid: int) -> torch.Tensor:
            return tokens.transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[-1], grid, grid
            )

        def normalize_tokens(tokens: torch.Tensor) -> torch.Tensor:
            return F.layer_norm(
                tokens.float(), (tokens.shape[-1],)
            ).to(dtype=tokens.dtype)

        future_map = token_map(
            normalize_tokens(future_appearance), self.appearance_grid
        )
        context_map = token_map(
            normalize_tokens(context_appearance), self.appearance_grid
        )
        value = self.future_appearance(future_map)
        value = value + self.context_appearance(context_map)
        value = value + self.appearance_delta(future_map - context_map)
        geometry_map = self.geometry(
            token_map(normalize_tokens(geometry), self.geometry_grid)
        )
        if geometry_map.shape[-2:] != value.shape[-2:]:
            geometry_map = F.interpolate(
                geometry_map.float(),
                size=value.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=value.dtype)
        value = value + geometry_map
        value = value + self.action(action).to(dtype=value.dtype)[:, :, None, None]
        value = value + self.task(task).to(dtype=value.dtype)[:, :, None, None]
        value = value + self.time(self._time_features(future_time)).to(
            dtype=value.dtype
        )[:, :, None, None]
        value = value + self.view_embedding.index_select(0, view_ids).to(
            dtype=value.dtype
        )
        return self.condition_blocks(value)

    def _relative_correlation(
        self,
        future_appearance: torch.Tensor,
        context_appearance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return relative cost features, logits, and a confidence-gated prior."""

        if future_appearance.shape != context_appearance.shape:
            raise ValueError("correlation context/future appearance must align")
        count = future_appearance.shape[0]
        grid = self.appearance_grid
        tokens = grid * grid
        if future_appearance.shape[1] != tokens:
            raise ValueError("appearance token count does not match its grid")

        def descriptors(value: torch.Tensor) -> torch.Tensor:
            normalized = F.layer_norm(
                value.float(), (value.shape[-1],)
            ).to(dtype=value.dtype)
            projected = self.correlation_projection(normalized)
            return F.normalize(projected.float(), dim=-1)

        query = descriptors(future_appearance)
        key = descriptors(context_appearance)
        all_pairs = torch.bmm(query, key.transpose(1, 2))

        axis = torch.arange(grid, device=all_pairs.device)
        target_y, target_x = torch.meshgrid(axis, axis, indexing="ij")
        target_y = target_y.reshape(tokens, 1)
        target_x = target_x.reshape(tokens, 1)
        offsets = torch.arange(
            -self.correlation_radius,
            self.correlation_radius + 1,
            device=all_pairs.device,
        )
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_y = offset_y.reshape(1, self.correlation_bins)
        offset_x = offset_x.reshape(1, self.correlation_bins)
        source_y = target_y + offset_y
        source_x = target_x + offset_x
        valid = (
            (source_y >= 0)
            & (source_y < grid)
            & (source_x >= 0)
            & (source_x < grid)
        )
        source_index = (
            source_y.clamp(0, grid - 1) * grid
            + source_x.clamp(0, grid - 1)
        ).long()
        relative = all_pairs.gather(
            2, source_index[None].expand(count, -1, -1)
        )
        logits = relative / self.correlation_temperature
        logits = logits.masked_fill(~valid[None], -1.0e4)
        volume = relative.masked_fill(~valid[None], 0.0)
        features = self.correlation_encoder(
            volume.transpose(1, 2).reshape(
                count,
                self.correlation_bins,
                grid,
                grid,
            ).to(dtype=future_appearance.dtype)
        )

        probability = torch.softmax(logits, dim=-1)
        displacement = torch.stack(
            (offset_x.reshape(-1), offset_y.reshape(-1)), dim=0
        ).to(dtype=probability.dtype)
        displacement = displacement * (
            float(self.cfg.rgb_size - 1) / float(max(1, grid - 1))
        )
        expected = torch.einsum("ntd,cd->nct", probability, displacement)
        valid_count = valid.sum(dim=-1).clamp_min(1).to(probability.dtype)
        uniform_peak = valid_count.reciprocal()[None]
        peak = probability.amax(dim=-1)
        confidence = ((peak - uniform_peak) / (1.0 - uniform_peak).clamp_min(1.0e-6))
        confidence = confidence.clamp(0.0, 1.0)
        prior = (expected * confidence[:, None]).reshape(count, 2, grid, grid)
        return (
            features,
            logits.transpose(1, 2).reshape(
                count, self.correlation_bins, grid, grid
            ),
            prior,
        )

    def forward(
        self,
        *,
        appearance_tokens: torch.Tensor,
        appearance_context_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        factual_action_summary: torch.Tensor,
        task_embedding: torch.Tensor,
        future_times_s: torch.Tensor,
        context_latent: torch.Tensor,
        frame_indices: Optional[Sequence[int]] = None,
        target_view_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        expected_appearance = (
            appearance_tokens.shape[0],
            cfg.K,
            cfg.num_views,
            cfg.appearance_P,
            cfg.token_dim,
        )
        if tuple(appearance_tokens.shape) != expected_appearance:
            raise ValueError(f"appearance tokens must be {expected_appearance}")
        batch = appearance_tokens.shape[0]
        if tuple(appearance_context_tokens.shape) != (
            batch,
            cfg.num_views,
            cfg.appearance_P,
            cfg.token_dim,
        ):
            raise ValueError("context appearance tokens are incompatible")
        if tuple(geometry_tokens.shape) != (
            batch,
            cfg.K,
            cfg.P,
            cfg.token_dim,
        ):
            raise ValueError("renderer geometry tokens are incompatible")
        if tuple(factual_action_summary.shape) != (
            batch,
            cfg.K,
            cfg.state_hidden,
        ):
            raise ValueError("renderer factual action summary is incompatible")
        if tuple(future_times_s.shape) != (batch, cfg.K):
            raise ValueError("renderer future times must be [B,K]")
        if tuple(context_latent.shape) != (
            batch,
            cfg.num_views,
            cfg.rgb_latent_channels,
            self.latent_grid,
            self.latent_grid,
        ):
            raise ValueError("context RGB latent is incompatible")

        indices = tuple(
            cfg.rgb_decode_indices if frame_indices is None else frame_indices
        )
        if any(index < 0 or index >= cfg.K for index in indices):
            raise ValueError("latent RGB decode index is outside K")
        index_tensor = torch.tensor(
            indices, dtype=torch.long, device=appearance_tokens.device
        )
        frames = len(indices)
        if frames == 0:
            empty_latent = appearance_tokens.new_empty(
                batch,
                0,
                cfg.num_views,
                cfg.rgb_latent_channels,
                self.latent_grid,
                self.latent_grid,
            )
            empty_flow = appearance_tokens.new_empty(
                batch, 0, cfg.num_views, 2, self.latent_grid, self.latent_grid
            )
            empty_mask = appearance_tokens.new_empty(
                batch, 0, cfg.num_views, 1, self.latent_grid, self.latent_grid
            )
            empty_correlation = appearance_tokens.new_empty(
                batch,
                0,
                cfg.num_views,
                self.correlation_bins,
                self.appearance_grid,
                self.appearance_grid,
            )
            return {
                "rgb_latent": empty_latent,
                "rgb_frame_indices": index_tensor,
                "rgb_flow_pixels": empty_flow,
                "rgb_disocclusion_logit": empty_mask,
                "rgb_disocclusion": empty_mask,
                "rgb_warped_latent": empty_latent,
                "rgb_synthesis_latent": empty_latent,
                "rgb_warp_valid": empty_mask.bool(),
                "rgb_flow_correlation_logits": empty_correlation,
            }

        views = int(cfg.num_views)
        selected_appearance = appearance_tokens.index_select(1, index_tensor)
        selected_geometry = geometry_tokens.index_select(1, index_tensor)
        selected_action = factual_action_summary.index_select(1, index_tensor)
        selected_time = future_times_s.index_select(1, index_tensor)
        future_flat = selected_appearance.reshape(
            batch * frames * views, cfg.appearance_P, cfg.token_dim
        )
        context_appearance_flat = (
            appearance_context_tokens[:, None]
            .expand(-1, frames, -1, -1, -1)
            .reshape(batch * frames * views, cfg.appearance_P, cfg.token_dim)
        )
        geometry_flat = (
            selected_geometry[:, :, None]
            .expand(-1, -1, views, -1, -1)
            .reshape(batch * frames * views, cfg.P, cfg.token_dim)
        )
        action_flat = (
            selected_action[:, :, None]
            .expand(-1, -1, views, -1)
            .reshape(batch * frames * views, cfg.state_hidden)
        )
        task_flat = (
            task_embedding[:, None, None]
            .expand(-1, frames, views, -1)
            .reshape(batch * frames * views, cfg.task_dim)
        )
        time_flat = (
            selected_time[:, :, None]
            .expand(-1, -1, views)
            .reshape(batch * frames * views)
        )
        view_ids = (
            torch.arange(views, device=appearance_tokens.device)
            .view(1, 1, views)
            .expand(batch, frames, -1)
            .reshape(-1)
        )
        context_latent_flat = (
            context_latent[:, None]
            .expand(-1, frames, -1, -1, -1, -1)
            .reshape(
                batch * frames * views,
                cfg.rgb_latent_channels,
                self.latent_grid,
                self.latent_grid,
            )
        )

        condition = self._condition(
            future_appearance=future_flat,
            context_appearance=context_appearance_flat,
            geometry=geometry_flat,
            action=action_flat,
            task=task_flat,
            future_time=time_flat,
            view_ids=view_ids,
        )
        correlation_features, correlation_logits, correlation_prior = (
            self._relative_correlation(future_flat, context_appearance_flat)
        )
        flow_features = self.flow_fusion(
            torch.cat((condition, correlation_features), dim=1)
        )
        flow_low = self.flow_head(self.flow_tower(flow_features)).float()
        # Regress residuals in pixel units.  Dividing before tanh keeps the
        # zero-point derivative at one instead of rgb_flow_max_pixels, which
        # avoids the early overshoot followed by zero-flow collapse seen in
        # the first Teacher run.
        max_flow = float(cfg.rgb_flow_max_pixels)
        flow_residual = max_flow * torch.tanh(flow_low / max_flow)
        flow_residual = F.interpolate(
            flow_residual,
            size=(self.latent_grid, self.latent_grid),
            mode="bilinear",
            align_corners=True,
        )
        correlation_prior = F.interpolate(
            correlation_prior,
            size=(self.latent_grid, self.latent_grid),
            mode="bilinear",
            align_corners=True,
        )
        flow = (flow_residual + correlation_prior).clamp(
            -max_flow, max_flow
        ).to(dtype=condition.dtype)

        context_high = self.context_high(context_latent_flat)
        context_low = self.context_low(context_high)
        warped_latent, valid = warp_with_pixel_flow(
            context_latent_flat,
            flow,
            image_height=cfg.rgb_size,
            image_width=cfg.rgb_size,
        )
        warped_high, _ = warp_with_pixel_flow(
            context_high,
            flow,
            image_height=cfg.rgb_size,
            image_width=cfg.rgb_size,
        )
        warped_low, _ = warp_with_pixel_flow(
            context_low,
            flow,
            image_height=cfg.rgb_size,
            image_width=cfg.rgb_size,
        )
        warped_low = F.interpolate(
            warped_low.float(),
            size=(self.latent_grid, self.latent_grid),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=warped_high.dtype)
        condition_high = F.interpolate(
            condition.float(),
            size=(self.latent_grid, self.latent_grid),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=warped_high.dtype)
        synthesis_features = self.synthesis_stem(
            torch.cat(
                (condition_high, warped_high, warped_low, warped_latent), dim=1
            )
        )
        synthesis = self.synthesis_head(synthesis_features)
        disocclusion_logit = self.disocclusion_head(synthesis_features)
        predicted_disocclusion = torch.sigmoid(disocclusion_logit)
        disocclusion = 1.0 - (
            (1.0 - predicted_disocclusion) * valid.to(predicted_disocclusion.dtype)
        )
        # Reconstruction must not be allowed to expand the mask just because
        # synthesis is easier than correspondence.  The mask is trained by its
        # factual occlusion losses; its forward value still controls the exact
        # inference compositor.
        composition_mask = disocclusion.detach()
        latent = (
            (1.0 - composition_mask) * warped_latent
            + composition_mask * synthesis
        )

        if target_view_mask is None:
            slot_valid = torch.ones(
                batch,
                frames,
                views,
                1,
                1,
                1,
                dtype=torch.bool,
                device=appearance_tokens.device,
            )
        else:
            if tuple(target_view_mask.shape) != (batch, frames, views):
                raise ValueError("target view mask must be [B,F,V]")
            slot_valid = target_view_mask[..., None, None, None].bool()

        def restore(value: torch.Tensor, channels: int) -> torch.Tensor:
            restored = value.reshape(
                batch,
                frames,
                views,
                channels,
                self.latent_grid,
                self.latent_grid,
            )
            return restored * slot_valid.to(dtype=restored.dtype)

        return {
            "rgb_latent": restore(latent, cfg.rgb_latent_channels),
            "rgb_frame_indices": index_tensor,
            "rgb_flow_pixels": restore(flow, 2),
            "rgb_disocclusion_logit": restore(disocclusion_logit, 1),
            "rgb_disocclusion": restore(disocclusion, 1),
            "rgb_warped_latent": restore(warped_latent, cfg.rgb_latent_channels),
            "rgb_synthesis_latent": restore(synthesis, cfg.rgb_latent_channels),
            "rgb_warp_valid": (
                valid.reshape(
                    batch,
                    frames,
                    views,
                    1,
                    self.latent_grid,
                    self.latent_grid,
                )
                & slot_valid
            ),
            "rgb_flow_correlation_logits": (
                correlation_logits.reshape(
                    batch,
                    frames,
                    views,
                    self.correlation_bins,
                    self.appearance_grid,
                    self.appearance_grid,
                )
                * slot_valid[..., :1, :1, :1].to(
                    dtype=correlation_logits.dtype
                )
            ),
        }


__all__ = [
    "NativeLatentFlowRGBDecoder",
    "warp_with_pixel_flow",
]
