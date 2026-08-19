"""Runtime assembly for the model-independent WM3D episode cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from safetensors import safe_open
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .cache_codec import JpegPackReader, dequantize_per_vector
from .episode_robot import (
    PreparedEpisodeRobot,
    assemble_robot_window_from_prepared_episode,
    prepare_episode_robot_tensors,
    validate_episode_robot_tensors,
)
from .grouped_robot import GroupedRobotLimits
from .grouped_normalization import (
    GroupedNormalizationError,
    GroupedRobotNormalizer,
    normalize_grouped_masked,
    validate_grouped_lane_mask,
)
from .manifest_contract import CacheIndexEntry, DataProfile, load_cache_index, sha256_file
from wm3d.models.model_factory import validate_model_data_compatibility
from wm3d.models.native_world_model import native_config_from_mapping


class CacheDataError(RuntimeError):
    pass


def _active_source_names(
    *,
    source_order: Sequence[str],
    selected_sources: Sequence[str],
    entries: Sequence[CacheIndexEntry],
    split: str,
) -> tuple[str, ...]:
    """Return sources that actually contribute to one sealed split.

    Every configured training source is contractual: silently dropping one
    would change the training mixture.  Validation and test closures, however,
    may legitimately omit a source as long as the split remains non-empty.
    """

    selected = set(selected_sources)
    present = {entry.source for entry in entries if entry.split == split}
    active = tuple(name for name in source_order if name in selected and name in present)
    if not active:
        raise CacheDataError("cache selection produced no samples")
    if split == "train":
        missing = tuple(name for name in source_order if name in selected and name not in present)
        if missing:
            raise CacheDataError(
                f"training sources have no cache windows: {list(missing)}"
            )
    return active


class _ShardStore:
    def __init__(
        self,
        root: Path,
        expected_sha_by_relative: Mapping[str, str],
        *,
        verify_on_open: bool,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        self.expected_sha = dict(expected_sha_by_relative)
        self.verify_on_open = bool(verify_on_open)
        self._resolved: dict[str, Path] = {}
        self._verified: set[str] = set()

    def register(
        self,
        relative: str,
        expected_sha256: str,
        *,
        verified: bool = False,
        allow_verified_replacement: bool = False,
    ) -> None:
        """Register a lazily materialized shard without weakening SHA checks."""

        previous = self.expected_sha.get(relative)
        if previous is not None and previous != expected_sha256:
            if not (verified and allow_verified_replacement):
                raise CacheDataError(
                    f"cache shard {relative} changed digest within one dataset process"
                )
            # A bounded streaming LRU may delete an old episode and later
            # recreate the same task path.  The streaming manager verifies the
            # replacement before registration; discard all state tied to the
            # old inode and digest before accepting that verified replacement.
            self._resolved.pop(relative, None)
            self._verified.discard(relative)
        self.expected_sha[relative] = expected_sha256
        if verified:
            self._verified.add(relative)

    def path(self, relative: str) -> Path:
        path = self._resolved.get(relative)
        if path is None:
            candidate = self.root / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise CacheDataError(f"cache shard is not a regular file: {candidate}")
            path = candidate.resolve(strict=True)
            try:
                path.relative_to(self.root)
            except ValueError as exc:
                raise CacheDataError(f"cache shard escapes root: {path}") from exc
            self._resolved[relative] = path
        if self.verify_on_open and relative not in self._verified:
            expected = self.expected_sha.get(relative)
            if expected is None:
                raise CacheDataError(f"cache index has no SHA for {relative}")
            observed = sha256_file(path)
            if observed != expected:
                raise CacheDataError(
                    f"cache shard SHA mismatch {relative}: {observed} != {expected}"
                )
            self._verified.add(relative)
        return path

    def read_many(
        self, relative: str, names: Sequence[str], rows: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        if not rows or any(int(row) < 0 for row in rows):
            raise CacheDataError("cache row selection must be non-empty/non-negative")
        path = self.path(relative)
        output: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            missing = set(names) - keys
            if missing:
                raise CacheDataError(f"{relative} misses tensors {sorted(missing)}")
            for name in names:
                tensor_slice = handle.get_slice(name)
                shape = tensor_slice.get_shape()
                if not shape or max(rows) >= shape[0]:
                    raise CacheDataError(
                        f"rows {list(rows)} outside {relative}:{name} shape {shape}"
                    )
                output[name] = torch.stack(
                    [tensor_slice[int(row) : int(row) + 1][0] for row in rows]
                )
        return output

    def read_quantized_many(
        self, relative: str, name: str, rows: Sequence[int]
    ) -> torch.Tensor:
        value = self.read_many(relative, (f"{name}_q", f"{name}_scale"), rows)
        return dequantize_per_vector(
            value[f"{name}_q"], value[f"{name}_scale"], dtype=torch.bfloat16
        )

    def read_all(self, relative: str) -> dict[str, torch.Tensor]:
        path = self.path(relative)
        with safe_open(path, framework="pt", device="cpu") as handle:
            return {name: handle.get_tensor(name) for name in handle.keys()}


class _JpegStore:
    def __init__(self, shards: _ShardStore, cache_size: int) -> None:
        self.shards = shards
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[str, JpegPackReader] = OrderedDict()

    def reader(self, relative: str) -> JpegPackReader:
        reader = self.cache.pop(relative, None)
        if reader is None:
            reader = JpegPackReader(self.shards.path(relative))
        self.cache[relative] = reader
        while len(self.cache) > self.cache_size:
            _name, old = self.cache.popitem(last=False)
            old.close()
        return reader


class _RobotStore:
    def __init__(self, shards: _ShardStore, cache_size: int) -> None:
        self.shards = shards
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[
            str, tuple[Mapping[str, torch.Tensor], PreparedEpisodeRobot]
        ] = OrderedDict()

    def read(
        self, relative: str, *, embodiment: Any
    ) -> tuple[Mapping[str, torch.Tensor], PreparedEpisodeRobot]:
        cached = self.cache.pop(relative, None)
        if cached is None:
            loaded = self.shards.read_all(relative)
            validate_episode_robot_tensors(loaded)
            cached = (
                loaded,
                prepare_episode_robot_tensors(loaded, embodiment=embodiment),
            )
        self.cache[relative] = cached
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return cached


def _sha_map(entries: Sequence[CacheIndexEntry]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        for relative, digest in (
            (entry.feature_shard, entry.feature_sha256),
            (entry.robot_shard, entry.robot_sha256),
            (entry.rgb_pack, entry.rgb_pack_sha256),
        ):
            previous = result.setdefault(relative, digest)
            if previous != digest:
                raise CacheDataError(
                    f"cache index assigns conflicting SHA values to {relative}"
                )
    return result


def _square_grid(tokens: int, *, label: str) -> int:
    grid = int(round(int(tokens) ** 0.5))
    if grid <= 0 or grid * grid != int(tokens):
        raise CacheDataError(f"{label} spatial token count must be a square")
    return grid


def _pool_tokens(value: torch.Tensor, *, source_grid: int, target_grid: int) -> torch.Tensor:
    if source_grid == target_grid:
        return value
    if value.shape[-2] != source_grid * source_grid:
        raise CacheDataError("token tensor disagrees with cache grid")
    leading = value.shape[:-2]
    channels = value.shape[-1]
    image = value.float().reshape(-1, source_grid, source_grid, channels)
    image = image.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(image, (target_grid, target_grid))
    return pooled.permute(0, 2, 3, 1).reshape(
        *leading, target_grid * target_grid, channels
    ).to(dtype=value.dtype)


def _pool_valid_mask(
    value: torch.Tensor, *, source_grid: int, target_grid: int
) -> torch.Tensor:
    if value.shape[-1] != source_grid * source_grid:
        raise CacheDataError("token mask disagrees with cache grid")
    if source_grid == target_grid:
        return value.bool()
    leading = value.shape[:-1]
    pooled = F.adaptive_max_pool2d(
        value.float().reshape(-1, 1, source_grid, source_grid),
        (target_grid, target_grid),
    )
    return pooled.reshape(*leading, target_grid * target_grid).bool()


def _pool_masked(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    source_grid: int,
    target_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask-aware spatial pooling for scalar or vector geometry targets."""

    if value.shape[: mask.ndim] != mask.shape:
        raise CacheDataError("geometry value/mask shapes disagree")
    channels = 1 if value.ndim == mask.ndim else int(value.shape[-1])
    if channels == 1:
        flat_value = value.float().reshape(-1, 1, source_grid, source_grid)
    else:
        flat_value = value.float().reshape(-1, source_grid, source_grid, channels)
        flat_value = flat_value.permute(0, 3, 1, 2)
    flat_mask = mask.float().reshape(-1, 1, source_grid, source_grid)
    numerator = F.adaptive_avg_pool2d(flat_value * flat_mask, (target_grid, target_grid))
    denominator = F.adaptive_avg_pool2d(flat_mask, (target_grid, target_grid))
    pooled = numerator / denominator.clamp_min(1.0e-12)
    valid = denominator > 0
    leading = mask.shape[:-1]
    if channels == 1:
        return (
            pooled.reshape(*leading, target_grid * target_grid).to(value.dtype),
            valid.reshape(*leading, target_grid * target_grid),
        )
    return (
        pooled.permute(0, 2, 3, 1)
        .reshape(*leading, target_grid * target_grid, channels)
        .to(value.dtype),
        valid.reshape(*leading, target_grid * target_grid),
    )


def _fuse_target_tokens(
    view_tokens: torch.Tensor,
    confidence: torch.Tensor,
    view_mask: torch.Tensor,
    world_mask: torch.Tensor,
    *,
    source_grid: int,
    target_grid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if view_tokens.ndim != 4 or confidence.shape != view_tokens.shape[:-1]:
        raise CacheDataError("future view token/confidence shapes disagree")
    if view_mask.shape != view_tokens.shape[:2] or world_mask.shape != (
        view_tokens.shape[0],
        view_tokens.shape[2],
    ):
        raise CacheDataError("future view/world masks disagree with tokens")
    weight = confidence.float().clamp_min(0) * view_mask[..., None].float()
    weight = weight * world_mask[:, None].float()
    numerator = (view_tokens.float() * weight[..., None]).sum(dim=1)
    denominator = weight.sum(dim=1)
    channels = view_tokens.shape[-1]
    numerator_grid = numerator.reshape(-1, source_grid, source_grid, channels)
    numerator_grid = numerator_grid.permute(0, 3, 1, 2)
    denominator_grid = denominator.reshape(-1, 1, source_grid, source_grid)
    pooled_num = F.adaptive_avg_pool2d(
        numerator_grid, (target_grid, target_grid)
    )
    pooled_den = F.adaptive_avg_pool2d(
        denominator_grid, (target_grid, target_grid)
    )
    target = (pooled_num / pooled_den.clamp_min(1.0e-12)).permute(0, 2, 3, 1)
    return (
        target.reshape(-1, target_grid * target_grid, channels).to(torch.bfloat16),
        pooled_den.reshape(-1, target_grid * target_grid) > 0,
    )


class UnifiedCacheDataset(Dataset[dict[str, torch.Tensor]]):
    """One source/model-independent cache ABI for 1B, 5B and all robots."""

    FRAME_REQUIRED = (
        "source_observation_row",
        "frame_time_s",
        "view_mask",
        "world_token_mask",
        "depth",
        "depth_mask",
        "point",
        "point_mask",
        "camera_pose",
        "camera_pose_mask",
        "geometry_confidence",
        "rgb_offsets",
        "rgb_lengths",
    )

    def __init__(
        self,
        *,
        cache_root: Path,
        index_path: Path,
        index_sha256: str,
        data_profile: DataProfile,
        model_profile: Mapping[str, Any],
        split: str,
        selected_sources: Optional[Sequence[str]] = None,
        verify_shard_sha_on_open: bool = True,
        jpeg_reader_cache_size: int = 8,
        robot_reader_cache_size: int = 16,
        appearance_cache_grid: int | None = None,
        grouped_normalizer: GroupedRobotNormalizer,
    ) -> None:
        validate_model_data_compatibility(
            model_profile,
            data_profile,
            appearance_cache_grid=appearance_cache_grid,
        )
        self.root = Path(cache_root).resolve(strict=True)
        entries = load_cache_index(index_path, expected_sha256=index_sha256)
        if split not in {"train", "val", "test"}:
            raise CacheDataError(f"invalid split {split!r}")
        model = native_config_from_mapping(model_profile["model"]).__dict__
        sampling = model_profile["sampling"]
        self.T = int(model["T"])
        self.K = int(model["K"])
        self.model_grid = _square_grid(int(model["P"]), label="model")
        representation = data_profile.cache_representation
        self.cache_grid = int(representation["token_grid"])
        if self.cache_grid * self.cache_grid != int(representation["spatial_tokens"]):
            raise CacheDataError("cache token grid/profile is inconsistent")
        if self.model_grid > self.cache_grid:
            raise CacheDataError(
                "model spatial grid exceeds cached representation; recache at a higher grid"
            )
        if int(model["token_dim"]) != int(representation["token_dim"]):
            raise CacheDataError("model/cache token dimensions disagree")
        if int(model["num_views"]) != int(representation["num_views"]):
            raise CacheDataError("model/cache canonical view counts disagree")
        if int(model["rgb_size"]) > int(representation["rgb_size"]):
            raise CacheDataError("model RGB target exceeds cached RGB resolution")
        self.model = model
        self.appearance_enabled = bool(model["appearance_enabled"])
        self.appearance_context_frames = int(model["appearance_context_frames"])
        self.appearance_grid = _square_grid(
            int(model["appearance_P"]), label="appearance model"
        )
        self.appearance_cache_grid = (
            self.cache_grid if appearance_cache_grid is None else int(appearance_cache_grid)
        )
        if self.appearance_enabled and self.appearance_grid > self.appearance_cache_grid:
            raise CacheDataError(
                "appearance model grid exceeds cached per-view representation"
            )
        self.sampling = sampling
        self.rgb_size = int(model["rgb_size"])
        self.rgb_indices = tuple(int(item) for item in model["rgb_decode_indices"])
        if not self.rgb_indices or any(not 0 <= item < self.K for item in self.rgb_indices):
            raise CacheDataError("model RGB supervision indices lie outside K")
        self.limits = GroupedRobotLimits(
            max_groups=int(model["max_action_groups"]),
            max_substeps=int(model["max_action_substeps"]),
            max_action_dim=int(model["max_action_dim"]),
            max_state_dim=int(model["max_state_dim"]),
        )
        if grouped_normalizer.window_index_sha256 != index_sha256:
            raise CacheDataError(
                "grouped normalization belongs to a different window index"
            )

        source_order = data_profile.source_order
        selected = set(source_order if selected_sources is None else selected_sources)
        unknown = sorted(selected - set(source_order))
        if unknown:
            raise CacheDataError(f"selected sources not in data profile: {unknown}")
        self.source_to_id = {name: index for index, name in enumerate(source_order)}
        self.data_profile = data_profile
        active_sources = _active_source_names(
            source_order=source_order,
            selected_sources=tuple(selected),
            entries=entries,
            split=split,
        )
        active = set(active_sources)
        filtered = [
            entry for entry in entries if entry.split == split and entry.source in active
        ]
        for entry in filtered:
            if len(entry.context_feature_rows) != self.T:
                raise CacheDataError(
                    f"sample {entry.sample_id} context rows do not match T={self.T}"
                )
            if len(entry.future_feature_rows) != self.K:
                raise CacheDataError(
                    f"sample {entry.sample_id} future rows do not match K={self.K}"
                )
        order = {name: index for index, name in enumerate(source_order)}
        self.entries = tuple(
            sorted(filtered, key=lambda item: (order[item.source], item.sample_id))
        )
        self._source_names = active_sources
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name in self._source_names:
            start = cursor
            while cursor < len(self.entries) and self.entries[cursor].source == name:
                cursor += 1
            if cursor == start:
                raise CacheDataError(f"source {name!r} has no {split!r} cache windows")
            spans[name] = (start, cursor)
        if cursor != len(self.entries):
            raise CacheDataError("cache entries do not form sealed source spans")
        self._source_spans = spans
        for entry in self.entries:
            source = data_profile.sources[order[entry.source]]
            if entry.embodiment != source.embodiment:
                raise CacheDataError(
                    f"sample {entry.sample_id} embodiment differs from its source profile"
                )
        self.shards = _ShardStore(
            self.root, _sha_map(self.entries), verify_on_open=verify_shard_sha_on_open
        )
        self.jpeg = _JpegStore(self.shards, jpeg_reader_cache_size)
        self.robot = _RobotStore(self.shards, robot_reader_cache_size)
        self.grouped_normalizer = grouped_normalizer

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def source_names(self) -> tuple[str, ...]:
        return self._source_names

    @property
    def source_spans(self) -> Mapping[str, tuple[int, int]]:
        return dict(self._source_spans)

    @staticmethod
    def _validate_sample(result: Mapping[str, torch.Tensor], entry: CacheIndexEntry) -> None:
        for name, value in result.items():
            if not isinstance(value, torch.Tensor):
                raise CacheDataError(
                    f"sample {entry.sample_id} field {name!r} is not a tensor"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise CacheDataError(
                    f"sample {entry.sample_id} field {name!r} contains NaN/Inf"
                )
        if not bool(torch.diff(result["world_times_s"]).gt(0).all()):
            raise CacheDataError(f"sample {entry.sample_id} world times are not increasing")
        valid = result["policy_query_mask"].bool()
        pairs = valid[:, 1:] & valid[:, :-1]
        if bool((torch.diff(result["policy_query_dt"], dim=-1)[pairs] <= 0).any()):
            raise CacheDataError(f"sample {entry.sample_id} policy times are not increasing")
        if not bool(result["action_group_mask"].any()):
            raise CacheDataError(f"sample {entry.sample_id} has no action group")
        if not bool(result["current_state_mask"].any()):
            raise CacheDataError(f"sample {entry.sample_id} has no measured current state")

    def _sample_from_entry(
        self, entry: CacheIndexEntry, *, sample_index: int
    ) -> dict[str, torch.Tensor]:
        context_rows = entry.context_feature_rows
        future_rows = entry.future_feature_rows
        all_rows = context_rows + future_rows
        frame = self.shards.read_many(entry.feature_shard, self.FRAME_REQUIRED, all_rows)
        boundary_rows = (entry.leading_feature_row,) + all_rows
        boundary_source = self.shards.read_many(
            entry.feature_shard, ("source_observation_row",), boundary_rows
        )["source_observation_row"].to(torch.int64)
        if not bool(torch.diff(boundary_source).gt(0).all()):
            raise CacheDataError(
                f"sample {entry.sample_id} source observation boundaries are invalid"
            )

        context_view_tokens = self.shards.read_quantized_many(
            entry.feature_shard, "view_tokens", context_rows
        )
        context_tokens = _pool_tokens(
            context_view_tokens,
            source_grid=self.cache_grid,
            target_grid=self.model_grid,
        )
        future_view_tokens = self.shards.read_quantized_many(
            entry.feature_shard, "view_tokens", future_rows
        )
        future = slice(self.T, None)
        target_tokens, target_token_mask = _fuse_target_tokens(
            future_view_tokens,
            frame["geometry_confidence"][future],
            frame["view_mask"][future].bool(),
            frame["world_token_mask"][future].bool(),
            source_grid=self.cache_grid,
            target_grid=self.model_grid,
        )
        target_depth, target_depth_mask = _pool_masked(
            frame["depth"][future],
            frame["depth_mask"][future].bool(),
            source_grid=self.cache_grid,
            target_grid=self.model_grid,
        )
        target_point, target_point_mask = _pool_masked(
            frame["point"][future],
            frame["point_mask"][future].bool(),
            source_grid=self.cache_grid,
            target_grid=self.model_grid,
        )

        appearance: dict[str, torch.Tensor] = {}
        if self.appearance_enabled:
            context_appearance_tokens = context_view_tokens
            future_appearance_tokens = future_view_tokens
            if self.appearance_cache_grid != self.cache_grid:
                context_appearance_tokens = self.shards.read_quantized_many(
                    entry.feature_shard, "appearance_tokens", context_rows
                )
                future_appearance_tokens = self.shards.read_quantized_many(
                    entry.feature_shard, "appearance_tokens", future_rows
                )
            context_start = self.T - self.appearance_context_frames
            appearance_context = _pool_tokens(
                context_appearance_tokens[context_start:].contiguous(),
                source_grid=self.appearance_cache_grid,
                target_grid=self.appearance_grid,
            ).contiguous()
            appearance_target = _pool_tokens(
                future_appearance_tokens,
                source_grid=self.appearance_cache_grid,
                target_grid=self.appearance_grid,
            ).contiguous()
            appearance["appearance_context_tokens"] = appearance_context
            appearance["appearance_context_mask"] = (
                frame["view_mask"][context_start : self.T, :, None].bool()
                .expand(-1, -1, self.appearance_grid * self.appearance_grid)
            )
            appearance["target_appearance_tokens"] = appearance_target
            appearance["target_appearance_mask"] = (
                frame["view_mask"][future, :, None].bool()
                .expand(-1, -1, self.appearance_grid * self.appearance_grid)
            )

        rgb_rows = torch.tensor(self.rgb_indices, dtype=torch.long) + self.T
        reader = self.jpeg.reader(entry.rgb_pack)
        target_rgb = torch.stack(
            [
                reader.decode(
                    frame["rgb_offsets"][int(row)].tolist(),
                    frame["rgb_lengths"][int(row)].tolist(),
                )
                for row in rgb_rows.tolist()
            ]
        ).float()
        if tuple(target_rgb.shape[-2:]) != (self.rgb_size, self.rgb_size):
            leading = target_rgb.shape[:-3]
            target_rgb = F.interpolate(
                target_rgb.reshape(-1, 3, *target_rgb.shape[-2:]),
                size=(self.rgb_size, self.rgb_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).reshape(*leading, 3, self.rgb_size, self.rgb_size)
        target_rgb = target_rgb.div_(255.0)
        rgb_view_mask = frame["view_mask"].index_select(0, rgb_rows).bool()

        embodiment = self.data_profile.embodiments[entry.embodiment]
        robot_values, prepared_robot = self.robot.read(
            entry.robot_shard, embodiment=embodiment
        )
        action = assemble_robot_window_from_prepared_episode(
            prepared=prepared_robot,
            embodiment=embodiment,
            selected_source_boundary_indices=boundary_source.tolist(),
            limits=self.limits,
            context_samples=self.T,
            max_policy_queries=int(self.model["max_policy_queries"]),
            policy_target_horizon_s=float(
                self.sampling["policy_target_horizon_seconds"]
            ),
        )
        normalization = self.grouped_normalizer.tensors_for(
            source=entry.source,
            embodiment_id=int(action["embodiment_ids"]),
            group_ids=action["action_group_ids"],
            action_semantic_ids=action["action_semantic_ids"],
            state_semantic_ids=action["state_semantic_ids"],
        )
        for slot, (group, series) in enumerate(
            zip(embodiment.groups, prepared_robot.action_series)
        ):
            fine_available = bool(
                normalization.fine_action_available[slot, : group.action_dim].all()
            )
            coarse_available = bool(
                normalization.coarse_action_available[slot, : group.action_dim].all()
            )
            if (series.supervision == "fine_command") != fine_available or (
                series.supervision == "coarse_effect"
            ) != coarse_available:
                raise CacheDataError(
                    f"sample {entry.sample_id} action lane differs from normalization artifact"
                )
        fine_masks = (
            (action["history_fine_action_mask"], 1),
            (action["future_factual_fine_action_mask"], 1),
            (action["target_fine_action_mask"], 0),
        )
        coarse_masks = (
            (action["history_coarse_action_mask"], 1),
            (action["future_factual_coarse_action_mask"], 1),
            (action["target_coarse_action_mask"], 1),
        )
        try:
            for mask, group_axis in fine_masks:
                validate_grouped_lane_mask(
                    mask,
                    available=normalization.fine_action_available,
                    group_axis=group_axis,
                    lane="fine_command",
                )
            for mask, group_axis in coarse_masks:
                validate_grouped_lane_mask(
                    mask,
                    available=normalization.coarse_action_available,
                    group_axis=group_axis,
                    lane="coarse_effect",
                )
            validate_grouped_lane_mask(
                action["current_state_mask"],
                available=normalization.state_available,
                group_axis=0,
                lane="current_state",
            )
        except GroupedNormalizationError as exc:
            raise CacheDataError(f"sample {entry.sample_id}: {exc}") from exc
        for name in (
            "history_fine_action_values",
            "future_factual_fine_action_values",
        ):
            action[name] = normalize_grouped_masked(
                action[name],
                action[name.replace("values", "mask")],
                offset=normalization.fine_action_offset,
                scale=normalization.fine_action_scale,
                group_axis=1,
            )
        for name in (
            "history_coarse_action_values",
            "future_factual_coarse_action_values",
        ):
            action[name] = normalize_grouped_masked(
                action[name],
                action[name.replace("values", "mask")],
                offset=normalization.coarse_action_offset,
                scale=normalization.coarse_action_scale,
                group_axis=1,
            )
        action["current_state_values"] = normalize_grouped_masked(
            action["current_state_values"],
            action["current_state_mask"],
            offset=normalization.state_offset,
            scale=normalization.state_scale,
            group_axis=0,
        )
        action["target_fine_action"] = normalize_grouped_masked(
            action["target_fine_action"],
            action["target_fine_action_mask"],
            offset=normalization.fine_action_offset,
            scale=normalization.fine_action_scale,
            group_axis=0,
        )
        action["target_coarse_action_normalized"] = normalize_grouped_masked(
            action["target_coarse_action"],
            action["target_coarse_action_mask"],
            offset=normalization.coarse_action_offset,
            scale=normalization.coarse_action_scale,
            group_axis=1,
        )
        action["action_normalization_offset"] = torch.where(
            normalization.fine_action_available,
            normalization.fine_action_offset,
            normalization.coarse_action_offset,
        )
        action["action_normalization_scale"] = torch.where(
            normalization.fine_action_available,
            normalization.fine_action_scale,
            normalization.coarse_action_scale,
        )
        task_embedding = prepared_robot.task_embedding
        if task_embedding.numel() != int(self.model["task_dim"]):
            raise CacheDataError(
                f"sample {entry.sample_id} task embedding dimension is incompatible"
            )
        aux_values = torch.zeros(
            self.T,
            int(self.model["max_aux_tokens"]),
            int(self.model["aux_dim"]),
            dtype=torch.float32,
        )
        aux_mask = torch.zeros(aux_values.shape[:-1], dtype=torch.bool)
        aux_type_ids = torch.zeros(aux_values.shape[:-1], dtype=torch.int64)
        result = {
            **action,
            **appearance,
            "world_tokens": context_tokens,
            "view_mask": frame["view_mask"][: self.T].bool(),
            "world_times_s": frame["frame_time_s"],
            "task_embedding": task_embedding,
            "aux_values": aux_values,
            "aux_mask": aux_mask,
            "aux_type_ids": aux_type_ids,
            "target_tokens": target_tokens,
            "target_token_mask": target_token_mask,
            "target_depth": target_depth,
            "target_depth_mask": target_depth_mask,
            "target_point": target_point,
            "target_point_mask": target_point_mask,
            "target_camera_pose": frame["camera_pose"][future],
            "target_camera_pose_mask": frame["camera_pose_mask"][future].bool(),
            "rgb_frame_indices": torch.tensor(self.rgb_indices, dtype=torch.int64),
            "target_rgb": target_rgb,
            "target_rgb_mask": rgb_view_mask[..., None, None, None],
            "source_id": torch.tensor(self.source_to_id[entry.source], dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }
        self._validate_sample(result, entry)
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._sample_from_entry(self.entries[index], sample_index=index)
