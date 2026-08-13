"""Compact episode-level robot cache shared by every WM3D model size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .grouped_robot import (
    ACTION_SEMANTIC_IDS,
    COMPOSITION_OPERATOR_IDS,
    STATE_SEMANTIC_IDS,
    EmbodimentSpec,
    GroupedRobotLimits,
    GroupedRobotContractError,
    RawActionSeries,
    RawStateSeries,
    RawStateSnapshot,
    compose_coarse_effects_for_observed_window,
    pack_grouped_robot_window,
    slice_fine_action_series,
)


EPISODE_ROBOT_SCHEMA = "wm3d_v8_episode_robot_ragged_v1"
SUPERVISION_IDS: Mapping[str, int] = {
    "unused": 0,
    "fine_command": 1,
    "coarse_effect": 2,
}


@dataclass(frozen=True)
class EpisodeRobotCache:
    observation_times_s: torch.Tensor
    embodiment_id: torch.Tensor
    group_ids: torch.Tensor
    group_mask: torch.Tensor
    supervision_ids: torch.Tensor
    action_semantic_ids: torch.Tensor
    state_semantic_ids: torch.Tensor
    composition_operator_ids: torch.Tensor
    action_offsets: torch.Tensor
    action_values: torch.Tensor
    action_value_mask: torch.Tensor
    action_times_s: torch.Tensor
    action_time_mask: torch.Tensor
    action_world_interval_indices: torch.Tensor
    state_offsets: torch.Tensor
    state_values: torch.Tensor
    state_value_mask: torch.Tensor
    state_times_s: torch.Tensor
    task_embedding: torch.Tensor

    def as_tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().contiguous()
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class PreparedEpisodeRobot:
    observation_times_s: np.ndarray
    action_series: tuple[RawActionSeries, ...]
    state_series: tuple[RawStateSeries, ...]
    task_embedding: torch.Tensor


def _mask(value: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=np.bool_)
    result = np.asarray(value, dtype=np.bool_)
    if result.shape != shape:
        raise GroupedRobotContractError(
            f"robot episode mask {result.shape} != values {shape}"
        )
    return result


def build_episode_robot_cache(
    *,
    embodiment: EmbodimentSpec,
    action_series: Sequence[RawActionSeries],
    state_series: Sequence[RawStateSeries],
    task_embedding: torch.Tensor,
    observation_times_s: Sequence[float],
    max_groups: int,
    max_action_dim: int,
    max_state_dim: int,
) -> EpisodeRobotCache:
    """Pack source-native ragged streams once, without T/K or rate padding."""

    if not 0 < len(embodiment.groups) <= int(max_groups):
        raise GroupedRobotContractError("embodiment group count exceeds cache capacity")
    task = torch.as_tensor(task_embedding, dtype=torch.float32).reshape(-1)
    if task.numel() <= 0 or not bool(torch.isfinite(task).all()):
        raise GroupedRobotContractError("task embedding must be finite/non-empty")
    observation_clock = np.asarray(observation_times_s, dtype=np.float64)
    if (
        observation_clock.ndim != 1
        or observation_clock.size < 2
        or not np.isfinite(observation_clock).all()
        or np.any(np.diff(observation_clock) <= 0)
    ):
        raise GroupedRobotContractError(
            "episode observation clock must be finite and strictly increasing"
        )
    action_by_group: dict[str, RawActionSeries] = {}
    for item in action_series:
        if item.group in action_by_group:
            raise GroupedRobotContractError(f"duplicate action series {item.group!r}")
        action_by_group[item.group] = item
    state_by_group: dict[str, RawStateSeries] = {}
    for item in state_series:
        if item.group in state_by_group:
            raise GroupedRobotContractError(f"duplicate state series {item.group!r}")
        state_by_group[item.group] = item
    expected_actions = {group.name for group in embodiment.groups}
    if set(action_by_group) != expected_actions:
        raise GroupedRobotContractError(
            f"episode action groups {sorted(action_by_group)} != {sorted(expected_actions)}"
        )
    expected_states = {
        group.name for group in embodiment.groups if group.state_dim > 0
    }
    if set(state_by_group) != expected_states:
        raise GroupedRobotContractError(
            f"episode state groups {sorted(state_by_group)} != {sorted(expected_states)}"
        )

    groups = int(max_groups)
    action_dim = int(max_action_dim)
    state_dim = int(max_state_dim)
    group_ids = torch.zeros(groups, dtype=torch.int64)
    group_mask = torch.zeros(groups, dtype=torch.bool)
    supervision = torch.zeros(groups, dtype=torch.int64)
    action_semantics = torch.zeros(groups, action_dim, dtype=torch.int64)
    state_semantics = torch.zeros(groups, state_dim, dtype=torch.int64)
    composition = torch.zeros(groups, action_dim, dtype=torch.int64)
    action_offsets = [0]
    state_offsets = [0]
    action_values: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    action_times: list[torch.Tensor] = []
    action_time_masks: list[torch.Tensor] = []
    action_intervals: list[torch.Tensor] = []
    state_values: list[torch.Tensor] = []
    state_masks: list[torch.Tensor] = []
    state_times: list[torch.Tensor] = []

    for slot in range(groups):
        if slot >= len(embodiment.groups):
            action_offsets.append(action_offsets[-1])
            state_offsets.append(state_offsets[-1])
            continue
        spec = embodiment.groups[slot]
        if spec.action_dim > action_dim or spec.state_dim > state_dim:
            raise GroupedRobotContractError(
                f"group {spec.name!r} exceeds episode-cache dimension capacity"
            )
        group_ids[slot] = spec.group_id
        group_mask[slot] = True
        action_semantics[slot, : spec.action_dim] = torch.tensor(
            [ACTION_SEMANTIC_IDS[item] for item in spec.action_semantics]
        )
        state_semantics[slot, : spec.state_dim] = torch.tensor(
            [STATE_SEMANTIC_IDS[item] for item in spec.state_semantics]
        )
        composition[slot, : spec.action_dim] = torch.tensor(
            [COMPOSITION_OPERATOR_IDS[item] for item in spec.composition_operators]
        )

        raw_action = action_by_group[spec.name]
        values = np.asarray(raw_action.values, dtype=np.float32)
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != spec.action_dim
            or not np.isfinite(values).all()
        ):
            raise GroupedRobotContractError(f"invalid action values for {spec.name!r}")
        padded = np.zeros((len(values), action_dim), dtype=np.float32)
        padded[:, : spec.action_dim] = values
        padded_mask = np.zeros_like(padded, dtype=np.bool_)
        padded_mask[:, : spec.action_dim] = _mask(
            raw_action.value_mask, values.shape
        )
        if raw_action.supervision == "fine_command":
            if raw_action.timestamps_s is None or raw_action.world_interval_indices is not None:
                raise GroupedRobotContractError("fine episode actions require only timestamps")
            timestamps = np.asarray(raw_action.timestamps_s, dtype=np.float64)
            if (
                timestamps.shape != (len(values),)
                or not np.isfinite(timestamps).all()
                or np.any(np.diff(timestamps) <= 0)
            ):
                raise GroupedRobotContractError("fine action clock is invalid")
            time_mask = np.ones(len(values), dtype=np.bool_)
            intervals = np.full(len(values), -1, dtype=np.int64)
        elif raw_action.supervision == "coarse_effect":
            if raw_action.timestamps_s is not None or raw_action.world_interval_indices is None:
                raise GroupedRobotContractError("coarse episode actions require interval indices")
            intervals = np.asarray(raw_action.world_interval_indices, dtype=np.int64)
            if (
                intervals.shape != (len(values),)
                or np.any(intervals < 0)
                or np.any(intervals >= observation_clock.size - 1)
                or np.unique(intervals).size != intervals.size
            ):
                raise GroupedRobotContractError("coarse action interval indices are invalid")
            timestamps = np.zeros(len(values), dtype=np.float64)
            time_mask = np.zeros(len(values), dtype=np.bool_)
        else:
            raise GroupedRobotContractError(
                f"unknown action supervision {raw_action.supervision!r}"
            )
        supervision[slot] = SUPERVISION_IDS[raw_action.supervision]
        action_values.append(torch.from_numpy(padded))
        action_masks.append(torch.from_numpy(padded_mask))
        action_times.append(torch.from_numpy(timestamps))
        action_time_masks.append(torch.from_numpy(time_mask))
        action_intervals.append(torch.from_numpy(intervals))
        action_offsets.append(action_offsets[-1] + len(values))

        if spec.state_dim == 0:
            state_offsets.append(state_offsets[-1])
            continue
        raw_state = state_by_group[spec.name]
        values_state = np.asarray(raw_state.values, dtype=np.float32)
        times_state = np.asarray(raw_state.timestamps_s, dtype=np.float64)
        if (
            values_state.ndim != 2
            or values_state.shape[0] < 1
            or values_state.shape[1] != spec.state_dim
            or times_state.shape != (len(values_state),)
            or not np.isfinite(values_state).all()
            or not np.isfinite(times_state).all()
            or np.any(np.diff(times_state) <= 0)
        ):
            raise GroupedRobotContractError(f"invalid state series for {spec.name!r}")
        padded_state = np.zeros((len(values_state), state_dim), dtype=np.float32)
        padded_state[:, : spec.state_dim] = values_state
        padded_state_mask = np.zeros_like(padded_state, dtype=np.bool_)
        padded_state_mask[:, : spec.state_dim] = _mask(
            raw_state.value_mask, values_state.shape
        )
        state_values.append(torch.from_numpy(padded_state))
        state_masks.append(torch.from_numpy(padded_state_mask))
        state_times.append(torch.from_numpy(times_state))
        state_offsets.append(state_offsets[-1] + len(values_state))

    return EpisodeRobotCache(
        observation_times_s=torch.from_numpy(observation_clock.copy()),
        embodiment_id=torch.tensor(embodiment.embodiment_id, dtype=torch.int64),
        group_ids=group_ids,
        group_mask=group_mask,
        supervision_ids=supervision,
        action_semantic_ids=action_semantics,
        state_semantic_ids=state_semantics,
        composition_operator_ids=composition,
        action_offsets=torch.tensor(action_offsets, dtype=torch.int64),
        action_values=torch.cat(action_values, dim=0),
        action_value_mask=torch.cat(action_masks, dim=0),
        action_times_s=torch.cat(action_times, dim=0),
        action_time_mask=torch.cat(action_time_masks, dim=0),
        action_world_interval_indices=torch.cat(action_intervals, dim=0),
        state_offsets=torch.tensor(state_offsets, dtype=torch.int64),
        state_values=(
            torch.cat(state_values, dim=0)
            if state_values
            else torch.empty(0, state_dim, dtype=torch.float32)
        ),
        state_value_mask=(
            torch.cat(state_masks, dim=0)
            if state_masks
            else torch.empty(0, state_dim, dtype=torch.bool)
        ),
        state_times_s=(
            torch.cat(state_times, dim=0)
            if state_times
            else torch.empty(0, dtype=torch.float64)
        ),
        task_embedding=task,
    )


def validate_episode_robot_tensors(values: Mapping[str, torch.Tensor]) -> None:
    required = set(EpisodeRobotCache.__dataclass_fields__)
    if set(values) != required:
        raise GroupedRobotContractError(
            f"episode robot tensors mismatch: missing={sorted(required-set(values))} "
            f"unknown={sorted(set(values)-required)}"
        )
    group_ids = values["group_ids"]
    groups = int(group_ids.numel())
    if group_ids.ndim != 1 or groups < 1:
        raise GroupedRobotContractError("episode robot group_ids must be non-empty rank-1")
    observation_clock = values["observation_times_s"]
    if (
        observation_clock.ndim != 1
        or observation_clock.numel() < 2
        or not bool(torch.isfinite(observation_clock).all())
        or not bool(torch.diff(observation_clock).gt(0).all())
    ):
        raise GroupedRobotContractError(
            "episode observation clock must be finite and strictly increasing"
        )
    for name in ("group_mask", "supervision_ids"):
        if tuple(values[name].shape) != (groups,):
            raise GroupedRobotContractError(f"{name} must align to group_ids")
    action_dim = int(values["action_values"].shape[-1])
    state_dim = int(values["state_values"].shape[-1])
    for name in ("action_semantic_ids", "composition_operator_ids"):
        if tuple(values[name].shape) != (groups, action_dim):
            raise GroupedRobotContractError(f"{name} shape is invalid")
    if tuple(values["state_semantic_ids"].shape) != (groups, state_dim):
        raise GroupedRobotContractError("state_semantic_ids shape is invalid")
    for offsets_name, values_name in (
        ("action_offsets", "action_values"),
        ("state_offsets", "state_values"),
    ):
        offsets = values[offsets_name]
        if (
            tuple(offsets.shape) != (groups + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != len(values[values_name])
            or bool((torch.diff(offsets) < 0).any())
        ):
            raise GroupedRobotContractError(f"{offsets_name} is invalid")
    for tensor in values.values():
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise GroupedRobotContractError("episode robot tensors contain NaN/Inf")


def unpack_episode_robot_tensors(
    values: Mapping[str, torch.Tensor], *, embodiment: EmbodimentSpec
) -> tuple[list[RawActionSeries], list[RawStateSeries]]:
    validate_episode_robot_tensors(values)
    if int(values["embodiment_id"]) != embodiment.embodiment_id:
        raise GroupedRobotContractError("episode robot embodiment id mismatch")
    actions: list[RawActionSeries] = []
    states: list[RawStateSeries] = []
    action_offsets = values["action_offsets"].tolist()
    state_offsets = values["state_offsets"].tolist()
    for slot, spec in enumerate(embodiment.groups):
        if int(values["group_ids"][slot]) != spec.group_id or not bool(
            values["group_mask"][slot]
        ):
            raise GroupedRobotContractError("episode robot group layout mismatch")
        left, right = int(action_offsets[slot]), int(action_offsets[slot + 1])
        supervision_id = int(values["supervision_ids"][slot])
        supervision = next(
            (name for name, identity in SUPERVISION_IDS.items() if identity == supervision_id),
            None,
        )
        if supervision not in {"fine_command", "coarse_effect"}:
            raise GroupedRobotContractError("episode robot supervision id is invalid")
        action_values = values["action_values"][left:right, : spec.action_dim].numpy()
        action_mask = values["action_value_mask"][left:right, : spec.action_dim].numpy()
        actions.append(
            RawActionSeries(
                group=spec.name,
                supervision=supervision,  # type: ignore[arg-type]
                values=action_values,
                timestamps_s=(
                    values["action_times_s"][left:right].numpy()
                    if supervision == "fine_command"
                    else None
                ),
                world_interval_indices=(
                    values["action_world_interval_indices"][left:right].numpy()
                    if supervision == "coarse_effect"
                    else None
                ),
                value_mask=action_mask,
            )
        )
        state_left, state_right = int(state_offsets[slot]), int(state_offsets[slot + 1])
        if spec.state_dim > 0:
            states.append(
                RawStateSeries(
                    group=spec.name,
                    values=values["state_values"][
                        state_left:state_right, : spec.state_dim
                    ].numpy(),
                    timestamps_s=values["state_times_s"][state_left:state_right].numpy(),
                    value_mask=values["state_value_mask"][
                        state_left:state_right, : spec.state_dim
                    ].numpy(),
                )
            )
        elif state_left != state_right:
            raise GroupedRobotContractError("stateless group owns cached state rows")
    return actions, states


def prepare_episode_robot_tensors(
    values: Mapping[str, torch.Tensor], *, embodiment: EmbodimentSpec
) -> PreparedEpisodeRobot:
    """Validate/decode one episode once per dataloader or index worker."""

    actions, states = unpack_episode_robot_tensors(values, embodiment=embodiment)
    clock = values["observation_times_s"].detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    task = values["task_embedding"].detach().cpu().to(torch.float32)
    return PreparedEpisodeRobot(
        observation_times_s=clock,
        action_series=tuple(actions),
        state_series=tuple(states),
        task_embedding=task,
    )


def _exact_current_states(
    state_series: Sequence[RawStateSeries], *, timestamp_s: float
) -> list[RawStateSnapshot]:
    output: list[RawStateSnapshot] = []
    for series in state_series:
        times = np.asarray(series.timestamps_s, dtype=np.float64)
        rows = np.flatnonzero(times == np.float64(timestamp_s))
        if len(rows) != 1:
            raise GroupedRobotContractError(
                f"group {series.group!r} has {len(rows)} exact states at policy start"
            )
        row = int(rows[0])
        value_mask = None
        if series.value_mask is not None:
            value_mask = np.asarray(series.value_mask, dtype=np.bool_)[row]
        output.append(
            RawStateSnapshot(
                group=series.group,
                timestamp_s=float(times[row]),
                values=np.asarray(series.values, dtype=np.float32)[row],
                value_mask=value_mask,
            )
        )
    return output


def assemble_robot_window_from_episode(
    *,
    values: Mapping[str, torch.Tensor],
    embodiment: EmbodimentSpec,
    selected_source_boundary_indices: Sequence[int],
    limits: "GroupedRobotLimits",
    context_samples: int,
    max_policy_queries: int,
    policy_target_horizon_s: float,
) -> dict[str, torch.Tensor]:
    """Assemble one model-profile window from compact source-native streams."""

    prepared = prepare_episode_robot_tensors(values, embodiment=embodiment)
    return assemble_robot_window_from_prepared_episode(
        prepared=prepared,
        embodiment=embodiment,
        selected_source_boundary_indices=selected_source_boundary_indices,
        limits=limits,
        context_samples=context_samples,
        max_policy_queries=max_policy_queries,
        policy_target_horizon_s=policy_target_horizon_s,
    )


def assemble_robot_window_from_prepared_episode(
    *,
    prepared: PreparedEpisodeRobot,
    embodiment: EmbodimentSpec,
    selected_source_boundary_indices: Sequence[int],
    limits: "GroupedRobotLimits",
    context_samples: int,
    max_policy_queries: int,
    policy_target_horizon_s: float,
) -> dict[str, torch.Tensor]:
    """Assemble a window without re-decoding the episode ragged streams."""

    if not isinstance(limits, GroupedRobotLimits):
        raise GroupedRobotContractError("limits must be GroupedRobotLimits")
    boundaries_indices = np.asarray(selected_source_boundary_indices, dtype=np.int64)
    clock = prepared.observation_times_s
    if (
        boundaries_indices.ndim != 1
        or boundaries_indices.size < 3
        or boundaries_indices[0] < 0
        or boundaries_indices[-1] >= clock.size
        or np.any(np.diff(boundaries_indices) <= 0)
    ):
        raise GroupedRobotContractError(
            "selected source boundaries must be valid strictly increasing rows"
        )
    boundaries = clock[boundaries_indices]
    interval_count = len(boundaries) - 1
    if interval_count < 2:
        raise GroupedRobotContractError("robot window needs history and future intervals")
    actions, states = prepared.action_series, prepared.state_series
    specs = {group.name: group for group in embodiment.groups}
    selected_actions: list[RawActionSeries] = []
    for series in actions:
        if series.supervision == "fine_command":
            selected_actions.append(
                slice_fine_action_series(
                    series, start_s=float(boundaries[0]), stop_s=float(boundaries[-1])
                )
            )
        else:
            selected_actions.append(
                compose_coarse_effects_for_observed_window(
                    series,
                    group=specs[series.group],
                    source_world_times_s=clock,
                    selected_boundary_indices=boundaries_indices,
                )
            )
    # Boundaries are exactly [leading, context(T), future(K)].  Therefore the
    # final context state/policy start is boundary position T.
    T = int(context_samples)
    K = interval_count - T
    if T <= 0 or K <= 0:
        raise GroupedRobotContractError("context_samples does not split history/future")
    policy_start = float(boundaries[T])
    packed = pack_grouped_robot_window(
        embodiment=embodiment,
        limits=limits,
        world_boundaries_s=boundaries,
        action_series=selected_actions,
        current_state=_exact_current_states(states, timestamp_s=policy_start),
        policy_chunk_start_s=policy_start,
    )

    G, A = limits.max_groups, limits.max_action_dim
    C = int(max_policy_queries)
    if C <= 0:
        raise GroupedRobotContractError("max_policy_queries must be positive")
    policy_dt = np.zeros((G, C), dtype=np.float32)
    policy_mask = np.zeros((G, C), dtype=np.bool_)
    target_fine = np.zeros((G, C, A), dtype=np.float32)
    target_fine_mask = np.zeros((G, C, A), dtype=np.bool_)
    for slot, spec in enumerate(embodiment.groups):
        series = actions[slot]
        if series.supervision == "coarse_effect":
            # Coarse-only sources supervise one real world-interval query per
            # future interval.  They do not become high-rate fine labels.
            query_times = boundaries[T:-1] - policy_start
            if len(query_times) > C:
                raise GroupedRobotContractError(
                    f"group {spec.name!r} coarse policy queries exceed capacity {C}"
                )
            policy_dt[slot, : len(query_times)] = query_times.astype(np.float32)
            policy_mask[slot, : len(query_times)] = True
            continue
        if series.timestamps_s is None:
            raise GroupedRobotContractError("fine action series lost its recorded clock")
        timestamps = np.asarray(series.timestamps_s, dtype=np.float64)
        # Policy chunks use the same half-open ownership rule as world
        # intervals.  A command exactly at the horizon end belongs to the next
        # chunk; including it here would supervise a query that cannot
        # contribute to any current factual world interval.
        keep = (timestamps >= policy_start) & (
            timestamps < policy_start + float(policy_target_horizon_s)
        )
        indices = np.flatnonzero(keep)
        if len(indices) > C:
            raise GroupedRobotContractError(
                f"group {spec.name!r} has {len(indices)} policy targets, capacity {C}"
            )
        # The policy-query clock is independent of the world-state clock.  A
        # source is not required to execute a command at the exact camera/state
        # timestamp; every recorded command inside the requested horizon is a
        # query at its real positive offset.
        count = len(indices)
        policy_dt[slot, :count] = (timestamps[indices] - policy_start).astype(np.float32)
        policy_mask[slot, :count] = True
        target_fine[slot, :count, : spec.action_dim] = np.asarray(
            series.values, dtype=np.float32
        )[indices]
        raw_mask = _mask(series.value_mask, np.asarray(series.values).shape)
        target_fine_mask[slot, :count, : spec.action_dim] = raw_mask[indices]

    return {
        "history_fine_action_values": torch.from_numpy(packed.fine_action_values[:T]),
        "history_fine_action_mask": torch.from_numpy(packed.fine_action_mask[:T]),
        "history_fine_action_dt": torch.from_numpy(packed.fine_action_dt[:T]),
        "history_fine_sample_mask": torch.from_numpy(packed.fine_sample_mask[:T]),
        "history_coarse_action_values": torch.from_numpy(packed.coarse_action_values[:T]),
        "history_coarse_action_mask": torch.from_numpy(packed.coarse_action_mask[:T]),
        "future_factual_fine_action_values": torch.from_numpy(packed.fine_action_values[T:]),
        "future_factual_fine_action_mask": torch.from_numpy(packed.fine_action_mask[T:]),
        "future_factual_fine_action_dt": torch.from_numpy(packed.fine_action_dt[T:]),
        "future_factual_fine_sample_mask": torch.from_numpy(packed.fine_sample_mask[T:]),
        "future_factual_coarse_action_values": torch.from_numpy(packed.coarse_action_values[T:]),
        "future_factual_coarse_action_mask": torch.from_numpy(packed.coarse_action_mask[T:]),
        "action_group_ids": torch.from_numpy(packed.group_ids),
        "action_group_mask": torch.from_numpy(packed.group_mask),
        "action_semantic_ids": torch.from_numpy(packed.action_semantic_ids),
        "composition_operator_ids": torch.from_numpy(packed.composition_operator_ids),
        "current_state_values": torch.from_numpy(packed.current_state_values),
        "current_state_mask": torch.from_numpy(packed.current_state_mask),
        "state_semantic_ids": torch.from_numpy(packed.state_semantic_ids),
        "embodiment_ids": torch.as_tensor(packed.embodiment_id),
        "policy_query_dt": torch.from_numpy(policy_dt),
        "policy_query_mask": torch.from_numpy(policy_mask),
        "target_fine_action": torch.from_numpy(target_fine),
        "target_fine_action_mask": torch.from_numpy(target_fine_mask),
        "target_coarse_action": torch.from_numpy(packed.coarse_action_values[T:]),
        "target_coarse_action_mask": torch.from_numpy(packed.coarse_action_mask[T:]),
        "future_world_boundaries_dt": torch.from_numpy(
            (boundaries[T:] - policy_start).astype(np.float32)
        ),
    }
