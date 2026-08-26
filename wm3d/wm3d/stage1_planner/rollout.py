"""One-shot candidate-conditioned rollout through the frozen unified Stage0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class NativeRollout:
    tokens: torch.Tensor
    future_dt_s: torch.Tensor
    token_mask: torch.Tensor
    depth: torch.Tensor
    depth_mask: torch.Tensor
    point: torch.Tensor
    point_mask: torch.Tensor
    pose: torch.Tensor
    pose_mask: torch.Tensor
    confidence: torch.Tensor
    view_mask: torch.Tensor


_CONTEXT_KEYS = (
    "world_tokens",
    "view_mask",
    "world_times_s",
    "task_embedding",
    "history_fine_action_values",
    "history_fine_action_mask",
    "history_fine_action_dt",
    "history_fine_sample_mask",
    "history_coarse_action_values",
    "history_coarse_action_mask",
    "action_group_ids",
    "action_group_mask",
    "action_semantic_ids",
    "current_state_values",
    "current_state_mask",
    "state_semantic_ids",
    "embodiment_ids",
    "policy_query_dt",
    "policy_query_mask",
    "action_normalization_offset",
    "action_normalization_scale",
    "state_normalization_offset",
    "state_normalization_scale",
    "aux_values",
    "aux_mask",
    "aux_type_ids",
)


def _expand_candidates(value: torch.Tensor, candidates: int) -> torch.Tensor:
    return value[:, None].expand(-1, candidates, *([-1] * (value.ndim - 1))).flatten(0, 1)


def _world_config(world: torch.nn.Module):
    current = world
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "cfg"):
            return current.cfg
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, torch.nn.Module):
            break
        current = wrapped
    raise ValueError("frozen Stage0 wrapper does not expose its sealed model config")


def single_horizon_native_rollout(
    world: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    horizon: int,
    candidate_microbatch: int = 0,
) -> NativeRollout:
    """Run exactly one trained Stage0 horizon, then slice to ``H <= K``.

    There is intentionally no autoregressive loop: the current native model
    has only been trained for one factual K-window.  Candidate actions already
    use the sealed grouped/timestamped/normalized ABI from the Stage1 branch
    materializer.
    """

    required = set(_CONTEXT_KEYS) | {
        "candidate_fine_action_values",
        "candidate_fine_action_mask",
        "candidate_fine_action_dt",
        "candidate_fine_sample_mask",
        "candidate_coarse_action_values",
        "candidate_coarse_action_mask",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"rollout batch misses unified fields: {missing}")
    cfg = _world_config(world)
    if not 0 < int(horizon) <= int(cfg.K):
        raise ValueError(f"Stage1 horizon H={horizon} must satisfy 0 < H <= K={cfg.K}")
    fine = batch["candidate_fine_action_values"]
    if fine.ndim != 6:
        raise ValueError("candidate fine actions must be [B,C,K,G,S,A]")
    bsz, candidates = fine.shape[:2]
    if candidates < 2:
        raise ValueError("Stage1 planning requires at least two candidates")
    for name in _CONTEXT_KEYS:
        value = batch[name]
        if not isinstance(value, torch.Tensor) or value.shape[0] != bsz:
            raise ValueError(f"unified context field {name} is not batched")
    expected_fine = (
        bsz, candidates, cfg.K, cfg.max_action_groups,
        fine.shape[-2], cfg.max_action_dim,
    )
    if tuple(fine.shape) != expected_fine or not 0 < fine.shape[-2] <= cfg.max_action_substeps:
        raise ValueError("candidate fine actions differ from sealed Stage0 grouped capacities")
    if batch["candidate_fine_action_mask"].shape != fine.shape:
        raise ValueError("candidate fine action mask mismatch")
    if (
        batch["candidate_fine_action_dt"].shape != fine.shape[:-1]
        or batch["candidate_fine_sample_mask"].shape != fine.shape[:-1]
    ):
        raise ValueError("candidate fine action timestamps/mask mismatch")
    coarse_shape = (bsz, candidates, cfg.K, cfg.max_action_groups, cfg.max_action_dim)
    if (
        tuple(batch["candidate_coarse_action_values"].shape) != coarse_shape
        or batch["candidate_coarse_action_mask"].shape != batch["candidate_coarse_action_values"].shape
    ):
        raise ValueError("candidate coarse actions differ from sealed Stage0 capacities")
    candidate_fields = {
        "future_factual_fine_action_values": "candidate_fine_action_values",
        "future_factual_fine_action_mask": "candidate_fine_action_mask",
        "future_factual_fine_action_dt": "candidate_fine_action_dt",
        "future_factual_fine_sample_mask": "candidate_fine_sample_mask",
        "future_factual_coarse_action_values": "candidate_coarse_action_values",
        "future_factual_coarse_action_mask": "candidate_coarse_action_mask",
    }
    micro = candidates if int(candidate_microbatch) <= 0 else int(candidate_microbatch)
    if micro <= 0:
        raise ValueError("candidate_microbatch must be non-negative")
    collected: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("pred_tokens", "depth", "point", "camera_pose", "geometry_confidence")
    }
    for start in range(0, candidates, micro):
        stop = min(candidates, start + micro)
        local = stop - start
        kwargs = {name: _expand_candidates(batch[name], local) for name in _CONTEXT_KEYS}
        for target, source in candidate_fields.items():
            value = batch[source][:, start:stop]
            kwargs[target] = value.flatten(0, 1)
        kwargs["rgb_frame_indices"] = ()
        output = world(**kwargs)
        for name in collected:
            value = output[name][:, :horizon]
            collected[name].append(value.reshape(bsz, local, *value.shape[1:]))
    merged = {name: torch.cat(values, dim=1) for name, values in collected.items()}
    future_dt = batch["world_times_s"][:, cfg.T : cfg.T + horizon]
    future_dt = future_dt - batch["world_times_s"][:, cfg.T - 1 : cfg.T]
    future_dt = future_dt[:, None].expand(-1, candidates, -1)
    token_mask = torch.ones(
        bsz, candidates, horizon, cfg.P, dtype=torch.bool, device=fine.device
    )
    view_mask = torch.ones(
        bsz, candidates, horizon, cfg.num_views, dtype=torch.bool, device=fine.device
    )
    evidence_mask = view_mask[..., None].expand(-1, -1, -1, -1, cfg.P)
    return NativeRollout(
        tokens=merged["pred_tokens"],
        future_dt_s=future_dt,
        token_mask=token_mask,
        depth=merged["depth"],
        depth_mask=evidence_mask,
        point=merged["point"],
        point_mask=evidence_mask,
        pose=merged["camera_pose"],
        pose_mask=view_mask,
        confidence=merged["geometry_confidence"],
        view_mask=view_mask,
    )
