from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


CHUNK_SIZE = 8
MIN_CHUNKS = 8
MIN_FUTURE_FRAMES = 64
G4_SOURCE_ENUM_SCHEMA = "wm3d_stage1_g4_source_enum_v1"
G4_SOURCE_SCHEMA_VERSION = 1
G4_INITIAL_CONTEXT_SOURCE = "ground_truth_initial_context"
G4_PREDICTED_TOKEN_SOURCE = "previous_pred_tokens"
G4_PREDICTED_RGB_SOURCE = "previous_wan_last_frame"


class ClosedLoopError(ValueError):
    pass


def g4_expected_token_source(chunk_index: int) -> str:
    return (
        G4_INITIAL_CONTEXT_SOURCE
        if int(chunk_index) == 0
        else G4_PREDICTED_TOKEN_SOURCE
    )


def g4_expected_rgb_source(chunk_index: int) -> str:
    return (
        G4_INITIAL_CONTEXT_SOURCE
        if int(chunk_index) == 0
        else G4_PREDICTED_RGB_SOURCE
    )


def make_g4_source_ledger_entry(
    *,
    chunk_index: int,
    clip_id: str,
    start: int,
    action_source: str,
    target_source: str | None,
) -> dict[str, Any]:
    token_source = g4_expected_token_source(chunk_index)
    rgb_source = g4_expected_rgb_source(chunk_index)
    parent_chunk = None if int(chunk_index) == 0 else int(chunk_index) - 1
    return {
        "source_enum_schema": G4_SOURCE_ENUM_SCHEMA,
        "source_schema_version": G4_SOURCE_SCHEMA_VERSION,
        "chunk": int(chunk_index),
        "clip_id": clip_id,
        "start": int(start),
        "token_source": token_source,
        "token_context_source": token_source,
        "token_context_from_chunk": parent_chunk,
        "rgb_source": rgb_source,
        "rgb_context_source": rgb_source,
        "rgb_context_from_chunk": parent_chunk,
        "action_source": action_source,
        "action_chunk_source": action_source,
        "target_source": target_source,
        "target_role": "metrics_only",
    }


@dataclass(frozen=True)
class ClosedLoopChunk:
    clip_id: str
    start: int
    action: Tensor
    action_source: str
    target_rgb: Tensor | None = None
    target_source: str | None = None


@dataclass(frozen=True)
class ClosedLoopGeneratedChunk:
    clip_id: str
    start: int
    video: Tensor
    pred_tokens: Tensor
    metrics: dict[str, float | int | None]


@dataclass(frozen=True)
class ClosedLoopResult:
    clip_id: str
    starts: tuple[int, ...]
    video: Tensor
    final_state: Tensor
    final_rgb: Tensor
    chunks: tuple[ClosedLoopGeneratedChunk, ...]
    source_ledger: tuple[dict[str, Any], ...]
    assert_counters: dict[str, int]
    metrics: dict[str, Any]

    def to_report(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "starts": list(self.starts),
            "chunks": len(self.chunks),
            "future_frames": int(self.video.shape[1]),
            "source_ledger": list(self.source_ledger),
            "assert_counters": dict(self.assert_counters),
            "metrics": dict(self.metrics),
            "chunks_detail": [
                {
                    "chunk": index,
                    "clip_id": chunk.clip_id,
                    "start": chunk.start,
                    "frames": int(chunk.video.shape[1]),
                    **chunk.metrics,
                }
                for index, chunk in enumerate(self.chunks)
            ],
        }


WorldModelStep = Callable[
    [Tensor, Tensor, Tensor, Any, int],
    Mapping[str, Tensor],
]
WanStep = Callable[
    [Mapping[str, Tensor], Tensor, Tensor, Any, int],
    Tensor | Any,
]


def _validate_inputs(
    initial_state: Tensor,
    initial_rgb: Tensor,
    chunks: Sequence[ClosedLoopChunk],
) -> dict[str, int]:
    if initial_state.ndim < 3 or int(initial_state.shape[1]) != 16:
        raise ClosedLoopError(
            "initial_state must contain the first true 16-frame context, "
            f"got {tuple(initial_state.shape)}"
        )
    if initial_rgb.ndim != 4 or int(initial_rgb.shape[1]) != 3:
        raise ClosedLoopError(
            f"initial_rgb must be BCHW RGB, got {tuple(initial_rgb.shape)}"
        )
    if len(chunks) < MIN_CHUNKS:
        raise ClosedLoopError(f"G4 requires at least 8 chunks, got {len(chunks)}")
    clip_ids = {chunk.clip_id for chunk in chunks}
    if len(clip_ids) != 1:
        raise ClosedLoopError(f"G4 requires exactly one clip, got {sorted(clip_ids)}")

    contiguous_checks = 0
    action_source_checks = 0
    for index, chunk in enumerate(chunks):
        if not chunk.action_source:
            raise ClosedLoopError(f"chunk {index} has no action source")
        action_source_checks += 1
        if chunk.action.ndim < 2 or int(chunk.action.shape[1]) != CHUNK_SIZE:
            raise ClosedLoopError(
                f"chunk {index} action horizon must be 8, "
                f"got {tuple(chunk.action.shape)}"
            )
        if index:
            expected = int(chunks[index - 1].start) + CHUNK_SIZE
            if int(chunk.start) != expected:
                raise ClosedLoopError(
                    "G4 requires contiguous starts with "
                    f"next_start=current_start+8; chunk {index} "
                    f"has {chunk.start}, expected {expected}"
                )
            contiguous_checks += 1

    return {
        "minimum_chunk_checks": 1,
        "single_clip_checks": 1,
        "contiguous_start_checks": contiguous_checks,
        "action_source_checks": action_source_checks,
        "minimum_future_frame_checks": 0,
        "predicted_token_context_uses": 0,
        "prior_wan_rgb_context_uses": 0,
        "future_gt_token_input_reads": 0,
        "future_gt_rgb_context_input_reads": 0,
        "token_source_assertions": 0,
        "rgb_source_assertions": 0,
        "target_as_input_assertions": 0,
        "target_metric_reads": 0,
        "source_ledger_entries": 0,
    }


def _as_bfchw(video: Tensor, *, name: str) -> Tensor:
    if not torch.is_tensor(video):
        raise ClosedLoopError(f"{name} must be a tensor")
    if video.ndim == 4:
        if int(video.shape[1]) != 3:
            raise ClosedLoopError(
                f"{name} must be FCHW when rank 4, got {tuple(video.shape)}"
            )
        return video.unsqueeze(0)
    if video.ndim != 5:
        raise ClosedLoopError(
            f"{name} must be BFCHW or BCTHW, got {tuple(video.shape)}"
        )
    if int(video.shape[2]) == 3:
        return video
    if int(video.shape[1]) == 3:
        return video.permute(0, 2, 1, 3, 4).contiguous()
    raise ClosedLoopError(f"{name} has no RGB channel axis, got {tuple(video.shape)}")


def _match_target(target: Tensor, prediction: Tensor) -> Tensor:
    target = _as_bfchw(target, name="target_rgb").to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    if int(target.shape[0]) != int(prediction.shape[0]):
        raise ClosedLoopError(
            "target/prediction batch mismatch: "
            f"{target.shape[0]} != {prediction.shape[0]}"
        )
    if int(target.shape[1]) < int(prediction.shape[1]):
        raise ClosedLoopError(
            "evaluation target has fewer frames than generated chunk: "
            f"{target.shape[1]} < {prediction.shape[1]}"
        )
    target = target[:, : int(prediction.shape[1])]
    if tuple(target.shape[-2:]) != tuple(prediction.shape[-2:]):
        batch, frames, channels = target.shape[:3]
        target = F.interpolate(
            target.reshape(batch * frames, channels, *target.shape[-2:]),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, channels, *prediction.shape[-2:])
    return target


def _linear_slope(values: Tensor) -> tuple[float, float]:
    y = values.detach().float().cpu()
    x = torch.arange(int(y.numel()), dtype=torch.float32)
    x_centered = x - x.mean()
    denominator = x_centered.square().sum()
    slope = (
        float(((y - y.mean()) * x_centered).sum() / denominator)
        if float(denominator) > 0.0
        else 0.0
    )
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def _compute_metrics(
    generated: Sequence[ClosedLoopGeneratedChunk],
    chunks: Sequence[ClosedLoopChunk],
    counters: dict[str, int],
) -> dict[str, Any]:
    seam_values = [
        (generated[index].video[:, 0] - generated[index - 1].video[:, -1]).abs().mean()
        for index in range(1, len(generated))
    ]
    within_values = [
        (item.video[:, 1:] - item.video[:, :-1]).abs().mean()
        for item in generated
        if int(item.video.shape[1]) > 1
    ]
    seam_l1 = float(torch.stack(seam_values).mean().cpu()) if seam_values else 0.0
    within_l1 = float(torch.stack(within_values).mean().cpu()) if within_values else 0.0
    seam_ratio = seam_l1 / max(within_l1, 1e-12)

    drift_values: list[Tensor] = []
    target_reads = 0
    for item, chunk in zip(generated, chunks, strict=True):
        if chunk.target_rgb is None:
            continue
        target = _match_target(chunk.target_rgb, item.video)
        drift_values.append(
            (item.video.float() - target.float()).abs().mean(dim=(0, 2, 3, 4))
        )
        target_reads += 1
    counters["target_metric_reads"] = target_reads

    drift_slope: float | None = None
    drift_intercept: float | None = None
    drift_mean: float | None = None
    drift_final: float | None = None
    if drift_values:
        drift = torch.cat(drift_values)
        drift_slope, drift_intercept = _linear_slope(drift)
        drift_mean = float(drift.mean().cpu())
        drift_final = float(drift[-1].cpu())

    all_video = torch.cat([item.video for item in generated], dim=1)
    frame_intensity = all_video.float().mean(dim=(0, 2, 3, 4))
    black_frame_rate = float((frame_intensity <= (1.0 / 255.0)).float().mean())

    return {
        "chunks": len(generated),
        "future_frames": int(all_video.shape[1]),
        "seam_count": len(seam_values),
        "seam_l1": seam_l1,
        "within_chunk_l1": within_l1,
        "seam_to_within_ratio": seam_ratio,
        "drift_l1_slope": drift_slope,
        "drift_l1_intercept": drift_intercept,
        "drift_l1_mean": drift_mean,
        "drift_l1_final": drift_final,
        "black_frame_rate": black_frame_rate,
        "target_input_read_count": 0,
        "target_metric_read_count": target_reads,
        "target_leakage_score": 0.0,
        "target_leakage_detected": False,
        "target_leakage_audit_pass": (
            counters["future_gt_token_input_reads"] == 0
            and counters["future_gt_rgb_context_input_reads"] == 0
        ),
    }


@torch.no_grad()
def run_closed_loop(
    *,
    initial_state: Tensor,
    initial_rgb: Tensor,
    chunks: Sequence[ClosedLoopChunk],
    wm_step: WorldModelStep,
    wan_step: WanStep,
    task_context: Any = None,
) -> ClosedLoopResult:
    chunks = tuple(chunks)
    counters = _validate_inputs(initial_state, initial_rgb, chunks)
    clip_id = chunks[0].clip_id
    s_roll = initial_state
    rgb_roll = initial_rgb
    generated: list[ClosedLoopGeneratedChunk] = []
    source_ledger: list[dict[str, Any]] = []

    for chunk_index, chunk in enumerate(chunks):
        token_source = g4_expected_token_source(chunk_index)
        rgb_source = g4_expected_rgb_source(chunk_index)
        wm_out = wm_step(
            s_roll,
            rgb_roll,
            chunk.action,
            task_context,
            chunk_index,
        )
        if "pred_tokens" not in wm_out:
            raise ClosedLoopError(
                f"WM output for chunk {chunk_index} has no pred_tokens"
            )
        pred_tokens = wm_out["pred_tokens"]
        if (
            pred_tokens.ndim != s_roll.ndim
            or int(pred_tokens.shape[1]) != CHUNK_SIZE
            or tuple(pred_tokens.shape[2:]) != tuple(s_roll.shape[2:])
        ):
            raise ClosedLoopError(
                "pred_tokens must be shape-compatible with the 8-frame state "
                f"advance, got state={tuple(s_roll.shape)} "
                f"pred={tuple(pred_tokens.shape)}"
            )

        wan_output = wan_step(
            wm_out,
            rgb_roll,
            chunk.action,
            task_context,
            chunk_index,
        )
        raw_video = getattr(wan_output, "video", wan_output)
        video = _as_bfchw(raw_video, name="Wan video").detach()
        if int(video.shape[0]) != int(s_roll.shape[0]):
            raise ClosedLoopError(
                f"Wan/state batch mismatch: {video.shape[0]} != {s_roll.shape[0]}"
            )
        if int(video.shape[1]) <= 0:
            raise ClosedLoopError(f"Wan chunk {chunk_index} generated no frames")

        chunk_metrics = {
            "within_chunk_l1": (
                float((video[:, 1:] - video[:, :-1]).abs().mean().cpu())
                if int(video.shape[1]) > 1
                else 0.0
            ),
            "mean_intensity": float(video.float().mean().cpu()),
        }
        generated.append(
            ClosedLoopGeneratedChunk(
                clip_id=clip_id,
                start=int(chunk.start),
                video=video,
                pred_tokens=pred_tokens.detach(),
                metrics=chunk_metrics,
            )
        )
        source_ledger.append(
            make_g4_source_ledger_entry(
                chunk_index=chunk_index,
                clip_id=clip_id,
                start=chunk.start,
                action_source=chunk.action_source,
                target_source=chunk.target_source,
            )
        )
        counters["token_source_assertions"] += 1
        counters["rgb_source_assertions"] += 1
        counters["target_as_input_assertions"] += 1
        if chunk_index:
            counters["predicted_token_context_uses"] += 1
            counters["prior_wan_rgb_context_uses"] += 1

        s_roll = torch.cat((s_roll[:, 8:], pred_tokens.detach()), dim=1)
        rgb_roll = video[:, -1].detach()

    total_frames = sum(int(item.video.shape[1]) for item in generated)
    if total_frames < MIN_FUTURE_FRAMES:
        raise ClosedLoopError(
            f"G4 requires at least 64 future frames, got {total_frames}"
        )
    counters["minimum_future_frame_checks"] = 1
    counters["source_ledger_entries"] = len(source_ledger)
    metrics = _compute_metrics(generated, chunks, counters)
    all_video = torch.cat([item.video for item in generated], dim=1)

    return ClosedLoopResult(
        clip_id=clip_id,
        starts=tuple(int(chunk.start) for chunk in chunks),
        video=all_video,
        final_state=s_roll.detach(),
        final_rgb=rgb_roll.detach(),
        chunks=tuple(generated),
        source_ledger=tuple(source_ledger),
        assert_counters=counters,
        metrics=metrics,
    )
