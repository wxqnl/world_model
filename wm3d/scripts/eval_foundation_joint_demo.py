"""Export deterministic WSA-backed WM3D video demos from a committed checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch

from wm3d.data.foundation_input import build_foundation_input_adapter
from wm3d.models.foundation_world_model import FoundationWorldModel
from wm3d.models.model_factory import build_world_model
from wm3d.training.distributed_checkpoint import DistributedCheckpointManager
from wm3d.training.distributed_runtime import (
    destroy_distributed,
    initialize_distributed,
    strategy_from_mapping,
    wrap_model,
)
from wm3d.training.pretrain import (
    _batch_to_device,
    _build_mixed_dataset,
    _configure_reproducibility,
    _make_loader,
    _resume_expectations,
)
from wm3d.training.runtime_contract import load_materialized_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7340)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--duration-ms", type=int, default=350)
    parser.add_argument("--skip-zero-action", action="store_true")
    return parser.parse_args()


def _target_motion_score(batch: Mapping[str, Any], *, context: int, future: int) -> float:
    images = batch["direct_rgb_uint8"][0, context - 1 : context + future].float()
    mask = batch["direct_view_mask"][0, context - 1 : context + future].bool()
    delta = images[1:] - images[:-1]
    pair_mask = mask[1:] & mask[:-1]
    if not bool(pair_mask.any()):
        return 0.0
    per_view = delta.square().mean(dim=(2, 3, 4)).sqrt()
    return float(per_view[pair_mask].mean().item() / 255.0)


def _select_high_motion_batch(
    config: Mapping[str, Any], *, rank: int, world_size: int, count: int, seed: int
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    dataset, profile = _build_mixed_dataset(
        config, split="val", device=torch.device("cpu"), rank=rank
    )
    loader = _make_loader(
        dataset,
        profile,
        config["runtime_profile"],
        rank=rank,
        world_size=world_size,
        start_step=0,
        num_steps=count,
        seed=seed,
        gradient_accumulation=1,
        micro_batch_size=1,
    )
    model_config = config["model_profile"]["model"]
    best: dict[str, Any] | None = None
    best_score = -1.0
    for batch in loader:
        score = _target_motion_score(
            batch,
            context=int(model_config["T"]),
            future=int(model_config["K"]),
        )
        if score > best_score:
            best = dict(batch)
            best_score = score
    if best is None:
        raise RuntimeError("validation sampler produced no demo candidate")
    source_id = int(best["source_id"].reshape(-1)[0].item())
    source_names = tuple(str(name) for name in profile.source_order)
    metadata = {
        "candidate_count": count,
        "selection_motion_rms": best_score,
        "sample_index": int(best["sample_index"].reshape(-1)[0].item()),
        "source_id": source_id,
        "source_name": source_names[source_id],
        "task_text": str(best["task_text"][0]),
    }
    return best, metadata


def _tensor_video_to_images(video: torch.Tensor) -> list[Image.Image]:
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"target video has invalid shape {tuple(video.shape)}")
    value = video[0].detach().float().clamp(-1, 1).add(1).mul(127.5)
    value = value.round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in value]


def _normalize_images(frames: Sequence[Image.Image], expected: int) -> list[Image.Image]:
    result = [frame.convert("RGB") for frame in frames]
    if len(result) != expected:
        raise ValueError(f"generated video has {len(result)} frames, expected {expected}")
    if len({frame.size for frame in result}) != 1:
        raise ValueError("generated video frame sizes disagree")
    return result


def _save_gif(path: Path, frames: Sequence[Image.Image], duration_ms: int) -> None:
    values = list(frames)
    values[0].save(
        path,
        save_all=True,
        append_images=values[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def _labelled_panel(
    columns: Sequence[tuple[str, Sequence[Image.Image]]]
) -> list[Image.Image]:
    frame_count = len(columns[0][1])
    if any(len(frames) != frame_count for _name, frames in columns):
        raise ValueError("panel columns have different temporal lengths")
    width, height = columns[0][1][0].size
    header = 28
    result: list[Image.Image] = []
    for index in range(frame_count):
        canvas = Image.new("RGB", (width * len(columns), height + header), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (name, frames) in enumerate(columns):
            frame = frames[index]
            if frame.size != (width, height):
                raise ValueError("panel frame sizes changed")
            canvas.paste(frame, (column * width, header))
            draw.text((column * width + 8, 7), f"{name}  t={index}", fill="black")
        result.append(canvas)
    return result


def _split_views(frames: Sequence[Image.Image]) -> dict[str, list[Image.Image]]:
    width, height = frames[0].size
    top_height = height - height // 3
    bottom_height = height // 3
    half_width = width // 2
    boxes = {
        "view0": (0, 0, width, top_height),
        "view1": (0, top_height, half_width, top_height + bottom_height),
        "view2": (half_width, top_height, width, top_height + bottom_height),
    }
    return {
        name: [
            frame.crop(box).resize((width, top_height), Image.Resampling.LANCZOS)
            for frame in frames
        ]
        for name, box in boxes.items()
    }


def _video_array(frames: Sequence[Image.Image]) -> np.ndarray:
    return np.stack([np.asarray(frame, dtype=np.float32) / 255.0 for frame in frames])


def _temporal_metrics(
    target: Sequence[Image.Image],
    factual: Sequence[Image.Image],
    zero: Sequence[Image.Image] | None,
) -> dict[str, Any]:
    target_array = _video_array(target)
    factual_array = _video_array(factual)
    if target_array.shape != factual_array.shape:
        raise ValueError("target and factual videos have different shapes")
    target_delta = np.diff(target_array, axis=0)
    factual_delta = np.diff(factual_array, axis=0)

    def rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))

    target_delta_rms = rms(target_delta)
    factual_delta_rms = rms(factual_delta)
    denominator = float(
        np.linalg.norm(target_delta.reshape(-1))
        * np.linalg.norm(factual_delta.reshape(-1))
    )
    temporal_direction = (
        0.0
        if denominator <= 1.0e-12
        else float(np.dot(target_delta.reshape(-1), factual_delta.reshape(-1)) / denominator)
    )
    spatial_std = factual_array[1:].std(axis=(1, 2, 3))
    metrics: dict[str, Any] = {
        "target_frame_delta_rms": target_delta_rms,
        "factual_frame_delta_rms": factual_delta_rms,
        "factual_frame_delta_retention": factual_delta_rms / max(target_delta_rms, 1.0e-12),
        "factual_temporal_direction_cosine": temporal_direction,
        "factual_target_l1": float(np.mean(np.abs(factual_array - target_array))),
        "factual_spatial_std_min": float(spatial_std.min()),
        "factual_is_single_color": bool(float(spatial_std.min()) < 0.01),
        "factual_is_temporally_identical": bool(np.max(np.abs(factual_delta)) < (0.5 / 255.0)),
    }
    if zero is not None:
        zero_array = _video_array(zero)
        if zero_array.shape != factual_array.shape:
            raise ValueError("zero-action and factual videos have different shapes")
        metrics.update(
            {
                "zero_frame_delta_rms": rms(np.diff(zero_array, axis=0)),
                "factual_zero_response_rms": rms(factual_array[1:] - zero_array[1:]),
            }
        )
    if not all(
        math.isfinite(value)
        for value in metrics.values()
        if isinstance(value, float)
    ):
        raise FloatingPointError("demo metrics contain non-finite values")
    return metrics


def _save_outputs(
    output: Path,
    *,
    target: Sequence[Image.Image],
    factual: Sequence[Image.Image],
    zero: Sequence[Image.Image] | None,
    duration_ms: int,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=False)
    columns: list[tuple[str, Sequence[Image.Image]]] = [
        ("target", target),
        ("factual action", factual),
    ]
    if zero is not None:
        columns.append(("zero action", zero))
    paths: list[Path] = []
    for name, frames in columns:
        path = output / f"{name.replace(' ', '_')}.gif"
        _save_gif(path, frames, duration_ms)
        paths.append(path)
    panel = output / "comparison.gif"
    _save_gif(panel, _labelled_panel(columns), duration_ms)
    paths.append(panel)
    split = {name: _split_views(frames) for name, frames in columns}
    for view_name in ("view0", "view1", "view2"):
        view_columns = [(name, value[view_name]) for name, value in split.items()]
        path = output / f"comparison_{view_name}.gif"
        _save_gif(path, _labelled_panel(view_columns), duration_ms)
        paths.append(path)
    return [str(path) for path in paths]


def main() -> None:
    args = parse_args()
    config, config_digest = load_materialized_runtime(args.runtime)
    runtime = config["runtime_profile"]
    strategy = strategy_from_mapping(runtime["distributed"])
    context = initialize_distributed(strategy)
    try:
        if context.world_size != 1 or context.rank != 0:
            raise RuntimeError("foundation demo export requires exactly one rank")
        if args.candidate_count <= 0 or args.num_inference_steps <= 0:
            raise ValueError("candidate count and inference steps must be positive")
        _configure_reproducibility(args.seed)
        cpu_batch, selection = _select_high_motion_batch(
            config,
            rank=context.rank,
            world_size=context.world_size,
            count=args.candidate_count,
            seed=args.seed,
        )
        input_adapter = build_foundation_input_adapter(config, device=context.device)
        input_adapter.eval()
        foundation_batch = input_adapter.materialize(cpu_batch)
        batch = _batch_to_device(foundation_batch, context.device)
        raw_model = build_world_model(config["model_profile"], device=context.device)
        if not isinstance(raw_model, FoundationWorldModel):
            raise RuntimeError("runtime did not build a foundation world model")
        wrapped = wrap_model(raw_model, context, strategy)
        checkpoint_step = int(args.checkpoint.name.split("_")[-1])
        expectations = _resume_expectations(
            config,
            config_digest,
            step=checkpoint_step,
            world_size=context.world_size,
        )
        manager = DistributedCheckpointManager(args.checkpoint.parent)
        manager.load_model_for_evaluation(
            path=args.checkpoint,
            model=wrapped.model,
            expected=expectations,
        )
        raw_model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            factual_result = raw_model.predict_joint(
                batch,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
            )
        zero_result = None
        if not args.skip_zero_action:
            torch.cuda.empty_cache()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zero_result = raw_model.predict_joint(
                    batch,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed,
                    zero_future_action=True,
                )
        frame_count = int(batch["video"].shape[2])
        target = _tensor_video_to_images(batch["video"])
        factual = _normalize_images(factual_result["video"], frame_count)
        zero = (
            None
            if zero_result is None
            else _normalize_images(zero_result["video"], frame_count)
        )
        paths = _save_outputs(
            args.output,
            target=target,
            factual=factual,
            zero=zero,
            duration_ms=args.duration_ms,
        )
        metrics = _temporal_metrics(target, factual, zero)
        predicted_action = factual_result.get("action_normalized")
        if isinstance(predicted_action, torch.Tensor):
            predicted_action = predicted_action.detach().float().cpu()
            target_action = batch["action"][0].detach().float().cpu()
            mask = batch["action_element_mask"][0].detach().bool().cpu()
            metrics["joint_action_normalized_mae"] = float(
                (predicted_action - target_action).abs()[mask].mean().item()
            )
        receipt = {
            "schema": "wm3d_foundation_joint_demo_v1",
            "checkpoint_step": checkpoint_step,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "frame_count": frame_count,
            "selection": selection,
            "metrics": metrics,
            "files": paths,
        }
        receipt_path = args.output / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
