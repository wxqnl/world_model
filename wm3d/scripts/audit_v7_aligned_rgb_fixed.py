"""Single-GPU fixed-sample audit for the V8 model with aligned RGB renderer."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Mapping

import lpips
from PIL import Image, ImageDraw
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.native_objective import objective_config_from_mapping
from wm3d.training.pretrain import _batch_to_device, _forward_with_action_counterfactual
from wm3d.training.runtime_contract import load_materialized_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-gif-root", type=Path)
    parser.add_argument("--old-audit", type=Path)
    return parser.parse_args()


def load_full_model(model: torch.nn.Module, checkpoint: Path) -> None:
    options = StateDictOptions(full_state_dict=False, cpu_offload=False)
    state = get_model_state_dict(model, options=options)
    dcp.load({"model": state}, checkpoint_id=checkpoint / "distcp")
    incompatible = set_model_state_dict(model, state, options=options)
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"model load incomplete: missing={missing[:8]} unexpected={unexpected[:8]}"
        )


def slot_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return batch["target_rgb_mask"][..., 0, 0, 0].bool()


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = torch.broadcast_to(mask, value.shape).to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def motion_amplitude(
    video: torch.Tensor, context: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    value = (video.float() - context[:, None].float()).square().mean(
        dim=(3, 4, 5)
    ).sqrt()
    return masked_mean(value, slots)


def frame_delta(
    video: torch.Tensor, context: torch.Tensor, slots: torch.Tensor
) -> torch.Tensor:
    clip = torch.cat((context[:, None], video), dim=1).float()
    value = (clip[:, 1:] - clip[:, :-1]).square().mean(
        dim=(3, 4, 5)
    ).sqrt()
    return masked_mean(value, slots)


def gradient_energy(video: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
    value = video.float()
    dx = (value[..., :, 1:] - value[..., :, :-1]).abs().mean(dim=3)
    dy = (value[..., 1:, :] - value[..., :-1, :]).abs().mean(dim=3)
    mask_x = motion[..., :, 1:] | motion[..., :, :-1]
    mask_y = motion[..., 1:, :] | motion[..., :-1, :]
    return 0.5 * (masked_mean(dx, mask_x) + masked_mean(dy, mask_y))


@torch.no_grad()
def lpips_distance(
    network: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    valid = torch.nonzero(slots.reshape(-1), as_tuple=False).flatten()
    prediction = prediction.float().reshape(-1, *prediction.shape[-3:])
    target = target.float().reshape(-1, *target.shape[-3:])
    prediction = prediction.index_select(0, valid)
    target = target.index_select(0, valid)
    values = []
    with torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, prediction.shape[0], 8):
            values.append(
                network(
                    prediction[start : start + 8].mul(2).sub(1),
                    target[start : start + 8].mul(2).sub(1),
                    normalize=False,
                ).reshape(-1)
            )
    return torch.cat(values).mean()


def audit_metrics(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    perceptual: torch.nn.Module,
    threshold: float,
) -> dict[str, float]:
    target = batch["target_rgb"].float().clamp(0, 1)
    context = batch["context_rgb"].float().clamp(0, 1)
    prediction = prediction.float().clamp(0, 1)
    copy_last = context[:, None].expand_as(target)
    slots = slot_mask(batch)
    pixel_slots = slots[..., None, None]
    motion = ((target - context[:, None]).abs().mean(dim=3) > threshold) & pixel_slots
    static = (~motion) & pixel_slots
    target_motion = motion_amplitude(target, context, slots)
    target_delta = frame_delta(target, context, slots)
    target_gradient = gradient_energy(target, motion)
    copy_motion_error = masked_mean(
        (copy_last - target).abs().mean(dim=3), motion
    )
    result: dict[str, torch.Tensor] = {
        "target_motion_amplitude_rms": target_motion,
        "target_frame_delta_amplitude_rms": target_delta,
        "target_motion_gradient_energy": target_gradient,
    }
    for name, video in (("prediction", prediction), ("copy_last", copy_last)):
        absolute = (video - target).abs().mean(dim=3)
        square = (video - target).square().mean(dim=3)
        amplitude = motion_amplitude(video, context, slots)
        delta = frame_delta(video, context, slots)
        sharpness = gradient_energy(video, motion)
        result.update(
            {
                f"{name}_l1": masked_mean(absolute, pixel_slots),
                f"{name}_psnr_db": -10
                * torch.log10(masked_mean(square, pixel_slots).clamp_min(1e-10)),
                f"{name}_lpips": lpips_distance(perceptual, video, target, slots),
                f"{name}_motion_l1": masked_mean(absolute, motion),
                f"{name}_static_l1": masked_mean(absolute, static),
                f"{name}_motion_amplitude_rms": amplitude,
                f"{name}_to_target_motion_amplitude_ratio": amplitude
                / (target_motion + 1e-6),
                f"{name}_frame_delta_amplitude_rms": delta,
                f"{name}_to_target_frame_delta_ratio": delta / (target_delta + 1e-6),
                f"{name}_motion_gradient_energy": sharpness,
                f"{name}_to_target_motion_gradient_ratio": sharpness
                / (target_gradient + 1e-6),
                f"{name}_moving_region_residual_vs_copy_last": masked_mean(
                    absolute, motion
                )
                / (copy_motion_error + 1e-6),
            }
        )
    values = {name: float(value) for name, value in result.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("non-finite RGB audit metric")
    return values


def rgb_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def tile(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    output = Image.new("RGB", (image.width, image.height + 40), "white")
    draw = ImageDraw.Draw(output)
    draw.text((6, 4), title, fill="black")
    draw.text((6, 21), subtitle, fill="black")
    output.paste(image, (0, 40))
    return output


def old_prediction_frames(path: Path) -> list[Image.Image]:
    image = Image.open(path)
    frames = []
    for index in range(getattr(image, "n_frames", 1)):
        image.seek(index)
        frame = image.convert("RGB")
        frames.append(frame.crop((512, 0, 768, frame.height)))
    return frames


def save_gifs(
    root: Path,
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    old_root: Path | None,
) -> list[str]:
    root.mkdir(parents=True, exist_ok=False)
    target = batch["target_rgb"][0]
    context = batch["context_rgb"][0]
    slots = slot_mask(batch)[0]
    source_id = int(batch["source_id"][0])
    paths = []
    for view in range(target.shape[1]):
        old = None
        if old_root is not None:
            old_path = old_root / f"sample_000_source_05_view_{view}.gif"
            if old_path.is_file():
                old = old_prediction_frames(old_path)
        frames = []
        old_index = 0
        for frame in range(target.shape[0]):
            if not bool(slots[frame, view]):
                continue
            columns = [
                tile(rgb_image(target[frame, view]), "Target", f"future {frame + 1}"),
                tile(
                    rgb_image(prediction[0, frame, view]),
                    "Aligned prediction",
                    "step100 teacher=0",
                ),
            ]
            if old is not None and old_index < len(old):
                columns.append(old[old_index])
            columns.append(tile(rgb_image(context[view]), "Copy last", "static baseline"))
            old_index += 1
            canvas = Image.new(
                "RGB",
                (sum(column.width for column in columns), max(column.height for column in columns)),
                "white",
            )
            left = 0
            for column in columns:
                canvas.paste(column, (left, 0))
                left += column.width
            frames.append(canvas)
        if not frames:
            continue
        path = root / f"sample_000_source_{source_id:02d}_view_{view}.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=320,
            loop=0,
            optimize=False,
            disposal=2,
        )
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise RuntimeError("output root must be new")
    args.output_root.mkdir(parents=True)
    config, _ = load_materialized_runtime(args.runtime)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with torch.device(device):
        model = build_world_model(config["model_profile"])
    if not isinstance(model, NativeWorldModel):
        raise RuntimeError("audit requires NativeWorldModel")
    load_full_model(model, args.checkpoint.resolve(strict=True))
    model.eval()
    flow_outputs: list[torch.Tensor] = []
    flow_modules = [
        module
        for name, module in model.named_modules()
        if name.endswith("flow_head")
    ]
    if len(flow_modules) != 1:
        raise RuntimeError(f"expected one flow head, found {len(flow_modules)}")
    flow_hook = flow_modules[0].register_forward_hook(
        lambda _module, _inputs, output: flow_outputs.append(output.detach())
    )

    raw_batch = torch.load(args.batch, map_location="cpu", weights_only=False)
    fixed_seed = int(raw_batch.pop("_fixed_validation_seed"))
    raw_batch.pop("_schema", None)
    batch = _batch_to_device(raw_batch, device)
    if (
        fixed_seed != 7340
        or int(batch["source_id"][0]) != 5
        or int(batch["embodiment_ids"][0]) != 206
        or int(batch["sample_index"][0]) != 141
    ):
        raise RuntimeError("fixed sample identity mismatch")
    objective = replace(
        objective_config_from_mapping(config["objective_profile"]["objective"]),
        action_counterfactual_token_advantage=0.0,
        action_counterfactual_rgb_advantage=0.0,
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _forward_with_action_counterfactual(
            model,
            batch,
            appearance_teacher_ratio=0.0,
            objective=objective,
        )
    flow_hook.remove()
    if not flow_outputs:
        raise RuntimeError("flow head did not run")
    prediction = output["rgb"].float().clamp(0, 1)
    perceptual = (
        lpips.LPIPS(net="alex", verbose=False).eval().requires_grad_(False).to(device)
    )
    values = audit_metrics(
        prediction,
        batch,
        perceptual,
        float(objective.rgb_motion_threshold),
    )
    raw_flow = torch.cat(flow_outputs, dim=0).float()
    flow_pixels = 0.5 * model.cfg.rgb_size * torch.tanh(
        raw_flow / (0.5 * model.cfg.rgb_size)
    )
    flow_magnitude = flow_pixels.square().sum(dim=1).sqrt()
    blend = output["rgb_blend"].float()
    motion_probability = output["rgb_motion_logit"].float().sigmoid()
    values.update(
        {
            "predicted_flow_mean_pixels": float(flow_magnitude.mean()),
            "predicted_flow_p95_pixels": float(torch.quantile(flow_magnitude, 0.95)),
            "predicted_redraw_blend_mean": float(blend.mean()),
            "predicted_redraw_blend_p95": float(torch.quantile(blend, 0.95)),
            "predicted_motion_probability_mean": float(motion_probability.mean()),
        }
    )
    gif_paths = save_gifs(
        args.output_root / "gifs", prediction, batch, args.old_gif_root
    )
    result = {
        "schema": "wm3d_v7_aligned_rgb_early_audit_v1",
        "checkpoint_step": int(args.checkpoint.name.split("_")[1]),
        "fixed_validation_seed": fixed_seed,
        "sample_index": 141,
        "source_id": 5,
        "embodiment_id": 206,
        "appearance_teacher_ratio": 0.0,
        "gif_paths": gif_paths,
        "metrics": values,
    }
    if args.old_audit is not None:
        result["old_audit_path"] = str(args.old_audit)
        result["old_metrics"] = json.loads(args.old_audit.read_text())["metrics"]
    (args.output_root / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit_complete": str(args.output_root), **values}, sort_keys=True))


if __name__ == "__main__":
    main()
