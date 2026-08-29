"""Real fixed-batch single-GPU proof for the V7-aligned native RGB renderer."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
import yaml

from wm3d.models.model_factory import build_world_model
from wm3d.models.native_world_model import NativeWorldModel
from wm3d.training.native_objective import (
    build_rgb_perceptual_model,
    compute_native_objective,
    objective_config_from_mapping,
)
from wm3d.training.pretrain import (
    _batch_to_device,
    _configure_reproducibility,
    _forward_with_action_counterfactual,
)


NEW_ALIGNMENT_PREFIXES = (
    "rgb_head.image_decoder.motion_token_stem.",
    "rgb_head.image_decoder.motion_view_proj.",
    "rgb_head.image_decoder.motion_geometry_stem.",
    "rgb_head.image_decoder.motion_action_proj.",
    "rgb_head.image_decoder.motion_task_proj.",
    "rgb_head.image_decoder.motion_to_synthesis.",
    "rgb_head.image_decoder.motion_ups.",
    "rgb_head.image_decoder.flow_head.",
    "rgb_head.image_decoder.disocclusion_head.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialized-batch", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--model-profile", type=Path, required=True)
    parser.add_argument("--objective-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_shared_warmstart(
    model: torch.nn.Module, checkpoint: Path
) -> tuple[int, list[str]]:
    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        strict=False,
    )
    state = get_model_state_dict(model, options=options)
    source_metadata = FileSystemReader(
        checkpoint / "distcp"
    ).read_metadata().state_dict_metadata
    source_names = {
        name.removeprefix("model.")
        for name in source_metadata
        if name.startswith("model.")
    }
    new_names = sorted(
        name
        for name in state
        if any(name.startswith(prefix) for prefix in NEW_ALIGNMENT_PREFIXES)
    )
    required_names = set(state) - set(new_names)
    missing = sorted(required_names - source_names)
    if missing:
        raise RuntimeError(f"warmstart misses shared tensors: {missing[:8]}")
    shared_names = sorted(required_names & source_names)
    if not any(name.startswith("dynamics_blocks.") for name in shared_names):
        raise RuntimeError("warmstart did not find factual dynamics")
    shared = {name: state[name] for name in shared_names}
    dcp.load(
        {"model": shared},
        checkpoint_id=checkpoint / "distcp",
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    set_model_state_dict(model, shared, options=options)
    return len(shared_names), new_names


def gradient_norm(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
        raise RuntimeError(f"missing or non-finite gradient: {name}")
    norm = float(parameter.grad.detach().float().norm())
    if norm <= 0.0:
        raise RuntimeError(f"zero gradient: {name}")
    return norm


def first_parameter(module: torch.nn.Module) -> torch.nn.Parameter:
    return next(module.parameters())


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError("output receipt must be new")
    seed = 7340
    _configure_reproducibility(seed)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    cpu_batch = torch.load(
        args.materialized_batch,
        map_location="cpu",
        weights_only=False,
    )
    if (
        cpu_batch.get("_schema") != "wm3d_fixed_materialized_batch_v1"
        or int(cpu_batch.get("_fixed_validation_seed", -1)) != seed
    ):
        raise RuntimeError("fixed validation batch contract is invalid")
    batch = _batch_to_device(
        {
            name: value
            for name, value in cpu_batch.items()
            if not name.startswith("_")
        },
        device,
    )
    del cpu_batch
    if int(batch["source_id"][0]) != 5:
        raise RuntimeError("fixed validation source is not Austin Sailor")

    model_profile = yaml.safe_load(
        args.model_profile.read_text(encoding="utf-8")
    )
    objective_profile = yaml.safe_load(
        args.objective_profile.read_text(encoding="utf-8")
    )
    model = build_world_model(model_profile).to(device)
    if not isinstance(model, NativeWorldModel):
        raise RuntimeError("single-step proof requires NativeWorldModel")
    shared_tensors, new_names = load_shared_warmstart(
        model,
        args.warmstart.resolve(strict=True),
    )
    model.train()
    objective = objective_config_from_mapping(objective_profile["objective"])
    perceptual = build_rgb_perceptual_model(objective, device=device)

    started = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _forward_with_action_counterfactual(
            model,
            batch,
            appearance_teacher_ratio=0.0,
            objective=objective,
        )
        losses = compute_native_objective(
            output=output,
            batch=batch,
            config=objective,
            perceptual_model=perceptual,
            rgb_perceptual_chunk_size=8,
        )
    forward_seconds = time.monotonic() - started
    if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
        raise RuntimeError("full fixed-batch objective is non-finite")

    rgb_objective = replace(
        objective,
        token_mse=0.0,
        token_cosine=0.0,
        appearance_l1=0.0,
        appearance_teacher_l1=0.0,
        appearance_autoregressive_l1=0.0,
        appearance_motion_l1=0.0,
        appearance_mse=0.0,
        appearance_cosine=0.0,
        appearance_motion_mse=0.0,
        appearance_delta_cosine=0.0,
        depth_log=0.0,
        point=0.0,
        camera_pose=0.0,
        action_fine=0.0,
        action_coarse=0.0,
        action_counterfactual_token_advantage=0.0,
        action_counterfactual_rgb_advantage=0.0,
        action_velocity=0.0,
    )
    rgb_losses = compute_native_objective(
        output=output,
        batch=batch,
        config=rgb_objective,
        perceptual_model=perceptual,
        rgb_perceptual_chunk_size=8,
    )
    started = time.monotonic()
    rgb_losses["total"].backward()
    backward_seconds = time.monotonic() - started

    decoder = model.rgb_head.image_decoder
    probes = {
        "factual_dynamics": first_parameter(model.dynamics_blocks[-1]),
        "p256_synthesis": first_parameter(decoder.token_stem),
        "p64_motion": first_parameter(decoder.motion_token_stem),
        "flow": decoder.flow_head.weight,
        "disocclusion": decoder.disocclusion_head.weight,
        "motion_mask": decoder.motion_head.weight,
    }
    probe_norms = {
        name: gradient_norm(parameter, name)
        for name, parameter in probes.items()
    }
    nonfinite = []
    nonzero = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif bool(parameter.grad.count_nonzero()):
            nonzero += 1
    if nonfinite:
        raise RuntimeError(f"non-finite model gradients: {nonfinite[:8]}")

    receipt = {
        "schema": "wm3d_v7_aligned_rgb_single_step_v1",
        "fixed_validation_seed": seed,
        "source_id": int(batch["source_id"][0]),
        "world_tokens_shape": list(batch["world_tokens"].shape),
        "target_rgb_shape": list(batch["target_rgb"].shape),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "warmstart_shared_tensors": shared_tensors,
        "new_alignment_tensors": len(new_names),
        "total_loss": float(losses["total"].detach()),
        "rgb_only_loss": float(rgb_losses["total"].detach()),
        "probe_gradient_norms": probe_norms,
        "nonzero_gradient_tensors": nonzero,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
