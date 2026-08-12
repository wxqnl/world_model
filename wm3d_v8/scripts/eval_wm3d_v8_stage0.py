#!/usr/bin/env python3
"""在固定真实验证样本上比较 WM3D V8 Stage0 checkpoint。

本脚本只复用正式训练的 config、dataset、model、action contract 和 forward
路径；不构造伪样本，不以训练日志中的随机 batch 代替验证集。输出同时覆盖
native 3D world prediction、统一 20 Hz action owner 和 teacher-action 因果隔离。
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from wm3d_v3.data.v8_action_contract import (
    SUBSTEPS_PER_WORLD,
    compose_base_delta_actions_torch,
)
from wm3d_v3.losses import (
    _normalize_depth,
    _normalise_points_relative,
    _point_mask_hw,
    _resize_point_pred,
)
from wm3d_v3.training.train import (
    _forward_joint_model,
    action_policy_kwargs_from_targets,
    batch_to_device,
    build_datasets,
    build_model,
    compute_configured_action_policy_loss,
    decode_codec_targets,
    load_action_stats_if_available,
    load_train_config,
    multiview_kwargs_from_targets,
    normalize_action_grip_contract,
    targets_with_close01_grip,
)


class Moments:
    def __init__(self) -> None:
        self.sum: dict[str, float] = defaultdict(float)
        self.sumsq: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)

    def update(self, values: Mapping[str, torch.Tensor | float], repeat: int = 1) -> None:
        for key, raw in values.items():
            if isinstance(raw, torch.Tensor):
                array = raw.detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            else:
                array = np.full(max(1, int(repeat)), float(raw), dtype=np.float64)
            array = array[np.isfinite(array)]
            if array.size == 0:
                continue
            self.sum[key] += float(array.sum())
            self.sumsq[key] += float(np.square(array).sum())
            self.count[key] += int(array.size)

    def report(self) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for key in sorted(self.sum):
            count = self.count[key]
            mean = self.sum[key] / count
            variance = max(0.0, self.sumsq[key] / count - mean * mean)
            result[key] = {
                "mean": mean,
                "sem": math.sqrt(variance / count),
                "count": count,
            }
        return result


def _sample_indices(length: int, count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(length, generator=generator)[: min(length, count)].tolist()


def _source_datasets(cfg: dict) -> tuple[dict[str, Any], dict[str, int]]:
    _train, val = build_datasets(cfg)
    if not hasattr(val, "source_names") or not hasattr(val, "datasets"):
        raise RuntimeError("V8 formal validation dataset must expose source_names/datasets")
    sources = {
        str(name): dataset for name, dataset in zip(val.source_names, val.datasets)
    }
    return sources, {name: len(dataset) for name, dataset in sources.items()}


def _load_checkpoint_model(
    cfg: dict, checkpoint: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(int(cfg["train"].get("seed", 0)))
    model = build_model(cfg)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    contract = payload.get("action_policy_contract")
    if not isinstance(contract, Mapping) or contract.get("schema") != "wm3d_v8_stage0_action_policy_contract_v3":
        raise RuntimeError(f"{checkpoint} is not a sealed V8 action-policy-v3 checkpoint")
    result = model.load_state_dict(payload["model"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {result}")
    metadata = {
        "path": str(checkpoint),
        "step": int(payload["step"]),
        "epoch": int(payload.get("epoch", 0)),
        "stored_val_total": payload.get("val_total"),
        "action_policy_contract_schema": contract["schema"],
    }
    del payload
    gc.collect()
    model = model.to(device).eval()
    load_action_stats_if_available(model, cfg, 1, device)
    return model, metadata


def _resize_depth(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == target.shape[-2:]:
        return pred
    b, t = pred.shape[:2]
    return F.interpolate(
        pred.float().reshape(b * t, 1, *pred.shape[-2:]),
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).reshape(b, t, *target.shape[-2:])


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float() - target.float()).pow(2).mean(dim=(1, 2, 3, 4))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1.0e-8))


def _weighted_sample_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(device=values.device, dtype=values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(values)
    numerator = (values * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1)
    result = numerator / denominator.clamp_min(1.0)
    return torch.where(denominator > 0.0, result, torch.nan)


def _world_metrics(
    out: Mapping[str, torch.Tensor],
    tgt: Mapping[str, torch.Tensor],
    context_rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred_tokens = out["pred_tokens"].float()
    target_tokens = tgt["s_tgt"].float()
    rgb_pred = out["rgb"].float().clamp(0.0, 1.0)
    rgb_target = tgt["rgb_tgt_p"].float()
    last = context_rgb.float().unsqueeze(1).expand_as(rgb_target)
    depth_pred = _normalize_depth(_resize_depth(out["depth"].float(), tgt["depth_tgt"]))
    depth_target = _normalize_depth(tgt["depth_tgt"].float())
    metrics = {
        "world/token_mse": (pred_tokens - target_tokens).pow(2).mean(dim=(1, 2, 3)),
        "world/token_cosine": F.cosine_similarity(
            pred_tokens.flatten(2), target_tokens.flatten(2), dim=-1
        ).mean(dim=1),
        "world/rgb_l1": (rgb_pred - rgb_target).abs().mean(dim=(1, 2, 3, 4)),
        "world/rgb_psnr": _psnr(rgb_pred, rgb_target),
        "world/last_frame_rgb_l1": (last - rgb_target).abs().mean(dim=(1, 2, 3, 4)),
        "world/rgb_l1_gain_vs_last": (
            (last - rgb_target).abs().mean(dim=(1, 2, 3, 4))
            - (rgb_pred - rgb_target).abs().mean(dim=(1, 2, 3, 4))
        ),
        "world/depth_relative_l1": (depth_pred - depth_target).abs().mean(dim=(1, 2, 3)),
    }
    if "point" in out and "point_tgt" in tgt:
        point_tgt = tgt["point_tgt"].float()
        horizon = min(out["point"].shape[1], point_tgt.shape[1])
        point_pred = _resize_point_pred(
            out["point"][:, :horizon].float(), point_tgt.shape[2:4]
        )
        point_tgt = point_tgt[:, :horizon]
        point_mask = _point_mask_hw(tgt.get("point_conf_tgt"), point_pred)
        if point_mask is None:
            point_mask = torch.ones(point_pred.shape[:-1], device=point_pred.device)
        else:
            point_mask = point_mask[:, :horizon]
        pred_norm = _normalise_points_relative(point_pred, point_mask)
        tgt_norm = _normalise_points_relative(point_tgt, point_mask)
        metrics["world/point_normalized_l1"] = _weighted_sample_mean(
            (pred_norm - tgt_norm).abs(), point_mask
        )
    if "pose_geom" in out and "pose_geom_tgt" in tgt:
        pred = out["pose_geom"].float()
        target = tgt["pose_geom_tgt"].float()
        horizon = min(pred.shape[1], target.shape[1])
        pred, target = pred[:, :horizon], target[:, :horizon]
        mask = tgt.get("pose_geom_conf_tgt")
        if mask is None:
            mask = torch.ones(pred.shape[:2], device=pred.device)
        else:
            mask = mask[:, :horizon].reshape(pred.shape[0], horizon)
        pred_translation = pred[..., :3] - pred[..., :3][:, :1]
        target_translation = target[..., :3] - target[..., :3][:, :1]
        scale = target_translation.pow(2).sum(dim=-1, keepdim=True).mean(
            dim=1, keepdim=True
        ).sqrt().clamp_min(1.0e-3)
        metrics["world/pose_translation_normalized_l1"] = _weighted_sample_mean(
            ((pred_translation - target_translation) / scale).abs(), mask
        )
        pred_q = F.normalize(pred[..., 3:7], dim=-1)
        target_q = F.normalize(target[..., 3:7], dim=-1)
        cosine = (pred_q * target_q).sum(dim=-1).abs().clamp(0.0, 1.0)
        angle = 2.0 * torch.acos(cosine) * (180.0 / math.pi)
        metrics["world/pose_rotation_deg"] = _weighted_sample_mean(angle, mask)
    return metrics


def _action_metrics(
    out: Mapping[str, torch.Tensor], tgt: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    pose_norm = out["base_policy_pose_norm"].float()
    grip_prob = torch.sigmoid(out["base_policy_gripper_logit"].float())
    fine = tgt["policy_action_tgt"].float()
    fine_norm = tgt["policy_action_tgt_norm"].float()
    fine_valid = tgt["policy_action_valid_mask"].bool()
    coarse = tgt["policy_action_coarse_tgt"].float()
    coarse_norm = tgt["policy_action_coarse_tgt_norm"].float()
    coarse_valid = tgt["policy_action_coarse_valid_mask"].bool()
    fine_mean = tgt["policy_action_pose_mean"].float()[:, None]
    fine_std = tgt["policy_action_pose_std"].float()[:, None]
    fine_physical = pose_norm * fine_std + fine_mean
    predicted_actions = torch.cat((fine_physical, grip_prob.unsqueeze(-1)), dim=-1)
    coarse_horizon = pose_norm.shape[1] // SUBSTEPS_PER_WORLD
    composed = compose_base_delta_actions_torch(
        predicted_actions.reshape(
            pose_norm.shape[0], coarse_horizon, SUBSTEPS_PER_WORLD, 7
        )
    )
    coarse_mean = tgt["policy_action_coarse_pose_mean"].float()[:, None]
    coarse_std = tgt["policy_action_coarse_pose_std"].float()[:, None]
    composed_norm = (composed[..., :6] - coarse_mean) / coarse_std
    coarse_grip_prob = grip_prob.reshape(
        pose_norm.shape[0], coarse_horizon, SUBSTEPS_PER_WORLD
    )[..., -1]
    fine_abs = (fine_physical - fine[..., :6]).abs()
    coarse_abs = (composed[..., :6] - coarse[..., :6]).abs()
    return {
        "action/fine_label_fraction": fine_valid.float().mean(dim=1),
        "action/coarse_label_fraction": coarse_valid.float().mean(dim=1),
        "action/fine_pose_norm_l1": _weighted_sample_mean(
            (pose_norm - fine_norm).abs(), fine_valid
        ),
        "action/fine_translation_mae_mm": 1000.0
        * _weighted_sample_mean(fine_abs[..., :3], fine_valid),
        "action/fine_rotation_mae_deg": (180.0 / math.pi)
        * _weighted_sample_mean(fine_abs[..., 3:6], fine_valid),
        "action/fine_grip_accuracy": _weighted_sample_mean(
            ((grip_prob >= 0.5) == (fine[..., 6] >= 0.5)).float(), fine_valid
        ),
        "action/coarse_pose_norm_l1": _weighted_sample_mean(
            (composed_norm - coarse_norm).abs(), coarse_valid
        ),
        "action/coarse_translation_mae_mm": 1000.0
        * _weighted_sample_mean(coarse_abs[..., :3], coarse_valid),
        "action/coarse_rotation_mae_deg": (180.0 / math.pi)
        * _weighted_sample_mean(coarse_abs[..., 3:6], coarse_valid),
        "action/coarse_grip_accuracy": _weighted_sample_mean(
            ((coarse_grip_prob >= 0.5) == (coarse[..., 6] >= 0.5)).float(),
            coarse_valid,
        ),
    }


def _counterfactual_action(action_cond: torch.Tensor) -> torch.Tensor:
    changed = action_cond.clone()
    if changed.shape[-1] == 36:
        actions = changed[..., :28].reshape(*changed.shape[:-1], 4, 7)
        actions[..., :6].neg_()
        actions[..., 6].neg_()
    else:
        changed.neg_()
    return changed


@torch.no_grad()
def _evaluate_one(
    cfg: dict,
    checkpoint: Path,
    device: torch.device,
    sources: Mapping[str, Any],
    selected: Mapping[str, list[int]],
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    model, checkpoint_metadata = _load_checkpoint_model(cfg, checkpoint, device)
    train_cfg = cfg["train"]
    grip_contract = normalize_action_grip_contract(
        train_cfg.get("action_grip_contract", "close01")
    )
    per_source = {name: Moments() for name in [*sources, "ALL"]}
    causal: dict[str, dict[str, float]] = {}
    for source_name, dataset in sources.items():
        loader = DataLoader(
            Subset(dataset, selected[source_name]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for batch in loader:
            s, c, action_cond, context_rgb, tgt = batch_to_device(
                batch,
                device,
                cfg["model"]["state"]["k"],
                action_grip_contract=grip_contract,
                source_name=source_name,
                require_factual_action_contract=True,
            )
            decode_codec_targets(model, tgt)
            loss_tgt = targets_with_close01_grip(tgt, grip_contract)
            policy_kwargs = action_policy_kwargs_from_targets(loss_tgt)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _forward_joint_model(
                    model,
                    s,
                    c,
                    action_cond=action_cond,
                    context_rgb=context_rgb,
                    pixel=True,
                    bridging=False,
                    policy_kwargs=policy_kwargs,
                    multiview_kwargs=multiview_kwargs_from_targets(tgt),
                    native_action_no_teacher=True,
                )
                direct = compute_configured_action_policy_loss(
                    out, loss_tgt, train_cfg, step=checkpoint_metadata["step"]
                )
            metrics = _world_metrics(out, tgt, context_rgb)
            metrics.update(_action_metrics(out, loss_tgt))
            scalar = {f"objective/{key}": float(value.detach()) for key, value in direct.items()}
            for key in (source_name, "ALL"):
                per_source[key].update(metrics)
                per_source[key].update(scalar, repeat=int(s.shape[0]))
            if source_name not in causal:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    changed = _forward_joint_model(
                        model,
                        s,
                        c,
                        action_cond=_counterfactual_action(action_cond),
                        context_rgb=context_rgb,
                        pixel=False,
                        bridging=False,
                        policy_kwargs=policy_kwargs,
                        multiview_kwargs=multiview_kwargs_from_targets(tgt),
                        native_action_no_teacher=True,
                    )
                causal[source_name] = {
                    "policy_pose_teacher_action_max_abs": float(
                        (out["base_policy_pose_norm"] - changed["base_policy_pose_norm"])
                        .abs().max().float().cpu()
                    ),
                    "policy_grip_teacher_action_max_abs": float(
                        (out["base_policy_gripper_logit"] - changed["base_policy_gripper_logit"])
                        .abs().max().float().cpu()
                    ),
                    "policy_context_teacher_action_max_abs": float(
                        (out["policy_context_tokens"] - changed["policy_context_tokens"])
                        .abs().max().float().cpu()
                    ),
                    "factual_world_teacher_action_mean_abs": float(
                        (out["pred_tokens"] - changed["pred_tokens"])
                        .abs().mean().float().cpu()
                    ),
                }
                del changed
            del s, c, action_cond, context_rgb, tgt, loss_tgt, out
    report = {
        "checkpoint": checkpoint_metadata,
        "causal_runtime_audit": causal,
        "metrics": {name: moments.report() for name, moments in per_source.items()},
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=/absolute/path.pt")
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-source", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    cfg = load_train_config(args.cfg)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    sources, lengths = _source_datasets(cfg)
    selected = {
        name: _sample_indices(len(dataset), args.samples_per_source, args.seed + i * 1009)
        for i, (name, dataset) in enumerate(sources.items())
    }
    checkpoints: dict[str, Any] = {}
    for label, checkpoint in args.checkpoint:
        checkpoints[label] = _evaluate_one(
            cfg,
            checkpoint,
            device,
            sources,
            selected,
            args.batch_size,
            args.num_workers,
        )
    report = {
        "schema": "wm3d_v8_stage0_fixed_validation_comparison_v1",
        "config": str(args.cfg),
        "sampling": {
            "seed": args.seed,
            "samples_per_source": args.samples_per_source,
            "source_lengths": lengths,
            "selected_local_indices": selected,
        },
        "checkpoints": checkpoints,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        label: {
            key: value["mean"]
            for key, value in payload["metrics"]["ALL"].items()
        }
        for label, payload in checkpoints.items()
    }
    print(json.dumps({"out": str(args.out), "ALL": summary}, indent=2))


if __name__ == "__main__":
    main()
