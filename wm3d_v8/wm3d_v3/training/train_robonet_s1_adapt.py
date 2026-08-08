"""Protocol-correct RoboNet K8+2 adaptation from the WM3D-v7 S1-30K model."""
from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from wm3d_v3.benchmarks.robonet_protocol import pad_two_frame_history
from wm3d_v3.models.model_factory import build_joint_world_model


def validate_window_policy_rows(
    rows: Sequence[Mapping[str, Any]], expected_window_policy: str | None
) -> None:
    """Fail closed when a split is not the precommitted causal window policy."""
    if expected_window_policy is None:
        return
    if expected_window_policy != "fixed_start0":
        raise RuntimeError(f"unsupported RoboNet window policy: {expected_window_policy}")
    for row in rows:
        valid = (
            int(row.get("window_start", -1)) == 0
            and row.get("window_policy") == "fixed_start0"
            and row.get("context_future_leakage") is False
            and row.get("target_joint_observed_anchor") is True
            and row.get("official_test_excluded") is True
        )
        if not valid:
            raise RuntimeError(
                f"fixed_start0 causal/test-exclusion contract failed for {row.get('id')}"
            )


class RoboNetAdaptDataset(Dataset):
    def __init__(
        self,
        index: Path,
        action_stats: Path,
        *,
        split: str,
        expected_window_policy: str | None = None,
        remote_cache_base_url: str | None = None,
        remote_cache_path_prefix: str | None = None,
        remote_cache_timeout_seconds: float = 30.0,
    ) -> None:
        self.rows = [
            json.loads(line) for line in Path(index).read_text().splitlines()
            if line.strip() and json.loads(line).get("split") == split
        ]
        if not self.rows:
            raise RuntimeError(f"RoboNet adaptation split is empty: {split}")
        ids = [str(row["id"]) for row in self.rows]
        if len(ids) != len(set(ids)) or any(not row.get("official_test_excluded", True) for row in self.rows):
            raise RuntimeError("adaptation index is duplicate or not test-exclusion locked")
        validate_window_policy_rows(self.rows, expected_window_policy)
        with np.load(action_stats, allow_pickle=False) as stats:
            if str(stats["split"].item()) != "train":
                raise RuntimeError("RoboNet action statistics must be train-only")
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
        if self.mean.shape != (6,) or self.std.shape != (6,) or np.any(self.std <= 0):
            raise RuntimeError("invalid RoboNet pose statistics")
        self.remote_cache_base_url = (
            str(remote_cache_base_url).rstrip("/") if remote_cache_base_url else None
        )
        self.remote_cache_path_prefix = (
            Path(remote_cache_path_prefix) if remote_cache_path_prefix else None
        )
        self.remote_cache_timeout_seconds = float(remote_cache_timeout_seconds)
        if bool(self.remote_cache_base_url) != bool(self.remote_cache_path_prefix):
            raise RuntimeError(
                "remote RoboNet cache requires both base URL and path prefix"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _cache_source(self, cache_path: str) -> Path | io.BytesIO:
        path = Path(cache_path)
        if self.remote_cache_base_url is None:
            return path
        assert self.remote_cache_path_prefix is not None
        try:
            relative = path.relative_to(self.remote_cache_path_prefix)
        except ValueError as exc:
            raise RuntimeError(
                f"remote cache path escapes configured prefix: {path}"
            ) from exc
        url = f"{self.remote_cache_base_url}/{quote(relative.as_posix(), safe='/')}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(url, timeout=self.remote_cache_timeout_seconds) as response:
                    payload = response.read()
                if not payload:
                    raise RuntimeError(f"empty remote RoboNet cache response: {url}")
                return io.BytesIO(payload)
            except Exception as exc:  # DataLoader worker must retry transient HTTP faults.
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(f"failed to fetch remote RoboNet cache: {url}") from last_error

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(self._cache_source(str(row["cache_path"])), allow_pickle=False) as archive:
            if str(archive["schema"].item()) != "wm3d_v7_robonet_adapt_v2_causal":
                raise RuntimeError(f"unexpected cache schema: {row['id']}")
            context = np.asarray(archive["context_codes"], np.float32) * np.asarray(archive["context_scale"], np.float32)
            target = np.asarray(archive["target_codes"], np.float32) * np.asarray(archive["target_scale"], np.float32)
            context_rgb = np.asarray(archive["context_rgb"], np.float32) / 255.0
            target_rgb = np.asarray(archive["target_rgb"], np.float32) / 255.0
            actions = np.asarray(archive["actions"], np.float32)
            task = np.asarray(archive["task_emb"], np.float32)
        if context.shape[0:2] != (2, 64) or target.shape[0:2] != (10, 64) or context.shape[-1] != target.shape[-1]:
            raise RuntimeError(f"cache must preserve 2 observed and 10 future tokens: {row['id']}")
        if context_rgb.shape != (64, 64, 3) or target_rgb.shape != (10, 64, 64, 3):
            raise RuntimeError(f"cache must preserve 64x64 RGB 2+10 protocol: {row['id']}")
        if actions.shape != (10, 7) or task.shape != (2048,):
            raise RuntimeError(f"cache action/task contract mismatch: {row['id']}")
        pose = (actions[:, :6] - self.mean) / self.std
        grip = (actions[:, 6:7] > 0.5).astype(np.float32)
        action_cond = np.concatenate((pose, grip), axis=-1)
        zero_pose = np.broadcast_to((-self.mean / self.std)[None], (10, 6)).astype(np.float32)
        zero_action_cond = np.concatenate((zero_pose, np.zeros((10, 1), np.float32)), axis=-1)
        return {
            "id": str(row["id"]),
            "context_latent": torch.from_numpy(context.copy()),
            "target_latent": torch.from_numpy(target.copy()),
            "context_rgb": torch.from_numpy(context_rgb).permute(2, 0, 1).contiguous(),
            "target_rgb": torch.from_numpy(target_rgb).permute(0, 3, 1, 2).contiguous(),
            "action_tgt": torch.from_numpy(actions.copy()),
            "action_tgt_norm": torch.from_numpy(pose.copy()),
            "action_pose_mean": torch.from_numpy(self.mean.copy()),
            "action_pose_std": torch.from_numpy(self.std.copy()),
            "action_cond": torch.from_numpy(action_cond.copy()),
            "zero_action_cond": torch.from_numpy(zero_action_cond.copy()),
            "task_emb": torch.from_numpy(task.copy()),
        }


class RoboNetWanWindowDataset(Dataset):
    """Expose causal RoboNet caches through the native S2+Wan batch contract.

    Wan TI2V owns an anchor plus eight predicted frames, while the official
    RoboNet cache preserves two observed plus ten future frames.  Training
    therefore samples one of the three legal K8 windows.  Offset zero is the
    exact first official chunk; offsets one and two teacher-force only a past
    frame and expand coverage without exposing a future frame to its own
    prediction.
    """

    def __init__(
        self,
        index: Path,
        action_stats: Path,
        *,
        split: str,
        seed: int = 0,
        window_offset_policy: str = "fixed_zero",
        expected_window_policy: str | None = None,
        remote_cache_base_url: str | None = None,
        remote_cache_path_prefix: str | None = None,
        remote_cache_timeout_seconds: float = 30.0,
    ) -> None:
        self.base = RoboNetAdaptDataset(
            index,
            action_stats,
            split=split,
            expected_window_policy=expected_window_policy,
            remote_cache_base_url=remote_cache_base_url,
            remote_cache_path_prefix=remote_cache_path_prefix,
            remote_cache_timeout_seconds=remote_cache_timeout_seconds,
        )
        self.split = str(split)
        self.seed = int(seed)
        self.epoch = 0
        self.window_offset_policy = str(window_offset_policy)
        if self.window_offset_policy not in {"fixed_zero", "cycle_0_2"}:
            raise ValueError(
                "RoboNet Wan window_offset_policy must be fixed_zero or cycle_0_2"
            )

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _offset(self, index: int) -> int:
        if self.window_offset_policy == "fixed_zero":
            return 0
        return int((self.seed + self.epoch + int(index)) % 3)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        offset = self._offset(index)
        context_all = torch.cat(
            (item["context_latent"], item["target_latent"]), dim=0
        )
        context = context_all[offset : offset + 2]
        target = context_all[offset + 2 : offset + 10]
        if context.shape[0] != 2 or target.shape[0] != 8:
            raise RuntimeError("RoboNet Wan K8 window construction failed")

        target_rgb = item["target_rgb"]
        anchor_rgb = (
            item["context_rgb"] if offset == 0 else target_rgb[offset - 1]
        )
        future_rgb = target_rgb[offset : offset + 8]
        action_tgt = item["action_tgt"][offset : offset + 8]
        action_tgt_norm = item["action_tgt_norm"][offset : offset + 8]
        if future_rgb.shape[0] != 8 or action_tgt.shape[0] != 8:
            raise RuntimeError("RoboNet Wan RGB/action horizon mismatch")

        # The selected S2 core was trained with T=16.  Preserve the official
        # two-real-frame protocol while left-padding exactly as the RoboNet
        # evaluator does; feeding a raw T=2 tensor would change the native WM
        # contract rather than merely adapting its renderer.
        state = pad_two_frame_history(context.unsqueeze(0), native_t=16)[0]
        view_mask = torch.zeros((16, 2), dtype=torch.bool)
        view_mask[:, 0] = True
        return {
            "s_in": state,
            "s_wrist": torch.zeros_like(state),
            "view_mask": view_mask,
            "s_tgt_codec": target,
            "rgb_in": anchor_rgb.permute(1, 2, 0).unsqueeze(0).contiguous(),
            "rgb_tgt": future_rgb.permute(0, 2, 3, 1).contiguous(),
            "action_tgt": action_tgt,
            "action_tgt_norm": action_tgt_norm,
            "action_pose_mean": item["action_pose_mean"],
            "action_pose_std": item["action_pose_std"],
            "action_grip_close01": (action_tgt[:, 6] > 0.5).float(),
            "c": item["task_emb"],
            "clip_id": item["id"],
            "start": offset,
            "dataset": "robonet_train_only",
            "task_text": "robot manipulation scene, close-up tabletop robot arm",
        }


def combine_native_chunks(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key in ("pred_tokens", "rgb"):
        if key not in first or key not in second or first[key].shape[1] != 8 or second[key].shape[1] != 8:
            raise RuntimeError(f"native K8 output missing for {key}")
        result[key] = torch.cat((first[key], second[key][:, :2]), dim=1)
    for key in ("motion_logit", "motion_hint", "rgb_blend"):
        if (key in first) != (key in second):
            raise RuntimeError(f"native K8 auxiliary output is inconsistent for {key}")
        if key in first:
            if first[key].shape[1] != 8 or second[key].shape[1] != 8:
                raise RuntimeError(f"native K8 auxiliary output has invalid horizon for {key}")
            result[key] = torch.cat((first[key], second[key][:, :2]), dim=1)
    return result


def select_trainable_names(names: Iterable[str], prefixes: Sequence[str]) -> list[str]:
    prefixes = tuple(str(prefix) for prefix in prefixes)
    return [str(name) for name in names if str(name).startswith(prefixes)]


def adapted_checkpoint_cfg(parent_cfg: Mapping[str, Any], action_stats: str) -> dict[str, Any]:
    """Record the normalization used for adaptation without mutating lineage config."""
    resolved = copy.deepcopy(dict(parent_cfg))
    data = dict(resolved.get("data", {}))
    data["action_stats"] = str(action_stats)
    resolved["data"] = data
    return resolved


def validate_parent_checkpoint(
    checkpoint: Mapping[str, Any], *, expected_kind: str, expected_step: int
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Validate exact parent lineage before building or loading the model."""
    step = int(checkpoint.get("step", -1))
    if step != int(expected_step):
        raise RuntimeError(f"parent checkpoint step mismatch: {step} != {expected_step}")
    if "cfg" not in checkpoint:
        raise RuntimeError("parent checkpoint is missing cfg")
    if expected_kind == "robonet_adapt":
        if checkpoint.get("schema") != "wm3d_v7_robonet_s1_adapt_v1":
            raise RuntimeError("RoboNet parent checkpoint schema mismatch")
        adapt_data = checkpoint.get("adapt_config", {}).get("data", {})
        if adapt_data.get("official_test_excluded") is not True:
            raise RuntimeError("RoboNet parent checkpoint lacks test-exclusion lineage")
        audit = {
            "kind": expected_kind,
            "step": step,
            "official_test_excluded": True,
            "parent_checkpoint": checkpoint.get("parent_checkpoint"),
        }
    elif expected_kind in {"formal_s1", "s1_formal"}:
        audit = {
            "kind": expected_kind,
            "step": step,
            "official_test_excluded": None,
            "parent_checkpoint": None,
        }
    else:
        raise RuntimeError(f"unsupported parent checkpoint kind: {expected_kind}")
    return checkpoint["cfg"], audit


def load_renderer_initialization(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    prefix: str = "context_pixel.",
) -> dict[str, Any]:
    """Transplant only a leakage-safe adapted renderer onto the formal S1 core."""
    path = Path(checkpoint_path).resolve(strict=True)
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "wm3d_v7_robonet_s1_adapt_v1"
        or int(payload.get("step", -1)) <= 0
    ):
        raise RuntimeError("renderer initialization is not a RoboNet adaptation checkpoint")
    adapt = payload.get("adapt_config")
    data = adapt.get("data", {}) if isinstance(adapt, Mapping) else {}
    if (
        data.get("official_test_excluded") is not True
        or data.get("context_future_leakage") is not False
        or data.get("target_joint_observed_anchor") is not True
    ):
        raise RuntimeError("renderer initialization has unsafe RoboNet lineage")
    source = payload.get("model")
    if not isinstance(source, Mapping):
        raise RuntimeError("renderer initialization has no model state")
    target = model.state_dict()
    names = sorted(name for name in target if name.startswith(prefix))
    if not names:
        raise RuntimeError(f"model has no renderer parameters under {prefix}")
    missing = [name for name in names if name not in source]
    mismatched = [
        name for name in names
        if name in source and tuple(source[name].shape) != tuple(target[name].shape)
    ]
    if missing or mismatched:
        raise RuntimeError(
            f"renderer initialization mismatch: missing={missing[:8]} shape={mismatched[:8]}"
        )
    transplanted = dict(target)
    for name in names:
        transplanted[name] = source[name]
    model.load_state_dict(transplanted, strict=True)
    return {
        "checkpoint": str(path),
        "step": int(payload["step"]),
        "prefix": prefix,
        "tensors": len(names),
        "official_test_excluded": True,
    }


def select_dual_validation_checkpoint(
    records: Sequence[Mapping[str, Any]],
    *,
    checkpoint_root: Path,
    parent_checkpoint: Path,
    random_regression_limit: float,
    allow_step0_parent: bool = True,
) -> dict[str, Any]:
    """Pick by early FVD while protecting general random-window validation."""
    by_step: dict[int, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        split = str(record.get("validation_split", ""))
        if split not in {"early_start0", "random_window"}:
            continue
        by_step.setdefault(int(record["step"]), {})[split] = record
    if 0 not in by_step or "random_window" not in by_step[0]:
        raise RuntimeError("dual validation selection requires step-0 random-window FVD")
    baseline_random = float(by_step[0]["random_window"]["fvd"])
    maximum_random = baseline_random * (1.0 + float(random_regression_limit))
    candidates: list[tuple[float, float, int, Path]] = []
    for step, split_records in by_step.items():
        if not {"early_start0", "random_window"}.issubset(split_records):
            continue
        if step == 0 and not allow_step0_parent:
            # A transplanted renderer makes the in-memory step-0 model differ
            # from the formal parent file, so pointing selection at that parent
            # would silently select the wrong model.
            continue
        early_fvd = float(split_records["early_start0"]["fvd"])
        random_fvd = float(split_records["random_window"]["fvd"])
        if not np.isfinite(early_fvd) or not np.isfinite(random_fvd) or random_fvd > maximum_random:
            continue
        path = Path(parent_checkpoint) if step == 0 else Path(checkpoint_root) / f"step_{step:08d}.pt"
        if step != 0 and not path.is_file():
            continue
        candidates.append((early_fvd, random_fvd, step, path))
    if not candidates:
        raise RuntimeError("no checkpoint passes the dual-validation random-regression gate")
    early_fvd, random_fvd, step, path = min(candidates)
    return {
        "selected_step": step,
        "selected_checkpoint": str(path),
        "selected_early_fvd": early_fvd,
        "selected_random_fvd": random_fvd,
        "random_step0_fvd": baseline_random,
        "random_fvd_gate": maximum_random,
        "random_regression_limit": float(random_regression_limit),
        "selection_used_official_test": False,
    }


def _setup() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
    else:
        rank, world, local = 0, 1, 0
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def _target(model):
    return model.module if isinstance(model, DDP) else model


def _decode(model, latent: torch.Tensor) -> torch.Tensor:
    return _target(model).decode_input_tokens(latent).detach()


def _pad_actions(last_two: torch.Tensor) -> torch.Tensor:
    if tuple(last_two.shape[1:]) != (2, 7):
        raise RuntimeError("second RoboNet action chunk must be [B,2,7]")
    return torch.cat((last_two, last_two[:, -1:].expand(-1, 6, -1)), dim=1)


def rollout_k8_plus_2(
    model,
    state: torch.Tensor,
    task: torch.Tensor,
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
) -> dict[str, torch.Tensor]:
    kwargs = dict(
        pixel=True,
        skip_action_proposer=True,
        skip_action_policy=True,
        skip_native_prediction_heads=True,
    )
    first = model(state, task, action_cond=action_cond[:, :8], context_rgb=context_rgb, **kwargs)
    second_state = torch.cat((state[:, 8:], first["pred_tokens"]), dim=1)
    second = model(
        second_state, task, action_cond=_pad_actions(action_cond[:, 8:10]),
        context_rgb=first["rgb"][:, 7], **kwargs,
    )
    return combine_native_chunks(first, second)


def _gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        F.l1_loss(pred[..., 1:], target[..., 1:])
        + F.l1_loss(pred[..., 1:, :], target[..., 1:, :])
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def _i3d_alignment_loss(
    rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    i3d: torch.nn.Module,
    *,
    smooth_l1_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable video-feature alignment using the exact evaluation I3D."""
    guard = (
        torch.autocast(device_type="cuda", enabled=False)
        if rgb.device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with guard:
        pred_input = rgb.float().clamp(0, 1).permute(0, 2, 1, 3, 4).contiguous() * 255.0
        target_input = target_rgb.float().permute(0, 2, 1, 3, 4).contiguous() * 255.0
        pred_features = i3d(
            pred_input, rescale=True, resize=True, return_features=True
        ).float()
        with torch.no_grad():
            target_features = i3d(
                target_input, rescale=True, resize=True, return_features=True
            ).float()
        cosine = 1.0 - F.cosine_similarity(
            pred_features, target_features, dim=-1
        ).mean()
        feature_l1 = F.smooth_l1_loss(pred_features, target_features)
        total = cosine + float(smooth_l1_weight) * feature_l1
    return total, cosine, feature_l1


def _losses(
    pred: Mapping[str, torch.Tensor],
    target_tokens: torch.Tensor,
    target_rgb: torch.Tensor,
    context_rgb: torch.Tensor,
    lpips_model,
    cfg: Mapping[str, float],
    zero_pred: Mapping[str, torch.Tensor] | None = None,
    *,
    i3d: torch.nn.Module | None = None,
    compute_i3d: bool = False,
) -> dict[str, torch.Tensor]:
    tokens = pred["pred_tokens"].float()
    target_tokens = target_tokens.float()
    token_l1 = F.smooth_l1_loss(tokens, target_tokens)
    token_cos = 1.0 - F.cosine_similarity(tokens, target_tokens, dim=-1).mean()
    rgb = pred["rgb"].float()
    target_rgb = target_rgb.float()
    horizon = int(rgb.shape[1])
    last_horizon_weight = float(cfg.get("last_horizon_weight", 1.0))
    horizon_weights = torch.linspace(
        1.0, last_horizon_weight, horizon, device=rgb.device, dtype=rgb.dtype
    )
    horizon_weights = horizon_weights / horizon_weights.mean().clamp_min(1.0e-6)
    frame_l1 = (rgb - target_rgb).abs().mean(dim=(2, 3, 4))
    rgb_l1 = frame_l1.mean()
    horizon_rgb_l1 = (frame_l1 * horizon_weights[None]).mean()

    target_previous = torch.cat(
        (context_rgb[:, None].float(), target_rgb[:, :-1]), dim=1
    )
    pred_previous = torch.cat((context_rgb[:, None].float(), rgb[:, :-1]), dim=1)
    target_delta = target_rgb - target_previous
    pred_delta = rgb - pred_previous
    target_motion = target_delta.abs().mean(dim=2, keepdim=True)
    pred_motion = pred_delta.abs().mean(dim=2, keepdim=True)
    motion_mask = (
        target_motion > float(cfg.get("motion_threshold", 0.03))
    ).float()
    motion_mask = F.max_pool2d(
        motion_mask.flatten(0, 1), kernel_size=5, stride=1, padding=2
    ).reshape_as(motion_mask)
    motion_l1 = _masked_mean((rgb - target_rgb).abs(), motion_mask)
    motion_delta_l1 = _masked_mean(
        (pred_delta - target_delta).abs(), motion_mask
    )
    target_motion_mean = _masked_mean(target_motion, motion_mask)
    pred_motion_mean = _masked_mean(pred_motion, motion_mask)
    motion_magnitude = (pred_motion_mean - target_motion_mean).abs()
    background_mask = 1.0 - motion_mask
    background_tolerance = float(cfg.get("background_motion_tolerance", 0.005))
    background_motion = _masked_mean(
        F.relu(pred_motion - target_motion - background_tolerance),
        background_mask,
    )
    motion_ratio = pred_motion_mean.detach() / target_motion_mean.detach().clamp_min(1.0e-6)
    temporal = F.l1_loss(pred_delta, target_delta)
    spatial_gradient = _gradient_l1(rgb, target_rgb)

    motion_mask_bce = rgb.new_zeros(())
    motion_mask_dice = rgb.new_zeros(())
    if (
        float(cfg.get("motion_mask_bce", 0.0)) > 0.0
        or float(cfg.get("motion_mask_dice", 0.0)) > 0.0
    ):
        if "motion_logit" not in pred:
            raise RuntimeError(
                "motion-head supervision requested but rollout omitted motion_logit"
            )
        motion_logit = pred["motion_logit"].float()
        if motion_logit.shape != motion_mask.shape:
            raise RuntimeError(
                f"motion-head shape mismatch: {tuple(motion_logit.shape)} != {tuple(motion_mask.shape)}"
            )
        positives = motion_mask.sum()
        negatives = motion_mask.numel() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(
            min=1.0, max=float(cfg.get("motion_pos_weight_cap", 20.0))
        )
        motion_mask_bce = F.binary_cross_entropy_with_logits(
            motion_logit, motion_mask, pos_weight=pos_weight
        )
        probability = torch.sigmoid(motion_logit)
        intersection = (probability * motion_mask).sum(dim=(2, 3, 4))
        denominator = (probability + motion_mask).sum(dim=(2, 3, 4))
        motion_mask_dice = (
            1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        ).mean()

    flat_pred = rgb.reshape(-1, 3, 64, 64)
    flat_target = target_rgb.reshape(-1, 3, 64, 64)
    lpips_per_frame = lpips_model(
        flat_pred * 2.0 - 1.0, flat_target * 2.0 - 1.0
    ).reshape(rgb.shape[0], horizon, -1).mean(dim=-1)
    lpips = (lpips_per_frame * horizon_weights[None]).mean()
    i3d_alignment = rgb.new_zeros(())
    i3d_cosine = rgb.new_zeros(())
    i3d_smooth_l1 = rgb.new_zeros(())
    if compute_i3d:
        if i3d is None:
            raise RuntimeError("I3D alignment was scheduled without an I3D model")
        i3d_alignment, i3d_cosine, i3d_smooth_l1 = _i3d_alignment_loss(
            rgb,
            target_rgb,
            i3d,
            smooth_l1_weight=float(cfg.get("i3d_smooth_l1_weight", 0.25)),
        )
    zero = total_zero = rgb.new_zeros(())
    action_token_l1 = rgb.new_zeros(())
    action_rgb_l1 = rgb.new_zeros(())
    if zero_pred is not None:
        action_token_l1 = (tokens - zero_pred["pred_tokens"].float()).abs().mean()
        action_rgb_l1 = (rgb - zero_pred["rgb"].float()).abs().mean()
        token_margin = rgb.new_tensor(float(cfg.get("action_token_margin", 0.0)))
        rgb_margin = rgb.new_tensor(float(cfg.get("action_rgb_margin", 0.0)))
        zero = F.relu(token_margin - action_token_l1)
        total_zero = F.relu(rgb_margin - action_rgb_l1)
    action_gt_rank = rgb.new_zeros(())
    action_gt_win_rate = rgb.new_zeros(())
    if zero_pred is not None and float(cfg.get("action_gt_rank_weight", 0.0)) > 0.0:
        factual_error = (rgb - target_rgb).abs().mean(dim=(1, 2, 3, 4))
        negative_error = (
            zero_pred["rgb"].float() - target_rgb
        ).abs().mean(dim=(1, 2, 3, 4))
        rank_margin = float(cfg.get("action_gt_rank_margin", 0.002))
        action_gt_rank = F.relu(rank_margin + factual_error - negative_error).mean()
        action_gt_win_rate = (factual_error < negative_error).float().mean().detach()
    total = (
        float(cfg.get("token_l1", 1.0)) * token_l1
        + float(cfg.get("token_cos", 0.1)) * token_cos
        + float(cfg.get("rgb_l1", 1.2)) * rgb_l1
        + float(cfg.get("rgb_lpips", 0.4)) * lpips
        + float(cfg.get("motion_l1", 1.0)) * motion_l1
        + float(cfg.get("temporal_l1", 0.2)) * temporal
        + float(cfg.get("motion_delta_l1", 0.0)) * motion_delta_l1
        + float(cfg.get("motion_magnitude", 0.0)) * motion_magnitude
        + float(cfg.get("background_motion", 0.0)) * background_motion
        + float(cfg.get("motion_mask_bce", 0.0)) * motion_mask_bce
        + float(cfg.get("motion_mask_dice", 0.0)) * motion_mask_dice
        + float(cfg.get("horizon_rgb_l1", 0.0)) * horizon_rgb_l1
        + float(cfg.get("spatial_gradient", 0.0)) * spatial_gradient
        + float(cfg.get("i3d_alignment", 0.0)) * i3d_alignment
        + float(cfg.get("action_token_weight", 0.0)) * zero
        + float(cfg.get("action_rgb_weight", 0.0)) * total_zero
        + float(cfg.get("action_gt_rank_weight", 0.0)) * action_gt_rank
    )
    return {
        "total": total, "token_l1": token_l1, "token_cos": token_cos,
        "rgb_l1": rgb_l1, "lpips": lpips, "motion_l1": motion_l1,
        "temporal_l1": temporal, "motion_delta_l1": motion_delta_l1,
        "motion_magnitude": motion_magnitude,
        "background_motion": background_motion,
        "motion_mask_bce": motion_mask_bce,
        "motion_mask_dice": motion_mask_dice,
        "horizon_rgb_l1": horizon_rgb_l1,
        "spatial_gradient": spatial_gradient,
        "pred_motion_mean": pred_motion_mean.detach(),
        "target_motion_mean": target_motion_mean.detach(),
        "motion_ratio": motion_ratio,
        "i3d_alignment": i3d_alignment,
        "i3d_cosine": i3d_cosine,
        "i3d_smooth_l1": i3d_smooth_l1,
        "i3d_applied": rgb.new_tensor(float(compute_i3d)),
        "action_token_l1": action_token_l1,
        "action_rgb_l1": action_rgb_l1,
        "action_token_margin_loss": zero,
        "action_rgb_margin_loss": total_zero,
        "action_gt_rank": action_gt_rank,
        "action_gt_win_rate": action_gt_win_rate,
    }


def _batch(batch: Mapping[str, Any], device: torch.device, model) -> tuple[torch.Tensor, ...]:
    context_latent = batch["context_latent"].to(device, non_blocking=True)
    target_latent = batch["target_latent"].to(device, non_blocking=True)
    context_two = _decode(model, context_latent)
    target_tokens = _decode(model, target_latent)
    state = pad_two_frame_history(context_two, native_t=16)
    return (
        state,
        target_tokens,
        batch["task_emb"].to(device, non_blocking=True),
        batch["context_rgb"].to(device, non_blocking=True),
        batch["target_rgb"].to(device, non_blocking=True),
        batch["action_cond"].to(device, non_blocking=True),
        batch["zero_action_cond"].to(device, non_blocking=True),
    )


def _frechet(a: np.ndarray, b: np.ndarray) -> float:
    import scipy.linalg
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ma, mb = a.mean(0), b.mean(0)
    ca = a.T @ a / len(a) - np.outer(ma, ma)
    cb = b.T @ b / len(b) - np.outer(mb, mb)
    root, _ = scipy.linalg.sqrtm(ca @ cb, disp=False)
    return float(np.real(np.square(ma - mb).sum() + np.trace(ca + cb - 2 * root)))


@torch.inference_mode()
def _evaluate(model, dataset, cfg, rank, world, device, lpips_model, i3d) -> dict[str, float]:
    import piqa

    indices = list(range(rank, len(dataset), world))
    loader = DataLoader(Subset(dataset, indices), batch_size=int(cfg["eval"]["batch_size_per_gpu"]), shuffle=False, num_workers=1)
    sums = torch.zeros(13, device=device, dtype=torch.float64)
    pred_features: list[np.ndarray] = []
    target_features: list[np.ndarray] = []
    ssim_fn = piqa.SSIM(window_size=11, sigma=1.5, n_channels=3, reduction="none").to(device).eval()
    model.eval()
    for batch in loader:
        state, target_tokens, task, context, target_rgb, actions, zero_actions = _batch(batch, device, model)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = rollout_k8_plus_2(model, state, task, context, actions)
            zero = rollout_k8_plus_2(model, state, task, context, zero_actions)
        rgb = pred["rgb"].float().clamp(0, 1)
        mse = (rgb - target_rgb).square().mean(dim=(1, 2, 3, 4))
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
        # Match the pinned iVideoGPT frame metric implementation.
        ssim = ssim_fn(target_rgb.flatten(0, 1), rgb.flatten(0, 1)).reshape(len(rgb), 10).mean(1)
        lp = lpips_model(target_rgb.flatten(0, 1) * 2 - 1, rgb.flatten(0, 1) * 2 - 1).reshape(len(rgb), 10).mean(1)
        token_sens = (pred["pred_tokens"].float() - zero["pred_tokens"].float()).abs().mean(dim=(1, 2, 3))
        rgb_sens = (rgb - zero["rgb"].float().clamp(0, 1)).abs().mean(dim=(1, 2, 3, 4))
        pred_previous = torch.cat((context[:, None].float(), rgb[:, :-1]), dim=1)
        target_previous = torch.cat((context[:, None].float(), target_rgb[:, :-1]), dim=1)
        pred_motion = (rgb - pred_previous).abs().mean(dim=(1, 2, 3, 4))
        target_motion = (target_rgb - target_previous).abs().mean(dim=(1, 2, 3, 4))
        pred_range = (rgb[:, -1] - context.float()).abs().mean(dim=(1, 2, 3))
        target_range = (target_rgb[:, -1] - context.float()).abs().mean(dim=(1, 2, 3))
        factual_error = (rgb - target_rgb).abs().mean(dim=(1, 2, 3, 4))
        zero_rgb = zero["rgb"].float().clamp(0, 1)
        zero_error = (zero_rgb - target_rgb).abs().mean(dim=(1, 2, 3, 4))
        action_wins = (factual_error < zero_error).float()
        threshold = float(cfg["loss"].get("motion_threshold", 0.03))
        pred_motion_fraction = (
            (rgb - pred_previous).abs().mean(dim=2) > threshold
        ).float().mean(dim=(1, 2, 3))
        target_motion_fraction = (
            (target_rgb - target_previous).abs().mean(dim=2) > threshold
        ).float().mean(dim=(1, 2, 3))
        sums += torch.tensor([
            float(psnr.sum()), float(ssim.sum()), float(lp.sum()),
            float(token_sens.sum()), float(rgb_sens.sum()),
            float(pred_motion.sum()), float(target_motion.sum()),
            float(pred_range.sum()), float(target_range.sum()),
            float(action_wins.sum()), float(pred_motion_fraction.sum()),
            float(target_motion_fraction.sum()), float(len(rgb)),
        ], device=device, dtype=torch.float64)
        for source, sink in ((rgb, pred_features), (target_rgb, target_features)):
            features = i3d(
                source.permute(0, 2, 1, 3, 4).contiguous() * 255.0,
                rescale=True, resize=True, return_features=True,
            )
            sink.append(features.float().cpu().numpy())
    if world > 1:
        dist.all_reduce(sums)
    local = (np.concatenate(pred_features), np.concatenate(target_features))
    gathered: list[Any] = [None for _ in range(world)]
    if world > 1:
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    result: dict[str, float] = {}
    if rank == 0:
        pred_f = np.concatenate([item[0] for item in gathered])
        target_f = np.concatenate([item[1] for item in gathered])
        count = max(1.0, float(sums[-1]))
        result = {
            "psnr": float(sums[0] / count), "ssim": float(sums[1] / count),
            "lpips": float(sums[2] / count), "action_token_l1": float(sums[3] / count),
            "action_rgb_l1": float(sums[4] / count),
            "pred_motion_mean": float(sums[5] / count),
            "target_motion_mean": float(sums[6] / count),
            "motion_ratio": float(sums[5] / sums[6].clamp_min(1.0e-12)),
            "pred_range_mean": float(sums[7] / count),
            "target_range_mean": float(sums[8] / count),
            "range_ratio": float(sums[7] / sums[8].clamp_min(1.0e-12)),
            "action_gt_win_rate": float(sums[9] / count),
            "pred_motion_fraction": float(sums[10] / count),
            "target_motion_fraction": float(sums[11] / count),
            "count": int(count),
            "fvd": _frechet(pred_f, target_f),
        }
    objects: list[Any] = [result]
    if world > 1:
        dist.broadcast_object_list(objects, src=0)
    model.train()
    return objects[0]


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--disable-checkpoint", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.out_root is not None:
        cfg["out_root"] = str(args.out_root)
    if args.disable_eval:
        cfg["eval"]["enabled"] = False
    if args.disable_checkpoint:
        cfg["train"]["disable_checkpoint"] = True
    rank, world, _local, device = _setup()
    seed = int(cfg["train"].get("seed", 1707))
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    data_cfg = cfg["data"]
    window_policy = data_cfg.get("window_policy")
    remote_kwargs = {
        "remote_cache_base_url": data_cfg.get("remote_cache_base_url"),
        "remote_cache_path_prefix": data_cfg.get("remote_cache_path_prefix"),
        "remote_cache_timeout_seconds": float(
            data_cfg.get("remote_cache_timeout_seconds", 30.0)
        ),
    }
    primary_train_ds = RoboNetAdaptDataset(
        Path(data_cfg["index"]), Path(data_cfg["action_stats"]),
        split="train", expected_window_policy=window_policy, **remote_kwargs,
    )
    train_parts: list[Dataset] = [primary_train_ds]
    extra_train_index = data_cfg.get("extra_train_index")
    extra_train_repeats = int(data_cfg.get("extra_train_repeats", 0))
    if extra_train_index and extra_train_repeats > 0:
        extra_train_ds = RoboNetAdaptDataset(
            Path(extra_train_index), Path(data_cfg["action_stats"]),
            split="train",
            expected_window_policy=str(
                data_cfg.get("extra_train_window_policy", "fixed_start0")
            ),
            **remote_kwargs,
        )
        train_parts.extend([extra_train_ds] * extra_train_repeats)
    train_ds: Dataset = (
        train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    )
    early_val_index = Path(data_cfg.get("early_val_index", data_cfg["index"]))
    early_val_ds = RoboNetAdaptDataset(
        early_val_index, Path(data_cfg["action_stats"]), split="val",
        expected_window_policy=data_cfg.get(
            "early_val_window_policy", window_policy
        ),
        **remote_kwargs,
    )
    random_val_ds = RoboNetAdaptDataset(
        Path(data_cfg["random_val_index"]), Path(data_cfg["action_stats"]),
        split="val", **remote_kwargs,
    )
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        train_ds, batch_size=int(cfg["train"]["batch_size_per_gpu"]), sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 2)), pin_memory=True,
        drop_last=True, persistent_workers=int(cfg["train"].get("num_workers", 2)) > 0,
    )
    checkpoint = torch.load(cfg["parent_checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    parent_cfg, parent_audit = validate_parent_checkpoint(
        checkpoint,
        expected_kind=str(cfg.get("parent_kind", "formal_s1")),
        expected_step=int(cfg.get("parent_step", 30000)),
    )
    output_cfg = adapted_checkpoint_cfg(parent_cfg, cfg["data"]["action_stats"])
    model = build_joint_world_model(parent_cfg["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    renderer_init_audit: dict[str, Any] | None = None
    if cfg.get("renderer_init_checkpoint"):
        renderer_init_audit = load_renderer_initialization(
            model,
            Path(cfg["renderer_init_checkpoint"]),
            prefix=str(cfg.get("renderer_init_prefix", "context_pixel.")),
        )
    prefixes = tuple(cfg["train"]["trainable_prefixes"])
    selected = set(select_trainable_names((name for name, _ in model.named_parameters()), prefixes))
    if not selected or any(name.startswith(("action_policy.", "future_value_head.", "action_proj.")) for name in selected):
        raise RuntimeError("trainable filter must isolate dynamics+renderer and freeze policy/value/action heads")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in selected
    model.to(device)
    if world > 1:
        # A training item performs native K8 followed by K2 before one backward.
        # Re-broadcasting buffers on the second DDP forward mutates buffers that
        # the first autograd graph still references, so buffer broadcast must be
        # disabled for this two-forward/one-backward contract.
        model = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    import lpips
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in lpips_model.parameters():
        parameter.requires_grad = False
    i3d = torch.jit.load(cfg["eval"]["i3d_path"], map_location=device).eval()
    for parameter in i3d.parameters():
        parameter.requires_grad = False
    renderer, renderer_action, core = [], [], []
    for name, parameter in _target(model).named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("context_pixel.action_proj."):
            renderer_action.append(parameter)
        elif name.startswith("context_pixel."):
            renderer.append(parameter)
        else:
            core.append(parameter)
    parameter_groups = []
    if core:
        parameter_groups.append({
            "params": core,
            "lr": float(cfg["train"]["core_lr"]),
            "group_name": "core",
        })
    if renderer:
        parameter_groups.append({
            "params": renderer,
            "lr": float(cfg["train"]["renderer_lr"]),
            "group_name": "renderer",
        })
    if renderer_action:
        parameter_groups.append({
            "params": renderer_action,
            "lr": float(cfg["train"].get("renderer_action_lr", cfg["train"]["renderer_lr"])),
            "group_name": "renderer_action",
        })
    optimizer = torch.optim.AdamW(
        parameter_groups, betas=(0.9, 0.95),
        weight_decay=float(cfg["train"].get("weight_decay", 0.02)),
    )
    max_steps = int(cfg["train"]["max_steps"]); warmup = int(cfg["train"].get("warmup_steps", 0))
    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(cfg["out_root"]); log_path = output / "metrics.jsonl"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({
            "status": "initialized", "world": world, "train": len(train_ds),
            "primary_train": len(primary_train_ds),
            "extra_train_repeats": extra_train_repeats,
            "early_val": len(early_val_ds), "random_val": len(random_val_ds),
            "trainable_millions": sum(p.numel() for p in _target(model).parameters() if p.requires_grad) / 1e6,
            "parent": cfg["parent_checkpoint"], "parent_audit": parent_audit,
            "renderer_init_audit": renderer_init_audit,
            "native_chunks": [8, 2],
        }), flush=True)
    step = 0; epoch = 0; started = time.perf_counter(); milestones = set(map(int, cfg["train"]["checkpoint_steps"]))
    model.train()
    def run_dual_validation(validation_step: int) -> None:
        for validation_split, dataset in (
            ("early_start0", early_val_ds),
            ("random_window", random_val_ds),
        ):
            metrics = _evaluate(model, dataset, cfg, rank, world, device, lpips_model, i3d)
            if rank == 0:
                record = {
                    "step": validation_step,
                    "validation_split": validation_split,
                    "elapsed_seconds": time.perf_counter() - started,
                    **metrics,
                }
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps({"status": "validation", **record}), flush=True)

    validation_enabled = bool(cfg["eval"].get("enabled", True))
    if validation_enabled and bool(cfg["eval"].get("at_start", False)):
        run_dual_validation(0)
    while step < max_steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            if step >= max_steps:
                break
            state, target_tokens, task, context_rgb, target_rgb, actions, zero_actions = _batch(batch, device, model)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = rollout_k8_plus_2(model, state, task, context_rgb, actions)
                use_counterfactual = (
                    float(cfg["loss"].get("action_token_weight", 0.0)) > 0.0
                    or float(cfg["loss"].get("action_rgb_weight", 0.0)) > 0.0
                    or float(cfg["loss"].get("action_gt_rank_weight", 0.0)) > 0.0
                )
                negative_actions = zero_actions
                negative_mode = str(
                    cfg["loss"].get("action_negative_mode", "zero")
                ).strip().lower()
                if negative_mode == "shuffle" and actions.shape[0] > 1:
                    negative_actions = torch.roll(actions, shifts=1, dims=0)
                elif negative_mode not in {"zero", "shuffle"}:
                    raise RuntimeError(f"unsupported action negative mode: {negative_mode}")
                zero_pred = (
                    rollout_k8_plus_2(model, state, task, context_rgb, negative_actions)
                    if use_counterfactual else None
                )
                i3d_every = max(1, int(cfg["loss"].get("i3d_every_steps", 1)))
                compute_i3d = (
                    float(cfg["loss"].get("i3d_alignment", 0.0)) > 0.0
                    and step % i3d_every == 0
                )
                losses = _losses(
                    pred, target_tokens, target_rgb, context_rgb, lpips_model,
                    cfg["loss"], zero_pred=zero_pred, i3d=i3d,
                    compute_i3d=compute_i3d,
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"non-finite RoboNet adaptation loss at step {step + 1}")
            losses["total"].backward()
            grad = torch.nn.utils.clip_grad_norm_([
                p for p in _target(model).parameters() if p.requires_grad
            ], float(cfg["train"].get("grad_clip", 1.0)))
            if not torch.isfinite(grad):
                raise RuntimeError(f"non-finite RoboNet adaptation gradient at step {step + 1}")
            optimizer.step(); scheduler.step(); step += 1
            if rank == 0 and (step == 1 or step % int(cfg["train"].get("print_every", 10)) == 0):
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "status": "train", "step": step, "seconds_per_step": elapsed / step,
                    "total": float(losses["total"].detach()), "token_l1": float(losses["token_l1"].detach()),
                    "rgb_l1": float(losses["rgb_l1"].detach()), "lpips": float(losses["lpips"].detach()),
                    "motion_delta_l1": float(losses["motion_delta_l1"].detach()),
                    "motion_mask_bce": float(losses["motion_mask_bce"].detach()),
                    "motion_mask_dice": float(losses["motion_mask_dice"].detach()),
                    "motion_ratio": float(losses["motion_ratio"]),
                    "pred_motion_mean": float(losses["pred_motion_mean"]),
                    "target_motion_mean": float(losses["target_motion_mean"]),
                    "i3d_alignment": float(losses["i3d_alignment"].detach()),
                    "i3d_applied": bool(losses["i3d_applied"].item()),
                    "action_token_l1": float(losses["action_token_l1"].detach()),
                    "action_rgb_l1": float(losses["action_rgb_l1"].detach()),
                    "action_gt_rank": float(losses["action_gt_rank"].detach()),
                    "action_gt_win_rate": float(losses["action_gt_win_rate"]),
                    "grad_norm": float(grad),
                    "learning_rates": {
                        str(group.get("group_name", index)): float(lr)
                        for index, (group, lr) in enumerate(
                            zip(optimizer.param_groups, scheduler.get_last_lr())
                        )
                    },
                    "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                }), flush=True)
            evaluate = validation_enabled and (
                step in milestones
                or step == max_steps
                or step % int(cfg["eval"]["every_steps"]) == 0
            )
            if evaluate:
                run_dual_validation(step)
            if (
                step in milestones or step == max_steps
            ) and not bool(cfg["train"].get("disable_checkpoint", False)):
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    _atomic_checkpoint(output / "ckpt" / f"step_{step:08d}.pt", {
                        "schema": "wm3d_v7_robonet_s1_adapt_v1", "step": step,
                        "model": _target(model).state_dict(), "cfg": output_cfg,
                        "adapt_config": cfg, "parent_checkpoint": cfg["parent_checkpoint"],
                        "renderer_init_audit": renderer_init_audit,
                    })
                if world > 1:
                    dist.barrier()
        epoch += 1
    if rank == 0 and validation_enabled:
        records = [
            json.loads(line) for line in log_path.read_text().splitlines() if line.strip()
        ]
        selection = select_dual_validation_checkpoint(
            records,
            checkpoint_root=output / "ckpt",
            parent_checkpoint=Path(cfg["parent_checkpoint"]),
            random_regression_limit=float(cfg["eval"].get("random_regression_limit", 0.05)),
            allow_step0_parent=renderer_init_audit is None,
        )
        _atomic_json(output / "selected_checkpoint.json", selection)
        print(json.dumps({"status": "selected", **selection}), flush=True)
    if rank == 0:
        print(json.dumps({"status": "complete", "step": step, "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
