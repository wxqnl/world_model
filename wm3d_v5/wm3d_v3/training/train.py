"""v3 joint training. DDP-aware, bf16, gradient checkpointing optional."""
from __future__ import annotations
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.splits import episode_split, load_clip_split_file, random_window_indices
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import LossWeights, compute_losses
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.hunyuan_latent_adapter import (
    HunyuanLatentAdapter,
    HunyuanLatentAdapterConfig,
)
from wm3d_v3.training.lr_schedule import (
    build_lr_scheduler,
    resolve_optimizer_settings,
)
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.state_stream import StateConfig


def setup_ddp():
    if "RANK" in os.environ:
        backend = os.environ.get("WM3D_DDP_BACKEND", "nccl")
        dist.init_process_group(backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def build_model(cfg: dict) -> JointWorldModel:
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    jc = JointConfig(
        dual=dc,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        geom_upsample_mode=cfg["model"].get("geom_upsample_mode", "transpose"),
        enable_geom_extra=cfg["model"].get("enable_geom_extra", True),
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_context_pixel=cfg["model"].get("enable_context_pixel", False),
        context_pixel_hidden=cfg["model"].get("context_pixel_hidden", 384),
        context_pixel_action_dim=cfg["model"].get("context_pixel_action_dim", 7),
        context_pixel_task_dim=cfg["model"].get("context_pixel_task_dim"),
        context_pixel_residual_scale=cfg["model"].get("context_pixel_residual_scale", 0.75),
        context_pixel_use_action=cfg["model"].get("context_pixel_use_action", True),
        context_pixel_use_task=cfg["model"].get("context_pixel_use_task", True),
        context_pixel_predict_motion=cfg["model"].get("context_pixel_predict_motion", False),
        context_pixel_motion_blend_gain=cfg["model"].get("context_pixel_motion_blend_gain", 0.0),
        enable_control_head=cfg["model"].get("enable_control_head", False),
        control_hidden=cfg["model"].get("control_hidden", 128),
        control_output_size=cfg["model"].get("control_output_size", 256),
        control_fuse_size=cfg["model"].get("control_fuse_size", 64),
        control_refine_channels=cfg["model"].get("control_refine_channels", 16),
        control_use_refine=cfg["model"].get("control_use_refine", True),
        control_action_dim=cfg["model"].get("control_action_dim", 7),
        control_task_dim=cfg["model"].get("control_task_dim"),
        control_use_context=cfg["model"].get("control_use_context", True),
        control_use_action=cfg["model"].get("control_use_action", True),
        control_use_task=cfg["model"].get("control_use_task", True),
        enable_progress_head=cfg["model"].get("enable_progress_head", False),
        progress_hidden=cfg["model"].get("progress_hidden", 256),
        progress_layers=cfg["model"].get("progress_layers", 2),
        progress_heads=cfg["model"].get("progress_heads", 4),
        progress_action_dim=cfg["model"].get("progress_action_dim", 7),
        progress_task_dim=cfg["model"].get("progress_task_dim"),
        progress_max_horizon=cfg["model"].get("progress_max_horizon", 32),
        progress_use_action=cfg["model"].get("progress_use_action", True),
        progress_use_task=cfg["model"].get("progress_use_task", True),
        enable_action_proposer=cfg["model"].get("enable_action_proposer", False),
        proposer_hidden=cfg["model"].get("proposer_hidden", 512),
        proposer_layers=cfg["model"].get("proposer_layers", 3),
        proposer_candidates=cfg["model"].get("proposer_candidates", 4),
        proposer_horizon=cfg["model"].get("proposer_horizon"),
        proposer_task_dim=cfg["model"].get("proposer_task_dim"),
        proposer_dropout=cfg["model"].get("proposer_dropout", 0.0),
        proposer_use_task=cfg["model"].get("proposer_use_task", True),
        enable_action_policy=cfg["model"].get("enable_action_policy", False),
        policy_hidden=cfg["model"].get("policy_hidden", 768),
        policy_layers=cfg["model"].get("policy_layers", 6),
        policy_heads=cfg["model"].get("policy_heads", 8),
        policy_chunk_layers=cfg["model"].get("policy_chunk_layers", 2),
        policy_horizon=cfg["model"].get("policy_horizon"),
        policy_task_dim=cfg["model"].get("policy_task_dim"),
        policy_max_context=cfg["model"].get("policy_max_context"),
        policy_dropout=cfg["model"].get("policy_dropout", 0.1),
        policy_use_task=cfg["model"].get("policy_use_task", True),
        policy_lowdim_dim=cfg["model"].get("policy_lowdim_dim", 0),
        policy_object_state_dim=cfg["model"].get("policy_object_state_dim", 0),
        policy_plan_state_dim=cfg["model"].get("policy_plan_state_dim", 0),
        policy_action_history_len=cfg["model"].get("policy_action_history_len", 0),
        policy_action_history_dim=cfg["model"].get("policy_action_history_dim", 7),
        policy_use_progress=cfg["model"].get("policy_use_progress", False),
        policy_progress_dim=cfg["model"].get("policy_progress_dim", 1),
        policy_progress_mode=cfg["model"].get("policy_progress_mode", "token"),
        policy_enable_local_residual=cfg["model"].get("policy_enable_local_residual", False),
        policy_local_hidden=cfg["model"].get("policy_local_hidden", 256),
        policy_local_layers=cfg["model"].get("policy_local_layers", 2),
        policy_local_residual_scale=cfg["model"].get("policy_local_residual_scale", 1.0),
        policy_local_use_lowdim=cfg["model"].get("policy_local_use_lowdim", True),
        policy_local_use_plan_state=cfg["model"].get("policy_local_use_plan_state", True),
        policy_local_use_progress=cfg["model"].get("policy_local_use_progress", True),
        policy_local_use_action_history=cfg["model"].get("policy_local_use_action_history", True),
        policy_enable_waypoint_head=cfg["model"].get("policy_enable_waypoint_head", False),
        policy_waypoint_hidden=cfg["model"].get("policy_waypoint_hidden", 256),
        policy_waypoint_layers=cfg["model"].get("policy_waypoint_layers", 2),
        policy_waypoint_num_stages=cfg["model"].get("policy_waypoint_num_stages", 4),
        policy_waypoint_stage_dim=cfg["model"].get("policy_waypoint_stage_dim", 4),
        policy_waypoint_active_stages=tuple(cfg["model"].get("policy_waypoint_active_stages", ())),
        policy_waypoint_residual_scale=cfg["model"].get("policy_waypoint_residual_scale", 1.0),
        policy_waypoint_mode=cfg["model"].get("policy_waypoint_mode", "residual"),
        policy_waypoint_use_summary=cfg["model"].get("policy_waypoint_use_summary", True),
        policy_waypoint_use_lowdim=cfg["model"].get("policy_waypoint_use_lowdim", True),
        policy_waypoint_use_plan_state=cfg["model"].get("policy_waypoint_use_plan_state", True),
        policy_waypoint_use_progress=cfg["model"].get("policy_waypoint_use_progress", True),
        policy_waypoint_use_action_history=cfg["model"].get("policy_waypoint_use_action_history", True),
        policy_enable_prior=cfg["model"].get("policy_enable_prior", False),
        policy_prior_chunk_layers=cfg["model"].get("policy_prior_chunk_layers", 1),
        enable_bridging=cfg["model"].get("enable_bridging", True),
        enable_aux_idm=cfg["model"].get("enable_aux_idm", False),
        aux_idm_hidden=cfg["model"].get("aux_idm_hidden", 1024),
        aux_idm_layers=cfg["model"].get("aux_idm_layers", 3),
        enable_world_prior=cfg["model"].get("enable_world_prior", False),
        world_prior_hidden=cfg["model"].get("world_prior_hidden", cfg["model"].get("state", {}).get("hidden", 768)),
        world_prior_layers=cfg["model"].get("world_prior_layers", 4),
        world_prior_heads=cfg["model"].get("world_prior_heads", cfg["model"].get("state", {}).get("n_heads", 8)),
        world_prior_mlp_mult=cfg["model"].get("world_prior_mlp_mult", 4),
        world_prior_dropout=cfg["model"].get("world_prior_dropout", 0.0),
        world_prior_task_dim=cfg["model"].get("world_prior_task_dim"),
        world_prior_action_dim=cfg["model"].get("world_prior_action_dim", 7),
        world_prior_use_context=cfg["model"].get("world_prior_use_context", True),
        world_prior_use_action=cfg["model"].get("world_prior_use_action", True),
        world_prior_predict_initial=cfg["model"].get("world_prior_predict_initial", True),
    )
    return JointWorldModel(jc)


def _data_split_cfg(data_cfg: dict) -> dict:
    split_cfg = data_cfg.get("split") or {}
    if isinstance(split_cfg, (str, Path)):
        return {"file": str(split_cfg)}
    if not isinstance(split_cfg, dict):
        raise ValueError("data.split must be a mapping or split-file path")
    return dict(split_cfg)


def _split_value(data_cfg: dict, split_cfg: dict, key: str, default=None):
    return split_cfg.get(key, data_cfg.get(key, default))


def _explicit_clip_ids(data_cfg: dict, split_cfg: dict) -> tuple[list[str] | None, list[str] | None]:
    train_ids = split_cfg.get("train_clip_ids")
    val_ids = split_cfg.get("val_clip_ids")
    split_file = data_cfg.get("split_file") or split_cfg.get("file") or split_cfg.get("path")
    if split_file:
        file_split = load_clip_split_file(split_file)
        train_ids = train_ids if train_ids is not None else file_split["train_clip_ids"]
        val_ids = val_ids if val_ids is not None else file_split["val_clip_ids"]
    return train_ids, val_ids


def _window_config(data_cfg: dict, model_cfg: dict | None = None) -> WindowConfig:
    model_cfg = model_cfg or {}
    policy_state_default = any(
        int(model_cfg.get(k, 0) or 0) > 0
        for k in ("policy_lowdim_dim", "policy_object_state_dim", "policy_plan_state_dim", "policy_action_history_len")
    ) or bool(model_cfg.get("policy_use_progress", False))
    return WindowConfig(T=data_cfg["T"], k=data_cfg["k"],
                        stride=data_cfg["stride"],
                        cache_root=Path(data_cfg["cache_root"]),
                        tokens_subdir=data_cfg.get("tokens_subdir", "vggt_pooled"),
                        action_stats=Path(data_cfg["action_stats"])
                        if data_cfg.get("action_stats") else None,
                        require_task_emb=bool(data_cfg.get("require_task_emb", False)),
                        load_task_text=bool(data_cfg.get("load_task_text", False)),
                        load_rgb=bool(data_cfg.get("load_rgb", True)),
                        load_geom=bool(data_cfg.get("load_geom", True)),
                        load_state_tgt=bool(data_cfg.get("load_state_tgt", True)),
                        load_geom_extra=bool(data_cfg.get("load_geom_extra", False)),
                        require_geom_extra=bool(data_cfg.get("require_geom_extra", False)),
                        window_geom_subdir=data_cfg.get("window_geom_subdir", "vggt_window_geom_p64"),
                        use_window_tokens=bool(data_cfg.get("use_window_tokens", False)),
                        max_windows_per_episode=int(data_cfg.get("max_windows_per_episode", 0) or 0),
                        trust_window_geom_cache=bool(data_cfg.get("trust_window_geom_cache", False)),
                        allow_pseudo_progress_targets=bool(data_cfg.get("allow_pseudo_progress_targets", False)),
                        require_progress=bool(data_cfg.get("require_progress", bool(model_cfg.get("policy_use_progress", False)))),
                        load_policy_state=bool(data_cfg.get("load_policy_state", policy_state_default)),
                        require_policy_state=bool(data_cfg.get("require_policy_state", False)),
                        policy_lowdim_dim=int(data_cfg.get("policy_lowdim_dim", model_cfg.get("policy_lowdim_dim", 0)) or 0),
                        policy_object_state_dim=int(data_cfg.get("policy_object_state_dim", model_cfg.get("policy_object_state_dim", 0)) or 0),
                        policy_plan_state_dim=int(data_cfg.get("policy_plan_state_dim", model_cfg.get("policy_plan_state_dim", 0)) or 0),
                        policy_action_history_len=int(data_cfg.get("policy_action_history_len", model_cfg.get("policy_action_history_len", 0)) or 0),
                        policy_action_history_dim=int(data_cfg.get("policy_action_history_dim", model_cfg.get("policy_action_history_dim", 7)) or 7))


def _sample_record(dataset, sample_idx: int):
    if isinstance(dataset, Subset):
        return _sample_record(dataset.dataset, int(dataset.indices[sample_idx]))
    if hasattr(dataset, "index") and hasattr(dataset, "records"):
        record_idx, _start = dataset.index[int(sample_idx)]
        return dataset.records[record_idx]
    if hasattr(dataset, "records"):
        return dataset.records[int(sample_idx)]
    return None


def build_dataset_sample_weights(dataset, sampler_cfg: dict | bool | None) -> torch.Tensor | None:
    if not sampler_cfg:
        return None
    if sampler_cfg is True:
        sampler_cfg = {"enabled": True}
    if not isinstance(sampler_cfg, dict) or not bool(sampler_cfg.get("enabled", False)):
        return None
    n = len(dataset)
    if n <= 0:
        return None
    dataset_weights = {str(k): float(v) for k, v in (sampler_cfg.get("dataset_weights") or {}).items()}
    balance_by_dataset = bool(sampler_cfg.get("balance_by_dataset", True))
    records = [_sample_record(dataset, i) for i in range(n)]
    counts: dict[str, int] = {}
    for rec in records:
        name = str(getattr(rec, "dataset", "unknown")) if rec is not None else "unknown"
        counts[name] = counts.get(name, 0) + 1
    weights = []
    for rec in records:
        name = str(getattr(rec, "dataset", "unknown")) if rec is not None else "unknown"
        repeat_weight = float(getattr(rec, "repeat_weight", 1.0) or 1.0) if rec is not None else 1.0
        w = repeat_weight * dataset_weights.get(name, 1.0)
        if balance_by_dataset:
            w /= max(1, counts.get(name, 1))
        weights.append(max(0.0, w))
    out = torch.as_tensor(weights, dtype=torch.double)
    if not torch.isfinite(out).all() or float(out.sum()) <= 0.0:
        raise ValueError("weighted_sampler produced no positive finite sample weights")
    return out


class WeightedDistributedSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        self.weights = weights.double().cpu()
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        positive_count = int((self.weights > 0).sum().item())
        if positive_count <= 0:
            raise ValueError("WeightedDistributedSampler needs at least one positive sample weight")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if self.replacement:
            self.num_samples = int(num_samples) if num_samples is not None else int(math.ceil(len(weights) / self.num_replicas))
        else:
            max_per_rank = positive_count // self.num_replicas
            if max_per_rank <= 0:
                raise ValueError(
                    "WeightedDistributedSampler with replacement=False needs at least "
                    "num_replicas positive sample weights"
                )
            requested = int(num_samples) if num_samples is not None else max_per_rank
            self.num_samples = min(requested, max_per_rank)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        return iter(sampled[self.rank:self.total_size:self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_datasets(cfg: dict, overfit_ids=None):
    records = read_manifest(cfg["data"]["manifest"])
    if overfit_ids:
        records = [r for r in records if r.clip_id in overfit_ids]
        if not records:
            raise RuntimeError(f"no records matched overfit ids: {overfit_ids}")
    data_cfg = cfg["data"]
    split_cfg = _data_split_cfg(data_cfg)
    has_episode_split_keys = (
        data_cfg.get("split_file")
        or any(k in split_cfg for k in ("file", "path", "train_clip_ids", "val_clip_ids", "heldout_dataset"))
    )
    mode = split_cfg.get("mode", "episode" if has_episode_split_keys else "random_window")
    val_frac = float(_split_value(data_cfg, split_cfg, "val_frac", 0.0))
    seed = int(_split_value(data_cfg, split_cfg, "seed", 0))
    wcfg = _window_config(data_cfg, cfg.get("model"))

    if mode == "episode":
        train_ids, val_ids = _explicit_clip_ids(data_cfg, split_cfg)
        clip_split = episode_split(
            records,
            val_frac=val_frac,
            seed=seed,
            train_clip_ids=train_ids,
            val_clip_ids=val_ids,
            heldout_dataset=_split_value(data_cfg, split_cfg, "heldout_dataset"),
        )
        train_records = [r for r in records if r.clip_id in clip_split.train_clip_ids]
        val_records = [r for r in records if r.clip_id in clip_split.val_clip_ids]
        tr_ds = OXEWindowDataset(train_records, wcfg)
        val_ds = OXEWindowDataset(val_records, wcfg)
        if len(tr_ds) == 0:
            raise RuntimeError("episode train split empty — caches missing?")
        if len(val_ds) == 0:
            raise RuntimeError("episode val split empty — caches missing?")
        return tr_ds, val_ds

    if mode != "random_window":
        raise ValueError(f"unsupported data.split.mode: {mode}")

    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    if n == 0:
        raise RuntimeError("OXEWindowDataset empty — caches missing?")
    train_idx, val_idx = random_window_indices(n, val_frac=val_frac, seed=seed)
    return Subset(ds, train_idx), Subset(ds, val_idx)


def batch_to_device(batch: dict, device: torch.device, k: int, direct_policy_only: bool = False) -> tuple:
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    context_rgb = None
    tgt = {}
    if not direct_policy_only:
        tgt["s_init_tgt"] = s[:, -1]
        if "s_tgt" in batch:
            tgt["s_tgt"] = batch["s_tgt"].to(device, non_blocking=True)
        if "depth_tgt" in batch:
            tgt["depth_tgt"] = batch["depth_tgt"].to(device, non_blocking=True)
        if "depth_in" in batch:
            tgt["depth_ref"] = batch["depth_in"][:, -1].to(device, non_blocking=True)
        if "rgb_in" in batch and "rgb_tgt" in batch:
            context_rgb = batch["rgb_in"][:, -1].to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous()
            # rgb_tgt: [B, k, 256, 256, 3]; permute -> [B, k, 3, 256, 256]
            rgb_tgt_p = batch["rgb_tgt"].to(device, non_blocking=True).permute(0, 1, 4, 2, 3)
            tgt["rgb_tgt_p"] = rgb_tgt_p
            tgt["rgb_ref_p"] = context_rgb
    action_tgt = batch["action_tgt"].to(device, non_blocking=True)
    action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
    tgt["action_tgt"] = action_tgt
    tgt["action_tgt_norm"] = action_tgt_norm
    for key in (
        "point_tgt",
        "point_conf_tgt",
        "pose_geom_tgt",
        "pose_geom_conf_tgt",
        "depth_conf_tgt",
        "lowdim_state",
        "object_state",
        "plan_state",
        "action_history",
    ):
        if key in batch:
            tgt[key] = batch[key].to(device, non_blocking=True)
    if "progress_tgt" in batch:
        tgt["progress_tgt"] = batch["progress_tgt"].to(device, non_blocking=True)
    if "terminal_success_tgt" in batch:
        tgt["terminal_success_tgt"] = batch["terminal_success_tgt"].to(device, non_blocking=True)
    if "plausibility_tgt" in batch:
        tgt["plausibility_tgt"] = batch["plausibility_tgt"].to(device, non_blocking=True)
    action_cond = make_action_condition(action_tgt, action_tgt_norm)
    return s, c, action_cond, context_rgb, tgt


def action_policy_kwargs_from_targets(tgt: dict) -> dict:
    kwargs = {}
    for src, dst in (
        ("lowdim_state", "lowdim_state"),
        ("object_state", "object_state"),
        ("plan_state", "plan_state"),
        ("action_history", "action_history"),
    ):
        if src in tgt:
            kwargs[dst] = tgt[src]
    if "progress_state" in tgt:
        kwargs["progress_state"] = tgt["progress_state"]
    elif "progress_tgt" in tgt:
        progress = tgt["progress_tgt"]
        if progress.ndim > 1:
            progress = progress[:, :1]
        kwargs["progress_state"] = progress
    return kwargs


def _forward_joint_model(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    *,
    action_cond: torch.Tensor | None,
    context_rgb: torch.Tensor | None,
    prior_clean_tokens: torch.Tensor | None = None,
    pixel: bool = False,
    bridging: bool = False,
    policy_kwargs: dict | None = None,
) -> dict:
    kwargs = {
        "action_cond": action_cond,
        "context_rgb": context_rgb,
        "prior_clean_tokens": prior_clean_tokens,
        "pixel": pixel,
        "bridging": bridging,
    }
    if policy_kwargs:
        kwargs.update(policy_kwargs)
    return model(s, c, **kwargs)


def _mask_to_shape(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return mask.reshape((mask.shape[0],) + (1,) * (x.ndim - 1))


def apply_condition_dropout(
    s: torch.Tensor,
    c: torch.Tensor,
    action_cond: torch.Tensor | None,
    context_rgb: torch.Tensor | None,
    train_cfg: dict,
    *,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, torch.Tensor]]:
    cfg = train_cfg.get("condition_dropout") or {}
    if not isinstance(cfg, dict):
        raise ValueError("train.condition_dropout must be a mapping when provided")
    enabled = bool(cfg.get("enabled", False))
    action_p = float(cfg.get("action_p", 0.0))
    context_p = float(cfg.get("context_p", cfg.get("rgb_context_p", 0.0)))
    text_only_p = float(cfg.get("text_only_p", 0.0))
    text_context_p = float(cfg.get("text_context_p", 0.0))
    text_action_p = float(cfg.get("text_action_p", 0.0))
    if not training or not enabled or (
        action_p <= 0.0 and context_p <= 0.0
        and text_only_p <= 0.0 and text_context_p <= 0.0 and text_action_p <= 0.0
    ):
        zero = s.new_zeros(())
        return s, c, action_cond, context_rgb, {
            "drop_action_frac": zero,
            "drop_context_frac": zero,
            "drop_text_only_frac": zero,
            "drop_text_context_frac": zero,
            "drop_text_action_frac": zero,
        }

    bsz = s.shape[0]
    explicit_total = max(0.0, text_only_p) + max(0.0, text_context_p) + max(0.0, text_action_p)
    if explicit_total > 1.0 + 1e-6:
        raise ValueError(
            "condition_dropout text_only_p + text_context_p + text_action_p must be <= 1.0"
        )
    u = torch.rand(bsz, device=s.device)
    text_only_mask = u < max(0.0, text_only_p)
    text_context_mask = (u >= max(0.0, text_only_p)) & (
        u < max(0.0, text_only_p) + max(0.0, text_context_p)
    )
    text_action_mask = (u >= max(0.0, text_only_p) + max(0.0, text_context_p)) & (
        u < explicit_total
    )
    residual_mask = u >= explicit_total

    action_mask = text_only_mask | text_context_mask
    context_mask = text_only_mask | text_action_mask
    if residual_mask.any():
        residual_action = torch.rand(bsz, device=s.device) < max(0.0, min(1.0, action_p))
        residual_context = torch.rand(bsz, device=s.device) < max(0.0, min(1.0, context_p))
        action_mask = action_mask | (residual_mask & residual_action)
        context_mask = context_mask | (residual_mask & residual_context)

    s_out = torch.where(_mask_to_shape(context_mask, s), torch.zeros_like(s), s)
    rgb_out = None
    if context_rgb is not None:
        rgb_out = torch.where(_mask_to_shape(context_mask, context_rgb), torch.zeros_like(context_rgb), context_rgb)
    action_out = action_cond
    if action_cond is not None:
        action_out = torch.where(_mask_to_shape(action_mask, action_cond), torch.zeros_like(action_cond), action_cond)
    return s_out, c, action_out, rgb_out, {
        "drop_action_frac": action_mask.float().mean(),
        "drop_context_frac": context_mask.float().mean(),
        "drop_text_only_frac": text_only_mask.float().mean(),
        "drop_text_context_frac": text_context_mask.float().mean(),
        "drop_text_action_frac": text_action_mask.float().mean(),
    }


def prior_clean_tokens_from_targets(tgt: dict) -> torch.Tensor | None:
    if "s_init_tgt" not in tgt or "s_tgt" not in tgt:
        return None
    return torch.cat([tgt["s_init_tgt"][:, None], tgt["s_tgt"]], dim=1)


def load_compatible_state_dict(model: torch.nn.Module, state: dict, strict: bool):
    if strict:
        return model.load_state_dict(state, strict=True)
    current = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    result = model.load_state_dict(compatible, strict=False)
    return SimpleNamespace(
        missing_keys=result.missing_keys,
        unexpected_keys=result.unexpected_keys,
        skipped_keys=skipped,
    )


def load_action_stats_if_available(model, cfg: dict, rank: int, device: torch.device) -> None:
    stats_path = cfg["data"].get("action_stats")
    if not stats_path:
        return
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"action_stats not found: {path}")
    stats = np.load(path)
    target = model.module if isinstance(model, DDP) else model
    mean = torch.as_tensor(stats["mean"][:6], device=device)
    std = torch.as_tensor(stats["std"][:6], device=device)
    target.load_action_stats(mean, std)
    if rank == 0:
        print(f"[rank0] loaded action_stats from {path}")


def apply_trainable_filter(model: torch.nn.Module, train_cfg: dict, rank: int) -> None:
    """Optionally freeze parameters by name prefix before DDP/optimizer setup."""
    trainable_prefixes = tuple(train_cfg.get("trainable_prefixes") or ())
    freeze_prefixes = tuple(train_cfg.get("freeze_prefixes") or ())
    if not trainable_prefixes and not freeze_prefixes:
        return

    frozen = 0
    trainable = 0
    trainable_tensors = 0
    for name, param in model.named_parameters():
        enabled = True
        if trainable_prefixes:
            enabled = name.startswith(trainable_prefixes)
        if freeze_prefixes and name.startswith(freeze_prefixes):
            enabled = False
        param.requires_grad = enabled
        if enabled:
            trainable += param.numel()
            trainable_tensors += 1
        else:
            frozen += param.numel()
    if trainable_tensors == 0:
        raise RuntimeError(
            "trainable filter left no trainable parameters; "
            f"trainable_prefixes={trainable_prefixes} freeze_prefixes={freeze_prefixes}"
        )
    if rank == 0:
        print(
            f"[rank0] trainable filter: trainable={trainable/1e6:.2f}M "
            f"frozen={frozen/1e6:.2f}M trainable_prefixes={list(trainable_prefixes)} "
            f"freeze_prefixes={list(freeze_prefixes)}"
        )


def _normalise_param_name(name: str) -> str:
    while name.startswith("module."):
        name = name[len("module."):]
    return name


def _prefix_matches(name: str, prefix: str) -> bool:
    prefix = str(prefix).strip()
    if not prefix:
        return False
    prefix = prefix[:-1] if prefix.endswith(".") else prefix
    return name == prefix or name.startswith(prefix + ".")


def _lr_multiplier_for_name(name: str, lr_multipliers: dict[str, float]) -> float:
    clean_name = _normalise_param_name(name)
    best_prefix = ""
    best_mult = 1.0
    for prefix, mult in lr_multipliers.items():
        if _prefix_matches(clean_name, prefix) and len(str(prefix)) > len(best_prefix):
            best_prefix = str(prefix)
            best_mult = float(mult)
    return best_mult


def build_optimizer_param_groups(
    model: torch.nn.Module,
    hunyuan_adapter: torch.nn.Module | None,
    train_cfg: dict,
    rank: int,
) -> list[dict]:
    """Build AdamW groups with optional prefix-based LR multipliers.

    Config format:

    train:
      lr_multipliers:
        context_pixel: 0.05
        geom: 0.05
        hunyuan_adapter: 1.0
    """
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    raw_multipliers = train_cfg.get("lr_multipliers") or {}
    if not isinstance(raw_multipliers, dict):
        raise ValueError("train.lr_multipliers must be a mapping of prefix -> multiplier")
    lr_multipliers = {str(k): float(v) for k, v in raw_multipliers.items()}

    grouped: dict[float, dict] = {}
    group_counts: dict[float, int] = {}

    def add_param(name: str, param: torch.nn.Parameter) -> None:
        if not param.requires_grad:
            return
        mult = _lr_multiplier_for_name(name, lr_multipliers)
        if mult <= 0:
            raise ValueError(f"lr multiplier must be positive for {name}, got {mult}")
        group = grouped.setdefault(
            mult,
            {
                "params": [],
                "lr": lr * mult,
                "weight_decay": weight_decay,
                "lr_multiplier": mult,
            },
        )
        group["params"].append(param)
        group_counts[mult] = group_counts.get(mult, 0) + param.numel()

    for name, param in model.named_parameters():
        add_param(name, param)
    if hunyuan_adapter is not None:
        for name, param in hunyuan_adapter.named_parameters():
            add_param(f"hunyuan_adapter.{name}", param)

    groups = [grouped[mult] for mult in sorted(grouped)]
    if not groups:
        raise RuntimeError("optimizer has no trainable parameters after filters")
    if rank == 0:
        if lr_multipliers:
            summary = ", ".join(
                f"mult={mult:g} lr={lr * mult:.3g} params={group_counts[mult] / 1e6:.2f}M"
                for mult in sorted(group_counts)
            )
            print(f"[rank0] optimizer lr groups: {summary}", flush=True)
        else:
            total = sum(group_counts.values())
            print(f"[rank0] optimizer single lr={lr:.3g} params={total / 1e6:.2f}M", flush=True)
    return groups


def _all_reduce_gradients(model: torch.nn.Module, world: int) -> None:
    backend = dist.get_backend()
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if backend == "gloo" and param.grad.is_cuda:
            grad_cpu = param.grad.detach().float().cpu()
            dist.all_reduce(grad_cpu, op=dist.ReduceOp.SUM)
            grad_cpu.div_(world)
            param.grad.copy_(grad_cpu.to(device=param.grad.device, dtype=param.grad.dtype))
        else:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world)


def _distributed_finite_count(value: torch.Tensor | bool, device: torch.device, world: int) -> int:
    if isinstance(value, torch.Tensor):
        finite = bool(value.detach().item())
    else:
        finite = bool(value)
    if world <= 1:
        return int(finite)
    flag = torch.tensor([1 if finite else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return int(flag.item())


def _score_evaluator(out: dict, train_cfg: dict) -> torch.Tensor:
    progress_w = float(train_cfg.get("evaluator_score_progress_weight", 1.0))
    terminal_w = float(train_cfg.get("evaluator_score_terminal_weight", 1.0))
    plaus_w = float(train_cfg.get("evaluator_score_plausibility_weight", 0.0))
    terms: list[torch.Tensor] = []
    weights: list[float] = []
    if progress_w and "progress" in out:
        terms.append(torch.sigmoid(out["progress"].float()).mean(dim=1) * progress_w)
        weights.append(abs(progress_w))
    if terminal_w and "terminal_success_logit" in out:
        terms.append(torch.sigmoid(out["terminal_success_logit"].float()) * terminal_w)
        weights.append(abs(terminal_w))
    if plaus_w and "plausibility_logit" in out:
        terms.append(torch.sigmoid(out["plausibility_logit"].float()) * plaus_w)
        weights.append(abs(plaus_w))
    if not terms:
        return out["pred_tokens"].new_zeros(out["pred_tokens"].shape[0], dtype=torch.float32)
    return torch.stack(terms, dim=0).sum(dim=0) / max(1e-6, sum(weights))


def _counterfactual_action_cond(
    action_cond: torch.Tensor,
    variant: str,
    step: int,
    scaled_factor: float = 2.0,
) -> torch.Tensor:
    cur = action_cond.clone()
    if variant == "zero":
        cur.zero_()
    elif variant == "shuffled":
        if cur.shape[0] > 1:
            cur = torch.roll(cur, shifts=1 + (step % max(1, cur.shape[0] - 1)), dims=0)
        elif cur.shape[1] > 1:
            cur = torch.roll(cur, shifts=1, dims=1)
    elif variant == "sign_flip":
        cur[..., :6].neg_()
    elif variant == "scaled":
        cur[..., :6].mul_(scaled_factor)
    elif variant == "grip_toggle":
        cur[..., 6:7] = 1.0 - cur[..., 6:7]
    else:
        raise ValueError(f"unknown evaluator counterfactual variant: {variant}")
    return cur


def compute_evaluator_pairwise_loss(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    real_out: dict,
    action_cond: torch.Tensor,
    train_cfg: dict,
    *,
    step: int,
    policy_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    weight = float(train_cfg.get("evaluator_pairwise_weight", 0.0))
    if weight <= 0:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_pairwise": zero,
            "evaluator_pairwise_acc": zero,
            "evaluator_pairwise_gap": zero,
        }
    variants = train_cfg.get("evaluator_pairwise_variants") or []
    if isinstance(variants, str):
        variants = [v.strip() for v in variants.split(",") if v.strip()]
    if not variants:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_pairwise": zero,
            "evaluator_pairwise_acc": zero,
            "evaluator_pairwise_gap": zero,
        }
    margin = float(train_cfg.get("evaluator_pairwise_margin", 0.1))
    scaled_factor = float(train_cfg.get("evaluator_pairwise_scaled_factor", 2.0))
    real_score = _score_evaluator(real_out, train_cfg)
    losses = []
    accs = []
    gaps = []
    for variant in variants:
        var_cond = _counterfactual_action_cond(action_cond, variant, step, scaled_factor=scaled_factor)
        var_out = _forward_joint_model(model, s, c, action_cond=var_cond, context_rgb=None, pixel=False, bridging=False, policy_kwargs=policy_kwargs)
        var_score = _score_evaluator(var_out, train_cfg)
        gap = real_score - var_score
        losses.append(torch.relu(margin - gap).mean())
        accs.append((gap > 0).float().mean())
        gaps.append(gap.mean())
    return {
        "L_evaluator_pairwise": torch.stack(losses).mean(),
        "evaluator_pairwise_acc": torch.stack(accs).mean(),
        "evaluator_pairwise_gap": torch.stack(gaps).mean(),
    }


def compute_evaluator_candidate_pairwise_loss(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    real_out: dict,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    train_cfg: dict,
    policy_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Rank proposer candidates by oracle action error.

    Real-vs-corrupted pairwise loss proves the evaluator notices obviously bad
    actions. TTC needs a sharper target: among the K candidates that the proposer
    actually emits, the learned score should prefer the candidate closest to the
    demonstrated action chunk.
    """
    weight = float(train_cfg.get("evaluator_candidate_pairwise_weight", 0.0))
    if weight <= 0 or "proposer_action_cond" not in real_out:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_candidate_pairwise": zero,
            "L_evaluator_candidate_ce": zero,
            "evaluator_candidate_pairwise_acc": zero,
            "evaluator_candidate_pairwise_gap": zero,
            "evaluator_candidate_selected_pose_l1": zero,
            "evaluator_candidate_anchor_pose_l1": zero,
            "evaluator_candidate_oracle_pose_l1": zero,
            "evaluator_candidate_oracle_match": zero,
        }

    candidate_cond = real_out["proposer_action_cond"]
    if candidate_cond.ndim != 4 or candidate_cond.shape[-1] != 7:
        raise ValueError(f"proposer_action_cond must be [B,K,k,7], got {tuple(candidate_cond.shape)}")
    bsz, n_candidates, _horizon, _dim = candidate_cond.shape
    if n_candidates < 2:
        zero = real_out["pred_tokens"].new_zeros(())
        return {
            "L_evaluator_candidate_pairwise": zero,
            "L_evaluator_candidate_ce": zero,
            "evaluator_candidate_pairwise_acc": zero,
            "evaluator_candidate_pairwise_gap": zero,
            "evaluator_candidate_selected_pose_l1": zero,
            "evaluator_candidate_anchor_pose_l1": zero,
            "evaluator_candidate_oracle_pose_l1": zero,
            "evaluator_candidate_oracle_match": zero,
        }

    pose_l1 = (
        candidate_cond[..., :6].float()
        - action_tgt_norm.float()[:, None]
    ).abs().mean(dim=(2, 3))
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_prob = candidate_cond[..., 6].float().clamp(1e-5, 1.0 - 1e-5)
    with torch.autocast(device_type="cuda", enabled=False):
        grip_bce = F.binary_cross_entropy(
            grip_prob.float(),
            grip_tgt[:, None].expand_as(grip_prob).float(),
            reduction="none",
        ).mean(dim=2)
    grip_weight = float(train_cfg.get("evaluator_candidate_grip_weight", 0.1))
    oracle_err = (pose_l1 + grip_weight * grip_bce).detach()
    oracle_idx = oracle_err.argmin(dim=1)

    scores = []
    for ci in range(n_candidates):
        cand_out = _forward_joint_model(
            model,
            s,
            c,
            action_cond=candidate_cond[:, ci].to(device=s.device, dtype=s.dtype),
            context_rgb=None,
            pixel=False,
            bridging=False,
            policy_kwargs=policy_kwargs,
        )
        scores.append(_score_evaluator(cand_out, train_cfg))
    score_t = torch.stack(scores, dim=1)
    best_score = score_t.gather(1, oracle_idx[:, None])
    non_oracle = torch.ones(bsz, n_candidates, dtype=torch.bool, device=score_t.device)
    non_oracle.scatter_(1, oracle_idx[:, None], False)
    other_scores = score_t[non_oracle].reshape(bsz, n_candidates - 1)
    gap = best_score - other_scores
    margin = float(train_cfg.get("evaluator_candidate_pairwise_margin", 0.05))
    ce_temp = float(train_cfg.get("evaluator_candidate_ce_temperature", 0.1))
    ce_temp = max(ce_temp, 1e-4)
    selected_idx = score_t.argmax(dim=1)
    selected_pose_l1 = pose_l1.gather(1, selected_idx[:, None]).squeeze(1)

    return {
        "L_evaluator_candidate_pairwise": torch.relu(margin - gap).mean(),
        "L_evaluator_candidate_ce": F.cross_entropy(score_t.float() / ce_temp, oracle_idx),
        "evaluator_candidate_pairwise_acc": (gap > 0).float().mean(),
        "evaluator_candidate_pairwise_gap": gap.mean(),
        "evaluator_candidate_selected_pose_l1": selected_pose_l1.mean(),
        "evaluator_candidate_anchor_pose_l1": pose_l1[:, 0].mean(),
        "evaluator_candidate_oracle_pose_l1": pose_l1.min(dim=1).values.mean(),
        "evaluator_candidate_oracle_match": (selected_idx == oracle_idx).float().mean(),
    }


def compute_direct_policy_loss(
    out: dict,
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    train_cfg: dict,
) -> dict[str, torch.Tensor]:
    weight = float(train_cfg.get("direct_policy_weight", 0.0))
    head = str(train_cfg.get("direct_policy_head", "policy")).strip().lower()
    if head in ("policy", "direct", "full"):
        pose_key, grip_key, cond_key = "policy_pose_norm", "policy_gripper_logit", "policy_action_cond"
    elif head in ("prior", "prior_policy", "oxe_prior"):
        pose_key, grip_key, cond_key = "prior_policy_pose_norm", "prior_policy_gripper_logit", "prior_policy_action_cond"
    elif head in ("base", "base_policy"):
        pose_key, grip_key, cond_key = "base_policy_pose_norm", "base_policy_gripper_logit", None
    else:
        raise ValueError(f"unknown direct_policy_head={head!r}; expected policy, prior, or base")
    if weight <= 0 or pose_key not in out or grip_key not in out or (cond_key is not None and cond_key not in out):
        zero = action_tgt_norm.new_zeros(())
        return {
            "L_direct_policy": zero,
            "direct_policy_pose_l1": zero,
            "direct_policy_first_pose_l1": zero,
            "direct_policy_grip_acc": zero,
            "direct_policy_grip_transition_acc": zero,
            "direct_policy_grip_transition_rate": zero,
        }
    pose_pred = out[pose_key].float()
    grip_logit = out[grip_key].float()
    horizon = min(pose_pred.shape[1], action_tgt_norm.shape[1], grip_logit.shape[1], action_tgt.shape[1])
    pose_pred = pose_pred[:, :horizon]
    grip_logit = grip_logit[:, :horizon]
    pose_tgt = action_tgt_norm[:, :horizon].float()
    grip_tgt = (action_tgt[:, :horizon, 6] > 0.5).float()
    huber_delta = float(train_cfg.get("direct_policy_huber_delta", train_cfg.get("huber_delta", 1.0)))

    pose_err = F.smooth_l1_loss(pose_pred, pose_tgt, beta=huber_delta, reduction="none").mean(dim=(1, 2))
    first_pose_err = F.smooth_l1_loss(pose_pred[:, 0], pose_tgt[:, 0], beta=huber_delta, reduction="none").mean(dim=1)
    if horizon > 1:
        delta_err = F.smooth_l1_loss(
            pose_pred[:, 1:] - pose_pred[:, :-1],
            pose_tgt[:, 1:] - pose_tgt[:, :-1],
            beta=huber_delta,
            reduction="none",
        ).mean(dim=(1, 2))
    else:
        delta_err = pose_err.new_zeros(pose_err.shape)

    grip_bce = F.binary_cross_entropy_with_logits(grip_logit, grip_tgt, reduction="none")
    prev_tgt = torch.cat([grip_tgt[:, :1], grip_tgt[:, :-1]], dim=1)
    grip_transition = (grip_tgt != prev_tgt).float()
    transition_weight = float(train_cfg.get("direct_policy_grip_transition_weight", 0.0))
    if transition_weight > 0:
        step_weight = 1.0 + transition_weight * grip_transition
        grip_err = (grip_bce * step_weight).sum(dim=1) / step_weight.sum(dim=1).clamp_min(1e-6)
    else:
        grip_err = grip_bce.mean(dim=1)
    first_grip_err = F.binary_cross_entropy_with_logits(grip_logit[:, 0], grip_tgt[:, 0], reduction="none")

    L_direct = (
        float(train_cfg.get("direct_policy_pose_weight", 1.0)) * pose_err.mean()
        + float(train_cfg.get("direct_policy_first_pose_weight", 2.0)) * first_pose_err.mean()
        + float(train_cfg.get("direct_policy_delta_weight", 0.2)) * delta_err.mean()
        + float(train_cfg.get("direct_policy_grip_weight", 0.3)) * grip_err.mean()
        + float(train_cfg.get("direct_policy_first_grip_weight", 0.5)) * first_grip_err.mean()
    )
    grip_prob = torch.sigmoid(grip_logit)
    grip_match = ((grip_prob > 0.5) == (grip_tgt > 0.5)).float()
    transition_mask = grip_transition > 0.5
    transition_acc = grip_match[transition_mask].mean() if bool(transition_mask.any().detach().cpu()) else grip_prob.new_zeros(())
    return {
        "L_direct_policy": L_direct,
        "direct_policy_pose_l1": (pose_pred - pose_tgt).abs().mean(),
        "direct_policy_first_pose_l1": (pose_pred[:, 0] - pose_tgt[:, 0]).abs().mean(),
        "direct_policy_grip_acc": grip_match.mean(),
        "direct_policy_grip_transition_acc": transition_acc,
        "direct_policy_grip_transition_rate": grip_transition.mean(),
    }


def build_hunyuan_latent_adapter(cfg: dict, device: torch.device) -> HunyuanLatentAdapter:
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    adapter_cfg = HunyuanLatentAdapterConfig(
        token_dim=model_cfg["state"]["D"],
        token_grid=int(model_cfg["state"]["P"] ** 0.5),
        hidden=int(train_cfg.get("hunyuan_adapter_hidden", 192)),
        latent_channels=int(train_cfg.get("hunyuan_latent_channels", 16)),
        latent_time=int(train_cfg.get("hunyuan_latent_time", 3)),
        latent_hw=int(train_cfg.get("hunyuan_latent_hw", 32)),
        action_dim=int(train_cfg.get("hunyuan_action_dim", 7)),
        task_dim=int(train_cfg.get("hunyuan_task_dim", model_cfg["state"].get("cond_dim", 2048))),
        n_blocks=int(train_cfg.get("hunyuan_adapter_blocks", 4)),
        use_motion=bool(train_cfg.get("hunyuan_use_motion", True)),
        use_rough_rgb=bool(train_cfg.get("hunyuan_use_rough_rgb", True)),
        use_context=bool(train_cfg.get("hunyuan_use_context", True)),
        use_action=bool(train_cfg.get("hunyuan_use_action", True)),
        use_task=bool(train_cfg.get("hunyuan_use_task", True)),
    )
    adapter = HunyuanLatentAdapter(adapter_cfg).to(device)
    if bool(train_cfg.get("hunyuan_zero_init_output", False)):
        adapter.zero_init_output()
    return adapter


def load_hunyuan_vae(train_cfg: dict, device: torch.device):
    model_base = Path(train_cfg.get("hunyuan_model_base", "/data/Minko/models/hunyuan_video"))
    repo = Path(train_cfg.get("hunyuan_repo", "/data/Minko/external/HunyuanVideo"))
    os.environ.setdefault("MODEL_BASE", str(model_base))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from hyvideo.vae import load_vae  # type: ignore

    vae, _, _, _ = load_vae(
        "884-16c-hy",
        train_cfg.get("hunyuan_vae_precision", "fp16"),
        device=device,
    )
    vae.requires_grad_(False)
    vae.eval()
    return vae


def target_video_from_batch(context_rgb: torch.Tensor, rgb_tgt_p: torch.Tensor) -> torch.Tensor:
    video = torch.cat([context_rgb[:, None], rgb_tgt_p], dim=1)
    return video.permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def encode_hunyuan_latents(vae, video_bcthw: torch.Tensor) -> torch.Tensor:
    x = video_bcthw.mul(2.0).sub(1.0)
    posterior = vae.encode(x.to(dtype=vae.dtype)).latent_dist
    latents = posterior.mode()
    return latents * float(vae.config.scaling_factor)


def _hunyuan_zero_losses(zero: torch.Tensor, prefix: str = "") -> dict[str, torch.Tensor]:
    head = f"{prefix}_" if prefix else ""
    return {
        f"L_{head}hunyuan_latent": zero,
        f"{head}hunyuan_latent_mse": zero,
        f"{head}hunyuan_latent_l1": zero,
        f"{head}hunyuan_latent_temporal_mse": zero,
        f"{head}hunyuan_latent_motion_l1": zero,
    }


def _compute_hunyuan_controls_loss(
    adapter: HunyuanLatentAdapter,
    vae,
    *,
    tokens: torch.Tensor,
    depth: torch.Tensor,
    target_latents: torch.Tensor,
    context_rgb: torch.Tensor,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
    rough_rgb: torch.Tensor | None = None,
    motion_hint: torch.Tensor | None = None,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    if bool(train_cfg.get("hunyuan_detach_world", False)):
        tokens = tokens.detach()
        depth = depth.detach()
        rough_rgb = rough_rgb.detach() if rough_rgb is not None else None

    delta_latents = adapter(
        tokens,
        depth,
        context_rgb=context_rgb,
        motion_hint=motion_hint,
        rough_rgb=rough_rgb,
        action_cond=action_cond,
        task_emb=task_emb,
        target_latents=target_latents,
    )
    pred_latents = delta_latents
    if bool(train_cfg.get("hunyuan_residual_from_rough", False)) and rough_rgb is not None:
        rough_video = target_video_from_batch(context_rgb, rough_rgb.float())
        rough_latents = encode_hunyuan_latents(vae, rough_video)
        pred_latents = rough_latents.to(dtype=delta_latents.dtype) + float(
            train_cfg.get("hunyuan_residual_scale", 1.0)
        ) * delta_latents

    zero = tokens.new_zeros(())
    latent_mse = F.mse_loss(pred_latents.float(), target_latents.float())
    latent_l1 = F.l1_loss(pred_latents.float(), target_latents.float())
    if pred_latents.shape[2] > 1 and target_latents.shape[2] > 1:
        pred_dt = pred_latents.float()[:, :, 1:] - pred_latents.float()[:, :, :-1]
        target_dt = target_latents.float()[:, :, 1:] - target_latents.float()[:, :, :-1]
        temporal_mse = F.mse_loss(pred_dt, target_dt)
        motion_l1 = F.l1_loss(pred_dt.abs(), target_dt.abs())
    else:
        temporal_mse = zero
        motion_l1 = zero
    loss = (
        float(train_cfg.get("hunyuan_latent_mse_weight", 1.0)) * latent_mse
        + float(train_cfg.get("hunyuan_latent_l1_weight", 0.05)) * latent_l1
        + float(train_cfg.get("hunyuan_latent_temporal_weight", 0.0)) * temporal_mse
        + float(train_cfg.get("hunyuan_latent_motion_weight", 0.0)) * motion_l1
    )
    head = f"{prefix}_" if prefix else ""
    return {
        f"L_{head}hunyuan_latent": loss,
        f"{head}hunyuan_latent_mse": latent_mse.detach(),
        f"{head}hunyuan_latent_l1": latent_l1.detach(),
        f"{head}hunyuan_latent_temporal_mse": temporal_mse.detach(),
        f"{head}hunyuan_latent_motion_l1": motion_l1.detach(),
    }


def compute_hunyuan_latent_loss(
    adapter: HunyuanLatentAdapter,
    vae,
    out: dict,
    tgt: dict,
    context_rgb: torch.Tensor | None,
    action_cond: torch.Tensor,
    task_emb: torch.Tensor,
    train_cfg: dict,
) -> dict[str, torch.Tensor]:
    zero = out["pred_tokens"].new_zeros(())
    losses = _hunyuan_zero_losses(zero)
    losses.update(_hunyuan_zero_losses(zero, prefix="prior"))
    if context_rgb is None or "rgb_tgt_p" not in tgt:
        return losses

    target_video = target_video_from_batch(context_rgb, tgt["rgb_tgt_p"])
    target_latents = encode_hunyuan_latents(vae, target_video)

    rough_rgb = out.get("rgb") if bool(train_cfg.get("hunyuan_use_rough_rgb", True)) else None
    losses.update(_compute_hunyuan_controls_loss(
        adapter,
        vae,
        tokens=out["pred_tokens"],
        depth=out["depth"],
        target_latents=target_latents,
        context_rgb=context_rgb,
        action_cond=action_cond,
        task_emb=task_emb,
        train_cfg=train_cfg,
        rough_rgb=rough_rgb,
        motion_hint=out.get("motion_hint"),
    ))

    prior_enabled = bool(train_cfg.get("enable_prior_hunyuan_latent_loss", False))
    prior_weight = float(train_cfg.get("prior_hunyuan_latent_weight", 0.0))
    if prior_enabled and prior_weight > 0 and "prior_hunyuan_tokens" in out and "prior_hunyuan_depth" in out:
        prior_rough = out.get("prior_rgb") if bool(train_cfg.get("prior_hunyuan_use_rough_rgb", train_cfg.get("hunyuan_use_rough_rgb", True))) else None
        losses.update(_compute_hunyuan_controls_loss(
            adapter,
            vae,
            tokens=out["prior_hunyuan_tokens"],
            depth=out["prior_hunyuan_depth"],
            target_latents=target_latents,
            context_rgb=context_rgb,
            action_cond=action_cond,
            task_emb=task_emb,
            train_cfg=train_cfg,
            rough_rgb=prior_rough,
            motion_hint=None,
            prefix="prior",
        ))
    return losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--disable_pixel_until", type=int, default=0,
                    help="train first N epochs without L_rgb (stage 1)")
    ap.add_argument(
        "--no_pixel",
        action="store_true",
        help="never activate RGB/video renderer or RGB/LPIPS losses; train latent/action/evaluator heads only",
    )
    ap.add_argument("--reset_optim", action="store_true",
                    help="on --resume, load model weights only; recreate fresh optimizer/scheduler/step")
    ap.add_argument("--print_every", type=int, default=0,
                    help="print stdout step log every N steps (0=off)")
    ap.add_argument("--strict_resume", action="store_true",
                    help="strict state_dict load on resume (default: allow mismatches)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")

    overfit_ids = cfg.get("overfit_clip_ids") if args.overfit else None
    tr_ds, val_ds = build_datasets(cfg, overfit_ids=overfit_ids)
    train_cfg = cfg["train"]
    bs = train_cfg["batch_size_per_gpu"]; nw = train_cfg["num_workers"]
    weighted_sampler_cfg = train_cfg.get("weighted_sampler") or cfg.get("weighted_sampler")
    sample_weights = build_dataset_sample_weights(tr_ds, weighted_sampler_cfg)
    if world > 1:
        if sample_weights is not None:
            sampler_num_samples = None
            if isinstance(weighted_sampler_cfg, dict) and weighted_sampler_cfg.get("num_samples_per_rank") is not None:
                sampler_num_samples = int(weighted_sampler_cfg["num_samples_per_rank"])
            tr_s = WeightedDistributedSampler(
                sample_weights,
                num_replicas=world,
                rank=rank,
                replacement=bool((weighted_sampler_cfg or {}).get("replacement", True)) if isinstance(weighted_sampler_cfg, dict) else True,
                num_samples=sampler_num_samples,
                seed=int((weighted_sampler_cfg or {}).get("seed", cfg["data"].get("seed", 0))) if isinstance(weighted_sampler_cfg, dict) else int(cfg["data"].get("seed", 0)),
            )
        else:
            tr_s = DistributedSampler(tr_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s, num_workers=nw,
                                pin_memory=True, drop_last=True)
        val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_s, num_workers=nw, pin_memory=True)
    else:
        if sample_weights is not None:
            num_samples = int(weighted_sampler_cfg.get("num_samples", len(sample_weights))) if isinstance(weighted_sampler_cfg, dict) else len(sample_weights)
            tr_s = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=bool(weighted_sampler_cfg.get("replacement", True)) if isinstance(weighted_sampler_cfg, dict) else True)
            tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s, num_workers=nw, pin_memory=True, drop_last=True)
        else:
            tr_s = None
            tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    weights = LossWeights(**cfg["loss"])
    direct_policy_only = bool(train_cfg.get("direct_policy_only", False))
    hunyuan_training_enabled = (
        bool(train_cfg.get("enable_hunyuan_latent_loss", False))
        and float(train_cfg.get("hunyuan_latent_weight", 0.0)) > 0
        and not args.no_pixel
        and not direct_policy_only
    )
    pixel_training_enabled = (
        bool(cfg["model"].get("enable_pixel", True))
        and (
            bool(train_cfg.get("enable_pixel_loss", True))
            or (hunyuan_training_enabled and bool(train_cfg.get("hunyuan_use_rough_rgb", True)))
        )
        and not args.no_pixel
        and not direct_policy_only
    )
    if int(args.disable_pixel_until) > 0 and world > 1 and not bool(train_cfg.get("manual_grad_sync", direct_policy_only)) and not bool(train_cfg.get("find_unused_parameters", False)):
        raise RuntimeError(
            "--disable_pixel_until with DDP find_unused_parameters=false can leave pixel/context-pixel "
            "parameters unused during early epochs. Set find_unused_parameters=true, use --no_pixel, "
            "manual_grad_sync=true, or do not disable pixel epochs."
        )
    model = build_model(cfg).to(device)
    if not pixel_training_enabled:
        for module_name in ("pixel", "context_pixel"):
            module = getattr(model, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
    apply_trainable_filter(model, cfg["train"], rank)
    hunyuan_adapter = None
    hunyuan_vae = None
    if hunyuan_training_enabled:
        hunyuan_adapter = build_hunyuan_latent_adapter(cfg, device)
        hunyuan_vae = load_hunyuan_vae(train_cfg, device)
        if rank == 0:
            h_params = sum(p.numel() for p in hunyuan_adapter.parameters() if p.requires_grad)
            print(f"[rank0] HunyuanLatentAdapter: {h_params/1e6:.2f}M", flush=True)

    if rank == 0:
        n_p = model.num_trainable_params()
        print(f"[rank0] JointWorldModel: {n_p/1e6:.1f}M; train_windows={len(tr_ds)} val_windows={len(val_ds)}")
    manual_grad_sync = bool(train_cfg.get("manual_grad_sync", direct_policy_only))
    use_ddp_wrapper = world > 1 and not manual_grad_sync
    if use_ddp_wrapper:
        model = DDP(
            model,
            device_ids=[local],
            find_unused_parameters=train_cfg.get("find_unused_parameters", False),
        )
        if hunyuan_adapter is not None:
            hunyuan_adapter = DDP(
                hunyuan_adapter,
                device_ids=[local],
                find_unused_parameters=train_cfg.get("find_unused_parameters", False),
            )

    lpips_fn = None
    if pixel_training_enabled and weights.rgb_lpips > 0:
        import lpips

        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
    if rank == 0:
        print(
            f"[rank0] pixel_training_enabled={pixel_training_enabled} "
            f"hunyuan_training_enabled={hunyuan_training_enabled} "
            f"lpips_active={lpips_fn is not None} disable_pixel_until={args.disable_pixel_until} "
            f"direct_policy_only={direct_policy_only} manual_grad_sync={manual_grad_sync}",
            flush=True,
        )

    optimizer_settings = resolve_optimizer_settings(cfg)
    train_cfg_for_opt = dict(train_cfg)
    train_cfg_for_opt["lr"] = optimizer_settings.peak_lr
    train_cfg_for_opt["weight_decay"] = optimizer_settings.weight_decay
    optimizer_param_groups = build_optimizer_param_groups(model, hunyuan_adapter, train_cfg_for_opt, rank)
    trainable_params = [
        param
        for group in optimizer_param_groups
        for param in group["params"]
    ]
    opt = torch.optim.AdamW(
        optimizer_param_groups,
        betas=optimizer_settings.betas,
    )
    grad_clip_value = float(optimizer_settings.grad_clip if optimizer_settings.grad_clip is not None else train_cfg["grad_clip"])
    max_steps = int(train_cfg.get("max_steps", 0) or 0)
    total_steps = max(1, len(tr_loader) * train_cfg["epochs"])
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    sched = build_lr_scheduler(opt, cfg, total_steps)
    if rank == 0:
        lr_schedule_type = str((cfg.get("lr_schedule") or train_cfg.get("lr_schedule") or {}).get("type", train_cfg.get("lr_schedule_type", "cosine"))).lower()
        print(
            f"[rank0] optimizer={optimizer_settings.type} betas={optimizer_settings.betas} "
            f"peak_lr={optimizer_settings.peak_lr:.3g} min_lr={optimizer_settings.min_lr:.3g} "
            f"weight_decay={optimizer_settings.weight_decay:.3g} grad_clip={grad_clip_value:.3g} "
            f"lr_schedule={lr_schedule_type} total_steps={total_steps}",
            flush=True,
        )
    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"]["ckpt_dir"]
    if rank == 0:
        (out_root / cfg["out"]["tb_dir"]).mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_root / cfg["out"]["tb_dir"])

    start_epoch = 0; step = 0; best_val = float("inf")
    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        target = model.module if isinstance(model, DDP) else model
        load_res = load_compatible_state_dict(target, sd["model"], strict=args.strict_resume)
        if hunyuan_adapter is not None and "hunyuan_adapter" in sd:
            adapter_target = hunyuan_adapter.module if isinstance(hunyuan_adapter, DDP) else hunyuan_adapter
            adapter_load_res = load_compatible_state_dict(
                adapter_target,
                sd["hunyuan_adapter"],
                strict=args.strict_resume,
            )
        else:
            adapter_load_res = None
        if args.reset_optim:
            if rank == 0:
                miss = len(getattr(load_res, "missing_keys", []) or [])
                un = len(getattr(load_res, "unexpected_keys", []) or [])
                skip = len(getattr(load_res, "skipped_keys", []) or [])
                print(f"[rank0] resumed weights only from {args.resume} (optim RESET) — "
                      f"missing={miss} unexpected={un} skipped={skip}")
                if adapter_load_res is not None:
                    amiss = len(getattr(adapter_load_res, "missing_keys", []) or [])
                    aun = len(getattr(adapter_load_res, "unexpected_keys", []) or [])
                    askip = len(getattr(adapter_load_res, "skipped_keys", []) or [])
                    print(f"[rank0] resumed hunyuan adapter from checkpoint — "
                          f"missing={amiss} unexpected={aun} skipped={askip}")
        else:
            opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
            start_epoch = sd["epoch"] + 1; step = sd["step"]; best_val = sd.get("best_val", best_val)
            if rank == 0:
                print(f"[rank0] resumed from {args.resume} at epoch {start_epoch}")
    load_action_stats_if_available(model, cfg, rank, device)

    k = cfg["data"]["k"]
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if max_steps > 0 and step >= max_steps:
            break
        if world > 1 and tr_s is not None:
            tr_s.set_epoch(epoch)
        do_pixel = pixel_training_enabled and epoch >= args.disable_pixel_until
        model.train()
        for batch in tr_loader:
            if max_steps > 0 and step >= max_steps:
                break
            s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k, direct_policy_only=direct_policy_only)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if direct_policy_only:
                    target_model = model.module if isinstance(model, DDP) else model
                    if target_model.action_policy is None:
                        raise RuntimeError("direct_policy_only requires enable_action_policy")
                    out = target_model.action_policy(s, task_emb=c, **action_policy_kwargs_from_targets(tgt))
                    losses = {"L_total": out["policy_pose_norm"].new_zeros(())}
                    zero = out["policy_pose_norm"].new_zeros(())
                    rank_losses = {
                        "L_evaluator_pairwise": zero,
                        "evaluator_pairwise_acc": zero,
                        "evaluator_pairwise_gap": zero,
                        "evaluator_real_progress": zero,
                        "evaluator_variant_progress": zero,
                    }
                    candidate_losses = {
                        "L_evaluator_candidate_pairwise": zero,
                        "L_evaluator_candidate_ce": zero,
                        "evaluator_candidate_pairwise_acc": zero,
                        "evaluator_candidate_pairwise_gap": zero,
                        "evaluator_candidate_selected_pose_l1": zero,
                        "evaluator_candidate_anchor_pose_l1": zero,
                        "evaluator_candidate_oracle_pose_l1": zero,
                        "evaluator_candidate_oracle_match": zero,
                    }
                else:
                    s_cond, c_cond, action_cond_model, context_rgb_cond, dropout_losses = apply_condition_dropout(
                        s, c, action_cond, context_rgb, train_cfg, training=True
                    )
                    prior_clean_tokens = prior_clean_tokens_from_targets(tgt)
                    policy_kwargs = action_policy_kwargs_from_targets(tgt)
                    out = _forward_joint_model(
                        model,
                        s_cond,
                        c_cond,
                        action_cond=action_cond_model,
                        context_rgb=context_rgb_cond,
                        prior_clean_tokens=prior_clean_tokens,
                        pixel=do_pixel,
                        bridging=False,
                        policy_kwargs=policy_kwargs,
                    )
                    losses = compute_losses(out, tgt, weights, lpips_fn if do_pixel else None)
                    losses.update({k: v.detach() for k, v in dropout_losses.items()})
                    rank_losses = compute_evaluator_pairwise_loss(
                        model,
                        s_cond,
                        c_cond,
                        out,
                        action_cond_model,
                        cfg["train"],
                        step=step,
                        policy_kwargs=policy_kwargs,
                    )
                    candidate_losses = compute_evaluator_candidate_pairwise_loss(
                        model,
                        s,
                        c,
                        out,
                        tgt["action_tgt"],
                        tgt["action_tgt_norm"],
                        cfg["train"],
                        policy_kwargs=policy_kwargs,
                    )
                direct_losses = compute_direct_policy_loss(
                    out,
                    tgt["action_tgt"],
                    tgt["action_tgt_norm"],
                    cfg["train"],
                )
                hunyuan_losses = {}
                if hunyuan_training_enabled and hunyuan_adapter is not None and hunyuan_vae is not None:
                    hunyuan_losses = compute_hunyuan_latent_loss(
                        hunyuan_adapter,
                        hunyuan_vae,
                        out,
                        tgt,
                        context_rgb_cond if 'context_rgb_cond' in locals() else context_rgb,
                        action_cond_model if 'action_cond_model' in locals() else action_cond,
                        c_cond if 'c_cond' in locals() else c,
                        train_cfg,
                    )
                if cfg["train"].get("evaluator_pairwise_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["evaluator_pairwise_weight"]) * rank_losses["L_evaluator_pairwise"]
                    )
                if cfg["train"].get("evaluator_candidate_pairwise_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["evaluator_candidate_pairwise_weight"])
                        * candidate_losses["L_evaluator_candidate_pairwise"]
                    )
                if cfg["train"].get("evaluator_candidate_ce_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["evaluator_candidate_ce_weight"])
                        * candidate_losses["L_evaluator_candidate_ce"]
                    )
                if cfg["train"].get("direct_policy_weight", 0.0):
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(cfg["train"]["direct_policy_weight"]) * direct_losses["L_direct_policy"]
                    )
                if train_cfg.get("hunyuan_latent_weight", 0.0) and hunyuan_losses:
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(train_cfg["hunyuan_latent_weight"]) * hunyuan_losses["L_hunyuan_latent"]
                    )
                if train_cfg.get("prior_hunyuan_latent_weight", 0.0) and hunyuan_losses:
                    losses["L_total"] = (
                        losses["L_total"]
                        + float(train_cfg["prior_hunyuan_latent_weight"])
                        * hunyuan_losses["L_prior_hunyuan_latent"]
                    )
                losses.update({k: v.detach() for k, v in rank_losses.items()})
                losses.update({k: v.detach() for k, v in candidate_losses.items()})
                losses.update({k: v.detach() for k, v in direct_losses.items()})
                losses.update({k: v.detach() for k, v in hunyuan_losses.items()})
            loss = losses["L_total"]
            finite_count = _distributed_finite_count(torch.isfinite(loss), device, world)
            if finite_count != world:
                if rank == 0:
                    print(
                        f"[rank0] WARN skip non-finite loss at step {step} "
                        f"finite_ranks={finite_count}/{world}",
                        flush=True,
                    )
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if world > 1 and manual_grad_sync:
                _all_reduce_gradients(model, world)
                if hunyuan_adapter is not None:
                    _all_reduce_gradients(hunyuan_adapter, world)
            gn = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_value)
            grad_finite_count = _distributed_finite_count(torch.isfinite(gn), device, world)
            if grad_finite_count != world:
                if rank == 0:
                    print(
                        f"[rank0] WARN skip non-finite grad_norm at step {step} "
                        f"finite_ranks={grad_finite_count}/{world}",
                        flush=True,
                    )
                opt.zero_grad(set_to_none=True); sched.step(); step += 1
                continue
            opt.step(); sched.step()
            if rank == 0 and args.print_every and step % args.print_every == 0:
                rgb_l1 = float(losses.get("L_rgb_l1", torch.tensor(0.)).detach().float())
                lpv = float(losses.get("L_rgb_lpips", torch.tensor(0.)).detach().float())
                prop = float(losses.get("L_proposer_pose", torch.tensor(0.)).detach().float())
                prog = float(losses.get("L_progress", torch.tensor(0.)).detach().float())
                pair = float(losses.get("L_evaluator_pairwise", torch.tensor(0.)).detach().float())
                pair_acc = float(losses.get("evaluator_pairwise_acc", torch.tensor(0.)).detach().float())
                cand = float(losses.get("L_evaluator_candidate_pairwise", torch.tensor(0.)).detach().float())
                cand_ce = float(losses.get("L_evaluator_candidate_ce", torch.tensor(0.)).detach().float())
                cand_acc = float(losses.get("evaluator_candidate_pairwise_acc", torch.tensor(0.)).detach().float())
                cand_sel = float(losses.get("evaluator_candidate_selected_pose_l1", torch.tensor(0.)).detach().float())
                cand_anchor = float(losses.get("evaluator_candidate_anchor_pose_l1", torch.tensor(0.)).detach().float())
                direct = float(losses.get("L_direct_policy", torch.tensor(0.)).detach().float())
                direct_pose = float(losses.get("direct_policy_pose_l1", torch.tensor(0.)).detach().float())
                direct_grip = float(losses.get("direct_policy_grip_acc", torch.tensor(0.)).detach().float())
                hunyuan = float(losses.get("L_hunyuan_latent", torch.tensor(0.)).detach().float())
                prior_hunyuan = float(losses.get("L_prior_hunyuan_latent", torch.tensor(0.)).detach().float())
                prior = float(losses.get("L_world_prior", torch.tensor(0.)).detach().float())
                prior_flow = float(losses.get("L_world_prior_flow", torch.tensor(0.)).detach().float())
                prior_depth = float(losses.get("L_world_prior_depth", torch.tensor(0.)).detach().float())
                drop_a = float(losses.get("drop_action_frac", torch.tensor(0.)).detach().float())
                drop_c = float(losses.get("drop_context_frac", torch.tensor(0.)).detach().float())
                drop_text = float(losses.get("drop_text_only_frac", torch.tensor(0.)).detach().float())
                h_mse = float(losses.get("hunyuan_latent_mse", torch.tensor(0.)).detach().float())
                h_tmp = float(losses.get("hunyuan_latent_temporal_mse", torch.tensor(0.)).detach().float())
                depth = float(losses.get("L_depth", torch.tensor(0.)).detach().float())
                depth_ch = float(losses.get("L_depth_change", torch.tensor(0.)).detach().float())
                depth_mo = float(losses.get("L_depth_motion_l1", torch.tensor(0.)).detach().float())
                print(f"[rank0] step {step} (ep {epoch}) L_total={float(loss.detach().float()):.4f} "
                      f"rgb_L1={rgb_l1:.4f} lpips={lpv:.4f} "
                      f"depth={depth:.4f} depth_ch={depth_ch:.4f} depth_mo={depth_mo:.4f} "
                      f"hunyuan={hunyuan:.4f} prior_hunyuan={prior_hunyuan:.4f} h_mse={h_mse:.4f} h_tmp={h_tmp:.4f} "
                      f"prior={prior:.4f} prior_flow={prior_flow:.4f} prior_depth={prior_depth:.4f} "
                      f"drop_a={drop_a:.2f} drop_ctx={drop_c:.2f} drop_text={drop_text:.2f} "
                      f"prop={prop:.4f} progress={prog:.4f} "
                      f"pair={pair:.4f} pair_acc={pair_acc:.3f} "
                      f"cand={cand:.4f} cand_ce={cand_ce:.4f} cand_acc={cand_acc:.3f} "
                      f"cand_sel={cand_sel:.3f} cand_anchor={cand_anchor:.3f} "
                      f"direct={direct:.4f} direct_pose={direct_pose:.3f} direct_grip={direct_grip:.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                for k_, v in losses.items():
                    tb.add_scalar(f"train/{k_}", float(v.detach().float()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
                tb.add_scalar("stage_pixel", float(do_pixel), step)
            ckpt_every_steps = int(cfg["train"].get("ckpt_every_steps", 0) or 0)
            if rank == 0 and ckpt_every_steps > 0 and (step + 1) % ckpt_every_steps == 0:
                step_ckpt = {
                    "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "epoch": epoch,
                    "step": step + 1,
                    "val_total": None,
                    "best_val": best_val,
                    "cfg": cfg,
                }
                if hunyuan_adapter is not None:
                    adapter_target = hunyuan_adapter.module if isinstance(hunyuan_adapter, DDP) else hunyuan_adapter
                    step_ckpt["hunyuan_adapter"] = adapter_target.state_dict()
                    step_ckpt["hunyuan_adapter_cfg"] = adapter_target.cfg.__dict__
                step_path = ckpt_dir / f"step_{step + 1:08d}.pt"
                tmp_path = ckpt_dir / f".{step_path.name}.tmp"
                torch.save(step_ckpt, tmp_path)
                os.replace(tmp_path, step_path)
                latest_tmp = ckpt_dir / ".latest.pt.tmp"
                torch.save(step_ckpt, latest_tmp)
                os.replace(latest_tmp, ckpt_dir / "latest.pt")
            step += 1
            if max_steps > 0 and step >= max_steps:
                if rank == 0:
                    print(f"[rank0] reached max_steps={max_steps}; running validation/checkpoint", flush=True)
                break
        # Val
        model.eval()
        agg = {}; nb = 0
        max_val_batches = int(train_cfg.get("max_val_batches", 0))
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if max_val_batches > 0 and bi >= max_val_batches:
                    break
                s, c, action_cond, context_rgb, tgt = batch_to_device(batch, device, k, direct_policy_only=direct_policy_only)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if direct_policy_only:
                        target_model = model.module if isinstance(model, DDP) else model
                        if target_model.action_policy is None:
                            raise RuntimeError("direct_policy_only requires enable_action_policy")
                        out = target_model.action_policy(s, task_emb=c, **action_policy_kwargs_from_targets(tgt))
                        losses = {"L_total": out["policy_pose_norm"].new_zeros(())}
                    else:
                        prior_clean_tokens = prior_clean_tokens_from_targets(tgt)
                        policy_kwargs = action_policy_kwargs_from_targets(tgt)
                        out = _forward_joint_model(
                            model,
                            s,
                            c,
                            action_cond=action_cond,
                            context_rgb=context_rgb,
                            prior_clean_tokens=prior_clean_tokens,
                            pixel=do_pixel,
                            bridging=False,
                            policy_kwargs=policy_kwargs,
                        )
                        losses = compute_losses(out, tgt, weights, lpips_fn if do_pixel else None)
                    direct_losses = compute_direct_policy_loss(
                        out,
                        tgt["action_tgt"],
                        tgt["action_tgt_norm"],
                        cfg["train"],
                    )
                    hunyuan_losses = {}
                    if hunyuan_training_enabled and hunyuan_adapter is not None and hunyuan_vae is not None:
                        hunyuan_losses = compute_hunyuan_latent_loss(
                            hunyuan_adapter,
                            hunyuan_vae,
                            out,
                            tgt,
                            context_rgb,
                            action_cond,
                            c,
                            train_cfg,
                        )
                    if cfg["train"].get("direct_policy_weight", 0.0):
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(cfg["train"]["direct_policy_weight"]) * direct_losses["L_direct_policy"]
                        )
                    if train_cfg.get("hunyuan_latent_weight", 0.0) and hunyuan_losses:
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(train_cfg["hunyuan_latent_weight"]) * hunyuan_losses["L_hunyuan_latent"]
                        )
                    if train_cfg.get("prior_hunyuan_latent_weight", 0.0) and hunyuan_losses:
                        losses["L_total"] = (
                            losses["L_total"]
                            + float(train_cfg["prior_hunyuan_latent_weight"])
                            * hunyuan_losses["L_prior_hunyuan_latent"]
                        )
                    losses.update({k: v.detach() for k, v in direct_losses.items()})
                    losses.update({k: v.detach() for k, v in hunyuan_losses.items()})
                for kk, v in losses.items():
                    agg[kk] = agg.get(kk, 0.0) + float(v.detach().float())
                nb += 1
        if world > 1:
            keys = sorted(agg.keys())
            reduce_device = "cpu" if dist.get_backend() == "gloo" else device
            v = torch.tensor([agg[kk] for kk in keys] + [float(nb)], device=reduce_device)
            dist.all_reduce(v)
            tot_nb = float(v[-1].item())
            for i, kk in enumerate(keys):
                agg[kk] = float(v[i].item()) / max(1.0, tot_nb)
            nb = 1
        if rank == 0:
            for kk, vv in agg.items():
                tb.add_scalar(f"val/{kk}", vv / max(1, nb), step)
            val_total = agg["L_total"] / max(1, nb)
            ckpt = {"model": (model.module if isinstance(model, DDP) else model).state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "val_total": val_total,
                    "best_val": best_val, "cfg": cfg}
            if hunyuan_adapter is not None:
                adapter_target = hunyuan_adapter.module if isinstance(hunyuan_adapter, DDP) else hunyuan_adapter
                ckpt["hunyuan_adapter"] = adapter_target.state_dict()
                ckpt["hunyuan_adapter_cfg"] = adapter_target.cfg.__dict__
            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0:
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_total < best_val:
                best_val = val_total
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
            val_direct = agg.get("L_direct_policy", 0.0) / max(1, nb)
            val_direct_pose = agg.get("direct_policy_pose_l1", 0.0) / max(1, nb)
            val_direct_first_pose = agg.get("direct_policy_first_pose_l1", 0.0) / max(1, nb)
            val_direct_grip = agg.get("direct_policy_grip_acc", 0.0) / max(1, nb)
            val_direct_grip_transition = agg.get("direct_policy_grip_transition_acc", 0.0) / max(1, nb)
            val_hunyuan = agg.get("L_hunyuan_latent", 0.0) / max(1, nb)
            val_prior_hunyuan = agg.get("L_prior_hunyuan_latent", 0.0) / max(1, nb)
            val_hunyuan_mse = agg.get("hunyuan_latent_mse", 0.0) / max(1, nb)
            print(
                f"[rank0] epoch {epoch}: val_total {val_total:.4f} "
                f"val_direct={val_direct:.4f} "
                f"val_direct_pose_l1={val_direct_pose:.4f} "
                f"val_direct_first_pose_l1={val_direct_first_pose:.4f} "
                f"val_direct_grip_acc={val_direct_grip:.3f} "
                f"val_direct_grip_transition_acc={val_direct_grip_transition:.3f} "
                f"val_hunyuan={val_hunyuan:.4f} "
                f"val_prior_hunyuan={val_prior_hunyuan:.4f} "
                f"val_hunyuan_mse={val_hunyuan_mse:.4f} "
                f"(best {best_val:.4f}, pixel={do_pixel})"
            )
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
