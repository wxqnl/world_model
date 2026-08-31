"""Small LIBERO success/proposer fine-tune from cached expert windows."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import h5py
from torch.utils.data import DataLoader, Dataset, default_collate, random_split

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.eval.run_eval import build_model


DEFAULT_PLAN_STATE_DIM = 8


def _is_task1_put_cream_butter(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key, "")) for key in ("task_name", "instruction")).lower()
    return ("cream_cheese" in text or "cream cheese" in text) and "butter" in text and "basket" in text


def _task1_plan_stage_from_progress(progress: float) -> int:
    p = min(1.0, max(0.0, float(progress)))
    if p < 0.23:
        return 0
    if p < 0.52:
        return 1
    if p < 0.77:
        return 2
    return 3


def _task1_plan_state_from_stage(stage: int, plan_state_dim: int = DEFAULT_PLAN_STATE_DIM) -> np.ndarray:
    """Encode task1 as [stage4, target2, subgoal2].

    Stages are pick/place cream, then pick/place butter. The boundaries are a
    deliberately simple expert-progress prior for the P0 gate; online rollout
    uses an object-state tracker instead of these fixed thresholds.
    """
    if plan_state_dim < DEFAULT_PLAN_STATE_DIM:
        raise ValueError(f"plan_state_dim must be >= {DEFAULT_PLAN_STATE_DIM}, got {plan_state_dim}")
    stage = min(3, max(0, int(stage)))
    target = 0 if stage < 2 else 1
    subgoal = 0 if stage in (0, 2) else 1
    out = np.zeros(int(plan_state_dim), dtype=np.float32)
    out[stage] = 1.0
    out[4 + target] = 1.0
    out[6 + subgoal] = 1.0
    return out


def _resize_plan_state(arr: np.ndarray, plan_state_dim: int) -> np.ndarray:
    out = np.zeros(int(plan_state_dim), dtype=np.float32)
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = min(len(out), len(flat))
    if n > 0:
        out[:n] = flat[:n]
    return out


def _infer_plan_state(row: dict[str, Any], data: np.lib.npyio.NpzFile, progress: float, plan_state_dim: int) -> torch.Tensor:
    if "plan_state" in data and np.asarray(data["plan_state"]).size > 0:
        arr = _resize_plan_state(np.asarray(data["plan_state"], dtype=np.float32), plan_state_dim)
        return torch.from_numpy(arr)
    if _is_task1_put_cream_butter(row):
        stage = _task1_plan_stage_from_progress(progress)
        return torch.from_numpy(_task1_plan_state_from_stage(stage, plan_state_dim))
    return torch.zeros(int(plan_state_dim), dtype=torch.float32)


class LiberoExpertCacheDataset(Dataset):
    def __init__(
        self,
        manifest: Path | list[Path],
        *,
        plan_state_dim: int = DEFAULT_PLAN_STATE_DIM,
        include_action_history: bool = True,
        require_multiview: bool = False,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        self._episode_len_cache: dict[tuple[str, str], int] = {}
        self.plan_state_dim = int(plan_state_dim)
        self.include_action_history = bool(include_action_history)
        self.require_multiview = bool(require_multiview)
        manifests = manifest if isinstance(manifest, list) else [manifest]
        for item in manifests:
            with Path(item).open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"empty LIBERO expert cache manifest: {manifest}")

    def _episode_len(self, row: dict[str, Any]) -> int | None:
        if "episode_len" in row:
            return int(row["episode_len"])
        hdf5_path = row.get("hdf5_path")
        demo_id = row.get("demo_id")
        if not hdf5_path or not demo_id:
            return None
        key = (str(hdf5_path), str(demo_id))
        if key not in self._episode_len_cache:
            with h5py.File(key[0], "r") as h5:
                self._episode_len_cache[key] = int(h5["data"][key[1]]["actions"].shape[0])
        return self._episode_len_cache[key]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        data = np.load(row["cache_path"])
        proposer_weight = float(data["proposer_weight"]) if "proposer_weight" in data else float(row.get("proposer_weight", 1.0))
        if "proposer_weight_override" in row:
            proposer_weight = float(row["proposer_weight_override"])
        if "proposer_weight_mult" in row:
            proposer_weight *= float(row["proposer_weight_mult"])
        sample = {
            "s_in": torch.from_numpy(data["s_in"].astype(np.float32)),
            "c": torch.from_numpy(data["c"].astype(np.float32)),
            "context_rgb": torch.from_numpy(data["context_rgb"].astype(np.float32)),
            "action_tgt": torch.from_numpy(data["action_tgt"].astype(np.float32)),
            "action_tgt_norm": torch.from_numpy(data["action_tgt_norm"].astype(np.float32)),
            "terminal_success_tgt": torch.tensor(float(data["terminal_success_tgt"]), dtype=torch.float32),
            "plausibility_tgt": torch.tensor(float(data["plausibility_tgt"]), dtype=torch.float32),
            "proposer_weight": torch.tensor(proposer_weight, dtype=torch.float32),
            "progress_tgt": torch.as_tensor(
                np.asarray(data["progress_tgt"], dtype=np.float32)
                if "progress_tgt" in data
                else np.asarray(0.0, dtype=np.float32)
            ),
            "task_name": row.get("task_name", ""),
        }
        has_wrist = "s_wrist" in data and "view_mask" in data
        if self.require_multiview and not has_wrist:
            raise RuntimeError(
                f"LIBERO cache is not explicit paired-view data: {row['cache_path']}"
            )
        if has_wrist:
            wrist = np.asarray(data["s_wrist"], dtype=np.float32)
            view_mask = np.asarray(data["view_mask"], dtype=np.bool_)
            if wrist.shape != np.asarray(data["s_in"]).shape:
                raise RuntimeError(
                    f"LIBERO wrist/anchor token shape mismatch: {wrist.shape} vs "
                    f"{np.asarray(data['s_in']).shape}"
                )
            if view_mask.shape != (wrist.shape[0], 2) or not view_mask[:, 0].all():
                raise RuntimeError(
                    f"invalid LIBERO view_mask {view_mask.shape} in {row['cache_path']}"
                )
            sample["s_wrist"] = torch.from_numpy(wrist)
            sample["view_mask"] = torch.from_numpy(view_mask)
        if "target_start" in row:
            episode_len = self._episode_len(row)
            if episode_len is not None:
                denom = max(1.0, float(episode_len) - 1.0)
                horizon = int(sample["action_tgt"].shape[0])
                progress = np.clip(
                    (float(row["target_start"]) + np.arange(horizon, dtype=np.float32)) / denom,
                    0.0,
                    1.0,
                )
                sample["progress_tgt"] = torch.from_numpy(progress)
        progress_for_plan = float(sample["progress_tgt"].reshape(-1)[-1])
        sample["plan_state"] = _infer_plan_state(row, data, progress_for_plan, self.plan_state_dim)
        if "lowdim_state" in data:
            sample["lowdim_state"] = torch.from_numpy(data["lowdim_state"].astype(np.float32))
        if "object_state" in data and np.asarray(data["object_state"]).size > 0:
            sample["object_state"] = torch.from_numpy(data["object_state"].astype(np.float32))
        else:
            sample["object_state"] = torch.zeros(112, dtype=torch.float32)
        if self.include_action_history and "action_history" in data:
            sample["action_history"] = torch.from_numpy(data["action_history"].astype(np.float32))
        return sample


def _load_action_stats(model: torch.nn.Module, stats_path: Path, device: torch.device) -> None:
    stats = np.load(stats_path)
    mean = torch.as_tensor(stats["mean"][:6], device=device, dtype=torch.float32)
    std = torch.as_tensor(stats["std"][:6], device=device, dtype=torch.float32).clamp_min(1e-6)
    model.load_action_stats(mean, std)


def _freeze(model: torch.nn.Module, prefixes: list[str]) -> tuple[int, int]:
    trainable_prefixes = tuple(prefixes)
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        enabled = name.startswith(trainable_prefixes)
        param.requires_grad = enabled
        if enabled:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return frozen, trainable


def _score(out: dict[str, torch.Tensor]) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    if "progress" in out:
        terms.append(torch.sigmoid(out["progress"].float()).mean(dim=1))
    if "terminal_success_logit" in out:
        terms.append(torch.sigmoid(out["terminal_success_logit"].float()))
    if not terms:
        return out["pred_tokens"].new_zeros(out["pred_tokens"].shape[0], dtype=torch.float32)
    return torch.stack(terms, dim=0).mean(dim=0)


def _progress_score(out: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sigmoid(out["progress"].float()).mean(dim=1)


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=values.device, dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1e-6)


def _proposer_losses(out: dict[str, torch.Tensor], action_tgt: torch.Tensor, action_tgt_norm: torch.Tensor, proposer_weight: torch.Tensor, cfg: dict) -> dict[str, torch.Tensor]:
    pose_pred = out["proposer_pose_norm"].float()
    pose_tgt = action_tgt_norm.float()
    pose_err = F.smooth_l1_loss(
        pose_pred,
        pose_tgt[:, None].expand_as(pose_pred),
        beta=float(cfg.get("huber_delta", 1.0)),
        reduction="none",
    ).mean(dim=(2, 3))
    best_idx = pose_err.argmin(dim=1)
    sample_weight = proposer_weight.float().to(device=pose_err.device)
    if float(sample_weight.sum().detach().cpu()) <= 0:
        zero = pose_err.new_zeros(())
        return {
            "L_proposer_pose": zero,
            "L_proposer_grip": zero,
            "L_proposer_anchor": zero,
            "L_proposer_anchor_grip": zero,
            "L_proposer_anchor_first_pose": zero,
            "L_proposer_anchor_first_grip": zero,
            "proposer_best_idx_mean": zero.detach(),
            "proposer_selected_pose_l1": zero.detach(),
            "proposer_anchor_pose_l1": zero.detach(),
            "proposer_anchor_first_pose_l1": zero.detach(),
        }
    L_pose = _weighted_mean(pose_err.gather(1, best_idx[:, None]).squeeze(1), sample_weight)
    L_anchor = _weighted_mean(pose_err[:, 0], sample_weight)

    grip_logits = out["proposer_gripper_logit"].float()
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_err = F.binary_cross_entropy_with_logits(
        grip_logits,
        grip_tgt[:, None].expand_as(grip_logits),
        reduction="none",
    ).mean(dim=2)
    L_grip = _weighted_mean(grip_err.gather(1, best_idx[:, None]).squeeze(1), sample_weight)
    L_anchor_grip = _weighted_mean(grip_err[:, 0], sample_weight)
    first_pose_err = F.smooth_l1_loss(
        pose_pred[:, 0, 0],
        pose_tgt[:, 0],
        beta=float(cfg.get("huber_delta", 1.0)),
        reduction="none",
    ).mean(dim=1)
    L_anchor_first_pose = _weighted_mean(first_pose_err, sample_weight)
    first_grip_err = F.binary_cross_entropy_with_logits(
        grip_logits[:, 0, 0],
        grip_tgt[:, 0],
        reduction="none",
    )
    L_anchor_first_grip = _weighted_mean(first_grip_err, sample_weight)

    selected_idx = pose_err.argmin(dim=1)
    return {
        "L_proposer_pose": L_pose,
        "L_proposer_grip": L_grip,
        "L_proposer_anchor": L_anchor,
        "L_proposer_anchor_grip": L_anchor_grip,
        "L_proposer_anchor_first_pose": L_anchor_first_pose,
        "L_proposer_anchor_first_grip": L_anchor_first_grip,
        "proposer_best_idx_mean": _weighted_mean(best_idx.float(), sample_weight).detach(),
        "proposer_selected_pose_l1": _weighted_mean(pose_err.gather(1, selected_idx[:, None]).squeeze(1), sample_weight).detach(),
        "proposer_anchor_pose_l1": L_anchor.detach(),
        "proposer_anchor_first_pose_l1": L_anchor_first_pose.detach(),
    }


def _candidate_rank_losses(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    out: dict[str, torch.Tensor],
    action_tgt: torch.Tensor,
    action_tgt_norm: torch.Tensor,
    proposer_weight: torch.Tensor,
    cfg: dict,
) -> dict[str, torch.Tensor]:
    candidate_cond = out["proposer_action_cond"].float()
    sample_weight = proposer_weight.float().to(device=candidate_cond.device)
    if float(sample_weight.sum().detach().cpu()) <= 0:
        zero = candidate_cond.new_zeros(())
        return {
            "L_candidate_ce": zero,
            "L_candidate_pairwise": zero,
            "candidate_rank_acc": zero.detach(),
            "candidate_anchor_acc": zero.detach(),
            "candidate_score_gap": zero.detach(),
            "candidate_selected_error": zero.detach(),
            "candidate_anchor_error": zero.detach(),
            "candidate_oracle_error": zero.detach(),
            "candidate_selected_vs_anchor_improvement": zero.detach(),
            "candidate_nonanchor_selection_rate": zero.detach(),
            "candidate_oracle_nonanchor_rate": zero.detach(),
        }
    bsz, n_candidates, _horizon, _dim = candidate_cond.shape
    pose_l1 = (candidate_cond[..., :6] - action_tgt_norm[:, None].float()).abs().mean(dim=(2, 3))
    grip_tgt = (action_tgt[..., 6] > 0.5).float()
    grip_prob = candidate_cond[..., 6].clamp(1e-5, 1.0 - 1e-5)
    with torch.autocast(device_type="cuda", enabled=False):
        grip_bce = F.binary_cross_entropy(
            grip_prob.float(),
            grip_tgt[:, None].expand_as(grip_prob).float(),
            reduction="none",
        ).mean(dim=2)
    oracle_err = (pose_l1 + float(cfg.get("candidate_grip_weight", 0.1)) * grip_bce).detach()
    oracle_idx = oracle_err.argmin(dim=1)

    if not bool(cfg.get("candidate_rank_enabled", True)):
        zero = candidate_cond.new_zeros(())
        anchor_err = oracle_err[:, 0]
        best_err = oracle_err.gather(1, oracle_idx[:, None]).squeeze(1)
        return {
            "L_candidate_ce": zero,
            "L_candidate_pairwise": zero,
            "candidate_rank_acc": zero.detach(),
            "candidate_anchor_acc": _weighted_mean((oracle_idx == 0).float(), sample_weight).detach(),
            "candidate_score_gap": zero.detach(),
            "candidate_selected_error": _weighted_mean(anchor_err, sample_weight).detach(),
            "candidate_anchor_error": _weighted_mean(anchor_err, sample_weight).detach(),
            "candidate_oracle_error": _weighted_mean(best_err, sample_weight).detach(),
            "candidate_selected_vs_anchor_improvement": zero.detach(),
            "candidate_nonanchor_selection_rate": zero.detach(),
            "candidate_oracle_nonanchor_rate": _weighted_mean((oracle_idx != 0).float(), sample_weight).detach(),
        }

    scores = []
    for ci in range(n_candidates):
        cand_out = model(
            s,
            c,
            action_cond=candidate_cond[:, ci].detach().to(device=s.device, dtype=s.dtype),
            pixel=False,
            bridging=False,
        )
        scores.append(_score(cand_out))
    score_t = torch.stack(scores, dim=1)
    ce_temp = max(1e-4, float(cfg.get("candidate_ce_temperature", 0.1)))
    ce_each = F.cross_entropy(score_t.float() / ce_temp, oracle_idx, reduction="none")
    L_ce = _weighted_mean(ce_each, sample_weight)

    best_score = score_t.gather(1, oracle_idx[:, None])
    non_oracle = torch.ones(bsz, n_candidates, dtype=torch.bool, device=score_t.device)
    non_oracle.scatter_(1, oracle_idx[:, None], False)
    gap = best_score - score_t[non_oracle].reshape(bsz, n_candidates - 1)
    margin = float(cfg.get("candidate_pairwise_margin", 0.05))
    selected_idx = score_t.argmax(dim=1)
    selected_err = oracle_err.gather(1, selected_idx[:, None]).squeeze(1)
    anchor_err = oracle_err[:, 0]
    best_err = oracle_err.gather(1, oracle_idx[:, None]).squeeze(1)
    relative_improvement = (anchor_err - selected_err) / anchor_err.clamp_min(1e-6)
    return {
        "L_candidate_ce": L_ce,
        "L_candidate_pairwise": _weighted_mean(torch.relu(margin - gap).mean(dim=1), sample_weight),
        "candidate_rank_acc": _weighted_mean((selected_idx == oracle_idx).float(), sample_weight).detach(),
        "candidate_anchor_acc": _weighted_mean((oracle_idx == 0).float(), sample_weight).detach(),
        "candidate_score_gap": _weighted_mean(gap.mean(dim=1), sample_weight).detach(),
        "candidate_selected_error": _weighted_mean(selected_err, sample_weight).detach(),
        "candidate_anchor_error": _weighted_mean(anchor_err, sample_weight).detach(),
        "candidate_oracle_error": _weighted_mean(best_err, sample_weight).detach(),
        "candidate_selected_vs_anchor_improvement": _weighted_mean(relative_improvement, sample_weight).detach(),
        "candidate_nonanchor_selection_rate": _weighted_mean((selected_idx != 0).float(), sample_weight).detach(),
        "candidate_oracle_nonanchor_rate": _weighted_mean((oracle_idx != 0).float(), sample_weight).detach(),
    }


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "s": batch["s_in"].to(device, non_blocking=True),
        "c": batch["c"].to(device, non_blocking=True),
        "context_rgb": batch["context_rgb"].to(device, non_blocking=True),
        "action_tgt": batch["action_tgt"].to(device, non_blocking=True),
        "action_tgt_norm": batch["action_tgt_norm"].to(device, non_blocking=True),
        "terminal_success_tgt": batch["terminal_success_tgt"].to(device, non_blocking=True),
        "plausibility_tgt": batch["plausibility_tgt"].to(device, non_blocking=True),
        "proposer_weight": batch["proposer_weight"].to(device, non_blocking=True),
        "progress_tgt": batch["progress_tgt"].to(device, non_blocking=True),
    }


def _binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = targets == 1
    neg = targets == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        rank_sums = np.bincount(inverse, weights=ranks)
        ranks = rank_sums[inverse] / counts[inverse]
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _classification_metrics(targets: list[float], logits: list[float]) -> dict[str, float]:
    y = np.asarray(targets, dtype=np.int64)
    z = np.asarray(logits, dtype=np.float64)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
    pred = p >= 0.5
    pos = y == 1
    neg = y == 0
    tpr = float(pred[pos].mean()) if pos.any() else float("nan")
    tnr = float((~pred[neg]).mean()) if neg.any() else float("nan")
    return {
        "outcome_auc": _binary_auc(y, p),
        "outcome_accuracy_at_0p5": float((pred == y).mean()),
        "outcome_balanced_accuracy_at_0p5": float(np.nanmean([tpr, tnr])),
        "outcome_prob_positive_mean": float(p[pos].mean()) if pos.any() else float("nan"),
        "outcome_prob_negative_mean": float(p[neg].mean()) if neg.any() else float("nan"),
        "outcome_prob_gap": float(p[pos].mean() - p[neg].mean()) if pos.any() and neg.any() else float("nan"),
        "outcome_count": int(len(y)),
        "outcome_positive_count": int(pos.sum()),
        "outcome_negative_count": int(neg.sum()),
    }


def _counterfactual_conditions(action_cond: torch.Tensor) -> dict[str, torch.Tensor]:
    reverse = action_cond.clone()
    reverse[..., :6] = -reverse[..., :6]
    reverse[..., 6] = 1.0 - reverse[..., 6]
    shuffle = action_cond.roll(1, dims=0) if action_cond.shape[0] > 1 else reverse
    temporal = action_cond.flip(dims=(1,))
    noop = torch.zeros_like(action_cond)
    noop[..., 6] = action_cond[:, :1, 6]
    return {"reverse": reverse, "shuffle": shuffle, "temporal": temporal, "noop": noop}


def _paired_counterfactual_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    factual_out: dict[str, torch.Tensor],
    action_cond: torch.Tensor,
    loss_cfg: dict,
) -> dict[str, torch.Tensor]:
    modes = tuple(loss_cfg.get("paired_counterfactual_modes", ()))
    zero = factual_out["terminal_success_logit"].new_zeros(())
    if not modes:
        return {
            "L_paired_counterfactual": zero,
            "L_paired_ce": zero.detach(),
            "L_paired_margin": zero.detach(),
            "paired_rank_acc": zero.detach(),
            "paired_gap_mean": zero.detach(),
        }
    conditions = _counterfactual_conditions(action_cond)
    unknown = sorted(set(modes) - set(conditions))
    if unknown:
        raise ValueError(f"unknown paired counterfactual modes: {unknown}")
    score_mode = str(loss_cfg.get("paired_counterfactual_score", "terminal")).strip().lower()
    if score_mode not in {"terminal", "outcome", "progress"}:
        raise ValueError(f"unknown paired_counterfactual_score: {score_mode}")
    score_fn = {
        "terminal": lambda item: item["terminal_success_logit"].float(),
        "outcome": _score,
        "progress": _progress_score,
    }[score_mode]
    negative_scores = []
    for mode in modes:
        candidate_out = model(
            batch["s"], batch["c"], action_cond=conditions[mode],
            context_rgb=batch["context_rgb"], pixel=False, bridging=False,
        )
        negative_scores.append(score_fn(candidate_out))
    factual_score = score_fn(factual_out)
    negative = torch.stack(negative_scores, dim=1)
    scores = torch.cat((factual_score[:, None], negative), dim=1)
    temperature = max(1e-4, float(loss_cfg.get("paired_counterfactual_temperature", 0.1)))
    labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    L_ce = F.cross_entropy(scores / temperature, labels)
    gap = factual_score[:, None] - negative
    margin = float(loss_cfg.get("paired_counterfactual_margin", 0.05))
    L_margin = torch.relu(margin - gap).mean()
    L_pair = (
        float(loss_cfg.get("paired_counterfactual_ce_weight", 1.0)) * L_ce
        + float(loss_cfg.get("paired_counterfactual_margin_weight", 0.25)) * L_margin
    )
    return {
        "L_paired_counterfactual": L_pair,
        "L_paired_ce": L_ce.detach(),
        "L_paired_margin": L_margin.detach(),
        "paired_rank_acc": (scores.argmax(dim=1) == 0).float().mean().detach(),
        "paired_gap_mean": gap.mean().detach(),
    }


def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, loss_cfg: dict) -> dict[str, float]:
    model.eval()
    evaluate_counterfactuals = bool(loss_cfg.get("evaluate_counterfactuals", True))
    agg: dict[str, float] = {}
    n = 0
    targets: list[float] = []
    logits: list[float] = []
    margins: dict[str, list[float]] = {mode: [] for mode in ("reverse", "shuffle", "temporal", "noop")}
    top1: list[float] = []
    factual_ranks: list[float] = []
    with torch.no_grad():
        for batch in loader:
            b = _batch_to_device(batch, device)
            action_cond = make_action_condition(b["action_tgt"], b["action_tgt_norm"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(b["s"], b["c"], action_cond=action_cond, context_rgb=b["context_rgb"], pixel=False, bridging=False)
                losses = _losses(model, b, out, loss_cfg)
                candidate_outputs = (
                    {
                        mode: model(
                            b["s"], b["c"], action_cond=condition,
                            context_rgb=b["context_rgb"], pixel=False, bridging=False,
                        )
                        for mode, condition in _counterfactual_conditions(action_cond).items()
                    }
                    if evaluate_counterfactuals
                    else {}
                )
            for key, value in losses.items():
                agg[key] = agg.get(key, 0.0) + float(value.detach().float())
            factual_logit = out["terminal_success_logit"].detach().float()
            score_mode = str(loss_cfg.get("paired_counterfactual_score", "terminal")).strip().lower()
            rank_score_fn = {
                "terminal": lambda item: item["terminal_success_logit"].float(),
                "outcome": _score,
                "progress": _progress_score,
            }[score_mode]
            target = b["terminal_success_tgt"].detach().float()
            targets.extend(target.cpu().tolist())
            logits.extend(factual_logit.cpu().tolist())
            positive = target > 0.5
            if evaluate_counterfactuals and positive.any():
                factual_rank_score = rank_score_fn(out).detach().float()
                candidate_scores = []
                for mode, candidate_out in candidate_outputs.items():
                    candidate_score = rank_score_fn(candidate_out).detach().float()
                    margins[mode].extend((factual_rank_score[positive] - candidate_score[positive]).cpu().tolist())
                    candidate_scores.append(candidate_score[positive])
                scores = torch.stack((factual_rank_score[positive], *candidate_scores), dim=1)
                top1.extend((scores.argmax(dim=1) == 0).float().cpu().tolist())
                factual_ranks.extend((1 + (scores[:, 1:] > scores[:, :1]).sum(dim=1)).float().cpu().tolist())
            n += 1
    metrics = {key: val / max(1, n) for key, val in sorted(agg.items())}
    metrics.update(_classification_metrics(targets, logits))
    metrics["paired_all_counterfactual_top1"] = float(np.mean(top1)) if top1 else float("nan")
    metrics["paired_factual_mean_rank"] = float(np.mean(factual_ranks)) if factual_ranks else float("nan")
    for mode, values in margins.items():
        arr = np.asarray(values, dtype=np.float64)
        name = f"positive_factual_minus_{mode}_logit"
        metrics[f"{name}_mean"] = float(arr.mean()) if len(arr) else float("nan")
        metrics[f"{name}_win_fraction"] = float((arr > 0).mean()) if len(arr) else float("nan")
    return metrics
def _losses(model: torch.nn.Module, batch: dict[str, Any], out: dict[str, torch.Tensor], loss_cfg: dict) -> dict[str, torch.Tensor]:
    terminal_tgt = batch["terminal_success_tgt"].float()
    terminal_prob = torch.sigmoid(out["terminal_success_logit"].float())
    L_terminal = F.binary_cross_entropy_with_logits(out["terminal_success_logit"].float(), terminal_tgt)
    plaus_tgt = batch["plausibility_tgt"].float()
    L_plaus = F.binary_cross_entropy_with_logits(out["plausibility_logit"].float(), plaus_tgt)
    progress_pred = torch.sigmoid(out["progress"].float())
    progress_tgt = batch["progress_tgt"].float()
    if progress_tgt.ndim == 1:
        progress_tgt = progress_tgt[:, None].expand(-1, progress_pred.shape[1])
    horizon = min(progress_pred.shape[1], progress_tgt.shape[1])
    progress_pred = progress_pred[:, :horizon]
    progress_tgt = progress_tgt[:, :horizon]
    L_progress = F.smooth_l1_loss(
        progress_pred,
        progress_tgt,
        beta=float(loss_cfg.get("huber_delta", 1.0)),
    )
    proposer = _proposer_losses(out, batch["action_tgt"], batch["action_tgt_norm"], batch["proposer_weight"], loss_cfg)
    candidate = _candidate_rank_losses(model, batch["s"], batch["c"], out, batch["action_tgt"], batch["action_tgt_norm"], batch["proposer_weight"], loss_cfg)
    L_total = (
        float(loss_cfg.get("terminal_weight", 0.2)) * L_terminal
        + float(loss_cfg.get("plausibility_weight", 0.0)) * L_plaus
        + float(loss_cfg.get("proposer_pose_weight", 1.0)) * proposer["L_proposer_pose"]
        + float(loss_cfg.get("progress_weight", 0.0)) * L_progress
        + float(loss_cfg.get("proposer_grip_weight", 0.2)) * proposer["L_proposer_grip"]
        + float(loss_cfg.get("proposer_anchor_weight", 0.1)) * proposer["L_proposer_anchor"]
        + float(loss_cfg.get("proposer_anchor_grip_weight", 0.0)) * proposer["L_proposer_anchor_grip"]
        + float(loss_cfg.get("proposer_anchor_first_pose_weight", 0.0)) * proposer["L_proposer_anchor_first_pose"]
        + float(loss_cfg.get("proposer_anchor_first_grip_weight", 0.0)) * proposer["L_proposer_anchor_first_grip"]
        + float(loss_cfg.get("candidate_ce_weight", 0.2)) * candidate["L_candidate_ce"]
        + float(loss_cfg.get("candidate_pairwise_weight", 0.1)) * candidate["L_candidate_pairwise"]
    )
    return {
        "L_total": L_total,
        "L_terminal": L_terminal.detach(),
        "terminal_prob_mean": terminal_prob.mean().detach(),
        "L_plausibility": L_plaus.detach(),
        **{k: v.detach() for k, v in proposer.items()},
        "L_progress": L_progress.detach(),
        "progress_mae": (progress_pred - progress_tgt).abs().mean().detach(),
        **{k: v.detach() for k, v in candidate.items()},
    }


_PROBE_BATCH_KEYS = (
    "s_in", "c", "context_rgb", "action_tgt", "action_tgt_norm",
    "terminal_success_tgt", "plausibility_tgt", "proposer_weight", "progress_tgt",
)


def _probe_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return default_collate([{key: sample[key] for key in _PROBE_BATCH_KEYS} for sample in samples])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--train_manifest", type=Path, default=None)
    ap.add_argument("--val_manifest", type=Path, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out_root", type=Path, default=None)
    ap.add_argument("--pair_modes", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    base_cfg = yaml.safe_load(Path(cfg["base_cfg"]).read_text())
    loss_cfg = dict(cfg["loss"])
    if args.pair_modes is not None:
        loss_cfg["paired_counterfactual_modes"] = [mode for mode in args.pair_modes.split(",") if mode]
    device = torch.device(args.device or cfg["train"].get("device", "cuda:0"))
    seed = int(args.seed if args.seed is not None else cfg["data"].get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if args.train_manifest is not None or args.val_manifest is not None:
        if args.train_manifest is None or args.val_manifest is None:
            raise ValueError("--train_manifest and --val_manifest must be provided together")
        train_ds = LiberoExpertCacheDataset(args.train_manifest)
        val_ds = LiberoExpertCacheDataset(args.val_manifest)
    else:
        manifest_cfg = cfg["data"]["manifest"]
        manifests = [Path(item) for item in manifest_cfg] if isinstance(manifest_cfg, list) else Path(manifest_cfg)
        ds = LiberoExpertCacheDataset(manifests)
        val_frac = float(cfg["data"].get("val_frac", 0.2))
        n_val = max(1, int(len(ds) * val_frac))
        n_train = max(1, len(ds) - n_val)
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    n_train = len(train_ds)
    n_val = len(val_ds)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)), pin_memory=True,
        drop_last=True, collate_fn=_probe_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)), pin_memory=True,
        collate_fn=_probe_collate,
    )

    model = build_model(base_cfg).to(device)
    init_ckpt = Path(cfg["train"]["init_ckpt"])
    sd = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"], strict=True)
    _load_action_stats(model, Path(cfg["data"]["action_stats"]), device)
    frozen, trainable = _freeze(model, list(cfg["train"].get("trainable_prefixes", ["progress_head.", "action_proposer."])))
    print(json.dumps({"loaded": str(init_ckpt), "train_windows": n_train, "val_windows": n_val, "frozen_M": frozen / 1e6, "trainable_M": trainable / 1e6}), flush=True)

    baseline_metrics = _evaluate(model, val_loader, device, loss_cfg)
    print(json.dumps({"baseline_metrics": baseline_metrics}, sort_keys=True), flush=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr if args.lr is not None else cfg["train"].get("lr", 1e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.02)),
        betas=(0.9, 0.95),
    )
    max_steps = int(args.max_steps or cfg["train"].get("max_steps", 20))
    warmup = int(cfg["train"].get("warmup_steps", 5))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, max_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    out_root = args.out_root or Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"].get("ckpt_dir", "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    while step < max_steps:
        for batch in train_loader:
            b = _batch_to_device(batch, device)
            action_cond = make_action_condition(b["action_tgt"], b["action_tgt_norm"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(b["s"], b["c"], action_cond=action_cond, context_rgb=b["context_rgb"], pixel=False, bridging=False)
                losses = _losses(model, b, out, loss_cfg)
                paired = _paired_counterfactual_losses(model, b, out, action_cond, loss_cfg)
                losses["L_total"] = losses["L_total"] + float(loss_cfg.get("paired_counterfactual_weight", 0.0)) * paired["L_paired_counterfactual"]
                losses.update(paired)
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            opt.step()
            sched.step()
            if args.print_every and step % args.print_every == 0:
                print(
                    f"[libero-success] step {step} "
                    f"L_total={float(losses['L_total'].detach().float()):.4f} "
                    f"prop={float(losses['L_proposer_pose']):.4f} "
                    f"term_p={float(losses['terminal_prob_mean']):.3f} "
                    f"prog_mae={float(losses['progress_mae']):.3f} "
                    f"cand_ce={float(losses['L_candidate_ce']):.4f} "
                    f"cand_acc={float(losses['candidate_rank_acc']):.3f} "
                    f"pair_acc={float(losses['paired_rank_acc']):.3f} "
                    f"pair_gap={float(losses['paired_gap_mean']):.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            step += 1
            if step >= max_steps:
                break

    metrics = _evaluate(model, val_loader, device, loss_cfg)
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline": baseline_metrics, "trained": metrics, "seed": seed,
        "lr": opt.defaults["lr"], "step": step,
        "paired_counterfactual_modes": loss_cfg.get("paired_counterfactual_modes", []),
    }
    (out_root / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    save_trainable_only = bool(cfg["out"].get("save_trainable_only", False))
    trainable_prefixes = tuple(cfg["train"].get("trainable_prefixes", []))
    model_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if not save_trainable_only or name.startswith(trainable_prefixes)
    }
    ckpt = {
        "model": model_state,
        "trainable_only": save_trainable_only,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "step": step,
        "cfg": cfg,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
    }
    torch.save(ckpt, ckpt_dir / "best.pt")
    print(json.dumps({"metrics": metrics, "ckpt": str(ckpt_dir / "best.pt"), "step": step}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
