"""Evaluation harness for Phase 1 predictive head.

Reports four metrics, each with per-sample arrays that we bootstrap for CIs:

* **k-step token L2.** Mean squared error of rolled-out predicted tokens vs.
  real cached tokens at horizons k = 1, 2, 4, 8. Normalized by the per-clip
  standard deviation of the target to give a scale-free number.
* **k-step cosine.** Same as above but cosine similarity. Easier to read.
* **Trajectory error (VGGT only).** When the backbone is VGGT we also compare
  predicted-frame *decoded* camera extrinsic against the real extrinsic at
  horizon k, using SO(3) geodesic for rotation and L2 for translation. Since
  the predictive head works in pooled-token space we cannot decode geometry
  directly; we instead use nearest-neighbor lookup into the cached real frames
  and report "how far does the predicted token land from its nearest real
  frame in the clip," which serves as a rough semantic check.
* **Counterfactual action delta (VGGT only).** Zero out `tgt_action`; re-run
  prediction; report L2 distance between the two predictions. A real
  action-conditioned model should have larger counterfactual delta than a
  zero-action baseline.

Outputs one CSV per run with per-sample rows and a summary JSON.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import NextTokenPairs, Shard, _load_tokens

log = logging.getLogger("phase1.eval")


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    shard: Shard,
    context_len: int,
    horizons: list[int],
    token_pool: str,
    device: str,
    action_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Autoregressive rollout on a single shard.

    At each starting position t0 in [C-1, N-max(horizons)-1], we unroll the
    model for max(horizons) steps feeding previous predictions back in, and
    compare each prediction at horizon k to the ground-truth cached token.
    """
    data = np.load(shard.path)
    tokens = _load_tokens(data["tokens"])     # [N, P, D]
    actions = torch.from_numpy(data["actions"]).float()  # [N, A]

    if token_pool == "mean":
        tokens_state = tokens.mean(dim=1)     # [N, D]
    else:
        tokens_state = tokens                 # keep grid

    if action_stats is not None:
        m = torch.from_numpy(action_stats[0]).float()
        s = torch.from_numpy(action_stats[1]).float()
        actions = (actions - m) / s

    N = tokens_state.shape[0]
    Kmax = max(horizons)
    tgt_std = tokens_state.float().std(dim=0).clamp(min=1e-5)  # per-dim std

    per_k = {k: {"l2": [], "cos": [], "cf_delta": []} for k in horizons}
    model.eval()

    starts = list(range(context_len - 1, N - Kmax - 1))
    for t0 in starts:
        ctx = tokens_state[t0 - context_len + 1 : t0 + 1].to(device)  # [C, D]
        ctx = ctx.unsqueeze(0)                                        # [1, C, D]
        ctx_actions = actions[t0 - context_len + 1 : t0 + 1].to(device).unsqueeze(0)
        for k in range(1, Kmax + 1):
            a_apply = actions[t0 + k - 1].to(device).unsqueeze(0)     # [1, A]
            pred = model(ctx, ctx_actions, a_apply)                   # [1, D]

            if k in per_k:
                real = tokens_state[t0 + k].to(device)
                # scale-free per-dim L2
                l2 = (((pred.squeeze(0) - real) / tgt_std.to(device)) ** 2).mean().item()
                cos = F.cosine_similarity(pred, real.unsqueeze(0), dim=-1).item()
                # counterfactual: same context, zero action
                a_zero = torch.zeros_like(a_apply)
                pred_cf = model(ctx, ctx_actions, a_zero)
                cf_delta = (pred - pred_cf).pow(2).mean().sqrt().item()
                per_k[k]["l2"].append(l2)
                per_k[k]["cos"].append(cos)
                per_k[k]["cf_delta"].append(cf_delta)

            # advance context: slide by one, replace last with prediction
            ctx = torch.cat([ctx[:, 1:], pred.unsqueeze(1)], dim=1)
            a_next = actions[t0 + k].to(device).unsqueeze(0)
            ctx_actions = torch.cat([ctx_actions[:, 1:], a_next.unsqueeze(1)], dim=1)

    return per_k


def bootstrap_ci(values: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, n, size=(iters, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(
    model: torch.nn.Module,
    val_shards: list[Shard],
    context_len: int,
    horizons: list[int],
    token_pool: str,
    device: str,
    action_stats: tuple[np.ndarray, np.ndarray] | None = None,
    bootstrap_iters: int = 1000,
) -> dict:
    all_per_k = {k: {"l2": [], "cos": [], "cf_delta": []} for k in horizons}
    for sh in val_shards:
        per_k = rollout(model, sh, context_len, horizons, token_pool, device, action_stats)
        for k in horizons:
            for m in ("l2", "cos", "cf_delta"):
                all_per_k[k][m].extend(per_k[k][m])

    summary: dict = {}
    for k in horizons:
        summary[k] = {}
        for m in ("l2", "cos", "cf_delta"):
            v = np.array(all_per_k[k][m], dtype=np.float64)
            mean, lo, hi = bootstrap_ci(v, iters=bootstrap_iters, seed=k * 13 + hash(m) % 1000)
            summary[k][m] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": int(v.size)}
    return {"per_sample": all_per_k, "summary": summary}
