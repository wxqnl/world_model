"""Phase 1 training loop.

Single-process training on one GPU. The predictive head is tiny (~20M params)
and the dataset is small (~3k pairs), so multi-GPU is not needed. We train
three runs back-to-back:

    1. vggt         — VGGT tokens, action-conditioned
    2. vggt_noact   — VGGT tokens, zero-action ablation (same architecture)
    3. dinov2       — DINOv2 tokens, action-conditioned

All three runs share the same head hyperparameters and training recipe. Only
the feature cache and the ``use_actions`` flag differ.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import (
    NextTokenPairs,
    compute_action_stats,
    discover_shards,
    split_shards,
)
from .eval import evaluate
from .heads import PredictiveHead, count_params

log = logging.getLogger("phase1.train")


def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * t))


def run_one(
    cfg: dict,
    cache_dir: Path,
    token_dim: int,
    run_name: str,
    use_actions: bool,
    device: str = "cuda",
    out_root: Path | None = None,
) -> dict:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    shards = discover_shards(cache_dir)
    if not shards:
        raise RuntimeError(f"no shards found in {cache_dir}")
    train_shards, val_shards = split_shards(shards, cfg["dataset"]["split"]["val_episode_ids"])
    log.info("  train shards: %d   val shards: %d", len(train_shards), len(val_shards))
    if not train_shards or not val_shards:
        raise RuntimeError("train or val split is empty")

    action_stats = None
    if cfg["dataset"]["normalize_actions"]:
        action_stats = compute_action_stats(train_shards)
        log.info("  action mean: %s", np.round(action_stats[0], 3))
        log.info("  action std:  %s", np.round(action_stats[1], 3))

    C = cfg["head"]["context_len"]
    pool = cfg["head"]["token_pool"]
    train_ds = NextTokenPairs(train_shards, C, pool, cfg["dataset"]["normalize_actions"], action_stats)
    val_ds = NextTokenPairs(val_shards, C, pool, cfg["dataset"]["normalize_actions"], action_stats)
    log.info("  train pairs: %d   val pairs: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_windows"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    head = PredictiveHead(
        token_dim=token_dim,
        action_dim=cfg["dataset"]["action_dim"],
        hidden_dim=cfg["head"]["hidden_dim"],
        n_layers=cfg["head"]["n_layers"],
        n_heads=cfg["head"]["n_heads"],
        context_len=C,
        action_embed_dim=cfg["head"]["action_embed_dim"],
        dropout=cfg["head"]["dropout"],
        use_actions=use_actions,
    ).to(device)
    log.info("  head params: %.2f M", count_params(head) / 1e6)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    total_steps = cfg["train"]["epochs"] * len(train_loader)
    warmup = cfg["train"]["warmup_steps"]

    log_rows = []
    best_val = float("inf")
    best_state = None
    best_summary = None
    step = 0
    t0 = time.time()

    for epoch in range(cfg["train"]["epochs"]):
        head.train()
        losses = []
        for batch in train_loader:
            for k in ("ctx_tokens", "ctx_actions", "tgt_action", "tgt_tokens"):
                batch[k] = batch[k].to(device, non_blocking=True)
            lr = cosine_lr(step, total_steps, warmup, cfg["train"]["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            pred = head(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
            loss = (pred - batch["tgt_tokens"]).pow(2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            losses.append(float(loss))
            step += 1

        # Per-epoch quick val loss (not full eval — that's expensive).
        head.eval()
        with torch.no_grad():
            vl = []
            vloader = DataLoader(val_ds, batch_size=cfg["train"]["batch_windows"],
                                 shuffle=False, num_workers=0, drop_last=False)
            for batch in vloader:
                for k in ("ctx_tokens", "ctx_actions", "tgt_action", "tgt_tokens"):
                    batch[k] = batch[k].to(device, non_blocking=True)
                pred = head(batch["ctx_tokens"], batch["ctx_actions"], batch["tgt_action"])
                vl.append((pred - batch["tgt_tokens"]).pow(2).mean().item())
            val_loss = float(np.mean(vl))
        train_loss = float(np.mean(losses))
        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
        log.info(
            "  ep %02d  train %.4e  val %.4e  lr %.2e  %.1fs",
            epoch, train_loss, val_loss, lr, time.time() - t0,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    # Restore best + full rollout eval.
    assert best_state is not None
    head.load_state_dict(best_state)
    log.info("  best val %.4e — running full k-step rollout eval", best_val)
    full = evaluate(
        head,
        val_shards,
        context_len=C,
        horizons=cfg["eval"]["horizons"],
        token_pool=pool,
        device=device,
        action_stats=action_stats,
        bootstrap_iters=cfg["eval"]["bootstrap_iters"],
    )

    # Persist.
    out_root = out_root or Path("results/phase1/runs")
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "best.pt")
    (run_dir / "train_log.json").write_text(json.dumps(log_rows, indent=2))
    (run_dir / "eval_summary.json").write_text(json.dumps(full["summary"], indent=2))

    # per-sample CSVs for later plotting
    import csv
    for k, per in full["per_sample"].items():
        with open(run_dir / f"eval_k{k}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["l2", "cos", "cf_delta"])
            for i in range(len(per["l2"])):
                w.writerow([per["l2"][i], per["cos"][i], per["cf_delta"][i]])

    elapsed = time.time() - t0
    log.info("  %s done in %.1fs   best val %.4e", run_name, elapsed, best_val)
    return {"run_name": run_name, "best_val": best_val, "elapsed": elapsed}


def main(cfg_path: str, runs: list[str], device: str = "cuda",
         action_embed_dim: int | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = yaml.safe_load(open(cfg_path))
    if action_embed_dim is not None:
        cfg["head"]["action_embed_dim"] = int(action_embed_dim)
        log.info("override head.action_embed_dim = %d", cfg["head"]["action_embed_dim"])

    results = []
    for run in runs:
        log.info("=== run: %s ===", run)
        # `vggt_bigact` is an action-conditioning triage variant: same cache as
        # `vggt` but caller overrides --action_embed_dim (default 64 -> e.g. 256).
        if run in ("vggt", "vggt_bigact"):
            cache = Path(cfg["cache"]["out_dir"])
            token_dim = cfg["cache"]["token_dim"]
            use_actions = True
        elif run == "vggt_noact":
            cache = Path(cfg["cache"]["out_dir"])
            token_dim = cfg["cache"]["token_dim"]
            use_actions = False
        elif run == "dinov2":
            cache = Path(cfg["cache"]["out_dir"]).parent / "cache_tokens_dinov2"
            token_dim = cfg["dinov2"]["token_dim"]
            use_actions = True
        else:
            raise ValueError(f"unknown run: {run}")
        try:
            info = run_one(cfg, cache, token_dim, run, use_actions, device=device)
            results.append(info)
        except Exception as e:
            log.exception("run %s failed: %s", run, e)
            results.append({"run_name": run, "error": str(e)})

    Path("results/phase1/runs/_summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/phase1/default.yaml")
    p.add_argument("--runs", nargs="+", default=["vggt", "vggt_noact", "dinov2"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--action_embed_dim", type=int, default=None,
                   help="override cfg.head.action_embed_dim (e.g. 256 for "
                        "action-conditioning capacity triage)")
    args = p.parse_args()
    main(args.cfg, args.runs, args.device, action_embed_dim=args.action_embed_dim)
