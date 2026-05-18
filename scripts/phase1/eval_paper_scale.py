"""Run M2 gate evaluation on a paper-scale D_ψ checkpoint.

Loads `results/phase1_scale/runs/{run}/best.pt`, evaluates k-step rollout L2
and cf_delta against the held-out val shards, and prints the M2 acceptance
verdict (from PHASE2_PAPER_PLAN §7):

    k=8 rollout L2 ≤ 3.34 (prototype vggt_noact baseline)  AND
    cf_delta significantly non-zero (> 0.05) on val

Writes a full summary JSON next to the checkpoint:
    results/phase1_scale/runs/{run}/eval_summary.json

Single-GPU (rank 0). Reuses src.phase1.eval.evaluate(), which is the same
evaluator the prototype trainer uses, so numbers are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running as `python scripts/phase1/eval_paper_scale.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import yaml

from src.phase1.dataset import (
    compute_action_stats,
    discover_shards,
    split_shards,
)
from src.phase1.eval import evaluate
from src.phase1.heads import PredictiveHead

log = logging.getLogger("eval_paper_scale")

PROTO_K8_L2_BASELINE = 3.34   # from PHASE1_REPORT vggt_noact
CF_DELTA_FLOOR = 0.05         # spec §7 M2 gate


def _load_val_ids(cfg: dict) -> list:
    split = cfg["dataset"]["split"]
    if "val_episode_ids" in split:
        return list(split["val_episode_ids"])
    if "val_episode_ids_file" in split:
        p = Path(split["val_episode_ids_file"])
        if p.exists():
            return list(json.loads(p.read_text()))
    raise RuntimeError("no val ids configured")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="configs/phase1/paper_scale.yaml")
    p.add_argument("--run", default="vggt_noact",
                   choices=["vggt", "vggt_noact", "vggt_bigact"])
    p.add_argument("--out_root", default="results/phase1_scale/runs")
    p.add_argument("--ckpt", default=None,
                   help="override checkpoint path; default = {out_root}/{run}/best.pt")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = yaml.safe_load(open(args.cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else Path(args.out_root) / args.run / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    log.info("loading checkpoint: %s", ckpt_path)

    # discover val shards (same logic as the trainer)
    cache_dir = Path(cfg["cache"]["out_dir"])
    shards = discover_shards(cache_dir)
    val_ids = _load_val_ids(cfg)
    train_shards, val_shards = split_shards(shards, val_ids)
    log.info("train shards: %d  val shards: %d", len(train_shards), len(val_shards))

    # action stats (always computed from train shards, even for vggt_noact)
    action_stats = None
    if cfg["dataset"]["normalize_actions"]:
        action_stats = compute_action_stats(train_shards)

    # build the head — must mirror trainer config exactly
    use_actions = args.run in ("vggt", "vggt_bigact")
    head_cfg = cfg["head"]
    head = PredictiveHead(
        token_dim=int(cfg["cache"]["token_dim"]),
        action_dim=int(cfg["dataset"]["action_dim"]),
        hidden_dim=int(head_cfg["hidden_dim"]),
        n_layers=int(head_cfg["n_layers"]),
        n_heads=int(head_cfg["n_heads"]),
        context_len=int(head_cfg["context_len"]),
        action_embed_dim=int(head_cfg["action_embed_dim"]),
        dropout=float(head_cfg["dropout"]),
        use_actions=use_actions,
        use_checkpoint=False,                  # not needed at eval
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    head.load_state_dict(state)
    head.eval()

    log.info("running k-step rollout eval at horizons %s on %d val shards",
             cfg["eval"]["horizons"], len(val_shards))
    t0 = time.time()
    full = evaluate(
        head,
        val_shards,
        context_len=int(head_cfg["context_len"]),
        horizons=cfg["eval"]["horizons"],
        token_pool=head_cfg["token_pool"],
        device=device,
        action_stats=action_stats,
        bootstrap_iters=int(cfg["eval"]["bootstrap_iters"]),
    )
    log.info("eval done in %.1fs", time.time() - t0)

    # ---- write summary
    run_dir = Path(args.out_root) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(full["summary"], indent=2))
    log.info("summary written: %s", summary_path)

    # ---- M2 gate verdict (per PHASE2_PAPER_PLAN §7)
    k8 = full["summary"].get(8, {})
    k8_l2 = k8.get("l2", {}).get("mean", float("nan"))
    cf_d  = k8.get("cf_delta", {}).get("mean", float("nan"))

    log.info("=== M2 gate ===")
    log.info("  k=8 rollout L2 = %.4f   (proto baseline ceiling %.2f)", k8_l2, PROTO_K8_L2_BASELINE)
    log.info("  k=8 cf_delta   = %.4f   (floor %.2f, only meaningful when use_actions=True)",
             cf_d, CF_DELTA_FLOOR)

    l2_pass = k8_l2 <= PROTO_K8_L2_BASELINE if not np.isnan(k8_l2) else False
    cf_pass = cf_d  > CF_DELTA_FLOOR        if not np.isnan(cf_d)  else False

    if not use_actions:
        log.info("  (vggt_noact: cf_delta is informational only — gate decided by k=8 L2)")
        gate_pass = l2_pass
    else:
        gate_pass = l2_pass and cf_pass

    log.info("  VERDICT: %s", "PASS" if gate_pass else "FAIL — investigate before scaling G_θ")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
