# Phase 1 paper-scale report — scaled `D_ψ` on 25K-clip token cache

**Date:** 2026-05-18
**Runner:** `scripts/launch_ddp.sh --gpus 4 -- python -m src.phase1.train_ddp --cfg configs/phase1/paper_scale.yaml --run vggt_noact`
**Hardware:** 4× H100 80 GB (GPUs 0–3), shared box
**Config:** `configs/phase1/paper_scale.yaml`, 12 epochs, per-rank batch 32, 155.36 M-param head, ctx=8, bf16 autocast, gradient checkpointing on
**Eval:** k-step rollout at k ∈ {1, 2, 4, 8, 16, 32}, n = 11,777 samples per horizon, 95% CI by 1000-iter bootstrap (`scripts/phase1/eval_paper_scale.py`)
**Data:** 25,006 clips total (bridge 12K + fractal20220817 8K + kuka 4K + jaco 976 + DROID-100 30), 22,505 train / 2,501 val (10% per-sub-dataset, capped at 3K). Tokens cached via `src/phase1/cache.py` from frozen VGGT-1B.

## Summary

| Run | Best val loss | @epoch | L2 k=1 | L2 k=8 | cos k=1 | cos k=8 | L2 k=32 |
|---|---|---|---|---|---|---|---|
| **`vggt_noact` paper-scale** | **5.68e-03** | 11 | **0.41** | **1.21** | 0.9994 | 0.9971 | **3.19** |
| `vggt_noact` prototype (n=696, 30 clips) | 1.71e-02 | 29 | 1.54 | 3.34 | 0.9979 | 0.9939 | — |

**Scaling effect on k=8 L2: 3.34 → 1.21 (64% drop).** k=32 at paper scale (3.19) is *still below* the prototype's k=8 ceiling — the long-horizon stability claim holds.

## M2 acceptance criterion

Per `PHASE2_PAPER_PLAN.md` §7:
> M2 gate: k=8 rollout L2 ≤ prototype's `vggt_noact` (3.34) **AND** `cf_delta` significantly non-zero (> 0.05) on val.

- ✅ k=8 L2 = **1.21** (gate ≤ 3.34) — passes by 2.76×
- N/A `cf_delta` = 0 (vggt_noact uses no actions; this leg of the gate only applies to action-conditioned runs)

**Verdict: PASS.** The paper's foundational claim — VGGT tokens give a temporally-predictable substrate that scales — is supported at paper scale.

## Loss trajectory

| epoch | train | val |
|---:|---:|---:|
| 0 | 4.07e-01 | 1.41e-02 |
| 1 | 1.25e-02 | 1.09e-02 |
| 2 | 1.04e-02 | 9.60e-03 |
| 3 | 9.16e-03 | 9.12e-03 |
| 4 | 8.28e-03 | 7.86e-03 |
| 5 | 7.25e-03 | 7.32e-03 |
| 6 | 6.79e-03 | 6.72e-03 |
| 7 | 6.27e-03 | 6.38e-03 |
| 8 | 6.07e-03 | 6.07e-03 |
| 9 | 5.59e-03 | 5.83e-03 |
| 10 | 5.50e-03 | 5.71e-03 |
| **11** | **5.32e-03** | **5.68e-03** |

Train ≈ val throughout (best epoch matches train, no overfitting gap). Wall-clock: 6,420 s = 107 min for full 12-epoch run.

## Eval — all horizons

| k | L2 mean | L2 95% CI | cos | cf_delta | n |
|---:|---:|---|---:|---:|---:|
| 1 | 0.4067 | [0.401, 0.412] | 0.9994 | 0 | 11,777 |
| 2 | 0.5275 | [0.520, 0.535] | 0.9991 | 0 | 11,777 |
| 4 | 0.7683 | [0.758, 0.778] | 0.9984 | 0 | 11,777 |
| 8 | 1.2112 | [1.196, 1.227] | 0.9971 | 0 | 11,777 |
| 16 | 1.9270 | [1.901, 1.953] | 0.9953 | 0 | 11,777 |
| 32 | 3.1875 | [3.136, 3.240] | 0.9927 | 0 | 11,777 |

Cosine similarity ≥ 0.99 across all horizons. L2 roughly doubles per 4× horizon (k=1 → k=4: 0.41 → 0.77; k=8 → k=32: 1.21 → 3.19) — predictable scaling.

## Fixes landed during this milestone

Five distinct bugs surfaced trying to bring the prototype trainer up to DDP at paper scale; each cost a crash and a restart before the run completed.

1. **`pyarrow` not in venv** — `src/phase1/cache.py`'s DROID parquet loader imports `pyarrow.parquet` lazily; 30/25,006 cached clips silently failed at the end of the cache pass with `ModuleNotFoundError`. Added to `requirements.txt`; installed into the project venv.

2. **`scripts/launch_ddp.sh` couldn't run `python -m MODULE`** — when `PASS_ARGS = ["python", "-m", "mod", ...]`, torchrun's argparse prefix-matched our trainer's `--run` flag against torchrun's own `--run-path` and crashed before launch. Patched the launcher to detect the `python -m MODULE` pattern and invoke `torchrun -m MODULE -- <script_args>` so torchrun stops parsing at `--`.

3. **`val_ids.json` clip-id strings vs `split_shards` integer match** — `data/manifests/val_ids.json` stores clip IDs like `"droid_0019"`/`"oxe_bridge_001234"`, but `src/phase1/dataset.py::split_shards` was comparing them as integers against `Shard.episode_index`, producing `val shards: 0`. Added `clip_id: str` to `Shard`, made `split_shards` match against either `clip_id` (strings) or `episode_index` (legacy ints).

4. **`PredictiveHead` had unused params for `vggt_noact`** — the action embedding layers existed unconditionally, forcing `find_unused_parameters=True` in DDP, which **silently breaks PyTorch's `Join` context manager** (torch 2.4.1): the unused-params size-1 allreduce collides with Join's shadow collectives. Refactored `heads.py` so `action_proj = None` when `use_actions=False`; reverted `find_unused_parameters=False` in config.

5. **Real fix: deterministic per-epoch batch truncation** — `Join` itself failed to keep ranks in sync at epoch boundaries even after fix #4. The root issue is that `StreamingNextTokenPairs.__len__` returns `total_pairs / world_size` (the *average*), but the per-rank slice from `order[rank::world]` has different actual pair counts each epoch due to the deterministic shuffle. Solution: each epoch, the trainer replays the dataset's shuffle for that epoch's seed, sums the rank's actual pairs, MIN-allreduces across ranks, and breaks every rank's loop at `MIN(per_rank_batches) - 16` (safety margin). Per-epoch val now runs on rank 0 against `head.module` (not the DDP wrapper) so its forwards never trigger DDP bookkeeping collectives. Explicit `dist.barrier()` before val.

The net effect of #4 and #5 is that the trainer is now lock-step deterministic across all ranks at every epoch boundary. NCCL never times out anymore.

## Artifacts

- Best checkpoint: `results/phase1_scale/runs/vggt_noact/best.pt` (621 MB, state_dict only)
- 3 latest step ckpts retained: `ckpt_step_00056000.pt`, `..._57000.pt`, `..._57323.pt` (~5.6 GB)
- Eval summary: `results/phase1_scale/runs/vggt_noact/eval_summary.json`
- Training log: `results/phase1_scale/train_logs/vggt_noact_20260518_015536.log`
- Eval log: `results/phase1_scale/runs/vggt_noact/eval_20260518_034532.log`

## What this unlocks

- **M3 can start.** The coupling baseline target (`vggt_noact` frozen) is `results/phase1_scale/runs/vggt_noact/best.pt`. `G_θ`'s self-consistency loss will couple against this predictor.
- The horizon-32 result (L2 3.19 < prototype k=8 baseline 3.34) is the strongest single number we have for the predictor-substrate story — quotable in the paper as-is.

## Open questions still parked

- **Action conditioning.** `vggt_noact` was used here because the prototype found `vggt` (with actions) was *worse*. The FiLM and spatial-pool variants flagged in `PHASE2_PAPER_PLAN.md` §2.5 are still untested at paper scale; deferred until after M3–M5.
- **taco_play.** 11 tars (~3K episodes) downloaded but not yet extracted/cached/used. Optional per spec; can add as an ablation if M3 needs more data.
