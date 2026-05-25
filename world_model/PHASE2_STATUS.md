# Phase 2 — live status & restart guide

**Scope of this session.** This is a **feasibility prototype** of Phase 2's coupling mechanism, not the full paper-scale Phase 2 (which per the roadmap needs 10–50K (text, scene) pairs and 4–8K H100-hours). We train the generative head `G_theta` and physics inference `g_phi` on the 30 DROID clips already cached in Phase 1, with two variants, to test whether the **predictor-self-consistency + physics-moment-match** coupling reduces held-out coupling losses relative to pure flow matching. If yes → justification to scale to real Phase 2 data. If no → useful negative result; revisit coupling design.

## What's trained / what's frozen

- **Frozen:** VGGT backbone (tokens cached in Phase 1), CLIP ViT-B/32 text encoder, Phase 1 predictor `D_psi` (`vggt_noact/best.pt`).
- **Trainable:** `FlowMatchingGenerator` (~22.6 M params), `PhysicsInference` (~6.2 M params, only in `flow_coupled` variant).

## Two variants

1. `flow_only`     — pure rectified flow matching on VGGT tokens, no coupling.
2. `flow_coupled`  — flow matching + `w_selfconsistency=1.0` + `w_physics=0.1`.

Each variant: 40 epochs, batch 8, ~90 train batches/epoch, ~80 min wall-clock on 1 H100 for coupled (shorter for flow_only).

## Environment

```bash
source /home/user01/Minko/reskip2/.venv/bin/activate
cd /home/user01/Simo/geophys-feasibility
```

GPU policy unchanged: GPUs 0/1/4/5 usually idle. `nvidia-smi` first.

## Checklist

- [x] Scaffolding smoke-tested (`tests/test_phase2_smoke.py`)
- [x] Training pipeline smoke-tested (`--smoke`, 2 epochs × 3 batches)
- [x] Full run: `flow_only`    →  `results/phase2/runs/flow_only/`    (6.2 min, best val_fm 0.938)
- [x] Full run: `flow_coupled` →  `results/phase2/runs/flow_coupled/` (16 min, best val_fm 0.967)
- [x] Comparison eval + write-up → `PHASE2_REPORT.md`, `results/phase2/comparison.json`, `plot_compare.png`

## Commands

Launch both variants in parallel (two idle GPUs):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.phase2.train_generative \
    --cfg configs/phase2/default.yaml --variant flow_only \
    2>&1 | tee results/phase2/train_flow_only.log &

CUDA_VISIBLE_DEVICES=1 python -m scripts.phase2.train_generative \
    --cfg configs/phase2/default.yaml --variant flow_coupled \
    2>&1 | tee results/phase2/train_flow_coupled.log &

wait
```

Smoke (quick sanity):

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.phase2.train_generative \
    --variant flow_coupled --smoke
```

## Prototype success criterion

`flow_coupled` has **lower** mean `val_sc` (predictor self-consistency on held-out generated rollouts) than `flow_only` evaluated on the same sampler and step count. Ideally `val_fm` is comparable (coupling shouldn't tank flow quality) and physics moment match also shrinks.

## How to pick up if things break

1. `cat PHASE2_STATUS.md` — see last checked-off step.
2. `ls results/phase2/runs/` — per-variant `train_log.json`, `summary.json`, `best.pt`.
3. `results/phase2/train_*.log` — per-variant stdout.

Known gotchas:
- Smoke log-accumulator warning was fixed (detach before `.item()`).
- Training uses `sample_with_grad` at `--train_sample_steps=8` Euler steps for the coupling loss — this dominates step time; eval uses 24 steps.

## Session log

- **2026-04-20** — Scaffolding + training script written. Smoke OK (22.6M gen params, 6.2M physics, 717 train / 186 val windows). Config at `configs/phase2/default.yaml`.
- **2026-04-20** — Both variants completed via `run_phase2.sh`. Post-hoc comparison (same 24-step sampler, same seed): `flow_coupled` cuts val_sc by **41.5%** (0.0392 → 0.0229) vs `flow_only`, at 3.6% cost to val_fm. Prototype success criterion met. See `PHASE2_REPORT.md`.
- **2026-04-21** — Pixel-space feasibility eval (`scripts/phase2/pixel_eval.py`): trained a 0.88M-param token→depth decoder on the real Phase 1 cache (80 epochs, best val L1 0.166 on unit-median-normalized depth), decoded 8 windows per variant with a shared seed. Coupling *regresses* both pixel metrics — temporal smoothness 0.096 → 0.180, Wasserstein-to-real 0.047 → 0.076. Token-plausibility ≠ perceptual-plausibility at this scale. Artifacts in `results/phase2/pixel_eval/`.
- **2026-04-21** — Coupling ablations. Extended `scripts/phase2/train_generative.py` with `--w_sc`, `--w_ph`, `--sc_warmup_epochs`. Launched three variants in parallel: `flow_coupled_sched20` (warmup 20, then full-strength coupling), `flow_coupled_w05` (w_sc=0.5, w_ph=0.05), `flow_coupled_w025` (w_sc=0.25, w_ph=0.025). Also running Phase 1 `vggt_bigact` with action_embed_dim=256 to test the action-conditioning regression. Results will land in `results/phase2/runs/flow_coupled_*` and `results/phase1/runs/vggt_bigact/`. Consolidation script: `scripts/phase2/compare_variants.py`.
