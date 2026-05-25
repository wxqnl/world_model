# Phase 1 Report — Frozen-token next-token prediction

**Date:** 2026-04-20
**Runner:** `run_phase1.sh` (GPUs 0/1/4, ~7.5 min wall-clock)
**Config:** `configs/phase1/default.yaml`, 30 epochs, batch 16, ~20M-param head
**Eval:** k-step rollout at k∈{1,2,4,8}, n=696 samples, 95% CI by bootstrap

## Summary

| Run           | Best val loss | @epoch | L2 k=1 | L2 k=8 | cos k=1 | cos k=8 | |Δ cf| k=1 |
|---------------|---------------|--------|--------|--------|---------|---------|-----------|
| `vggt`        | **0.0189**    | 21     | 1.73   | 3.83   | 0.9977  | 0.9934  | 0.0431    |
| `vggt_noact`  | **0.0171**    | 29     | 1.54   | 3.34   | 0.9979  | 0.9939  | 0.0000    |
| `dinov2`      | 0.1228        | 25     | 13.87  | 24.11  | 0.8838  | 0.8160  | 0.0283    |

(`cf_delta` = mean change in next-token prediction when action is counterfactually swapped. By construction zero for `vggt_noact`.)

## Findings

**1. VGGT features dominate DINOv2 for next-token prediction.** At every horizon, the frozen-VGGT head's L2 is ~7–8× lower and cosine similarity ~0.12 higher than the DINOv2 baseline. DINOv2 also severely overfits (train 0.007 vs val 0.123, 17× gap), while VGGT runs generalize cleanly (train ~0.012 vs val ~0.018). This validates VGGT tokens as a substantially richer substrate for temporal/geometric prediction than 2D self-supervised features.

**2. Action conditioning did *not* improve loss** — `vggt_noact` actually beats `vggt` by ~9% on best val loss (0.0171 vs 0.0189) and ~11% on L2 at every horizon. The action head does learn to respond to actions (cf_delta ≈ 0.04, clearly non-zero), but routing that signal through the predictor adds noise rather than useful conditioning on this dataset/head. Several possible causes to probe in Phase 2:
  - action embedding may be under-regularized relative to the small ~20 M-param head
  - dataset actions may be weakly predictive of token motion at these horizons
  - `cf_delta` is small in absolute terms (~2% of L2 magnitude) — the action axis carries little signal

**3. Phase 0 context.** Phase 0 frozen-VGGT baselines: token-flow Pearson r = 0.47 (manipulation) / 0.52 (driving); depth drift 5.5%. Phase 1's rollout degrades gracefully — L2 doubles from k=1 (1.73) to k=8 (3.83), cosine drops only 0.004 — consistent with the Phase 0 finding that frozen VGGT tokens have stable, predictable temporal structure.

## Success criteria

- ✅ VGGT beats DINOv2 at next-token prediction — **yes, by ~7–8×**
- ❌ Action conditioning reduces loss vs `vggt_noact` — **no, slight regression**

## Plots

- `results/phase1/plot_rollout.png` — L2 and cosine vs horizon, 95% CIs
- `results/phase1/plot_train.png` — train/val curves (log scale)

## Artifacts

- Per-run ckpts, logs, per-sample CSVs: `results/phase1/runs/<run>/`
- Master orchestrator log: `results/phase1/run_phase1.log`
- Per-run stdout: `results/phase1/train_{vggt,vggt_noact,dinov2}.log`

## Suggested next steps (Phase 2 triage)

1. Diagnose action-conditioning regression: try (a) larger action embedding, (b) FiLM vs concat injection, (c) higher-action-variance subset of clips.
2. Scale the head (50–100 M params) to see whether VGGT's advantage grows or saturates.
3. Longer horizons (k=16, 32) to find where VGGT's cosine collapses — currently still 0.99 at k=8.

## Action-conditioning triage addendum (2026-04-21)

Ran `vggt_bigact` — same architecture as `vggt` but `action_embed_dim=256`
(vs 64). Goal: test whether the regression is a capacity issue.

| Run | best val | L2 k=1 | L2 k=8 | cos k=8 | \|cf_delta\| k=1 |
|---|---|---|---|---|---|
| `vggt` (embed=64) | 0.0189 | 1.73 | 3.83 | 0.9934 | 0.0431 |
| `vggt_noact` | **0.0171** | **1.54** | **3.34** | **0.9939** | 0.0000 |
| `vggt_bigact` (embed=256) | 0.0182 | 1.68 | **4.03** | 0.9931 | 0.0455 |

**Finding: the regression is not capacity-limited.** `vggt_bigact` improves
marginally on best val loss and k=1 L2 (3–4%) and becomes slightly more
action-sensitive (cf_delta 0.0431 → 0.0455), but **k=8 L2 regresses**
(3.83 → 4.03) and it still loses to `vggt_noact` everywhere. More capacity
on the action path lets the head fit one-step transitions a hair better
while accumulating error faster over rollouts — consistent with the
hypothesis that DROID actions at 30-clip scale are a noisy predictor of
pooled-token change.

**Recommendation for full Phase 1.** Proceed with `vggt_noact`-style as the
primary predictor. Separately test two variants we haven't tried:
(a) **FiLM injection** (multiplicative conditioning) instead of additive;
(b) **spatial-pool tokens** instead of mean-pool, so actions condition
specific grid cells rather than all cells uniformly.
