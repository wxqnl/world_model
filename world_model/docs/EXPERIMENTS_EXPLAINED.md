# Project notes for cross-discipline readers

_Last updated: 2026-04-22. Written for colleagues (hardware engineers, non-ML researchers) who want to understand what this project is testing, why, and how to read the numbers without an ML background._

## 1. What problem we're actually solving

A "world model" is a neural network that learns a compact internal representation of how the physical world looks and behaves. There are two camps in 2026:

- **Generative world models** (World Labs Marble, DeepMind Genie 3). Given a text prompt like "a kitchen with a red mug on the counter," they synthesize a 3D scene you can walk through. They're good at *making up* plausible-looking environments. They are *not* good at answering "if the robot arm pushes this mug 5 cm right, what happens next?"
- **Predictive world models** (DeepMind Dreamer, Wayve GAIA-3, Meta V-JEPA 2). Given a video clip and an action, they predict what the next frame/state will look like. They're good at *forecasting dynamics*. They are *not* good at generating scenes from scratch.

These two capabilities are converging toward the same goal — a single model that can *invent* an environment and *evolve* it under agent actions — but they're built on different representations of the world, and no one has successfully merged them.

**Our bet:** there's a recent class of models called *geometric foundation models* (the one we use is called **VGGT**, from CVPR 2025) that output a compressed description of the 3D structure of a scene. That description is rich enough to regenerate the scene from scratch AND structured enough to predict how it'll evolve. If true, it's the shared "latent space" both camps have been missing.

This repo is the **feasibility study** for that bet. We're not training a giant model. We're running a series of small, targeted experiments to see if the bet holds.

---

## 2. The architecture in one paragraph

Take a video frame. Run it through a **frozen** (= not retrained) VGGT backbone. You get a set of ~64 tokens per frame, each a 2048-dim vector, that together describe the 3D geometry of what's in the frame. Everything we train in this project sits on top of those frozen tokens:

- **Predictor head `D_psi`** (Phase 1): takes 4 past frames' tokens + optional action, predicts the next frame's tokens.
- **Generator head `G_theta`** (Phase 2): takes a text prompt + an initial frame's tokens, synthesizes a short token sequence for future frames.
- **Physics-inference head `g_phi`** (Phase 2): takes a sequence of tokens, outputs a compact "physics summary" vector. Meant to capture things like friction/mass/contact stiffness without ever being told what those things are.

The interesting bit is the *coupling* between `G_theta` and `D_psi`: we train the generator so that the predictor can extrapolate its output easily. Intuition: real videos are "easy to predict" (they follow physics); if generated videos are also easy to predict, they probably look physical too.

---

## 3. What we've done so far

### Phase 0 — "does VGGT even work on moving scenes?"  (Apr 15–20)

VGGT was trained on *static* 3D reconstruction. Will it still work on videos of robot arms moving stuff around? We ran four diagnostic tests on 30 short clips:

| Test | What it asks | Pass? |
|---|---|---|
| Temporal coherence | Does depth stay consistent as the camera slides? | ~yes (5.5% drift) |
| Static-vs-dynamic | How much worse is reconstruction when things move? | Acceptable |
| Token ↔ flow correlation | Do token changes track pixel motion? | Moderate (r ≈ 0.47–0.52) |
| Cross-domain | Does it hold for driving videos too? | Yes |

**Verdict: cautious proceed.** VGGT's geometry holds up on dynamic scenes well enough to build on. Written up in `PAPER.md` and `REPORT.md`.

### Phase 1 — "can we predict the next frame?"  (Apr 20, ~7 min of GPU)

Trained three tiny predictor heads (~20 M parameters each, transformer architecture). Same data, same training recipe, three different inputs:

- `vggt` — our geometric-foundation-model tokens, with robot-action conditioning
- `vggt_noact` — same tokens but we zero out the actions (ablation: proves whether actions help)
- `dinov2` — a well-known *non-geometric* foundation model (a plain vision transformer). This is the baseline we want to beat.

**Headline:** VGGT crushes DINOv2. VGGT's tokens produce ~7–8× lower rollout error than DINOv2's. This is the core claim of the paper draft we're building toward.

**Surprise:** the ablation (`vggt_noact`) actually *beats* the version with actions. Adding action information makes the predictor worse. We ran a follow-up (`vggt_bigact`, bigger action embedding) on 2026-04-21 to test if this is a capacity problem. It isn't. The DROID dataset's robot-action signals are probably too noisy at 30-clip scale to help. Two untested fixes remain (FiLM conditioning, spatial-pooled tokens) — that's what I recommended as the next step.

Write-up: `PHASE1_REPORT.md`.

### Phase 2 prototype — "can we generate scenes the predictor accepts?"  (Apr 20–21)

Trained a flow-matching generator (22.6 M params) that takes text + an initial frame's tokens and produces 8 frames of future tokens. Two variants:

- `flow_only` — just the generative objective. Baseline.
- `flow_coupled` — generative objective + "predictor self-consistency" loss (penalize when the frozen predictor can't extrapolate the generated frames) + a small physics-matching loss.

**Token-space result:** coupling reduces predictor self-consistency loss by **−41.5%** on held-out windows. Big win in token space.

**Pixel-space reality check (2026-04-21):** we trained a small stand-in depth decoder on real frames and decoded samples from both variants to see if they *look* more real. Coupling actually *hurt* pixel-level metrics: temporal smoothness regressed from 0.096 to 0.180, Wasserstein-to-real from 0.047 to 0.076. **Token "easy-to-predict" ≠ "looks real."**

**Fix (also 2026-04-21):** we tried turning on the coupling loss only *after* the first 20 epochs of pure generative training (`flow_coupled_sched20`). That variant is Pareto-optimal — better than both `flow_only` and `flow_coupled` on basically every metric we care about. Turns out early coupling pushes the generator into a token region the predictor likes but the decoder finds adversarial. Late coupling refines instead.

Write-up: `PHASE2_REPORT.md`.

---

## 4. The numbers at a glance

Summary of all substantive runs, in the most-important-first order. All numbers are held-out (not training) losses. Lower is better everywhere except "cos" (cosine similarity: higher is better).

**Phase 1 — predictor heads** (30 DROID clips, 24 train / 6 val):

| Run | best val loss | L2 rollout @ k=8 | cos @ k=8 | take |
|---|---|---|---|---|
| `vggt`        | 0.019 | 3.83  | 0.993 | VGGT is the substrate |
| `vggt_noact`  | **0.017** | **3.34** | **0.994** | actions don't help here |
| `vggt_bigact` | 0.018 | 4.03  | 0.993 | bigger action head doesn't help either |
| `dinov2`      | 0.123 | 24.11 | 0.816 | baseline, much worse |

**Phase 2 — generator heads** (same dataset):

| Run | val_fm ↓ | val_sc ↓ | pixel smoothness ↓ | pixel Wasserstein ↓ |
|---|---|---|---|---|
| `flow_only`               | 0.952 | 0.039 | 0.096 | 0.047 |
| `flow_coupled` (original) | 0.986 | **0.023** | 0.180 | 0.076 |
| `flow_coupled_sched20`    | **0.957** | 0.030 | 0.105 | **0.035** |
| `flow_coupled_w05`        | 0.982 | 0.023 | 0.197 | 0.092 |
| `flow_coupled_w025`       | 0.974 | 0.030 | 0.110 | 0.047 |

`flow_coupled_sched20` is the winner: it's the *only* variant that matches `flow_only` in token quality AND beats it in pixel-quality. That tells us *when* you turn on a coupling loss matters more than *how strong* it is.

---

## 5. Why the statistics we use

A few metrics show up repeatedly. Here's what they mean and why they're the right tool:

### Bootstrap 95% confidence intervals (Phase 1 rollout table)

When you have a sample of losses — say 696 (context, next-frame) pairs from the validation set — you get a mean loss, but you also want to know *how confident* that mean is. The standard way to ask "is this difference real or noise?" is to compute a 95% confidence interval.

We use **bootstrap resampling** (1000 iterations) instead of assuming the losses are normally distributed. Bootstrap = repeatedly sample the 696 values with replacement, compute the mean each time, and use the 2.5th/97.5th percentile of the resulting means as the CI. This doesn't assume the distribution of losses has any particular shape — robust for heavy-tailed ML errors.

**How to read it:** if two runs have non-overlapping 95% CIs, the difference is almost certainly real. In Phase 1, VGGT and DINOv2's CIs don't come close to overlapping anywhere, which is why the 7–8× gap is a robust claim.

### Standard error of the mean (sem) (Phase 2 comparisons)

Standard error = std / √n. It measures how precisely we know the *mean* of a metric, given n samples. Smaller sem = more reliable mean estimate.

The most interesting single finding in Phase 2 is that the sem on `val_sc` drops ~25× under coupling (0.006 → 0.0002). That means: the coupled generator doesn't just have a lower average self-consistency error — it has a *much more consistent* error across held-out windows. Some windows don't trip it up while others do; all windows behave similarly. For a hardware analog: this is like a chip that's not only fast on average but also has tight clock-jitter — both matter.

### Pearson correlation coefficient (Phase 0 token ↔ flow)

Pearson r measures linear correlation between two quantities, ranging −1 to +1. We computed r between (change in VGGT tokens across adjacent frames) and (optical-flow magnitude, a standard measure of pixel motion). If tokens encode motion, r should be high.

We got r ≈ 0.47–0.52. That's "moderate" — enough to confirm tokens carry motion signal, not enough to say tokens are a *perfect* proxy for motion. This matches the "yellow-zone" verdict in Phase 0.

### Wasserstein distance (Phase 2 pixel eval)

Wasserstein (earth-mover) distance measures how different two distributions are. If you imagine each distribution as a pile of dirt, Wasserstein is the minimum dirt you'd have to move to make one pile into the other.

We used it on the histograms of decoded depth values from generated-vs-real clips. Why not just pixel-wise MSE? Because generators produce *plausible but different* scenes, not pixel-identical ones. A generator that produces a correct-looking kitchen from a different viewpoint should score well even though every pixel differs. Distributional metrics like Wasserstein are the right tool for "does it look statistically real."

### Temporal smoothness

Simply: mean squared difference between consecutive decoded depth frames. Real manipulation videos are *smooth* — things don't teleport. This metric is 0.015 for real clips. If a generator produces 0.180, it's producing jumpy, incoherent motion. Same unit as pixel error; calibrate by eye against the real-data baseline.

---

## 6. Where we are, in one sentence per phase

- **Phase 0** ✅ — VGGT is usable on dynamic robot videos.
- **Phase 1** ✅ — VGGT crushes the non-geometric baseline; action conditioning needs more work.
- **Phase 2 prototype** ✅ — Coupling works if you turn it on at the right time; Pareto winner identified.
- **Phase 2 proper** ⏳ — Needs 10–50K text–scene pairs (we have 30) and a proper VGGT depth decoder.
- **Paper** ⏳ — Phase 1 alone is publishable at NeurIPS 2026 per the original plan. The action-conditioning regression and the coupling-schedule finding are both paper-worthy bonuses.

---

## 7. Glossary (skim as needed)

- **Foundation model**: a large model pretrained on huge generic data, usable as a starting point for many downstream tasks. VGGT and DINOv2 are foundation models; we don't retrain them.
- **Frozen**: the model's weights are loaded and held fixed; we only train the heads on top.
- **Token**: a vector that summarizes a patch of input (a 1/64th of a frame in our case). Don't confuse with NLP tokens.
- **Head**: a small trainable neural network that consumes tokens and produces a task-specific output (next-token prediction, generation, etc.).
- **Flow matching**: a way to train a generator that's simpler than diffusion. We train it to predict a "velocity" that moves noise into real data; at inference we integrate the velocity with an ODE solver (Euler's method, 24 steps for us).
- **Rollout**: repeatedly applying a predictor to its own output to forecast k frames ahead. Long rollouts are where predictors usually drift and fail.
- **Ablation**: an intentionally-broken variant of a model used to prove a component matters (e.g., `vggt_noact` with zero actions proves whether actions actually help).
- **Pareto optimal / Pareto front**: a setting that isn't dominated by any other on every axis simultaneously. `flow_coupled_sched20` beats `flow_only` on pixel Wasserstein AND matches it on val_fm; neither is strictly better than the other on all axes, but the sched20 variant is strictly better than `flow_coupled` on everything. That makes it Pareto-optimal within this ablation set.
