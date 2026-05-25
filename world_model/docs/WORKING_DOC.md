# GeoPhys-WM Working Doc — Current Results, Architecture, Data

**Scope.** High-level summary of work in `Simo/geophys-feasibility/`, the working
repo for the GeoPhys-WM plan (unified generative + predictive world model on a
frozen VGGT geometric latent, with a physics-coupling module).

**Status snapshot.** Phase 0 feasibility → cautious proceed. Phase 1 predictive
head → VGGT beats DINOv2 by ~7–8×. Phase 2 coupling prototype → −41.5%
predictor self-consistency on held-out windows. Pixel-space feasibility eval →
coupling improves token-plausibility but **degrades** perceptual plausibility
(a meaningful negative finding for the current design).

---

## 0. TL;DR for a coworker glancing at this

- We treat **frozen VGGT-1B** as a 3D-aware image encoder. It emits per-frame
  geometric "tokens" (scene structure + camera + depth).
- On top of those tokens, we train two small heads in our shared latent:
  a **predictor** `D_ψ` (action-conditioned next-state), and a **generator**
  `G_θ` (flow-matching text-conditioned future synthesis).
- A **physics module** `g_φ` and a **predictor self-consistency loss** couple
  them: generated futures must look physically in-distribution AND be easy for
  the frozen predictor to extrapolate.
- **Phase 0**: VGGT tokens respond to scene motion (Pearson r=0.47 manipulation,
  0.52 driving) — weak-to-moderate but above the pivot threshold. Depth drift
  across sliding windows is 5.5% (yellow).
- **Phase 1**: small transformer predictor on frozen VGGT tokens → trained in 7
  minutes on 30 DROID clips; beats a DINOv2 baseline by ~7–8× on L2 at both
  k=1 and k=8 horizons. Action conditioning unexpectedly *regressed* slightly.
- **Phase 2**: flow-matching generator + coupling loss → **−41.5%** predictor
  self-consistency loss on held-out windows vs. the uncoupled baseline, at a
  3.6% cost to pure flow-matching.
- **Pixel-space eval (new)**: token-level plausibility does not transfer to
  pixel space on this prototype — coupled samples decode to *noisier*, less
  temporally smooth depth than uncoupled ones. Suggests the coupling loss
  optimizes a direction the decoder treats as adversarial.

All prototype-scale (30 clips). Numbers establish that the mechanism is live,
not final performance.

---

## 1. System architecture (single picture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FROZEN GEOMETRIC FOUNDATION MODEL                        │
│                                                                              │
│   images [T, 3, 518, 518]  ──►  VGGT-1B aggregator  ──►  tokens [T, 64, 2048]│
│                                     (frozen)          (pooled 8×8 grid, bf16)│
│                                                                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │   shared state space z
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────────┐        ┌──────────────────┐           ┌──────────────────┐
│  Predictor D_ψ    │        │  Generator G_θ   │           │ Physics g_φ      │
│  (Phase 1)        │        │  (Phase 2)       │           │ (Phase 2)        │
│                   │        │                  │           │                  │
│  [z_{t-3..t}, a]  │        │ [text, z_0] → z  │           │ [z_{0..3}] → φ   │
│       ↓           │        │  rectified flow  │           │   latent ∈ ℝ^64  │
│  small xformer    │        │  matching ODE    │           │   (not GT-sup)   │
│  ~20 M params     │        │  ~23 M params    │           │   ~6 M params    │
│       ↓           │        │        ↓         │           │        ↓         │
│   ẑ_{t+1}         │        │   z_gen [T,P,D]  │           │   g_φ(z_gen)  ≈  │
│                   │        │                  │           │   g_φ(z_real)    │
└────────┬──────────┘        └────────┬─────────┘           └────────┬─────────┘
         │                            │                              │
         └────────┐          ┌────────┘                              │
                  ▼          ▼                                       │
           Coupling: L_sc (predictor self-consistency on z_gen)  ←───┘
                 + L_ph (physics moment-match on φ)
```

- **Frozen** = no gradient, no fine-tuning: VGGT-1B, CLIP text encoder (Phase 2),
  and in Phase 2 the already-trained Phase 1 predictor.
- **Trained** = `D_ψ` in Phase 1 only; `G_θ` and `g_φ` in Phase 2 only.
- The `vggt_noact` ablation from Phase 1 is what Phase 2 uses as the frozen
  self-consistency predictor (generated trajectories lack real actions).
- State space is identical across heads so losses are directly comparable and
  the coupling is meaningful.

---

## 2. What each phase produced

### Phase 0 — Feasibility diagnostic (frozen VGGT-1B only)
Goal: verify that frozen VGGT tokens are a viable state space for dynamic
manipulation scenes. 1–2 H100s, ~40 GPU-hours, 4 diagnostics.

| Exp | Question | Result | Rating |
|---|---|---|---|
| 1 | Sliding-window temporal coherence (depth / pose / Chamfer) | Depth rel-err 5.5%, rotation 0.08°, Chamfer 3.4×10⁻² | Depth **Yellow**, pose **Green** |
| 2 | Static vs dynamic reconstruction quality | Smooth quality-vs-motion profile; raw ScanNet AbsRel uncalibrated (expected — VGGT is affine-invariant) | **Green** (no cliff) |
| 3 ★ | Do token deltas correlate with RAFT optical flow? (critical gate) | Median Pearson r = 0.470 (manipulation, n=30), 0.522 (driving, n=11); **22/30 DROID clips p<0.05**; min r = 0.162 | **Yellow** manipulation / **Green** driving |
| 4 | Cross-domain probe (manipulation vs KITTI) | Driving is bimodal (std 0.32 vs 0.16): highway uniform-motion scenes push to r≈0; curated scenes reach r=0.80 | Mixed |

**Verdict (`REPORT.md`).** Cautious proceed. No metric crosses the red/pivot
line; two land yellow. The honest reading is: tokens are motion-aware but not
dynamics-specialized, so a world model on top of them must either (a) fine-tune
VGGT's aggregator on a dynamics objective, or (b) keep VGGT frozen and let a
downstream head absorb the dynamics residual. We chose (b) for Phase 1.

### Phase 1 — Predictive head `D_ψ` on frozen tokens
30 epochs, batch 16, ~20 M-param transformer head, **~7.5 min wall-clock** on
3 GPUs (one GPU per run). Same transformer across runs; only the input
backbone/conditioning varies.

| Run | Val loss | L2 k=1 | L2 k=8 | cos k=8 | \|Δcf\| k=1 |
|---|---|---|---|---|---|
| `vggt` (VGGT + actions) | 0.0189 | 1.73 | 3.83 | 0.9934 | 0.0431 |
| `vggt_noact` (VGGT, actions zeroed) | **0.0171** | 1.54 | 3.34 | 0.9939 | 0.0000 |
| `dinov2` (DINOv2 + actions) | 0.1228 | 13.87 | 24.11 | 0.8160 | 0.0283 |

`|Δcf|` = mean shift in the predicted next-token when the action is counterfactually swapped (0 by construction for `vggt_noact`).

**Findings.**
- **VGGT dominates DINOv2** at every horizon: L2 ~7–8× lower, cosine ~0.12
  higher. DINOv2 overfits severely (train 0.007 vs val 0.123, 17× gap); VGGT
  runs generalize cleanly (train ~0.012 vs val ~0.018).
- **Action conditioning slightly hurts** on this dataset: `vggt_noact` wins by
  ~9% on val loss and ~11% on L2 at every horizon. The action head *does* learn
  (cf_delta ≈ 0.04 is clearly non-zero) but the signal adds more noise than
  it adds conditioning. Suspect under-regularized action embedding, narrow
  action variance in DROID at this scale, or FiLM-vs-additive routing choice.
- **Rollouts degrade gracefully**: L2 doubles from k=1 to k=8 while cosine
  drops only 0.004 — tokens stay nearly parallel even when their magnitude
  drifts. Consistent with the Phase 0 temporal-coherence picture.

Full write-up: `PHASE1_REPORT.md`. Training/rollout plots:
`results/phase1/plot_train.png`, `results/phase1/plot_rollout.png`.

### Phase 2 — Generator `G_θ` + coupling (feasibility prototype)
40 epochs, batch 8, **22 min wall-clock** on 2 GPUs (coupled 16 min, flow_only
6 min). 24 train / 6 val clips → 717 train / 186 val windows. **Not
paper-scale** — purely "does the coupling mechanism work as designed?".

| Variant | val_fm (flow-match) | val_sc (predictor self-consistency) | Δ vs baseline |
|---|---|---|---|
| `flow_only` | 0.9518 ± 0.0169 | 0.03921 ± 0.00583 | — |
| `flow_coupled` | 0.9863 ± 0.0212 | **0.02293 ± 0.00024** | **−41.5% val_sc**, +3.6% val_fm |

**Findings.**
- Coupling does what it's designed to: generated rollouts become dramatically
  easier for the frozen `D_ψ` to extrapolate.
- Variance **collapses ~25×** (sem 0.00583 → 0.00024) — coupling doesn't just
  lower the mean, it makes the generator uniformly predictor-plausible.
- The physics moment-match loss ended near 10⁻⁶. On 30 clips `g_φ` plausibly
  saturated to near-constant; essentially all lift came from `L_sc`. This
  branch needs real-scale data to test.
- Prototype success criterion met → green-light real Phase 2.

Full write-up: `PHASE2_REPORT.md`. Comparison plot:
`results/phase2/plot_compare.png`.

### Pixel-space feasibility eval (2026-04-21)
Question: do predictor-plausible samples also **look** plausible? Trained a
tiny learned decoder `pooled_tokens → 64×64 depth` on the real Phase 1 cache,
then applied it to samples from each generator variant.

**Why not use VGGT's own decoder?** Its depth_head needs the *full multi-layer
aggregator output* (1374 tokens per frame, all layers). We only cached the
last layer pooled to 8×8. The honest-but-small substitute is to learn a
token→depth mapping from real pairs and apply it to generated tokens.

Initial 2-variant result was a **negative finding** — the original
`flow_coupled` regressed on both pixel metrics. The ablation runs
(next section) then **turned this into a positive result** by showing which
coupling schedule closes the pixel-vs-token gap.

### Coupling ablations (2026-04-21)
Three variants run alongside the original to find the best coupling recipe:

| Variant | w_sc | w_ph | sc warmup | val_fm | val_sc | Δval_fm | Δval_sc | Pixel smooth ↓ | Pixel W₁ ↓ |
|---|---|---|---|---|---|---|---|---|---|
| `flow_only` | 0 | 0 | — | 0.952 | 0.039 | — | — | 0.097 | 0.047 |
| `flow_coupled` | 1.0 | 0.1 | 0 | 0.986 | **0.023** | +3.6% | **+41.5%** | 0.180 | 0.076 |
| **`sched20`** | 1.0 | 0.1 | **20 ep** | **0.957** | 0.030 | **+0.6%** | +24.5% | 0.105 | **0.035** |
| `w05` | 0.5 | 0.05 | 0 | 0.982 | 0.023 | +3.1% | +41.1% | 0.197 | 0.092 |
| `w025` | 0.25 | 0.025 | 0 | 0.974 | 0.030 | +2.3% | +24.2% | 0.110 | 0.047 |

**Findings (big).**

1. **`w_sc=1.0` from epoch 0 is overkill.** `w05` matches the original
   `flow_coupled` on val_sc at slightly lower val_fm cost. Half the coupling
   weight was enough.
2. **Schedule-based coupling is a Pareto win on two axes.** `sched20` gives
   val_fm *parity with flow_only* (+0.6% cost) AND the **best pixel-space
   Wasserstein of any variant** (0.035 < 0.047 flow_only baseline). Token-
   space val_sc gain is modest (24.5%) but the bundle is clearly preferable
   for downstream use.
3. **The token-vs-pixel gap is driven by early-training coupling.**
   Qualitatively (`results/phase2/pixel_eval_ablations/compare_grid.png`):
   `flow_only` and `sched20` produce depth maps with persistent large-scale
   structure; `w05` and the original `flow_coupled` produce high-frequency,
   incoherent textures. Coupling applied before flow matching has converged
   pushes `G_θ` into a token-space region that is predictor-plausible but
   decoder-adversarial. Coupling applied after convergence refines the sample
   distribution without breaking perceptual structure.
4. **`w_sc=0.25` is strictly dominated** by `w_sc=0.5`. Abandon.

**Recommendation for real Phase 2**: schedule-based coupling with warmup
~50% of training epochs, then `w_sc=1.0`, `w_ph=0.1`. Verified to beat
flow_only on both token- and pixel-space metrics at 30-clip prototype scale.

Artifacts: `results/phase2/pixel_eval/` (original 2-variant), `results/phase2/pixel_eval_ablations/` (all 4 variants), `results/phase2/ablations_partial.json`.

---

## 3. Model details

### Frozen backbone — VGGT-1B
`src/vggt_wrapper.py`. `facebook/VGGT-1B`, bf16 autocast on Ampere+:
- `encode(images)` → last-layer `aggregated_tokens` list + `ps_idx`. Images are
  518×518, 3-channel, float ∈ [0,1].
- `decode_geometry(tokens, ps_idx, images)` → depth, point_map, camera
  extrinsic/intrinsic, confidences. Used only for Phase 0 diagnostics and the
  one-time cache step; never at Phase 1/2 training time.
- VGGT returns 1374 tokens per frame (camera token + register token + special
  tokens + 37×37 patch grid). We split off the 4 leading specials and
  **adaptive-avg-pool the 37×37 patch grid to 8×8 = 64 patches per frame**,
  each with D=2048.
- Pool rationale: VGGT-World and VGGT-DP both report that a spatial pool of
  the patch grid preserves enough geometric information for downstream tasks,
  while cutting memory ~22× and making the per-epoch training cost trivial.

### Phase 1 predictor `D_ψ` (`src/phase1/heads.py`)
Vanilla pre-norm GELU Transformer encoder:

| Component | Value |
|---|---|
| Input | 4 past mean-pooled frame tokens (D=2048) + 4 past action embeddings (A=7) |
| Token projection | Linear 2048 → 512 |
| Action projection | Linear 7 → 64 → GELU → Linear 64 → 512, added to token at same time step |
| Query slot | Learnable D=512, additively conditioned on the *target* action |
| Positional | Learned over (context + 1) = 5 slots |
| Backbone | 4 × TransformerEncoderLayer, d_model=512, nhead=8, FF=2048, dropout=0.1, norm_first=True |
| Readout | LayerNorm on the query slot → Linear 512 → 2048 |
| Params | ~20 M trainable |
| Loss | MSE on next-frame mean-pooled token |
| Optim | AdamW lr=3e-4, wd=0.01, grad clip 1.0, 200-step linear warmup + cosine decay, 30 epochs |

Action conditioning is the only difference between `vggt` and `vggt_noact`:
the latter skips the additive action path entirely (the query slot starts as
the bare learnable parameter, no tgt-action term).

### Phase 2 generator `G_θ` (`src/phase2/generative.py`)
Rectified flow matching in VGGT token space. Predicts the velocity field of
the ODE that transports a standard Gaussian into real tokens.

| Component | Value |
|---|---|
| Latent | z ∈ ℝ^(T=8 × P=64 × D=2048) = 8 frames of 8×8 grid, D=2048 |
| Input projection | Linear 2048 → 512 on every token |
| Time embedding | Sinusoidal, added to every token (one value per sample) |
| Text conditioning | Frozen CLIP ViT-B/32 pooled output (512-dim); projected and added to every token |
| Init-frame conditioning | Linear 2048 → 512 on the 64 patches of the first frame, broadcast across T (acts as a "static reference" the generator deforms) |
| Positional | Learned over T·P = 512 slots (frame × patch grid) |
| Backbone | 6 × TransformerEncoderLayer, d_model=512, nhead=8, FF=2048, dropout=0.0, norm_first=True |
| Output | LayerNorm → Linear 512 → 2048 → reshape to [B, T, P, D] |
| Params | ~22.6 M trainable |
| Loss | `‖v_θ(z_t, t, c) − (z_1 − z_0)‖²` with `z_t = (1−t)z_0 + t·z_1`, `z_0 ∼ 𝒩(0,I)` |
| Sampler | Euler, 8 steps during training (gradients through the sampler via `sample_with_grad`), 24 steps at eval |
| Optim | AdamW lr=2e-4, 40 epochs |

### Phase 2 physics `g_φ` (`src/phase2/physics.py`)
Latent-variable physics inference. `φ ∈ ℝ^64` is **not** a supervised target —
no dataset provides "friction = 0.3" labels. Instead, `g_φ` is trained jointly
with `G_θ` through a distributional matching loss:

| Component | Value |
|---|---|
| Input | First 4 frames of a window, 4 × 64 tokens at D=2048 |
| Input projection | Linear 2048 → 384 |
| CLS slot | Learnable D=384 prepended |
| Positional | Learned over 4·64 + 1 = 257 slots |
| Backbone | 3 × TransformerEncoderLayer, d_model=384, nhead=6, FF=1536 |
| Head | LayerNorm → Linear 384 → 64 |
| Params | ~6.2 M trainable |
| Loss | `L_ph = ‖mean(φ_real) − mean(φ_gen)‖² + ‖std(φ_real) − std(φ_gen)‖²` (first + second moments per batch) |

This is a deliberately weak physics "inference" — more a latent summary than
an explicit parameter readout. The plan is to swap for MMD or a richer
structural match at paper scale.

### Phase 2 coupling (`src/phase2/coupling.py`)
Two losses composed on top of flow matching:

1. **Predictor self-consistency `L_sc`** (weight 1.0). Generate a trajectory,
   mean-pool to per-frame tokens, run the frozen `vggt_noact` predictor one
   step at a time with zero actions, report MSE. Low when the trajectory looks
   like a plausible evolution under `D_ψ`.
2. **Physics moment match `L_ph`** (weight 0.1).

Total: `L = L_fm + 1.0·L_sc + 0.1·L_ph` in the `flow_coupled` variant, `L_fm`
alone in `flow_only`.

Gradients flow through the 8-step Euler sampler that produces `z_gen`, so the
coupling is actually able to shape what `G_θ` samples. This is the single most
important implementation detail in Phase 2.

### Pixel decoder `TokenDepthDecoder` (new, `src/phase2/pixel_decode.py`)
Tiny learned stand-in for VGGT's depth head, for the pixel-space eval only:

| Component | Value |
|---|---|
| Input | pooled tokens [P=64, D=2048] reshaped to [D=2048, 8, 8] |
| Stem | 1×1 conv → 128 |
| Upsample | 3 × (ConvTranspose 2× stride-2 + 3×3 conv + GELU), channels 128 → 128 → 64 → 32 |
| Head | 1×1 conv to 1 channel + softplus (depth ≥ 0) |
| Output | 64×64 depth map |
| Params | ~0.88 M |
| Target | unit-median-normalized cached VGGT depth (shape-only) |
| Loss | L1 |
| Optim | AdamW lr=3e-4, wd=1e-4, 80 epochs |

---

## 4. Data details

All data frozen against JSON manifests under `data/manifests/`; raw frames live
outside the repo (`.gitignore`). Total disk <50 GB.

### Set A — manipulation (primary)
- **Source**: `lerobot/droid_100` on Hugging Face (30 episodes, public).
- **Selection**: third-person camera views; skip wrist-cam-only and
  severe-motion-blur clips; no camera cuts.
- **Size used**: 30 clips × up to 128 frames each = ~3,840 frames.
- **Actions (A=7)**: DROID end-effector deltas (x, y, z, rx, ry, rz, gripper),
  normalized to zero-mean unit-std using training-split statistics.
- **States (S=7)**: same space as actions (position + gripper state).
- **Task strings**: short natural-language instructions ("pick up the cup",
  "close the drawer"). Empty strings get `"robot manipulation task"` as
  placeholder. Used only in Phase 2 as frozen CLIP text conditioning.

### Set B — static scenes (Phase 0 baseline only)
- **Source**: ScanNet subset (20 scenes × ~20 frames; ScanNet requires a signed
  TOS form).
- **Role**: Exp 2 baseline with GT depth (uint16 mm). Used to demonstrate that
  VGGT outputs are affine-invariant and that evaluation needs median-ratio or
  affine alignment.

### Set C — driving (Phase 0 cross-domain only)
- **Source**: KITTI subset (11 drives from `2011_09_26_*`, forward-facing
  camera, up to 30 frames each).
- **Role**: Exp 4 cross-domain probe. Intentionally includes both highway
  uniform-motion drives (which break token-flow correlation) and curated
  dynamic-agent drives (which have the cleanest r signal in the whole study,
  up to r = 0.80 on drive 0009).

### Token cache (Phase 1/2 shared; `results/phase1/cache_tokens/`)
Built once by `src/phase1/cache.py`. One `.npz` per clip:

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `tokens` | (n_frames, 64, 2048) | int16 (bitcast of bf16) | pooled VGGT tokens |
| `depth` | (n_frames, 64, 64) | fp16 | downsampled VGGT-predicted depth |
| `extrinsic` | (n_frames, 3, 4) | fp32 | camera extrinsic (VGGT-predicted) |
| `intrinsic` | (n_frames, 3, 3) | fp32 | camera intrinsic (VGGT-predicted) |
| `conf` | (n_frames,) | fp32 | mean per-frame VGGT confidence |
| `actions` | (n_frames, 7) | fp32 | DROID end-effector delta |
| `states` | (n_frames, 7) | fp32 | DROID end-effector state |
| `frame_ids` | (n_frames,) | int32 | absolute frame index in the source clip |

Plus a `.json` sidecar with episode_index, task, fps, and the window plan.
Both Phase 1 and Phase 2 read this cache directly; neither re-runs VGGT. The
bf16→int16 bitcast is numpy-friendly (numpy has no bf16) and **lossless**.

Sliding-window config: W=8 frames, stride=4 (50% overlap). Up to 128 frames
per clip → ~16 windows per clip → 717 train / 186 val windows at the episode
split below.

### Split
| Split | Episodes | Phase 1 pairs | Phase 2 windows |
|---|---|---|---|
| Train | 24 (all except 3, 9, 14, 19, 24, 29) | ~2,900 (C=4) | 717 |
| Val | 6 (episodes 3, 9, 14, 19, 24, 29) | ~700 | 186 |

Same split across Phase 1 and Phase 2, so the frozen Phase 1 predictor is
never evaluated on data it trained on during Phase 2.

### DINOv2 parallel cache (`results/phase1/cache_tokens_dinov2/`)
Identical pipeline with `vit_base_patch14_dinov2.lvd142m` at 518×518, pooled
to 8×8, D=768. Used only for the Phase 1 baseline comparison.

### What's *not* in the cache
- Raw frames (outside repo; re-fetchable from DROID via HuggingFace).
- Full multi-layer VGGT aggregator outputs (we only cached last-layer + pooled).
  This is why the pixel-space eval uses a learned stand-in decoder instead of
  VGGT's own depth head.
- Ground-truth physical parameters — we don't have any; `g_φ` is latent-only.

### Data availability for scaling up
- Phase 1 paper-scale needs 50–200K episodes. Accessible via Open X-Embodiment
  (public, CC-BY), DROID full (~76K trajectories, license-gated), BridgeData V2.
- Phase 2 paper-scale needs 10–50K (text, scene) pairs. ScanNet + ARKitScenes
  with captions + bootstrap from Marble/Cosmos API.
- Phase 3 needs only self-generated data on top of Phase 2.

---

## 5. Repo map

```
configs/{phase1,phase2}/default.yaml   # one-place config per phase
data/manifests/set_{a,b,c}.json        # frozen clip lists
src/vggt_wrapper.py                    # VGGT encode/decode
src/{data_loader,metrics,viz}.py       # Phase 0 primitives
experiments/exp1..4.py                 # Phase 0 diagnostics
src/phase1/{cache,dataset,heads,train,eval}.py
src/phase2/{generative,physics,coupling,dataset,text_encoder,pixel_decode}.py
scripts/phase2/{train_generative,compare_eval,pixel_eval}.py
run_phase1.sh, run_phase2.sh           # multi-GPU orchestration
REPORT.md                              # Phase 0 synthesis (numbers match the CSVs)
PHASE1_{REPORT,STATUS}.md              # Phase 1 results + restart guide
PHASE2_{REPORT,STATUS}.md              # Phase 2 results + restart guide
PAPER.md                               # Phase 0 paper draft
results/                               # metrics, plots, ckpts (gitignored)
  └── phase2/pixel_eval/               # new: decoded-depth eval
```

## 6. Headline numbers

- Phase 0 token-flow Pearson r: **0.470** manipulation (n=30, 22/30 p<0.05) /
  **0.522** driving (n=11).
- Phase 0 depth drift across a 4-frame overlap: **5.5%**.
- Phase 1 VGGT vs DINOv2 next-token L2 at k=8: **3.83 vs 24.11** (~6.3×).
- Phase 1 action conditioning: not capacity-limited. Larger action embedding
  (64→256) improves k=1 by 3% but regresses k=8 (3.83→4.03); still loses to
  `vggt_noact` everywhere.
- Phase 2 coupling (original `flow_coupled`, w_sc=1.0 from ep 0): **−41.5%**
  val_sc vs flow_only, +3.6% val_fm cost. But **−86%** pixel-space perceptual
  quality (temporal smoothness 0.097 → 0.180).
- Phase 2 coupling (`sched20`, warmup 20 epochs then w_sc=1.0): **−24.5%**
  val_sc, **+0.6%** val_fm cost, and **BEST pixel-space Wasserstein of any
  variant (0.035, better than flow_only's 0.047)**. Best overall variant.
- Phase 2 coupling (`w05`, w_sc=0.5 from ep 0): matches `flow_coupled` on
  val_sc at lower val_fm cost. Confirms original w_sc=1.0 was overkill.

## 7. Open questions carried forward

1. **Phase 1 action conditioning is not a capacity problem.** Triage tested
   a 4× larger action embedding (64→256) with the `vggt_bigact` run: slight
   improvement on k=1 L2 (1.73→1.68) but k=8 L2 regresses (3.83→4.03) and
   still loses to `vggt_noact` everywhere. On 30 clips, the DROID action
   signal is genuinely noisy, not under-represented. For full scale,
   recommend: (a) `vggt_noact`-style architecture as the primary predictor,
   (b) test FiLM vs additive injection as a separate variant, (c) spatial-
   pool tokens instead of mean-pool so actions condition specific grid cells.
2. **Token-vs-pixel gap is resolved by schedule-based coupling.** Ablations
   above; use `sc_warmup_epochs ≈ 0.5 × total_epochs` at full scale.
3. **Does the physics loss become meaningful at scale?** Still open —
   Phase 2's `L_ph` saturated at 10⁻⁶ across all ablation variants, likely
   `g_φ` collapse on 30 clips. Needs real-scale data to test.
4. **Fine-tune VGGT aggregator vs keep fully frozen?** Still open. Phase 0
   yellow on depth drift is the strongest argument for light fine-tuning.
5. **Can we reinstate VGGT's own depth head?** The learned decoder stand-in
   is cheap but imperfect. Re-caching full multi-layer aggregator outputs
   for the val set (~6 clips, ~20 min) would let us use VGGT's real
   depth_head on generated samples for the gold-standard perceptual eval.

## 8. Current next actions (prototype → paper scale)

Most of the load-bearing prototype questions are now answered:

- ✅ Phase 1 action-conditioning triage: not a capacity issue. Recommend
  `vggt_noact`-style as primary; test FiLM/spatial-pool separately.
- ✅ Schedule-based coupling tested (`sched20`): Pareto win on both
  token- and pixel-space metrics.
- ✅ Coupling-weight sweep done: `w_sc=0.5` suffices, `w_sc=0.25` dominated.
- ⚠️ Pixel-space eval: positive on `sched20` but still uses a learned decoder
  stand-in, not VGGT's own depth head.
- ⏳ Larger-data coupling (500–2K clips) not yet run; would tell us whether
  `L_ph` becomes load-bearing and whether the schedule recipe generalizes.

**Remaining before committing to full Phase 1** (~1 week, <300 GPU-hours):

1. **Re-cache full multi-layer aggregator outputs on the val set** (~6 clips,
   ~20 min on 1 H100) so we can rerun pixel-eval with VGGT's real
   depth_head. Confirms the `sched20` pixel-space win holds under the
   gold-standard decoder.
2. **Phase 1 FiLM variant** (`vggt_film`) — replace additive action injection
   with FiLM. One more cheap run, tells us whether action routing rather
   than capacity was the issue.
3. **Mid-scale Phase 2 sanity** (500–2K clips, `sched20` variant only,
   ~2 days of training). Tells us whether `L_ph` wakes up at scale and
   whether the schedule recipe still works at larger data diversity.

If all three look positive, commit to full Phase 1 per the compute budget
(~3.5–5K H100-hours, 2–3 months on 4–8× H100). **The Phase 2 recipe is now
pinned**: schedule-based coupling with warmup ~50% of total epochs,
`w_sc=1.0`, `w_ph=0.1`. No weight sweep needed at scale.
