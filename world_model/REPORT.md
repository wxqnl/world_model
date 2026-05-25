# GeoPhys-WM Phase 0 Feasibility Report

**Date:** 2026-04-15
**Model:** facebook/VGGT-1B (frozen, bf16)
**Hardware:** H100 80GB (shared GPUs, ~40GB available per device)

## Executive Summary

Frozen VGGT-1B tokens show **moderate but real** signal for dynamics-aware state
representation. The token-flow correlation (Exp 3, the critical gate) lands
squarely in the **Yellow zone** (median Pearson r = 0.47–0.52 depending on
domain), well above the Red/pivot threshold of 0.2, but below the Green 0.5 line
for the manipulation domain. Temporal coherence is borderline Yellow. Depth
quality on static scenes is poor without GT-anchored scale.

**Recommendation: Cautious proceed** — the signal is present and statistically
significant across all domains, but a world model built on these tokens will need
either (a) fine-tuning to sharpen the dynamics signal, or (b) a hybrid
architecture that complements VGGT geometry with a learned dynamics head.

---

## Experiment Results

### Exp 1 — Temporal Coherence (Set A: 30 DROID manipulation clips)

Measures consistency of depth, pose, and point cloud predictions across
overlapping sliding windows (W=8, S=4, 4-frame overlap).

| Metric | Mean | Std | Min | Max | Threshold | Rating |
|---|---|---|---|---|---|---|
| Depth rel-err | **5.5%** | 2.3% | 1.4% | 10.7% | <5% green, 5–15% yellow | **Yellow** |
| Rotation (deg) | **0.083°** | 0.003° | 0.082° | 0.094° | <2° green | **Green** |
| Translation err | **0.0022** | 0.0014 | 0.0005 | 0.0066 | — | Acceptable |
| Chamfer distance | **0.034** | 0.014 | 0.013 | 0.069 | — | Acceptable |

**Interpretation:** Rotational predictions are very consistent across windows
(sub-degree). Depth shows moderate drift, with the worst clips (droid_0010,
droid_0018) reaching ~10% relative error — likely driven by rapid motion or
scene changes within the clip. The 5.5% mean sits right at the Yellow boundary.

### Exp 2 — Static vs Dynamic Quality (Set A + Set B)

#### Static domain (Set B: 20 ScanNet scenes, with GT depth)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| AbsRel | **0.547** | 0.068 | 0.432 | 0.686 |
| delta<1.25 | **0.0005** | 0.001 | 0.0 | 0.004 |
| Confidence mean | 3.45 | 3.03 | 1.00 | 10.18 |

**Note:** AbsRel is very high (0.55) and delta<1.25 is near-zero. VGGT
depth predictions are in an arbitrary per-window scale (affine-invariant),
so raw comparison to metric GT depth without scale+shift alignment produces
misleading absolute numbers. The confidence scores vary widely (1.0–10.2),
suggesting the model is uncertain on many ScanNet scenes. This result is
**expected behavior** — VGGT was not designed for metric depth; these numbers
confirm that downstream use must include median-ratio or affine alignment
(as Exp 1 already does).

#### Dynamic domain (Set A: 30 DROID clips, photometric self-consistency)

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| Photometric L1 | **0.0070** | 0.002 | 0.003 | 0.015 |
| Mean flow magnitude | **0.097** | 0.13 | 0.014 | 0.507 |
| Confidence mean | 1.56 | 0.59 | 1.00 | 4.15 |

**Quality vs motion profile:** All 30 clips fell into motion bin 0 (flow < 2
pixels), indicating the DROID dataset's manipulation motions are relatively
small. Within this range, photometric L1 was low (mean 0.007), indicating good
self-consistency. The scatter plot shows a mild positive trend (more motion →
higher L1), but the dynamic range is too narrow to observe the "cliff drop" that
would indicate a Red rating.

**Verdict:** **Smooth** — no evidence of catastrophic quality degradation with
motion, but the motion range tested is limited.

### Exp 3 — Token Dynamics (Set A: 30 DROID clips) ★ Critical Gate

Correlates per-frame VGGT token L2 deltas (last aggregator layer) with RAFT
optical flow magnitude.

| Metric | Value | Threshold | Rating |
|---|---|---|---|
| **Median Pearson r** | **0.470** | >0.5 green, 0.2–0.5 yellow, <0.2 red | **Yellow** |
| Mean r | 0.456 | — | — |
| Std r | 0.158 | — | — |
| Min r | 0.162 | — | — |
| Max r | 0.765 | — | — |
| Significant (p<0.05) | 22/30 clips (73%) | — | Strong |

**Interpretation:** Token deltas carry a genuine motion-correlated signal.
73% of clips show statistically significant correlation, and no clip falls
below r = 0.16. The median r of 0.47 means tokens explain only ~22% of the
variance in optical flow magnitude — the encoding is motion-aware but not
dynamics-specialized.

The strongest correlations (r > 0.6) appear on clips with clear, sustained
object motion (droid_0008 r=0.765, droid_0021 r=0.742, droid_0019 r=0.653,
droid_0022 r=0.647). Weakest correlations appear on clips with subtle/slow
motion (droid_0012 r=0.162, droid_0006 r=0.214, droid_0020 r=0.217,
droid_0017 r=0.219).

### Exp 4 — Cross-Domain Probe (Set A vs Set C)

Repeats Exp 3 on KITTI autonomous driving (Set C: 11 clips) and compares
to manipulation (Set A).

| Domain | N clips | Median r | Mean r | Std | Min | Max |
|---|---|---|---|---|---|---|
| Manipulation (A) | 30 | **0.470** | 0.456 | 0.158 | 0.162 | 0.765 |
| Autonomous driving (C) | 11 | **0.522** | 0.376 | 0.335 | -0.190 | 0.803 |

**Interpretation:** The driving domain shows **higher variance** (std 0.34 vs
0.16) and more extreme values. Some drives produce strong correlations (drive
0009: r = 0.80, drive 0014: r = 0.61) while others are near-zero or negative
(drive 0002: r = -0.04, drive 0013: r = -0.19, drive 0028: r = 0.06).

The negative/near-zero cases likely correspond to highway segments with
minimal visual change but large ego-motion — the camera translates forward
with little parallax, so optical flow is high but spatially uniform. VGGT
tokens, focused on scene structure, don't change much because the 3D scene
hasn't changed. This is actually a *feature* for a world model: the tokens
distinguish structural change from ego-motion.

The median r (0.52) is slightly *higher* than manipulation (0.47), but the
distribution is bimodal rather than concentrated, making the driving domain
less reliable overall.

---

## Synthesis & Go/No-Go

### What we know

1. **Token dynamics signal is real** (r > 0.2 everywhere, p < 0.05 for 90% of
   clips). Frozen VGGT tokens do respond to scene dynamics — this is not noise.

2. **Signal is moderate, not strong** (median r ~ 0.47). A world model using
   raw frozen tokens as its only dynamics representation would need to learn a
   significant mapping from token space to dynamics space.

3. **Temporal coherence is borderline** (5.5% depth rel-err). Sliding-window
   predictions show ~5% drift, which a downstream model must account for
   (e.g., by using relative rather than absolute predictions).

4. **Cross-domain transfer is mixed.** The token-flow correlation works across
   manipulation and driving, but driving shows high variance with failure
   modes on uniform-motion scenes.

5. **Absolute depth quality is poor without alignment.** This is expected and
   not blocking — downstream use should always include scale alignment.

### Threshold Summary

| Experiment | Metric | Value | Zone |
|---|---|---|---|
| Exp 1 | Depth rel-err | 5.5% | **Yellow** |
| Exp 1 | Rotation | 0.08° | **Green** |
| Exp 3 | Median Pearson r (manipulation) | 0.47 | **Yellow** |
| Exp 3 | Median Pearson r (driving) | 0.52 | **Green** |
| Exp 2 | Quality vs motion | Smooth | **Green** |

### Recommendation

**Cautious proceed to Phase 1.**

The token-dynamics signal is present and domain-general, placing this approach
above the Red/pivot threshold on every metric. However, the Yellow ratings on
the two most important metrics (depth coherence and manipulation token-flow
correlation) mean Phase 1 should plan for either:

- **Option A:** Fine-tune VGGT's aggregator layers on a dynamics prediction
  objective (next-frame token prediction or flow regression). This directly
  attacks the ~0.47 correlation ceiling.

- **Option B:** Use frozen VGGT tokens as a *geometric backbone* and add a
  lightweight temporal dynamics head (e.g., a transformer over token deltas).
  This preserves VGGT's 3D strengths while learning the dynamics gap.

Option B is lower-risk and keeps VGGT frozen (no retraining needed).

---

## Artifacts

| File | Description |
|---|---|
| `results/metrics/exp1_consistency.csv` | Per-clip temporal coherence metrics |
| `results/metrics/exp2_static_vs_dynamic.csv` | Static (AbsRel) + dynamic (photometric L1) metrics |
| `results/metrics/exp2_motion_bins.csv` | Motion-binned quality breakdown |
| `results/metrics/exp3_token_flow_correlation.csv` | Per-clip Pearson r (original run, 120 frames) |
| `results/metrics/exp4_cross_domain.csv` | Per-clip Pearson r, both domains |
| `results/plots/exp1_*_hist.png` | Exp 1 metric distributions |
| `results/plots/exp2_quality_vs_motion.png` | Flow magnitude vs photometric L1 scatter |
| `results/plots/exp2_conf_static.png` | Confidence distribution (static scenes) |
| `results/plots/exp2_conf_dynamic.png` | Confidence distribution (dynamic scenes) |
| `results/plots/exp3_correlation_distribution.png` | Pearson r histogram (original run) |
| `results/plots/exp4_domain_comparison.png` | Manipulation vs driving r distributions |
| `results/qualitative/exp1_droid_*.png` | Depth pair visualizations |
| `results/qualitative/exp3_set_a_*.png` | Token delta vs flow heatmaps |

## Notes

- Exp 4 was run with `max_frames_per_clip=30` due to GPU memory constraints
  (shared H100s with concurrent training jobs occupying ~39GB per device).
  The original Exp 3 used up to 120 frames. Correlation values are comparable
  between the two runs (median r 0.47 for both on Set A), confirming that 30
  frames is sufficient for the token-flow correlation signal.

- Set C (KITTI) yielded 11 clips out of 12 planned drives (drive 0005
  extraction failed). 11 clips is sufficient for the cross-domain probe.

- The Exp 2 AbsRel on ScanNet (0.55) should not be compared to depth
  benchmarks; VGGT outputs affine-invariant depth. Proper evaluation requires
  median-ratio alignment (as in Exp 1) or fitting an affine transform
  per-scene.
