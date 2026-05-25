# Paper-scale Phase 2 plan

_Written 2026-04-22, after the Phase 2 prototype validated the coupling mechanism (see PHASE2_REPORT.md)._

This plan scales the prototype into the paper-grade experiment the original GeoPhys-WM roadmap (Section 7, Phase 2) calls for. It is written to be concrete: named hypotheses, named deliverables, named budgets. It assumes the prototype's findings are load-bearing — `vggt_noact` as the predictor, `flow_coupled_sched20`-style schedule-based coupling as the default — and builds outward from there.

## 1. Research claim we want to publish

**Primary claim.** On shared geometric tokens from a frozen VGGT backbone, a generative head trained with physics-inference-based coupling to a frozen predictor produces scenes that (a) are perceptually plausible under a proper token→pixel decoder, (b) remain stable under long-horizon predictor rollouts (k=16, 32), and (c) improve downstream policy learning when used as augmentation data, compared to both real-data-only and unfiltered-generated baselines.

This is exactly the H3 sub-claim from Section 2 of the original plan, operationalized. It decomposes into three head-to-head experiments, each with a named baseline — see Section 4 below.

**Secondary claims** (bundled in the same paper):
- The predictor substrate story: frozen VGGT tokens dominate DINOv2 for next-token prediction (from Phase 1) and the advantage *grows* with head scale and horizon. (Current Phase 1 uses a 20 M head at k≤8; extend to 100 M and k=32.)
- Scheduled coupling is Pareto-optimal over immediate coupling. The prototype showed this at small scale; we confirm or falsify at paper scale.

## 2. What the prototype tells us to assume going in

1. **Predictor to couple against is `vggt_noact`.** In Phase 1 it was strongest; it also has zero dependency on action quality (a lingering problem — see the 2026-04-21 action-conditioning triage addendum in PHASE1_REPORT.md). Generated rollouts have no actions anyway, so the decision is forced.
2. **Coupling schedule is `sched20`-style** (pure flow matching for ~50% of training, then turn on `L_sc` and `L_ph`). The prototype showed this is the only variant that wins on both token and pixel metrics; we default to it and sweep only around it.
3. **Physics loss `L_ph` at the coupling stage was probably under-powered at 30-clip scale** (magnitudes ~10⁻⁶, saturated). At paper scale we need a data-scale sweep on `phi_dim` and re-check whether g_φ contributes beyond L_sc.
4. **Token-space ≠ pixel-space.** Any paper-scale run must include a real VGGT-decoded evaluation, not a learned stand-in. This is plumbing, not research.
5. **Phase 1 open question is still open**: action conditioning in `D_ψ` regressed, not fixable by capacity. Two untried variants (FiLM, spatial-pool tokens) must be resolved before we lock in `D_ψ` at scale — else the predictor we couple against is provisional.

## 3. Data plan

| Bucket | Purpose | Source(s) | Target size | Status |
|---|---|---|---|---|
| Predictor training | Retrain scaled `D_ψ` | Open X-Embodiment subset (5–10% = ~100–200K episodes) | 30K train / 3K val episodes | Need to acquire |
| Generator training — real | `G_θ` real-video anchor | Same Open X-Embodiment subset, plus DROID-100 (already cached) | ~30K clips, ~1–3 M frames | Need to acquire |
| Generator training — text | Task conditioning | Existing DROID task strings + VLM-captioned frames (Claude or GPT-4V) for clips without captions | 30K (text, scene) pairs | Need to build caption pipeline |
| Policy evaluation | Downstream augmentation benchmark | **LIBERO** (established BC benchmark, 130 tasks, well-documented) | 10 held-out tasks | Public, just clone |
| Out-of-distribution probe | H3 generalization claim | Held-out Franka-Kitchen + a small manual-curation set | 20 OOD clips | Manually curated |

**Storage budget.** VGGT tokens (int16 packed bfloat16, 64 patches × 2048 dim × 2 B) ≈ 256 KB/frame. At 3 M frames ≈ 750 GB. Need NVMe-backed cache (local SSD works; NFS too slow for random reads during training). Compressed to int8 with per-clip scale: halve to ~375 GB.

**Captioning pipeline.** For clips without task strings: sample 4 frames/clip, run through Claude Sonnet via the Anthropic API (~$0.003/clip captioning), post-filter by regex for "robot," "hand," "object." Costs ≈ $100 for 30K clips.

## 4. Three-horned experiment (the paper's main figure)

Policy training: train a small BC (behavior cloning) policy on top of frozen VGGT tokens on a fixed real-data seed. Augment training data three ways:

| Arm | Training data | What it tests |
|---|---|---|
| A. Real-only | 30K real clips | Baseline. |
| B. Real + unfiltered generated | 30K real + 30K samples from `flow_only` | Does generation help *at all*, or is it just noise? |
| C. Real + coupled generated | 30K real + 30K samples from `flow_coupled_sched20` | **The claim.** |

**Metrics** (on held-out LIBERO tasks):
- Task success rate (primary)
- Sample efficiency (success at 25%, 50%, 75%, 100% of real data)
- OOD success rate on the separate probe set

**Statistical protocol.** 5 seeds per arm. Report mean ± 95% bootstrap CI on success rate. Pre-register `w_sc` and `w_ph` values (from prototype: 1.0 / 0.1 post-warmup, sched=20 epochs-equivalent at our training schedule).

## 5. Model sizing for paper scale

| Component | Prototype | Paper scale | Rationale |
|---|---|---|---|
| `G_θ` generator | 22.6 M, 6 layers, hidden 512 | **~300 M**, 20 layers, hidden 1280 | Matches smaller video-diffusion references; still well below industry scale |
| `g_φ` physics inference | 6.2 M, phi_dim=64 | **~20 M**, phi_dim=128 or 256 (swept) | Prototype showed phi_dim=64 saturated at small scale |
| `D_ψ` predictor | 20 M, 4 layers, hidden 512 | **~100 M**, 12 layers, hidden 1024, context=8 | Extends horizons to k=16, 32; incorporates Phase 1 action-conditioning fix |
| VGGT backbone | Frozen, VGGT-1B | Frozen, VGGT-1B | Unchanged (project is adapter-scale by design) |
| Text encoder | CLIP ViT-B/32 (frozen) | CLIP ViT-L/14 (frozen) | ~5× better text embeddings; marginal compute cost (encoder runs once per batch) |
| Decoder (pixel eval) | Learned stand-in, 0.88 M | VGGT's real depth head | Gets us away from the token-vs-pixel gap |

Total trainable params ≈ 420 M. This sits in the "medium-scale world model" tier in Section 8 of the original plan, not "foundation pretraining."

## 6. Compute and time estimate

Unit cost: one H100-hour.

| Workstream | Estimate | Notes |
|---|---|---|
| Token caching | 120 H100-h | 3 M frames × ~15 ms VGGT forward, 4 GPUs × ~10 hr |
| `D_ψ` retraining + Phase 1 triage (FiLM + spatial-pool) | 250 H100-h | Three variants × 80 h each + final scale-up |
| `G_θ` main training (flow_only, flow_coupled_sched, 2 seeds each) | 800 H100-h | ~200 h per run on 8 H100s DDP |
| `G_θ` ablation sweep (w_sc, w_ph, phi_dim grid) | 500 H100-h | 6 settings × short 80-h runs |
| Policy training (3 arms × 5 seeds × 3 data fractions) | 1500 H100-h | LIBERO is light, but 45 runs total |
| Evaluation runs (k=16/32 rollout, pixel eval, OOD) | 200 H100-h | Mostly inference |
| Infra / debug budget | 600 H100-h | ~15% contingency |
| **Total** | **~4000 H100-h** | Matches original roadmap's 4–8K estimate |

**Wall-clock.** On a dedicated 8×H100 node: 4000 H100-h ÷ 8 = 500 hours ≈ **3 weeks of pure compute**. With iteration, debugging, failed runs, and waiting on captioning pipelines: **2.5–3 months elapsed**.

**Cost if renting.** On-demand 8×H100 at ~$25/hr ≈ $12.5K. Reserved/academic access significantly lower.

## 7. Milestones and gates

Each milestone has a named acceptance criterion. A failed gate triggers a scope review, not automatic continuation.

| # | Milestone | Gate | Est. wall-clock |
|---|---|---|---|
| M0 | Environment, distributed training boilerplate, token cache pipeline | `cache_vggt.py` writes 1 K clips reproducibly on 4 GPUs; DDP smoke test on 2 nodes | Week 1 |
| M1 | Full token cache (30 K clips, 3 M frames) | Cache verifies, storage < 400 GB after int8 quant | Week 2 |
| M2 | Scaled `D_ψ` retrained with Phase 1 action-conditioning fix | k=8 rollout L2 ≤ prototype's `vggt_noact` (3.34) AND cf_delta significantly non-zero (> 0.05) on val | Week 3 |
| M3 | `G_θ` baselines: `flow_only` and `flow_coupled_sched` converged | val_fm plateaus; `sched` variant matches `flow_only` on val_fm within ±2% | Week 5 |
| M4 | Pixel-space eval via real VGGT decoder on the two `G_θ` variants | `sched` variant Wasserstein-to-real ≤ `flow_only`'s (prototype showed 0.035 < 0.047 with stand-in decoder) | Week 6 |
| M5 | Ablation sweep: `phi_dim` ∈ {64, 128, 256}; `w_sc` grid {0.5, 1.0, 2.0}; `w_ph` grid {0, 0.1, 0.5}; warmup grid {10, 20, 30} | Pareto curve generated; best setting identified | Week 7 |
| M6 | Policy training infra on LIBERO; BC baseline at ~70% real-data success | Known LIBERO BC reference number reproduced within 5% | Week 8 |
| M7 | Three-horned policy experiment (A/B/C arms × 5 seeds × 3 data fractions) | Arm C > Arm B > Arm A with p < 0.05 on ≥ 2 of the 3 data fractions | Week 10 |
| M8 | OOD evaluation + long-horizon stability | Arm C wins on OOD with ≥ 5-point success-rate margin; drift at k=32 lower than Arm A | Week 11 |
| M9 | Paper draft, figures, ablation appendix | Internal review complete; submission-ready | Week 12 |

**M2 is the first real gate.** If the scaled predictor can't hit prototype-level rollout performance on 30× more data, the whole project's foundation is in doubt. Investigate before proceeding.

**M4 is the second real gate.** If the stand-in-decoder Wasserstein result doesn't reproduce with the real VGGT decoder, the headline "scheduled coupling is perceptually better" story collapses and the paper pivots to a token-space-only claim with appropriate scope reduction.

**M7 is the research gate.** If Arm C doesn't beat Arm B, the primary claim (H3) isn't supported at paper scale. Paper still publishable as a world-model contribution with a negative physics-coupling result, but the story changes substantially.

## 8. Key implementations

Below is a concrete checklist of code changes and new files. This is the "builder's todo," suitable for a kickoff task list.

### Infrastructure
- [ ] **Distributed training wrapper** for `scripts/phase2/train_generative.py`. Switch to `torch.distributed` with DDP or FSDP; add launcher script. Current code is single-GPU.
- [ ] **Mixed precision (bf16)** throughout. Prototype is fp32.
- [ ] **Gradient checkpointing** on the transformer backbones (generator especially — 300 M params × seq_len 8 × 64 patches will OOM without it).
- [ ] **Checkpoint resume** with step-level granularity. Prototype saves best.pt by val; scale needs every-N-step saves + a `--resume` flag.
- [ ] **Logging** to Weights & Biases (or a local TensorBoard) — prototype's JSON log is fine for 40 epochs but not for a multi-week run.

### Data
- [ ] **Open X-Embodiment downloader** — pick a pinned subset (likely the `fractal20220817_data` / `bridge` / `kuka` slices; check licenses), download to local NVMe.
- [ ] **Token cache pipeline at scale** — refactor `src/phase1/cache.py` to: (a) run distributed across GPUs, (b) write to sharded format (webdataset or a similar tar-based format is the robustness play over raw .npz), (c) verify checksums per shard.
- [ ] **Caption pipeline** — Claude API client that caption clips without `meta.task` strings. Rate-limited, resumable, with failure logging.
- [ ] **Filtering pipeline** — drop clips with severe motion blur, camera cuts, or wrist-cam-only views. Re-use Phase 0's `scripts/build_set_a_manifest.py` logic and scale it.

### Model code
- [ ] **`src/phase2/generative.py`**: scale parameters, add RoPE-style positional encoding for longer seq_len if needed, wire gradient checkpointing. Verify flow matching loss is unchanged.
- [ ] **`src/phase2/physics.py`**: parameterize `phi_dim` via config. Add a proper distribution-matching loss (MMD with RBF kernel is the right replacement for the prototype's moment-match; prototype explicitly flagged moment-match as a placeholder).
- [ ] **`src/phase1/heads.py`**: add FiLM and spatial-pool variants (Phase 1 triage cleanup). Settle on a fixed action-conditioning scheme for the scaled `D_ψ`.
- [ ] **`src/phase2/pixel_eval.py`**: replace learned stand-in decoder with a call to VGGT's actual depth head. Need to cache the right intermediate features (multi-layer aggregator) from VGGT during the token caching pass.
- [ ] **`src/phase3/policy/`**: new subpackage. BC policy on VGGT tokens, training loop, LIBERO eval harness. This is new scaffolding not present in the repo.

### Scripts
- [ ] `scripts/phase2_scale/cache_tokens.sh` — launcher for distributed caching
- [ ] `scripts/phase2_scale/train_generator.sh` — launcher for distributed generator training
- [ ] `scripts/phase2_scale/train_predictor.sh` — same, for the retrained `D_ψ`
- [ ] `scripts/phase2_scale/run_policy_experiment.sh` — the three-horned experiment
- [ ] `scripts/phase2_scale/build_final_figures.py` — paper figure generation from results

### Configs
- [ ] `configs/phase2_scale/predictor.yaml` — new scale for `D_ψ`
- [ ] `configs/phase2_scale/generator.yaml` — the 300 M generator config
- [ ] `configs/phase2_scale/ablation_grid.yaml` — the M5 sweep
- [ ] `configs/phase2_scale/policy.yaml` — BC on LIBERO

## 9. Risks (updated from original plan with prototype findings)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Scaled `D_ψ` doesn't improve with capacity + data | Low | High (breaks substrate claim) | M2 is an early gate; if we hit a ceiling at 100 M / 3 M frames, revisit token design (spatial grid vs mean pool) |
| Real VGGT decoder result contradicts stand-in decoder | Medium | Medium (pivots narrative) | M4 gate; paper still has a token-space story if this falls through |
| Policy training on LIBERO is flakier than expected | Medium | Medium (muddies headline) | Pick a known-reference BC implementation; reproduce published numbers before trusting deltas |
| Action-conditioning fix (FiLM / spatial-pool) doesn't beat `vggt_noact` | Medium | Low | Use `vggt_noact` and note this as an interesting open question; paper gains a side-claim about action-conditioning difficulty |
| Storage / data bandwidth becomes bottleneck | Medium | Low | Int8 token quantization; WebDataset sharding; prefetch workers |
| Physics-inference `g_φ` still saturates at scale | Medium | Medium | M5 sweep should catch this; contingency is to remove `L_ph` from the main run and include it only as an ablation (this changes the claim slightly: "predictor coupling is what matters, not physics loss per se") |
| Someone publishes the same idea first | Medium | High | Share the Phase 1 result at a workshop as soon as possible to establish priority |
| Compute allocation delays | High (shared cluster) | Medium | Reserve time blocks; prioritize M0–M2 in the first allocation |

## 10. Scope guardrails (what's NOT in this plan)

- **Phase 3 (closed-loop self-improvement)** is a separate project. Not in this plan, per original roadmap's phasing. Flag for consideration only after Phase 2 ships.
- **Non-VGGT geometric backbones** (MonST3R, CUT3R) are fallback, not primary. Swap only if M2 fails catastrophically.
- **Action-conditioned generation**. Generator stays text+init-frame conditioned. Adding action conditioning to `G_θ` changes the problem and is premature.
- **Sim-to-real for policy**. LIBERO is a sim benchmark; we claim sim results, not real-robot. Real-robot validation is follow-up work.
- **Pixel-space synthesis end-to-end**. We decode tokens to depth for eval, not a full RGB+depth pipeline. That's Phase 3 territory.

## 11. Team shape (for staffing estimates)

Minimum viable team to execute in 12 weeks:

- **1 lead** (architecture, physics-coupling design, paper writing)
- **1 engineer** (distributed training, data pipelines, infra)
- **0.5 collaborator** (evaluation, baselines, policy benchmark)
- **Optional**: VLM-captioning pass can be batch-outsourced

Add ~30% time if this is a single person doing all three.

---

## Ready to hand off

A companion document `PHASE2_KICKOFF.md` is written as a prompt for a future Claude Code session. It contains (a) the must-read order of existing docs, (b) the first three tasks in priority order with acceptance criteria, (c) the non-obvious gotchas learned from the prototype, and (d) open questions that need human decisions before autonomous work proceeds.
