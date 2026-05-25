# Session recap — through 2026-04-23

_Written so you can close the session and resume cold. A future Claude Code session should read this first, then jump to `PHASE2_KICKOFF.md` for the execution handoff._

## Short version (one paragraph)

Between 2026-04-20 and 2026-04-23 we took the project from "Phase 1 ready to launch" to "Phase 1 + Phase 2 prototype complete, paper-scale plan written, stretch-scale plan written, docs cleaned up." Nothing is running. Nothing is pending. All changes are committed and pushed to `origin/main` on `git@github.com:wusimo/world_model.git`. The repo is in a clean handoff state.

## What got executed

### Phase 1 (2026-04-20, ~7.5 min wall-clock)

Three predictor runs launched via `run_phase1.sh` detached with `setsid nohup` so tmux closure couldn't kill them (PGID 2035478). GPUs 0/1/4 concurrently.

- `vggt`          — best val 0.0189, L2 k=8 = 3.83
- `vggt_noact`    — **best val 0.0171, L2 k=8 = 3.34** ← best
- `dinov2`        — best val 0.1228, L2 k=8 = 24.11

Take: VGGT beats DINOv2 ~7–8× on rollout L2. Action conditioning regressed (`vggt_noact` > `vggt`). Writeup in `PHASE1_REPORT.md`. Committed as `80d6554`.

### Phase 2 prototype (2026-04-20, ~22 min wall-clock)

Two variants via `run_phase2.sh`, PGID 2386362, GPUs 0/1:

- `flow_only`    — pure flow matching, 6.2 min
- `flow_coupled` — + predictor self-consistency (w=1.0) + physics moment-match (w=0.1), 16 min

Post-hoc comparison via `scripts/phase2/compare_eval.py` (same 24-step sampler, same seed):

| Variant | val_fm ± sem | val_sc ± sem | Δval_sc |
|---|---|---|---|
| `flow_only`    | 0.9518 ± 0.0169 | 0.03921 ± 0.00583 | — |
| `flow_coupled` | 0.9863 ± 0.0212 | **0.02293 ± 0.00024** | **−41.5%** |

Success criterion met. sem on `val_sc` collapsed ~25× under coupling (consistency, not just mean). Writeup in `PHASE2_REPORT.md`. Committed as `3523947`.

### What happened between my sessions (2026-04-21, user-driven)

Between my sessions on 2026-04-20 and 2026-04-22, the user and/or a linter added three important things. I did NOT run these; they're documented in the updated PHASE2_REPORT, PHASE1_REPORT, and PHASE2_STATUS:

- **Pixel-space eval.** Trained a 0.88 M-param token→depth decoder on real Phase 1 cache (80 epochs). Decoded samples from both variants. Result: coupling *regressed* pixel metrics (temporal smoothness 0.096 → 0.180, Wasserstein 0.047 → 0.076). Token-space "easy to predict" ≠ perceptually plausible. Artifacts: `results/phase2/pixel_eval/`.
- **Ablation variants.** `flow_coupled_sched20` (pure-flow-matching for first 20 epochs, then full-strength coupling), `flow_coupled_w05` (weights ÷2), `flow_coupled_w025` (weights ÷4). Script now supports `--w_sc`, `--w_ph`, `--sc_warmup_epochs` flags. **`sched20` is Pareto-optimal** — matches `flow_only` on val_fm (+0.6%), wins on val_sc (−24.5%), and wins on pixel Wasserstein (0.035 < 0.047 baseline). That's the paper-scale default.
- **`vggt_bigact`.** Phase 1 follow-up with `action_embed_dim=256`. Confirms action-conditioning regression is *not* capacity-limited. More capacity → tighter one-step fit + faster rollout drift. Fix space shrinks to FiLM injection or spatial-pool tokens, neither tried yet.

Committed by user across `c8759a6`, `437c4c4`, `720c5b6`, `3861a98`.

## What I wrote this session (2026-04-22 → 2026-04-23)

All commits are on `main`. Latest head: `c35d91a`. Pushed.

### Cross-discipline explainer — `docs/EXPERIMENTS_EXPLAINED.md`

Plain-language doc aimed at hardware engineers / non-ML readers. Sections:
1. What problem we're solving (generative vs predictive world models)
2. Architecture in one paragraph
3. What each phase tested, in plain English
4. Numbers at a glance (two summary tables spanning all runs)
5. Why each statistic (bootstrap CIs, sem, Pearson r, Wasserstein, temporal smoothness)
6. Status per phase
7. Glossary (foundation model, frozen, token, head, flow matching, rollout, ablation, Pareto)

Committed `17b7943`.

### Paper-scale Phase 2 plan — `PHASE2_PAPER_PLAN.md`

Eleven sections. Decomposes H3 (physical coupling value) into a three-arm policy experiment on LIBERO:

- Arm A: real-only
- Arm B: real + unfiltered generated
- Arm C: real + coupling-generated (the claim)

Data: 30 K clips OXE + DROID. Captions via Claude API (~$100). Storage ~400 GB int8. Compute: ~4 K H100-hours, ~3 months wall-clock on 8×H100. Nine milestones M0–M9 with named gates. M2 (scaled D_ψ), M4 (real-VGGT-decoder pixel eval), M7 (policy experiment) are the research gates.

Committed `596b3b9`.

### Kickoff prompt — `PHASE2_KICKOFF.md`

Self-contained enough to paste into a cold Claude Code session. Contains:
- Must-read docs in order (with time budgets)
- Non-negotiables from the prototype (don't relitigate): `vggt_noact` as predictor, `sched20`-style schedule as default, VGGT stays frozen, int16-viewed-as-bf16 storage
- Known gotchas (vggt/ gitignore rule catches `results/phase1/runs/vggt/`, `setsid nohup` pattern for tmux survival, `.detach().item()` warning, DataLoader shard cache at scale)
- First three tasks with acceptance criteria (audit, DDP smoke, 100-clip data dry run)
- Five open questions that need human decisions before significant compute burns

Committed `596b3b9`.

### One-page executive summary — `docs/ONE_PAGER.md`

Single doc, covers:
1. The pitch (one paragraph)
2. Architecture at a glance — Mermaid system diagram
3. Scale positioning — two PNG figures (4-tier bar chart + capability heatmap)
4. Data recipe (one table)
5. Training recipe (stage table + ASCII coupling schedule + Gantt figure)
6. What we leverage (and don't)
7. Path to industry-scale (Phase 3+)
8. Answers to three strategic questions

Three supporting figures generated: `docs/figures/scale_positioning.png`, `capability_matrix.png`, `training_schedule.png`. Committed `4a2bd8f`.

### Strategic Q&A from this session

The user asked three questions; full answers are in `docs/ONE_PAGER.md §7`, but condensed:

- **Q: Action-conditioned generator?** Yes. Add actions as optional conditioning to G_θ, train with classifier-free-guidance dropout p=0.2. One checkpoint, two inference modes (text-only or text+actions). Recommended upgrade to the paper-scale plan.
- **Q: Industry-scale / leverage existing weights?** Deliberately no on scale (that's the positioning; Risk 3 of the original plan). We already leverage ~1.3B params of frozen VGGT + CLIP — the core bet. No useful world-model weights to reuse (Marble/Genie/Cosmos aren't public, wrong latent space anyway).
- **Q: Does this bridge generative + predictive?** Architecturally yes (shared latent, dual heads, coupling loss — the capability matrix row for "shared latent for both" is unique to us). Output-format no (we emit VGGT tokens, not explorable 3D). Scope claim as *methodological unification at medium scale*, not head-to-head with Marble.

### Stretch-scale plan — `docs/STRETCH_PLAN.md`

Answers "what if we had ~50 H100s?"

Budget: 25 K H100-hours over 6–8 weeks. Architecture delta: 2.6 B trainable params (G_θ 1.5 B, D_ψ 500 M, g_φ 100 M, new 500 M token→RGB decoder). Data delta: 400 K clips (OXE full + EgoVideo + SoMeThing + ScanNet). New: a real token→RGB decoder — the single biggest qualitative change vs paper-scale (we can finally publish images, not just metrics). Puts us in the medium-scale tier alongside GAIA-3 and Dreamer V3.

Recommendation: run paper-scale first, stretch immediately after. De-risks the 25 K H100-h commitment and produces two papers (methods + systems) at ~30% extra total compute vs stretch-alone.

Supporting figure: `docs/figures/scale_tiers.png`. Committed `c35d91a`.

### Housekeeping

Same commit `c35d91a`:

- Deleted stale `STATUS.md` (was Phase 0 day-1 progress, fully superseded)
- Moved `WORKING_DOC.md` → `docs/WORKING_DOC.md` (long-form technical reference, not entry-point)
- Moved `EXPERIMENTS_EXPLAINED.md` → `docs/EXPERIMENTS_EXPLAINED.md`
- Rewrote `README.md`: was Phase-0-focused, now has a multi-phase doc map ("I want to... Read this") and updated repo layout
- Updated one cross-reference in `PHASE2_KICKOFF.md`

## Repo state at end of session

**Head commit:** `c35d91a` on `main`, pushed to `origin/main`.

**Clean:** no uncommitted changes. All `results/phase{1,2}/*.log` files and `cache_tokens*/` directories are untracked-but-ignored noise, not pending work.

**Doc map:**

```
Top-level (working docs)
├── README.md                      — entry point with doc map
├── PHASE1_REPORT.md, PHASE2_REPORT.md — results writeups
├── PHASE1_STATUS.md, PHASE2_STATUS.md — handover live-status docs
├── PHASE2_PAPER_PLAN.md           — paper-scale plan with M0-M9 gates
├── PHASE2_KICKOFF.md              — self-contained prompt for future session
├── PAPER.md, REPORT.md            — Phase 0 artifacts (frozen)
├── GENSPARK_PROMPTS.md            — Phase 0 conceptual-figure prompts

docs/ (supporting)
├── ONE_PAGER.md                   — 5-minute executive summary with diagrams
├── EXPERIMENTS_EXPLAINED.md       — plain-language cross-discipline doc
├── WORKING_DOC.md                 — long-form technical reference across all phases
├── STRETCH_PLAN.md                — the 50-H100 medium-scale plan
├── SESSION_RECAP_2026-04-23.md    — this doc
└── figures/
    ├── scale_positioning.png      — used by ONE_PAGER
    ├── capability_matrix.png      — used by ONE_PAGER
    ├── training_schedule.png      — used by ONE_PAGER
    └── scale_tiers.png            — used by STRETCH_PLAN
```

## Five open questions still awaiting human decisions

Reproduced from `PHASE2_KICKOFF.md` because they gate real work and haven't been answered yet:

1. **Which Open X-Embodiment sub-datasets** for paper-scale data? (Action spaces differ; need to pick.)
2. **Paper venue + deadline?** (NeurIPS 2026, ICLR 2027, CoRL 2026 — determines schedule urgency.)
3. **Compute source?** (Shared cluster vs cloud rental; affects scale strategy and ~$12.5K budget question.)
4. **Phase 1 action-conditioning triage: run FiLM + spatial-pool now, or defer?** (Addenda named these as untried.)
5. **Captioning budget** (~$100 for 30 K clips; confirm authority before API calls).

Nothing in this project should advance past environment setup until these are answered. A Claude Code session picking this up should ask the user these before burning meaningful compute.

## For a new session picking this up

1. Read this recap.
2. Read `docs/ONE_PAGER.md` (5 min).
3. Read `PHASE2_KICKOFF.md` (10 min) — it's written to be the execution handoff.
4. If running paper-scale: answer the five questions above with the user, then follow `PHASE2_KICKOFF.md` Task 1 (audit), Task 2 (DDP smoke), Task 3 (data dry run).
5. If running stretch-scale: also read `docs/STRETCH_PLAN.md` and check whether the 50-H100 compute access has been arranged.
6. Reports to update as work progresses: `PHASE2_STATUS.md` (live status), `PHASE2_REPORT.md` (final results). Don't touch `PAPER.md` / `REPORT.md` — those are Phase 0, frozen.

## Git log excerpt (most recent first)

```
c35d91a Docs housekeeping + 50-H100 stretch-scale plan
4a2bd8f Add one-page architecture + plan doc with visualizations
596b3b9 Add paper-scale Phase 2 plan and kickoff prompt
17b7943 Add plain-language project note + sync flow_coupled re-run artifacts
3861a98 Phase 2 coupling ablations + Phase 1 action-conditioning triage       (user)
720c5b6 Polish docs: fix REPORT Exp3 stats, expand WORKING_DOC, log pixel eval (user)
437c4c4 Phase 2 pixel-space feasibility eval                                  (user)
c8759a6 Add WORKING_DOC: high-level summary of Phase 0/1/2 work, ...          (user)
3523947 Phase 2 feasibility prototype: flow-matching generator + coupling loss
80d6554 Phase 1: train predictive head on frozen tokens
3486fe3 Add paper writeup, paper-quality figures, and Genspark figure prompts (Phase 0)
07c3985 Phase 0 feasibility diagnostic: frozen VGGT-1B token evaluation
```
