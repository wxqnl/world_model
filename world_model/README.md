# GeoPhys-WM

Unified generative + predictive world model on a frozen VGGT geometric latent, with physics-inference coupling. This repo hosts the feasibility study (Phase 0), the predictive head (Phase 1), the generative head + coupling prototype (Phase 2), and the plan for paper-scale Phase 2.

**Current state (2026-04-22).** Phase 0, 1, and 2-prototype all complete. Paper-scale Phase 2 plan is written and committed; not yet executed. Short strategic summary in [`docs/ONE_PAGER.md`](docs/ONE_PAGER.md).

## Where to start reading

| I want to... | Read this |
|---|---|
| Get the whole pitch in 5 minutes | [`docs/ONE_PAGER.md`](docs/ONE_PAGER.md) |
| Understand the project without ML background | [`docs/EXPERIMENTS_EXPLAINED.md`](docs/EXPERIMENTS_EXPLAINED.md) |
| See the detailed technical working notes across all phases | [`docs/WORKING_DOC.md`](docs/WORKING_DOC.md) |
| See what Phase 2 paper-scale entails | [`PHASE2_PAPER_PLAN.md`](PHASE2_PAPER_PLAN.md) |
| Hand off the paper-scale work to Claude Code | [`PHASE2_KICKOFF.md`](PHASE2_KICKOFF.md) |
| See what 50 H100s unlocks (medium-scale tier) | [`docs/STRETCH_PLAN.md`](docs/STRETCH_PLAN.md) |
| Read the Phase 0 feasibility paper | [`PAPER.md`](PAPER.md) + [`REPORT.md`](REPORT.md) |
| See Phase 1 results | [`PHASE1_REPORT.md`](PHASE1_REPORT.md) |
| See Phase 2 prototype results | [`PHASE2_REPORT.md`](PHASE2_REPORT.md) |
| Resume work on an in-flight phase | [`PHASE1_STATUS.md`](PHASE1_STATUS.md), [`PHASE2_STATUS.md`](PHASE2_STATUS.md) |

## Repo layout

```
configs/
  default.yaml                 Phase 0 diagnostic config
  phase1/default.yaml          Phase 1 predictor config
  phase2/default.yaml          Phase 2 generator config
src/
  vggt_wrapper.py              VGGT encode / decode_geometry / full_inference
  data_loader.py               manifest-driven clip loader
  metrics.py, viz.py           shared evaluation utilities
  phase1/                      cache, dataset, heads, training, eval
  phase2/                      generator, physics, coupling, pixel decode
experiments/exp1..4.py         Phase 0 four diagnostics
scripts/
  phase2/                      Phase 2 training + comparison + pixel eval
  smoke_test.py, build_*.py    Phase 0 infra
tests/                         smoke tests for Phase 1 and Phase 2 code
data/
  manifests/*.json             clip lists (checked in)
  README.md                    data sources, licenses, format
results/                       generated outputs (gitignored for bulk, key artifacts committed)
docs/
  ONE_PAGER.md                 executive summary + architecture diagrams
  EXPERIMENTS_EXPLAINED.md     cross-discipline explainer
  WORKING_DOC.md               long-form technical reference
  STRETCH_PLAN.md              the 50-H100 medium-scale plan
  figures/                     generated figures used in the docs above
```

## Install

```bash
bash setup.sh
source .venv/bin/activate
python scripts/smoke_test.py              # Phase 0 sanity check (downloads ~5GB weights first time)
python tests/test_phase1_smoke.py         # Phase 1 sanity
python tests/test_phase2_smoke.py         # Phase 2 sanity
```

Requires: CUDA GPU (Ampere+ for bf16), Python 3.11, ~10 GB for weights + deps.

## Running experiments

### Phase 0 diagnostic (replicate REPORT.md)

```bash
python experiments/exp1_temporal_coherence.py
python experiments/exp2_static_vs_dynamic.py
python experiments/exp3_token_dynamics.py
python experiments/exp4_cross_domain.py
```

### Phase 1 (predictor heads — vggt, vggt_noact, dinov2, vggt_bigact)

```bash
bash run_phase1.sh                         # orchestrator, runs all variants
```

### Phase 2 prototype (flow_only vs flow_coupled vs ablations)

```bash
bash run_phase2.sh                         # orchestrator, runs the base 2 variants
python -m scripts.phase2.compare_eval      # post-hoc held-out comparison
python -m scripts.phase2.pixel_eval        # pixel-space eval via learned decoder
```

See the phase-specific STATUS / REPORT docs for what each run produces and how to interpret it.

## Citing / sharing

This is ongoing research. If you're discussing the project externally, please cite the specific phase report that contains the result you're using — the numbers in the top-level `REPORT.md` and `PAPER.md` are Phase 0 only and do not cover Phase 1 or 2 findings.

## Contact

See the git log for commit authors. The project lives in `/home/user01/Simo/geophys-feasibility` on the shared box.
