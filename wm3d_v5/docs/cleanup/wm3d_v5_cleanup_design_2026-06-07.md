# wm3d_v5 Codebase Cleanup & Documentation — Design Spec

Date: 2026-06-07
Status: awaiting approval to execute (Phases 1–4)
Companion: `CLASSIFICATION_2026-06-07.md` (file-level keep/archive manifest)

## Goal

Make the `wm3d_v5` tree clear and maintainable: one obvious place for the current
training flow, configs, and data pipeline; obsolete/abandoned experiments out of the
way; authoritative docs that match what actually runs. No model/behavior changes.

## Hard constraints (approved)

1. **No import / package / behavior changes.** The package stays importable as
   `wm3d_v3`; every checkpoint path, config key, and launcher contract is preserved.
2. **Git baseline first (done).** `wm3d_v5/` is now tracked; baseline committed and
   pushed to branch `wm3d-v5` on `origin` (via REST API — `github.com:443` is blocked
   from this server, `api.github.com` works). Every later change is a reviewable diff.
3. **Dead files are archived, not deleted.** They move to `archive/`, recoverable.
   Only pure junk (`.bak` family, `__pycache__`, `egg-info`) is hard-deleted (git
   baseline retains it anyway).
4. **Classification approved before any file moves.** See companion manifest.

## Current canonical flow (the thing being documented & protected)

`docs/training/300m_stage0_to_stage2_world_pretrain_v2.md` is the canonical recipe.
Pipeline: **Stage0 (visual+geom) → Stage1 (dynamics) → Stage1.5 (Hunyuan bridge) →
Stage2 (progress+proposer) → Stage3 (Hunyuan DiT-control generation) → Run3 (LIBERO /
formal benchmark)**. v5 changes the geometry path to VGGT-native 3D
(`docs/world_model_v5_changes.md`).

## Phases

### Phase 1 — De-clutter (hard-delete junk)
- Remove 11 `.bak/.pathbak/.gradbak/.stage3bak/.pointbak/.orig` files.
- Remove all `__pycache__/` and `wm3d_v3.egg-info/`.
- `.gitignore` already prevents these from returning.

### Phase 2 — Archive dead eras (reversible) → `archive/`
Per the manifest: 41 scripts + 103 configs move to `archive/scripts/` and
`archive/configs/`, preserving basenames. Loose top-level `REPORT_*.md` / roadmap
docs move to `docs/reports/`. Conservative default: anything ambiguous stays.

### Phase 3 — Organize survivors into intent subfolders
```
scripts/
  train/     run_300m_*, run_v5_stage_*, watch_*_flow_*, run_140m_*, *dit_control*
  cache/     cache_oxe.py, cache_geom_utils.py, cache_*_wm3d.py, validate_native3d_*
  manifest/  build_*_manifest*.py
  eval/      run_formal_*_benchmark_*, run_worldvla_*, run_*_native_benchmark_*
  setup/     download_*, setup_*, nccl_smoke.py, run_dist_smoke_*, prepare_worker_*, sync_*
configs/
  v5_native3d/   v5_* + _smoke_v5_* + _eval_v5_*
  flow_300m/     v3_p64_300m_*balanced* + runG + run1_droid_smoke
  flow_140m/     v3_p64_140m_stage[012]_*
  scaling_1b/    v3_p64_1b_stage*
  smoke/         scaling_smoke_* + smoke_300m_revised_*
```
Launcher path references that break on move are fixed in the same commit.
(If sub-foldering configs risks churn in many launchers, fall back to flat `configs/`
with the dead ones archived — decided at execution time per reference count.)

### Phase 4 — Authoritative docs under `docs/`
- `docs/README.md` — index + current-vs-archival map.
- `docs/training/TRAINING_FLOW.md` — per-stage: purpose, launcher, config, trained/frozen
  modules, eval gate. Anchored on the canonical 300m doc + verified stage3 facts
  (`PYTHONPATH=$V5_ROOT`, `torch.enable_grad()` around the adapter forward).
- `docs/CONFIGS.md` — meaningful `model./data./train.` keys (native3d toggles
  `enable_geom_extra`, `require_geom_extra`, `weighted_sampler`, policy-state flags) +
  table mapping each surviving config → stage.
- `docs/DATA_PIPELINE.md` — manifest build → `cache_oxe.py --geom_extra` →
  `vggt_geom/*.npz` keys → `OXEWindowDataset` targets → validation.

### Phase 5 — Verify
- `bash -n` every moved launcher; `py_compile` moved `.py`.
- Run `pytest tests/` — must stay green.
- Re-run the native3d stage3 smoke end-to-end (already green) post-reorg.
- Confirm editable import still resolves `wm3d_v3` → the v5 tree.

## Out of scope (explicitly not doing now)
- Renaming the `wm3d_v3` package.
- Refactoring module internals / merging `joint_model_b/c`.
- Touching `results/`, checkpoints, `external/`, or data caches.
