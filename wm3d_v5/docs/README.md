# wm3d_v5 Documentation Index

`wm3d_v5` is the active tree of the WM3D world model: an action-conditioned 3D world
model on frozen VGGT geometry tokens, evolving from depth-only (v3) to **VGGT-native 3D**
supervision (world points + camera pose). The Python package is imported as `wm3d_v3`
(name kept for checkpoint/config compatibility) but the live source lives **here** in
`wm3d_v5/`.

## Start here

| Doc | What it covers |
| --- | --- |
| [`training/300m_stage0_to_stage2_world_pretrain_v2.md`](training/300m_stage0_to_stage2_world_pretrain_v2.md) | **Canonical training recipe** — authoritative per-stage policy, configs, launchers |
| [`training/TRAINING_FLOW.md`](training/TRAINING_FLOW.md) | One-page stage ladder: stage → launcher → config → trained/frozen → eval gate |
| [`CONFIGS.md`](CONFIGS.md) | Config schema reference (`model.`/`data.`/`train.`/`loss.`) + config → stage map |
| [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | manifest → VGGT/RGB cache → `OXEWindowDataset` targets → cache validation |
| [`world_model_v5_changes.md`](world_model_v5_changes.md) | What v5 changed vs v3 (native-3D geometry path) |

## Repository map

```
wm3d_v5/
  wm3d_v3/            <- the importable package (models, data, training, eval, encoders, policy)
  scripts/           <- ACTIVE launchers/tools (training, caching, manifests, eval, setup)
  configs/           <- ACTIVE yaml configs (v5 native3d, 300m/140m/1b flows, smokes)
  tests/             <- pytest suite
  docs/              <- this directory
    training/        <- training recipes
    reports/         <- dated status reports & roadmaps (historical record)
    cleanup/         <- cleanup design + file classification manifest
  manifests/         <- data manifests (gitignored: large/regenerable)
  results/           <- training outputs & checkpoints (gitignored: ~161GB)
  external/          <- vendored HunyuanVideo / VGGT (gitignored)
  archive/           <- SUPERSEDED scripts/configs, kept for reference (not in active flow)
```

## Current vs archived

The **active** flow is the staged world-model pretraining ladder
(Stage0 → Stage1 → Stage1.5 → Stage2 → Stage3 generation → Run3 benchmark) plus its
v5 native-3D variants. Everything in `archive/` is a superseded experiment lineage
(v3/v3.5/vla, p256/rgb1b, p64 demo-heads, the LIBERO phasecond/teacher policy
ablations, 500m runs, the pre-DiT Hunyuan adapter, and the 300m fromscratch-v1
attempt). See [`cleanup/CLASSIFICATION_2026-06-07.md`](cleanup/CLASSIFICATION_2026-06-07.md)
for the exact per-file keep/archive decision and rationale.

Nothing in `archive/` is referenced by an active launcher (verified: 0 active→archive
references at archive time). To revive an archived file, `git mv` it back.

## Hard facts that must not regress

- **Package resolution:** the editable install must resolve `wm3d_v3` to **this** v5 tree.
  The Stage3 DiT-control launcher exports `PYTHONPATH=$V5_ROOT` for this reason — the
  editable finder otherwise points at the depth-only `/data/Minko/world_model/wm3d_v3`
  tree, which lacks the DiT-control adapter.
- **Stage3 grad:** loading the Hunyuan sampler globally calls
  `torch.set_grad_enabled(False)`; the DiT-control trainer wraps the adapter forward +
  loss in `torch.enable_grad()` so the (only) trainable adapter actually gets gradients.
- All commands run on the remote H100 server under `/data/Minko`, never locally.
