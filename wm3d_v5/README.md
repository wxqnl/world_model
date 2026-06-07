# WM3D — Native-3D Action-Conditioned World Model (`wm3d_v5`)

WM3D is an **action-conditioned 3D world model** built on frozen VGGT geometry tokens.
Given *past observation + task + a future action chunk*, it predicts *future VGGT tokens,
depth, rough RGB/motion*, and — in v5 — **native 3D geometry** (world points + camera
pose) instead of depth alone. A pretrained Hunyuan-Video model is bolted on as a
text→video renderer (Stage3), controlled by the frozen world model.

> This is the **active** tree. The Python package is imported as `wm3d_v3` (kept for
> checkpoint/config compatibility), but the live source lives **here** in `wm3d_v5/`.
> All training/eval runs on the remote H100 server under `/data/Minko`, never locally.

## Where to start

| You want to… | Go to |
| --- | --- |
| Understand the whole repo | [`docs/README.md`](docs/README.md) — docs index + current-vs-archived map |
| Run / understand training | [`docs/training/300m_stage0_to_stage2_world_pretrain_v2.md`](docs/training/300m_stage0_to_stage2_world_pretrain_v2.md) (canonical recipe) · [`docs/training/TRAINING_FLOW.md`](docs/training/TRAINING_FLOW.md) (one-page map) |
| Read/write a config | [`docs/CONFIGS.md`](docs/CONFIGS.md) |
| Build/validate the data cache | [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) |
| Know what v5 changed vs v3 | [`docs/world_model_v5_changes.md`](docs/world_model_v5_changes.md) |

## Layout

```
wm3d_v5/
  wm3d_v3/        importable package: models · data · training · eval · encoders · policy
  scripts/        active launchers/tools (train · cache · manifest · eval · setup) — flat
  configs/        active YAML configs (v5 native3d · 300m/140m/1b flows · smokes) — flat
  tests/          pytest suite (146 passing)
  docs/           README index + training recipes + reports/ + cleanup/
  archive/        SUPERSEDED scripts/configs, kept for reference (no active references)
  manifests/ results/ external/   data, checkpoints, vendored deps (gitignored)
```

## Training flow (one line)

```
Stage0 visual/geom → Stage1 dynamics → Stage1.5 Hunyuan bridge → Stage2 progress+proposer
  → Stage3 generation (Hunyuan DiT-control, text→video) → Run3 LIBERO/benchmark
```
Native-3D variants (`configs/v5_*`, `scripts/run_v5_*`) swap depth-only supervision for
VGGT world-points + camera-pose. See `TRAINING_FLOW.md` for the stage→launcher→config map.

## Invariants that must not regress

- The editable install must resolve `wm3d_v3` → **this** v5 tree. The Stage3 DiT-control
  launcher exports `PYTHONPATH=$V5_ROOT` (otherwise `wm3d_v3` resolves to the depth-only
  `/data/Minko/world_model/wm3d_v3` sibling, which lacks the DiT-control adapter).
- The Stage3 trainer wraps the adapter forward **and loss** in `torch.enable_grad()`
  because loading the Hunyuan sampler globally calls `torch.set_grad_enabled(False)`.
- Real text conditioning needs `data.load_task_text: true` (trainer sets `--load_task_text`).

## Tests

```bash
cd /data/Minko/world_model/wm3d_v5
PYTHONPATH=$PWD /data/Minko/.venvs/wm3d/bin/python -m pytest tests/ -q   # 146 passed
```

## Housekeeping

This tree was decluttered + documented on 2026-06-07 (no import/behavior/package changes):
superseded experiment lineages moved to `archive/`, junk removed, authoritative docs
added. Per-file keep/archive rationale: [`docs/cleanup/CLASSIFICATION_2026-06-07.md`](docs/cleanup/CLASSIFICATION_2026-06-07.md).
`scripts/` and `configs/` are intentionally **flat** — add new files there, don't recreate
the old per-experiment sprawl.
