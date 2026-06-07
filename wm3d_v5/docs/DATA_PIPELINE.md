# Data Pipeline

From raw OXE/DROID episodes to training-ready windows. All paths are on the H100 server.

```
OXE/DROID episodes
  -> manifest (jsonl, one record per clip)              scripts/build_*_manifest*.py
  -> VGGT + RGB + geometry cache (per clip)             scripts/cache_oxe.py [--geom_extra]
  -> window targets at load time                        wm3d_v3/data/window_dataset.py
  -> preflight validation                               scripts/validate_native3d_window_cache.py
```

## 1. Manifest

A manifest is a jsonl list of clips with dataset tags and frame counts. Canonical builders:

- `build_oxe_droid_balanced_manifest_v1.py` — rebalances a source manifest across
  datasets (e.g. `fractal=0.25,bridge=0.25,droid=0.25,small_robot=0.25`) to a target
  record count. Produces e.g. `manifests/oxe_droid20k_balanced_world_v2.jsonl`.
- `build_stage1_oxe_droid_manifest.py`, `build_oxe_trainable_manifest.py`,
  `build_v5_native3d_experiment_manifest.py` — stage/experiment-specific manifests.

Manifests are gitignored (large, regenerable). After building, sync cache files to all
nodes with `scripts/sync_oxe_droid_cache_for_manifest_v1.sh` /
`scripts/sync_manifest_cache_files_v1.sh`.

## 2. Cache (`scripts/cache_oxe.py`)

Encodes each clip once into `cache_root` (`/data/Minko/datasets/cache/wm3d_v3`):

- **VGGT pooled tokens** → `vggt_pooled/` (`[T, P=64, D=2048]`)
- **RGB** targets, **Qwen** task embeddings, **action stats**
- **Geometry** → `vggt_geom/<clip>.npz`. With `--geom_extra` (default on) writes the
  native-3D fields; `--no_geom_extra` keeps legacy depth-only.

`vggt_geom/*.npz` keys:

| Key | Native-3D? | Meaning |
| --- | --- | --- |
| `depth` | legacy | per-frame depth map |
| `depth_conf` | extra | depth confidence |
| `world_points` | extra | per-pixel 3D world points |
| `world_points_conf` | extra | world-point confidence |
| `pose_enc` | extra | encoded camera pose (trans + quat + fov) |

Helpers: `cache_geom_utils.py` (geom validation), `cache_lerobot_droid_wm3d.py`
(LeRobot/DROID ingestion). `geom_extra_complete()` / `validate_geom_npz(...,
require_geom_extra=True)` gate whether a clip counts as a complete native-3D cache —
a depth-only or partial npz is rejected for native-3D runs.

## 3. Window targets (`wm3d_v3/data/window_dataset.py`)

`OXEWindowDataset` slices clips into `(T, k, stride)` windows and emits future-window
targets. Fields emitted:

| Field | Meaning |
| --- | --- |
| `action_tgt`, `action_tgt_norm` | future action chunk (raw + normalized) |
| `depth_conf_tgt` | future depth confidence |
| `point_tgt`, `point_conf_tgt` | future world points + confidence (native-3D) |
| `pose_geom_tgt` | future camera pose `[B,T,9]` (native-3D) |
| `task_text` | raw instruction string (only when `data.load_task_text: true`) |

Behavior:

- Old depth-only caches stay readable. Native-3D targets are emitted only when present;
  missing targets contribute zero loss and are logged as active/missing metrics — unless
  `data.require_geom_extra: true`, which fails fast.
- Optional policy-state caches (`lowdim_state/`, `object_state/`, `plan_state/`,
  `action_history/`) are loaded when present; `action_history` is synthesized from past
  cached actions if its cache is absent. `data.require_policy_state: true` fails fast
  instead of silently omitting rich-state fields.
- Progress/success/plausibility targets come only from real manifest/cache fields unless
  `data.allow_pseudo_progress_targets: true`.

## 4. Preflight validation (`scripts/validate_native3d_window_cache.py`)

Before a native-3D run, validate the **window** geom cache. It checks per-window key
presence (`pooled`, `depth`, `depth_conf`, `point`, `point_conf`, `pose`, `pose_conf`),
shapes (`T`, `k`, `P`, `D`, `hw`), frame counts, and base-cache readiness (incl. Qwen
task embeddings with `--require_task_emb`). It reports missing windows/records so a
partial cache is caught before GPU time is spent.

```bash
/data/Minko/.venvs/wm3d/bin/python scripts/validate_native3d_window_cache.py \
  --manifest manifests/<your>.jsonl \
  --cache_root /data/Minko/datasets/cache/wm3d_v3 \
  --window_subdir <window_geom_subdir> --require_task_emb
```
