# World Model v5 Changes

v5 keeps the existing v3 training spine and package imports, but changes the geometry path from depth-only supervision toward VGGT-native 3D supervision.

## What changed from v3

- VGGT cache can now write optional geometry extras into `vggt_geom/<clip>.npz`: `world_points`, `world_points_conf`, `pose_enc`, and `depth_conf`, alongside the legacy `depth` key.
- `OXEWindowDataset` reads those extra fields when present and emits future-window targets: `point_tgt`, `point_conf_tgt`, `pose_geom_tgt`, and `depth_conf_tgt`.
- Old depth-only caches remain readable. Set `data.require_geom_extra: true` only for runs that must fail fast when point/pose targets are missing.
- `compute_losses` now uses real `out["point"]` vs `point_tgt` and `out["pose_geom"]` vs `pose_geom_tgt` losses, with confidence weighting where available. Missing targets contribute zero and log active/missing metrics.
- Stage2 progress-like targets are no longer fabricated by default. `progress_tgt`, `terminal_success_tgt`, and `plausibility_tgt` are emitted only from real manifest/cache fields, unless `data.allow_pseudo_progress_targets: true` is explicitly set.
- Direct-policy-only training now forwards rich state fields to `action_policy` when present: `lowdim_state`, `object_state`, `plan_state`, `action_history`, and a compact `progress_state` derived from `progress_tgt`.
- `OXEWindowDataset` can now load optional policy-state caches from `vggt_geom` keys or sidecar subdirectories (`lowdim_state/`, `object_state/`, `plan_state/`, `action_history/`). When an action-history cache is absent, it synthesizes `action_history` from past cached actions. Set `data.require_policy_state: true` to fail fast instead of silently omitting required rich-state fields.
- Training supports an optional `train.weighted_sampler` block for per-dataset weighting without duplicating manifest rows.
- `world3d_claim_eval` now reports optional `world_point_l1` and `camera_pose_enc_mse` when point/pose targets are present, so native-3D evaluation is not limited to depth dynamics.

## Cache requirement before v5 native-3D training

Before using a config with `data.require_geom_extra: true`, backfill the VGGT geometry cache. Example:

```bash
cd /data/Minko/world_model/wm3d_v5
PYTHONPATH=. python scripts/cache_oxe.py   --manifest /data/Minko/world_model/wm3d_v5/manifests/oxe_droid20k_balanced_world_v2.jsonl   --cache_root /data/Minko/datasets/cache/wm3d_v3   --geom_extra
```

Use the same sharding flags as the existing cache jobs (`--shard`, `--world`, `--batch_frames`). `--no_geom_extra` keeps legacy depth-only behavior. If a VGGT head is unavailable, the cache script warns and writes the fields it can produce; runs with `require_geom_extra: true` will then fail fast at dataset time.

Before launching training, run a light preflight by constructing `OXEWindowDataset` with the target config or by calling `scripts.cache_oxe.cache_complete(..., need_geom_extra=True)` for manifest clips. The validator checks key presence, shapes, frame counts, and token/action/rgb alignment; a depth-only or partial `.npz` is not considered a complete v5 native-3D cache.

## New v5 configs

- `configs/v5_p64_300m_stage0_native3d_oxe_droid20k_balanced_2node_v1.yaml`
- `configs/v5_p64_1b_stage0_native3d_wsd_4node_v1.yaml`

Both enable `model.enable_geom_extra: true`, set `data.require_geom_extra: true`, keep pseudo progress labels disabled, and set nonzero point/pose geometry losses.

## Optional weighted sampler

Example:

```yaml
train:
  weighted_sampler:
    enabled: true
    balance_by_dataset: true
    dataset_weights:
      droid: 2.0
      bridge: 1.0
    replacement: true
```

With `balance_by_dataset: true`, base weights are inverse per-dataset window counts, then multiplied by each record's optional `repeat_weight` and the explicit `dataset_weights` multiplier.

## Direct-policy rich-state notes

For OXE direct-policy runs that declare nonzero `policy_lowdim_dim`, `policy_object_state_dim`, `policy_plan_state_dim`, or `policy_action_history_len`, v5 automatically enables `load_policy_state` from the model config. Existing OXE caches may still lack lowdim/object/plan fields; for a defensible rich-state policy claim, either backfill those fields or set:

```yaml
data:
  require_policy_state: true
```

`action_history` is recoverable from `actions/<clip>.npy`, so it can be trained even before explicit sidecar action-history caches are added.


## Isolation note

`wm3d_v5/` was created as a separate source tree from the current `wm3d_v3/` code, excluding `results/` and runtime caches. The existing `wm3d_v3/` tree is already dirty because it is the active training/development tree; v5 changes should be reviewed and launched from `wm3d_v5/` and should not be treated as an in-place mutation of the running v3 Stage1 job.
