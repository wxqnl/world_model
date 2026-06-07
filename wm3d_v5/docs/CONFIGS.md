# Config Reference

WM3D configs are YAML with four top sections: `data:`, `model:`, `train:`, `loss:`
(plus `out:`). Configs are consumed by `wm3d_v3/training/train.py`. This page documents
the keys that actually change behavior; see a concrete current config such as
`configs/v5_p64_300m_stage0_native3d_oxe_droid20k_balanced_2node_v1.yaml` for a full example.

## `data:` — dataset & cache

| Key | Meaning |
| --- | --- |
| `manifest` | path to the jsonl manifest of clips |
| `cache_root` | root of the VGGT/RGB/geom cache (`/data/Minko/datasets/cache/wm3d_v3`) |
| `tokens_subdir` | pooled VGGT token subdir (`vggt_pooled`) |
| `action_stats` | npz of per-axis action mean/std for normalization |
| `T` / `k` / `stride` | window length / future horizon / frame stride |
| `load_rgb` / `load_geom` | load RGB targets / geometry (depth) targets |
| `require_geom_extra` | **native-3D switch**: fail fast unless `point`/`pose`/conf targets exist |
| `load_state_tgt` | load progress/object/plan rich-state targets |
| `load_task_text` | emit raw `task_text` strings (required for Stage3 text conditioning) |
| `require_task_emb` | fail fast if Qwen task embeddings are missing |
| `allow_pseudo_progress_targets` | if false, only real progress/success targets are used |
| `require_policy_state` | fail fast if rich policy-state fields are missing |
| `split.mode` | `episode` (episode-level val split, prevents window leakage) |

## `model:` — architecture & enabled heads

`state` / `action` blocks set the dual-stream trunk dims (`hidden`, `n_layers`,
`n_heads`, `P=64` tokens, `D=2048` VGGT dim, `action_cond_dim=7`). Head toggles:

| Key | Effect |
| --- | --- |
| `enable_geom_extra` | **native-3D**: GeomDecoder also predicts `point` + `pose_geom` (not just depth) |
| `geom_upsample_mode` | depth/point upsampler (`resize_conv`) |
| `enable_pixel` / `enable_context_pixel` | rough RGB decoder / context-pixel motion renderer |
| `context_pixel_*` | renderer width, residual scale, motion prediction, action/task conditioning |
| `enable_progress_head` | progress / terminal-success head (Stage2) |
| `enable_action_proposer` | action proposer head (Stage2) |
| `enable_action_policy` | direct policy head (downstream) |
| `enable_world_prior` | in-model text→token generator (Run-G alternative to DiT-control) |
| `enable_bridging` | bridging adapter |

Stage discipline = which heads are enabled. Stage0 enables geom/pixel only; Stage2 flips
on `progress_head`/`action_proposer`; world_prior/policy stay off unless that stage needs them.

## `train:` — optimization & stage gating

| Key | Meaning |
| --- | --- |
| `epochs`, `batch_size_per_gpu`, `lr`, `warmup_steps`, `weight_decay`, `grad_clip` | standard |
| `precision` | `bf16` |
| `trainable_prefixes` | **freeze control**: only params whose name starts with these prefixes train |
| `lr_multipliers.<prefix>` | per-prefix LR scaling (e.g. `context_pixel: 0.05`, `geom: 0.05`) |
| `condition_dropout.*` | classifier-free guidance dropout probs (`text_only_p`, `action_p`, `context_p`, …) |
| `enable_pixel_loss` | include RGB/depth pixel losses in the optimizer this stage |
| `enable_hunyuan_latent_loss` / `enable_prior_hunyuan_latent_loss` | Hunyuan bridge losses |
| `hunyuan_latent_weight`, `hunyuan_detach_world` | bridge weight; detach world trunk from bridge grad |
| `hunyuan_repo` / `hunyuan_model_base` | vendored HunyuanVideo + weights paths |
| `weighted_sampler.*` | optional per-dataset weighting (`balance_by_dataset`, `dataset_weights`) |
| `find_unused_parameters` | DDP flag (set true only when some heads get no grad) |

## `loss:` — objective weights

Geometry: `geom_depth`, `depth_change`, `depth_motion_l1`, `depth_tv`, `geom_point`
(native-3D world-point L1), `geom_pose` (native-3D camera-pose). RGB/motion: `rgb_l1`,
`rgb_lpips`, `rgb_motion_l1`, `rgb_edge`, `rgb_motion_bce/dice`. Action: `action`, `grip`,
`cos`. Generative: `world_prior*`, `progress`, `terminal_progress`, `plausibility`.
A weight of `0.0` disables that term for the stage (e.g. Stage0 sets `world_prior: 0.0`,
`progress: 0.0`).

## Active configs → stage map

| Group (`configs/`) | Count | Stage / use |
| --- | --- | --- |
| `v5_*`, `_smoke_v5_*`, `_eval_v5_*` | 7 | **v5 native-3D** (current direction) |
| `v3_p64_300m_*balanced*`, `*runG*`, `*run1_droid_smoke*` | 7 | canonical 300M staged flow + Run-G |
| `v3_p64_140m_stage[012]_*` | 18 | 140M diagnostic staged flow |
| `v3_p64_1b_stage*` | 7 | 1B-class formal target |
| `scaling_smoke_stage_*`, `smoke_300m_revised_flow_*` | 8 | staged smoke / CI |

Superseded configs (v3/v3.5/p256, LIBERO ablations, p64 demo-heads, 500m) live in
`archive/configs/` — see `docs/cleanup/CLASSIFICATION_2026-06-07.md`.
