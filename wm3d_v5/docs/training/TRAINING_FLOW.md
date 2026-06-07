# WM3D Training Flow — One-Page Map

Authoritative per-stage policy lives in
[`300m_stage0_to_stage2_world_pretrain_v2.md`](300m_stage0_to_stage2_world_pretrain_v2.md).
This page is the quick map: each stage's launcher, config, what trains vs freezes, and
the eval gate. Parameter-agnostic — the same ladder scales 140M (diagnostic) → 300M →
1B (formal target) by swapping width/config.

## The ladder

```
P0 preflight (no training)
  -> Stage0  visual/geom foundation        (context+action -> tokens/depth/rough RGB)
  -> Stage1  core 3D dynamics + visual replay
  -> Stage1.5 Hunyuan bridge alignment
  -> Stage2  progress + proposer action scaffold
  -> Stage2.5 optional visual/depth polish
  -> Stage3  generation (Hunyuan DiT-control, text->video)   [gated off by default]
  -> Run3    LIBERO/benchmark adaptation                     [downstream VLA]
```

Default automated watcher runs Stage0→Stage1→Stage1.5→Stage2:
`scripts/watch_300m_stage0_to_stage2_flow_v2.sh` (set `RUN_STAGE3_GENERATION=1` to
continue into Stage3; `RUN_GENERATION_CANARY=1` for Run-G diagnostics).

## Stage → launcher → config → trainable

| Stage | Launcher (`scripts/`) | Canonical 300M config (`configs/`) | Trains | Frozen |
| --- | --- | --- | --- | --- |
| Stage0 | `run_300m_stage0_visual_geom_oxe_droid20k_balanced_2node_v1.sh` | `v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1.yaml` | dual trunk, action_proj, geom heads, context_pixel | — (from scratch) |
| Stage1 | (watcher) `run_300m_stage1_world_..._from_stage0_2node_v2.sh` | `v3_p64_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2.yaml` | dynamics; `context_pixel`/`geom` at 0.05× LR | rest of trunk held by LR |
| Stage1.5 | `run_300m_stage1p5_hunyuan_bridge_..._v1.sh` | `v3_p64_300m_stage1p5_hunyuan_bridge_..._from_stage1_2node_v1.yaml` | Hunyuan latent adapter + `context_pixel` | world trunk (`hunyuan_detach_world=true`) |
| Stage2 | `run_300m_stage2_..._balanced_..._v2.sh` | `v3_p64_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2.yaml` | `progress_head`, `action_proposer` | world/visual/geom core |
| Stage2.5 | `run_300m_stage2p5_visual_depth_polish_..._v1.sh` | `v3_p64_300m_stage2p5_visual_depth_polish_..._from_stage2_2node_v1.yaml` | optional visual/depth repair | — |
| Stage3 | `run_stage3_generation_hunyuan_dit_control_v1.sh` (or `run_v5_generation_stage_hunyuan_dit_control_v1.sh`) | `WM_CFG`/`WM_CKPT` point at any trained world ckpt | DiT control adapter only (zero-init) | world model + Hunyuan VAE/DiT/text-encoders |

`trainable_prefixes` / `lr_multipliers` in each config encode the "trains vs frozen"
column precisely; read [the canonical doc](300m_stage0_to_stage2_world_pretrain_v2.md)
for the exact loss weights per stage.

## v5 native-3D variants

v5 swaps the geometry path from depth-only to VGGT-native (world points + camera pose).
Same ladder, native-3D configs/launchers:

| Purpose | Launcher | Config |
| --- | --- | --- |
| 300M stage0 native3d | `run_v5_stage_4node_v1.sh` | `v5_p64_300m_stage0_native3d_oxe_droid20k_balanced_2node_v1.yaml` |
| 1B stage0 native3d (formal) | `watch_native3d_v5_cache_then_train_1b_stage0_v1.sh` | `v5_p64_1b_stage0_native3d_wsd_4node_v1.yaml` |
| 140M native3d smoke | — | `_smoke_v5_p64_140m_stage0_native3d_exp8192_w2_1gpu.yaml` |

Enable native-3D with `model.enable_geom_extra: true` + `data.require_geom_extra: true`
(see [`CONFIGS.md`](../CONFIGS.md)). The Stage3 DiT-control adapter consumes native-3D
geometry (`point` `[B,T,H,W,3]`, `pose_geom` `[B,T,9]`) via `use_point`/`use_pose`
encoders; depth-only checkpoints remain backward-compatible (encoders gate on
availability, DiT-facing output projections are zero-init no-ops at init).

## Stage3 prerequisites (must not regress)

1. `run_v5_generation_stage_hunyuan_dit_control_v1.sh` exports `PYTHONPATH=$V5_ROOT` so
   `wm3d_v3` resolves to this v5 tree (which has the adapter).
2. The trainer wraps the adapter forward **and loss** in `torch.enable_grad()` because
   loading the Hunyuan sampler globally calls `torch.set_grad_enabled(False)`.
3. Real text conditioning needs `data.load_task_text: true` (the trainer sets
   `--load_task_text` by default); without it training collapses to a constant prompt.

## Eval gates

After each stage the watcher runs RGB/depth + native-3D evals. Key formal benchmarks:

- `run_formal_world_model_benchmark_v5_stage0_v1.sh` — stage0 world-model benchmark
- `run_formal_world_model_benchmark_v5_libero_long_v1.sh` — LIBERO long-horizon
- `run_world3d_native_benchmark_v2.sh` / `run_formal_native3d_benchmark_v1.sh` — native-3D
  real-action-vs-counterfactual depth/point/pose claim gate
- `run_worldvla_libero_benchmark_v1.sh` — WorldVLA LIBERO protocol
