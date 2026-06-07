# 140M Stage0 Visual Depth Stabilized 4-Node V1

Goal: verify that the 140M visual/depth Stage0 can recover clean RGB/depth demos on the current OXE+DROID depthplus data without warm-starting from the old 138M/140M checkpoints.

## Changes

- Use `geom_upsample_mode: resize_conv` to replace transposed-convolution depth upsampling for this run.
- Add low-weight `depth_tv` regularization to suppress high-frequency striping in predicted depth.
- Keep the same `manifests/oxe_droid20k_depthplus_world_v1.jsonl` data used by the failed depthplus visual test.
- Train for `max_steps: 20000` on 4 nodes / 32 GPUs instead of the previous 3000-step short test.
- Reduce depth dynamic losses from the failed short test:
  - `geom_depth: 3.5 -> 1.6`
  - `depth_change: 0.25 -> 0.05`
  - `depth_motion_l1: 0.20 -> 0.04`

## Run

```bash
bash scripts/run_140m_stage0_visual_depth_stabilized_4node_v1.sh
bash scripts/watch_140m_stage0_visual_depth_stabilized_4node_v1.sh
```

Outputs:

- Training: `results/wm3d_v3_p64_140m_stage0_visual_depth_stabilized_4node_v1`
- Node0 log: `/data/Minko/logs/train_140m_stage0_visual_depth_stabilized_4node_v1_node0.log`
- Watch/eval log: `/data/Minko/logs/watch_140m_stage0_visual_depth_stabilized_4node_v1.log`
- Post-train eval: `results/wm3d_v3_p64_140m_stage0_visual_depth_stabilized_4node_v1/eval_after_stage0_stabilized`

## Pass Criteria

- No recurring stripe/checkerboard artifacts in DROID/Bridge/TACO depth demo GIFs.
- DROID depth should improve materially over the failed Stage0 baseline (`DROID L_depth_rel_L1 ~= 0.09` on the 24b/bs4 review set).
- RGB should be judged by fixed demo GIFs first, not only by L1/LPIPS.
