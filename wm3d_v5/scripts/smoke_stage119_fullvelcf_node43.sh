#!/usr/bin/env bash
set -euo pipefail

RUN=/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_hunyuan_dit_wmlatent_fullvelcf_jointpt_stage119_smoke_20260622_1425
CFG=/data/Minko/world_model/wm3d_v5/configs/v5_p64_1b_stage0_native3d_hunyuan_dit_wmlatent_fullvelcf_jointpt_stage119_2node_v1.yaml
mkdir -p "$RUN/logs"

cd /data/Minko/world_model/wm3d_v5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WM3D_GRAD_BUCKET_MB=256
export PYTHONUNBUFFERED=1

/data/Minko/.venvs/wm3d/bin/torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=172.27.0.6 \
  --master_port=29829 \
  scripts/train_stage0_hunyuan_dit_joint_body.py \
  --cfg "$CFG" \
  --out_dir "$RUN" \
  --print_every 1 \
  --ckpt_every_steps 2500 \
  --max_steps 4 \
  2>&1 | tee "$RUN/logs/node43.log"
