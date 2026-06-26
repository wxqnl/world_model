#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN=/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_hunyuan_dit_jointpt_fromscratch_wsd_noemptycache_stage114c_16gpu_20260619_1930
CFG=/data/Minko/world_model/wm3d_v5/configs/v5_p64_1b_stage0_native3d_hunyuan_dit_jointpt_fromscratch_2node_v1.yaml
mkdir -p "$RUN/logs"

export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WM3D_GRAD_BUCKET_MB=256

/data/Minko/.venvs/wm3d/bin/torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=1 \
  --master_addr=172.27.0.6 \
  --master_port=29715 \
  scripts/train_stage0_hunyuan_dit_joint.py \
  --cfg "$CFG" \
  --out_dir "$RUN" \
  --print_every 20 \
  2>&1 | tee -a "$RUN/logs/node44.log"
