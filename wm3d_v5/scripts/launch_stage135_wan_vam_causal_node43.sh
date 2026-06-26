#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN=/0604-10T-test/wm3d_v5/results/wm3d_v5_p64_1b_stage0_native3d_wan_vam_cleanctrl_videocf_fsdp_jointpt_stage135_16gpu_20260625_1745
CFG=/data/Minko/world_model/wm3d_v5/configs/v5_p64_1b_stage0_native3d_wan_vam_stage135_2node_v1.yaml
mkdir -p "$RUN/logs"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WM3D_GRAD_BUCKET_MB=256
export PYTHONUNBUFFERED=1

/data/Minko/.venvs/wm3d/bin/torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=0 \
  --master_addr=172.27.0.6 \
  --master_port=29865 \
  scripts/train_stage0_wan_ti2v_joint_body.py \
  --cfg "$CFG" \
  --out_dir "$RUN" \
  --print_every 20 \
  --ckpt_every_steps 2500 \
  2>&1 | tee -a "$RUN/logs/node43.log"
