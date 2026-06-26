#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko/world_model/wm3d_v5
CFG=${CFG:-configs/v5_p64_1b_stage2p5_oxe_action_policy_pretrain_v2_libero_compat.yaml}
RESUME=${RESUME:-/0604-10T-test/wm3d_v5/checkpoints/wm3d_v5_p64_1b_stage2_action_scaffold_native3d_wsd_4node_v1_best.pt}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
LOG_DIR=${LOG_DIR:-/data/Minko/logs/wm3d_v5_stage2p5_oxe_action_policy_pretrain_v2_libero_compat}
mkdir -p "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export WM3D_DDP_BACKEND=${WM3D_DDP_BACKEND:-gloo}
# NCCL 2.26 on this H100 node fails first all_reduce unless NVLS is disabled.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
exec /data/Minko/.venvs/wm3d/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m wm3d_v3.training.train \
  --cfg "$CFG" \
  --resume "$RESUME" \
  --reset_optim \
  --no_pixel \
  --print_every "${PRINT_EVERY:-25}"
