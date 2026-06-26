#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5
export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="${CFG:-configs/v5_p64_1b_libero_action_policy_a2_nohistft_p2_noprog_v1.yaml}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs/wm3d_v5_p64_1b_libero_action_policy_p2_noprog_v1}"
MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29621}"
NNODES="${NNODES:-2}"
NPROC="${NPROC:-8}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on node43 and NODE_RANK=1 on node44}"
TORCHRUN="${TORCHRUN:-/data/Minko/.venvs/wm3d/bin/torchrun}"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" PYTHONUNBUFFERED=1 "${TORCHRUN}"   --nnodes="${NNODES}"   --nproc_per_node="${NPROC}"   --node_rank="${NODE_RANK}"   --master_addr="${MASTER_ADDR}"   --master_port="${MASTER_PORT}"   -m wm3d_v3.training.train_libero_action_policy   --cfg "${CFG}"   --print_every 25
