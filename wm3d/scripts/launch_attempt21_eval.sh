#!/usr/bin/env bash
set -euo pipefail

NODE_RANK=${1:?node rank is required}
ROOT=/data/Minko/wm3d_conditioning_canary_1b_action_rgb_clean_2node16_step500_20260826_attempt21
CODE=/data/Minko/wm3d_conditioning_fix_20260824/wm3d/wm3d
VENV=/data/Minko/.venvs/wm3d_direct_v8_20260821
CHECKPOINT="$ROOT/training/checkpoints/step_00000500"
OUTPUT="$ROOT/eval/step500_action_rgb_teacher0_v3_signaltrace.json"
DEMO_ROOT="$ROOT/eval/step500_teacher0_demos_v3_signaltrace"

mkdir -p "$DEMO_ROOT"
cd "$CODE"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$CODE"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0.1411
export NCCL_SOCKET_FAMILY=AF_INET
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MALLOC_ARENA_MAX=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WM3D_EXECUTION_HOTFIX=1
export WM3D_VGGT_SOURCE_ROOT=/data/Minko/wm3d_vggt_source_20260822
export WM3D_VGGT_MODEL_SNAPSHOT=/data/Minko/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9
export WM3D_DIRECT_PREPARED_ROW_CACHE_BYTES_PER_RANK=1073741824
export WM3D_DIRECT_PREFETCH_WORKERS=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=4096
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export PYTHONFAULTHANDLER=1

exec "$VENV/bin/torchrun" \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --master-addr=172.27.0.6 \
  --master-port=29823 \
  --max-restarts=0 \
  "$CODE/scripts/eval_action_conditioning.py" \
  --runtime "$ROOT/runtime.yaml" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --demo-root "$DEMO_ROOT" \
  --demo-samples 10 \
  --appearance-teacher-ratio 0
