#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-/data/Minko/world_model/wm3d_v5/results/wm3d_v5_p64_1b_libero_spatial_evalmatch_teacher_recovery_k32_sft_v1}"
CKPT="${CKPT:-${ROOT}/ckpt/best.pt}"
BASE_CFG="${BASE_CFG:-configs/v5_p64_1b_libero_action_policy_fastwam_spatial_dualcam_concat_padded_earlyw_rgb_lastobs_k32_base_v1.yaml}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/libero_spatial_evalmatch_recovery_k32_h8_gate000_${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-/data/Minko/logs/libero_evalmatch_recovery_k32}"

mkdir -p "${LOG_ROOT}"

env \
  BASE_CFG="${BASE_CFG}" \
  CKPT="${CKPT}" \
  BENCH_ROOT="${BENCH_ROOT}" \
  LOG_DIR="${BENCH_ROOT}/logs" \
  EP_DIR="${BENCH_ROOT}/episodes" \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  BASE_PORT="${BASE_PORT:-12880}" \
  SUITE=libero_spatial \
  MAX_TASKS="${MAX_TASKS:-10}" \
  TASK_IDS="${TASK_IDS:-}" \
  INIT_START="${INIT_START:-0}" \
  INIT_END="${INIT_END:-49}" \
  MAX_STEPS="${MAX_STEPS:-300}" \
  CAMERA_KEYS="${CAMERA_KEYS:-agentview_image,robot0_eye_in_hand_image}" \
  CAMERA_FUSION="${CAMERA_FUSION:-concat}" \
  CAMERA_SIZE="${CAMERA_SIZE:-256}" \
  ROTATE_180="${ROTATE_180:-1}" \
  CONTEXT_T="${CONTEXT_T:-16}" \
  WARMUP_STEPS="${WARMUP_STEPS:-0}" \
  EXEC_HORIZON="${EXEC_HORIZON:-8}" \
  ACTION_HISTORY_LEN="${ACTION_HISTORY_LEN:-0}" \
  SEND_PROGRESS="${SEND_PROGRESS:-0}" \
  SELECTION_MODE="${SELECTION_MODE:-direct}" \
  USE_POLICY_GRIPPER_PROB="${USE_POLICY_GRIPPER_PROB:-1}" \
  GRIPPER_CLOSED_THRESHOLD="${GRIPPER_CLOSED_THRESHOLD:-0.35}" \
  GRIPPER_MIN_CLOSE_Z="${GRIPPER_MIN_CLOSE_Z:-0.0}" \
  MAX_RETRIES="${MAX_RETRIES:-4}" \
  EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-1800}" \
  HF_HOME="${HF_HOME:-/data/Minko/.cache/huggingface}" \
  HF_HUB_CACHE="${HF_HUB_CACHE:-/data/Minko/.cache/huggingface/hub}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  MUJOCO_GL=egl \
  __EGL_VENDOR_LIBRARY_FILENAMES=/data/Minko/egl/10_nvidia.json \
  BENCHMARK_LABEL="LIBERO spatial closed-loop (WM3D-v5 evalmatch recovery k32 SFT, h8 gate0.0)" \
  bash scripts/run_libero_spatial_evalmatch_resumable_v1.sh all
