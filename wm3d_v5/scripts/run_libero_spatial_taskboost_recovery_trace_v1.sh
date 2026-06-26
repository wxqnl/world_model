#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-/data/Minko/world_model/wm3d_v5/results/wm3d_v5_p64_1b_libero_all_plus_spatial_dense_taskboost_k32_actpolicy_sft_v1}"
CKPT="${CKPT:-${ROOT}/ckpt/best.pt}"
BASE_CFG="${BASE_CFG:-configs/v5_p64_1b_libero_action_policy_fastwam_spatial_dualcam_concat_padded_earlyw_rgb_lastobs_k32_base_v1.yaml}"
BENCH_ROOT="${BENCH_ROOT:-${ROOT}/libero_spatial_taskboost_recovery_trace_${RUN_ID}}"
EPISODE_LIST="${EPISODE_LIST:?EPISODE_LIST is required; use scripts/make_libero_failed_episode_list_v1.py first}"
LOG_ROOT="${LOG_ROOT:-/data/Minko/logs/libero_taskboost_recovery_trace}"

mkdir -p "${LOG_ROOT}" "${BENCH_ROOT}"

env \
  BASE_CFG="${BASE_CFG}" \
  CKPT="${CKPT}" \
  BENCH_ROOT="${BENCH_ROOT}" \
  LOG_DIR="${BENCH_ROOT}/logs" \
  EP_DIR="${BENCH_ROOT}/episodes" \
  SAVE_FRAMES_ROOT="${BENCH_ROOT}/frames" \
  SAVE_FRAME_EVERY="${SAVE_FRAME_EVERY:-1}" \
  EPISODE_LIST="${EPISODE_LIST}" \
  GPUS="${GPUS:-0,1,2,3,4,5,6,7}" \
  BASE_PORT="${BASE_PORT:-13880}" \
  SUITE=libero_spatial \
  MAX_TASKS=10 \
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
  SEND_OBJECT_STATE="${SEND_OBJECT_STATE:-0}" \
  TRACE_OBJECT_STATE="${TRACE_OBJECT_STATE:-1}" \
  SEND_PLAN_STATE="${SEND_PLAN_STATE:-0}" \
  PLAN_STATE_DIM="${PLAN_STATE_DIM:-8}" \
  SELECTION_MODE="${SELECTION_MODE:-direct}" \
  USE_POLICY_GRIPPER_PROB="${USE_POLICY_GRIPPER_PROB:-1}" \
  GRIPPER_CLOSED_THRESHOLD="${GRIPPER_CLOSED_THRESHOLD:-0.35}" \
  GRIPPER_MIN_CLOSE_Z="${GRIPPER_MIN_CLOSE_Z:-0.0}" \
  MAX_RETRIES="${MAX_RETRIES:-2}" \
  EPISODE_TIMEOUT="${EPISODE_TIMEOUT:-1800}" \
  BENCHMARK_LABEL="LIBERO spatial recovery trace collection (WM3D-v5 taskboost best, gate0.0, frames)" \
  bash scripts/run_libero_spatial_evalmatch_resumable_v1.sh all
