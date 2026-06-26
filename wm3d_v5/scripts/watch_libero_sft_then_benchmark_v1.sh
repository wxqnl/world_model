#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

RUN_NAME="${RUN_NAME:-wm3d_v5_1b_libero_world_model_sft_from_stage2_v1}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs/${RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-/data/Minko/world_model/wm3d_v5/results/${RUN_NAME}}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-${LOG_DIR}/train_2node_node0.pid}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/watch_sft_then_benchmark.log}"
MAX_WAIT_TRAIN_START_SEC="${MAX_WAIT_TRAIN_START_SEC:-604800}"
POLL_SEC="${POLL_SEC:-120}"

BENCH_RESULT_ROOT="${BENCH_RESULT_ROOT:-/data/Minko/world_model/wm3d_v5/results/worldvla_libero_official_${RUN_NAME}}"
BENCH_RUN_ID="${BENCH_RUN_ID:-${RUN_NAME}_worldvla_official_512}"

mkdir -p "${LOG_DIR}"
exec >> "${WATCH_LOG}" 2>&1

echo "[watch] started $(date -Is)"
echo "[watch] train_pid_file=${TRAIN_PID_FILE}"
echo "[watch] out_root=${OUT_ROOT}"
echo "[watch] bench_result_root=${BENCH_RESULT_ROOT}"

deadline=$(( $(date +%s) + MAX_WAIT_TRAIN_START_SEC ))
while [[ ! -s "${TRAIN_PID_FILE}" ]]; do
  if (( $(date +%s) > deadline )); then
    echo "[watch] timed out waiting for train pid file"
    exit 5
  fi
  echo "[watch] waiting for train pid file $(date -Is)"
  sleep "${POLL_SEC}"
done

train_pid="$(cat "${TRAIN_PID_FILE}")"
echo "[watch] train_pid=${train_pid}"
while ps -p "${train_pid}" >/dev/null 2>&1; do
  echo "[watch] training still running $(date -Is)"
  sleep "${POLL_SEC}"
done
echo "[watch] training process ended $(date -Is)"

ckpt="${OUT_ROOT}/ckpt/best.pt"
if [[ ! -f "${ckpt}" ]]; then
  echo "[watch] missing checkpoint: ${ckpt}"
  exit 6
fi

echo "[watch] starting benchmark ckpt=${ckpt}"
CKPT="${ckpt}" \
RESULT_ROOT="${BENCH_RESULT_ROOT}" \
RUN_ID="${BENCH_RUN_ID}" \
VIDEO_SIZE=512 \
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  scripts/run_worldvla_libero_benchmark_v1.sh all
echo "[watch] benchmark finished $(date -Is)"
