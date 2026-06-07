#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
RUN="train_1b_stage0_visual_depth_wsd_smoke_4node_v1"
CFG="configs/v3_p64_1b_stage0_visual_depth_wsd_smoke_4node_v1.yaml"
ROOT="results/wm3d_v3_p64_1b_stage0_visual_depth_wsd_smoke_4node_v1"
CKPT="${ROOT}/ckpt/step_00002000.pt"
PID_FILE="/data/Minko/logs/${RUN}_node0.pid"
EVAL_DIR="${ROOT}/eval_step2000"
LOG_DIR="/data/Minko/logs"

echo "watch_start $(date -Is)"
echo "waiting_for=${CKPT}"

while [[ ! -f "${CKPT}" ]]; do
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "training_exited_before_step2000 pid=${pid} $(date -Is)"
      exit 2
    fi
  fi
  sleep 60
done

echo "ckpt_found $(date -Is)"

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  while kill -0 "${pid}" 2>/dev/null; do
    echo "waiting_for_training_exit pid=${pid} $(date -Is)"
    sleep 30
  done
fi

mkdir -p "${EVAL_DIR}"

echo "run_eval_start $(date -Is)"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${PY}" -m wm3d_v3.eval.run_eval \
  --cfg "${CFG}" \
  --ckpt "${CKPT}" \
  --out "${EVAL_DIR}/eval_rgb_depth_24b.json" \
  --max_batches 24 \
  --batch_size 1

echo "demo_gif_start $(date -Is)"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${PY}" -m wm3d_v3.eval.make_demo_gif \
  --cfg "${CFG}" \
  --ckpt "${CKPT}" \
  --out_dir "${EVAL_DIR}/demo_gifs" \
  --n_clips 8

echo "native3d_start $(date -Is)"
CFG="${CFG}" \
CKPT="${CKPT}" \
OUT_DIR="${EVAL_DIR}/native3d_benchmark_v2" \
GPU=0 \
BATCH_SIZE=1 \
MAX_BATCHES_PER_DATASET=2 \
N_VIZ=4 \
PY="${PY}" \
scripts/run_world3d_native_benchmark_v2.sh \
  > "${LOG_DIR}/${RUN}_native3d_eval.log" 2>&1 || echo "native3d_failed log=${LOG_DIR}/${RUN}_native3d_eval.log"
