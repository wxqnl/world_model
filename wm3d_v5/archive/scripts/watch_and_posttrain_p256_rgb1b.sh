#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/Minko/world_model}"
PROJ="${PROJ:-$ROOT/wm3d_v3}"
CFG="${CFG:-$PROJ/configs/v3_p256_rgb1b_oxe.yaml}"
OUT_ROOT="${OUT_ROOT:-$PROJ/results/wm3d_v3_p256_rgb1b}"
CKPT="${CKPT:-$OUT_ROOT/ckpt/best.pt}"
FINAL_CKPT="${FINAL_CKPT:-$OUT_ROOT/ckpt/epoch_019.pt}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-/data/Minko/logs/train_v3_p256_rgb1b_latest.pid}"
TRAIN_LOG="${TRAIN_LOG:-/data/Minko/logs/train_v3_p256_rgb1b_latest.log}"
POLL_SECONDS="${POLL_SECONDS:-300}"

RUN_ID="${RUN_ID:-rgb1b_$(date +%Y%m%d_%H%M%S)}"
WATCH_LOG_DIR="${WATCH_LOG_DIR:-/data/Minko/logs}"
WATCH_LOG="$WATCH_LOG_DIR/posttrain_p256_rgb1b_watcher_${RUN_ID}.log"

mkdir -p "$WATCH_LOG_DIR"
exec > >(tee -a "$WATCH_LOG") 2>&1

echo "[$(date)] watcher started"
echo "root=$ROOT"
echo "proj=$PROJ"
echo "cfg=$CFG"
echo "out_root=$OUT_ROOT"
echo "ckpt=$CKPT"
echo "final_ckpt=$FINAL_CKPT"
echo "train_pid_file=$TRAIN_PID_FILE"
echo "train_log=$TRAIN_LOG"

train_pid=""
if [[ -s "$TRAIN_PID_FILE" ]]; then
  train_pid="$(cat "$TRAIN_PID_FILE")"
fi

if [[ -n "$train_pid" ]]; then
  echo "[$(date)] waiting for training pid $train_pid"
  while kill -0 "$train_pid" 2>/dev/null; do
    tail -n 5 "$TRAIN_LOG" 2>/dev/null || true
    sleep "$POLL_SECONDS"
  done
else
  echo "[$(date)] no train pid found; falling back to process pattern"
  while pgrep -f "python.*-m wm3d_v3.training.train --cfg .*v3_p256_rgb1b_oxe.yaml" >/dev/null; do
    tail -n 5 "$TRAIN_LOG" 2>/dev/null || true
    sleep "$POLL_SECONDS"
  done
fi

echo "[$(date)] training process no longer alive; validating completion"

if grep -Eq "Traceback|RuntimeError|CUDA out of memory|FAILED|ncclUnhandledCudaError|SignalException" "$TRAIN_LOG"; then
  echo "[$(date)] training log contains an error marker; aborting posttrain generation"
  grep -En "Traceback|RuntimeError|CUDA out of memory|FAILED|ncclUnhandledCudaError|SignalException" "$TRAIN_LOG" | tail -n 40 || true
  exit 1
fi

if ! grep -q "\\[rank0\\] epoch 19:" "$TRAIN_LOG"; then
  echo "[$(date)] final epoch marker not found in train log; aborting"
  tail -n 80 "$TRAIN_LOG" || true
  exit 1
fi

if [[ ! -s "$FINAL_CKPT" ]]; then
  echo "[$(date)] final checkpoint missing: $FINAL_CKPT"
  exit 1
fi

if [[ ! -s "$CKPT" ]]; then
  echo "[$(date)] best checkpoint missing: $CKPT"
  exit 1
fi

echo "[$(date)] training completed cleanly; launching posttrain GIF generation"

export PATH="/data/Minko/.venvs/wm3d/bin:$PATH"
export ROOT
export PROJ
export CFG
export OUT_ROOT
export CKPT
export RUN_ID
export POST_DIR="${POST_DIR:-$OUT_ROOT/posttrain_$RUN_ID}"
export POLL_SECONDS=30
export SHORT_N_CLIPS="${SHORT_N_CLIPS:-16}"
export FULL_TASKS_N="${FULL_TASKS_N:-8}"
export FULL_MIN_FRAMES="${FULL_MIN_FRAMES:-64}"
export FULL_MAX_FRAMES="${FULL_MAX_FRAMES:-160}"
export EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-50}"

bash "$PROJ/scripts/posttrain_p256_demos.sh"

echo "[$(date)] watcher finished; outputs in $POST_DIR"
