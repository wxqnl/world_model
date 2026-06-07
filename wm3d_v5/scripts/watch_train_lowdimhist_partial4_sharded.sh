#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/Minko}"
PROJECT_ROOT="${PROJECT_ROOT:-$ROOT/world_model/wm3d_v3}"
WM3D_PY="${WM3D_PY:-$ROOT/.venvs/wm3d/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1_cache}"
CFG="${CFG:-configs/libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1.yaml}"
TRAIN_ROOT="${TRAIN_ROOT:-results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1}"
TRAIN_LOG="${TRAIN_LOG:-$ROOT/logs/train_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1.log}"
TRAIN_PID_FILE="${TRAIN_PID_FILE:-$ROOT/logs/train_libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1.pid}"
WATCH_LOG="${WATCH_LOG:-$ROOT/logs/watch_train_lowdimhist_partial4_sharded.log}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0}"

cd "$PROJECT_ROOT"
mkdir -p "$ROOT/logs"

{
  echo "[watch-sharded] started $(date -Is)"
  for si in 00 01 02 03 04 05 06; do
    pid_file="$ROOT/logs/libero_partial4_sharded_cache_${si}.pid"
    if [[ ! -f "$pid_file" ]]; then
      echo "[watch-sharded] missing pid file $pid_file"
      exit 2
    fi
    pid="$(cat "$pid_file")"
    echo "[watch-sharded] waiting shard_${si} pid $pid"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 60
    done
    manifest="$CACHE_ROOT/shard_${si}/manifest.jsonl"
    if [[ ! -f "$manifest" ]]; then
      echo "[watch-sharded] missing manifest $manifest"
      exit 3
    fi
    echo "[watch-sharded] shard_${si} ready lines=$(wc -l < "$manifest")"
  done

  total_lines=0
  for manifest in "$CACHE_ROOT"/shard_*/manifest.jsonl; do
    lines="$(wc -l < "$manifest")"
    total_lines=$((total_lines + lines))
  done
  echo "[watch-sharded] all shards ready total_lines=$total_lines"

  CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  "$WM3D_PY" -m wm3d_v3.training.train_libero_action_policy \
    --cfg "$CFG" \
    --print_every 25 \
    > "$TRAIN_LOG" 2>&1 &
  train_pid=$!
  echo "$train_pid" > "$TRAIN_PID_FILE"
  echo "[watch-sharded] train pid $train_pid"
  wait "$train_pid"
  echo "[watch-sharded] train done $(date -Is)"

  if [[ -f "$TRAIN_ROOT/ckpt/best.pt" ]]; then
    CUDA_VISIBLE_DEVICES="$ROLLOUT_CUDA_VISIBLE_DEVICES" \
    DEVICE=cuda:0 \
    CFG=configs/v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist.yaml \
    CKPT="$TRAIN_ROOT/ckpt/best.pt" \
    SELECTION_MODE=direct \
    PORT=8773 \
    MAX_TASKS=1 \
    INIT_STATE_HDF5=/data/Minko/benchmarks/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5 \
    INIT_STATE_DEMO_ID=demo_0 \
    MAX_STEPS=300 \
    CAMERA_SIZE=128 \
    CONTEXT_T=16 \
    WARMUP_STEPS=0 \
    GRIPPER_MODE=closed01_to_libero \
    SEND_LOWDIM=1 \
    ACTION_HISTORY_LEN=16 \
    OUT="$TRAIN_ROOT/libero_remote_rollout_hdf5init_task1_demo0_best_300step.json" \
    SAVE_FRAMES_DIR="$TRAIN_ROOT/hdf5init_task1_demo0_best_frames" \
    SAVE_FRAME_EVERY=25 \
    SERVER_LOG=/data/Minko/logs/wm3d_policy_server_lowdimhist_partial4_sharded_task1_demo0_best.log \
    bash scripts/run_libero_remote_smoke.sh
    echo "[watch-sharded] rollout done $(date -Is)"
  else
    echo "[watch-sharded] best checkpoint missing after train"
    exit 4
  fi
} >> "$WATCH_LOG" 2>&1
