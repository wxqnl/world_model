#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
ALL_PIDFILE=${ALL_PIDFILE:-$LOG_DIR/train_oxe_all_trainable_no_video_fullpolicy_v1_8gpu.pid}
ALL_BEST=${ALL_BEST:-results/wm3d_v3_p64_140m_p0_action_policy_oxe_all_trainable_no_video_fullpolicy_v1_8gpu/ckpt/best.pt}
LIBERO_CFG=${LIBERO_CFG:-configs/libero_action_policy_partial4_oxeall_fullpolicy_v1_8gpu.yaml}
LIBERO_LOG=${LIBERO_LOG:-$LOG_DIR/train_libero_action_policy_partial4_oxeall_fullpolicy_v1_8gpu.log}
LIBERO_PIDFILE=${LIBERO_PIDFILE:-$LOG_DIR/train_libero_action_policy_partial4_oxeall_fullpolicy_v1_8gpu.pid}

mkdir -p "$LOG_DIR"

while [[ ! -f "$ALL_PIDFILE" ]]; do
  echo "waiting_for_all_oxe_pidfile=$ALL_PIDFILE"
  sleep 60
done

all_pid="$(cat "$ALL_PIDFILE" || true)"
if [[ -n "${all_pid:-}" ]] && ps -p "$all_pid" >/dev/null 2>&1; then
  echo "waiting_for_all_oxe_pid=$all_pid"
  while ps -p "$all_pid" >/dev/null 2>&1; do
    sleep 120
  done
fi

if [[ ! -f "$ALL_BEST" ]]; then
  echo "missing_all_oxe_best=$ALL_BEST"
  exit 2
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WM3D_DDP_BACKEND=gloo \
OMP_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node=8 \
  -m wm3d_v3.training.train_libero_action_policy \
  --cfg "$LIBERO_CFG" \
  --print_every 25 \
  >>"$LIBERO_LOG" 2>&1 &

echo "$!" > "$LIBERO_PIDFILE"
echo "started_libero_pid=$(cat "$LIBERO_PIDFILE")"
