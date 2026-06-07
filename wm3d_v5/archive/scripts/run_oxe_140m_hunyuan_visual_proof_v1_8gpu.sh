#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

CFG=${CFG:-configs/v3_p64_140m_hunyuan_visual_proof_v1_8gpu.yaml}
RESUME=${RESUME:-results/wm3d_v3_p64_140m_actioncond_context_motion/ckpt/best.pt}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
RUN_NAME=${RUN_NAME:-train_oxe_140m_hunyuan_visual_proof_v1_8gpu}
LOG=${LOG:-$LOG_DIR/${RUN_NAME}.log}
PIDFILE=${PIDFILE:-$LOG_DIR/${RUN_NAME}.pid}

mkdir -p "$LOG_DIR"

if [[ -f "$PIDFILE" ]]; then
  old_pid=$(cat "$PIDFILE" || true)
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "already_running_pid=$old_pid"
    echo "log=$LOG"
    exit 0
  fi
fi

if [[ ! -f "$CFG" ]]; then
  echo "missing_cfg=$CFG" >&2
  exit 1
fi
if [[ ! -f "$RESUME" ]]; then
  echo "missing_resume=$RESUME" >&2
  exit 1
fi

export PYTHONPATH=/data/Minko/world_model/wm3d_v3:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}

nohup /data/Minko/.venvs/wm3d/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  -m wm3d_v3.training.train \
  --cfg "$CFG" \
  --resume "$RESUME" \
  --reset_optim \
  --print_every 25 \
  > "$LOG" 2>&1 &

pid=$!
echo "$pid" > "$PIDFILE"
echo "started_pid=$pid"
echo "cfg=$CFG"
echo "resume=$RESUME"
echo "log=$LOG"
echo "pidfile=$PIDFILE"
