#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/Minko/world_model/wm3d_v3
PIDFILE=/data/Minko/logs/train_p64_context_motion_latest.pid
CFG=$ROOT/configs/v3_p64_140m_actioncond_context_motion.yaml
CKPT=$ROOT/results/wm3d_v3_p64_140m_actioncond_context_motion/ckpt/best.pt
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$ROOT/results/wm3d_v3_p64_140m_actioncond_context_motion/demo_best_auto_${STAMP}
LOG=/data/Minko/logs/demo_p64_context_motion_auto_${STAMP}.log

cd "$ROOT"
echo "[watch] started at $(date '+%F %T %Z')" | tee -a "$LOG"
echo "[watch] pidfile=$PIDFILE" | tee -a "$LOG"

if [[ -s "$PIDFILE" ]]; then
  pid=$(cat "$PIDFILE")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
fi

echo "[watch] training process ended at $(date '+%F %T %Z')" | tee -a "$LOG"
if [[ ! -f "$CKPT" ]]; then
  echo "[watch] missing checkpoint: $CKPT" | tee -a "$LOG"
  exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /data/Minko/.venvs/wm3d/bin/python \
  -m wm3d_v3.eval.make_demo_gif \
  --cfg "$CFG" \
  --ckpt "$CKPT" \
  --out_dir "$OUT_DIR" \
  --n_clips 8 2>&1 | tee -a "$LOG"

echo "[watch] demo_dir=$OUT_DIR" | tee -a "$LOG"
ln -sfn "$OUT_DIR" "$ROOT/results/wm3d_v3_p64_140m_actioncond_context_motion/demo_best_auto_latest"
ln -sfn "$LOG" /data/Minko/logs/demo_p64_context_motion_auto_latest.log
