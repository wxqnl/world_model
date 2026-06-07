#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/Minko/world_model/wm3d_v3
PIDFILE=/data/Minko/logs/train_p64_control_head_v2_256_latest.pid
CFG=$ROOT/configs/v3_p64_140m_actioncond_control_head_v2_256.yaml
RUN=$ROOT/results/wm3d_v3_p64_140m_actioncond_control_head_v2_256
CKPT=$RUN/ckpt/best.pt
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$RUN/demo_best_auto_${STAMP}
LOG=/data/Minko/logs/demo_p64_control_head_v2_256_auto_${STAMP}.log
SENS_OUT=$RUN/action_sensitivity_${STAMP}.json
EVAL_OUT=$RUN/eval_episode_40b_${STAMP}.json

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

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /data/Minko/.venvs/wm3d/bin/python \
  -m wm3d_v3.eval.run_eval \
  --cfg "$CFG" \
  --ckpt "$CKPT" \
  --out "$EVAL_OUT" \
  --max_batches 40 2>&1 | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /data/Minko/.venvs/wm3d/bin/python \
  -m wm3d_v3.eval.action_sensitivity \
  --cfg "$CFG" \
  --ckpt "$CKPT" \
  --out "$SENS_OUT" \
  --max_batches 40 \
  --variants zero,shuffled,sign_flip,scaled,grip_toggle 2>&1 | tee -a "$LOG"

echo "[watch] demo_dir=$OUT_DIR" | tee -a "$LOG"
echo "[watch] eval=$EVAL_OUT" | tee -a "$LOG"
echo "[watch] action_sensitivity=$SENS_OUT" | tee -a "$LOG"
ln -sfn "$OUT_DIR" "$RUN/demo_best_auto_latest"
ln -sfn "$EVAL_OUT" "$RUN/eval_episode_40b_latest.json"
ln -sfn "$SENS_OUT" "$RUN/action_sensitivity_latest.json"
ln -sfn "$LOG" /data/Minko/logs/demo_p64_control_head_v2_256_auto_latest.log
