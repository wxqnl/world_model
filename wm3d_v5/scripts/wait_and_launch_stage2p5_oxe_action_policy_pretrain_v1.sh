#!/usr/bin/env bash
set -euo pipefail
cd /data/Minko/world_model/wm3d_v5
MIN_FREE_GPUS=${MIN_FREE_GPUS:-4}
MAX_USED_MB=${MAX_USED_MB:-1000}
POLL_SECONDS=${POLL_SECONDS:-120}
RUN_NAME=${RUN_NAME:-wm3d_v5_stage2p5_oxe_action_policy_pretrain_v1}
LOG_DIR=${LOG_DIR:-/data/Minko/logs/${RUN_NAME}}
mkdir -p "$LOG_DIR"
LOCK="$LOG_DIR/launch.lock"
if [ -e "$LOCK" ]; then
  echo "[$(date -Is)] lock exists: $LOCK" >&2
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
select_free_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v max="$MAX_USED_MB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 <= max) print $1}' \
    | head -n "$MIN_FREE_GPUS" \
    | paste -sd, -
}
while true; do
  GPUS=$(select_free_gpus || true)
  COUNT=0
  if [ -n "${GPUS:-}" ]; then
    COUNT=$(awk -F, '{print NF}' <<< "$GPUS")
  fi
  echo "[$(date -Is)] free_gpus=${GPUS:-none} count=$COUNT need=$MIN_FREE_GPUS max_used_mb=$MAX_USED_MB" | tee -a "$LOG_DIR/wait.log"
  if [ "$COUNT" -ge "$MIN_FREE_GPUS" ]; then
    export GPUS
    export NPROC="$MIN_FREE_GPUS"
    export LOG_DIR
    TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_gpus_${GPUS//,/}.log"
    echo "[$(date -Is)] launching on GPUS=$GPUS log=$TRAIN_LOG" | tee -a "$LOG_DIR/wait.log"
    nohup bash scripts/run_stage2p5_oxe_action_policy_pretrain_v1.sh > "$TRAIN_LOG" 2>&1 &
    echo $! > "$LOG_DIR/train.pid"
    echo "$GPUS" > "$LOG_DIR/gpus.txt"
    echo "$TRAIN_LOG" > "$LOG_DIR/train_log_path.txt"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
