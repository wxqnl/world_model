#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko

WORKER=${WORKER:-root@172.27.0.7}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
PIDFILE=${PIDFILE:-$LOG_DIR/prepare_worker_172_27_0_7_sync_v1.pid}
LOG=${LOG:-$LOG_DIR/prepare_worker_172_27_0_7_sync_v1.log}
RSH=${RSH:-ssh -o BatchMode=yes}
RSYNC_BWLIMIT_KB=${RSYNC_BWLIMIT_KB:-300000}

mkdir -p "$LOG_DIR"

if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "already_running pid=$old_pid log=$LOG"
    exit 0
  fi
fi

run_rsync() {
  local label="$1"
  local src="$2"
  local dst="$3"
  shift 3
  echo "[$(date -Is)] sync_start label=$label src=$src dst=$dst"
  ionice -c2 -n7 nice -n 10 rsync -aH --partial --inplace \
    --numeric-ids \
    --bwlimit="$RSYNC_BWLIMIT_KB" \
    --info=progress2,stats2 \
    -e "$RSH" \
    "$@" \
    "$src" "$dst"
  echo "[$(date -Is)] sync_done label=$label"
}

{
  echo "[$(date -Is)] prepare_worker_start worker=$WORKER bwlimit_kb=$RSYNC_BWLIMIT_KB"
  ssh -o BatchMode=yes "$WORKER" "mkdir -p /data/Minko/world_model /data/Minko/datasets/cache /data/Minko/models /data/Minko/external"

  run_rsync wm3d_code /data/Minko/world_model/wm3d_v3/ "$WORKER:/data/Minko/world_model/wm3d_v3/" \
    --exclude /results/ \
    --exclude /tb/ \
    --exclude '**/__pycache__/' \
    --exclude '*.pyc'

  run_rsync hunyuan_repo /data/Minko/external/HunyuanVideo/ "$WORKER:/data/Minko/external/HunyuanVideo/"
  run_rsync hunyuan_model /data/Minko/models/hunyuan_video/ "$WORKER:/data/Minko/models/hunyuan_video/"
  run_rsync oxe_cache /data/Minko/datasets/cache/wm3d_v3/ "$WORKER:/data/Minko/datasets/cache/wm3d_v3/"

  ssh -o BatchMode=yes "$WORKER" "mkdir -p /data/Minko/world_model/wm3d_v3/results"
  run_rsync libero_policy_cache \
    /data/Minko/world_model/wm3d_v3/results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_v1_cache/ \
    "$WORKER:/data/Minko/world_model/wm3d_v3/results/wm3d_libero_action_policy_lowdimhist_partial4_start_stride4_v1_cache/"

  ssh -o BatchMode=yes "$WORKER" "cd /data/Minko/world_model/wm3d_v3 && /data/Minko/.venvs/wm3d/bin/python - <<'PY'
import torch
from pathlib import Path
checks = [
    Path('/data/Minko/datasets/cache/wm3d_v3'),
    Path('/data/Minko/models/hunyuan_video'),
    Path('/data/Minko/external/HunyuanVideo'),
    Path('/data/Minko/world_model/wm3d_v3/configs'),
]
print({'cuda': torch.cuda.device_count(), 'checks': {str(p): p.exists() for p in checks}})
PY"
  echo "[$(date -Is)] prepare_worker_done"
} >> "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "started_prepare_worker_pid=$(cat "$PIDFILE") log=$LOG"
