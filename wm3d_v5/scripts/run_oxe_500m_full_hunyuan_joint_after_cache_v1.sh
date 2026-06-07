#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
CACHE_ROOT=${CACHE_ROOT:-/data/Minko/datasets/cache/wm3d_v3}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
WORLD=${WORLD:-8}
BATCH_FRAMES=${BATCH_FRAMES:-32}

SOURCE_MANIFEST=${SOURCE_MANIFEST:-manifests/oxe_all_downloaded.jsonl}
ELIGIBLE_MANIFEST=${ELIGIBLE_MANIFEST:-manifests/oxe_all_trainable_no_video_v1.jsonl}
CACHED_MANIFEST=${CACHED_MANIFEST:-manifests/oxe_all_trainable_cached_rgb_geom_v1.jsonl}
PRECACHE_PIDS=${PRECACHE_PIDS:-$LOG_DIR/cache_oxe_all_trainable_no_video_v1.pids}
FULL_CACHE_PIDS=${FULL_CACHE_PIDS:-$LOG_DIR/cache_oxe_all_trainable_rgb_geom_v1.pids}
OLD_PIPELINE_PIDFILE=${OLD_PIPELINE_PIDFILE:-$LOG_DIR/run_oxe_all_trainable_no_video_pipeline_v1.pid}

CURRENT_BEST=${CURRENT_BEST:-results/wm3d_v3_p64_140m_p0_action_policy_oxe_fullpolicy_cached_v4_8gpu/ckpt/best.pt}
ALL_CFG=${ALL_CFG:-configs/v3_p64_500m_full_hunyuan_joint_v1_8gpu.yaml}
ALL_LOG=${ALL_LOG:-$LOG_DIR/train_oxe_500m_full_hunyuan_joint_v1_8gpu.log}
ALL_PIDFILE=${ALL_PIDFILE:-$LOG_DIR/train_oxe_500m_full_hunyuan_joint_v1_8gpu.pid}

mkdir -p "$LOG_DIR"

wait_for_pids_file() {
  local pids_file="$1"
  local label="$2"
  if [[ ! -f "$pids_file" ]]; then
    return 0
  fi
  while true; do
    local running=0
    while read -r pid; do
      if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
        running=$((running + 1))
      fi
    done < "$pids_file"
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    echo "waiting_${label}_running=$running"
    sleep 120
  done
}

if [[ ! -s "$ELIGIBLE_MANIFEST" ]]; then
  "$PY" scripts/build_oxe_trainable_manifest.py \
    --input "$SOURCE_MANIFEST" \
    --output "$ELIGIBLE_MANIFEST" \
    --cache_root "$CACHE_ROOT" \
    --T 16 --k 8 --stride 4
fi

wait_for_pids_file "$PRECACHE_PIDS" "existing_no_rgb_cache"

if [[ -f "$OLD_PIPELINE_PIDFILE" ]]; then
  old_pid="$(cat "$OLD_PIPELINE_PIDFILE" || true)"
  if [[ -n "${old_pid:-}" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    kill "$old_pid" 2>/dev/null || true
  fi
fi

rm -f "$FULL_CACHE_PIDS"
touch "$FULL_CACHE_PIDS"

for shard in $(seq 0 $((WORLD - 1))); do
  shard_log="$LOG_DIR/cache_oxe_all_trainable_rgb_geom_v1_shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$shard" "$PY" scripts/cache_oxe.py \
    --manifest "$ELIGIBLE_MANIFEST" \
    --cache_root "$CACHE_ROOT" \
    --shard "$shard" \
    --world "$WORLD" \
    --batch_frames "$BATCH_FRAMES" \
    >>"$shard_log" 2>&1 &
  echo "$!" >> "$FULL_CACHE_PIDS"
done

cache_status=0
while read -r pid; do
  if [[ -n "$pid" ]]; then
    wait "$pid" || cache_status=$?
  fi
done < "$FULL_CACHE_PIDS"
if [[ "$cache_status" -ne 0 ]]; then
  echo "full_cache_status=$cache_status"
  exit "$cache_status"
fi

"$PY" scripts/build_oxe_trainable_manifest.py \
  --input "$ELIGIBLE_MANIFEST" \
  --output "$CACHED_MANIFEST" \
  --cache_root "$CACHE_ROOT" \
  --T 16 --k 8 --stride 4 \
  --require_policy_cache \
  --require_rgb \
  --require_geom

if [[ ! -s "$CACHED_MANIFEST" ]]; then
  echo "empty_cached_manifest=$CACHED_MANIFEST"
  exit 2
fi
if [[ ! -f "$CURRENT_BEST" ]]; then
  echo "missing_current_best=$CURRENT_BEST"
  exit 3
fi
if [[ ! -d /data/Minko/external/HunyuanVideo ]]; then
  echo "missing_hunyuan_repo=/data/Minko/external/HunyuanVideo"
  exit 4
fi
if [[ ! -d /data/Minko/models/hunyuan_video ]]; then
  echo "missing_hunyuan_model_base=/data/Minko/models/hunyuan_video"
  exit 5
fi

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
WM3D_DDP_BACKEND=gloo \
OMP_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TORCHRUN" --standalone --nproc_per_node=8 \
  -m wm3d_v3.training.train \
  --cfg "$ALL_CFG" \
  --resume "$CURRENT_BEST" \
  --reset_optim \
  --print_every 25 \
  >>"$ALL_LOG" 2>&1 &

echo "$!" > "$ALL_PIDFILE"
echo "started_full_hunyuan_joint_pid=$(cat "$ALL_PIDFILE")"
