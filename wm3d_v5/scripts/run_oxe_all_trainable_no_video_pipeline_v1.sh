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
CACHED_MANIFEST=${CACHED_MANIFEST:-manifests/oxe_all_trainable_cached_no_video_v1.jsonl}
CURRENT_PIDFILE=${CURRENT_PIDFILE:-$LOG_DIR/train_oxe_fullpolicy_cached_v4_8gpu.pid}
CURRENT_BEST=${CURRENT_BEST:-results/wm3d_v3_p64_140m_p0_action_policy_oxe_fullpolicy_cached_v4_8gpu/ckpt/best.pt}
ALL_CFG=${ALL_CFG:-configs/v3_p64_140m_p0_action_policy_oxe_all_trainable_no_video_fullpolicy_v1_8gpu.yaml}
ALL_LOG=${ALL_LOG:-$LOG_DIR/train_oxe_all_trainable_no_video_fullpolicy_v1_8gpu.log}
ALL_PIDFILE=${ALL_PIDFILE:-$LOG_DIR/train_oxe_all_trainable_no_video_fullpolicy_v1_8gpu.pid}
CACHE_PIDS=${CACHE_PIDS:-$LOG_DIR/cache_oxe_all_trainable_no_video_v1.pids}

mkdir -p "$LOG_DIR"

"$PY" scripts/build_oxe_trainable_manifest.py \
  --input "$SOURCE_MANIFEST" \
  --output "$ELIGIBLE_MANIFEST" \
  --cache_root "$CACHE_ROOT" \
  --T 16 --k 8 --stride 4

if [[ -f "$CURRENT_PIDFILE" ]]; then
  current_pid="$(cat "$CURRENT_PIDFILE" || true)"
  if [[ -n "${current_pid:-}" ]] && ps -p "$current_pid" >/dev/null 2>&1; then
    echo "waiting_for_current_pid=$current_pid"
    while ps -p "$current_pid" >/dev/null 2>&1; do
      sleep 60
    done
  fi
fi

rm -f "$CACHE_PIDS"
touch "$CACHE_PIDS"

for shard in $(seq 0 $((WORLD - 1))); do
  shard_log="$LOG_DIR/cache_oxe_all_trainable_no_video_v1_shard${shard}.log"
  CUDA_VISIBLE_DEVICES="$shard" "$PY" scripts/cache_oxe.py \
    --manifest "$ELIGIBLE_MANIFEST" \
    --cache_root "$CACHE_ROOT" \
    --shard "$shard" \
    --world "$WORLD" \
    --batch_frames "$BATCH_FRAMES" \
    --no_rgb \
    --no_geom \
    >>"$shard_log" 2>&1 &
  echo "$!" >> "$CACHE_PIDS"
done

cache_status=0
while read -r pid; do
  if [[ -n "$pid" ]]; then
    wait "$pid" || cache_status=$?
  fi
done < "$CACHE_PIDS"
if [[ "$cache_status" -ne 0 ]]; then
  echo "cache_status=$cache_status"
  exit "$cache_status"
fi

"$PY" scripts/build_oxe_trainable_manifest.py \
  --input "$ELIGIBLE_MANIFEST" \
  --output "$CACHED_MANIFEST" \
  --cache_root "$CACHE_ROOT" \
  --T 16 --k 8 --stride 4 \
  --require_policy_cache

if [[ ! -s "$CACHED_MANIFEST" ]]; then
  echo "empty_cached_manifest=$CACHED_MANIFEST"
  exit 2
fi
if [[ ! -f "$CURRENT_BEST" ]]; then
  echo "missing_current_best=$CURRENT_BEST"
  exit 3
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
  --no_pixel \
  --print_every 50 \
  >>"$ALL_LOG" 2>&1 &

echo "$!" > "$ALL_PIDFILE"
echo "started_all_oxe_pid=$(cat "$ALL_PIDFILE")"
