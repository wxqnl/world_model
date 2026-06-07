#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/Minko/world_model}"
PROJ="${PROJ:-$ROOT/wm3d_v3}"
RAW_ROOT="${RAW_ROOT:-/data/Minko/datasets/oxe_hf}"
CACHE_ROOT="${CACHE_ROOT:-/data/Minko/datasets/cache/wm3d_v3}"
MANIFEST="${MANIFEST:-$PROJ/manifests/oxe_all_downloaded.jsonl}"
CTRL_PID_FILE="${CTRL_PID_FILE:-/data/Minko/logs/p64_138m_actioncond_full_controller_latest.pid}"
CTRL_LOG="${CTRL_LOG:-/data/Minko/logs/p64_138m_actioncond_full_controller_latest.log}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs}"
POLL_SECONDS="${POLL_SECONDS:-300}"
WORLD="${WORLD:-8}"
BATCH_FRAMES="${BATCH_FRAMES:-16}"
RUN_P256="${RUN_P256:-1}"

RUN_ID="${RUN_ID:-full_oxe_$(date +%Y%m%d_%H%M%S)}"
LOG="$LOG_DIR/prepare_full_oxe_${RUN_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] full OXE preparation watcher started"
echo "root=$ROOT"
echo "proj=$PROJ"
echo "raw_root=$RAW_ROOT"
echo "cache_root=$CACHE_ROOT"
echo "manifest=$MANIFEST"
echo "ctrl_pid_file=$CTRL_PID_FILE"
echo "ctrl_log=$CTRL_LOG"
echo "world=$WORLD batch_frames=$BATCH_FRAMES run_p256=$RUN_P256"

export PATH="/data/Minko/.venvs/wm3d/bin:$PATH"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

wait_for_current_run() {
  local pid=""
  if [[ -s "$CTRL_PID_FILE" ]]; then
    pid="$(cat "$CTRL_PID_FILE")"
  fi
  if [[ -n "$pid" ]]; then
    echo "[$(date)] waiting for current training/posttrain controller pid $pid"
    while kill -0 "$pid" 2>/dev/null; do
      tail -n 8 "$CTRL_LOG" 2>/dev/null || true
      sleep "$POLL_SECONDS"
    done
  else
    echo "[$(date)] no controller pid file found; checking process pattern"
    while pgrep -f "v3_p64_138m_actioncond_full.yaml|p64_138m_actioncond_full_controller" >/dev/null; do
      tail -n 8 "$CTRL_LOG" 2>/dev/null || true
      sleep "$POLL_SECONDS"
    done
  fi

  if grep -Eq "TRAIN_EXIT=[1-9][0-9]*|Traceback|CUDA out of memory|FAILED" "$CTRL_LOG" 2>/dev/null; then
    echo "[$(date)] current run log contains a failure marker; aborting full OXE preparation"
    grep -En "TRAIN_EXIT=[1-9][0-9]*|Traceback|CUDA out of memory|FAILED" "$CTRL_LOG" | tail -n 40 || true
    exit 1
  fi
  echo "[$(date)] current run is complete; starting full OXE preparation"
}

print_manifest_summary() {
  python - <<'PY'
import json, os
from collections import Counter
from pathlib import Path
p = Path(os.environ["MANIFEST"])
counts = Counter()
frames = Counter()
n = 0
for line in p.open():
    r = json.loads(line)
    n += 1
    d = r.get("dataset", "?")
    counts[d] += 1
    frames[d] += int(r.get("n_frames", 0))
print(f"manifest={p}")
print(f"clips={n}")
print("counts=" + json.dumps(dict(counts), sort_keys=True))
print("frames=" + json.dumps(dict(frames), sort_keys=True))
print(f"total_frames={sum(frames.values())}")
PY
}

run_sharded() {
  local name="$1"
  shift
  local -a pids=()
  echo "[$(date)] launching $name across $WORLD shards"
  for shard in $(seq 0 $((WORLD - 1))); do
    local shard_log="$LOG_DIR/${name}_${RUN_ID}_shard${shard}.log"
    (
      cd "$ROOT"
      export CUDA_VISIBLE_DEVICES="$shard"
      "$@" --shard "$shard" --world "$WORLD" --batch_frames "$BATCH_FRAMES"
    ) >"$shard_log" 2>&1 &
    pids+=("$!")
    echo "  shard=$shard pid=${pids[-1]} log=$shard_log"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      :
    else
      local status=$?
      echo "[$(date)] $name worker failed pid=$pid status=$status"
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date)] $name failed; showing recent shard errors"
    grep -En "Traceback|RuntimeError|CUDA out of memory|FAILED|Error" "$LOG_DIR/${name}_${RUN_ID}"_shard*.log | tail -n 80 || true
    exit 1
  fi
  echo "[$(date)] $name complete"
}

cache_counts() {
  echo "[$(date)] cache counts"
  for d in "$CACHE_ROOT"/vggt_pooled "$CACHE_ROOT"/vggt_geom "$CACHE_ROOT"/rgb_256 "$CACHE_ROOT"/actions "$CACHE_ROOT"/qwen_taskemb "$CACHE_ROOT"/vggt_p256; do
    if [[ -d "$d" ]]; then
      printf "%s " "$d"
      find "$d" -maxdepth 1 -type f | wc -l
    fi
  done
  df -h "$CACHE_ROOT" || true
}

wait_for_current_run

cd "$ROOT"
echo "[$(date)] building full downloaded OXE manifest"
python "$PROJ/scripts/build_oxe_manifest.py" \
  --root "$RAW_ROOT" \
  --out "$MANIFEST" \
  --datasets bridge fractal20220817_data taco_play jaco_play kuka \
  --fractal_subsample 1.0 \
  --seed 42
export MANIFEST
print_manifest_summary

mkdir -p "$CACHE_ROOT"/{vggt_pooled,vggt_geom,rgb_256,actions,qwen_taskemb}
cache_counts

run_sharded "cache_full_oxe_base" \
  python "$PROJ/scripts/cache_oxe.py" \
    --manifest "$MANIFEST" \
    --cache_root "$CACHE_ROOT" \
    --skip_qwen
cache_counts
