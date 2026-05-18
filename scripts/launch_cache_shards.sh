#!/usr/bin/env bash
# Launch N parallel cache.py shards, one per GPU, pinned via CUDA_VISIBLE_DEVICES.
#
# Usage:
#   scripts/launch_cache_shards.sh --gpus 0,1,2,3 --cfg configs/phase1/paper_scale.yaml \
#                                  --log_dir results/phase1_scale/cache_logs
#
# Why a separate launcher (not launch_ddp.sh): cache.py is embarrassingly parallel
# across clips, not gradient-coupled. torchrun would just add NCCL overhead and a
# pointless rendezvous. Each shard reads a fixed slice of the manifest by index
# (i % num_shards == shard_index) so it's resumable: re-run after a crash and the
# survivor shards skip already-cached clips.
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=""
CFG="configs/phase1/paper_scale.yaml"
LOG_DIR="results/phase1_scale/cache_logs"
LIMIT=""
FORCE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)    GPUS="$2"; shift 2 ;;
        --cfg)     CFG="$2";  shift 2 ;;
        --log_dir) LOG_DIR="$2"; shift 2 ;;
        --limit)   LIMIT="--limit $2"; shift 2 ;;
        --force)   FORCE="--force"; shift ;;
        *)         echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$GPUS" ]]; then
    echo "ERROR: --gpus (comma list of GPU indices) required" >&2; exit 2
fi
mkdir -p "$LOG_DIR"

# Source the project venv if present (matches launch_ddp.sh behavior).
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [[ -f /home/user01/Minko/reskip2/.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /home/user01/Minko/reskip2/.venv/bin/activate
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_SHARDS=${#GPU_LIST[@]}
ts=$(date +%Y%m%d_%H%M%S)

echo "==========================================================="
echo "  cfg:         $CFG"
echo "  gpus:        ${GPU_LIST[*]}  (num_shards=$NUM_SHARDS)"
echo "  log dir:     $LOG_DIR"
echo "==========================================================="

pids=()
for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    log_file="$LOG_DIR/cache_shard_${i}_gpu${gpu}_${ts}.log"
    echo "  → shard $i on GPU $gpu → $log_file"
    CUDA_VISIBLE_DEVICES="$gpu" \
        nohup python -m src.phase1.cache \
            --cfg "$CFG" \
            --shard_index "$i" --num_shards "$NUM_SHARDS" \
            $LIMIT $FORCE \
        >"$log_file" 2>&1 &
    pids+=($!)
done

echo "  pids: ${pids[*]}"
echo "  wait pid example: tail -f $LOG_DIR/cache_shard_0_*.log"
echo "  kill all: kill ${pids[*]}"

# Keep this script alive so users can ctrl-C all shards at once.
wait "${pids[@]}"
echo "all shards complete"
