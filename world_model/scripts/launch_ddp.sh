#!/usr/bin/env bash
# Shared-machine-friendly DDP launcher.
#
# 1. Picks N free GPUs (memory.used < FREE_MEM_MIB and util < FREE_UTIL_PCT).
# 2. Refuses to launch if fewer than --gpus are free (so we never bump someone
#    else's job).
# 3. Pins CUDA_VISIBLE_DEVICES, runs torchrun, prints final pin + a release note.
#
# Usage:
#   scripts/launch_ddp.sh --gpus 8 -- \
#       python -m src.phase1.train_ddp --cfg configs/phase1/paper_scale.yaml --run vggt_noact
#
#   scripts/launch_ddp.sh --gpus 4 --port 29501 -- \
#       python scripts/phase2/train_generative_ddp.py --cfg configs/phase2/paper_scale.yaml --variant flow_coupled
#
# Tunables (env vars):
#   FREE_MEM_MIB   max memory.used to consider a GPU free   (default 500)
#   FREE_UTIL_PCT  max utilization.gpu to consider it free  (default 5)
#   PREFER_GPUS    comma list of preferred GPU indices (subset; rest auto-filled)
#
# Notes:
# * Always source the project venv inside this script to keep environment
#   identical between rank-0 logs and worker processes.
# * Set DRY_RUN=1 to print the chosen pin and command without launching.

set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=""
MASTER_PORT=29500
EXTRA_TORCHRUN_ARGS=""
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)        NUM_GPUS="$2"; shift 2 ;;
        --port)        MASTER_PORT="$2"; shift 2 ;;
        --torchrun)    EXTRA_TORCHRUN_ARGS="$2"; shift 2 ;;
        --)            shift; PASS_ARGS=("$@"); break ;;
        *)             echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$NUM_GPUS" ]]; then
    echo "ERROR: --gpus N required" >&2; exit 2
fi
if [[ ${#PASS_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: command after -- required" >&2; exit 2
fi

FREE_MEM_MIB="${FREE_MEM_MIB:-500}"
FREE_UTIL_PCT="${FREE_UTIL_PCT:-5}"
PREFER_GPUS="${PREFER_GPUS:-}"

# ---------- discover free GPUs
mapfile -t SMI < <(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
               --format=csv,noheader,nounits | tr -d ' '
)
FREE=()
ALL_INDICES=()
for line in "${SMI[@]}"; do
    IFS=',' read -r idx mem util <<< "$line"
    ALL_INDICES+=("$idx")
    if [[ "$mem" -lt "$FREE_MEM_MIB" && "$util" -lt "$FREE_UTIL_PCT" ]]; then
        FREE+=("$idx")
    fi
done

# ---------- honor PREFER_GPUS subset
PICK=()
if [[ -n "$PREFER_GPUS" ]]; then
    IFS=',' read -ra PREF <<< "$PREFER_GPUS"
    for g in "${PREF[@]}"; do
        for f in "${FREE[@]}"; do
            if [[ "$g" == "$f" ]]; then PICK+=("$g"); break; fi
        done
    done
fi
# fill remaining slots from the free pool, skipping any already picked
for f in "${FREE[@]}"; do
    [[ ${#PICK[@]} -ge $NUM_GPUS ]] && break
    skip=0
    for p in "${PICK[@]:-}"; do [[ "$p" == "$f" ]] && skip=1 && break; done
    [[ $skip -eq 0 ]] && PICK+=("$f")
done

if [[ ${#PICK[@]} -lt $NUM_GPUS ]]; then
    echo "==========================================================="
    echo "REFUSING TO LAUNCH: only ${#PICK[@]}/$NUM_GPUS GPUs free."
    echo "Free pool: ${FREE[*]:-none}"
    echo "Lower --gpus, set PREFER_GPUS=, or wait for other jobs to finish."
    echo "Currently busy GPUs (per nvidia-smi):"
    for line in "${SMI[@]}"; do
        IFS=',' read -r idx mem util <<< "$line"
        if [[ "$mem" -ge "$FREE_MEM_MIB" || "$util" -ge "$FREE_UTIL_PCT" ]]; then
            printf "  GPU %s  mem=%s MiB  util=%s%%\n" "$idx" "$mem" "$util"
        fi
    done
    echo "==========================================================="
    exit 3
fi

PICK_CSV="$(IFS=,; echo "${PICK[*]:0:$NUM_GPUS}")"
HOST="$(hostname -s)"
echo "==========================================================="
echo "  host:           $HOST"
echo "  free pool:      ${FREE[*]}"
echo "  pinning to:     CUDA_VISIBLE_DEVICES=$PICK_CSV  ($NUM_GPUS GPU(s))"
echo "  master_port:    $MASTER_PORT"
echo "  command:        ${PASS_ARGS[*]}"
echo "  release me:     pkill -f \"$(basename ${PASS_ARGS[0]})\"   # if needed"
echo "==========================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] not launching"
    exit 0
fi

# ---------- env hygiene for shared box
export CUDA_VISIBLE_DEVICES="$PICK_CSV"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_ASYNC_ERROR_HANDLING=1
# bind nccl to a stable interface if you have multiple — uncomment if needed:
# export NCCL_SOCKET_IFNAME=eth0

# Activate the project venv (matches run_phase1.sh / run_phase2.sh).
if [[ -f /home/user01/Minko/reskip2/.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /home/user01/Minko/reskip2/.venv/bin/activate
fi

# Single-GPU: avoid torchrun + nccl init entirely. Trainer's _ddp_setup falls
# back to (rank=0, world=1) when RANK is unset.
if [[ "$NUM_GPUS" -eq 1 ]]; then
    exec "${PASS_ARGS[@]}"
fi

exec torchrun \
    --nproc_per_node "$NUM_GPUS" \
    --nnodes 1 \
    --master_port "$MASTER_PORT" \
    $EXTRA_TORCHRUN_ARGS \
    "${PASS_ARGS[@]}"
