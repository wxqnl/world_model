#!/usr/bin/env bash
set -u
cd /home/user01/Simo/geophys-feasibility
source /home/user01/Minko/reskip2/.venv/bin/activate

LOGDIR=results/phase2
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/run_phase2.log"

echo "=== phase2 runner started $(date -Iseconds) ===" | tee -a "$MASTER"

CUDA_VISIBLE_DEVICES=0 python -m scripts.phase2.train_generative \
    --cfg configs/phase2/default.yaml --variant flow_only \
    >"$LOGDIR/train_flow_only.log" 2>&1 &
PID_A=$!
echo "launched flow_only pid=$PID_A on GPU0" | tee -a "$MASTER"

CUDA_VISIBLE_DEVICES=1 python -m scripts.phase2.train_generative \
    --cfg configs/phase2/default.yaml --variant flow_coupled \
    >"$LOGDIR/train_flow_coupled.log" 2>&1 &
PID_B=$!
echo "launched flow_coupled pid=$PID_B on GPU1" | tee -a "$MASTER"

wait $PID_A; echo "flow_only    done rc=$? at $(date -Iseconds)" | tee -a "$MASTER"
wait $PID_B; echo "flow_coupled done rc=$? at $(date -Iseconds)" | tee -a "$MASTER"

echo "=== phase2 runner finished $(date -Iseconds) ===" | tee -a "$MASTER"
