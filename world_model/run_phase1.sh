#!/usr/bin/env bash
set -u
cd /home/user01/Simo/geophys-feasibility
source /home/user01/Minko/reskip2/.venv/bin/activate

LOGDIR=results/phase1
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/run_phase1.log"

echo "=== phase1 runner started $(date -Iseconds) ===" | tee -a "$MASTER"

# Start vggt and vggt_noact immediately on GPUs 1 and 4 (no dep on dinov2 cache)
CUDA_VISIBLE_DEVICES=1 python -m src.phase1.train --cfg configs/phase1/default.yaml --runs vggt \
    >"$LOGDIR/train_vggt.log" 2>&1 &
PID_VGGT=$!
echo "launched vggt pid=$PID_VGGT on GPU1" | tee -a "$MASTER"

CUDA_VISIBLE_DEVICES=4 python -m src.phase1.train --cfg configs/phase1/default.yaml --runs vggt_noact \
    >"$LOGDIR/train_vggt_noact.log" 2>&1 &
PID_NOACT=$!
echo "launched vggt_noact pid=$PID_NOACT on GPU4" | tee -a "$MASTER"

# DINOv2 cache on GPU 0 (blocking this shell; other runs continue in bg)
echo "starting dinov2 cache on GPU0 at $(date -Iseconds)" | tee -a "$MASTER"
CUDA_VISIBLE_DEVICES=0 python -m src.phase1.cache_dinov2 --cfg configs/phase1/default.yaml \
    >"$LOGDIR/cache_dinov2.log" 2>&1
CACHE_RC=$?
echo "dinov2 cache exit=$CACHE_RC at $(date -Iseconds)" | tee -a "$MASTER"

if [ $CACHE_RC -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=0 python -m src.phase1.train --cfg configs/phase1/default.yaml --runs dinov2 \
        >"$LOGDIR/train_dinov2.log" 2>&1 &
    PID_DINO=$!
    echo "launched dinov2 pid=$PID_DINO on GPU0" | tee -a "$MASTER"
else
    echo "SKIPPING dinov2 training — cache failed" | tee -a "$MASTER"
    PID_DINO=
fi

# Wait for all training
wait $PID_VGGT;  echo "vggt done rc=$? at $(date -Iseconds)"       | tee -a "$MASTER"
wait $PID_NOACT; echo "vggt_noact done rc=$? at $(date -Iseconds)" | tee -a "$MASTER"
if [ -n "$PID_DINO" ]; then
    wait $PID_DINO; echo "dinov2 done rc=$? at $(date -Iseconds)" | tee -a "$MASTER"
fi

echo "=== phase1 runner finished $(date -Iseconds) ===" | tee -a "$MASTER"
