#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PY=${PY:-/data/Minko/.venvs/wm3d/bin/python}
GPU=${GPU:-7}
CFG=${CFG:-configs/v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1.yaml}
CKPT=${CKPT:-results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/best.pt}
OUT_DIR=${OUT_DIR:-results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_BATCHES_PER_DATASET=${MAX_BATCHES_PER_DATASET:-4}
N_VIZ=${N_VIZ:-6}

mkdir -p "$OUT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m wm3d_v3.eval.world3d_claim_eval \
  --cfg "$CFG" \
  --ckpt "$CKPT" \
  --out "$OUT_DIR/world3d_claim_balanced.json" \
  --split val \
  --batch_size "$BATCH_SIZE" \
  --balanced_datasets \
  --max_batches_per_dataset "$MAX_BATCHES_PER_DATASET" \
  --variants zero sign_flip scaled shuffled grip_toggle \
  --viz_dir "$OUT_DIR/visuals" \
  --n_viz "$N_VIZ" \
  --viz_per_dataset 1 \
  --log_every 1

echo "world3d_native_benchmark_v2_done out=$OUT_DIR"
