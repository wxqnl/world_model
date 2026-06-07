#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

PY=${PY:-/data/Minko/.venvs/wm3d/bin/python}
GPU=${GPU:-0}
GPU_WAIT_THRESHOLD_MIB=${GPU_WAIT_THRESHOLD_MIB:-5000}
GPU_WAIT_POLL_SEC=${GPU_WAIT_POLL_SEC:-300}

CFG=${CFG:-configs/_eval_v5_p64_140m_stage0_native3d_exp8192_w2_loadgeom_v1.yaml}
CKPT=${CKPT:-results/wm3d_v5_p64_140m_stage0_native3d_exp8192_w2_3node_v1/ckpt/best.pt}
MODEL_NAME=${MODEL_NAME:-WM3D-v5-140M-stage0}
SUITE=${SUITE:-WM3D-v5-OXE-val}
OUT_DIR=${OUT_DIR:-results/wm3d_v5_p64_140m_stage0_native3d_exp8192_w2_3node_v1/formal_world_model_benchmark_v1}

BATCH_SIZE=${BATCH_SIZE:-4}
MAX_BATCHES_PER_DATASET=${MAX_BATCHES_PER_DATASET:-8}
N_VIZ=${N_VIZ:-8}
FVD_CONTEXT_FRAMES=${FVD_CONTEXT_FRAMES:-0}
LITERATURE_BASELINES=${LITERATURE_BASELINES:-configs/literature_world_model_baselines_v1.json}

mkdir -p "$OUT_DIR/logs"

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | awk '{print $1}')
    if [[ "${used:-999999}" -lt "$GPU_WAIT_THRESHOLD_MIB" ]]; then
      echo "gpu_ready gpu=$GPU memory_used_mib=$used threshold_mib=$GPU_WAIT_THRESHOLD_MIB"
      return 0
    fi
    echo "waiting_for_gpu gpu=$GPU memory_used_mib=$used threshold_mib=$GPU_WAIT_THRESHOLD_MIB"
    sleep "$GPU_WAIT_POLL_SEC"
  done
}

BENCH_CFG="$OUT_DIR/v5_world_benchmark_cfg.yaml"
"$PY" - "$CFG" "$BENCH_CFG" <<'PY'
import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text())
model = cfg.setdefault("model", {})
model["enable_action_policy"] = False
train = cfg.setdefault("train", {})
train["num_workers"] = min(int(train.get("num_workers", 2)), 4)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote benchmark cfg {dst}")
PY

wait_for_gpu

VIDEO_JSON="$OUT_DIR/video_quality_i3d.json"
SAMPLE_MANIFEST="$OUT_DIR/sample_manifest.jsonl"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m wm3d_v3.eval.video_quality_eval \
  --cfg "$BENCH_CFG" \
  --ckpt "$CKPT" \
  --out "$VIDEO_JSON" \
  --split val \
  --batch_size "$BATCH_SIZE" \
  --num_workers 4 \
  --balanced_datasets \
  --max_batches_per_dataset "$MAX_BATCHES_PER_DATASET" \
  --include_lpips \
  --include_fvd \
  --fvd_backend i3d_torchscript \
  --i3d_model_path external/fvd_i3d/i3d_torchscript.pt \
  --fvd_context_frames "$FVD_CONTEXT_FRAMES" \
  --sample_manifest_out "$SAMPLE_MANIFEST" \
  --log_every 1 \
  > "$OUT_DIR/logs/video_quality_i3d.log" 2>&1

NATIVE_JSON="$OUT_DIR/native3d_action_counterfactual.json"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m wm3d_v3.eval.world3d_claim_eval \
  --cfg "$BENCH_CFG" \
  --ckpt "$CKPT" \
  --out "$NATIVE_JSON" \
  --split val \
  --batch_size "$BATCH_SIZE" \
  --balanced_datasets \
  --max_batches_per_dataset "$MAX_BATCHES_PER_DATASET" \
  --variants zero sign_flip scaled shuffled grip_toggle \
  --viz_dir "$OUT_DIR/native3d_visuals" \
  --n_viz "$N_VIZ" \
  --viz_per_dataset 1 \
  --log_every 1 \
  > "$OUT_DIR/logs/native3d_action_counterfactual.log" 2>&1

"$PY" -m wm3d_v3.eval.formal_world_model_benchmark \
  --video_quality_json "$VIDEO_JSON" \
  --native3d_json "$NATIVE_JSON" \
  --out_dir "$OUT_DIR/final" \
  --model_name "$MODEL_NAME" \
  --suite "$SUITE" \
  --cfg "$BENCH_CFG" \
  --ckpt "$CKPT" \
  --literature_baselines "$LITERATURE_BASELINES" \
  > "$OUT_DIR/logs/aggregate.log" 2>&1

echo "formal_world_model_benchmark_v5_done out=$OUT_DIR/final"
