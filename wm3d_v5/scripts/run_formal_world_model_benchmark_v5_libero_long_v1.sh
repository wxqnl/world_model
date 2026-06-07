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
SUITE=${SUITE:-LIBERO-Long}
OUT_DIR=${OUT_DIR:-results/wm3d_v5_p64_140m_stage0_native3d_exp8192_w2_3node_v1/formal_world_model_benchmark_libero_long_v1}

CACHE_ROOT=${CACHE_ROOT:-/data/Minko/world_model/wm3d_v5/results/wm3d_libero_action_policy_lowdimhist_libero10_full_start_stride4_v1_cache}
CACHE_MANIFEST=${CACHE_MANIFEST:-$OUT_DIR/libero10_current_cache_manifest.jsonl}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_WINDOWS_PER_TASK=${MAX_WINDOWS_PER_TASK:-16}
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
cfg.setdefault("model", {})["enable_action_policy"] = False
cfg.setdefault("train", {})["num_workers"] = min(int(cfg.get("train", {}).get("num_workers", 2)), 4)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"wrote benchmark cfg {dst}")
PY

"$PY" - "$CACHE_ROOT" "$CACHE_MANIFEST" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
parts = sorted(root.glob("shard_*/manifest.jsonl"))
rows = []
seen = set()
for path in parts:
    for line in path.read_text().splitlines():
        if not line.strip() or line in seen:
            continue
        seen.add(line)
        rows.append(line)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(rows) + ("\n" if rows else ""))
print(f"wrote cache manifest {out} rows={len(rows)} sources={len(parts)}")
if len(rows) < 2:
    raise SystemExit("not enough LIBERO cached windows for video benchmark")
PY

wait_for_gpu

VIDEO_JSON="$OUT_DIR/libero_long_video_quality_i3d.json"
SAMPLE_MANIFEST="$OUT_DIR/libero_long_sample_manifest.jsonl"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m wm3d_v3.eval.libero_video_quality_eval \
  --cfg "$BENCH_CFG" \
  --ckpt "$CKPT" \
  --cache_manifest "$CACHE_MANIFEST" \
  --out "$VIDEO_JSON" \
  --batch_size "$BATCH_SIZE" \
  --num_workers 4 \
  --balanced_tasks \
  --max_windows_per_task "$MAX_WINDOWS_PER_TASK" \
  --dataset_name libero_long \
  --include_lpips \
  --include_fvd \
  --fvd_backend i3d_torchscript \
  --i3d_model_path external/fvd_i3d/i3d_torchscript.pt \
  --sample_manifest_out "$SAMPLE_MANIFEST" \
  --log_every 1 \
  > "$OUT_DIR/logs/libero_long_video_quality_i3d.log" 2>&1

"$PY" -m wm3d_v3.eval.formal_world_model_benchmark \
  --video_quality_json "$VIDEO_JSON" \
  --out_dir "$OUT_DIR/final" \
  --model_name "$MODEL_NAME" \
  --suite "$SUITE" \
  --cfg "$BENCH_CFG" \
  --ckpt "$CKPT" \
  --literature_baselines "$LITERATURE_BASELINES" \
  > "$OUT_DIR/logs/aggregate.log" 2>&1

echo "formal_world_model_benchmark_v5_libero_long_done out=$OUT_DIR/final"
