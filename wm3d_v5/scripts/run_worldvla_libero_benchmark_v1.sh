#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/data/Minko/egl/10_nvidia.json}"

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/Minko/external/world_model_eval_sources/WorldVLA/rynnvla-002}"
PROCESSED_ROOT="${PROCESSED_ROOT:-/data/Minko/benchmarks/LIBERO/processed_data_worldvla_val}"
RESULT_ROOT="${RESULT_ROOT:-/data/Minko/world_model/wm3d_v5/results/worldvla_libero_official_v1}"
CKPT="${CKPT:-/data/Minko/world_model/wm3d_v3/results/wm3d_v3_p64_1b_stage2_action_scaffold_from_stage1p5_3node_v1/ckpt/best.pt}"
I3D="${I3D:-/data/Minko/world_model/wm3d_v5/external/fvd_i3d/i3d_torchscript.pt}"
SUITES="${SUITES:-10,goal,object,spatial}"
RUN_ID="${RUN_ID:-wm3d_v5_1b_stage2_worldvla_official}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

mkdir -p "$RESULT_ROOT"/logs "$RESULT_ROOT"/summaries "$RESULT_ROOT"/videos "$RESULT_ROOT"/metrics

cmd="${1:-all}"

run_prepare() {
  IFS=',' read -r -a gpu_array <<< "$GPUS"
  world_size="${#gpu_array[@]}"
  for rank in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$rank]}"
    MUJOCO_EGL_DEVICE_ID="$gpu" "$PY" -m wm3d_v3.eval.worldvla_libero_protocol prepare-val \
      --official_root "$OFFICIAL_ROOT" \
      --processed_root "$PROCESSED_ROOT" \
      --suites "$SUITES" \
      --resolution 512 \
      --skip_existing \
      --rank "$rank" \
      --world_size "$world_size" \
      --out_summary "$RESULT_ROOT/summaries/prepare_${RUN_ID}_rank${rank}.json" \
      2>&1 | tee "$RESULT_ROOT/logs/prepare_${RUN_ID}_rank${rank}.log" &
  done
  wait
}

run_export() {
  IFS=',' read -r -a gpu_array <<< "$GPUS"
  world_size="${#gpu_array[@]}"
  for rank in "${!gpu_array[@]}"; do
    gpu="${gpu_array[$rank]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m wm3d_v3.eval.worldvla_libero_protocol export-videos \
      --official_root "$OFFICIAL_ROOT" \
      --processed_root "$PROCESSED_ROOT" \
      --suites "$SUITES" \
      --ckpt "$CKPT" \
      --out_dir "$RESULT_ROOT/videos/$RUN_ID" \
      --out_summary "$RESULT_ROOT/summaries/export_${RUN_ID}_rank${rank}.json" \
      --device cuda:0 \
      --qwen_device cuda:0 \
      --rank "$rank" \
      --world_size "$world_size" \
      --run_id "$RUN_ID" \
      --log_every 50 \
      2>&1 | tee "$RESULT_ROOT/logs/export_${RUN_ID}_rank${rank}.log" &
  done
  wait
}

run_metric() {
  "$PY" - <<PY 2>&1 | tee "$RESULT_ROOT/metrics/official_metric_${RUN_ID}.log"
import argparse
import runpy
import sys

sys.argv = [
    "calculate_world_model_performance.py",
    "--i3d_model_path", "$I3D",
    "--folder_world_model", "$RESULT_ROOT/videos/$RUN_ID",
    "--folder_action_world_model", "$RESULT_ROOT/videos/$RUN_ID",
]
runpy.run_path(
    "$OFFICIAL_ROOT/exps_libero_world_model/calculate_world_model_performance.py",
    run_name="__main__",
    init_globals={"argparse": argparse},
)
PY
}

case "$cmd" in
  prepare)
    run_prepare
    ;;
  export)
    run_export
    ;;
  metric)
    run_metric
    ;;
  all)
    run_prepare
    run_export
    run_metric
    ;;
  *)
    echo "usage: $0 [prepare|export|metric|all]" >&2
    exit 2
    ;;
esac
