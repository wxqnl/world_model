#!/usr/bin/env bash
set -euo pipefail

V5_ROOT=${V5_ROOT:-/data/Minko/world_model/wm3d_v5}
cd "$V5_ROOT"
export PYTHONPATH="$V5_ROOT:${PYTHONPATH:-}"  # ensure wm3d_v3 resolves to the v5 tree (adapter + stage3 edits live here)

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
WM_CFG=${WM_CFG:?WM_CFG is required}
WM_CKPT=${WM_CKPT:?WM_CKPT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

args=(
  scripts/train_hunyuan_dit_control_adapter.py
  --wm_cfg "$WM_CFG"
  --wm_ckpt "$WM_CKPT"
  --out_dir "$OUT_DIR"
)

append_arg() {
  local env_name=$1
  local flag=$2
  local value=${!env_name:-}
  if [[ -n "$value" ]]; then
    args+=("$flag" "$value")
  fi
}

append_flag() {
  local env_name=$1
  local flag=$2
  local value=${!env_name:-0}
  case "${value,,}" in
    1|true|yes|y|on) args+=("$flag") ;;
  esac
}

append_arg HUNYUAN_REPO --hunyuan_repo
append_arg HUNYUAN_MODEL_BASE --hunyuan_model_base
append_arg HUNYUAN_DIT_WEIGHT --hunyuan_dit_weight
append_arg HUNYUAN_MODEL_RESOLUTION --hunyuan_model_resolution
append_arg VAE_PRECISION --vae_precision
append_arg HUNYUAN_PRECISION --hunyuan_precision
append_arg TEXT_ENCODER_PRECISION --text_encoder_precision
append_arg TEXT_ENCODER_PRECISION_2 --text_encoder_precision_2
append_arg PRECISION --precision
append_arg EPOCHS --epochs
append_arg BATCH_SIZE_PER_GPU --batch_size_per_gpu
append_arg NUM_WORKERS --num_workers
append_arg LR --lr
append_arg WEIGHT_DECAY --weight_decay
append_arg WARMUP_STEPS --warmup_steps
append_arg GRAD_CLIP --grad_clip
append_arg VELOCITY_MSE_WEIGHT --velocity_mse_weight
append_arg VELOCITY_L1_WEIGHT --velocity_l1_weight
append_arg MAX_TRAIN_WINDOWS --max_train_windows
append_arg MAX_VAL_WINDOWS --max_val_windows
append_arg EVAL_BATCHES --eval_batches
append_arg PRINT_EVERY --print_every
append_arg SEED --seed
append_arg CONTROL_HIDDEN --hidden
append_arg CONTROL_SCALE --control_scale
append_arg FLOW_SHIFT --flow_shift
append_arg EMBEDDED_CFG_SCALE --embedded_cfg_scale
append_arg PATH_TYPE --path_type
append_arg CONTROL_CKPT --control_ckpt
append_flag HUNYUAN_USE_FP8 --hunyuan_use_fp8
append_flag HUNYUAN_NO_FP8 --no_hunyuan_use_fp8

mkdir -p "$OUT_DIR"
echo "hunyuan_dit_control_stage_start wm_cfg=$WM_CFG wm_ckpt=$WM_CKPT out=$OUT_DIR nproc=$NPROC_PER_NODE"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  echo "hunyuan_dit_control_stage_error nproc_gt_1_not_supported_in_v1 use_NPROC_PER_NODE_1" >&2
  exit 2
fi
"$PY" "${args[@]}"

echo "hunyuan_dit_control_stage_done out=$OUT_DIR"
