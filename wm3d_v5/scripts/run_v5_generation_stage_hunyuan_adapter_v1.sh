#!/usr/bin/env bash
set -euo pipefail

V5_ROOT=${V5_ROOT:-/data/Minko/world_model/wm3d_v5}
cd "$V5_ROOT"

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
WM_CFG=${WM_CFG:?WM_CFG is required}
WM_CKPT=${WM_CKPT:?WM_CKPT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

args=(
  scripts/train_hunyuan_latent_adapter.py
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
append_arg VAE_PRECISION --vae_precision
append_arg PRECISION --precision
append_arg EPOCHS --epochs
append_arg BATCH_SIZE_PER_GPU --batch_size_per_gpu
append_arg NUM_WORKERS --num_workers
append_arg LR --lr
append_arg WEIGHT_DECAY --weight_decay
append_arg WARMUP_STEPS --warmup_steps
append_arg GRAD_CLIP --grad_clip
append_arg LATENT_MSE_WEIGHT --latent_mse_weight
append_arg LATENT_L1_WEIGHT --latent_l1_weight
append_arg DECODED_L1_WEIGHT --decoded_l1_weight
append_arg DECODED_MOTION_L1_WEIGHT --decoded_motion_l1_weight
append_arg MAX_TRAIN_WINDOWS --max_train_windows
append_arg MAX_VAL_WINDOWS --max_val_windows
append_arg EVAL_BATCHES --eval_batches
append_arg PRINT_EVERY --print_every
append_arg SEED --seed
append_arg ADAPTER_HIDDEN --hidden
append_arg ADAPTER_N_BLOCKS --n_blocks
append_arg RESIDUAL_SCALE --residual_scale
append_flag RESIDUAL_FROM_ROUGH --residual_from_rough

mkdir -p "$OUT_DIR"
echo "hunyuan_adapter_stage_start wm_cfg=$WM_CFG wm_ckpt=$WM_CKPT out=$OUT_DIR nproc=$NPROC_PER_NODE"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  "$TORCHRUN" --nproc_per_node "$NPROC_PER_NODE" "${args[@]}"
else
  "$PY" "${args[@]}"
fi

echo "hunyuan_adapter_stage_done out=$OUT_DIR"
