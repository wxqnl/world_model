#!/usr/bin/env bash
# =============================================================================
# Stage3: text->video generation (Hunyuan DiT-control) on top of a FROZEN world
# model. Parameterized — point WM_CFG/WM_CKPT at any trained world checkpoint:
#   - now:   v3 300m stage2 (depth-only)   e.g. results/wm3d_v3_p64_300m_stage2_.../ckpt/best.pt
#   - later: a native3d stage checkpoint   (re-point WM_CFG/WM_CKPT, no code change)
#
# Frozen: world model + Hunyuan VAE/DiT/text-encoders. Trainable: zero-init DiT
# control adapter only. Text conditioning is ON by default (the trainer passes
# --load_task_text; requires the window_dataset load_task_text passthrough fix).
#
# Required env:  WM_CFG  WM_CKPT  OUT_DIR
# Optional env:  EPOCHS MAX_TRAIN_WINDOWS LR CONTROL_HIDDEN FLOW_SHIFT
#                EMBEDDED_CFG_SCALE PATH_TYPE HUNYUAN_REPO HUNYUAN_MODEL_BASE ...
# =============================================================================
set -euo pipefail

V5_ROOT=${V5_ROOT:-/data/Minko/world_model/wm3d_v5}
cd "$V5_ROOT"

# ---- required: which world model to condition on, and where to write ----
export WM_CFG=${WM_CFG:?set WM_CFG (world-model config used to rebuild+condition, e.g. configs/v3_p64_300m_stage2_...yaml)}
export WM_CKPT=${WM_CKPT:?set WM_CKPT (trained world-model checkpoint, e.g. results/.../ckpt/best.pt)}
export OUT_DIR=${OUT_DIR:?set OUT_DIR (stage3 generation output dir)}

# ---- Hunyuan assets (already installed on this box) ----
export HUNYUAN_REPO=${HUNYUAN_REPO:-/data/Minko/external/HunyuanVideo}
export HUNYUAN_MODEL_BASE=${HUNYUAN_MODEL_BASE:-/data/Minko/models/hunyuan_video}

# ---- stage3 training defaults (override via env) ----
export EPOCHS=${EPOCHS:-2}
export BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-1}
export NUM_WORKERS=${NUM_WORKERS:-4}
export LR=${LR:-1e-4}
export WARMUP_STEPS=${WARMUP_STEPS:-100}
export MAX_TRAIN_WINDOWS=${MAX_TRAIN_WINDOWS:-20000}
export MAX_VAL_WINDOWS=${MAX_VAL_WINDOWS:-512}
export EVAL_BATCHES=${EVAL_BATCHES:-16}
export PRINT_EVERY=${PRINT_EVERY:-20}
export CONTROL_HIDDEN=${CONTROL_HIDDEN:-256}
export FLOW_SHIFT=${FLOW_SHIFT:-7.0}
export EMBEDDED_CFG_SCALE=${EMBEDDED_CFG_SCALE:-6.0}
export PATH_TYPE=${PATH_TYPE:-noise}
export NPROC_PER_NODE=1   # DiT-control trainer is single-process in v1

echo "stage3_generation_start wm_cfg=${WM_CFG} wm_ckpt=${WM_CKPT} out=${OUT_DIR}"
echo "  text-conditioning=ON (trainer --load_task_text default) epochs=${EPOCHS} max_train_windows=${MAX_TRAIN_WINDOWS} control_hidden=${CONTROL_HIDDEN}"
mkdir -p "${OUT_DIR}"
bash "${V5_ROOT}/scripts/run_v5_generation_stage_hunyuan_dit_control_v1.sh"
echo "stage3_generation_done out=${OUT_DIR}"
