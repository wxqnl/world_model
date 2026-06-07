#!/usr/bin/env bash
set -euo pipefail

V5_ROOT=${V5_ROOT:-/data/Minko/world_model/wm3d_v5}
cd "$V5_ROOT"

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
CFG=${CFG:?CFG is required}
CKPT_DIR=${CKPT_DIR:?CKPT_DIR is required}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
CANARY_GPU=${CANARY_GPU:-7}
MAX_BATCHES=${MAX_BATCHES:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
N_GIFS=${N_GIFS:-2}
N_HUNYUAN_GIFS=${N_HUNYUAN_GIFS:-1}
RUN_RGB_METRICS=${RUN_RGB_METRICS:-0}
RUN_WORLD_PRIOR_EVAL=${RUN_WORLD_PRIOR_EVAL:-auto}
RUN_WORLD_PRIOR_PIXEL=${RUN_WORLD_PRIOR_PIXEL:-0}
RUN_ROUGH_GIF=${RUN_ROUGH_GIF:-auto}
RUN_HUNYUAN_LATENT_DEMO=${RUN_HUNYUAN_LATENT_DEMO:-0}
RUN_HUNYUAN_DIT_GENERATION=${RUN_HUNYUAN_DIT_GENERATION:-0}
ALLOW_REAL_HUNYUAN_GPU_GENERATION=${ALLOW_REAL_HUNYUAN_GPU_GENERATION:-0}
N_HUNYUAN_DIT_CLIPS=${N_HUNYUAN_DIT_CLIPS:-1}
HUNYUAN_DIT_HEIGHT=${HUNYUAN_DIT_HEIGHT:-320}
HUNYUAN_DIT_WIDTH=${HUNYUAN_DIT_WIDTH:-512}
HUNYUAN_DIT_FRAMES=${HUNYUAN_DIT_FRAMES:-9}
HUNYUAN_DIT_STEPS=${HUNYUAN_DIT_STEPS:-8}
HUNYUAN_DIT_SEED=${HUNYUAN_DIT_SEED:-0}
HUNYUAN_DIT_CONTROL_SCALE=${HUNYUAN_DIT_CONTROL_SCALE:-1.0}

mkdir -p "$OUT_ROOT" "$LOG_DIR"

last_key=""
echo "watch_canary_start cfg=$CFG ckpt_dir=$CKPT_DIR out_root=$OUT_ROOT gpu=$CANARY_GPU interval=$INTERVAL_SECONDS"

while true; do
  ckpt=""
  if [[ -s "$CKPT_DIR/latest.pt" ]]; then
    ckpt="$CKPT_DIR/latest.pt"
  elif [[ -s "$CKPT_DIR/best.pt" ]]; then
    ckpt="$CKPT_DIR/best.pt"
  fi

  if [[ -n "$ckpt" ]]; then
    key="$(stat -c %n:%Y:%s "$ckpt" 2>/dev/null || true)"
    if [[ -n "$key" && "$key" != "$last_key" ]]; then
      step="$($PY - "$ckpt" <<PY 2>/dev/null || true
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(sd.get("step", "unknown") if isinstance(sd, dict) else "unknown")
PY
)"
      if [[ -z "$step" ]]; then
        step="unknown"
      fi
      stamp="$(date +%Y%m%d_%H%M%S)"
      out_dir="$OUT_ROOT/step_${step}_${stamp}"
      log="$LOG_DIR/canary_step_${step}_${stamp}.log"
      echo "canary_launch ckpt=$ckpt step=$step out=$out_dir log=$log"
      if CUDA_VISIBLE_DEVICES="$CANARY_GPU"         V5_ROOT="$V5_ROOT"         VENV_BIN="$VENV_BIN"         PY="$PY"         CFG="$CFG"         CKPT="$ckpt"         OUT_DIR="$out_dir"         MAX_BATCHES="$MAX_BATCHES"         BATCH_SIZE="$BATCH_SIZE"         N_GIFS="$N_GIFS"         N_HUNYUAN_GIFS="$N_HUNYUAN_GIFS"         RUN_RGB_METRICS="$RUN_RGB_METRICS"         RUN_WORLD_PRIOR_EVAL="$RUN_WORLD_PRIOR_EVAL"         RUN_WORLD_PRIOR_PIXEL="$RUN_WORLD_PRIOR_PIXEL"         RUN_ROUGH_GIF="$RUN_ROUGH_GIF"         RUN_HUNYUAN_LATENT_DEMO="$RUN_HUNYUAN_LATENT_DEMO"         HUNYUAN_ADAPTER_CKPT="${HUNYUAN_ADAPTER_CKPT:-}"         HUNYUAN_FLOW_CKPT="${HUNYUAN_FLOW_CKPT:-}"         HUNYUAN_WM_CKPT="${HUNYUAN_WM_CKPT:-$ckpt}"         RUN_HUNYUAN_DIT_GENERATION="$RUN_HUNYUAN_DIT_GENERATION"         ALLOW_REAL_HUNYUAN_GPU_GENERATION="$ALLOW_REAL_HUNYUAN_GPU_GENERATION"         HUNYUAN_DIT_CONTROL_CKPT="${HUNYUAN_DIT_CONTROL_CKPT:-}"         N_HUNYUAN_DIT_CLIPS="$N_HUNYUAN_DIT_CLIPS"         HUNYUAN_DIT_HEIGHT="$HUNYUAN_DIT_HEIGHT"         HUNYUAN_DIT_WIDTH="$HUNYUAN_DIT_WIDTH"         HUNYUAN_DIT_FRAMES="$HUNYUAN_DIT_FRAMES"         HUNYUAN_DIT_STEPS="$HUNYUAN_DIT_STEPS"         HUNYUAN_DIT_SEED="$HUNYUAN_DIT_SEED"         HUNYUAN_DIT_CONTROL_SCALE="$HUNYUAN_DIT_CONTROL_SCALE"         MAX_WORLD_PRIOR_BATCHES="${MAX_WORLD_PRIOR_BATCHES:-}"         WORLD_PRIOR_STEPS="${WORLD_PRIOR_STEPS:-}"         bash scripts/run_generation_canary_v1.sh >"$log" 2>&1; then
        echo "canary_ok step=$step out=$out_dir"
      else
        status=$?
        echo "canary_failed step=$step status=$status log=$log"
      fi
      last_key="$key"
    fi
  fi
  sleep "$INTERVAL_SECONDS"
done
