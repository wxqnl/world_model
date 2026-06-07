#!/usr/bin/env bash
set -euo pipefail

V5_ROOT=${V5_ROOT:-/data/Minko/world_model/wm3d_v5}
cd "$V5_ROOT"

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
CFG=${CFG:?CFG is required}
CKPT=${CKPT:?CKPT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required}
MAX_BATCHES=${MAX_BATCHES:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
N_GIFS=${N_GIFS:-2}
N_HUNYUAN_GIFS=${N_HUNYUAN_GIFS:-1}
RUN_RGB_METRICS=${RUN_RGB_METRICS:-0}
RUN_WORLD_PRIOR_EVAL=${RUN_WORLD_PRIOR_EVAL:-auto}
RUN_WORLD_PRIOR_PIXEL=${RUN_WORLD_PRIOR_PIXEL:-0}
RUN_ROUGH_GIF=${RUN_ROUGH_GIF:-auto}
RUN_HUNYUAN_LATENT_DEMO=${RUN_HUNYUAN_LATENT_DEMO:-0}
HUNYUAN_ADAPTER_CKPT=${HUNYUAN_ADAPTER_CKPT:-}
HUNYUAN_FLOW_CKPT=${HUNYUAN_FLOW_CKPT:-}
HUNYUAN_WM_CKPT=${HUNYUAN_WM_CKPT:-$CKPT}
RUN_HUNYUAN_DIT_GENERATION=${RUN_HUNYUAN_DIT_GENERATION:-0}
ALLOW_REAL_HUNYUAN_GPU_GENERATION=${ALLOW_REAL_HUNYUAN_GPU_GENERATION:-0}
HUNYUAN_DIT_CONTROL_CKPT=${HUNYUAN_DIT_CONTROL_CKPT:-}
N_HUNYUAN_DIT_CLIPS=${N_HUNYUAN_DIT_CLIPS:-1}
HUNYUAN_DIT_HEIGHT=${HUNYUAN_DIT_HEIGHT:-320}
HUNYUAN_DIT_WIDTH=${HUNYUAN_DIT_WIDTH:-512}
HUNYUAN_DIT_FRAMES=${HUNYUAN_DIT_FRAMES:-9}
HUNYUAN_DIT_STEPS=${HUNYUAN_DIT_STEPS:-8}
HUNYUAN_DIT_SEED=${HUNYUAN_DIT_SEED:-0}
HUNYUAN_DIT_CONTROL_SCALE=${HUNYUAN_DIT_CONTROL_SCALE:-1.0}

mkdir -p "$OUT_DIR"

echo "canary_start cfg=$CFG ckpt=$CKPT out=$OUT_DIR max_batches=$MAX_BATCHES"

is_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

is_false() {
  case "${1,,}" in
    0|false|no|n|off) return 0 ;;
    *) return 1 ;;
  esac
}

cfg_bool() {
  local key=$1
  "$PY" - "$CFG" "$key" <<PY
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
cur = cfg
for part in sys.argv[2].split("."):
    if not isinstance(cur, dict) or part not in cur:
        cur = False
        break
    cur = cur[part]
print("true" if bool(cur) else "false")
PY
}

checkpoint_has_hunyuan_adapter() {
  local ckpt_path=$1
  local result
  result="$($PY - "$ckpt_path" <<PY 2>/dev/null || true
import sys
import torch
try:
    sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception:
    print("false")
else:
    print("true" if isinstance(sd, dict) and "hunyuan_adapter" in sd else "false")
PY
)"
  [[ "$result" == "true" ]]
}

should_run_world_prior() {
  if is_true "$RUN_WORLD_PRIOR_EVAL"; then
    return 0
  fi
  if is_false "$RUN_WORLD_PRIOR_EVAL"; then
    return 1
  fi
  [[ "$(cfg_bool model.enable_world_prior)" == "true" && "$(cfg_bool data.require_task_emb)" == "true" ]]
}

should_run_rough_gif() {
  if is_true "$RUN_ROUGH_GIF"; then
    return 0
  fi
  if is_false "$RUN_ROUGH_GIF"; then
    return 1
  fi
  [[ "$(cfg_bool data.load_rgb)" == "true" ]] || return 1
  [[ "$(cfg_bool model.enable_pixel)" == "true" || "$(cfg_bool model.enable_context_pixel)" == "true" ]]
}

should_run_hunyuan_dit_generation() {
  is_true "$RUN_HUNYUAN_DIT_GENERATION" || return 1
  is_true "$ALLOW_REAL_HUNYUAN_GPU_GENERATION" || return 1
  [[ -n "$HUNYUAN_DIT_CONTROL_CKPT" ]]
}

should_run_hunyuan_latent_demo() {
  if is_true "$RUN_HUNYUAN_LATENT_DEMO"; then
    return 0
  fi
  [[ -n "$HUNYUAN_ADAPTER_CKPT" ]] && return 0
  checkpoint_has_hunyuan_adapter "$CKPT"
}

core_cmd=(
  "$PY" -m wm3d_v3.eval.run_eval
  --cfg "$CFG"
  --ckpt "$CKPT"
  --out "$OUT_DIR/core_eval.json"
  --max_batches "$MAX_BATCHES"
  --batch_size "$BATCH_SIZE"
)
if ! is_true "$RUN_RGB_METRICS"; then
  core_cmd+=(--skip_rgb_metrics)
fi

echo "canary_stage core_eval rgb_metrics=$RUN_RGB_METRICS"
"${core_cmd[@]}"

if should_run_world_prior; then
  prior_cmd=(
    "$PY" -m wm3d_v3.eval.world_prior_eval
    --cfg "$CFG"
    --ckpt "$CKPT"
    --out "$OUT_DIR/world_prior_eval.json"
    --max_batches "${MAX_WORLD_PRIOR_BATCHES:-$MAX_BATCHES}"
    --batch_size "$BATCH_SIZE"
    --steps "${WORLD_PRIOR_STEPS:-8}"
  )
  if is_true "$RUN_WORLD_PRIOR_PIXEL"; then
    prior_cmd+=(--pixel)
  fi
  echo "canary_stage world_prior_eval pixel=$RUN_WORLD_PRIOR_PIXEL"
  "${prior_cmd[@]}"
else
  echo "canary_skip world_prior_eval RUN_WORLD_PRIOR_EVAL=$RUN_WORLD_PRIOR_EVAL enable_world_prior=$(cfg_bool model.enable_world_prior) require_task_emb=$(cfg_bool data.require_task_emb)"
fi

if should_run_rough_gif; then
  echo "canary_stage rough_gif n_clips=$N_GIFS"
  "$PY" -m wm3d_v3.eval.make_demo_gif     --cfg "$CFG"     --ckpt "$CKPT"     --out_dir "$OUT_DIR/rough_gifs"     --n_clips "$N_GIFS"
else
  echo "canary_skip rough_gif RUN_ROUGH_GIF=$RUN_ROUGH_GIF load_rgb=$(cfg_bool data.load_rgb) enable_pixel=$(cfg_bool model.enable_pixel) enable_context_pixel=$(cfg_bool model.enable_context_pixel)"
fi

if should_run_hunyuan_latent_demo; then
  demo_ckpt="$CKPT"
  if [[ -n "$HUNYUAN_ADAPTER_CKPT" ]]; then
    demo_ckpt="$HUNYUAN_ADAPTER_CKPT"
  fi
  demo_cmd=(
    "$PY" -m wm3d_v3.eval.make_hunyuan_latent_demo
    --cfg "$CFG"
    --ckpt "$demo_ckpt"
    --out_dir "$OUT_DIR/hunyuan_latent_demos"
    --n_clips "$N_HUNYUAN_GIFS"
  )
  if [[ "$demo_ckpt" != "$CKPT" ]]; then
    demo_cmd+=(--wm_ckpt "$HUNYUAN_WM_CKPT")
  fi
  echo "canary_stage hunyuan_latent_demo ckpt=$demo_ckpt wm_ckpt=${HUNYUAN_WM_CKPT:-}"
  "${demo_cmd[@]}"
else
  echo "canary_skip hunyuan_latent_demo RUN_HUNYUAN_LATENT_DEMO=$RUN_HUNYUAN_LATENT_DEMO adapter_ckpt_set=$([[ -n "$HUNYUAN_ADAPTER_CKPT" ]] && echo 1 || echo 0) flow_ckpt_set=$([[ -n "$HUNYUAN_FLOW_CKPT" ]] && echo 1 || echo 0) flow_ckpt_note=flow_demo_not_supported_by_latent_adapter_demo"
fi


if should_run_hunyuan_dit_generation; then
  echo "canary_stage hunyuan_dit_control_generation control_ckpt=$HUNYUAN_DIT_CONTROL_CKPT clips=$N_HUNYUAN_DIT_CLIPS size=${HUNYUAN_DIT_HEIGHT}x${HUNYUAN_DIT_WIDTH} frames=$HUNYUAN_DIT_FRAMES steps=$HUNYUAN_DIT_STEPS"
  "$PY" -m wm3d_v3.eval.make_hunyuan_dit_control_demo     --cfg "$CFG"     --wm_ckpt "$CKPT"     --control_ckpt "$HUNYUAN_DIT_CONTROL_CKPT"     --out_dir "$OUT_DIR/hunyuan_dit_control_videos"     --n_clips "$N_HUNYUAN_DIT_CLIPS"     --height "$HUNYUAN_DIT_HEIGHT"     --width "$HUNYUAN_DIT_WIDTH"     --frames "$HUNYUAN_DIT_FRAMES"     --steps "$HUNYUAN_DIT_STEPS"     --seed "$HUNYUAN_DIT_SEED"     --control_scale "$HUNYUAN_DIT_CONTROL_SCALE"
else
  echo "canary_skip hunyuan_dit_control_generation RUN_HUNYUAN_DIT_GENERATION=$RUN_HUNYUAN_DIT_GENERATION allow_real=$ALLOW_REAL_HUNYUAN_GPU_GENERATION control_ckpt_set=$([[ -n "$HUNYUAN_DIT_CONTROL_CKPT" ]] && echo 1 || echo 0)"
fi

find "$OUT_DIR" -maxdepth 3 -type f -print | sort
echo "canary_done out=$OUT_DIR"
