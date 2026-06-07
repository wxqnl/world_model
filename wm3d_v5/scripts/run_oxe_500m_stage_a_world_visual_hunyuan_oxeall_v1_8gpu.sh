#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

VENV_BIN=${VENV_BIN:-/data/Minko/.venvs/wm3d/bin}
PY=${PY:-$VENV_BIN/python}
TORCHRUN=${TORCHRUN:-$VENV_BIN/torchrun}
LOG_DIR=${LOG_DIR:-/data/Minko/logs}
RUN_NAME=${RUN_NAME:-train_oxe_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu}
CFG=${CFG:-configs/v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.yaml}
RESUME=${RESUME:-results/wm3d_v3_p64_140m_hunyuan_visual_proof_v1_8gpu/ckpt/best.pt}
PIDFILE=${PIDFILE:-$LOG_DIR/${RUN_NAME}.pid}
LOG=${LOG:-$LOG_DIR/${RUN_NAME}.log}
CANARY_PIDFILE=${CANARY_PIDFILE:-$LOG_DIR/${RUN_NAME}_canary.pid}
CANARY_LOG=${CANARY_LOG:-$LOG_DIR/${RUN_NAME}_canary.log}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6}
NPROC_PER_NODE=${NPROC_PER_NODE:-7}

mkdir -p "$LOG_DIR"

if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old_pid:-}" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "already_running pid=$old_pid pidfile=$PIDFILE"
    exit 10
  fi
fi

for p in "$CFG" "$RESUME" /data/Minko/datasets/cache/wm3d_v3 /data/Minko/external/HunyuanVideo /data/Minko/models/hunyuan_video; do
  if [[ ! -e "$p" ]]; then
    echo "missing_required_path=$p"
    exit 20
  fi
done

"$PY" - <<'PY'
import json, yaml
from pathlib import Path
from wm3d_v3.training.train import build_model, build_datasets, build_hunyuan_latent_adapter

cfg = yaml.safe_load(Path("configs/v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.yaml").read_text())
model = build_model(cfg)
model_params = sum(p.numel() for p in model.parameters())
trainable_model = sum(
    p.numel()
    for n, p in model.named_parameters()
    if any(n.startswith(prefix) for prefix in cfg["train"].get("trainable_prefixes", []))
)
adapter = build_hunyuan_latent_adapter(cfg, "cpu")
adapter_params = sum(p.numel() for p in adapter.parameters())
tr, val = build_datasets(cfg)
import os
nproc = int(os.environ.get("NPROC_PER_NODE", "7"))
print(json.dumps({
    "cfg": "configs/v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.yaml",
    "model_params_M": round(model_params / 1e6, 3),
    "trainable_model_params_M": round(trainable_model / 1e6, 3),
    "hunyuan_adapter_params_M": round(adapter_params / 1e6, 3),
    "train_windows": len(tr),
    "val_windows": len(val),
    "nproc_per_node": nproc,
    "global_batch": int(cfg["train"]["batch_size_per_gpu"]) * nproc,
}, ensure_ascii=False))
PY

OUT_ROOT="$("$PY" - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.yaml"))
print(cfg["out"]["root"])
PY
)"
CKPT_DIR="$OUT_ROOT/ckpt"
CANARY_OUT="$OUT_ROOT/canary"
mkdir -p "$CKPT_DIR" "$CANARY_OUT"

if [[ -f "$CANARY_PIDFILE" ]]; then
  old_canary="$(cat "$CANARY_PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old_canary:-}" ]] && ps -p "$old_canary" >/dev/null 2>&1; then
    kill "$old_canary" 2>/dev/null || true
  fi
fi

CFG="$CFG" \
CKPT_DIR="$CKPT_DIR" \
OUT_ROOT="$CANARY_OUT" \
LOG_DIR="$LOG_DIR" \
CANARY_GPU="${CANARY_GPU:-7}" \
INTERVAL_SECONDS="${CANARY_INTERVAL_SECONDS:-300}" \
MAX_BATCHES="${CANARY_MAX_BATCHES:-8}" \
N_GIFS="${CANARY_N_GIFS:-2}" \
N_HUNYUAN_GIFS="${CANARY_N_HUNYUAN_GIFS:-1}" \
nohup scripts/watch_generation_canary_v1.sh >"$CANARY_LOG" 2>&1 &
echo "$!" > "$CANARY_PIDFILE"

WM3D_DDP_BACKEND=nccl \
NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" \
NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_bond_0}" \
NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}" \
NCCL_ASYNC_ERROR_HANDLING=1 \
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" \
nohup "$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
  -m wm3d_v3.training.train \
  --cfg "$CFG" \
  --resume "$RESUME" \
  --reset_optim \
  --print_every 25 \
  >"$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "started_training pid=$(cat "$PIDFILE") log=$LOG"
echo "started_canary pid=$(cat "$CANARY_PIDFILE") log=$CANARY_LOG"
echo "out_root=$OUT_ROOT"
