#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/data/Minko/egl/10_nvidia.json}"

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
LIBERO_PY="${LIBERO_PY:-/data/Minko/.conda-envs/libero-py38/bin/python}"
BASE_CFG="${BASE_CFG:-configs/v5_p64_1b_libero_action_policy_base_from_stage2_v1.yaml}"
SFT_CFG="${SFT_CFG:-configs/v5_p64_1b_libero_action_policy_sft_from_stage2_v1.yaml}"
STAGE2_ROOT="${STAGE2_ROOT:-results/wm3d_v5_p64_1b_stage2_action_scaffold_native3d_wsd_4node_v1}"
OUT_ROOT="${OUT_ROOT:-results/wm3d_v5_p64_1b_libero_action_policy_sft_from_stage2_v1}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs/wm3d_v5_p64_1b_libero_action_policy_v1}"
ACTION_CACHE_ROOT="${ACTION_CACHE_ROOT:-/0604-10T-test/wm3d_v5/cache/libero_action_policy_full_T16_k8_s4_v1}"
ACTION_MANIFEST="${ACTION_MANIFEST:-/data/Minko/world_model/wm3d_v5/manifests/libero_action_policy_full_T16_k8_s4_v1.jsonl}"
POLICY_PORT="${POLICY_PORT:-8765}"
SERVER_GPU="${SERVER_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-0}"

mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

stage2_ckpt() {
  if [[ -s "${STAGE2_ROOT}/ckpt/best.pt" ]]; then
    echo "${STAGE2_ROOT}/ckpt/best.pt"
  elif [[ -s "${STAGE2_ROOT}/ckpt/latest.pt" ]]; then
    echo "${STAGE2_ROOT}/ckpt/latest.pt"
  else
    echo "missing Stage2 checkpoint under ${STAGE2_ROOT}/ckpt" >&2
    return 4
  fi
}

require_action_cache() {
  if [[ ! -s "${ACTION_MANIFEST}" || ! -s "${ACTION_CACHE_ROOT}/action_stats.npz" ]]; then
    echo "missing full LIBERO action cache; run scripts/build_v5_libero_action_policy_cache_full_v1.sh all first" >&2
    return 5
  fi
}

runtime_cfg() {
  require_action_cache
  local ckpt
  ckpt="$(stage2_ckpt)"
  local out="${LOG_DIR}/runtime_libero_action_policy_sft.yaml"
  INIT_CKPT="${ckpt}" ACTION_MANIFEST="${ACTION_MANIFEST}" ACTION_STATS="${ACTION_CACHE_ROOT}/action_stats.npz" OUT_ROOT="${OUT_ROOT}" "${PY}" - <<'PY'
import os
from pathlib import Path
import yaml

cfg_path = Path("configs/v5_p64_1b_libero_action_policy_sft_from_stage2_v1.yaml")
cfg = yaml.safe_load(cfg_path.read_text())
cfg["train"]["init_ckpt"] = os.environ["INIT_CKPT"]
cfg["data"]["manifest"] = [os.environ["ACTION_MANIFEST"]]
cfg["data"]["action_stats"] = os.environ["ACTION_STATS"]
cfg["out"]["root"] = os.environ["OUT_ROOT"]
out = Path(os.environ.get("RUNTIME_CFG", "/data/Minko/logs/wm3d_v5_p64_1b_libero_action_policy_v1/runtime_libero_action_policy_sft.yaml"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(out)
PY
}

smoke() {
  require_action_cache
  "${PY}" - <<'PY'
import json
from pathlib import Path
import yaml
from wm3d_v3.training.train_libero_success_p0 import LiberoExpertCacheDataset
from wm3d_v3.training.train import build_model

sft = yaml.safe_load(Path("configs/v5_p64_1b_libero_action_policy_sft_from_stage2_v1.yaml").read_text())
base = yaml.safe_load(Path("configs/v5_p64_1b_libero_action_policy_base_from_stage2_v1.yaml").read_text())
ds = LiberoExpertCacheDataset(Path("/data/Minko/world_model/wm3d_v5/manifests/libero_action_policy_full_T16_k8_s4_v1.jsonl"), plan_state_dim=17)
sample = ds[0]
model = build_model(base)
print(json.dumps({
    "windows": len(ds),
    "sample_shapes": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")},
    "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
}, sort_keys=True))
PY
}

train() {
  local cfg
  cfg="$(runtime_cfg | tail -n 1)"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}" \
  WM3D_DDP_BACKEND="${WM3D_DDP_BACKEND:-gloo}" \
  NCCL_NVLS_ENABLE=0 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
  HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  setsid "${TORCHRUN:-/data/Minko/.venvs/wm3d/bin/torchrun}" --standalone --nproc_per_node="${NPROC:-8}" -m wm3d_v3.training.train_libero_action_policy \
    --cfg "${cfg}" \
    --print_every 25 \
    < /dev/null > "${LOG_DIR}/train_action_policy.log" 2>&1 &
  echo $! > "${LOG_DIR}/train_action_policy.pid"
  echo "train_pid=$(cat "${LOG_DIR}/train_action_policy.pid")"
  echo "train_log=${LOG_DIR}/train_action_policy.log"
}

server() {
  local ckpt="${CKPT:-${OUT_ROOT}/ckpt/best.pt}"
  if [[ ! -s "${ckpt}" ]]; then
    echo "missing action-policy checkpoint: ${ckpt}" >&2
    return 6
  fi
  CUDA_VISIBLE_DEVICES="${SERVER_GPU}" PYTHONUNBUFFERED=1 nohup "${PY}" -m wm3d_v3.policy.http_policy_server \
    --cfg "${BASE_CFG}" \
    --ckpt "${ckpt}" \
    --host 127.0.0.1 \
    --port "${POLICY_PORT}" \
    --device cuda:0 \
    --qwen_device cuda:0 \
    --task_cache_dir /0604-10T-test/wm3d_v5/cache/libero_taskemb_online \
    --selection_mode action_policy \
    > "${LOG_DIR}/policy_server.log" 2>&1 &
  echo $! > "${LOG_DIR}/policy_server.pid"
  echo "server_pid=$(cat "${LOG_DIR}/policy_server.pid")"
  echo "server_log=${LOG_DIR}/policy_server.log"
}

benchmark_smoke() {
  local ckpt="${CKPT:-${OUT_ROOT}/ckpt/best.pt}"
  if [[ ! -s "${ckpt}" ]]; then
    echo "missing action-policy checkpoint: ${ckpt}" >&2
    return 6
  fi
  if ! curl -fsS "http://127.0.0.1:${POLICY_PORT}/health" >/dev/null; then
    server
    sleep 15
  fi
  PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}" "${LIBERO_PY}" -m wm3d_v3.benchmarks.libero_remote_runner \
    --server_url "http://127.0.0.1:${POLICY_PORT}" \
    --suite libero_10 \
    --max_tasks 2 \
    --init_states 0 \
    --max_steps 300 \
    --camera_key agentview_image \
    --camera_size 256 \
    --context_T 16 \
    --warmup_steps 5 \
    --gripper_mode closed01_to_libero \
    --send_lowdim \
    --action_history_len 16 \
    --send_progress \
    --out "${OUT_ROOT}/libero_closed_loop_smoke_libero10_2tasks_init0.json" \
    > "${LOG_DIR}/libero_closed_loop_smoke.log" 2>&1
}

status() {
  echo "stage2_ckpt=$(stage2_ckpt 2>/dev/null || true)"
  echo "action_manifest=${ACTION_MANIFEST}"
  [[ -s "${ACTION_MANIFEST}" ]] && wc -l "${ACTION_MANIFEST}" || true
  echo "out_root=${OUT_ROOT}"
  echo "log_dir=${LOG_DIR}"
  for f in "${LOG_DIR}"/*.pid; do
    [[ -e "${f}" ]] || continue
    pid="$(cat "${f}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "RUNNING $(basename "${f}") pid=${pid}"
    else
      echo "STOPPED $(basename "${f}") pid=${pid}"
    fi
  done
}

cmd="${1:-status}"
case "${cmd}" in
  smoke) smoke ;;
  train) train ;;
  server) server ;;
  benchmark_smoke) benchmark_smoke ;;
  status) status ;;
  *)
    echo "usage: $0 [smoke|train|server|benchmark_smoke|status]" >&2
    exit 2
    ;;
esac
