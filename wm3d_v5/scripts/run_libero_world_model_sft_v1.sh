#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v5

export PYTHONPATH="/data/Minko/world_model/wm3d_v5:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/root/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri/}"
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/data/Minko/egl/10_nvidia.json}"

PY="${PY:-/data/Minko/.venvs/wm3d/bin/python}"
MASTER_ADDR="${MASTER_ADDR:-172.27.0.6}"
MASTER_PORT="${MASTER_PORT:-29673}"
IFACE="${IFACE:-bond0.1411}"
WORKER_HOSTS=(${WORKER_HOSTS:-root@172.27.0.7})
NNODES=$((1 + ${#WORKER_HOSTS[@]}))

SHARED_ROOT="${SHARED_ROOT:-/0604-10T-test/wm3d_v5}"
CACHE_ROOT="${CACHE_ROOT:-${SHARED_ROOT}/cache/libero_world_model_sft_official_replay_v1}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${SHARED_ROOT}/processed/libero_world_model_sft_train_512_v1}"
SHARED_CKPT="${SHARED_CKPT:-${SHARED_ROOT}/checkpoints/wm3d_v3_p64_1b_stage2_action_scaffold_best.pt}"
SOURCE_STAGE2_CKPT="${SOURCE_STAGE2_CKPT:-/data/Minko/world_model/wm3d_v3/results/wm3d_v3_p64_1b_stage2_action_scaffold_from_stage1p5_3node_v1/ckpt/best.pt}"

CFG_NAME="${CFG_NAME:-libero_world_model_sft_1b_from_stage2_v1.yaml}"
CFG_PATH="${CFG_PATH:-configs/${CFG_NAME}}"
MANIFEST="${MANIFEST:-/data/Minko/world_model/wm3d_v5/manifests/libero_world_model_sft_train_v1.jsonl}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/data/Minko/benchmarks/LIBERO/datasets}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/Minko/external/world_model_eval_sources/WorldVLA/rynnvla-002}"
RUN_NAME="${RUN_NAME:-wm3d_v5_1b_libero_world_model_sft_from_stage2_v1}"
OUT_ROOT="${OUT_ROOT:-/data/Minko/world_model/wm3d_v5/results/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-/data/Minko/logs/${RUN_NAME}}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
RESET_OPTIM="${RESET_OPTIM:-1}"

mkdir -p "${LOG_DIR}" "${SHARED_ROOT}/checkpoints" "${CACHE_ROOT}"

require_shared_mount() {
  if ! mountpoint -q /0604-10T-test; then
    echo "shared flash cache is not mounted at /0604-10T-test" >&2
    exit 3
  fi
}

setup_shared_checkpoint() {
  require_shared_mount
  if [[ ! -s "${SHARED_CKPT}" ]]; then
    mkdir -p "$(dirname "${SHARED_CKPT}")"
    cp -f "${SOURCE_STAGE2_CKPT}" "${SHARED_CKPT}"
  fi
  ls -lh "${SHARED_CKPT}"
}

sync_worker_code() {
  local host="$1"
  ssh "${host}" "mkdir -p /data/Minko/world_model/wm3d_v5 /data/Minko/logs/${RUN_NAME}"
  rsync -a --delete \
    --exclude=results \
    --exclude=.git \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    ./ "${host}:/data/Minko/world_model/wm3d_v5/"
}

manifest() {
  require_shared_mount
  "${PY}" scripts/build_libero_world_model_cache.py \
    --mode manifest \
    --raw_data_root "${RAW_DATA_ROOT}" \
    --official_root "${OFFICIAL_ROOT}" \
    --manifest "${MANIFEST}" \
    --cache_root "${CACHE_ROOT}" \
    --out_summary "${LOG_DIR}/manifest_summary.json" \
    --suites "10,goal,object,spatial" \
    --min_frames 24
}

cache() {
  require_shared_mount
  if [[ ! -f "${MANIFEST}" ]]; then
    manifest
  fi
  IFS=',' read -r -a gpu_array <<< "${GPUS}"
  local world_size="${#gpu_array[@]}"
  for rank in "${!gpu_array[@]}"; do
    local gpu="${gpu_array[$rank]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/build_libero_world_model_cache.py \
      --mode cache \
      --raw_data_root "${RAW_DATA_ROOT}" \
      --official_root "${OFFICIAL_ROOT}" \
      --manifest "${MANIFEST}" \
      --cache_root "${CACHE_ROOT}" \
      --processed_root "${PROCESSED_ROOT}" \
      --source processed \
      --resolution 512 \
      --out_summary "${LOG_DIR}/cache_rank${rank}.json" \
      --suites "10,goal,object,spatial" \
      --min_frames 24 \
      --batch_frames 16 \
      --shard "${rank}" \
      --world "${world_size}" \
      2>&1 | tee "${LOG_DIR}/cache_rank${rank}.log" &
  done
  wait
}

prepare_replay() {
  require_shared_mount
  if [[ ! -f "${MANIFEST}" ]]; then
    manifest
  fi
  IFS=',' read -r -a gpu_array <<< "${GPUS}"
  local world_size="${#gpu_array[@]}"
  for rank in "${!gpu_array[@]}"; do
    local gpu="${gpu_array[$rank]}"
    MUJOCO_GL="${MUJOCO_GL}" \
    LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH}" \
    __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES}" \
    MUJOCO_EGL_DEVICE_ID="${gpu}" "${PY}" scripts/build_libero_world_model_cache.py \
      --mode prepare_replay \
      --raw_data_root "${RAW_DATA_ROOT}" \
      --official_root "${OFFICIAL_ROOT}" \
      --manifest "${MANIFEST}" \
      --processed_root "${PROCESSED_ROOT}" \
      --out_summary "${LOG_DIR}/prepare_replay_rank${rank}.json" \
      --resolution 512 \
      --min_frames 24 \
      --skip_existing \
      --state_render_fallback \
      --shard "${rank}" \
      --world "${world_size}" \
      2>&1 | tee "${LOG_DIR}/prepare_replay_rank${rank}.log" &
  done
  wait
}

validate_cache() {
  require_shared_mount
  "${PY}" scripts/build_libero_world_model_cache.py \
    --mode validate \
    --manifest "${MANIFEST}" \
    --cache_root "${CACHE_ROOT}" \
    --out_summary "${LOG_DIR}/cache_validate.json"
}

action_stats() {
  require_shared_mount
  "${PY}" archive/scripts/compute_action_stats.py --cache_root "${CACHE_ROOT}" \
    2>&1 | tee "${LOG_DIR}/action_stats.log"
}

smoke_dataset() {
  require_shared_mount
  "${PY}" - <<'PY'
import yaml
from wm3d_v3.training.train import build_datasets
cfg = yaml.safe_load(open("configs/libero_world_model_sft_1b_from_stage2_v1.yaml"))
tr, val = build_datasets(cfg)
print({"train_windows": len(tr), "val_windows": len(val)})
sample = tr[0]
print({k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in sample.items() if k in {"s_in","s_tgt","rgb_in","rgb_tgt","depth_tgt","action_tgt","c"}})
PY
}

train_1node() {
  require_shared_mount
  setup_shared_checkpoint
  mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
  local train_args=(--cfg "${CFG_PATH}" --print_every 25 --resume "${SHARED_CKPT}")
  if [[ "${RESET_OPTIM}" == "1" ]]; then
    train_args+=(--reset_optim)
  fi
  nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    PYTHONUNBUFFERED=1 \
    WM3D_DDP_BACKEND=nccl \
    NCCL_DEBUG=INFO \
    NCCL_IB_DISABLE=0 \
    NCCL_NVLS_ENABLE=0 \
    NCCL_NET_GDR_LEVEL=2 \
    NCCL_SOCKET_IFNAME="${IFACE}" \
    GLOO_SOCKET_IFNAME="${IFACE}" \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -m torch.distributed.run \
      --nnodes=1 \
      --nproc_per_node=8 \
      -m wm3d_v3.training.train \
      "${train_args[@]}" \
    > "${LOG_DIR}/train_1node.log" 2>&1 &
  echo $! > "${LOG_DIR}/train_1node.pid"
  echo "train_1node_pid=$(cat "${LOG_DIR}/train_1node.pid")"
  echo "train_1node_log=${LOG_DIR}/train_1node.log"
}

train_2node() {
  require_shared_mount
  setup_shared_checkpoint
  for host in "${WORKER_HOSTS[@]}"; do
    sync_worker_code "${host}"
    ssh "${host}" "mountpoint -q /0604-10T-test"
  done

  local train_args=(--cfg "${CFG_PATH}" --print_every 25 --resume "${SHARED_CKPT}")
  if [[ "${RESET_OPTIM}" == "1" ]]; then
    train_args+=(--reset_optim)
  fi
  local common_env=(
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    "PYTHONUNBUFFERED=1"
    "WM3D_DDP_BACKEND=nccl"
    "NCCL_DEBUG=INFO"
    "NCCL_IB_DISABLE=0"
    "NCCL_NVLS_ENABLE=0"
    "NCCL_NET_GDR_LEVEL=2"
    "NCCL_SOCKET_IFNAME=${IFACE}"
    "GLOO_SOCKET_IFNAME=${IFACE}"
    "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    "HF_ENDPOINT=${HF_ENDPOINT}"
    "TOKENIZERS_PARALLELISM=false"
  )

  local rank=1
  for host in "${WORKER_HOSTS[@]}"; do
    ssh "${host}" "cd /data/Minko/world_model/wm3d_v5 && mkdir -p ${LOG_DIR} && (setsid env ${common_env[*]} ${PY} -m torch.distributed.run --nnodes=${NNODES} --nproc_per_node=8 --node_rank=${rank} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} -m wm3d_v3.training.train ${train_args[*]} > ${LOG_DIR}/train_2node_node${rank}.log 2>&1 < /dev/null & echo \$! > ${LOG_DIR}/train_2node_node${rank}.pid)"
    rank=$((rank + 1))
  done

  nohup env "${common_env[@]}" "${PY}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m wm3d_v3.training.train \
    "${train_args[@]}" \
    > "${LOG_DIR}/train_2node_node0.log" 2>&1 &
  echo $! > "${LOG_DIR}/train_2node_node0.pid"

  echo "nnodes=${NNODES}"
  echo "node0_pid=$(cat "${LOG_DIR}/train_2node_node0.pid")"
  echo "node0_log=${LOG_DIR}/train_2node_node0.log"
  rank=1
  for host in "${WORKER_HOSTS[@]}"; do
    echo "node${rank}_pid=$(ssh "${host}" "cat ${LOG_DIR}/train_2node_node${rank}.pid")"
    echo "node${rank}_log=${host}:${LOG_DIR}/train_2node_node${rank}.log"
    rank=$((rank + 1))
  done
}

benchmark() {
  local ckpt="${CKPT:-${OUT_ROOT}/ckpt/best.pt}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "checkpoint not found for benchmark: ${ckpt}" >&2
    exit 4
  fi
  CKPT="${ckpt}" \
  RESULT_ROOT="${RESULT_ROOT:-/data/Minko/world_model/wm3d_v5/results/worldvla_libero_official_${RUN_NAME}}" \
  RUN_ID="${BENCH_RUN_ID:-${RUN_NAME}_worldvla_official}" \
  HF_ENDPOINT="${HF_ENDPOINT}" \
    scripts/run_worldvla_libero_benchmark_v1.sh all
}

status() {
  echo "shared_cache=${CACHE_ROOT}"
  echo "processed_root=${PROCESSED_ROOT}"
  echo "manifest=${MANIFEST}"
  echo "cfg=${CFG_PATH}"
  echo "out_root=${OUT_ROOT}"
  echo "log_dir=${LOG_DIR}"
  echo "shared_ckpt=${SHARED_CKPT}"
  for f in "${LOG_DIR}"/*.pid; do
    [[ -e "${f}" ]] || continue
    local pid
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
  setup_shared) setup_shared_checkpoint ;;
  manifest) manifest ;;
  prepare_replay) prepare_replay ;;
  cache) cache ;;
  validate_cache) validate_cache ;;
  action_stats) action_stats ;;
  smoke_dataset) smoke_dataset ;;
  train_1node) train_1node ;;
  train_2node) train_2node ;;
  benchmark) benchmark ;;
  status) status ;;
  prepare_train)
    setup_shared_checkpoint
    manifest
    prepare_replay
    cache
    validate_cache
    action_stats
    smoke_dataset
    ;;
  all_2node)
    setup_shared_checkpoint
    manifest
    prepare_replay
    cache
    validate_cache
    action_stats
    smoke_dataset
    train_2node
    ;;
  *)
    echo "usage: $0 [setup_shared|manifest|prepare_replay|cache|validate_cache|action_stats|smoke_dataset|train_1node|train_2node|benchmark|status|prepare_train|all_2node]" >&2
    exit 2
    ;;
esac
