#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806
CFG=configs/wm3d_v7_1b_native_actionpolicy_joint_canary1000_node43_v3.yaml
NAME=wm3d_v7_1b_native_actionpolicy_joint_canary1000_node43_v3
LOG_DIR=${ROOT}/logs/actionrepair_1b_20260807
LOG=${LOG_DIR}/joint_canary1000_node43_v3.log
PID_FILE=${LOG_DIR}/joint_canary1000_node43_v3.pid
OUT=${ROOT}/results/${NAME}
PY=/data/Minko/.venvs/wm3d/bin/python

cd "${ROOT}"
test -s "${CFG}"
if pgrep -af "[w]m3d_v3.training.train.*${CFG}" >/dev/null; then
  echo "duplicate V7 v3 canary process" >&2
  exit 1
fi
if find "${OUT}/ckpt" -maxdepth 1 -name 'step_*.pt' -print -quit 2>/dev/null | grep -q .; then
  echo "v3 canary checkpoint directory is not empty" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "node43 has an active GPU compute process" >&2
  exit 1
fi
free_bytes=$(df -B1 --output=avail /data | tail -1 | tr -d ' ')
if [[ "${free_bytes}" -lt 200000000000 ]]; then
  echo "node43 /data free space is below 200 GB: ${free_bytes}" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits | grep -Ev '^0, 0$' | grep -q .; then
  echo "node43 has non-zero uncorrected ECC" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
sha256sum "${CFG}" wm3d_v3/training/train.py wm3d_v3/models/action_policy.py > "${LOG_DIR}/joint_canary1000_node43_v3.inputs.sha256"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0.1411
export GLOO_SOCKET_IFNAME=bond0.1411
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

setsid "${PY}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m wm3d_v3.training.train \
  --cfg "${CFG}" \
  --print_every 10 \
  > "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "${pid}"
