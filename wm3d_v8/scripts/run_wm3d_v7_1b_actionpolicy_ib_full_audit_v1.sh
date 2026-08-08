#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806
PY=/data/Minko/.venvs/wm3d/bin/python
CFG=configs/wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3.yaml
SMOKE=scripts/smoke_wm3d_v7_1b_actionpolicy_distributed_audit_v3.py
MASTER_ADDR=172.27.0.7
MASTER_PORT=${MASTER_PORT:-29941}
ATTEMPT_LABEL=${ATTEMPT_LABEL:-$(date -u +%Y%m%dT%H%M%SZ)}
ATTEMPT_DIR=${ROOT}/audits/actionrepair_1b_20260807/ib_full_data_audit/${ATTEMPT_LABEL}
NOT_BEFORE=$(( $(date +%s) + 12 ))

declare -A HOSTS=( [0]=172.27.0.7 [1]=172.27.0.4 [2]=172.27.0.6 )
declare -A HCAS=(
  [0]=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10
  [1]=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10
  [2]=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8
)

cd "${ROOT}"
test -s "${CFG}"
test -s "${SMOKE}"
mkdir -m 0755 -p "${ATTEMPT_DIR}"

for rank in 0 1 2; do
  if [[ "${rank}" == "2" ]]; then
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
  else
    apps=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "root@${HOSTS[${rank}]}" \
      "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits")
  fi
  if [[ -n "${apps//[[:space:]]/}" ]]; then
    echo "node rank ${rank} has active GPU applications; refusing full audit" >&2
    exit 1
  fi
done

declare -a jobs=()
for rank in 0 1 2; do
  launch_cmd="cd '${ROOT}' && while [ \"\$(date +%s)\" -lt '${NOT_BEFORE}' ]; do sleep 0.1; done; exec env PYTHONPATH='${ROOT}' CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 WM3D_NODE_RANK='${rank}' NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_IB_HCA='${HCAS[${rank}]}' NCCL_NET_GDR_LEVEL=2 NCCL_SOCKET_IFNAME=bond0.1411 NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=bond0.1411 NCCL_NVLS_ENABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,COLL TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 timeout --signal=TERM --kill-after=60s 1800s '${PY}' -m torch.distributed.run --nnodes=3 --nproc_per_node=8 --node_rank='${rank}' --master_addr='${MASTER_ADDR}' --master_port='${MASTER_PORT}' '${SMOKE}' --cfg '${CFG}'"
  if [[ "${rank}" == "2" ]]; then
    (bash -c "${launch_cmd}") > "${ATTEMPT_DIR}/rank${rank}.log" 2>&1 &
  else
    (ssh -o BatchMode=yes -o ConnectTimeout=15 "root@${HOSTS[${rank}]}" "${launch_cmd}") \
      > "${ATTEMPT_DIR}/rank${rank}.log" 2>&1 &
  fi
  jobs[${rank}]=$!
  echo "armed node_rank=${rank} orchestration_pid=${jobs[${rank}]} not_before=${NOT_BEFORE}"
done

failed=0
for rank in 0 1 2; do
  if wait "${jobs[${rank}]}"; then
    echo "node_rank=${rank} rc=0"
  else
    rc=$?
    echo "node_rank=${rank} rc=${rc}" >&2
    failed=1
  fi
done

for rank in 0 1 2; do
  log=${ATTEMPT_DIR}/rank${rank}.log
  ib_count=$(grep -c "Using network IB" "${log}" || true)
  if [[ "${ib_count}" -ne 8 ]]; then
    echo "node_rank=${rank} ib_selection_count=${ib_count}, expected 8" >&2
    failed=1
  fi
  if grep -q "Using network Socket" "${log}"; then
    echo "node_rank=${rank} fell back to Socket" >&2
    failed=1
  fi
  if grep -Eiq "wrong type|NCCL WARN|Traceback|DistBackendError|CUDA error|unhandled system error|timed out" "${log}"; then
    echo "node_rank=${rank} contains a transport/runtime error" >&2
    failed=1
  fi
done

if ! grep -q "WM3D_V7_FORMAL_DISTRIBUTED_AUDIT_SMOKE_OK" "${ATTEMPT_DIR}/rank0.log"; then
  echo "rank0 is missing the global five-source audit success payload" >&2
  failed=1
fi

if [[ "${failed}" -ne 0 ]]; then
  echo "WM3D_V7_IB_FULL_DATA_AUDIT_FAIL attempt_dir=${ATTEMPT_DIR}" >&2
  exit 1
fi

echo "WM3D_V7_IB_FULL_DATA_AUDIT_PASS attempt_dir=${ATTEMPT_DIR}"
