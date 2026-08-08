#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806
CFG=configs/wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3.yaml
NAME=wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3
NODE_RANK=${NODE_RANK:?NODE_RANK must be 0 (node44), 1 (node41), or 2 (node43)}
MASTER_ADDR=${MASTER_ADDR:-172.27.0.7}
MASTER_PORT=${MASTER_PORT:-29871}
LOG_DIR=${ROOT}/logs/${NAME}
LOG=${LOG_DIR}/train_rank${NODE_RANK}.log
PID_FILE=${LOG_DIR}/launcher_rank${NODE_RANK}.pid
PY=/data/Minko/.venvs/wm3d/bin/python
CONFIRM=${WM3D_V7_FORMAL_RETRAIN:-}
EXPECTED_CONFIRM=EXECUTE_WM3D_V7_1B_ACTIONPOLICY_FORMAL100K_V3
EXPECTED_RESOLVED_SHA=${WM3D_V7_PREFLIGHT_RESOLVED_SHA:?WM3D_V7_PREFLIGHT_RESOLVED_SHA is required}
PREFLIGHT_REPORT=${WM3D_V7_PREFLIGHT_REPORT:?WM3D_V7_PREFLIGHT_REPORT is required}

cd "${ROOT}"
test -s "${CFG}"
if [[ "${CONFIRM}" != "${EXPECTED_CONFIRM}" ]]; then
  echo "formal confirmation mismatch" >&2
  exit 1
fi
"${PY}" - "${PREFLIGHT_REPORT}" "${EXPECTED_RESOLVED_SHA}" "${NODE_RANK}" <<'PY'
import json
import sys

path, expected_sha, node_rank = sys.argv[1:]
report = json.load(open(path))
if report.get("schema") != "wm3d_v7_1b_native_actionpolicy_joint_preflight_report_v3":
    raise SystemExit("unexpected formal preflight schema")
if report.get("passed") is not True or report.get("launch_ready") is not True:
    raise SystemExit("formal preflight is not launch-ready")
if report.get("resolved_config_sha256") != expected_sha:
    raise SystemExit("formal preflight/config digest mismatch")
if report.get("health", {}).get("compute_apps"):
    raise SystemExit("formal preflight observed active GPU applications")
print(f"validated formal preflight for node rank {node_rank}: {expected_sha}")
PY
if pgrep -af "[w]m3d_v3.training.train.*${CFG}" >/dev/null; then
  echo "duplicate formal V7 v3 process on node rank ${NODE_RANK}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
# Never inherit the cluster-wide Socket fallback from a login shell.  The
# physical mlx5 numbering differs on node43, so bind only the eight verified
# 400G ports on each host and exclude both the bond pseudo-device and the
# single 100G port.
case "${NODE_RANK}" in
  0|1)
    NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10
    ;;
  2)
    NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8
    ;;
  *)
    echo "unsupported node rank for RDMA HCA mapping: ${NODE_RANK}" >&2
    exit 1
    ;;
esac
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA="${NCCL_IB_HCA_ALLOWLIST}"
export NCCL_NET_GDR_LEVEL=2
export NCCL_SOCKET_IFNAME=bond0.1411
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME=bond0.1411
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WM3D_DDP_TIMEOUT_MINUTES=60
export WM3D_GRAD_BUCKET_MB=256
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

setsid "${PY}" -m torch.distributed.run \
  --nnodes=3 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m wm3d_v3.training.train \
  --cfg "${CFG}" \
  --print_every 20 \
  > "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "${pid}"
