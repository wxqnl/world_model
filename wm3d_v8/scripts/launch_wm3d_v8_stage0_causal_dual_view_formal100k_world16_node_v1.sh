#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT=/data/Minko/wm3d_v8_stage0_formal_code_20260810_v2
RUNTIME_ROOT=/data/Minko/world_model/wm3d_v8_stage0_causal_dual_view_20260809
CFG=${RUNTIME_ROOT}/manifests/formal_world16_v1/runtime_config_formal100k_world16_v1.yaml
SEAL=${RUNTIME_ROOT}/manifests/formal_world16_v1/seal_report_formal100k_world16_v1.json
NAME=formal100k_world16_node43_node44_v1
LOG_DIR=${RUNTIME_ROOT}/logs/${NAME}
NODE_RANK=${NODE_RANK:?NODE_RANK must be 0 (node43) or 1 (node44)}
MASTER_ADDR=${MASTER_ADDR:-172.27.0.6}
MASTER_PORT=${MASTER_PORT:-29981}
LOG=${LOG_DIR}/train_rank${NODE_RANK}_step_00000000_to_00001000.log
PID_FILE=${LOG_DIR}/launcher_rank${NODE_RANK}_step_00000000_to_00001000.pid
INPUTS_FILE=${LOG_DIR}/inputs_rank${NODE_RANK}_step_00000000_to_00001000.sha256
PY=/data/Minko/.venvs/wm3d/bin/python
CONFIRM=${WM3D_V8_STAGE0_FORMAL:-}
EXPECTED_CONFIRM=EXECUTE_WM3D_V8_STAGE0_FORMAL_WORLD16_V1
EXPECTED_RESOLVED_SHA=${WM3D_V8_PREFLIGHT_RESOLVED_SHA:?WM3D_V8_PREFLIGHT_RESOLVED_SHA is required}
PREFLIGHT_REPORT=${WM3D_V8_PREFLIGHT_REPORT:?WM3D_V8_PREFLIGHT_REPORT is required}
EXPECTED_SEAL_SHA=${WM3D_V8_FORMAL_SEAL_SHA256:?WM3D_V8_FORMAL_SEAL_SHA256 is required}

if [[ "${CONFIRM}" != "${EXPECTED_CONFIRM}" ]]; then
  echo "formal confirmation mismatch" >&2
  exit 1
fi
if [[ ! "${EXPECTED_RESOLVED_SHA}" =~ ^[0-9a-f]{64}$ ||
      ! "${EXPECTED_SEAL_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "formal resolved/seal SHA must be exact lowercase SHA256" >&2
  exit 1
fi

cd "${PROJECT}"
test -s "${CFG}"
test -s "${SEAL}"
test -s "${PREFLIGHT_REPORT}"
observed_seal_sha=$(sha256sum "${SEAL}" | awk '{print $1}')
if [[ "${observed_seal_sha}" != "${EXPECTED_SEAL_SHA}" ]]; then
  echo "formal seal SHA mismatch" >&2
  exit 1
fi

PYTHONPATH="${PROJECT}" "${PY}" - \
  "${PREFLIGHT_REPORT}" "${EXPECTED_RESOLVED_SHA}" "${CFG}" "${SEAL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    load_config,
    resolved_config_sha256,
)

report_path, expected_sha, config_path, seal_path = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
if report.get("schema") != "wm3d_v8_stage0_causal_dual_view_preflight_report_v1":
    raise SystemExit("unexpected formal preflight schema")
if report.get("passed") is not True or report.get("launch_ready") is not True:
    raise SystemExit("formal preflight is not launch-ready")
if report.get("errors") != [] or report.get("warnings") != [] or report.get("blockers") != []:
    raise SystemExit("formal preflight contains findings")
if report.get("resolved_config_sha256") != str(expected_sha):
    raise SystemExit("formal preflight/config digest mismatch")
if (report.get("health") or {}).get("compute_apps"):
    raise SystemExit("formal preflight observed active GPU applications")

seal = json.loads(seal_path.read_text())
if seal.get("schema") != "wm3d_v8_stage0_causal_dual_view_canary_seal_v1":
    raise SystemExit("unexpected formal seal schema")
if seal.get("passed") is not True or seal.get("launch_ready") is not True:
    raise SystemExit("formal seal is not launch-ready")
if Path(str(seal.get("runtime_config") or "")).resolve() != config_path.resolve():
    raise SystemExit("formal seal is bound to another runtime config")
if seal.get("resolved_config_sha256") != str(expected_sha):
    raise SystemExit("formal seal/config digest mismatch")
if hashlib.sha256(config_path.read_bytes()).hexdigest() != seal.get("runtime_config_sha256"):
    raise SystemExit("formal runtime config content SHA mismatch")

config = load_config(config_path)
if resolved_config_sha256(config) != str(expected_sha):
    raise SystemExit("resolved formal config changed after preflight")
contract = config.get("contract") or {}
train = config.get("train") or {}
out = config.get("out") or {}
expected_lineage = (
    "wm3d_v8_stage0_causal_dual_view_actionpolicy_"
    "formal100k_world16_node43_node44_20260810_v1"
)
if contract.get("schema") != "wm3d_v8_stage0_causal_dual_view_actionpolicy_formal_v1":
    raise SystemExit("formal contract schema mismatch")
if train.get("run_lineage") != expected_lineage:
    raise SystemExit("formal run lineage mismatch")
expected_train = {
    "num_nodes": 2,
    "gpus_per_node": 8,
    "batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 2,
    "effective_global_batch": 64,
    "max_steps": 100000,
}
for key, expected in expected_train.items():
    if int(train.get(key, -1)) != expected:
        raise SystemExit(f"formal {key} mismatch")
if any(train.get(key) is not None for key in (
    "resume_checkpoint", "pretrained_world_checkpoint", "stage_transition"
)):
    raise SystemExit("formal fresh lineage contains a resume/warm-start source")
if train.get("fresh_initialization_required") is not True:
    raise SystemExit("formal lineage is not fresh")
expected_out = Path(
    "/data/Minko/world_model/wm3d_v8_stage0_causal_dual_view_20260809/"
    "results/formal100k_world16_node43_node44_v1"
)
if Path(str(out.get("root") or "")).resolve() != expected_out:
    raise SystemExit("formal output root mismatch")
PY

if pgrep -af "[w]m3d_v3.training.train.*${CFG}" >/dev/null; then
  echo "duplicate formal V8 Stage0 process on node rank ${NODE_RANK}" >&2
  exit 1
fi
if [[ -e "${LOG}" || -e "${PID_FILE}" ]]; then
  echo "formal node-launch output already exists" >&2
  exit 1
fi
OUT=${RUNTIME_ROOT}/results/${NAME}
if find "${OUT}/ckpt" -maxdepth 1 -name 'step_*.pt' -print -quit 2>/dev/null | grep -q .; then
  echo "formal fresh checkpoint directory is not empty" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -Eq '[0-9]'; then
  echo "formal host has an active GPU compute process" >&2
  exit 1
fi
free_bytes=$(df -B1 --output=avail /data | tail -1 | tr -d ' ')
if [[ "${free_bytes}" -lt 160000000000 ]]; then
  echo "formal host /data free space is below 160 GB: ${free_bytes}" >&2
  exit 1
fi
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits |
  grep -Ev '^0, 0$' | grep -q .; then
  echo "formal host has non-zero uncorrected ECC" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
sha256sum "${CFG}" "${SEAL}" \
  wm3d_v3/data/v8_causal_dual_view.py \
  wm3d_v3/data/window_dataset.py \
  wm3d_v3/data/v7_compact_dataset.py \
  wm3d_v3/training/train.py > "${INPUTS_FILE}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
case "${NODE_RANK}" in
  0)
    NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8
    ;;
  1)
    NCCL_IB_HCA_ALLOWLIST=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10
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
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m wm3d_v3.training.train \
  --cfg "${CFG}" \
  --print_every 20 \
  --stop_after_step 1000 \
  > "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "${pid}"
