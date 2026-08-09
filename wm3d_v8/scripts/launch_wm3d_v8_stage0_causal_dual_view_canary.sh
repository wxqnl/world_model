#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"
if [[ "${MODE}" != "check" && "${MODE}" != "launch" ]]; then
  echo "usage: $0 [check|launch]" >&2
  exit 2
fi

PROJECT=/data/Minko/wm3d_v8_release_worktree/wm3d_v8
RUNTIME_ROOT=/data/Minko/world_model/wm3d_v8_stage0_causal_dual_view_20260809
CFG=${WM3D_V8_CANARY_CFG:-${RUNTIME_ROOT}/manifests/canary/runtime_config_smoke20_v1.yaml}
SEAL=${WM3D_V8_CANARY_SEAL:-${RUNTIME_ROOT}/manifests/canary/seal_report_smoke20_v1.json}
OUT=${RUNTIME_ROOT}/results/training_canary20_v1
LOG_DIR=${RUNTIME_ROOT}/logs/training_canary20_v1
LOG=${LOG_DIR}/train_rank0.log
PREFLIGHT_LOG=${LOG_DIR}/preflight_full_v2.json
PID_FILE=${LOG_DIR}/launcher_rank0.pid
PY=/root/miniconda3/envs/starvla/bin/python

EXPECTED_SEAL_SHA=${WM3D_V8_CANARY_SEAL_SHA256:-}
if [[ ! "${EXPECTED_SEAL_SHA}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "WM3D_V8_CANARY_SEAL_SHA256 must be an exact lowercase SHA256" >&2
  exit 1
fi

cd "${PROJECT}"
test -s "${CFG}"
test -s "${SEAL}"
observed_seal_sha=$(sha256sum "${SEAL}" | awk '{print $1}')
if [[ "${observed_seal_sha}" != "${EXPECTED_SEAL_SHA}" ]]; then
  echo "seal report SHA mismatch: expected=${EXPECTED_SEAL_SHA} observed=${observed_seal_sha}" >&2
  exit 1
fi

"${PY}" - "${SEAL}" "${CFG}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from scripts.preflight_wm3d_v8_stage0_causal_dual_view import (
    load_config,
    resolved_config_sha256,
)

seal_path = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
report = json.loads(seal_path.read_text())
if report.get("schema") != "wm3d_v8_stage0_causal_dual_view_canary_seal_v1":
    raise SystemExit("unexpected seal report schema")
if report.get("passed") is not True or report.get("launch_ready") is not True:
    raise SystemExit("seal report is not launch-ready")
if Path(str(report.get("runtime_config") or "")).resolve() != config_path:
    raise SystemExit("seal report is bound to a different runtime config")
observed_config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
if observed_config_sha != report.get("runtime_config_sha256"):
    raise SystemExit("runtime config content SHA does not match seal report")
config = load_config(config_path)
if resolved_config_sha256(config) != report.get("resolved_config_sha256"):
    raise SystemExit("resolved runtime config SHA does not match seal report")
train = config.get("train") or {}
out = config.get("out") or {}
if int(train.get("max_steps", -1)) != 20:
    raise SystemExit("bounded canary must hard-stop at step 20")
if train.get("resume_checkpoint") is not None:
    raise SystemExit("bounded canary forbids resume")
if not bool(train.get("fresh_initialization_required")):
    raise SystemExit("bounded canary requires fresh initialization")
if Path(str(out.get("root") or "")).resolve() != Path(
    "/data/Minko/world_model/wm3d_v8_stage0_causal_dual_view_20260809/"
    "results/training_canary20_v1"
):
    raise SystemExit("bounded canary output root mismatch")
PY

if pgrep -af "[w]m3d_v3.training.train.*${CFG}" >/dev/null; then
  echo "duplicate V8 causal dual-view canary process" >&2
  exit 1
fi
if find "${OUT}/ckpt" -maxdepth 1 -name 'step_*.pt' -print -quit 2>/dev/null | grep -q .; then
  echo "bounded canary checkpoint directory is not empty" >&2
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
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits |
  grep -Ev '^0, 0$' | grep -q .; then
  echo "node43 has non-zero uncorrected ECC" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
"${PY}" scripts/preflight_wm3d_v8_stage0_causal_dual_view.py --config "${CFG}" --mode full > "${PREFLIGHT_LOG}"

if [[ "${MODE}" == "check" ]]; then
  echo "V8 causal dual-view canary launch check passed"
  exit 0
fi

sha256sum "${CFG}" "${SEAL}" wm3d_v3/data/v8_causal_dual_view.py wm3d_v3/data/window_dataset.py wm3d_v3/data/v7_compact_dataset.py wm3d_v3/training/train.py > "${LOG_DIR}/inputs.sha256"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

setsid "${PY}" -m torch.distributed.run --standalone --nproc_per_node=8 -m wm3d_v3.training.train --cfg "${CFG}" --print_every 1 > "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "${pid}"
