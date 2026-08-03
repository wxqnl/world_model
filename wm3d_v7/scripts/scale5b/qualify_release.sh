#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/wm3d/bin/torchrun}"
ENV_PREFIX="$(dirname "$(dirname "${PYTHON_BIN}")")"
ENVIRONMENT_CONTRACT="${ENVIRONMENT_CONTRACT:-${REPO_ROOT}/environments/scale5b/environment_contract.json}"
ENVIRONMENT_RECEIPT="${TRAIN_ENV_RECEIPT:-${ENV_PREFIX}/environment_receipt.json}"
export ENVIRONMENT_RECEIPT

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXPECTED_ENV_SHA="$(
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from wm3d_v3.data.scale5b_contracts import canonical_sha256
with open(os.environ["ENVIRONMENT_RECEIPT"], encoding="utf-8") as handle:
    print(canonical_sha256(json.load(handle)))
PY
)"
"${PYTHON_BIN}" scripts/scale5b/verify_environment.py \
  --contract "${ENVIRONMENT_CONTRACT}" \
  --receipt "${ENVIRONMENT_RECEIPT}" \
  --expected-sha256 "${EXPECTED_ENV_SHA}"

"${PYTHON_BIN}" -m compileall -q \
  wm3d_v3/data \
  wm3d_v3/encoders/native5b_vggt.py \
  wm3d_v3/encoders/vggt_encoder.py \
  wm3d_v3/models/native5b.py \
  wm3d_v3/training \
  scripts/scale5b \
  run_v7.sh \
  tests/test_scale5b_native.py \
  tests/test_scale5b_data_pipeline.py \
  tests/test_scale5b_handoff.py \
  tests/scale5b_fsdp2_smoke.py

"${PYTHON_BIN}" -m ruff check \
  wm3d_v3/data/scale5b_*.py \
  wm3d_v3/encoders/native5b_vggt.py \
  wm3d_v3/encoders/vggt_encoder.py \
  wm3d_v3/models/native5b.py \
  wm3d_v3/training/scale5b_*.py \
  wm3d_v3/training/train_native5b.py \
  wm3d_v3/training/eval_native5b.py \
  scripts/scale5b \
  tests/test_scale5b_*.py \
  tests/scale5b_fsdp2_smoke.py

"${PYTHON_BIN}" -m pytest -q \
  tests/test_scale5b_native.py \
  tests/test_scale5b_data_pipeline.py \
  tests/test_scale5b_handoff.py

"${PYTHON_BIN}" - <<'PY'
import json
import torch
from wm3d_v3.models.native5b import Native5BConfig, NativeWM3D5B
with torch.device("meta"):
    model = NativeWM3D5B(Native5BConfig())
counts = model.parameter_counts()
assert counts["total"] == 4_956_589_929, counts
print(json.dumps({"pass": True, "parameter_counts": counts}, sort_keys=True))
PY

while IFS= read -r script; do
  bash -n "${script}"
done < <(
  find scripts/scale5b environments/scale5b -type f -name '*.sh' -print
  printf '%s\n' run_v7.sh
)

if [[ "${RUN_GPU_SMOKE:-0}" == "1" ]]; then
  : "${GPU_SMOKE_ROOT:?Set a new absolute checkpoint root for the 2-GPU smoke}"
  if [[ "${GPU_SMOKE_ROOT}" != /* || -e "${GPU_SMOKE_ROOT}" ]]; then
    echo "GPU_SMOKE_ROOT must be an unused absolute path" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "${TORCHRUN_BIN}" --standalone --nproc-per-node=2 \
    tests/scale5b_fsdp2_smoke.py --root "${GPU_SMOKE_ROOT}"
fi

echo "WM3D-V7 native 5B release qualification PASS"
