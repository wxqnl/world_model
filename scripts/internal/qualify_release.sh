#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-/opt/wm3d/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/wm3d/bin/torchrun}"
ENV_PREFIX="$(dirname "$(dirname "${PYTHON_BIN}")")"
ENVIRONMENT_CONTRACT="${ENVIRONMENT_CONTRACT:-${REPO_ROOT}/environments/environment_contract.json}"
ENVIRONMENT_RECEIPT="${TRAIN_ENV_RECEIPT:-${ENV_PREFIX}/environment_receipt.json}"
export ENVIRONMENT_RECEIPT

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

EXPECTED_ENV_SHA="$(
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from wm3d.data.contracts import canonical_sha256
with open(os.environ["ENVIRONMENT_RECEIPT"], encoding="utf-8") as handle:
    print(canonical_sha256(json.load(handle)))
PY
)"
"${PYTHON_BIN}" scripts/internal/verify_environment.py \
  --contract "${ENVIRONMENT_CONTRACT}" \
  --receipt "${ENVIRONMENT_RECEIPT}" \
  --expected-sha256 "${EXPECTED_ENV_SHA}"

"${PYTHON_BIN}" -m compileall -q \
  wm3d/data \
  wm3d/encoders/vggt_features.py \
  wm3d/encoders/vggt_encoder.py \
  wm3d/models/wm3d.py \
  wm3d/training \
  scripts \
  wm3d.sh \
  tests/test_model.py \
  tests/test_data_pipeline.py \
  tests/test_handoff.py \
  tests/fsdp2_smoke.py

"${PYTHON_BIN}" -m ruff check \
  wm3d/data/*.py \
  wm3d/encoders/vggt_features.py \
  wm3d/encoders/vggt_encoder.py \
  wm3d/models/wm3d.py \
  wm3d/training/*.py \
  wm3d/training/train.py \
  wm3d/training/eval.py \
  scripts \
  tests/test_*.py \
  tests/fsdp2_smoke.py

"${PYTHON_BIN}" -m pytest -q \
  tests/test_model.py \
  tests/test_data_pipeline.py \
  tests/test_handoff.py

"${PYTHON_BIN}" - <<'PY'
import json
import torch
from wm3d.models.wm3d import WM3DConfig, WM3D
with torch.device("meta"):
    model = WM3D(WM3DConfig())
counts = model.parameter_counts()
assert counts["total"] == 4_956_589_929, counts
print(json.dumps({"pass": True, "parameter_counts": counts}, sort_keys=True))
PY

while IFS= read -r script; do
  bash -n "${script}"
done < <(
  find scripts environments -type f -name '*.sh' -print
  printf '%s\n' wm3d.sh
)

if [[ "${RUN_GPU_SMOKE:-0}" == "1" ]]; then
  : "${GPU_SMOKE_ROOT:?Set a new absolute checkpoint root for the 2-GPU smoke}"
  if [[ "${GPU_SMOKE_ROOT}" != /* || -e "${GPU_SMOKE_ROOT}" ]]; then
    echo "GPU_SMOKE_ROOT must be an unused absolute path" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    "${TORCHRUN_BIN}" --standalone --nproc-per-node=2 \
    tests/fsdp2_smoke.py --root "${GPU_SMOKE_ROOT}"
fi

echo "WM3D WM3D release qualification PASS"
