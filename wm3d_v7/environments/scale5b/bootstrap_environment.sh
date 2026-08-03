#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?请通过 run_v7.sh 调用}"
: "${SYSTEM_PYTHON:=python3.10}"
: "${PYTHON_BIN:?site.env 缺少 PYTHON_BIN}"

ENV_PREFIX="$(dirname "$(dirname "${PYTHON_BIN}")")"
CONTRACT="${REPO_ROOT}/environments/scale5b/environment_contract.json"
REQUIREMENTS="${REPO_ROOT}/environments/scale5b/requirements.lock"
RECEIPT="${TRAIN_ENV_RECEIPT:-${ENV_PREFIX}/environment_receipt.json}"

command -v "${SYSTEM_PYTHON}" >/dev/null || {
  echo "找不到 ${SYSTEM_PYTHON}。请加载集群 Python 3.10 module，或修改 SYSTEM_PYTHON。" >&2
  exit 2
}
"${SYSTEM_PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"需要 Python 3.10，当前为 {sys.version.split()[0]}")
PY

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

receipt_sha() {
  "${PYTHON_BIN}" - "$RECEIPT" <<'PY'
import json
import sys
from wm3d_v3.data.scale5b_contracts import canonical_sha256
with open(sys.argv[1], encoding="utf-8") as handle:
    print(canonical_sha256(json.load(handle)))
PY
}

if [[ -f "${RECEIPT}" && -x "${PYTHON_BIN}" ]]; then
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/scale5b/verify_environment.py" \
    --contract "${CONTRACT}" --receipt "${RECEIPT}" \
    --expected-sha256 "$(receipt_sha)"
  echo "V7 环境已通过：${ENV_PREFIX}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  "${SYSTEM_PYTHON}" -m venv "${ENV_PREFIX}"
fi
"${PYTHON_BIN}" -m pip install --upgrade pip==25.1.1
"${PYTHON_BIN}" -m pip install \
  --index-url "${PIP_INDEX_URL:-https://pypi.org/simple}" \
  --extra-index-url "${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  --requirement "${REQUIREMENTS}"
"${PYTHON_BIN}" -m pip check

if [[ -e "${RECEIPT}" || -L "${RECEIPT}" ]]; then
  echo "环境 receipt 已存在但未通过，拒绝覆盖：${RECEIPT}" >&2
  exit 2
fi
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/scale5b/verify_environment.py" \
  --contract "${CONTRACT}" --output "${RECEIPT}"
echo "V7 环境安装完成：${ENV_PREFIX}"
