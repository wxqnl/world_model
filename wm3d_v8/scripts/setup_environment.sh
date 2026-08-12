#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SYSTEM_PYTHON=${SYSTEM_PYTHON:-python3.10}
ENV_DIR=${ENV_DIR:-${ROOT}/.venv}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.org/simple}
cd "${ROOT}"

RECEIPT_PATH="${ENV_DIR}/environment_receipt.json"
VERIFY_ONLY=0
if [[ -e "${RECEIPT_PATH}" || -L "${RECEIPT_PATH}" ]]; then
  [[ -f "${RECEIPT_PATH}" && ! -L "${RECEIPT_PATH}" ]] || {
    echo "环境 receipt 必须是普通文件，禁止目录、设备或符号链接：${RECEIPT_PATH}" >&2
    exit 2
  }
  [[ -x "${ENV_DIR}/bin/python" ]] || {
    echo "receipt 已存在但环境 Python 不可执行：${ENV_DIR}/bin/python" >&2
    exit 2
  }
  VERIFY_ONLY=1
  PYTHON_BIN="${ENV_DIR}/bin/python"
fi

if [[ "${VERIFY_ONLY}" == 0 ]]; then
  command -v "${SYSTEM_PYTHON}" >/dev/null || {
    echo "找不到 ${SYSTEM_PYTHON}；请加载 Python 3.10 module 或设置 SYSTEM_PYTHON。" >&2
    exit 2
  }
  "${SYSTEM_PYTHON}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"要求 Python 3.10，当前为 {sys.version.split()[0]}")
PY

  if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    "${SYSTEM_PYTHON}" -m venv "${ENV_DIR}"
  fi
  PYTHON_BIN="${ENV_DIR}/bin/python"
  "${PYTHON_BIN}" -m pip install --upgrade pip==25.1.1
  "${PYTHON_BIN}" -m pip install \
    --index-url "${PYPI_INDEX_URL}" \
    --extra-index-url "${PYTORCH_INDEX_URL}" \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128
  "${PYTHON_BIN}" -m pip install \
    --index-url "${PYPI_INDEX_URL}" \
    --requirement "${ROOT}/requirements.txt"
fi

# 无论首次建环境还是复用封存环境，都必须先做离线依赖一致性检查。
# 复用路径不会执行任何 pip install，因此不会访问网络或改写环境。
"${PYTHON_BIN}" -m pip check

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export WM3D_SETUP_ROOT="${ROOT}"
export WM3D_SETUP_PYTHON="${PYTHON_BIN}"
export WM3D_SETUP_RECEIPT="${RECEIPT_PATH}"
export WM3D_SETUP_VERIFY_ONLY="${VERIFY_ONLY}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
from pathlib import Path
import subprocess
import torch
from wm3d_v3.data.manifest_contract import sha256_file

root = Path(os.environ["WM3D_SETUP_ROOT"]).resolve(strict=True)
# 保留 venv/bin/python 这个启动路径。若 resolve 到 /usr/bin/python，
# `python -m pip freeze` 会错误地审计系统 Python，而不是当前 venv。
python = Path(os.environ["WM3D_SETUP_PYTHON"]).absolute()
if not python.is_file() or not os.access(python, os.X_OK):
    raise RuntimeError(f"环境 Python 不可执行：{python}")
path = Path(os.environ["WM3D_SETUP_RECEIPT"]).absolute()
receipt = {
    "schema": "wm3d_v8_environment_receipt_v1",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "requirements_path": str(root / "requirements.txt"),
    "requirements_sha256": sha256_file(root / "requirements.txt"),
    "pip_check": "pass",
    "pip_freeze": subprocess.check_output(
        [str(python), "-m", "pip", "freeze"],
        text=True,
    ).splitlines(),
}
payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
verify_only = os.environ["WM3D_SETUP_VERIFY_ONLY"] == "1"
if verify_only:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"环境 receipt 不是普通文件：{path}")
    sealed_payload = path.read_text(encoding="utf-8")
    if sealed_payload != payload:
        try:
            sealed = json.loads(sealed_payload)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"环境 receipt 不是合法 JSON：{path}") from error
        changed = sorted(
            key
            for key in set(sealed) | set(receipt)
            if sealed.get(key) != receipt.get(key)
        )
        raise RuntimeError(
            "当前环境与封存 receipt 不一致；"
            f"变化字段={changed}。禁止覆盖，请指定新的 ENV_DIR。"
        )
    print(
        json.dumps(
            {"passed": True, "receipt": str(path), "status": "verified-skip"},
            sort_keys=True,
        )
    )
    raise SystemExit(0)

if path.exists() or path.is_symlink():
    raise RuntimeError(f"环境 receipt 竞态冲突，禁止覆盖：{path}")
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
try:
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise RuntimeError(
            f"环境 receipt 竞态冲突，禁止覆盖：{path}"
        ) from error
finally:
    temporary.unlink(missing_ok=True)
print(
    json.dumps(
        {"passed": True, "receipt": str(path), "status": "created"},
        sort_keys=True,
    )
)
PY
