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
    --requirement "${ROOT}/environments/requirements.lock"

  # PyPI's decord 0.6.0 Linux payload is Python-version agnostic (the Python
  # package loads libdecord.so through ctypes), but some mirrors serve a wheel
  # whose internal WHEEL tag is incorrectly left at cp36.  pip can install and
  # import it on Python 3.10, then `pip check` rejects the stale internal tag.
  # Normalize only this known payload, update RECORD, and prove the installed
  # library imports before sealing the environment receipt.
  "${PYTHON_BIN}" - <<'PY'
import base64
import csv
import hashlib
import io
from pathlib import Path
import platform
import sys

import decord

if sys.version_info[:2] != (3, 10):
    raise RuntimeError("decord wheel normalization is sealed for Python 3.10")
if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise RuntimeError("decord wheel normalization requires Linux x86_64")
site_root = Path(decord.__file__).resolve().parent.parent
dist_roots = sorted(site_root.glob("decord-0.6.0.dist-info"))
if len(dist_roots) != 1:
    raise RuntimeError(f"expected one decord 0.6.0 dist-info, found {dist_roots}")
dist_root = dist_roots[0]
wheel = dist_root / "WHEEL"
record = dist_root / "RECORD"
payload = wheel.read_text(encoding="utf-8")
stale = "Tag: cp36-cp36m-manylinux2010_x86_64"
correct = "Tag: cp310-cp310-manylinux2010_x86_64"
if stale in payload:
    payload = payload.replace(stale, correct)
    temporary_wheel = wheel.with_name(f".{wheel.name}.tmp")
    temporary_wheel.write_text(payload, encoding="utf-8")
    temporary_wheel.replace(wheel)
elif correct not in payload:
    raise RuntimeError(f"unexpected decord wheel tag in {wheel}: {payload!r}")

rows = list(csv.reader(io.StringIO(record.read_text(encoding="utf-8"))))
wheel_relative = wheel.relative_to(site_root).as_posix()
wheel_bytes = wheel.read_bytes()
digest = base64.urlsafe_b64encode(hashlib.sha256(wheel_bytes).digest()).rstrip(b"=")
matched = 0
for row in rows:
    if row and row[0] == wheel_relative:
        row[1] = f"sha256={digest.decode()}"
        row[2] = str(len(wheel_bytes))
        matched += 1
if matched != 1:
    raise RuntimeError(f"decord RECORD does not uniquely contain {wheel_relative}")
temporary = record.with_name(f".{record.name}.tmp")
with temporary.open("w", encoding="utf-8", newline="") as handle:
    csv.writer(handle, lineterminator="\n").writerows(rows)
temporary.replace(record)
if decord.__version__ != "0.6.0":
    raise RuntimeError(f"unexpected decord version {decord.__version__}")
decord.cpu(0)
PY
fi

# 无论首次建环境还是复用封存环境，都必须先做离线依赖一致性检查。
# 复用路径不会执行任何 pip install，因此不会访问网络或改写环境。
"${PYTHON_BIN}" -m pip check

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export WM3D_ENV_ROOT="${ROOT}"
export WM3D_ENV_PYTHON="${PYTHON_BIN}"
export WM3D_ENV_RECEIPT="${RECEIPT_PATH}"
export WM3D_ENV_VERIFY_ONLY="${VERIFY_ONLY}"
"${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
from pathlib import Path
import subprocess
import torch
from wm3d.data.manifest_contract import sha256_file

root = Path(os.environ["WM3D_ENV_ROOT"]).resolve(strict=True)
# 保留 venv/bin/python 这个启动路径。若 resolve 到 /usr/bin/python，
# `python -m pip freeze` 会错误地审计系统 Python，而不是当前 venv。
python = Path(os.environ["WM3D_ENV_PYTHON"]).absolute()
if not python.is_file() or not os.access(python, os.X_OK):
    raise RuntimeError(f"环境 Python 不可执行：{python}")
path = Path(os.environ["WM3D_ENV_RECEIPT"]).absolute()
receipt = {
    "schema": "wm3d_v8_environment_receipt_v1",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "requirements_path": str(root / "environments/requirements.lock"),
    "requirements_sha256": sha256_file(root / "environments/requirements.lock"),
    "pip_check": "pass",
    "pip_freeze": subprocess.check_output(
        [str(python), "-m", "pip", "freeze"],
        text=True,
    ).splitlines(),
}
payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
verify_only = os.environ["WM3D_ENV_VERIFY_ONLY"] == "1"
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
