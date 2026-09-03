#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_PYTHON=${BASE_PYTHON:-${ROOT}/.venv/bin/python}
ENV_ROOT=${ENV_ROOT:-${ROOT}/.venv-foundation}
BASE_RECEIPT=${BASE_RECEIPT:-$(dirname "$(dirname "${BASE_PYTHON}")")/environment_receipt.json}
REQUIREMENTS=${ROOT}/environments/foundation_requirements.lock
RECEIPT=${ENV_ROOT}/environment_receipt.json
WSA_SOURCE_ROOT=${WM3D_WSA_SOURCE_ROOT:-/data/Minko/third_party/WSA}

[[ -x "${BASE_PYTHON}" ]] || {
  echo "missing base WM3D Python: ${BASE_PYTHON}" >&2
  exit 2
}
[[ -f "${BASE_RECEIPT}" && ! -L "${BASE_RECEIPT}" ]] || {
  echo "missing sealed base WM3D environment receipt: ${BASE_RECEIPT}" >&2
  exit 2
}
[[ -f "${REQUIREMENTS}" && ! -L "${REQUIREMENTS}" ]] || {
  echo "missing foundation requirements lock: ${REQUIREMENTS}" >&2
  exit 2
}
[[ -f "${WSA_SOURCE_ROOT}/src/lerobot/policies/WSA_Large/modeling_wsa_large.py" ]] || {
  echo "missing WSA-Large source tree: ${WSA_SOURCE_ROOT}" >&2
  exit 2
}

VERIFY_ONLY=0
if [[ -e "${RECEIPT}" || -L "${RECEIPT}" ]]; then
  [[ -f "${RECEIPT}" && ! -L "${RECEIPT}" ]] || {
    echo "foundation receipt must be a regular file: ${RECEIPT}" >&2
    exit 2
  }
  [[ -x "${ENV_ROOT}/bin/python" ]] || {
    echo "foundation receipt exists but Python is missing: ${ENV_ROOT}/bin/python" >&2
    exit 2
  }
  VERIFY_ONLY=1
fi

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv "${ENV_ROOT}"
fi

# A venv cannot inherit another venv through --system-site-packages. Reuse the
# already validated WM3D CUDA/PyTorch stack read-only, while keeping WSA-only
# packages isolated in ENV_ROOT (which remains first on sys.path).
BASE_SITE=$("${BASE_PYTHON}" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
FOUNDATION_SITE=$("${ENV_ROOT}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
printf '%s\n' "${BASE_SITE}" > "${FOUNDATION_SITE}/wm3d_base_environment.pth"

if [[ "${VERIFY_ONLY}" == 0 ]]; then
  "${ENV_ROOT}/bin/python" -m pip install --requirement "${REQUIREMENTS}"
fi

"${ENV_ROOT}/bin/python" -m pip check

export WM3D_WSA_SOURCE_ROOT="${WSA_SOURCE_ROOT}"
PYTHONPATH="${ROOT}:${WSA_SOURCE_ROOT}/src" \
  "${ENV_ROOT}/bin/python" - <<'PY'
import os
import torch
from wm3d.models.foundation_world_model import _load_upstream_symbols

WSALargeConfig, WSALargePolicy = _load_upstream_symbols(
    os.environ["WM3D_WSA_SOURCE_ROOT"]
)

assert torch.cuda.is_available()
print(
    "foundation environment ready:",
    torch.__version__,
    WSALargeConfig.__name__,
    WSALargePolicy.__name__,
)
PY

export WM3D_FOUNDATION_ROOT="${ROOT}"
export WM3D_FOUNDATION_ENV="${ENV_ROOT}"
export WM3D_FOUNDATION_BASE_RECEIPT="${BASE_RECEIPT}"
export WM3D_FOUNDATION_REQUIREMENTS="${REQUIREMENTS}"
export WM3D_FOUNDATION_RECEIPT="${RECEIPT}"
export WM3D_FOUNDATION_VERIFY_ONLY="${VERIFY_ONLY}"
PYTHONPATH="${ROOT}:${WSA_SOURCE_ROOT}/src" \
  "${ENV_ROOT}/bin/python" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path

import torch
import transformers


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


root = Path(os.environ["WM3D_FOUNDATION_ROOT"]).resolve(strict=True)
environment = Path(os.environ["WM3D_FOUNDATION_ENV"]).resolve(strict=True)
base_receipt = Path(os.environ["WM3D_FOUNDATION_BASE_RECEIPT"]).resolve(strict=True)
requirements = Path(os.environ["WM3D_FOUNDATION_REQUIREMENTS"]).resolve(strict=True)
receipt_path = Path(os.environ["WM3D_FOUNDATION_RECEIPT"]).absolute()
site_root = next((environment / "lib").glob("python*/site-packages")).resolve(strict=True)
overlay = sorted(
    {
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in metadata.distributions(path=[str(site_root)])
        if distribution.metadata.get("Name") not in {"pip", "setuptools"}
    },
    key=str.casefold,
)
receipt = {
    "schema": "wm3d_foundation_environment_receipt_v1",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "transformers": transformers.__version__,
    "base_environment_receipt_path": str(base_receipt),
    "base_environment_receipt_sha256": digest(base_receipt),
    "foundation_requirements_path": str(requirements),
    "foundation_requirements_sha256": digest(requirements),
    "foundation_overlay": overlay,
    "pip_check": "pass",
}
payload = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
if os.environ["WM3D_FOUNDATION_VERIFY_ONLY"] == "1":
    if receipt_path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(
            "foundation environment differs from its sealed receipt; use a new ENV_ROOT"
        )
    print(json.dumps({"passed": True, "receipt": str(receipt_path), "status": "verified"}))
else:
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError(f"refusing to replace foundation receipt: {receipt_path}")
    temporary = receipt_path.with_name(f".{receipt_path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, receipt_path)
    temporary.unlink()
    print(json.dumps({"passed": True, "receipt": str(receipt_path), "status": "created"}))
PY
