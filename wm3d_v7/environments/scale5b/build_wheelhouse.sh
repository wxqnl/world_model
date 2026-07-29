#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LOCK_FILE="${SCRIPT_DIR}/requirements.lock"
OUTPUT="${1:-${SCRIPT_DIR}/wheelhouse}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "The formal H200 wheelhouse must be built on x86_64 Linux" >&2
  exit 2
fi
if [[ "${OUTPUT}" != /* ]]; then
  OUTPUT="$(readlink -f "$(dirname "${OUTPUT}")")/$(basename "${OUTPUT}")"
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse wheelhouse output: ${OUTPUT}" >&2
  exit 2
fi
"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"wheelhouse requires CPython 3.10, got {sys.version}")
PY

mkdir -p "$(dirname "${OUTPUT}")"
TEMPORARY="$(mktemp -d "${OUTPUT}.incomplete.XXXXXXXX")"
echo "Incomplete wheelhouse evidence: ${TEMPORARY}"

"${PYTHON_BIN}" -m pip download \
  --dest "${TEMPORARY}" \
  --no-deps \
  --only-binary=:all: \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --requirement "${LOCK_FILE}"

cp "${LOCK_FILE}" "${TEMPORARY}/requirements.lock"
(
  cd "${TEMPORARY}"
  LC_ALL=C sha256sum ./*.whl | LC_ALL=C sort > WHEELHOUSE.SHA256
  sha256sum requirements.lock > REQUIREMENTS.SHA256
  sha256sum --check WHEELHOUSE.SHA256
  sha256sum --check REQUIREMENTS.SHA256
)
mv "${TEMPORARY}" "${OUTPUT}"
echo "Published immutable wheelhouse: ${OUTPUT}"
