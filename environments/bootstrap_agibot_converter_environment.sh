#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?请通过 wm3d.sh 调用}"
: "${SYSTEM_PYTHON:=python3.10}"
: "${ENV_ROOT:?site.env 缺少 ENV_ROOT}"
: "${STAGING_ROOT:?site.env 缺少 STAGING_ROOT}"
: "${CONVERTER_PYTHON_BIN:?site.env 缺少 CONVERTER_PYTHON_BIN}"

REVISION=8e7d6970eaf5a64b8af6ec45586d201b8ca9ef16
SOURCE_SHA=51e51e7e2d91c46db3bb4ccb9604d55776d9f9f90389465ed2603fe9f9bbc702
PYPROJECT_SHA=34a923b9d6739c52d63af14d20282d5cbebbc78a46a81d76600ad33ae4057d66
LOCK_SHA=fabc7c9544e0073cabc2cf351c38b92a6f6578f343320be91a65c4050a7a10d3

ENV_PREFIX="$(dirname "$(dirname "${CONVERTER_PYTHON_BIN}")")"
CONTRACT_SOURCE="${REPO_ROOT}/environments/agibot_converter_environment_contract.json"
CONTRACT="${ENV_PREFIX}/environment_contract.json"
REVISION_FILE="${ENV_PREFIX}/LEROBOT_REVISION"
RECEIPT="${CONVERTER_ENV_RECEIPT:-${ENV_PREFIX}/environment_receipt.json}"
CACHE="${STAGING_ROOT}/lerobot_converter"
ARCHIVE="${CACHE}/lerobot-${REVISION}.tar.gz"
SOURCE_ROOT="${CACHE}/lerobot-${REVISION}"
BUILD_ROOT="${CACHE}/lerobot-${REVISION}-wm3d-build"
POETRY_ENV="${ENV_ROOT}/poetry-1.8.5"
PREPARE_SCRIPT="${REPO_ROOT}/environments/prepare_lerobot_converter_build.py"

command -v "${SYSTEM_PYTHON}" >/dev/null || {
  echo "找不到 ${SYSTEM_PYTHON}" >&2
  exit 2
}
command -v ffmpeg >/dev/null || {
  echo "缺少系统 ffmpeg。请用集群包管理器安装后重试。" >&2
  exit 2
}
"${SYSTEM_PYTHON}" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"AgiBot converter 需要 Python 3.10，当前为 {sys.version.split()[0]}")
if sys.platform != "linux" or platform.machine() != "x86_64":
    raise SystemExit(
        "AgiBot converter 当前只支持 Linux x86_64: "
        f"actual={sys.platform}/{platform.machine()}"
    )
PY
mkdir -p "${CACHE}" "${ENV_ROOT}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

for guarded in \
  "${ARCHIVE}" "${SOURCE_ROOT}" "${BUILD_ROOT}" \
  "${CONTRACT}" "${REVISION_FILE}" "${RECEIPT}"; do
  if [[ -L "${guarded}" ]]; then
    echo "拒绝使用 symlink：${guarded}" >&2
    exit 2
  fi
done
if [[ -e "${ARCHIVE}" && ! -f "${ARCHIVE}" ]]; then
  echo "LeRobot archive 必须是普通文件：${ARCHIVE}" >&2
  exit 2
fi
for directory in "${SOURCE_ROOT}" "${BUILD_ROOT}"; do
  if [[ -e "${directory}" && ! -d "${directory}" ]]; then
    echo "LeRobot 工作路径必须是目录：${directory}" >&2
    exit 2
  fi
done
for regular in "${CONTRACT}" "${REVISION_FILE}" "${RECEIPT}"; do
  if [[ -e "${regular}" && ! -f "${regular}" ]]; then
    echo "环境证据必须是普通文件：${regular}" >&2
    exit 2
  fi
done
if [[ -f "${CONTRACT}" ]] && ! cmp --silent "${CONTRACT_SOURCE}" "${CONTRACT}"; then
  echo "已安装 converter contract 与当前 release 不一致：${CONTRACT}" >&2
  exit 2
fi
if [[ -f "${REVISION_FILE}" && "$(<"${REVISION_FILE}")" != "${REVISION}" ]]; then
  echo "已安装 LeRobot revision 与当前 release 不一致：${REVISION_FILE}" >&2
  exit 2
fi

if [[ -f "${RECEIPT}" && -x "${CONVERTER_PYTHON_BIN}" ]]; then
  "${CONVERTER_PYTHON_BIN}" \
    "${REPO_ROOT}/environments/verify_agibot_converter_environment.py" \
    --contract "${CONTRACT}" --revision-file "${REVISION_FILE}" \
    --receipt "${RECEIPT}"
  echo "AgiBot 转换环境已通过：${ENV_PREFIX}"
  exit 0
fi

if [[ ! -x "${CONVERTER_PYTHON_BIN}" ]]; then
  "${SYSTEM_PYTHON}" -m venv "${ENV_PREFIX}"
  "${CONVERTER_PYTHON_BIN}" -m pip install --upgrade pip==24.3.1
fi
if [[ ! -x "${POETRY_ENV}/bin/poetry" ]]; then
  "${SYSTEM_PYTHON}" -m venv "${POETRY_ENV}"
  "${POETRY_ENV}/bin/python" -m pip install \
    pip==24.3.1 poetry==1.8.5 poetry-plugin-export==1.8.0
fi

if [[ -f "${ARCHIVE}" ]]; then
  echo "${SOURCE_SHA}  ${ARCHIVE}" | sha256sum --check -
else
  tmp="${ARCHIVE}.incomplete.$$"
  curl --fail --location --retry 5 --output "${tmp}" \
    "https://github.com/huggingface/lerobot/archive/${REVISION}.tar.gz"
  echo "${SOURCE_SHA}  ${tmp}" | sha256sum --check -
  mv "${tmp}" "${ARCHIVE}"
fi
if [[ ! -d "${SOURCE_ROOT}" ]]; then
  tmp="${SOURCE_ROOT}.incomplete.$$"
  mkdir "${tmp}"
  tar -xzf "${ARCHIVE}" -C "${tmp}" --strip-components=1
  mv "${tmp}" "${SOURCE_ROOT}"
fi
if [[ -L "${SOURCE_ROOT}" || ! -d "${SOURCE_ROOT}" ]]; then
  echo "LeRobot source root 必须是目录且不能是 symlink：${SOURCE_ROOT}" >&2
  exit 2
fi
echo "${PYPROJECT_SHA}  ${SOURCE_ROOT}/pyproject.toml" | sha256sum --check -
echo "${LOCK_SHA}  ${SOURCE_ROOT}/poetry.lock" | sha256sum --check -

raw_requirements="${CACHE}/requirements-${REVISION}.raw.txt"
requirements="${CACHE}/requirements-${REVISION}.linux-x86_64.txt"
(cd "${SOURCE_ROOT}" && "${POETRY_ENV}/bin/poetry" check --lock)
(cd "${SOURCE_ROOT}" && "${POETRY_ENV}/bin/poetry" export \
  --only main --format requirements.txt --output "${raw_requirements}")
if [[ ! -e "${BUILD_ROOT}" ]]; then
  tmp="${BUILD_ROOT}.incomplete.$$"
  mkdir "${tmp}"
  cp -a "${SOURCE_ROOT}/." "${tmp}/"
  mv "${tmp}" "${BUILD_ROOT}"
fi
if [[ -L "${BUILD_ROOT}" || ! -d "${BUILD_ROOT}" ]]; then
  echo "LeRobot build root 必须是目录且不能是 symlink：${BUILD_ROOT}" >&2
  exit 2
fi
"${SYSTEM_PYTHON}" "${PREPARE_SCRIPT}" \
  --requirements-input "${raw_requirements}" \
  --requirements-output "${requirements}" \
  --build-pyproject "${BUILD_ROOT}/pyproject.toml"
"${CONVERTER_PYTHON_BIN}" -m pip install \
  --require-hashes --requirement "${requirements}"
(cd "${BUILD_ROOT}" && POETRY_VIRTUALENVS_CREATE=false \
  "${POETRY_ENV}/bin/poetry" build --format wheel)
"${CONVERTER_PYTHON_BIN}" -m pip install --no-deps \
  "${BUILD_ROOT}/dist/lerobot-0.1.0-py3-none-any.whl"
"${CONVERTER_PYTHON_BIN}" -m pip check

if [[ ! -e "${CONTRACT}" ]]; then
  install -m 0640 "${CONTRACT_SOURCE}" "${CONTRACT}"
fi
if [[ ! -e "${REVISION_FILE}" ]]; then
  printf '%s\n' "${REVISION}" > "${REVISION_FILE}"
  chmod 0640 "${REVISION_FILE}"
fi
if [[ -e "${RECEIPT}" || -L "${RECEIPT}" ]]; then
  echo "converter receipt 已存在但未通过，拒绝覆盖：${RECEIPT}" >&2
  exit 2
fi
"${CONVERTER_PYTHON_BIN}" \
  "${REPO_ROOT}/environments/verify_agibot_converter_environment.py" \
  --contract "${CONTRACT}" --revision-file "${REVISION_FILE}" \
  --output "${RECEIPT}"
echo "AgiBot 转换环境安装完成：${ENV_PREFIX}"
