#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXAMPLE="${ROOT}/configs/examples/v7_native5b_h200.env"

usage() {
  cat <<'EOF'
WM3D-V7

  ./run_v7.sh init [site.env]              # 生成站点配置
  ./run_v7.sh setup site.env               # 创建普通 Python venv 并安装依赖
  ./run_v7.sh plan site.env                # 打印将执行的完整流程
  ./run_v7.sh data site.env                # 下载、转换、cache、seal
  ./run_v7.sh train site.env               # 1k canary + eval，通过后提交 formal
  ./run_v7.sh eval site.env STEP_CHECKPOINT
  ./run_v7.sh all site.env                 # setup + data + train
  ./run_v7.sh smoke /abs/work-root         # 少量公开数据，2 卡真实 5B 全流程

辅助命令：doctor、status。每个阶段都会生成 receipt，重跑时从已完成阶段继续。
smoke 默认只使用当前机器 GPU0–1，不需要 site.env，不会提交 Slurm 任务。
EOF
}

action="${1:-help}"
if [[ "${action}" == "help" || "${action}" == "-h" || "${action}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${action}" == "init" ]]; then
  target="${2:-site.env}"
  if [[ -e "${target}" || -L "${target}" ]]; then
    echo "拒绝覆盖：${target}" >&2
    exit 2
  fi
  cp -- "${EXAMPLE}" "${target}"
  chmod 0600 "${target}"
  echo "已生成 ${target}。填写标有“必填”的项目后运行："
  echo "  ./run_v7.sh all ${target}"
  exit 0
fi

if [[ "${action}" == "smoke" ]]; then
  if [[ $# -ne 2 || "$2" != /* ]]; then
    echo "用法：./run_v7.sh smoke /abs/work-root" >&2
    exit 2
  fi
  exec "${ROOT}/scripts/scale5b/run_public_smoke.sh" "$2"
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi
SITE_CONFIG="$(realpath -e -- "$2")"
if [[ -L "${SITE_CONFIG}" || ! -f "${SITE_CONFIG}" ]]; then
  echo "site.env 必须是普通文件：${SITE_CONFIG}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${SITE_CONFIG}"
set +a
export SITE_CONFIG REPO_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON_BIN}"):$(dirname "${CONVERTER_PYTHON_BIN}"):${PATH}"

setup() {
  "${ROOT}/environments/scale5b/bootstrap_environment.sh"
  "${ROOT}/environments/scale5b/bootstrap_agibot_converter_environment.sh"
}

require_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "环境尚未安装，请先运行 ./run_v7.sh setup ${SITE_CONFIG}" >&2
    exit 2
  fi
}

pipeline() {
  "${PYTHON_BIN}" "${ROOT}/scripts/scale5b/pipeline_native5b.py" \
    "$@" --site "${SITE_CONFIG}"
}

case "${action}" in
  setup)
    setup
    ;;
  doctor)
    require_python
    pipeline doctor
    ;;
  plan)
    require_python
    pipeline all --dry-run
    ;;
  data)
    require_python
    pipeline data
    ;;
  train)
    require_python
    pipeline canary
    pipeline train
    ;;
  eval)
    require_python
    if [[ $# -ne 3 ]]; then
      echo "用法：./run_v7.sh eval site.env /abs/.../step_XXXXXXXX" >&2
      exit 2
    fi
    pipeline eval --checkpoint "$3"
    ;;
  status)
    require_python
    pipeline status
    ;;
  all)
    setup
    pipeline data
    pipeline canary
    pipeline train
    ;;
  *)
    echo "未知命令：${action}" >&2
    usage >&2
    exit 2
    ;;
esac
