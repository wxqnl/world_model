#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EXAMPLE="${ROOT}/configs/cluster/h200.env.example"

usage() {
  cat <<'EOF'
WM3D

  ./wm3d.sh init [site.env]                  生成站点配置
  ./wm3d.sh setup site.env                   创建 Python 3.10 venv
  ./wm3d.sh doctor site.env                  检查环境、权限和集群配置
  ./wm3d.sh plan site.env                    打印完整执行计划

  ./wm3d.sh lock site.env                    固定公开数据 revision
  ./wm3d.sh download site.env                下载或续传原始数据
  ./wm3d.sh prepare site.env                 转换数据并生成 episode plan
  ./wm3d.sh cache site.env                   生成 cache 与 dataset seal
  ./wm3d.sh data site.env                    依次执行上述四个数据阶段

  ./wm3d.sh train site.env                   canary + eval + 正式训练
  ./wm3d.sh eval site.env STEP_CHECKPOINT    评测完整编号 checkpoint
  ./wm3d.sh status site.env                  查看数据与训练状态
  ./wm3d.sh all site.env                     setup + data + train
  ./wm3d.sh params site.env [TRAIN_YAML]     计算模型参数组成

  ./wm3d.sh smoke /abs/work-root             GPU0-1 公开小样本全流程

数据、代码、环境、配置和 checkpoint 都会生成可哈希 receipt。重复执行同一阶段会验证
已有结果并从未完成处继续；训练只从含 COMMITTED.json 的编号 checkpoint 恢复。
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
  echo "已生成 ${target}。填写必填项后运行："
  echo "  ./wm3d.sh setup ${target}"
  echo "  ./wm3d.sh doctor ${target}"
  exit 0
fi

if [[ "${action}" == "smoke" ]]; then
  if [[ $# -ne 2 || "$2" != /* ]]; then
    echo "用法：./wm3d.sh smoke /abs/work-root" >&2
    exit 2
  fi
  exec "${ROOT}/scripts/smoke/run.sh" "$2"
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
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname "${PYTHON_BIN}"):$(dirname "${CONVERTER_PYTHON_BIN}"):${PATH}"

setup() {
  "${ROOT}/environments/bootstrap_environment.sh"
}

require_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "环境尚未安装，请先运行 ./wm3d.sh setup ${SITE_CONFIG}" >&2
    exit 2
  fi
}

pipeline() {
  "${PYTHON_BIN}" "${ROOT}/scripts/pipeline.py" "$@" --site "${SITE_CONFIG}"
}

case "${action}" in
  setup)
    setup
    ;;
  doctor|plan|lock|download|prepare|cache|data|status)
    require_python
    if [[ "${action}" == "plan" ]]; then
      pipeline all --dry-run
    else
      pipeline "${action}"
    fi
    ;;
  train)
    require_python
    pipeline canary
    pipeline train
    ;;
  eval)
    require_python
    if [[ $# -ne 3 ]]; then
      echo "用法：./wm3d.sh eval site.env /abs/.../step_XXXXXXXX" >&2
      exit 2
    fi
    pipeline eval --checkpoint "$3"
    ;;
  params)
    require_python
    if [[ $# -gt 3 ]]; then
      echo "用法：./wm3d.sh params site.env [TRAIN_YAML]" >&2
      exit 2
    fi
    config="${3:-${ROOT}/configs/train/5b_h200.yaml}"
    if [[ "${config}" != /* ]]; then
      config="${ROOT}/${config}"
    fi
    config="$(realpath -e -- "${config}")"
    exec "${PYTHON_BIN}" "${ROOT}/scripts/tools/report_parameters.py" \
      --config "${config}"
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
