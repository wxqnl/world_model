#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
用法：
  ./run_v8.sh check
  ./run_v8.sh stage0-static
  ./run_v8.sh stage1-static [phase-config]
  ./run_v8.sh stage1-data [phase-config]

正式训练仍由封存的 Stage0/Stage1 启动脚本执行；本入口不自动跨阶段晋级。
EOF
}

stage1_cfg=${2:-configs/wm3d_v7_stage1_planner_dynamics10k.yaml}

case "${1:-}" in
  check)
    "${PYTHON_BIN}" -m compileall -q wm3d_v3 scripts tests
    bash -n \
      scripts/launch_wm3d_v7_1b_actionpolicy_joint_canary1000_node43_v3.sh \
      scripts/launch_wm3d_v7_1b_actionpolicy_joint_formal100k_node_v3.sh \
      scripts/start_wm3d_v7_1b_actionpolicy_joint_formal100k_3node24_v3.sh \
      scripts/launch_wm3d_v7_stage1_planner.sh
    "${PYTHON_BIN}" -m pytest -q \
      tests/test_v7_action_loss_contract.py \
      tests/test_v7_native_action_loss.py \
      tests/test_v7_action_policy_transition.py \
      tests/test_v7_distributed_transport.py \
      tests/test_v7_compact_dataset.py \
      tests/test_v7_compact_sharding.py \
      tests/test_v7_data_contracts.py \
      tests/test_v7_stage1_planner.py \
      tests/test_v7_stage1_planner_contract.py
    ;;
  stage0-static)
    "${PYTHON_BIN}" scripts/preflight_wm3d_v7_1b_actionpolicy_joint.py \
      --config configs/wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3.yaml \
      --mode static
    ;;
  stage1-static)
    "${PYTHON_BIN}" scripts/preflight_wm3d_v7_stage1_planner.py \
      --cfg "${stage1_cfg}" --mode static
    ;;
  stage1-data)
    "${PYTHON_BIN}" scripts/preflight_wm3d_v7_stage1_planner.py \
      --cfg "${stage1_cfg}" --mode data
    ;;
  *)
    usage
    exit 2
    ;;
esac
