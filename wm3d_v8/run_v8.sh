#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
用法：
  ./run_v8.sh check
  ./run_v8.sh static [config]
  ./run_v8.sh full <sealed-runtime-config> <report.json>
  ./run_v8.sh transition <step_XXXXXXXX.pt> <sealed-stage0-config> <report.json>

该入口负责代码自检、preflight 和 Stage0→下游继承审计，不自动启动或跨 milestone 晋级训练。
EOF
}

DEFAULT_CONFIG=configs/wm3d_v8_stage0_causal_dual_view_unified_action_formal100k_world16_node43_node44_v3.yaml

case "${1:-}" in
  check)
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

roots = (Path("wm3d_v3"), Path("scripts"), Path("tests"))
files = sorted(path for root in roots for path in root.rglob("*.py"))
for path in files:
    compile(path.read_bytes(), str(path), "exec")
print(f"compiled {len(files)} Python files")
PY
    bash -n run_v8.sh
    "${PYTHON_BIN}" -m pytest -q -p no:cacheprovider tests
    ;;
  static)
    config=${2:-${DEFAULT_CONFIG}}
    "${PYTHON_BIN}" scripts/preflight_wm3d_v8_stage0_causal_dual_view.py       --config "${config}" --mode static
    ;;
  full)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    "${PYTHON_BIN}" scripts/preflight_wm3d_v8_stage0_causal_dual_view.py       --config "$2" --mode full --json-out "$3"
    ;;
  transition)
    [[ $# -eq 4 ]] || { usage; exit 2; }
    "${PYTHON_BIN}" scripts/audit_wm3d_v8_stage0_libero_transition.py       --checkpoint "$2" --expected-config "$3" --report "$4"
    ;;
  *)
    usage
    exit 2
    ;;
esac
