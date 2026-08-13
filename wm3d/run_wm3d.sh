#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv/bin/python}
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
WM3D 从零入口（1B/5B 与数据源均由 profile 决定）：

  ./run_wm3d.sh env
  ./run_wm3d.sh lock-resolve <resolve_source_lock.py 参数...>
  ./run_wm3d.sh download <download_sources.py 参数...>
  ./run_wm3d.sh archive-collection <materialize_archive_collection.py 参数...>
  ./run_wm3d.sh external-convert <run_external_converter.py 参数...>
  ./run_wm3d.sh beta-task-list <list_agibot_beta_tasks.py 参数...>
  ./run_wm3d.sh schema-audit <inspect_source_schema.py 参数...>
  ./run_wm3d.sh adapter-audit <audit_adapter_contract.py 参数...>
  ./run_wm3d.sh inventory <materialize_source_inventory.py 参数...>
  ./run_wm3d.sh legacy-residual-import <materialize_legacy_residual_inventory.py 参数...>
  ./run_wm3d.sh collection-inventory <materialize_collection_inventory.py 参数...>
  ./run_wm3d.sh data-profile <materialize_data_profile.py 参数...>
  ./run_wm3d.sh task-bank <build_task_embeddings.py 参数...>
  ./run_wm3d.sh cache-plan <plan_cache_tasks.py 参数...>
  ./run_wm3d.sh cache-worker <run_cache_worker.py 参数...>
  ./run_wm3d.sh cache-seal <seal_episode_cache.py 参数...>
  ./run_wm3d.sh window <materialize_window_index.py 参数...>
  ./run_wm3d.sh normalization <build_grouped_normalization.py 参数...>
  ./run_wm3d.sh runtime <materialize_runtime.py 参数...>
  ./run_wm3d.sh preflight <torchrun 参数...> -- --runtime <runtime.yaml>
  ./run_wm3d.sh train <torchrun 参数...> -- <训练参数...>
  ./run_wm3d.sh eval <torchrun 参数...> -- <评测参数...>
  ./run_wm3d.sh smoke-real --work-root <空目录> --operator <姓名或工号> \
    --accept-dataset-license --confirm-adapter-semantics [--gpus 0,1]
  ./run_wm3d.sh 5b <init|doctor|download|cache|train|status|verify...> [参数...]
  ./run_wm3d.sh stage1-seal-selection <正式四 root selection seal 参数...>
  ./run_wm3d.sh stage1-replay-authority <真实 simulator replay 参数...>
  ./run_wm3d.sh stage1-audit-rollouts <audit_robocasa_real_rollouts.py 参数...>
  ./run_wm3d.sh stage1-produce <produce_robocasa_stage1_candidates.py 参数...>
  ./run_wm3d.sh stage1-materialize <materialize_stage1_branches.py 参数...>
  ./run_wm3d.sh stage1-train <torchrun 参数...> -- --runtime <stage1.yaml> [...]
  ./run_wm3d.sh stage1-eval <torchrun 参数...> -- --runtime <stage1.yaml> [...]
  ./run_wm3d.sh check

`preflight`/`train`/`eval`/`stage1-train`/`stage1-eval` 把 `--` 左侧原样交给
torchrun，右侧交给对应统一入口。Stage1 只读取同一份 sealed Stage0 runtime、
committed DCP 与真实 simulator branch receipt。
未知 revision、字段、单位、坐标系、夹爪语义或时钟都会 fail closed。
EOF
}

need_python() {
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "找不到 ${PYTHON_BIN}；先运行 ./run_wm3d.sh env 或设置 PYTHON_BIN。" >&2
    exit 2
  }
}

run_py() {
  need_python
  "${PYTHON_BIN}" "$@"
}

torchrun_split() {
  local module=$1
  shift
  local torch_args=()
  local app_args=()
  local seen=0
  for argument in "$@"; do
    if [[ "${argument}" == -- && ${seen} -eq 0 ]]; then
      seen=1
    elif [[ ${seen} -eq 0 ]]; then
      torch_args+=("${argument}")
    else
      app_args+=("${argument}")
    fi
  done
  [[ ${seen} -eq 1 && ${#app_args[@]} -gt 0 ]] || {
    echo "${module}: 必须用 -- 分隔 torchrun 和应用参数。" >&2
    exit 2
  }
  need_python
  "$(dirname "${PYTHON_BIN}")/torchrun" "${torch_args[@]}" -m "${module}" "${app_args[@]}"
}

torchrun_preflight_split() {
  local torch_args=()
  local app_args=()
  local seen=0
  for argument in "$@"; do
    if [[ "${argument}" == -- && ${seen} -eq 0 ]]; then
      seen=1
    elif [[ ${seen} -eq 0 ]]; then
      torch_args+=("${argument}")
    else
      [[ "${argument}" != --preflight-only ]] || {
        echo "preflight: --preflight-only 由统一入口自动追加，禁止重复传入。" >&2
        exit 2
      }
      app_args+=("${argument}")
    fi
  done
  [[ ${seen} -eq 1 && ${#app_args[@]} -gt 0 ]] || {
    echo "preflight: 必须用 -- 分隔 torchrun 和应用参数。" >&2
    exit 2
  }
  need_python
  "$(dirname "${PYTHON_BIN}")/torchrun" "${torch_args[@]}" \
    -m wm3d.training.pretrain "${app_args[@]}" --preflight-only
}

command=${1:-}
[[ -n "${command}" ]] || { usage; exit 2; }
shift || true
need_python_unless_env() {
  [[ "${command}" == env || "${command}" == smoke-real || "${command}" == 5b || \
     "${command}" == -h || "${command}" == --help || "${command}" == help ]] || need_python
}
need_python_unless_env

case "${command}" in
  env)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    exec bash environments/bootstrap_environment.sh
    ;;
  lock-resolve) exec "${PYTHON_BIN}" scripts/data/resolve_source_lock.py "$@" ;;
  download) exec "${PYTHON_BIN}" scripts/data/download_sources.py "$@" ;;
  archive-collection) exec "${PYTHON_BIN}" scripts/data/materialize_archive_collection.py "$@" ;;
  external-convert) exec "${PYTHON_BIN}" scripts/data/run_external_converter.py "$@" ;;
  beta-task-list) exec "${PYTHON_BIN}" scripts/data/list_agibot_beta_tasks.py "$@" ;;
  schema-audit) exec "${PYTHON_BIN}" scripts/data/inspect_source_schema.py "$@" ;;
  adapter-audit) exec "${PYTHON_BIN}" scripts/data/audit_adapter_contract.py "$@" ;;
  inventory) exec "${PYTHON_BIN}" scripts/data/materialize_source_inventory.py "$@" ;;
  legacy-residual-import) exec "${PYTHON_BIN}" scripts/data/materialize_legacy_residual_inventory.py "$@" ;;
  collection-inventory) exec "${PYTHON_BIN}" scripts/data/materialize_collection_inventory.py "$@" ;;
  data-profile) exec "${PYTHON_BIN}" scripts/data/materialize_data_profile.py "$@" ;;
  task-bank) exec "${PYTHON_BIN}" scripts/data/build_task_embeddings.py "$@" ;;
  cache-plan) exec "${PYTHON_BIN}" scripts/data/plan_cache_tasks.py "$@" ;;
  cache-worker) exec "${PYTHON_BIN}" scripts/data/run_cache_worker.py "$@" ;;
  cache-seal) exec "${PYTHON_BIN}" scripts/data/seal_episode_cache.py "$@" ;;
  window) exec "${PYTHON_BIN}" scripts/data/materialize_window_index.py "$@" ;;
  normalization) exec "${PYTHON_BIN}" scripts/data/build_grouped_normalization.py "$@" ;;
  runtime) exec "${PYTHON_BIN}" scripts/materialize_runtime.py "$@" ;;
  preflight) torchrun_preflight_split "$@" ;;
  train) torchrun_split wm3d.training.pretrain "$@" ;;
  eval) torchrun_split wm3d.training.offline_eval "$@" ;;
  smoke-real)
    exec "${SYSTEM_PYTHON:-python3.10}" scripts/run_real_smoke.py "$@"
    ;;
  5b) exec bash scripts/cluster/wm3d_5b.sh "$@" ;;
  stage1-seal-selection) exec "${PYTHON_BIN}" scripts/data/seal_robocasa_stage1_selection.py "$@" ;;
  stage1-replay-authority) exec "${PYTHON_BIN}" scripts/data/replay_robocasa_stage1_authority.py "$@" ;;
  stage1-audit-rollouts) exec "${PYTHON_BIN}" scripts/data/audit_robocasa_real_rollouts.py "$@" ;;
  stage1-produce) exec "${PYTHON_BIN}" scripts/produce_robocasa_stage1_candidates.py "$@" ;;
  stage1-materialize) exec "${PYTHON_BIN}" scripts/materialize_stage1_branches.py "$@" ;;
  stage1-train) torchrun_split wm3d.stage1_planner.train "$@" ;;
  stage1-eval) torchrun_split scripts.eval_stage1 "$@" ;;
  check)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    run_py - <<'PY'
from pathlib import Path
files = sorted(path for root in (Path("wm3d"), Path("scripts"), Path("tests")) for path in root.rglob("*.py"))
for path in files:
    compile(path.read_bytes(), str(path), "exec")
print(f"compiled {len(files)} Python files")
PY
    bash -n run_wm3d.sh environments/bootstrap_environment.sh
    run_py -m pytest -q -p no:cacheprovider tests
    ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
