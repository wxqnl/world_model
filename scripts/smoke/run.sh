#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ $# -ne 1 || "$1" != /* ]]; then
  echo "用法：$0 /abs/work-root" >&2
  exit 2
fi

WORK_INPUT="$1"
if [[ -L "${WORK_INPUT}" ]]; then
  echo "work-root 不能是符号链接：${WORK_INPUT}" >&2
  exit 2
fi
mkdir -p -- "${WORK_INPUT}"
WORK_ROOT="$(realpath -e -- "${WORK_INPUT}")"
if [[ ! -d "${WORK_ROOT}" || -L "${WORK_ROOT}" ]]; then
  echo "work-root 必须是真实目录：${WORK_ROOT}" >&2
  exit 2
fi

DEVICES="${WM3D_SMOKE_CUDA_DEVICES:-0,1}"
EXPECTED_IP="${WM3D_SMOKE_EXPECTED_IP:-172.27.0.5}"
if [[ "${DEVICES}" != "0,1" ]]; then
  echo "本 smoke 固定使用 node42 GPU0–1，当前为 ${DEVICES}" >&2
  exit 2
fi
WORLD_SIZE=2
VGGT_SOURCE_COMMIT=a288dd0f14786c93483e45524328726ab7b1b4ce
VGGT_SOURCE_ARCHIVE_SHA256=df4e7de1184bcb28ad6b4a83ead828f34ba42fb18be03c034801ffeb3a058f91
VGGT_SOURCE_TREE_SHA256=afc51bd052a538736830c33651f8f59f087629754298dd95d5d97a1a8bf99fa1
VGGT_REVISION=860abec7937da0a4c03c41d3c269c366e82abdf9
TASK_REVISION=52e6cc877548ebd0de720a7fe86177f8a5593a673f40162aa9006a3877fa97c1

export REPO_ROOT="${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHON_BIN="${WORK_ROOT}/venv/bin/python"
export TRAIN_ENV_RECEIPT="${WORK_ROOT}/venv/environment_receipt.json"
export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf_home}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORK_ROOT}/pip_cache}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONDONTWRITEBYTECODE=1
mkdir -p -- "${HF_HOME}" "${PIP_CACHE_DIR}" "${WORK_ROOT}/logs" \
  "${WORK_ROOT}/release" "${WORK_ROOT}/raw" "${WORK_ROOT}/build" \
  "${WORK_ROOT}/receipts"

RUN_LOG="${WORK_ROOT}/logs/smoke.log"
RUN_STATUS="${WORK_ROOT}/smoke_status.json"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CURRENT_STAGE="environment"

publish_status() {
  rc=$?
  trap - EXIT
  set +e
  state="failed"
  if [[ "${rc}" -eq 0 ]]; then
    state="passed"
  fi
  ended_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  status_tmp="${RUN_STATUS}.tmp.$$"
  printf '%s\n' \
    "{\"schema\":\"wm3d_v7_smoke_status_v1\",\"state\":\"${state}\",\"exit_code\":${rc},\"stage\":\"${CURRENT_STAGE}\",\"attempt_id\":\"${ATTEMPT_ID}\",\"started_at_utc\":\"${STARTED_AT_UTC}\",\"ended_at_utc\":\"${ended_at_utc}\",\"log\":\"logs/smoke.log\"}" \
    >"${status_tmp}"
  mv -f -- "${status_tmp}" "${RUN_STATUS}"
  printf 'smoke %s：stage=%s exit=%s；日志=%s；状态=%s\n' \
    "${state}" "${CURRENT_STAGE}" "${rc}" "${RUN_LOG}" "${RUN_STATUS}"
  exit "${rc}"
}

trap publish_status EXIT
exec > >(tee -a "${RUN_LOG}") 2>&1
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
printf '\n=== WM3D smoke attempt %s started %s ===\n' \
  "${ATTEMPT_ID}" "${STARTED_AT_UTC}"
echo "Hugging Face endpoint: ${HF_ENDPOINT}"

echo "[1/10] 创建并核验固定 Python 环境"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3.10}" \
  "${ROOT}/environments/bootstrap_environment.sh"
PY="${PYTHON_BIN}"
TORCHRUN="${WORK_ROOT}/venv/bin/torchrun"

CURRENT_STAGE="download_aloha"
echo "[2/10] 下载固定 revision 的 91 MB ALOHA 公开样本"
if ! "${PY}" "${ROOT}/scripts/data/download_raw_snapshots.py" \
  --lock "${ROOT}/configs/smoke/aloha_sources.lock.yaml" \
  --raw-root "${WORK_ROOT}/raw" --source aloha_smoke --resume; then
  echo "ALOHA 公开样本下载失败。若 huggingface.co 不可达，请原地重试：" >&2
  echo "  HF_ENDPOINT=https://hf-mirror.com ./wm3d.sh smoke ${WORK_ROOT}" >&2
  exit 1
fi
export ALOHA_SMOKE_ROOT="${WORK_ROOT}/raw/aloha_smoke"

DATASET_ROOT="${WORK_ROOT}/dataset"
CONTRACT_BUILD="${WORK_ROOT}/build/dataset_contract.json"
CURRENT_STAGE="prepare_dataset"
if [[ ! -f "${DATASET_ROOT}/receipts/source_scan.json" ]]; then
  echo "[3/10] 编译数据契约并扫描 train/val episode"
  if [[ ! -f "${CONTRACT_BUILD}" ]]; then
    "${PY}" "${ROOT}/scripts/data/compile_dataset_contract.py" \
      --inventory "${ROOT}/configs/smoke/aloha_dataset.yaml" \
      --output "${CONTRACT_BUILD}"
  fi
  "${PY}" "${ROOT}/scripts/data/scan_sources.py" \
    --dataset-contract "${CONTRACT_BUILD}" \
    --source-layouts "${ROOT}/configs/smoke/aloha_layouts.json" \
    --output-root "${DATASET_ROOT}"
fi

CURRENT_STAGE="action_statistics"
if [[ ! -f "${DATASET_ROOT}/control/action_stats.json" ]]; then
  echo "[4/10] 统计 grouped action"
  PARTIAL="${WORK_ROOT}/build/action_stats_00000.npz"
  if [[ ! -f "${PARTIAL}" ]]; then
    "${PY}" "${ROOT}/scripts/data/build_action_stats.py" partial \
      --episode-plan "${DATASET_ROOT}/control/episode_plan.jsonl" \
      --output "${PARTIAL}" --shard-id 0 --num-shards 1 \
      --global-sample-budget 25000
  fi
  "${PY}" "${ROOT}/scripts/data/build_action_stats.py" merge \
    --partials "${PARTIAL}" \
    --output "${DATASET_ROOT}/control/action_stats.json"
fi

CURRENT_STAGE="encoder_assets"
echo "[5/10] 准备固定 VGGT 资产和 smoke task bank"
VGGT_SOURCE="${WORK_ROOT}/assets_source/vggt"
"${PY}" "${ROOT}/scripts/assets/materialize_vggt_source.py" \
  --output-root "${VGGT_SOURCE}" \
  --archive-root "${WORK_ROOT}/assets_source/archives" \
  --commit "${VGGT_SOURCE_COMMIT}" \
  --archive-sha256 "${VGGT_SOURCE_ARCHIVE_SHA256}" \
  --tree-sha256 "${VGGT_SOURCE_TREE_SHA256}"

if [[ -n "${VGGT_MODEL_SNAPSHOT:-}" ]]; then
  VGGT_SNAPSHOT="$(realpath -e -- "${VGGT_MODEL_SNAPSHOT}")"
else
  VGGT_SNAPSHOT="$("${PY}" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="facebook/VGGT-1B",
    revision="${VGGT_REVISION}",
    cache_dir="${HF_HOME}",
))
PY
)"
fi
if [[ "$(basename "${VGGT_SNAPSHOT}")" != "${VGGT_REVISION}" ]]; then
  echo "VGGT model snapshot revision 漂移：${VGGT_SNAPSHOT}" >&2
  exit 2
fi
TASK_SNAPSHOT="${WORK_ROOT}/assets_source/task/${TASK_REVISION}"
"${PY}" - "${TASK_SNAPSHOT}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
path = root / "SMOKE_ONLY.txt"
payload = "deterministic SHA-256 task vectors; not used by formal training\n"
if path.exists() and path.read_text(encoding="utf-8") != payload:
    raise SystemExit(f"task marker drift: {path}")
if not path.exists():
    path.write_text(payload, encoding="utf-8")
PY
ASSET_ROOT="${WORK_ROOT}/encoder_assets_nopyc_v1"
if [[ ! -f "${ASSET_ROOT}/receipt.json" ]]; then
  "${PY}" "${ROOT}/scripts/assets/seal_encoder_assets.py" \
    --vggt-source-root "${VGGT_SOURCE}" \
    --vggt-source-commit "${VGGT_SOURCE_COMMIT}" \
    --vggt-source-archive-sha256 "${VGGT_SOURCE_ARCHIVE_SHA256}" \
    --vggt-source-tree-sha256 "${VGGT_SOURCE_TREE_SHA256}" \
    --vggt-model facebook/VGGT-1B \
    --vggt-snapshot "${VGGT_SNAPSHOT}" --vggt-revision "${VGGT_REVISION}" \
    --task-model wm3d/smoke-hash-2048 \
    --task-snapshot "${TASK_SNAPSHOT}" --task-revision "${TASK_REVISION}" \
    --output-root "${ASSET_ROOT}"
else
  "${PY}" "${ROOT}/scripts/assets/verify_encoder_assets.py" \
    --asset-root "${ASSET_ROOT}"
fi
if [[ ! -f "${DATASET_ROOT}/control/task_index.json" ]]; then
  "${PY}" "${ROOT}/scripts/data/build_task_bank.py" \
    --backend smoke-hash \
    --episode-plan "${DATASET_ROOT}/control/episode_plan.jsonl" \
    --output-root "${DATASET_ROOT}" --asset-root "${ASSET_ROOT}" \
    --confirmation EXECUTE_V7_PUBLIC_SMOKE_HASH_TASK_BANK
fi

resource_guard() {
  "${PY}" "${ROOT}/scripts/smoke/verify_resources.py" \
    --devices "${DEVICES}" --work-root "${WORK_ROOT}" \
    --expected-ip "${EXPECTED_IP}" --minimum-free-bytes 50000000000
}

CURRENT_STAGE="vggt_cache"
if [[ ! -f "${DATASET_ROOT}/receipts/dataset_seal.json" ]]; then
  echo "[6/10] GPU0–1 编码真实 RGB/action 并发布 VGGT cache"
  resource_guard
  pids=()
  for shard in 0 1; do
    gpu="${shard}"
    log="${WORK_ROOT}/logs/encode_${shard}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
      "${ROOT}/scripts/data/cache_vggt_shard.py" \
      --dataset-contract "${DATASET_ROOT}/control/dataset_contract.json" \
      --episode-plan "${DATASET_ROOT}/control/episode_plan.jsonl" \
      --action-stats "${DATASET_ROOT}/control/action_stats.json" \
      --task-index "${DATASET_ROOT}/control/task_index.json" \
      --output-root "${DATASET_ROOT}" --asset-root "${ASSET_ROOT}" \
      --shard-id "${shard}" --num-shards 2 --max-part-frames 128 \
      --window-stride 8 --encoder-batch-frames 1 \
      --vggt-revision "${VGGT_REVISION}" --device cuda \
      >"${log}" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "encoder shard 失败；证据保留在 ${WORK_ROOT}/logs" >&2
    exit 1
  fi
  "${PY}" "${ROOT}/scripts/data/seal_dataset.py" \
    --dataset-root "${DATASET_ROOT}" --num-encoder-shards 2
fi
"${PY}" "${ROOT}/scripts/data/verify_dataset.py" \
  --dataset-root "${DATASET_ROOT}" --mode deep --sample-windows-per-source 2

CURRENT_STAGE="release_config"
echo "[7/10] 固化代码、环境和训练配置"
CODE_RECEIPT="${WORK_ROOT}/receipts/code.json"
if [[ ! -f "${CODE_RECEIPT}" ]]; then
  "${PY}" "${ROOT}/scripts/cluster/seal_code.py" \
    --repo-root "${ROOT}" --output "${CODE_RECEIPT}"
fi
CONFIG="${WORK_ROOT}/release/train_smoke2.yaml"
TRAIN_ROOT="${WORK_ROOT}/train"
if [[ ! -f "${CONFIG}" ]]; then
  RUN_LINEAGE="$("${PY}" - "${DATASET_ROOT}/receipts/dataset_seal.json" "${CODE_RECEIPT}" <<'PY'
import hashlib
import sys
from pathlib import Path
value = b"wm3d_v7_public_smoke_v1"
for name in sys.argv[1:]:
    value += hashlib.sha256(Path(name).read_bytes()).digest()
print(hashlib.sha256(value).hexdigest())
PY
)"
  "${PY}" "${ROOT}/scripts/cluster/materialize_config.py" \
    --template "${ROOT}/configs/train/5b_smoke.yaml" \
    --dataset-root "${DATASET_ROOT}" --code-receipt "${CODE_RECEIPT}" \
    --code-root "${ROOT}" \
    --environment-contract "${ROOT}/environments/environment_contract.json" \
    --environment-receipt "${TRAIN_ENV_RECEIPT}" \
    --output-root "${TRAIN_ROOT}" --output-config "${CONFIG}" \
    --run-name wm3d_v7_public_aloha_smoke2 \
    --run-lineage "${RUN_LINEAGE}" --world-size 2 --shard-degree 2 \
    --global-batch-size 2 --micro-batch-size 1 \
    --smoke-confirmation EXECUTE_WM3D_PUBLIC_SMOKE
fi

CHECKPOINT="${TRAIN_ROOT}/checkpoints/step_00000001"
CURRENT_STAGE="train"
if [[ ! -f "${CHECKPOINT}/COMMITTED.json" ]]; then
  echo "[8/10] GPU0–1 运行真实约 4.96B 模型的一步 FSDP2 训练"
  resource_guard
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  CUDA_VISIBLE_DEVICES="${DEVICES}" "${TORCHRUN}" --standalone \
    --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    -m wm3d.training.train --config "${CONFIG}" \
    2>&1 | tee "${WORK_ROOT}/logs/train.log"
fi

EVAL_ROOT="${WORK_ROOT}/eval"
CURRENT_STAGE="eval"
if [[ ! -f "${EVAL_ROOT}/report.json" ]]; then
  echo "[9/10] 从显式 step_00000001 checkpoint 运行一步 eval"
  resource_guard
  CUDA_VISIBLE_DEVICES="${DEVICES}" "${TORCHRUN}" --standalone \
    --nnodes=1 --nproc-per-node="${WORLD_SIZE}" \
    -m wm3d.training.eval --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" --output-root "${EVAL_ROOT}" --steps 1 \
    2>&1 | tee "${WORK_ROOT}/logs/eval.log"
fi

CURRENT_STAGE="report"
echo "[10/10] 发布全流程证据报告"
REPORT="${WORK_ROOT}/smoke_report.json"
if [[ ! -f "${REPORT}" ]]; then
  "${PY}" "${ROOT}/scripts/smoke/report.py" \
    --work-root "${WORK_ROOT}" --dataset-root "${DATASET_ROOT}" \
    --train-config "${CONFIG}" --train-root "${TRAIN_ROOT}" \
    --eval-root "${EVAL_ROOT}" \
    --raw-receipt "${ALOHA_SMOKE_ROOT}/.wm3d_v7_download_receipt.json" \
    --code-receipt "${CODE_RECEIPT}" \
    --environment-receipt "${TRAIN_ENV_RECEIPT}" --output "${REPORT}"
fi
"${PY}" - "${REPORT}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("pass") is not True:
    raise SystemExit("smoke report did not pass")
print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
PY
CURRENT_STAGE="complete"
