#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENTRY="${ROOT}/run_wm3d.sh"
SCALE=${WM3D_CLUSTER_SCALE:-5b}
case "${SCALE}" in
  1b)
    SCALE_LABEL=1B
    PRESET_VAR=WM3D_1B_PRESET
    RUN_ID_VAR=WM3D_1B_RUN_ID
    TEMPLATE="${ROOT}/configs/cluster/h100_1b_direct.env.example"
    GUIDE=docs/WM3D_DIRECT_RAW.md
    ;;
  5b)
    SCALE_LABEL=5B
    PRESET_VAR=WM3D_5B_PRESET
    RUN_ID_VAR=WM3D_5B_RUN_ID
    TEMPLATE="${ROOT}/configs/cluster/h200_5b_direct.env.example"
    GUIDE=docs/WM3D_5B_SCALING.md
    ;;
  *)
    echo "WM3D_CLUSTER_SCALE 必须是 1b 或 5b" >&2
    exit 2
    ;;
esac

usage() {
  cat <<EOF
WM3D ${SCALE_LABEL} 集群操作入口：

  ./run_wm3d.sh ${SCALE} configure <model-root> <data-root> [site.env]
  ./run_wm3d.sh ${SCALE} init <preset> <site.env> [direct_raw|streaming_raw|episode_cache]
  ./run_wm3d.sh ${SCALE} env <site.env>
  ./run_wm3d.sh ${SCALE} data-template <site.env>
  ./run_wm3d.sh ${SCALE} doctor <site.env>
  ./run_wm3d.sh ${SCALE} plan <site.env>
  ./run_wm3d.sh ${SCALE} lock <site.env>
  ./run_wm3d.sh ${SCALE} download <site.env>
  ./run_wm3d.sh ${SCALE} task-bank <site.env>
  ./run_wm3d.sh ${SCALE} cache-plan <site.env>
  ./run_wm3d.sh ${SCALE} cache-worker <site.env> <global-worker-index> <worker-count> <local-gpu|inherited>
  ./run_wm3d.sh ${SCALE} cache-seal <site.env>
  ./run_wm3d.sh ${SCALE} streaming-prepare <site.env>
  ./run_wm3d.sh ${SCALE} window <site.env>
  ./run_wm3d.sh ${SCALE} normalization <site.env>
  ./run_wm3d.sh ${SCALE} runtime <site.env>
  ./run_wm3d.sh ${SCALE} preflight <site.env>
  ./run_wm3d.sh ${SCALE} train <site.env> [sealed-stop-step]
  ./run_wm3d.sh ${SCALE} resume <site.env> <step|checkpoint> [sealed-stop-step]
  ./run_wm3d.sh ${SCALE} eval <site.env> [step|checkpoint] [output.json]
  ./run_wm3d.sh ${SCALE} slurm <site.env> <preflight|train|resume|eval> [参数...]
  ./run_wm3d.sh ${SCALE} status <site.env>
  ./run_wm3d.sh ${SCALE} verify <site.env> [step|checkpoint] [eval.json]

完整顺序见 ${GUIDE}。
EOF
}

die() {
  echo "${SCALE}: $*" >&2
  exit 2
}

absolute_existing_file() {
  local value=$1
  [[ "${value}" == /* ]] || die "site 文件必须是绝对路径：${value}"
  [[ -f "${value}" && ! -L "${value}" ]] || die "site 文件必须是普通文件且禁止符号链接：${value}"
}

repo_path() {
  local value=$1
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${ROOT}/${value}"
  fi
}

require_var() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "site 文件缺少 ${name}"
}

require_file() {
  local label=$1
  local value=$2
  [[ -f "${value}" && ! -L "${value}" ]] || die "${label} 不存在、不是普通文件或是符号链接：${value}"
}

validate_5b_v8_contract() {
  [[ "${SCALE}" == 5b ]] || return 0
  "${PYTHON_BIN}" - "${MODEL_PROFILE}" "${ENCODER_CONTRACT}" "${OBJECTIVE_PROFILE}" <<'PY'
from pathlib import Path
import sys

import torch
import yaml

from wm3d.models.model_factory import build_world_model, validate_model_profile

model = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
encoder = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
objective = yaml.safe_load(Path(sys.argv[3]).read_text(encoding="utf-8"))
validate_model_profile(model)
cfg = model["model"]
required = {
    "T": 24,
    "P": 144,
    "K": 16,
    "policy_task_modulation": True,
    "policy_calibration_conditioning": True,
    "dynamics_layers": 2,
    "factual_dynamics_repeats": 1,
    "factual_action_residual_scale": 1.0,
    "factual_v7_early_action_conditioning": False,
    "factual_v7_early_action_scale": 0.0,
    "factual_v7_bridge_layers_state": [],
    "appearance_enabled": False,
    "rgb_context_enabled": True,
    "rgb_original_v7_context": False,
    "rgb_action_owned_transport": True,
    "rgb_v7_high_frequency_refiner": True,
    "rgb_v7_high_frequency_scale": 0.0625,
    "rgb_context_alignment_enabled": False,
    "rgb_render_action_free_prior": False,
    "rgb_context_motion_blend_gain": 0.0,
    "rgb_context_action_scale": 1.0,
    "rgb_context_appearance_delta_scale": 0.0,
    "rgb_detail_residual_scale": 0.0,
}
wrong = {
    key: (cfg.get(key), expected)
    for key, expected in required.items()
    if cfg.get(key) != expected
}
if wrong:
    raise SystemExit(f"5B V8 model contract mismatch: {wrong}")
if encoder.get("token_grid") != 12 or encoder.get("target_rgb_size") != 384:
    raise SystemExit("5B V8 encoder must be P144/384px")
if "appearance_token_grid" in encoder or "appearance_feature_layer" in encoder:
    raise SystemExit("5B V8 production encoder must not extract absolute P256")
weights = objective.get("objective", {})
required_weights = {
    "rgb_l1": 1.2,
    "rgb_perceptual": 0.55,
    "rgb_gradient": 0.08,
    "rgb_charbonnier": 0.0,
    "rgb_motion_l1": 1.0,
    "rgb_flow_teacher": 0.20,
    "rgb_disocclusion_bce": 0.0,
    "rgb_disocclusion_dice": 0.0,
    "appearance_l1": 0.0,
    "appearance_teacher_l1": 0.0,
    "appearance_autoregressive_l1": 0.0,
    "action_counterfactual_token_advantage": 1.0,
    "action_counterfactual_token_margin": 0.005,
    "context_pixel_action_rank_weight": 2.0,
    "context_pixel_action_separation_weight": 0.5,
    "context_pixel_action_rank_start_step": 30000,
    "context_pixel_action_rank_ramp_steps": 10000,
    "context_pixel_action_rank_every": 8,
    "context_pixel_action_rank_batch_size": 1,
    "context_pixel_action_rank_margin": 0.003,
    "context_pixel_action_separation_margin": 0.006,
    "context_pixel_action_motion_threshold": 0.03,
    "context_pixel_action_motion_gain": 4.0,
    "context_pixel_action_negative_min_distance": 0.05,
}
wrong_weights = {
    key: (weights.get(key), expected)
    for key, expected in required_weights.items()
    if weights.get(key) != expected
}
if wrong_weights:
    raise SystemExit(f"5B V8 objective mismatch: {wrong_weights}")
with torch.device("meta"):
    built = build_world_model(model)
print(
    "5B V8 contract passed: "
    f"parameters={sum(parameter.numel() for parameter in built.parameters()):,}"
)
PY
}

validate_preset() {
  if [[ "${SCALE}" == 1b ]]; then
    case "$1" in
      canary1k|formal100k) ;;
      *) die "未知 1B preset：$1（可选 canary1k、formal100k）" ;;
    esac
  else
    case "$1" in
      canary1k|validation100k|formal600k) ;;
      *) die "未知 5B preset：$1（可选 canary1k、validation100k、formal600k）" ;;
    esac
  fi
}

apply_preset() {
  validate_preset "${WM3D_PRESET}"
  case "${SCALE}:${WM3D_PRESET}" in
    1b:canary1k)
      RUNTIME_PROFILE=configs/runtime/h100_8_fsdp2_streaming_canary1k.yaml
      TOTAL_STEPS=1000
      MILESTONES="100 -> 500 -> 1000"
      ;;
    1b:formal100k)
      RUNTIME_PROFILE=configs/runtime/h100_8_fsdp2_streaming_formal100k.yaml
      TOTAL_STEPS=100000
      MILESTONES="1000 -> 5000 -> 10000 -> 100000"
      ;;
    5b:canary1k)
      RUNTIME_PROFILE=configs/runtime/h200_64_fsdp2_canary1k.yaml
      TOTAL_STEPS=1000
      MILESTONES="100 -> 500 -> 1000"
      ;;
    5b:validation100k)
      RUNTIME_PROFILE=configs/runtime/h200_64_fsdp2_validation100k.yaml
      TOTAL_STEPS=100000
      MILESTONES="1000 -> 100000"
      ;;
    5b:formal600k)
      RUNTIME_PROFILE=configs/runtime/h200_64_fsdp2.yaml
      TOTAL_STEPS=600000
      MILESTONES="1000 -> 5000 -> 20000 -> 600000"
      ;;
  esac
  RUN_ROOT="${WORK_ROOT}/runs/${SCALE}_${WM3D_RUN_ID}"
  RUNTIME_YAML="${CONTROL_ROOT}/runtime_${SCALE}_${WM3D_RUN_ID}.yaml"
  RUN_NAME="wm3d_${SCALE}_${WM3D_RUN_ID}"
  RUN_LINEAGE="wm3d_${SCALE}_${DATA_FAMILY}_${WM3D_RUN_ID}_v1"
  printf -v FINAL_STEP_PADDED '%08d' "${TOTAL_STEPS}"
  EVAL_OUTPUT="${RUN_ROOT}/eval_step_${FINAL_STEP_PADDED}.json"
}

load_site() {
  local site=$1
  absolute_existing_file "${site}"
  # Site files are operator-owned shell configuration, as in the previous cluster handoff.
  # They must live outside Git and are never accepted as a data/model receipt.
  set -a
  # shellcheck disable=SC1090
  source "${site}"
  set +a
  require_var "${PRESET_VAR}"
  require_var "${RUN_ID_VAR}"
  WM3D_PRESET=${!PRESET_VAR}
  WM3D_RUN_ID=${!RUN_ID_VAR}
  WM3D_DATA_MODE=${WM3D_DATA_MODE:-episode_cache}
  STREAMING_METADATA_ROOT=${STREAMING_METADATA_ROOT:-${CONTROL_ROOT}/streaming_metadata}
  STREAMING_METADATA_SEAL=${STREAMING_METADATA_SEAL:-${STREAMING_METADATA_ROOT}/metadata_seal_${SCALE}.json}
  STREAMING_LRU_ROOT=${STREAMING_LRU_ROOT:-${WORK_ROOT}/streaming_lru}
  STREAMING_LRU_GIB_PER_RANK=${STREAMING_LRU_GIB_PER_RANK:-64}
  STREAMING_METADATA_WORKERS=${STREAMING_METADATA_WORKERS:-32}
  STREAMING_ENCODE_BATCH_FRAMES=${STREAMING_ENCODE_BATCH_FRAMES:-16}
  STREAMING_DECODE_WORKERS=${STREAMING_DECODE_WORKERS:-4}
  DIRECT_INPUT_RGB_SIZE=${DIRECT_INPUT_RGB_SIZE:-518}
  DIRECT_DECODE_WORKERS=${DIRECT_DECODE_WORKERS:-1}
  DIRECT_ROBOT_CACHE_EPISODES=${DIRECT_ROBOT_CACHE_EPISODES:-8}
  DIRECT_PREFETCH_WINDOWS=${DIRECT_PREFETCH_WINDOWS:-32}
  DIRECT_PREFETCH_WORKERS=${DIRECT_PREFETCH_WORKERS:-4}
  DIRECT_VIDEO_INDEX_CACHE_ASSETS=${DIRECT_VIDEO_INDEX_CACHE_ASSETS:-128}
  DIRECT_ENCODE_CHUNK_ROWS=${DIRECT_ENCODE_CHUNK_ROWS:-32}
  DIRECT_MINIMUM_CHUNK_ROWS=${DIRECT_MINIMUM_CHUNK_ROWS:-4}
  DIRECT_APPEARANCE_FEATURE_LAYER=${DIRECT_APPEARANCE_FEATURE_LAYER:--1}
  DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK=${DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK:-8}
  [[ "${DIRECT_PREFETCH_WINDOWS}" =~ ^[1-9][0-9]*$ ]] || \
    die "DIRECT_PREFETCH_WINDOWS 必须是正整数"
  [[ "${DIRECT_PREFETCH_WORKERS}" =~ ^[1-9][0-9]*$ ]] || \
    die "DIRECT_PREFETCH_WORKERS 必须是正整数"
  (( DIRECT_PREFETCH_WORKERS <= DIRECT_PREFETCH_WINDOWS )) || \
    die "DIRECT_PREFETCH_WORKERS 不能超过 DIRECT_PREFETCH_WINDOWS"
  [[ "${DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK}" =~ ^[0-9]+$ ]] ||     die "DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK 必须是非负整数"
  WM3D_DIRECT_PREFETCH_WORKERS=${DIRECT_PREFETCH_WORKERS}
  WM3D_DIRECT_PREPARED_ROW_CACHE_BYTES_PER_RANK=$((
    DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK * 1024 * 1024 * 1024
  ))
  MINIMUM_RAW_FILESYSTEM_BYTES=${MINIMUM_RAW_FILESYSTEM_BYTES:-0}
  INCLUDE_AGIBOT_2026=${INCLUDE_AGIBOT_2026:-YES}
  INCLUDE_AGIBOT_BETA=${INCLUDE_AGIBOT_BETA:-NO}
  case "${WM3D_DATA_MODE}" in
    episode_cache|streaming_raw|direct_raw) ;;
    *) die "WM3D_DATA_MODE 必须是 episode_cache、streaming_raw 或 direct_raw" ;;
  esac
  case "${INCLUDE_AGIBOT_BETA}" in
    YES|NO) ;;
    *) die "INCLUDE_AGIBOT_BETA 必须是 YES 或 NO" ;;
  esac
  case "${INCLUDE_AGIBOT_2026}" in
    YES|NO) ;;
    *) die "INCLUDE_AGIBOT_2026 必须是 YES 或 NO" ;;
  esac
  for name in DATA_FAMILY WORK_ROOT CONTROL_ROOT RAW_ROOT \
    CACHE_ROOT ENV_DIR PYTHON_BIN \
    ACCEPT_DATA_LICENSES SOURCE_TEMPLATE SOURCE_LOCK DATA_TEMPLATE DATA_PROFILE TASK_BANK_ROOT \
    TASK_BANK_INDEX TASK_MANIFEST EPISODE_INDEX EPISODE_SEAL WINDOW_INDEX WINDOW_SEAL \
    GROUPED_NORMALIZATION MODEL_PROFILE ENCODER_CONTRACT \
    TASK_ENCODER_CONTRACT OBJECTIVE_PROFILE NNODES \
    GPUS_PER_NODE MASTER_ADDR PREFLIGHT_PORT TRAIN_PORT EVAL_PORT; do
    require_var "${name}"
  done
  apply_preset
  SOURCE_TEMPLATE=$(repo_path "${SOURCE_TEMPLATE}")
  DATA_TEMPLATE=$(repo_path "${DATA_TEMPLATE}")
  MODEL_PROFILE=$(repo_path "${MODEL_PROFILE}")
  ENCODER_CONTRACT=$(repo_path "${ENCODER_CONTRACT}")
  TASK_ENCODER_CONTRACT=$(repo_path "${TASK_ENCODER_CONTRACT}")
  OBJECTIVE_PROFILE=$(repo_path "${OBJECTIVE_PROFILE}")
  RUNTIME_PROFILE=$(repo_path "${RUNTIME_PROFILE}")
  export PYTHON_BIN HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  export WM3D_VGGT_SOURCE_ROOT WM3D_VGGT_MODEL_SNAPSHOT QWEN3_VL_EMBEDDING_PATH
  export WM3D_DIRECT_PREFETCH_WORKERS
  export WM3D_DIRECT_PREPARED_ROW_CACHE_BYTES_PER_RANK
}

checkpoint_for_step() {
  local raw=$1
  [[ "${raw}" =~ ^[0-9]+$ ]] || die "step 必须是非负整数：${raw}"
  local step=$((10#${raw}))
  (( step > 0 && step <= TOTAL_STEPS )) || \
    die "step 必须位于 1..${TOTAL_STEPS}：${raw}"
  printf '%s/checkpoints/step_%08d\n' "${RUN_ROOT}" "${step}"
}

resolve_checkpoint() {
  local value=$1
  if [[ "${value}" =~ ^[0-9]+$ ]]; then
    checkpoint_for_step "${value}"
  else
    printf '%s\n' "${value}"
  fi
}

step_from_checkpoint() {
  local value=$1
  local name=${value##*/}
  [[ "${name}" =~ ^step_([0-9]{8})$ ]] || \
    die "checkpoint 目录名必须为 step_XXXXXXXX：${value}"
  printf '%d\n' "$((10#${BASH_REMATCH[1]}))"
}

eval_output_for_step() {
  local raw=$1
  local step=$((10#${raw}))
  printf '%s/eval_step_%08d.json\n' "${RUN_ROOT}" "${step}"
}

sha256() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys
path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
}

torch_args() {
  printf '%s\n' \
    "--nnodes=${NNODES}" \
    "--nproc_per_node=${GPUS_PER_NODE}" \
    "--node_rank=${NODE_RANK:-0}" \
    "--master_addr=${MASTER_ADDR}" \
    "--master_port=$1"
}

action=${1:-help}
shift || true

case "${action}" in
  configure)
    [[ "${SCALE}" == 5b ]] || die "configure 只用于 5B 交付"
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    configure_args=(
      "${ROOT}/scripts/cluster/configure_5b_inputs.py"
      --model-root "$1"
      --data-root "$2"
    )
    if [[ -n "${WM3D_WORK_ROOT:-}" ]]; then
      configure_args+=(--work-root "${WM3D_WORK_ROOT}")
    fi
    if [[ $# -eq 3 ]]; then
      configure_args+=(--site-output "$3")
    fi
    exec "${SYSTEM_PYTHON:-python3}" "${configure_args[@]}"
    ;;
  init)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    preset=$1
    target=$2
    data_mode=${3:-direct_raw}
    validate_preset "${preset}"
    case "${data_mode}" in
      direct_raw|streaming_raw|episode_cache) ;;
      *) die "未知数据访问方式：${data_mode}（可选 direct_raw、streaming_raw、episode_cache）" ;;
    esac
    [[ "${target}" == /* ]] || die "init 目标必须是绝对路径"
    [[ ! -e "${target}" && ! -L "${target}" ]] || die "拒绝覆盖已有 site 文件：${target}"
    mkdir -p "$(dirname "${target}")"
    install -m 600 "${TEMPLATE}" "${target}"
    sed -i "s/^${PRESET_VAR}=.*/${PRESET_VAR}=${preset}/" "${target}"
    sed -i "s/^WM3D_DATA_MODE=.*/WM3D_DATA_MODE=${data_mode}/" "${target}"
    echo "已创建 ${target}（preset=${preset}, data_mode=${data_mode}）；编辑站点路径后运行 doctor。"
    exit 0
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
esac

[[ $# -ge 1 ]] || { usage; exit 2; }
site=$1
shift
load_site "${site}"

case "${action}" in
  doctor)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    require_file "model profile" "${MODEL_PROFILE}"
    require_file "vision encoder contract" "${ENCODER_CONTRACT}"
    require_file "task encoder contract" "${TASK_ENCODER_CONTRACT}"
    require_file "objective profile" "${OBJECTIVE_PROFILE}"
    require_file "runtime profile" "${RUNTIME_PROFILE}"
    [[ -x "${PYTHON_BIN}" ]] || die "Python 环境不存在：${PYTHON_BIN}；先运行 ENV_DIR=... ./run_wm3d.sh env"
    validate_5b_v8_contract
    "${PYTHON_BIN}" -m pip check
    "${PYTHON_BIN}" - "${MODEL_PROFILE}" "${RUNTIME_PROFILE}" <<'PY'
from pathlib import Path
import sys
import yaml
from wm3d.models.model_factory import validate_model_profile
from wm3d.training.runtime_contract import validate_runtime_profile
model = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_model_profile(model)
runtime = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
validate_runtime_profile(runtime)
print(f"model={model['name']} expected_parameters={int(model['expected_parameter_count']):,}")
print(f"world_size={runtime['expected_world_size']} total_steps={runtime['train']['total_steps']}")
PY
    [[ "${MASTER_ADDR}" != REQUIRED_MASTER_ADDR ]] || die "MASTER_ADDR 仍是 REQUIRED_MASTER_ADDR"
    if (( MINIMUM_RAW_FILESYSTEM_BYTES > 0 )); then
      raw_probe=${RAW_ROOT}
      while [[ ! -e "${raw_probe}" && "${raw_probe}" != / ]]; do
        raw_probe=$(dirname "${raw_probe}")
      done
      [[ -d "${raw_probe}" && ! -L "${raw_probe}" ]] || \
        die "RAW_ROOT 的已有父目录无效或是符号链接：${raw_probe}"
      raw_filesystem_bytes=$(df -B1 --output=size "${raw_probe}" | tail -n 1 | tr -d ' ')
      (( raw_filesystem_bytes >= MINIMUM_RAW_FILESYSTEM_BYTES )) || \
        die "RAW_ROOT 文件系统容量不足：${raw_filesystem_bytes} < ${MINIMUM_RAW_FILESYSTEM_BYTES} bytes"
    fi
    [[ -d "${WM3D_VGGT_SOURCE_ROOT}" && ! -L "${WM3D_VGGT_SOURCE_ROOT}" ]] || \
      die "VGGT source root 缺失或是符号链接：${WM3D_VGGT_SOURCE_ROOT}"
    [[ -d "${WM3D_VGGT_MODEL_SNAPSHOT}" && ! -L "${WM3D_VGGT_MODEL_SNAPSHOT}" ]] || \
      die "VGGT model snapshot 缺失或是符号链接：${WM3D_VGGT_MODEL_SNAPSHOT}"
    [[ -d "${QWEN3_VL_EMBEDDING_PATH}" && ! -L "${QWEN3_VL_EMBEDDING_PATH}" ]] || \
      die "Qwen embedding snapshot 缺失或是符号链接：${QWEN3_VL_EMBEDDING_PATH}"
    if [[ -f "${DATA_PROFILE}" && ! -L "${DATA_PROFILE}" ]]; then
      "${PYTHON_BIN}" - "${DATA_PROFILE}" <<'PY'
from pathlib import Path
import sys
from wm3d.data.manifest_contract import load_data_profile
profile = load_data_profile(Path(sys.argv[1]))
print(f"data_profile={profile.name} sources={len(profile.sources)}")
PY
    else
      echo "data_profile=WAITING (${DATA_PROFILE})"
      echo "下载可以先做；cache/training 要等项目负责人交付已审计 data profile。"
    fi
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,ecc.errors.uncorrected.volatile.total --format=csv,noheader || true
    df -h "${WORK_ROOT}" 2>/dev/null || true
    ;;
  plan)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    if [[ "${WM3D_DATA_MODE}" == streaming_raw ]]; then
      data_steps="task-bank -> cache-plan -> streaming-prepare"
      data_detail="stream LRU ${STREAMING_LRU_ROOT} (${STREAMING_LRU_GIB_PER_RANK} GiB/rank)"
    elif [[ "${WM3D_DATA_MODE}" == direct_raw ]]; then
      data_steps="task-bank -> cache-plan -> streaming-prepare"
      data_detail="direct VGGT chunk=${DIRECT_ENCODE_CHUNK_ROWS}, batch-coalesced decode, prefetch workers=${DIRECT_PREFETCH_WORKERS}, row cache=${DIRECT_PREPARED_ROW_CACHE_GIB_PER_RANK} GiB/rank, no latent cache"
    else
      data_steps="task-bank -> cache-plan -> cache-worker[*] -> cache-seal -> window -> normalization"
      data_detail="episode cache ${CACHE_ROOT}"
    fi
    cat <<EOF
WM3D ${SCALE_LABEL} site plan
  preset:     ${WM3D_PRESET}
  work:       ${WORK_ROOT}
  raw:        ${RAW_ROOT}
  data:       ${DATA_PROFILE}
  cache:      ${CACHE_ROOT}
  data mode:  ${WM3D_DATA_MODE}
  mode detail: ${data_detail}
  runtime:    ${RUNTIME_YAML}
  run:        ${RUN_ROOT}
  topology:   ${NNODES} nodes x ${GPUS_PER_NODE} GPUs
  model:      ${MODEL_PROFILE}
  schedule:   ${RUNTIME_PROFILE}
  final step: ${TOTAL_STEPS}

Order:
  env -> data-template -> doctor -> lock -> download -> adapter/inventory approval
  -> ${data_steps} -> runtime -> preflight
  -> train/resume milestones: ${MILESTONES}
  -> eval -> verify
EOF
    ;;
  env)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    ENV_DIR="${ENV_DIR}" SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3.10}" "${ENTRY}" env
    ;;
  lock)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ -n "${HF_TOKEN_FILE:-}" ]] || die "lock 需要在 site 中设置 HF_TOKEN_FILE"
    require_file "Hugging Face token" "${HF_TOKEN_FILE}"
    require_file "source template" "${SOURCE_TEMPLATE}"
    token_mode=$(stat -c '%a' "${HF_TOKEN_FILE}")
    (( (8#${token_mode} & 8#077) == 0 )) || die "HF token 禁止 group/world 权限；推荐 chmod 600"
    [[ "${ACCEPT_DATA_LICENSES}" == YES ]] || \
      die "先接受全部上游数据许可，并在 site 文件设置 ACCEPT_DATA_LICENSES=YES"
    mkdir -p "${CONTROL_ROOT}"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${ENTRY}" lock-resolve \
      --template "${SOURCE_TEMPLATE}" --output "${SOURCE_LOCK}" \
      --token-file "${HF_TOKEN_FILE}" \
      --confirm-licenses YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES
    ;;
  data-template|oxe-replacement)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ "${DATA_FAMILY}" == public_robot_oxe ]] || \
      die "data-template 只用于 DATA_FAMILY=public_robot_oxe"
    mkdir -p "${CONTROL_ROOT}/adapters"
    template_args=(
      "${ROOT}/scripts/data/materialize_oxe_default.py"
      --base-source-template "${ROOT}/configs/sources/public_sources.template.yaml"
      --base-data-template "${ROOT}/configs/data/public_robot_6106h.template.yaml"
      --output-source-template "${SOURCE_TEMPLATE}"
      --output-data-template "${DATA_TEMPLATE}"
      --output-adapter-root "${CONTROL_ROOT}/adapters"
    )
    if [[ "${SCALE}" == 1b ]]; then
      template_args+=(
        --model-profile "${MODEL_PROFILE}"
        --encoder-contract "${ENCODER_CONTRACT}"
        --profile-name public_robot_1b_oxe
        --profile-role default_1b_public_profile
      )
    fi
    if [[ "${INCLUDE_AGIBOT_BETA}" == YES ]]; then
      template_args+=(--include-agibot-beta)
    fi
    if [[ "${INCLUDE_AGIBOT_2026}" == NO ]]; then
      template_args+=(--exclude-agibot-2026)
    fi
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${PYTHON_BIN}" "${template_args[@]}"
    ;;
  download)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ -n "${HF_TOKEN_FILE:-}" ]] || die "download 需要在 site 中设置 HF_TOKEN_FILE"
    require_file "Hugging Face token" "${HF_TOKEN_FILE}"
    require_file "source lock" "${SOURCE_LOCK}"
    mkdir -p "${RAW_ROOT}"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "${ENTRY}" download \
      --lock "${SOURCE_LOCK}" --raw-root "${RAW_ROOT}" \
      --token-file "${HF_TOKEN_FILE}" --max-workers "${DOWNLOAD_WORKERS:-32}"
    ;;
  task-bank)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    require_file "data profile" "${DATA_PROFILE}"
    mkdir -p "${TASK_BANK_ROOT}"
    CUDA_VISIBLE_DEVICES="${TASK_GPU:-0}" "${ENTRY}" task-bank \
      --data-profile "${DATA_PROFILE}" --encoder-contract "${TASK_ENCODER_CONTRACT}" \
      --output-root "${TASK_BANK_ROOT}" --device "${TASK_DEVICE:-cuda:0}"
    ;;
  cache-plan)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    require_file "data profile" "${DATA_PROFILE}"
    require_file "task bank index" "${TASK_BANK_INDEX}"
    "${ENTRY}" cache-plan --data-profile "${DATA_PROFILE}" \
      --encoder-contract "${ENCODER_CONTRACT}" \
      --task-encoder-contract "${TASK_ENCODER_CONTRACT}" \
      --task-bank-index "${TASK_BANK_INDEX}" --output "${TASK_MANIFEST}"
    ;;
  cache-worker)
    [[ $# -eq 3 ]] || { usage; exit 2; }
    [[ "${WM3D_DATA_MODE}" == episode_cache ]] || \
      die "${WM3D_DATA_MODE} 不运行 cache-worker；请运行 streaming-prepare"
    worker_index=$1
    worker_count=$2
    local_gpu=$3
    require_file "cache task manifest" "${TASK_MANIFEST}"
    require_file "task bank index" "${TASK_BANK_INDEX}"
    task_bank_sha=$(sha256 "${TASK_BANK_INDEX}")
    worker_command=("${ENTRY}" cache-worker \
      --task-manifest "${TASK_MANIFEST}" --data-profile "${DATA_PROFILE}" \
      --encoder-contract "${ENCODER_CONTRACT}" --task-bank-root "${TASK_BANK_ROOT}" \
      --task-bank-index-sha256 "${task_bank_sha}" --cache-root "${CACHE_ROOT}" \
      --worker-index "${worker_index}" --worker-count "${worker_count}" \
      --device cuda:0 --batch-frames "${CACHE_BATCH_FRAMES:-16}" \
      --decode-workers "${CACHE_DECODE_WORKERS:-4}" \
      --writer-threads "${CACHE_WRITER_THREADS:-2}" --fail-fast)
    if [[ "${local_gpu}" == inherited ]]; then
      [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || die "inherited 模式要求调度器设置 CUDA_VISIBLE_DEVICES"
      "${worker_command[@]}"
    else
      [[ "${local_gpu}" =~ ^[0-9]+$ ]] || die "local-gpu 必须是整数或 inherited"
      CUDA_VISIBLE_DEVICES="${local_gpu}" "${worker_command[@]}"
    fi
    ;;
  cache-seal)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ "${WM3D_DATA_MODE}" == episode_cache ]] || \
      die "${WM3D_DATA_MODE} 不运行 cache-seal；请运行 streaming-prepare"
    "${ENTRY}" cache-seal --task-manifest "${TASK_MANIFEST}" \
      --receipt-root "${CACHE_ROOT}/receipts" \
      --episode-index-fragment-root "${CACHE_ROOT}/episode_index_fragments" \
      --output-index "${EPISODE_INDEX}" --output-seal "${EPISODE_SEAL}"
    ;;
  streaming-prepare)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ "${WM3D_DATA_MODE}" != episode_cache ]] || \
      die "streaming-prepare 只用于 streaming_raw 或 direct_raw"
    require_file "cache task manifest" "${TASK_MANIFEST}"
    require_file "task bank index" "${TASK_BANK_INDEX}"
    task_bank_sha=$(sha256 "${TASK_BANK_INDEX}")
    mkdir -p "${STREAMING_METADATA_ROOT}"
    "${ENTRY}" streaming-prepare \
      --task-manifest "${TASK_MANIFEST}" --data-profile "${DATA_PROFILE}" \
      --model-profile "${MODEL_PROFILE}" --encoder-contract "${ENCODER_CONTRACT}" \
      --task-bank-root "${TASK_BANK_ROOT}" \
      --task-bank-index-sha256 "${task_bank_sha}" \
      --output-root "${STREAMING_METADATA_ROOT}/payload" \
      --episode-index "${EPISODE_INDEX}" --window-index "${WINDOW_INDEX}" \
      --grouped-normalization "${GROUPED_NORMALIZATION}" \
      --output-seal "${STREAMING_METADATA_SEAL}" \
      --workers "${STREAMING_METADATA_WORKERS}"
    ;;
  window)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ "${WM3D_DATA_MODE}" == episode_cache ]] || \
      die "${WM3D_DATA_MODE} 的 window 已由 streaming-prepare 生成"
    "${ENTRY}" window --episode-index "${EPISODE_INDEX}" --episode-seal "${EPISODE_SEAL}" \
      --cache-root "${CACHE_ROOT}" --data-profile "${DATA_PROFILE}" \
      --model-profile "${MODEL_PROFILE}" --output-index "${WINDOW_INDEX}" \
      --output-seal "${WINDOW_SEAL}"
    ;;
  normalization)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    [[ "${WM3D_DATA_MODE}" == episode_cache ]] || \
      die "${WM3D_DATA_MODE} 的 normalization 已由 streaming-prepare 生成"
    window_sha=$(sha256 "${WINDOW_INDEX}")
    "${ENTRY}" normalization --data-profile "${DATA_PROFILE}" \
      --model-profile "${MODEL_PROFILE}" --window-index "${WINDOW_INDEX}" \
      --window-index-sha256 "${window_sha}" --cache-root "${CACHE_ROOT}" \
      --output "${GROUPED_NORMALIZATION}"
    ;;
  runtime)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    require_file "environment receipt" "${ENV_DIR}/environment_receipt.json"
    validate_5b_v8_contract
    mkdir -p "${CONTROL_ROOT}" "${RUN_ROOT}"
    common=(runtime --model "${MODEL_PROFILE}" --data "${DATA_PROFILE}" \
      --runtime "${RUNTIME_PROFILE}" --objective "${OBJECTIVE_PROFILE}" \
      --environment-lock "${ENV_DIR}/environment_receipt.json" --run-name "${RUN_NAME}" \
      --run-lineage "${RUN_LINEAGE}" --output-root "${RUN_ROOT}" --output "${RUNTIME_YAML}")
    if [[ "${WM3D_DATA_MODE}" == streaming_raw ]]; then
      require_file "streaming metadata seal" "${STREAMING_METADATA_SEAL}"
      "${ENTRY}" "${common[@]}" --data-mode streaming_raw \
        --streaming-metadata-seal "${STREAMING_METADATA_SEAL}" \
        --streaming-lru-root "${STREAMING_LRU_ROOT}" \
        --streaming-lru-gib-per-rank "${STREAMING_LRU_GIB_PER_RANK}" \
        --streaming-encode-batch-frames "${STREAMING_ENCODE_BATCH_FRAMES}" \
        --streaming-decode-workers "${STREAMING_DECODE_WORKERS}"
    elif [[ "${WM3D_DATA_MODE}" == direct_raw ]]; then
      require_file "direct metadata seal" "${STREAMING_METADATA_SEAL}"
      "${ENTRY}" "${common[@]}" --data-mode direct_raw \
        --streaming-metadata-seal "${STREAMING_METADATA_SEAL}" \
        --direct-input-rgb-size "${DIRECT_INPUT_RGB_SIZE}" \
        --direct-decode-workers "${DIRECT_DECODE_WORKERS}" \
        --direct-robot-cache-episodes "${DIRECT_ROBOT_CACHE_EPISODES}" \
        --direct-prefetch-windows "${DIRECT_PREFETCH_WINDOWS}" \
        --direct-video-index-cache-assets "${DIRECT_VIDEO_INDEX_CACHE_ASSETS}" \
        --direct-encode-chunk-rows "${DIRECT_ENCODE_CHUNK_ROWS}" \
        --direct-minimum-chunk-rows "${DIRECT_MINIMUM_CHUNK_ROWS}" \
        --direct-appearance-feature-layer "${DIRECT_APPEARANCE_FEATURE_LAYER}"
    else
      "${ENTRY}" "${common[@]}" --data-mode episode_cache \
        --cache-root "${CACHE_ROOT}" --episode-cache-index "${EPISODE_INDEX}" \
        --episode-cache-seal "${EPISODE_SEAL}" --cache-index "${WINDOW_INDEX}" \
        --cache-seal "${WINDOW_SEAL}" \
        --grouped-normalization "${GROUPED_NORMALIZATION}"
    fi
    ;;
  preflight)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    mapfile -t distributed < <(torch_args "${PREFLIGHT_PORT}")
    "${ENTRY}" preflight "${distributed[@]}" -- --runtime "${RUNTIME_YAML}"
    ;;
  train)
    [[ $# -le 1 ]] || { usage; exit 2; }
    mapfile -t distributed < <(torch_args "${TRAIN_PORT}")
    app=(--runtime "${RUNTIME_YAML}")
    [[ $# -eq 0 ]] || app+=(--stop-after-step "$1")
    "${ENTRY}" train "${distributed[@]}" -- "${app[@]}"
    ;;
  resume)
    [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
    checkpoint=$(resolve_checkpoint "$1")
    mapfile -t distributed < <(torch_args "${TRAIN_PORT}")
    app=(--runtime "${RUNTIME_YAML}" --resume "${checkpoint}")
    [[ $# -eq 1 ]] || app+=(--stop-after-step "$2")
    "${ENTRY}" train "${distributed[@]}" -- "${app[@]}"
    ;;
  eval)
    [[ $# -le 2 ]] || { usage; exit 2; }
    checkpoint=$(resolve_checkpoint "${1:-${TOTAL_STEPS}}")
    checkpoint_step=$(step_from_checkpoint "${checkpoint}")
    output=${2:-$(eval_output_for_step "${checkpoint_step}")}
    mapfile -t distributed < <(torch_args "${EVAL_PORT}")
    "${ENTRY}" eval "${distributed[@]}" -- --runtime "${RUNTIME_YAML}" \
      --checkpoint "${checkpoint}" --output "${output}"
    ;;
  slurm)
    [[ "${SCALE}" == 5b ]] || die "slurm 交付入口只用于 5B"
    [[ $# -ge 1 ]] || { usage; exit 2; }
    operation=$1
    shift
    case "${operation}" in
      preflight|train|resume|eval) ;;
      *) die "slurm operation 必须是 preflight、train、resume 或 eval" ;;
    esac
    [[ -n "${SLURM_JOB_NODELIST:-}" ]] || \
      die "slurm 必须在已分配的 Slurm allocation 内运行"
    command -v scontrol >/dev/null || die "找不到 scontrol"
    command -v srun >/dev/null || die "找不到 srun"
    master_addr=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
    [[ -n "${master_addr}" ]] || die "无法从 SLURM_JOB_NODELIST 解析 master 节点"
    exec srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
      --kill-on-bad-exit=1 --export="ALL,MASTER_ADDR=${master_addr}" \
      bash -lc 'code_root=$1; shift; cd "$code_root"; exec "$@"' \
      _ "${ROOT}" "${ENTRY}" "${SCALE}" "${operation}" "${site}" "$@"
    ;;
  status)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    args=(--allow-incomplete --data-profile "${DATA_PROFILE}" --task-manifest "${TASK_MANIFEST}" \
      --episode-index "${EPISODE_INDEX}" --window-index "${WINDOW_INDEX}" \
      --runtime "${RUNTIME_YAML}" --run-root "${RUN_ROOT}")
    if [[ "${WM3D_DATA_MODE}" == episode_cache ]]; then
      args+=(--episode-seal "${EPISODE_SEAL}" --window-seal "${WINDOW_SEAL}")
    fi
    [[ ! -f "${EVAL_OUTPUT}" ]] || args+=(--eval "${EVAL_OUTPUT}")
    "${PYTHON_BIN}" "${ROOT}/scripts/tools/report_5b_run.py" "${args[@]}"
    ;;
  verify)
    [[ $# -le 3 ]] || { usage; exit 2; }
    if [[ $# -eq 3 ]]; then
      expected_step=$1
      checkpoint=$2
      eval_receipt=$3
    else
      checkpoint=$(resolve_checkpoint "${1:-${TOTAL_STEPS}}")
      expected_step=$(step_from_checkpoint "${checkpoint}")
      eval_receipt=${2:-$(eval_output_for_step "${expected_step}")}
    fi
    report_args=(--data-profile "${DATA_PROFILE}" --task-manifest "${TASK_MANIFEST}" \
      --episode-index "${EPISODE_INDEX}" --window-index "${WINDOW_INDEX}" \
      --runtime "${RUNTIME_YAML}" --run-root "${RUN_ROOT}" \
      --expected-step "${expected_step}" --checkpoint "${checkpoint}" \
      --eval "${eval_receipt}" --require-complete)
    if [[ "${WM3D_DATA_MODE}" == episode_cache ]]; then
      report_args+=(--episode-seal "${EPISODE_SEAL}" --window-seal "${WINDOW_SEAL}")
    fi
    "${PYTHON_BIN}" "${ROOT}/scripts/tools/report_5b_run.py" \
      "${report_args[@]}"
    ;;
  *) usage; exit 2 ;;
esac
