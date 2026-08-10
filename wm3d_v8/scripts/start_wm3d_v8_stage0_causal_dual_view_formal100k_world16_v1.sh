#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

# Sole formal entry point. Run on node43 after the world16 cache is sealed and
# byte-identical on node43/node44. Both hosts must pass the full preflight.
PROJECT=/data/Minko/wm3d_v8_stage0_formal_code_20260810_v2
RUNTIME_ROOT=/data/Minko/world_model/wm3d_v8_stage0_causal_dual_view_20260809
CFG=${RUNTIME_ROOT}/manifests/formal_world16_v1/runtime_config_formal100k_world16_v1.yaml
SEAL=${RUNTIME_ROOT}/manifests/formal_world16_v1/seal_report_formal100k_world16_v1.json
PREFLIGHT=${PROJECT}/scripts/preflight_wm3d_v8_stage0_causal_dual_view.py
NODE_LAUNCHER=${PROJECT}/scripts/launch_wm3d_v8_stage0_causal_dual_view_formal100k_world16_node_v1.sh
NAME=formal100k_world16_node43_node44_v1
LOG_ROOT=${RUNTIME_ROOT}/logs/${NAME}
PY=/data/Minko/.venvs/wm3d/bin/python
CONFIRM=${WM3D_V8_STAGE0_FORMAL:-}
EXPECTED_CONFIRM=EXECUTE_WM3D_V8_STAGE0_FORMAL_WORLD16_V1

if [[ "${CONFIRM}" != "${EXPECTED_CONFIRM}" ]]; then
  echo "set WM3D_V8_STAGE0_FORMAL=${EXPECTED_CONFIRM}" >&2
  exit 1
fi

cd "${PROJECT}"
test -s "${CFG}"
test -s "${SEAL}"
test -x "${PREFLIGHT}"
test -x "${NODE_LAUNCHER}"
mkdir -p "${LOG_ROOT}"

seal_sha=$(sha256sum "${SEAL}" | awk '{print $1}')
resolved_sha=$(PYTHONPATH="${PROJECT}" "${PY}" - "${CFG}" <<'PY'
import sys
from pathlib import Path
from scripts.preflight_wm3d_v8_stage0_causal_dual_view import load_config, resolved_config_sha256
print(resolved_config_sha256(load_config(Path(sys.argv[1]))))
PY
)

declare -A HOSTS=( [0]=172.27.0.6 [1]=172.27.0.7 )
declare -A NAMES=( [0]=node43 [1]=node44 )

run_host() {
  local rank=$1
  shift
  if [[ "${rank}" == "0" ]]; then
    "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=15 "root@${HOSTS[$rank]}" "$@"
  fi
}

CODE_FILES=(
  configs/wm3d_v8_stage0_causal_dual_view_actionpolicy_formal100k_world16_node43_node44_v1.yaml
  scripts/preflight_wm3d_v8_stage0_causal_dual_view.py
  scripts/launch_wm3d_v8_stage0_causal_dual_view_formal100k_world16_node_v1.sh
  wm3d_v3/data/v8_causal_dual_view.py
  wm3d_v3/data/window_dataset.py
  wm3d_v3/data/v7_compact_dataset.py
  wm3d_v3/training/train.py
)
local_code_manifest=$(sha256sum "${CODE_FILES[@]}")
remote_code_manifest=$(ssh -o BatchMode=yes -o ConnectTimeout=15 root@172.27.0.7 \
  "cd '${PROJECT}' && sha256sum ${CODE_FILES[*]}")
if [[ "${local_code_manifest}" != "${remote_code_manifest}" ]]; then
  echo "node43/node44 formal code manifests differ" >&2
  exit 1
fi
code_manifest_sha=$(printf '%s\n' "${local_code_manifest}" | sha256sum | awk '{print $1}')

echo "formal resolved config SHA256: ${resolved_sha}"
for rank in 0 1; do
  report="${LOG_ROOT}/preflight_full_rank${rank}.json"
  stdout_log="${LOG_ROOT}/preflight_full_rank${rank}.stdout.log"
  echo "full preflight ${NAMES[$rank]} rank=${rank}"
  if [[ "${rank}" == "0" ]]; then
    PYTHONPATH="${PROJECT}" "${PY}" "${PREFLIGHT}" \
      --config "${CFG}" --mode full --json-out "${report}" \
      > "${stdout_log}"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=15 "root@${HOSTS[$rank]}" \
      "cd /data/Minko && PYTHONPATH='${PROJECT}' '${PY}' '${PREFLIGHT}' --config '${CFG}' --mode full --json-out '${report}' > '${stdout_log}'"
  fi
done

ssh root@172.27.0.7 "cd /data/Minko && cat '${LOG_ROOT}/preflight_full_rank1.json'" \
  > "${LOG_ROOT}/preflight_full_rank1.remote_copy.json"

"${PY}" - "${resolved_sha}" "${LOG_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

expected, root = sys.argv[1], Path(sys.argv[2])
paths = [
    root / "preflight_full_rank0.json",
    root / "preflight_full_rank1.remote_copy.json",
]
reports = [json.loads(path.read_text()) for path in paths]
for rank, report in enumerate(reports):
    if report.get("passed") is not True or report.get("launch_ready") is not True:
        raise SystemExit(f"rank {rank} preflight is not launch-ready")
    if report.get("errors") != [] or report.get("warnings") != [] or report.get("blockers") != []:
        raise SystemExit(f"rank {rank} preflight contains findings")
    if report.get("resolved_config_sha256") != expected:
        raise SystemExit(f"rank {rank} resolved config mismatch")
    if (report.get("health") or {}).get("compute_apps"):
        raise SystemExit(f"rank {rank} preflight observed active GPU applications")
for key in (
    "runtime",
    "verified_artifacts",
    "source_coverage",
    "cache_contract_hashes",
    "training_assets",
    "dataset_probe",
    "action_objective",
):
    if reports[1].get(key) != reports[0].get(key):
        raise SystemExit(f"rank 1 {key} differs from rank 0")
print("two-host formal preflight closure is equivalent")
PY

started_ranks=()
started_pids=()
cleanup_partial() {
  local index rank pid
  for ((index=${#started_ranks[@]}-1; index>=0; index--)); do
    rank=${started_ranks[$index]}
    pid=${started_pids[$index]}
    echo "partial launch cleanup: exact rank=${rank} launcher=${pid}" >&2
    if [[ "${rank}" == "0" ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    else
      ssh "root@${HOSTS[$rank]}" "kill -TERM -- -${pid} 2>/dev/null || true" || true
    fi
  done
}
trap cleanup_partial ERR

attempt_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
attempt_dir=${LOG_ROOT}/orchestration_attempts/${attempt_id}
mkdir -m 0755 -p "${attempt_dir}"
launch_not_before_epoch=$(( $(date +%s) + 20 ))
declare -a launch_jobs=()

for rank in 0 1; do
  report="${LOG_ROOT}/preflight_full_rank${rank}.json"
  launch_cmd="cd /data/Minko && while [ \"\$(date +%s)\" -lt '${launch_not_before_epoch}' ]; do sleep 0.1; done; WM3D_V8_STAGE0_FORMAL='${EXPECTED_CONFIRM}' WM3D_V8_PREFLIGHT_RESOLVED_SHA='${resolved_sha}' WM3D_V8_PREFLIGHT_REPORT='${report}' WM3D_V8_FORMAL_SEAL_SHA256='${seal_sha}' NODE_RANK='${rank}' MASTER_ADDR='172.27.0.6' MASTER_PORT='29981' '${NODE_LAUNCHER}'"
  if [[ "${rank}" == "0" ]]; then
    (bash -c "${launch_cmd}") \
      > "${attempt_dir}/rank${rank}.stdout.tmp" \
      2> "${attempt_dir}/rank${rank}.stderr.tmp" &
  else
    (ssh -o BatchMode=yes -o ConnectTimeout=15 \
      "root@${HOSTS[$rank]}" "${launch_cmd}") \
      > "${attempt_dir}/rank${rank}.stdout.tmp" \
      2> "${attempt_dir}/rank${rank}.stderr.tmp" &
  fi
  launch_jobs[$rank]=$!
  echo "armed ${NAMES[$rank]} rank=${rank} orchestration_pid=${launch_jobs[$rank]} not_before=${launch_not_before_epoch}"
done

launch_failed=0
for rank in 0 1; do
  if wait "${launch_jobs[$rank]}"; then
    echo "launcher command returned successfully for ${NAMES[$rank]} rank=${rank}"
  else
    status=$?
    echo "launcher command failed for ${NAMES[$rank]} rank=${rank} rc=${status}" >&2
    launch_failed=1
  fi
  mv "${attempt_dir}/rank${rank}.stdout.tmp" "${attempt_dir}/rank${rank}.stdout"
  mv "${attempt_dir}/rank${rank}.stderr.tmp" "${attempt_dir}/rank${rank}.stderr"
done

for rank in 0 1; do
  pid=$(tail -n 1 "${attempt_dir}/rank${rank}.stdout" 2>/dev/null || true)
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    pid_file="${LOG_ROOT}/launcher_rank${rank}_step_00000000_to_00001000.pid"
    if [[ "${rank}" == "0" ]]; then
      pid=$(tail -n 1 "${pid_file}" 2>/dev/null || true)
    else
      pid=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
        "root@${HOSTS[$rank]}" "tail -n 1 '${pid_file}' 2>/dev/null || true" || true)
    fi
  fi
  if [[ "${pid}" =~ ^[0-9]+$ ]]; then
    started_ranks+=("${rank}")
    started_pids+=("${pid}")
    echo "started ${NAMES[$rank]} rank=${rank} launcher=${pid}"
  else
    echo "missing valid launcher PID for ${NAMES[$rank]} rank=${rank}" >&2
    launch_failed=1
  fi
done

if [[ "${launch_failed}" -ne 0 || "${#started_pids[@]}" -ne 2 ]]; then
  trap - ERR
  cleanup_partial
  exit 1
fi

trap - ERR
sleep 20
for index in 0 1; do
  rank=${started_ranks[$index]}
  pid=${started_pids[$index]}
  if [[ "${rank}" == "0" ]]; then
    kill -0 "${pid}"
    child_count=$(ps --ppid "${pid}" -o pid= | wc -l)
  else
    ssh "root@${HOSTS[$rank]}" "kill -0 '${pid}'"
    child_count=$(ssh "root@${HOSTS[$rank]}" "ps --ppid '${pid}' -o pid= | wc -l")
  fi
  if [[ "${child_count}" -ne 8 ]]; then
    echo "WARNING: ${NAMES[$rank]} launcher ${pid} currently has ${child_count} direct children" >&2
  fi
done

receipt="${LOG_ROOT}/formal_launch_receipt_step_00000000_to_00001000.json"
"${PY}" - "${receipt}" "${resolved_sha}" "${seal_sha}" "${code_manifest_sha}" \
  "${started_pids[*]}" "${launch_not_before_epoch}" "${attempt_dir}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path, config_sha, seal_sha, code_sha, raw_pids, launch_epoch, attempt_dir = sys.argv[1:]
payload = {
    "schema": "wm3d_v8_stage0_causal_dual_view_formal_launch_receipt_v1",
    "committed": True,
    "resolved_config_sha256": config_sha,
    "seal_report_sha256": seal_sha,
    "code_manifest_sha256": code_sha,
    "launch_mode": "concurrent_two_host_absolute_time_gate",
    "launch_not_before_epoch": int(launch_epoch),
    "orchestration_attempt_dir": attempt_dir,
    "nodes": [
        {"rank": 0, "host": "node43", "ip": "172.27.0.6", "launcher_pid": int(raw_pids.split()[0])},
        {"rank": 1, "host": "node44", "ip": "172.27.0.7", "launcher_pid": int(raw_pids.split()[1])},
    ],
    "formal_target_step": 100000,
    "invocation_hard_stop_step": 1000,
    "initialization": "fresh_random_world_with_frozen_pinned_codec",
    "unix_time": time.time(),
}
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
if target.exists():
    if target.read_bytes() != encoded:
        raise SystemExit(f"existing launch receipt is non-identical: {target}")
else:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
