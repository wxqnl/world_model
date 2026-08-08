#!/usr/bin/env bash
set -euo pipefail

# Sole formal entry point. Run on node43 after the 1K gate receipt is pinned in
# the formal config. It verifies all three hosts before starting any rank.
ROOT=/data/Minko/world_model/wm3d_v7_actionrepair1b_20260806
CFG=configs/wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3.yaml
PREFLIGHT=scripts/preflight_wm3d_v7_1b_actionpolicy_joint.py
NODE_LAUNCHER=scripts/launch_wm3d_v7_1b_actionpolicy_joint_formal100k_node_v3.sh
NAME=wm3d_v7_1b_native_actionpolicy_joint_formal100k_3node24_v3
LOG_ROOT=${ROOT}/logs/${NAME}
PY=/data/Minko/.venvs/wm3d/bin/python
CONFIRM=${WM3D_V7_FORMAL_RETRAIN:-}
EXPECTED_CONFIRM=EXECUTE_WM3D_V7_1B_ACTIONPOLICY_FORMAL100K_V3

if [[ "${CONFIRM}" != "${EXPECTED_CONFIRM}" ]]; then
  echo "set WM3D_V7_FORMAL_RETRAIN=${EXPECTED_CONFIRM}" >&2
  exit 1
fi

cd "${ROOT}"
test -s "${CFG}"
test -x "${PREFLIGHT}"
test -x "${NODE_LAUNCHER}"
mkdir -p "${LOG_ROOT}"

resolved_sha=$(PYTHONPATH="${ROOT}" "${PY}" - "${CFG}" <<'PY'
import sys
from pathlib import Path
from scripts.preflight_wm3d_v7_stage0_actiondynamics import load_config, resolved_config_sha256
print(resolved_config_sha256(load_config(Path(sys.argv[1]))))
PY
)

declare -A HOSTS=( [0]=172.27.0.7 [1]=172.27.0.4 [2]=172.27.0.6 )
declare -A NAMES=( [0]=node44 [1]=node41 [2]=node43 )

run_host() {
  local rank=$1
  shift
  if [[ "${rank}" == "2" ]]; then
    "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=15 "root@${HOSTS[$rank]}" "$@"
  fi
}

echo "formal resolved config SHA256: ${resolved_sha}"
for rank in 0 1 2; do
  report="${LOG_ROOT}/preflight_full_rank${rank}.json"
  echo "full preflight ${NAMES[$rank]} rank=${rank}"
  if [[ "${rank}" == "2" ]]; then
    PYTHONPATH="${ROOT}" "${PY}" "${PREFLIGHT}" \
      --config "${CFG}" --mode full --json-out "${report}" >/dev/null
  else
    ssh -o BatchMode=yes -o ConnectTimeout=15 "root@${HOSTS[$rank]}" \
      "cd /data/Minko && PYTHONPATH='${ROOT}' '${PY}' '${ROOT}/${PREFLIGHT}' --config '${ROOT}/${CFG}' --mode full --json-out '${report}' >/dev/null"
  fi
done

# Compare the complete immutable runtime/data closure before launch. Reports
# live on each node's local /data, so stream the two remote reports to node43.
for rank in 0 1; do
  ssh "root@${HOSTS[$rank]}" "cat '${LOG_ROOT}/preflight_full_rank${rank}.json'" \
    > "${LOG_ROOT}/preflight_full_rank${rank}.remote_copy.json"
done

"${PY}" - "${resolved_sha}" "${LOG_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

expected, root = sys.argv[1], Path(sys.argv[2])
paths = [
    root / "preflight_full_rank0.remote_copy.json",
    root / "preflight_full_rank1.remote_copy.json",
    root / "preflight_full_rank2.json",
]
reports = [json.load(path.open()) for path in paths]
for index, report in enumerate(reports):
    if report.get("passed") is not True or report.get("launch_ready") is not True:
        raise SystemExit(f"rank {index} preflight is not launch-ready")
    if report.get("resolved_config_sha256") != expected:
        raise SystemExit(f"rank {index} resolved config mismatch")
for key in ("runtime", "verified_artifacts", "source_cycle_counts", "global_batch"):
    first = reports[0].get(key)
    for index, report in enumerate(reports[1:], 1):
        # runtime embeds local paths, which are identical by contract.
        if report.get(key) != first:
            raise SystemExit(f"rank {index} {key} differs from rank 0")
print("three-host preflight closure is byte-equivalent")
PY

started_ranks=()
started_pids=()
cleanup_partial() {
  local index rank pid
  for ((index=${#started_ranks[@]}-1; index>=0; index--)); do
    rank=${started_ranks[$index]}
    pid=${started_pids[$index]}
    echo "partial launch cleanup: exact rank=${rank} launcher=${pid}" >&2
    if [[ "${rank}" == "2" ]]; then
      kill -TERM -- "-${pid}" 2>/dev/null || true
    else
      ssh "root@${HOSTS[$rank]}" "kill -TERM -- -${pid} 2>/dev/null || true" || true
    fi
  done
}
trap cleanup_partial ERR

# The representative 24-rank transport audit starts all hosts concurrently.
# Do the same here, and additionally make every host cross an absolute-time
# gate before it spawns torchrun.  Starting rank 0 and waiting for its launcher
# response before contacting ranks 1/2 repeatedly caused NCCL's first lazy
# communicator handshake to observe a partially formed world.
attempt_id=$(date -u +%Y%m%dT%H%M%SZ)_$$
attempt_dir=${LOG_ROOT}/orchestration_attempts/${attempt_id}
mkdir -m 0755 -p "${attempt_dir}"
launch_not_before_epoch=$(( $(date +%s) + 20 ))
declare -a launch_jobs=()

for rank in 0 1 2; do
  report="${LOG_ROOT}/preflight_full_rank${rank}.json"
  launch_cmd="cd /data/Minko && while [ \"\$(date +%s)\" -lt '${launch_not_before_epoch}' ]; do sleep 0.1; done; WM3D_V7_FORMAL_RETRAIN='${EXPECTED_CONFIRM}' WM3D_V7_PREFLIGHT_RESOLVED_SHA='${resolved_sha}' WM3D_V7_PREFLIGHT_REPORT='${report}' NODE_RANK='${rank}' '${ROOT}/${NODE_LAUNCHER}'"
  if [[ "${rank}" == "2" ]]; then
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
for rank in 0 1 2; do
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

# Parse all successful launcher responses before deciding whether cleanup is
# needed, so a partial concurrent start is still owned and can be terminated
# exactly.  Fall back to the node-local PID file if SSH returned no stdout.
for rank in 0 1 2; do
  pid=$(tail -n 1 "${attempt_dir}/rank${rank}.stdout" 2>/dev/null || true)
  if [[ ! "${pid}" =~ ^[0-9]+$ ]]; then
    pid_file="${LOG_ROOT}/launcher_rank${rank}.pid"
    if [[ "${rank}" == "2" ]]; then
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

if [[ "${launch_failed}" -ne 0 || "${#started_pids[@]}" -ne 3 ]]; then
  trap - ERR
  cleanup_partial
  exit 1
fi

# Once all three torchrun launchers exist, ownership is committed. Never let a
# later monitoring failure terminate the formal job.
trap - ERR
sleep 20
for index in 0 1 2; do
  rank=${started_ranks[$index]}
  pid=${started_pids[$index]}
  if [[ "${rank}" == "2" ]]; then
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

receipt="${LOG_ROOT}/formal_launch_receipt.json"
"${PY}" - "${receipt}" "${resolved_sha}" "${started_pids[*]}" "${launch_not_before_epoch}" "${attempt_dir}" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

path, digest, raw_pids, launch_epoch, attempt_dir = sys.argv[1:]
payload = {
    "schema": "wm3d_v7_1b_native_actionpolicy_joint_formal_launch_receipt_v3",
    "committed": True,
    "resolved_config_sha256": digest,
    "launch_mode": "concurrent_three_host_absolute_time_gate",
    "launch_not_before_epoch": int(launch_epoch),
    "orchestration_attempt_dir": attempt_dir,
    "nodes": [
        {"rank": 0, "host": "node44", "ip": "172.27.0.7", "launcher_pid": int(raw_pids.split()[0])},
        {"rank": 1, "host": "node41", "ip": "172.27.0.4", "launcher_pid": int(raw_pids.split()[1])},
        {"rank": 2, "host": "node43", "ip": "172.27.0.6", "launcher_pid": int(raw_pids.split()[2])},
    ],
    "target_step": 100000,
    "initialization": "fresh_random_world_with_frozen_pinned_codec",
    "serving_action_owner": "direct_pose_plus_delta_composed_gripper",
    "auxiliary_action_owner": "pose_only_flow_matching",
    "unix_time": time.time(),
}
temporary = Path(path).with_name("." + Path(path).name + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
