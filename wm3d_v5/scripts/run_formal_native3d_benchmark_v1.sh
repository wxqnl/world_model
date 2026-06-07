#!/usr/bin/env bash
set -euo pipefail

cd /data/Minko/world_model/wm3d_v3

PY=${PY:-/data/Minko/.venvs/wm3d/bin/python}
LIBERO_PY=${LIBERO_PY:-/data/Minko/.conda-envs/libero-py38/bin/python}
GPU=${GPU:-7}
CFG=${CFG:-configs/v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1.yaml}
CKPT=${CKPT:-results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/best.pt}
OUT_DIR=${OUT_DIR:-results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/formal_native3d_benchmark_v1}
MAX_BATCHES_PER_DATASET=${MAX_BATCHES_PER_DATASET:-8}
N_VIZ=${N_VIZ:-6}
LIBERO_SUITE=${LIBERO_SUITE:-libero_10}
LIBERO_ROOT=${LIBERO_ROOT:-/data/Minko/benchmarks/LIBERO}

# Optional closed-loop trace evidence. If provided, this turns the report from an
# offline 3D world-model benchmark into a public-simulator task-progress report.
LIBERO_ROLLOUT_JSON=${LIBERO_ROLLOUT_JSON:-}
LIBERO_EXPERT_REF_NPZ=${LIBERO_EXPERT_REF_NPZ:-}
LIBERO_TARGET_OBJECTS=${LIBERO_TARGET_OBJECTS:-cream_cheese_1,butter_1}
LIBERO_RECEPTACLE=${LIBERO_RECEPTACLE:-basket_1}

mkdir -p "$OUT_DIR/logs"

native_out="$OUT_DIR/native3d_world_model"
OUT_DIR="$native_out" GPU="$GPU" MAX_BATCHES_PER_DATASET="$MAX_BATCHES_PER_DATASET" N_VIZ="$N_VIZ" \
  scripts/run_world3d_native_benchmark_v2.sh \
  > "$OUT_DIR/logs/native3d_world_model.log" 2>&1

"$LIBERO_PY" -m wm3d_v3.benchmarks.libero_probe \
  --root "$LIBERO_ROOT" \
  --suite "$LIBERO_SUITE" \
  --out "$OUT_DIR/libero_probe.json" \
  > "$OUT_DIR/logs/libero_probe.log" 2>&1 || true

trace_summary=""
object3d_eval=""
if [[ -n "$LIBERO_ROLLOUT_JSON" && -f "$LIBERO_ROLLOUT_JSON" ]]; then
  trace_summary="$OUT_DIR/libero_trace_summary.json"
  "$PY" -m wm3d_v3.benchmarks.libero_trace_summary \
    --input "$LIBERO_ROLLOUT_JSON" \
    --out "$trace_summary" \
    > "$OUT_DIR/logs/libero_trace_summary.log" 2>&1 || true

  if [[ -n "$LIBERO_EXPERT_REF_NPZ" && -f "$LIBERO_EXPERT_REF_NPZ" ]]; then
    object3d_eval="$OUT_DIR/libero_object3d_progress.json"
    "$PY" -m wm3d_v3.benchmarks.libero_object_contact_eval \
      --rollout_json "$LIBERO_ROLLOUT_JSON" \
      --expert_ref_npz "$LIBERO_EXPERT_REF_NPZ" \
      --target_objects "$LIBERO_TARGET_OBJECTS" \
      --receptacle "$LIBERO_RECEPTACLE" \
      --out "$object3d_eval" \
      > "$OUT_DIR/logs/libero_object3d_progress.log" 2>&1 || true
  fi
fi

"$PY" - "$OUT_DIR" "$CFG" "$CKPT" "$native_out/world3d_claim_balanced.json" \
  "$OUT_DIR/libero_probe.json" "$trace_summary" "$object3d_eval" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
cfg, ckpt = sys.argv[2], sys.argv[3]
native_path = Path(sys.argv[4])
libero_probe_path = Path(sys.argv[5])
trace_summary_path = Path(sys.argv[6]) if sys.argv[6] else None
object3d_path = Path(sys.argv[7]) if sys.argv[7] else None

def read(path):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())

native = read(native_path) or {}
libero = read(libero_probe_path) or {}
trace = read(trace_summary_path)
object3d = read(object3d_path)
core = (native.get("core_contribution") or {}).get("evidence") or {}

visuals = sorted(str(p) for p in (out_dir / "native3d_world_model" / "visuals").glob("*.gif"))
card = {
    "name": "WM3D Formal Native-3D Benchmark Card",
    "version": "v1",
    "cfg": cfg,
    "ckpt": ckpt,
    "benchmarks": {
        "public_task_benchmark": {
            "name": "LIBERO",
            "suite": libero.get("suite"),
            "task_api_available": bool(libero.get("task_api_available")),
            "env_api_available": bool(libero.get("env_api_available")),
            "num_tasks": libero.get("num_tasks"),
            "closed_loop_success_rate": None if trace is None else trace.get("success_rate"),
        },
        "native3d_world_prediction": {
            "source": str(native_path),
            "sampling": native.get("sampling"),
            "counts": native.get("counts"),
            "depth_future_l1": core.get("depth_future_l1"),
            "depth_change_l1": core.get("depth_change_l1"),
            "depth_change_cos": core.get("depth_change_cos"),
            "depth_change_sign_acc": core.get("depth_change_sign_acc"),
            "motion_region_depth_l1": core.get("motion_region_depth_l1"),
            "motion_region_depth_change_l1": core.get("motion_region_depth_change_l1"),
            "motion_region_depth_temporal_delta_l1": core.get("motion_region_depth_temporal_delta_l1"),
            "real_action_depth_win_rate": core.get("mean_depth_win_rate_vs_counterfactual"),
            "real_action_depth_change_win_rate": core.get("mean_depth_change_win_rate_vs_counterfactual"),
            "real_action_motion_region_depth_win_rate": core.get("mean_motion_region_depth_win_rate_vs_counterfactual"),
            "real_action_token_win_rate": core.get("mean_token_win_rate_vs_counterfactual"),
        },
        "libero_object_state_3d_progress": None if object3d is None else {
            "source": str(object3d_path),
            "success_rate": object3d.get("success_rate"),
            "stage_score_mean": object3d.get("stage_score_mean"),
            "target_objects": object3d.get("target_objects"),
            "receptacle": object3d.get("receptacle"),
            "episodes": object3d.get("episodes"),
        },
    },
    "visual_artifacts": {
        "native3d_counterfactual_gifs": visuals,
    },
    "claim_boundary": {
        "what_this_proves": [
            "The model predicts future depth/3D-change signals, not just RGB.",
            "The real robot action beats counterfactual actions on future depth, depth-change, motion-region depth, and latent tokens.",
            "When a LIBERO rollout trace is supplied, object/eef/receptacle distances are evaluated in simulator 3D object-state coordinates.",
        ],
        "what_this_does_not_prove": [
            "It is not a public leaderboard submission unless closed-loop LIBERO episodes are run at scale.",
            "It is not a tau0 simulator comparison because tau0's simulator weights/test-time compute code are not public in the pulled repo.",
        ],
    },
}
out = out_dir / "benchmark_card.json"
out.write_text(json.dumps(card, indent=2, sort_keys=True))
print(json.dumps({"benchmark_card": str(out)}, indent=2, sort_keys=True))
PY

echo "formal_native3d_benchmark_done out=$OUT_DIR"
