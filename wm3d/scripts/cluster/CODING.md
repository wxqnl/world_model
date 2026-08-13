# Cluster orchestration guide

- Cluster scripts are thin wrappers around `run_wm3d.sh`; training and data semantics stay in
  the package and existing materializers.
- Every action must be safe to run from any node when the site file points to shared storage.
- Do not infer adapter units, frames, gripper polarity, source cadence, or source revisions.
- A failed preflight, missing seal, stale resource receipt, or incomplete checkpoint is terminal.
- Keep scheduler-specific examples in documentation rather than hiding them in training code.
