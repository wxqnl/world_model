# Script guide

- Scripts are thin orchestration or materialization boundaries; reusable semantics live in the
  `wm3d` package.
- Every publishing script must be no-clobber and emit a receipt or seal that binds its inputs.
- Expensive stages must support deterministic partitioning and safe re-entry.
- Do not add a second model factory, trainer, checkpoint format, or Stage1 action policy.
- User-facing scripts are exposed through `run_wm3d.sh`; private workers may remain direct test
  utilities.
