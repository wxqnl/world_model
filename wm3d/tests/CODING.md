# Test guide

- Tests encode release contracts, not only happy-path numerical behavior.
- Data and receipt loaders need missing-field, tamper, mask, shape, and lineage failures.
- Training changes need finite-loss and gradient-owner evidence.
- Distributed changes need real multi-process coverage proportional to risk.
- Keep tests deterministic and runnable with `./run_wm3d.sh check` without filters or exclusions.
