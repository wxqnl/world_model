# Operator reporting guide

- Reporting tools are read-only unless an explicit `--output` summary is requested.
- A report may summarize receipts; it must not manufacture or replace training authority.
- Pipeline health and model quality are different. Finite losses and valid coverage prove the
  run is wired correctly, not that the learned model is useful.
- Any reported PASS must be derived from sealed runtime, committed checkpoint, finite metrics,
  gradient ownership, and evaluation coverage.
