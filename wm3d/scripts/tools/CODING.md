# Diagnostic and reporting tools

- Keep the reusable factual-motion microprobe, production RGB/Action A/B and checkpoint exports.
  A historical tiny-model or transport probe is not evidence for the current full native-direct run.
- Use production model/data/objective code for current-path qualification. No parallel trainer,
  fabricated targets or hand-picked success-only samples.
- Run probes in separate output roots. Never overwrite formal checkpoints, alter a running tree,
  change source weights or take GPUs owned by another job.
- Reporting tools are read-only except for their requested output artifacts.
- Separate local-input checks, parameter/meta construction, real distributed execution and model
  quality. None can stand in for the others.
- Preserve diagnostic evidence; qualify resume only through the production checkpoint contract.
