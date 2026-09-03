# Training guide

- Stage0 training and offline evaluation use the same model factory and objective.
- DDP and FSDP2 are runtime choices, not different trainers.
- Checkpoints are complete numbered DCP directories with `COMMITTED.json`; never restore from an
  unverified `latest` alias.
- Required gradient owners must be covered exactly once and have finite, nonzero gradients after
  the first real accumulated backward.
- Resource qualification is launch-specific; stable run contracts must remain resumable.
- Foundation profiles use the joint WSA-owned objective and the foundation environment. Do not mix
  native RGB/action losses into that objective or silently fall back to the base environment.
