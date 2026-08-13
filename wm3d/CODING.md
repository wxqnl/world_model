# WM3D coding guide

## Project rules

- Product and code names are `WM3D` and `wm3d`; do not add release numbers to new paths,
  Python modules, commands, or user-facing documentation.
- The only exception is an existing serialized schema/receipt identifier. Those strings are
  immutable disk ABI and may keep their historical version tag.
- `run_wm3d.sh` is the single user entry point. New workflows must compose its existing stages
  or add one explicit subcommand; do not create a parallel trainer.
- Stage0 owns the native world model and executable action policy. Stage1 freezes Stage0 and
  owns only candidate planning/ranking.
- Preserve source-native timestamps, grouped robot semantics, masks, physical units, and sealed
  SHA lineage. Dataset-specific behavior belongs in adapters and profiles, not the trainer.
- Never commit data, caches, model weights, checkpoints, local environments, receipts, or logs.

## Verification

Run from this directory:

```bash
PYTHON_BIN=.venv/bin/python ./run_wm3d.sh check
```

Changes to distributed training or checkpoint code also require the relevant multi-process
worker tests. Changes to data contracts require tamper/fail-closed tests.
