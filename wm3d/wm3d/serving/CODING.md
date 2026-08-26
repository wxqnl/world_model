# Serving guide

- Serving must consume the same grouped-action ABI, physical units, masks, offsets, and scales as
  Stage0 training; do not add inference-only action conversions.
- Panda/LIBERO uses H16 fine action history, K8 candidate actions, and executes H1 at 20 Hz.
- Future candidate actions must remain outside the action-free state and policy trunks.
- Reject malformed shapes or missing normalization metadata instead of padding or guessing.
