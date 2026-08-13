# Configuration guide

- Configuration files describe contracts; they must not hide dataset-specific defaults in code.
- Model, data, objective, encoder, runtime, and Stage1 profiles remain orthogonal.
- Units, coordinate frames, action groups, state groups, masks, cadence, and supervision lanes
  must be explicit.
- Formal templates use placeholders when real revisions, paths, or operator attestations are not
  known. Do not guess production values.
- Existing schema identifiers are immutable compatibility ABI. New profile names use `wm3d` or
  neutral capability names, not release numbers.
