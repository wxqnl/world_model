# Data module guide

- Preserve source-native cadence and physical values in episode caches.
- Adapters are the only place that map raw fields into grouped action/current-state semantics.
- Fine command, coarse effect, and current-state normalization lanes are separate.
- Cache identity binds source row, adapter, representation, encoders, and task bank. Model window
  shape and training budget belong downstream.
- Missing views, groups, state, or supervision remain explicitly masked; never synthesize them.
- The foundation bridge may reshape source-native commands but must never interpolate them. Three
  views form one WSA canvas; exact command timestamps, grouped masks and source normalization remain
  recoverable at serving. Future commands retain their original K world intervals: trim only a
  common unused suffix of per-interval capacity, never globally compact active rows. Irregular
  command counts remain masked slots; do not shift commands between intervals or synthesize values.
- Legacy foundation mode encodes three timestamps (current, sampled horizon, final).
  Native trajectory mode encodes the current observation plus all K future timestamps,
  in small independent-time VGGT batches. Future targets never enter model inputs.
  Spatial resizing is per view and must not mix cameras or temporal rows.
- Reuse an approved whole-episode selection verbatim. Filter training only; keep validation
  unchanged and preserve the saved source-window proportions. Never count overlapping windows
  as hours or replace the source mixture with a motion-selected/equal-source subset.
  Missing raw mounts are an infrastructure error, not permission to silently drop sources.
