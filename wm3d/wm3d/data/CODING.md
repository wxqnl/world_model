# Data module guide

- Preserve source-native cadence and physical values in episode caches.
- Adapters are the only place that map raw fields into grouped action/current-state semantics.
- Fine command, coarse effect, and current-state normalization lanes are separate.
- Cache identity binds source row, adapter, representation, encoders, and task bank. Model window
  shape and training budget belong downstream.
- Missing views, groups, state, or supervision remain explicitly masked; never synthesize them.
