# Data module guide

- Preserve source-native cadence and physical values in episode caches.
- Adapters are the only place that map raw fields into grouped action/current-state semantics.
- Fine command, coarse effect, and current-state normalization lanes are separate.
- Cache identity binds source row, adapter, representation, encoders, and task bank. Model window
  shape and training budget belong downstream.
- Missing views, groups, state, or supervision remain explicitly masked; never synthesize them.
- The foundation bridge may reshape source-native commands but must never interpolate them. Three
  views form one WSA canvas; exact command timestamps, grouped masks and source normalization remain
  recoverable at serving.
- The foundation native-3D teacher encodes exactly three timestamps in one frozen VGGT call: last
  observation, one deterministic sampled K8 target, and the final target. The sampled horizon comes
  from the sealed sample index; targets never enter the model input. Spatial resizing is per view and
  must not mix cameras.
