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
- Frozen feature reuse is instance-local to the encoder/preprocessing/task manifest.
  Key it by packed episode/frame identity plus the complete camera-availability mask;
  never use sample_index alone. Keep ordinary output tensors and preserve CPU/CUDA RNG.
  A cache speed claim requires actual encoder A/B on unmodified sampler batches,
  including current/K8 ordering and future-label isolation, not only a fake encoder test.
- Reuse an approved whole-episode selection verbatim. Filter training only; keep validation
  unchanged and preserve the saved source-window proportions. Never count overlapping windows
  as hours or replace the source mixture with a motion-selected/equal-source subset.
  Missing raw mounts are an infrastructure error, not permission to silently drop sources.
- Episode selection supplies integer window counts to the exact source sampler, not fractional
  probabilities. Keep the selected-window ratios exact. Cached schedules must reproduce the
  original per-cycle shuffle and per-source occurrence for arbitrary optimizer-step access.
- `parsed_index_cache.py` accelerates only canonical window-index parsing. Keep the original
  source-content check on every call, including memory hits; filesystem timestamps are not enough.
  Cached immutable rows must preserve every field and exact order. Do not cache a modified mixture.
- The trainer shares parsed rows between train/validation and uses a private node-local disk cache
  across preflight/training processes. Serialize cache construction, not warm readers. Publish
  atomically; reject stale/corrupt payloads and rebuild with the original validated parser.
  Release the full-corpus memory reference after both datasets are assembled.
- Bump the parsed-cache format when canonical row validation or `CacheIndexEntry` semantics change.
  `WM3D_INDEX_CACHE_DIR` selects the private cache root; an explicit empty value disables disk reuse.
