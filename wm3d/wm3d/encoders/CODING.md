# Encoder guide

- Encoders are frozen, version-pinned boundaries that produce explicit cache representations.
- Cache identity binds the encoder contract, source revision, task bank, and representation.
- Keep external model loading offline-capable and fail closed on dimension or revision drift.
- Model-capacity-specific T/P/K choices belong in profiles and window indices, not hidden encoder
  defaults.
