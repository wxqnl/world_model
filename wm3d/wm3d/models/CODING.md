# Model guide

- `NativeWorldModel` is the shared implementation for every capacity profile.
- Keep policy state action-free and factual world refinement explicitly action-conditioned.
- Preserve continuous time, masks, grouped embodiment identity, and explicit RGB/depth/point/pose
  heads.
- A capacity change belongs in a model profile, not a forked model class.
