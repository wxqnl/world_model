# Model guide

- `NativeWorldModel` remains the shared implementation for native 1B/5B profiles.
- `FoundationWorldModel` is the V10 WSA-Large adapter. Keep the official Wan2.2 video,
  ActionDiT, Future3D and MoT owners intact; WM3D additions are limited to grouped robot,
  exact-time and frozen-VGGT target adapters.
- WSA-Large does not use Qwen3-VL. Do not add WSA-Base/Qwen components to the foundation profile.
- Keep policy state action-free and factual world refinement explicitly action-conditioned.
- Preserve continuous time, masks, grouped embodiment identity, and explicit RGB/depth/point/pose
  heads.
- A capacity change belongs in a model profile, not a forked model class.
